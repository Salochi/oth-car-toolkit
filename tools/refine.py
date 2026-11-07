"""
Refine a low-poly GLB before conversion: more polygons, with the curves the model's normals describe made real.

Run:  blender -b --factory-startup --python tools/refine.py -- <profile.py> <model.glb> <out.glb>

Phong tessellation (Boubekeur & Alexa 2008): every triangle is split into k x k smaller ones, and the new points are
lifted onto the curved patch that the triangle's corner normals describe (recomputed from the welded shape:
smooth below REFINE_SMOOTH_ANGLE, split above it). Flat panels
stay flat (their normals agree), curved ones (wheel arches, tyres, a rounded bonnet) get round silhouettes, and a hard
edge stays hard (each side keeps its own normals). Unlike Catmull-Clark subdivision it is local to each triangle, so
the long thin triangles of a decimated model don't turn into lumps.

The objects of each group the profile names (REFINE_GROUPS: output name -> source object name prefixes) are tessellated
together: a point on an edge two triangles share (also across objects, by position) is the average of both sides', so
no cracks open. The result is one object per material, named <group>_<material>, scaled by REFINE_SCALE (to real-world
metres) and written as a GLB the converters read like any other.

Then the ripples a fan of long thin source triangles leaves are smoothed out (REFINE_FAIR passes, along the normals
only, the hard edges held) and the normals are recomputed from the final shape.

Profile settings: REFINE_GROUPS, REFINE_SPLITS (group -> k, default 2), REFINE_FAIR (group -> passes, default 0), REFINE_ALPHA (how far towards the curved patch,
0..1, default 0.75), REFINE_SMOOTH_ANGLE (degrees: edges sharper than this stay hard, default 30), REFINE_SCALE,
REFINE_ROTATE_Z (degrees about the vertical: the converters want the front along Blender -y or +y), REFINE_WELD (source units: corners closer than this are one point).
"""
import bpy, os, sys, importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # profiles import pipeline's class names
argv = sys.argv[sys.argv.index('--') + 1:]
spec = importlib.util.spec_from_file_location('profile', argv[0]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GLB, OUT = os.path.abspath(argv[1]), os.path.abspath(argv[2])
ALPHA = float(os.environ.get('REFINE_ALPHA', getattr(P, 'REFINE_ALPHA', 0.75)))
SPLITS = getattr(P, 'REFINE_SPLITS', {})
NORMALS = os.environ.get('REFINE_NORMALS', getattr(P, 'REFINE_NORMALS', 'source'))
FAIR = {g: int(os.environ['REFINE_FAIR']) for g in P.REFINE_GROUPS} if 'REFINE_FAIR' in os.environ else getattr(P, 'REFINE_FAIR', {})
SCALE = getattr(P, 'REFINE_SCALE', 1.0)
_r = np.radians(getattr(P, 'REFINE_ROTATE_Z', 0.0))
ROT = np.array([[np.cos(_r), -np.sin(_r), 0], [np.sin(_r), np.cos(_r), 0], [0, 0, 1]])
WELD = getattr(P, 'REFINE_WELD', 1e-5)
WHEELS = getattr(P, 'REFINE_WHEELS', None)
FAIR_MAX = getattr(P, 'REFINE_FAIR_MAX', 0.015)   # metres
SMOOTH_ANGLE = float(os.environ.get('REFINE_SMOOTH_ANGLE', getattr(P, 'REFINE_SMOOTH_ANGLE', 30.0)))


def log(*a):
    print('[refine]', *a, flush=True)


def to_out(co):
    """Source world positions -> the output's (turned and scaled, metres)."""
    return np.asarray(co) @ ROT.T * SCALE


def drop_pieces(bm, boxes, names=()):
    """Deletes the pieces lying wholly inside one of the boxes ((x0, x1, y0, y1, z0, z1) in output metres). A piece is
    a run of faces of one material joined by edges: once welded, a part that only touches another (a leaf spring
    under the floor) is still its own piece. A seventh value, a tuple of material names, drops only pieces of those
    (a wheel tub's black walls, not the paint of the flare's lip beside them)."""
    import bmesh
    seen, gone = set(), []
    for f in bm.faces:
        if f in seen:
            continue
        comp, stack = [], [f]
        seen.add(f)
        while stack:
            x = stack.pop(); comp.append(x)
            for e in x.edges:
                for y in e.link_faces:
                    if y not in seen and y.material_index == x.material_index:
                        seen.add(y); stack.append(y)
        p = to_out([tuple(v.co) for x in comp for v in x.verts])
        lo, hi = p.min(0), p.max(0)
        mat = names[comp[0].material_index] if comp[0].material_index < len(names) else None
        if any(b[0] <= lo[0] and hi[0] <= b[1] and b[2] <= lo[1] and hi[1] <= b[3] and b[4] <= lo[2] and hi[2] <= b[5]
               and (len(b) < 7 or mat in b[6]) for b in boxes):
            gone += comp
    if gone:
        bmesh.ops.delete(bm, geom=gone, context='FACES')
    return len(gone)


def move_verts(bm, moves, tol=0.0015):
    """Hand fixes to the source's shape (REFINE_MOVE: [(from, to)] output points, metres): every vertex within tol of
    a from goes to its to; vertices landing together are welded, the faces collapsing with them dropped, and the
    faces round a moved vertex shade flat (their source normals were for the old shape)."""
    import bmesh
    sl = bm.loops.layers.float_vector.get('src_n')
    co = np.array([tuple(v.co) for v in bm.verts])
    out = to_out(co)
    moved = set()
    for frm, to in moves:
        hit = np.nonzero(np.linalg.norm(out - np.asarray(frm), axis=1) < tol)[0]
        src = (np.asarray(to, float) / SCALE) @ ROT
        bm.verts.ensure_lookup_table()
        for i in hit:
            bm.verts[i].co = tuple(src); moved.add(bm.verts[i])
        log(f'  moved {len(hit)} vertices to {tuple(to)}')
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=WELD)
    bad = [f for f in bm.faces if f.calc_area() < 1e-12]
    if bad:
        bmesh.ops.delete(bm, geom=bad, context='FACES_ONLY')
    bm.normal_update()
    if sl is not None:
        live = {v for v in moved if v.is_valid}
        for f in {f for v in live for f in v.link_faces}:
            for l in f.loops:
                l[sl] = f.normal


def close_tjunctions(bm, tol=5e-4):
    """Splits an open edge where another piece's corner lies on it (a T-junction) and welds the two: flat, such a seam
    is invisible, but once the tessellation bends the edge one way and the corner the other, a crack opens there."""
    import bmesh
    total = 0
    for _ in range(40):
        bnd = [e for e in bm.edges if len(e.link_faces) == 1]
        if not bnd:
            break
        bm.verts.index_update()
        bv = sorted({v for e in bnd for v in e.verts}, key=lambda v: v.index)   # a fixed order: the same splits every run
        A = np.array([tuple(e.verts[0].co) for e in bnd]); B = np.array([tuple(e.verts[1].co) for e in bnd])
        V = np.array([tuple(v.co) for v in bv])
        d = B - A; L2 = np.maximum((d * d).sum(1), 1e-20)
        found = {}
        for i in range(0, len(V), 400):
            p = V[i:i + 400, None, :]
            t = ((p - A) * d).sum(2) / L2
            q = A + np.clip(t, 0, 1)[..., None] * d
            m = (np.linalg.norm(p - q, axis=2) < tol) & (t > 0.01) & (t < 0.99)
            for vi, ei in zip(*np.nonzero(m)):
                if ei not in found:              # one split per edge per round
                    found[ei] = (bv[i + vi], float(t[vi, ei]))
        if not found:
            break
        pairs = []
        for ei, (v, t) in found.items():
            e = bnd[ei]
            if not e.is_valid or not v.is_valid:
                continue
            ne, nv = bmesh.utils.edge_split(e, e.verts[0], t)
            nv.co = v.co.copy()
            pairs += [nv, v]
        bmesh.ops.remove_doubles(bm, verts=list(dict.fromkeys(pairs)), dist=tol)
        total += len(found)
    return total


def taper_crease_ends(bm, sl, mats=None, angle=5.0, skip=()):
    """Where a crease in the shading (the two sides' source normals differ) stops in the middle of a panel, its end
    shows as a dark spike at some angles: the last two corners along it are blended toward one smooth normal."""
    from math import radians
    cos = np.cos(radians(angle))

    def hard(e):
        if len(e.link_faces) != 2:
            return False
        f0, f1 = e.link_faces
        if f0.material_index != f1.material_index or (mats is not None and f0.material_index not in mats):
            return False
        v = e.verts[0]
        l0 = next(l for l in f0.loops if l.vert == v); l1 = next(l for l in f1.loops if l.vert == v)
        return l0[sl].dot(l1[sl]) < cos

    ends = 0
    inside = lambda q: any(b[0] <= q[0] <= b[1] and b[2] <= q[1] <= b[3] and b[4] <= q[2] <= b[5] for b in skip)
    for v in bm.verts:
        if skip and inside(to_out([tuple(v.co)])[0]):
            continue
        hs = [e for e in v.link_edges if hard(e)]
        if len(hs) != 1:
            continue
        ends += 1
        w = hs[0].other_vert(v)
        for vert, k in ((v, 1.0), (w, 0.5)):
            loops = list(vert.link_loops)
            m = sum((l[sl] for l in loops), type(loops[0][sl])((0.0, 0.0, 0.0)))
            if m.length < 1e-9:
                continue
            m.normalize()
            for l in loops:
                n = l[sl] * (1 - k) + m * k
                l[sl] = n.normalized()
    return ends


def soften(ob, rules):
    """Smooth shading over a panel (REFINE_SOFTEN: (box, materials)): the source's split normals along a panel's soft
    folds (a bonnet's bulge) draw lines and specks once the panel is refined; inside the box, every corner of those
    materials takes its vertex's area-weighted normal over the box's faces of them. Written straight into the src_n
    attribute (no bmesh round trip: the mesh stays exactly as it is)."""
    me = ob.data
    names = [m.name if m else None for m in me.materials]
    nP, nL, nV = len(me.polygons), len(me.loops), len(me.vertices)
    fn = np.zeros((nP, 3)); me.polygons.foreach_get('normal', fn.ravel())
    fa = np.zeros(nP); me.polygons.foreach_get('area', fa)
    fc = np.zeros((nP, 3)); me.polygons.foreach_get('center', fc.ravel())
    fm = np.zeros(nP, np.int64); me.polygons.foreach_get('material_index', fm)
    ls = np.zeros(nP, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(nP, np.int64); me.polygons.foreach_get('loop_total', lt)
    lv = np.zeros(nL, np.int64); me.loops.foreach_get('vertex_index', lv)
    lp = np.repeat(np.arange(nP), lt)                      # each loop's polygon
    sn = np.zeros((nL, 3)); me.attributes['src_n'].data.foreach_get('vector', sn.ravel())
    C = to_out(fc)
    for box, materials in rules:
        mi = [names.index(m) for m in materials if m in names]
        sel = (np.isin(fm, mi) & (box[0] <= C[:, 0]) & (C[:, 0] <= box[1]) & (box[2] <= C[:, 1]) & (C[:, 1] <= box[3]) &
               (box[4] <= C[:, 2]) & (C[:, 2] <= box[5]))
        acc = np.zeros((nV, 3))
        # the region's border keeps the source's normals (a corner shared with a face outside: the nose's crest)
        edge = np.zeros(nV, bool); edge[lv[np.nonzero(np.isin(fm, mi)[lp] & ~sel[lp])[0]]] = True
        li = np.nonzero(sel[lp] & ~edge[lv])[0]
        la = np.nonzero(sel[lp])[0]
        np.add.at(acc, lv[la], fn[lp[la]] * fa[lp[la], None])
        n = acc[lv[li]]
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        sn[li[ok]] = n[ok] / ln[ok, None]
        log(f'  softened {sel.sum()} faces ({ok.sum()} corners)')
    me.attributes['src_n'].data.foreach_set('vector', sn.ravel())


def bevel(ob, cfg):
    """A rounded bevel on the hard edges between the materials cfg['materials'] names (edges sharper than cfg['angle']
    degrees): edges that catch the light instead of razor lines. 'boxes': only edges inside one of them (output
    metres); 'safe': only edges long enough for it, among long enough neighbours (a bevel wider than the edges around
    it folds over itself). On this kind of decimated mesh a bevel everywhere crumples the big panels, so it goes where
    it holds up. Shading normals hardened by hand into src_n: the faces the bevel leaves keep the source's, a bevel
    corner on a flat face takes that face's, and the middle of a bevel bends smoothly between them."""
    import bmesh
    from math import radians
    me = ob.data
    mats = {i for i, m in enumerate(me.materials) if m and m.name in cfg['materials']}
    bm = bmesh.new(); bm.from_mesh(me)
    bm.normal_update()
    boxes = cfg.get('boxes')
    w = cfg['width'] / SCALE
    safe = cfg.get('safe', False)
    minlen = lambda v, e: min((x.calc_length() for x in v.link_edges if x is not e), default=9.0)
    inside = lambda q, b: b[0] <= q[0] <= b[1] and b[2] <= q[1] <= b[3] and b[4] <= q[2] <= b[5]
    edges = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f0, f1 = e.link_faces
        # one material on both sides only: a bevel between two (paint and a black flare, a lamp) takes one side's
        # material and spreads it onto the other
        if f0.material_index in mats and f0.material_index == f1.material_index and f0.normal.angle(f1.normal, 0.0) > radians(cfg['angle']):
            qs = to_out([tuple(v.co) for v in e.verts])
            if boxes is not None and not any(all(inside(q, b) for q in qs) for b in boxes):
                continue
            if any(any(inside(q, b) for q in qs) for b in cfg.get('exclude', [])):
                continue
            if safe and (e.calc_length() < 3 * w or min(minlen(v, e) for v in e.verts) < 1.3 * w):
                continue
            edges.append(e)
    old = set(bm.verts)
    res = bmesh.ops.bevel(bm, geom=edges, offset=w, offset_type='OFFSET', profile_type='SUPERELLIPSE',
                          segments=cfg.get('segments', 2), profile=0.5, affect='EDGES', clamp_overlap=cfg.get('clamp', False),
                          loop_slide=True, miter_outer='SHARP', miter_inner='SHARP', vmesh_method='ADJ')
    new = set(res['faces'])
    slivers = [f for f in new if f.calc_area() < 1e-14]
    if slivers:
        bmesh.ops.delete(bm, geom=slivers, context='FACES_ONLY')
        new -= set(slivers)
    bm.normal_update()
    sl = bm.loops.layers.float_vector.get('src_n')
    # the corners the bevel added to the faces beside it: their own face's nearest original corner's normal
    for f in bm.faces:
        if f in new:
            continue
        keep = [l for l in f.loops if l.vert in old]
        if keep:
            for l in f.loops:
                if l.vert not in old:
                    l[sl] = min(keep, key=lambda k: (k.vert.co - l.vert.co).length)[sl].copy()
    for f in new:
        for l in f.loops:
            flat = [x for x in l.vert.link_loops if x.face not in new]
            if flat:
                n = sum((x[sl] for x in flat), type(f.normal)((0.0, 0.0, 0.0)))
            else:
                n = sum((g.normal * g.calc_area() for g in l.vert.link_faces if g in new), type(f.normal)((0.0, 0.0, 0.0)))
            l[sl] = n.normalized() if n.length > 1e-9 else f.normal
    # the shape: the bevel's borders hard, so a flat panel next to a bevel doesn't bend toward it
    for f in new:
        for e in f.edges:
            if any(g not in new for g in e.link_faces):
                e.smooth = False
    bm.to_mesh(me); bm.free()
    log(f'  bevel: {len(edges)} edges, {cfg["width"] * 1000:.0f} mm, {cfg.get("segments", 2)} segments, '
        f'{len(new)} faces, {len(slivers)} slivers out')


def joined(objs, name, group):
    """One welded world-space copy of the objects, with normals recomputed from its shape: smooth across edges
    flatter than SMOOTH_ANGLE, split at sharper ones (source normals can be flipped or smoothed over hard corners,
    and the tessellation turns any normal error into a dent). The profile's dropped pieces go and its bevel goes on
    first (REFINE_DROP, REFINE_BEVEL: per group)."""
    import bmesh
    from math import radians
    copies = []
    for o in objs:
        c = o.copy(); c.data = o.data.copy(); bpy.context.scene.collection.objects.link(c); copies.append(c)
    bpy.ops.object.select_all(action='DESELECT')
    for c in copies:
        c.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for c in copies:   # the source's own normals, kept for the shading (REFINE_NORMALS 'source')
        c.data.calc_normals_split()
        ln = np.zeros((len(c.data.loops), 3)); c.data.loops.foreach_get('normal', ln.ravel())
        c.data.attributes.new('src_n', 'FLOAT_VECTOR', 'CORNER').data.foreach_set('vector', ln.ravel())
    if len(copies) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active; ob.name = name
    me = ob.data
    if me.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD)
    n = close_tjunctions(bm)
    if n:
        log(f'  closed {n} T-junctions')
    if getattr(P, 'REFINE_MOVE', {}).get(group):
        move_verts(bm, P.REFINE_MOVE[group])
    bm.normal_update()
    for rule in getattr(P, 'REFINE_RECOLOR', {}).get(group, []):
        # (box, material, material[, max normal z]): faces of one material inside the box (output metres) take
        # another; with a fourth value, only those facing down at least that much (a fender's lip under its flare)
        box, frm, to = rule[:3]
        nzmax = rule[3] if len(rule) > 3 else None
        names = [m.name if m else None for m in me.materials]
        fi, ti = names.index(frm), names.index(to)
        k = 0
        for f in bm.faces:
            if f.material_index == fi:
                c = to_out([tuple(f.calc_center_median())])[0]
                if box[0] <= c[0] <= box[1] and box[2] <= c[1] <= box[3] and box[4] <= c[2] <= box[5]:
                    if nzmax is not None and (np.asarray(tuple(f.normal)) @ ROT.T)[2] > nzmax:
                        continue
                    f.material_index = ti; k += 1
        log(f'  {k} {frm} faces -> {to}')
    if getattr(P, 'REFINE_TAPER_CREASES', {}).get(group):
        sl = bm.loops.layers.float_vector.get('src_n')
        names = P.REFINE_TAPER_CREASES[group]
        mats = {i for i, m in enumerate(me.materials) if m and m.name in names}
        n_t = taper_crease_ends(bm, sl, mats, skip=getattr(P, 'REFINE_FLAT', {}).get(group, ()))
        log(f'  tapered {n_t} crease ends')
    boxes = None if os.environ.get('REFINE_NO_DROP') else getattr(P, 'REFINE_DROP', {}).get(group)
    if boxes:
        bm.verts.ensure_lookup_table(); bm.verts.index_update()
        log(f'  dropped {drop_pieces(bm, boxes, [m.name if m else None for m in me.materials])} faces in {len(boxes)} boxes')
    bm.to_mesh(me); bm.free()
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    cfg = None if os.environ.get('REFINE_NO_BEVEL') else getattr(P, 'REFINE_BEVEL', {}).get(group)
    for c in ([cfg] if isinstance(cfg, dict) else cfg or []):     # one pass, or several (each its own width)
        bevel(ob, c)
    if getattr(P, 'REFINE_SOFTEN', {}).get(group):
        soften(ob, P.REFINE_SOFTEN[group])
    if me.has_custom_normals:
        bpy.ops.object.select_all(action='DESELECT'); ob.select_set(True); bpy.context.view_layer.objects.active = ob
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    me.use_auto_smooth = True
    me.auto_smooth_angle = radians(SMOOTH_ANGLE)
    return ob


def gather(objs):
    """World-space triangles of the objects: positions (T,3,3), split normals (T,3,3), the source's normals (T,3,3),
    uvs (T,3,2), materials (T,)."""
    Ps, Ns, Ss, Us, Ms, mats = [], [], [], [], [], []
    for o in objs:
        me = o.data
        me.calc_normals_split()
        me.calc_loop_triangles()
        T = len(me.loop_triangles)
        if T == 0:
            continue
        lt = np.zeros((T, 3), np.int64); me.loop_triangles.foreach_get('loops', lt.ravel())
        lv = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
        co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
        ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
        uv = np.zeros((len(me.loops), 2))
        if me.uv_layers.active:
            me.uv_layers.active.data.foreach_get('uv', uv.ravel())
        sn = ln.copy()
        if 'src_n' in me.attributes:
            me.attributes['src_n'].data.foreach_get('vector', sn.ravel())
        mi = np.zeros(T, np.int64); me.loop_triangles.foreach_get('material_index', mi)
        mw = np.array(o.matrix_world)
        nm = np.array(o.matrix_world.to_3x3().inverted_safe().transposed())
        co = co @ mw[:3, :3].T + mw[:3, 3]
        ln = ln @ nm.T
        ln /= np.maximum(np.linalg.norm(ln, axis=1, keepdims=True), 1e-12)
        sn = sn @ nm.T
        sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-12)
        base = len(mats)
        mats += [me.materials[i] if i < len(me.materials) else None for i in range(max(len(me.materials), 1))]
        Ps.append(co[lv[lt]]); Ns.append(ln[lt]); Ss.append(sn[lt]); Us.append(uv[lt][..., :2]); Ms.append(mi + base)
    cat = np.concatenate
    return cat(Ps), cat(Ns), cat(Ss), cat(Us), cat(Ms), mats


def tessellate(P, N, S, U, k, alpha, flat=()):
    """Phong tessellation of triangles (T,3,3) into k*k each, the shape from the normals N. Returns vertex positions,
    triangles (vertex ids, welded along shared edges) and their per-corner normals N and S (interpolated) and uvs."""
    T = len(P)
    grid = [(i, j) for i in range(k + 1) for j in range(k + 1 - i)]                 # barycentric (k-i-j, i, j) / k
    gidx = {g: n for n, g in enumerate(grid)}
    Wi = np.array([(k - i - j, i, j) for i, j in grid], np.int64)
    W = Wi / k
    G = len(grid)
    sub = []
    for i in range(k):
        for j in range(k - i):
            sub.append((gidx[(i, j)], gidx[(i + 1, j)], gidx[(i, j + 1)]))
            if i + j < k - 1:
                sub.append((gidx[(i + 1, j)], gidx[(i + 1, j + 1)], gidx[(i, j + 1)]))
    sub = np.array(sub)

    # p*(w) = sum_i sum_j w_i w_j pi_i(p_j), pi_i(q) = q - ((q - p_i).n_i) n_i
    pi = np.empty((T, 3, 3, 3))                                                      # [t, i, j] = pi_i(p_j)
    for a in range(3):
        d = P - P[:, a:a + 1]
        pi[:, a] = P - np.einsum('tjc,tc->tj', d, N[:, a])[..., None] * N[:, a][:, None]
    lin = np.einsum('gi,tic->tgc', W, P)
    cur = np.einsum('gi,gj,tijc->tgc', W, W, pi)
    pos = (1 - alpha) * lin + alpha * cur                                            # (T,G,3)
    # a piece's open edge stays straight: two pieces that only touch there (a bonnet's edge on the grille surround)
    # would each bend theirs their own way and open a gap
    q0 = np.round(P.reshape(-1, 3) / WELD).astype(np.int64)
    _, pid0 = np.unique(q0, axis=0, return_inverse=True)
    pid0 = pid0.reshape(T, 3)
    ekeys = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ekeys.append(np.minimum(pid0[:, a], pid0[:, b]) * (len(q0) + 1) + np.maximum(pid0[:, a], pid0[:, b]))
    ek, ecount = np.unique(np.concatenate(ekeys), return_counts=True)
    open_edge = dict(zip(ek.tolist(), (ecount == 1).tolist()))
    for g in range(len(Wi) if not os.environ.get('REFINE_CURVED_OPEN') else 0):
        c = np.nonzero(Wi[g])[0]
        if len(c) != 2:
            continue
        a, b = c
        key = np.minimum(pid0[:, a], pid0[:, b]) * (len(q0) + 1) + np.maximum(pid0[:, a], pid0[:, b])
        m = np.array([open_edge[x] for x in key.tolist()])
        pos[m, g] = lin[m, g]
    if len(flat):   # (REFINE_FLAT) triangles kept as the source has them: a corner the curve would bulge
        C = to_out(P.mean(1))
        m = np.zeros(T, bool)
        for b in flat:
            m |= (b[0] <= C[:, 0]) & (C[:, 0] <= b[1]) & (b[2] <= C[:, 1]) & (C[:, 1] <= b[3]) & (b[4] <= C[:, 2]) & (C[:, 2] <= b[5])
        pos[m] = lin[m]
    nrm = np.einsum('gi,tic->tgc', W, N)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    snrm = np.einsum('gi,tic->tgc', W, S)
    snrm /= np.maximum(np.linalg.norm(snrm, axis=-1, keepdims=True), 1e-12)
    uvs = np.einsum('gi,tic->tgc', W, U)

    # weld: corners by position, edge points by (edge, step along it), interior points stay their own
    q = np.round(P.reshape(-1, 3) / WELD).astype(np.int64)
    _, pid = np.unique(q, axis=0, return_inverse=True)
    pid = pid.reshape(T, 3)
    nz = Wi > 0
    cnt = nz.sum(1)
    keys = np.full((T, G, 4), -1, np.int64)
    for g in range(G):
        c = np.nonzero(nz[g])[0]
        if cnt[g] == 1:
            keys[:, g, 0] = 0; keys[:, g, 1] = pid[:, c[0]]
        elif cnt[g] == 2:
            a, b = c
            ia, ib = pid[:, a], pid[:, b]
            swap = ia > ib
            keys[:, g, 0] = 1
            keys[:, g, 1] = np.where(swap, ib, ia)
            keys[:, g, 2] = np.where(swap, ia, ib)
            keys[:, g, 3] = np.where(swap, Wi[g, b], Wi[g, a])
        else:
            keys[:, g, 0] = 2; keys[:, g, 1] = np.arange(T); keys[:, g, 2] = g
    flat = keys.reshape(-1, 4)
    _, vid, inv_cnt = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    V = np.zeros((len(inv_cnt), 3))
    np.add.at(V, vid, pos.reshape(-1, 3))
    V /= inv_cnt[:, None]
    vid = vid.reshape(T, G)
    tri = vid[:, sub].reshape(-1, 3)
    tri_n = nrm[:, sub].reshape(-1, 3, 3)
    tri_s = snrm[:, sub].reshape(-1, 3, 3)
    tri_uv = uvs[:, sub].reshape(-1, 3, 2)
    return V, tri, tri_n, tri_s, tri_uv


def fair(V, tri, tri_n, fm, iters):
    """Taubin smoothing (lambda/mu, no shrinking) of the tessellated surface along its normals only, so the ripples a
    fan of long thin source triangles leaves are smoothed out and nothing slides sideways. The vertices on hard edges
    (the two sides' normals differ), on material borders and on open borders stay where they are."""
    if iters <= 0:
        return V
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    en = np.concatenate([tri_n[:, [0, 1]], tri_n[:, [1, 2]], tri_n[:, [2, 0]]])        # (E,2,3) loop normals at the ends
    em = np.tile(fm, 3)
    swap = e[:, 0] > e[:, 1]
    es = np.where(swap[:, None], e[:, ::-1], e)
    ens = np.where(swap[:, None, None], en[:, ::-1], en)
    order = np.lexsort((es[:, 1], es[:, 0]))
    es, ens, em = es[order], ens[order], em[order]
    _, start, count = np.unique(es, axis=0, return_index=True, return_counts=True)
    fixed = np.zeros(len(V), bool)
    fixed[es[start[count != 2]].ravel()] = True                                       # open or non-manifold edges
    two = start[count == 2]
    hard = (np.abs(ens[two] - ens[two + 1]).max(axis=(1, 2)) > 1e-3) | (em[two] != em[two + 1])
    fixed[es[two[hard]].ravel()] = True
    ue = es[start]
    def nb_sum(X):   # per vertex, the sum of its neighbours' positions
        return np.stack([np.bincount(ue[:, 0], X[ue[:, 1], c], len(V)) + np.bincount(ue[:, 1], X[ue[:, 0], c], len(V))
                         for c in range(3)], 1)
    deg = np.bincount(ue[:, 0], minlength=len(V)) + np.bincount(ue[:, 1], minlength=len(V))
    deg = np.maximum(deg, 1)[:, None]
    fn = np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]])           # area-weighted
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, tri[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    free = ~fixed[:, None]
    X = V.copy()
    for _ in range(iters):
        for f in (0.5, -0.53):
            L = nb_sum(X) / deg - X
            X = X + f * free * np.einsum('ij,ij->i', L, vn)[:, None] * vn
    # ripples are millimetres to a centimetre or two: a vertex that would move further sits among odd neighbours
    # (a sliver next to huge triangles), so it moves at most FAIR_MAX
    D = X - V
    d = np.linalg.norm(D, axis=1) * SCALE
    cap = FAIR_MAX / np.maximum(d, 1e-12)
    X = V + D * np.minimum(cap, 1.0)[:, None]
    log(f'  fair: {iters} passes, {fixed.sum()} of {len(V)} vertices held, {(d > FAIR_MAX).sum()} capped at '
        f'{FAIR_MAX * 1000:.0f} mm (the furthest wanted {d.max() * 1000:.0f} mm)')
    return X


def shape_normals(V, tri, angle):
    """Normals from the final shape, for the whole group at once (continuous across material borders): smooth across
    edges flatter than angle, split at sharper ones."""
    me = bpy.data.meshes.new('normals')
    me.from_pydata(V.tolist(), [], tri.tolist())
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.use_auto_smooth = True
    me.auto_smooth_angle = np.radians(angle)
    me.calc_normals_split()
    ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
    bpy.data.meshes.remove(me)
    return ln.reshape(-1, 3, 3)


def nearest_in_front(trees, o, near, p, d, dist):
    """The nearest surface within dist in front of point p along d, over all the parts (o's faces in near skipped:
    across a crease a ray skims into the neighbours): (distance, part, face, facing . d) or None."""
    best = None
    for oo, t in trees:
        org, gone_by = p + d * 1e-5, 0.0
        for _ in range(4):
            h = t.ray_cast(org, d, dist - gone_by)
            if h[0] is None:
                break
            if not (oo is o and h[2] in near):
                if best is None or gone_by + h[3] < best[0]:
                    best = (gone_by + h[3], oo.name, h[2], h[1].dot(d))
                break
            gone_by += h[3] + 1e-6; org = h[0] + d * 1e-6
            if gone_by >= dist:
                break
    return best


def cull_buried(gname, dist, parts, sink=0.0008, sink_parts=('body',)):
    """Faces with another surface right in front of them (within dist along their own normal) out. The refining
    leaves a fold's sliver lying a fraction of a millimetre under the panel it folded from, or a paint lip just
    under a black part: the game draws both, they fight for the pixels (jagged black streaks on the paint, paint
    hairlines on the black), and the buried one bakes black too. Hidden either way, so gone; but only while what
    covers it stays (two faces over each other both going would leave a hole)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    skip = ('_lines', '_wipers', '_blackout', '_chassis')
    objs = [o for o in parts if o.type == 'MESH' and o.name.startswith(gname + '_') and not o.name.endswith(skip)]
    dg = bpy.context.evaluated_depsgraph_get()
    trees = [(o, BVHTree.FromObject(o, dg)) for o in objs]
    data, cover, partial = {}, {}, {}          # cover[(obj, face)] = (obj, face) right in front of it
    for o in objs:
        me = o.data
        me.calc_normals_split()
        ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
        ls = np.zeros(len(me.polygons), np.int64); me.polygons.foreach_get('loop_start', ls)
        lt = np.zeros(len(me.polygons), np.int64); me.polygons.foreach_get('loop_total', lt)
        data[o.name] = (ln, ls, lt)
        if o.name.endswith('_back'):
            continue
        # along the face's own (geometric) normal; its neighbours (sharing a corner) don't count: across a crease a
        # ray skims into them
        n = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('normal', n.ravel())
        c = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('center', c.ravel())
        fv = [set(p.vertices) for p in me.polygons]
        vf = {}
        for i, vs in enumerate(fv):
            for v in vs:
                vf.setdefault(v, []).append(i)
        co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
        for i in range(len(me.polygons)):
            d = Vector(n[i].tolist())
            near = {j for v in fv[i] for j in vf[v]}
            # covered all over: the centre and a point near each corner (a face only partly under another stays)
            pts = [c[i]] + [c[i] * 0.15 + co[v] * 0.85 for v in me.polygons[i].vertices]
            hits = [nearest_in_front(trees, o, near, Vector(pt.tolist()), d, dist) for pt in pts]
            ok = [h is not None and h[3] > 0.2 for h in hits]
            if all(ok):
                cover[(o.name, i)] = (hits[0][1], hits[0][2])
            elif any(ok):
                partial.setdefault(o.name, set()).add(i)
    # partly covered: sunk sink metres back (the top skin wins the pixels cleanly where they overlap; the rest of it
    # still shows, a hair deeper)
    n_sunk = 0
    for o in objs:
        fs = partial.get(o.name)
        # the paint only, and less than the back skins' gap (sunk onto a back skin, the exporter turned faces round)
        if not fs or o.name[len(gname) + 1:] not in sink_parts:
            continue
        me = o.data
        co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
        fn = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('normal', fn.ravel())
        mv = np.zeros_like(co); cnt = np.zeros(len(co))
        for i in fs:
            for v in me.polygons[i].vertices:
                mv[v] += fn[i]; cnt[v] += 1
        k = cnt > 0
        mv[k] /= np.maximum(np.linalg.norm(mv[k], axis=1, keepdims=True), 1e-12)
        co[k] -= mv[k] * sink
        me.vertices.foreach_set('co', co.ravel()); me.update()
        n_sunk += len(fs)
    log(f'  {n_sunk} faces partly under another sunk {sink * 1000:.1f} mm')
    gone = set(cover)
    while True:                                 # a face whose cover goes too stays
        back = {k for k in gone if cover[k] in gone}
        if not back:
            break
        # of each pair over each other, keep the outer: the one covering the other first in the list stays
        keep = set()
        for k in sorted(back):
            if k not in keep and cover[k] not in keep:
                keep.add(k)
        gone -= keep
    total = 0
    for o in objs:
        idx = sorted(i for (nm, i) in gone if nm == o.name)
        if not idx:
            continue
        me = o.data
        ln, ls, lt = data[o.name]
        keep = np.ones(len(me.polygons), bool); keep[idx] = False
        kept = np.nonzero(keep)[0]
        # out through a new mesh, the custom normals of the rest kept as they were
        fv = [list(p.vertices) for p in me.polygons]
        uv = np.zeros((len(me.loops), 2)); me.uv_layers[0].data.foreach_get('uv', uv.ravel())
        co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
        sk = {tuple(sorted(e.vertices)) for e in me.edges if e.use_edge_sharp}
        loops = np.concatenate([np.arange(ls[i], ls[i] + lt[i]) for i in kept])
        new = bpy.data.meshes.new(me.name)
        new.from_pydata(co.tolist(), [], [fv[i] for i in kept])
        new.uv_layers.new(name=me.uv_layers[0].name)
        new.uv_layers[0].data.foreach_set('uv', uv[loops].ravel())
        new.polygons.foreach_set('use_smooth', [True] * len(new.polygons))
        new.use_auto_smooth = True
        new.edges.foreach_set('use_edge_sharp', [tuple(sorted(e.vertices)) in sk for e in new.edges])
        new.normals_split_custom_set(ln[loops].tolist())
        for m in me.materials:
            new.materials.append(m)
        o.data = new
        bpy.data.meshes.remove(me)
        total += len(idx)
        log(f'  {o.name}: {len(idx)} buried faces out')
    log(f'{gname}: {total} buried faces out ({len(cover) - len(gone)} kept: their cover was buried too)')


def edge_on(V, tri, tri_n, lim=0.3):
    """Triangles seen nearly edge-on to their own normals (|face . normal| < lim): slivers along creases and seams."""
    P = V[tri]
    g = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(g, axis=1)
    g = g / np.maximum(area, 1e-30)[:, None]
    m = tri_n.sum(1); m /= np.maximum(np.linalg.norm(m, axis=1), 1e-12)[:, None]
    return (np.abs((g * m).sum(1)) < lim) & (area > 1e-14)


def boundary_slivers(V, tri, tri_n, fm):
    """An edge-on sliver on the border of two materials (a paint lip folded into a black bumper's pad) takes the
    material its corners share most with the other side: it showed as a hairline of the wrong colour."""
    e = edge_on(V, tri, tri_n)
    mats = np.unique(fm)
    if len(mats) < 2 or not e.any():
        return fm
    cnt = np.zeros((len(tri), len(mats)), np.int64)
    for j, m in enumerate(mats):
        at = np.zeros(len(V), bool); at[tri[(fm == m) & ~e].ravel()] = True
        cnt[:, j] = at[tri].sum(1)
    own = np.searchsorted(mats, fm)
    other = cnt.copy(); other[np.arange(len(tri)), own] = -1
    best = other.argmax(1)
    take = e & (other[np.arange(len(tri)), best] > 0) & (cnt[np.arange(len(tri)), own] <= other[np.arange(len(tri)), best])
    log(f'  {take.sum()} border slivers take the neighbouring material')
    return np.where(take, mats[best], fm)


def fix_folds(name, V, tri, tri_n, tri_uv, tol=0.05):
    """Triangles facing away from their own normals: the refining folds a sliver over now and then (where a crease
    runs out, round a tight corner) and the game culls it, a hole showing the dark inside (black streaks, slivers);
    or the triangle sits right but its corners took normals from the far side of a fold (it shades black). A folded
    one (facing away from its neighbourhood too) is turned round; the others take the neighbourhood's normal."""
    P = V[tri]
    g = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(g, axis=1)
    g = g / np.maximum(area, 1e-30)[:, None]
    nn = tri_n / np.maximum(np.linalg.norm(tri_n, axis=-1, keepdims=True), 1e-12)
    bad = (np.einsum('ijk,ik->ij', nn, g).min(1) < tol) & (area > 1e-14)
    if not bad.any():
        return tri, tri_n, tri_uv
    acc = np.zeros_like(V)
    np.add.at(acc, tri.ravel(), np.repeat(g * area[:, None], 3, 0))
    nb = acc[tri].sum(1)
    nb /= np.maximum(np.linalg.norm(nb, axis=1), 1e-30)[:, None]
    m = nn.sum(1); m /= np.maximum(np.linalg.norm(m, axis=1), 1e-12)[:, None]
    gm = (g * m).sum(1)                  # edge-on slivers (|gm| small) are left as they are: either way is a guess
    fold = bad & (gm < -0.3) & ((m * nb).sum(1) > 0)
    tri, tri_n, tri_uv = tri.copy(), tri_n.copy(), tri_uv.copy()
    tri[fold] = tri[fold][:, ::-1]
    tri_uv[fold] = tri_uv[fold][:, ::-1]
    tri_n[fold] = tri_n[fold][:, ::-1]
    rest = bad & ~fold & (np.abs(gm) >= 0.3)
    use = np.where(((nb * g).sum(1) > 0.2)[:, None], nb, g)
    tri_n[rest] = np.repeat(use[rest][:, None, :], 3, 1)
    log(f'  {name}: {fold.sum()} folded triangles turned round, {rest.sum()} given their neighbourhood normal')
    return tri, tri_n, tri_uv


def build(name, V, tri, tri_n, tri_uv, mat):
    # a triangle whose corners welded into one vertex is no face: Blender keeps it, and the custom normals of every
    # face after it in the mesh come out scrambled (glossy blotches over a whole part)
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    if not ok.all():
        log(f'  {name}: {(~ok).sum()} collapsed triangles out')
        tri, tri_n, tri_uv = tri[ok], tri_n[ok], tri_uv[ok]
    tri, tri_n, tri_uv = fix_folds(name, V, tri, tri_n, tri_uv)
    used, local = np.unique(tri, return_inverse=True)
    me = bpy.data.meshes.new(name)
    me.from_pydata((V[used] @ ROT.T * SCALE).tolist(), [], local.reshape(-1, 3).tolist())
    if len(me.polygons) != len(tri) or len(me.loops) != 3 * len(tri):
        log(f'  {name}: {len(tri)} triangles in, {len(me.polygons)} faces / {len(me.loops)} corners out')
    me.uv_layers.new(name='UVMap')
    me.uv_layers[0].data.foreach_set('uv', tri_uv.reshape(-1))
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.use_auto_smooth = True
    # Blender keeps one custom normal per smooth fan (the corners round a vertex joined by smooth edges), averaging
    # what it's given: where the two sides of an edge want normals far apart (a fold, a crease, a back-to-back skin)
    # the edge is marked sharp, so each side keeps its own
    t3 = local.reshape(-1, 3)
    nn = tri_n / np.maximum(np.linalg.norm(tri_n, axis=-1, keepdims=True), 1e-12)
    side = []
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        lo, hi = np.minimum(t3[:, a_], t3[:, b_]), np.maximum(t3[:, a_], t3[:, b_])
        na = np.where((t3[:, a_] < t3[:, b_])[:, None], nn[:, a_], nn[:, b_])
        nb = np.where((t3[:, a_] < t3[:, b_])[:, None], nn[:, b_], nn[:, a_])
        side.append((lo, hi, na, nb))
    lo = np.concatenate([x[0] for x in side]); hi = np.concatenate([x[1] for x in side])
    na = np.concatenate([x[2] for x in side]); nb = np.concatenate([x[3] for x in side])
    key = lo * (len(used) + 1) + hi
    order = np.argsort(key, kind='stable')
    key, na, nb = key[order], na[order], nb[order]
    same = key[1:] == key[:-1]
    differ = same & (((na[1:] * na[:-1]).sum(1) < 0.9) | ((nb[1:] * nb[:-1]).sum(1) < 0.9))
    sharp_keys = set(key[1:][differ].tolist())
    if sharp_keys:
        ek = np.array([k[0] * (len(used) + 1) + k[1] for k in me.edge_keys])
        sharp = np.isin(ek, np.fromiter(sharp_keys, np.int64))
        me.edges.foreach_set('use_edge_sharp', sharp.tolist())
    me.normals_split_custom_set((tri_n.reshape(-1, 3) @ ROT.T).tolist())
    if os.environ.get('REFINE_DUMP'):
        me.calc_normals_split()
        got = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', got.ravel())
        want = tri_n.reshape(-1, 3) @ ROT.T
        want /= np.maximum(np.linalg.norm(want, axis=1, keepdims=True), 1e-12)
        log(f'  {name}: custom normals off by >60 deg at {((got * want).sum(1) < 0.5).sum()} corners')
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def back_faces(V, tri, tri_n, tri_uv, gap=0.0015):
    """Back sides for one-sided faces (seen from behind the game draws nothing: an arch's inner panel, the underbody):
    the same triangles turned round, their vertices moved gap metres behind the front."""
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, tri[:, c], tri_n[:, c])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    used = np.unique(tri)
    V2 = V.copy()
    V2[used] -= vn[used] * gap / SCALE
    return V2, tri[:, ::-1], -tri_n[:, ::-1], tri_uv[:, ::-1]


def blackout(cfg):
    """A black sheet a little behind every window, facing out: the cab has no inside, so through the glass the game
    showed whatever lay behind (the windscreen read grey); now all windows read dark like the stock cars'."""
    glass = bpy.data.objects.get('body_' + cfg['glass'])
    me = glass.data
    me.calc_normals_split(); me.calc_loop_triangles()
    co = np.array([tuple(v.co) for v in me.vertices])
    centre = np.array([0.0, 0.0, 1.0])
    vs, fs = [], []
    for t in me.loop_triangles:
        n = np.array(tuple(t.normal))
        c = np.array(tuple(t.center))
        if n @ (c - centre) <= 0:          # an inner face of the glass: the sheet goes behind the outer one only
            continue
        b = len(vs)
        for vi in t.vertices:
            vs.append(co[vi] - n * cfg.get('depth', 0.03))
        fs.append((b, b + 1, b + 2))
    bm_me = bpy.data.meshes.new('body_blackout')
    bm_me.from_pydata(vs, [], fs)
    bm_me.materials.append(bpy.data.materials[cfg['material']])
    ob = bpy.data.objects.new('body_blackout', bm_me)
    bpy.context.scene.collection.objects.link(ob)
    log(f'blackout: {len(fs)} faces behind the windows')
    return [ob]


class Solid:
    """A small bmesh kit for the made-up parts (liners, frame, wipers): every face wound to face the given way."""

    def __init__(self):
        import bmesh
        self.bm = bmesh.new()

    def face(self, pts, out_dir):
        vs = [self.bm.verts.new(tuple(p)) for p in pts]
        f = self.bm.faces.new(vs)
        f.normal_update()
        if np.dot(np.array(tuple(f.normal)), out_dir) < 0:
            f.normal_flip()
        return f

    def box(self, c, ax, ay, az, hx, hy, hz):
        c, ax, ay, az = (np.asarray(x, float) for x in (c, ax, ay, az))
        P = lambda sx, sy, sz: c + sx * hx * ax + sy * hy * ay + sz * hz * az
        for axis, (u, v, w) in ((ax, (1, 0, 0)), (ay, (0, 1, 0)), (az, (0, 0, 1))):
            for sgn in (-1, 1):
                if u:
                    q = [P(sgn, a, b) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                elif v:
                    q = [P(a, sgn, b) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                else:
                    q = [P(a, b, sgn) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                self.face(q, sgn * axis)

    def prism(self, c, axis, r, h, sides=8, ref=(1.0, 0.0, 0.0)):
        """An n-sided prism standing on c along axis (a pivot cap, a nut)."""
        c, axis = np.asarray(c, float), np.asarray(axis, float)
        a = np.asarray(ref, float); a = a - axis * (a @ axis); a /= np.linalg.norm(a); b = np.cross(axis, a)
        ring = [c + r * (np.cos(t) * a + np.sin(t) * b) for t in np.linspace(0, 2 * np.pi, sides, endpoint=False)]
        top = [p + axis * h for p in ring]
        self.face(ring, -axis); self.face(top, axis)
        for i in range(sides):
            j = (i + 1) % sides
            mid = (ring[i] + ring[j]) / 2 - c
            self.face([ring[i], ring[j], top[j], top[i]], mid)

    def sweep(self, sections, caps=True):
        """Quads between consecutive 4-point sections (a frame rail), facing away from the sections' middles."""
        for s0, s1 in zip(sections, sections[1:]):
            m = (np.mean(s0, 0) + np.mean(s1, 0)) / 2
            for j in range(4):
                k = (j + 1) % 4
                q = [s0[j], s0[k], s1[k], s1[j]]
                self.face(q, np.mean(q, 0) - m)
        if caps:
            d = np.mean(sections[-1], 0) - np.mean(sections[0], 0)
            self.face(sections[0], -d); self.face(sections[-1], d)

    def object(self, name, material):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me); self.bm.free()
        me.materials.append(bpy.data.materials[material])
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def chassis(cfg, hubs):
    """What a real truck shows through its wheel arches and under it, the model having nothing there:
    a wheel-well liner over each wheel and a ladder frame (two rails kicking up over the axles, crossmembers).
    A liner follows its arch's opening a little way in (so none of it shows below or beside the fender), from the
    body's side in to liner_inner (liner_inner_rear behind: inside the bed's wheel humps, under liner_top_rear),
    a rounded shell turning down into an inner wall, both sides drawn."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    parts = [bpy.data.objects.get('body_' + m) for m in (cfg['paint'], 'plastic', 'dark')]
    parts = [o for o in parts if o]
    V, F = [], []
    for o in parts:
        k = len(V)
        V += [tuple(v.co) for v in o.data.vertices]
        F += [[k + i for i in poly.vertices] for poly in o.data.polygons]
    tree = BVHTree.FromPolygons(V, F)
    pc = np.array(V)
    S = Solid()
    gap, chamf, seg = cfg.get('liner_gap', 0.03), cfg.get('liner_chamfer', 0.045), 36
    ang = np.radians(np.linspace(-cfg.get('liner_sweep', 80), cfg.get('liner_sweep', 80), seg + 1))
    for corner, h in hubs.items():
        side = 1.0 if h[0] > 0 else -1.0
        rear = h[1] > 0
        cy, cz = h[1], h[2]
        xin = cfg.get('liner_inner_rear', cfg['liner_inner']) if rear else cfg['liner_inner']
        top = cfg.get('liner_top_rear') if rear else None
        # the opening: from the hub out to the fender's edge per angle, just outside the body's side (the flares)
        ropen = []
        for t in ang:
            dv = Vector((0.0, np.sin(t), np.cos(t)))
            hs = [tree.ray_cast(Vector((side * x, cy, cz)), dv, 1.0) for x in cfg.get('liner_probe', (0.92, 0.98, 1.03))]
            ds = [hh[3] for hh in hs if hh[0] is not None]
            ropen.append(min(ds) if ds else np.nan)
        ropen = np.array(ropen)
        ropen = np.where(np.isnan(ropen), np.nanmedian(ropen), ropen)
        ropen = np.array([np.median(ropen[max(0, i - 2):i + 3]) for i in range(len(ropen))])
        r = ropen + gap
        if top is not None:
            r = np.minimum(r, np.where(np.cos(ang) > 0.05, (top - cz) / np.maximum(np.cos(ang), 0.05), 9.0))
        # how far out: just inside the body's side at each point (cast in from outside along x)
        xo = []
        for t, ri in zip(ang, r):
            y, z = cy + ri * np.sin(t), cz + ri * np.cos(t)
            hit = tree.ray_cast(Vector((side * 1.5, y, z)), Vector((-side, 0.0, 0.0)), 1.5 - xin)
            xo.append(abs(hit[0].x) - 0.02 if hit[0] is not None else np.nan)
        xo = np.array(xo)
        xo = np.where(np.isnan(xo), np.nanmin(xo) if np.isfinite(xo).any() else 0.85, xo)
        xo = np.array([xo[max(0, i - 1):i + 2].min() for i in range(len(xo))])
        log(f'  liner {corner}: radius {r.min():.3f}-{r.max():.3f} (top {r[len(r) // 2]:.3f}), out to {xo.min():.3f}-{xo.max():.3f}')
        P = lambda x, ri, t: np.array([side * x, cy + ri * np.sin(t), cz + ri * np.cos(t)])
        out = [P(xo[i], r[i], t) for i, t in enumerate(ang)]
        mid = [P(xin + chamf, r[i], t) for i, t in enumerate(ang)]
        low = [P(xin, r[i] - chamf, t) for i, t in enumerate(ang)]
        for skin in (0.0, 0.008):          # the inner skin faces the wheel, the outer the body
            sgn = -1.0 if skin == 0.0 else 1.0
            o = lambda p, t: p + skin * np.array([0.0, np.sin(t), np.cos(t)])
            for i in range(seg):
                t0, t1 = ang[i], ang[i + 1]
                radial = np.array([0.0, np.sin((t0 + t1) / 2), np.cos((t0 + t1) / 2)])
                S.face([o(mid[i], t0), o(mid[i + 1], t1), o(out[i + 1], t1), o(out[i], t0)], sgn * radial)
                cn = radial - np.array([side, 0.0, 0.0])
                S.face([o(low[i], t0), o(low[i + 1], t1), o(mid[i + 1], t1), o(mid[i], t0)], sgn * cn / np.linalg.norm(cn))
            wall = [p + np.array([-side * skin, 0.0, 0.0]) for p in low]
            S.face(wall, np.array([side * (1.0 if skin == 0.0 else -1.0), 0.0, 0.0]))
    # the frame: rails at +-rail_x, kicking up over the axles (the donor's axles travel under them)
    ys = sorted(h[1] for h in hubs.values())
    fy, ry = ys[0], ys[-1]
    rx, rw, rh = cfg['rail_x'], cfg.get('rail_w', 0.07), cfg.get('rail_h', 0.15)
    low, high = cfg['rail_low'], cfg['rail_high']
    prof = [(cfg['rail_front'], low), (fy - 0.45, low), (fy - 0.3, high), (fy + 0.3, high), (fy + 0.45, low),
            (ry - 0.45, low), (ry - 0.3, high), (ry + 0.3, high), (ry + 0.45, low), (cfg['rail_rear'], low)]
    for sx in (-1, 1):
        secs = []
        for y, zb in prof:
            x0, x1 = sx * rx - rw / 2, sx * rx + rw / 2
            secs.append([np.array(p) for p in ((x0, y, zb), (x1, y, zb), (x1, y, zb + rh), (x0, y, zb + rh))])
        S.sweep(secs)
    for y in cfg.get('crossmembers', []):
        zb = low + 0.02
        S.box((0.0, y, zb + 0.05), (1, 0, 0), (0, 1, 0), (0, 0, 1), rx, 0.035, 0.045)
    ob = S.object('body_chassis', cfg['material'])
    log(f'chassis: liners at {sorted(hubs)}, frame rails at +-{rx} m, {len(cfg.get("crossmembers", []))} crossmembers')
    return [ob]


def stock_wheels(hubs):
    """New wheels in place of the source's (REFINE_WHEELS: rim style and tyre size), from offroad_wheel.py, at the
    source tyres' hubs: Wheel_<corner> and Tire_<corner> objects like the source's."""
    import offroad_wheel as OW
    from mathutils import Vector
    made = []
    rim = {'split6': OW.make_split6, 'beadlock': OW.make_rim}[WHEELS['rim']]
    d_in, w_in = WHEELS['tyre']
    for corner, c in hubs.items():
        c = Vector(tuple(c))
        if c.x > 0:      # the converters' Unity x is -Blender x: Blender +x is the left side
            place = lambda v, c=c: c + Vector((v.x, -v.z, v.y))
        else:
            place = lambda v, c=c: c + Vector((-v.x, v.z, v.y))
        made += [rim(f'Wheel_{corner}', place), OW.make_tyre(f'Tire_{corner}', place, d_in, w_in)]
    log(f'stock wheels: {WHEELS["rim"]} rim, {d_in} x {w_in} in tyres at {sorted(hubs)}')
    return made


def trim_lines(cfg):
    """Thin trim strips laid on a body part (REFINE_LINES: on, material, width, lines = polylines of output (x, y)):
    each polyline sampled every step (5 mm), dropped onto the part from above, lifted off it by lift (1.5 mm)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    on = bpy.data.objects.get('body_' + cfg['on'])
    tree = BVHTree.FromPolygons([tuple(v.co) for v in on.data.vertices], [list(f.vertices) for f in on.data.polygons])
    S = Solid()
    w, lift, n_pts = cfg.get('width', 0.007) / 2, cfg.get('lift', 0.0015), 0
    for line in cfg['lines']:
        # points (x, y): dropped onto the part from above; (x, y, z): moved to the part's nearest point (a line
        # running down a steep face, the bonnet's seam turning down to the headlamps)
        line = np.asarray(line, float)
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        ts = np.linspace(0.0, t[-1], max(2, int(t[-1] / cfg.get('step', 0.005)) + 1))
        pts = np.stack([np.interp(ts, t, line[:, c]) for c in range(line.shape[1])], 1)
        P, N = [], []
        for p in pts:
            if len(p) == 3:
                h = tree.find_nearest(Vector(tuple(p)), 0.1)
            else:
                h = tree.ray_cast(Vector((p[0], p[1], 3.0)), Vector((0.0, 0.0, -1.0)), 4.0)
            if h[0] is not None:
                P.append(np.array(h[0])); N.append(np.array(h[1]))
        P, N = np.array(P), np.array(N)
        # each cross-section's edges laid on the part too (a flat strip across a fold, the nose's crease, floats
        # off it): three points, each lifted off the part along its own normal
        rows = []
        for i in range(len(P)):
            tg = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
            sd = np.cross(N[i], tg); sd /= np.linalg.norm(sd)
            row = []
            for off in (-w, 0.0, w):
                h = tree.find_nearest(Vector(tuple(P[i] + sd * off)), 0.05) if off else (P[i], N[i])
                q, n = (np.array(h[0]), np.array(h[1])) if h[0] is not None else (P[i] + sd * off, N[i])
                row.append((q + n * lift, n))
            rows.append(row)
        for i in range(len(rows) - 1):
            for k in range(2):
                a0, a1, b0, b1 = rows[i][k], rows[i][k + 1], rows[i + 1][k], rows[i + 1][k + 1]
                S.face([a0[0], b0[0], b1[0], a1[0]], a0[1] + a1[1] + b0[1] + b1[1])
        n_pts += len(P)
    ob = S.object('body_lines', cfg['material'])
    log(f'trim lines: {len(cfg["lines"])} on {cfg["on"]}, {n_pts} points')
    return [ob]


def wipers(cfg):
    """Parked wiper arms on the windscreen (body_wipers, in the profile's material): each a pivot cap, an arm and a
    blade lying along the screen's lower edge; cfg['blades'] = [(pivot Unity x, length m)], blades running to +Unity x."""
    import bmesh
    glass = bpy.data.objects.get('body_' + cfg['glass'])
    me = glass.data
    me.calc_loop_triangles()
    pts, nrm = [], []
    for t in me.loop_triangles:
        if t.normal.y < -0.5:     # facing forward (the front is Blender -y): the windscreen
            pts.append(tuple(t.center)); nrm.append(tuple(t.normal))
    pts, n = np.array(pts), np.array(nrm).mean(0)
    n /= np.linalg.norm(n)
    e = np.array([1.0, 0.0, 0.0]); e -= n * (e @ n); e /= np.linalg.norm(e)
    u = np.cross(n, e); u *= np.sign(u[2])          # up the screen
    base = pts[pts[:, 2] < pts[:, 2].min() + 0.03].mean(0)
    bm = bmesh.new()

    def box(c, ax, ay, az, hx, hy, hz):
        vs = [bm.verts.new(tuple(c + sx * hx * ax + sy * hy * ay + sz * hz * az))
              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for f in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            bm.faces.new([vs[i] for i in f])

    bm.free()
    S = Solid()
    up = cfg.get('up', 0.02)                       # how far up the screen the blades lie, from its lower edge
    for px, length in cfg['blades']:
        bx = -px                                   # Unity x -> Blender x
        pivot = base + e * (bx - base[0]) - u * cfg.get('pivot_down', 0.035) + n * 0.004   # in the cowl's black trim
        S.prism(pivot, n, 0.019, 0.016, 10, e)                       # the pivot cap
        S.prism(pivot + n * 0.016, n, 0.008, 0.008, 6, e)            # its nut
        # the arm: a short riser up the screen, then along it, a little above the glass
        a0 = pivot + n * 0.018
        a1 = base + e * (bx - base[0]) + u * (up - 0.004) + n * 0.022
        S.box((a0 + a1) / 2, u, e, n, np.linalg.norm(a1 - a0) / 2, 0.009, 0.005)
        tip = a1 - e * length * 0.95
        S.box((a1 + tip) / 2, e, u, n, length * 0.475, 0.007, 0.005)
        S.box(tip - e * 0.015 + n * 0.0, e, u, n, 0.02, 0.008, 0.006)  # the arm's end cap
        # the blade: a spine and a thinner rubber strip on the glass, clipped to the arm in the middle
        bc = base + e * (bx - base[0] - length * 0.5) + u * (up + 0.012)
        S.box(bc + n * 0.011, e, u, n, length * 0.48, 0.008, 0.005)
        S.box(bc + n * 0.004, e, u, n, length * 0.47, 0.003, 0.003)
        S.box(bc + n * 0.016 - u * 0.006, e, u, n, 0.025, 0.012, 0.006)
    ob = S.object('body_wipers', cfg['material'])
    log(f'wipers: {len(cfg["blades"])} on the screen, base at {np.round(base, 3)}')
    return [ob]


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
src = [o for o in bpy.context.scene.objects if o.type == 'MESH']
out, total, hubs = [], 0, {}
for gname, prefixes in P.REFINE_GROUPS.items():
    members = [o for o in src if o.name.startswith(tuple(prefixes))]
    if not members:
        log('no objects for', gname); continue
    if WHEELS and gname.startswith(('Wheel_', 'Tire_')):
        if gname.startswith('Tire_'):   # only the hub: the wheel itself is built new
            co = to_out(np.concatenate([[tuple(o.matrix_world @ v.co) for v in o.data.vertices] for o in members]))
            hubs[gname[5:]] = (co.min(0) + co.max(0)) / 2
        continue
    work = joined(members, gname + '_work', gname)
    Pt, Nt, St, Ut, Mt, mats = gather([work])
    if os.environ.get('REFINE_DUMP'):
        np.savez(os.environ['REFINE_DUMP'] + f'_{gname}.npz', P=Pt, N=Nt, S=St, M=Mt, names=[m.name if m else '' for m in mats])
    # slivers (the bevel's clamped corners) have no normal: out, and any other unusable normal falls back
    area = np.linalg.norm(np.cross(Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]), axis=1)
    ok = area > 1e-12
    Pt, Nt, St, Ut, Mt = Pt[ok], Nt[ok], St[ok], Ut[ok], Mt[ok]
    bad = ~np.isfinite(Nt).all(-1) | (np.linalg.norm(Nt, axis=-1) < 0.5)
    Nt[bad] = 0.0                                   # no normal: that corner doesn't bend the patch
    sbad = ~np.isfinite(St).all(-1) | (np.linalg.norm(St, axis=-1) < 0.5)
    St[sbad] = np.where(bad[sbad][:, None], np.cross(Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0])[np.nonzero(sbad)[0]] /
                        area[ok][np.nonzero(sbad)[0]][:, None], Nt[sbad])
    if (~ok).sum() or bad.sum() or sbad.sum():
        log(f'  {(~ok).sum()} slivers out, {bad.sum()} shape and {sbad.sum()} shading normals unusable')
    bpy.data.objects.remove(work)
    k = SPLITS.get(gname, 2)
    V, tri, tri_n, tri_s, tri_uv = tessellate(Pt, Nt, St, Ut, k, ALPHA, getattr(P, 'REFINE_FLAT', {}).get(gname, ()))
    fm = np.repeat(Mt, k * k)
    V0 = V.copy()
    V = fair(V, tri, tri_n, fm, FAIR.get(gname, 0))
    flat = getattr(P, 'REFINE_FLAT', {}).get(gname, ())
    if len(flat):   # the flat spots keep out of the fairing too
        C = to_out(V0[tri].mean(1))
        m = np.zeros(len(tri), bool)
        for b in flat:
            m |= (b[0] <= C[:, 0]) & (C[:, 0] <= b[1]) & (b[2] <= C[:, 1]) & (C[:, 1] <= b[3]) & (b[4] <= C[:, 2]) & (C[:, 2] <= b[5])
        keep = np.unique(tri[m])
        V[keep] = V0[keep]
        log(f'  flat: {m.sum()} triangles kept as modelled')
    for box, floor in getattr(P, 'REFINE_LIFT', {}).get(gname, []):
        # (box, height): what hangs below the height inside the box (output metres) comes up to it, e.g. a
        # differential moulded into the floor, under the donor's own
        O = to_out(V)
        sel = ((box[0] <= O[:, 0]) & (O[:, 0] <= box[1]) & (box[2] <= O[:, 1]) & (O[:, 1] <= box[3]) &
               (box[4] <= O[:, 2]) & (O[:, 2] < floor))
        O[sel, 2] = floor
        V = O / SCALE @ ROT
        log(f'  lifted {sel.sum()} vertices to {floor} m')
    for rule in getattr(P, 'REFINE_RECOLOR_FINAL', {}).get(gname, []):
        # (box, material, material[, rings[, (min, max) normal z]]) on the final triangles (output metres): for big
        # source faces whose own centre lies elsewhere, e.g. a bumper's underside bulging down through the valance's
        # slots. With rings, only the triangles within that many rings of the target material (a thin paint lip round
        # a black part); with a normal range, only those facing that way
        box, frm, to = rule[:3]
        rings = rule[3] if len(rule) > 3 else None
        nzr = rule[4] if len(rule) > 4 else None
        amax = rule[5] if len(rule) > 5 else None          # and only triangles smaller than this (m2): a thin lip
        idx = {(mats[m].name if mats[m] else None): m for m in np.unique(fm)}
        if frm in idx and to in idx:
            C = to_out(V[tri].mean(1))
            sel = ((fm == idx[frm]) & (box[0] <= C[:, 0]) & (C[:, 0] <= box[1]) & (box[2] <= C[:, 1]) & (C[:, 1] <= box[3]) &
                   (box[4] <= C[:, 2]) & (C[:, 2] <= box[5]))
            if nzr is not None or amax is not None:
                Q = to_out(V[tri].reshape(-1, 3)).reshape(-1, 3, 3)
                fc = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
                fa = np.linalg.norm(fc, axis=1)
                if nzr is not None:
                    fz = fc[:, 2] / np.maximum(fa, 1e-20)
                    sel &= (nzr[0] <= fz) & (fz <= nzr[1])
                if amax is not None:
                    sel &= fa / 2 < amax
            if rings is not None:
                near = np.zeros(len(V), bool); near[tri[fm == idx[to]].ravel()] = True
                grown = np.zeros(len(fm), bool)
                for _ in range(rings):
                    hit = sel & ~grown & near[tri].any(1)
                    grown |= hit
                    near[tri[hit].ravel()] = True
                sel = grown
            fm = np.where(sel, idx[to], fm)
            log(f'  {sel.sum()} final {frm} triangles -> {to}')
    tri_n = tri_s if NORMALS == 'source' else shape_normals(V, tri, SMOOTH_ANGLE)
    fm = boundary_slivers(V, tri, tri_n, fm)
    names = []
    double = getattr(P, 'REFINE_DOUBLE', {}).get(gname, ())
    for m in np.unique(fm):
        sel = fm == m
        mname = mats[m].name if mats[m] else 'none'
        out.append(build(f'{gname}_{mname}', V, tri[sel], tri_n[sel], tri_uv[sel], mats[m])); names.append(mname)
        if mname in double:
            Vb, tb, nb, ub = back_faces(V, tri[sel], tri_n[sel], tri_uv[sel])
            out.append(build(f'{gname}_{mname}_back', Vb, tb, nb, ub, mats[m]))
            log(f'  {mname}: {len(tb)} back faces')
    total += len(tri)
    log(f'{gname}: {len(members)} objects, {len(Pt)} -> {len(tri)} tris (k={k}), materials {sorted(set(names))}')
for gname in ({} if os.environ.get('REFINE_NO_BURIED') else getattr(P, 'REFINE_BURIED', {})):
    cull_buried(gname, P.REFINE_BURIED[gname], out)
if WHEELS:
    out += stock_wheels(hubs)
if getattr(P, 'REFINE_WIPERS', None):
    out += wipers(P.REFINE_WIPERS)
if getattr(P, 'REFINE_LINES', None):
    out += trim_lines(P.REFINE_LINES)
if getattr(P, 'REFINE_BLACKOUT', None):
    out += blackout(P.REFINE_BLACKOUT)
if getattr(P, 'REFINE_CHASSIS', None):
    out += chassis(P.REFINE_CHASSIS, hubs)
for o in list(bpy.context.scene.objects):
    if o not in out:
        bpy.data.objects.remove(o)
log('total', total, 'tris in', len(out), 'objects')
# and as a .blend beside it: the converter reads that (a glTF round trip scrambled some custom normals of the
# black trim, which then showed as glossy blotches or, flipped, as holes)
bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(OUT)[0] + '.blend')
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_normals=True, export_materials='EXPORT', export_yup=True)
log('wrote', OUT)
