"""
The Baja Bug's first step, before convert_materials.py: "VW Beetle" by 3DDomino (Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/vw-beetle-1ffa1ce092184da8bab996bb3a481cf2), a detailed CAD-style beetle coloured by
materials, made into a Baja Bug. The fenders stay; the stock bumpers and wheels go; the lower engine lid and the rear
apron are cut away round a flat four out in the open, inside a two-loop tube basket with a stinger pipe; a small tube
bar hugs the nose with two rally lamps on it. Placeholder wheels mark the hubs (the car drives on the donor's wheels).
Writes a .blend for convert_materials.py with profiles/bajabug_3dd.py.

    blender -b --factory-startup --python profiles/bajabug_3dd_prep.py -- input/beetle-3ddomino/vw_beetle.glb build/bajabug-3dd/prepped.blend
    blender -b --factory-startup --python tools/convert_materials.py -- profiles/bajabug_3dd.py build/bajabug-3dd/prepped.blend build/bajabug-3dd <aro.mesh> bajabug

Everything below is built in the source's own units and axes (Blender: x right, y back with the nose near y=0, z up;
stock hubs at z 0.33), then scaled to the real wheelbase and moved so the hubs sit 0.37 m up and the axles' middle at
y=0.
"""
import bpy, bmesh, sys, os, math, mathutils
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUT = argv[0], os.path.abspath(argv[1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def log(*a): print('[prep]', *a, flush=True)


# ---- the source's measures
AXLES = (0.73, 3.05)              # stock axle y (the tyres' centres)
STOCK_HUB_Z = 0.33
SCALE = 2.40 / (AXLES[1] - AXLES[0])   # the real car's wheelbase, 2.40 m
# ---- the build (source units unless said)
BODY_LIFT = 0.16                  # metres the body sits higher over the hubs than stock (room for the 0.40 m tyres)
HUB_Y = 0.37                      # metres: the hubs' height in the output (as the earlier Bug's)
TYRE_R = 0.40                     # metres: the donor's (the Brick's) tyres
TRACK = 1.42                      # metres between the tyres' centres
CUT_Y = 3.28                      # the engine bay's front (the firewall)
WALL_X = 0.42                     # inside the rear wells' walls
TUBE_R = 0.024
_HALF = [(0.0, 0.875), (0.14, 0.875), (0.26, 0.865), (0.30, 0.78), (0.36, 0.73), (0.395, 0.58), (0.43, 0.50)]


def _chaikin(p, k=4):
    p = np.asarray(p, float)
    for _ in range(k):
        q = [p[0]]
        for a, b in zip(p, p[1:]): q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1]); p = np.array(q)
    return p


_PROF = _chaikin(_HALF)
ARCH = lambda x: float(np.interp(abs(x), _PROF[:, 0], _PROF[:, 1]))   # the rear cut's top, round the engine (from behind)

GROUPS = {}                       # output object name -> the objects that make it (joined at the end)


def put(group, o):
    GROUPS.setdefault(group, []).append(o); return o


def mat(name, rgb, metal=0.0, rough=0.5):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1); b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    return m


def bbox(o):
    w = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    return np.array([[min(v[i] for v in w) for i in range(3)], [max(v[i] for v in w) for i in range(3)]])


def fillet(pts, r, n=8):
    pts = [np.array(p, float) for p in pts]; out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        d0 = (b - a) / np.linalg.norm(b - a); d1 = (c - b) / np.linalg.norm(c - b)
        rr = min(r, np.linalg.norm(b - a) / 2, np.linalg.norm(c - b) / 2)
        p0, p2 = b - d0 * rr, b + d1 * rr
        for t in np.linspace(0, 1, n):
            out.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(pts[-1]); return out


def tube(group, pts, m, r=TUBE_R, bend=0.12, res=4):
    pts = fillet(pts, bend) if len(pts) > 2 and bend > 0 else [np.array(p, float) for p in pts]
    cu = bpy.data.curves.new('tube', 'CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = r; cu.bevel_resolution = res; cu.use_fill_caps = True
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts): p.co = (*q, 1)
    o = bpy.data.objects.new('tube', cu); o.data.materials.append(m)
    bpy.context.scene.collection.objects.link(o)
    return put(group, o)


def cyl(group, loc, r, depth, rot, m, verts=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object; o.data.materials.append(m); bpy.ops.object.shade_smooth()
    o.data.use_auto_smooth = True; o.data.auto_smooth_angle = math.radians(40)
    return put(group, o)


def box(group, c, s, m, bev=0.012):
    bpy.ops.mesh.primitive_cube_add(size=1, location=c); o = bpy.context.object
    o.scale = s; o.data.materials.append(m)
    if bev > 0:
        b = o.modifiers.new('b', 'BEVEL'); b.width = bev; b.segments = 2; b.limit_method = 'ANGLE'
    o.data.use_auto_smooth = True; o.data.auto_smooth_angle = math.radians(40)
    bpy.ops.object.shade_smooth()
    return put(group, o)


def C(c, axis):   # a cylinder along an axis
    return {'x': (0, math.pi / 2, 0), 'y': (math.pi / 2, 0, 0), 'z': (0, 0, 0)}[axis]


# ---------------------------------------------------------------- import, sort the source
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
for o in list(bpy.data.objects):
    if o.name == 'Cube': bpy.data.objects.remove(o)
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
body = max(meshes, key=lambda o: len(o.data.polygons))
log('source', len(meshes), 'mesh objects,', sum(len(o.data.polygons) for o in meshes), 'faces; body', len(body.data.polygons))

head_lens = mat('bb_headlens', (0.85, 0.85, 0.82), 0.0, 0.1)
gone = 0
for o in meshes:
    if o is body: continue
    lo, hi = bbox(o); c = (lo + hi) / 2
    mname = o.data.materials[0].name if o.data.materials else ''
    wheel = abs(c[0]) > 0.5 and any(math.hypot(c[1] - a, c[2] - STOCK_HUB_Z) < 0.36 for a in AXLES)
    bumper = (hi[0] - lo[0]) > 1.2 and (hi[1] < 0.4 or lo[1] > 3.55)
    if wheel or bumper:
        bpy.data.objects.remove(o); gone += 1; continue
    # name the lamps (no mud on them); the headlamps' grey glass becomes a lens
    if c[1] < 0.5 and 0.40 < abs(c[0]) < 0.7 and 0.45 < c[2] < 0.80:
        o.name = 'lamp_head'
        if 'Glass_Basic_Grey' in mname: o.data.materials[0] = head_lens
    elif c[1] > 3.4 and 0.42 < abs(c[0]) < 0.62 and 0.45 < c[2] < 0.72:
        o.name = 'lamp_tail'
    elif 'Glass_Basic_Orange' in mname:
        o.name = 'lamp_indicator'
log('dropped', gone, 'objects (stock wheels, bumpers)')


# ---------------------------------------------------------------- the rear cut
def rear_cut():
    """The lower engine lid and the apron cut away round the engine (seen from behind), between the wells' walls:
    bisected along the outline, the faces inside deleted (the body keeps its own normals)."""
    M = body.matrix_world; Mi = M.inverted()
    bm = bmesh.new(); bm.from_mesh(body.data)
    def region(f, pad=0.04):
        c = M @ f.calc_center_median()
        return c.y > CUT_Y - pad and abs(c.x) < WALL_X + pad
    def bis(pt, n, sel):
        geom = [f for f in bm.faces if sel(f)]
        if not geom: return
        edges = list({e for f in geom for e in f.edges}); verts = list({v for f in geom for v in f.verts})
        p = Mi @ mathutils.Vector(pt); nn = (Mi.to_3x3() @ mathutils.Vector(n)).normalized()
        bmesh.ops.bisect_plane(bm, geom=verts + edges + geom, plane_co=p, plane_no=nn)
    bis((0, CUT_Y, 0), (0, 1, 0), lambda f: region(f, 0.06))
    for sx in (-1, 1): bis((sx * WALL_X, 0, 0), (1, 0, 0), lambda f: region(f))
    xs = np.linspace(-WALL_X, WALL_X, 57)
    for x0, x1 in zip(xs, xs[1:]):
        z0, z1 = ARCH(x0), ARCH(x1)
        bis((x0, 0, z0), (-(z1 - z0), 0, x1 - x0),
            lambda f, x0=x0, x1=x1: region(f) and x0 - 0.02 <= (M @ f.calc_center_median()).x <= x1 + 0.02)
    dead = [f for f in bm.faces if (lambda c: c.y > CUT_Y and abs(c.x) < WALL_X and c.z < ARCH(c.x))(M @ f.calc_center_median())]
    bmesh.ops.delete(bm, geom=dead, context='FACES'); bm.to_mesh(body.data); bm.free()
    log('rear cut:', len(dead), 'faces out')
    for o in list(bpy.data.objects):
        if o.type != 'MESH' or o is body: continue
        lo, hi = bbox(o); c = (lo + hi) / 2
        if c[1] > CUT_Y and abs(c[0]) < WALL_X and c[2] < ARCH(c[0]):
            log('  and', o.name); bpy.data.objects.remove(o)


rear_cut()


def fix_inside_out():
    """Pieces made inside out (the source's mirrored side was exported without turning its faces round, and Sketchfab
    draws both sides; the game draws only the front): per loose piece, sampled faces vote by which way a ray from
    them gets out of the car; a piece whose faces mostly see out from their backs is turned round (faces and normals)."""
    import random
    from mathutils.bvhtree import BVHTree
    random.seed(1)
    V = mathutils.Vector
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
    allbm = bmesh.new()
    for o in objs:
        me = o.data.copy(); me.transform(o.matrix_world); allbm.from_mesh(me); bpy.data.meshes.remove(me)
    tree = BVHTree.FromBMesh(allbm)
    faces = 0
    for o in objs:
        me = o.data; M = o.matrix_world; N3 = M.to_3x3().inverted().transposed()
        me.calc_normals_split()
        ln = {(p.index, me.loops[li].vertex_index): V(me.loops[li].normal) for p in me.polygons for li in p.loop_indices}
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        seen, flip = set(), []
        for f0 in bm.faces:
            if f0.index in seen: continue
            comp, stack = [], [f0]; seen.add(f0.index)
            while stack:
                f = stack.pop(); comp.append(f)
                for e in f.edges:
                    for g in e.link_faces:
                        if g.index not in seen: seen.add(g.index); stack.append(g)
            good = inv = 0
            for f in random.sample(comp, min(60, len(comp))):
                n = (N3 @ f.normal).normalized(); c = M @ f.calc_center_median()
                if n.length < 0.5: continue
                out_front = tree.ray_cast(c + n * 2e-4, n, 5.0)[0] is None
                out_back = tree.ray_cast(c - n * 2e-4, -n, 5.0)[0] is None
                if out_front and not out_back: good += 1
                elif out_back and not out_front: inv += 1
            if inv > good and inv >= 2: flip += comp
        if not flip:
            bm.free(); continue
        flipped = {f.index for f in flip}
        bmesh.ops.reverse_faces(bm, faces=flip)
        bm.to_mesh(me); bm.free()
        new = []
        for p in me.polygons:
            s = -1.0 if p.index in flipped else 1.0
            new += [tuple(ln[(p.index, me.loops[li].vertex_index)] * s) for li in p.loop_indices]
        me.use_auto_smooth = True
        me.normals_split_custom_set(new)
        faces += len(flipped)
        log(f'  turned round in {o.name}: {len(flipped)} faces')
    allbm.free()
    log(f'inside-out pieces turned round: {faces} faces')


fix_inside_out()

trim = mat('bb_trim', (0.06, 0.06, 0.065), 0.5, 0.45)        # the tubes (the trim colour picker)
alu, black = mat('bb_alu', (0.20, 0.20, 0.21), 0.8, 0.35), mat('bb_black', (0.03, 0.03, 0.035), 0.2, 0.6)
chrome, fins = mat('bb_chrome', (0.34, 0.34, 0.36), 1.0, 0.12), mat('bb_fins', (0.10, 0.10, 0.11), 0.6, 0.5)
under = mat('bb_under', (0.035, 0.035, 0.038), 0.0, 0.8)     # the underside, the wheel wells, the body's insides
bay_mat = mat('bb_under_bay', (0.035, 0.035, 0.038), 0.0, 0.8)  # the same for the engine bay's walls


def underside_dark():
    """The floor's underside and the wheel wells' insides dark, not body colour (material only: the source's normals
    stay)."""
    M = body.matrix_world; N3 = M.to_3x3().inverted().transposed()
    me = body.data
    if under.name not in [m.name for m in me.materials if m]: me.materials.append(under)
    ui = [m.name if m else '' for m in me.materials].index(under.name)
    hubs = [mathutils.Vector((0, a, STOCK_HUB_Z)) for a in AXLES]
    # the open air, for faces high on a well's edge: the nose panel's flanks tuck under the front fenders, facing out
    # and near the hub, but you see them from the front (dark smears along the fender seams on light paint)
    from mathutils.bvhtree import BVHTree
    abm = bmesh.new()
    for o in bpy.data.objects:
        if o.type == 'MESH':
            m2 = o.data.copy(); m2.transform(o.matrix_world); abm.from_mesh(m2); bpy.data.meshes.remove(m2)
    atree = BVHTree.FromBMesh(abm); abm.free()
    looks = [mathutils.Vector(v).normalized() for v in ((0, -1, 0), (0, -1, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0))]
    def in_view(c, n):
        return any(atree.ray_cast(c + n * 0.002 + d * 0.002, d, 5.0)[0] is None for d in looks if d.dot(n) > 0.05)
    dark = 0
    for p in me.polygons:
        c = M @ p.center; n = (N3 @ p.normal).normalized()
        # facing down, low; not the nose's (and tail's) lower curve, facing mostly forward and seen from the front
        down = n.z < -0.35 and c.z < 0.45 and not (abs(n.y) > 0.45 and in_view(c, n))
        well = False
        if 0.25 < abs(c.x) < 0.85:       # facing the tyre (its middle ~0.62 out) and near its axle: the well's insides
            for h in hubs:
                r = mathutils.Vector((0, h.y - c.y, h.z - c.z)).length
                d = mathutils.Vector((math.copysign(0.62, c.x) - c.x, h.y - c.y, h.z - c.z))
                if r < 0.58 and n.dot(d.normalized()) > 0.2 and not (c.z > h.z + 0.05 and in_view(c, n)): well = True
        if down or well: p.material_index = ui; dark += 1
    log(f'underside and wells dark: {dark} faces')



underside_dark()


def bay_walls(cell=0.02):
    """The engine bay's walls, dark, wherever they are under the bodywork: the firewall (from the floor up to the lid)
    and the sides between the bay and the rear wells (under the lid, ahead of the rear skin). Two-sided."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    bmb = bmesh.new(); bmb.from_object(body, dg); bmb.transform(body.matrix_world)
    tree = BVHTree.FromBMesh(bmb)
    V = mathutils.Vector
    def hit(p, d, far=2.0):
        h = tree.ray_cast(V(p), V(d), far); return h[0] is not None
    allbm = bmesh.new()   # the roof: every source piece (the lid, the glass), not the body alone
    for o in bpy.data.objects:
        if o.type == 'MESH' and not o.name.startswith(('Tire', 'Wheel', 'bb_', 'lamp_rally')):
            me = o.data.copy(); me.transform(o.matrix_world); allbm.from_mesh(me); bpy.data.meshes.remove(me)
    rtree = BVHTree.FromBMesh(allbm)
    def roof(x, y):   # the top at (x, y): the lowest first hit from above within 4 cm along y that isn't down
        # through a seam or a slot to the floor (never higher than the surface: nothing pokes out)
        zs = [h[0].z for d in (-0.04, -0.02, 0, 0.02, 0.04)
              for h in [rtree.ray_cast(V((x, y + d, 3.0)), V((0, 0, -1)), 4.0)] if h[0] is not None and h[0].z > 0.45]
        return min(zs) if zs else np.nan
    def roofline(pts):   # along a line (-1: no top)
        r = np.array([roof(x, y) for x, y in pts]); return np.where(np.isnan(r), -1.0, r)
    def skin(x, y):   # the paint's top at (x, y) (nan: none, or through a gap in it to below the bay's top)
        h = tree.ray_cast(V((x, y, 3.0)), V((0, 0, -1)), 4.0)[0]
        return h.z if h is not None and h.z > 0.70 else np.nan
    def fill(r):      # gaps filled from the neighbours
        ok = ~np.isnan(r)
        if ok.sum() < 2: return r
        k = np.arange(len(r)); return np.where(ok, r, np.interp(k, k[ok], r[ok]))
    def skinline(pts):
        # never over the paint: roof() and top_x() see every piece, and where the paint has gaps (at the cut, the louvre's
        # ends) they saw a piece 8 cm higher, and the walls' corners came out through the paint's sides
        return fill(np.array([skin(x, y) for x, y in pts]))
    # each wall: (point at grid (u, w), u range, w range, snap): snap moves a grid point onto the wall's region (its
    # edges then follow the body instead of stepping), None where it can't. A cell is made when one corner is inside.
    walls = []
    # the firewall: x across, z up, at y = CUT_Y - 0.012; from the floor's level up to 15 mm under the roof
    fy_ = CUT_Y - 0.012; fx = np.linspace(-WALL_X - 0.01, WALL_X + 0.01, 300)
    ftop = roofline([(x, fy_) for x in fx])
    fsk = skinline([(x, fy_) for x in fx]); ftop = np.where((ftop < 0) | np.isnan(fsk), ftop, np.minimum(ftop, fsk))
    def fire_snap(u, w):
        t = np.interp(u, fx, ftop) - 0.015
        return None if t < 0.30 else (u, fy_, min(w, t))
    walls.append(('fire', fire_snap, (-WALL_X - 0.01, WALL_X + 0.01), (0.24, 1.30), (0, -1, 0)))
    for sx in (-1, 1):
        x = sx * (WALL_X + 0.008); sy = np.linspace(CUT_Y - 0.012, 4.20, 300)
        stop = roofline([(x, y) for y in sy]) - 0.015
        # roof() is the lowest top within 4 cm along y: where the lid's lip slopes down at the bay's rear that is well
        # under the shell right above, and a wedge stayed open over the wall (seen through). There: up to the shell's
        # underside right above (the first down-facing body surface under the local top, the top no lower than
        # roof(): not through a slot)
        for k, y in enumerate(sy):
            if stop[k] < 0: continue
            o, t0 = V((x, y, 3.0)), None
            for _ in range(8):
                h = tree.ray_cast(o, V((0, 0, -1)), 4.0)
                if h[0] is None: break
                if t0 is None:
                    t0 = h[0].z
                    if t0 < stop[k] + 0.010: break
                elif h[0].z < t0 - 0.08: break
                if h[1].z < -0.2:
                    stop[k] = max(stop[k], h[0].z - 0.004); break
                o = h[0] - V((0, 0, 1e-4))
        ssk = skinline([(x, y) for y in sy]); stop = np.where((stop < 0) | np.isnan(ssk), stop, np.minimum(stop, ssk - 0.004))
        def side_snap(u, w, x=x, sy=sy, stop=stop):
            h = tree.ray_cast(V((x, CUT_Y - 0.012, w)), V((0, 1, 0)), 0.8)[0]   # the rear skin at this height
            if h is None: return None
            y = min(u, h.y - 0.003); t = np.interp(y, sy, stop)
            return None if t < w - 0.25 else (x, y, min(w, t))
        walls.append((f'side{sx}', side_snap, (CUT_Y - 0.012, 4.20), (0.18, 1.20), (-sx, 0, 0)))
    bm = bmesh.new(); made = 0
    for name, snap, (u0, u1), (w0, w1), nrm in walls:
        nu, nw = int(round((u1 - u0) / cell)), int(round((w1 - w0) / cell))
        us = np.linspace(u0, u1, nu + 1); ws = np.linspace(w0, w1, nw + 1)
        pts = [[snap(u, w) for w in ws] for u in us]
        # a corner is inside when the snap left it where it was (on the wall's plane only its u and w can move)
        ok = np.array([[p is not None and abs((p[0] if name == 'fire' else p[1]) - u) < 1e-9 and abs(p[2] - w) < 1e-9
                        for p, w in zip(row, ws)] for row, u in zip(pts, us)])
        vs = {}
        def vert(i, j, side):
            k = (i, j, side)
            if k not in vs:
                vs[k] = bm.verts.new(V(pts[i][j]) + V(nrm) * (0.002 if side else -0.002))
            return vs[k]
        for i in range(nu):
            for j in range(nw):
                cs = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                if any(pts[a][b] is None for a, b in cs) or not any(ok[a, b] for a, b in cs): continue
                # as two triangles, each turned to face its side: the snap can fold a cell over (a corner pulled
                # back past its neighbour onto the rear skin), and a folded quad's one normal leaves half of it
                # facing the wrong way, culled in game (see-through slivers at the walls' rear ends)
                for tri in ((cs[0], cs[1], cs[2]), (cs[0], cs[2], cs[3])):
                    q = [V(pts[a][b]) for a, b in tri]
                    if (q[1] - q[0]).cross(q[2] - q[0]).length < 1e-7: continue  # squashed flat by the snap
                    for side, sgn in ((1, 1), (0, -1)):
                        try: f = bm.faces.new([vert(a, b, side) for a, b in tri])
                        except ValueError: continue
                        f.normal_update()
                        if f.normal.dot(V(nrm)) * sgn < 0: f.normal_flip()
                        made += 1
    # the ceiling: under the engine lid and the louvre (single-sheet in the source: from the bay and through the louvre
    # you saw out through it), following the lid 12 mm under it, two-sided
    xs = np.arange(-0.46, 0.46 + 1e-6, cell); ys = np.arange(2.78, 4.20 + 1e-6, cell)
    zt = np.full((len(xs), len(ys)), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            h = tree.ray_cast(V((x, y, 0.72)), V((0, 0, 1)), 1.0)
            if h[0] is not None and h[0].z > 0.70: zt[i, j] = h[0].z - 0.012
    for zz in (zt, zt.T):           # the louvre's slots let rays through: fill the gaps between hits, along y then x
        for row in zz:
            ok = ~np.isnan(row)
            if ok.sum() >= 2:
                k = np.arange(len(row)); a, b = k[ok][0], k[ok][-1]
                row[a:b + 1] = np.interp(k[a:b + 1], k[ok], row[ok])
    def top_x(x, y):    # the top (glass and paint) at (x, y), over the louvre's slots (narrow across x)
        zs = [h[0].z for d in (-0.012, -0.006, 0, 0.006, 0.012)
              for h in [rtree.ray_cast(V((x + d, y, 3.0)), V((0, 0, -1)), 4.0)] if h[0] is not None]
        return max(zs) if zs else np.nan
    for i, x in enumerate(xs):      # never over the glass or the paint
        for j, y in enumerate(ys):
            t = top_x(x, y)
            if not np.isnan(t) and not np.isnan(zt[i, j]): zt[i, j] = min(zt[i, j], t - 0.012)
    sk = np.array([[skin(x, y) for y in ys] for x in xs])
    sk = np.array([fill(r) for r in sk]); sk = np.array([fill(c) for c in sk.T]).T
    gx, gy = np.gradient(sk, cell)   # 12 mm square to the paint, not straight down (on its steep sides that was nothing)
    zt = np.where(np.isnan(sk), zt, np.minimum(zt, sk - 0.012 * np.minimum(np.sqrt(1 + gx ** 2 + gy ** 2), 5.0)))
    cv = {}
    def cvert(i, j, side):
        k = (i, j, side)
        if k not in cv: cv[k] = bm.verts.new((xs[i], ys[j], zt[i, j] + (0.002 if side else -0.002)))
        return cv[k]
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            q = zt[i:i + 2, j:j + 2]
            if np.isnan(q).any() or q.max() - q.min() > 0.12: continue
            # over the cut, never down past its outline (it hung out under the lid's edge)
            if any(ys[jj] > CUT_Y - 0.02 and abs(xs[ii]) < WALL_X + 0.03 and zt[ii, jj] < ARCH(xs[ii]) + 0.03
                   for ii in (i, i + 1) for jj in (j, j + 1)): continue
            for side, sgn in ((1, 1), (0, -1)):
                f = bm.faces.new([cvert(i, j, side), cvert(i + 1, j, side), cvert(i + 1, j + 1, side), cvert(i, j + 1, side)])
                f.normal_update()
                if f.normal.z * sgn < 0: f.normal_flip()
                made += 1
    bmb.free()
    me = bpy.data.meshes.new('bb_bay'); bm.to_mesh(me); bm.free()
    me.materials.append(bay_mat)   # its own material: the converter keeps it out of the decimation (DECIMATE_KEEP)
    o = bpy.data.objects.new('bb_bay', me); bpy.context.scene.collection.objects.link(o)
    log(f'bay walls: {made} faces')
    return put('bb_bay', o)


bay_walls()
housing, lens = mat('bb_housing', (0.05, 0.05, 0.05), 0.5, 0.4), mat('bb_lens', (0.95, 0.93, 0.80), 0.0, 0.1)

# the flat four
E = 'bb_engine'
box(E, (0, 3.50, 0.40), (0.24, 0.44, 0.24), alu, 0.03)                              # crankcase
box(E, (0, 3.50, 0.27), (0.20, 0.36, 0.06), alu, 0.02)                              # sump
for sx in (-1, 1):
    for y in (3.41, 3.59):
        cyl(E, (sx * 0.19, y, 0.40), 0.052, 0.17, C(0, 'x'), fins, 16)              # barrel
        for k in range(6):
            cyl(E, (sx * (0.125 + k * 0.026), y, 0.40), 0.074, 0.008, C(0, 'x'), fins, 20)   # fins
    box(E, (sx * 0.30, 3.50, 0.40), (0.05, 0.42, 0.15), alu, 0.015)                 # heads
    box(E, (sx * 0.345, 3.50, 0.40), (0.04, 0.34, 0.11), black, 0.015)             # rocker covers
    box(E, (sx * 0.20, 3.50, 0.505), (0.20, 0.44, 0.015), black, 0.0)              # cylinder tins
    tube(E, [(sx * 0.29, 3.50, 0.49), (sx * 0.24, 3.42, 0.62), (sx * 0.17, 3.38, 0.70)], alu, r=0.018, bend=0.05, res=2)
    cyl(E, (sx * 0.17, 3.38, 0.72), 0.03, 0.06, C(0, 'z'), alu, 12)                # carb
    cyl(E, (sx * 0.17, 3.38, 0.78), 0.085, 0.07, C(0, 'z'), chrome, 28)            # air cleaner
    cyl(E, (sx * 0.17, 3.38, 0.818), 0.03, 0.012, C(0, 'z'), chrome, 12)
    for y in (3.41, 3.59):                                                          # headers to the collector
        tube(E, [(sx * 0.30, y, 0.32), (sx * 0.30, y, 0.25), (sx * 0.17, 3.79, 0.25)], chrome, r=0.019, bend=0.06, res=2)
box(E, (0, 3.47, 0.60), (0.62, 0.36, 0.15), black, 0.04)                            # fan shroud
cyl(E, (0, 3.64, 0.62), 0.15, 0.10, C(0, 'y'), black, 32)                           # fan housing
cyl(E, (0, 3.70, 0.74), 0.065, 0.14, C(0, 'y'), alu, 20)                            # alternator
cyl(E, (0, 3.78, 0.74), 0.055, 0.025, C(0, 'y'), alu, 20)                           # its pulley
cyl(E, (0, 3.78, 0.42), 0.085, 0.03, C(0, 'y'), alu, 28)                            # crank pulley
for sx in (-1, 1):
    tube(E, [(sx * 0.068, 3.78, 0.43), (sx * 0.05, 3.78, 0.74)], black, r=0.006, bend=0, res=1)   # belt
cyl(E, (0.08, 3.30, 0.66), 0.03, 0.10, C(0, 'z'), black, 12)                        # distributor
cyl(E, (0, 3.80, 0.25), 0.045, 0.40, C(0, 'x'), chrome, 24)                         # collector
# the stinger: up inside the basket, then back over its top loop; open at the end, sooty inside
STINGER = [(0.18, 3.80, 0.25), (0.27, 3.82, 0.26), (0.27, 3.82, 0.66), (0.29, 3.93, 0.96)]
tube('bb_stinger', STINGER, chrome, r=0.038, bend=0.08).data.use_fill_caps = False
soot = tube('bb_stinger', STINGER, under, r=0.034, bend=0.08)
soot.data.use_fill_caps = False
soot['inside_out'] = 1


def ring(group, c, d, r0, r1, m, n=32):
    """A flat ring (the pipe's lip) at c, facing d."""
    c, d = mathutils.Vector(c), mathutils.Vector(d).normalized()
    a = d.orthogonal().normalized(); b = d.cross(a)
    bm = bmesh.new()
    inner = [bm.verts.new(c + (a * math.cos(t) + b * math.sin(t)) * r0) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    outer = [bm.verts.new(c + (a * math.cos(t) + b * math.sin(t)) * r1) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    for i in range(n):
        f = bm.faces.new((inner[i], inner[i - 1], outer[i - 1], outer[i])); f.normal_update()
        if f.normal.dot(d) < 0: f.normal_flip()
    me = bpy.data.meshes.new('ring'); bm.to_mesh(me); bm.free(); me.materials.append(m)
    o = bpy.data.objects.new('ring', me); bpy.context.scene.collection.objects.link(o)
    return put(group, o)


ring('bb_stinger', STINGER[-1], np.subtract(STINGER[-1], STINGER[-2]), 0.034, 0.038, chrome)

# the rear basket: two loops round the engine, posts between them
lo_z, hi_z, bx, by0, by1 = 0.28, 0.50, 0.40, 3.55, 3.89
for z in (lo_z, hi_z):
    tube('bb_rear', [(-bx, by0, z), (-bx, by1, z), (bx, by1, z), (bx, by0, z)], trim, bend=0.10)
for x in (-0.28, -0.08, 0.08, 0.28):
    tube('bb_rear', [(x, by1, lo_z), (x, by1, hi_z)], trim, bend=0)
for sx in (-1, 1):
    tube('bb_rear', [(sx * bx, 3.72, lo_z), (sx * bx, 3.72, hi_z)], trim, bend=0)
    tube('bb_rear', [(sx * bx, by0, lo_z), (sx * bx, by0 - 0.02, lo_z + 0.11), (sx * bx, by0, hi_z)], trim, bend=0.05)

# reverse lamps: small round lamps on the top loop's rear corners, facing back (clear of the stinger; far enough out
# that each side's reverse glow comes from its own lamp)
for sx in (-1, 1):
    x, y, z, r = sx * 0.355, 3.865, hi_z + TUBE_R + 0.045, 0.034
    cyl('lamp_reverse', (x, y, z), r + 0.006, 0.045, C(0, 'y'), housing, 24)
    cyl('lamp_reverse', (x, y + 0.0235, z), r, 0.004, C(0, 'y'), lens, 24)
    cyl('lamp_reverse', (x, y, hi_z + TUBE_R + 0.004), 0.008, 0.03, C(0, 'z'), housing, 8)

# the front: a small bar hugging the apron, two rally lamps on it
# (the apron's face is at y ~0.21 at x 0.22 and ~0.27 at x 0.46; its lower edge at z ~0.36): the bar's ends and its
# two struts run back into the apron, on plates, and a skid plate closes the gap under the nose
fy, fz = 0.07, 0.38
tube('bb_front', [(-0.46, 0.29, fz + 0.03), (-0.44, fy + 0.04, fz), (-0.30, fy, fz), (0.30, fy, fz), (0.44, fy + 0.04, fz),
                  (0.46, 0.29, fz + 0.03)], trim, bend=0.07)
for sx in (-1, 1):
    tube('bb_front', [(sx * 0.22, fy, fz), (sx * 0.22, 0.235, fz + 0.03)], trim, bend=0)
    box('bb_front', (sx * 0.22, 0.212, fz + 0.03), (0.09, 0.012, 0.08), trim, 0.004)          # strut plates
    x, y, z, r = sx * 0.19, fy - 0.01, fz + 0.115, 0.075
    cyl('lamp_rally', (x, y, z), r, r * 0.9, C(0, 'y'), housing, 28)
    cyl('lamp_rally', (x, y - r * 0.46, z), r * 0.86, 0.01, C(0, 'y'), lens, 28)
    cyl('lamp_rally', (x, y, z - r * 0.9), 0.012, r * 0.9, C(0, 'z'), housing, 10)
box('bb_front', (0, (fy + 0.33) / 2, fz - 0.016), (0.62, 0.26, 0.008), under, 0.0)           # skid plate

# placeholder wheels (the car drives on the donor's): they set the hubs. In source units, so that after the scale and
# the move the hubs land HUB_Y up with the body BODY_LIFT higher over them than stock
hub_z = STOCK_HUB_Z - BODY_LIFT / SCALE
tyre_m, rim_m = mat('bb_tyre', (0.03, 0.03, 0.03), 0, 0.9), mat('bb_rim', (0.30, 0.30, 0.32), 0.6, 0.4)
for a, fr in zip(AXLES, 'FR'):
    for sx, lr in ((-1, 'L'), (1, 'R')):
        x = sx * TRACK / 2 / SCALE
        put(f'Tire_{fr}{lr}', cyl('tmp', (x, a, hub_z), TYRE_R / SCALE, 0.25 / SCALE, C(0, 'x'), tyre_m, 40))
        put(f'Wheel_{fr}{lr}', cyl('tmp', (x + sx * 0.005, a, hub_z), 0.20 / SCALE, 0.24 / SCALE, C(0, 'x'), rim_m, 24))
GROUPS.pop('tmp')

# ---------------------------------------------------------------- join the groups, scale and place
for name, objs in GROUPS.items():
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    curves = [o for o in objs if o.type == 'CURVE']
    if curves:
        bpy.ops.object.convert(target='MESH')
        objs = [o for o in bpy.context.selected_objects]
    for o in objs:   # bevels into the mesh; inside-out parts (the pipe's soot) turned round
        bpy.context.view_layer.objects.active = o
        for md in list(o.modifiers): bpy.ops.object.modifier_apply(modifier=md.name)
        if o.get('inside_out'):
            bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
            bm.to_mesh(o.data); bm.free()
            if o.data.has_custom_normals:
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    j = bpy.context.view_layer.objects.active; j.name = name; j.data.name = name
    j.data.use_auto_smooth = True; j.data.auto_smooth_angle = math.radians(40)

dz = HUB_Y - SCALE * hub_z
root = bpy.data.objects.new('bb_root', None); bpy.context.scene.collection.objects.link(root)
for o in bpy.data.objects:
    if o.parent is None and o is not root:
        mw = o.matrix_world.copy(); o.parent = root; o.matrix_world = mw
root.scale = (SCALE,) * 3
root.location = (0, -SCALE * (AXLES[0] + AXLES[1]) / 2, dz)
bpy.context.view_layer.update()
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
log(f'scale x{SCALE:.4f}, hubs {HUB_Y} m up (body {BODY_LIFT} m over stock), {tris} triangles')
bpy.ops.wm.save_as_mainfile(filepath=OUT)
log('saved', OUT)
