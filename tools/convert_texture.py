"""
Converter for models with one baked texture (a photo-style or hand-painted map over a proper unwrap, where windows,
lamps, trim and badges are painted across faces rather than modelled): GLB -> <name>.mesh + maps + previews.

Run:  blender -b --factory-startup --python tools/convert_texture.py -- <profile.py> <model.glb> <out_dir> <reference.mesh> [name]

Nothing is classed per face: the source UVs stay as the atlas, every texel's car-space position and normal are baked
into texture space, and the profile's classify() decides each texel's class from its colour and where it sits on the
car (a blue-grey texel up high is a window, low at the front a headlamp). The maps follow from the texel classes, so
the paint zone and the lamps keep the texture's own edges. The model may be one mesh: the wheels are found as the loose
pieces inside the profile's wheel search volumes. The model is scaled to a real wheelbase.
"""
import bpy, bmesh, sys, os, math, importlib.util
import numpy as np
from mathutils import Vector, Matrix

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import pipeline as pl
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER, IDQ, BACKQ, log

argv = sys.argv[sys.argv.index('--') + 1:]
spec = importlib.util.spec_from_file_location('profile', argv[0]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GLB, OUT, REF = os.path.abspath(argv[1]), os.path.abspath(argv[2]), os.path.abspath(argv[3])
NAME = argv[4] if len(argv) > 4 else P.NAME
os.makedirs(OUT, exist_ok=True)
WELD = getattr(P, 'WELD', 1e-5)
S = P.FRONT_SIGN   # +1: the front is Blender -y
PART_IDS = {'body': 0, 'glass': 1, 'wheel': 2, 'tyre': 3}


def to_unity(v):  # Blender (x,y,z) -> Unity (x right, y up, z forward); a mirror, fixed by the winding check later
    return np.stack([-S * v[..., 0], v[..., 2], -S * v[..., 1]], -1)


# ---------------------------------------------------------------- import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o.name.startswith(tuple(P.DROP_OBJECTS)):
        log('drop object', o.name); bpy.data.objects.remove(o)
bpy.context.view_layer.update()
src_img = bpy.data.images[P.TEXTURE]
TEX = src_img.size[0]
assert src_img.size[0] == src_img.size[1], 'the source texture must be square'
log(f'texture {src_img.name} {TEX}x{TEX}')

# one mesh in world space
meshes = []
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = bpy.data.meshes.new_from_object(o, preserve_all_data_layers=True); me.transform(o.matrix_world); meshes.append(me)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
src = pl.make_obj('src', meshes, True)   # welded only after the wheels are split off (welding joins touching pieces)
me = src.data
me.uv_layers[0].name = 'src'
for uv in list(me.uv_layers)[1:]: me.uv_layers.remove(uv)


# ---------------------------------------------------------------- wheels: loose pieces inside the search volumes
def loose_parts(bm):
    bm.faces.ensure_lookup_table(); seen = set(); out = []
    for f in bm.faces:
        if f.index in seen: continue
        st, comp = [f], []; seen.add(f.index)
        while st:
            g = st.pop(); comp.append(g)
            for e in g.edges:
                for n in e.link_faces:
                    if n.index not in seen: seen.add(n.index); st.append(n)
        out.append(comp)
    return out


W = P.WHEEL_SEARCH   # source units: axle_y (front, rear), axle_z, radius, x (inner, outer) from the middle
bm = bmesh.new(); bm.from_mesh(me)
corner_faces = {}
for comp in loose_parts(bm):
    v = np.array([tuple(l.vert.co) for f in comp for l in f.loops])
    ax = np.abs(v[:, 0])
    if ax.min() < W['x'][0] or ax.max() > W['x'][1] or np.ptp(np.sign(v[:, 0])) > 0: continue
    for ai, ay in enumerate(W['axle_y']):
        if np.hypot(v[:, 1] - ay, v[:, 2] - W['axle_z']).max() <= W['radius']:
            key = ('F' if ai == 0 else 'R', 'L' if -S * np.sign(v[0, 0]) < 0 else 'R')   # Unity x = -S * Blender x; left is -x
            corner_faces.setdefault(key[0] + key[1], []).extend(f.index for f in comp)
            break
log('wheel pieces (faces per corner):', {k: len(v) for k, v in sorted(corner_faces.items())})
if len(corner_faces) != 4:
    for comp in loose_parts(bm):   # what lies near the search volumes, to fix WHEEL_SEARCH by
        v = np.array([tuple(l.vert.co) for f in comp for l in f.loops])
        d = min(np.hypot(v[:, 1] - ay, v[:, 2] - W['axle_z']).max() for ay in W['axle_y'])
        if d < 3 * W['radius']:
            log(f'  piece {len(comp)} faces, min {np.round(v.min(0), 3)} max {np.round(v.max(0), 3)}, reach from axle {d:.3f}')
    raise SystemExit('expected wheel pieces at four corners: check WHEEL_SEARCH')
wheel_face_ids = {i for v in corner_faces.values() for i in v}
fl_ids = set(corner_faces['FL'])

# wheel centres (source units): the middle of each corner's pieces
bm.faces.ensure_lookup_table()
centres_src = {}
for k, ids in corner_faces.items():
    v = np.array([tuple(l.vert.co) for i in ids for l in bm.faces[i].loops])
    centres_src[k] = (v.min(0) + v.max(0)) / 2
fy = np.mean([c[1] for k, c in centres_src.items() if k[0] == 'F']); ry = np.mean([c[1] for k, c in centres_src.items() if k[0] == 'R'])
ground = min(np.array([tuple(l.vert.co) for i in wheel_face_ids for l in bm.faces[i].loops])[:, 2])
K = P.WHEELBASE / abs(fy - ry)
PLACE = Matrix.Scale(K, 4) @ Matrix.Translation((0.0, -(fy + ry) / 2, -ground))
log(f'scale x{K:.3f}, ground at {ground:.4f}, wheelbase {abs(fy - ry):.4f} source units')

# part per face: the FL pieces split into rim and tyre (by distance from the axle), the other wheels dropped
fl_c = centres_src['FL']
part = np.zeros(len(bm.faces), np.int64)
gone = []
for f in bm.faces:
    if f.index in fl_ids:
        c = f.calc_center_median()
        part[f.index] = PART_IDS['tyre'] if math.hypot(c[1] - fl_c[1], c[2] - fl_c[2]) * K > P.RIM_RADIUS else PART_IDS['wheel']
    elif f.index in wheel_face_ids:
        gone.append(f.index)
pl.face_attr(me, 'part', part.tolist())
bm.free()
bm = bmesh.new(); bm.from_mesh(me)
lay = bm.faces.layers.int['part']
bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.faces[i] for i in gone], context='FACES')
bm.to_mesh(me); bm.free()
me.transform(PLACE)
me.update()
pl.weld(src, WELD * K)
me = src.data

wheel_centres = {}
for k, c in centres_src.items():
    wheel_centres[k] = to_unity(np.array(PLACE @ Vector(c))[None])[0]
log('wheels', {k: np.round(c, 3) for k, c in sorted(wheel_centres.items())})


# ---------------------------------------------------------------- the profile's body edits (optional)
def to_blender(u):
    u = np.asarray(u, float)
    return Vector((-S * u[0], -S * u[2], u[1]))


def free_cells(mesh, cell_px, gap_px):
    """Squares of cell_px texels the source unwrap leaves empty (gap_px clear all round), as (x0, y0) pixel corners."""
    g = 4; n = TEX // g
    used = np.zeros((n, n), bool)
    uv = np.zeros(len(mesh.loops) * 2); mesh.uv_layers['src'].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2) % 1.0
    for p in mesh.polygons:
        t = uv[p.loop_start:p.loop_start + p.loop_total] * n
        x0, y0 = np.floor(t.min(0)).astype(int); x1, y1 = np.ceil(t.max(0)).astype(int)
        used[max(y0, 0):min(y1, n), max(x0, 0):min(x1, n)] = True
    out = []
    step = cell_px // g; pad = gap_px // g
    for y in range(0, n - step + 1, step):
        for x in range(0, n - step + 1, step):
            if not used[max(y - pad, 0):y + step + pad, max(x - pad, 0):x + step + pad].any(): out.append((x * g, y * g))
    return out


def _fillet(pts, r, closed):
    """Each corner of a polyline rounded (a quadratic curve between the points r back and r on, r clamped to half of
    either leg)."""
    n = len(pts); out = []
    for i in range(n):
        if not closed and i in (0, n - 1): out.append(pts[i]); continue
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        a, b = p1 - p0, p2 - p1
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if la < 1e-9 or lb < 1e-9: out.append(p1); continue
        phi = math.acos(np.clip(np.dot(a / la, b / lb), -1, 1))
        if phi < 1e-3: out.append(p1); continue
        t = min(r * math.tan(phi / 2), 0.5 * la, 0.5 * lb)
        q0, q1 = p1 - a / la * t, p1 + b / lb * t
        k = max(2, int(phi / (math.pi / 48)) + 1)
        for s in np.linspace(0, 1, k + 1):
            out.append((1 - s) ** 2 * q0 + 2 * (1 - s) * s * p1 + s ** 2 * q1)
    return np.array(out)


def _resample(pts, step, closed, passes=1):
    """A polyline resampled at even spacing (about step), light smoothing passes over the result (the ends held)."""
    P_ = np.vstack([pts, pts[:1]]) if closed else pts
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P_, axis=0), axis=1))]
    if d[-1] < 1e-6: return pts[:1]
    k = max(2, int(round(d[-1] / step)) + (0 if closed else 1))
    s = np.linspace(0, d[-1], k, endpoint=not closed)
    Q = np.stack([np.interp(s, d, P_[:, c]) for c in range(3)], 1)
    for _ in range(passes if len(Q) > 4 else 0):
        R = Q.copy()
        if closed: R = (np.roll(Q, 1, 0) + 2 * Q + np.roll(Q, -1, 0)) / 4
        else: R[1:-1] = (Q[:-2] + 2 * Q[1:-1] + Q[2:]) / 4
        Q = R
    return Q


def _frames(pts, closed):
    """Twist-free frames (tangent, normal, binormal) along a polyline (parallel transport)."""
    n = len(pts)
    if closed: t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    else: t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    up = np.array([0.0, 1.0, 0.0]) if abs(t[0][1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nv = up - np.dot(up, t[0]) * t[0]; nv /= np.linalg.norm(nv)
    out = []
    for i in range(n):
        nv = nv - np.dot(nv, t[i]) * t[i]; nv /= max(np.linalg.norm(nv), 1e-12)
        out.append((t[i], nv, np.cross(t[i], nv)))
    return out


FLAT_CELLS = []   # (x0, y0, size) texel squares holding a lining's flat colour
ADDED_CELLS = []  # (x0, y0, size) every swatch cell (linings and new parts): not the source's texels


class Edit:
    """What a profile's edit(e) gets: the body (Blender object, placed in metres) and helpers that take Unity points
    (x right, y up, z forward, metres, y = 0 the ground): pieces(), delete(), cut(), and new parts (box(), cylinder())
    each coloured by one of the profile's SWATCHES, painted into texture space the unwrap leaves empty."""

    def __init__(self, obj):
        self.obj = obj
        self.bm = bmesh.new(); self.bm.from_mesh(obj.data)
        self.part = self.bm.faces.layers.int['part']
        self.wheels = wheel_centres
        self.new = []   # (bmesh, swatch name)
        self.group = None   # while set, new shapes of one swatch share one bmesh (and one swatch cell)
        self._groups = {}
        self.flat_swatches = []   # the linings' swatches (face layer 'flat': index + 1)

    def _body(self):
        return [f for f in self.bm.faces if f[self.part] == 0]

    def centre(self, f):
        return to_unity(np.array(f.calc_center_median())[None])[0]

    def pieces(self):
        """The body's loose pieces: [(faces, lo, hi)] (Unity bounds)."""
        out = []
        for comp in loose_parts(self.bm):
            if comp[0][self.part] != 0: continue
            v = to_unity(np.array([tuple(l.vert.co) for f in comp for l in f.loops]))
            out.append((comp, v.min(0), v.max(0)))
        return out

    def delete(self, faces, why=''):
        faces = list({f for f in faces if f.is_valid})
        bmesh.ops.delete(self.bm, geom=faces, context='FACES')
        log(f'edit: {len(faces)} faces out' + (f' ({why})' if why else ''))

    def cut(self, region, point, normal, why=''):
        """Splits the body faces in region (a test on Unity face centres) along a plane (Unity point and normal), then
        takes out those of them on the normal's side: a straight, clean edge where a face test would leave a zigzag."""
        faces = [f for f in self._body() if region(self.centre(f))]
        if not faces: log(f'edit: cut {why}: nothing in the region'); return
        geom = list({e for f in faces for e in f.edges}) + faces + list({v for f in faces for v in f.verts})
        n = to_blender(normal) - to_blender((0, 0, 0))
        bmesh.ops.bisect_plane(self.bm, geom=geom, plane_co=to_blender(point), plane_no=n, clear_inner=False, clear_outer=False)
        side = np.asarray(normal, float)
        gone = [f for f in self._body() if region(self.centre(f)) and np.dot(self.centre(f) - np.asarray(point, float), side) > 0]
        self.delete(gone, why)

    def slice(self, region, point, normal):
        """Splits the body faces in region along a plane, taking nothing out: a clean edge for a later cut's region to
        end on."""
        faces = [f for f in self._body() if region(self.centre(f))]
        geom = list({e for f in faces for e in f.edges}) + faces + list({v for f in faces for v in f.verts})
        bmesh.ops.bisect_plane(self.bm, geom=geom, plane_co=to_blender(point), plane_no=to_blender(normal) - to_blender((0, 0, 0)))

    def drop_small(self, max_faces, why='loose fragments'):
        """Takes out the body's loose pieces of at most max_faces faces (what the cuts leave floating)."""
        gone = [f for faces, lo, hi in self.pieces() if len(faces) <= max_faces for f in faces]
        self.delete(gone, why)

    def shell(self, region, swatch, inset=0.006):
        """An inside for a hollow shell: the body faces in region copied, turned to face inward, set in by inset metres
        and coloured by one swatch (flat). Through a cut you then see a dark lining that follows the body, not
        through the car (the game draws one side of a face only)."""
        lay = self.bm.faces.layers.int.get('flat') or self.bm.faces.layers.int.new('flat')   # first: a new layer invalidates faces
        faces = [f for f in self._body() if region(self.centre(f))]
        verts = list({v for f in faces for v in f.verts}); edges = list({e for f in faces for e in f.edges})
        dup = bmesh.ops.duplicate(self.bm, geom=verts + edges + faces)
        nf = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMFace)]
        nv = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.reverse_faces(self.bm, faces=nf)
        self.bm.normal_update()
        for v in nv: v.co += v.normal * inset
        self.flat_swatches.append(swatch)
        for f in nf: f[lay] = len(self.flat_swatches)
        log(f'edit: inside lining of {len(nf)} faces')

    def wall(self, fixed, value, a0, a1, y0, ymax, swatch, step=0.03, gap=0.012, top=None):
        """A vertical panel (both sides) at Unity x = value (fixed 'x', running along z from a0 to a1) or z = value (fixed
        'z', running along x), from y0 up to the body's underside: its top follows the skin (raycast down, gap below
        it), never higher than ymax (nor top(x, z), when given). y0 may be a function of (x, z) (an arch over a
        wheel). For closing a hollow
        shell behind a cut (wheel tubs)."""
        from mathutils.bvhtree import BVHTree
        tree = BVHTree.FromBMesh(self.bm)
        bottom = y0 if callable(y0) else (lambda x, z: y0)
        top_fn = top
        cols = []
        for t in np.linspace(a0, a1, max(2, int(abs(a1 - a0) / step) + 1)):
            x, z = (value, t) if fixed == 'x' else (t, value)
            top = ymax
            hit = tree.ray_cast(to_blender((x, 3.0, z)), Vector((0, 0, -1)))
            if hit[0] is not None: top = min(top, hit[0][2] - gap)
            if top_fn is not None: top = min(top, top_fn(x, z))
            b = bottom(x, z)
            cols.append((x, z, b, max(top, b + 0.01)))
        def build(bm):
            for back in (False, True):   # each side its own vertices (bmesh keeps one face per vertex set)
                vs = [(bm.verts.new(to_blender((x, b, z))), bm.verts.new(to_blender((x, top, z)))) for x, z, b, top in cols]
                for (a, b), (c, d) in zip(vs, vs[1:]):
                    bm.faces.new((b, d, c, a) if back else (a, c, d, b))
        self._shape(swatch, build)

    def _shape(self, swatch, build):
        bm = bmesh.new(); build(bm)
        if self.group is None: self.new.append((bm, swatch)); return
        key = (self.group, swatch)
        if key not in self._groups:
            self._groups[key] = bm; self.new.append((bm, swatch)); return
        me_ = bpy.data.meshes.new('grp'); bm.to_mesh(me_); bm.free()
        self._groups[key].from_mesh(me_); bpy.data.meshes.remove(me_)

    def box(self, centre, size, swatch, bevel=0.0):
        """An axis-aligned box (Unity centre and size)."""
        def build(bm):
            m = Matrix.Translation(to_blender(centre)) @ Matrix.Diagonal((size[0], size[2], size[1], 1.0))
            bmesh.ops.create_cube(bm, size=1.0, matrix=m)
            if bevel > 0: bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, affect='EDGES')
        self._shape(swatch, build)

    def cylinder(self, centre, radius, length, axis, swatch, segments=20, radius2=None, caps=True):
        """A cylinder (or cone, radius2) along a Unity axis 'x', 'y' or 'z'."""
        rot = {'x': Matrix.Rotation(math.pi / 2, 4, 'Y'), 'y': Matrix.Identity(4), 'z': Matrix.Rotation(math.pi / 2, 4, 'X')}[axis]
        def build(bm):
            bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=segments, radius1=radius,
                                  radius2=radius if radius2 is None else radius2, depth=length,
                                  matrix=Matrix.Translation(to_blender(centre)) @ rot)
        self._shape(swatch, build)

    def tube(self, path, radius, swatch, bend=0.0, segments=12, caps=True, closed=False, step=0.0, smooth=1):
        """A round tube along a Unity polyline: each corner rounded to an arc of radius bend (metres, clamped to the
        legs), the profile carried along without twisting. step: resample the path at about this spacing first (for
        dense or uneven paths such as a cut's edge)."""
        pts = np.asarray(path, float)
        if closed and np.allclose(pts[0], pts[-1]): pts = pts[:-1]
        if bend > 0: pts = _fillet(pts, bend, closed)
        if step > 0: pts = _resample(pts, step, closed, smooth)
        if len(pts) < 2: return
        frames = _frames(pts, closed)
        def build(bm):
            ring = []
            for p, (t, n, b) in zip(pts, frames):
                ring.append([bm.verts.new(to_blender(p + radius * (math.cos(a) * n + math.sin(a) * b)))
                             for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)])
            pairs = list(zip(ring, ring[1:])) + ([(ring[-1], ring[0])] if closed else [])
            for r0, r1 in pairs:
                for k in range(segments):
                    k1 = (k + 1) % segments
                    bm.faces.new((r0[k], r0[k1], r1[k1], r1[k]))
            if caps and not closed:
                bm.faces.new(list(reversed(ring[0]))); bm.faces.new(ring[-1])
            bm.normal_update(); bm.faces.ensure_lookup_table()
            f = bm.faces[0]   # a side face: outward is away from the path (to_blender is a mirror, so check)
            if (f.calc_center_median() - to_blender(pts[0])).dot(f.normal) < 0:
                bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
        self._shape(swatch, build)

    def poly(self, verts, faces, swatch, thickness=0.0, out=None):
        """Faces from Unity vertices, turned to face along out (a Unity direction) when given; thickness > 0 makes it
        a solid plate (pushed back from the faces by it)."""
        def build(bm):
            vs = [bm.verts.new(to_blender(v)) for v in verts]
            fs = [bm.faces.new([vs[i] for i in f]) for f in faces]
            bm.normal_update()
            if out is not None:
                o = to_blender(out) - to_blender((0, 0, 0))
                flip = [f for f in fs if f.normal.dot(o) < 0]
                if flip: bmesh.ops.reverse_faces(bm, faces=flip); bm.normal_update()
            if thickness > 0:
                bmesh.ops.solidify(bm, geom=fs, thickness=thickness)
        self._shape(swatch, build)

    def skin_x(self, y, z, sx):
        """The skin's outer x at height y, along-car z, on side sx (raycast in from outside), or None."""
        from mathutils.bvhtree import BVHTree
        hit = BVHTree.FromBMesh(self.bm).ray_cast(to_blender((sx * 2.0, y, z)), to_blender((-sx, 0, 0)) - to_blender((0, 0, 0)))
        return None if hit[0] is None else float(to_unity(np.array(hit[0])[None])[0][0])

    def bulkhead(self, z, x_inner, y0, y1, swatch, sx, step=0.02, gap=0.004):
        """A panel in the plane z = value closing the body shell's open end there, on one side (sx -1 or 1): from
        x_inner out to the skin (raycast in from outside at each height, gap short of it), heights y0 to y1 where the
        ray finds the skin. Both faces, so it reads from either side."""
        from mathutils.bvhtree import BVHTree
        tree = BVHTree.FromBMesh(self.bm)
        cols = []
        for y in np.arange(y0, y1 + 1e-6, step):
            hit = tree.ray_cast(to_blender((sx * 2.0, y, z)), to_blender((-sx, 0, 0)) - to_blender((0, 0, 0)))
            if hit[0] is None: continue
            xs = to_unity(np.array(hit[0])[None])[0][0]
            if sx * xs <= sx * x_inner + 0.01: continue
            cols.append((y, xs - sx * gap))
        if len(cols) < 2: log(f'edit: bulkhead at z {z}: no skin found'); return
        def build(bm):
            for back in (False, True):
                vs = [(bm.verts.new(to_blender((x_inner, y, z))), bm.verts.new(to_blender((xo, y, z)))) for y, xo in cols]
                for (a, b), (c, d) in zip(vs, vs[1:]):
                    q = (a, b, d, c) if (sx > 0) != back else (a, c, d, b)
                    bm.faces.new(q)
        self._shape(swatch, build)

    def edges(self, region):
        """The skin's open edges (one body face, not a lining) whose middles pass region (Unity test), as chains of
        Unity points with the skin's outward normal at each: [(points (N,3), normals (N,3), closed)]."""
        self.bm.normal_update()
        flat = self.bm.faces.layers.int.get('flat')
        skin = lambda f: f[self.part] == 0 and (flat is None or f[flat] == 0)
        nb = {}
        for e in self.bm.edges:
            fs = [f for f in e.link_faces if skin(f)]
            if len(fs) != 1: continue
            m = to_unity(np.array((e.verts[0].co + e.verts[1].co) / 2)[None])[0]
            if not region(m): continue
            a, b = e.verts
            nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
        out, used = [], set()
        def walk(start):
            chain = [start]; prev = None; cur = start
            while True:
                nxt = [v for v in nb[cur] if v is not prev and (min(cur.index, v.index), max(cur.index, v.index)) not in used]
                if not nxt: return chain, False
                v = nxt[0]; used.add((min(cur.index, v.index), max(cur.index, v.index)))
                if v is start: return chain, True
                chain.append(v); prev, cur = cur, v
                if len(nb[cur]) != 2: return chain, False
        self.bm.verts.index_update()
        for v in sorted(nb, key=lambda v: len(nb[v]) == 2):   # chain ends (and branch points) first
            while any((min(v.index, w.index), max(v.index, w.index)) not in used for w in nb[v]):
                chain, closed = walk(v)
                if len(chain) < 2: break
                pts = to_unity(np.array([tuple(c.co) for c in chain]))
                nrm = to_unity(np.array([tuple(c.normal) for c in chain]))   # direction only: to_unity is linear
                out.append((pts, nrm, closed))
        return out

    def lip(self, region, radius, swatch, inset=0.003, step=0.025, min_len=0.06, smooth=6):
        """A rolled edge along the skin's open edges in region: a tube just inside the skin's edge, thick enough to
        close the gap to the lining, so a cut reads as a rounded lip instead of a paper-thin one."""
        n_ch = 0
        for pts, nrm, closed in self.edges(region):
            if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < min_len: continue
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            self.tube(pts - nrm * inset, radius, swatch, segments=8, closed=closed, step=step, caps=True, smooth=smooth)
            n_ch += 1
        log(f'edit: lip along {n_ch} edge chains')

    def cut_plan(self, region, zw, why=''):
        """Takes out the body faces in region outside a top-view curve through (z, half-width) points (z rising), on
        both sides: vertical cuts, so the edge follows the skin's steep sides cleanly (a near-level cut grazing a
        near-level surface wanders)."""
        zw = np.asarray(zw, float)
        for sx in (-1, 1):
            reg = lambda c, sx=sx: region(c) and sx * c[0] > 0
            for z in zw[1:-1, 0]:
                self.slice(lambda c, z=z: reg(c) and abs(c[2] - z) < 0.2, (0, 0, z), (0, 0, 1))
            for (z0, w0), (z1, w1) in zip(zw, zw[1:]):
                n = np.array([sx * (z1 - z0), 0.0, -(w1 - w0)])   # square to the segment, outward (towards sx)
                n /= np.linalg.norm(n)
                self.cut(lambda c, z0=z0, z1=z1, reg=reg: reg(c) and z0 - 1e-4 <= c[2] <= z1 + 1e-4,
                         np.array([sx * w0, 0.0, z0]), n, f'{why} {z0:.2f}..{z1:.2f}')

    def cut_curve(self, region, pts, why='', side=-1, axis='z'):
        """Takes out the body faces in region on one side of a curve through (a, y) points (a rising): a is Unity z
        (axis 'z', a side-view curve) or |x| (axis 'x', a front-view curve, mirrored to both sides); side -1 below the
        curve, +1 above it. The faces are first split at each point's a, then each span cut along its own chord, so
        the edge follows the curve with no notches at the joins."""
        pts = np.asarray(pts, float)
        sides = (1,) if axis == 'z' else (-1, 1)
        for sx in sides:
            ax = (lambda c: c[2]) if axis == 'z' else (lambda c, sx=sx: sx * c[0])
            unit = (lambda a: np.array([0.0, 0.0, a])) if axis == 'z' else (lambda a, sx=sx: np.array([sx * a, 0.0, 0.0]))
            for a in pts[1:-1, 0]:
                self.slice(lambda c, a=a, ax=ax: region(c) and abs(ax(c) - a) < 0.2, tuple(unit(a)), tuple(unit(1.0)))
            for (a0, y0), (a1, y1) in zip(pts, pts[1:]):
                d = unit(a1 - a0) + np.array([0.0, y1 - y0, 0.0])
                along = d / np.linalg.norm(d)
                n = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], along) * along   # in the curve's plane, square to it
                n /= np.linalg.norm(n)
                if side < 0: n = -n
                lo_, hi_ = min(a0, a1), max(a0, a1)
                self.cut(lambda c, lo_=lo_, hi_=hi_, ax=ax: region(c) and lo_ - 1e-4 <= ax(c) <= hi_ + 1e-4,
                         unit(a0) + np.array([0.0, y0, 0.0]), n, f'{why} {a0:.2f}..{a1:.2f}')

    def finish(self):
        self.bm.to_mesh(self.obj.data); self.bm.free()
        if not self.new and not self.flat_swatches: return
        cell, gap = getattr(P, 'SWATCH_CELL', 192), 12
        cells = free_cells(self.obj.data, cell, gap)
        need = len(self.new) + len(self.flat_swatches)
        assert len(cells) >= need, f'{need} swatch cells needed, room for {len(cells)}: smaller SWATCH_CELL'
        px = np.array(src_img.pixels[:], np.float32).reshape(TEX, TEX, 4)
        # the linings: every corner on its cell's middle (one flat colour; their AO is set flat after the bake)
        me_ = self.obj.data
        if self.flat_swatches:
            fl = pl.get_face_attr(me_, 'flat')
            uvl = me_.uv_layers['src']
            uv = np.zeros(len(me_.loops) * 2); uvl.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
            for k, sw in enumerate(self.flat_swatches, 1):
                x0, y0 = cells.pop()
                FLAT_CELLS.append((x0, y0, cell)); ADDED_CELLS.append((x0, y0, cell))
                for p_ in me_.polygons:
                    if fl[p_.index] == k: uv[p_.loop_start:p_.loop_start + p_.loop_total] = (np.array([x0, y0]) + cell / 2) / TEX
                px[max(y0 - gap // 2, 0):y0 + cell + gap // 2, max(x0 - gap // 2, 0):x0 + cell + gap // 2, :3] = np.array(P.SWATCHES[sw], np.float32) / 255
            uvl.data.foreach_set('uv', uv.ravel())
        objs = []
        for i, (bm, sw) in enumerate(self.new):
            me2 = bpy.data.meshes.new(f'new{i}'); bm.to_mesh(me2); bm.free()
            o = bpy.data.objects.new(f'new{i}', me2); pl.scene().collection.objects.link(o)
            me2.uv_layers.new(name='src')
            pl.select_only(o); bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(island_margin=0.03); bpy.ops.object.mode_set(mode='OBJECT')
            x0, y0 = cells[i]; m = 6
            log(f'  part {i} {sw}: cell {x0},{y0}, {len(o.data.polygons)} faces')
            ADDED_CELLS.append((x0, y0, cell))
            uvl = o.data.uv_layers['src']   # fetched again after the mode switch (a stale layer crashes)
            uv = np.zeros(len(o.data.loops) * 2); uvl.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
            uv = (np.array([x0 + m, y0 + m]) + uv * (cell - 2 * m)) / TEX
            uvl.data.foreach_set('uv', uv.ravel())
            px[max(y0 - gap // 2, 0):y0 + cell + gap // 2, max(x0 - gap // 2, 0):x0 + cell + gap // 2, :3] = np.array(P.SWATCHES[sw], np.float32) / 255
            pl.face_attr(o.data, 'part', [0] * len(o.data.polygons))
            objs.append(o)
        src_img.pixels.foreach_set(px.ravel())
        pl.select_only(self.obj)
        for o in objs: o.select_set(True)
        bpy.ops.object.join()
        log(f'edit: {len(objs)} new parts on swatches ({len(cells)} free cells)')


if hasattr(P, 'edit'):
    e = Edit(src)
    P.edit(e, log)
    e.finish()
    me = src.data

if os.environ.get('OTH_EDIT_ONLY'):
    # quick textured renders of the edited body, for iterating on edit(): no bake, no export
    mat = bpy.data.materials.new('quick'); mat.use_nodes = True
    t = mat.node_tree.nodes.new('ShaderNodeTexImage'); t.image = src_img
    uvn = mat.node_tree.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'src'
    mat.node_tree.links.new(uvn.outputs['UV'], t.inputs['Vector'])
    mat.node_tree.links.new(t.outputs['Color'], mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    me.materials.clear(); me.materials.append(mat)
    sc = pl.scene(); sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'TEXTURE'; sc.display.shading.show_cavity = True
    sc.display.shading.show_backface_culling = True   # as the game draws
    sc.world = bpy.data.worlds.new('grey'); sc.world.color = (0.35, 0.37, 0.4)   # a background the black parts stand out on
    sc.render.resolution_x, sc.render.resolution_y = 1400, 900
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    for nm, az, el, d, tz in (('front34', 35, 15, 8.5, 0), ('rear34', 150, 20, 8.5, 0), ('side', 90, 3, 8.5, 0), ('rear', 180, 10, 6.5, 0),
                              ('front', 0, 8, 6.5, 0), ('under', 90, -60, 8.5, 0), ('rear_low', 160, 2, 5.5, 0),
                              ('close_front34', 40, 12, 3.6, 1.5), ('close_rear34', 140, 15, 3.6, -1.6),
                              ('close_front_low', 15, -4, 3.4, 1.6), ('close_side_rear', 75, 25, 3.0, -1.2)):
        a_, e_ = math.radians(az), math.radians(el)
        dv = (Vector((0, -S, 0)) * math.cos(a_) + Vector((S, 0, 0)) * math.sin(a_)) * math.cos(e_) + Vector((0, 0, math.sin(e_)))
        tgt = to_blender((0, 0.7, tz)); cam.location = tgt + dv * d
        cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(OUT, f'edit_{nm}.png'); bpy.ops.render.render(write_still=True)
    # orthographic, to measure from: 280 px per metre, image centre at Unity (y 0.85, z 0) (side) / (x 0, z 0) (top)
    cam.data.type = 'ORTHO'; cam.data.ortho_scale = 1400 / 280
    for nm, loc, rot in (('ortho_side', Vector((20 * S, 0, 0.85)), Vector((-1, 0, 0))), ('ortho_top', Vector((0, 0, 20)), Vector((0, 0, -1)))):
        cam.location = loc
        cam.rotation_euler = rot.to_track_quat('-Z', 'Y' if nm == 'ortho_side' else 'X').to_euler()
        if nm == 'ortho_top': cam.rotation_euler = Vector((0, 0, -1)).to_track_quat('-Z', 'X').to_euler()
        sc.render.filepath = os.path.join(OUT, f'edit_{nm}.png'); bpy.ops.render.render(write_still=True)
    src_img.filepath_raw = os.path.join(OUT, 'edit_texture.png'); src_img.file_format = 'PNG'; src_img.save()
    log('edit-only renders written'); raise SystemExit(0)

# the atlas is the source unwrap
me.uv_layers.new(name='atlas')
uvs = np.zeros(len(me.loops) * 2); me.uv_layers['src'].data.foreach_get('uv', uvs); me.uv_layers['atlas'].data.foreach_set('uv', uvs)
me.uv_layers.active = me.uv_layers['atlas']; me.uv_layers['atlas'].active_render = True
src.name = 'atlas'
part = pl.get_face_attr(me, 'part')
log(f'faces: body {(part == 0).sum()}, wheel {(part == 2).sum()}, tyre {(part == 3).sum()}')

# ---------------------------------------------------------------- texel positions, normals and parts
co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
lv = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
me.calc_normals_split()
ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
upos, unrm = to_unity(co[lv]), to_unity(ln)
lo, hi = upos.min(0), upos.max(0)
loop_face = np.zeros(len(me.loops), np.int64)
for p in me.polygons: loop_face[p.loop_start:p.loop_start + p.loop_total] = p.index


def corner_colour(name, rgb):
    a = me.color_attributes.new(name, 'FLOAT_COLOR', 'CORNER')
    rgba = np.ones((len(me.loops), 4), np.float32); rgba[:, :3] = rgb; a.data.foreach_set('color', rgba.ravel())


corner_colour('c_pos', (upos - lo) / (hi - lo))
corner_colour('c_nrm', unrm * 0.5 + 0.5)
corner_colour('c_part', np.stack([part[loop_face] / 4.0, np.zeros(len(lv)), np.zeros(len(lv))], 1))

baker = pl.Baker(src, OUT, NAME, TEX, ao_distance=P.AO_DISTANCE, albedo_textures=())


def bake_float(attr):
    """An attribute baked into texture space as floats (not saved), with a coverage channel."""
    img = bpy.data.images.new('f_' + attr, TEX, TEX, alpha=True, float_buffer=True)
    img.colorspace_settings.name = 'Non-Color'
    for m in (baker.attr, baker.albedo): m.node_tree.nodes['target'].image = img
    baker.attr.node_tree.nodes['attr'].attribute_name = attr
    me.polygons.foreach_set('material_index', [0] * len(me.polygons)); me.update()
    pl.scene().cycles.samples = 1
    pl.scene().render.bake.margin = 4
    pl.select_only(src); bpy.ops.object.bake(type='EMIT')
    pl.scene().render.bake.margin = 16
    a = np.array(img.pixels[:], np.float32).reshape(TEX, TEX, 4)
    bpy.data.images.remove(img)
    return a


pos = bake_float('c_pos')
covered = pos[..., :3].sum(-1) > 0          # bake clears to black: (0,0,0) is the bounding box corner, never a texel
pos = lo + pos[..., :3] * (hi - lo)
nrm = bake_float('c_nrm')[..., :3] * 2 - 1
tpart = np.round(bake_float('c_part')[..., 0] * 4).astype(np.int64)
log(f'texels covered: {covered.mean():.1%}')

# ---------------------------------------------------------------- texel classes
srgb = np.array(src_img.pixels[:], np.float32).reshape(TEX, TEX, 4)[..., :3]   # an 8-bit image's pixels: its sRGB values


def lin_to_srgb(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, 1), 1 / 2.4) - 0.055)


added = np.zeros((TEX, TEX), bool)
for x0, y0, sz in ADDED_CELLS: added[max(y0 - 6, 0):y0 + sz + 6, max(x0 - 6, 0):x0 + sz + 6] = True
cls = P.classify(srgb, pos, nrm, (tpart == PART_IDS['body']) & ~added)   # body: the source's own body texels
cls = np.where(tpart == PART_IDS['tyre'], TYRE, np.where(tpart == PART_IDS['wheel'], RIM, cls))
cls[~covered] = OTHER
for c in np.unique(cls[covered]):
    log(f'class {pl.CLASS_NAMES[c]}: {(cls[covered] == c).mean():.1%} of the covered texels')

# a picture of the classes, for checking the profile's rules
CLASS_RGB = {PAINT: (230, 200, 30), TRIM: (40, 40, 160), DARK: (20, 20, 20), METAL: (170, 170, 170), LENS: (60, 220, 255),
             EMIT: (255, 255, 255), REDLENS: (230, 20, 20), AMBER: (255, 140, 0), DETAIL: (120, 70, 30), GLASS: (40, 90, 220),
             TYRE: (60, 60, 60), RIM: (210, 210, 230), INTERIOR: (120, 0, 120), OTHER: (255, 0, 255)}
vis = np.zeros((TEX, TEX, 4), np.float32); vis[..., 3] = 1
for c, col in CLASS_RGB.items(): vis[cls == c, :3] = np.array(col) / 255
vis[~covered, :3] = 0.15
vimg = bpy.data.images.new('classes', TEX, TEX); vimg.pixels.foreach_set(vis.ravel()); baker.save(vimg, 'classes')

# ---------------------------------------------------------------- maps
front, rear = pos[..., 2] > 0, pos[..., 2] < 0
lamp = np.isin(cls, pl.LAMP_CLASSES)


def save_map(name, arr, colour=False):
    img = bpy.data.images.new(name, TEX, TEX, alpha=False)
    img.colorspace_settings.name = 'sRGB' if colour else 'Non-Color'
    a = np.ones((TEX, TEX, 4), np.float32); a[..., :3] = np.clip(arr, 0, 1); img.pixels.foreach_set(a.ravel())
    baker.save(img, name)
    return img


albedo = srgb.copy()   # written as an sRGB image: sRGB values throughout
for c, lin in P.ALBEDO_OVERRIDE.items(): albedo[cls == c] = lin_to_srgb(lin)
if getattr(P, 'TYRE_DARKEN', None): albedo[cls == TYRE] *= P.TYRE_DARKEN
if hasattr(P, 'albedo'): albedo = P.albedo(albedo, srgb, cls, pos, nrm, ~added)

zones = np.zeros((TEX, TEX, 3), np.float32)
zones[..., 0] = (cls == PAINT) * P.paint_texels(albedo)   # the final albedo: a profile may repaint texels
# Under the paint zone the base shows wherever the zone is partial (seams, the zone's edges, mip levels): the source's
# own paint colour there mixes with the player's (yellow under blue reads green). PAINT_BASE_GREY: the paint
# texels' base turned grey, keeping their lightness (times the factor), so the seams stay dark and nothing tints.
if getattr(P, 'PAINT_BASE_GREY', None) is not None:
    pt = (cls == PAINT) | (~covered & (P.paint_texels(albedo) > 0))   # and the paint round the islands (filtering reads it)
    lum = (0.3 * albedo[..., 0] + 0.59 * albedo[..., 1] + 0.11 * albedo[..., 2]) * P.PAINT_BASE_GREY
    albedo[pt] = lum[pt][:, None]
albedo_img = save_map('albedo_base', albedo, colour=True)
zones[..., 1] = cls == TRIM
zones_img = save_map('zones', zones)

mask = np.zeros((TEX, TEX, 3), np.float32)
mask[..., 0] = (cls == GLASS) & front & (nrm[..., 2] > 0.3)                    # the windscreen
mask[..., 1] = lamp | (cls == GLASS)                                           # glossy
mask[..., 2] = ~lamp & ~np.isin(cls, [GLASS, TYRE])                            # dirt, mud, snow, wear
save_map('mask', mask)

lights = np.zeros((TEX, TEX, 3), np.float32)
lights[..., 0] = rear & (cls == REDLENS)
lights[..., 1] = front & np.isin(cls, [EMIT, LENS])
lights_img = save_map('lights', lights)

ao_img = baker.bake('ao', 'AO', samples=P.AO_SAMPLES)
aoa = pl.pixels(ao_img, TEX); aoa[lamp, :3] = 1.0
for x0, y0, sz in FLAT_CELLS: aoa[y0:y0 + sz, x0:x0 + sz, :3] = getattr(P, 'LINING_AO', 0.6)
ao_img.pixels.foreach_set(aoa.ravel()); ao_img.save()
log(f'paint zone: {(zones[..., 0] > 0.5).mean():.1%} of the atlas')

# ---------------------------------------------------------------- parts, LODs, collider, export
parts = pl.split_parts(src, PART_IDS)
lods = pl.make_lods(parts['body'], P.LOD_TRIS, report_tris=True)
coll = pl.hull_collider(parts['body'].data)
hub = wheel_centres['FL']
out_parts = [('body', 0, lods[0], np.zeros(3))]
out_parts += [(f'body_lod{i}', 0, o, np.zeros(3)) for i, o in enumerate(lods[1:], 1)]
if len(parts['glass'].data.polygons): out_parts.append(('glass', 1, parts['glass'], np.zeros(3)))
out_parts += [('wheel', 2, parts['wheel'], hub), ('tyre', 3, parts['tyre'], hub), ('collider', 4, coll, np.zeros(3))]
arrays = pl.export_parts(out_parts, to_unity, math.radians(getattr(P, 'SMOOTH_DEG', 35)))
tyre_P = arrays['tyre'][2]
log(f'tyre radius {np.hypot(tyre_P[:, 1], tyre_P[:, 2]).max():.3f} m, width {np.ptp(tyre_P[:, 0]):.3f} m')

# ---------------------------------------------------------------- anchors
anchors, ref = pl.map_ref_anchors(REF, arrays, wheel_centres, prefixes=('cosm_', 'fog_', 'exhaust_', 'engine_smoke'))
bpos = arrays['body'][2]
tRL = min(c[2] for c in wheel_centres.values())
ex = P.exhaust(bpos, tRL) if hasattr(P, 'exhaust') else None
if ex is not None:
    rot = pl.yaw_quat(P.EXHAUST_YAW) if getattr(P, 'EXHAUST_YAW', None) is not None else next((r for n, (p_, r) in ref.items() if n.startswith('exhaust_')), BACKQ)
    for n in [n for n in anchors if n.startswith('exhaust_')]: del anchors[n]
    anchors['exhaust_L' if ex[0] < 0 else 'exhaust_R'] = (ex, rot)
    log(f'anchor exhaust: {np.round(ex, 3)}')
for cn, c in wheel_centres.items():
    anchors['wheel_' + cn] = (c, IDQ if cn.endswith('L') else BACKQ)


def centroid(sel):
    return pos[sel].mean(0) if sel.any() else None


for side, sgn in (('L', -1), ('R', 1)):
    sd = np.sign(pos[..., 0]) == sgn
    h = centroid(front & np.isin(cls, [EMIT, LENS]) & sd)
    t = centroid(rear & (cls == REDLENS) & sd)
    if h is not None: anchors['headlight_' + side] = (h + np.array([0, 0, getattr(P, 'LAMP_FORWARD', 0.03)]), IDQ)
    if t is not None: anchors['tail_' + side] = (t - np.array([0, 0, getattr(P, 'LAMP_FORWARD', 0.03)]), BACKQ)
    log(f'lamps {side}: head {None if h is None else np.round(h, 3)}, tail {None if t is None else np.round(t, 3)}')
tl_ = centroid(rear & (cls == REDLENS))
if tl_ is not None: anchors['reverse'] = (np.array([0.0, tl_[1], tl_[2] - 0.03]), BACKQ)   # no reverse lamp: between the tails
pl.report_missing(anchors)
mesh_path = os.path.join(OUT, f'{NAME}.mesh')
pl.write_mesh(mesh_path, arrays, anchors)
import colliders
colliders.apply(mesh_path, getattr(P, 'COLLIDER_SLICES', 6))

# ---------------------------------------------------------------- previews
pv = pl.Previews(OUT, TEX, albedo_img, zones_img, ao_img, lights_img, P.PREVIEW_PAINT, P.PREVIEW_TRIM, S, uv_map='atlas')
pv.use([lods[0], parts['wheel'], parts['tyre']], set_uv=True)
for o in lods[1:] + [coll, parts['glass']]: o.hide_render = True
standin = getattr(P, 'PREVIEW_STANDIN_WHEEL', None)   # (radius, width): the donor's wheel, when the car keeps it
if standin:
    parts['wheel'].hide_render = True; parts['tyre'].hide_render = True
    mat = bpy.data.materials.new('standin'); mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1)
    for cn, c in wheel_centres.items():
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=standin[0], depth=standin[1],
                                            location=(-S * c[0], -S * c[2], c[1]), rotation=(0, math.pi / 2, 0))
        bpy.context.active_object.data.materials.append(mat)
else:
    pl.mirror_wheels(parts, wheel_centres, hub, 'FL', lambda k: k.endswith('R'), S)
pv.shoot('front34', 35, 15, 8.5, (0, 0, 0.75)); pv.shoot('rear34', 150, 20, 8.5, (0, 0, 0.75))
pv.shoot('side', 90, 5, 8.5, (0, 0, 0.75)); pv.shoot('top', 60, 55, 8.5, (0, 0, 0.75))
pv.shoot('close_front', 0, 8, 3.4, (0, -S * 1.6, 0.7)); pv.shoot('close_rear', 180, 8, 3.4, (0, S * 1.6, 0.8))
pv.lit_look(True)
pv.shoot('lit_front', 0, 8, 6.0, (0, 0, 0.75)); pv.shoot('lit_rear', 180, 12, 6.0, (0, 0, 0.75))
pv.lit_look(False)
for i, o in enumerate(lods[1:], 1):
    lods[i - 1].hide_render = True; o.hide_render = False
    pv.use([o], set_uv=True)
    pv.shoot(f'side_lod{i}', 90, 5, 8.5, (0, 0, 0.75))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f'{NAME}_build.blend'))
log('done')
