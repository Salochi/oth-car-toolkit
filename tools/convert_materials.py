"""
Converter for models coloured by their materials (one flat colour per material, e.g. a detailed Sketchfab car):
GLB -> <name>.mesh + <name>_albedo_base / _zones / _mask / _lights / _ao PNGs + preview renders.

Run:  blender -b --factory-startup --python tools/convert_materials.py -- <profile.py> <model.glb> <out_dir> <reference.mesh> [name]

The profile (profiles/*.py, see profiles/example_materials.py) says what the source's materials and objects are: which
material is paint, trim, glass, a lamp...; which objects are the wheels, tyres and bumpers; what to drop. The reference
.mesh is a built car of a similar size whose cosmetic anchors are mapped onto this body.
"""
import bpy, bmesh, sys, os, math, importlib.util
import numpy as np
from mathutils import Vector

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import pipeline as pl
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER, IDQ, BACKQ, log

argv = sys.argv[sys.argv.index('--') + 1:]
spec = importlib.util.spec_from_file_location('profile', argv[0]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GLB, OUT, REF = argv[1], os.path.abspath(argv[2]), argv[3]
NAME = argv[4] if len(argv) > 4 else P.NAME
os.makedirs(OUT, exist_ok=True)
TEX = getattr(P, 'TEX', 2048)
LOD_TRIS = [int(os.environ.get('LOD0', 10**9))] + list(P.LOD_TRIS)


def mat_class(name):
    for sub, c in P.CLASS_RULES:
        if sub.lower() in (name or '').lower():
            return c
    return OTHER


def mat_color(m):
    if m is None or not m.use_nodes:
        return (0.5, 0.5, 0.5)
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            inp = n.inputs['Base Color']
            if inp.is_linked:
                src = inp.links[0].from_node
                if src.type == 'TEX_IMAGE' and src.image and src.image.size[0] > 0:
                    px = np.array(src.image.pixels[:]).reshape(-1, 4)[:, :3]
                    return tuple(px.mean(0))
                return (0.3, 0.3, 0.3)
            return tuple(inp.default_value[:3])
    return (0.5, 0.5, 0.5)


# ---------------------------------------------------------------- import, flatten to world space
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
scene = bpy.context.scene
# animated objects (doors, a tailgate): frozen at the frame where they are least rotated (closed)
for o in list(bpy.data.objects):
    ad = o.animation_data
    if ad and ad.action:
        f0, f1 = (int(x) for x in ad.action.frame_range)
        best = None
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            ang = o.matrix_local.to_quaternion().angle
            if best is None or ang < best[0]:
                best = (ang, f)
        scene.frame_set(best[1])
        mw = o.matrix_local.copy()
        o.animation_data_clear()
        o.matrix_local = mw
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
flat = []
for o in list(scene.objects):
    if o.type != 'MESH':
        continue
    chain, p = [], o
    while p:
        chain.append(p.name); p = p.parent
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    # carry the source's split normals through the world transform ourselves (Mesh.transform doesn't, which leaves
    # objects under rotated empties, e.g. doors on their hinges, with wrong normals)
    me.calc_normals_split()
    ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
    nm3 = np.array(o.matrix_world.to_3x3().inverted_safe().transposed())
    ln = ln @ nm3.T
    ln /= np.maximum(np.linalg.norm(ln, axis=1, keepdims=True), 1e-9)
    me.transform(o.matrix_world)
    me.use_auto_smooth = True
    me.normals_split_custom_set(ln.tolist())
    flat.append((me, chain))
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def wheel_tag(chain):
    for n in chain:
        for kind in (P.TYRE_PREFIX, P.WHEEL_PREFIX):
            for corner in P.CORNERS:
                if n.startswith(f'{kind}_{corner}'):
                    return kind, corner
    return None


# front direction: front bumper vs rear bumper (Blender y)
def centre(me):
    v = np.array([x.co for x in me.vertices]); return v.mean(0) if len(v) else np.zeros(3)
fy = [centre(me)[1] for me, ch in flat if any(n.startswith(P.FRONT_OBJECT) for n in ch)]
ry = [centre(me)[1] for me, ch in flat if any(n.startswith(P.REAR_OBJECT) for n in ch)]
S = 1.0 if np.mean(fy) < np.mean(ry) else -1.0   # +1: front is Blender -y
log(f'front is Blender {"-" if S > 0 else "+"}y')


def to_unity(v):  # Blender (x,y,z) -> Unity (x right, y up, z forward); a mirror, fixed by the winding check later
    return np.stack([-S * v[..., 0], v[..., 2], -S * v[..., 1]], -1)


# ---------------------------------------------------------------- sort into parts
groups = {'body': [], 'glass': [], 'wheel': [], 'tyre': []}
exhaust_pts = []
wheel_centres = {}
for me, chain in flat:
    if any(n.startswith(P.DROP_OBJECTS) for n in chain):
        log('drop object', chain[0], len(me.polygons), 'faces')
        bpy.data.meshes.remove(me); continue
    for prefix, sub, repl in P.MATERIAL_SWAPS:
        if any(n.startswith(prefix) for n in chain):
            to = next(m for m in bpy.data.materials if repl in m.name)
            for i, m in enumerate(me.materials):
                if m and sub in m.name: me.materials[i] = to
    classes = [mat_class(m.name if m else '') for m in me.materials]
    dropped = {INTERIOR} | ({LENS} if any(n.startswith(P.NO_LENS_OBJECTS) for n in chain) else set())
    if dropped & set(classes):
        bm = bmesh.new(); bm.from_mesh(me)
        gone = [f for f in bm.faces if classes[f.material_index] in dropped]
        log('drop interior faces', chain[0], len(gone), 'of', len(bm.faces))
        bmesh.ops.delete(bm, geom=gone, context='FACES'); bm.to_mesh(me); bm.free()
        if len(me.polygons) == 0:
            bpy.data.meshes.remove(me); continue
    if P.EXHAUST_OBJECT and any(n.startswith(P.EXHAUST_OBJECT) for n in chain):
        exhaust_pts.append(to_unity(np.array([v.co for v in me.vertices])))
    tag = wheel_tag(chain)
    if tag:
        kind, corner = tag
        if kind == P.TYRE_PREFIX:
            c = to_unity(centre(me)[None])[0]
            wheel_centres[corner] = c
        groups.setdefault(f'{kind}_{corner}', []).append(me)
        continue
    lamp = 1 if any(n.startswith(P.LAMP_OBJECTS) for n in chain) else 0
    la = me.attributes.new('lamp', 'INT', 'FACE'); la.data.foreach_set('value', [lamp] * len(me.polygons))
    groups['body'].append(me)


def unity_corner(c):   # which named corner is really FL in Unity (x<0, z>0)
    return ('F' if c[2] > 0 else 'R') + ('L' if c[0] < 0 else 'R')
corner_of = {k: unity_corner(c) for k, c in wheel_centres.items()}
log('wheels', {k: (corner_of[k], np.round(c, 3)) for k, c in wheel_centres.items()})
fl_name = [k for k, v in corner_of.items() if v == 'FL'][0]

body = pl.make_obj('body', groups['body'])
wheel = pl.make_obj('wheel', groups[f'{P.WHEEL_PREFIX}_{fl_name}'])
tyre = pl.make_obj('tyre', groups[f'{P.TYRE_PREFIX}_{fl_name}'])
for k in list(groups):
    if k.startswith((P.WHEEL_PREFIX + '_', P.TYRE_PREFIX + '_')) and fl_name not in k:
        for me in groups[k]: bpy.data.meshes.remove(me)

# glass out of the body (by material)
pl.select_only(body)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='DESELECT'); bpy.ops.object.mode_set(mode='OBJECT')
for poly in body.data.polygons:
    m = body.data.materials[poly.material_index]
    poly.select = mat_class(m.name if m else '') == GLASS
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
glass = [o for o in scene.objects if o.name.startswith('body.')][0]; glass.name = 'glass'

log(f'body {pl.tris(body)} tris, glass {pl.tris(glass)}, wheel {pl.tris(wheel)}, tyre {pl.tris(tyre)}')
pl.decimate(body, LOD_TRIS[0])
log(f'body decimated to {pl.tris(body)}')

# ---------------------------------------------------------------- optional: the off-road wheel set (garage tyre sizes)
# Built by offroad_wheel.py beside the car (the AO bake must not see the stock wheel through it): the rim with the
# smaller tyre at OT_A, the bigger tyre at OT_B around an occluder rim (baked, not exported).
part_objs = [body, glass, wheel, tyre]
PART_IDS = {'body': 0, 'glass': 1, 'wheel': 2, 'tyre': 3}
OT = getattr(P, 'OFFROAD_WHEELS', None)
if OT:
    import offroad_wheel as OW
    OT_A = wheel_centres[fl_name] + np.array([-3.0, 0.0, 0.0])
    OT_B = wheel_centres[fl_name] + np.array([-5.0, 0.0, 0.0])
    def ot_place(c):
        # local (x out, y up, z forward) of a left wheel at Unity point c -> Blender
        cb = Vector((-S * c[0], -S * c[2], c[1]))
        return lambda v: cb + Vector((S * v.x, -S * v.z, v.y))
    (small, small_w), (big, big_w) = OT['tyres']
    wheel_ot = OW.make_rim('wheel_ot', ot_place(OT_A))
    tyre_a = OW.make_tyre(f'tyre_ot{small}', ot_place(OT_A), small, small_w)
    tyre_b = OW.make_tyre(f'tyre_ot{big}', ot_place(OT_B), big, big_w)
    ot_occluder = OW.make_rim('ot_occluder', ot_place(OT_B))
    log(f'off-road set: rim {pl.tris(wheel_ot)} tris, {small}in {pl.tris(tyre_a)}, {big}in {pl.tris(tyre_b)}')
    TYRE_A, TYRE_B = tyre_a.name, tyre_b.name   # (the objects go into the atlas: keep their names)
    for o in (wheel_ot, tyre_a, tyre_b):
        PART_IDS[o.name] = len(PART_IDS); part_objs.append(o)

# ---------------------------------------------------------------- one atlas: join, classes, looks
TYRE_PARTS = [PART_IDS[n] for n in PART_IDS if n.startswith('tyre')]
for o in part_objs:
    pl.face_attr(o.data, 'part', [PART_IDS[o.name]] * len(o.data.polygons))
    if 'lamp' not in o.data.attributes:
        pl.face_attr(o.data, 'lamp', [0] * len(o.data.polygons))
pl.select_only(body)
for o in part_objs[1:]: o.select_set(True)
bpy.ops.object.join()
atlas = bpy.context.view_layer.objects.active; atlas.name = 'atlas'
me = atlas.data
for uv in list(me.uv_layers): me.uv_layers.remove(uv)
me.uv_layers.new(name='atlas')

part = pl.get_face_attr(me, 'part')
cls = np.array([mat_class(me.materials[p.material_index].name if me.materials[p.material_index] else '') for p in me.polygons])
base = np.array([mat_color(me.materials[p.material_index]) for p in me.polygons])
fc = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('center', fc.ravel())
fn = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('normal', fn.ravel())
uc, un = to_unity(fc), to_unity(fn)
for c, rgb in P.ALBEDO_OVERRIDE.items(): base[cls == c] = rgb
front, rear = uc[:, 2] > 0, uc[:, 2] < 0
zones = np.zeros_like(base); mask = np.zeros_like(base); lights = np.zeros_like(base)
zones[cls == PAINT, 0] = 1; zones[cls == TRIM, 1] = 1                      # R paint, G trim (the colour pickers)
lampf = pl.get_face_attr(me, 'lamp')
dirt = np.isin(cls, [PAINT, TRIM, DARK, METAL, DETAIL, RIM, OTHER]) & ~np.isin(part, TYRE_PARTS) & (lampf == 0)
mask[dirt, 2] = 1                                                           # B takes dirt, mud, snow
mask[np.isin(cls, [LENS, EMIT, REDLENS, AMBER, GLASS]), 1] = 1              # G glossy (glass, lenses)
windshield = (part == PART_IDS['glass']) & (un[:, 2] > 0.3) & (uc[:, 2] > 0)
mask[windshield, 0] = 1                                                     # R the windscreen (rain, wipers)
lights[front & np.isin(cls, [EMIT, LENS]), 1] = 1          # G headlights
lights[rear & (cls == REDLENS), 0] = 1                     # R tail / brake
chmsl = rear & (cls == EMIT) & (uc[:, 1] > P.HIGH_BRAKE_Y)  # rear emitters this high: the third brake light
lights[chmsl, 0] = 1                                       # R brake (third light)
lights[rear & (cls == EMIT) & ~chmsl, 2] = 1               # B reverse
for nm, arr in (('c_albedo', base), ('c_zones', zones), ('c_mask', mask), ('c_lights', lights)):
    pl.set_corner_colour(me, nm, arr)

# each distinct look (albedo, zones, mask, lights, part) gets its own rectangle of the atlas
key = np.round(np.concatenate([base, zones, mask, lights, part[:, None]], 1), 3)
_, gid = np.unique(key, axis=0, return_inverse=True)
prio = [P.UV_PRIORITY.get(n, 1.0) for n in PART_IDS]   # by part id (dict order)
rects, gcls, gpart = pl.unwrap_looks(atlas, gid, part, cls, prio, P.CLASS_UV_WEIGHT, TEX, margins_px=(0.5, 0.0))

# ---------------------------------------------------------------- bake
baker = pl.Baker(atlas, OUT, NAME, TEX, ao_distance=P.AO_DISTANCE)
albedo_img = baker.bake('albedo_base', colour=True)
zones_img = baker.bake('zones', attr_name='c_zones')
baker.bake('mask', attr_name='c_mask')
lights_img = baker.bake('lights', attr_name='c_lights')
ao_img = baker.bake('ao', 'AO', samples=P.AO_SAMPLES)
if OT: bpy.data.objects.remove(ot_occluder)   # only there to shade the bigger tyre's inside
pl.blank_looks(ao_img, rects, [g for g in range(len(gcls)) if gcls[g] in pl.LAMP_CLASSES], TEX)

# ---------------------------------------------------------------- parts, LODs, collider, export
parts = pl.split_parts(atlas, PART_IDS)
lods = pl.make_lods(parts['body'], LOD_TRIS[1:])
coll = pl.hull_collider(parts['body'].data)
hub = wheel_centres[fl_name]
out_parts = [('body', 0, lods[0], np.zeros(3))]
out_parts += [(f'body_lod{i}', 0, o, np.zeros(3)) for i, o in enumerate(lods[1:], 1)]
out_parts += [('glass', 1, parts['glass'], np.zeros(3)), ('wheel', 2, parts['wheel'], hub),
              ('tyre', 3, parts['tyre'], hub), ('collider', 4, coll, np.zeros(3))]
if OT:   # the off-road set, each part centred on its own hub like the stock wheel (the car mod swaps them in)
    out_parts += [('wheel_ot', 2, parts['wheel_ot'], OT_A), (TYRE_A, 3, parts[TYRE_A], OT_A),
                  (TYRE_B, 3, parts[TYRE_B], OT_B)]
arrays = pl.export_parts(out_parts, to_unity, math.radians(getattr(P, 'SMOOTH_DEG', 35)))

# ---------------------------------------------------------------- anchors
anchors, ref = pl.map_ref_anchors(REF, arrays, wheel_centres)
if exhaust_pts:
    # smoke from the pipe's rear end (on its side only), facing the reference's way unless the profile says
    ep = np.concatenate(exhaust_pts); tip = ep[ep[:, 2] < ep[:, 2].min() + 0.03].mean(0)
    side = 'exhaust_L' if tip[0] < 0 else 'exhaust_R'
    rot = next((r for n, (p_, r) in ref.items() if n.startswith('exhaust_')), BACKQ)
    if getattr(P, 'EXHAUST_YAW', None) is not None: rot = pl.yaw_quat(P.EXHAUST_YAW)
    for n in [n for n in anchors if n.startswith('exhaust_')]: del anchors[n]
    anchors[side] = (tip, rot)
    log(f'anchor {side}: {np.round(tip, 3)} (tail pipe end)')
for corner_name, c in wheel_centres.items():
    cn = corner_of[corner_name]
    anchors['wheel_' + cn] = (c, IDQ if cn.endswith('L') else BACKQ)
def centroid(sel):
    return uc[sel].mean(0) if sel.any() else None
for side, sgn in (('L', -1), ('R', 1)):
    h = centroid(front & np.isin(cls, [EMIT, LENS]) & (np.sign(uc[:, 0]) == sgn) & (np.abs(uc[:, 0]) > P.LAMP_MIN_X))
    t = centroid(rear & (cls == REDLENS) & (np.sign(uc[:, 0]) == sgn) & (np.abs(uc[:, 0]) > P.LAMP_MIN_X))
    if h is not None: anchors['headlight_' + side] = (h, IDQ)
    if t is not None: anchors['tail_' + side] = (t, BACKQ)
r = centroid(rear & (cls == EMIT) & ~chmsl)
if r is not None: anchors['reverse'] = (np.array([0.0, r[1], r[2]]), BACKQ)
b = centroid(chmsl)
if b is not None: anchors['brake'] = (np.array([0.0, b[1], b[2]]), BACKQ)
pl.report_missing(anchors)

mesh_path = os.path.join(OUT, f'{NAME}.mesh')
pl.write_mesh(mesh_path, arrays, anchors)
import colliders
colliders.apply(mesh_path, getattr(P, 'COLLIDER_SLICES', 6))

# ---------------------------------------------------------------- previews
pv = pl.Previews(OUT, TEX, albedo_img, zones_img, ao_img, lights_img, P.PREVIEW_PAINT, P.PREVIEW_TRIM, S)
ot_names = ('wheel_ot', TYRE_A, TYRE_B) if OT else ()
pv.use([lods[0], parts['glass'], parts['wheel'], parts['tyre']] + [parts[n] for n in ot_names])
for o in lods[1:] + [coll]: o.hide_render = True
for n_ in ot_names: parts[n_].hide_render = True
pl.mirror_wheels(parts, wheel_centres, hub, fl_name, lambda k: corner_of[k].endswith('R'), S)
pv.shoot('front34', 35, 15); pv.shoot('rear34', 150, 20); pv.shoot('side', 90, 5); pv.shoot('top', 60, 55)
L = getattr(P, 'PREVIEW_LENGTH', 2.6)   # half the car's length, roughly: the close-ups' distance from the middle
pv.shoot('close_front', 0, 5, 2.2, (0, -S * L, 1.0)); pv.shoot('close_frontleft', -40, 10, 3.2, (S * 0.6, -S * (L - 0.7), 0.8))
pv.shoot('close_rear', 180, 5, 2.6, (0, S * (L - 0.1), 1.05)); pv.shoot('close_rearright', 130, 10, 3.2, (-S * 0.7, S * (L - 1.0), 0.9))
pv.lit_look(True)
pv.shoot('lit_front', 0, 8, 6.5); pv.shoot('lit_rear', 180, 12, 6.5)
pv.lit_look(False)
if OT:   # the off-road set on its own
    for n_ in ot_names: parts[n_].hide_render = False
    lods[0].hide_render = True
    for o in scene.objects:
        if o.name.startswith(('wheel', 'tyre')) and o.name not in ot_names: o.hide_render = True
    ca, cb_ = Vector((-S * OT_A[0], -S * OT_A[2], OT_A[1])), Vector((-S * OT_B[0], -S * OT_B[2], OT_B[1]))
    pv.shoot('offroad_wheels', 90 if S > 0 else -90, 12, 3.4, tuple((ca + cb_) / 2), lens=45)
    for n_ in ot_names: parts[n_].hide_render = True
    lods[0].hide_render = False
    for o in scene.objects:
        if o.name.startswith(('wheel', 'tyre')) and o.name not in ot_names: o.hide_render = False
for i, o in enumerate(lods[1:], 1):   # the distance versions, close up, to check them
    lods[i - 1].hide_render = True; o.hide_render = False
    pv.use([o])
    pv.shoot(f'side_lod{i}', 90, 5)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f'{NAME}_build.blend'))
log('done')
