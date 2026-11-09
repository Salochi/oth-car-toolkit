"""
The 4th-gen 4Runner's first step, before convert_materials.py: "Toyota 4Runner" by sadiqminhas (Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/toyota-4runner-3afb8daa0a4a4bfea99e1af74eca1bdb), a detailed, lifted 2006-2009 4Runner
coloured by materials, its objects merged by material (all four tyres in the same objects, the rear axle in the rims').
Writes a .blend for convert_materials.py with profiles/4runner4.py:

- the wheels split per corner (Wheel_FL ... and Tire_FL ..., by which hub each loose piece is round) and decimated
- the model's own running gear out (the rear axle, the front knuckles and shafts): the Burro's rigged axles show instead
- the belly and the front skid left whole: the rigged axles ride up into them (axle tunnels cut there read as a chunk missing)
- the lamp covers out of the window glass (glass would make them dark and see-through)
- scaled to the real wheelbase (2.789 m)

    blender -b --factory-startup --python profiles/4runner4_prep.py -- input/4runner4/toyota_4runner.glb build/4runner4/prepped.blend
    blender -b --factory-startup --python tools/convert_materials.py -- profiles/4runner4.py build/4runner4/prepped.blend build/4runner4 <hilo22.mesh> 4runner4

Built in the source's axes (Blender: x right, nose toward -y, z up, ground at z 0).
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
GLB, OUT = argv[0], os.path.abspath(argv[1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def log(*a): print('[prep]', *a, flush=True)


# ---- the source's measures (source units: about metres)
HUBS = {'FL': (-1, -1.665), 'FR': (1, -1.665), 'RL': (-1, 1.200), 'RR': (1, 1.200)}   # (side, y)
HUB_Z = 0.44
SCALE = 2.789 / (1.200 + 1.665)          # the real 4th gen's wheelbase, 109.8 in
WHEEL_TRIS, TYRE_TRIS = 9000, 12000      # per corner, after decimation (the source's are 30k and 60k)

# ---------------------------------------------------------------- import, flatten to world space
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
scene = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
src = []
for o in [o for o in scene.objects if o.type == 'MESH']:
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    me.calc_normals_split()
    ln = np.zeros((len(me.loops), 3)); me.loops.foreach_get('normal', ln.ravel())
    ln = ln @ np.array(o.matrix_world.to_3x3().inverted_safe().transposed()).T
    ln /= np.maximum(np.linalg.norm(ln, axis=1, keepdims=True), 1e-9)
    me.transform(o.matrix_world)
    me.use_auto_smooth = True
    me.normals_split_custom_set(ln.tolist())
    src.append((o.name, me))
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
for name, me in src:
    o = bpy.data.objects.new(name, me); scene.collection.objects.link(o)


def mat(name, rgb, metal=0.0, rough=0.5, like=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1); b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    return m


def islands(me):
    """Loose piece id per face (faces joined by shared vertices)."""
    nv = len(me.vertices)
    parent = np.arange(nv)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ev = np.zeros(len(me.edges) * 2, int); me.edges.foreach_get('vertices', ev); ev = ev.reshape(-1, 2)
    for a, b in ev:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    root = np.array([find(i) for i in range(nv)])
    lt = np.zeros(len(me.polygons), int); me.polygons.foreach_get('loop_start', lt)
    lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
    _, isl = np.unique(root[lv[lt]], return_inverse=True)
    return isl


def face_centres(me):
    c = np.zeros((len(me.polygons), 3)); me.polygons.foreach_get('center', c.ravel()); return c


def split(o, labels):
    """Split object o by a label per face: one object per label (a None label: dropped). Separated by selection, which
    keeps the custom normals. Returns {label: object}."""
    me = o.data
    lt = np.zeros(len(me.polygons), int); me.polygons.foreach_get('loop_start', lt)
    out = {}
    labs = sorted({('_drop' if l is None else l) for l in labels})
    labels = ['_drop' if l is None else l for l in labels]
    for lab in labs[:-1]:
        me = o.data
        sel = np.array([l == lab for l in labels])
        vsel = np.zeros(len(me.vertices), bool)
        for p in np.flatnonzero(sel): vsel[list(me.polygons[p].vertices)] = True
        me.vertices.foreach_set('select', vsel)
        me.edges.foreach_set('select', np.zeros(len(me.edges), bool))
        me.polygons.foreach_set('select', sel)
        bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_mode(use_extend=False, type='FACE', action='ENABLE')
        bpy.ops.mesh.select_all(action='DESELECT'); bpy.ops.object.mode_set(mode='OBJECT')
        o.data.vertices.foreach_set('select', vsel); o.data.polygons.foreach_set('select', sel)
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
        out[lab] = [x for x in bpy.context.selected_objects if x is not o][0]
        labels = [l for l, s_ in zip(labels, sel) if not s_]
    out[labs[-1]] = o
    for lab, x in out.items(): x.name = lab
    if '_drop' in out:
        bpy.data.objects.remove(out.pop('_drop'))
    return out


def labelled(o, fn):
    """fn(island bbox lo, hi, face centre) -> label or None (dropped), per loose piece."""
    me = o.data
    isl = islands(me); fc = face_centres(me)
    lab = [None] * len(isl)
    cache = {}
    order = np.argsort(isl)
    bounds = np.split(order, np.flatnonzero(np.diff(isl[order])) + 1)
    for faces in bounds:
        lo, hi = fc[faces].min(0), fc[faces].max(0)
        L = fn(lo, hi)
        for f in faces: lab[f] = L
    return lab


def hub_of(lo, hi, reach=0.36):
    c = (lo + hi) / 2
    for corner, (sx, y) in HUBS.items():
        if np.sign(c[0]) == sx and abs(c[0]) > 0.6 and math.hypot(c[1] - y, c[2] - HUB_Z) < reach \
                and hi[1] - lo[1] < 0.9:
            return corner
    return None


def inbox(lo, hi, b):
    return lo[0] >= b[0] and hi[0] <= b[1] and lo[1] >= b[2] and hi[1] <= b[3] and lo[2] >= b[4] and hi[2] <= b[5]


objs = {o.name: o for o in scene.objects}
dropped = 0


def apply_labels(name, fn):
    global dropped
    lab = labelled(objs[name], fn)
    n = sum(1 for l in lab if l is None); dropped += n
    if n: log(f'  {name}: {n} of {len(lab)} faces out')
    return split(objs[name], lab)


# ---------------------------------------------------------------- the wheels: per corner; the rest of their objects out
for name in ('Object_27', 'Object_29', 'Object_31', 'Object_33'):            # tyres
    apply_labels(name, lambda lo, hi: (f'Tire_{hub_of(lo, hi, 0.5)}_' if hub_of(lo, hi, 0.5) else None))   # tread blocks: 0.4 out
for name in ('Object_35', 'Object_37', 'Object_39', 'Object_41'):            # rims, inner wheel, hub locks
    apply_labels(name, lambda lo, hi: (f'Wheel_{hub_of(lo, hi)}_' if hub_of(lo, hi) else None))
log('wheels split;', dropped, 'faces of running gear out (axle, shafts)')

# ---------------------------------------------------------------- the body's black: the front knuckles and shafts out
FRONT_GEAR = [(0.30, 0.80, -1.82, -1.55, 0.30, 0.63), (-0.80, -0.30, -1.82, -1.55, 0.30, 0.63)]
before = dropped
# the exterior black plastic (the trim colour): the roof rack, the running boards, the grille and the lower grille
TRIM_BOXES = [(-1.0, 1.0, -1.2, 2.3, 1.885, 2.1),                              # roof rails and cross bars
              (0.70, 1.0, -1.25, 0.80, 0.35, 0.62), (-1.0, -0.70, -1.25, 0.80, 0.35, 0.62),   # running boards
              (-0.55, 0.55, -2.70, -2.25, 0.50, 1.20)]                          # grille, lower grille


def black_label(lo, hi):
    if any(inbox(lo, hi, b) for b in FRONT_GEAR): return None
    if any(inbox(lo, hi, b) for b in TRIM_BOXES): return 'rr_trim'
    return 'rr_black'


blk = apply_labels('Object_10', black_label)
trim_mat = mat('rr_trim', (0.02, 0.02, 0.022), 0.2, 0.5)
if 'rr_trim' in blk:
    blk['rr_trim'].data.materials.clear(); blk['rr_trim'].data.materials.append(trim_mat)
    log('trim:', len(blk['rr_trim'].data.polygons), 'faces')
log(dropped - before, 'faces of the front suspension out')

# ---------------------------------------------------------------- the lamps' chrome reflectors out
# the lamp covers are opaque in game (a lens), so the reflectors behind them never show; they were most of the chrome's
# 230k triangles. What is left (the badges) stays out of the decimation (DECIMATE_KEEP: decimated, letters went missing)
LAMP_BOXES = [(0.40, 1.0, -2.75, -1.95, 0.80, 1.30), (-1.0, -0.40, -2.75, -1.95, 0.80, 1.30),   # headlamps
              (0.40, 1.0, -2.75, -1.95, 0.45, 0.82), (-1.0, -0.40, -2.75, -1.95, 0.45, 0.82),   # fog lamps
              (0.70, 1.0, 1.85, 2.40, 0.95, 1.60), (-1.0, -0.70, 1.85, 2.40, 0.95, 1.60),      # tail lamps
              (-0.25, 0.25, 2.15, 2.30, 1.78, 1.90)]                                           # third brake lamp
before = dropped
for name in ('Object_14', 'Object_16', 'Object_18'):
    apply_labels(name, lambda lo, hi: None if any(inbox(lo, hi, b) for b in LAMP_BOXES) else 'rr_chrome')
log(dropped - before, 'faces of lamp reflectors out')

# ---------------------------------------------------------------- lamp covers out of the window glass
lens = mat('rr_lens', (0.85, 0.85, 0.82), 0.0, 0.1)


def glass_label(lo, hi):
    c = (lo + hi) / 2
    if c[1] < -2.0 and c[2] < 0.80: return 'lamp_front_fog'        # fog lamp covers (no light: off the anchors)
    if c[1] < -2.0 and c[2] < 1.25: return 'lamp_front_cover'      # headlamp covers
    if c[1] > 1.9 and c[2] < 1.45:
        # the tail lamps' clear parts: the lower one is the reverse lamp, the upper one (the indicator, behind clear
        # glass) stays unlit. Between them, bulb beads under the red lens (12k faces, never seen) that pulled the
        # reverse anchor into the red middle
        if c[2] < 1.11: return 'lamp_rear_cover'
        if c[2] > 1.27: return 'lamp_rear_signal'
        return None
    return 'rr_glass'


g = apply_labels('Object_20', glass_label)
fog = mat('rr_foglens', (0.75, 0.75, 0.70), 0.0, 0.1)
for k, o in g.items():
    if k.startswith('lamp_'):
        o.data.materials.clear(); o.data.materials.append(fog if k in ('lamp_front_fog', 'lamp_rear_signal') else lens)
objs['Object_22'].name = 'lamp_rear_tail'
objs['Object_12'].name = 'lamp_front_indicator'
log('glass:', {k: len(o.data.polygons) for k, o in g.items()})

# ---------------------------------------------------------------- decimate the wheels (the converter keeps them whole)
groups = {}
for o in list(scene.objects):
    if o.type == 'MESH' and o.name.startswith(('Tire_', 'Wheel_')):
        groups.setdefault(o.name.rstrip('.0123456789').rstrip('_'), []).append(o)
for gname, os_ in groups.items():
    bpy.ops.object.select_all(action='DESELECT')
    for o in os_: o.select_set(True)
    bpy.context.view_layer.objects.active = os_[0]
    if len(os_) > 1: bpy.ops.object.join()
    j = bpy.context.view_layer.objects.active; j.name = gname
    n = sum(len(p.vertices) - 2 for p in j.data.polygons)
    target = TYRE_TRIS if gname.startswith('Tire') else WHEEL_TRIS
    if n > target:
        md = j.modifiers.new('d', 'DECIMATE'); md.ratio = target / n
        bpy.ops.object.modifier_apply(modifier='d')
    log(f'{gname}: {n} -> {sum(len(p.vertices) - 2 for p in j.data.polygons)} tris')

# ---------------------------------------------------------------- name the rest
names = {'Object_8': 'rr_paint', 'Object_24': 'rr_mirrorglass'}
for k, v in names.items():
    if k in objs and objs[k].name == k: objs[k].name = v

# ---------------------------------------------------------------- scale and place: the axles' middle at y 0
root = bpy.data.objects.new('rr_root', None); scene.collection.objects.link(root)
for o in bpy.data.objects:
    if o.parent is None and o is not root:
        mw = o.matrix_world.copy(); o.parent = root; o.matrix_world = mw
root.scale = (SCALE,) * 3
root.location = (0, -SCALE * (HUBS['FL'][1] + HUBS['RL'][1]) / 2, 0)
bpy.context.view_layer.update()
log('chrome left:', sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects
                        if o.type == 'MESH' and o.name.startswith('rr_chrome')), 'tris')
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
log(f'scale x{SCALE:.4f}, {tris} triangles; objects:', sorted(o.name for o in bpy.data.objects if o.type == 'MESH'))
bpy.ops.wm.save_as_mainfile(filepath=OUT)
log('saved', OUT)
