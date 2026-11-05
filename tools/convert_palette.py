"""
Converter for low-poly models coloured by one palette texture (POLYGON-style asset packs: each face sits on one palette
cell, painted panels on a texture strip with wear on it): GLB -> <name>.mesh + maps + previews.

Run:  blender -b --factory-startup --python tools/convert_palette.py -- <profile.py> <model.glb> <out_dir> <reference.mesh> [name]

Faces are classed by the texture colour under them (the profile's PALETTE), the albedo is baked from the source
texture, and the paint zone is wherever the painted texels are (blue over red by a margin, or the profile's rule), so
wear and dirt stay on top of any paint colour. The model is scaled to a real wheelbase. Optional (all in the profile):
loose pieces of an "armour" object kept only at the front, the model's own suspension and hub pieces taken out (for a
car on the donor's wheels and rigged axles), the tyre darkened.
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
GLB, OUT, REF = argv[1], os.path.abspath(argv[2]), argv[3]
NAME = argv[4] if len(argv) > 4 else P.NAME
os.makedirs(OUT, exist_ok=True)
TEX = getattr(P, 'TEX', 2048)
WELD = P.WELD


def palette_class(rgb, src):
    for ref, c, only in P.PALETTE:
        if only and only != src: continue
        if np.abs(np.array(rgb, float) - ref).max() <= P.PALETTE_TOL: return c
    lum = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    return DARK if lum < 90 else OTHER


# ---------------------------------------------------------------- import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
scene = bpy.context.scene
for o in list(bpy.data.objects):
    if o.type != 'MESH' or o.name.startswith(P.DROP_OBJECTS):
        if o.type == 'MESH': log('drop object', o.name)
        if o.type != 'EMPTY': bpy.data.objects.remove(o)
bpy.context.view_layer.update()
tex_body = bpy.data.images[P.BODY_TEXTURE]; tex_wheel = bpy.data.images[P.WHEEL_TEXTURE]
tw, th = tex_body.size
tex_px = np.array(tex_body.pixels[:], dtype=np.float32).reshape(th, tw, 4)[..., :3]
kind_of = P.kind_of   # object name -> 'body', 'glass', 'armour' or 'wheel'

# scale and place: the wheels' centres give the wheelbase (Blender y), the tyres' bottoms the ground
meshes = [o for o in scene.objects if o.type == 'MESH']
wc = {}
for o in meshes:
    if kind_of(o.name) == 'wheel':
        v = np.array([o.matrix_world @ x.co for x in o.data.vertices]); wc[o.name] = (v.min(0), v.max(0))
cy = sorted(((lo + hi) / 2)[1] for lo, hi in wc.values())
K = P.WHEELBASE / (np.mean(cy[2:]) - np.mean(cy[:2]))
ground = min(lo[2] for lo, hi in wc.values())
mid_y = (np.mean(cy[2:]) + np.mean(cy[:2])) / 2
PLACE = Matrix.Scale(K, 4) @ Matrix.Translation((0.0, -mid_y, -ground))
log(f'scale x{K:.2f}, ground at {ground:.5f}')
S = P.FRONT_SIGN   # +1: the front is Blender -y


def to_unity(v):  # Blender (x,y,z) -> Unity (x right, y up, z forward); a mirror, fixed by the winding check later
    return np.stack([-S * v[..., 0], v[..., 2], -S * v[..., 1]], -1)


flat = []
for o in meshes:
    me = bpy.data.meshes.new_from_object(o, preserve_all_data_layers=True)
    me.transform(PLACE @ o.matrix_world)
    flat.append((me, kind_of(o.name), o.name))
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def bm_weld(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD)


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


# ---------------------------------------------------------------- per-face class, colour and texture
def face_info(me, src):
    """Per face: class, sampled sRGB 0..255, the texture to bake from (0 none, 1 body, 2 wheel)."""
    uvd = me.uv_layers[0].data
    cls, rgb, tex = [], [], []
    for p in me.polygons:
        uv = np.array([uvd[i].uv for i in p.loop_indices])
        span = float((uv.max(0) - uv.min(0)).max())
        if src == 'glass':
            cls.append(GLASS); rgb.append((0, 0, 0)); tex.append(0); continue
        u, v = np.clip(uv.mean(0), 0, 0.9999)
        c = tuple(int(x) for x in np.round(tex_px[int(v * th), int(u * tw)] * 255))
        rgb.append(c); tex.append(1)
        if span > 0.004 or P.is_paint_colour(c):
            # the painted strip (paint, wear, dirt) and paint-coloured palette faces: the paint goes where the texels
            # are paint, decided per texel after the bake
            cls.append(PAINT)
            continue
        cl = palette_class(c, src)
        if cl == TRIM and not P.is_trim(to_unity(np.array(p.center)[None])[0]): cl = DARK
        cls.append(cl)
    return cls, rgb, tex


def set_face_attrs(me, cls, rgb, tex, part):
    for nm, vals in (('cls', cls), ('tex', tex), ('part', [part] * len(me.polygons))):
        pl.face_attr(me, nm, vals)
    pl.face_attr(me, 'rgb', [r << 16 | g << 8 | b for r, g, b in rgb])


PART_IDS = {'body': 0, 'glass': 1, 'wheel': 2, 'tyre': 3}
groups = {'body': [], 'glass': []}
wheel_centres, wheel_meshes = {}, {}
for me, src, oname in flat:
    if src == 'wheel':
        v = np.array([x.co for x in me.vertices]); c = (v.min(0) + v.max(0)) / 2
        cu = to_unity(c[None])[0]
        corner = ('F' if cu[2] > 0 else 'R') + ('L' if cu[0] < 0 else 'R')
        wheel_centres[corner] = cu; wheel_meshes[corner] = (me, c)

HUB = getattr(P, 'HUB_CLEAR', None)       # (radius, inward) metres: the donor's wheel volume at each hub
SUSP = getattr(P, 'SUSPENSION', None)     # the rules that find the model's own suspension pieces


def in_wheel(pu):
    """Inside the volume the game's wheel takes at a corner (Unity point): hub and spindle pieces sized for the model's
    wider wheels poke through the donor's narrower ones."""
    for c in wheel_centres.values():
        if math.hypot(pu[1] - c[1], pu[2] - c[2]) < HUB[0] and abs(pu[0]) > abs(c[0]) - HUB[1] and np.sign(pu[0]) == np.sign(c[0]):
            return True
    return False


hub_cleared = 0
for me, src, oname in flat:
    if src == 'wheel': continue
    bm = bmesh.new(); bm.from_mesh(me)
    if src == 'body' and (SUSP or HUB):
        bm_weld(bm)
        axles_z = sorted({round(float(c[2]), 3) for c in wheel_centres.values()})
        susp = []
        for comp in (loose_parts(bm) if SUSP else []):
            if len(comp) >= SUSP['max_faces']: continue
            v = to_unity(np.array([tuple(l.vert.co) for f in comp for l in f.loops]))
            lo, hi = v.min(0), v.max(0); cz = (lo[2] + hi[2]) / 2
            near = min(abs(cz - z) for z in axles_z) < SUSP['axle_dz']
            if near and hi[1] < SUSP['top'] and lo[1] < SUSP['bottom'] and np.abs(v[:, 0]).max() > SUSP['reach']:
                susp += comp
            elif near and len(comp) < SUSP['diff_max_faces'] and np.abs(v[:, 0]).max() < SUSP['diff_half_width'] and lo[1] < SUSP['diff_bottom']:
                log(f'centre block out: {len(comp)} faces, y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f}')
                susp += comp
        susp_ids = {f.index for f in susp}
        gone = susp + ([f for f in bm.faces if f.index not in susp_ids and in_wheel(to_unity(np.array(f.calc_center_median())[None])[0])] if HUB else [])
        hub_cleared += len(gone) - len(susp)
        log(f'suspension: {len(susp)} body faces taken out (the donor axles show instead)')
        bmesh.ops.delete(bm, geom=gone, context='FACES')
    if src == 'armour':
        bm_weld(bm)
        gone = []
        for comp in loose_parts(bm):
            zc = np.mean([to_unity(np.array(l.vert.co)[None])[0][2] for f in comp for l in f.loops])
            if zc < P.ARMOUR_KEEP_Z: gone += comp
        log(f'armour: drop {len(gone)} of {len(bm.faces)} faces (keeping the pieces at the front)')
        bmesh.ops.delete(bm, geom=gone, context='FACES')
    bm.to_mesh(me); bm.free()
    cls, rgb, tex = face_info(me, src)
    set_face_attrs(me, cls, rgb, tex, PART_IDS['glass' if src == 'glass' else 'body'])
    groups['glass' if src == 'glass' else 'body'].append(me)
log('wheels', {k: np.round(c, 3) for k, c in wheel_centres.items()})
log(f'hubs: {hub_cleared} body faces inside the wheels taken out')
fl_me, fl_c = wheel_meshes['FL']
for k, (me, c) in wheel_meshes.items():
    if k != 'FL': bpy.data.meshes.remove(me)


def split_wheel(me, c):
    """The FL wheel: rim and tyre by distance from the axle (Blender x)."""
    rim, tyre = me.copy(), me.copy()
    for m, keep_tyre in ((rim, False), (tyre, True)):
        bm = bmesh.new(); bm.from_mesh(m)
        gone = [f for f in bm.faces if (math.hypot(f.calc_center_median()[1] - c[1], f.calc_center_median()[2] - c[2]) > P.RIM_RADIUS) != keep_tyre]
        bmesh.ops.delete(bm, geom=gone, context='FACES'); bm.to_mesh(m); bm.free()
    bpy.data.meshes.remove(me)
    return rim, tyre
rim_me, tyre_me = split_wheel(fl_me, fl_c)
for m, cl, pn in ((rim_me, RIM, 'wheel'), (tyre_me, TYRE, 'tyre')):
    n = len(m.polygons)
    set_face_attrs(m, [cl] * n, [(0, 0, 0)] * n, [2] * n, PART_IDS[pn])
tyre_r = max(math.hypot(v.co[1] - fl_c[1], v.co[2] - fl_c[2]) for v in tyre_me.vertices)
log(f'tyre radius {tyre_r:.3f} m, width {np.ptp([v.co[0] for v in tyre_me.vertices]):.3f} m')

# the source's custom normals are flat per triangle: dropped so the export smooths the welded body
body = pl.make_obj('body', groups['body'], True); glass = pl.make_obj('glass', groups['glass'], True)
wheel = pl.make_obj('wheel', [rim_me], True); tyre = pl.make_obj('tyre', [tyre_me], True)
for o in (body, glass, wheel, tyre): pl.weld(o, WELD)
log(f'body {pl.tris(body)} tris, glass {pl.tris(glass)}, wheel {pl.tris(wheel)}, tyre {pl.tris(tyre)}')

# ---------------------------------------------------------------- one atlas: join, classes, looks
pl.select_only(body)
for o in (glass, wheel, tyre): o.select_set(True)
bpy.ops.object.join()
atlas = bpy.context.view_layer.objects.active; atlas.name = 'atlas'
me = atlas.data
me.uv_layers[0].name = 'src'
for uv in list(me.uv_layers)[1:]: me.uv_layers.remove(uv)
me.uv_layers.new(name='atlas')
me.uv_layers.active = me.uv_layers['atlas']; me.uv_layers['atlas'].active_render = True

nf = len(me.polygons)
part, cls, texsrc, rgbi = (pl.get_face_attr(me, n).astype(np.int64) for n in ('part', 'cls', 'tex', 'rgb'))
rgb = np.stack([(rgbi >> 16) & 255, (rgbi >> 8) & 255, rgbi & 255], 1)
fc = np.zeros((nf, 3)); me.polygons.foreach_get('center', fc.ravel())
fn = np.zeros((nf, 3)); me.polygons.foreach_get('normal', fn.ravel())
uc, un = to_unity(fc), to_unity(fn)
front, rear = uc[:, 2] > 0, uc[:, 2] < 0
log('classes:', {pl.CLASS_NAMES[c]: int((cls == c).sum()) for c in np.unique(cls)})

base = np.full((nf, 3), 0.5); has_override = np.zeros(nf, bool)
for c, lin in P.ALBEDO_OVERRIDE.items(): base[cls == c] = lin; has_override |= cls == c
tex_w = np.zeros((nf, 3))   # R: bake from the body texture, G: from the wheel texture
tex_w[(texsrc == 1) & ~has_override, 0] = 1; tex_w[(texsrc == 2) & ~has_override, 1] = 1
zones = np.zeros((nf, 3)); mask = np.zeros((nf, 3)); lights = np.zeros((nf, 3))
zones[cls == PAINT, 0] = 1; zones[cls == TRIM, 1] = 1
lamp = np.isin(cls, [LENS, EMIT, REDLENS, AMBER])
dirt = np.isin(cls, [PAINT, TRIM, DARK, METAL, DETAIL, RIM, OTHER]) & (part != PART_IDS['tyre']) & ~lamp
mask[dirt, 2] = 1
mask[np.isin(cls, [LENS, EMIT, REDLENS, AMBER, GLASS]), 1] = 1
windshield = (part == PART_IDS['glass']) & (un[:, 2] > 0.3) & (uc[:, 2] > 0)
mask[windshield, 0] = 1
lights[front & (cls == EMIT), 1] = 1        # G headlights
lights[rear & (cls == REDLENS), 0] = 1      # R tail / brake
for nm, arr in (('c_albedo', base), ('c_tex', tex_w), ('c_zones', zones), ('c_mask', mask), ('c_lights', lights)):
    pl.set_corner_colour(me, nm, arr)

# each distinct look (class, palette colour, part, zones/mask/lights) gets its own rectangle; the painted strip's faces
# are one look per class
key = np.concatenate([cls[:, None], np.where(texsrc[:, None] == 1, rgb, 0) * (cls[:, None] != PAINT), part[:, None],
                      np.round(np.concatenate([zones, mask, lights], 1), 3)], 1)
_, gid = np.unique(key, axis=0, return_inverse=True)
prio = [P.UV_PRIORITY.get(n, 1.0) for n in PART_IDS]
rects, gcls, gpart = pl.unwrap_looks(atlas, gid, part, cls, prio, P.CLASS_UV_WEIGHT, TEX, margins_px=(2.0, 0.5, 0.0))

# ---------------------------------------------------------------- bake
baker = pl.Baker(atlas, OUT, NAME, TEX, ao_distance=P.AO_DISTANCE, albedo_textures=((tex_body, 'Red'), (tex_wheel, 'Green')))
albedo_img = baker.bake('albedo_base', colour=True)
if getattr(P, 'TYRE_DARKEN', None):
    # a light grey photo tread reads as white tyres in the game's light: darkened to rubber
    alb = pl.pixels(albedo_img, TEX)
    for g in range(len(gcls)):
        if gcls[g] == TYRE:
            r, c = pl.rect_pixels(rects, g, TEX); alb[r, c, :3] *= P.TYRE_DARKEN
    albedo_img.pixels.foreach_set(alb.ravel()); albedo_img.save()
zones_img = baker.bake('zones', attr_name='c_zones')
baker.bake('mask', attr_name='c_mask')
lights_img = baker.bake('lights', attr_name='c_lights')
ao_img = baker.bake('ao', 'AO', samples=P.AO_SAMPLES)
# the paint zone: only the painted texels (wear and dirt keep their own colour over any paint)
alb, zon = pl.pixels(albedo_img, TEX), pl.pixels(zones_img, TEX)
zon[..., 0] *= P.paint_texels(alb[..., :3])
zones_img.pixels.foreach_set(zon.ravel()); zones_img.save()
log(f'paint zone: {(zon[..., 0] > 0.5).mean():.1%} of the atlas')
pl.blank_looks(ao_img, rects, [g for g in range(len(gcls)) if gcls[g] in pl.LAMP_CLASSES], TEX)

# ---------------------------------------------------------------- parts, LODs, collider, export
parts = pl.split_parts(atlas, PART_IDS)
lods = pl.make_lods(parts['body'], P.LOD_TRIS, report_tris=True)
coll = pl.hull_collider(parts['body'].data)
hub = wheel_centres['FL']
out_parts = [('body', 0, lods[0], np.zeros(3))]
out_parts += [(f'body_lod{i}', 0, o, np.zeros(3)) for i, o in enumerate(lods[1:], 1)]
out_parts += [('glass', 1, parts['glass'], np.zeros(3)), ('wheel', 2, parts['wheel'], hub),
              ('tyre', 3, parts['tyre'], hub), ('collider', 4, coll, np.zeros(3))]
arrays = pl.export_parts(out_parts, to_unity, math.radians(getattr(P, 'SMOOTH_DEG', 35)))

# ---------------------------------------------------------------- anchors
anchors, ref = pl.map_ref_anchors(REF, arrays, wheel_centres, prefixes=('cosm_', 'fog_', 'exhaust_', 'engine_smoke'))
tRL = min(c[2] for c in wheel_centres.values())
ex = P.exhaust(uc, part, tRL)    # the profile's exhaust point (or None: the reference's, mapped)
if ex is not None:
    rot = next((r for n, (p_, r) in ref.items() if n.startswith('exhaust_')), BACKQ)
    for n in [n for n in anchors if n.startswith('exhaust_')]: del anchors[n]
    anchors['exhaust_L' if ex[0] < 0 else 'exhaust_R'] = (ex, rot)
    log(f'anchor exhaust: {np.round(ex, 3)}')
for cn, c in wheel_centres.items():
    anchors['wheel_' + cn] = (c, IDQ if cn.endswith('L') else BACKQ)
def centroid(sel):
    return uc[sel].mean(0) if sel.any() else None
for side, sgn in (('L', -1), ('R', 1)):
    h = centroid(front & (cls == EMIT) & (np.sign(uc[:, 0]) == sgn))
    t = centroid(rear & (cls == REDLENS) & (np.sign(uc[:, 0]) == sgn))
    if h is not None: anchors['headlight_' + side] = (h, IDQ)
    if t is not None: anchors['tail_' + side] = (t, BACKQ)
    log(f'lamps {side}: head {None if h is None else np.round(h, 3)}, tail {None if t is None else np.round(t, 3)}')
tl_ = centroid(rear & (cls == REDLENS))
if tl_ is not None: anchors['reverse'] = (np.array([0.0, tl_[1], tl_[2]]), BACKQ)   # no reverse lamp: between the tails
pl.report_missing(anchors)
mesh_path = os.path.join(OUT, f'{NAME}.mesh')
pl.write_mesh(mesh_path, arrays, anchors)
import colliders
colliders.apply(mesh_path, getattr(P, 'COLLIDER_SLICES', 6))

# ---------------------------------------------------------------- previews
pv = pl.Previews(OUT, TEX, albedo_img, zones_img, ao_img, lights_img, P.PREVIEW_PAINT, P.PREVIEW_TRIM, S, uv_map='atlas')
pv.use([lods[0], parts['glass'], parts['wheel'], parts['tyre']], set_uv=True)
for o in lods[1:] + [coll]: o.hide_render = True
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
pv.shoot('front34', 35, 15, 8.5, (0, 0, 0.85)); pv.shoot('rear34', 150, 20, 8.5, (0, 0, 0.85))
pv.shoot('side', 90, 5, 8.5, (0, 0, 0.85)); pv.shoot('top', 60, 55, 8.5, (0, 0, 0.85))
pv.shoot('close_front', 0, 8, 3.4, (0, -S * 1.6, 0.8)); pv.shoot('close_rear', 180, 8, 3.4, (0, S * 1.6, 0.9))
pv.lit_look(True)
pv.shoot('lit_front', 0, 8, 6.0, (0, 0, 0.85)); pv.shoot('lit_rear', 180, 12, 6.0, (0, 0, 0.85))
pv.lit_look(False)
for i, o in enumerate(lods[1:], 1):
    lods[i - 1].hide_render = True; o.hide_render = False
    pv.use([o], set_uv=True)
    pv.shoot(f'side_lod{i}', 90, 5, 8.5, (0, 0, 0.85))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f'{NAME}_build.blend'))
log('done')
