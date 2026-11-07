"""
The conversion stages every converter shares (run inside Blender 3.6): one texture atlas with a rectangle per face
"look", the baked maps (albedo, paint zones, mask, lights, ambient occlusion), the body's distance versions, the
collider, the export to the game's axes and the .mesh with its anchors, and the preview renders.

A converter (convert_materials.py, convert_palette.py) reads its source model, sorts its faces into parts and classes
and puts per-face colours on one joined object ("atlas"); these stages do the rest. See docs/pipeline.md.
"""
import bpy, bmesh, os, math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import vera_mesh

# ---------------------------------------------------------------- face classes (what a face is, for the maps)
PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER = range(14)
CLASS_NAMES = ['PAINT', 'TRIM', 'DARK', 'METAL', 'LENS', 'EMIT', 'REDLENS', 'AMBER', 'DETAIL', 'GLASS', 'TYRE', 'RIM',
               'INTERIOR', 'OTHER']
LAMP_CLASSES = (LENS, EMIT, REDLENS, AMBER)
IDQ, BACKQ = (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, -4.371139e-08)   # facing forward / backward (Unity)
KIND = {'body': vera_mesh.BODY, 'glass': vera_mesh.GLASS, 'wheel': vera_mesh.WHEEL, 'tyre': vera_mesh.TYRE,
        'collider': vera_mesh.COLLIDER}


def log(*a):
    print('[build]', *a)


# ---------------------------------------------------------------- objects
def scene():
    return bpy.context.scene


def select_only(o):
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.context.view_layer.objects.active = o


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def weld(o, dist=1e-5):
    select_only(o); bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=dist); bpy.ops.object.mode_set(mode='OBJECT')


def decimate(o, target):
    t = tris(o)
    if t <= target: return
    # no welding: separate pieces (doors, panels, trim) must not collapse into each other
    m = o.modifiers.new('dec', 'DECIMATE'); m.decimate_type = 'COLLAPSE'; m.ratio = target / t
    m.use_collapse_triangulate = True
    select_only(o); bpy.ops.object.modifier_apply(modifier='dec')


def make_obj(name, meshes, clear_custom_normals=False):
    """One object from meshes (joined)."""
    objs = []
    for me in meshes:
        o = bpy.data.objects.new(name + '_src', me); scene().collection.objects.link(o); objs.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1: bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active; o.name = name
    if clear_custom_normals and o.data.has_custom_normals: bpy.ops.mesh.customdata_custom_splitnormals_clear()
    return o


def face_attr(me, name, values, kind='INT'):
    a = me.attributes.new(name, kind, 'FACE'); a.data.foreach_set('value', list(values))


def get_face_attr(me, name):
    a = np.zeros(len(me.polygons), dtype=np.int32); me.attributes[name].data.foreach_get('value', a)
    return a


def set_corner_colour(me, name, face_rgb):
    """A per-face colour as a corner colour attribute (the bake reads it)."""
    a = me.color_attributes.new(name, 'FLOAT_COLOR', 'CORNER')
    li = np.zeros(len(me.loops), dtype=np.int64)
    for p in me.polygons: li[p.loop_start:p.loop_start + p.loop_total] = p.index
    rgba = np.ones((len(me.loops), 4), dtype=np.float32); rgba[:, :3] = face_rgb[li]
    a.data.foreach_set('color', rgba.ravel())


# ---------------------------------------------------------------- the atlas
def squarify(areas, x, y, w, h):
    """Squarified treemap: rectangles of the given areas (summing to w*h), as close to square as it can."""
    out, items = [], list(areas)
    def worst(row, short):
        s = sum(row); return max(max(short * short * a / (s * s), s * s / (short * short * a)) for a in row)
    while items:
        short = min(w, h); row = [items[0]]; i = 1
        while i < len(items) and worst(row + [items[i]], short) <= worst(row, short):
            row.append(items[i]); i += 1
        thick = sum(row) / short; off = 0.0
        for a in row:
            l = a / thick
            out.append((x, y + off, thick, l) if w >= h else (x + off, y, l, thick)); off += l
        if w >= h: x += thick; w -= thick
        else: y += thick; h -= thick
        items = items[i:]
    return out


def unwrap_looks(atlas, gid, part, cls, part_prio, class_weight, tex, margins_px=(0.5, 0.0), uv_name='atlas', big_px=256):
    """
    The atlas UVs: each look (gid: per-face look index) gets its own rectangle, so texture filtering and mipmaps never
    mix looks (no tail-lamp red on the bodywork, faces smaller than a texel still read their own look). Rectangle
    areas follow 3D area times the part's priority (part_prio, by part id) and the class weight. margins_px: the
    island margins tried in turn, in texels of the rectangle (the packer gives up when thousands of islands share a
    small rectangle: then the next, smaller one). Returns (rects by look, class per look, part per look).
    """
    me = atlas.data
    gid = gid.ravel(); NG = int(gid.max()) + 1
    nf = len(me.polygons)
    loop_start = np.zeros(nf, dtype=np.int32); me.polygons.foreach_get('loop_start', loop_start); loop_start = loop_start.astype(np.int64)
    loop_total = np.zeros(nf, dtype=np.int32); me.polygons.foreach_get('loop_total', loop_total); loop_total = loop_total.astype(np.int64)
    loop_face = np.repeat(np.arange(nf), loop_total)
    nxt = np.arange(len(me.loops)) + 1; nxt[loop_start + loop_total - 1] = loop_start
    area3 = np.zeros(nf, np.float32); me.polygons.foreach_get('area', area3); area3 = area3.astype(np.float64)
    uvl = lambda: me.uv_layers[uv_name]  # fetched fresh: mode switches reallocate the layer
    def get_uv():
        a = np.zeros(len(me.loops) * 2, np.float32); uvl().data.foreach_get('uv', a); return a.reshape(-1, 2).astype(np.float64)
    def set_uv(a):
        uvl().data.foreach_set('uv', a.astype(np.float32).ravel())
    def uv_area(a):
        cr = a[:, 0] * a[nxt, 1] - a[nxt, 0] * a[:, 1]
        return 0.5 * np.abs(np.add.reduceat(cr, loop_start))
    def edit_select(sel):
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='DESELECT')
        bm = bmesh.from_edit_mesh(me); bm.faces.ensure_lookup_table(); ul = bm.loops.layers.uv[uv_name]
        for i in np.nonzero(sel)[0]:
            f = bm.faces[i]; f.select_set(True)
            for l in f.loops: l[ul].select = True
        bmesh.update_edit_mesh(me)

    select_only(atlas)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_mode(type='FACE'); bpy.ops.object.mode_set(mode='OBJECT')
    for g in range(NG):
        edit_select(gid == g)
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0, area_weight=0.0, correct_aspect=True,
                                 scale_to_bounds=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    # same texel density everywhere (UV units = metres), times each part's priority
    uv = get_uv(); a2 = uv_area(uv)
    prio = np.asarray(part_prio, float)
    gpart = np.zeros(NG, dtype=np.int64); gpart[gid] = part
    s3 = np.bincount(gid, area3, NG); s2 = np.bincount(gid, a2, NG)
    gcls = np.zeros(NG, dtype=np.int64); gcls[gid] = cls
    gprio = prio[gpart] * np.array([class_weight.get(c, 1.0) for c in gcls])
    k = np.sqrt(s3 / np.maximum(s2, 1e-12)) * gprio
    uv *= k[gid[loop_face]][:, None]
    share = s3 * gprio ** 2
    share = np.maximum(share / share.sum(), (28 / tex) ** 2)   # every look gets at least a 28 px square
    share /= share.sum()
    order = np.argsort(-share)
    rects = dict(zip(order.tolist(), squarify(share[order].tolist(), 0.0, 0.0, 1.0, 1.0)))
    GAP = 5 / tex   # per side: 10 texels between looks
    set_uv(uv)
    for g in range(NG):
        x, y, w, h = rects[g]
        x, y, w, h = x + GAP, y + GAP, max(w - 2 * GAP, 2 / tex), max(h - 2 * GAP, 2 / tex)
        lsel = gid[loop_face] == g
        uv = get_uv(); uv[lsel, 0] /= w; uv[lsel, 1] /= h; set_uv(uv)   # pre-squash so the square pack lands undistorted
        pre = get_uv()
        # wide margins only where the rectangle can afford them (a small look's islands shrank to a texel or two:
        # a rim's spoke took the black of the margin round it)
        mp = margins_px if tex * min(w, h) >= big_px else tuple(m for m in margins_px if m <= 0.5) or (0.0,)
        for margin in (m / (tex * min(w, h)) for m in mp):
            bpy.ops.object.mode_set(mode='OBJECT'); set_uv(pre); edit_select(gid == g)
            bpy.ops.uv.pack_islands(rotate=False, scale=True, margin_method='FRACTION', margin=margin, shape_method='CONCAVE')
            bpy.ops.object.mode_set(mode='OBJECT')
            uv = get_uv(); lo, hi = uv[lsel].min(0), uv[lsel].max(0)
            if lo.min() >= -1e-4 and hi.max() <= 1 + 1e-4: break
        else:
            raise RuntimeError(f'look {g} does not fit its rectangle')
        uv[lsel, 0] = x + uv[lsel, 0] * w; uv[lsel, 1] = y + uv[lsel, 1] * h; set_uv(uv)
    uv = get_uv(); a2 = uv_area(uv)
    log(f'unwrapped: {NG} looks, uv in [{uv.min():.3f}, {uv.max():.3f}], fill {a2.sum():.3f}, '
        f'body {a2[part == 0].sum() / area3[part == 0].sum() * tex * tex:.0f} px2/m2')
    return rects, gcls, gpart


def rect_pixels(rects, g, tex):
    """The pixel window (rows, cols slices) of look g's rectangle."""
    x, y, w, h = rects[g]
    return slice(int(y * tex), int(np.ceil((y + h) * tex))), slice(int(x * tex), int(np.ceil((x + w) * tex)))


def pixels(img, tex):
    return np.array(img.pixels[:], dtype=np.float32).reshape(tex, tex, 4)


# ---------------------------------------------------------------- the bake
class Baker:
    """Bakes the atlas's corner colours (and the AO) into PNGs <out>/<name>_<map>.png. The albedo is the c_albedo
    colour, or where c_tex says so (R, G: weights per source texture), the source textures through the src UVs."""

    def __init__(self, atlas, out, name, tex, ao_distance, albedo_textures=()):
        self.atlas, self.out, self.name, self.tex = atlas, out, name, tex
        me = atlas.data
        self.attr = self._material(False, ())
        self.albedo = self._material(True, albedo_textures)
        me.materials.clear(); me.materials.append(self.attr); me.materials.append(self.albedo)
        sc = scene()
        sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
        sc.render.bake.margin = 16; sc.render.bake.use_clear = True
        if not sc.world: sc.world = bpy.data.worlds.new('w')
        sc.world.light_settings.distance = ao_distance

    @staticmethod
    def _material(albedo, textures):
        m = bpy.data.materials.new('bake_albedo' if albedo else 'bake'); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        emis = nt.nodes.new('ShaderNodeEmission'); out = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(emis.outputs[0], out.inputs['Surface'])
        attr = nt.nodes.new('ShaderNodeAttribute'); attr.name = 'attr'
        if albedo:
            attr.attribute_name = 'c_albedo'
            prev = attr.outputs['Color']
            if textures:
                w = nt.nodes.new('ShaderNodeAttribute'); w.attribute_name = 'c_tex'
                sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(w.outputs['Color'], sep.inputs[0])
                src = nt.nodes.new('ShaderNodeUVMap'); src.uv_map = 'src'
                for img, ch in textures:
                    t = nt.nodes.new('ShaderNodeTexImage'); t.image = img; t.interpolation = 'Linear'
                    nt.links.new(src.outputs['UV'], t.inputs['Vector'])
                    mix = nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MIX'
                    nt.links.new(sep.outputs[ch], mix.inputs['Fac']); nt.links.new(prev, mix.inputs['Color1'])
                    nt.links.new(t.outputs['Color'], mix.inputs['Color2']); prev = mix.outputs['Color']
            nt.links.new(prev, emis.inputs['Color'])
        else:
            nt.links.new(attr.outputs['Color'], emis.inputs['Color'])
        tgt = nt.nodes.new('ShaderNodeTexImage'); tgt.name = 'target'; nt.nodes.active = tgt; tgt.select = True
        return m

    def bake(self, map_name, kind='EMIT', attr_name=None, colour=False, samples=1):
        tex = self.tex
        img = bpy.data.images.new(map_name, tex, tex, alpha=False)
        img.colorspace_settings.name = 'sRGB' if colour else 'Non-Color'
        use = self.albedo if map_name == 'albedo_base' else self.attr
        for m in (self.attr, self.albedo): m.node_tree.nodes['target'].image = img
        if attr_name: self.attr.node_tree.nodes['attr'].attribute_name = attr_name
        me = self.atlas.data
        me.polygons.foreach_set('material_index', [1 if use is self.albedo else 0] * len(me.polygons)); me.update()
        scene().cycles.samples = samples
        select_only(self.atlas)
        bpy.ops.object.bake(type=kind)
        self.save(img, map_name)
        log('baked', map_name)
        return img

    def save(self, img, map_name):
        img.filepath_raw = os.path.join(self.out, f'{self.name}_{map_name}.png'); img.file_format = 'PNG'; img.save()


def ao_floor(img, rects, gcls, floors, tex, lift=0.08):
    """The AO of the looks of the given classes ({class: floor}) kept at or above floor, and any texel more than lift
    darker than the median round it (3 x 3) brought up to that: a texel of a fold's island, or one the filtering takes
    from next door, reads as a black speck on the panel."""
    px = pixels(img, tex)
    for g, c in enumerate(gcls):
        if c not in floors:
            continue
        r, cc = rect_pixels(rects, g, tex)
        a = px[r, cc, 0].copy()
        pad = np.pad(a, 1, mode='edge')
        win = np.stack([pad[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
        med = np.median(win, 0)
        a = np.maximum(a, med - lift)
        a = np.maximum(a, floors[c])
        px[r, cc, :3] = a[..., None]
    img.pixels.foreach_set(px.ravel()); img.save()


def flatten_looks(img, me, gid, rects, tex, uv_name='atlas'):
    """Each look's rectangle filled with the look's colour, read at the UV centres of its biggest faces (surely
    baked). Only for maps that are one flat colour per look (the albedo with no source textures, the masks)."""
    px = pixels(img, tex)
    nf = len(me.polygons)
    ls = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_total', lt)
    uv = np.zeros(len(me.loops) * 2); me.uv_layers[uv_name].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2) * tex
    nxt = np.arange(len(uv)) + 1; nxt[ls + lt - 1] = ls
    cr = uv[:, 0] * uv[nxt, 1] - uv[nxt, 0] * uv[:, 1]
    ua = 0.5 * np.abs(np.add.reduceat(cr, ls))
    cen = np.add.reduceat(uv, ls) / lt[:, None]
    gid = np.asarray(gid).ravel()
    for g in rects:
        f = np.nonzero(gid == g)[0]
        if not len(f):
            continue
        f = f[np.argsort(-ua[f])][:25]
        ix = np.clip(cen[f].astype(np.int64), 0, tex - 1)
        col = np.median(px[ix[:, 1], ix[:, 0], :3], 0)
        r, c = rect_pixels(rects, g, tex)
        px[r, c, :3] = col
    img.pixels.foreach_set(px.ravel()); img.save()


def blank_looks(img, rects, looks, tex, value=1.0):
    """Sets the given looks' rectangles to a value (lamp lenses and bulbs wrap their own insides: their AO is a noisy
    dark mess, so they get none)."""
    px = pixels(img, tex)
    for g in looks:
        r, c = rect_pixels(rects, g, tex); px[r, c, :3] = value
    img.pixels.foreach_set(px.ravel()); img.save()


def fix_ao_specks(img, me, tex, drop=0.25, max_area=5e-4, grow=1.5, passes=12, uv_name='atlas', only=None):
    """Small faces baked far darker than the faces round them take their neighbours' AO. A sliver tucked into a
    fold (a crease's end, a panel seam) bakes black, buried by its own neighbours, and the game draws it as a black
    streak or speck on the paint; its texels (and grow texels round them: the filtering reaches that far) are
    refilled with the area-weighted AO of the faces sharing its corners. Returns the number of faces refilled."""
    px = pixels(img, tex)
    nf = len(me.polygons)
    ls = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_total', lt)
    area = np.zeros(nf); me.polygons.foreach_get('area', area)
    lv = np.zeros(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', lv)
    uv = np.zeros(len(me.loops) * 2); me.uv_layers[uv_name].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2) * tex
    co = np.zeros((len(me.vertices), 3)); me.vertices.foreach_get('co', co.ravel())
    lf = np.repeat(np.arange(nf), lt)
    # each face's AO: its corners pulled a little toward its centre, and the centre
    cen = np.add.reduceat(uv, ls) / lt[:, None]
    smp = np.concatenate([cen, uv * 0.7 + cen[lf] * 0.3])
    sf = np.concatenate([np.arange(nf), lf])
    ix = np.clip(smp.astype(np.int64), 0, tex - 1)
    val = px[ix[:, 1], ix[:, 0], 0]
    fao = np.bincount(sf, val, nf) / np.bincount(sf, None, nf)
    # the neighbourhood: faces sharing a corner position (the atlas is split along seams: positions, not indices)
    _, vkey = np.unique(np.round(co, 5), axis=0, return_inverse=True); vkey = vkey.ravel()
    lk = vkey[lv]
    # a whole cluster of slivers in a fold bakes black together: passes, each lifting the ones darker than round
    # them, so the light works in from the cluster's edge
    nao = fao.copy(); out = np.zeros(nf, bool)
    for _ in range(passes):
        acc = np.bincount(lk, (nao * area)[lf], vkey.max() + 1); wsum = np.bincount(lk, area[lf], vkey.max() + 1)
        nb_a = np.bincount(lf, acc[lk], nf) - nao * area * lt
        nb_w = np.bincount(lf, wsum[lk], nf) - area * lt
        nb = np.where(nb_w > 1e-12, nb_a / np.maximum(nb_w, 1e-12), nao)
        hit = (area < max_area) & (nao < nb - drop)
        if only is not None:
            hit &= only
        if not hit.any():
            break
        nao = np.where(hit, nb, nao); out |= hit
    for f in np.nonzero(out)[0]:
        t = uv[ls[f]:ls[f] + lt[f]]
        c = t.mean(0)
        d = t - c; n = np.linalg.norm(d, axis=1, keepdims=True)
        t = t + d / np.maximum(n, 1e-9) * grow                     # the polygon grown by grow texels
        x0, y0 = np.maximum(np.floor(t.min(0)).astype(int), 0); x1, y1 = np.minimum(np.ceil(t.max(0)).astype(int), tex - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        q = np.stack([gx.ravel(), gy.ravel()], 1)
        inside = np.ones(len(q), bool); sgn = None
        for i in range(len(t)):
            e = t[(i + 1) % len(t)] - t[i]
            cr = e[0] * (q[:, 1] - t[i][1]) - e[1] * (q[:, 0] - t[i][0])
            s0 = np.sign(e[0] * (c[1] - t[i][1]) - e[1] * (c[0] - t[i][0]))
            inside &= cr * s0 >= 0
        yy, xx = (q[inside, 1]).astype(int), (q[inside, 0]).astype(int)
        px[yy, xx, :3] = np.maximum(px[yy, xx, :3], nao[f])
    img.pixels.foreach_set(px.ravel()); img.save()
    return int(out.sum())


def fix_ao_buried(img, me, tex, distance, start=0.003, rays=48, below=0.2, gain=0.25, grow=1.5, uv_name='atlas',
                  only=None):
    """Faces baked near black re-measured: AO by rays of our own, leaving from start metres off the face (past a
    second skin a fraction of a millimetre over it, which buries it for the bake: the game shows the face, black).
    Where that comes out gain lighter, the face's texels take it; a real crevice stays as dark as it baked."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    px = pixels(img, tex)
    nf = len(me.polygons)
    ls = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(nf, np.int64); me.polygons.foreach_get('loop_total', lt)
    uv = np.zeros(len(me.loops) * 2); me.uv_layers[uv_name].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2) * tex
    lf = np.repeat(np.arange(nf), lt)
    cen = np.add.reduceat(uv, ls) / lt[:, None]
    smp = np.concatenate([cen, uv * 0.7 + cen[lf] * 0.3]); sf = np.concatenate([np.arange(nf), lf])
    ix = np.clip(smp.astype(np.int64), 0, tex - 1)
    fao = np.bincount(sf, px[ix[:, 1], ix[:, 0], 0], nf) / np.bincount(sf, None, nf)
    co = [tuple(v.co) for v in me.vertices]
    tree = BVHTree.FromPolygons(co, [tuple(p.vertices) for p in me.polygons])
    fn = np.zeros((nf, 3)); me.polygons.foreach_get('normal', fn.ravel())
    fc = np.zeros((nf, 3)); me.polygons.foreach_get('center', fc.ravel())
    rng = np.random.default_rng(1)
    u1, u2 = rng.random(rays), rng.random(rays)                # cosine-weighted hemisphere directions
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    hemi = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], 1)
    k = 0
    sel = fao < below if only is None else (fao < below) & only   # only: the faces it may touch (the paint's)
    for f in np.nonzero(sel)[0]:
        n = fn[f]
        t1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0]); t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        dirs = hemi[:, :1] * t1 + hemi[:, 1:2] * t2 + hemi[:, 2:] * n
        o = fc[f] + n * start
        free = sum(tree.ray_cast(Vector(o.tolist()), Vector(d.tolist()), distance)[0] is None for d in dirs)
        ao = free / rays
        if ao < fao[f] + gain:
            continue
        t = uv[ls[f]:ls[f] + lt[f]]; c = t.mean(0)
        dd = t - c; t = t + dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-9) * grow
        x0, y0 = np.maximum(np.floor(t.min(0)).astype(int), 0); x1, y1 = np.minimum(np.ceil(t.max(0)).astype(int), tex - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        q = np.stack([gx.ravel(), gy.ravel()], 1)
        inside = np.ones(len(q), bool)
        for i in range(len(t)):
            e = t[(i + 1) % len(t)] - t[i]
            cr = e[0] * (q[:, 1] - t[i][1]) - e[1] * (q[:, 0] - t[i][0])
            s0 = np.sign(e[0] * (c[1] - t[i][1]) - e[1] * (c[0] - t[i][0]))
            inside &= cr * s0 >= 0
        yy, xx = q[inside, 1].astype(int), q[inside, 0].astype(int)
        px[yy, xx, :3] = np.maximum(px[yy, xx, :3], ao)
        k += 1
    img.pixels.foreach_set(px.ravel()); img.save()
    return k


# ---------------------------------------------------------------- parts, LODs, collider, export
def split_parts(atlas, part_ids):
    """The atlas split back into its parts (by the 'part' face attribute)."""
    def split_part(pid, name):
        o = atlas.copy(); o.data = atlas.data.copy(); o.name = name; scene().collection.objects.link(o)
        bm = bmesh.new(); bm.from_mesh(o.data)
        lay = bm.faces.layers.int['part']
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f[lay] != pid], context='FACES')
        bm.to_mesh(o.data); bm.free()
        return o
    parts = {n: split_part(i, n) for n, i in part_ids.items()}
    bpy.data.objects.remove(atlas)
    return parts


def make_lods(body, lod_tris, report_tris=False):
    """The body's distance versions (decimated copies, stray vertices put back on the full body)."""
    lods = [body]
    full = body.data
    full_bvh = BVHTree.FromPolygons([tuple(v.co) for v in full.vertices], [tuple(p.vertices) for p in full.polygons])
    def snap_strays(o, tol=0.02):
        """Collapsing unwelded pieces can fling a vertex far off; put any vertex more than tol from the full body back on it."""
        co = np.zeros((len(o.data.vertices), 3)); o.data.vertices.foreach_get('co', co.ravel())
        n = 0
        for i, c in enumerate(co):
            loc, _, _, dist = full_bvh.find_nearest(Vector(c))
            if loc is not None and dist > tol:
                co[i] = loc; n += 1
        o.data.vertices.foreach_set('co', co.ravel()); o.data.update()
        return n
    for i, t in enumerate(lod_tris, 1):
        o = body.copy(); o.data = body.data.copy(); o.name = f'body_lod{i}'; scene().collection.objects.link(o)
        decimate(o, t); lods.append(o)
        log(f'lod{i}: ' + (f'{tris(o)} tris, ' if report_tris else '') + f'snapped {snap_strays(o)} stray vertices')
    return lods


def hull_collider(full, max_tris=250):
    """Convex hull of the body's points, planar-dissolved (only removes points, so the shape stays on the hull) until it
    is under max_tris (Unity convex colliders stop at 255). colliders.py slices it along the car afterwards."""
    bm = bmesh.new()
    for v in full.vertices: bm.verts.new(v.co)
    hull = bmesh.ops.convex_hull(bm, input=list(bm.verts))
    bmesh.ops.delete(bm, geom=list({v for v in hull['geom_interior'] + hull['geom_unused'] if isinstance(v, bmesh.types.BMVert)}), context='VERTS')
    cme = bpy.data.meshes.new('collider'); bm.to_mesh(cme); bm.free()
    coll = bpy.data.objects.new('collider', cme); scene().collection.objects.link(coll)
    for deg in (2, 4, 7, 10, 14, 18, 24, 30):
        if tris(coll) <= max_tris: break
        bm = bmesh.new(); bm.from_mesh(cme)
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(deg), verts=bm.verts, edges=bm.edges)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(cme); bm.free()
    log(f'collider {tris(coll)} tris')
    return coll


def export_arrays(o, to_unity, origin=np.zeros(3), smooth=True, smooth_angle=math.radians(35)):
    """A part's vertices, normals, UVs and triangles in the game's axes (split normals; winding by the normals)."""
    m = o.data
    m.use_auto_smooth = smooth; m.auto_smooth_angle = smooth_angle if smooth else math.pi
    m.calc_normals_split(); m.calc_loop_triangles()
    nl = len(m.loops)
    lv = np.zeros(nl, dtype=np.int64); m.loops.foreach_get('vertex_index', lv)
    ln = np.zeros((nl, 3)); m.loops.foreach_get('normal', ln.ravel())
    co = np.zeros((len(m.vertices), 3)); m.vertices.foreach_get('co', co.ravel())
    uv = np.zeros((nl, 2))
    if m.uv_layers: m.uv_layers['atlas'].data.foreach_get('uv', uv.ravel())
    P = to_unity(co[lv]) - origin; N = to_unity(ln)
    key = np.concatenate([lv[:, None].astype(np.float64), np.round(N, 4), np.round(uv, 6)], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    tl = np.zeros(len(m.loop_triangles) * 3, dtype=np.int64); m.loop_triangles.foreach_get('loops', tl)
    I = inv[tl].reshape(-1, 3)
    Pv, Nv, Uv = P[first], N[first], uv[first]
    # the axes swap may be a mirror: then every triangle turns round; after that, only one whose normals clearly face
    # the other way (a refined fold) is turned. A triangle seen edge-on by its normals keeps its winding (by the sign
    # alone it was a coin toss: a bed wall's faces turned away, slits in the bed)
    if np.linalg.det(np.asarray(to_unity(np.eye(3)), float)) < 0:
        I = I[:, ::-1].copy()
    fnrm = np.cross(Pv[I[:, 1]] - Pv[I[:, 0]], Pv[I[:, 2]] - Pv[I[:, 0]])
    fnrm /= np.maximum(np.linalg.norm(fnrm, axis=1, keepdims=True), 1e-20)
    mn = Nv[I].mean(1); mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-20)
    flip = np.einsum('ij,ij->i', fnrm, mn) < -0.3
    I[flip] = I[flip][:, ::-1]
    return Pv.astype('<f4'), Nv.astype('<f4'), Uv.astype('<f4'), I.astype('<u4')


def export_parts(out_parts, to_unity, smooth_angle):
    """[(name, kind, object, origin)] -> {name: (kind, origin, P, N, U, I)}; the collider unsmoothed."""
    arrays = {}
    for nm, k, o, origin in out_parts:
        arrays[nm] = (k, origin) + export_arrays(o, to_unity, origin, smooth=(nm != 'collider'), smooth_angle=smooth_angle)
        log(f'part {nm}: {len(arrays[nm][2])} verts {len(arrays[nm][5])} tris')
    return arrays


# ---------------------------------------------------------------- anchors
TOP = ('cosm_Hood', 'cosm_Roof', 'cosm_Roof_Bar', 'cosm_Bed', 'cosm_Bushcables', 'cosm_Antenna')
SIDE = ('cosm_Snorkel', 'cosm_Side_Left', 'cosm_Side_Right', 'cosm_Mudflap_FL', 'cosm_Mudflap_FR', 'cosm_Mudflap_RL', 'cosm_Mudflap_RR')
FRONT = ('cosm_Plate_Front',)  # Front, Front_Bumper, Winch: mapped only (a reference's recessed lower bumper misleads the fit)
REAR = ('cosm_Rear', 'cosm_Rear_Bumper', 'cosm_Plate_Rear', 'cosm_Ladder', 'cosm_WhipLights')


def read_ref(path):
    """A reference car's anchors and body (Unity axes)."""
    m = vera_mesh.read(path)
    body = m.part('body')
    return m.anchors, body.P.copy(), body.I.copy()


def map_ref_anchors(ref_path, arrays, wheel_centres, prefixes=('cosm_', 'fog_', 'exhaust_')):
    """
    The reference car's cosmetic (and fog / exhaust) anchors put on our body: along the car, the overhangs scale
    between axle and bumper and between the axles each keeps its distance to the nearer axle; across and up by the
    bodies' bounds; then each cosmetic slot keeps the reference's offset from its body surface along the slot's
    natural axis (roof down, sides in, bumpers along the car). cosmscale_* anchors are copied as they are.
    """
    ref, RP, RI = read_ref(ref_path)
    rmin, rmax = RP.min(0), RP.max(0)
    P0 = arrays['body'][2]; tmin, tmax = P0.min(0), P0.max(0)
    rbvh = BVHTree.FromPolygons([tuple(v) for v in RP.tolist()], RI.tolist())
    bvh = BVHTree.FromPolygons([tuple(v) for v in P0.tolist()], arrays['body'][5].tolist())
    rFL, rRL = ref['wheel_FL'][0][2], ref['wheel_RL'][0][2]
    tFL = max(c[2] for c in wheel_centres.values()); tRL = min(c[2] for c in wheel_centres.values())

    def zmap(z):
        if z >= rFL: return tFL + (z - rFL) / (rmax[2] - rFL) * (tmax[2] - tFL)
        if z <= rRL: return tRL - (rRL - z) / (rRL - rmin[2]) * (tRL - tmin[2])
        return tFL - (rFL - z) if rFL - z < z - rRL else tRL + (z - rRL)

    def mapped(pos):
        return np.array([pos[0] * tmax[0] / rmax[0], pos[1] * tmax[1] / rmax[1], zmap(pos[2])])

    def hit(tree, origin, direction, axis):
        h = tree.ray_cast(Vector(origin), Vector(direction))
        return None if h[0] is None else h[0][axis]

    def surface_fit(nm, rpos, tpos):
        def probe(tree, p, lo, hi):
            if nm in TOP: return 1, hit(tree, (p[0], hi[1] + 1, p[2]), (0, -1, 0), 1)
            if nm in SIDE:
                sg = 1 if p[0] >= 0 else -1
                return 0, hit(tree, (sg * (hi[0] + 1), p[1], p[2]), (-sg, 0, 0), 0)
            if nm in FRONT: return 2, hit(tree, (p[0], p[1], hi[2] + 1), (0, 0, -1), 2)
            if nm in REAR: return 2, hit(tree, (p[0], p[1], lo[2] - 1), (0, 0, 1), 2)
            if nm == 'cosm_Side_Step': return 1, hit(tree, (-0.75 * hi[0], lo[1] - 1, p[2]), (0, 1, 0), 1)
            return None, None
        axis, rh = probe(rbvh, rpos, rmin, rmax)
        if axis is None or rh is None: return tpos, 'mapped'
        _, th = probe(bvh, tpos, tmin, tmax)
        if th is None: return tpos, 'mapped (no surface)'
        delta = rpos[axis] - rh
        if abs(delta) > 0.5: return tpos, f'mapped (ref offset {delta:.2f} too big)'
        out = tpos.copy(); out[axis] = th + delta
        return out, f'surface {"xyz"[axis]} {th:.3f} {delta:+.3f}'

    anchors = {}
    for nm, (pos, rot) in ref.items():
        if nm.startswith('cosmscale_'):
            anchors[nm] = (pos, rot)
        elif nm.startswith(tuple(prefixes)):
            tp = mapped(pos)
            if nm.startswith('cosm_'):
                tp, how = surface_fit(nm, pos, tp)
                log(f'anchor {nm}: {np.round(tp, 3)} ({how})')
            anchors[nm] = (tp, rot)
    return anchors, ref


def yaw_quat(deg):
    h = math.radians(deg) / 2
    return (0.0, math.sin(h), 0.0, math.cos(h))


def report_missing(anchors):
    missing = [a for a in ('wheel_FL', 'wheel_FR', 'wheel_RL', 'wheel_RR', 'headlight_L', 'headlight_R', 'tail_L', 'tail_R', 'reverse') if a not in anchors]
    log('missing anchors:', missing or 'none')


def write_mesh(path, arrays, anchors):
    parts = [vera_mesh.Part(nm, k, P, N, U, I.reshape(-1, 3), tuple(float(x) for x in origin))
             for nm, (k, origin, P, N, U, I) in arrays.items()]
    vera_mesh.write(path, vera_mesh.Mesh(1, parts, anchors))
    log('wrote', path)


# ---------------------------------------------------------------- previews
class Previews:
    """Workbench renders of the car in the game's look (albedo with the paint and trim zones coloured, times the AO)."""

    def __init__(self, out, tex, albedo, zones, ao, lights, paint_rgb, trim_rgb, S, uv_map=None):
        self.out, self.S = out, S
        alb, zon, aoa, lig = pixels(albedo, tex), pixels(zones, tex), pixels(ao, tex), pixels(lights, tex)
        paint_c = np.array(paint_rgb, np.float32) / 255; trim_c = np.array(trim_rgb, np.float32) / 255
        look = alb[..., :3] * (1 - zon[..., :1]) + paint_c * zon[..., :1]
        look = look * (1 - zon[..., 1:2]) + trim_c * zon[..., 1:2]
        look *= aoa[..., :1]
        def as_img(name, rgb):
            img = bpy.data.images.new(name, tex, tex, alpha=False); img.colorspace_settings.name = 'sRGB'
            a = np.ones((tex, tex, 4), np.float32); a[..., :3] = np.clip(rgb, 0, 1); img.pixels.foreach_set(a.ravel()); return img
        self.game = as_img('game_look', look)
        self.lit = as_img('game_lit', np.maximum(look, lig[..., :1] * [1, 0.08, 0.05]) + lig[..., 1:2] * 0.9 + lig[..., 2:3] * 0.9)
        prev = bpy.data.materials.new('preview'); prev.use_nodes = True
        pn = prev.node_tree
        self.t1 = pn.nodes.new('ShaderNodeTexImage'); self.t1.image = self.game
        if uv_map:
            uvn = pn.nodes.new('ShaderNodeUVMap'); uvn.uv_map = uv_map; pn.links.new(uvn.outputs['UV'], self.t1.inputs['Vector'])
        pn.nodes.active = self.t1
        bsdf = pn.nodes['Principled BSDF']; pn.links.new(self.t1.outputs['Color'], bsdf.inputs['Base Color'])
        self.material = prev
        sc = scene()
        sc.render.engine = 'BLENDER_WORKBENCH'
        sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'TEXTURE'
        sc.display.shading.show_cavity = True
        sc.display.shading.show_backface_culling = True   # as the game draws: no backs (inside-out parts show)
        sc.render.resolution_x, sc.render.resolution_y = 1400, 900
        self.cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(self.cam); sc.camera = self.cam
        self.cam.data.lens = 50

    def use(self, objects, set_uv=False):
        for o in objects:
            o.data.materials.clear(); o.data.materials.append(self.material)
            if set_uv:
                for p_ in o.data.polygons: p_.material_index = 0
                o.data.uv_layers.active = o.data.uv_layers['atlas']

    def lit_look(self, on):
        self.t1.image = self.lit if on else self.game

    def shoot(self, name, az_deg, el_deg, dist=10.5, target=(0, 0, 0.9), lens=50):
        S, cam = self.S, self.cam
        az, el = math.radians(az_deg), math.radians(el_deg)
        fwd = Vector((0, -S, 0)); side = Vector((S, 0, 0))   # Blender front and right
        dirv = (fwd * math.cos(az) + side * math.sin(az)) * math.cos(el) + Vector((0, 0, math.sin(el)))
        target = Vector(target); cam.data.lens = lens
        cam.location = target + dirv * dist
        cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
        scene().render.filepath = os.path.join(self.out, f'preview_{name}.png')
        bpy.ops.render.render(write_still=True)


def mirror_wheels(parts, wheel_centres, hub, fl_key, right_of, S):
    """Copies of the FL rim and tyre at the other three corners (mirrored on the right), for the previews."""
    for key, c in wheel_centres.items():
        if key == fl_key: continue
        for src in (parts['wheel'], parts['tyre']):
            o = src.copy(); scene().collection.objects.link(o)
            d = c - hub
            bd = Vector((-S * d[0], -S * d[2], d[1]))   # Unity delta -> Blender delta
            if right_of(key):
                ctr = Vector((-S * hub[0], -S * hub[2], hub[1]))
                o.matrix_world = Matrix.Translation(ctr + bd) @ Matrix.Scale(-1, 4, (1, 0, 0)) @ Matrix.Translation(-ctr)
            else:
                o.location = bd
