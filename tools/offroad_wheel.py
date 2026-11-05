"""
Procedural off-road wheel and tyres for build_car.py (Blender 3.6, bmesh), low-poly to match the game.

  make_rim(...)  an 18 x 9 inch beadlock wheel in the style of a modern UTV/truck beadlock: gloss black barrel and
                 beadlock ring with 32 bolt heads, 10 curved spokes (gunmetal sides, machined faces), recessed hub with
                 6 lug nuts and a square centre cap.
  make_tyre(...) an aggressive mud-terrain tyre of a given outer diameter and section width on an 18 inch bead: big
                 staggered centre blocks and shoulder blocks that wrap down onto the sidewall.

Local frame: X along the axle (outer face +X), Y up, Z forward; metres; centred on the tyre's middle (the hub point
the car's wheel part uses). place() maps it to Blender world space for a left wheel at a Unity-space centre.
Materials are named so build_car.py's CLASS_RULES give them their looks: OT_Black (DARK), OT_Gunmetal (RIM),
OT_Machined (METAL), OT_Tyre (TYRE).
"""
import math
import bmesh
import bpy
from mathutils import Vector

RIM_BEAD_R = 18 * 0.0254 / 2      # 0.2286 m: 18 inch bead seat
RIM_HALF_W = 9 * 0.0254 / 2       # 0.1143 m: 9 inch wide

MATERIALS = {  # name: linear base colour
    'OT_Black': (0.014, 0.014, 0.016),
    'OT_Gunmetal': (0.018, 0.018, 0.02),      # all black: spoke sides, hub
    'OT_Machined': (0.03, 0.03, 0.033),       # a touch lighter (satin) on the spoke faces and lugs
    'OT_Tyre': (0.024, 0.024, 0.026),
}


def material(name):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*MATERIALS[name], 1.0)
        m.diffuse_color = (*[c ** (1 / 2.2) for c in MATERIALS[name]], 1.0)   # the viewport / preview colour
    return m


class Builder:
    """A bmesh with a material index per named material, and solid primitives in the wheel's cylindrical frame."""

    def __init__(self, names):
        self.bm = bmesh.new()
        self.names = list(names)

    def mi(self, name):
        return self.names.index(name)

    @staticmethod
    def cyl(x, r, t):
        """(x, radius, angle) -> local (x, y, z); angle 0 is straight up."""
        return Vector((x, r * math.cos(t), r * math.sin(t)))

    def face(self, verts, mat):
        f = self.bm.faces.new(verts)
        f.material_index = self.mi(mat)
        return f

    def revolve(self, profile, seg, mat, t0=0.0, closed_profile=False, mats=None):
        """Surface of revolution of a (x, r) polyline around the axle."""
        rings = []
        for i in range(seg):
            t = t0 + 2 * math.pi * i / seg
            rings.append([self.bm.verts.new(self.cyl(x, r, t)) for x, r in profile])
        n = len(profile)
        for i in range(seg):
            a, b = rings[i], rings[(i + 1) % seg]
            for j in range(n if closed_profile else n - 1):
                k = (j + 1) % n
                m_ = mats[j] if mats else mat
                self.face([a[j], b[j], b[k], a[k]], m_)

    def sector(self, poly, t0, t1, mat, steps=1, top_mat=None, top_edge=None):
        """A convex (x, r) polygon swept around the axle from angle t0 to t1 (a tread block, a ring segment).
        top_edge: index of the polygon edge drawn with top_mat (the block's running face)."""
        secs = []
        for s in range(steps + 1):
            t = t0 + (t1 - t0) * s / steps
            secs.append([self.bm.verts.new(self.cyl(x, r, t)) for x, r in poly])
        n = len(poly)
        self.face(list(reversed(secs[0])), mat)
        self.face(secs[-1], mat)
        for s in range(steps):
            a, b = secs[s], secs[s + 1]
            for j in range(n):
                k = (j + 1) % n
                self.face([a[j], a[k], b[k], b[j]], top_mat if (top_mat and j == top_edge) else mat)

    def prism_x(self, cy, cz, rr, sides, x0, x1, mat, rot=0.0, cap_mat=None):
        """An n-gon prism along the axle at (y, z) = (cy, cz): lug nuts, beadlock bolts."""
        lo, hi = [], []
        for i in range(sides):
            a = rot + 2 * math.pi * i / sides
            y, z = cy + rr * math.cos(a), cz + rr * math.sin(a)
            lo.append(self.bm.verts.new((x0, y, z)))
            hi.append(self.bm.verts.new((x1, y, z)))
        self.face(list(reversed(lo)), mat)
        self.face(hi, cap_mat or mat)
        for i in range(sides):
            j = (i + 1) % sides
            self.face([lo[i], lo[j], hi[j], hi[i]], mat)

    def loft(self, sections, mat, top_mat=None, top_index=None):
        """Quads between consecutive 4-point sections (a spoke), with end caps."""
        vs = [[self.bm.verts.new(p) for p in sec] for sec in sections]
        self.face(list(reversed(vs[0])), mat)
        self.face(vs[-1], mat)
        for s in range(len(vs) - 1):
            a, b = vs[s], vs[s + 1]
            for j in range(4):
                k = (j + 1) % 4
                self.face([a[j], a[k], b[k], b[j]], top_mat if j == top_index else mat)

    def finish(self, name, place):
        bm = self.bm
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        for n in self.names:
            me.materials.append(material(n))
        for v in me.vertices:
            v.co = place(v.co)
        me.update()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        return o


def make_rim(name, place, seg=48):
    b = Builder(['OT_Black', 'OT_Gunmetal', 'OT_Machined'])
    Rb, hw = RIM_BEAD_R, RIM_HALF_W
    # barrel: inner flange, bead seat, drop centre, outer bead seat (gloss black)
    barrel = [(-hw - 0.012, Rb + 0.018), (-hw, Rb + 0.018), (-hw + 0.004, Rb), (-0.03, Rb - 0.012),
              (0.02, Rb - 0.03), (0.06, Rb - 0.03), (hw - 0.01, Rb - 0.004), (hw, Rb)]
    # a solid wall (outer and inner skin), so it shows from both sides through the spokes
    b.revolve(barrel + [(x, r - 0.006) for x, r in reversed(barrel)], seg, 'OT_Black', closed_profile=True)
    # beadlock ring on the outer face, with its bolt circle
    ring_in, ring_out, ring_x0, ring_x1 = Rb - 0.03, Rb + 0.026, hw - 0.004, hw + 0.03
    b.revolve([(ring_x0, ring_out), (ring_x1 - 0.004, ring_out), (ring_x1, ring_out - 0.006), (ring_x1, ring_in + 0.004),
               (ring_x1 - 0.006, ring_in), (ring_x0, ring_in)], seg, 'OT_Black', closed_profile=True)
    bolt_r = (ring_in + ring_out) / 2 + 0.002
    for i in range(32):
        t = 2 * math.pi * (i + 0.5) / 32
        b.prism_x(bolt_r * math.cos(t), bolt_r * math.sin(t), 0.0075, 6, ring_x1 - 0.001, ring_x1 + 0.007, 'OT_Black', rot=t)
    # hub: recessed face, ring, lug nuts, square centre cap
    hub_x = 0.05
    b.revolve([(0.0, 0.212), (0.0, 0.07), (-0.012, 0.07), (-0.012, 0.212)], seg, 'OT_Gunmetal', closed_profile=True)  # the disc behind the spokes
    b.revolve([(hub_x - 0.03, 0.083), (hub_x + 0.022, 0.083), (hub_x + 0.026, 0.078), (hub_x + 0.026, 0.066),
               (hub_x + 0.006, 0.066), (hub_x + 0.006, 0.02), (hub_x - 0.03, 0.02)], seg, 'OT_Gunmetal', closed_profile=True,
              mats=['OT_Gunmetal', 'OT_Machined', 'OT_Gunmetal', 'OT_Gunmetal', 'OT_Black', 'OT_Black', 'OT_Gunmetal'])
    for i in range(6):
        t = 2 * math.pi * i / 6
        b.prism_x(0.048 * math.cos(t), 0.048 * math.sin(t), 0.0095, 6, hub_x + 0.006, hub_x + 0.024, 'OT_Machined', rot=t + 0.5)
    s = 0.024
    for poly, x0, x1, m in (((-s, -s, s, s), hub_x + 0.006, hub_x + 0.03, 'OT_Black'),
                            ((-s * 0.55, -s * 0.55, s * 0.55, s * 0.55), hub_x + 0.03, hub_x + 0.034, 'OT_Machined')):
        y0, z0, y1, z1 = poly
        lo = [b.bm.verts.new((x0, y, z)) for y, z in ((y0, z0), (y1, z0), (y1, z1), (y0, z1))]
        hi = [b.bm.verts.new((x1, y, z)) for y, z in ((y0, z0), (y1, z0), (y1, z1), (y0, z1))]
        b.face(list(reversed(lo)), m); b.face(hi, m)
        for j in range(4):
            k = (j + 1) % 4
            b.face([lo[j], lo[k], hi[k], hi[j]], m)
    # ten curved spokes: from the hub ring out under the beadlock, concave (the hub sits deeper), machined faces
    r0, r1, steps = 0.078, ring_in + 0.006, 6
    for i in range(10):
        base = 2 * math.pi * i / 10
        secs = []
        for s_ in range(steps + 1):
            u = s_ / steps
            r = r0 + (r1 - r0) * u
            t = base + 0.05 * u * u                       # a slight sweep toward the tips
            w = (0.054 - 0.022 * u) / 2                   # chunky at the hub, tapering out
            xf = hub_x + 0.02 + 0.045 * math.sin(u * math.pi * 0.5)   # face: concave, meets the ring
            xb = xf - (0.03 - 0.008 * u)
            # (x, tangential offset) corners; the face (x = xf) is the 2nd edge
            tan = Vector((0.0, -math.sin(t), math.cos(t)))
            rad = Vector((0.0, math.cos(t), math.sin(t)))
            c = rad * r
            secs.append([Vector((xb, 0, 0)) + c - tan * w, Vector((xf, 0, 0)) + c - tan * w * 0.8,
                         Vector((xf, 0, 0)) + c + tan * w * 0.8, Vector((xb, 0, 0)) + c + tan * w])
        b.loft(secs, 'OT_Gunmetal', top_mat='OT_Machined', top_index=1)
    return b.finish(name, place)


def make_tyre(name, place, outer_d_in, width_in, seg=56, pitches=None):
    """A mud-terrain tyre: outer diameter and section width in inches, on the 18 inch bead."""
    b = Builder(['OT_Tyre'])
    R = outer_d_in * 0.0254 / 2
    W = width_in * 0.0254
    hw = W / 2
    Rb = RIM_BEAD_R
    block = 0.024 if outer_d_in >= 35 else 0.021          # tread depth
    Rt = R - block                                       # carcass under the blocks
    side = R - Rb
    # carcass profile, bead to bead over the tread (outer half then mirrored)
    half = [(RIM_HALF_W + 0.004, Rb + 0.004), (hw - 0.012, Rb + 0.03), (hw, Rb + side * 0.5),
            (hw - 0.006, Rt - 0.03), (hw - 0.03, Rt - 0.004), (hw - 0.05, Rt)]
    prof = half + [(-x, r) for x, r in reversed(half)]
    b.revolve(prof, seg, 'OT_Tyre')
    # tread: per pitch, staggered centre blocks and long/short shoulder blocks that wrap onto the sidewall
    n = pitches or int(round(2 * math.pi * R / 0.105))
    gap = 0.012 / R                                       # groove between blocks, as an angle
    for p in range(n):
        t0 = 2 * math.pi * p / n
        dt = 2 * math.pi / n
        for sgn in (1, -1):
            off = 0 if sgn > 0 else dt * 0.5               # the other half staggered by half a pitch
            # centre block (two per pitch across the middle, angled a little by the stagger)
            xa, xb = 0.012, hw * 0.48
            poly = [(xa, Rt - 0.002), (xb, Rt - 0.002), (xb, R), (xa, R)]
            if sgn < 0: poly = [(-x, r) for x, r in reversed(poly)]
            b.sector(poly, t0 + off + gap / 2, t0 + off + dt * 0.62, 'OT_Tyre')
            # shoulder block: big, alternating long/short, wrapping over the shoulder and down the sidewall
            long_ = p % 2 == 0
            xs0 = hw * 0.56
            drop = 0.07 if long_ else 0.045
            poly = [(xs0, Rt - 0.002), (hw - 0.05, Rt - 0.002), (hw - 0.006, Rt - 0.03), (hw + 0.004, R - block - drop),
                    (hw + 0.016, R - block - drop + 0.012), (hw + 0.012, R - 0.02), (hw - 0.02, R), (xs0, R)]
            if sgn < 0: poly = [(-x, r) for x, r in reversed(poly)]
            b.sector(poly, t0 + off + dt * 0.30, t0 + off + dt * (0.95 if long_ else 0.86), 'OT_Tyre')
    return b.finish(name, place)
