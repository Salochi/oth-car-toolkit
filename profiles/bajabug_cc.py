"""
Profile for convert_texture.py: "Volkswagen Beetle" by yugengen (Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/volkswagen-beetle-d29dbdb9a90146169a8b3261f298145d), one mesh over one baked texture.
Built on the Brick, keeping the Brick's wheels and rigged axles; anchors from the OTH+ Carpati 24.
The model itself is not part of this toolkit: bring your own copy (input/bajabug-cc/volkswagen_beetle.glb).

    blender -b --factory-startup --python tools/convert_texture.py -- profiles/bajabug_cc.py <model.glb> build/bajabug-cc <aro.mesh> bajabug
"""
import numpy as np
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'bajabug'
TEXTURE = 'Image_1'            # the car's texture, as the GLB import names it
DROP_OBJECTS = ('Plane',)      # the asphalt ground plane the model was shown on
FRONT_SIGN = 1.0               # the front (the bonnet and headlamps) is Blender -y
WHEELBASE = 2.40               # metres (the real car's): sets the model's scale
WELD = 1e-5
LOD_TRIS = [14000, 7000, 3000]

# where the wheels are, in the source's units (Blender axes as imported): pieces wholly inside are wheel
WHEEL_SEARCH = dict(axle_y=(-0.46, 0.38), axle_z=0.122, radius=0.14, x=(0.15, 0.36))
RIM_RADIUS = 0.25              # metres from the axle: further out is tyre

# ---- texel classes (vectorised: srgb 0..1 (H,W,3), Unity position and normal per texel, body texels)
BELT_Y = 0.95                  # metres: the side windows' bottom edge is just above this
HEADLAMP = (0.34, 0.92, 1.905)    # |x|, y, z of the headlamp lenses (metres): on the front hoop
TAIL_LAMP = (0.46, 0.67, -2.020)   # on the rear bar
_CACHE = {}                    # what classify() worked out, for albedo() (the same texels, the same run)


def classify(c, pos, nrm, body):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    out = np.full(r.shape, DETAIL, np.int64)
    grey = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) < 0.14
    out[grey] = METAL
    out[lum < 0.16] = DARK
    blue = (b > r + 0.04) & (b > 0.22)
    out[blue] = METAL                                                   # chrome mirrors the sky
    out[blue & (y > BELT_Y)] = GLASS                                    # windows
    near = lambda c, rad: np.hypot(np.hypot(np.abs(x) - c[0], y - c[1]), z - c[2]) < rad
    out[near(HEADLAMP, 0.12) & (blue | grey) & (lum > 0.2)] = LENS
    red = (r > g + 0.07) & (r > b + 0.04)
    out[red & near(TAIL_LAMP, 0.11)] = REDLENS
    yellow = ((r + g) / 2 - b > 0.12) & (r > 0.25) & (g > 0.2)
    out[yellow] = PAINT
    was = out.copy()
    glass0 = body & (out == GLASS) & ~_handles(pos)
    out[_repaint(pos, nrm) & body & np.isin(out, [METAL, DETAIL, DARK, GLASS, LENS])] = PAINT
    # every window redrawn: its outline smoothed, an even chrome ring round it, the source's uneven trim beyond it paint
    glass, ring, band = _windows(pos, glass0, out)
    out[body & (out == GLASS) & ~glass] = PAINT          # glass specks off the windows
    out[body & glass] = GLASS
    out[body & band & ~glass & ~_handles(pos)] = PAINT
    out[body & ring] = METAL
    # specks left on the panels (dirt and smudges baked into the source): paint where paint surrounds them
    out[body & _specks(out, pos)] = PAINT
    # the door handles' own (embossed) shape: the trim colour (black, as the bumpers), the recess round it dark
    out[body & _handles(pos) & np.isin(out, [METAL, GLASS, DETAIL]) & (lum > 0.3)] = TRIM
    # the tubes: the trim colour (the swatch's exact colour, on the added parts only)
    tc = np.array(SWATCHES['trim']) / 255
    out[~body & (np.abs(c - tc).max(-1) < 2.5 / 255)] = TRIM
    _CACHE.update(ring=ring, band=band, filled=body & (out == PAINT) & (was != PAINT))
    return out


def _repaint(pos, nrm):
    """Texels painted over with the body colour: the louvres left on the cut-down engine lid (below the rear window's
    trim), the bonnet badge, the old headlamp rings' rims on the nose's sides, the fender beading left along the hood
    (up to the A pillars) and the rear quarters."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    louvres = (z < -1.38) & (y > 0.88) & (y < 1.32) & (np.abs(x) < 0.5)
    badge = (z > 1.35) & (np.abs(x) < 0.16) & (nrm[..., 1] > -0.2) & (y > 0.55)
    rings = (z > 1.40) & (np.abs(x) > 0.36) & (y < 1.0) & (y > 0.5)
    seams = (z > 0.55) & (z < 1.75) & (np.abs(x) > 0.2) & (y > 0.8) & (y < 1.22)   # windows() puts the windscreen back
    quarters = (z < -1.05) & (np.abs(x) > 0.42) & (y > 0.95) & (y < 1.06)
    return louvres | badge | rings | seams | quarters


def _handles(pos):
    """The door handles (the source's chrome mirrors the sky there, so it read as glass)."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    return (np.abs(x) > 0.6) & (z > -0.53) & (z < -0.11) & (y > 0.93) & (y < 1.065)


TRIM_WIDTH, TRIM_CLEAR = 0.013, 0.035   # metres: the chrome ring round each window, the paint beyond it
WINDOW_SMOOTH = 0.008                   # metres: notches and bumps in a window's outline smaller than this go
GRID = 0.0025


def _grid_ops(cell):
    def dilate(g, rad):
        out_ = g.copy(); r_ = int(round(rad / cell))
        for dx in range(-r_, r_ + 1):
            for dy in range(-r_, r_ + 1):
                if (dx or dy) and dx * dx + dy * dy <= r_ * r_: out_ |= np.roll(np.roll(g, dx, 0), dy, 1)
        return out_
    erode = lambda g, rad: ~dilate(~g, rad)
    return dilate, erode


def _windows(pos, glass, cls):
    """Every window redrawn on a grid across its own plane (2.5 mm): the glass outline closed and opened by
    WINDOW_SMOOTH (no notches, no specks), a TRIM_WIDTH ring round it, a band past that to TRIM_CLEAR.
    Returns (glass, ring, band) texel masks."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    G = np.zeros(x.shape, bool); R = G.copy(); B = G.copy()
    dilate, erode = _grid_ops(GRID)
    # (area, u, v, glass's own extra test, more glass, texels left as they are, smoothing): the rear window, the
    # windscreen (shaded dark towards its top, its wipers across it: both count as glass for the outline, and stay
    # as they are), the side windows each side
    dark = cls == DARK
    views = [((z < -1.0) & (np.abs(x) < 0.65) & (y > 1.0) & (y < 1.75), x, y * 0.587 + z * 0.81, y > 1.255, None, None, WINDOW_SMOOTH),
             ((z > 0.15) & (z < 1.0) & (np.abs(x) < 0.68) & (y > 1.05), x, y * 0.83 - z * 0.56, y > 1.12, dark & (y > 1.17), dark, 0.03)]
    for sx in (-1, 1):
        views.append(((sx * x > 0.45) & (np.abs(x) < 0.82) & (z > -1.2) & (z < 0.7) & (y > 0.98) & (y < 1.62),
                      z, y, y > 1.07, None, None, WINDOW_SMOOTH))
    for area, u, v, extra, more, keep, sm in views:
        gl = (glass | (more if more is not None else False)) & area & extra
        if gl.sum() < 200: continue
        u0, v0 = u[area].min() - 0.05, v[area].min() - 0.05
        nu, nv = int((u[area].max() + 0.05 - u0) / GRID) + 1, int((v[area].max() + 0.05 - v0) / GRID) + 1
        ij = lambda m: (np.clip(((u[m] - u0) / GRID).astype(int), 0, nu - 1), np.clip(((v[m] - v0) / GRID).astype(int), 0, nv - 1))
        g = np.zeros((nu, nv), bool); g[ij(gl)] = True
        g = erode(dilate(g, sm), sm)          # close: notches and pinholes
        g = dilate(erode(g, sm), sm)          # open: specks and spurs
        n1, n2 = dilate(g, TRIM_WIDTH), dilate(g, TRIM_CLEAR)
        i, j = ij(area)
        kp = keep[area] if keep is not None else np.zeros(i.shape, bool)
        G[area] |= g[i, j] & ~kp
        R[area] |= n1[i, j] & ~g[i, j]
        B[area] |= n2[i, j] & ~n1[i, j]
    R &= ~G; B &= ~(G | R)
    return G, R, B


def _box_mean(m, r):
    """The share of True in each (2r+1)^2 texel square (an integral image)."""
    a = np.pad(m.astype(np.float32), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    s = a[2 * r + 1:, 2 * r + 1:] - a[:-2 * r - 1, 2 * r + 1:] - a[2 * r + 1:, :-2 * r - 1] + a[:-2 * r - 1, :-2 * r - 1]
    return s / (2 * r + 1) ** 2


def _specks(cls, pos):
    """Non-paint texels on the plain panels (the cowl by the A pillars, the pillars, the lower doors) that paint all
    but surrounds: the source's baked smudges. The door seams stay (they run outside these boxes)."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    ax = np.abs(x)
    panels = (((z > 0.45) & (z < 1.0) & (ax > 0.3) & (ax < 0.8) & (y > 0.95) & (y < 1.25))      # the cowl
              | ((ax > 0.55) & (z > -0.62) & (z < 0.62) & (y > 0.32) & (y < 0.93))              # the lower doors
              | ((ax > 0.5) & (z > -1.45) & (z < -0.95) & (y > 0.95) & (y < 1.45))              # the C pillars
              | ((ax > 0.55) & (z > -0.35) & (z < -0.05) & (y > 1.05) & (y < 1.5)))             # the B pillars
    panels &= ~_handles(pos)
    share = _box_mean(cls == PAINT, 5)
    return panels & np.isin(cls, [METAL, DETAIL, DARK]) & (share > 0.55)


CHROME = (0.80, 0.80, 0.82)       # sRGB: the windows' trim
HANDLE = (0.10, 0.10, 0.11)       # sRGB: the handles' base (the trim colour replaces it)
PAINT_FILL = (0.80, 0.76, 0.22)   # sRGB: the source's paint yellow, for texels painted over
PAINT_BASE_GREY = 0.6             # the paint texels' base: grey (see convert_texture.py)


def albedo(alb, srgb, cls, pos, nrm, source):
    """The texels classify() turned to paint get full paint (paint_texels reads the result), and the louvres all of
    theirs; the windows' rings one even chrome; the handles dark under the trim colour; the source's yellow tint gone
    from what isn't paint."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    ring, band, filled = _CACHE['ring'], _CACHE['band'], _CACHE['filled']
    lid = (_repaint(pos, nrm) & (z < -1.38) & (np.abs(x) < 0.5)) | band
    over = source & (cls == PAINT) & (filled | (lid & (paint_texels(alb) < 0.9)))
    alb[over] = PAINT_FILL
    alb[source & (cls == METAL) & ring] = CHROME
    alb[source & (cls == TRIM)] = HANDLE
    # the window seals, the mirrors' and the trim's edges: the source's yellow tints them olive (green under a blue
    # paint once it mixes in): grey, their lightness kept
    tint = source & np.isin(cls, [METAL, DARK, DETAIL]) & ((alb[..., 0] + alb[..., 1]) / 2 - alb[..., 2] > 0.04)
    lum = 0.3 * alb[..., 0] + 0.59 * alb[..., 1] + 0.11 * alb[..., 2]
    alb[tint] = lum[tint][:, None]
    return alb


def paint_texels(c):
    """How much of a paint texel is paint: the yellow over blue margin (seams and panel gaps stay dark)."""
    return np.clip(((c[..., 0] + c[..., 1]) / 2 - c[..., 2] - 0.12) / 0.10, 0.0, 1.0)


def exhaust(bpos, rear_axle_z):
    """The left tailpipe's end: up and out over the rear bar."""
    return np.array(EXHAUST_TIP(-1))


EXHAUST_YAW = 180

# ---- maps
ALBEDO_OVERRIDE = {GLASS: (0.02, 0.02, 0.025), LENS: (0.6, 0.6, 0.6)}
AO_DISTANCE, AO_SAMPLES = 0.5, 64

# ---- previews
PREVIEW_PAINT, PREVIEW_TRIM = (52, 80, 104), (26, 26, 28)   # Faded Blue: a cool paint shows a yellow base bleeding as green
PREVIEW_STANDIN_WHEEL = (0.404, 0.31)   # the Brick's wheel, drawn plain at each hub
COLLIDER_SLICES = 6


# ---- the Baja conversion (Unity metres: x right, y up, z forward; front axle z +1.2, rear -1.2, hubs at y 0.37)
# Fenderless: the fenders cut back to arches that follow the wheels (a rolled lip on every cut), tube bumpers front and
# rear carry round lamps, a skid plate under the nose, the flat four out in the open behind a tube cage.
SWATCHES = {'alu': (170, 170, 166), 'black': (30, 30, 32), 'chrome': (205, 205, 210), 'trim': (36, 36, 40),
            'paint': tuple(round(c * 255) for c in PAINT_FILL), 'lens': (214, 218, 224), 'red': (200, 26, 22)}
SWATCH_CELL = 112
FRONT_FENDER_Z, REAR_FENDER_Z = 0.72, -0.68   # the doors' front and rear edges
SILL_Y = 0.28                                 # the doors' bottom edge: where the arches start
SHOULDER_Y = 0.97                             # the fender line along the body sides
TUBE_R, TUBE_SIDES = 0.024, 20               # the bumpers' tubes: radius, sides round
LIP_R = 0.011
# The arches: a quarter ellipse from the door's bottom corner up over the wheel (side view), as the old fenders ran:
# about 0.1 m over the tyre round its upper front (rear) quarter. Rear: up to the engine lid's edge behind the wheel;
# front: up to the hood's edge ahead of it, then along it (seen from above) to the nose.
REAR_ARCH = dict(zc=-1.80, yc=SILL_Y, a=1.12, b=0.74)      # centre (z, y), half-axes (along, up)
FRONT_ARCH = dict(zc=1.45, yc=SILL_Y, a=0.73, b=0.74)    # its top at the crown where the hood-edge cut takes over
FRONT_SPLIT_Z = 1.45                          # ahead of this the front fender comes off along the hood's edge (z, |x|):
HOOD_EDGE = [(1.45, 0.545), (1.55, 0.53), (1.65, 0.515), (1.80, 0.50), (2.40, 0.50)]
HOOD_UNDER = [(1.45, SHOULDER_Y), (1.55, 0.92), (1.62, 0.87), (1.70, 0.80), (2.40, 0.80)]   # side view, just under it
FRONT_Z = 1.85                                # the front hoop (0.10 m in from where it was: two tubes' widths)
SKID = dict(top=(0.42, 0.64, 1.70), bottom=(0.32, 0.34, 1.38))   # (half-width, y, z) of its top and bottom edges
ENGINE_Z = -1.62                              # the engine's middle (0.18 m forward of where it was: the departure angle)
REAR_BAR_Z, REAR_BAR_Y = -1.98, 0.58
EXHAUST_TIP = lambda sx: (sx * 0.17, 0.62, ENGINE_Z - 0.56)   # up and back, over the bar


def _ellipse(zc, yc, a, b, sign, n=14):
    """A quarter ellipse (side view, (z, y) points, z rising) from the door's bottom corner (zc + sign * a, yc) up to
    its top (zc, yc + b)."""
    t = np.linspace(0, np.pi / 2, n)
    pts = [(zc + sign * a * np.cos(s), yc + b * np.sin(s)) for s in t]
    return pts[::-1] if sign > 0 else pts


def _lamp(e, centre, radius, depth, facing, lens):
    """A round lamp pod facing along z (facing +1 front, -1 rear): a black can, its lens a disc on the open end, a
    stem down to the tube under it."""
    cx, cy, cz = centre
    e.cylinder((cx, cy, cz), radius, depth, 'z', 'black', segments=24)
    e.cylinder((cx, cy, cz + facing * (depth / 2 + 0.004)), radius * 0.86, 0.01, 'z', lens, segments=24)
    e.cylinder((cx, cy - radius - 0.03, cz), 0.014, 0.07, 'y', 'trim', segments=8)


def _engine(e, zc):
    """The flat four round its middle zc (Unity z; the fan to the rear), in three swatch cells (black, alu, chrome)."""
    e.group = 'engine'
    e.box((0, 0.47, zc), (0.34, 0.26, 0.44), 'alu', bevel=0.04)                         # crankcase
    e.box((0, 0.64, zc + 0.12), (0.10, 0.07, 0.08), 'alu', bevel=0.015)                 # oil filler and breather
    e.cylinder((0, 0.70, zc + 0.12), 0.02, 0.07, 'y', 'black', segments=10)
    for sx in (-1, 1):
        for dz in (-0.095, 0.095):                                                       # the cylinders, finned
            e.cylinder((sx * 0.29, 0.50, zc + dz), 0.058, 0.16, 'x', 'black', segments=16)
            for k in range(5):
                e.cylinder((sx * (0.22 + 0.03 * k), 0.50, zc + dz), 0.074, 0.008, 'x', 'black', segments=16)
        e.box((sx * 0.405, 0.50, zc), (0.07, 0.17, 0.40), 'alu', bevel=0.02)            # head
        e.cylinder((sx * 0.455, 0.50, zc), 0.055, 0.36, 'z', 'alu', segments=16)        # rocker cover
        e.box((sx * 0.29, 0.585, zc), (0.26, 0.012, 0.44), 'black', bevel=0.004)        # cylinder tin
        # heat exchanger: from under the heads forward, and back up to the muffler
        e.tube([(sx * 0.35, 0.36, zc + 0.34), (sx * 0.35, 0.36, zc - 0.08), (sx * 0.22, 0.39, zc - 0.28)], 0.038, 'black',
               bend=0.08, segments=14)
        e.tube([(0.0, 0.80, zc - 0.03), (sx * 0.14, 0.80, zc - 0.03), (sx * 0.34, 0.64, zc)], 0.018, 'alu',
               bend=0.05, segments=10)                                                  # intake manifold
        for dz in (-0.095, 0.095):                                                       # plug leads
            e.tube([(0.0, 0.72, zc + 0.10), (sx * 0.18, 0.71, zc + 0.10), (sx * 0.43, 0.58, zc + dz)], 0.006, 'black',
                   bend=0.04, segments=6, caps=False)
    e.cylinder((0, 0.68, zc + 0.10), 0.03, 0.10, 'y', 'black', segments=12)              # distributor
    # fan shroud and housing (the fan at the rear), alternator on it, the belt to the crank pulley
    e.box((0, 0.68, zc - 0.06), (0.62, 0.15, 0.24), 'black', bevel=0.05)
    e.cylinder((0, 0.72, zc - 0.21), 0.165, 0.10, 'z', 'black', segments=32)
    e.cylinder((0, 0.88, zc - 0.10), 0.07, 0.20, 'z', 'alu', segments=16)                # alternator
    e.cylinder((0, 0.88, zc - 0.23), 0.045, 0.03, 'z', 'alu', segments=16)               # its pulley
    e.cylinder((0, 0.55, zc - 0.245), 0.10, 0.04, 'z', 'alu', segments=32)               # crank pulley
    e.tube([(-0.10, 0.55, zc - 0.245), (-0.045, 0.88, zc - 0.245), (0.0, 0.93, zc - 0.245), (0.045, 0.88, zc - 0.245),
            (0.10, 0.55, zc - 0.245), (0.0, 0.45, zc - 0.245)], 0.008, 'black', bend=0.04, closed=True, segments=6)
    # carburettor and air cleaner on top
    e.cylinder((0, 0.86, zc - 0.03), 0.035, 0.14, 'y', 'alu', segments=16)
    e.cylinder((0, 0.97, zc - 0.03), 0.14, 0.08, 'y', 'black', segments=32)
    e.cylinder((0, 1.012, zc - 0.03), 0.145, 0.006, 'y', 'alu', segments=32)
    e.cylinder((0, 1.02, zc - 0.03), 0.02, 0.02, 'y', 'alu', segments=8)                # wing nut
    # muffler (a can across the back, tucked up) and two tailpipes up and out over the bar
    e.cylinder((0, 0.39, zc - 0.315), 0.055, 0.60, 'x', 'black', segments=24)
    for sx in (-1, 1):
        e.cylinder((sx * 0.30, 0.39, zc - 0.315), 0.058, 0.012, 'x', 'alu', segments=24)  # end caps
        tip = EXHAUST_TIP(sx)
        e.tube([(sx * 0.17, 0.39, zc - 0.33), (sx * 0.17, 0.42, zc - 0.45), tip], 0.026, 'chrome', bend=0.06, segments=16)
    e.group = None


def edit(e, log):
    side = lambda c: abs(c[0]) > 0.5   # the fenders (the hood, apron and engine lid are narrower)
    # clean edges for the cuts to end on: the door edges, and the engine bay sides
    e.slice(lambda c: side(c) and abs(c[2] - FRONT_FENDER_Z) < 0.15, (0, 0, FRONT_FENDER_Z), (0, 0, 1))
    e.slice(lambda c: side(c) and abs(c[2] - REAR_FENDER_Z) < 0.15, (0, 0, REAR_FENDER_Z), (0, 0, 1))
    for sx in (-1, 1):
        e.slice(lambda c: abs(abs(c[0]) - 0.5) < 0.15 and c[2] < REAR_FENDER_Z, (sx * 0.5, 0, 0), (1, 0, 0))
    # bumpers, the front plate and the apron under the nose
    e.cut(lambda c: c[2] > 1.66, np.array([0, 0.63, 0]), np.array([0, -1.0, 0]), 'front bumper, plate and apron')
    e.cut(lambda c: c[2] > 1.38 and abs(c[0]) <= 0.5, np.array([0, 0.62, 0]), np.array([0, -1.0, 0]), 'the valance under the nose')
    e.cut(lambda c: c[2] < -1.94, np.array([0, 0.63, 0]), np.array([0, -1.0, 0]), 'rear bumper')
    # the front fenders: an arch from the door's bottom corner up over the wheel; ahead of FRONT_SPLIT_Z (where the
    # arch has risen over the fender's crown) a cut along the hood's edge seen from above, down to the nose
    front = lambda c: side(c) and c[2] > FRONT_FENDER_Z
    e.cut_curve(lambda c: front(c) and c[2] <= FRONT_SPLIT_Z + 0.02,
                [p for p in _ellipse(**FRONT_ARCH, sign=-1) if p[0] <= FRONT_SPLIT_Z] + [(FRONT_SPLIT_Z + 0.02, SHOULDER_Y)],
                'front arch')
    e.cut_plan(lambda c: abs(c[0]) > 0.45 and c[2] > FRONT_SPLIT_Z, HOOD_EDGE, 'front fender (hood edge)')
    e.cut_curve(lambda c: side(c) and c[2] > FRONT_SPLIT_Z, HOOD_UNDER, 'front fender (under the hood edge)')
    # the nose's lower corners rounded off (seen from the front)
    e.cut_curve(lambda c: c[2] > 1.38 and c[1] < 0.95, [(0.0, 0.63), (0.26, 0.63), (0.36, 0.645), (0.42, 0.67), (0.46, 0.71),
                                                         (0.49, 0.77), (0.51, 0.85)], 'nose corners', axis='x')
    # the rear fenders: an arch from the door's bottom corner up over the wheel to the engine lid's edge behind it
    rear = lambda c: side(c) and c[2] < REAR_FENDER_Z
    e.cut_curve(rear, [(-2.40, REAR_ARCH['yc'] + REAR_ARCH['b'])] + _ellipse(**REAR_ARCH, sign=1), 'rear arch')
    # the engine lid, rear apron and pipes: an open engine bay below the louvres' top
    e.cut(lambda c: abs(c[0]) <= 0.5 and c[2] < -1.50, np.array([0, 1.02, 0]), np.array([0, -1.0, 0]), 'engine lid and apron')
    # the source's own inner wheel-well walls (|x| 0.53, under the hood and the rear quarters): ours replace them
    e.delete([f for f in e._body() for c in [e.centre(f)]
              if 0.505 < abs(c[0]) < 0.62 and (0.72 < c[2] < 1.70 or -2.4 < c[2] < REAR_FENDER_Z) and c[1] < 0.86],
             'the inner wheel wells')
    e.drop_small(150)
    # rolled lips on the cuts' edges (before the lining, which has open edges of its own)
    e.lip(lambda c: True, LIP_R, 'paint')
    # the shell is hollow: a black lining inside it, wherever a cut lets you look in (the lower body, all of the rear)
    e.shell(lambda c: c[1] < 1.3 or c[2] < -0.6, 'black')
    for sx in (-1, 1):
        # the wheel wells: a black wall from the floor up to the skin (closing the trunk, the space under the rear seat
        # and the engine bay's sides); at the front its foot follows the skid plate's top edge
        sk_t, sk_b = SKID['top'], SKID['bottom']
        foot = lambda x, z: SILL_Y + 0.02 if z < sk_b[2] else sk_b[1] + (z - sk_b[2]) / (sk_t[2] - sk_b[2]) * (sk_t[1] - sk_b[1]) - 0.02
        e.wall('x', sx * 0.49, FRONT_FENDER_Z - 0.01, sk_t[2], foot, 1.2, 'black')
        e.wall('x', sx * 0.49, -1.60, REAR_FENDER_Z + 0.01, SILL_Y, 1.05, 'black')
    e.box((0, 0.66, -1.52), (1.00, 0.76, 0.03), 'black')   # the firewall behind the rear seat
    _engine(e, ENGINE_Z)
    # ---- front: a tube hoop with two lamps on top (its legs on the lower bar's straight), a skid plate under the nose
    fz = FRONT_Z
    e.tube([(-0.44, 0.42, 1.50), (-0.58, 0.44, fz), (0.58, 0.44, fz), (0.44, 0.42, 1.50)], TUBE_R, 'trim', bend=0.12, segments=TUBE_SIDES)
    e.tube([(-0.48, 0.44, fz), (-0.48, 0.80, fz), (0.48, 0.80, fz), (0.48, 0.44, fz)], TUBE_R, 'trim', bend=0.10, segments=TUBE_SIDES)
    for sx in (-1, 1):
        _lamp(e, (sx * HEADLAMP[0], HEADLAMP[1], HEADLAMP[2] - 0.055), 0.085, 0.10, 1, 'lens')
    _skid(e)
    # ---- rear: a tube bumper round the engine, a cage over it, two lamps on the bar
    bz, by = REAR_BAR_Z, REAR_BAR_Y
    e.tube([(-0.55, by, -1.66), (-0.55, by, bz), (0.55, by, bz), (0.55, by, -1.66)], TUBE_R, 'trim', bend=0.12, segments=TUBE_SIDES)
    for sx in (-1, 1):
        e.tube([(sx * 0.40, by, bz), (sx * 0.40, 0.98, bz + 0.13), (sx * 0.40, 1.03, -1.70)], TUBE_R, 'trim', bend=0.12, segments=TUBE_SIDES)
        _lamp(e, (sx * TAIL_LAMP[0], TAIL_LAMP[1], TAIL_LAMP[2] + 0.035), 0.045, 0.06, -1, 'red')
    e.tube([(-0.40, 0.965, bz + 0.12), (0.40, 0.965, bz + 0.12)], TUBE_R * 0.85, 'trim', caps=False, segments=TUBE_SIDES)


def _skid(e):
    """The skid plate: narrower at its foot, folded down at the sides, two ribs and its bolts."""
    (wt, yt, zt), (wb, yb, zb) = SKID['top'], SKID['bottom']
    out = (0, -0.7, 1)
    e.group = 'skid'
    e.poly([(-wt, yt, zt), (wt, yt, zt), (wb, yb, zb), (-wb, yb, zb)], [(0, 1, 2, 3)], 'alu', thickness=0.01, out=out)
    for sx in (-1, 1):   # the folded sides
        e.poly([(sx * wt, yt, zt), (sx * wb, yb, zb), (sx * (wb - 0.01), yb + 0.05, zb + 0.01), (sx * (wt - 0.01), yt + 0.05, zt - 0.03)],
               [(0, 1, 2, 3)], 'alu', thickness=0.006, out=(sx, -0.3, 0))
        e.tube([(sx * 0.16, yb - 0.006, zb + 0.02), (sx * 0.20, yt - 0.006, zt - 0.02)], 0.008, 'alu', segments=8)   # a rib
        for t in (0.08, 0.92):   # bolts
            p = np.array([sx * (wb + (wt - wb) * t - 0.04), yb + (yt - yb) * t, zb + (zt - zb) * t])
            e.cylinder(tuple(p + np.array([0, -0.008, 0.006])), 0.012, 0.012, 'y', 'alu', segments=8)
    e.group = None
