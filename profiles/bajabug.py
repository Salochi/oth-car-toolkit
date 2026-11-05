"""
Profile for convert_palette.py: "Apocalyptic Volkswagen Baja Bug" (Sketchfab), a low-poly POLYGON-style model with one
palette texture. Built on the Brick, keeping the Brick's wheels and rigged axles; anchors from the OTH+ Carpati 24.
The model itself is not part of this toolkit: bring your own copy.

    blender -b --factory-startup --python tools/convert_palette.py -- profiles/bajabug.py <model.glb> build/bajabug <aro.mesh> bajabug
"""
import numpy as np
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'bajabug'
TEX = 2048
LOD_TRIS = [3600, 2200, 1200]   # LOD1..3; the whole car is ~6k triangles, LOD0 keeps it all
WHEELBASE = 2.40                # metres (the real car's): sets the model's scale
FRONT_SIGN = 1.0                # +1: the front (the bull bar) is Blender -y (checked on renders of the source)
WELD = 2e-4                     # metres: the source is unwelded triangles; welding lets the export smooth the curves
RIM_RADIUS = 0.275              # metres: wheel faces further out from the axle are the tyre

# ---- objects
DROP_OBJECTS = ('Cube', 'SM_Veh_Buggy_Glass_Armour')   # Sketchfab's leftover cube; the window louvres
BODY_TEXTURE, WHEEL_TEXTURE = 'Image_0', 'Image_2'      # the palette atlas and the wheel texture, as the GLB names them


def kind_of(name):
    if name.startswith('Wheel2'): return 'wheel'
    if 'Glass' in name: return 'glass'
    if 'Armour' in name: return 'armour'
    return 'body'


# The armour (one object of loose pieces): pieces whose centre is further forward than this (Unity z, metres) stay
# (the bull bar, skid plate and its four lamps); the roof cage and lamps, door plates, spoiler and pipes go.
ARMOUR_KEEP_Z = 1.0

# ---- palette cells (sRGB 0..255, as sampled at the face) -> class; first within PALETTE_TOL wins; the third value
# limits a cell to one object kind
PALETTE = [
    ((136, 136, 136), EMIT, 'armour'),   # the bull bar's lamp lenses
    ((120, 32, 32), REDLENS, None),      # tail lamps
    ((120, 88, 32), AMBER, None),        # indicators over the tail lamps
    ((168, 176, 168), METAL, None),      # the side trim strip
    ((112, 120, 128), METAL, None), ((96, 104, 112), METAL, None),   # shocks, engine tin, the bar's lamp shells
    ((40, 40, 48), TRIM, 'body'),        # fenders, running boards, bumpers
]
PALETTE_TOL = 14
TRIM_MAX_Y = 1.22   # metres: the fender tops; black above (window seals) is not trim


def is_paint_colour(c):
    """A palette face in the paint's colour (blue)."""
    return c[2] > c[0] + 20


def is_trim(pu):
    """Trim-coloured faces that really are the outside black (Unity point): not the engine out in the open at the back,
    not the window seals."""
    return not (pu[1] > TRIM_MAX_Y or (abs(pu[0]) < 0.48 and pu[2] < -1.05))


def paint_texels(rgb):
    """Per texel, how much is paint (0..1) on the painted faces: blue over red by a margin (wear and dirt are red over
    blue, greys even); the texture's own light and shade don't matter (the paint replaces the texel)."""
    return np.clip((rgb[..., 2] - rgb[..., 0] - 0.07) / 0.05, 0.0, 1.0)


def exhaust(uc, part, rear_axle_z):
    """The engine sits out in the open at the back: smoke from under its rear edge, on the left."""
    eng = uc[(part == 0) & (uc[:, 2] < rear_axle_z - 0.2) & (np.abs(uc[:, 0]) < 0.35) & (uc[:, 1] < 0.9)]
    if not len(eng): return None
    return np.array([-0.2, max(0.45, eng[:, 1].min()), eng[:, 2].min()])


# ---- the car runs on the donor's wheels (KeepDonorWheels: the Brick's, radius 0.40 m, 0.31 m wide) and rigged axles
# (KeepDonorAxles): the model's own hub pieces inside that wheel (within radius of the axle, further out than inward
# inside the hub's centre plane) go, and so does its suspension
HUB_CLEAR = (0.30, 0.17)
# a loose piece of the body is suspension when it is small, near an axle, below the fender tops and reaches out toward
# a wheel; the small blocks on the centre line at each axle that hang lowest (the transaxle housing) go too
SUSPENSION = dict(max_faces=200, axle_dz=0.35, top=1.12, bottom=0.85, reach=0.5,
                  diff_max_faces=40, diff_half_width=0.2, diff_bottom=0.45)

# ---- maps
# albedo overrides (linear rgb) where the source colour makes no sense without the source shader
ALBEDO_OVERRIDE = {GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8)}
TYRE_DARKEN = 0.42   # the source's light grey photo tread, darkened to rubber
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.6, 'tyre': 0.6}
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45, METAL: 0.6}
AO_DISTANCE, AO_SAMPLES = 0.5, 64

# ---- previews
PREVIEW_PAINT, PREVIEW_TRIM = (52, 80, 104), (26, 26, 28)
PREVIEW_STANDIN_WHEEL = (0.404, 0.31)   # the Brick's wheel, drawn plain at each hub
COLLIDER_SLICES = 6
