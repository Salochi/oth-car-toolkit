"""
Profile for convert_materials.py: the Baja Bug from "VW Beetle" by 3DDomino (Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/vw-beetle-1ffa1ce092184da8bab996bb3a481cf2), after profiles/bajabug_3dd_prep.py (the
fenders kept, the engine out back in a tube basket, a small front bar with rally lamps). Built on the Brick, keeping the
Brick's wheels and rigged axles (the placeholder wheels only set the hubs); anchors from the OTH+ Carpati 24 (aro.mesh).
Bring your own copy of the model (input/beetle-3ddomino/).

    blender -b --factory-startup --python profiles/bajabug_3dd_prep.py -- input/beetle-3ddomino/vw_beetle.glb build/bajabug-3dd/prepped.blend
    blender -b --factory-startup --python tools/convert_materials.py -- profiles/bajabug_3dd.py build/bajabug-3dd/prepped.blend build/bajabug-3dd <aro.mesh> bajabug
"""
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'bajabug'
TEX = 2048
# the body's distance versions (LOD1..3, triangles); LOD0 is the decimated body (LOD0 in the environment, else below)
LOD0_TRIS = 163000   # 150k for the decimated car + the bay walls, kept whole (DECIMATE_KEEP)
# the engine bay's flat two-sided walls: decimated they fold over and lose a side (see-through)
DECIMATE_KEEP = ('bb_under_bay',)
LOD_TRIS = [50000, 20000, 8000]

# ---- which objects are what (by name prefix, anywhere up an object's parents; the prep names them)
FRONT_OBJECT, REAR_OBJECT = 'bb_front', 'bb_rear'
TYRE_PREFIX, WHEEL_PREFIX = 'Tire', 'Wheel'
CORNERS = ('FL', 'FR', 'RL', 'RR')
LAMP_OBJECTS = ('lamp_',)        # the head, tail and rally lamps and the indicators: no mud on them
NO_LENS_OBJECTS = ()
MATERIAL_SWAPS = []
DROP_OBJECTS = ()
EXHAUST_OBJECT = 'bb_stinger'
EXHAUST_YAW = 180                # the stinger points up and back: smoke out behind
HIGH_BRAKE_Y = 1.3
LAMP_MIN_X = 0.3                 # the fender lamps (the rally lamps, nearer the middle, light up but aren't anchors)

# ---- material names -> face classes, first match wins
CLASS_RULES = [
    ('bb_tyre', TYRE), ('bb_rim', RIM),
    ('bb_trim', TRIM),                                         # the tubes: the trim colour
    ('bb_headlens', EMIT), ('bb_lens', EMIT),
    # the engine's metal as DARK: its own grey, and it keeps its AO (METAL's is blanked below)
    ('bb_alu', DARK), ('bb_chrome', DARK), ('bb_black', DARK), ('bb_fins', DARK), ('bb_housing', DARK), ('bb_under', DARK),
    ('Paint_Metallic', PAINT),
    ('Glass_Basic_Red', REDLENS), ('Glass_Basic_Orange', AMBER), ('Glass_Basic_Grey', GLASS),
    ('Steel_Rough', METAL), ('color150150160', METAL), ('Paint_Matte_White', METAL),
    ('Hard_Rough_Plastic_Black', DARK), ('color252525', DARK), ('Tire', DARK), ('Dots', DARK),
]
# linear rgb for classes whose source colour makes no sense without the source shader
ALBEDO_OVERRIDE = {GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8), PAINT: (0.8, 0.8, 0.8), TRIM: (0.018, 0.018, 0.02)}

# ---- the atlas and the bake
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.3, 'tyre': 0.3}
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45}
# texels between the UV islands, tried in turn (the first that packs): half a texel let a panel's edge take the next
# island's colour
UV_MARGINS = (2.0, 1.5, 1.0, 0.5, 0.0)
AO_DISTANCE, AO_SAMPLES = 0.5, 48
# the paint's AO no darker than this: the hood's panel gaps and ridge baked as black lines (the gaps show as gaps anyway)
AO_FLOOR = {PAINT: 0.6}
OFFROAD_WHEELS = None
AO_WHEELS_APART = True

# ---- previews, collider
PREVIEW_PAINT, PREVIEW_TRIM = (52, 80, 104), (40, 40, 48)
PREVIEW_LENGTH = 2.2
COLLIDER_SLICES = 6
