"""
Profile for convert_materials.py: the 4th-gen (2003-2009) 4Runner from "Toyota 4Runner" by sadiqminhas (Sketchfab,
CC BY 4.0: https://sketchfab.com/3d-models/toyota-4runner-3afb8daa0a4a4bfea99e1af74eca1bdb), a lifted 2006-2009 Limited
4x4 on mud-terrain tyres, after profiles/4runner4_prep.py (wheels per corner, the model's axles out, lamp
covers out of the glass, the trim split off). Built on the Burro with its rigged axles; anchors from the OTH+ Hilo 22.
Bring your own copy of the model (input/4runner4/).

    blender -b --factory-startup --python profiles/4runner4_prep.py -- input/4runner4/toyota_4runner.glb build/4runner4/prepped.blend
    blender -b --factory-startup --python tools/convert_materials.py -- profiles/4runner4.py build/4runner4/prepped.blend build/4runner4 <hilo22.mesh> 4runner4
"""
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = '4runner4'
TEX = 2048
# the body's distance versions; LOD0 is the source decimated (about 235k triangles with the interior, the badges kept)
LOD0_TRIS = 180000
# the chrome badges (lettering, emblems: about 20k triangles) kept whole: decimated, letters went missing
DECIMATE_KEEP = ('Chrome',)
LOD_TRIS = [50000, 20000, 8000]

# ---- which objects are what (the prep names them)
FRONT_OBJECT, REAR_OBJECT = 'lamp_front', 'lamp_rear'
TYRE_PREFIX, WHEEL_PREFIX = 'Tire', 'Wheel'
CORNERS = ('FL', 'FR', 'RL', 'RR')
LAMP_OBJECTS = ('lamp_',)        # head, fog and tail lamps, indicators: no mud
NO_LENS_OBJECTS = ()
MATERIAL_SWAPS = []
DROP_OBJECTS = ()
EXHAUST_OBJECT = None            # no tail pipe in the model: the Hilo 22's exhaust anchor
HIGH_BRAKE_Y = 1.6
LAMP_MIN_X = 0.3

# ---- material names -> face classes, first match wins
CLASS_RULES = [
    ('rr_lens', EMIT), ('rr_foglens', OTHER), ('Tail_Lights', REDLENS), ('Indicators', AMBER),
    ('Main_Paint', PAINT), ('rr_trim', TRIM), ('Black', DARK),
    ('Glass', GLASS), ('Mirrors', METAL), ('Chrome', METAL),
    ('Rims', RIM), ('Hub_Lock', DETAIL), ('Tyres', TYRE),
]
ALBEDO_OVERRIDE = {GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8), PAINT: (0.8, 0.8, 0.8), TRIM: (0.018, 0.018, 0.02),
                   RIM: (0.03, 0.03, 0.032), METAL: (0.55, 0.55, 0.56), DARK: (0.02, 0.02, 0.022),
                   TYRE: (0.025, 0.025, 0.025)}

# ---- the atlas and the bake
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.6, 'tyre': 0.6}
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45}
UV_MARGINS = (2.0, 1.5, 1.0, 0.5, 0.0)
AO_DISTANCE, AO_SAMPLES = 0.6, 48
AO_FLOOR = {PAINT: 0.7, TRIM: 0.35}
OFFROAD_WHEELS = None
AO_WHEELS_APART = True

# ---- previews, collider
PREVIEW_PAINT, PREVIEW_TRIM = (74, 84, 62), (26, 26, 28)
PREVIEW_LENGTH = 2.45
COLLIDER_SLICES = 6
