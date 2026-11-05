"""
A starting point for a convert_materials.py profile: copy it, rename it after your car and fill in the names from your
model (Blender's outliner for the objects, the material list for the materials). profiles/tacoma.py is a complete one.

    blender -b --factory-startup --python tools/convert_materials.py -- profiles/my_car.py my_car.glb build/my_car <reference.mesh> my_car
"""
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'my_car'
TEX = 2048
LOD_TRIS = [60000, 25000, 10000]          # LOD1..3 triangles (LOD0: the whole body)

# ---- objects, by name prefix (anywhere up an object's parents)
FRONT_OBJECT, REAR_OBJECT = 'bumper_front', 'bumper_rear'     # one object at each end: tells the front from the back
TYRE_PREFIX, WHEEL_PREFIX = 'Tire', 'Wheel'                   # tyres and rims are named <prefix>_<corner>
CORNERS = ('Front_Left', 'Front_Right', 'Rear_Left', 'Rear_Right')
LAMP_OBJECTS = ('headlight',)    # lamp assemblies: no dirt on them
NO_LENS_OBJECTS = ()             # lamps whose clear outer cover should go (it z-fights the coloured lens under it)
MATERIAL_SWAPS = []              # (object prefix, material substring, replacement material substring)
DROP_OBJECTS = ()                # whole objects to leave out (an undercarriage, a leftover plane...)
EXHAUST_OBJECT = 'exhaust'       # the tail pipe (None: the reference car's exhaust anchor, mapped)
EXHAUST_YAW = None               # degrees the smoke leaves at (Unity: 0 forward, 180 back, -90 left); None: the reference's
HIGH_BRAKE_Y = 1.3               # rear emitters above this (metres) are a third brake light
LAMP_MIN_X = 0.3                 # head and tail lamps sit further out than this (metres from the middle)

# ---- material name substring -> class, first match wins (the rest: OTHER)
CLASS_RULES = [
    ('Paint', PAINT),            # the body colour (the player's main colour replaces it)
    ('Plastic_Black', TRIM),     # the second colour (bumpers, flares, mirrors)
    ('Black', DARK), ('Chrome', METAL),
    ('Headlight_Glass', LENS), ('Light_Bulb', EMIT), ('Tail_Red', REDLENS), ('Indicator', AMBER),
    ('Window', GLASS), ('Tyre', TYRE), ('Rim', RIM),
    ('Interior', INTERIOR),      # dropped: the game's glass is dark
]
# linear rgb for classes whose source colour makes no sense without the source shader
ALBEDO_OVERRIDE = {LENS: (0.03, 0.03, 0.035), GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8),
                   PAINT: (0.8, 0.8, 0.8), TRIM: (0.018, 0.018, 0.02)}

# ---- atlas and bake
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.6, 'tyre': 0.6}   # texture space per part (the body is 1)
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45}                  # and per class
AO_DISTANCE, AO_SAMPLES = 0.6, 48
OFFROAD_WHEELS = None            # or {'tyres': [(33, 12.5), (35, 12.5)]}: a beadlock set for tyre-size upgrades

# ---- previews
PREVIEW_PAINT, PREVIEW_TRIM = (180, 30, 30), (26, 26, 28)
PREVIEW_LENGTH = 2.3             # about half the car's length (metres): where the close-ups look
COLLIDER_SLICES = 6
