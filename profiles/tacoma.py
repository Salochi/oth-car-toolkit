"""
Profile for convert_materials.py: "Toyota Tacoma TRD Pro 2024" (Sketchfab), a detailed pickup coloured by materials.
Built on the Burro, anchors from the OTH+ Hilo 22. The model itself is not part of this toolkit: bring your own copy.

    blender -b --factory-startup --python tools/convert_materials.py -- profiles/tacoma.py <model.glb> build <hilo22.mesh> tacoma
"""
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'tacoma'
TEX = 2048
# the body's distance versions (LOD1..3, triangles); LOD0 keeps the full body (decimating it welded and collapsed
# overlapping pieces, e.g. the doors); the LOD0 environment variable caps it
LOD_TRIS = [60000, 25000, 10000]

# ---- which objects are what (by name prefix, anywhere up the object's parents)
FRONT_OBJECT, REAR_OBJECT = 'bumper_front', 'bumper_rear'     # tell the front from the back
TYRE_PREFIX, WHEEL_PREFIX = 'Tire', 'Wheel'                   # <prefix>_<corner>: the tyres and rims (brakes go with the rims)
CORNERS = ('Front_Left', 'Front_Right', 'Rear_Left', 'Rear_Right')
LAMP_OBJECTS = ('headlight', 'gablight')     # lamp assemblies: no mud on their insides (chrome, black housings)
# tail lamps: the clear outer cover can't be drawn see-through and z-fights the red lens under it, so it goes
NO_LENS_OBJECTS = ('gablight',)
# (object name prefix, material substring, replacement material substring): the tailgate camera sits in the badge's
# second O, so it takes the badge's chrome
MATERIAL_SWAPS = [('trunk_41', 'Clear polycarbonate', 'Metal_Chrome')]
# the model's own frame, arms and shocks: static in the body they part from the wheels as the suspension works, so they
# go and the car shows the donor's rigged axles instead (the framework's KeepDonorAxles)
DROP_OBJECTS = ('Undercarriage_Procedural',)
EXHAUST_OBJECT = 'exhaust'   # the tail pipe: its rear end is where the exhaust smoke comes out
# the way the smoke leaves the pipe, as a yaw in degrees (Unity: 0 forward, 180 back, -90 left): the TRD Pro's pipe
# turns out to the driver's side under the bed, so not the reference's straight back. None: the reference's.
EXHAUST_YAW = -90.0
HIGH_BRAKE_Y = 1.45          # rear emitters above this (metres) are the cab's third brake light
LAMP_MIN_X = 0.3             # head and tail lamps sit further out than this (the light anchors' centroids)

# ---- material names -> face classes: (substring of material name, class), first match wins
CLASS_RULES = [
    ('OT_Machined', METAL), ('OT_Gunmetal', RIM), ('OT_Black', DARK), ('OT_Tyre', TYRE),   # offroad_wheel.py
    ('Deep Ocean Blue', PAINT), ('Plastic_Black', TRIM), ('Matte_DarkBlack', DARK), ('Coat_Black', DARK),
    ('Metal_Chrome', METAL), ('Metal_Steel', METAL), ('Galv', METAL), ('Clear polycarbonate', LENS),
    ('Light_White', EMIT), ('LightBulb', EMIT), ('Medium Bulb', EMIT), ('Plastic_Red', REDLENS),
    ('Plastic_Orange', AMBER), ('Red Detail', DETAIL), ('Glass_Window', GLASS), ('Tire_Truck', TYRE),
    ('Satin graphite', RIM), ('Fabric_Light', INTERIOR), ('Dash_Gray', INTERIOR),
]
# albedo overrides (linear rgb) for classes whose source colour makes no sense without the source shader
ALBEDO_OVERRIDE = {LENS: (0.03, 0.03, 0.035), GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8),
                   PAINT: (0.8, 0.8, 0.8), TRIM: (0.018, 0.018, 0.02)}

# ---- the atlas: linear texture-space share per part (the body is 1) and per class (black parts show their AO least)
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.6, 'tyre': 0.6, 'wheel_ot': 0.6, 'tyre_ot33': 0.55, 'tyre_ot35': 0.55}
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45}
AO_DISTANCE, AO_SAMPLES = 0.6, 48

# ---- the garage's bigger tyres (offroad_wheel.py): a beadlock rim with 33 and 35 inch tyres, (inches, width inches)
OFFROAD_WHEELS = {'tyres': [(33, 12.5), (35, 12.5)]}

# ---- previews: the default paint and trim, and half the car's length (for the close-ups)
PREVIEW_PAINT, PREVIEW_TRIM = (34, 74, 140), (26, 26, 28)
PREVIEW_LENGTH = 2.6
COLLIDER_SLICES = 6
