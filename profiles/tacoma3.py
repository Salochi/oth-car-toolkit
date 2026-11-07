"""
Profile for refine.py and convert_materials.py: "Toyota Tacoma TRD" by LagzDesign (Sketchfab, CC BY 4.0:
https://sketchfab.com/3d-models/toyota-tacoma-trd-f13b2c73a13147a69a6d2a83f72b2505), a low-poly 3rd-gen (2016-2023)
Tacoma double cab coloured by materials. Refined first (Phong tessellation: rounder arches, tyres, mirrors, about 9x the
triangles), then converted like the 4th gen: built on the Burro, anchors from the OTH+ Hilo 22. Bring your own copy of
the model (input/tacoma3/).

    blender -b --factory-startup --python tools/refine.py -- profiles/tacoma3.py input/tacoma3/toyota_tacoma_trd.glb build/tacoma3/refined.glb
    blender -b --factory-startup --python tools/convert_materials.py -- profiles/tacoma3.py build/tacoma3/refined.blend build/tacoma3 <hilo22.mesh> tacoma3
"""
from pipeline import PAINT, TRIM, DARK, METAL, LENS, EMIT, REDLENS, AMBER, DETAIL, GLASS, TYRE, RIM, INTERIOR, OTHER

NAME = 'tacoma3'
TEX = 2048

# ---- refine.py: groups (output name -> source object prefixes), splits per triangle edge, scale and turn
_CORNERS = (('fl', 'FL'), ('fr', 'FR'), ('rl', 'RL'), ('rr', 'RR'))
REFINE_GROUPS = {
    'body': ['body_'],
    **{f'Wheel_{c}': [f'{k}_rim_'] for k, c in _CORNERS},
    **{f'Tire_{c}': [f'{k}_tire_'] for k, c in _CORNERS},
}
REFINE_SPLITS = {'body': 3, **{f'Wheel_{c}': 2 for _, c in _CORNERS}, **{f'Tire_{c}': 3 for _, c in _CORNERS}}
REFINE_ALPHA = 0.75
# the shape follows normals recomputed from the welded source (smooth below 30 degrees); the shading keeps the model's
# own normals; 10 fairing passes take out the ripples its fans of long thin triangles leave
REFINE_SMOOTH_ANGLE = 30.0
REFINE_NORMALS = 'source'
REFINE_FAIR = {'body': 10, **{f'Tire_{c}': 10 for _, c in _CORNERS}}
# the model's wheelbase is 1.994 units; the real double cab's is 3.236 m (127.4 in)
REFINE_SCALE = 3.236 / 1.994
REFINE_ROTATE_Z = -90.0        # the source faces Blender +x; the converter wants the front along y
# (output metres, Blender axes, front -y) the model's own front and rear axle beams and rear leaf springs: the
# Burro's rigged axles show instead, and these were the lowest points of the belly
REFINE_DROP = {'body': [(-0.70, 0.70, -1.80, -1.58, 0.30, 0.50), (-0.70, 0.70, 1.43, 1.65, 0.30, 0.50),
                        (0.50, 0.95, 1.00, 2.10, 0.25, 0.70), (-0.95, -0.50, 1.00, 2.10, 0.25, 0.70),
                        # the source's flat black arch panels in the wheel wells (the liners take their place)
                        (0.50, 0.66, -2.40, -1.15, 0.30, 1.00), (-0.66, -0.50, -2.40, -1.15, 0.30, 1.00),
                        (0.55, 0.63, 1.00, 2.05, 0.45, 1.00), (-0.63, -0.55, 1.00, 2.05, 0.45, 1.00),
                        # and the rest of the source's square wheel tubs: their front, top and rear walls and the hub
                        # discs (just inside the tyre: a lift or the bigger tyres showed a box round the wheel)
                        *[(lo * sx if sx > 0 else -hi, hi if sx > 0 else -lo, y0, y1, z0, z1, ('plastic', 'dark'))
                          for sx in (1, -1) for lo, hi, y0, y1, z0, z1 in (
                              (0.47, 0.95, -2.20, -1.15, 0.30, 0.95),
                              (0.50, 0.95, 0.97, 2.10, 0.50, 0.98))]]}
# where a crease in the paint's shading stops mid-panel (the bonnet's lines from the scoop) it fades out instead
REFINE_TAPER_CREASES = {'body': ('body',)}
# axle tunnels: the underbody is one solid slab at 0.3-0.5 m, and the Burro's rigged axles (top about 0.45 m at rest)
# went through it as the suspension worked. Over each axle the slab goes up to 0.76 m (the bed floor is at 0.8 m,
# the engine bay is empty): about 0.3 m of travel before the axle reaches anything
# (the rear differential moulded into the floor reaches a little ahead of the tunnel: that lip goes to 0.45 m)
REFINE_LIFT = {'body': [((-0.50, 0.50, 1.19, 1.89, 0.0, 1.0), 0.76), ((-0.50, 0.50, -2.04, -1.34, 0.0, 1.0), 0.76),
                        ((-0.72, 0.72, 1.00, 1.95, 0.0, 1.0), 0.45)]}
# a rounded edge on the hard edges of the panels, trim and lamps where a bevel holds up on this decimated mesh (over
# the big flat panels it crumples their shading): the rear end, the bed's sides and rails, the back of the cab.
# 'safe': only edges long enough for the width, among long enough neighbours
REFINE_BEVEL = {'body': [dict(materials=('body', 'plastic', 'dark', 'rear_lights', 'brake_lights', 'turnsignal_lights',
                                        'reverse_lights'), angle=35.0, width=0.012, segments=3, clamp=False, safe=True,
                             boxes=[(-1.2, 1.2, 2.05, 3.0, 0.0, 1.45),      # tailgate, tail lamps, bumper, fender ends
                                    (-1.2, 1.2, 0.85, 3.0, 1.0, 1.45),      # bed rails and the bed sides' tops
                                    (-1.0, 1.0, 0.35, 1.0, 1.15, 2.1),      # the back of the cab and its roof edge
                                    # the front end, round the grille and the headlamps (where it folds into their
                                    # slats and bezels): bumper and valance, bonnet and fender tops, the outer corners
                                    (-1.2, 1.2, -3.0, -1.55, 0.0, 0.80), (-1.2, 1.2, -3.0, -1.55, 1.12, 1.45),
                                    (0.82, 1.2, -3.0, -1.55, 0.0, 1.45), (-1.2, -0.82, -3.0, -1.55, 0.0, 1.45),
                                    (-1.0, 1.0, -1.2, 0.4, 1.6, 2.1)],      # the roof's edges
                             # and none round the small parts (it crumples or tears them): the headlamps, the fog lamps,
                             # the bonnet scoop, the roof's radio fin, the join of cab and bed
                             exclude=[(0.38, 0.95, -2.85, -2.15, 0.92, 1.25), (-0.95, -0.38, -2.85, -2.15, 0.92, 1.25),
                                      (0.40, 0.92, -2.85, -2.25, 0.45, 0.90), (-0.92, -0.40, -2.85, -2.25, 0.45, 0.90),
                                      (-0.45, 0.45, -2.10, -1.30, 1.15, 1.45),
                                      (-0.15, 0.15, 0.10, 0.90, 1.70, 2.10),
                                      (0.70, 1.20, 0.30, 1.15, 0.75, 1.50), (-1.20, -0.70, 0.30, 1.15, 0.75, 1.50),
                                      # the black flares: rounding a thin trim that lies on the paint shows the paint
                                      (0.70, 1.20, -2.40, -0.95, 0.0, 1.15), (-1.20, -0.70, -2.40, -0.95, 0.0, 1.15),
                                      (0.70, 1.20, 0.85, 2.25, 0.0, 1.15), (-1.20, -0.70, 0.85, 2.25, 0.0, 1.15),
                                      # the roof's back edge (the next pass, wider) and the cab's back corners
                                      # (rounded, they pinched the black roof trim turning down them)
                                      (-0.60, 0.60, 0.82, 1.02, 1.70, 1.86),
                                      (0.55, 0.95, 0.70, 1.05, 1.15, 1.90), (-0.95, -0.55, 0.70, 1.05, 1.15, 1.90),
                                      # the tailgate's face (the rounding flattened its pressed shape into facets)
                                      (-0.80, 0.80, 2.45, 2.80, 0.70, 1.25),
                                      # the bed's back corners over the tail lamps (rounded, they bulged out and back)
                                      (0.68, 0.98, 2.35, 2.80, 1.12, 1.40), (-0.98, -0.68, 2.35, 2.80, 1.12, 1.40)]),
                    # the roof's back edge: a wider round over the top of the cab
                    dict(materials=('body',), angle=20.0, width=0.022, segments=4, clamp=False, safe=False,
                         boxes=[(-0.50, 0.50, 0.82, 1.02, 1.70, 1.86)]),
                    # the bed's back corners round the tail lamps: a light round off their razor edges
                    dict(materials=('body',), angle=30.0, width=0.007, segments=3, clamp=True, safe=False,
                         boxes=[(0.68, 0.98, 2.35, 2.80, 1.12, 1.40), (-0.98, -0.68, 2.35, 2.80, 1.12, 1.40)])]}
# the tail lamps' black housings are the trim material in the source: with the trim set to the body colour they
# came out painted. Inside the lamps they are the fixed black instead
REFINE_RECOLOR = {'body': [((-0.98, -0.74, 2.42, 2.78, 0.82, 1.28), 'plastic', 'dark'),
                           # the cab floor's underside (painted in the source: a body-colour patch on the belly)
                           ((-0.79, 0.79, -1.25, 1.00, 0.30, 0.60), 'body', 'dark', -0.5),
                           ((0.74, 0.98, 2.42, 2.78, 0.82, 1.28), 'plastic', 'dark'),
                           # the fenders' paint lip curling under the black flares: it showed between flare and liner
                           ((0.75, 1.10, -2.12, -1.08, 0.30, 1.12), 'body', 'dark', -0.2),
                           ((-1.10, -0.75, -2.12, -1.08, 0.30, 1.12), 'body', 'dark', -0.2),
                           ((0.75, 1.10, 0.95, 2.15, 0.30, 1.12), 'body', 'dark', -0.2),
                           ((-1.10, -0.75, 0.95, 2.15, 0.30, 1.12), 'body', 'dark', -0.2),
                           # the bumper's paint underside, poking through the lower valance's slots
                           ((-0.85, 0.85, -2.95, -2.10, 0.25, 0.64), 'body', 'plastic', -0.5),
                           ]}
# the bumper band's underside bulging down through the lower valance's slots (big source faces: by final triangle)
REFINE_RECOLOR_FINAL = {'body': [((-0.45, 0.45, -2.60, -2.40, 0.40, 0.665), 'body', 'plastic'),
                                 # the rear bumper's black middle: every paint strip left in it (along the recess's
                                 # top, under it, beside the step pads) black
                                 ((-0.36, 0.36, 2.55, 2.72, 0.55, 0.715), 'body', 'plastic'),
                                 # the rear bumper's middle: the paint lips round its black step pads
                                 ((-0.72, 0.72, 2.45, 2.85, 0.60, 0.70), 'body', 'plastic', 1, (0.3, 1.0)),
                                 # and the paint strip along the bottom of its black middle
                                 ((-0.64, 0.64, 2.40, 2.85, 0.45, 0.575), 'body', 'plastic', None, (-1.0, -0.25)),
                                 # and the thin paint rim round its middle recess (small triangles only)
                                 ((-0.47, 0.47, 2.40, 2.85, 0.53, 0.72), 'body', 'plastic', 3, (-1.0, 1.0), 6e-4),
                                 # the bed's wheel humps: their rounded inner edge is paint in the source
                                 ((0.54, 0.70, 0.95, 2.30, 1.00, 1.13), 'body', 'plastic'),
                                 ((-0.70, -0.54, 0.95, 2.30, 1.00, 1.13), 'body', 'plastic')]}
# the black trim line along the bonnet's sides carried on round its nose, between the headlamps' inner corners
REFINE_LINES = dict(on='body', material='dark', width=0.008, lift=0.0015,
                    # the seam: across the nose on the model's own crease, then straight down to each headlamp's
                    # inner corner
                    lines=[[(-0.506, -2.372, 1.111), (-0.48, -2.372, 1.143), (-0.462, -2.373, 1.167), (-0.424, -2.373, 1.199),
                            (-0.39, -2.374, 1.211), (0.0, -2.374, 1.211), (0.39, -2.374, 1.211), (0.424, -2.373, 1.199),
                            (0.462, -2.373, 1.167), (0.48, -2.372, 1.143), (0.506, -2.372, 1.111)]])
# the bed's back corners over the tail lamps kept flat as modelled (curved, they bulged out and back in)
# the tailgate's lower side corners (crumpled) and the tops of the C-pillars, where the black roof trim turns down
REFINE_FLAT = {'body': [(0.68, 0.98, 2.35, 2.80, 1.12, 1.40), (-0.98, -0.68, 2.35, 2.80, 1.12, 1.40),
                        (0.50, 0.80, 2.55, 2.80, 0.70, 0.98), (-0.80, -0.50, 2.55, 2.80, 0.70, 0.98),
                        (0.55, 0.95, 0.70, 1.05, 1.45, 1.90), (-0.95, -0.55, 0.70, 1.05, 1.45, 1.90)]}
# the bed's back corners: the source steps them in over the tail lamps (a notch, dark underneath); the corner's edge
# carried straight on down to the lamp's top instead
REFINE_MOVE = {'body': [(p, q) for sx in (1, -1) for p, q in (
    ((sx * 0.830, 2.590, 1.204), (sx * 0.820, 2.621, 1.204)),
    ((sx * 0.823, 2.594, 1.209), (sx * 0.820, 2.621, 1.204)))]}
# the bonnet's shading smooth over its bulge (its folds drew lines and specks either side of the scoop)
# and the bed sides' character line, from over the rear arch back to the tail lamps (it shaded jagged)
REFINE_SOFTEN = {'body': [((-0.74, 0.74, -2.40, -1.22, 1.15, 1.50), ('body',)),
                          ((0.78, 1.00, 1.55, 2.60, 0.62, 1.20), ('body',)),
                          ((-1.00, -0.78, 1.55, 2.60, 0.62, 1.20), ('body',))]}
# faces with another surface under a millimetre in front of them out (folded slivers, paint lips under black)
REFINE_BURIED = {'body': 0.001}
# the inner panels and the underbody drawn from both sides (from behind, the game draws nothing: see-through)
REFINE_DOUBLE = {'body': ('plastic', 'dark')}
# a black sheet 3 cm behind every window: the cab is empty, so the windscreen showed grey through
REFINE_BLACKOUT = dict(glass='glass', material='dark', depth=0.03)
# wheel-well liners and a frame (the model has none: through the arches it was empty and see-through); the liners
# follow the arches 3 cm in, and behind they stay inside the bed's wheel humps (inner face at 0.55 m, top 1.03 m)
REFINE_CHASSIS = dict(paint='body', material='dark', liner_inner=0.52, liner_inner_rear=0.58, liner_top_rear=1.0,
                      rail_x=0.42, rail_low=0.53, rail_high=0.61, rail_front=-2.30, rail_rear=2.45,
                      crossmembers=(-1.0, -0.2, 0.6, 2.3))
# the source's wheels are flat discs on plain tubes: a stock-style 18 inch six-split-spoke wheel on a 32 x 11.5 inch
# mud-terrain tyre instead (offroad_wheel.py, like the garage's 33 / 35 inch set)
REFINE_WHEELS = dict(rim='split6', tyre=(32, 11.5))
# parked wipers: (pivot Unity x, blade length m), blades toward the passenger side
REFINE_WIPERS = dict(glass='glass', material='dark', blades=[(-0.56, 0.55), (0.02, 0.50)], up=-0.06,
                     pivot_down=0.085)

# ---- the body's distance versions (LOD1..3, triangles); LOD0 is the refined body as is
LOD_TRIS = [45000, 20000, 8000]

# ---- which objects are what (by name prefix; refine.py names them <group>_<material>)
FRONT_OBJECT, REAR_OBJECT = 'body_front_lights', 'body_rear_lights'
TYRE_PREFIX, WHEEL_PREFIX = 'Tire', 'Wheel'
CORNERS = ('FL', 'FR', 'RL', 'RR')
LAMP_OBJECTS = ('body_headlights', 'body_blackout')   # the headlamp housings, the sheet behind the glass: no mud
NO_LENS_OBJECTS = ()
MATERIAL_SWAPS = []
# the headlamps' clear outer cover: can't be drawn see-through, and opaque it would hide the lamps under it
DROP_OBJECTS = ('body_headlights_glass',)
EXHAUST_OBJECT = None          # no tail pipe in the model: the Hilo 22's exhaust anchor
HIGH_BRAKE_Y = 1.45
LAMP_MIN_X = 0.3

# ---- material names -> face classes: (substring of material name, class), first match wins
CLASS_RULES = [
    ('OT_Machined', METAL), ('OT_Gunmetal', RIM), ('OT_Black', DARK), ('OT_Tyre', TYRE),   # offroad_wheel.py
    ('turnsignal_lights', AMBER), ('reverse_lights', EMIT), ('brake_lights', REDLENS), ('rear_lights', REDLENS),
    ('front_lights', EMIT), ('headlights', EMIT),
    ('body', PAINT), ('plastic', TRIM), ('dark', DARK), ('glass', GLASS), ('mirrors', METAL),
    ('rim_color', METAL), ('rim_main', RIM), ('Tires', TYRE),
]
# albedo overrides (linear rgb) for classes whose source colour makes no sense without the source shader
ALBEDO_OVERRIDE = {GLASS: (0.02, 0.02, 0.025), EMIT: (0.8, 0.8, 0.8), PAINT: (0.8, 0.8, 0.8), TRIM: (0.018, 0.018, 0.02),
                   RIM: (0.03, 0.03, 0.032), METAL: (0.55, 0.55, 0.56)}

# ---- the atlas: linear texture-space share per part (the body is 1) and per class (black parts show their AO least)
UV_PRIORITY = {'glass': 0.45, 'wheel': 0.6, 'tyre': 0.6, 'wheel_ot': 0.6, 'tyre_ot33': 0.55, 'tyre_ot35': 0.55}
CLASS_UV_WEIGHT = {DARK: 0.3, TRIM: 0.45}
# texels between the UV islands, tried in turn (the first that packs): the refined body has thousands of small faces,
# and half a texel let the filtering mix a panel's edge with the next island's (a fold in a gap bakes black: black
# dashes along the paint)
UV_MARGINS = (2.0, 1.5, 1.0, 0.5, 0.0)
AO_DISTANCE, AO_SAMPLES = 0.6, 48
# the paint's AO no darker than this (the refined body's folds and tiny faces baked black specks into it; its gaps
# and seams are black trim of their own)
AO_FLOOR = {PAINT: 0.75, TRIM: 0.35}

# ---- the garage's bigger tyres (offroad_wheel.py): a beadlock rim with 33 and 35 inch tyres, (inches, width inches)
OFFROAD_WHEELS = {'tyres': [(33, 12.5), (35, 12.5)]}

# ---- previews: the default paint and trim, and half the car's length (for the close-ups)
PREVIEW_PAINT, PREVIEW_TRIM = (52, 66, 48), (26, 26, 28)
PREVIEW_LENGTH = 2.6
COLLIDER_SLICES = 6
# between the axles the collider stops this high (metres): the side steps hang lower than the floor
COLLIDER_FLOOR = 0.47
# the wheels' ambient occlusion baked away from the body (no fender shadow baked onto one side of the tyre)
AO_WHEELS_APART = True
