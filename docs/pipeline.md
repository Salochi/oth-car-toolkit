# What the converters do

Both converters (`convert_materials.py`, `convert_palette.py`) run the same way: read the source model, put it in the
game's axes and scale, sort its faces into **parts** and **classes**, then hand one joined object to the shared stages
in `pipeline.py`. Only the first half differs: how a face's class and colour are found.

## 1. Reading the source

- **convert_materials.py**: every material is one flat colour (or a texture, averaged). Objects are told apart by name:
  the profile says which are the tyres and rims (`<Tire>_<Front_Left>`...), the front and rear bumpers (to find the
  front), the lamps, the exhaust, what to drop. Animated objects (doors) are frozen closed. The model is used at its
  own scale (it must be in metres).
- **convert_palette.py**: one palette texture for the whole car. A face's colour is the texel under its UV centre;
  faces with a UV spread (painted panels) are paint. The model is scaled so its wheelbase is the profile's
  `WHEELBASE`; the ground is the bottom of the tyres. Wheels with the tyre and rim in one mesh are split by distance
  from the axle (`RIM_RADIUS`).

## 2. Parts and classes

Parts: `body`, `glass`, `wheel`, `tyre` (and any extra wheel sets). Only the front left wheel and tyre are kept; the
framework mirrors them.

Classes (`pipeline.PAINT`, `TRIM`, `DARK`, `METAL`, `LENS`, `EMIT`, `REDLENS`, `AMBER`, `DETAIL`, `GLASS`, `TYRE`,
`RIM`, `INTERIOR`, `OTHER`) decide the maps:

| map | channel | from |
|---|---|---|
| `_albedo_base` | rgb | the face colour (or the source texture), the profile's overrides for glass and lenses |
| `_zones` | R | **paint**: replaced by the player's main colour |
| | G | **trim**: replaced by the trim colour (black plastic, chrome...) |
| `_mask` | R | the windscreen (rain, wipers) |
| | G | glossy: glass and lenses |
| | B | takes dirt, mud, snow and wear (not lamps, not tyres) |
| `_lights` | R | tail and brake lamps |
| | G | headlights (front `EMIT` / `LENS`) |
| | B | reverse lamps |
| `_ao` | | ambient occlusion (lamp looks blanked: their insides make a noisy mess) |

The game recolours `base × (1 − R − G) + main × R + trim × G`, so the zones replace the albedo where they are.

## 3. The atlas

Every distinct face **look** (its albedo, zones, mask, lights and part) gets its own rectangle in one 2048² atlas,
packed as a squarified treemap with a 10-texel gap between rectangles. Inside a rectangle only the AO varies, so
mipmaps and filtering never mix two looks (no tail-lamp red bleeding onto the paint) and faces smaller than a texel
still show their own look. Rectangle sizes follow 3D area, scaled per part (`UV_PRIORITY`) and per class
(`CLASS_UV_WEIGHT`: black parts show their AO least).

## 4. Bake, LODs, collider, export

- The maps above are baked from per-face colours (Cycles, emission bakes; AO with `AO_SAMPLES` samples).
- The body's distance versions are decimated copies (`LOD_TRIS`), with vertices flung off by the collapse put back
  on the full body.
- The collider: a convex hull of the body, reduced to ≤ 250 triangles, then sliced along the car by `colliders.py`
  (`COLLIDER_SLICES`) so each piece only reaches as low as the body does there.
- Export: Unity axes, split normals (smoothing angle 35°), counter-clockwise winding checked against the normals.

## 5. Anchors

- Wheels: the tyres' centres. Lamps: the centroids of the head / tail lamp faces per side; reverse between the
  reverse lamps (or the tails); a third brake light when there is one.
- Accessories (`cosm_*`): taken from the reference car. Along the car, the overhangs scale between axle and bumper and
  between the axles each slot keeps its distance to the nearer axle; across and up by the bodies' bounds; then each
  slot keeps the reference's offset from the body surface along its natural axis (the roof down, the sides in, the
  bumpers along the car).
- Exhaust: the tail pipe's end (from the pipe object) or the profile's own rule, facing the reference's way unless
  the profile gives a yaw.

## 6. Previews

`preview_*.png`: the car in the game's look (paint and trim in the profile's preview colours, times the AO), from the
front, rear, side, top, close up, with the lights on, and each LOD. Check these before you build the mod.
