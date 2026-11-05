# The `.mesh` format ("VERA" version 1)

One file per car: every part of the model and the car's anchors. Little-endian throughout. Read and written by
`tools/vera_mesh.py`; read in the game by the OTH+ car framework (`MeshBlob`).

## Layout

```
"VERA"                       4 bytes
version          u16         1
part count       u16
part × count:
    name         u8 length + UTF-8 bytes
    kind         u8          0 body, 1 glass, 2 wheel, 3 tyre, 4 collider
    origin       3 × f32     where the part sits on the car (a wheel or tyre: its hub)
    rotation     4 × f32     quaternion x, y, z, w (the converters write identity)
    scale        3 × f32     (the converters write 1, 1, 1)
    vertex count u32 (n)
    positions    n × 3 f32   relative to the origin
    normals      n × 3 f32
    uvs          n × 2 f32   into the car's texture atlas, v up
    index count  u32 (m)     a multiple of 3
    indices      m × u32     triangles, counter-clockwise seen from outside (Unity winding)
anchor count     u16
anchor × count:
    name         u8 length + UTF-8 bytes
    position     3 × f32     car space
    rotation     4 × f32     quaternion x, y, z, w
```

## Axes

Unity's: x to the right, y up, z forward, in metres; y = 0 is the ground under the tyres. Seen from above the car
faces +z.

## Parts

| name | kind | what |
|---|---|---|
| `body` | 0 | the whole body, every look in one mesh |
| `body_lod1` .. `body_lod3` | 0 | the body for the distance levels (fewer triangles) |
| `glass` | 1 | the windows |
| `wheel`, `tyre` | 2, 3 | the front left rim and tyre, centred on their hub (`origin`); the framework mirrors them to the other corners |
| `collider` | 4 | the body's physics shape: a convex hull of at most 255 triangles |
| `collider_1` .. | 4 | more convex pieces of the body (colliders.py slices the body along its length) |

Other wheel and tyre parts may follow (e.g. `wheel_ot`, `tyre_ot33`: a car mod can swap them in for tyre-size
upgrades).

## Anchors

| name | |
|---|---|
| `wheel_FL`, `wheel_FR`, `wheel_RL`, `wheel_RR` | hub centres; the left ones face +z, the right ones -z |
| `headlight_L/R`, `tail_L/R`, `reverse`, `brake` | where the lamps' light sources go |
| `exhaust_L` / `exhaust_R`, `engine_smoke`, `fog_L/R` | effects; the exhaust's rotation is the way the smoke leaves the pipe |
| `cosm_*` | where the game's accessory slots go (roof, hood, bumpers, sides, mudflaps...) |
| `cosmscale_*` | a slot's scale, copied from the reference car |
