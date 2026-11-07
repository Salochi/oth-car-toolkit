# oth-car-toolkit

Tools for turning a 3D car model into a drivable car for **Over The Hill**, through the car framework of
[OverTheHillPlus](https://github.com/legomaster188/OverTheHillPlus-QoL) (a BepInEx 6 IL2CPP mod). The framework builds
a new networked car on a stock "donor" car from a `.mesh` model file and a few textures; this toolkit makes those files
from a GLB in Blender, and draws the car's garage icon.

```
your model.glb ──► convert (Blender) ──► <name>.mesh + <name>_albedo_base/_zones/_mask/_lights/_ao.png + previews
                                     └─► colliders.py (sliced body collider)      make_icon.py ──► <name>_icon.png
```

No game files and no car models are included: you bring the model (check its licence first, see below) and the
reference car the accessory anchors are copied from.

## What's in it

| | |
|---|---|
| `tools/refine.py` | Optional first step for a **low-poly** model: Phong tessellation (each triangle split k x k, the new points lifted onto the curve the normals describe), a fairing pass against the ripples of a decimated mesh, the model's own normals kept for the shading; also scales and turns it. Writes a GLB, and a .blend beside it for `convert_materials.py` (a glTF round trip can scramble custom normals). |
| `tools/convert_materials.py` | Converter for models coloured by their **materials** (one flat colour per material: detailed Sketchfab-style cars). |
| `tools/convert_palette.py` | Converter for low-poly models coloured by one **palette texture** (POLYGON-style packs): faces classed by the texture colour under them, the albedo baked from the texture, the paint zone on the painted texels only (wear stays on top of any paint). |
| `tools/pipeline.py` | The stages both share: the texture atlas (a rectangle per face "look"), the bakes, the body's distance versions (LODs), the collider, the export to the game's axes, the accessory anchors, the `.mesh`, preview renders. |
| `tools/colliders.py` | Slices the body collider into convex pieces along the car, so it follows a belly that is higher than the bumpers (one convex hull bottoms out on rocks the body never touches). |
| `tools/make_icon.py` | The garage icon in the game's style: a white side silhouette with window cut-outs, belt and sill lines, door outlines with handles, the trim pieces' outlines, tall-sidewall wheels. |
| `tools/offroad_wheel.py` | A procedural beadlock rim with tyres of given sizes (for garage tyre-size upgrades). |
| `tools/vera_mesh.py` | Reads and writes the `.mesh` format ([docs/mesh-format.md](docs/mesh-format.md)). |
| `profiles/` | What a converter needs to know about one source model (which material is paint, which object is a wheel, what to drop...). `tacoma.py`, `tacoma3.py` (refined first) and `bajabug.py` are working examples; `example_materials.py` is a commented starting point. |

## Requirements

- **Blender 3.6** (the converters run inside it, headless; other versions are untested).
- **Python 3** with `numpy` and `Pillow` for `make_icon.py` (the others run in Blender's own Python).
- A **reference car**: a built `.mesh` of a similar car whose accessory anchors get mapped onto yours. The OTH+
  repository has one per car it adds, under `src/Mods/<Car>/Assets/` (e.g. `hilo22.mesh` for a pickup, `aro.mesh` for a
  small 4x4).

## Quick start

```bash
# 1. convert (pick the converter that matches how the model is coloured, and write a profile for it)
blender -b --factory-startup --python tools/convert_materials.py -- profiles/my_car.py my_car.glb build/my_car ref/hilo22.mesh my_car

# 2. look at build/my_car/preview_*.png (front, rear, side, top, close-ups, lit, the LODs)

# 3. the garage icon (door seams by the side windows; --tyre when the car keeps the donor's wheels)
python tools/make_icon.py build/my_car/my_car.mesh build/my_car/my_car_icon.png --tex build/my_car/my_car --seams 0F,0-1,1R
```

Then embed the `.mesh` and the PNGs in a car mod for the framework (resources `<name>/<file>`), with a car definition
that names the donor. [docs/new-car.md](docs/new-car.md) walks through all of it, [docs/pipeline.md](docs/pipeline.md)
explains what the converters do and what each map means.

## Model licences

A car mod embeds a converted copy of the model, and anything embedded in a mod can be pulled back out. So only publish
a car whose model licence allows that kind of redistribution. Some things worth knowing:

- **CC BY / CC BY-SA / CC0** models (Sketchfab's free downloads often are): fine with credit (and share-alike where it
  says so). Check that the uploader really made it: re-uploads of paid asset packs under a "free" licence happen.
- **Sketchfab's Standard licence** (paid and "Free Standard" models) forbids making the model available "in a way that
  allows third parties to use, download, extract or access" it: keep such cars to yourself.
- **Asset-store packs** (Synty, Unity Asset Store...) usually allow use in a game you make, not in a mod that ships the
  meshes to others.

This is not legal advice: read the licence of your model.

## Credits

The car framework, the `.mesh` format and the reference cars are from
[OverTheHillPlus](https://github.com/legomaster188/OverTheHillPlus-QoL) (MIT). Over The Hill is made by Funselektor
Labs; this is an unofficial fan project.

Licence: MIT (see [LICENSE](LICENSE)).
