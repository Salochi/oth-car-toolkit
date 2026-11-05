# Adding a car

## 1. Pick the model and the donor

- **The model**: a GLB with separate wheels (or at least wheels that can be told apart), real-world scale for a
  material-coloured model, and a licence that allows what you mean to do with the car (see the README).
- **The donor**: the stock car the new one is built on. Its physics skeleton, garage data and accessory slots become
  the new car's, so pick one close in **wheelbase** and size. The framework moves the wheels to your model's track and
  wheelbase, but a close donor drives most like the real thing.
- **The reference**: a built `.mesh` of a similar car (body shape and size), for the accessory anchors.

## 2. Look at the source

Import it in Blender and note: object names (tyres, rims, bumpers, lamps, exhaust), material names, which way the front
faces, anything to drop (interiors, a leftover cube, add-ons you don't want). For a palette model, sample the palette
colours of the lamps, trim and paint.

## 3. Write a profile

Copy `profiles/example_materials.py` (or `bajabug.py` for a palette model) and fill it in. The class rules decide
almost everything, so take care with them: paint, trim (the second colour), glass, lamp lenses and emitters, tyres.

## 4. Convert and check

```bash
blender -b --factory-startup --python tools/convert_materials.py -- profiles/my_car.py my_car.glb build/my_car ref/<similar>.mesh my_car
```

Read the log (`[build]` lines: which object was dropped, the wheel corners, the anchors and how each was fitted, missing
anchors) and the previews. Typical fixes: a material in the wrong class, the front the wrong way round, lamps not
found (check `LAMP_MIN_X`), a look that does not fit its rectangle (more texture space: `UV_PRIORITY`).

## 5. The icon

```bash
python tools/make_icon.py build/my_car/my_car.mesh build/my_car/my_car_icon.png --tex build/my_car/my_car --seams 0F,0-1,1R
```

`--seams` places the doors by the side windows, front to back: `0F` in front of window 0, `0-1` between windows 0 and 1,
`1R` behind window 1 (a four-door cab: `0F,0-1,1R`; a two-door with a rear quarter window: `0F,0-1`). `--tyre` is the
radius the car drives on when it keeps the donor's wheels.

## 6. The car mod

A small BepInEx plugin that hands a car definition and the files to the framework: the files embedded as resources
`<name>/<file>` (`<name>.mesh`, the five maps, `_icon.png`), the definition with the car's id, item id, guid, name,
donor and garage base, tuning, paint and trim presets, engine sounds. The OTH+ cars (e.g. `src/Mods/Aro`) are complete
examples. Things to tune after the first drive:

- **Tuning** (`CarTuningProfile`, multipliers on the donor): mass, torque, power, gearing, suspension. Keep the torque
  at the wheels near the donor's (torque × final drive ÷ the tyre radius ratio): far below it the car may not pull
  away in the high range.
- **Wheels**: the model's own, or the donor's (an option of some framework versions: then take the model's own
  hub and suspension pieces out of the body, and keep the donor's rigged axles).
- **Hidden items**: donor accessories that don't fit the body (a spare wheel on a car without a rear door).
