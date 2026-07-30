# Tactile product-flow example

`metalrobo_tactile_example` exercises the normal
`EpisodeTwin -> WorldTemplate -> WorldFamily -> MRWorldPack` path for three
authored embodiments:

- `franka-grasp`: two 32x32 finger pads, each with one backing shape;
- `g1-balance`: two 32x32 plantar atlases, each spanning four foot spheres;
- `psm-needle`: two 32x32 inner-jaw atlases, each spanning one capsule and two
  tooth spheres.

All scenarios run actual rigid contact, publish metric normal penetration,
solver wrench and center of pressure, friction utilization, and
`tangentialMotion = [du, dv, vu, vv]`. The displacement values are bounded
rigid-body anchor motion in each cooked tangent frame. They are not a membrane
material model.

The same explicit authored pack can be written for
`metalrobo.compile_world_pack(...)`. MLX carries tactile history in
`WorldState`, publishes named arrays, and feeds every two-sensor embodiment
through one 64-value-per-sensor canonical metric stem plus presence and
confidence. A trained cross-sensor encoder may replace that stem only when its
fingerprinted asset is supplied; this example does not claim trained weights
or hardware transfer.

Run:

```sh
./build/bin/metalrobo_tactile_example franka-grasp
./build/bin/metalrobo_tactile_example g1-balance \
  --write-world-pack /tmp/g1-tactile.mrworld
./build/bin/metalrobo_tactile_example psm-needle \
  --debug-dir /tmp/metalrobo-tactile-debug \
  --write-world-pack /tmp/psm-tactile.mrworld
```

The optional debug directory contains a 16-bit depth PGM, validity PGM,
metric CSV, 3D query OBJ, and summary JSON per sensor. These are inspection
artifacts; captured RL execution consumes native Metal/MLX buffers without
CPU readback.
