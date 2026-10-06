# Cardboard from manufactured structure

The native Matter instrument represents two paper liners, a corrugated paper
medium, and finite starch-glue regions as distinct three-dimensional FEM
materials. It uses published inputs and keeps missing measurements explicit.
The purpose is to derive folding from material response and contact, rather than
assigning a weak hinge to a box-shaped elastic object.

## Starting material

The default `--preset literature2009` builds a 31.6 × 20 × 4.210 mm C-flute coupon with
7.9 mm pitch, 0.277 mm liners, and a 0.191 mm medium. The sinusoidal medium has
constant normal thickness. Separate upper and lower glue bridges connect the
paper with conforming shared interfaces; there are no direct liner-to-medium
bonds. The lower glue region has a nominal 0.8 mm horizontal footprint and
0.1 mm minimum gap; the upper uses 0.6 mm and 0.03 mm. End patches are clipped
to the coupon. The manifest records their actual width and area. Its
`medium_centerline_height_m` denotes peak-to-trough rise, not an absolute
vertical coordinate.

This is a **cross-source recipe**, not an exact reconstruction of a measured
board lot. [The manufacturing sources](examples/cardboard/manufacturing-sources.md)
and [machine-readable manifest](examples/cardboard/manufacturing-source-manifest.json)
identify each source value, proxy, derived quantity and numerical assumption.
In particular, the paper calipers/densities and finished-board caliper are
borrowed from separately reported specimens. Glue density is assumed.

The initial condition is preformed, bonded, conditioned and stress-free.
Paper forming through corrugating rolls, residual manufacturing stresses,
starch gelatinization/curing, drying and moisture transport are not simulated.
These omissions cannot be hidden by calling the starting geometry a simulated
manufacturing process.

## Paper and adhesive mechanics

The Haj-Ali 2009 paper assets use direction-dependent elastic constants in the
machine, cross-machine and thickness axes. The medium's material frame follows
the flute tangent. Additive plastic Green strain and an associated Hill yield
surface provide irreversible material memory through Matter's implicit-state
constitutive solver. The constants describe a compression-calibrated,
ideal-plastic approximation: the source's bilinear hardening slopes were not
tabulated, and a zero slope is an explicit modeling choice. The symmetric Hill
surface does not reproduce the source's distinct tension/compression strengths.
It is not a complete paper failure or crease law.

The six `ep` components are plastic strain; algebraic stress variables are
scratch state and must not be counted as permanent deformation. Source-bound
CPU and Metal material-point checks compare stresses, return states, yield,
work and tangents. They verify the implemented law, not agreement with a real
folded box. Matching elastic-control assets hold plastic strain at zero while
preserving the elastic constants and density.

Finite glue uses the reported 400 MPa modulus with an assumed Poisson ratio
0.3, density 1500 kg/m³ and a Neo-Hookean finite-rotation extension. It resolves
bond geometry and elastic compliance. Its bonded interfaces cannot debond;
there is no calibrated adhesive strength, fracture energy or curing law.
Deformable self-contact is enabled for finite-glue geometry; its friction is
currently zero. `--without-self-contact` is a numerical diagnostic control.

The original `--preset nagasawa2013` remains available for the retained elastic
line-tie experiments. Its source-arc profile check preserves an inconsistency:
interpreting the reported 1.7 mm radius as a centerline radius with the stated
pitch and angle gives approximately 5.728 mm total caliper, not 5 mm. The check
does not silently rescale those inputs. The executable geometry is explicitly
a sinusoidal approximation.

## Build and verify

```sh
cmake -S matter -B build-cardboard -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-cardboard --target numi-matter-cardboard-probe \
  numi-matter-cardboard-profile-check numi-matter-cardboard-glue-mesh-check \
  numi-matter-cardboard-paper-material-check numi-matter-cardboard-paper-gpu-check \
  numi-matter-cardboard-material-check numi-matter-cardboard-gpu-check -j4
ctest --test-dir build-cardboard -R '^matter\.(compiler|metal)\.cardboard_' --output-on-failure
build-cardboard/numi-matter-cardboard-probe --help
```

Compile and inspect the structure without advancing physics:

```sh
build-cardboard/numi-matter-cardboard-probe --preset literature2009 \
  --compile-only --nx 8 --ny 1 --thickness-slices 1 --output /tmp/cardboard-geometry
python3 matter/tools/audit_cardboard.py /tmp/cardboard-geometry/mesh.json \
  /tmp/cardboard-geometry/initial.obj /tmp/cardboard-geometry/audit.json
```

Output directories must be new or empty. Mesh JSON stores authored FP64
coordinates while OBJ snapshots contain the native FP32 positions; small
initial differences include quantization. The audit requires NumPy; the
renderer additionally requires Matplotlib. Rendering uses exported geometry
and does not manufacture or amplify deformation.

## Loading protocol and evidence

The literature preset defaults to a coarse diagnostic mesh (8 subdivisions
per pitch, 1 across width and 1 through paper thickness), a 0.25-degree grip
rotation, 25 µs timesteps, and 32 loading + 16 hold + 32 unloading + 64 constrained
settling steps. Its total physical duration is 3.6 ms. These are working
numerical instrument settings, not a mesh-converged folding experiment.

The left grip is fixed. The right grip rotates about its section center with
translation constrained, so this is a combined bend/shear test, not pure-moment
or four-point bending. `--steps`, `--hold-steps`, `--unload-steps` and
`--relax-steps` specify a continuous loading cycle. Unloading rotates the
right grip back to zero, and relaxation keeps **both grips clamped** at that
angle. This is constrained settling, not free release; its residual shape cannot
establish the shape of a freely unloaded crease. The probe runs an unloaded
reference in a separate environment, exports phase-end geometry and material
history, and records accepted-state solver certificates on every step.
A short dynamic cycle does not establish a quasi-static fold.

`--newton-iterations` controls the global nonlinear budget (7 legacy, 14
literature preset). `--fgmres-iterations` controls the global Krylov budget; `--material-iterations`
controls local constitutive Newton iterations in [1,16]. Legacy defaults remain
10 and 8 respectively; the literature recipe selects 32 and 16. Neither option
changes acceptance tolerances. The material budget is bound into the compiled
world fingerprint. A state with both `update` and `implicit` uses `update` as
the local Newton starting guess and `implicit` as the equation to solve; an
initializer never replaces the constitutive residual. Raw status diagnostic slots have status-dependent meanings;
use the separately exported certificate residual rather than a successful
status slot that happens to be zero. The certificate raw accepted flag can
remain one on a rejected global step in this runtime; `step_accepted` therefore
requires successful status as well as finite, in-tolerance residuals.

A lasting crease requires more than a residual displacement: examine the six
plastic-strain components, unloaded-reference drift, residual speed/kinetic
energy, reaction forces, and sensitivity to mesh/time resolution. Elastic
material may retain visible motion during relaxation. Do not label that motion
plasticity. An unsuccessful return map or nonlinear step remains failed
numerical evidence even if a smaller load later passes.

The final default cycle and a half-angle cycle each accepted 144 native steps,
retaining nonzero plastic strain while the unloaded references retained zero.
An otherwise matching elastic control also completed 144 steps. Nine targeted
compiler/Metal checks passed. These are implementation checks, not physical
folding validation.

See [retained evidence](examples/cardboard/evidence/README.md) for exact source
and binary identities, accepted runs, failed trials and unresolved gates.
Published crease curves, adhesive failure, creasing-tool contact and full-box
folding/assembly remain separate validation gates. This instrument does not
claim physical qualification of those behaviors.
