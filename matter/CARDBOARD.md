# Cardboard from manufactured structure

This development slice represents paper liners and a corrugated paper medium
as three-dimensional FEM solids in the existing Numi Matter Metal runtime.
The purpose is to establish an inspectable physical instrument before claiming
creasing or box-assembly fidelity. It does not collect new physical measurements.

## Scientific question

Can a structure assembled from published paper properties, layer thicknesses,
and flute geometry transmit load and bend through the native continuum solve?
Does adding a bonded corrugated medium change the response relative to the same
two liners without the medium? A subsequent question is whether irreversible
creasing follows from paper plasticity and interface damage under a real tool.

The first question concerns numerical mechanics. Answering it does not answer
the second or establish agreement with an actual box. Geometry, material tests,
native execution, comparison with published experiments, and complete assembly
are separate evidence gates.

## Sources

- Nagasawa, Komiyama and Mitsomwang (2013), *Finite Element Analysis of
  Corrugated Board on Rotary Creasing Process*, DOI
  [10.1299/jamdsm.7.103](https://doi.org/10.1299/jamdsm.7.103).
  Table 3 (printed p.105) supplies directional elastic constants; Fig.8 and
  section 3 (pp.107–108) supply A-flute geometry. Table 4 supplies an equivalent
  ring-crush medium modulus and yield stress. This is an effective structural
  characterization, not an independently measured constitutive plastic law.
- Aduke, Venter and Coetzee (2024), *Numerical Modelling of Corrugated
  Paperboard Boxes*, DOI
  [10.3390/mca29040070](https://doi.org/10.3390/mca29040070).
  The structural model uses explicit sinusoidal fluting and shared-node ties.
  Its comparison with experiment identifies bond compliance and geometry as
  remaining sources of disagreement. Fitted, weakened crease regions in that
  paper do not reproduce the process of manufacturing a crease.

The source manifest in `examples/cardboard` distinguishes reported quantities,
derived quantities, and model assumptions. Original PDFs are research inputs;
they are not redistributed here.

## Constitutive boundary

The initial liner and medium laws use objective orthotropic Saint Venant–Kirchhoff energy
in material coordinates (machine, cross-machine, thickness). Its infinitesimal
moduli reproduce the paper's elastic constants, including reciprocal Poisson
coupling. It is a finite-rotation extension chosen for the instrument, not a
validated large-strain paper law. Both use the reported sheet constants from
Table 3. The medium's RCT elastic modulus is retained as a separate effective
comparison with an explicitly assumed Poisson ratio. Because RCT includes
structural buckling, it must not silently replace the sheet modulus in an
explicit flute model.

An elastic model cannot leave a permanent crease. A deformed snapshot under
applied boundary conditions cannot establish springback, plasticity, tearing,
adhesive failure, or self-contact. Published calibration and validation must
refer to the same material grade, orientation, conditioning, geometry and
loading protocol; constants from different papers must not silently be mixed.

## Build

```sh
cmake -S matter -B build-cardboard -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-cardboard --target numi-matter-cardboard-material-check numi-matter-cardboard-gpu-check numi-matter-cardboard-probe -j4
ctest --test-dir build-cardboard -R '^matter\.(compiler|metal)\.cardboard_material$' --output-on-failure
build-cardboard/numi-matter-cardboard-probe --help
```

The native probe exports its authored and accepted geometry, step diagnostics,
and execution identity. An unsuccessful nonlinear solve is failed numerical
evidence and must remain in the record. Output paths must be new. The study
record is stored outside the source checkout so ignored build cleanup cannot
silently remove the retained attempts.

## Required progression

1. Verify dimensional material inputs, stress and tangent consistency,
   zero stress under rigid rotation, mass, connected topology and material axes.
2. Run no-load and loaded native controls; inspect actual accepted free-node
   motion, boundary reaction, determinant bounds and failed steps.
3. Check spatial and temporal sensitivity before promoting response values.
4. Reproduce a published loading protocol and compare like-for-like observables.
5. Add source-supported plasticity/interface laws and perform loading/unloading
   and a held-out creasing condition before attempting box assembly.

There is no closure, tab insertion, glue curing, humidity evolution or full-box
assembly claim in the initial elastic instrument.

## Native strip use

```sh
build-cardboard/numi-matter-cardboard-probe --output /tmp/cardboard-new-run \
  --nx 8 --ny 1 --thickness-slices 1 --steps 8 --bend-angle 0.25 --dt 0.0001
python3 matter/tools/audit_cardboard.py /tmp/cardboard-new-run/mesh.json \
  /tmp/cardboard-new-run/accepted_step_000008_bent.obj /tmp/cardboard-new-run/audit.json
python3 matter/tools/render_cardboard.py /tmp/cardboard-new-run/initial.obj \
  /tmp/cardboard-new-run/accepted_step_000008_bent.obj /tmp/cardboard-new-run/strip.png
```

The audit requires NumPy; the renderer additionally requires Matplotlib.
The renderer reads actual exported positions without amplifying displacement.
The initial OBJ contains native FP32 positions; the mesh JSON preserves authored
FP64 coordinates, so their tiny initial differences include quantization.

`--fgmres-iterations 32` increases the Krylov iteration budget while retaining
the acceptance tolerance. The default remains 10; record the budget when
comparing runs.

The left end is clamped. The right end rotates about its own section center
with that center fixed. This constrains translation and produces shear as well
as bending; it is not a pure-moment or four-point-bending experiment. With the
example settings the motion takes 0.8 ms and is dynamic. Reaction sign is force
exerted on the prescribed right boundary; the reported moment is about that
section's center. No left/right force balance or quasi-static limit is claimed.
`--without-medium` preserves the two liners and their grip constraints, leaving
two disconnected components. This is a numerical removal control, not a
specimen with the same mass or boundary-force distribution.

See [the retained development evidence](examples/cardboard/evidence/README.md)
for successful runs, failed refinement, and unresolved numerical sensitivity.
