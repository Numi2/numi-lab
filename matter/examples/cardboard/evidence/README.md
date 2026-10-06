# Cardboard instrument evidence — 6 October 2026

The initial elastic instrument is recorded below. The subsequent
[paper plasticity and finite-glue evidence](structured-paper-and-glue/README.md)
records the new default recipe, rejected trials, solver corrections, and native
loading/unloading checks. Keep the two model versions distinct.

This is native numerical development evidence, not experimental validation of
paperboard or a demonstration of permanent folding. No new physical data was
collected. The source baseline is `c3d3641f06c80b7723324a39c9e0f8e9af2f431e`.
The runtime device was Apple M4. Invocation receipts bind the executable,
Metal library, authored material files, and probe source with SHA-256.

## What was verified

The CPU check compares the DSL energy, stress, and tangent with an independent
orthotropic stiffness calculation and finite differences. The Metal check
compares the production FEM internal-force and tangent kernels with a rotated
tetrahedron oracle. Both CTests passed; see `constitutive-ctest.log.gz`. These
constitutive tests take zero physical timesteps and do not qualify a box.

An independent integration of the sinusoidal flute gives total sheet mass
0.486768504 g for the 36 × 20 × 5 mm coupon, excluding glue. The mesh mass is
0.477802689, 0.484486034, and 0.486194612 g at 8, 16, and 32 subdivisions per
pitch. All have positive tetrahedron volumes and one connected component.
The geometry error decreases with refinement; the original arc-and-tangent
flute profile in Nagasawa is still replaced by an explicit sinusoidal assumption.
See `geometry-reference.json` and the three `geometry-nx*` directories.

## Native motion and controls

All cases below hold the left end, rotate the right end about its fixed section
center, and carry an independent unloaded environment through the same runtime.
The ramp lasts 0.8 ms. Width and through-paper subdivisions are both 1: this is
a coarse, high-aspect-ratio mesh. Reactions are dynamic and are not pure-bending
or quasi-static stiffness measurements.

| Attempt | Pitch subdivisions | Steps × dt | Outcome | Final right moment, N·m |
|---|---:|---:|---|---:|
| `pilot-001` | 8 | 8 × 100 µs | 8 accepted | −0.0837986 |
| `refine-x-001` | 16 | 8 × 100 µs | 7 accepted; eighth rejected, code 10 | Invalid endpoint |
| `refine-time-001` | 8 | 16 × 50 µs | 16 accepted | −0.0975654 |
| `no-medium-001` | 8 | 8 × 100 µs | 8 accepted; two disconnected liners | −0.0449905 |

In the pilot, the independent exported-position audit measures 16.4512 µm
maximum **free-node** displacement, versus 4.23 nm in the unloaded reference.
Those audit differences include FP32 serialization relative to the authored
FP64 mesh. The loaded determinant range is approximately 0.99401–1.00603.
This demonstrates solved interior deformation, rather than imposed motion of
the entire mesh. Removing the medium changes the response, but also removes
mass and changes connectivity; the table alone does not isolate a physical
stiffening coefficient.

Halving the timestep changes final moment magnitude by 16.4%. The spatial
refinement fails, so neither spatial nor temporal convergence is established.
The rejected eighth step's geometry is the retained/rolled-back accepted state;
its reaction must not be interpreted as a valid response at 0.25 degrees.
The separate `bent-audit.json` in that directory audits step 7's accepted export.

## Telemetry correction and retention

The original pilot CSV labels `diagnostics.w` as `correction`; this is incorrect.
On ordinary finalization it is maximum contact speed. The raw diagnostic vector
also changes interpretation on failure: `refine-x-001` has code 10 and x =
0.0001003895013, which is not a minimum determinant. Its independently computed
J remains positive. The corrected probe names all four fields as raw status
diagnostics. Successful status z reads scheduler numerical.y; that field is not populated
as the successful KKT residual in this path and may stay zero. No zero
diagnostic is promoted into a convergence proof.
The original CSV and exact original probe source (`pilot-probe-source.mm`) are
retained unchanged, with the source SHA matching the original receipts.

`sensitivity-plan.json` was recorded before the three sensitivity runs.
The complete native package, meshes, every exported step, and render are retained
locally at `/Users/home/cardboard-evidence-20261006`; this directory publishes
compact receipts and observations. These are instrument-development records,
not a registered `numi.science.plan.v2` physical study.

## Budget revision and unused condition

The failure-site source audit identifies x = 0.0001003895013 as relative
residual, just above the unchanged 0.0001 acceptance threshold. The original
budget was ten Krylov columns. The revision exposes the budget as a CLI argument
and records it, without altering any tolerance. Its predictions and stopping
rule were written before execution in `budget-revision-plan.json`.

With `--fgmres-iterations 32`, `refine-x-budget32-001` completes all eight
steps on the 16-subdivision mesh. The final step uses 20 columns. Its final
right moment is −0.0810481 N·m, maximum free-node displacement 13.6661 µm,
and accepted J interval 0.994558–1.004242. The unused 0.125-degree condition
(`unused-angle-budget32-001`) also completes eight steps; its moment is
−0.0525450 N·m and free-node displacement 7.04569 µm. The predeclared decrease
in moment occurs, but the response is not proportional to the imposed angle.

This supports iteration budget as a cause of the earlier rejected endpoint.
It does not establish mesh/time convergence: the budget changed together with
the successful refined comparison, width/thickness are still coarse, and
temporal sensitivity remains unresolved. A next numerical test should hold
budget 32 fixed while refining time, width, and thickness independently. No
physical constitutive parameters were fitted to these native observations.

## Remaining physical work

Before interpreting real folding/assembly: resolve numerical sensitivity,
reproduce a matching published fixture and specimen, then add source-supported
irreversible paper response, adhesive/interface damage, and self-contact.
Loading/unloading and a held-out crease condition must distinguish permanent
folding from a temporarily bent elastic sheet. Glue cure, humidity, tearing,
tabs, and full-box closure are not implemented by this slice.

The final source additionally clarifies the diagnostic semantics in its manifest;
`budget32-probe-source.mm` retains the exact source of the two budget-32 runs.
