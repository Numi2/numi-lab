# Source201: preserve completed regional RAP restart state

Source201 is a bounded solver-conditioning change based on source198. When an
environment has completed its current FGMRES restart cycle, restart-only RAP
build/factor passes preserve that environment's same-Newton augmented RAP
matrix, pivots, LU, factor-valid and fallback state. Active environments still
build/factor their current matrices. The original base RAP kernels remain
unchanged. The restart RAP dispatch now covers E times mode-count squared
groups, fixing the prior E2 launch that covered only one environment.

The per-Newton reset and restart completion marker remain generation-local.
The final true-residual checks, nonlinear Armijo path, update ordering and
transaction rollback are unchanged. This is conditioning-state reuse; it does
not alter the physical operator or raise a solver limit.

The source201 frozen map contains 817 selected files and is based on frozen
source198. Its only source delta is matter/CMakeLists.txt,
matter/src/metal/fgmres.metalinc, matter/src/runtime.mm, and the new
matter/tools/cardboard_regional_rap_tail_preserve_native_check.mm. Source198's
base RAP build/factor kernels are byte-identical. The publication checkout also
retains source199's stricter full-contact checker unchanged; source201 helper
build receipts remain bound to the standalone frozen201 snapshot and do not
claim a rebuild of the composed source199-plus-source201 checkout. Source-bound Debug and
Release helper builds passed; the two earlier Release build failures are
retained in the build evidence. A separate Metal O0 compile and the
integrated Metal O3 artifact were both exercised by native534. Both runs
passed the named E2 helper groups with identical output and zero physical
transactions. This synthetic kernel check is not a FEM solve, full solver
parity, performance measurement or physical qualification.

The earlier source198 control17 replay evidence is separately bound:
native531/native532 output parity covers the captured failure, native530's
saved prefix is accepted only for controls0-16, and the accepted-prefix package contains 34 accepted rows across two
environments (17 per environment). Separately, the source198 strict layer
covers 1,728 paper centers and 17,280 directional actions at Newton34. The
nine-basis FP64 sweep resolves 15,529 of 15,552 row/direction pairs; the
remaining 23 have no adjacent same-branch pair meeting the declared gate. These results do not
qualify a full nonlinear coupon or general native FP32 derivative behavior.

The fresh source201 60-control coupon was in progress when this publication
cut was prepared and is excluded without an outcome. All six qualification
axes remain open: solver correctness, numerical convergence, component
calibration, analogue validation, physical assembly in simulation, and
crush/impact prediction. No complete coupon, shipping assembly, mesh/time
refinement series, qualified compression peak, crush, or drop result is claimed.

Exact source, build and test bindings are indexed in
evidence/native-implementation-20261014/publication-binding-reference035.json.
See also solver-milestone-source198.md and solver-milestone-source199.md.
