# Bounded native CGLS and the remaining nonlinear failure

Source142 adds an opt-in, bounded CGLS direction to the existing native FEM
driver. It uses the current unsymmetric mechanical operator and its transpose;
it does not change masses, material parameters, physical loading or production
tolerances. The original Newton certificate is cleared before the replacement
direction enters the ordinary nonlinear residual line search. Final residual,
geometry and transaction rollback rules remain in force. The feature is off by
default and restricted to the recorded small, contact-free paper coupon scope.

The first direct helper test failed. A stage diagnostic located an inverted
constrained-row mask predicate in `measure_initial`. Repairing that predicate
makes the unchanged native checker pass 678 checks, including quiet-environment
and invalid-input controls. This is a synthetic CSR helper check, with zero
physical steps. Failed tests and the diagnostic are retained alongside it.

Native450 then constructs the direction inside the actual coupon solve at
control 1, Newton 48. Its forward and transpose actions agree with the bound
independently assembled captured-cache operator to relative errors 1.45e-7 and 1.44e-7 along the captured
nonzero directions. All 846 free-node masks match; constrained and unused rows
remain zero. The ordinary nonlinear trial accepts fraction 1 and reduces the
residual from 1.84444e-4 to 4.85484e-5 N s, a 73.68% decrease. These comparisons
qualify the selected actions and trial, not the full matrix or trajectory.
The CGLS snapshot lacks candidate-node bytes, so matching package, world, RHS
and earlier trace records do not establish complete candidate-state identity.

The coupon still fails the bounded control solve and rolls back after one
accepted control. Observed Newton 49–62 steps accept small fractions and make
little further progress. The trace's 64-record limit omits Newton 63's Armijo
outcome, so the last branch is unknown. Selecting fallback at every Newton
would leave the observed accepted steps unchanged and could affect that final
unobserved branch. A separate explicit residual-minimizing direction policy is
the next experiment; no unexecuted result is included here.

Native452 removes only the optional CGLS capture observer. Its complete
attempted history matches native450: all 35 output files, stdout, observations,
64 Armijo records, 4,096 true-residual records and the status5 payload are
identical. The remaining stderr agrees after path normalization and removal of
observer/timing metadata. Both terminal receipts report failure without a
timeout. Source141's separate feature-disabled comparison also preserves its
prior complete attempted history; it is not a fresh source142-disabled run.

The earlier native448 attempt ran out of disk space and has no complete native
receipt or CGLS capture. Its partial logs are retained as an operational
failure, with no inferred solver outcome. Subsequent runs use an external disk
reserve and a guard that can terminate only their owned native process group.

The derivative investigation now resolves each of the 1,728 captured paper
directions over at least four adjacent predefined FP64 perturbations at the
1e-6 gate. Those intervals differ by row; there is no universal perturbation.
The assembled ideal FP64 comparison meets the 1% gate over a resolved range.
The native FP32 finite-difference material/element gate remains inconclusive
because the tested differences do not resolve the derivative above arithmetic
noise. Directional evidence does not establish every column of every tangent.

A separate decomposition covers all 1,920 tetrahedra. The captured negative
stiffness work is dominated by compressive initial-stress/geometric stiffness:
-0.142631 J, offset by +0.126816 J from the algorithmic material contribution.
The resulting -0.015815 J agrees with independent and native element actions.
This identifies a finite-state prestress effect along this direction; it does
not demonstrate a residual-sign defect. Compression/shear hardening and grade
transfer assumptions remain physical-model limitations.

| Verdict axis | Current boundary |
|---|---|
| Solver correctness | Selected helper, operator actions, nonlinear descent and observer parity pass; full trajectory, native derivative and nonlinear contact qualification remain open. |
| Numerical convergence | No completed coupon spatial, timestep or slower-loading series. |
| Component calibration | Source-bound tensile hardening fits and component instruments exist; calibrated corrugated damage and independent starch debonding remain open. |
| Analogue validation | No complete held-out pin-adhesion or box fixture validation. |
| Physical assembly in simulation | No scored, folded, bonded, released and settled shipping checkpoint. |
| Crush/impact prediction | No qualified first peak, capacity-derived masses, gravity crush or drop result. |

[Reference024](evidence/native-implementation-20261014/publication-binding-reference024.json)
binds the source restoration packages, commands, complete attempted histories,
failed and superseded evidence, independent comparisons and calibration map.
The [six-axis ledger](evidence/native-implementation-20261014/summary.json)
preserves the remaining qualification boundaries.
