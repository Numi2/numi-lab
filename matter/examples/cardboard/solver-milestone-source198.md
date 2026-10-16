# Source198: exact contact trials and a bounded damped solver direction

Source198 implements shared FEM/contact trial geometry, full-residual descent for
pure-FEM mixed contact, and an experimental damped correction. The original
kraft/NSSC/starch recipe and miniature regression remain unchanged. The
300 × 200 × 200 mm explicit-flute shipping box is still unqualified.

The [publication binding](evidence/native-implementation-20261014/publication-binding-reference033.json)
identifies the exact source, build, payload hashes and six separate verdicts.
Source198 has 816 selected source files. Its build passes seven targeted CPU
CTest checks, 13 solver-environment checks and the independent damped-CGLS oracle.

## Solver changes

Private FEM and contact trials now construct the same endpoint from accepted
history, candidate velocity and the exact tested update fraction. Constitutive
candidates, contact geometry and CCD are recomputed from that history. Pure-FEM
deformable-contact trials use the complete coupled residual as the merit test;
the local prepass still enforces finite barrier evaluation and certified CCD.
MPM and mixed FEM/MPM supports retain their local barrier-energy descent path.
The physical operator, final production certificate and transactional rollback
remain in force.

The opt-in CGLS direction minimizes
`||A x - b||² + μ² ||x||²` from zero, with the sole admitted positive damping
setting `NM_FEM_BASE_CGLS_DAMPING_KG=1e-3`. It has a bounded, contact-free scope,
requires the existing tangent and true-residual safeguards, and is disabled by
default. Its numerical configuration participates in checkpoint identity.
The old Newton direction's certificate does not transfer to this new direction:
the exact selected update must pass physical residual descent and the final
production certificate. No global tolerance was relaxed.

## Measured contact result

The small synthetic fixture contains 36 FEM nodes, 48 tetrahedra and one
cohesive face. It does not calibrate corrugated board.

With the regional preconditioner and expanded numerical budget, native526
accepts two controls and then reaches the intended postCommit fault. The third
solve has a residual of 9.63568425e-8 against the unchanged 1e-7 gate before the
prescribed-motion validator rejects the deliberately dynamic punch. Its status1,
failing index0 and diagnostics[-3,-3,-3,-3] identify that fault. Accepted histories
and cursors roll back, and the certificate is invalidated.

Within each run, all 13 baseline/shared-scratch output pairs are byte-identical.
The [independent transaction audit](evidence/native-implementation-20261014/analysis/native526-527-independent-transaction-audit-001/report.json)
checks source/build/material bindings, raw snapshots, trace ordering and rollback.

Scalar native525 and source198 repeat527 accept two controls but fail residual
descent at Newton18 before reaching the injected fault. Their checker exits0
because its old assertion accepts any rejection with rollback. That proves
rollback for the actual failure; it does **not** isolate the intended fault.
Their complete output payloads are byte-identical. The expanded budget changes
FGMRES64→256, restart10→16, line search8→24 and Newton16→32 together; this is
not an isolated restart comparison. Exact same-state operator equality between
the scalar and regional selector is not measured by these trajectories.

The earlier native520/522 failure is observer-invariant. Independent replay
finds `||b-Jδ||/||b||=0.999982264`, and the saved residual workspace agrees with
freshly formed `b-Jδ`. This capture demonstrates linear stagnation, not an
action mismatch or universal tangent correctness.

## Damped recurrence and retained failures

Native523 executes the actual production CGLS recurrence on a nonsymmetric
three-degree-of-freedom test. O0/O3 outputs are byte-identical. All four positive
damping cases complete caps32/64 with zero error bits; solution errors are about
1e-9 and independently formed residual-vector differences about1e-11.
These are private kernel checks with zero physical FEM transactions.

The four undamped controls underflow the denominator at iteration6 and report
breakdown. That remains a limitation. Native518 failed before dispatch because
the host lookup omitted the exported kernel namespace. Native519 corrected the
lookup but its helper misreported the residual by subtracting `b` twice and
stopped after six iterations with status running. Its exit0 does not qualify
residual accuracy or termination. Native523 corrects both helper defects.
Failed builds, wrapper admissions, the helper errata and a retracted false
duplicate-writer review finding remain archived.

## Shipping memory and open qualification

Accounting now counts the private trial arena once and includes the optional
sorted-incidence workspace, `12 E bit_ceil(6 C)` bytes. Malformed selectors and
overflow reject. The shipping mesh still has528,444 nodes,1,477,728 tetrahedra
and260,835 cohesive faces. Its bounded native arena limits remain unresolved.
Planning estimates are not measured whole-process allocation or a16GiB pass.

All six overall axes remain open: solver correctness, numerical convergence,
component calibration, analogue validation, physical assembly in simulation,
and crush/impact prediction. No complete coupon or requested mesh/time series
passes. No fixture-free shipping checkpoint, qualified compression peak,
gravity crush or drop trajectory exists. Public-data identifiability limits on
exact-recipe damage, local starch/hot-melt fracture and bare-box impact remain
explicit.

The fresh native521 coupon is excluded while active. Its outcome cannot be
inferred from the component checks above.
