# First rejected reacting-body solve: bounded trace

The existing legacy mixed-FEM, articulated-box fixture was rerun at its
unchanged `1e-7` scalar tolerance and original 16 Newton / existing FGMRES
budgets. It again rejected the first step with Matter status 10 and the same
diagnostic values as the uninstrumented run. This is a poroelastic mixed
fixture, not the reference-area midpoint plane or a force-qualified control.

The new opt-in trace copies only bounded existing solver/line-search/status
arenas. At each of the 16 `final_line_search` boundaries the recorded
selected alpha and object limit were 1. The reassembled nonlinear norm fell
from 0.500015974 at iteration 0 to 1.43790231e-7 at iteration 15; the
terminal status reported 1.34402e-7. Both norms include mixed-coordinate
rows and are not a pure physical impulse measurement. The data do not show a
clearance-limited tiny alpha. They do not independently recompute the linear
residual, certify an Armijo sufficient-decrease decision, or establish why
progress slowed. The reported FGMRES values are solver diagnostics only.
No tolerance or budget was changed to obtain this result.

On the default-tolerance fixture, traced and untraced runs produced bytewise
identical audit snapshots of rigid q/v, FEM nodes and fields, contact history,
material state, schedulers and generalized candidate
(SHA-256 `d29bfa0715403f073e10e1e5365b4d9034941767ca742176d8d9f9bd9a0f294c`).
Each run also passed late geometric rejection and exact retry from initial
and already-loaded checkpoints. The diagnostic snapshot is not a restart
archive and does not include every possible owner or fixture. Four scoped
CTest regressions passed; an earlier expanded run had an unrelated existing
articulated topology-guard failure, retained in the rollback report.

`coupled-tight-newton-018.log` and its analysis retain the 16 observations;
`coupled-tight-newton-016.log` and `-017.log` retain prior trace-scope attempts
that did not capture iterations. The relevant executable/source hashes and
traced/untraced artifacts are retained alongside this report. The opt-in
trace does not establish independent force, torque, work, or midpoint
accuracy, and the reacting-body midpoint transaction remains unimplemented.

A second opt-in capture retained the actual assembled RHS, proposed
correction, accepted and working nodes and fields, contact samples and
histories, rigid candidate, body projection and point Jacobians at the
assembly and final-line-search boundaries. The raw records are in
`coupled-tight-newton-021.log` and `coupled-default-rich-traced-021.log`;
`analyze_coupled_rigid_rhs.py` reproduces the following checks.

At the default stopping configuration the captured rigid RHS is
2.2572639863938093e-5 N s. The sum of the recorded contact impulses is
9.650689753470942e-5 N s; subtracting the one-kilogram body mass times the
captured candidate velocity change, 7.393425767077133e-5 N s, gives exactly
the same displayed value. Thus the previously observed discrepancy is
present in the assembled rigid residual at the accepted stopping point; it
is not merely a mismatch in the printed contact and owner outputs. The
`1e-7` run has the identical configuration at this point and continues. At
its 15th zero-based iteration, the captured rigid RHS has magnitude
3.31e-9 N s, while the FEM mechanical and pressure block norms remain about
1.00e-7 and 1.03e-7 respectively. These block norms use different coordinate
units and cannot be interpreted as one physical force error.

The rich traced and untraced default runs again had byte-identical captured
physical state and passed both rejection/retry checks. The four scoped tests
passed after the diagnostic change. The capture does not independently
recompute the mixed FEM equations or `J*delta`; it establishes neither a
correct tangent nor the cause of the final residual floor. It also does not
qualify the reference-area law, midpoint coupling, or work balance.
