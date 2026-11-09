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
