# Full-contact true linear residual instrument

`NM_FEM_TRUE_RESIDUAL_ROOT=<control-step>` recomputes the complete immutable
FEM `b - A x` after each bounded FGMRES restart using the current candidate
and the same accepted constitutive and cohesive history as the production
matvec. The node gather includes mass, bulk, kinematic rigid contact and
self-contact. The diagnostic additionally dispatches the production cohesive
face derivative and nodal scatter into a private face-vector arena. Schema2
trace records declare this operator scope and the cohesive face count.
Earlier schema1 traces did not admit cohesive faces and remain separate.

The supported instrument has one explicit FEM object, at most four
environments,4096 nodes and4096 immutable cohesive faces, no dynamic or
articulated rigid unknowns, fields, MPM, Human constraints or mutable topology.
An unsupported scope or missing private arena is reported as a skipped
instrument. The byte budget is bounded independently of the production face
operator, which this read-only path does not overwrite. The freshly evaluated
operator uses private status and face-vector output; no accepted history,
material candidate or cohesive reference is committed by the instrument.

`NM_FEM_TRUE_RESIDUAL_RESTART=0` (the default) reports the difference between
the Arnoldi least-squares estimate and the fresh full-operator residual. It
must be checked against an otherwise identical diagnostic-off run before
using its numerical observations. `=1` also replaces the restart residual
with this same physical operator; this is an explicitly different numerical
algorithm experiment and is not diagnostic-on/off parity evidence.
Neither mode changes the nonlinear tolerance, contact feasibility or final
production certificate. No measured true-residual result is asserted by this
source change; native results must be source/executable/shader bound.


The current implementation isolates element, cohesive and assembled nodal
operator output in dedicated bounded diagnostic buffers. It admits at most
4 environments, 4096 nodes, 32768 tetrahedra and 4096 cohesive faces in the
existing immutable explicit-FEM scope. Nonfinite metrics fail the private
instrument before any optional residual replacement. Healthy finite metrics
are kept separate from trace rows emitted after production failure.

Integrated source 45 measured the full operator with unchanged physical
history: scalar final-cycle true/Arnoldi differences were at most 2.873e-5
relative, while the strict nonlinear transaction still rejected. That capture
predates the dedicated element/nodal scratch isolation and has its own exact
source binding. A fresh native parity run is required for the current buffers.
