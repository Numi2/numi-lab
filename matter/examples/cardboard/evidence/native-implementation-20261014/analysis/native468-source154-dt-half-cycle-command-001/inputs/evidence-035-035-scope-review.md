# Source154 Newton-budget boundary review

Source154 is a single-CLI-bound change over frozen source153. The accepted maximum for `--newton-iterations` and its help text changes from 64 to 128 in `matter/tools/cardboard_probe.mm`; all other 786 source-hash entries match source153.

The runtime already treats this value as an iteration budget. In source153 `matter/src/runtime.mm:16123-16126` chooses either the override or `state.mixedSolverValue.nonlinearIterations.x`, and `:16138-16140` loops while `nonlinearIteration < nonlinearIterationBudget`; this path has no 64-iteration clamp. `matter/src/compiler_impl.cpp:275-279` cooks `source.newtonIterations` into `nonlinearIterations.x`. `matter/src/validation_impl.cpp:405-415` rejects zero and checks the other solver fields against their own limits, but imposes no upper bound on `.x`.

CGLS remains independently limited. Source153 `matter/src/runtime.mm:11919-11933` accepts only the explicit strings `32` or `64` for `NM_FEM_BASE_CGLS_ITERATIONS`; source154 does not change that path. FGMRES restart stays 16 in the prepared experiment, and physical tolerances/operator/acceptance code are unchanged.

Build and host evidence: the source-bound probe target builds cleanly; `--help` advertises `1..128`; a compile-only 882-node/1,920-tetrahedron package records `solver.newton_iteration_budget=128`, while preserving 32 local material iterations, FGMRES restart16, CGLS64 command intent and `1e-6` tolerance. Compile-only reports `runtime_execution=false`, `physical_validation=false`, and zero accepted steps. Inputs 0 and 129 are rejected with the explicit `1..128` error before an output directory is created.

The earlier source149 outer-128 preparation is retained as a rejected preflight only. Source149 itself advertises and enforces `1..64`; no source149 probe was launched. No native simulation or GPU run occurred for source154. This work demonstrates parser/compiler admission only, not convergence, performance, or physical qualification.
