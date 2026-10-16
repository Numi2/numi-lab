# Selected Newton FEM capture

`NM_FEM_OPERATOR_FD_CAPTURE_PATH=/new/path.jsonl` enables the rich readback when the selected diagnostic is also requested with `NM_FEM_OPERATOR_FD_SELECTOR=<control-step>:<zero-based-Newton-iteration>`. The destination must not already exist; the runtime opens it exclusively. The capture has a 64 MiB scratch budget and a separate 32 MiB readback ceiling; oversized captures remain explicitly rejected. With either selector unset, this export path is inactive.

The runtime copies the assembled center residual into private scratch immediately before FGMRES. FGMRES restart cycles may reuse `femResidual`, so the capture reads that private pre-FGMRES copy. The capture itself runs after the selected Newton solve and before its correction is applied. It records diagnostic buffers and does not replace production state.

The file is JSON Lines:

- `header`: control-step and Newton selection, world/execution fingerprints, counts, actual epsilon values, timestep by environment, constitutive material names and parameter layout, actual environment parameter values, units, and center-residual provenance.
- `node`: one record per environment/node. `accepted` and `center` contain `x_m`, mass, velocity, inverse mass, rest position, constraint tag, delta velocity, persistent reference displacement/mode, and the raw state float4s. Four `epsilon_pair_*` states contain the actual side candidates. Their differences from center are emitted as realized position, velocity, and persistent-displacement deltas. `center_residual_impulse_Ns` is the privately copied assembled RHS; `production_A_delta_v_impulse_Ns` is the production gathered operator action; `epsilon_pair_residual_impulses_Ns` contains the four reassembled side residuals.
- `tetrahedron`: one record per environment/tet. The center record includes material identity, node IDs, rest volume, inverse rest matrix, local material frame, world/local F, Fdot and directions, P, algorithmic tangent, accepted/projected state, raw projection diagnostics, status/flags, Hill indicators when the 13-state law applies, and actual/operator element-force vectors. Each of the four `sides` records contains its epsilon, candidate F/Fdot, P, projected state, and projection/status diagnostics.
- `integrity`: reports whether the existing diagnostic's before/after bitwise check found its protected production inputs unchanged.

Matrix arrays are row-major 3×3. Material-local fields are the inputs and outputs after the production frame rotation; world fields are included to audit element assembly. `local_F_direction` is per unit velocity correction (`grad(delta_v)`); each side uses `F_center +/- epsilon*dt*local_F_direction`. Residuals and operator actions are impulses in N·s. P is in Pa; tangent and operator element-force outputs are per second. Hill `q`, `lambda_increment`, `a`, and `b` are dimensionless. Nonfinite floats serialize as JSON `null`.

The center residual is labeled `pre_fgmres_private_copy` because it is copied from the assembled RHS before FGMRES can overwrite restart residuals. Per-side residuals are the actual production residual reassemblies at the recorded side candidates. The rich output supports replay and localization; it is not a convergence certificate or a physical validation result.

The current tet layout uses 251 float4 slots, including `projected_state_low`
for the center and four perturbations. Those words are temporary projection
precision, not accepted persistent history. Cases
`solver/regional-bending-captured-cycle-057` and
`solver/regional-bending-capture-control-058` have byte-identical production
outputs across all 20 files, including rejected conditioning buffers. The
source/build/control checks and hashes are recorded by
`cardboard_diagnostic_history_compare.py`. Both attempts failed on their second
loading step; parity of this bounded failed attempt is separate from full-cycle
invariance, solver convergence and physical qualification.

For the compact contact snapshot, set `NM_FEM_CONTACT_TRACE_ROOT=<control-step>`. By default it keeps the legacy first 32 Newton iterations, with at most 128 stage records and a 1 MiB cap per stage. To inspect one later iteration without producing the full hex dumps, also set `NM_FEM_CONTACT_TRACE_ITERATION=<zero-based-iteration>`; this captures at most the four `before_contact_limits`, `after_deformable_limit`, `after_rigid_limit`, and `final_line_search` stages (4 MiB maximum total). Iteration selection is diagnostic only; an oversized stage emits a skip record, and all trace paths are read-only with respect to solver state.
