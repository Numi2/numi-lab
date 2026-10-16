# Source154: complete-cycle solver controls and regional-state evidence

The native driver now admits an outer Newton budget of 128 and an exact opt-in all-control CGLS policy. The runtime routes `NM_FEM_TRUE_RESIDUAL_ROOT=all` and `NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=all:all` through the existing capability and certificate checks on each control. `all:all` requires the bounded CGLS primary policy and all-Newton suffix. Numeric selectors and defaults retain their behavior. No material parameter, physical operator, tolerance, final certificate or rollback rule changes.

The extracted parser checks pass 44 cases. The 128 budget is present in the compiled protocol; 0 and 129 are rejected. Source153 was built through the root project. Source154 was built through standalone `matter`, with three artifacts, rather than the wider root-project artifact set. This build-topology difference is explicit in the retained erratum and run receipts.

Native466 tests the existing numeric selector at 64 Newton iterations across that change. Its complete attempted history matches native463: 35 output files, 112 Armijo records, 4096 true-residual records, 64 CGLS records, and exact CGLS/state/status payloads. Only run/source paths require mapping; no program-identity normalization is needed. Both attempts fail control 1 after one accepted control. This establishes compatibility, not convergence.

Native467 is the separately source-bound 128-Newton all-control experiment. It requests all loading, hold, withdrawal, relaxation, release and post-release phases. The capacity guard interrupted it after 703 seconds when available disk space fell below its 512 MiB floor. The native process received SIGTERM, with exit -15, no timeout and unchanged inputs. Control 0 completed; control 1 reached encoding but has no completion record. This is an incomplete capacity interruption, not a nonlinear failure or a completed-cycle result. No accepted prefix qualifies a complete coupon.

## Regional observer evidence

The isolated source151 observer captures 30 regional arrays, 77,619,992 bytes, at the same physical state as the prior419/455 center. It is not integrated into source154. Native464 preserves native462's full attempted history: all 4282 common non-timing event rows match, as do the legacy CGLS/state/status payloads and stdout. The additional regional capture is diagnostic only.

The sidecar has a retained metadata defect: `primary_policy_selected_environment=[0,0]` is copied before CGLS layout/selection fills that scratch. It must not be interpreted as current selected-environment data. The same-point Armijo and FGMRES state and final CGLS selection are separate evidence. Isolated source156 corrects the metadata and reader contracts; its 23 CPU tests and Objective-C++ syntax check pass. It has no native build or observer-parity run and is not integrated here. The original payload and misleading field are preserved.

The current-state AP and RAP comparisons have relative errors of approximately 1.04e-7 and 3.63e-8. The graph-patch comparison retains a failed absolute entry gate: maximum difference 7.34e-8 exceeds its predeclared 1e-9 limit. Its free/free Frobenius relative error is 7.47e-8; that separate statistic does not change the failed gate or resolve an FP32 absolute noise floor.

A bounded CPU experiment compares 64 CGLS iterations over the captured operator. FP64 relative true residuals for identity, node-block and unlatched base-regional scaling are 0.22967, 0.08845 and 0.57905; CPU-FP32 results are 0.23562, 0.08979 and 0.92340. All cases reach the cap with finite iterates. The regional formula is worse in this experiment. It is not evidence of improved native conditioning or a consumed restart-augmented map.

The external reader verifies hashes, exact joined state, graph topology/free mask/layout checksum, finite arrays and valid factor pivot permutations. The restart-augmented RAP remains unadmitted because its independently consuming restart cycle was not captured. Current projected material history is also unavailable in the captured candidate-history arena. CPU graph-map/transpose tests do not establish native arbitrary-vector action or transpose parity.

## Reproduction and remaining qualification

The retained source package reconstructs sources151,153 and154 from published base `67ba06be7f48981dda142ea35eedd792d7cd2ca5`, with complete on-disk restoration and hash verification. Native466's first launch preflight refused colliding external input basenames before creating a native case. Versioned command packages repair only those names using byte-identical copies; the refused command remains intact.

See the archived `analysis/source154-default64-native466-parity-001`, `analysis/source151-regional-observer464-parity-001`, command packages and native receipts for exact inputs and complete failed histories. Singleton CGLS/state captures must be omitted from all-control commands because their `O_EXCL` paths lack a control namespace. Single-root Armijo trace selection remains numeric-only.

| Verdict axis | Current result |
| --- | --- |
| Solver correctness | Default64 and observer full-attempt compatibility controls pass. Complete nonlinear/contact solve and native regional-map/transpose qualification remain open. |
| Numerical convergence | No complete coupon mesh/time/slower-loading comparison; native467 is capacity-interrupted. No downstream damage/interface/scoring/assembly/crush convergence. |
| Component calibration | Published-data package retained. Exact kraft/NSSC progressive failure and local starch fracture parameters remain unidentified. |
| Analogue validation | Grade, fixture, conditioning and rate transfer limits remain explicit. |
| Physical assembly in simulation | No completed 12-score, lap/end-bond, released and settled 300×200×200 mm shipping box. Shipping-scale contact/memory admission remains open. |
| Crush and impact prediction | No qualified compression peak, capacity-derived scenario masses, gravity crush or 100 mm drop result. |
