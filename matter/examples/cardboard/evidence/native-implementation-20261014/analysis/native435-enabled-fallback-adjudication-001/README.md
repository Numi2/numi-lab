# Native435 base-state Cauchy fallback adjudication

## Result

Native435 shows a local fallback acceptance, not a completed coupon. The ordinary Newton attempt at environment 0 / Newton 48 rejected all 24 Armijo trials. Its numeric primary history through Newton 47 is byte-for-byte equal to native434, and the primary Newton-48 residual/trial values also match. The only primary-48 trace differences are status fields: after the primary rejection, native435 clears the status so its explicitly selected fallback can run; native434 retains status 10.

At Newton 48, native435 evaluates the base-state Cauchy direction `p = tau * A^T b`. Its recorded gradient norm, `A`-gradient norm, and FP32 `tau` match the independent native-kernel report433 within the serialized trace precision. The original Newton solve was marked converged, but its certificate is explicitly not applied to the Cauchy direction. Residual Armijo selects alpha 1: the environment-0 metric drops from `1.84443750e-4` to `1.60052223e-4` (ratio 0.867756, reduction 13.224%). The held reference environment is a certified no-op.

The next ordinary Newton updates 49–52 are accepted, with alphas 0.03125, 0.000244140625, 3.81469727e-6, and 1.1920929e-7. Newton 53 rejects after 24 trials: its residual metric rises from `1.55870541e-4` to `1.55870759e-4`, status 10, failing index 53. The second bent loading control therefore fails; the run exits 2 with one accepted control and `physical_validation=false`.

## History and rollback evidence

Both native receipts report execution started, unchanged protected inputs, no timeout, and no termination errors. The true-residual stream has 4,096 records in each run: 64 restart cycles at each Newton iteration 0–63. The records are exact through Newton 48. After the failure status propagates at Newton 54, environment 0 records status 10 / operator status 4 and `true_l2=-1`; these are explicit status-gated sentinels, not residual measurements. The Armijo logger has a 64-record cap. Native435 uses one slot for the Cauchy attempt, so its Armijo trace stops at Newton 62; it has no Armijo record for Newton 63 even though the true-residual event stream reaches it.

The 35 output files share the same paths; 31 are byte-identical. The accepted first bent snapshot, held-reference snapshots, mesh, compiled package, and material state are exact matches. In both runs, the rejected bent diagnostic OBJ is byte-identical to the last accepted bent OBJ, consistent with visible geometry rollback. The final rejected candidate-node buffer differs and is treated only as an uncommitted candidate. The only observation-row differences are the failed bent control’s failure index, FGMRES count, and final base/trial residual diagnostics. No complete coupon or final accepted certificate exists.

## Source and command scope

Native434 uses frozen source133; native435 uses source135. Their frozen-source manifests have the same parent revision, and only `matter/src/runtime.mm` differs. Source135 describes the added change as fail-closed positive-finite immutable-mass admission for the opted-in fallback. Liner and medium material files have identical SHA-256 hashes. The physical probe argv matches after normalizing only executable, output, material, and metallib paths. All common solver environment variables match after run-path normalization. Native435 adds `NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=1:48` and passes the exact report433 as an external input. Question/prediction strings and source/build paths differ as run metadata.

The run used source-bound native artifacts and preserved source inputs. `report.json` binds command, receipt, logs, run outputs, external source-build records, binaries, metallibs, source snapshot manifests, and report433. `manifest.json` hashes every file in this adjudication package. `audit.py` rechecks the comparisons using Python's standard library; it performs no simulation or GPU work.

## Claim boundary

The result establishes that the fallback direction was formed and passed this local residual-Armijo check after the primary Newton rejection. It does not show that the Cauchy direction is linearly certified, that the 27-control protocol converges, or that the board is physically calibrated or qualified. The final status-10 rejection and absence of a completed coupon remain the governing outcome.
