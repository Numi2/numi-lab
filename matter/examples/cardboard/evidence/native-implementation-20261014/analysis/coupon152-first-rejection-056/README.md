# Coupon 152 first rejection audit

Classification: **failed native solver run**, not a completed cycle. The probe accepted 3 of the planned 27 bent-arm controls, then rejected loading control step 3 (zero-based) for the bent environment. It exited 2 without timeout; receipt says source inputs were unchanged. The held-reference environment had a successful CSV row at that control index, but it does not make the bent run successful.

## First rejecting gate

The failed CSV row records status 10 (`NM_STATUS_NONLINEAR_SOLVER_FAILURE`), object 0, `failing_index=23`, and 24 line-search trials. Its Armijo diagnostics are:

- base free-DOF residual norm: `4.4076828089600895e-6` (about 4.41 times the configured `1e-6` tolerance)
- last trial norm: the same `4.4076828089600895e-6`
- last trial alpha: `1.1920928955078125e-7` (`2^-23`)
- trial count: `24`

The source requires strict residual decrease (`state.y < state.x`) as well as the Armijo bound, and writes code 10 after the trial budget is exhausted: [`fem.metalinc` lines 4297–4300 and 4338–4343 in the bound source54 snapshot](../../snapshots/source-tangent-cache-trial-scratch-and-exact-AP-integrated-054/matter/src/metal/fem.metalinc). Thus the concrete first failure is **residual-Armijo exhaustion**, not a final certificate or determinant rejection.

The microstep FGMRES count reached the configured 1024 maximum. The capture semantics in [`runtime.mm` line 10648](../../snapshots/source-tangent-cache-trial-scratch-and-exact-AP-integrated-054/matter/src/runtime.mm) say that this is the maximum `totalUsed` observed in the microstep, not a per-Newton count. No per-Newton true linear residual or convergence flag is present, so this does not prove that the Krylov solve failed. The available evidence cannot yet distinguish an inexact Krylov direction from a local tangent/precision issue or another source of a non-descent Newton direction.

The recorded trial residual is finite and candidate trial statuses were healthy enough for a finite norm; there is no material-projection failure status. `min_J` and `max_J` on the rejected row describe the retained accepted state, per the output manifest, so they do not establish a failing candidate determinant. The failed row has `certificate_raw_accepted_flag=0`; the rejection occurred before successful publication, not at a final accepted-state certificate.

## Conditioning capture and checker-sign question

`rejected-conditioning-000004` is marked `accepted_physical_state=false`, control step 3, with 2 environments, 882 nodes, and 26 modes. The physics fingerprint matches the probe result. Both factors are marked valid; 846/882 node-block rows are valid per environment (the remaining 36 correspond to grip-constrained nodes).

Recomputing `RAP = effective_P^T AP` from the byte-hashed arenas gives valid symmetric RAP spectra and pivoted-LU reconstruction errors of `8.04e-8` (bent) and `7.90e-8` (held reference). This rules out a sign/reconstruction discrepancy inside the captured coarse matrix. The RAP condition numbers are 2.15 and 2,330.72 respectively. The full-space A-invariance residuals are large (0.36–0.39 for bent; 0.73–1.00 for held reference), so these regional modes have limited coverage of the full operator; these are not full FEM condition estimates and do not prove why the Armijo line search failed.

The snapshot’s usual `cardboard_regional_conditioning_audit.py` CLI rejects this run at its outer-protocol gate, correctly for that tool’s narrower case: it expects one accepted step, Newton budget 20, and FGMRES budget 256. Coupon 152 has three accepted steps and budgets 64/1024. That gate mismatch is **not** a checker sign bug. For the descriptive arena calculations here, only that outer-run predicate was bypassed; the arena schema/hash/RAP checks were retained. The run’s sealed source/runtime/external inputs were independently verified: all 755 declared hashes match.

The earlier run 135 is only context: it used source52/build052 and timed out after two accepted bent controls at 600.21 seconds. It is not a matched source comparison to this source54/build054 attempt.

## Reproducibility and evidence limits

`report.json` contains the source snapshot, per-source-file hashes, all 755 verified input-hash count, run receipt, solver settings, output hashes, and hashes for every conditioning arena. The report is diagnostic-only. To identify what made the Newton direction stop descending, repeat with the same source54 recipe and capture per-Newton FGMRES true residual/convergence plus the Armijo base/trial sequence at bent environment, control step 3, Newton index 23. Do not attribute the failure to FGMRES based only on the microstep maximum.
