# Native442 repeated-Cauchy adjudication

Status: **failed_accepted_prefix_only**.

The source-bound run reports 1 accepted control steps out of the 27-control protocol; completed coupon = false. An accepted prefix is not a completed coupon.

It retained 128 residual-Armijo events, 64 Cauchy-labeled pass events (128 per-environment rows), and 4096 true-residual cycle events. Only 2 rows were actual Cauchy invocations; 126 are gated duplicate primary telemetry. Primary Newton records 0–47 and the first Newton-48 Cauchy event are compared field-for-field to native435.

## Actual fallback trials

| Newton | Env | Accepted | Alpha | Base L2 | Trial L2 | Ratio | Outcome |
|---:|---:|:---:|---:|---:|---:|---:|---|
| 48 | 0 | True | 1 | 0.00018444375 | 0.000160052223 | 0.8677562834197418 | selected_trial_accepted |
| 53 | 0 | True | 1 | 0.000155870541 | 0.000142245088 | 0.9125848097236027 | selected_trial_accepted |

## Cauchy telemetry classification

The source emits a Cauchy-labeled event after each selected Newton iteration. Source137 schedules the second pass, but the direction builder only creates a Cauchy vector when that environment’s primary Armijo pass exhausted and set `metrics.w == 3`. An accepted primary leaves the marker at 1 (or 2 for a certified no-op); the second-pass trial is gated and the audit verified that every non-status metric exactly duplicates the primary payload.

The 64 Cauchy-labeled events contain 2 invoked environment trials and 126 gated duplicate rows. Invocations occurred only at Newton 48, 53 for env0; each was accepted at alpha 1. Env1 remained a certified no-op throughout.

The JSON report retains the per-iteration, per-environment classification and any status-only differences.

## Limits

- No physical calibration or convergence claim follows unless all 27 controls complete and terminal acceptance is present.
- Status-gated post-failure residual sentinels are not measurements.
- A Newton-direction true-residual certificate does not certify the Cauchy fallback; record each fallback Armijo result and rely on the final transaction certificate for a committed step.
- The coupled rich capture is explicitly skipped for this two-environment run; no coupled operator audit is claimed.
- The source137 runtime messages contain stale trace-cap labels; selected all scope is 512 as encoded in frozen source.
- Under selector :all, a Cauchy-labeled second-pass event is not necessarily an executed fallback. Only environments whose primary Newton Armijo exhausted with metrics.w set to 3 invoke the Cauchy builder; otherwise the second pass is gated and repeats primary metrics. This run had two actual Cauchy environment trials (Newton 48 and 53, env0); all remaining Cauchy-labeled environment rows are duplicate telemetry.
- The probe option --cache-fem-algorithmic-tangent was explicitly enabled in both native435 and native442. Source139 keeps the authored default false and allocates the cache only for selector word 1; these runs therefore do not measure default-off runtime behavior.
