# Native American-crow object-push development

The final v14 engineering fixture is wired into the native compiler, C API,
Swift rollout/trainer and `tools/numi crow-env`. Canonical assets are in
`environments/american-crow/object-push-v1`. No learned manipulation, calibrated
material model, full lifelong ecology, social learning or sustained production
performance is claimed. Existing v13 deployment97 remains the cued-foraging
candidate; it is not rebound to this new task.

## Final controls

`object-push-native-11` passes7296 accepted transitions, using8 identical lanes
and a scripted native head-yaw pulse. Opposite, neutral and contact-disabled arms
miss without horizontal object displacement. The positive arm moves a resting
2g/8mm sphere by solved head/beak contacts toward the original lateral target,
ending5.54mm away. The object stays on its support, initially occupies4200 pixels
across8lanes x2eyes, and has no actor object-truth channel. Impulse samples are
from the last physics substep per control tick, not the total impulse.

The goal reward agrees with independent FP64 accepted-position evaluation.
Full world/retina/q/v/scene and actor/critic/transition hashes are exact across
chunks1/8, and after tick48 world+retina restoration into fresh native contexts.
This is a native checkpoint control, not ordinary v14 coupled-life support.
All18 native controls remain available; actor1247 includes93 proprioception/history
and1154 delayed RGB/validity/age features. Critic100 adds7 privileged relative
object channels. Normal crow/support contacts remain enabled.

The explicit32-substep hybrid-CCD profile retains50Hz control/15Hz retina.
The ordinary C API rejects a4-substep request; the CLI also rejects unsupported
v14 whole-life checkpoints before simulation. No implicit quality override.

## Physics refinement and failed attempts

Resting depth error at4/8/16/32 substeps is2.217/.562/.148/.0444mm. Compiled solver
slop was10micrometers throughout. The earlier attribution to2mm slop was wrong;
changing robot-level slop did not change run-owned compiled physics. That failed
hypothesis and its unchanged result are retained.

Push endpoints at16/32/64 substeps are6.43/5.54/10.35mm from the final lateral
engineering target. Maximum path differences are1.80mm (16vs32) and4.86mm
(32vs64). This is not monotonic trajectory convergence or material calibration.
The4-substep trajectory moved far forward;8 substeps rejected a CCD step.
The target was briefly moved to the coarse endpoint during development, then
returned to the original lateral apparatus target after refinement. The failed
coarse and fine goal checks are retained; none is held-out qualification.

Native attempts01/02 fail schema/policy-authority admission. Attempt03 has beak
candidate contacts but zero solved impulse/motion at the higher support.04 moves
the object but lacks initial camera coverage and has stale inactive checkpoint
tails.05 passes the then-speculative fixture, which did not match ordinary API
hybrid CCD.06 exposes hybrid CCD event-budget exhaustion.07 fixes that failure
but the sphere leaves the narrow support.08/09 expose excess coarse-step resting
depth; the attempted slop change is ineffective.10 uses32 substeps and passes
contact/partition/continuation but fails the forward coarse-goal progress gate.
11 passes the final bounded fixture. Build and ordinary-learning initialization
failures are retained, including metric visual-sphere scaling, CCD and Swift
argument errors. Learning03 uses the temporary forward target; learning04 and
throughput01 use the final lateral target. No negative result is overwritten.

## Owning fixes and regressions

Sphere visual tessellation is generated from the analytic native collider;
existing box/plane bytes are preserved. Retinal packs use metric geometry and
avoid legacy tracked-unit-sphere scaling. Hybrid near-contact CCD uses the
existing complete-sweep speculative-envelope coverage certificate, retaining
literal CCD/budgets for uncovered/distant impacts. Checkpoint readback alone
zeroes inactive manifold tails; resident GPU state and all live records remain
untouched.

The full Release integration build passes. Nine focused native crow regressions
pass after the shader change: embodiment, finite resource, head camera, retinal
timing, life checkpoint, recurrent rollout, retinal API/live and temporal history.
The generic native contact probe also passes, including high-speed mesh/convex
CCD, large pair tails, overflow isolation and its throughput release gate. These
bounded controls do not qualify all physics scenes.

## Ordinary learning and performance

Fresh8-lane recurrent training accepts2560 samples across2updates, fixed LR.0003,
gamma.999/lambda1,160-tick sequences,2full-batch optimizer steps/update,GRU32 and
encoder32. All18 final factor KL/counts and effective actor-update metrics are
finite, counts exact, actual actor displacement positive; no physics rejection.
Native generating-policy/MLX replay parity is enforced by the ordinary worker.
This is engineering admission, not learned object-push behavior.

A separate predeclared128-lane two-update trial accepts40960 samples with zero
reported physics rejection, in55.7515s at734.689 transitions/s. MLX peak is
579605392B and swap remains0. The8-lane trial reaches70.9633 transitions/s.
The physical profile/RGB remain intact. Batch and full-batch learner sizes differ,
so no pure optimizer speedup or sustained production qualification is inferred.
MLX peak excludes other native/device/system allocations. No adaptive-rate search.

The user's retained-rollout optimizer investigation is separately archived at
`artifacts/crow-rl-optimizer-audit-20261108`: final head/beak KL is now measured
after the last step; actual audited gamma.999/lambda1 retains head reward credit;
critic-dominated clipping changes actor updates at97 but did not affect65/81.
No learning-rate success was selected from KL. Held-out Gaussian intake26→37/192
remains statistically inconclusive despite improved correct-side choice.

## Git recovery and disk discipline

The content-deduplicated manifest retains every selected raw trial/failure,
source, exact binary/library/shader and learner/checkpoint output. Existing Git
chunks are reused without recreating their checkout. Canonical product assets
remain present. The recovery helpers restore from Git objects, isolate the
actually loaded library, rerun the full native controls and the128-lane ordinary
learner, and compare policy/rollout bytes, all model/Adam tensors/metadata and
scientific metrics. Measured update time and process-dependent allocator peaks are recorded separately
and excluded from causal learner equality. The initial overstrict allocator
equality failure is preserved in recovery-attempt-01.
Path-bound visual identities must be explained by native owning fingerprint
functions; all remaining fields/files are exact.

Raw/duplicate removal is permitted only after pushed Git recovery succeeds and
hash/mode/mtime/closed-handle checks pass. Recovery and cleanup receipts carry
the observed results. No shared Git provider cleanup or unrelated deletion.

Git recovery passes:74 native files byte-exact,2 path-bound visual manifests
explained by native owning fingerprints,9 policy/rollout files byte-exact, and
123 model/Adam arrays across3 learner files with exact metadata/scientific metrics.
The corrected check reuses the same successful native/learner executions; it
does not rerun or select a new training result.269199820 temporary bytes are
removed after hash/mode/mtime/closed-handle validation.
