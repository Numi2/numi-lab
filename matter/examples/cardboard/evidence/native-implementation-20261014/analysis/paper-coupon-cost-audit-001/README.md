# Paper coupon material-cost audit and projection-reuse prototype

Status: source-bound callgraph audit is complete. `projection-reuse.patch` is an external prototype against immutable source123; it has **not** been applied to the shipping source, compiled, or run on Metal. It makes no speed or solver-qualification claim.

## Source and evidence binding

The audit source is `/Users/home/cardboard-implementation-20261014/snapshots/source-two-stage-material-merit-123`. The source-build record SHA-256 is `fc96e2521b363c821132d7a493da091aa7d36d1c125951ffa09e2d38e91d0952`. Relevant source hashes are recorded in `manifest.json`; the prototype modifies only `matter/src/metal/common.metalinc` and `matter/src/metal/fem.metalinc`.

Native evidence is from:

- `solver/two-stage-certified-coupon-409`: source-build hash above; receipt `c769db491e80efcc5c415bdd3277174c33437665fee1c7985825ef790ba3ee7e`; result `6600e9b15bba128d2ef4ed2c5c4a181e91fdc8600a4171ecdd7e1c2eb04efcba`; observations `e2d7d7c09d87aa6989f4923691a57b2e99c6db7e6babaacb585cad8ace138d17`; stderr `7fa9d24eec8c3a66aa7cb2bdb3962cac7eb669b6de7055078dbf578b10bd4bcb`.
- `native/two-stage-controlled-material-sweep-410`: same source-build hash; receipt `f7b385ca9f15f0b5993c5887d78aedc31faaf3141893944dcb466122adced5cf`; accurate-policy output `56de51013ea87338b8410a93d7ca078504dfb9d4d2db9553e312bbca96331c58`.

Run409 compiled an 882-node, 1,920-tet explicit coupon with 1,152 liner, 576 medium, and 192 starch tets. It accepted the first bent control step; its observation rows reported 22 FGMRES iterations for the bent environment and 4 for the quiet reference. It then rejected control step 1, ending with `accepted_steps=1`, `status=failed`, `physical_validation=false`, exit 2, and a 676.329 s runner wall time. The source log's command-buffer timing measured control 0 at 1.01519 s encoding and 153.327 s GPU; control 1 at 0.756079 s encoding and 280.228 s GPU before the failed run ended. These are whole prescribed FEM transactions, not per-kernel timings.

Run410 emitted 1,536 selected-point samples (24 center repeats, 72 basis actions, 1,440 perturbed basis samples); all reported projection, stress, and tangent valid. It took 0.906242 s wall time, performed zero physical steps, and declared `physical_validation=false`. This is not a matched production coupon benchmark and cannot be divided into a per-tet or per-kernel cost.

## Callgraph finding

For a stateful generic material, `evaluateAlgorithmicStress` calls `projectMaterialState` once and then evaluates stress from that candidate. The private Armijo path dispatches `nm_fem_internal_forces` for each distinct trial candidate (`runtime.mm` around 11995; `fem.metalinc` 982–1047). It uses the stress-only evaluator and does **not** construct a 9-column tangent. Each Armijo alpha changes the candidate deformation/rate, so its material projection must remain fresh against the same accepted history.

The once-per-Newton `nm_fem_build_algorithmic_tangent_cache` kernel (`fem.metalinc` 2882–3000; runtime dispatch around 18798) loops the nine row-major unit directions. Each call to `evaluateAlgorithmicStressTangent` currently calls `projectMaterialState` with the same accepted state, high/low deformation, deformation rate, timestep, temperature, material, parameters, and a zero projection direction. The projection function has no tangent-direction input. Thus the projection candidate is invariant across the nine columns within this one tet invocation. The nine tangent evaluations and their direction-dependent derivative solves remain necessary.

For the source123 1,152+576 stateful paper tets and two compiled environments, the manifest implies up to 3,456 tet-environment cache entries. If all are due in a Newton build, current code makes 31,104 local projections for nine columns; the prototype makes 3,456 (one candidate per entry). This is a **modeled call-count reduction**, not a measured GPU speedup; scheduler eligibility and runtime cost distribution must be verified on-device.

The existing GPU timing helper (`metal_encoder_timing.hpp`) opts in only for `root == 100` and times the whole encoder. It does not give dispatch-level timings for this coupon's control roots 0/1, nor isolate the cache-builder dispatch from other work. `NUMI_CARDBOARD_STEP_TIMING=1` similarly reports whole encode/commit/GPU command-buffer time. These logs cannot identify how much of the 153–280 s was projection, tangent, Krylov, Armijo, or other runtime work.

## Prototype behavior

The patch adds an optional projected-state pair to `evaluateAlgorithmicStressTangent`. The tangent-cache kernel enables it only for a generic, compensated, scaled-correction 15-state material. Column zero follows the existing path, projects once, and returns the exact high/low candidate into two local arrays. Columns one through eight copy that pair into their normal evaluator candidate storage and skip only the repeated projection. The stress program, all nine direction-dependent tangent programs, implicit Jacobian/solve, low-word deformation inputs, validity checks, cache writes, and valid-entry latch remain in place.

There is no persistent cache arena, no cross-tet/environment/transaction reuse, and therefore no stale-generation key to maintain. The candidate lifetime is one thread's one tet's nine-column loop. The added caller arrays reserve 30 floats for eligible and ineligible tet invocations unless the Metal compiler proves the condition away; register/local-memory pressure and occupancy are unknown until compiled/measured. Invalid or nonfinite reused values fail the tangent-cache entry. Diagnostics that request projection iteration/residual details are rejected on the reuse path rather than fabricated. All other policies retain their old evaluator path.

Apply only to a scratch checkout of the exact source123 base:

```sh
cd /Users/home/cardboard-implementation-20261014/snapshots/source-two-stage-material-merit-123
patch -p1 < /Users/home/cardboard-implementation-20261014/analysis/paper-coupon-cost-audit-001/projection-reuse.patch
```

`verify_package.py` verifies base-file hashes, dry-applies the patch to a temporary copy, and checks the expected output hashes and untouched call sites. It is a source-shape check, not a compiler check.

## Required follow-up before adoption

1. Compile the exact patch in a fresh source-bound build; record Metal compiler diagnostics and register/threadgroup resource usage if available.
2. On identical captured source inputs, compare old and new 9x9 element tangent-cache matrices and valid/status data for liner and medium across elastic, yielded, unloading, rotated-frame, nonzero-rate cases. Bind exact F high/low words, Fdot, accepted 15-state high/low history, dt, parameters, material assets, frame, and build hashes. Require exact equality if operation ordering permits; otherwise report componentwise absolute and relative errors with an independently established floor. Do not infer parity from run-level convergence alone.
3. Compare trace-on/off cache input-output parity and transaction accepted state, residual/correction certificates, Armijo decisions, and rollback behavior on a fixed source-bound coupon trajectory.
4. Add opt-in timestamps around just the cache-builder dispatch and a representative private `nm_fem_internal_forces` dispatch, selected by source-bound step/Newton and capped to a small number of samples. Report unsupported counter sampling as unavailable; keep whole-encoder timings separate.
5. Benchmark matched warm builds and identical inputs; report projection count, cache-build GPU time, private trial GPU time, whole command time, and resource usage separately. The existing 409 failure remains a failure; no physical validation follows from this optimization.

Do not reuse projected state across Armijo alphas, accepted Newton states, different rate/dt values, changed material history, or another tet. Do not reuse the implicit Jacobian/factor or skip direction-dependent tangent programs in this prototype.
