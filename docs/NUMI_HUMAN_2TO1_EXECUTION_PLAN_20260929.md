# Human and Brain 2:1: research, progress and next execution plan

2026-09-29, Europe/Oslo. Work paused at the user's request after documenting this plan. **The target remains unmet: one Human must advance 10 simulated seconds in at most 5 wall seconds.** Feasibility on the current M4 Pro has not been demonstrated.

This document sets the next execution order. The [implementation history](NUMI_HUMAN_2TO1_PLAN_20260928.md) and [architecture analysis](NUMI_HUMAN_2TO1_ARCHITECTURE_SLICES_20260928.md) retain earlier measurements, unsuccessful experiments and qualification boundaries.

## Verified progress at pause

- Canonical source at inspection: `c170bbfb5ddbdb4888eb021fb07c734ea2f2dd92`, clean `main`, matching GitHub `origin/main` before this documentation change.
- Two retained complete, uninstrumented full-feedback launches took **152.982522209 s and 174.782947667 s**. Mean **163.882734938 s**; required improvement to 5 s **32.7765x**. The 14.25% spread has no established cause.
- The subsequent complete counter run took **153.382876125 s**, with 10,000 accepted physical rows, 10,000 accepted receptor rows, all 64 sweeps and zero assistance. Re-reading its retained native log reproduced the exact physical, receptor and terminal witnesses below. This was instrumented; it is not an additional uninstrumented latency sample.
- Its sampled finish median, excluding initialization, was **4.557438 ms/root**. Samples span roots 100, 1,000, 5,000 and 9,999. Each root reported 1,624–2,112 nonzero limit updates and 26,427–32,001 limit previews. These counts identify repetition; they do not measure instruction cost or memory stalls.
- The preview-bound cache executed and replayed the 128-root target exactly, but its sampled finish increased from **4.160083 to 4.304458 ms**. It remains rejected for performance. Earlier register variants also regressed. Their sources, binaries and evidence remain retained.
- A Metal timeline was captured. The requested detailed counter profile was unsupported in that attempt. Occupancy, spills and the cause of timing spread remain unmeasured.
- This research turn changed documentation only. The response-arena worktree `/Users/home/numi-human-response-arena-20260929` is clean at `c170bbfb`; its new cache has **not** been implemented, built or measured.

Complete counter-run witnesses:

| Stream | Rows | SHA-256 of canonical newline-delimited rows |
| --- | ---: | --- |
| Physical acceptance | 10,000 | `9f273366f5be7f15074104d3e35ea5bdd6f4b1daa950ab6a4a85fa3a77840230` |
| Receptor acceptance | 10,000 | `dea82d088111927c9e9ada6cbf8af71e060a42f27f4bf7ff3517155897154209` |
| Terminal state | 1 | `f9d3b429c92254fb77a6400f451f610c381ac500cc640bbbc3f117001f567066` |

Published receipts: [full finish counts](evidence/numi-human-finish-counters-20260928.json), [rejected preview cache](evidence/numi-human-limit-preview-cache-20260929.json), [full-feedback reference](evidence/numi-human-2to1-finish-research-20260928.json).

## What reaching five seconds requires

The absolute limit is 500 microseconds per physical root before launch costs. Reserve 1.0 s for startup/teardown and 0.5 s of margin, leaving **350 microseconds/root** over 10,000 roots. The measured finish alone is about 13 times that entire allocation. This requires coordinated changes to mechanics, Brain, geometry and scheduling.

| Complete owner | Final allocation | First implementation gate |
| --- | ---: | ---: |
| Geometry, authored routes, muscle/tendon dynamics | 70 us/root | <400 us/root |
| Complete mechanics, including all 64 sweeps | 100 us/root | <1 ms/root |
| Every scheduled Brain decision and consequence | 130 us/root | <1 ms/root |
| Receptors, joint admission, accepted journal | 30 us/root | Measure actual combined cost |
| Amortized host encoding/submission | 20 us/root | Measure before and after chunks |
| Horizon total | 350 us/root | Complete launch <=30 s |

These are engineering allocations, not predicted speedups. Stage samples from different roots, binaries or feedback programs must not be added into a purported exact wall-time breakdown. Intermediate whole-launch gates are **30 s, then 10 s, then 5 s**.

## Fixed workload and physical contract

Use one 157-body, 128-DOF Human with 416 muscle channels, the selected frozen Brain runtime and v6 program with 104 balance routes, unchanged joint/path feedback and seven receptor streams. Apply the existing +50 N pulse at root 100 for 20 roots. Retain 10,000 one-millisecond physical roots, every scheduled neural tick/event, all 64 ordered coupled sweeps, the complete effective operator, passive cross-couplings, tendon/muscle dynamics and native admission tolerances.

The program SHA-256 is `2615b7a1f8168038b98ae93314721235a3ad0fc544c47bf46b97aeefb4d9990a`; the retained Brain library is `dfa3b6cec6a5a65748233a2c3ab5b224deb401d9e969f46e89cce20e1056b650`. The Brain seed is `1314213193`. Bind source, runtime resources, inputs and seed before and after each candidate. Preserve zero root assistance and muscle-driven actuation.

## Next development slices

### 1. Establish the critical path and settle the solver representation

Use the existing complete counts and timeline first. Add narrowly selected same-root timing only where Brain, admission or host attribution is missing; include late states. Try an Xcode GPU capture for instruction costs once. If detailed profiling remains unavailable, use controlled native A/B measurements and compiler resource reports without claiming a hardware cause.

The immediate bounded solver experiment is a same-root response-column cache using the existing 16-KiB equality-factor arena. For 128 DOFs it fits 32 FP32 columns. Measure distinct columns, hits, misses and capacity fallback. Keep coefficient values and response FMA order unchanged.

**New lifetime finding:** `solveBilateral()` runs before the sweeps when `useProjectedContacts` is true, but runs inside the sweeps when it is false. Arena reuse must require projected contacts, the native non-CPU finish and the supported SIMD shape. Overwriting the factor in the other modes would corrupt subsequent equality solves. The map resets every physical root; all uncached responses retain the original device path.

Run the owning native check, then a same-binary off/on 128-root comparison with the actual cache path and target witnesses. Advance only if the intended stage improves in repeat comparisons. A failed experiment ends at that gate; another full horizon is unnecessary.

The main architectural solver slice follows: build **constraint velocities `y = Jv` and same-root response coefficients `C = JR`** from the existing complete, equality-conditioned response operator. Preserve the authored decision order and 64 sweeps; update constraint velocities after each impulse rather than repeatedly contracting contact axes against the full velocity vector. Preserve equality/source-limit work and materialize the final full velocity through the native response map.

This is a numerical implementation change. Require FP64 response/error comparison, unchanged native contact/equality/limit admission and deterministic candidate replay across the complete horizon. A different floating-point trace from the former implementation needs explicit numerical qualification.

The potential 158-row dense FP32 coefficient matrix takes 99,856 bytes, above the 32-KiB threadgroup limit. Prepare independent columns/tiles across GPU cores and use bounded device storage with a compact sweep working set. Retain all structurally possible couplings. The ordered impulse decision chain remains sequential; adding GPU lanes cannot make dependent decisions independent.

As a bounded placement comparison, measure the **existing native CPU finish** under the exact same full-feedback target, including its handoff/wait costs and all physical gates. The older 86.46-second hybrid result used a different workload and does not establish an advantage here. This comparison determines whether a small sequential phase benefits from the Apple CPU; it is not evidence that the CPU path meets five seconds.

### 2. Compile and fuse the complete Brain schedule

Refresh the frozen-runtime phase ranking under full feedback and late accepted states. The retained short attribution ranks regional acceptance, cognitive transition, cerebellar policy, prediction, decision memory and motor-ready work highly; it is not a complete current critical-path decomposition.

Compile immutable route and consumer incidence once. Replace repeated searches and state materialization with consumer-oriented fields. Fuse measured producers/consumers while preserving committed-versus-staged reads, every neural cadence, event deadline, protective/affect consequence and random stream. Use compact active indices only where all scheduled decay/background work and future activation semantics remain equivalent.

Preserve required hash byte order and identity. Reusing immutable metadata is valid; reusing a state-dependent digest across roots is not. Evaluate SIMD matrix/MPP operations only for measured dense shapes with a qualified precision contract. M4 is Apple9; Metal 4 API support does not grant Apple10 neural hardware.

### 3. Compile shared geometry and muscle traversal

First profile the existing topology-cache/local-motion/kinematics selectors that are disabled in the selected baseline. Reuse a qualified owner implementation where it improves the target.

Compile ancestry, dependency levels and authored body/site/muscle incidence. Evaluate current transforms once and share them with the full operator, forces and receptors. Parallelize independent bodies, sites and muscles. Remove a dense Jacobian only after every consumer has an equivalent compact representation. Preserve compensated coordinate semantics, source path length/velocity, wrapping, fibre/tendon dynamics and force closure.

### 4. Execute causally ordered GPU chunks

The current app explicitly sets `segmentSteps = 1` when the Brain controller is present. Completion hashes the accepted physical state on the host, validates receptors, submits an accepted-consequence command and waits before publishing the next neural generation. This is an ownership dependency requiring implementation, not just deletion of a wait.

Move prerequisite fingerprint calculation, receptor validity, admission status and generation publication into the native GPU schedule. Preserve digest values/order and joint ownership. Start with 16 roots per submission, then 32. Within a chunk, root `r+1` consumes only the accepted Human/receptor/Brain state of root `r`. A failed root stops dependent work and retains the exact accepted prefix and committed state.

Require chunk-size replay, correct event/random clocks and an executed joint failure/restore check before promotion. Publish compact per-root journals; format their same values after completion, with output included in total timing. Freeze immutable model resources through an explicit compiled identity so structural metadata can be retained without rebuilding/copying it at every root.

Begin with the existing command model. Compare Metal 4 allocators, argument tables and barriers if host encoding remains a measured cost. Compile/source-bound pipeline archives and immutable startup assets, then measure startup rather than moving required work outside the clock.

## Experiment and completion rules

For each slice: one measured bottleneck, one implementation, its owning native numerical/physical check, then repeated complete uninstrumented launches if the stage gate passes. Preserve failed candidates. Judge the integrated change in wall seconds; do not multiply isolated improvements into a forecast.

Final proof requires **five complete launches <=5 s each**, including startup, teardown and required journal output, with 10,000 accepted physical and receptor roots, all 64 sweeps, unchanged Brain clocks, zero assistance, exact candidate replay, source/runtime/program/seed identity, bounded memory and no increasing swap. Check GPU errors and thermal state. Physical/receptor replay does not by itself establish full neural checkpoint identity, joint rollback, energy closure, physiological calibration or general perturbation recovery.

After improving single-Human latency, batch 8 then 16 independent Humans for learning throughput. Measure accepted simulated seconds/hour, replay per environment, failures, native/device memory and swap. Keep single-run latency separately measured. SSH is for development and evidence coordination between the Book and Mini; dependent roots remain on one owner.

## Resources and restart handoff

At the closing inspection, free storage was approximately **43.3 GiB on the Book and 38.7 GiB on the Mini**; Mini swap was 149.19 MiB. No Human probe, build or xctrace job was active on the Mini. Keep a measured 12-GiB free-storage floor and one timing workload on its GPU. Retain sole evidence and unrelated dirty work, including the Brain development checkout.

Evidence root on both hosts: `NumiHumanBrainRuns/2to1-finish-20260928`. Book handoff: `CONTINUE-2TO1-20260928.md`; complete log: `mini-evidence/finish-counters-mini-full-10000/native.log`. The new response-arena worktree contains no implementation yet. Resume with its guarded native cache experiment and the missing critical-path attribution; advance Brain/scheduling architecture as independent source work without competing Mini GPU jobs.

## Primary research and its application

- [Apple: Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/) explains work distribution, CPU/GPU timeline gaps and synchronization costs. It supports distributing independent preparation and compiling causal submissions; it predicts no Numi speedup.
- [Apple: Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/) explains resource limits and tuning. Cache lifetime and measured working sets determine this implementation.
- [Apple: profiling tools for M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111374/) describes shader cost graphs and execution history. The retained unsupported counter attempt does not provide that evidence.
- [Apple: Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/) describes allocators and resource binding. API migration follows measured encoding cost.
- [Apple feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) identify M4 as Apple9 and its threadgroup limits.
- [Featherstone: branch-induced sparsity](https://royfeatherstone.org/papers/sparse.pdf) motivates compiled structural dependencies. This Human's additional passive cross-couplings remain part of the full operator.
- [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) supplies constraint-space and PGS background. Applying `C = JR` to Numi's own response operator is a design inference, requiring native qualification; MuJoCo's soft-contact model and timing are not imported.
