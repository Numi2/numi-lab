# Numi Human: architecture work toward 2:1 execution

2026-09-28. Research and implementation plan; five-second feasibility on the current M4 Pro remains unproven. See the [measured implementation history](NUMI_HUMAN_2TO1_PLAN_20260928.md).

## The decision

Treat five seconds as a whole-program latency target. Two complete launches of the selected full balance-feedback workload took 152.9825 and 174.7829 seconds for ten simulated seconds, with exact candidate replay. Their mean is 163.8827 seconds: another 32.78x is needed at that mean. The second was 14.25% slower; the cause has not been established. Recent sparse factor and coefficient caching work saved seconds, while both register-distributed sweep prototypes regressed. Further thread-count or storage-layout variants need a measured hardware cause before proceeding.

Preserve one Human, 10,000 one-millisecond physical roots, all 64 ordered coupled sweeps, the complete effective operator and every scheduled neural update. Compile static topology and schedules; recompute state-dependent values at their existing clocks. The experiment must still demonstrate native physical acceptance, source muscle control, receptor delivery and zero assistance.

The target program is the frozen v6 source program with 104 balance routes, existing joint/path feedback, seven receptor streams and the selected Brain runtime. Apply the existing +50 N pulse at root 100 for 20 roots. This defines the performance workload; it does not establish general perturbation recovery. The older 86.46-second hybrid run used a different feedback/audit scope and cannot serve as its matched control.

## Five-second allocation

The absolute ceiling is 500 us per root before launch cost. Reserve 1.0 s for startup/teardown and 0.5 s of margin, leaving 3.5 s, or 350 us per root. These allocations are requirements, not forecasts.

| Complete owner | Final allocation | Earlier delivery gate |
| --- | ---: | ---: |
| Geometry, source muscle routes and tendon/muscle dynamics | 70 us/root | 400 us/root |
| Mechanics, including all 64 sweeps and admission | 100 us/root | 1,000 us/root |
| All scheduled Brain decision and consequence work | 130 us/root | 1,000 us/root |
| Receptor delivery, joint admission and accepted journal | 30 us/root | Measure before fusion |
| Amortized host encoding/submission | 20 us/root | Measure before chunking |
| Total horizon allocation | 350 us/root | Whole launch <=30 s |

The currently sampled 4.18-ms finish exceeds the entire final root allocation by almost 12x. The v3 and v4 register variants sampled 5.62 and 5.92 ms after excluding root 0 from samples at roots 0--7, and remain rejected early-root performance experiments. Geometry previously sampled at 1.58 ms and muscle dynamics at 0.49 ms also exceed it. Brain motor and consequences previously sampled around 2.35 and 1.70 ms. These are attribution samples from different captures, not an additive decomposition of the new full-feedback horizon.

## Ordered development slices

| Order | Slice and owning files | Concrete deliverable | Advance or stop decision |
| --- | --- | --- | --- |
| 0 | Matched profile: `apps/numilab_human_myosim_visual_probe.mm`, `apps/NumiHumanBrainController.hpp`, Brain standing bridge | Baseline pair completed; resolve the timing spread with before/during/after load and thermal observations, a correlated CPU/GPU timeline, finish instruction attribution, and early/pulse/middle/late accepted-root samples | Identify actual repeated work and synchronization. Do not infer spills, thermal throttling or bandwidth limits from elapsed time alone. |
| 1 | Ordered finish: `src/metal/NumiHumanStandSolve.metalinc`, `include/metalrobo/numi_human_friction.h` | Counts of nonzero contact/limit impulses, interval previews, interior/boundary friction decisions and boundary iterations; then one measured replacement of the dominant repeated computation | Preserve all 64 sweeps and source order; require an actual stage gain and native admission before a complete-horizon comparison. |
| 2 | Brain graph: `MetalDecisionRuntime.swift`, `MetalEmbodiedBrainRuntime.swift`, `MetalTissueRuntime.swift` and their owning shaders | Compact consumer indices and versioned committed/staged state; remove measured full-capacity copies/scans and fuse adjacent producers/consumers | Complete scheduled Brain below 1 ms/root first. Every event, protective/affect consequence and transaction remains represented. |
| 3 | Geometry: `src/metal/ArticulatedOperator.metal` and source joint/route preparation | Compile transform dependency levels and authored incidence; evaluate each state-dependent joint/transform once and share it across native consumers | Complete geometry/routes/dynamics below 400 us/root first; preserve compensated positions, route velocity, angular Jacobians, forces and passive terms. |
| 4 | Root scheduling: Human controller, receptor owner, Brain standing bridge and `MetalArticulatedOperator.mm` | Device-resident joint state; bounded 16-root then 32-root schedules with per-root admission/status and accepted journal | Exact candidate replay across chunk sizes, actual failure/restore, fixed neural clocks and no rejected root influencing later published roots. |
| 5 | Launch path and integration | Source/device/OS-bound pipeline archives, immutable startup assets, compact deferred journal formatting, repeated complete launches | Advance through <=30 s, <=10 s and <=5 s; include startup, teardown and the entire target journal. |

Slices 1--3 can be developed independently by the team against the same frozen target. Integrate each measured gain separately. Use the Book for source work, native owner checks and preparation; reserve the Mini for one GPU timing child at a time. Single-run latency remains the acceptance metric for this objective.

### Slice 1: remove arithmetic repetition from the 64-sweep owner

First measure the split between contact contractions, source-limit previews/responses, equality bookkeeping and friction-boundary work. The native friction helper currently permits up to 40 bisection iterations when the interior solution is outside its disk. This is a source-level opportunity; its frequency and time contribution have not been measured. If boundary work dominates, compare a safeguarded root solve that preserves the feasible endpoint and the native friction metric, with residual/error checks against the owning reference. Never substitute extra compliance for the boundary multiplier.

If repeated contractions dominate, prototype a constraint-space representation inside the existing solver:

1. Start from the existing complete effective operator and same-root equality-conditioned response columns. Retain all off-diagonal passive couplings, regularization and pivot checks.
2. Form constraint velocity `y = J v` and response coefficients `C = J R`, where `R` is the actual native response map, not an inverse of a simplified mass matrix.
3. Keep each ordered projection decision and all 64 sweeps. Update the affected constraint velocities with each ordered impulse, instead of repeatedly contracting every contact row against all 128 DOFs.
4. Materialize the complete final velocity through the original response map and preserve equality/source-limit impulse work and the existing position/integration/admission owner.

This changes floating-point expression order and needs an explicit numerical qualification. Require FP64 response/error evidence, actual Human contact/equality/limit residuals, full native physical gates and exact replay of the new candidate. Do not require an unsupported claim that its trace must match the old floating-point implementation.

Capacity matters: ten contact bindings provide up to 30 contact axes; adding 128 potential limit rows yields 158 rows. A dense 158x158 FP32 coefficient matrix alone takes 99,856 bytes, exceeding the 32-KiB threadgroup arena. Measure actual dimensions and coupling structure. Tile native response preparation across independent GPU work; keep sweep working sets compact and use device storage where needed. Do not create another large threadgroup cache by default or drop a coefficient because it was zero in one captured pose.

The ordered decision chain remains sequential. The independent response preparation is parallel work. A single SIMD32 group for an entire root does not fill a 16-core GPU, and two register variants already lost performance. Measure critical-path latency, active lanes, instruction costs and traffic rather than equating more register ownership with more speed.

### Slice 2: compile the Brain's existing schedule and state traffic

The selected frozen Brain runtime's retained 112-root attribution ranked regional acceptance (about 309 us), cognitive transition (304 us), cerebellar policy (243 us), prediction (219 us), decision memory (194 us) and motor-ready work (165 us) among its largest passes. Refresh this ranking under the selected full-feedback program and late accepted states before choosing a pass. It is not a prediction of total Brain time.

For the dominant passes, compile immutable route/consumer maps, process active authored rows without omitting scheduled work, and store committed/staged fields in layouts suited to their consumers. Replace a full-capacity copy only with a versioned representation that preserves old committed reads until publication. Preserve deterministic hash order where the hash contract requires it; parallel reductions need their own numerical/replay contract.

Retain every neural tick, event deadline, emergency capacity, protective response, affect consequence and random stream. Cached structural metadata is valid across roots; values depending on the current state are not. Dense operations are candidates for SIMD matrix/MPP only after measured shape and precision qualification. Apple's feature table identifies M4 as Apple9; do not assume Apple10 neural-accelerator hardware from Metal 4 API availability.

### Slice 3: share geometry with all native force consumers

The selected timing baseline leaves the existing topology-cache, local-motion and specialized-kinematics selectors disabled. Inspect and profile those existing implementations before adding another path, then qualify any useful combination on the full target. A retained ancestry-only prefix does not qualify a full geometry replacement.

Compile the body/joint dependency DAG and authored coordinate/route incidence once. At each root, evaluate current joint functions and transforms once, then parallelize independent bodies, sites and muscles. Share resulting geometry with the full effective operator, force reduction and receptor consumers. Remove a dense materialization only when every consumer can read an equivalent compact representation.

Keep compensated coordinate semantics, source path lengths/velocities, fibre/tendon dynamics, wrapping, angular Jacobians and force closure. This work concerns the actual Human geometry owner, not replacement mechanics or fewer muscles.

### Slice 4: remove per-root host round trips without breaking causality

Current source explicitly forces one-step segments when a Brain controller is present. The controller computes the accepted physical fingerprint on the host, waits for a follow-up consequence command and publishes the neural/receptor generation before the next root. This is a concrete scheduling dependency, not a removable wait in isolation.

Move the necessary fingerprint, receptor validity checks, root status and generation state onto the owning GPU schedule. Encode the existing operations in 16-root chunks, then 32-root chunks. Within a chunk, root `r+1` consumes only the accepted Human/receptor/Brain state of root `r`; roots cannot execute independently in time. A failure must stop dependent work, preserve the exact accepted prefix and support the real joint rollback contract. Double buffering alone does not establish this.

Reuse resource bindings and compiled schedules; publish a compact per-root accepted journal, then format its same values after chunk completion. Include formatting/output in end-to-end timing. Start with the current Metal command model; compare Metal 4 command allocators/argument tables only when host encoding remains measurable. An API migration by itself is not the 30.6x solution.

## Integration and final proof

Use a finite experiment loop: one measured bottleneck, one candidate, owning native precision/physical check, then a matched complete launch. Retain failed experiments. Stop a variant that does not improve its intended stage before spending ten-second runs on it. Report the change in whole-program wall time; do not multiply isolated gains into a forecast.

For <=30 s and <=10 s gates, allocate the remaining latency from correlated measurements again. The five-second gate requires five complete launches at or below five wall seconds, 10,000 admitted physical roots and receptor deliveries, all 64 sweeps, unchanged neural clocks, zero assistance, source/binary/program/seed identity, exact candidate replay, bounded process/device memory and no growing swap. Record GPU errors and thermal state. Full neural checkpoint identity, joint failure/restore and energy closure remain separate executed gates.

After the single-run owner improves, batch 8 and 16 independent Humans to improve learning throughput. Report accepted simulated seconds/hour, replay per environment, retained/peak device and process memory, failures and swap. An eight-environment throughput gain does not establish one Human finishing in five seconds. SSH connects the two development hosts; distributing dependent roots across their network is not the proposed latency path.

Keep at least the existing 12-GiB free-storage launch/build floor. Measured free space before evidence mirroring was approximately 45.7 GiB on the Book and 39.4 GiB on the Mini. Retain sole evidence and rejected sources/binaries, and measure `df` before new long captures. RSS alone is not total process footprint or GPU residency.

## Research grounding

- [Apple: Metal compute, submission, memory and occupancy](https://developer.apple.com/videos/play/tech-talks/10580/) supports batching useful work, reducing unnecessary copies and diagnosing register/threadgroup resource limits. These mechanisms motivate measurement; they do not establish this solver's bottleneck or predict its speedup.
- [Apple: finding GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy) distinguishes small work grids from exhausted resources. [Shader profiling](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance) is needed for instruction/spill attribution; those hardware counters have not yet been collected here.
- [Apple: Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api) describes command allocators, argument tables and explicit synchronization. Use these where the measured host/submission cost warrants them.
- [Apple: binary archives](https://developer.apple.com/videos/play/wwdc2020/10615/) supports source/device/OS-compatible pipeline caching to reduce launch compilation. Measure archive misses and cold versus cached startup explicitly.
- [Apple feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) binds the plan to M4/Apple9 capabilities. [Inline Metal ML](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) does not authorize assuming newer neural hardware or reducing the precision contract.
- [Featherstone: branch-induced sparsity](https://royfeatherstone.org/papers/sparse.pdf) supports compiling structural operator dependencies. Passive cross-branch couplings still belong to this Human's full operator.
- [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html) explains the constraint-space Jacobian/response formulation. The proposed `C = J R` is an inference applied to Numi's own complete, equality-conditioned operator; it does not import MuJoCo's soft-contact model or claim MuJoCo timing.

See the [portable qualification receipt](evidence/numi-human-2to1-finish-research-20260928.json). Evidence roots: `NumiHumanBrainRuns/2to1-finish-20260928` and the preceding `2to1-topology-20260928` on both hosts. Main source baseline: `f570ca3a846b291771230ec990e9e206f0c0d3a7`. Register prototypes remain isolated and unpromoted.
