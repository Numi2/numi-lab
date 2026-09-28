# Numi Human and Brain: 10 simulated seconds in 5 wall seconds

Current execution order and pause handoff: [2026-09-29 research and execution plan](NUMI_HUMAN_2TO1_EXECUTION_PLAN_20260929.md). This file retains the implementation history.

2026-09-28. The target remains unachieved. Feasibility on this M4 Pro is not established. This plan preserves 10,000 one-millisecond physical roots, every scheduled neural update, all 64 ordered sweeps, all effective-operator/passive/tendon/muscle terms and native physical admission.

## Measured gap

The retained hybrid development reference is 86.46 seconds, with a no-push feedback ablation and different receptor-audit scope. The paired cached all-GPU reference averages 157.04 seconds with 10,000 accepted physical and receptor rows, also using the no-push feedback ablation. These are different workloads. Neither establishes full-feedback five-second performance. The required overall speedup is 17.29x or 31.41x respectively.

The latest same-binary sparse-on/off ten-second pair accepted all 10,000 roots and receptor rows, retained all 64 sweeps and used zero assistance. Sparse on took 153.9486 seconds; off took 158.1849 seconds. This one pair saves 4.2362 seconds (2.68%), not a repeated throughput result. The default-off physical, receptor and terminal witnesses exactly match the retained original implementation. The candidate has a different numerical trace. Peak child RSS was approximately 300 MiB (RSS only; peak process memory footprint is larger) and swap remained at 149.19 MiB. The subsequent content-cached candidate took 151.1460 and 151.1291 seconds on two complete launches (mean 151.1375 seconds, requiring another 30.23x for five seconds) and matched all three full candidate witnesses from the earlier run. The two same-runtime launches exactly replayed one another. These launches used the feedback ablation.

The full selected v6 balance-feedback program is now measured with its 104 balance routes, unchanged joint/path feedback and +50 N root pulse at root 100 for 20 roots. The same frozen sparse/cached runtime took 152.982522209 and 174.782947667 seconds on two complete uninstrumented launches. Both accepted all 10,000 physical and receptor rows, all 64 sweeps and zero assistance; physical, receptor and terminal witnesses exactly replayed. The mean is 163.882734938 seconds, requiring another 32.78x for five seconds. The second was 14.25% slower; its cause is unmeasured. Both kept swap at 149.19 MiB and peak child RSS below 315 MiB. This establishes the target workload and candidate replay, not stable latency, general recovery, full neural checkpoint identity or energy closure.

The [revised architecture slices](NUMI_HUMAN_2TO1_ARCHITECTURE_SLICES_20260928.md) prioritize correlated profiling, measured repetition in the ordered finish, complete Brain state traffic, shared geometry and causally ordered device chunks.

The canonical rotating native profile samples one mechanical stage per root; it cannot be summed as an exact wall-time decomposition. Excluding initialization, measured medians are approximately:

| Stage | Current attribution |
| --- | ---: |
| Kinematics | 1.58 ms |
| Muscle dynamics | 0.49 ms |
| Stand prerequisites | 0.425 ms |
| Full effective-matrix assembly | 0.053 ms |
| Factorization | 0.940 ms |
| Equality responses | 0.504 ms |
| Equality factor | 0.325 ms |
| Projected responses | 0.890 ms |
| Original ordered finish | 4.66 ms |
| Brain motor, coarse GPU attribution | 2.35 ms |
| Brain accepted consequences, excluding initialization | 1.70 ms |

The cached finish has separately measured approximately 4.17--4.23 ms on the retained root-100 workload. A four-millisecond finish cannot fit a complete 0.35-millisecond root.

## Budget, not a speed prediction

Reserve 1.0 seconds for startup/teardown, 3.5 seconds for the horizon and 0.5 seconds of margin. The complete root allocation is:

| Owner | Target |
| --- | ---: |
| Geometry, source routes, muscle/tendon dynamics | 70 us |
| Complete mechanics, including every one of 64 sweeps | 100 us |
| Complete scheduled Brain | 130 us |
| Receptors, joint admission and accepted journal | 30 us |
| Amortized encoding/submission | 20 us |
| Total | 350 us |

With the retained 1.4349-second hybrid startup, only 356.5 us/root remains. Startup therefore belongs in the final gate, along with the complete intended feedback program and journal scope.

## 0. Freeze the complete workload and separate its timing scopes

Bind the target to one Human, the intended feedback and perturbation program, 10,000 physical roots, unchanged neural clocks/cadences, native admission and the accepted journal. Retain the ablation as an attribution control. Before judging five-second feasibility, time the complete intended feedback with the same source, binary, inputs and seed. Include startup and teardown. Keep static-equilibrium and pipeline cache state explicit.

Record host preparation/submission, GPU work, waits and output separately without treating rotating stage samples as a complete additive decomposition. Use instrumented short captures to identify causes, then uninstrumented whole launches to judge improvement. Hardware shader occupancy, spills and instruction attribution have not yet been collected; the current timings do not prove those causes.

## 1. Compile the operator and responses, then remove sweep repetition

**Current experiment:** construct a conservative graph from authored body ancestry, all root coordinates and every passive off-diagonal coupling. Add symbolic fill without a numerical threshold. Validate every assembled nonzero against that graph on the GPU. Keep the original lower input coefficients, row scales and pivot failure checks, and return solutions in original coordinate order.

The actual native graph has 3,352 factor entries and 43 dependency levels. The sampled raw matrix has 3,189 upper nonzero edges. The cold graph deliberately includes additional structurally possible entries. Sampling zeros does not authorize dropping future coefficients.

The first sparse implementation was a performance regression: factorization remained near 0.95 ms, equality responses rose to 1.03 ms and projected responses to 1.42 ms. A subsequent dependency-level implementation reduced response medians to 0.369 and 0.760 ms, but assigning an entire factor column to one lane raised factor time to 5.18 ms. Both variants and their sources/binaries are retained. Neither is a throughput-qualified replacement.

Assigning independent diagonals and factor entries across lanes within each level reduced factor time to approximately 0.699 ms, equality responses to 0.396 ms and projected responses to 0.756 ms. The original finish fell to 4.405 ms with the faster free solve. These are instrumented 112-root attribution results. The later full-horizon specialization passed one ten-second run; the content-cached version then passed two complete launches with exact candidate replay. The candidate physical, receptor and terminal witnesses match across the three sparse implementations on the 112-root workload, and differ from the original numerical implementation.

The factor-only specialization measured approximately 0.686 ms, with equality/projected responses at 0.394/0.754 ms and the original finish at 4.409 ms. Its native root-100 operator/factor/raw equality response/q/equality capture is byte-identical to the previous parallel-row variant. This isolated specialization produced little further improvement; register/occupancy causality has not been established by a shader profiler. Measure the complete owner; symbolic operation counts alone have already failed to predict latency.

The full 128-DOF operator oracle reports no missing nonzero coefficient, factor reconstruction error of 5.93e-7 relative to the peak coefficient, maximum raw equality-response forward error of 1.89e-4 versus FP64 and componentwise backward error of 2.77e-6. RHS vectors are reconstructed from captured native q/source equality records with explicit FP32 expression ordering; they are not directly captured pre-solve shader RHS buffers. This is operator arithmetic evidence, not complete contact/limit/time-integration or energy qualification.

The integrated Brain path submits one-millisecond segments. The sparse prototype originally rebuilt its symbolic graph at every root. The new host cache is per context and invalidates on model-record or passive-program contents; it caches structural metadata only. It still constructs a content key, copies graph metadata and uploads that metadata per root. The complete launch took 151.1460 seconds and matched all three earlier candidate witnesses. Its native resource log reports two graph compilations and 10,001 cache hits across setup and physical submissions, with a 121,916-byte content key and 42,696-byte graph. A second complete launch took 151.1291 seconds and exactly replayed the physical, receptor and terminal witnesses. The default-off v9 112-root run also exactly matches the original control; the preceding v8 default-off ten-second run matched the original full horizon.

The factor graph has enough entries for a roughly 13.4-KiB packed float factor. A future dedicated factor kernel can test a compact threadgroup representation and compiled dot-product incidence to reduce device traffic/barriers, subject to the full kernel's threadgroup/register limits. This is an experiment, not a measured improvement, and has lower priority than the four-millisecond finish.

The owner already prepares same-root equality-conditioned contact and potential source-limit responses in independent GPU work before the sweep. Reuse these responses and move only the remaining same-root derivative/coefficient passes out of the sweep. Use bounded device storage, source ordering and exact same-root derivatives. The earlier large threadgroup response cache regressed, so this requires measured layout and occupancy rather than another larger cache.

Profile the finish's ordered impulse decisions and nonlinear equality/source-limit preview. Compile independent equality/preview dependencies into levels and distribute response updates across lanes. Preserve every sweep and every active-set/physical gate. Consider an equivalent constraint-space Schur-complement representation only with the complete operator and actual production RHS/contact/limit reference; it is a numerical implementation change needing its own physical qualification and exact candidate replay.

Prototype a register-distributed SIMD32 finish: four of the 128 DOFs per lane, compact source/equality incidence, explicit SIMD communication and the same ordered decision owner. Reducing the old kernel to 32 threads already regressed; the proposed change needs a new data flow that removes repeated threadgroup barriers and preview traffic. Measure registers, spills, instruction costs and complete finish latency, then native admission and replay. A thread-count change alone is insufficient.

This prototype has now executed. The v3 natural-DOF register representation and v4 packed scalar representation each retained every sweep, executed their actual new finish pipeline, accepted the full 128-root target prefix and exactly matched the original target's physical, receptor and terminal witnesses. Native owner probes passed on the Book and Mini. The measured finish medians, excluding root 0 from eight samples at roots 0--7, were 5.618250 and 5.922625 ms, against 4.176333 ms for the retained control. This is early-root attribution; the physical prefix includes the later pulse. Both are rejected performance variants, with sources and binaries retained outside main. Their physical prefixes and minimal owner probes do not qualify a full ten-second replacement. A v2 selector limited to eight contact bindings did not execute the new pipeline for the actual ten-binding Human; that retained run is fallback evidence. Register/spill/occupancy causality remains unmeasured. Close blind register-layout tuning and measure contact/limit/friction repetition before the next solver implementation.

The complete sparse counter run took 153.382876 seconds and exactly matched all v9 target physical, receptor and terminal witnesses, with zero assistance and unchanged swap. Samples at roots 0, 100, 1,000, 5,000 and 9,999 retained all 64 sweeps and reported 26,427--32,001 limit previews and 1,624--2,112 nonzero limit updates, versus 0--13 boundary-friction solves. A native Metal timeline also passed; the selected detailed counter profile was unsupported. These are operation counts, not instruction-cost attribution. Prioritize same-root limit preview/response traffic and measure a candidate before extending it. See [the counts and next experiment](NUMI_HUMAN_2TO1_ARCHITECTURE_SLICES_20260928.md#executed-finish-workload-counts).

The first same-root preview-bound cache passed the 128-root native target and exact witnesses but gave no finish-stage improvement: 4.160083 ms off versus 4.304458 ms on for roots 1--7. It remains an isolated, retained performance experiment. The next bounded test reuses the existing factor arena only after its last solve for a same-root response-column cache, with measured distinct columns/hits and a full fallback for every uncached response. See [the rejected experiment and revised next slice](NUMI_HUMAN_2TO1_ARCHITECTURE_SLICES_20260928.md#rejected-preview-cache-and-the-next-solver-experiment).

**First gate:** complete mechanics below 1 ms/root on accepted native Human states. Advance toward 100 us only after that gate. Factor-only gains cannot satisfy it.

## 2. Compile the complete Brain graph

Profile the entire motor and accepted-consequence graphs on the selected frozen runtime. Rank active-site traversal, neuron/tissue updates, protective/affect work, memory and hash/checkpoint passes by measured time and traffic. Bind caches to committed state and immutable resource versions, use compact active indices and consumer-oriented layouts, and fuse repeated passes when counters and the complete graph improve.

Compile the existing cadence into a reusable schedule. Preserve the neural time base, event deadlines, emergency capacity and every required consequence. A longer neural tick or omitted protective work is not an optimization of this workload.

Use MPP/cooperative matrix operations for measured dense opportunities with a qualified numerical contract. M4 is Apple9; Apple's documented per-core neural acceleration for inline Metal tensor operations begins at Apple10. Metal 4 availability alone does not grant that hardware to this Mini.

**First gate:** complete scheduled Brain below 1 ms/root. Advance toward 130 us after this gate, with full intended feedback and the established perturbation comparison kept separate from a general recovery claim.

## 3. Compile kinematics and authored muscle routes

Precompute body/joint ancestry, transform dependency levels, authored coordinate incidence and route traversal. Evaluate q-dependent transforms once, share them through the owning geometry buffers and eliminate repeated full ancestry traversal. Use sparse/consumer-oriented Jacobian layouts where all consumers can preserve the complete force operator. Parallelize independent bodies, sites and muscles; avoid introducing another force or rate owner.

Target the measured 1.58-ms kinematics pass before the 0.49-ms muscle pass. Preserve fibre/tendon dynamics, route lengths/velocities, angular Jacobians, force reduction and calibration. Recheck source path and force closures on native accepted states.

## 4. Move root scheduling and acceptance into bounded device chunks

Compose the Brain decision, physical root, receptors, admission and neural consequences on the owning GPU timeline. Reuse compiled bindings and pipelines; avoid a CPU wait or state copy at each substage. Start with bounded 16- and 32-root chunks and keep an accepted prefix/status journal.

Failures must stop dependent work before accepted state can publish. Device chunking needs the actual joint Brain/Human transaction and checkpoint/rollback behavior, not only exact physical traces. Preserve random streams and scheduled events across chunk sizes. Keep full neural checkpoint, joint rollback and energy closure as separate qualification boundaries until executed.

Precompile or archive source-bound pipelines and retain verified immutable startup assets. Measure cold and cached startup separately, and include the stated launch scope in the final five-second result.

## Advancement and completion

Advance complete accepted ten-second launches through 30 seconds, 10 seconds, then 5 seconds. Promote a slice only after its owning precision and physical gates and a repeated complete-run comparison. Preserve rejected variants and sole evidence.

Final gate: five complete launches at or below five wall seconds, including startup/teardown, with the intended full feedback, 10,000 accepted roots, all 64 sweeps, zero assistance, source/binary/input/seed identity, exact candidate replay, bounded memory and no growing swap. Static operation counts, short prefixes and aggregate environment throughput do not satisfy this gate.

Batch 8 then 16 independent Humans for learning after the single-run owner is qualified. Measure accepted simulated seconds/hour, replay per environment, retained/peak memory and swap. Keep native heaps, private arenas and learner caches within the measured device working set. Current inspected free storage is roughly 46 GiB on the Book and 40 GiB on the Mini; continue measured checks before long captures or builds.

## Primary research

- [Featherstone: branch-induced sparsity and articulated factorization](https://royfeatherstone.org/papers/sparse.pdf). Supports authored structural sparsity and appropriate elimination order; it does not justify omitting passive couplings or contact/equality terms.
- [MuJoCo computation and sparse inertia factorization](https://mujoco.readthedocs.io/en/3.5.0/computation.html). A reference for tree-aware operators and constraint-space reasoning, not a substitute simulation or matched performance result.
- [Apple: Metal compute submission, occupancy and unified memory](https://developer.apple.com/videos/play/tech-talks/10580/). Supports batching useful work, reducing waits and accounting for the working set.
- [Apple: GPU performance on Apple silicon](https://developer.apple.com/videos/play/wwdc2020/10632/). Supports measuring register/threadgroup use and execution overlap.
- [Apple GPU feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) and [inline Metal ML operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4). Bind architecture choices to actual supported hardware.

Evidence and versioned experiments: `NumiHumanBrainRuns/2to1-topology-20260928` and `NumiHumanBrainRuns/2to1-finish-20260928` on both hosts. Runtime/source version v9 contains the host graph cache and passed repeated complete physical horizons with exact candidate physical/receptor/terminal replay, including the newly selected full balance-feedback target. Native neural checkpoint identity, joint failure/restore and energy closure remain separate unqualified boundaries. Published sparse source is main `f570ca3a846b291771230ec990e9e206f0c0d3a7`, corresponding to `56a7ce28` plus the four source changes bound in `sparse-operator-source-bindings-v9.json`. The sparse selector remains disabled by default; register variants are unpromoted isolated experiments.
