# Numi Human / Brain: ten simulated seconds in five wall seconds

Updated 2026-09-28. This is a development plan with measured starting points and acceptance gates. The five-second target is unachieved, and feasibility on this M4 Pro has not been established.

## Measured starting points

| Workload | Complete launch | Scope |
| --- | ---: | --- |
| Retained fastest development run | 86.457746 s; repeat 86.515307 s | CPU-one-handoff, no-push feedback ablation, full timing receptor audit disabled |
| All-GPU finish, original, two opposite-order pairs | Mean 160.257761 s | Actual 10,000 accepted physical and receptor rows, no-push feedback ablation, receptor audit retained |
| All-GPU finish, equality cache, same pairs | Mean 157.038299 s | Same clocks, all 64 ordered sweeps, byte-identical physical/receptor/terminal witnesses, zero assistance |

The cache saves 3.219463 s, or 2.009%, in its paired all-GPU workload. It does not speed up the selected hybrid CPU finish. The remaining whole-launch speedup is 17.29x from the hybrid development reference, or 31.41x from the cached all-GPU diagnostic reference. Full-feedback speed remains a separate unmeasured gate.

## Working backward from five seconds

Keep 10,000 one-millisecond physical roots. The engineering allocation is one second for startup/teardown, 3.5 seconds for the complete horizon, and a 0.5-second margin. These allocations are targets, not predictions.

| Work per complete root | Target mean |
| --- | ---: |
| Kinematics, authored routes, muscle/tendon dynamics | 70 us |
| Complete mechanics: assembly, factorization, responses, all 64 sweeps, projection | 100 us |
| Complete scheduled Brain: perception, tissue, policy, memory, motor, accepted consequences | 130 us |
| Receptors, joint admission and committed journal | 30 us |
| Amortized encoding/submission | 20 us |
| Total | 350 us |

With the retained hybrid startup of 1.434893 s, the hard allowance is only 356.51 us/root. A four-millisecond finish cannot fit, even if every other stage becomes free.

## First major implementation: compile the mechanical factor and response graph

The captured native effective operators at roots 1, 100 and 500 have the same observed upper-triangle nonzero pattern: 3,189 edges, with every nonzero retained and no magnitude threshold. The current ascending elimination produces 4,906 new fill edges and 8,223 lower factor entries. Reverse authored elimination and a minimum-degree diagnostic each produce only six new fill edges and 3,323 factor entries. Estimated symmetric rank-one update entries fall from 345,510 to 52,891, about 6.53x less symbolic update work.

This is a read-only operation-count result, not a latency measurement or a proof that sampled zeros remain zero in future configurations. The factor scratch is unchanged by each captured complete finish; the owning Cholesky leaves the original upper triangle intact. The lower input coefficients have already been overwritten by that factorization. Do not infer missing lower coefficients or authorize future pruning from these snapshots.

Implement in this order:

1. Compile a conservative graph from authored body/joint connectivity and every passive off-diagonal coupling. Preserve all coefficients, pivot checks, row scales and typed failures. Bind topology/parameters to the pack fingerprint; do not discover sparsity through per-root CPU readback.
2. Prepare a leaf-first permutation and checked factor/solve index tables. Prototype the complete permuted factor and many-right-hand-side response construction. Expose independent branches, rows and RHS columns across lanes.
3. Keep equality conditioning, tendon/force ownership, original contact/source-limit ordering and all 64 sweeps. A tree-only rigid operator is insufficient if it omits the complete effective operator's couplings.
4. Qualify the changed numerical implementation against the owning FP64 reference and native physical gates. Establish deterministic replay of the candidate; do not label a reordered factor as byte-identical to the original without measuring that.
5. First feasibility gate: the **complete** mechanical owner below 1 ms/root on real accepted states. If it passes, advance toward the 100-us allocation with whole-launch measurements.

Evidence: `mechanical-symbolic-fill-v2.json`, `analyze-mechanical-symbolic-fill-v1.py`. The owning source, full native snapshots, exact runtime bindings and rejected earlier response-layout experiments remain retained.

## Complete Brain graph

Compile immutable incident indices and a generation-specific shared feature graph. Share physical receptor/belief reductions among policy, planning, motor, cerebellar and consequence consumers. Preserve ascending reductions where exact agreement is required, every authored timestamp/rate, stochastic counters, history and emergency capacity.

The regional scheduler already averages 12.015 nominal module updates per millisecond; that reduction already exists. Do not count it again or arbitrarily slow neural modules. The separate instrumented 111-root sample sums to 4.512 ms across 92 phase leaves; it is not the latest complete-run mean. Use the full 500-ms nominal schedule, adaptive/event cases and later history states to qualify the compiled graph.

First feasibility gate: the complete Brain below 1 ms/root. Target allocation: 130 us/root. Fuse dependency-safe consumers and retain intermediate values in lane registers where measurement supports it. No single motor or cerebellar optimization qualifies the full graph.

## Geometry, submission and joint publication

- Compile and share the kinematics/path traversal that feeds body mechanics, 416 muscles, tendon loads and receptors. Use one accepted geometry generation and measured layouts; preserve all authored active and stateful muscle dynamics.
- Establish joint physical/neural publication for two successive device roots. Inject rejection after each owner, restore both states and replay the next root. Full neural checkpoint and joint rollback remain open; accepted trace equality does not close them.
- After joint publication passes, grow bounded GPU submissions from 2 to 8 to 32 roots, retaining the one-millisecond clock and device admission at each root. Remove repeated host synchronization and unnecessary materialization while preserving error publication.
- Use prewarmed source-bound pipelines and bounded reusable arenas. Keep compact committed witnesses available; avoid unneeded printing/readback in normal timing. Audit/replay runs remain separate measurements of the same admitted computation.

## Apple hardware choices

M4 is Apple9 in Apple's feature tables. Apple's documented per-core neural acceleration for inline tensor operations begins at Apple10. Metal 4 API availability on M4 does not imply that newer hardware acceleration exists on this Mini.

Cooperative tensors and fusion can still avoid intermediate memory stores. Treat MPP matrix operations as measured candidates for appropriate dense Brain work, with their numerical contract qualified. They do not parallelize the dependencies of ordered contact impulses. Avoid adding threadgroup staging or increasing thread count without measuring register pressure, accesses and whole-graph latency.

Source-bearing original/cached root-100 GPU traces are retained. Shader line costs, active/total instructions, divergence and register/threadgroup constraints have not yet been collected because the MacBook UI is locked. Trace creation and timeline coverage are not instruction/occupancy profiling. Continue graph work independently of that UI gate.

Primary research: [Apple GPU feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), [inline Metal 4 operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4), [Metal compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/), [Xcode shader profiling](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance), [Featherstone's sparse dynamics research](https://royfeatherstone.org/papers/sparse.pdf), [MuJoCo's computation description](https://mujoco.readthedocs.io/en/3.5.0/computation.html).

## Advancement and completion

Advance complete accepted 10-second launches through **30 s, 10 s, then 5 s**. Promote each slice only after its owning numerical/physical gate and actual whole-launch measurement. Preserve failed candidates and qualify any numerical change explicitly.

The final 2:1 gate is five complete cached launches at or below five wall seconds, including startup/teardown, with the intended full feedback, 10,000 accepted roots, all 64 sweeps, zero assistance, retained source/binary/seed identity, exact candidate replay, bounded memory and no growing swap. Recheck the established perturbation behavior independently; directional correction is not general recovery. Full neural checkpoint, joint rollback and energy closure must keep their own evidence status.

Batch 8 and 16 independent Humans afterward for learning throughput. Report accepted simulated seconds/hour, exact replay per environment, memory and swap. Aggregating independent environments does not satisfy single-Human latency.

## Current implementation closure

The equality cache remains opt-in and default-disabled. Canonical Book and Mini owning checks pass, including the typed two-environment/four-root/all-64-sweep fixture. Eight canonical shader replays reproduce all 24 unique output buffers at initialization and roots 1/100/500. Canonical Mini 112-root physical/receptor/terminal witnesses match the original exactly. The source-bound canonical ten-second launch completed in 157.199836 s with 10,000 physical and receptor rows, the same complete trace/terminal witnesses as the original, 314,195,968 bytes peak child RSS and unchanged swap. These checks qualify the opt-in cache on the measured standing workload; the cache stays default-disabled.

The initial canonical small-model check exposed a preexisting cancellation-order failure. Restoring the ordered scalar contact contraction for NV <= SIMD32 fixes that failure without changing the NV128 Human path. Failed and corrected receipts are retained.

The measurements and source-bound native receipts are retained under `NumiHumanBrainRuns/2to1-topology-20260928` on the MacBook and Mac mini. The paired-repeat archive SHA-256 is `7ad10f1b1897695c3a756e7c9fdc321e3104636d9c73d8d910cf9449b6511486`. Canonical Mini qualification uses shader SHA-256 `2ee9533671fc2e040e2597702d333de17a3b1735540348f215bc48063131d83f` and Brain library SHA-256 `dfa3b6cec6a5a65748233a2c3ab5b224deb401d9e969f46e89cce20e1056b650`.
