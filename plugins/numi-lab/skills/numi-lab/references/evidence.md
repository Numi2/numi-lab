# Numi Lab completion and evidence

For discovery commands, return the relevant version, resolved path, status, and
output or typed failure. For commands that execute or produce artifacts, return
the exact runtime revision and worktree state, arguments, artifact directory,
relevant runtime and artifact hashes, stdout/stderr or typed failure, and the
actual device/runtime used. For simulation, training, evaluation, and profiling,
also return failed environment steps, throughput, retained and peak memory,
replay/fingerprint evidence, available traces or counters, and task-specific
physical outcomes. State when a requested profiler gate or physical outcome was
unavailable. A build, test, reward, liveness check, or timeline-only trace is not
physical or detailed GPU performance proof.

Before a long Metal, training, evaluation, or profiling run, inspect active
workloads and existing artifacts. Do not duplicate a live run or contend for a
dedicated GPU; use isolated build/worktree paths and checkpointed execution
when the workload warrants them.

Retain every physically valid candidate and its measured outcome. Changing the
configured production policy is an explicit evidence-backed selection, not a
binary verdict that erases partial progress.

For molecular and tissue results, also preserve model/parameter provenance,
units, numerical tolerances, conservation and convergence evidence, uncertainty,
and the exact observable. Electronic energies, activation free energies, rates,
occupancy, cellular response, and organism outcomes are separate claims.
Synthetic culture, tissue, and artificial-life simulations do not establish
biological calibration or clinical validity.

Simulator evidence is not hardware evidence. Simulation, authoring, and local
training may be autonomous. Before real hardware can move, stop and obtain the
owner approval required by the configured arming policy, and verify limits and
an emergency stop; never infer that authority from approval to simulate.
