# Benchmark accounting: correctness smoke, 2026-11-07

The existing Franka benchmark previously divided requested environment steps by
wall time without establishing that those steps completed. Its revised output
uses the runtime's accepted count and includes rejected work in elapsed time.
It reports initialization, warmup, GPU and submission durations separately.
Accepted simulated seconds sum physical time over environments; they are not
the elapsed duration of one environment.

The change rejects an incomplete or failed warmup, incomplete measured
rollout, physics-error transition, or non-finite reward. The fault-injection
CPU test exercises those cases and input limits, and is registered with CTest.
A fault-injection test is not a native failure or rollback qualification.

On Apple M6, the native correctness smoke used source commit 6ef808aa, seed
7, the CMake/Ninja target metalrobo_bench and its compiled Metal libraries.
The 1-environment, 2-step run accepted 2/2 environment steps; the 32-environment,
10-step run accepted 320/320. Neither reported failed steps, transition physics
errors or non-finite rewards. The retained logs include source, executable,
library and metallib hashes and exact output.

These short runs establish that the new success accounting is exercised
through the native API. They are not physical validation, GPU failure
injection, a latency/throughput benchmark, a trained policy, a comparison
against prior revisions, or an accuracy-adjusted performance claim.
The control is a host-generated sinusoid. Later performance runs must use
qualified fixed workloads, exclusive device ownership, appropriate warmup,
repetitions and variability, and matched physical accuracy. Rendering,
learning and large coupled scenes need separate measurements.

Artifacts: [native smoke evidence](evidence/benchmark-accounting-20261107/).
