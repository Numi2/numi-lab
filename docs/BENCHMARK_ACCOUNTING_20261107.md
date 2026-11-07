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

A subsequent input/reporting correction at source commit 402bbcfa rejects
non-decimal numeric strings and suppresses mean reward for invalid transitions.
Its CPU fault-injection suite passed, and a rebuilt 1-environment native smoke
accepted 2/2 steps with no reported errors. The new binary's source and hash
are retained in the final-reporting subdirectory; the earlier results remain
bound to their original source and binary.

An independent CMake Release build at source revision dfa906b5 completed.
Both CPU accounting/adapter CTests passed, and a 1-environment native smoke
accepted 2/2 steps. Exact executable, library and shader hashes and the
native result are retained in the release-smoke subdirectory. This verifies
the Release target and success accounting, not an optimized performance
comparison or contact qualification.

A persistent single-environment latency instrument now measures individual
native advance calls and reports p50, p95 and maximum call latency, together
with total series wall time, accepted physical time, failures, GPU time and
retained runtime buffers. It uses a host-generated sinusoid and does not
measure rendering, learning or end-to-end interactive performance. The 20
warmup calls and every measured transition are checked for invalid physics
or non-finite rewards. The batch instrument applies the same warmup transition
checks.

CPU fault-injection tests cover failed warmup, incomplete or failed measured
steps, invalid transitions and malformed inputs. The final Release native
smoke at source revision d13b6334 accepted 2/2 batched steps and 5/5 separate
single-environment steps. It is a correctness smoke, not a performance sample.
Exact source, binary, library, shader hashes and output are in the retained
warmup-smoke directory.

Two previously registered but unstarted follow-up studies were explicitly
stopped when the benchmark source changed to validate warmup transitions;
their stop records are retained. New measurements require a new registration
bound to the revised source and binary.
