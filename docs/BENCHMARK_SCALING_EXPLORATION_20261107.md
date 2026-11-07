# Accepted-step batching exploration

A preregistered, paired science study compared 1 and 32 environments in the
existing Franka host-sinusoidal task on Apple M6. Each arm ran 100 control
steps after 20 warmup steps. Five fresh seeds were paired and the arm order
alternated. This was an exploration of accounting-valid operational scaling:
the study did not establish physical accuracy, policy quality, comparable work
in another engine, or improvements over a prior Numi revision.

All ten runs passed the declared accepted-step and output checks. Across five
runs per arm, median accepted environment steps per wall second were 325.36
(range 311.68–326.82) for one environment and 8,326.14
(range 8,104.70–8,538.13) for 32. The mean paired increase was 7,988.03
environment steps per second, within the broad preregistered interval.
The median wall durations for the 100-step calls were 0.30735 s and 0.38433 s;
the median reported GPU durations were 291.50 ms and 358.00 ms respectively.
The runtime reported 1,088,458 and 31,989,754 retained buffer bytes.
These byte counts describe runtime buffers, not process-wide or peak memory.

The binary was built with an empty CMAKE_BUILD_TYPE and no explicit global C++
flags; this is not a release-build performance qualification. Rendering,
learning, coupled FEM and accuracy-matched alternatives were not included.
Seed variation on one host is descriptive and is not hardware replication;
thermal conditions were not controlled. Failed or rejected physical steps
would invalidate the affected run rather than count as successful work.

The full content-addressed notebook including executable, dynamic library,
shader, input and model snapshots was archived and verified at
/Users/numi/numi-lab-benchmark-science-20261107/archive-revised on the measured
host. This repository retains a compact copy of registration, analysis,
revision and per-trial outputs in
[benchmark scaling evidence](evidence/benchmark-scaling-exploration-20261107/).
The compact copy is not itself the complete portable notebook. The archived
registration identifies source revision f4dd2312 and all bound hashes; later
documentation does not alter the measured identity. A follow-up using unused
seeds was registered but has not yet been run.

The selected developer directory on this host was
/Library/Developer/CommandLineTools. Metal compilation was available, but
xcrun could not find xctrace and /Applications did not contain Xcode.
Consequently these observations do not identify occupancy, bandwidth, kernel
bottlenecks or execution limiters. Those require an appropriate Metal profiling
setup before making a targeted GPU optimization claim.

## Post-run host-contention finding

After the study, process start times showed two background Git index-pack
processes had been running throughout the ten trials. The GPU trials did not
overlap another scheduled GPU experiment, but CPU and disk activity were not
isolated. Their potential effect on command submission and wall time was not
measured. The sealed exploratory analysis and revision are retained unchanged;
their supported prediction is therefore limited to the observed, potentially
contended conditions and is not a clean-host latency or throughput result.
A separate Release study was preregistered before its trials; it must record
the actual host load and cannot silently relabel this study.

The reported measured wall interval begins after host action generation and
allocation. Initialization and warmup are separately reported, but action
preparation and policy learning are not included in the throughput denominator.
The observation is therefore not end-to-end time to skill or complete
interactive latency.

A replacement Release study was preregistered after the warmup-accounting
change, with its plan and registration retained in
[release study evidence](evidence/benchmark-release-plan-v2-20261107/).
The earlier unstarted registrations were stopped explicitly when their source
artifact changed. The replacement subsequently ran; its observations and host
activity are recorded separately in [the Release study](BENCHMARK_RELEASE_STUDY_20261107.md).
