# Persistent single-environment latency exploration

A preregistered Release-build study measured one persistent Franka environment
on an Apple M6 with host-sinusoidal actions. Five seeds were paired across
100- and 200-call measured series, with 20 warmup calls and alternating arm
order. All ten series completed their declared accepted-step and output checks;
all reported zero terminal or timeout events and no nonzero GPU status.

For the 100-call series, the median across seeds of each series' p50 was
3.283 ms, p95 3.306 ms, and maximum 3.312 ms. For the 200-call series, those
medians were 3.282, 3.303, and 3.312 ms. These are medians of per-series
statistics, not pooled percentiles. The mean paired p95 difference was
-0.0145 ms, inside the deliberately broad registered -2 to 2 ms interval.
The two lengths include different trajectory portions and produce separate p95
estimates; they are not a per-step paired latency comparison.

The median aggregate series-wall rate was 5.08 accepted simulated seconds per
wall second for each length; the wall interval includes action filling and
transition validation. The reported per-call latency measures the rollout API
call itself and includes its diagnostic timing overhead. Initialization and
warmup are separate; rendering, policy inference, learning, continuum and an
accuracy-matched competitor are absent. This is not interactive end-to-end
latency or a physical or learned-task qualification. The runtime retained
882,538 buffer bytes, not process-wide peak memory.

Task-owned CPU builds and GPU runs were paused for the measurement. The
before/after host process snapshots are retained; macOS metadata services were
active after the run. They do not prove uninterrupted system GPU exclusivity or
controlled thermals. Five seeds on one machine are descriptive, not independent
hardware replications.

An earlier latency registration was stopped before any trials after the
instrument was found not to record terminal counts or preserve the first
nonzero GPU status. That stopped record is retained in
[the earlier plan](evidence/benchmark-latency-plan-20261107/). The corrected
instrument passed fault-injected CPU checks and a native success smoke before
the new [registration](evidence/benchmark-latency-plan-v2-20261107/).

The full content-addressed notebook with executable, library, shader, input,
trial and revision snapshots was archived and integrity-verified at
/Users/numi/numi-lab-latency-science-v2-20261107/archive-final.
Compact raw outputs, receipts, host snapshots, analysis and revision are in
[latency results](evidence/benchmark-latency-results-20261107/). The portable
archive, not the compact repository copy, contains all bound artifacts.
Registration is f4435b6c5fdb1e7cd62aab772e4aa31abbbb5492f3c195eea34473d6b6b70619.
A meaningful next measurement requires a physically qualified closed-loop task,
with rendering and policy inference visible separately, and fixed error gates.
