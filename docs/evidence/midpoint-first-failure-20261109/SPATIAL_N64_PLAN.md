# Independent n=64 spatial follow-up

The n=16 to n=32 reference comparison did not meet the original spatial
force and cumulative-impulse limits. The requested spatial accuracy is not
yet established. Before running, compare one further mesh at n=64 with n=32,
keeping the physical geometry, material parameters, initial conditions,
contact activation width, area coefficient, simulated duration and 2 us output
clock unchanged. Check n=64 at both 0.5 and 0.25 us using the original temporal
limits, and only interpret the spatial comparison when both meshes pass their
temporal checks. Verify total reference area and finite positive gap.

Prediction: the n=32 to n=64 pair will still fail at least one original spatial
gate. Reject that prediction if all gates pass with a valid temporal check.
This is a CPU-only independent reference; no claim of native n=64 behavior or
material calibration follows. Preserve both successful and failed attempts.

Resource bound: roughly 274625 nodes and 1572864 tetrahedra, one low-priority
CPU process, run the two timesteps sequentially with a 2-hour cap on each. The
n=32 runs took 255 and 518 seconds; cubic growth suggests an hour or longer in
total, but is not a guaranteed estimate. Do not run GPU timing benchmarks
concurrently. Before the large runs, build a separately named binary with only
the n limit raised; pass its energy-gradient self-test and reproduce the
retained n=2 0.25 us output byte-for-byte. Record source, executable and input
hashes and execution metadata. Stop after this mesh; further refinement needs
another explicit resource and validation plan. A timeout, instability, temporal
failure or missed spatial gate is a result, not a reason to relax a threshold.
