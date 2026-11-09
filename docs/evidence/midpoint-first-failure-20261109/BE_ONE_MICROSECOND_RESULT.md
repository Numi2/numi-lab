# Backward Euler at one microsecond

The 10,000-step request at one microsecond, using the frozen executable and
shader from the 1.953125 microsecond trial, rejected at zero-based step 3186
after 3,186 accepted steps. Matter reported nonlinear-solver failure status
10; the gap was 150.195 micrometres, outside the 100 micrometre contact band.
The previous physical state was restored and the recorded replay,
continuation, forced rollback, and fingerprint mismatch checks passed. The
retained trace includes the full status and diagnostic values.

Because the simulation did not complete, no force, peak or impulse accuracy
comparison is available for this step. The result does not show whether a
completed one-microsecond backward Euler rollout would pass or fail those
gates, and it does not identify the cause of the nonlinear failure. There is
still no matched-accuracy performance comparison. The measured wall time
includes diagnostics and replay while the CPU reference was running and is not
a benchmark. The initial 0.9765625 microsecond attempt also remains recorded:
it hit the probe workload limit before any physical steps.
