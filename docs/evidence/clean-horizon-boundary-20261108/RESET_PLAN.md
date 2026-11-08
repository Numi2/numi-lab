# Reset-aware boundary qualification

The original matrix assumed both environments would timeout at 256. The actual scheduler requests a periodic reset for environment 1 at zero-based step 220; retained retry-e2-c1-s256 output confirms only environment 0 timed out. A mid-horizon reset interrupts the claimed trajectory even if no task termination is reported. Mark any host-requested reset after the initial step as interruption, without changing reset behavior or termination counts.

Before rerun, predict for one environment and 255/256/257 steps: termination counts 0/1/1, clean rates 1/1/0. For two environments: counts 0/1/1 and clean rates 0.5/0.5/0.0. Run both chunk sizes 1 and 8 and require zero physics failures. The 257-step result is zero because environment 0 times out before the endpoint and environment 1 is interrupted by its scheduled reset. Record any discrepancy. This does not validate other task terminations or policy skill.
