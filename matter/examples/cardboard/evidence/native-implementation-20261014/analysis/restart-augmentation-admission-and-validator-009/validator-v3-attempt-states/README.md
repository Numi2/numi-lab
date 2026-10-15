# Restart augmentation validator v3

This offline validator separates three facts that the prior checker conflated: a restart attempt was formed, that attempted mode was admitted, and a factor is available. A valid generation tuple plus a positive attempt count records an attempt even when `mode_valid=0`; in that rejected case `q` and `Aq` must have been cleared. `factor_valid` is reported independently and may describe the existing base/identity fallback. When that fallback is present, v3 reconstructs and checks its pivoted nonsymmetric RAP/LU rather than treating it as an augmented-mode success.

For no-attempt environments, the generation sentinel, zero attempt count, zero mode validity, zero raw norm/mode scale, and zero `q/Aq` remain required. `target_scale` is current base-RAP diagnostic metadata; it may be nonzero even if no restart attempt occurred. It is not itself an attempt latch.

The new `--require-attempt` option requires evidence that at least one environment formed an attempt, whether admitted or rejected. `--require-admitted-augmentation` requires an admitted mode with its q/Aq and RAP checks. The legacy `--require-augmentation` name remains an alias for the admitted-mode requirement. These options are supported in `--conditioning-only` mode. Existing hard caps on numerical thresholds remain in force.

Run the CPU tests with:

```sh
python -m unittest -v /Users/home/cardboard-implementation-20261014/analysis/restart-augmentation-admission-and-validator-009/validator-v3-attempt-states/test_restart_augmentation_validation_v3.py
```

The report `reports/capture341-rejected-attempt-v3.json` is bound to the source-100 N=28/Newton-27 diagnostic run. It verifies that the failed run recorded one environment's mode-generation attempt, then rejected the mode and cleared `q/Aq`; the other environment had no attempt. The valid factor bit in the attempted environment is checked as the base-plus-identity fallback, not as augmented-factor evidence. The rich base operator and fallback factor reconstruction passed their diagnostic checks. The run accepted zero physical steps, so this is not evidence of solver convergence or benefit.

That run also exposed a source-order defect: the target-scale capture was reset later in the same Newton setup. Frozen source101 moves the same reset dispatch before basis projection, base RAP construction, and target-scale capture. The `restart-augmentation-target-reset-101` package has a static ordering regression for that repair. Source101 has been build-checked by root; the matched native rerun is separate and must determine whether the mode becomes admissible. This v3 report intentionally remains bound to source100's pre-fix native output.

The tests also retain the source-182 AP mismatch, source-323 base-action control, source-337 no-attempt/fallback control, synthetic admitted branch, and threshold-cap checks. No failed or unattempted capture is promoted to an admitted restart mode.
