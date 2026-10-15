# Medium tangent finite-difference sensitivity at tets 1634 and 1638

This package independently reprojects the captured 15-state material point in MP90 and computes full 9x9 stress-tangent finite differences for two source-selected medium tets. It addresses whether the previously retained FP64 finite-difference mismatch at epsilon `1e-7` could be an inaccurate endpoint root or a finite-step effect near the smoothed Hill norm.

The sweep uses the exact captured `F_high_f32 + F_low_f32`, the captured parameter words widened from FP32, and the same accepted history for every plus and minus endpoint. MP precision is set to 90 digits before endpoint operands are constructed. Each endpoint is solved anew with the independent literal 15-state equations and safeguarded Newton line search; both scaled residual and the final unscaled Newton correction must be at most `1e-70`. Nine matrix directions perturb dimensionless row-major F entries. The separate Fdot sweep perturbs `F` by `epsilon_seconds * captured F_direction_local_per_s`; it must not be conflated with the nine dimensionless basis columns.

Run from any directory with the environment used for the report:

```sh
python3 mp90_endpoint_fd.py
python3 common-epsilon-1e-7.py
python3 summarize.py
python3 verify_package.py
```

The full MP90 sweep took about 297 CPU seconds. It imports the independent reference equations from package `medium-tangent-state-sensitivity-1634-001`; that package and its inputs remain unchanged. `manifest.json` binds all package outputs and source inputs by SHA-256.

At the shared epsilon `1e-7`, fresh MP90 endpoint projections give 9x9 relative errors of `6.3493266303622349e-6` and `1.53712534361450375e-6` for tets 1634 and 1638. The previous FP64 sweep gives `6.349292567568564e-6` and `1.5371274494284187e-6`; absolute differences between the error metrics are `3.4063e-11` and `2.1058e-12`. That comparison makes endpoint-root stopping error an implausible explanation for the old mismatch at that step size.

In the fresh MP90 sweep, central 9x9 FD errors decrease to `1.8218e-15` and `3.8219e-17` at epsilon `1e-14`. All basis endpoints converge, with no detected yield-sign or Fischer-Burmeister selection changes. Center multipliers are about `5e-17`, yield gaps are positive, and the captured q smoothing parameter is about `2e-6`. At epsilon `1e-8`, the largest endpoint q shift is about 40 times the q smoothing scale; by `1e-10`, it is about 0.061 and 0.027 times that scale. This is consistent with finite perturbations traversing local smoothed-norm curvature. It is a local interpretation for these two states, not a general error bound.

The separate captured-Fdot action sweep also converges with smaller epsilon and holds accepted history fixed. Its full per-side lambda, yield gap, q, branch, projection status, residual, and terminal correction are retained in `summary.json`; every basis-side record is retained there and in `report.json`.

The associated native controlled sweep 306 is included only as diagnostic context. It reports native stress/action differences along captured Fdot and has `physical_steps: 0` and `physical_validation: false`. It is not the independent endpoint oracle. This package does not establish a native finite-difference gate, all-point derivative behavior, material qualification, physical validation, or a general arithmetic noise floor.
