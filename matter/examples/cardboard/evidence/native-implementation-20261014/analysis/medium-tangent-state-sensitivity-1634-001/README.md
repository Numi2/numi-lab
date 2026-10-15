# Medium 15-state tangent sensitivity replay

This package resolves the reported medium-material tangent discrepancy at two captured states: env 0/tet 1634 (the worst row in the earlier all-point comparison) and env 0/tet 1638 (a nearby low-error control). It is a bounded CPU reference study, not a material or solver qualification.

`report.json` is the result artifact; `manifest.json` binds it and `replay.py` to the native capture, source snapshots, materials, native run receipts, executable/metallib hashes, and prior all-point comparison. The script consumes those exact external paths and verifies the captured high/low `F` words, accepted state, direction, parameter words, and capture checksum before computing.

The independent constitutive replay uses the literal 15-state Hill/tensile-hardening equations, with exact widened FP32 inputs and 70- and 90-digit mpmath roots/Jacobians. At each point, the precision runs agree far below the reported error scale. An independent FP64 safeguarded, scaled-correction solve takes two iterations, resolves the tiny positive multiplier, and its full 9×9 implicit tangent matrix agrees with the MP90 matrix to roughly `3e-16`. The stable Green-strain form `D=F-I; E=(D+D.T+D.T@D)/2` is bit-identical to direct `F.T@F-I` for these two center inputs; that result is local to these captured `F` values.

The bounded derivative check uses all nine row-major unit directions at each center. Independent FP64 central differences use `epsilon = 1e-5, 1e-6, 1e-7`; plus/minus branches, local solve iterations, and terminal residuals are retained. These perturbations stay on the elastic-smoothed side at both points. Adjacent epsilon matrices differ by about `2.7e-9` and `2.7e-8` relative, while their errors against MP90 are about `6.4e-6` and `1.6e-6`. This is a local epsilon-sensitivity record; it does not establish a general floating-point noise floor.

The native300 accurate-policy result supplies one captured directional action at each point, with relative errors to MP90 of `3.66e-8` and `4.42e-9`. The 9×9 native tangent matrix included in the data is a separate read-only cache capture from run283. Its command enabled `--compensated-source-paper-projection` but omitted `--accurate-source-paper-projection`; the cached-matrix comparison is therefore diagnostic-only and is not attributed to native300's accurate policy.

The earlier NumPy reference at tet 1634 stopped at iteration zero under its `2e-10` residual threshold with the multiplier still zero. Evaluating the MP tangent at that NumPy state reproduces the NumPy tangent, which localizes the old discrepancy to the projected state in this example. This does not prove the same cause for every former outlier. The all-point native300 result made the six earlier-invalid liner rows valid, but its one-direction FP64 comparison is not a per-row high-precision proof.

## Reproduce

Run on a Python environment with NumPy and mpmath, and with the source-bound capture and native artifacts still present at their recorded paths:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/medium-tangent-state-sensitivity-1634-001/replay.py
```

The script rewrites `report.json`; compare its output hash to `manifest.json` only when the bound inputs are unchanged. It performs no GPU work and edits no solver source.

## Errata and limits

The first exploratory MP numbers were invalid because mpmath operands were created before precision was raised from its 15-digit default. They are superseded by this 70/90-digit replay. An intermediate report also assembled 9×9 matrices by reshaping unflattened 3×3 actions; that output was invalid and is identified by its superseded report hash in `report.json`. The current assembler flattens each action before placing it in a matrix column.

Only two medium points are examined. Native all-nine-direction finite differences, all-nine-direction source rows, mesh convergence, complete nonlinear-step behavior, grade calibration, and physical material validation remain untested here.
