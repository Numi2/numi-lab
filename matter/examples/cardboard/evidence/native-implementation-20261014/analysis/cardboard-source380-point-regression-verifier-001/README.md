# Source-380 exact-point numerical regression verifier

This opt-in verifier checks the exact published env0/tet674 source-380 fixture against a completed native point diagnostic, its receipt, the source package and material asset bytes, the separate native F-high/F-low/Fdot readback, and frozen independent FP64 state/stress/tangent vectors. It checks the existing local residual and scaled-correction thresholds (`2e-8` and `2e-8*(1+||state||∞)`), accepted dimensionless Armijo merits, row scales, and direct/tangent/trace-on-off bit parity.

It is a diagnostic regression only. It does not establish material calibration, finite-difference/noise-floor qualification, a physical step, or full coupon convergence. The tangent vector covers one declared nonzero deformation direction.

Run the check with Python 3.10 and NumPy:

```sh
python3.10 verify_point.py \
  --fixture /Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014/analysis/source380-generalized-material-point-001/native_point_input_674_v2.json \
  --package /Users/home/cardboard-implementation-20261014/solver/explicit32-certified-coupon-380/output/compiled.nmatterpack \
  --medium-asset /Users/home/cardboard-implementation-20261014/snapshots/source-scaled-material-merit-118/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter \
  --reference-vectors /Users/home/cardboard-implementation-20261014/analysis/cardboard-source380-point-regression-verifier-001/fp64-reference-vectors-674.json \
  --receipt /Users/home/cardboard-implementation-20261014/native/dimensionless-material674-point-390/receipt.json \
  --result /Users/home/cardboard-implementation-20261014/native/dimensionless-material674-point-390/output/source380-point-run.json \
  --points /Users/home/cardboard-implementation-20261014/native/dimensionless-material674-point-390/output/paper-points-source380_exact_compiled_budget32.jsonl \
  --trace /Users/home/cardboard-implementation-20261014/native/dimensionless-material674-point-390/output/tet674-env0-source380_exact_compiled_budget32-trace.json \
  --f-readback /Users/home/cardboard-implementation-20261014/native/exact-source380-material674-point-382/output/budget32.json \
  --f-readback-manifest /Users/home/cardboard-implementation-20261014/native/exact-source380-material674-point-382/output/point-replay.jsonl
```

The expected FP64 vectors are frozen at the root-rechecked point from `dimensionless-material674-point-390-strict-reference-addendum-001`. Inputs and outputs are hash-bound by the manifests. This package is external analysis tooling; it does not change CMake or default CTest/GPU behavior.
