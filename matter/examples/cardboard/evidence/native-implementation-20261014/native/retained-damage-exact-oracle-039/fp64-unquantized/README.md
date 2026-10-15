# FP64 tangent and quantization audit for native capture 039

This is a CPU-only numerical follow-up to the retained-matrix-damage capture. It compares three different questions without merging their verdicts:

1. Does an independent FP64 implementation differentiate itself consistently on unquantized FP64 inputs?
2. For the exact plus/minus deformation-gradient values that the native FP32 instrument actually stored, does an independent FP64 replay reproduce the finite difference, and does the midpoint tangent match the realized direction?
3. Does the requested direction survive FP32 quantization closely enough to support a requested-direction tangent claim?

The analysis is in `analysis.json`; the reproducible driver is `analyze_unquantized_fp64.py`, which generates and compiles the two region-specific C++ oracle drivers beside it. Its `source_bound_inputs` section and per-case hashes bind the capture JSON, captured executable/metallib, source constituent material, hardening data, builder/oracle headers, and compiler library. The input captures are retained at the absolute paths in the JSON. No physical parameters were changed, no GPU work was done, and no native result was modified.

## Results

The independent FP64 oracle is internally consistent at a `5e-7` requested-direction perturbation. Centered-FD versus AD errors are `2.37394e-7` for liner and `3.35416e-7` for medium; half-step spreads are `1.77731e-7` and `2.51250e-7`. Richardson comparisons are smaller still. This verifies the FP64 oracle's local derivative calculation only.

For the exact captured FP32 plus/minus pairs at `epsilon=1e-5`, native finite-difference stress versus the independent FP64 replay of those same stored values differs by about `2.04e-3` for both regions. At smaller epsilons this grows as FP32 realization error dominates. The FP64 midpoint tangent agrees closely with the FP64 finite difference of the same pair (`9.52e-5` liner, `1.34e-4` medium at `1e-5`), but the realized midpoint direction itself differs from the requested center direction by `1.48e-3` and `1.56e-3`. Thus the requested-direction gate remains **inconclusive**, not a pass or a demonstrated constitutive tangent failure.

At the unquantized requested direction, the `1e-6` oracle self-check passes only at the smaller steps: `5e-7` gives `2.37e-7` and `3.35e-7`; `1e-6` gives `9.49e-7` for liner but `1.34e-6` for medium. This is a diagnostic self-check and does not qualify native FP32 tangent accuracy.

The projection-state comparison separately identifies a precision issue in the generic 30-state path. Medium's native reported projection residual is `5.9654e-6`, while the FP64 re-projection residual is `2.22e-15`; its largest state discrepancy is `1.5326e-6` in `sbar11` (state L2 `1.9768e-6`). Liner's corresponding native residual is `1.1642e-10` versus FP64 `6.4083e-13`, and its largest state discrepancy is `1.7544e-6` in `sbar11`. The center stress/tangent values reconstruct closely from the captured FP64 oracle inputs; that does not erase the native medium state/residual mismatch.

## Limits

This evidence is CPU oracle analysis over two captured synthetic material points. It is not native Metal qualification, a mesh or box result, a physical calibration, or a determination that all solver states share this behavior. The results keep FP32 input quantization, native finite-difference output error, generic 30-state projection residual, and FP64 oracle self-consistency as separate observations. No tolerance was relaxed and no law or parameter was changed.

To reproduce from the retained environment, run:

```sh
python3 /Users/home/cardboard-implementation-20261014/native/retained-damage-exact-oracle-039/fp64-unquantized/analyze_unquantized_fp64.py
```

