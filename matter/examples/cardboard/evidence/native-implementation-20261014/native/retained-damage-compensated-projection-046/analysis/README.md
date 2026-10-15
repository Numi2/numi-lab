# Generic compensated projection: exact-pair audit

This CPU-only audit separates the derivative at the requested FP64 direction from the derivative along each plus/minus deformation pair actually stored by the native FP32 instrument. The native run remains a synthetic material-point instrument: it contains zero FEM steps and no calibration/physical qualification.

The native 046 receipt is successful (`exit_code=0`, `source_inputs_unchanged=true`). Its sealed source snapshot is `generic-damage-regional-block-032`; all 664 captured source-file hashes rechecked with no mismatches. The independent replay rebuilds the source constitutive program and asserts that its compiled constitutive fingerprint equals the fingerprint captured from the native run. It reconstructs each native perturbation from the exact stored FP32 values, widens those values to FP64, then projects both states independently. Two FP64 projection tolerances, `2e-11` and `2e-13`, are retained in the analysis output.

At the center state, native-to-FP64 relative stress/tangent errors were `4.381e-7` / `4.214e-7` for liner and `1.723e-8` / `5.047e-8` for medium. Native local projection residuals were `8.47e-11` and `7.20e-9`.

For the exact stored pair at requested epsilon `1e-5`, native AD versus the captured CPU-oracle pair FD was `9.57e-5` (liner) and `1.32e-4` (medium); native pair FD versus that captured oracle FD was `2.82e-6` and `1.71e-7`. A separate source recompile produced native-AD versus rebuilt-FP64-FD errors of `2.36e-4` and `2.71e-4`, also below `1e-3`. However, the realized direction differed from the requested direction by `2.367e-3`, so this pair does not pass the requested-direction gate.

There is no pair in this sweep that simultaneously meets the `1e-3` direction and local-derivative gates in both materials. At `1e-4`, the direction error is `2.580e-4`, while local AD/FD errors are `9.43e-3` and `1.329e-2`. At `3e-5`, the direction error is `1.334e-3`; liner local error is `8.56e-4`, while medium is `1.206e-3`. At smaller epsilons, the direction error grows as float32 quantization dominates, even when the exact-pair derivative itself compares well. The original native reference-epsilon `1e-3` gate remains not passed; the exact-pair results are a distinct diagnostic, not a replacement verdict.

The unquantized FP64 requested-direction sweep is internally consistent: centered-FD versus AD at `5e-7` is `2.374e-7` (liner) and `3.354e-7` (medium); the `5e-7` versus `2.5e-7` step spreads are `1.777e-7` and `2.512e-7`, and Richardson at `1e-6` differs from AD by `2.90e-10` and `1.73e-10`. This is an oracle self-check only. It does not make the native FP32 requested-direction result qualified.

The independent source recompile and captured CPU oracle return slightly different stresses for the same exact pair: at epsilon `1e-5`, maximum pair-stress relative differences are `3.33e-6` / `1.74e-6` (liner) and `5.26e-7` / `2.74e-7` (medium). The resulting captured-FD versus rebuilt-FD differences are about `1.4e-4` at `1e-5`; changing the recompiled oracle tolerance from `2e-11` to `2e-13` changes the same-pair FD by about `6.4e-10` or less. The audit retains this replay discrepancy rather than assigning it to tolerance or hiding it. At the smallest pairs, stress-level replay differences amplify under division by `2 epsilon`.

Historical capture 039 and its analysis remain unchanged and are referenced by their hashes. Those earlier FP32-only projection observations included medium residual `5.965e-6` and a larger `sbar11` parity discrepancy; 046 improves the center residual/parity but does not erase the earlier run or change the requested-direction verdict.

Reproduce the CPU analysis with:

```sh
python3 /Users/home/cardboard-implementation-20261014/native/retained-damage-compensated-projection-046/analysis/fp64-unquantized/analyze_unquantized_fp64.py --projection-tolerance 2e-11
python3 /Users/home/cardboard-implementation-20261014/native/retained-damage-compensated-projection-046/analysis/fp64-unquantized/analyze_unquantized_fp64.py --projection-tolerance 2e-13
python3 /Users/home/cardboard-implementation-20261014/native/retained-damage-compensated-projection-046/analysis/fp64-unquantized/summarize_compensated_projection.py
```

The source-bound numerical report is `generic-compensated-projection-fd-audit.json`; both FP64 tolerance outputs and all native capture, executable, metallib, source, and replay hashes are listed there. No GPU run, parameter change, or physical measurement was performed for this analysis.
