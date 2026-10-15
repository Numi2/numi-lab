# Restart augmentation validator v4: A-conjugacy

This package corrects the v3 check of the source-101 two-environment, Newton-27 restart readback. The production restart projection solves the base nonsymmetric coarse system with right-hand side `PᵀAq`, then updates both `q` and `Aq`. Its invariant is therefore `PᵀAq ≈ 0`. Euclidean `Pᵀq ≈ 0` is not an invariant for this projection and is reported only as a diagnostic.

The v3 344 report is retained unchanged, but its `augmentation_capture_inconsistent_diagnostic` verdict is superseded: it treated `Pᵀq = 0` as a gate and rejected the valid A-conjugate mode. The v4 unit tests include a nonsymmetric counterexample where the solver projection has small `PᵀAq` but nonzero `Pᵀq`.

The v4 validator checks captured per-environment attempt/generation state, current base target scale, reconstructed base `AP`, fresh independent `Aq`, mode norm scaling, FP64-widened `PᵀAq`, a labeled non-bitwise FP32 lane-reduction emulation, source-FP32 energy sign resolution, target-scaled `|qᵀAq|`, and the augmented nonsymmetric RAP/LU reconstruction. It does not claim exact Metal reduction parity. A scale floor of `32 * eps32 * sum(abs(q_i * Aq_i))` is reported as a conservative sign-resolution diagnostic, not as a measured physical or solver error bound.

Run the tests with:

```sh
python -m unittest -v test_restart_augmentation_validation_v4.py
```

The bound report `reports/capture344-a-conjugate-v4.json` uses the native run's operator capture and conditioning sidecar, source-build receipt, result, receipt, and manifest. It observes one admitted augmentation in environment 0 at generation `[0,0,27,0]`; the quiet environment has no attempt. For the admitted mode, the captured `PᵀAq` relative norm is `2.95e-9` in FP64 widened arithmetic and `3.57e-9` in the lane-strided FP32 emulation, while Euclidean `Pᵀq` is `0.0444`. The captured and fresh `Aq` energy magnitudes match the target scale to below `1e-7` relative; the sign is resolved by the stated source-FP32 scale. The fresh-Aq relative discrepancy is `3.24e-5`, base-AP maximum relative discrepancy is `5.09e-7`, and augmented RAP/LU reconstruction passes its existing `2e-4` threshold.

The native run itself exited with failure after zero accepted physical steps. This report validates the internal readback for that one captured restart generation; it does not demonstrate convergence benefit, a complete physical trajectory, or qualification of the cardboard model. Preserve the separate on/off and full-cycle native outcomes.
