# Base-state residual Cauchy experiment — source125 capture 419

This CPU-only experiment uses the exact rejected source125 control-step 1 / Newton-48 capture and its guarded cached local FEM tangents. It constructs the free FEM operator, checks its transpose action, and exports candidate directions for a root-owned production nonlinear-residual replay. No production code, force law, tolerance, or acceptance logic was changed.

## Correct residual and direction semantics

The current Newton-center nonlinear residual is the captured pre-FGMRES RHS `b`, with `||b||₂ = 1.8444380e-4 N·s`. The captured FGMRES correction has `||δv||₂ = 188.1204 m/s`; the cached-tangent linear remainder after that correction is `b - Aδv = 2.4927731e-6 N·s` in the offline assembled operator and `2.5024084e-6 N·s` in the 422 native cached-action replay.

For the fallback from the **current center**, the merit is `φ(v)=1/2 ||b-A v||₂²`, so its gradient is `-Aᵀb`. The descent direction is `+Aᵀb`. The primary exported Cauchy `p` is an **alternative total correction direction from this center**, substituting for `δv`; it is not an increment added to `δv`.

The operator is unsymmetrized, shape `2538 × 2538`, with 71,046 nonzeros, 846 free nodes and 36 constrained nodes. Its relative skew norm is `9.43358e-5`. The direct elementwise forward and transpose actions match CSR at `2.57e-16` and `2.68e-16`; the independent adjoint dot identity error is `2.81e-16`. Captured fixed-node correction and action rows are zero.

## Linear-model predictions

The raw Euclidean Cauchy direction has `||p||₂ = 1.42948e-4 m/s`; the linear model predicts `||b-Ap||₂ = 1.60055e-4 N·s` (ratio `0.86777`, merit decrease `24.70%`).

The column-equilibrated Cauchy direction uses `Sjj = 1 / ||A[:,j]||₂`, forms `B=A S`, `q=Bᵀb`, and sets `p=S ((qᵀq)/||Bq||²) q`. It has `||p||₂ = 3.34163e-3 m/s`, `||p||∞ = 1.63011e-3 m/s`; the linear model predicts `||b-Ap||₂ = 1.29575e-4 N·s` (ratio `0.70252`, merit decrease `50.65%`). Its simple position-rounding estimate is `dt ||p||∞ = 1.63011e-7 m`; 569 free nodes have at least one position coordinate word change under a single FP32 rounding estimate. This is not the production candidate-preparation order.

For the alternate interpretation, the package also evaluates the **additive** total correction `δv+p`: the linear model predicts residual `1.31365e-4 N·s` (ratio `0.71222`, merit decrease `49.27%`). Its total position increment is approximately `9.43e-3 m`. This result is separate from the primary substitution direction.

A norm trust-radius family is included as a diagnostic scale sweep, with radii `1e-6`, `1e-5`, and `1e-4` times `||δv||₂`. The first two cap the column-equilibrated Cauchy vector; the last does not. These are not calibrated physical trust radii.

## Evidence boundary and replay

The 422 and 424 diagnostics bind to this same capture. The native action replays are close but not bitwise identical to the independent assembled action; the differences are retained and no numerical noise floor is inferred. The 424 endpoint sweep matched its retained endpoints bitwise but did not qualify an FD derivative.

The capture is a rejected Newton candidate with zero physical transactions. This compiled scenario has zero rigid proxies, contact pairs, fields, and cohesive faces. The report only covers its FEM block. Predicted `b-Ap` values are linear-model results, not nonlinear trial residuals, Armijo outcomes, or physical qualification. Root should inject the exported primary `base_column_equilibrated_cauchy` total-node correction into a production residual replay at the captured center; the raw and trust-radius variants are also exported. `delta_plus_base_column_equilibrated_cauchy.total-node-correction.float32x4.bin` is the separately labeled additive option.

## Reproduction

The script revalidates capture 419 against its run prediction, receipt, stderr and exact source125 hashes. It also checks the 422 and 424 receipts and capture bindings. It refuses to overwrite an existing output directory containing a report or manifest. Use a fresh destination:

```sh
python3 -W error /Users/home/cardboard-implementation-20261014/analysis/captured-base-state-cauchy-419-001/run_experiment.py --output /tmp/captured-base-state-cauchy-419-recheck
```

`A_free_csr.npz` and the little-endian binary files are listed with dtype, shape and SHA-256 in `report.json` and the package manifest. Free-vector files are in node-major xyz order over sorted free node IDs; full-node float32x4 files are N×4, xyz correction in lanes 0–2 and zero lane 3, with constrained rows zero.
