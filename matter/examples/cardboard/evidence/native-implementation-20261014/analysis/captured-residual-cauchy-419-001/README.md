# Captured residual-merit Cauchy experiment — source125 capture 419

This package is a CPU-only, source-bound experiment on the rejected source125 control-step 1 / Newton-48 FEM operator capture. It computes a residual-merit gradient and candidate correction increments for a later native production-residual replay. It does not alter the Matter source or certify a nonlinear accepted step.

## Result

For environment 0, the offline assembled free FEM block is an unsymmetrized `2538 × 2538` matrix with 71,046 nonzeros. It has 846 free nodes and 36 constrained nodes; fixed correction and captured action rows are zero. The assembled operator has relative skew norm `9.43358e-5`, so the transpose was applied explicitly rather than assuming symmetry.

With `r = b - A δv`, `φ = 1/2 ||r||²`, and the production convention `A δv ≈ -dR/d(δv)`, an increment `p` has first-order merit derivative `-rᵀ A p`. The descent direction is therefore `+Aᵀr`.

- Assembled residual: `2.4927731e-6 N·s` (`1.35151e-2` of `||b||`). The 422 native cached-action replay gives `2.5024084e-6 N·s`; capture-native and replay action differences from the independent assembled action are retained separately.
- Raw Euclidean Cauchy increment: `||p||₂ = 2.07317e-6 m/s`, `||p||∞ = 7.92624e-7 m/s`. The linear model predicts `||r-Ap||₂ = 1.94321e-6 N·s`, a `39.23%` merit decrease.
- Column-equilibrated Cauchy increment, using `Sjj = 1 / ||A[:,j]||₂` and solving in `p = S z`: `||p||₂ = 1.07553e-4 m/s`, `||p||∞ = 7.15648e-5 m/s`. The linear model predicts `||r-Ap||₂ = 1.58330e-6 N·s`, a `59.66%` merit decrease. A nominal trust cap of `1e-6 ||δv||₂` is nonbinding for this direction; it is illustrative, not a calibrated physical radius.
- The alternate captured-action residuals preserve the same gradient direction: gradient cosine versus the assembled residual is `0.99918` for the capture-native action and `0.99901` for the 422 cached action. Their predicted column-equilibrated residual ratios are `0.63558` and `0.63570`, respectively.

A seeded matrix-free element gather agrees with CSR `A x` and `Aᵀ y` at relative errors `2.57e-16` and `2.68e-16`; its adjoint dot identity error is `2.81e-16`. These check the assembly/gather and transpose path, not constitutive calibration.

The column-scaled step has an estimated `dt ||p||∞ = 7.16e-9 m`; the simple single-add FP32 position proxy changes 473 free-node position words. This is only a representability estimate and does not reproduce production's accepted/current position composition order.

## Limits and next use

The 424 endpoint sweep was bitwise consistent with its retained endpoints but did not qualify a finite-difference derivative; this analysis does not assign a noise floor. The captured scenario has zero rigid proxies, contacts, fields, and cohesive faces, so the assembled block is only the complete FEM slice for this specific source-bound test. Both the capture and the 422 replay belong to a rejected physical attempt with zero physical transactions. Linear residual reduction is a candidate for native replay, not evidence of nonlinear descent, Armijo acceptance, or a completed cycle.

`run_experiment.py` revalidates the capture against its prediction, receipt, stderr, and source125 hashes before reassembly. It refuses to overwrite an existing report or manifest. To rerun, write to a fresh directory:

```sh
python3 -W error /Users/home/cardboard-implementation-20261014/analysis/captured-residual-cauchy-419-001/run_experiment.py --output /tmp/captured-residual-cauchy-419-recheck
```

`report.json` contains all source/native input bindings, operator dimensions, residual variants, matrix-free checks, direction metrics, and artifact names. `A_free_csr.npz` and the little-endian vector files are provided for the root-owned native residual replay. Each direction file contains an increment to the captured correction in m/s (not a replacement total correction); fixed node rows in the full-node float32x4 files are zero.
