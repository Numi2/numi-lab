# Exact-state FP64 replay of selected paper tets

This is an offline constitutive replay of the retained Newton-23 diagnostic captures 216, 218, and 220. Each native run rejected step 0 (`result.status=failed`, `accepted_steps=0`, runner exit 2), but the complete selected-Newton captures report `production_inputs_bitwise_unchanged=true`. The comparison therefore diagnoses the rejected transaction's local material and perturbation path; it does not establish an accepted step, solver convergence, or physical validation.

The six env0 targets are liner tets 1388 and 441, medium tet 633, and active/near-yield controls 518, 702, and 776. For each capture the audit retains the raw capture path/SHA, source build, prediction and receipt hashes; replay input/output; exact source and material hashes; and all center/side comparison fields in `exact-pair-replay-audit.json`. The capture source build is snapshot 069. Replay source 071 has matching SHA-256 for the paper replay law, projection scaling, compensated bytecode, and liner/medium source material files. The per-run copied `.nmatter` files also match those bytes. Run captures all retain unchanged-input receipts; each independent replay has 42 valid rows and no invalid row.

## Numerical contract

Native JSON floats are reconstructed as FP32 and widened exactly. For every center and plus/minus endpoint, the replay preserves F high/low and candidate-state high/low separately, then sums each pair in FP64. Accepted state, nominal direction, parameters, and native stresses/tangents are also interpreted as FP32 words widened to FP64. A separate offline replay-reader extension accepts two-word directions: the exact endpoint direction `(F_plus - F_minus)/(2 epsilon dt)` is encoded as FP32 high plus FP32 residual low. This avoids rounding that derived direction to a single FP32 value a second time.

The audit distinguishes (1) native stress vs independent FP64 stress at the exact captured candidate and exact endpoint F, (2) native directional tangent vs independent tangent at the captured candidate, (3) centered differences from the native endpoint stresses, (4) centered differences from independent FP64 stresses at those same native F endpoints, and (5) the same independent finite difference against tangents along nominal and exactly realized endpoint directions. The local force check maps each stress to the captured element reference geometry and compares per-element nodal force; it is not a global reaction-force validation.

## Result

At the centers, all six tets' native stress and directional tangent match the independent FP64 evaluator closely. Across the three captures and six targets, the largest observed center/side candidate-stress discrepancy is 0.01510 Pa, the largest relative candidate-tangent discrepancy is `5.09e-8`, and the largest center element-nodal-force discrepancy is `2.16e-8 N`. Captured projected states differ from independent re-projections by at most `2.32e-10` in a state component. These are deterministic evaluator/rounding discrepancies over these samples, **not** a repeatability-derived stochastic noise floor.

For the three smooth elastic targets, the exact native plus/minus F endpoints cease to follow the captured nominal direction as epsilon shrinks. The endpoint `F` infinity-norm difference is `6.2918e-7` for the 216 pair, `3.8990e-8` for 218, and `2.0521e-9` for 220; nominal values vary by tet because the captured direction magnitudes differ. The full per-tet endpoint error is recorded as `native_F_endpoint_delta_inf_error` and `exact_realized_F_direction_vs_nominal`.

At the smaller 218 epsilon, nominal-direction centered-FD mismatch versus the native tangent is 3.47% for tet 441, 5.37% for tet 633, and 7.83% for tet 1388. Yet FP64 finite differences at the *exact same captured endpoint F values* agree with the independent tangent along the exact two-word realized direction to `1.34e-8`, `3.87e-8`, and `2.12e-8`, respectively. Realized direction differs from nominal by 0.306%, 1.519%, and 1.690% for that pair.

At the smallest 220 epsilon, realized-direction deviations rise to 8.36% (441), 41.35% (633), and 16.73% (1388); nominal-direction FD errors are 6.78%, 106.93%, and 59.89%. Replaying the exact endpoint pairs against the exact realized-direction tangent reduces the independent FP64 discrepancies to `1.53e-8`, `4.90e-7`, and `6.42e-8`. Thus the first mismatch is already present in the *captured F endpoint perturbation relative to nominal dF*, before constitutive stress/tangent evaluation. At the smallest steps the stress changes are also small enough that FP32 endpoint-stress rounding matters: e.g. medium tet 633's 220 pair has a 0.195 Pa native stress difference and a 0.00521 Pa native-minus-FP64 difference in that response. That is an observed deterministic response discrepancy, not a calibrated random-noise bound.

Yield controls confirm why branch-aware admission matters. Center trial q is 2.30947 for tet 518, 1.02146 for tet 702, and 1.000229 for tet 776. In 216, the sampled pairs for 518, 702, and 776 cross elastic/plastic trial branches. In 218, 702 stays plastic but 776 crosses; 518 stays plastic. In 220 the sampled sides remain plastic, but tet 776's minimum side trial q is only 1.0000093, so that pair is still a near-yield control, not evidence about a well-separated smooth branch. The full q and plastic-multiplier triples are in the JSON. Branch-crossing centered differences are reported as observed responses only and are not used as smooth-derivative tests.

## Verdict and limits

The evidence does **not** support a defect in the local 13-state projection, stress, or directional tangent for these samples: exact-input center stress/tangent agree with the independent FP64 law, and exact endpoint stress finite differences agree with the FP64 tangent when evaluated along the endpoint direction that was actually represented. The discrepant requested-direction FD comparison is first localized to endpoint `F` realization/input quantization; the smaller 220 endpoint responses also expose a measurable FP32 stress-response discrepancy. This does not determine which upstream node-perturbation or kinematic reconstruction operation caused the `F` mismatch, and the captures are from rejected solver transactions. No native accepted-step, full-operator, material calibration, box, or physical qualification claim is made.

## Reproduction

The replay tool is CPU-only. Its isolated source extension and CMake build are included. The exact command used for the audit is:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/smooth-target-pair-audit-228/replay_exact_targets.py
```

That script references the immutable captures under `/Users/home/cardboard-implementation-20261014/solver/graph-schwarz-*{216,218,220}*`, source snapshots 069/071, and the replay executable built under `analysis/smooth-target-pair-audit-225/build`. It writes only a fresh output directory and fails if it already exists. CMake used Xcode AppleClang 21.0.0, Ninja, Release, CPU target `cardboard_solver_pair_replay_realized`; no GPU or native simulation was run for this audit.
