# Native source15 local-accuracy point summary

Run 285 completed with 16/16 pointwise checks passing. This is a bounded material-point replay; it is not a full solver, mesh, or physical validation.

## Provenance

- Receipt: exit `0`, timed out `False`, unchanged inputs `True`, wall 22.913 s.
- Frozen source snapshot manifest SHA-256: `769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752`.
- Checker header compile-time base-lineage token: `98fe026ee2007982e48f89547a435a2957db3978fb3730e008fdbb99d08f5704`. This is distinct from the frozen-source manifest hash above.
- Source material SHA-256: `76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9`; captured input JSON SHA-256: `4147dfbdd75587dd932d9e22d210b3725bbf8d38f24c0494151f5474dbf9d66b`.
- Prediction binds 782 inputs (772 source, 9 runtime, 1 external); all match. Build record SHA-256 `120c809d6d0ea9a0c1ed27a25df3c20748f186f62c64164eb54f09e745524c1b`; build success, no changed sources, and all 73 listed artifact hashes verified.
- Direct native artifacts:
  - `libmetalrobo.dylib`: `b88ee5fdacb91a8909d3ee97c9a86da090aeac0cedc34fef9d0dac9c7d50e64d`
  - `numi-matter-cardboard-bilinear15-captured-point-check`: `0ec673f6a60870233f115fca980964ab4840b43d5e179261b61c020a2467d805`
  - `NumiMatter.metallib`: `51c38a9b27b750227d61fae7f38a09de4030ede739d00e29557adfcf3db27bca`
  - `cardboard_paper_check.metallib`: `a129573c4f5f67de208ddac16afaa032c09d595cc2b9f4c7137bcf37b7e0bbc6`
  - `MetalRobo.metallib`: `61a548c69e6ec9c5c95480d54dbcf2fd851744e2a5e799397c59960e2fb27334`

## Results

| Capture | Cases | Branches | Default max stress/tangent/state relative error | Opt-in max stress/tangent/state relative error | Opt-in iterations |
|---|---:|---|---|---|---|
| N0 | 8/8 pass | no_resolved_multiplier_increase | 0/0/3.30431482e-08 | 4.26473663e-08/0.00080227165/8.26481775e-09 | 9–15 |
| N63 | 8/8 pass | multiplier_increased | 0/0/3.51759049e-08 | 3.80894298e-08/2.72136785e-08/1.61747107e-10 | 6–7 |

The opt-in policy stayed below the 1e-3 pointwise relative gate in each nonzero response. Its largest directional tangent error was 0.00080227165 on N0 and 2.72136785e-08 on N63. Default stress and tangent matched the captured production action exactly in these rows; state differences were small but nonzero.

Repeated identical evaluations differed by zero in the recorded rows. This is only a determinism observation; it is not an FP32 accuracy or representability bound. The N0 rows are classified `no_resolved_multiplier_increase`, while N63 rows are `multiplier_increased`; these are pointwise reference branch indicators, not evidence of branch coverage across all perturbation directions.

## Directional coverage and limits

Each of the 16 points tests one captured tangent direction, not the complete 9x9 local derivative. The exact same point inputs are sufficient to extend the test: use the nine matrix basis directions and the runtime relation `rateDirection = deformationDirection / dt`, producing nine independent FP64 reference actions per point. This would be a new stronger test and should preserve each direction’s projection validity/branch diagnostics. No such basis sweep was executed here.

The native run used zero physical solver steps and reports `physical_validation=false`. It does not qualify the full coupon solve, mesh convergence, or a calibrated production default.

## Files

- Original immutable run: `/Users/home/cardboard-implementation-20261014/native/captured-source15-local-accuracy-285`
- Verification script: `/Users/home/cardboard-implementation-20261014/analysis/captured-source15-native-285-summary/verify_native285.py`
- Machine-readable summary: `/Users/home/cardboard-implementation-20261014/analysis/captured-source15-native-285-summary/summary.json`
