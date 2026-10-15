# Tet 1186 MP90 direction adjudication

This bounded CPU analysis replays the captured environment-0 tet 1186 material point from rejected Newton iteration 23 of world `6de3475316b7e8b9`. It keeps the captured FP32 material parameters, accepted 15-state history, high/low deformation-gradient words, timestep, and direction fixed. It uses the independent source-equation implementation and MP90 arithmetic; no material law, solver setting, or physical parameter is changed.

The center and both endpoint projections converge for target deformation increments `||ΔF||F = 1e-7` and `3e-8`, using 90 decimal digits, a 1e-70 residual/correction gate, and at most 48 Newton updates. The center root's scaled residual is `3.410449816e-87`; the dynamic correction is `4.773554127e-94`. MP90 center and endpoint states agree with the v3 FP64 projections to relative L2 between about `5.1e-14` and `2.8e-13`.

For `1e-7`, the MP90 central finite difference agrees with the tangent applied to the realized endpoint rate direction at relative L2 `5.89e-12`, while comparison to the nominal captured direction is `9.60e-7`. At `3e-8`, those values are `5.30e-13` and `1.586e-6`. The endpoint pair's realized rate direction differs from the nominal one by relative L2 `1.51e-9` and `3.07e-9`, respectively; its tangent action differs by `9.60e-7` and `1.586e-6`. This is a point-specific sensitivity result, not a universal noise floor.

The v3 endpoint stresses differ from MP90 by only `4.6e-8` to `2.5e-7 Pa`; division by the small perturbation interval amplifies that into v3-FD versus MP90-FD relative differences of `2.56e-7` and `5.25e-7`. The captured material point is extremely close to yield (`yield_gap ≈ 9.576e-12`), although the recorded sign and smoothed FB-selection predicates remain the same at center and endpoints. This diagnoses the specific row's epsilon-sensitive comparison; it does not erase the uniform 1728-row v3 result's one `1e-6` exceedance at the common `1e-7` step.

## Limits

This is one captured tet and two perturbation magnitudes. It does not qualify native FP32, all material points/directions, the coupled solve, or the physical model. Do not promote one selected epsilon to a global threshold or revise the uniform v3 gate from this point-specific result. Original failed attempt files remain under `output-001/`; completed root results are `output-002/result.json` and `output-003/direction-roundoff.json`.

## Reproduction and verification

From this directory, run:

```sh
python3 verify.py
```

The MP90 root run was bounded to 115 seconds and completed in 2.93 seconds. Its runner is `replay_mp90.py`; the realized-direction and endpoint-stress decomposition is `analyze_direction_roundoff.py`. Their source inputs and result hashes are recorded in `manifest.json`. The verifier checks immutable file hashes, case identity, root completion, the two target magnitudes, and the numerical assertions summarized above.
