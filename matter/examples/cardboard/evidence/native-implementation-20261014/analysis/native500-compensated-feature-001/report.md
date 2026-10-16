# Native500 compensated VT audit

This is a bounded CPU replay of source176/native500 contact slot 7, not a production patch or a solver qualification. It asks whether carrying two FP32 words through vertex-triangle closest-point projection and barrier-energy evaluation would turn the recorded contact Armijo failure into an accepted step.

## Finding

For the exact captured FP32 endpoint words, compensated projection resolves separation accurately but does **not** make the contact-limiter path satisfy Armijo. All eight contact_limiter endpoints remain on the edge_bc branch and have positive exact barrier-energy change. In contrast, the corresponding FEM-applied endpoint path has positive changes at alpha 1, 1/2, and 1/4, then negative changes from alpha 1/8 through 1/128; those latter exact-energy trials satisfy the recorded Armijo bound.

The source paths differ at FP32 rounding. The limiter tests candidate.position + alpha * (dt * correction). FEM application instead adds alpha * correction to candidate velocity, then rebuilds position from accepted position/displacement and updated velocity. On this fixture the two position paths are visibly different even though both use the same accepted/candidate/correction input words. Thus separation cancellation is real, but correcting separation alone is not a repair: the limiter's tested path is the primary discrepancy against the endpoint that FEM actually applies. A safe implementation must use the same FEM endpoint construction for energy trials and must also certify the sweep to that exact endpoint; changing the energy calculation while retaining the old idealized sweep is insufficient.

The independent continuous derivative along the ideal unrounded direction is -4.447532027789818e-15 J/alpha, close to the recorded source derivative -4.447521825378786e-15 J/alpha. The center gradient dotted with the realized rounded endpoint jump, however, agrees in sign and magnitude with the exact finite energy change on each path. This explains the apparent sign conflict without attributing it to a different material branch.

## Eight alpha trials

The energy values below use the high-precision distance from the exact endpoint words and Decimal evaluation of the source barrier formula. Armijo uses the native-recorded center derivative and coefficient 1e-4; no gate or tolerance was changed. “Margin” is trial energy minus the Armijo right-hand side, so a negative value passes.

| alpha | limiter ΔE (J) | limiter margin (J) | FEM-applied ΔE (J) | FEM-applied margin (J) |
|---:|---:|---:|---:|---:|
| 1 | +2.7714335e-15 | +2.7718782e-15 | +2.2438342e-15 | +2.2442789e-15 |
| 1/2 | +1.3866731e-15 | +1.3868955e-15 | +8.8586100e-16 | +8.8608337e-16 |
| 1/4 | +6.9334478e-16 | +6.9345597e-16 | +1.9253279e-16 | +1.9264398e-16 |
| 1/8 | +3.4667424e-16 | +3.4672984e-16 | -1.5413754e-16 | -1.5408194e-16 |
| 1/16 | +1.7380002e-16 | +1.7382782e-16 | -3.2701165e-16 | -3.2698385e-16 |
| 1/32 | +8.6899969e-17 | +8.6913868e-17 | -4.1391165e-16 | -4.1389775e-16 |
| 1/64 | +4.3450019e-17 | +4.3456968e-17 | -4.5736158e-16 | -4.5735463e-16 |
| 1/128 | +2.0338489e-17 | +2.0341964e-17 | -4.7954863e-16 | -4.7954515e-16 |

For the current center, native separation is 8.652231372252572e-6 m; exact-word Decimal separation is 8.652200537702782e-6 m. The two-word result differs from Decimal by 3.015e-18 m, while native FP32 separation differs by 3.083e-11 m. Across all 24 slot-7 endpoints (current, limiter, FEM-applied for eight alpha values), the maximum two-word/Decimal distance error is 3.222e-18 m, all branches agree (edge_bc), and the maximum native separation error is 3.0835e-11 m.

The exact mollifier calculation stays on the unit plateau (phi=1) for these endpoints, so the comparison isolates the VT separation/barrier path. The contact and FEM-applied endpoint paths are both retained in the JSON outputs. For limiter trials, center-gradient dot realized node jump is positive from 2.77156e-15 J at alpha 1 to 2.03385e-17 J at 1/128 and agrees with the exact energy increase to within the small second-order remainder. FEM-applied alpha 1/8 has a negative -1.54135e-16 J realized first-order change, matching its exact energy decrease.

## Implementation guidance for the follow-on

Use one production-order FEM position helper in both nm_fem_apply_solution and nm_fem_armijo_prepare_candidate, and call it from the FEM branch of nm_contact_limit_deformable_line_search for each tested alpha. It must preserve the existing due/object/fixed decisions and exactly reproduce:
1. candidateVelocity + alpha * correction;
2. deltaVelocity relative to accepted velocity;
3. ordinary position: acceptedPosition + dt * updatedVelocity;
4. reference-displacement position: restPosition + (acceptedReferenceDisplacement + dt * updatedVelocity).

The limiter then derives mapped-feature endpoints from those node endpoints using the existing barycentric map. Its CCD sweep must start at the current candidate feature and end at the helper's exact applied feature. MPM is a separate existing path and must remain unchanged. The helper must not update material/history fields as a side effect; FEM apply retains its existing field and solution-scaling behavior.

This audit does not prove that a linear CCD chord is a conservative bound for any future interpolation policy, nor does it qualify full contact/CCD behavior. Re-run direct negative collision and mapped-feature controls after integration. The exact candidate/helper overlay is a separate next artifact; no production files were edited here.

## Reproduction

Run from /Users/n on the Mac mini:

- venvs/cardboard-reference-185/bin/python3 .../analysis/native500-compensated-feature-001/closest_vt_dd.cpp was compiled as recorded in the package receipt with the include root at the source176 snapshot. The binary is closest_vt_dd.
- ./closest_vt_dd .../native/contact-barrier-energy-replay-500/output.bin > dd-output.jsonl
- venvs/cardboard-reference-185/bin/python3 decimal_reference.py > decimal-reference.jsonl
- venvs/cardboard-reference-185/bin/python3 energy_armijo_analysis.py > energy-armijo.json
- venvs/cardboard-reference-185/bin/python3 add_slope.py
- venvs/cardboard-reference-185/bin/python3 realized_gradient.py > realized-gradient-summary.json

The native output is a preserved read-only replay (physical_transactions=0, ccd_qualification=false) bound to source closure 8427a7f2...03ca41, host 97c0b495...77b5f4, metallib 5e91e698...4a9f9e, and input manifest dc62eb04...83c70. No GPU was run for this audit.
