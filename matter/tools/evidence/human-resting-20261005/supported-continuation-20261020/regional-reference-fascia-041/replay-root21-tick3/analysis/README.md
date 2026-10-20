# Root 21, tick 3, VF Armijo replay

This diagnostic replay reproduces the terminal failure at root 21, microtick 3, selected deformable contact slot 2623. It is bound to the exact binary, source base, invocation, provenance, and stderr hashes in [witness-summary.json](witness-summary.json). CSV DictReader comparison confirms all 21 data rows are identical; the entire native and replay CSV files are byte-identical (SHA-256 4e1e3b038f148585576dce7f72ff141d4e764dfbe4a1912736e43c00d4406414). The replay also reproduces the failure tuple.

The selected contact is vertex-triangle kind 0, object 19, nodes 10165, 10162, 10167, 9917, with reference-local coordinates at origin (0.0370209105, -0.579743266, 0.114348516). The triangle witness is on its first edge: weights (1, -0.461908221, -0.538091779, 0).

All 12 tested alphas are full-fraction CCD-certified, pass the initial-touch and finite-barrier checks, and are rejected specifically by barrier Armijo. Base energy is 3.712967e-8, base directional derivative is -4.69789587e-15, and base trace separation is 8.48235504e-6 m. The captured JSON high/low coordinates were rehydrated to binary32 before summing in binary64, then projected onto the triangle edge. That edge distance decreases for every alpha, while every computed FP32 trial energy exceeds base. Thus this is not only a scalar-energy rounding plateau: the actual trial geometry closes even though the stored analytic directional derivative predicts slight descent.

| Alpha | Trial energy minus base | Trace separation minus base | FP64 edge separation change |
|---:|---:|---:|---:|
| 1 | +2.06e-13 | -2.73e-12 | -2.13e-12 m |
| 0.5 | +1.49e-13 | -1.82e-12 | -1.03e-12 m |
| 0.25 | +2.06e-13 | -2.73e-12 | -2e-12 m |
| 0.125 | +5.68e-14 | -9.1e-13 | -9.78e-13 m |
| 0.0625 | +5.68e-14 | -9.1e-13 | -5.29e-13 m |
| 0.03125 | +5.68e-14 | -9.1e-13 | -2.29e-13 m |
| 0.015625 | +5.68e-14 | -9.1e-13 | -9.08e-13 m |
| 0.0078125 | +2.06e-13 | -2.73e-12 | -1.94e-12 m |
| 0.00390625 | +2.06e-13 | -2.73e-12 | -2.5e-12 m |
| 0.001953125 | +1.49e-13 | -1.82e-12 | -1.23e-12 m |
| 0.0009765625 | +5.68e-14 | -9.1e-13 | -5.96e-13 m |
| 0.00048828125 | +5.68e-14 | -9.1e-13 | -3.13e-13 m |

The likely mismatch is in the finite-precision path definition. deformableBarrierEnergyAt evaluates reference-mode trial geometry from each femTrialNodeAtAlpha result. For alpha greater than zero, that helper recomposes reference displacement from accepted displacement and the corrected candidate velocity; the base potential is evaluated from the candidate state already in memory. This creates an FP32 recomposition/quantization path that can differ from the smooth dt * correction direction contracted by contactBarrierDirectionalDerivative. The trace supports this diagnosis, but a targeted state-path test is still needed before treating it as proven.

Replay stopped after the same 21 accepted steps with the same contact failure. This is diagnostic evidence only; it does not establish a repaired solver, sustained support, or physical qualification.
