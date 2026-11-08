# Unused-motion and friction controls

The revised research candidate and the archived fixed-128 discrete reference each completed 20/20 steps in all four predeclared variants. The ordinary and traced candidate outputs matched on exposed non-scratch fields; capture-free and repeated final-state and trace checks passed. The fresh reference shader was first checked against the retained original fixed-128 output. The host library identity changed since the older study; LIBRARY-AMENDMENT.md and the matching freshly compiled unmodified probe checks retain that distinction.

| Variant | Max 1 ms force error / reference peak | Total normal impulse error | Max strict normal / tangent residual (m/s) |
| --- | ---: | ---: | ---: |
| Reverse tangential motion | 0.00265% | 0.0000934% | 1.22e-6 / 7.34e-7 |
| Raised rod | 0.00483% | 0.0002165% | 2.99e-7 / 7.34e-8 |
| Zero scene friction | 0.00122% | 0.0000114% | 2.37e-7 / 0 |
| Separating sphere | 0 absolute | 0 absolute | 5.97e-7 / 1.85e-5 |

All comparisons used the predeclared 0.25% limits where a nonzero ground reference existed. In the frictionless case every captured contact had zero effective friction. In the separating case, five successive steps with the sphere outside the ground contact offset and moving away had zero ground impulse. The joint block was marked in 60/60 rows in each of the first three controls. The separating control had 15 marked and 30 unmarked rows; those unmarked rows used the retained fallback path. The largest captured cone excess was below `3e-10` in these fixtures. Maximum independently summed momentum-balance components were between `2.70e-9` and `1.78e-8 N s`; the estimates and work traces are in controls/analysis.json.

These are four synthetic variations of one island topology and a discrete reference, not physical force calibration, broad geometry qualification, independent geometric velocity reconstruction or a performance benchmark. A forced-failure restoration test remains. The candidate is isolated and unpromoted; the first candidate’s failed strict-normal evidence is retained separately. No directional CCD or augmented-Lagrangian option was enabled.
