# Native physical initialization fixes 216

Exact zero muscle activation now evaluates passive muscle mechanics and residuals without dividing by a zero recruitment interval. Recruitment and global activation polish are skipped for this fixed interval. Unsupported zero-cap pose search rejects transactionally. The existing positive-activation static-support checks and focused zero-activation cases pass.

A fresh FEM-bed release retains source-witness identities and gaps but does not recruit or seed duplicate rigid proxy reactions. The existing Matter FEM contact owns the bed load. Source placement uses actual Float32 outer FEM nodes, a declared 10 nm positive initial-clearance target and at most 16 outward root-coordinate ULPs. It retains the 1 um maximum first-touch and common-translation checks; no contact/material acceptance tolerance is changed.

The Release native app and both focused checks pass on SSH Numi. Control014 passed initialization and reached the shared solve, then rejected a -5.777 nm initial plane penetration. Control015 initialized at +24.025 nm with zero proxy reactions, passed that contact gate, and rejected Newton iteration5: tet29419 currentJ=2.495642185, final trialJ=2.505839825 exceeds the existing maximumJ=2.5. Both controls accepted zero physical steps. Respiratory/Brain/common-state receipts are retained and neither failed candidate is promoted as motion.

Candidate033 remains the zero-crossing outer reference, not an admitted physical replacement. Positive replacement cells, actual child mass mapping, native scene integration and accepted motion remain pending. These controls use the unchanged historical geometry as compatibility diagnostics and do not qualify supported rest. No images were inspected.
