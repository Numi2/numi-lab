# Bounded rejection recovery experiments

These are experimental, default-off numerical controls and diagnostics. No full-scene physical step was accepted. Kernel checks, initialization and remeshing do not qualify motion or performance.

- A bounded, timeline-ordered backtracking trace records per-alpha determinant and constitutive witnesses without changing acceptance.
- Reference self-contact width uses a conservative separating-plane reference gap; physical hard clearance remains unchanged. This changes numerical barrier regularization and is not force-equivalent to the baseline. Its trial reduced the initial nonlinear norm from 0.709982 to 0.266064, then failed with a collision certificate and zero accepted steps.
- The optional EW maximum-eta cap leaves its default at 0.25. A 1e-4 cap reduced the first linear relative residual to 1e-4 using 208 columns, but the skin/fat geometry limit remained 1/512 and no step was accepted.
- Fixed-boundary cavity conditioning retired six captured limiting cells and improved the first skin/fat alpha from 1/512 to 1/256. Its final nonlinear residual was 0.709383, compared with 0.708254 before the remesh: no convergence gain is established.

Source and binary identities are retained in each manifest. Full execution used a dirty integrated checkout, not a clean build of this publication commit. The bounded cavity initialization is retained as a separate Human-side candidate; it is not enabled by these Matter changes. The typed volume still contains many extremely thin cells: positive Jacobians alone do not establish conditioning.

No physical thresholds, mass, material law, topology or rollback policy are loosened by the eta control or diagnostic trace. The reference-width experiment is opt-in and unqualified.
