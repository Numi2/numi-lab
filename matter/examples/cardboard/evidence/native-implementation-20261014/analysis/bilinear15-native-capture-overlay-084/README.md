# Bilinear-15 local projection policy overlay, rebased on frozen083

This package carries the scoped scaled-correction policy onto `source-tiled-fixture-and-memory-accounting-083`. The earlier 082 form is preserved as `evidence-source-overlay-082-original.patch`; it excluded `paperCompensatedArithmetic` and therefore did not admit the actual source15 CLI combination.

The corrected compiler and validator accept the source15 pair `genericCompensatedImplicitProjection=true` plus `paperCompensatedArithmetic=true`. The first selects compensated generic implicit evaluation; the second supplies the compensated paper arithmetic/reference-deformation path. Neither turns on the 13-state `paperHillAccurateProjection` or scalar initializer. The policy-off world keeps the 8-step default; the opt-in gets a separately fingerprinted 16-step local budget.

This overlay passed its two CPU compiler tests on the external frozen083 copy. No GPU execution occurred. Root should apply `source-overlay.patch` against the exact base in `source-overlay.json`, freeze the resulting source, and run the bounded native captured-state checker supplied in the next overlay.
