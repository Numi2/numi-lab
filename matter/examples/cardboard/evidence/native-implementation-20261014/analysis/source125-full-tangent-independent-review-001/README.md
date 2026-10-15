# Independent review of full 9x9 tangent audit 147

Verdict: **PASS_WITH_SCOPED_ERRATUM**. The 147 audit is hash-complete and its independent rerun reproduces the matrix NPZ and per-column CSV byte-for-byte. The report matches semantically except for runtime timing; the prediction differs only in its generation timestamp.

The source contract names parameter slot 26 (zero-based) `fb_epsilon`. The inherited sweep-006 output field `rate_exponent_max_by_paper_material` is therefore mislabeled: its values are approximately `1e-8` and are the FB epsilon values. This sweep field is an inherited metadata defect, retained here as an erratum; it is not used by 147's calculation. The selected run's source-bound probe sets `maximumRateExponent = 0u`, and the run prediction binds that exact probe SHA.

The selected state is environment 0, control 1, Newton 48, with 1,920 tetrahedra and 17,280 tangent columns. All 1,728 paper projections are valid; all column references are nonzero and errors are under `1e-3`. Maximum per-column relative errors are liner `3.29483e-6`, medium `6.13590e-5`, and glue `5.18882e-8`. Fifty-eight FP64 projection warnings remain in the source report and are not suppressed.

This is a selected-state analytic derivative check. It does not qualify FP32 endpoint/secant noise, a complete nonlinear solve, nonzero-rate laws, damage/interfaces, or physical assembly. Two medium projected multiplier increments differ from captured zero only at approximately `5.1e-17`; this is recorded as a near-zero activity-label caveat.

Run `verify.py` with the qualified Python 3.10.12 environment to verify all declared hashes and rerun comparisons.
