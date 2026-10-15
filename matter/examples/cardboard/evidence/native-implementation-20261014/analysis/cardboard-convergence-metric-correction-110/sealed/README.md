# Cardboard resolution metric correction (overlay 110)

This CPU-only evaluator overlay is based exactly on snapshot `source-ee-ccd-continuation-107`. It changes only the resolution-study evaluator, its synthetic tests, and its documentation. Apply `patch.diff` with `-p1` to that snapshot; a dry-run against the bound snapshot succeeded. The sealed files are also under `files/matter/tools/`.

## Audit findings

The run loader and comparison admission already fail closed on incomplete accepted histories, rejected steps/certificates, mesh or material mismatch, changed controls, and mismatched phase schedules. The plan independently varies `nx`, `ny`, and thickness, includes a combined refined point, preserves all phase durations and loading speed in time refinement, and separates slower loading. Thresholds cover peak force/moment (2%), path work (2%), full curves (5%), release-angle change (1 degree), and enabled damage/debond area (2 percentage points). No admitted native calibration or physical qualification was found in the source-data package; missing or unidentifiable evidence remains unavailable.

Two evaluator defects required correction:

1. Absolute moment-path work summed the absolute value of each signed trapezoid. If a linearly interpolated moment crossed zero within a measured rotation interval, opposite-sign endpoint moments canceled and understated the path work. The corrected integral splits at the analytic zero crossing. Signed external work remains the signed trapezoid.
2. Springback comparison used only the magnitudes of the accepted pre-release and post-release measured right-grip angles. Equal-magnitude releases in opposite directions therefore had zero difference. The corrected 1-degree gate compares the signed measured angle changes. Historical magnitude fields remain present for readers.

The result schema is bumped from `numi.cardboard.resolution-study.v1` to `.v2` so previous screens cannot be mistaken for the corrected definitions. Existing v1 convergence outcomes require reevaluation before any promotion; this patch does not retroactively promote or invalidate a physical result.

## Verification

Using Python 3.10.12:

```sh
python3 matter/tools/test_cardboard_resolution_study.py
python3 matter/tools/test_cardboard_convergence.py
```

Results: 13/13 resolution-study synthetic contract tests and 19/19 convergence-reader synthetic contract tests passed. Added cases independently check a +1 to -1 moment zero-crossing (triangle-area oracle), the reported integrated path work, and opposite-sign springback changes whose magnitudes match but whose signed difference exceeds the 1-degree gate. These are evaluator contract tests only, not native solver or physical evidence.

See `manifest.json` for base, context, replacement, patch, and test-log SHA-256 bindings. No GPU or native run was performed.
