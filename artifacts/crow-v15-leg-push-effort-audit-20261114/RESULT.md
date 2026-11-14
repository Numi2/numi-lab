# Retrospective leg-drive saturation audit

The native v15 model reports hip and knee effort limits of 0.75 N m and ankle limits of 0.60 N m on each side. In the retained two-low wing-trigger trial, all six leg drives reached their respective limits during push intervals 369–374. Each hip was at its limit for about 7.3% of observed substeps, each knee 21.9%, and each ankle 26.0%; the preceding 350–368 window showed none at those thresholds. Full per-lane extrema, velocities, unique-substep counts, the source trace hash, observer field schema and model-inspection build hashes are retained.

This establishes intermittent modeled drive saturation, not that every substep is saturated, that faster commands cannot help, or that slower extension will help. The trace reports modeled effort and velocity; observer q/v-to-contact epoch alignment in these lanes remains inferred. The slower-extension experiment is a separate physical test.
