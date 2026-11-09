# Audit of the retained mesh ladder

This is a reanalysis of existing independent CPU references, not a new native rollout. The machine-readable audit records the SHA-256 of each input and validator. It checks each nodal reference-area weight against the triangulated reference surface, total area and mass, positive clearance, finite complete force histories, and impulse accounting. Each mesh passes its own 0.5-to-0.25 microsecond temporal refinement gates. The output does not itself prove identity of the material and law inputs: retain the associated source and execution evidence for that claim.

| Divisions, coarse to fine | 1 ms-binned force | Step-sampled peak force | Total impulse | Cumulative impulse |
| --- | ---: | ---: | ---: | ---: |
| 2 to 4 | 92.9039% | 56.0138% | 20.0090% | 29.2305% |
| 4 to 8 | 70.4261% | 28.4352% | 3.3740% | 12.5638% |
| 8 to 16 | 27.2376% | 38.4490% | 6.6589% | 8.3168% |
| 16 to 32 | 12.4253% | 48.0665% | 0.6280% | 3.6823% |
| 32 to 64 | 5.1604% | 43.8439% | 0.5249% | 1.5207% |
| 64 to 128 | 1.7274% | 20.6927% | 0.1618% | 0.5037% |

All six adjacent comparisons fail at least one predeclared spatial refinement gate. Peak means the maximum sampled at integration steps; it is not the 1 ms-binned metric. The historical 64-division pair is missing direct binary identity for both executions, so the separately supervised provenance-bound repeat must be assessed before upgrading that comparison. These results do not establish continuum convergence, real-material accuracy, or a native 128-division rollout.

Reproduce from the repository root with `python3 tests/contact/audit_historical_mesh_ladder.py`. The checked inputs remain unchanged; the script rewrites only its designated audit receipt.
