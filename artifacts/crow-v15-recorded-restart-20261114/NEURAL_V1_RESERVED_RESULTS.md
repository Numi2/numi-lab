# Neural balance v1: reserved 400-step comparison

The v1 policy was frozen before these tests. These are development branch-start fixtures, not autonomous arrivals. The bird has a scripted terminal carrier and neural residuals at four hip/ankle drives. All results use first episodes and 16 substeps; reset transitions do not count.

| Seed | Policy | First termination | Longest qualified hold | Hold reaches interval 400? | Result |
|---|---|---:|---:|---|---|
| 20310111 | Initial | none | 295 intervals (38–332) | no | fail |
| 20310111 | v1 | none | 363 intervals (38–400) | yes | pass |
| 20310129 | Initial | 379, reason 2 | 260 intervals (42–301) | no | fail |
| 20310129 | v1 | 370, reason 2 | 252 intervals (41–292) | no | fail |

The trained residual improves the first test but fails and terminates earlier than the initial policy in the second. It is not qualified. These seeds are consumed for further development. The gate is a model-specific physical contact check, not evidence of real-crow behavior. The task actor has root linear and angular velocity, projected gravity, joint observations, visual input, and 13 support-patch channels per foot, as defined by `src/core/RunProgram.cpp` and `src/core/CrowEmbodiment.cpp`; the separate scripted carrier remains active. The original 48-second recording, forced phase, branch relocation and reconstruction limitations still apply.
