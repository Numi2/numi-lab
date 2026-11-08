# Matched free-fall cross-engine gate

The [preregistered](FREE_FALL_CROSS_ENGINE_PLAN_20261108.md) one-second, collider-free ballistic test passed for both engines against the independent semi-implicit Euler reference, with 1,000 accepted Numi steps and no failed steps.

| Engine | Final y (m) | Final y velocity (m/s) | Absolute y error (m) | Absolute velocity error (m/s) |
| --- | ---: | ---: | ---: | ---: |
| Numi MetalWorld, FP32, Apple M6 | 5.0900983810 | -9.8100633621 | 0.0000033810 | 0.0000633621 |
| MuJoCo 4.0.0, FP64, same Mac | 5.0900950000 | -9.8100000000 | 3.02e-14 | 9.06e-14 |
| Independent discrete reference | 5.090095 | -9.81 | — | — |

The gate limits were 0.001 m and 0.001 m/s. Numi's tested runtime was the existing Release library and Metal binary whose exact hashes are recorded in [the evidence](evidence/free-fall-cross-engine-20261108); the MuJoCo 4.0.0 and Python package versions are pinned there too. The final comparison instrument is at source commit `95af5cbea`, and the runtime was already built before this instrument. No timing comparison was performed.

The first instrument build failed because the library path was assigned to the wrong configuration type; its compiler error is retained. The first Numi run was rejected before GPU dispatch because the free-motion world still contained rigid colliders. Removing the collider shapes made the implementation match the original declared no-contact setup; that failed attempt remains in the evidence.

This qualifies only a simple free-body ballistic integration gate at one timestep. It does not qualify contact, articulated dynamics, other integrators, training, latency or throughput, and it does not establish competitive performance. A later speed comparison requires matched physical behavior, fixed full workloads, explicit inclusion of setup and rejected work, and exclusive GPU measurements.
