# Same-mesh midpoint accuracy ladder

The default-disabled precision midpoint option ran the same n=2 cube, reference-area contact law, material, 100 micrometre activation width, physical tolerances and 10 ms duration as the BE control. All runs used the frozen 005 native binary and the independently temporally qualified n=2 FP64 reference. This does not establish spatial convergence.

| Timestep | Binned force error | Sampled peak error | Total impulse error | Cumulative impulse error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 125 us | 5.6621% | 6.2583% | 0.6322% | 2.6708% | fail |
| 62.5 us | 1.5656% | 3.2205% | 0.2360% | 0.7545% | fail |
| 31.25 us | 0.3142% | 0.5008% | 0.03747% | 0.2012% | fail |
| 15.625 us | 0.08504% | 0.2584% | 0.01094% | 0.05104% | pass |

The registered limits are 0.25%, 1%, 0.25%, and 1%, respectively. At the passing step all 640 steps were accepted, minimum gap was 87.0996 micrometres, and saved-state replay, continuation, forced rollback and fingerprint rejection passed. The largest *tested* passing timestep is 15.625 us; an untested intermediate timestep might also pass. The 0.08504% number is the binned force error, not peak error. The native peak was 0.722801 N, reference peak 0.724674 N; sampling occurs once per native timestep.

Per-run traces, checks and binary/reference hashes have prefix `stage-ladder-n2-`. A speed comparison is not yet possible: the BE control at 1 us completed but failed force and impulse gates, and finer control qualification is pending. All reported observational times include diagnostic snapshots and replay and are not benchmark results.
