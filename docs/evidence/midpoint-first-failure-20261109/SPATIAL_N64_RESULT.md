# n=64 fixed-law spatial follow-up

The independently evaluated 64-division reference completed at both 0.5 and
0.25 microseconds; both the 32-division and 64-division temporal comparisons
pass the preregistered limits. The 64-division reference area was
0.0003999999999999718 m² and its minimum gap was 94.4606 micrometres.
The 0.25 microsecond reference has peak force 0.1226473 N and total impulse
0.0004100167 N s. Its ordered-parallel evaluator had been checked byte for
byte against retained n=2, n=8 and n=32 outputs at the specified timesteps.

The temporally qualified 32-to-64 comparison still fails spatial qualification:

| Measure | Relative difference | Limit |
|---|---:|---:|
| 1 ms-binned force | 5.1604% | 0.25% |
| Sampled peak force | 43.8439% | 1% |
| Total impulse | 0.5249% | 0.25% |
| Maximum cumulative impulse | 1.5207% | 1% |

The prediction of at least one failed gate is supported; all four fail. The
binned and peak measurements are distinct, and neither establishes material
calibration. The physical law, activation width, area coefficient, geometry,
initial state and output clock were unchanged. The 64-division values are
independent CPU reference results, not native Matter or GPU timing evidence.
The cause of the residual spatial discrepancy remains unproven; further mesh
refinement would require a new bounded resource and validation plan.

The completed coarse output and ordered-parallel fine output, execution
records and analysis are retained. The pre-existing `rk4-n64-h025.json` is a
not-run sentinel for the superseded sequential supervisor, not the fine
result. The 2,128-second fine elapsed time includes concurrent reference work
and is not an application performance measurement.

The retained 2 microsecond common-clock curve inspection puts the n=32 peak
at 4.974 ms and the n=64 peak at 9.858 ms. Its largest pointwise difference
occurs at 5.980 ms (0.16652 N versus 0.02597 N), while the largest cumulative
impulse gap occurs at 6.044 ms. These sampled differences show that matching
final impulse alone can conceal substantial force-history differences; they do
not identify their cause. The common-clock peak is separate from the peak
measured at every 0.25 microsecond reference step used in the gate.
