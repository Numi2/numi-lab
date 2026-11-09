# Fixed-plane midpoint native rollout, n=8

The isolated opt-in stage-precision candidate completed the 10 ms, n=8
frictionless FEM–fixed-plane rollout at 3.90625 microseconds. All 2560 steps
were accepted, 1289 had active contact, and the restored full replay,
continuation replay, identity mismatch rejection and forced rollback checks
passed. The final physical digest is recorded in the trace. The separate short
run crossed the prior step-204 failure. The candidate remains disabled by
default and is limited to the stated fixture.

Against the independently evaluated, temporally checked n=8 reference:

| Measure | Relative error | Limit |
|---|---:|---:|
| 1 ms-binned force | 0.01731% | 0.25% |
| Sampled peak force | 0.02893% | 1% |
| Total impulse | 0.001973% | 0.25% |
| Maximum cumulative impulse | 0.009494% | 1% |

The native sampled peak was 0.36155154 N; the reference peak was 0.36165618 N.
These are distinct from the binned-force metric and have 3.90625 and 0.25
microsecond respective peak sampling. The native weight sum was
0.00039999998080007 m². The minimum gap was 91.7964 micrometres. The maximum
independently evaluated local residual without a fixed denominator floor was
0.00035594 versus the 0.005 limit, the independent maximum kinematic defect
was 3.823e-18 m, and maximum summed momentum error was 2.491e-12 N s.
Accumulated reported contact work was -1.43851028e-7 J; the independently
accumulated impulse/midvelocity work was -1.43851022e-7 J. These do not by
themselves close a complete material-plus-contact energy balance.

This qualifies the stated mesh-specific fixed-plane test, not spatial
convergence: the independently checked n=32 to n=64 refinement fails all four
spatial gates. It does not establish a cost advantage over backward Euler or a
two-way body. The near-contact initialization, full default regression suite,
and genuine dynamic-owner qualification remain separate. This run included
snapshots and replay, so its elapsed time is not a timing benchmark.
