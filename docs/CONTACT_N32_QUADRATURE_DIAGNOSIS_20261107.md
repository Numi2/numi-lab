# Dynamic n=32 surface quadrature comparison

The independently integrated n=32 three-point contact rule completed both
10 ms runs under the declared timeouts. Its 2 µs to 1 µs changes were
7.13e-8 in peak-normalized 1 ms force bins and 6.45e-9 in total impulse;
clearance and momentum checks passed. The retained n=16 comparison also
passes its independent temporal checks. Physical geometry, material,
initial state, finite-width normal contact energy and 1 ms bins were fixed.

At the common 1 µs timestep:

| Comparison | Maximum binned-force difference | Integrated impulse difference |
| --- | --- | --- |
| n=16 → n=32, nodal rule | 12.43% | 0.628% |
| n=16 → n=32, three-point rule | 14.75% | 0.362% |
| n=32 nodal → three-point | 6.59% | 0.907% |

All of these force and impulse differences exceed the unchanged 0.25% gates.
The prior static manufactured-control result showed that the three-point rule
can reduce local transfer error in that specific static configuration. It
does not resolve this dynamic mesh discrepancy.

The predeclared diagnostic prediction was that if quadrature were the main
cause of the 12.43% nodal spatial difference, switching to three-point
quadrature would reduce that difference below half its value. The measured
14.75% does not support that simple hypothesis. At the same time, the 6.59%
difference between rules on n=32 shows that quadrature remains consequential;
the test cannot assign the discrepancy to a unique cause or exclude
interactions between quadrature and bulk wave resolution.

Both n=32 trajectories retain contact throughout the post-onset portion
of the measured 10 ms. Minimum gaps are 94.17 µm for nodal and 92.68 µm
for three-point quadrature. The new runs took 65.9 s and 130.1 s, with
the fine run peaking at 57.0 MB RSS. These are CPU diagnostic costs, not
Metal performance. The physical contact law remains regularized and finite
width; no hard-contact, real-material, spatial-reference or production
qualification follows.

[Evidence](evidence/contact-n32-gaussian-20261107/) retains the prediction,
plan, exact source/binary identity, timeouts, measured resources, raw
trajectories and gated comparison. No GPU run or Matter integration was
performed.
