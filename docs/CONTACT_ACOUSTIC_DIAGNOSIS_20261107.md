# Contact onset: independent acoustic diagnostic

A one-dimensional small-strain elastic half-space provides a diagnostic for
the early compressional wave response. With the same density and elastic
constants, its longitudinal speed is 2.86 m/s and impedance 3146 Pa s/m.
Starting at the analytically computed free-fall contact time, an independent
FP64 RK4 integration uses the same pressure–gap law, reference area and
gravity. Its 1 µs to 0.5 µs force and impulse changes are below 8e-13 and
2e-13, respectively.

| Mesh | First 1 ms-bin force | Difference from 1D acoustic diagnostic |
| --- | --- | --- |
| Half-space diagnostic | 0.025265 N | — |
| 2 divisions, nodal | 0.199321 N | 688.9% |
| 4 divisions, nodal | 0.102786 N | 306.8% |
| 8 divisions, nodal | 0.051194 N | 102.6% |
| 16 divisions, nodal | 0.025818 N | 2.19% |
| 16 divisions, three-point triangle | 0.025794 N | 2.10% |

The first bin contains the onset at 4.5152 ms. A pilot-informed prediction
was that this early force would move toward the acoustic result with mesh
resolution; the listed data support that prediction for these samples.
A rough 1 µs pilot preceded the written prediction, so this is not a blind
prediction. It supports the idea that compressional-wave resolution
contributes to the onset error; it does not isolate every source of spatial
error.

The half-space is linear, one-dimensional and laterally infinite. The FEM
sample is finite, three-dimensional, nonlinear and has free sides, so the
half-space is not an exact solution or a qualified reference for that body.
The first bin predates a longitudinal wave travelling from the contact plane
to the top and back, but free-side effects are not ruled out. Later force
bins are especially unsuitable for treating this comparison as a finite-body
reference. The 2.19% agreement also does not satisfy the unchanged 0.25%
scientific gate. The FEM mesh and both candidate contact quadratures remain
unqualified.

The equations, pilot, plan, refined outputs and comparison are retained in
[the evidence](evidence/contact-acoustic-diagnosis-20261107/). No GPU or
production integrator change was involved.
