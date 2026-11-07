# Fixed-law FEM plane reference: spatial refinement remains unqualified

The independent FP64 FEM reference does not yet provide a spatially converged
force reference. Across 2, 4, 8 and 16 divisions per side, geometry, total
mass, material parameters, gravity, activation width, contact energy density,
surface quadrature rule, initial gap and 10 ms duration were held fixed.
Every mesh's 1 µs versus 0.5 µs RK4 temporal comparison passed the unchanged
0.25% force and impulse gates, as well as clearance and momentum checks.
The fine timestep was then held at 0.5 µs for spatial comparisons.

| Mesh pair | Maximum peak-normalized 1 ms force-bin difference | Impulse difference |
| --- | --- | --- |
| 2 → 4 | 92.90% | 20.01% |
| 4 → 8 | 70.43% | 3.374% |
| 8 → 16 | 27.24% | 6.659% |

All pairs fail the 0.25% reference gates. These are differences between
successive discrete models, not errors against a known converged solution.
Refining the mesh also refines nodal surface quadrature, so this comparison
does not separate volume discretization from contact quadrature error.

The fine-timestep runs have the same 4.5155 ms first onset, as expected
from identical free fall. Their first contact episodes end at 5.173, 5.0355,
4.9295 and 4.850 ms, respectively, followed by later renewed contact.
Observed minimum gaps remain positive. More spatial resolution substantially
changes the transient force and contact release behavior.

Mass and contact area are conserved at each resolution to FP64 rounding.
Using the same six-tetrahedra-per-cell connectivity and lumping, the mass
on contact-supporting nodes is 25%, 12.5%, 6.25% and 3.125% of total mass as
the mesh is refined. This is a property of the discretization, not a separate
physical contact layer. Linearized material wave speeds are approximately
2.86 m/s longitudinal and 1.35 m/s shear. The longitudinal cell-crossing
time falls from 3.50 ms to 0.437 ms across the sequence; the 20 mm sample
crossing time is about 6.99 ms. These scales and the shifting contact-supported
mass are plausible explanations for sensitivity of the short contact
transient, but do not identify the error source by themselves. The estimates
do not include contact-induced modes and do not certify explicit stability.
The observed temporal refinement is the relevant time-integration evidence.

The n=4 runs took 0.31 s and 0.68 s, n=8 took 2.50 s and 4.81 s, and n=16
took 16.58 s and 34.31 s on the recorded host. These are observed CPU
diagnostic times, not Metal performance claims. Plans, output, source/binary
hashes, wave calculations and the earlier n=2/4/8-only result are retained in
[the evidence directory](evidence/contact-mesh-reference-20261107/). The
previous 2-division midpoint result remains valid only for that fixed
semidiscrete model. Neither the reference nor the Matter control is
force-qualified across meshes or against a physical material experiment.
