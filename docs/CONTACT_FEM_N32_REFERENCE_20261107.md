# Fixed-law n=32 reference: spatial force gate still fails

The independent FP64 nodal-contact reference completed its bounded n=32
10 ms runs at 2 µs and 1 µs, preserving the same geometry, material, gravity,
initial state and regularized contact parameters. It contains 35,937 nodes
and 196,608 tetrahedra. Both runs passed clearance, momentum and the unchanged
0.25% temporal gates: the 2 µs to 1 µs force-bin change was 2.52e-8 and
impulse change 6.56e-9.

At the common 1 µs timestep, n=16 to n=32 changes the maximum
peak-normalized 1 ms force bin by **12.43%** and the integrated impulse by
**0.628%**. Both exceed the unchanged 0.25% spatial reference gates. The
n=32 minimum gap was 94.17 µm; its contact episode remained active from
4.516 ms through the end of the 10 ms observation. No n=32 Gaussian
quadrature comparison was performed.

The CPU runs took 68.4 and 147.6 seconds and peaked at about 65.5 and 67.2 MB
RSS. Before the run a conservative row-sum bound at the stress-free reference
shape and one uniform active gap gave h·ω ≤ 0.071 at 2 µs. That bound does
not cover the full nonlinear trajectory; the two-timestep, clearance and
momentum checks are the relevant observed evidence. Plans, timeouts, source
and binary hashes, resource logs and all output are retained
[in the evidence](evidence/contact-n32-reference-20261107/). These CPU costs
are not Metal performance claims.

The new first-bin force is 0.028501 N. It is **12.81% above** the approximate
1D acoustic half-space value, whereas n=16 had been 2.19% above it. The
previous apparent approach toward the half-space result through n=16 does not
continue monotonically. The n=16 agreement was not evidence of a converged 3D
onset; the finite block and half-space have different boundary conditions.
The 3D force reference and Matter control remain spatially unqualified.
