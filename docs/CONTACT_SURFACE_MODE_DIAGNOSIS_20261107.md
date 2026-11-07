# FEM contact: spatial response and quadrature diagnosis

The mesh-refinement failure persists. The independent CPU reference was
instrumented at the same physical parameters and 0.5 µs timestep to retain
surface force distribution and displacement projections at 0.1 ms intervals.
For every mesh, its force bins and total impulse reproduce the previously
retained reference exactly. Sampling at 0.1 ms does not locate an exact
between-sample force peak.

| Divisions | First contact episode | Largest sampled force | Maximum surface displacement RMS |
| --- | --- | --- | --- |
| 2 | 4.5155–5.173 ms | 0.725 N at 4.8 ms | 16.74 µm at 7.9 ms |
| 4 | 4.5155–5.0355 ms | 0.425 N at 4.7 ms | 12.94 µm at 8.6 ms |
| 8 | 4.5155–4.9295 ms | 0.335 N at 5.9 ms | 6.56 µm at 6.9 ms |
| 16 | 4.5155–4.850 ms | 0.257 N at 5.3 ms | 3.25 µm at 8.0 ms |

The common onset follows free fall; contact duration, later episodes, force
distribution and surface deformation vary strongly. At the maximum surface
RMS time, the two lowest single-axis cosine projections are respectively
8.12, 1.56, 0.508 and 0.221 µm for the four meshes. These projections are
partial diagnostics, not a complete eigenmode decomposition or evidence that
the remaining response is converged. The retained data also records nodal
corner and edge force, though those groups cover shrinking support areas as
the mesh is refined and their raw fractions should not be compared as fixed
physical regions.

A small-strain longitudinal wave crosses the 20 mm body in approximately
6.99 ms. Cell crossings fall from 3.50 ms at two divisions to 0.437 ms at
16. This is consistent with an under-resolved short contact transient, but
the scale estimate alone does not prove the cause. Each reported mesh passed
the independent 1 µs to 0.5 µs temporal checks; contact modes are not bounded
by a wave-speed estimate alone.

## Same-mesh contact quadrature

The smallest completed discriminating experiment uses n=8 and changes only
the quadrature of the same regularized surface energy: the original
one-third triangle vertex rule versus a positive three-point triangular
rule with barycentric interpolation and force transfer. Both retain the
same reference area, activation width, coefficient, material, geometry,
initial state and duration. Each rule's CPU force passes an independent
finite-difference energy-gradient check. The three-point rule's 1 µs versus
0.5 µs change is 2.94e-9 in peak-normalized force bins and 2.56e-11 in
impulse, meeting the unchanged 0.25% temporal gates.

At the common 0.5 µs timestep, the two n=8 rules differ by **8.40%** in
the largest peak-normalized 1 ms force bin and **3.37%** in total impulse.
The three-point rule retains positive clearance and momentum balance.
Changing surface quadrature therefore materially changes this transient,
but the test cannot apportion the larger mesh changes between volume
discretization and quadrature. The two rules are discrete approximations to
the same finite-width contact energy; neither establishes a continuum or
hard-contact reference.

A subsequently planned n=16 comparison held the FEM space and physical law
fixed while repeating both quadratures. The new rule again passed its
1 µs to 0.5 µs temporal, clearance and momentum gates. Its difference from
the nodal rule fell to 2.20% in force bins and 0.637% in impulse, but both
still exceed the 0.25% gates. Under the three-point rule itself, n=8 to
n=16 changes force bins by 27.62% and impulse by 3.84%. Thus changing this
surface rule alone does not resolve the mesh-refinement failure. No runtime
integrator or production contact law is implied.

[Evidence](evidence/contact-surface-diagnosis-20261107/) retains the source
and binary hashes, pilot, predeclared run plan, per-mesh surface traces,
quadrature runs, CPU costs and checks. These are CPU diagnostics, not Metal
performance measurements or physical material validation.
