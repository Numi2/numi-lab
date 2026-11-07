# Fixed reference-area FEM–plane contact: first temporal qualification

The new law remains an opt-in research instrument. It is not a qualified
control or a replacement for Matter's default contact law. The directional CCD
and augmented-Lagrangian options remain disabled by default. This experiment
does not alter the earlier contact-onset or CCD evidence.

The narrow fixture uses a flat, coplanar plane-facing exterior FEM patch.
Reference-surface lumped quadrature gives each node one third of each projected
face area. For positive gap g below d, its energy per reference area is
-k(g²-d²)² log(g²/d²), with fixed k = 1e16 N/m⁵ and d = 1e-4 m.
The area multiplies the same energy, force and tangent. This does not implement
general deformable surface quadrature. Unsupported geometry is rejected.
The implementation and its area are fixed independently of timestep and mesh.
The scope is a single static plane, reference-displacement FEM, and no friction
or coupling; it is disabled by default.

An independent FP64 scalar pressure–gap check gives 9.6970704133e-5 m
equilibrium gap under 0.086328 N on 0.0004 m², with energy derivative checked
by finite difference. This only validates the uniform scalar contact law: it
is not an independent elastic FEM impact reference. The CPU quadrature check
verifies constant total area across four triangulation resolutions and rejects
a tilted patch.

## Native temporal test

Four 10 ms runs used the same physical law, 2 divisions per axis, 0.2 mm initial
gap, 16 Newton and 64 FGMRES iterations, and the physical residual policy.
Plan, source commit, binary and metallib hashes, detailed traces, diagnostics,
and analysis are in [the retained evidence](evidence/contact-area-reference-20261107/).
The observed source commit is cac0c53c, Apple M6. Tests were sequential and
instrumented; their timings are not performance claims.

| Coarse h | Fine h | Peak 1 ms-bin force change | Integrated impulse change |
| --- | --- | --- | --- |
| 0.5 ms | 0.25 ms | 20.27% | 1.918% |
| 0.25 ms | 0.125 ms | 14.86% | 0.998% |
| 0.125 ms | 0.0625 ms | 23.92% | 0.579% |

Every rollout completed all steps and met the instrument's local residual,
clearance, momentum balance, rollback and deterministic replay checks.
Nevertheless, **none of the force comparisons meets the predeclared 0.25%
reference gate**. Contact episodes also differ; the two finest runs show
release and renewed contact at different times. Completion and residual checks
do not establish accurate force evolution. The scalar law check is not enough
to decide whether the apparent oscillatory behavior is physically correct.

No mesh qualification, hard-contact limit, independent dynamic reference,
near-contact initial-state qualification, or accepted-time performance
comparison was performed. The necessary next investigation is a separately
verified physical dynamic reference and refinement that resolves force and
release behavior without changing this law or relaxing its gates. The retained
failed temporal series must remain part of that assessment.
