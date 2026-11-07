# Static FEM–plane contact controls

A zero-gravity, laterally constrained homogeneous slab under a prescribed top
height has an independent analytic Neo-Hookean stress and the same regularized
plane pressure law. The uniform-gap solution is affine and exactly representable
on every tested mesh. For n=2, 4, 8, 16 and 32, both nodal and three-point
surface quadrature reproduced the analytic 0.008849532 N top and contact
force. The worst relative top-force error was 4.94e-13; the largest
normalized local z residual was 2.50e-12. Force moments also matched.

This is a narrowly prescribed static control with artificial lateral
constraints and no gravity. It checks an affine solution and contact transfer;
it does not qualify nonuniform static fields, the gravity-driven impact,
mesh convergence of the force trace or a physical material model.

A nonuniform manufactured affine-shear extension varies the bottom gap
linearly from 93 to 99 µm within the same activation band. An independently
integrated prescribed surface load balances the analytic material stress
and the varying contact pressure. Constant side shear traction is also
applied. This artificial loading produces an exactly representable affine
field, isolating the surface-force transfer from the bulk interpolation.

The first runs omitted the side traction and produced a nonvanishing
0.136% normalized local residual. Their outputs and exact source are retained.
The corrected reference traction uses independent 4×4 and 8×8
Gauss–Legendre/Duffy integration; their nodal loads, total force and first
moments agree within 1e-7 of the declared load scale before comparison.

| Divisions | Nodal integrated-force error | Nodal maximum local residual | Three-point integrated-force error | Three-point maximum local residual |
| --- | --- | --- | --- | --- |
| 2 | 7.537% | 31.97% | 0.0000280% | 0.0229% |
| 4 | 1.884% | 16.88% | 0.00000175% | 0.00563% |
| 8 | 0.471% | 8.67% | 0.000000109% | 0.00140% |
| 16 | 0.1178% | 4.39% | 0.00000000684% | 0.000349% |
| 32 | 0.0294% | 2.21% | 0.000000000427% | 0.0000871% |

The local residual is normalized by the total independently integrated contact
force divided by n²; it is distinct from summed force error. The nodal rule
first meets the unchanged 0.25% integrated-force gate at n=16, while substantial
opposing local errors remain. Both rules produce the same uniform affine
control above, but nonuniform contact highlights the quadrature and force
distribution difference.

This manufactured problem uses additional applied tractions and constrained
lateral motion. It does not reproduce the original free impact or establish
that its remaining mesh error has one cause. Three CPU tests check the
analytic uniform solution and the independent-integration nonuniform
control; [the evidence](evidence/contact-static-diagnosis-20261107/)
retains both the failed initial loading and the corrected runs.
