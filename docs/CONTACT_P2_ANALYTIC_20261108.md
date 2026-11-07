# P1/P2 fixed-state continuum comparison

This tests a genuinely different finite-element spatial approximation against
independent continuum integrals. Both P1 and P2 use the same compressible
Neo-Hookean volume energy and the same finite-band plane energy density as
the fixed-law experiment. Both use high-order surface quadrature here: this
is a different contact discretization from the existing nodal P1 rule, so it
is not a parity test of that discrete rule and cannot alone qualify the
original impact trajectory.

## Prediction registered before execution

For the 20 mm cube and existing material constants, prescribe a smooth
longitudinal deformation u_z=A sin(pi z/L), A=0.2 mm, with zero lateral
displacement. Its exact continuum energy is a scalar 1D quadrature of the
known diagonal Neo-Hookean density. Independently prescribe a surface gap
g(x,y)=95 µm+2 µm cos(pi x/L)cos(pi y/L). Its exact plane energy and total
normal force are independent tensor-product Gauss-Legendre integrals, checked
at orders 48 and 96 to relative error <=1e-10. Extend the second field
smoothly through the block, with no change of geometry or material law.

On n=2,4,8,16 tetrahedral meshes, interpolate each field with P1 and P2
and evaluate volume and boundary integrals at scikit-fem quadrature order 8.
Check positive deformation determinants and positive surface quadrature gaps.
Predict that at n>=4 P2 will have smaller relative volume-energy error and
smaller relative normal-force error than P1. Report actual errors and
unchanged 0.25% force criterion as a fixed-state comparison only. The 0.25%
dynamic force-bin and impulse gates remain unchanged and are not passed by
a static field.

Run the four meshes sequentially, each under a 180-second process-group
timeout with single-thread BLAS and no GPU. Retain source, environment,
reference values, all outputs, elapsed time and failures. This calculation
establishes approximation properties of manufactured smooth fields, not a
bound on impact-force error or material validation.

The first n=2 attempt stopped at output serialization because the element
count was a NumPy int32. No numerical result was emitted. Its traceback is
retained in the first-run evidence. The separately recorded corrected
attempt changes only JSON type conversion.

## Result

All four corrected runs completed. The independent reference Gauss-Legendre
orders agreed to 1.22e-15 relative, all sampled determinants and contact gaps
were positive, and P2 had lower bulk-energy and surface-force error than P1
at n>=4, as predicted.

| Mesh | P1 volume-energy error | P2 volume-energy error | P1 plane-force error | P2 plane-force error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 18.960% | 0.3963% | 8.112% | 0.1771% |
| 4 | 5.038% | 0.02598% | 2.229% | 0.01343% |
| 8 | 1.279% | 0.001641% | 0.5737% | 0.0008820% |
| 16 | 0.3210% | 0.0001028% | 0.1445% | 0.00005581% |

Each error is absolute difference divided by the independently integrated
continuum reference for the **separate** manufactured field. P2 fixed-state
force error meets 0.25% at these mesh sizes and P1 meets it at n=16. This is
not a dynamic 1 ms bin or impulse qualification. The surface law here is
integrated with order-8 face quadrature, not the existing nodal P1 discrete
rule; both represent the same continuum energy but their finite-mesh
responses need not agree.

This study shows that a P2 spatial discretization can substantially reduce
bulk and surface approximation errors for these smooth fields. It does not
identify the actual n=16→32 impact discrepancy or prove that P2 dynamics,
nonpenetration, contact work, replay or force are qualified. The
[first failure](evidence/contact-p2-analytic-20261108/) and
[corrected full results](evidence/contact-p2-analytic-corrected-20261108/)
are both retained.
