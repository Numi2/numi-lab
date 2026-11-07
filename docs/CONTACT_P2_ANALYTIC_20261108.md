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
