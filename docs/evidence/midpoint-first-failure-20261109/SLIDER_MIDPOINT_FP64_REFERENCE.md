# Two-owner midpoint equation reference

This independent CPU FP64 diagnostic solves the coupled FEM and freely reacting, 10 g, vertical prismatic slider midpoint equations over 10 ms. It includes gravity, the same reference-area pressure law and the same initial state as the retained refined RK4 slider reference. It is a diagnostic reference for implementing and checking the native transaction; it does not run Matter or MetalWorld and it is not a replay, rollback, interface-impulse, or dynamically coupled native qualification.

The two numerical methods share the independently authored semidiscrete FEM and contact model, but use different temporal formulations: an implicit midpoint endpoint solve with finite-difference Jacobian versus explicit RK4. The midpoint experiment solves all nodal coordinates and the slider coordinate together. The slider force is constructed as the opposite of the FEM contact force, so this test alone cannot prove independently published owner reactions. It evaluates a mass-weighted finite-difference symmetry check of the FEM/slider contact cross-block after the force exceeds 0.01 N. For the constrained slider, the reported x/y contact torque is the FEM-side moment about the origin; a real owner must also account for its constraint reaction. The midpoint contact work is evaluated from each owner’s displacement at the common stage force.

| Comparison with 0.25 µs RK4 | 15.625 µs midpoint | 7.8125 µs midpoint |
| --- | ---: | ---: |
| 1 ms-binned force | 0.00178% | 0.000459% |
| Sampled peak force | 0.108% | 0.0262% |
| Total normal impulse | 0.00158% | 0.000418% |
| Maximum cumulative impulse | 0.0851% | 0.0213% |
| FEM contact work | 0.00651% | 0.00166% |
| Slider contact work | 0.00672% | 0.00171% |

Cumulative impulse is linearly interpolated from midpoint step integrals onto the retained 2 µs RK4 output clock. The sampled peaks use each method’s own sample times, so their differences include sampling error. At 15.625 µs the maximum independent reconstructed per-step FEM and slider momentum residuals were 1.28e-12 and 4.52e-12 N·s; the maximum summed momentum residual was 7.00e-16 N·s. The maximum cross-block symmetry error was 2.73e-13 relative. The finer step reduces the reported trajectory errors. These numbers apply to this 27-node CPU fixture only and do not establish spatial convergence or native physical accuracy.

Reproduce with the two `numi_contact_fem_slider_fp64_midpoint` invocations at the step sizes above and `python3 tests/contact/analyze_slider_midpoint_reference.py`. The retained inputs include source, executable and RK4 reference hashes; the two relevant CTests passed. An initial attempt used the wrong build directory and found no tests; that failure is preserved separately as `slider-midpoint-fp64-wrong-build*.log`, followed by the successful build and tests in the contact build directory.
