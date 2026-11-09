# Native coupled slider fixture: prepared, not run

The separate `NM_ENABLE_UNQUALIFIED_SLIDER_DIAGNOSTIC` CMake option defaults
OFF. Default builds retain the unconditional runtime rejection; a diagnostic
build still requires the separate default-off slider configuration and has a
distinct device fingerprint. This switch is for collecting evidence, not
promotion or routine simulation.

A native coupled diagnostic has been prepared using the actual Matter program
and a dynamically reacting MetalWorld slider. For each accepted step it
records FEM contact impulse and moment, the independently observed rigid-body
momentum change and input wrench, ABA support moment, clearance, and the two
contact works. It captures the owner data after the transaction rather than
pretending a prescribed plane is dynamic. Its source and both diagnostic and
default runtime branches pass host syntax checks.

The fixture now also has separate pre-contact and after-nonzero-contact
late-failure modes. They compare the entire physical Matter snapshot and rigid
q/v with the checkpoint, and compare a retry to an uninterrupted continuation.
Those modes have passed host syntax checking but have not been run. The fixture
has not been built or run with full MetalWorld and Matter shader libraries, so
it establishes no whole-environment restoration, force accuracy, work
convergence, torque balance, or spatial convergence. All predeclared gates
remain in force and the production guard remains closed.

The first fixture draft incorrectly initialized the slider at rest. Before any
native run, review against `fem_slider_fp64_reference.cpp` caught this mismatch;
it now uses the reference's 0.02 m/s upward initial velocity. It also requires
the two compensated midpoint diagnostic settings, emits the actual float
timestep, contact coefficient, activation width and every reference-area
weight, and records an independent combined FEM/rigid physical residual.

`tests/contact/analyze_native_slider.py` applies the preregistered temporal,
force, impulse, clearance, balance and residual gates; it separately reports
contact work and does not invent a work threshold. A synthetic trace produced
from the retained reference passed, while separate injected impulse, torque
and residual failures were rejected by
`tests/contact/test_analyze_native_slider.py`. This checks analysis behavior,
not native physics.
