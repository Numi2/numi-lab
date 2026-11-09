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
