# Reacting slider contact wiring: not qualified

The disabled research path now has initial wiring for evaluating the existing
reference-area law at a nonrotating articulated stage plane. The same final
contact reduction can also forward the body's actual contact moment to
MetalWorld's body-wrench input, while the existing midpoint publisher remains
the sole source of axial generalized effort. The moment is taken about the
body center and converted from the full-step impulse by the physical timestep.

The fixed-plane admission guard remains in force. No coupled Matter/MetalWorld
contact step can use this wiring yet. In particular, this code does not by
itself certify the body's support reaction, full-step momentum and kinematics,
interface work, the actual start and end plane clearances, or whole-world
failure restoration and retry. Those remain necessary before enabling it.
Forwarding a contact moment is not evidence of an independently measured
constraint reaction.

The source passed host and Metal syntax checks; see
slider-moving-plane-wrench-syntax-001.json and its warning logs. No rebuilt
native shader or runtime test was run for these edits. Existing fixed-plane
and owner-only evidence refers to its separately hashed binaries.

The midpoint FEM certificate also includes twice the one-axis stage residual
in its combined physical momentum norm if that row is present, and rejects
unsupported rigid capacities. This covers the stage equation only; rounding
in the owner's actual published endpoint still needs an independent check.
The changed shader passed syntax only; see
slider-rigid-stage-certificate-syntax-001.json.
