# Mixed cohesive FEM residual Armijo slice

`NM_FEM_COHESIVE_RESIDUAL_ARMIJO_ROOT=<step|all>` enables a bounded
experimental line search for one static explicit-FEM object with active paired
mixed-mode cohesive faces. It tests the same eight `1, 1/2, ..., 1/128`
candidate fractions used by the existing no-contact path. Each candidate
reassembles bulk FEM forces and the production cohesive face force kernel,
then measures the exact free-node residual in impulse units. It uses the
existing final nonlinear residual threshold and does not change solver
tolerances or geometry gates.

Every trial reads the step-start accepted cohesive faces and writes only to
private candidate-face and force buffers. A rejected trial therefore cannot
advance opening, damage, or work history. Cohesive activation is still handled
by the normal accepted-step transaction. The current scope requires no
capsules, static boxes, self-contact, deformable contact, MPM, mutable
topology, mixed fields, or other generalized unknowns. Active faces must all
use the paired mixed-mode law. Unsupported worlds emit a skip reason.

Private face arenas are allocated only when the selector is present before
runtime construction, with caps of four environments, 4,096 faces, and 8 MiB
for the face-history plus force arenas. The CPU-only frame checker can be run
with:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  matter/tools/cardboard_mixed_cohesive_frame_check.cpp \
  -o /tmp/cardboard_mixed_cohesive_frame_check
/tmp/cardboard_mixed_cohesive_frame_check
```

Its asymmetric-penalty torque, rotation, tangent, and intact-work cases use
synthetic values. They are mathematical checks, not starch calibration or
native board qualification. Native line-search qualification remains a
separate required experiment.
