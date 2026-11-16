# 3D rod and v15 free-hand research regression

The CTest `metalrobo.crow.rod_v15_rigid_hand_reference` checks a linearized 3D rod attached to a free six-degree-of-freedom body using the v15 left-hand fixture's mass and inertia. It verifies hard attachment, force and torque, linear and angular momentum, node ordering and a double-precision KKT reference in four one-step controls. The lever arm is synthetic and the hand is not attached to the articulated skeleton. The v15 inertial properties are engineering priors, not measured American-crow data.

This standalone kernel is not MetalWorld, a nonlinear wing or feather model, contact, aerodynamics, a performance benchmark or flight qualification. The inputs, full output, source, failed earlier candidate and independent analysis are archived in [evidence commit 8fd4991](https://github.com/Numi2/numi-gpu-physics-pigeon-sim/commit/8fd4991).
