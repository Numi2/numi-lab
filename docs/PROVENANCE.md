# Provenance

MetalRobo's runtime physics and Metal kernels are original implementations.
No external physics engine is linked or called.

The Franka kinematic and inertial constants are factual model data adapted
from Franka Robotics' official `franka_description` package:

- <https://github.com/frankarobotics/franka_description>
- Pinned source: tag `2.8.1`, commit
  `02afaae282d4a8e10d7d2f781b23b3515c303ce5`
- Source records: `robots/fer/inertials.yaml`, `kinematics.yaml`,
  `joint_limits.yaml`, and `dynamics.yaml`
- Upstream license file:
  <https://github.com/frankarobotics/franka_description/blob/2.8.1/LICENSE>

The primitive collision spheres are a MetalRobo approximation of the upstream
coarse link geometry; no upstream mesh is copied into this repository.

The Unitree G1 topology, transforms, mass properties, joint limits, official
primitive colliders, and IMU frames are factual model data adapted from:

- `unitreerobotics/unitree_ros`
- Pinned commit `aa0f5c68b5aba347bad409e71b6430407da758d7`
- `robots/g1_description/g1_29dof_rev_1_0.{urdf,xml}`
- Upstream BSD-3-Clause license:
  <https://github.com/unitreerobotics/unitree_ros/blob/aa0f5c68b5aba347bad409e71b6430407da758d7/LICENSE>

The named G1 reset/PD/armature training preset is adapted from the pinned
Unitree RL Lab file documented in [G1_SPEC](G1_SPEC.md). That file carries an
Isaac Lab BSD-3-Clause SPDX notice; the containing repository license is
Apache-2.0. Full retained notices and the exact boundary are in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

All imported G1 link-frame joint origins and primitive positions are compiled
into MetalRobo's COM-centred runtime coordinates. This coordinate conversion,
fixed-link inertial folding, inverse-tensor calculation, runtime ABI, solvers,
collision implementation, and Metal kernels are MetalRobo implementation
work.

Algorithm references used for the independent articulated-body
implementation:

- Roy Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
- MuJoCo programming and computation documentation:
  <https://mujoco.readthedocs.io/>

References guide equations and compatibility targets. Their source code is not
part of the MetalRobo runtime.
