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

Algorithm references used for the independent articulated-body
implementation:

- Roy Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
- MuJoCo programming and computation documentation:
  <https://mujoco.readthedocs.io/>

References guide equations and compatibility targets. Their source code is not
part of the MetalRobo runtime.
