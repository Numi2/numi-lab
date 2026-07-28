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

The dVRK-style surgical PSM research model is pinned to:

- `orbit-surgical/orbit-surgical` commit
  `6e47534f7d412e4be523116f250c992a63146883`
- Source records `psm_col.usd` and
  `orbit/surgical/assets/psm.py`
- Upstream BSD-3-Clause license:
  <https://github.com/orbit-surgical/orbit-surgical/blob/6e47534f7d412e4be523116f250c992a63146883/LICENCE>
- JHU Classic shaft/wrist controller records and kinematic cross-check at
  `jhu-dvrk/sawIntuitiveResearchKit` commit
  `53a401d014e5ef8a7d5e3ad05f0680084507662c`, using `PSM.json` and
  `LARGE_NEEDLE_DRIVER_400006.json`

The serial remote-center construction, COM conversion, primitive collision
decomposition, approximate inertias, and independent jaw coordinates are
MetalRobo research representations rather than hardware calibration. The
upstream fixed 0.1 kg `psm_tool_tip_link` is folded into its 0.1 kg moving yaw
parent; because the USD authors no inertia tensor for either prim, the
combined COM and tensor use a documented point-mass/parallel-axis
approximation in the canonical serial frame.

The procedural curved needle uses the official Medtronic GS-21 product facts
of a 37 mm, half-circle taper needle. Cross-section, density, tip/swage
profile, contact material, and grasp-zone values are named research defaults.
The training ring and peg-board dimensions are also research defaults. No
third-party needle mesh is redistributed.

Algorithm references used for the independent articulated-body
implementation:

- Roy Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
- MuJoCo programming and computation documentation:
  <https://mujoco.readthedocs.io/>

References guide equations and compatibility targets. Their source code is not
part of the MetalRobo runtime.
