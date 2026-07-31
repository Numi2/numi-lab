# Unitree G1 29-DoF model specification

This document freezes the first Unitree G1 target for MetalRobo. It is an
implementation contract, not a claim that every robot sold as “G1” has the
same motors or locked joints.

## Frozen target

The first target is Unitree's **`g1_29dof_rev_1_0`**, with
`mode_machine = 5`:

- 29 actuated revolute joints: 12 leg, 3 waist, and 14 arm joints.
- A free `pelvis` root.
- No articulated hand. The two rubber hands are fixed inertial attachments.
- Hip pitch/roll gear ratios `{14.3, 22.5}` and W4010 wrist motors.
- All three waist joints are revolute in the selected URDF.

This revision is the best first baseline because Unitree marks it up to date,
gives it a stable revisioned filename, and Unitree's current RL Lab G1
configuration points to the same URDF/USD revision. Before deploying a policy
to hardware, read the robot's `mode_machine` value; Unitree says it is shown in
the app under **Device → Data → Robot → Machine Type**. A mode-5 model must not
be presented as compatible with mode 11, 13, 15, 18, or a locked-waist robot
without a separately compiled asset.

Snapshot date: 2026-07-28.

## Pinned primary sources and licenses

| Purpose | Repository and immutable revision | Records used | License |
|---|---|---|---|
| Canonical model data | [`unitreerobotics/unitree_ros@aa0f5c68b5aba347bad409e71b6430407da758d7`](https://github.com/unitreerobotics/unitree_ros/tree/aa0f5c68b5aba347bad409e71b6430407da758d7/robots/g1_description) | [`README.md`](https://github.com/unitreerobotics/unitree_ros/blob/aa0f5c68b5aba347bad409e71b6430407da758d7/robots/g1_description/README.md), [`g1_29dof_rev_1_0.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/aa0f5c68b5aba347bad409e71b6430407da758d7/robots/g1_description/g1_29dof_rev_1_0.urdf), companion [`g1_29dof_rev_1_0.xml`](https://github.com/unitreerobotics/unitree_ros/blob/aa0f5c68b5aba347bad409e71b6430407da758d7/robots/g1_description/g1_29dof_rev_1_0.xml), and `meshes/*.STL` | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/aa0f5c68b5aba347bad409e71b6430407da758d7/LICENSE) |
| Hardware-facing joint ABI and command example | [`unitreerobotics/unitree_sdk2@21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b`](https://github.com/unitreerobotics/unitree_sdk2/tree/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b) | [`defines.h`](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/include/unitree/dds_wrapper/robots/g1/defines.h) and [`g1_ankle_swing_example.cpp`](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/example/g1/low_level/g1_ankle_swing_example.cpp) | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/LICENSE) |
| Simulator-side low-level command semantics | [`unitreerobotics/unitree_mujoco@ae6a8403e272733e9996ef59990880330496177f`](https://github.com/unitreerobotics/unitree_mujoco/tree/ae6a8403e272733e9996ef59990880330496177f) | [`unitree_sdk2_bridge.h`](https://github.com/unitreerobotics/unitree_mujoco/blob/ae6a8403e272733e9996ef59990880330496177f/simulate/src/unitree_sdk2_bridge.h) and [`g1_joint_index_dds.md`](https://github.com/unitreerobotics/unitree_mujoco/blob/ae6a8403e272733e9996ef59990880330496177f/unitree_robots/g1/g1_joint_index_dds.md) | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_mujoco/blob/ae6a8403e272733e9996ef59990880330496177f/LICENSE) |
| Official training preset and contact-body naming | [`unitreerobotics/unitree_rl_lab@4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) | [`assets/robots/unitree.py`](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) and [`velocity_env_cfg.py`](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py) | [Apache-2.0](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/LICENCE) |

If model constants or meshes are copied into MetalRobo, retain the applicable
Unitree copyright notice, BSD-3-Clause text, and disclaimer. If RL Lab
configuration is adapted, meet Apache-2.0 attribution and changed-file notice
requirements. The licenses do not grant trademark endorsement.

### Source precedence

The selected URDF is canonical for topology, transforms, full inertia tensors,
position limits and declared collision geometry. The SDK is canonical for the
29-element motor order and low-level command fields. URDF, companion MJCF, and
RL Lab actuator limits are independent named presets because the official
sources conflict. The companion MJCF also supplies an explicit free root and
IMU declarations. RL Lab supplies a reproducible training controller, not
hardware truth.

This precedence matters because the official files disagree in a few places;
those differences are recorded below rather than averaged away.

## Generalized state and MetalRobo mapping

The selected URDF contains 39 links and 38 joints: 29 revolute and 9 fixed.
Fold the fixed links into their nearest dynamic ancestors, producing 30
dynamic links (`pelvis` plus one child for each actuated joint). The raw URDF
mass is **33.34114202 kg** and is preserved by the fold.

The companion MJCF explicitly gives `pelvis` a `free` joint. Therefore:

- actuated joint position/velocity/action count: 29;
- floating-root velocity coordinates: 6;
- `nv = 6 + 29 = 35`;
- with root translation plus quaternion, `nq = 7 + 29 = 36`.

MetalRobo must keep root position, root quaternion, root linear velocity, and
root angular velocity in dedicated state arrays. `MR_MAX_DOF = 32` then covers
the 29 actuated joints; it would not cover a flattened 35-element generalized
velocity vector. The 30-link folded tree fits `MR_MAX_LINKS = 33`.

Use these import conventions:

- URDF distances are metres, angles radians, mass kilograms, inertia kg·m².
- URDF `origin rpy="r p y"` becomes
  `R0 = Rz(y) * Ry(p) * Rx(r)`.
- `R0` is the child-to-parent orientation at `q = 0`; store it as MetalRobo
  quaternion `(x, y, z, w)` without inversion.
- The URDF joint axis is in the joint/child frame and maps directly to
  `MRJointGPU.axis`.
- For a revolute joint,
  `R_parent_child(q) = R0 * Rot(axis, q)`.
- Link 0 is the floating `pelvis`. Joint index `j` drives child link `j + 1`.

## Topologically ordered joint and link table

The joint indices exactly match Unitree SDK2's `JointIndex` enum and the
official Unitree MuJoCo DDS index document.

| j | joint | parent → child | origin xyz | origin rpy | axis | range rad | effort N·m | velocity rad/s |
|---:|---|---|---|---|---|---|---:|---:|
| 0 | `left_hip_pitch_joint` | 0 `pelvis` → 1 `left_hip_pitch_link` | `0 0.064452 -0.1027` | `0 0 0` | `0 1 0` | [-2.5307, 2.8798] | 88 | 32 |
| 1 | `left_hip_roll_joint` | 1 `left_hip_pitch_link` → 2 `left_hip_roll_link` | `0 0.052 -0.030465` | `0 -0.1749 0` | `1 0 0` | [-0.5236, 2.9671] | 139 | 20 |
| 2 | `left_hip_yaw_joint` | 2 `left_hip_roll_link` → 3 `left_hip_yaw_link` | `0.025001 0 -0.12412` | `0 0 0` | `0 0 1` | [-2.7576, 2.7576] | 88 | 32 |
| 3 | `left_knee_joint` | 3 `left_hip_yaw_link` → 4 `left_knee_link` | `-0.078273 0.0021489 -0.17734` | `0 0.1749 0` | `0 1 0` | [-0.087267, 2.8798] | 139 | 20 |
| 4 | `left_ankle_pitch_joint` | 4 `left_knee_link` → 5 `left_ankle_pitch_link` | `0 -9.4445e-05 -0.30001` | `0 0 0` | `0 1 0` | [-0.87267, 0.5236] | 35 | 30 |
| 5 | `left_ankle_roll_joint` | 5 `left_ankle_pitch_link` → 6 `left_ankle_roll_link` | `0 0 -0.017558` | `0 0 0` | `1 0 0` | [-0.2618, 0.2618] | 35 | 30 |
| 6 | `right_hip_pitch_joint` | 0 `pelvis` → 7 `right_hip_pitch_link` | `0 -0.064452 -0.1027` | `0 0 0` | `0 1 0` | [-2.5307, 2.8798] | 88 | 32 |
| 7 | `right_hip_roll_joint` | 7 `right_hip_pitch_link` → 8 `right_hip_roll_link` | `0 -0.052 -0.030465` | `0 -0.1749 0` | `1 0 0` | [-2.9671, 0.5236] | 139 | 20 |
| 8 | `right_hip_yaw_joint` | 8 `right_hip_roll_link` → 9 `right_hip_yaw_link` | `0.025001 0 -0.12412` | `0 0 0` | `0 0 1` | [-2.7576, 2.7576] | 88 | 32 |
| 9 | `right_knee_joint` | 9 `right_hip_yaw_link` → 10 `right_knee_link` | `-0.078273 -0.0021489 -0.17734` | `0 0.1749 0` | `0 1 0` | [-0.087267, 2.8798] | 139 | 20 |
| 10 | `right_ankle_pitch_joint` | 10 `right_knee_link` → 11 `right_ankle_pitch_link` | `0 9.4445e-05 -0.30001` | `0 0 0` | `0 1 0` | [-0.87267, 0.5236] | 35 | 30 |
| 11 | `right_ankle_roll_joint` | 11 `right_ankle_pitch_link` → 12 `right_ankle_roll_link` | `0 0 -0.017558` | `0 0 0` | `1 0 0` | [-0.2618, 0.2618] | 35 | 30 |
| 12 | `waist_yaw_joint` | 0 `pelvis` → 13 `waist_yaw_link` | `0 0 0` | `0 0 0` | `0 0 1` | [-2.618, 2.618] | 88 | 32 |
| 13 | `waist_roll_joint` | 13 `waist_yaw_link` → 14 `waist_roll_link` | `-0.0039635 0 0.044` | `0 0 0` | `1 0 0` | [-0.52, 0.52] | 35 | 30 |
| 14 | `waist_pitch_joint` | 14 `waist_roll_link` → 15 `torso_link` | `0 0 0` | `0 0 0` | `0 1 0` | [-0.52, 0.52] | 35 | 30 |
| 15 | `left_shoulder_pitch_joint` | 15 `torso_link` → 16 `left_shoulder_pitch_link` | `0.0039563 0.10022 0.24778` | `0.27931 5.4949e-05 -0.00019159` | `0 1 0` | [-3.0892, 2.6704] | 25 | 37 |
| 16 | `left_shoulder_roll_joint` | 16 `left_shoulder_pitch_link` → 17 `left_shoulder_roll_link` | `0 0.038 -0.013831` | `-0.27925 0 0` | `1 0 0` | [-1.5882, 2.2515] | 25 | 37 |
| 17 | `left_shoulder_yaw_joint` | 17 `left_shoulder_roll_link` → 18 `left_shoulder_yaw_link` | `0 0.00624 -0.1032` | `0 0 0` | `0 0 1` | [-2.618, 2.618] | 25 | 37 |
| 18 | `left_elbow_joint` | 18 `left_shoulder_yaw_link` → 19 `left_elbow_link` | `0.015783 0 -0.080518` | `0 0 0` | `0 1 0` | [-1.0472, 2.0944] | 25 | 37 |
| 19 | `left_wrist_roll_joint` | 19 `left_elbow_link` → 20 `left_wrist_roll_link` | `0.1 0.00188791 -0.01` | `0 0 0` | `1 0 0` | [-1.972222054, 1.972222054] | 25 | 37 |
| 20 | `left_wrist_pitch_joint` | 20 `left_wrist_roll_link` → 21 `left_wrist_pitch_link` | `0.038 0 0` | `0 0 0` | `0 1 0` | [-1.614429558, 1.614429558] | 5 | 22 |
| 21 | `left_wrist_yaw_joint` | 21 `left_wrist_pitch_link` → 22 `left_wrist_yaw_link` | `0.046 0 0` | `0 0 0` | `0 0 1` | [-1.614429558, 1.614429558] | 5 | 22 |
| 22 | `right_shoulder_pitch_joint` | 15 `torso_link` → 23 `right_shoulder_pitch_link` | `0.0039563 -0.10021 0.24778` | `-0.27931 5.4949e-05 0.00019159` | `0 1 0` | [-3.0892, 2.6704] | 25 | 37 |
| 23 | `right_shoulder_roll_joint` | 23 `right_shoulder_pitch_link` → 24 `right_shoulder_roll_link` | `0 -0.038 -0.013831` | `0.27925 0 0` | `1 0 0` | [-2.2515, 1.5882] | 25 | 37 |
| 24 | `right_shoulder_yaw_joint` | 24 `right_shoulder_roll_link` → 25 `right_shoulder_yaw_link` | `0 -0.00624 -0.1032` | `0 0 0` | `0 0 1` | [-2.618, 2.618] | 25 | 37 |
| 25 | `right_elbow_joint` | 25 `right_shoulder_yaw_link` → 26 `right_elbow_link` | `0.015783 0 -0.080518` | `0 0 0` | `0 1 0` | [-1.0472, 2.0944] | 25 | 37 |
| 26 | `right_wrist_roll_joint` | 26 `right_elbow_link` → 27 `right_wrist_roll_link` | `0.1 -0.00188791 -0.01` | `0 0 0` | `1 0 0` | [-1.972222054, 1.972222054] | 25 | 37 |
| 27 | `right_wrist_pitch_joint` | 27 `right_wrist_roll_link` → 28 `right_wrist_pitch_link` | `0.038 0 0` | `0 0 0` | `0 1 0` | [-1.614429558, 1.614429558] | 5 | 22 |
| 28 | `right_wrist_yaw_joint` | 28 `right_wrist_pitch_link` → 29 `right_wrist_yaw_link` | `0.046 0 0` | `0 0 0` | `0 0 1` | [-1.614429558, 1.614429558] | 5 | 22 |

## Folded dynamic-link inertials

Every selected URDF inertial frame has zero RPY, so each tensor below is the
full symmetric inertia about the COM in its link frame. Entries marked `†`
include fixed child inertials and are deterministic derivatives of the URDF;
the other rows are verbatim URDF values.

For fixed transform `(R, t)`, fold child inertials using:

`M = Σm`, `c = Σm(R c_child + t) / M`, and
`I_c = Σ[R I_child Rᵀ + m((d·d)E - ddᵀ)]`, where
`d = R c_child + t - c`.

| link | name | mass kg | COM xyz m | `(Ixx,Ixy,Ixz,Iyy,Iyz,Izz)` kg·m² |
|---:|---|---:|---|---|
| 0 | `pelvis` † | 3.814 | `0 0 -0.076030060304` | `(0.010554882086,0,2.1e-06,0.0093147820861,0,0.0079185)` |
| 1 | `left_hip_pitch_link` | 1.35 | `0.002741 0.047791 -0.02606` | `(0.001811,3.68e-05,-3.44e-05,0.0014193,0.000171,0.0012812)` |
| 2 | `left_hip_roll_link` | 1.52 | `0.029812 -0.001045 -0.087934` | `(0.0023773,-3.8e-06,-0.0003908,0.0024123,1.84e-05,0.0016595)` |
| 3 | `left_hip_yaw_link` | 1.702 | `-0.057709 -0.010981 -0.15078` | `(0.0057774,-0.0005411,-0.0023948,0.0076124,-0.0007072,0.003149)` |
| 4 | `left_knee_link` | 1.932 | `0.005457 0.003964 -0.12074` | `(0.011329,4.82e-05,-4.49e-05,0.011277,-0.0007146,0.0015168)` |
| 5 | `left_ankle_pitch_link` | 0.074 | `-0.007269 0 0.011137` | `(8.4e-06,0,-2.9e-06,1.89e-05,0,1.26e-05)` |
| 6 | `left_ankle_roll_link` | 0.608 | `0.026505 0 -0.016425` | `(0.0002231,2e-07,8.91e-05,0.0016161,-1e-07,0.0016667)` |
| 7 | `right_hip_pitch_link` | 1.35 | `0.002741 -0.047791 -0.02606` | `(0.001811,-3.68e-05,-3.44e-05,0.0014193,-0.000171,0.0012812)` |
| 8 | `right_hip_roll_link` | 1.52 | `0.029812 0.001045 -0.087934` | `(0.0023773,3.8e-06,-0.0003908,0.0024123,-1.84e-05,0.0016595)` |
| 9 | `right_hip_yaw_link` | 1.702 | `-0.057709 0.010981 -0.15078` | `(0.0057774,0.0005411,-0.0023948,0.0076124,0.0007072,0.003149)` |
| 10 | `right_knee_link` | 1.932 | `0.005457 -0.003964 -0.12074` | `(0.011329,-4.82e-05,4.49e-05,0.011277,0.0007146,0.0015168)` |
| 11 | `right_ankle_pitch_link` | 0.074 | `-0.007269 0 0.011137` | `(8.4e-06,0,-2.9e-06,1.89e-05,0,1.26e-05)` |
| 12 | `right_ankle_roll_link` | 0.608 | `0.026505 0 -0.016425` | `(0.0002231,-2e-07,8.91e-05,0.0016161,1e-07,0.0016667)` |
| 13 | `waist_yaw_link` | 0.214 | `0.003494 0.000233 0.018034` | `(0.00010673,2.703e-06,-7.631e-06,0.00010422,-2.01e-07,0.0001625)` |
| 14 | `waist_roll_link` | 0.086 | `0 2.3e-05 0` | `(7.079e-06,0,0,6.339e-06,0,8.245e-06)` |
| 15 | `torso_link` † | 7.817 | `0.0020313344633 0.00033972674939 0.1845971452` | `(0.12164651964,3.1110210298e-05,-0.0037428195854,0.10977258486,-1.5429925458e-05,0.027521919415)` |
| 16 | `left_shoulder_pitch_link` | 0.718 | `0 0.035892 -0.011628` | `(0.0004291,-9.2e-06,6.4e-06,0.000453,2.26e-05,0.000423)` |
| 17 | `left_shoulder_roll_link` | 0.643 | `-0.000227 0.00727 -0.063243` | `(0.0006177,-1e-06,8.7e-06,0.0006912,-5.3e-06,0.0003894)` |
| 18 | `left_shoulder_yaw_link` | 0.734 | `0.010773 -0.002949 -0.072009` | `(0.0009988,7.9e-06,0.0001412,0.0010605,-2.86e-05,0.0004354)` |
| 19 | `left_elbow_link` | 0.6 | `0.064956 0.004454 -0.010062` | `(0.0002891,6.53e-05,1.72e-05,0.0004152,-5.6e-06,0.0004197)` |
| 20 | `left_wrist_roll_link` | 0.08544498 | `0.01713944778 0.00053759094 0.00000048864` | `(0.00004821544023,-0.00000424511021,0.00000000510599,0.00003722899093,-0.00000000123525,0.00005482106541)` |
| 21 | `left_wrist_pitch_link` | 0.48404956 | `0.02299989837 -0.00111685314 -0.00111658096` | `(0.00016579646273,-0.00001231206746,0.00001231699194,0.00042954057410,0.00000081417712,0.00042953697654)` |
| 22 | `left_wrist_yaw_link` † | 0.25457647 | `0.070824430201 0.0001917452912 0.0016174161392` | `(0.00015044517917,0.00003760275409,-0.0000029549415859,0.00064311326853,0.0000037754842413,0.0005601139475)` |
| 23 | `right_shoulder_pitch_link` | 0.718 | `0 -0.035892 -0.011628` | `(0.0004291,9.2e-06,6.4e-06,0.000453,-2.26e-05,0.000423)` |
| 24 | `right_shoulder_roll_link` | 0.643 | `-0.000227 -0.00727 -0.063243` | `(0.0006177,1e-06,8.7e-06,0.0006912,5.3e-06,0.0003894)` |
| 25 | `right_shoulder_yaw_link` | 0.734 | `0.010773 0.002949 -0.072009` | `(0.0009988,-7.9e-06,0.0001412,0.0010605,2.86e-05,0.0004354)` |
| 26 | `right_elbow_link` | 0.6 | `0.064956 -0.004454 -0.010062` | `(0.0002891,-6.53e-05,1.72e-05,0.0004152,5.6e-06,0.0004197)` |
| 27 | `right_wrist_roll_link` | 0.08544498 | `0.01713944778 -0.00053759094 0.00000048864` | `(0.00004821544023,0.00000424511021,0.00000000510599,0.00003722899093,0.00000000123525,0.00005482106541)` |
| 28 | `right_wrist_pitch_link` | 0.48404956 | `0.02299989837 0.00111685314 -0.00111658096` | `(0.00016579646273,0.00001231206746,0.00001231699194,0.00042954057410,-0.00000081417712,0.00042953697654)` |
| 29 | `right_wrist_yaw_link` † | 0.25457647 | `0.070824430201 -0.0001917452912 0.0016174161392` | `(0.00015044517917,-0.00003760275409,-0.0000029549415859,0.00064311326853,-0.0000037754842413,0.0005601139475)` |

The fixed-link fold is:

| Fixed child | Dynamic parent | transform xyz / rpy | treatment |
|---|---|---|---|
| `pelvis_contour_link` | `pelvis` | `0 0 0` / `0 0 0` | Fold 0.001 kg inertia and retain collision mesh. |
| `logo_link` | `torso_link` | `0.0039635 0 -0.044` / `0 0 0` | Fold 0.001 kg inertia and retain collision mesh. |
| `head_link` | `torso_link` | `0.0039635 0 -0.044` / `0 0 0` | Fold 1.036 kg inertia and retain collision mesh. |
| `imu_in_torso` | `torso_link` | `-0.03959 -0.00224 0.14792` / `0 0 0` | Massless sensor frame. |
| `imu_in_pelvis` | `pelvis` | `0.04525 0 -0.08339` / `0 0 0` | Massless sensor frame. |
| `d435_link` | `torso_link` | `0.0576235 0.01753 0.42987` / `0 0.8307767239493009 0` | Massless camera frame. |
| `mid360_link` | `torso_link` | `0.0002835 0.00003 0.428434` / `3.141592653589793 0.05112069379091391 0` | Massless lidar frame. |
| `left_rubber_hand` | `left_wrist_yaw_link` | `0.0415 0.003 0` / `0 0 0` | Fold 0.170 kg inertia; no collision in URDF. |
| `right_rubber_hand` | `right_wrist_yaw_link` | `0.0415 -0.003 0` / `0 0 0` | Fold 0.170 kg inertia; no collision in URDF. |

The companion MJCF is not numerically identical to this exact URDF fold. For
example, it reports pelvis mass 3.813 kg and torso mass 7.818 kg, while the
URDF's 0.001 kg pelvis contour produces 3.814 and its torso/head/logo sum is
7.817. MetalRobo's canonical model must preserve the selected URDF, not move
that gram to match the generated MJCF.

## Actuation semantics

Unitree SDK2 exposes, per motor, `mode`, feed-forward torque `tau`, desired
position `q`, desired velocity `dq`, `kp`, and `kd`. Unitree's MuJoCo bridge
implements:

`tau_command = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)`

MetalRobo's FP64 articulated-actuation evaluator implements that command law
and then clamps the actuator contribution to the selected model's effort
range before passive dry friction is added. It provides disabled, named-model
PD, command-local PD, and direct-effort modes; floating-root coordinates
cannot be actuated. Continuous-joint position error uses the shortest signed
modulo-\(2\pi\) displacement. Near-zero dry friction is explicitly a
controller-local cancellation approximation, not a complete coupled
set-valued stiction model. `mode = 1` means enabled and `mode = 0` means
disabled in the official SDK example. The example publishes at 2 ms. Its
particular Kp/Kd values are an example controller, not motor constants.

For this target, use `mode_pr = 0`: SDK coordinates 4/5 and 10/11 represent
ankle pitch/roll, and coordinates 13/14 represent waist roll/pitch. SDK
`mode_pr = 1` reinterprets the coupled ankle channels as motor A/B coordinates.
That transmission mapping belongs in a hardware adapter; it must not create
additional simulated DoFs.

The canonical saturation values are those in the joint table. The official
source discrepancies are:

| Joint family | Selected URDF | Companion MJCF | Unitree RL Lab preset |
|---|---|---|---|
| hip pitch/yaw and waist yaw | 88 N·m, 32 rad/s | ±88 N·m | 88 N·m, 32 rad/s |
| hip roll and knee | 139 N·m, 20 rad/s | ±139 N·m | 139 N·m, 20 rad/s |
| ankle pitch/roll and waist roll/pitch | **35 N·m, 30 rad/s** | **±50 N·m** | **25 N·m, 37 rad/s** |
| shoulder, elbow, wrist roll | 25 N·m, 37 rad/s | ±25 N·m | 25 N·m, 37 rad/s |
| wrist pitch/yaw | 5 N·m, 22 rad/s | ±5 N·m | 5 N·m, 22 rad/s |

Do not silently use MJCF force ranges or RL training limits as hardware
limits. If parity work needs them, expose named model presets such as
`unitree_urdf_rev_1_0`, `unitree_mjcf_rev_1_0`, and
`unitree_rl_lab_4960b84`.

The RL Lab preset uses armature `0.01` on every joint and these implicit PD
gains:

- hip pitch/roll/yaw: Kp 100, Kd 2;
- knee: Kp 150, Kd 4;
- waist yaw: Kp 200, Kd 5;
- waist roll/pitch: Kp 40, Kd 5;
- ankle: Kp 40, Kd 2;
- shoulder, elbow, and all wrists: Kp 40, Kd 1.

The selected URDF itself supplies no armature or controller gains.
`makeUnitreeG1EngineModel()` defaults to the exact, explicitly named
`unitree_rl_lab_4960b84` actuator preset: effort and velocity limits, Kp/Kd,
and armature all come from the pinned RL Lab record. The source-exact
`unitree_urdf_rev_1_0` and `unitree_mjcf_rev_1_0` presets are separately
selectable. Their fingerprints differ and therefore change the compiled world
and task fingerprints; values are never silently mixed across sources. In
particular, ankle and waist roll/pitch effort is 25 N m for RL Lab, 35 N m for
URDF, and 50 N m for the companion MJCF.

An official RL Lab-compatible reset pose uses pelvis link-origin position
`(0, 0, 0.8)`, zero joint velocities, and zero joint positions except:

| joints | position rad |
|---|---:|
| left/right hip pitch | -0.1 |
| left/right knee | 0.3 |
| left/right ankle pitch | -0.2 |
| left/right shoulder pitch | 0.3 |
| left/right shoulder roll | +0.25 / -0.25 |
| left/right elbow | 0.97 |
| left/right wrist roll | +0.15 / -0.15 |

`MRBodyStateGPU` stores COM translation, so the compiled
identity-orientation root reset is `(0, 0, 0.723969939696)` after adding the
pelvis COM offset `(0, 0, -0.076030060304)`. Runtime joint anchors and
primitive collision positions, sole frames, and IMU frames are likewise
shifted from their source link frames into COM-centred runtime body
coordinates. The source tables in this document remain link-local.

## Collision geometry and feet

The URDF declares 36 collision elements:

- 24 STL meshes: pelvis contour; both hip pitch, hip roll, hip yaw, knee, and
  ankle-pitch links; torso, logo, and head; both shoulder-yaw, elbow,
  wrist-roll, wrist-pitch, and wrist-yaw links.
- Four cylinders: one shoulder-pitch and one shoulder-roll primitive per arm.
- Eight foot spheres: four on each ankle-roll link.

The current compiled model retains all 12 primitive records. The eight foot
spheres participate in simulation; the four shoulder cylinders carry
`MR_SHAPE_FLAG_SIMULATION_DISABLED`. Generic cylinder/plane narrowphase now
exists, but G1 cannot enable these shapes until the required cylinder
self-collision pair classes and exclusions are executable. They must not be
silently treated as complete collision geometry.

The shoulder-pitch cylinders have radius 0.03 m and length 0.05 m, centered at
`(0, ±0.04, -0.01)` with RPY `(0, π/2, 0)`. The shoulder-roll cylinders have
radius 0.03 m and length 0.03 m, centered at
`(-0.004, ±0.006, -0.053)` with identity orientation.

Each `*_ankle_roll_link` has the same four official contact spheres:

| point | center xyz m | radius m |
|---|---|---:|
| rear-left | `-0.05 0.025 -0.03` | 0.005 |
| rear-right | `-0.05 -0.025 -0.03` | 0.005 |
| front-left | `0.12 0.03 -0.03` | 0.005 |
| front-right | `0.12 -0.03 -0.03` | 0.005 |

The URDF does not name a `foot` or `sole` frame. The official RL Lab task
identifies feet by matching `.*ankle_roll.*`. MetalRobo should therefore use:

- canonical foot bodies: `left_ankle_roll_link` and
  `right_ankle_roll_link`;
- derived, MetalRobo-owned sole frames:
  `T_ankle_roll_sole = translation(0.035, 0, -0.035)`, identity rotation.

The derived x/y values are the centroid of the four sphere centers; z is the
bottom surface of their 5 mm radius. Label these as derived frames, not
official Unitree frames. A foot contact signal is the sum (or maximum, when
explicitly requested by a task) of impulses from its four sphere colliders.

The current MetalRobo ABI represents the eight spheres and finite cylinders
exactly, but not STL collision meshes. For cylinders, only cylinder/plane
collision is executable today. The first locomotion target should:

1. preserve the eight official foot spheres exactly;
2. add explicitly provenance-marked MetalRobo box/capsule proxies for gross
   body contacts, or add mesh/convex and cylinder collider support;
3. never describe generated proxies as Unitree geometry;
4. retain the official mesh path and source SHA in any compiled-asset
   manifest so exact geometry can replace proxies without changing topology.

Do not infer friction, restitution, solver stiffness, or damping from the
URDF; it specifies none. Those are simulator/task parameters.

## IMUs and contact sensors

The model defines two identity-oriented IMU frames:

| logical sensor | parent | position xyz m | hardware/API mapping |
|---|---|---|---|
| pelvis IMU | `pelvis` | `0.04525 0 -0.08339` | `LowState.imu_state` |
| torso IMU | `torso_link` | `-0.03959 -0.00224 0.14792` | DDS topic `rt/secondary_imu` in the SDK example |

The companion MJCF declares gyro and accelerometer sensors at both sites. For
each, it records gyro `noise="5e-4" cutoff="34.9"` and accelerometer
`noise="1e-2" cutoff="157"`. Keep noise disabled for deterministic parity by
default; expose those exact attributes as an optional named sensor preset.

For a sensor fixed at world offset `r` from its body origin, compute:

- `gyro = R_world_sensorᵀ * omega_world`;
- `a_site = a_origin + alpha × r + omega × (omega × r)`;
- `accelerometer = R_world_sensorᵀ * (a_site - gravity)`.

Also expose the pelvis orientation as a normalized `(x, y, z, w)` quaternion
and derive RPY only at the API edge. Unitree messages and MuJoCo records may
use different serialized quaternion orders; adapters must reorder explicitly.

Neither the selected URDF nor its companion MJCF declares a physical contact
sensor. Unitree RL Lab creates a simulator contact-force sensor over all robot
bodies and selects `.*ankle_roll.*` for foot rewards. MetalRobo contact-force,
air-time, and undesired-contact observations are therefore derived simulation
signals, not emulated hardware sensors.

## Variant boundary

Unitree's pinned README currently lists these relevant variants:

| Asset family | `mode_machine` | hip pitch/roll ratio | wrist | waist model | body DoF |
|---|---:|---|---|---|---:|
| `g1_29dof_rev_1_0` (selected) | 5 | 14.3 / 22.5 | 4010 | three revolute joints | 29 |
| `g1_29dof_mode_11` | 11 | 22.5 / 22.5 | 4010 | unlocked | 29 |
| `g1_29dof_mode_12` | 12 | 22.5 / 22.5 | 4010 | roll and pitch fixed in URDF | 27 actuated in URDF |
| `g1_29dof_mode_13` | 13 | 14.3 / 22.5 | 5010 new | unlocked | 29 |
| `g1_29dof_mode_14` | 14 | 14.3 / 22.5 | 5010 new | roll and pitch fixed in URDF | 27 actuated in URDF |
| `g1_29dof_mode_15` | 15 | 22.5 / 22.5 | 5010 new | unlocked | 29 |
| `g1_29dof_mode_16` | 16 | 22.5 / 22.5 | 5010 new | roll and pitch fixed in URDF | 27 actuated in URDF |
| `g1_29dof_mode_18` | 18 | 22.5 / 22.5 | 5010 new | unlocked; different waist limits/actuators | 29 |
| `g1_29dof_lock_waist_rev_1_0` | 6 | 14.3 / 22.5 | 4010 | locked revision | separate asset |
| `g1_23dof_rev_1_0` | 4 | 14.3 / 22.5 | none | one waist and five joints per arm | 23 |

Mode-5 hand variants add either 7 actuated joints per hand or 12 Inspire-hand
joints per hand. A mode-15 Dex1 variant adds 2 per hand. These exceed the
29-joint ABI and require separate topology and capacity decisions. The old
unrevisioned `g1_29dof` (`mode_machine = 2`, ratios 14.3/14.5) is explicitly
deprecated by Unitree and must not be used as the initial target.

The “29dof” filenames for locked mode 12/14/16 describe the robot family, not
the count of revolute joints in those URDFs: waist roll and pitch are fixed, so
the files contain 27 actuated revolute joints. Compile each variant from its
own pinned asset rather than changing limits on the selected mode-5 tree.

## Implementation acceptance facts

The first compiled G1 asset is internally consistent when it has:

- source identity `unitree_ros@aa0f5c68.../g1_29dof_rev_1_0.urdf`;
- 29 joints in the exact SDK order above and 30 folded dynamic links;
- separate floating-root state (`nq = 36`, `nv = 35` conceptually);
- folded mass 33.34114202 kg;
- full off-diagonal inertia terms retained;
- an authoritative per-DoF stream containing URDF effort/velocity/position
  limits plus separately identified preset armature and gains;
- executable feed-forward/model-PD/custom-PD/direct-effort semantics with
  effort saturation and transactional rejection;
- exact eight foot spheres and derived sole frames clearly labeled;
- two IMU frames and no falsely claimed hardware contact sensor;
- `mode_machine = 5` and `mode_pr = 0` metadata;
- a redistribution notice for any copied Unitree model data or meshes.

These are model-ingestion and command-law facts, not actuator identification
or sim-to-real safety. The named RL Lab armature and gains reproduce a
training preset; they are not identified motor physics. Contact material,
motor delay,
friction, compliance, backlash, thermal limits, battery effects, and
controller frequency variation require separate calibration.
