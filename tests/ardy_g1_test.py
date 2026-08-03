import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from metalrobo.ardy_g1 import (
    _G1_SKEL_JOINTS,
    _JOINT_FRAME_CONTRACT,
    _joint_frame_offsets,
    _project_mechanism_trajectory,
    _qpos_from_motion,
)
from metalrobo.ardy_interaction_convert import (
    _G1_JOINT_LOWER,
    _G1_JOINT_UPPER,
    _G1_JOINT_VELOCITY,
)
from metalrobo.g1_motion_retarget import G1Kinematics


_G1_NAMES = (
    "pelvis_skel", "left_hip_pitch_skel", "left_hip_roll_skel",
    "left_hip_yaw_skel", "left_knee_skel", "left_ankle_pitch_skel",
    "left_ankle_roll_skel", "left_toe_base", "right_hip_pitch_skel",
    "right_hip_roll_skel", "right_hip_yaw_skel", "right_knee_skel",
    "right_ankle_pitch_skel", "right_ankle_roll_skel", "right_toe_base",
    "waist_yaw_skel", "waist_roll_skel", "waist_pitch_skel",
    "left_shoulder_pitch_skel", "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel", "left_elbow_skel", "left_wrist_roll_skel",
    "left_wrist_pitch_skel", "left_wrist_yaw_skel", "left_hand_roll_skel",
    "right_shoulder_pitch_skel", "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel", "right_elbow_skel", "right_wrist_roll_skel",
    "right_wrist_pitch_skel", "right_wrist_yaw_skel", "right_hand_roll_skel",
)
_G1_PARENTS = np.asarray(
    (-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13,
     0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 17, 26, 27,
     28, 29, 30, 31, 32),
    dtype=np.int64,
)


class ARDYG1Test(unittest.TestCase):
    def test_solver_com_pose_converts_to_root_link_origin(self) -> None:
        model = G1Kinematics(
            links=("pelvis",),
            joints=(),
            joint_indices={},
            root_center_of_mass=np.asarray((0.01, -0.02, -0.08)),
            source_sha256="fixture",
        )
        rotation = Rotation.from_euler("xyz", (0.3, -0.2, 0.4)).as_matrix()
        link_origin = np.asarray((0.5, -0.25, 0.9))
        com_position = link_origin + rotation @ model.root_center_of_mass
        np.testing.assert_allclose(
            model.root_link_position(com_position, rotation),
            link_origin,
            atol=1.0e-12,
        )

    def test_native_joint_frame_conversion_round_trips_hinge_angles(self) -> None:
        desired = np.linspace(-0.2, 0.2, len(_G1_SKEL_JOINTS))
        local = np.repeat(np.eye(3)[None], len(_G1_NAMES), axis=0)
        offsets = _joint_frame_offsets()
        name_to_index = {name: index for index, name in enumerate(_G1_NAMES)}
        for output, (mechanism, axis, _) in enumerate(_JOINT_FRAME_CONTRACT):
            _, frame_offset = offsets[mechanism]
            ardy_axis = {"x": "z", "y": "x", "z": "y"}[axis]
            adjusted = Rotation.from_euler(ardy_axis, desired[output]).as_matrix()
            local[name_to_index[_G1_SKEL_JOINTS[output]]] = (
                frame_offset.T @ adjusted
            )
        global_rotations = np.empty_like(local)
        for joint, parent in enumerate(_G1_PARENTS):
            global_rotations[joint] = (
                local[joint]
                if parent < 0
                else global_rotations[parent] @ local[joint]
            )
        rotations_6d = np.concatenate(
            (global_rotations[:, :, 0], global_rotations[:, :, 1]), axis=1
        )[None]
        qpos = _qpos_from_motion(
            np.zeros((1, 3)), rotations_6d, _G1_NAMES, _G1_PARENTS
        )
        np.testing.assert_allclose(qpos[0, 7:], desired, atol=2.0e-6)
        np.testing.assert_allclose(qpos[0, :3], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(qpos[0, 3:7], (1.0, 0.0, 0.0, 0.0))

    def test_mechanism_projection_preserves_source_and_enforces_limits(self) -> None:
        source = np.zeros((5, len(_G1_SKEL_JOINTS)), dtype=np.float64)
        source[2, 5] = float(_G1_JOINT_UPPER[5]) + 1.0
        retained = source.copy()
        projected, evidence = _project_mechanism_trajectory(source, 25.0)
        np.testing.assert_array_equal(source, retained)
        self.assertTrue(np.all(projected >= _G1_JOINT_LOWER))
        self.assertTrue(np.all(projected <= _G1_JOINT_UPPER))
        velocity_ratio = (
            np.abs(np.diff(projected, axis=0)) * 25.0
            / _G1_JOINT_VELOCITY[None]
        )
        self.assertLessEqual(float(np.max(velocity_ratio)), 1.0 + 1.0e-6)
        self.assertGreater(evidence["corrected_sample_count"], 0)
        self.assertTrue(evidence["source_preserved"])


if __name__ == "__main__":
    unittest.main()
