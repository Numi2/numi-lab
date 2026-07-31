"""Learning, data, and deployment tools for MetalRobo's native runtime.

The Python package deliberately has no physics executor or rollout scheduler.
Swift owns rollout orchestration, and persistent Metal contexts own simulation
state. MLX consumes compact batches and publishes fingerprinted PolicyPacks.
"""

from .episode import compile_episode_manifest
from .g1_policy import (
    G1_ACTOR_FRAME_SIZE,
    G1_ACTOR_HISTORY,
    G1_ACTOR_OBSERVATION_SIZE,
    G1_ACTION_COUNT,
    G1DeployableSensors,
    G1NumpyPolicy,
    G1ObservationHistory,
    UnitreeG1MuJoCoRunner,
    export_g1_coreml,
    export_g1_mlx,
    export_g1_onnx,
)
from .mlx_policy_learning import (
    MLXActorCritic,
    MLXPPOConfiguration,
    MLXPolicyBatch,
    MLXPolicyLearner,
    NativePolicyDenseLayer,
    NativePolicyPack,
    NativePolicyRollout,
    read_policy_pack,
    read_policy_rollout_pack,
)
from .native import (
    MetalRoboError,
    library_version,
    unitree_g1_deployment_contract,
)
from .unitree_mjlab_policy import import_unitree_g1_mjlab_policy
from .tactile import (
    CONTACT_VALID,
    DEPTH_SATURATED,
    FILTERED_TARGET,
    SAMPLE_VALID,
    TACTILE_ABI_VERSION,
    NativeTactileObservation,
    TactileCalibrationRecord,
    TactileDeviceBuffers,
    TactileLayout,
    TactileObservationContract,
    TactileObservationSnapshot,
    TactileSensorContract,
    append_calibration_record,
    load_tactile_checkpoint_contract,
    save_tactile_checkpoint_contract,
    write_depth_preview_pgm,
)

__all__ = [
    "CONTACT_VALID",
    "DEPTH_SATURATED",
    "FILTERED_TARGET",
    "G1_ACTOR_FRAME_SIZE",
    "G1_ACTOR_HISTORY",
    "G1_ACTOR_OBSERVATION_SIZE",
    "G1_ACTION_COUNT",
    "G1DeployableSensors",
    "G1NumpyPolicy",
    "G1ObservationHistory",
    "MLXActorCritic",
    "MLXPPOConfiguration",
    "MLXPolicyBatch",
    "MLXPolicyLearner",
    "MetalRoboError",
    "NativePolicyDenseLayer",
    "NativePolicyPack",
    "NativePolicyRollout",
    "NativeTactileObservation",
    "SAMPLE_VALID",
    "TACTILE_ABI_VERSION",
    "TactileCalibrationRecord",
    "TactileDeviceBuffers",
    "TactileLayout",
    "TactileObservationContract",
    "TactileObservationSnapshot",
    "TactileSensorContract",
    "UnitreeG1MuJoCoRunner",
    "append_calibration_record",
    "compile_episode_manifest",
    "export_g1_coreml",
    "export_g1_mlx",
    "export_g1_onnx",
    "library_version",
    "import_unitree_g1_mjlab_policy",
    "load_tactile_checkpoint_contract",
    "read_policy_pack",
    "read_policy_rollout_pack",
    "save_tactile_checkpoint_contract",
    "unitree_g1_deployment_contract",
    "write_depth_preview_pgm",
]

__version__ = "0.4.0"
