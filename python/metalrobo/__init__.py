"""Learning, data, and deployment tools for MetalRobo's native runtime.

The Python package deliberately has no physics executor or rollout scheduler.
Swift owns rollout orchestration, persistent Metal contexts own simulation
state, and MLX is loaded only when a learning export is requested. Keeping
the package surface lazy lets artifact, motion, and contract tools run without
initializing an otherwise unused Apple GPU context.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CONTACT_VALID": (".tactile", "CONTACT_VALID"),
    "DEPTH_SATURATED": (".tactile", "DEPTH_SATURATED"),
    "FILTERED_TARGET": (".tactile", "FILTERED_TARGET"),
    "G1_ACTOR_FRAME_SIZE": (".g1_policy", "G1_ACTOR_FRAME_SIZE"),
    "G1_ACTOR_HISTORY": (".g1_policy", "G1_ACTOR_HISTORY"),
    "G1_ACTOR_OBSERVATION_SIZE": (
        ".g1_policy", "G1_ACTOR_OBSERVATION_SIZE"
    ),
    "G1_ACTION_COUNT": (".g1_policy", "G1_ACTION_COUNT"),
    "G1DeployableSensors": (".g1_policy", "G1DeployableSensors"),
    "G1NumpyPolicy": (".g1_policy", "G1NumpyPolicy"),
    "G1ObservationHistory": (".g1_policy", "G1ObservationHistory"),
    "MLXActorCritic": (".mlx_policy_learning", "MLXActorCritic"),
    "MLXPPOConfiguration": (
        ".mlx_policy_learning", "MLXPPOConfiguration"
    ),
    "MLXPolicyBatch": (".mlx_policy_learning", "MLXPolicyBatch"),
    "MLXPolicyLearner": (".mlx_policy_learning", "MLXPolicyLearner"),
    "MetalRoboError": (".native", "MetalRoboError"),
    "NativePolicyDenseLayer": (
        ".mlx_policy_learning", "NativePolicyDenseLayer"
    ),
    "NativePolicyPack": (".mlx_policy_learning", "NativePolicyPack"),
    "NativePolicyRollout": (
        ".mlx_policy_learning", "NativePolicyRollout"
    ),
    "NativeTactileObservation": (
        ".tactile", "NativeTactileObservation"
    ),
    "SAMPLE_VALID": (".tactile", "SAMPLE_VALID"),
    "TACTILE_ABI_VERSION": (".tactile", "TACTILE_ABI_VERSION"),
    "TactileCalibrationRecord": (
        ".tactile", "TactileCalibrationRecord"
    ),
    "TactileDeviceBuffers": (".tactile", "TactileDeviceBuffers"),
    "TactileLayout": (".tactile", "TactileLayout"),
    "TactileObservationContract": (
        ".tactile", "TactileObservationContract"
    ),
    "TactileObservationSnapshot": (
        ".tactile", "TactileObservationSnapshot"
    ),
    "TactileSensorContract": (".tactile", "TactileSensorContract"),
    "UnitreeG1MuJoCoRunner": (".g1_policy", "UnitreeG1MuJoCoRunner"),
    "append_calibration_record": (".tactile", "append_calibration_record"),
    "compile_episode_manifest": (".episode", "compile_episode_manifest"),
    "export_g1_coreml": (".g1_policy", "export_g1_coreml"),
    "export_g1_mlx": (".g1_policy", "export_g1_mlx"),
    "export_g1_onnx": (".g1_policy", "export_g1_onnx"),
    "import_unitree_g1_mjlab_policy": (
        ".unitree_mjlab_policy", "import_unitree_g1_mjlab_policy"
    ),
    "library_version": (".native", "library_version"),
    "load_tactile_checkpoint_contract": (
        ".tactile", "load_tactile_checkpoint_contract"
    ),
    "read_policy_pack": (".mlx_policy_learning", "read_policy_pack"),
    "read_policy_rollout_pack": (
        ".mlx_policy_learning", "read_policy_rollout_pack"
    ),
    "save_tactile_checkpoint_contract": (
        ".tactile", "save_tactile_checkpoint_contract"
    ),
    "unitree_g1_deployment_contract": (
        ".native", "unitree_g1_deployment_contract"
    ),
    "write_depth_preview_pgm": (".tactile", "write_depth_preview_pgm"),
}

__all__ = sorted(_EXPORTS)
__version__ = "0.4.0"


def __getattr__(name: str) -> Any:
    """Resolve public tools on first use and cache the resulting object."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
