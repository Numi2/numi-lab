"""Provenance-checked pinned G1 legs locomotion artifact inspection.

The published policy is legs-only: it owns the first twelve Unitree G1 motor
lanes while the native task remains responsible for the waist/arm hold lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np


G1_LEGS_FRAME_SIZE = 48
G1_LEGS_HISTORY = 5
G1_LEGS_OBSERVATION_SIZE = G1_LEGS_FRAME_SIZE * G1_LEGS_HISTORY
G1_LEGS_ACTIONS = 12
G1_LEGS_ACTION_SCALE = 0.5
G1_LEGS_CONTROL_DT = 0.02
G1_LEGS_GAIT_PERIOD = 0.7
G1_LEGS_ONNX_SHA256 = "1e21412a09f3af7fa2dbdec58de4d4600e2679862a1b24c502c0a02916bd440f"


@dataclass(frozen=True, slots=True)
class G1LegsPolicyArtifact:
    path: Path
    sha256: str
    layer_shapes: tuple[tuple[int, int], ...]


def inspect_onnx(path: str | Path) -> G1LegsPolicyArtifact:
    """Fail closed unless this is the pinned G1 legs actor graph."""
    target = Path(path).expanduser().resolve()
    digest_builder = hashlib.sha256()
    with target.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if digest != G1_LEGS_ONNX_SHA256:
        raise ValueError("ONNX SHA-256 does not match the pinned G1 legs artifact")
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("G1 legs ONNX inspection requires the onnx package") from error
    model = onnx.load(target)
    inputs = model.graph.input
    outputs = model.graph.output
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("G1 legs actor must have exactly one input and output")
    input_width = inputs[0].type.tensor_type.shape.dim[-1].dim_value
    output_width = outputs[0].type.tensor_type.shape.dim[-1].dim_value
    if input_width != G1_LEGS_OBSERVATION_SIZE or output_width != G1_LEGS_ACTIONS:
        raise ValueError("G1 legs actor ABI is not 240 observations to 12 leg actions")
    expected_ops = ("Sub", "Div", "Clip", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm")
    if tuple(node.op_type for node in model.graph.node) != expected_ops:
        raise ValueError("G1 legs actor graph is not the supported normalized ELU MLP")
    initializers = {value.name: tuple(value.dims) for value in model.graph.initializer}
    shapes = tuple(initializers[f"fcs.{index}.weight"] for index in range(4))
    if shapes != ((256, 240), (256, 256), (128, 256), (12, 128)):
        raise ValueError("G1 legs actor layer topology is unexpected")
    return G1LegsPolicyArtifact(target, digest, shapes)


def expand_leg_actions(actions: np.ndarray) -> np.ndarray:
    """Map the pinned actor's leg residuals into the full native G1 action ABI."""
    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != G1_LEGS_ACTIONS or not np.isfinite(values).all():
        raise ValueError("G1 legs action must be finite with exactly 12 leg lanes")
    result = np.zeros((*values.shape[:-1], 29), dtype=np.float32)
    result[..., :G1_LEGS_ACTIONS] = values
    return result
