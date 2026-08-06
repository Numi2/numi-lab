from pathlib import Path

import numpy as np

from metalrobo.g1_legs_policy import expand_leg_actions, inspect_onnx


def test_expand_leg_actions_preserves_legs_and_holds_upper_body() -> None:
    legs = np.arange(12, dtype=np.float32)[None, :]
    full = expand_leg_actions(legs)
    assert full.shape == (1, 29)
    np.testing.assert_array_equal(full[:, :12], legs)
    np.testing.assert_array_equal(full[:, 12:], 0.0)


def test_inspect_onnx_rejects_unknown_artifact(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.onnx"
    unknown.write_bytes(b"not an onnx model")
    try:
        inspect_onnx(unknown)
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("unknown artifact was accepted")
