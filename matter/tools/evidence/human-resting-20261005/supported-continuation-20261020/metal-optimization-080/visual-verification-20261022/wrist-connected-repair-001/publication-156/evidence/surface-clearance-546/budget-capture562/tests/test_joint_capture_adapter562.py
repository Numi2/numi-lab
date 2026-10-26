"""Focused tests for the separate 562 raw-joint capture hooks."""
from pathlib import Path
import hashlib
import importlib.util
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WRAPPER = HERE / "run_joint_full_coordinate_fit562.py"
spec = importlib.util.spec_from_file_location("joint_fit562_test", WRAPPER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeCapture:
    def __init__(self):
        self.calls = []

    def capture_solver_return(self, result, basis, **kwargs):
        self.calls.append((result, basis, kwargs))
        return result


def test_successful_solver_return_is_captured_unchanged():
    capture = FakeCapture()
    records = []
    basis = {"token": "full-coordinate-basis"}
    receipt = {"delta_semantics": "absolute_base_relative_millimeters"}
    result = {"success": True, "delta": [0.25]}

    def original(*args, **kwargs):
        return result

    wrapped = m.wrap_solver_for_raw_capture(
        original, capture, [basis], records, lambda returned: receipt
    )
    actual = wrapped("lp-input")
    assert actual is result
    assert capture.calls == [
        (
            result,
            basis,
            {
                "fit_receipt": receipt,
                "stage": "joint_lp_raw_before_area_alpha",
                "component": None,
                "records": records,
            },
        )
    ]


def test_failed_solver_return_is_not_captured():
    capture = FakeCapture()
    result = {"success": False, "status": 2}

    def original(*args, **kwargs):
        return result

    wrapped = m.wrap_solver_for_raw_capture(original, capture, [], [], lambda _: {})
    assert wrapped() is result
    assert capture.calls == []


def test_success_without_basis_fails_closed():
    capture = FakeCapture()
    result = {"success": True}

    def original(*args, **kwargs):
        return result

    wrapped = m.wrap_solver_for_raw_capture(original, capture, [], [], lambda _: {})
    with pytest.raises(ValueError, match="no captured basis token"):
        wrapped()
    assert capture.calls == []


def test_v2_pins_capture_source_and_records_units_and_pre_alpha_stage():
    text = WRAPPER.read_text()
    helper = m.CAPTURE_HELPER
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    assert digest == "fa7df8771a37200c4ed30e61d92ebf1b6227eed9da8a5ff45ab54b4ff299975e"
    assert f'EXPECTED_CAPTURE_HELPER_SHA = "{digest}"' in text
    assert '"delta_semantics": "absolute_base_relative_millimeters"' in text
    assert 'stage="joint_lp_raw_before_area_alpha"' in text
    assert '"raw_joint_directions_are_unadmitted": True' in text
    assert '"accepted_physical_steps": 0' in text


def test_v2_does_not_edit_frozen_v1_wrapper():
    v1 = HERE / "frozen-v1" / "run_joint_full_coordinate_fit561.py"
    assert v1.is_file()
    before = hashlib.sha256(v1.read_bytes()).hexdigest()
    assert before == "48742a5fcd6eb711847da808da938200cdc04889deaa6df37220079b3252b19a"
