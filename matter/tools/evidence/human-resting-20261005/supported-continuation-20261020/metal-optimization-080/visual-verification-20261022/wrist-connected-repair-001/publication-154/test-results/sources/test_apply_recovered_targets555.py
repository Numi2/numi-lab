import importlib.util
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("apply_recovered_targets555", HERE / "apply_recovered_targets555.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
TARGETS = Path(__file__).resolve().parents[2] / "current-parent-normal-targets-555" / "targets-001-recovered"


def test_actual_recovered_receipt_and_endpoint_rows_are_accepted():
    checked = MOD.verify_recovered_targets(TARGETS)
    assert checked["receipt_sha256"] == MOD.EXPECTED_RECOVERY_SHA256
    assert checked["report"]["status"] == "INCOMPLETE_CURRENT_PARENT_NORMAL_OUTPUT_ANNOTATION"
    assert checked["report"]["output_annotation_status"] == "PASS_POSTPROCESS_ONLY_RECOVERY"
    assert len(checked["report"]["targets"]) == 1248


def test_wrong_expected_recovery_pin_rejected():
    with pytest.raises(ValueError, match="recovery receipt pin mismatch"):
        MOD.verify_recovered_targets(TARGETS, "0" * 64)


def test_missing_recovery_receipt_rejected(tmp_path):
    copied = tmp_path / "target"
    shutil.copytree(TARGETS, copied)
    (copied / "postprocess-recovery-555.json").unlink()
    with pytest.raises(ValueError, match="missing postprocess recovery receipt"):
        MOD.verify_recovered_targets(copied)


def test_tampered_report_rejected_by_receipt(tmp_path):
    copied = tmp_path / "target"
    shutil.copytree(TARGETS, copied)
    p = copied / "nearest-union-exits.json"
    p.write_text(p.read_text().replace('"query_rows": 1248', '"query_rows": 1247', 1))
    with pytest.raises(ValueError, match="recovered report hash does not match receipt"):
        MOD.verify_recovered_targets(copied)


def test_tampered_npz_rejected_by_receipt(tmp_path):
    copied = tmp_path / "target"
    shutil.copytree(TARGETS, copied)
    import numpy as np
    p = copied / "nearest-union-exits.npz"
    with np.load(p, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["target_world_f32_m"] = payload["target_world_f32_m"].copy()
    payload["target_world_f32_m"][0, 0] += np.float32(1e-5)
    np.savez_compressed(p, **payload)
    with pytest.raises(ValueError, match="recovered NPZ hash does not match receipt"):
        MOD.verify_recovered_targets(copied)
