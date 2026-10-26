import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).with_name("finalize_existing_checks546.py")
SPEC = importlib.util.spec_from_file_location("finalize546_tested", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Merit:
    def __init__(self, point=True, joint=False, depth=True):
        self.point = point
        self.joint = joint
        self.depth = depth
        self.calls = []

    def assess_point_clearance_construction_progress(self, before, after):
        self.calls.append("point")
        return {"maximum_depth_nonincreasing": self.point,
                "squared_depth_nonincreasing": self.point}

    def assess_joint_clearance_construction_progress(self, *args):
        self.calls.append("joint")
        return {"construction_progress": self.joint}

    def assess_construction_depth_progress(self, *args):
        self.calls.append("depth")
        return {"construction_progress": self.depth}


def test_default_mode_and_trust_are_unchanged():
    args = M.make_parser().parse_args([
        "--candidate", "c.npz", "--baseline-state", "b.json",
        "--point-summary", "p.json", "--surface-report", "s.json", "--out", "out",
    ])
    assert args.point_continuation is False
    assert args.trust_mm == 1.0
    command = M.arithmetic_command("c.npz", "b.json", "out/arithmetic", 0.375)
    assert command[-2:] == ["--trust-mm", "0.375"]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "x"])
def test_trust_must_be_finite_positive(bad):
    with pytest.raises(ValueError):
        M.validate_trust_mm(bad)


def test_point_mode_uses_pinned_merits_and_existing_or_policy():
    owner = Merit(point=True, joint=False, depth=True)
    result = M.assess_point_continuation(owner, {}, {}, {}, {})
    assert result["accepted"] is True
    assert result["point_depth_nonincrease"] is True
    assert result["joint_or_depth_construction_progress"] is True
    assert owner.calls == ["point", "joint", "depth"]


def test_point_mode_accepts_joint_route_but_rejects_depth_regression():
    assert M.assess_point_continuation(Merit(point=True, joint=True, depth=False),
                                       {}, {}, {}, {})["accepted"] is True
    rejected = M.assess_point_continuation(Merit(point=False, joint=True, depth=True),
                                           {}, {}, {}, {})
    assert rejected["accepted"] is False
    assert rejected["point_depth_nonincrease"] is False
    assert M.assess_point_continuation(Merit(point=True, joint=False, depth=False),
                                       {}, {}, {}, {})["accepted"] is False


def test_reused_point_evidence_is_candidate_bound_and_self_free(tmp_path):
    candidate = tmp_path / "candidate.npz"
    np.savez(candidate,
             points_world_m=np.zeros((3, 3), np.float32),
             source_O_faces=np.array([[0, 1, 2]], np.int32))
    digest = M.sha(candidate)
    report = {"candidate_sha256": digest, "complete_surface_audit": True,
              "exact_PLC_surface_pairs": 0, "inside_node_bank_pairs": 7,
              "points": 3, "faces": 1}
    assert M._checked_point_report(report, digest, candidate) == 7
    with pytest.raises(ValueError):
        M._checked_point_report({**report, "candidate_sha256": "0" * 64},
                                digest, candidate)
    self_pairs = tmp_path / "surface-self-pairs.npy"
    np.save(self_pairs, np.empty((0, 2), np.int64))
    M._checked_self_pairs(self_pairs)
    np.save(self_pairs, np.array([[0, 1]], np.int64))
    with pytest.raises(ValueError):
        M._checked_self_pairs(self_pairs)


def test_pinned_audit_reuse_detects_modified_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}\n")
    pins = {report: M.sha(report)}
    M.verify_pins(pins)
    report.write_text("{\"changed\": true}\n")
    with pytest.raises(ValueError, match="pinned input/source changed"):
        M.verify_pins(pins)


def test_actual_545_arithmetic_report_schema_is_accepted():
    import json

    fixture = (SOURCE.parent / "test-fixtures" /
               "arithmetic-report-point-fit001-fca5eaf622d88336038b46c57f7171c872e82b9754c3287fcf63138d948f4fb7.json")
    assert M.sha(fixture) == "fca5eaf622d88336038b46c57f7171c872e82b9754c3287fcf63138d948f4fb7"
    report = json.loads(fixture.read_text())
    checked = M.validate_arithmetic_report(report, 1.0, 237)
    assert checked["maximum_source_distance_mm"] == report["maximum_source_distance_mm"]
    assert checked["maximum_source_distance_mm"] <= 12.0


def test_arithmetic_schema_does_not_accept_guessed_source_alias():
    import json

    fixture = (SOURCE.parent / "test-fixtures" /
               "arithmetic-report-point-fit001-fca5eaf622d88336038b46c57f7171c872e82b9754c3287fcf63138d948f4fb7.json")
    report = json.loads(fixture.read_text())
    report["maximum_source_displacement_mm"] = report.pop("maximum_source_distance_mm")
    with pytest.raises(ValueError, match="maximum_source_distance_mm"):
        M.validate_arithmetic_report(report, 1.0, 237)
