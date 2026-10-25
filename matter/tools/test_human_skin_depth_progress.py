
import copy
import json
import math
from pathlib import Path

from human_skin_fat_bone_audit import (
    assess_construction_depth_progress,
    assess_joint_clearance_construction_progress,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def synthetic_200_reports(*, after_mode="progress"):
    """Build matched strict-schema reports using the measured run-200 metrics."""
    before_points = load("before-points.json")
    before_surface = load("before-surface.json")
    candidate_before = "a" * 64
    candidate_after = "b" * 64
    point_count = 18430
    face_count = 36815
    before_max = 0.0033851283017981575
    before_squared = 0.0001887646246939005
    after_squared = 0.00018832885822506846

    template_surface = copy.deepcopy(before_surface)

    def set_surface_total(report, target):
        current = sum(0 if b["result"] is None else b["result"]["count"]
                      for b in report["banks"])
        delta = target - current
        if delta < 0:
            for bank in reversed(report["banks"]):
                result = bank["result"]
                if not result or not result["triangle_pairs"]:
                    continue
                take = min(-delta, result["count"])
                result["triangle_pairs"] = result["triangle_pairs"][:-take]
                result["count"] -= take
                delta += take
                if not delta:
                    break
        elif delta > 0:
            bank = next(b for b in report["banks"] if b["result"] is not None)
            result = bank["result"]
            start = result["count"]
            result["triangle_pairs"].extend([[900000 + i, 800000 + i]
                                             for i in range(delta)])
            result["count"] += delta
            delta = 0
        assert delta == 0
        report["surface_point_count"] = point_count
        report["surface_face_count"] = face_count
        return report

    before_surface = set_surface_total(template_surface, 3414)
    before_surface["candidate_sha256"] = candidate_before

    x_before = math.sqrt((before_squared - before_max ** 2) / 299)
    x_after = math.sqrt((after_squared - before_max ** 2) / 299)

    def point_report(candidate, nonmax):
        report = copy.deepcopy(before_points)
        report["candidate_sha256"] = candidate
        report["points"] = point_count
        report["faces"] = face_count
        report["query_pair_count"] = point_count * 185
        report["inside_nodes"] = 300
        report["inside_node_bank_pairs"] = 300
        report["inside_witnesses"] = [
            {"query_row": i, "bank_body": 1, "bank_stable_id": 1,
             "classification": "inside",
             "nearest_distance_m": repr(before_max if i == 0 else nonmax)}
            for i in range(300)
        ]
        return report

    bp = point_report(candidate_before, x_before)
    ap = point_report(candidate_after, x_after if after_mode == "progress" else x_before)
    after_surface = copy.deepcopy(before_surface)
    after_surface["candidate_sha256"] = candidate_after
    return bp, ap, before_surface, after_surface


def test_measured_run200_equal_counts_and_maximum_but_strict_squared_depth_progress():
    bp, ap, bs, ass = synthetic_200_reports()
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert result["classification"] == "CONSTRUCTION_DEPTH_PROGRESS"
    assert result["construction_progress"]
    assert (result["before_point_pairs"], result["after_point_pairs"]) == (300, 300)
    assert (result["before_surface_pairs"], result["after_surface_pairs"]) == (3414, 3414)
    assert result["before_maximum_depth_m"] == result["after_maximum_depth_m"] == 0.0033851283017981575
    assert result["before_squared_depth_m2"] == pytest.approx(0.0001887646246939005, rel=1e-12)
    assert result["after_squared_depth_m2"] == pytest.approx(0.00018832885822506846, rel=1e-12)
    assert result["no_introduced_typed_pairs"] and result["typed_bank_count"] == 185
    assert not result["geometry_admitted"] and not result["native_admission"]
    assert result["accepted_physical_steps"] == 0
    assert not assess_joint_clearance_construction_progress(bp, ap, bs, ass)["construction_progress"]


def test_unchanged_squared_depth_is_not_progress():
    bp, ap, bs, ass = synthetic_200_reports(after_mode="same")
    # Set every after witness to the exact baseline distance.
    ap["inside_witnesses"] = copy.deepcopy(bp["inside_witnesses"])
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["strict_squared_depth_progress"]
    assert result["classification"] == "NO_CONSTRUCTION_DEPTH_PROGRESS"


def test_point_count_increase_is_not_progress_even_if_depth_falls():
    bp, ap, bs, ass = synthetic_200_reports()
    extra = copy.deepcopy(ap["inside_witnesses"][-1])
    extra["query_row"] = 400
    extra["nearest_distance_m"] = "0.000001"
    ap["inside_witnesses"].append(extra)
    ap["inside_nodes"] += 1
    ap["inside_node_bank_pairs"] += 1
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["point_pair_count_nonincreasing"]
    assert not result["construction_progress"]


def test_new_typed_point_pair_is_not_progress_at_equal_count():
    bp, ap, bs, ass = synthetic_200_reports()
    ap["inside_witnesses"][1]["query_row"] = 400
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["no_introduced_typed_pairs"]
    assert result["introduced_point_pairs"]
    assert not result["construction_progress"]


def test_surface_count_or_new_surface_pair_is_not_progress():
    bp, ap, bs, ass = synthetic_200_reports()
    ass = copy.deepcopy(ass)
    bank = next(b for b in ass["banks"] if b["result"] is not None and b["result"]["count"] > 0)
    bank["result"]["triangle_pairs"][0] = [999999, 999998]
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["no_introduced_typed_pairs"]
    assert result["introduced_surface_pair_count"] == 1
    assert not result["construction_progress"]

    bp, ap, bs, ass = synthetic_200_reports()
    ass = copy.deepcopy(ass)
    bank = next(b for b in ass["banks"] if b["result"] is not None)
    bank["result"]["triangle_pairs"].append([999999, 999998])
    bank["result"]["count"] += 1
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["surface_pair_count_nonincreasing"]
    assert not result["construction_progress"]


def test_maximum_depth_increase_is_not_progress():
    bp, ap, bs, ass = synthetic_200_reports()
    ap["inside_witnesses"][0]["nearest_distance_m"] = "0.0033851283017982"
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert not result["maximum_depth_nonincreasing"]
    assert not result["construction_progress"]


def test_audit_completeness_hash_topology_and_typed_bank_checks_are_reused():
    bp, ap, bs, ass = synthetic_200_reports()
    malformed = copy.deepcopy(ass)
    malformed["banks"] = malformed["banks"][:-1]
    with pytest.raises(ValueError):
        assess_construction_depth_progress(bp, ap, bs, malformed)
    malformed = copy.deepcopy(ap)
    malformed["exact_PLC_surface_pairs"] = 1
    with pytest.raises(ValueError):
        assess_construction_depth_progress(bp, malformed, bs, ass)


def test_roundoff_only_squared_change_is_not_progress():
    bp, ap, bs, ass = synthetic_200_reports()
    ap["inside_witnesses"] = copy.deepcopy(bp["inside_witnesses"])
    ap["inside_witnesses"][1]["nearest_distance_m"] = repr(
        float(ap["inside_witnesses"][1]["nearest_distance_m"]) * (1.0 - 1e-15))
    result = assess_construction_depth_progress(bp, ap, bs, ass)
    assert result["squared_depth_improvement_m2"] <= result["squared_depth_roundoff_threshold_m2"]
    assert not result["construction_progress"]


import pytest
