import copy
import json
from pathlib import Path

import pytest

from human_skin_fat_bone_audit import assess_joint_clearance_construction_progress as assess

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def actual_before():
    return load("before-points.json"), load("before-surface.json")


def pair_depth_squared(point_report):
    depths = [float(row["nearest_distance_m"]) for row in point_report["inside_witnesses"]]
    return max(depths, default=0.0), sum(v * v for v in depths)


def set_surface_pair_count(report, target):
    """Adjust a fixture's explicit per-bank arrays, retaining strict P089 schema."""
    result = copy.deepcopy(report)
    banks = result["banks"]
    current = sum((b["result"]["count"] if b["result"] is not None else 0) for b in banks)
    delta = target - current
    if delta < 0:
        for bank in reversed(banks):
            row = bank["result"]
            if row is None or not row["triangle_pairs"]:
                continue
            take = min(-delta, row["count"])
            row["triangle_pairs"] = row["triangle_pairs"][:-take]
            row["count"] -= take
            delta += take
            if delta == 0:
                break
    elif delta > 0:
        bank = next(b for b in banks if b["result"] is not None)
        row = bank["result"]
        for index in range(delta):
            row["triangle_pairs"].append({"construction_fixture_pair": index})
        row["count"] += delta
        delta = 0
    assert delta == 0
    return result


def test_actual_alpha_0125_joint_trial_passes_and_reports_construction_only():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0125-points.json"), load("alpha-0125-surface.json")
    result = assess(before_points, after_points, before_surface, after_surface)
    assert result["construction_progress"]
    assert result["classification"] == "JOINT_CLEARANCE_CONSTRUCTION_PROGRESS"
    assert (result["before_point_pairs"], result["after_point_pairs"]) == (14, 18)
    assert (result["before_surface_pairs"], result["after_surface_pairs"]) == (3687, 3658)
    assert (result["before_total_pairs"], result["after_total_pairs"]) == (3701, 3676)
    assert result["before_maximum_point_depth_m"] == pytest.approx(18.80935394156552e-6)
    assert result["after_maximum_point_depth_m"] == pytest.approx(16.79e-6, abs=0.02e-6)
    assert result["before_squared_point_depth_m2"] == pytest.approx(1.385e-9, rel=2e-3)
    assert result["after_squared_point_depth_m2"] == pytest.approx(8.428e-10, rel=2e-3)
    assert result["typed_bank_count"] == 185
    assert not result["geometry_admitted"] and not result["native_admission"]
    assert result["accepted_physical_steps"] == 0


def test_actual_alpha_00625_surface_decrease_but_equal_total_is_not_progress():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0625-points.json"), load("alpha-0625-surface.json")
    result = assess(before_points, after_points, before_surface, after_surface)
    assert (result["before_point_pairs"], result["after_point_pairs"]) == (14, 20)
    assert (result["before_surface_pairs"], result["after_surface_pairs"]) == (3687, 3681)
    assert result["strict_surface_pair_reduction"]
    assert not result["strict_total_pair_reduction"]
    assert result["before_total_pairs"] == result["after_total_pairs"] == 3701
    assert not result["construction_progress"]


def test_surface_3650_passes_but_3683_fails_strict_total_reduction():
    before_points, before_surface = actual_before()
    after_points = load("alpha-0125-points.json")
    after_surface = load("alpha-0125-surface.json")
    result_3650 = assess(before_points, after_points, before_surface,
                         set_surface_pair_count(after_surface, 3650))
    assert result_3650["construction_progress"]
    assert result_3650["after_total_pairs"] == 3668
    result_3683 = assess(before_points, after_points, before_surface,
                         set_surface_pair_count(after_surface, 3683))
    assert result_3683["strict_surface_pair_reduction"]
    assert result_3683["after_total_pairs"] == result_3683["before_total_pairs"]
    assert not result_3683["strict_total_pair_reduction"]
    assert not result_3683["construction_progress"]


def test_depth_regression_fails_even_when_pair_counts_improve():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0125-points.json"), load("alpha-0125-surface.json")
    after_points["inside_witnesses"][0]["nearest_distance_m"] = "0.000019"
    result = assess(before_points, after_points, before_surface, after_surface)
    assert not result["maximum_depth_nonincreasing"]
    assert not result["construction_progress"]


def test_squared_depth_regression_fails_even_when_maximum_does_not():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0125-points.json"), load("alpha-0125-surface.json")
    for row in after_points["inside_witnesses"]:
        row["nearest_distance_m"] = "0.000014"
    result = assess(before_points, after_points, before_surface, after_surface)
    assert result["maximum_depth_nonincreasing"]
    assert not result["squared_depth_nonincreasing"]
    assert not result["construction_progress"]


def test_missing_or_invalid_count_bank_hash_topology_completeness_or_self_fails():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0125-points.json"), load("alpha-0125-surface.json")
    mutations = [
        (before_points, before_surface, after_points, {**after_surface, "banks": after_surface["banks"][:-1]}),
        (before_points, before_surface, after_points, {**after_surface, "bank_count": 184}),
        (before_points, before_surface, after_points, {**after_surface, "complete": False}),
        (before_points, before_surface, after_points, {**after_surface, "candidate_sha256": "0" * 64}),
        (before_points, before_surface, {**after_points, "faces": after_points["faces"] + 1}, after_surface),
        ({k: v for k, v in before_points.items() if k != "inside_node_bank_pairs"}, before_surface, after_points, after_surface),
        (before_points, before_surface, {**after_points, "query_pair_count": True}, after_surface),
        (before_points, before_surface, {**after_points, "exact_PLC_surface_pairs": 1}, after_surface),
    ]
    for bp, bs, ap, ass in mutations:
        with pytest.raises(ValueError):
            assess(bp, ap, bs, ass)


def test_changed_typed_bank_identity_fails():
    before_points, before_surface = actual_before()
    after_points, after_surface = load("alpha-0125-points.json"), load("alpha-0125-surface.json")
    after_surface = copy.deepcopy(after_surface)
    after_surface["banks"][0]["stable_bone_id"] += 10000
    with pytest.raises(ValueError, match="bank|typed"):
        assess(before_points, after_points, before_surface, after_surface)


def test_zero_clearance_unchanged_is_not_reported_as_progress():
    points = {"candidate_sha256": "a" * 64, "points": 3, "faces": 1,
              "complete_surface_audit": True, "exact_PLC_surface_pairs": 0,
              "inside_node_bank_pairs": 0, "inside_nodes": 0, "query_pair_count": 555,
              "inside_witnesses": []}
    banks = [{"body": i, "stable_bone_id": i + 1, "skin_faces": 1,
              "bank_faces": 1, "skin_aabb_active": 0, "bank_aabb_active": 0,
              "result": None} for i in range(185)]
    surface = {"candidate_sha256": "a" * 64, "complete": True,
               "bank_count": 185, "expected_bank_count": 185, "banks": banks,
               "surface_point_count": 3, "surface_face_count": 1}
    result = assess(points, points, surface, surface)
    assert result["final_zero_clearance_unchanged"]
    assert not result["construction_progress"]
    assert not result["geometry_admitted"]
