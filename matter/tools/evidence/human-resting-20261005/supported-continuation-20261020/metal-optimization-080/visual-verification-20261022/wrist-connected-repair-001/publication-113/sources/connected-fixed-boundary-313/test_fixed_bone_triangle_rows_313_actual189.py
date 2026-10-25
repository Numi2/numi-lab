from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
MODULE_PATH = ROOT / "connected-fixed-boundary-313" / "fixed_bone_triangle_rows_313.py"
PAIR_OWNER_PATH = ROOT / "barycentric-connected-clearance-170" / "connected-local-continuation-189-run001" / "round-005" / "proposal" / "round-001" / "fit" / "self_separation_026.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load(MODULE_PATH, "fixed_bone_rows_313_test")
pair_owner = load(PAIR_OWNER_PATH, "self_separation_026_313_test")


def scene():
    points = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    bone = np.array([
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
    ])
    record = {
        "skin_face_row": 0,
        "bank_body": 7,
        "stable_bone_id": 21,
        "bank_triangle_stable_id": 104,
        "fixed_triangle_world_mm": bone,
    }
    return points, faces, bone, record


def build(points, faces, record, owner=pair_owner):
    return helper.build_fixed_triangle_rows(points, faces, [record], owner)


def deficits(rows, displacement):
    return [float(row["lower"]) - float(np.sum(row["coefficients"] * displacement[row["nodes"]]))
            for row in rows]


def test_fixed_bone_rows_keep_skin_schema_and_typed_provenance():
    points, faces, _, record = scene()
    rows, typed = build(points, faces, record)
    assert len(rows) == 6
    assert len(typed) == 1
    provenance = typed[0]
    assert {k: provenance[k] for k in (
        "skin_face_row", "bank_body", "stable_bone_id", "bank_triangle_stable_id"
    )} == {"skin_face_row": 0, "bank_body": 7, "stable_bone_id": 21, "bank_triangle_stable_id": 104}
    assert provenance["kind"] == "fixed_bone_triangle"
    assert provenance["rows_start"] == 0 and provenance["rows_stop"] == 6
    for row in rows:
        assert set(row) == {"nodes", "coefficients", "lower", "label"}
        assert row["nodes"].ndim == 1
        assert row["coefficients"].shape == (len(row["nodes"]), 3)
        assert set(row["nodes"].tolist()) <= {0, 1, 2}
        assert row["label"].startswith("fixedbone:0:7:21:104:")
        assert np.isfinite(row["coefficients"]).all()
        assert np.isfinite(row["lower"])


def test_zero_increment_is_feasible_and_inward_skin_motion_violates():
    points, faces, _, record = scene()
    rows, _ = build(points, faces, record)
    zero = np.zeros_like(points)
    assert max(deficits(rows, zero)) <= 2e-8
    inward = np.zeros_like(points)
    inward[:, 2] = 3.0
    assert max(deficits(rows, inward)) > 0.1


def test_outward_and_tangential_skin_motion_are_safe():
    points, faces, _, record = scene()
    rows, _ = build(points, faces, record)
    outward = np.zeros_like(points)
    outward[:, 2] = -1.0
    tangent = np.zeros_like(points)
    tangent[:, 0] = 0.2
    assert max(deficits(rows, outward)) <= 2e-8
    assert max(deficits(rows, tangent)) <= 2e-8


def test_joint_translation_leaves_rows_unchanged():
    points, faces, bone, record = scene()
    rows_a, typed_a = build(points, faces, record)
    shift = np.array([128.0, -64.0, 32.0])
    moved_record = dict(record, fixed_triangle_world_mm=bone + shift)
    rows_b, typed_b = build(points + shift, faces, moved_record)
    assert len(rows_a) == len(rows_b)
    for left, right in zip(rows_a, rows_b):
        np.testing.assert_array_equal(left["nodes"], right["nodes"])
        np.testing.assert_allclose(left["coefficients"], right["coefficients"], atol=0.0, rtol=0.0)
        assert abs(left["lower"] - right["lower"]) < 1e-12
    assert typed_a[0]["skin_face_row"] == typed_b[0]["skin_face_row"]


def test_nonseparable_base_pair_fails_closed():
    points, faces, bone, record = scene()
    crossing = bone.copy()
    crossing[:, 2] = 0.0
    with pytest.raises(ValueError, match="rejected by 026 owner"):
        build(points, faces, dict(record, fixed_triangle_world_mm=crossing))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"skin_face_row": 4}, "out of range"),
        ({"skin_face_row": -1}, "nonnegative"),
        ({"bank_body": True}, "integer ID"),
        ({"stable_bone_id": 2.5}, "integer ID"),
        ({"fixed_triangle_world_mm": np.full((3, 3), np.nan)}, "finite with shape"),
    ],
)
def test_invalid_typed_records_fail_closed(changes, message):
    points, faces, _, record = scene()
    with pytest.raises(ValueError, match=message):
        build(points, faces, dict(record, **changes))


def test_duplicate_typed_pair_rejected_deterministically():
    points, faces, _, record = scene()
    errors = []
    for source in ([record, dict(record)], [dict(record), record]):
        with pytest.raises(ValueError, match="duplicate typed skin-bone pair") as exc:
            helper.build_fixed_triangle_rows(points, faces, source, pair_owner)
        errors.append(str(exc.value))
    assert errors[0] == errors[1]


def test_zero_skin_coefficient_rows_omit_only_feasible_constant_rows():
    points, faces, _, record = scene()

    class FixedOnlyOwner:
        @staticmethod
        def build_pair_rows(*args, **kwargs):
            return ([{
                "nodes": np.array([3, 4, 5], dtype=np.int64),
                "coefficients": np.eye(3),
                "lower": 0.0,
                "label": "constant",
            }], [{"kind": "stub"}])

    rows, typed = build(points, faces, record, FixedOnlyOwner())
    assert rows == []
    assert typed[0]["rows_start"] == typed[0]["rows_stop"] == 0
    assert typed[0]["omitted_fixed_only_rows"] == 1

    class InfeasibleFixedOnlyOwner:
        @staticmethod
        def build_pair_rows(*args, **kwargs):
            return ([{
                "nodes": np.array([3, 4, 5], dtype=np.int64),
                "coefficients": np.eye(3),
                "lower": 1e-9,
                "label": "constant-infeasible",
            }], [{"kind": "stub"}])

    with pytest.raises(ValueError, match="fixed-only row infeasible"):
        build(points, faces, record, InfeasibleFixedOnlyOwner())


def test_records_and_rows_are_deterministic_under_record_order():
    points, faces, bone, first = scene()
    second = dict(first, bank_body=8, stable_bone_id=22, bank_triangle_stable_id=9)
    second["fixed_triangle_world_mm"] = bone + np.array([0.0, 0.0, 4.0])
    forward = helper.build_fixed_triangle_rows(points, faces, [second, first], pair_owner)
    reverse = helper.build_fixed_triangle_rows(points, faces, [first, second], pair_owner)
    rows_a, typed_a = forward
    rows_b, typed_b = reverse
    assert [x["bank_body"] for x in typed_a] == [7, 8]
    assert [x["bank_body"] for x in typed_a] == [x["bank_body"] for x in typed_b]
    assert len(rows_a) == len(rows_b)
    for left, right in zip(rows_a, rows_b):
        np.testing.assert_array_equal(left["nodes"], right["nodes"])
        np.testing.assert_array_equal(left["coefficients"], right["coefficients"])
        assert left["lower"] == right["lower"]
        assert left["label"] == right["label"]
