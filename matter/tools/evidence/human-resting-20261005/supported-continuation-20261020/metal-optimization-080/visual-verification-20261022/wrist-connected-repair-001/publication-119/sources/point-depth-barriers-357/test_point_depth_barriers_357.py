from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).with_name("point_depth_barriers_357.py")
spec = importlib.util.spec_from_file_location("point_depth_barriers_357_tested", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _fixture(tmp_path):
    bodies = tmp_path / "bodies.csv"
    body = {
        "body_index": "0", "position_x_m": "0", "position_y_m": "0", "position_z_m": "0",
        "orientation_qx": "0", "orientation_qy": "0", "orientation_qz": "0", "orientation_qw": "1",
        "world_to_body_m00": "1", "world_to_body_m01": "0", "world_to_body_m02": "0",
        "world_to_body_m10": "0", "world_to_body_m11": "1", "world_to_body_m12": "0",
        "world_to_body_m20": "0", "world_to_body_m21": "0", "world_to_body_m22": "1",
        "com_offset_x_m": "0", "com_offset_y_m": "0", "com_offset_z_m": "0",
    }
    _write_csv(bodies, [body])
    points = tmp_path / "points.csv"
    _write_csv(points, [
        {"query_row": "0", "x_m": "0", "y_m": "0", "z_m": "0"},
        {"query_row": "1", "x_m": "2", "y_m": "0", "z_m": "0"},
    ])
    pairs = tmp_path / "pairs.csv"
    _write_csv(pairs, [
        {"query_row":"0","bank_body":"0","bank_stable_id":"7","classification":"inside","inside_rays":"3","outside_rays":"0","ambiguous":"0","nearest_bone":"7","nearest_triangle":"4","nearest_distance_m":"0.1","query_local_x":"0","query_local_y":"0","query_local_z":"0","closest_x":"0.1","closest_y":"0","closest_z":"0"},
        {"query_row":"1","bank_body":"0","bank_stable_id":"7","classification":"outside","inside_rays":"0","outside_rays":"3","ambiguous":"0","nearest_bone":"7","nearest_triangle":"5","nearest_distance_m":"0.1","query_local_x":"2","query_local_y":"0","query_local_z":"0","closest_x":"1.9","closest_y":"0","closest_z":"0"},
    ])
    nodes = tmp_path / "nodes.csv"
    _write_csv(nodes, [
        {"query_row":"0","total_bone_banks":"185","aabb_proven_outside_banks":"184","parity_outside_banks":"0","inside_banks":"1","ambiguous_banks":"0","invalid_banks":"0","parity_query_banks":"1"},
        {"query_row":"1","total_bone_banks":"185","aabb_proven_outside_banks":"185","parity_outside_banks":"0","inside_banks":"0","ambiguous_banks":"0","invalid_banks":"0","parity_query_banks":"0"},
    ])
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "argv":["binary","triangles.csv",str(bodies),str(points),str(pairs),str(nodes),"185"],
        "binary_sha256":"b"*64,"points_sha256":mod._sha(points),"pairs_sha256":mod._sha(pairs),
        "nodes_sha256":mod._sha(nodes),"query_rows":2,"physical_steps":0,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    common = {"points_csv":str(points),"pairs_csv":str(pairs),"nodes_csv":str(nodes),
              "query_receipt_json":str(receipt_path)}
    inside = {**common,"kind":"inside","expected_selected_count":1,"require_points_match":True}
    outside = {**common,"kind":"outside","selected_typed_pairs":[(1,0,7,5)],"expected_selected_count":1}
    return bodies, points, pairs, nodes, receipt_path, inside, outside


def test_inside_and_outside_rows_are_typed_and_baseline_feasible(tmp_path):
    bodies, *_rest, inside, outside = _fixture(tmp_path)
    p = np.asarray([[0,0,0],[2,0,0]], dtype=np.float32)
    rows, normals, lower, typed, records, report = mod.build_point_depth_rows(
        p, [inside, outside], bodies, expected_point_count=2
    )
    assert rows.tolist() == [0,1]
    assert np.allclose(normals, [[1,0,0],[1,0,0]])
    assert np.allclose(lower, [0,-0.1])
    assert typed.tolist() == [[0,0,7,4],[1,0,7,5]]
    assert report["inside_row_count"] == 1 and report["outside_row_count"] == 1
    assert report["maximum_zero_step_violation_m"] == 0
    assert records[0]["kind"] == "inside_nonregression"


def test_nonorthogonal_raw_transform_covector_matches_local_directional_derivative():
    M = np.asarray([[1.0,0.2,0.0],[0.0,1.0,0.1],[0.03,0.0,1.0]])
    nlocal = np.asarray([0.6,0.8,0.0])
    world_n, rhs = mod.local_halfspace_world_row(M, nlocal, -0.004)
    delta = np.asarray([0.002,-0.003,0.001])
    local_change = M @ delta
    left = float(np.dot(world_n, delta))
    right = float(np.dot(nlocal, local_change)) / np.linalg.norm(M.T @ nlocal)
    assert np.isclose(left, right, rtol=1e-14, atol=1e-15)
    assert np.isclose(rhs, -0.004 / np.linalg.norm(M.T @ nlocal), rtol=1e-14)


def test_inside_row_rebases_and_rejects_infeasible_zero_step(tmp_path):
    bodies, *_rest, inside, _outside = _fixture(tmp_path)
    inside = dict(inside)
    inside["require_points_match"] = False
    p = np.asarray([[-0.01,0,0],[2,0,0]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero-step baseline violates"):
        mod.build_point_depth_rows(p, [inside], bodies, expected_point_count=2)


def test_outside_selected_witness_rebases_without_regression(tmp_path):
    bodies, *_rest, _inside, outside = _fixture(tmp_path)
    p = np.asarray([[0,0,0],[1.95,0,0]], dtype=np.float64)
    rows, normals, lower, typed, records, report = mod.build_point_depth_rows(
        p, [outside], bodies, expected_point_count=2
    )
    assert rows.tolist() == [1]
    assert np.allclose(normals[0], [1,0,0])
    assert np.isclose(lower[0], -0.05)
    assert lower[0] <= 0
    assert np.isclose(lower[0], -0.05)


def test_query_receipt_hash_mismatch_fails_closed(tmp_path):
    bodies, points, pairs, nodes, receipt, inside, _outside = _fixture(tmp_path)
    with pairs.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="pairs_sha256 mismatch"):
        mod.build_point_depth_rows(np.asarray([[0,0,0],[2,0,0]], dtype=np.float32),
                                   [inside], bodies, expected_point_count=2)


def test_incomplete_bank_accounting_fails_closed(tmp_path):
    bodies, points, pairs, nodes, receipt, inside, _outside = _fixture(tmp_path)
    rows = list(csv.DictReader(nodes.open()))
    rows[0]["aabb_proven_outside_banks"] = "183"
    _write_csv(nodes, rows)
    data = json.loads(receipt.read_text())
    data["nodes_sha256"] = mod._sha(nodes)
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete valid bank accounting"):
        mod.build_point_depth_rows(np.asarray([[0,0,0],[2,0,0]], dtype=np.float32),
                                   [inside], bodies, expected_point_count=2)
