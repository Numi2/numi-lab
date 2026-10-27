#!/usr/bin/env python3
"""Independent array-and-receipt check for component-merit-retention-631 v2."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-merit-retention-631/from-run006-002")
ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
SELECTOR = D.parents[0] / "select_components631_v2.py"
TESTS = D.parents[0] / "test_select_components631_v2.py"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def readj(p):
    return json.loads(Path(p).read_text())

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

sel = readj(D / "selection/report.json")
baseline_state = readj(sel["baseline_state"])
baseline = Path(baseline_state["candidate"])
target = Path(sel["target"])
output = D / "selection/candidate.npz"
base_arrays, target_arrays, output_arrays = map(load, (baseline, target, output))
assert sha(baseline) == baseline_state["candidate_sha256"] == sel["base_candidate_sha256"]
assert sha(target) == sel["target_sha256"]
assert sha(output) == sel["candidate_sha256"]
assert set(base_arrays) == set(target_arrays) == set(output_arrays)
for k in base_arrays:
    assert base_arrays[k].dtype == target_arrays[k].dtype == output_arrays[k].dtype
    assert base_arrays[k].shape == target_arrays[k].shape == output_arrays[k].shape
    if k != "points_world_m":
        assert same(base_arrays[k], output_arrays[k]), f"nonpoint payload changed: {k}"

x, y, z = base_arrays["points_world_m"], target_arrays["points_world_m"], output_arrays["points_world_m"]
target_changed = np.any(x != y, axis=1)
selected = np.asarray(sel["selected_node_rows"], dtype=np.int64)
assert len(selected) == len(np.unique(selected))
assert np.array_equal(np.flatnonzero(np.any(x != z, axis=1)), selected)
outside = np.setdiff1d(np.arange(len(x)), selected)
assert same(x[outside], z[outside])
anchors = base_arrays["typed_coarse_anchor_output_rows_018"]
assert len(anchors) == 237 and len(np.unique(anchors)) == 237
assert same(x[anchors], y[anchors]) and same(x[anchors], z[anchors])

faces = base_arrays["source_O_faces"]
edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
changed_edges = edges[target_changed[edges].all(axis=1)]
graph = coo_matrix((np.ones(len(changed_edges)), (changed_edges[:, 0], changed_edges[:, 1])), shape=(len(x), len(x))).tocsr()
_, labels = connected_components(graph, directed=False)
labels[~target_changed] = -1
owners = np.max(labels[faces], axis=1)
for tri in faces:
    labs = labels[tri]
    assert len(set(labs[labs >= 0])) <= 1, "a face spans changed components"
selected_components = set(map(int, labels[selected]))
selected_components.discard(-1)

def bank_pairs(report):
    assert report["complete"] is True and report["bank_count"] == 185 and report["expected_bank_count"] == 185
    assert report["exact_predicate_sha256"] == "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
    result = {}
    for b in report["banks"]:
        r = b.get("result") or {}
        result[(int(b["body"]), int(b["stable_bone_id"]))] = set(map(tuple, r.get("triangle_pairs", [])))
    assert len(result) == 185
    return result

baseline_report = readj(baseline_state["surface_report"])
target_check = readj(sel["target_check"])
target_report_path = Path(sel["target_check"]).parent / "audit-159/report.json"
target_report = readj(target_report_path)
final_report = readj(D / "full-check/audit-159/report.json")
bp, tp, fp = map(bank_pairs, (baseline_report, target_report, final_report))
assert sha(baseline_report_path := Path(baseline_state["surface_report"])) == baseline_state["surface_report_sha256"]
assert sha(target_report_path) == sel["target_surface_report_sha256"]
assert target_check["candidate_sha256"] == sel["target_sha256"]
assert target_check["candidate_path"] == str(target)
assert target_report["candidate_sha256"] == sel["target_sha256"]
assert final_report["candidate_sha256"] == sel["candidate_sha256"]

deltas = defaultdict(lambda: defaultdict(int))
for bank in bp:
    for pair in bp[bank]:
        deltas[int(owners[pair[0]])][bank] -= 1
    for pair in tp[bank]:
        deltas[int(owners[pair[0]])][bank] += 1
    for pair in bp[bank]:
        if owners[pair[0]] < 0:
            assert pair in tp[bank], "unchanged face lost/changed a pair"
    for pair in tp[bank]:
        if owners[pair[0]] < 0:
            assert pair in bp[bank], "unchanged face gained/changed a pair"
keep = {c for c, ds in deltas.items() if c >= 0 and sum(ds.values()) < 0 and all(v <= 0 for v in ds.values())}
assert keep == selected_components
predicted = sum(map(len, bp.values())) + sum(sum(deltas[c].values()) for c in keep)
actual = sum(map(len, fp.values()))
assert predicted == actual == sel["predicted_surface_pairs"] == 772
bank_deltas = {f"{k[0]}/{k[1]}": len(fp[k]) - len(bp[k]) for k in bp if len(fp[k]) != len(bp[k])}
assert bank_deltas == {"115/118": -6, "124/123": -4}
assert all(len(fp[k]) <= len(bp[k]) for k in bp)

check = readj(D / "full-check/report.json")
final_state = readj(D / "final/state.json")
assert check["status"] == "COMPLETE_EXACT_CONSTRUCTION_CHECKS"
assert check["candidate_sha256"] == sel["candidate_sha256"]
assert check["self_pairs"] == check["contained_pairs"] == 0 and check["complete_banks"] == 185 and check["arithmetic_pass"] is True
assert final_state["candidate_sha256"] == sel["candidate_sha256"]
assert final_state["native_admission"] is False and final_state["accepted_physical_steps"] == 0

print(json.dumps({
    "status": "PASS_INDEPENDENT_ARRAY_AND_RECEIPT_CHECK",
    "selector_sha256": sha(SELECTOR),
    "test_sha256": sha(TESTS),
    "baseline_candidate_sha256": sha(baseline),
    "target_candidate_sha256": sha(target),
    "selection_candidate_sha256": sha(output),
    "selected_rows": int(len(selected)),
    "target_changed_rows": int(np.sum(target_changed)),
    "retained_components": sorted(keep),
    "typed_anchor_rows": int(len(anchors)),
    "nonpoint_payload_bitwise_baseline": True,
    "outside_selected_coordinates_bitwise_baseline": True,
    "predicted_and_actual_pairs": [int(predicted), int(actual)],
    "per_bank_deltas": bank_deltas,
    "exact_gate": {"self_pairs": 0, "contained_pairs": 0, "banks": 185, "arithmetic_pass": True},
    "native_admission": False
}, sort_keys=True))
