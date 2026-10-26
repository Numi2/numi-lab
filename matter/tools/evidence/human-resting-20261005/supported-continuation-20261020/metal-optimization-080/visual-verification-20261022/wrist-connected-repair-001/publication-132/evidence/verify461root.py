"""Read-only independent construction verification for selected 461 pass 011.

Consumes the existing exact 130/159 receipts, fit-460 report/support, and
459 preprojection certificates. It launches no solver and reruns no audits.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

R = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = R / "D"
RUN = D / "outward-plane-backtrack-461/run-001"
BACK = RUN / "backtrack-373"
PASS = BACK / "pass-011"
BASE = D / "constraint-project-backtrack-456/run-001/backtrack-373/pass-006/candidate.npz"
BASE_RUN = BASE.parent
FIT = R / "barycentric-connected-clearance-170/outward-plane-fit-460/run-002"
TRIAL = FIT / "candidate.npz"
FIT_REPORT = FIT / "report.json"
CANDIDATE = PASS / "candidate.npz"
STATE = BACK / "selected-result.json"
CONFIG = RUN / "backtrack-config.json"
INPUTS = BACK / "source-inputs.json"
PREPROJECTION = RUN / "preprojection.json"
SEED_REPORT = RUN / "seed-scale-report.json"
SEED_CACHE = RUN / "seed-rows.npz"
PREP = D / "connected-continuation-prep-457/run-001/self-seeds"
SEED_PAIRS = PREP / "merged-self-pair-seeds.npy"
GUARD_DIR = D / "fixed-bone-guard-refresh-457/run-001"
GUARD_NPZ = GUARD_DIR / "introduced-skin-bone-barriers.npz"
GUARD_REPORT = GUARD_DIR / "report.json"
GUARD_OWNER = R / "connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
SOURCE = R / "full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
OWNER401 = D / "retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
OUT = BACK / "root-independent-construction-verification.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def point_set(report):
    rows = report.get("inside_witnesses")
    require(isinstance(rows, list), "130 receipt lacks inside_witnesses")
    out = {(int(r["query_row"]), int(r["bank_body"]), int(r["bank_stable_id"]))
           for r in rows}
    require(len(out) == len(rows), "130 receipt has duplicate typed inside identities")
    return out


def surface_set(report):
    require(report.get("status") == "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
            and report.get("complete") is True, "159 receipt is incomplete")
    banks = report.get("banks")
    require(isinstance(banks, list) and len(banks) == 185
            and report.get("bank_count") == 185
            and report.get("expected_bank_count") == 185,
            "159 receipt does not cover all 185 typed banks")
    pairs, count = set(), 0
    for bank in banks:
        result = bank.get("result")
        if result is None:
            continue
        records = result.get("triangle_pairs")
        require(isinstance(records, list) and int(result.get("count", -1)) == len(records),
                "159 bank pair count disagrees with its records")
        count += len(records)
        for face, triangle in records:
            pair = (int(face), int(bank["body"]),
                    int(bank["stable_bone_id"]), int(triangle))
            require(pair not in pairs, "159 receipt has duplicate typed surface identities")
            pairs.add(pair)
    require(len(pairs) == count, "159 total pair count disagrees with typed identities")
    return pairs, count


def load_owner():
    sys.path.insert(0, str(OWNER401.parent))
    spec = importlib.util.spec_from_file_location("verify461_owner401", OWNER401)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    state = read(STATE)
    config = read(CONFIG)
    inputs = read(INPUTS)
    fit_report = read(FIT_REPORT)
    candidate_sha = sha(CANDIDATE)
    base_sha = sha(BASE)
    trial_sha = sha(TRIAL)

    require(state.get("candidate") == str(CANDIDATE)
            and state.get("candidate_sha256") == candidate_sha
            and candidate_sha == "742315227c1d8225aac9307af828251a73a98a9c40b4f59fe791aeef5da1ba0f",
            "selected 461 pass-011 candidate identity mismatch")
    require(base_sha == "1ac004ef6f6646ce4720360c50b7fd9855336a4967c3e97b199cf9bc0350049a",
            "456 baseline candidate identity mismatch")
    require(inputs.get("start") == str(BASE) and inputs.get("start_sha256") == base_sha,
            "461 start does not bind exact 456 pass-006")
    require(inputs.get("trial") == str(TRIAL) and inputs.get("trial_sha256") == trial_sha,
            "461 trial does not bind exact fit-460 candidate")
    require(fit_report.get("status") == "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED"
            and fit_report.get("native_admission") is False
            and fit_report.get("accepted_physical_steps") == 0,
            "fit-460 report status/admission mismatch")
    require(fit_report["inputs"].get("base_candidate_sha256") == base_sha
            and fit_report["inputs"].get("area_reference_sha256") == base_sha,
            "fit-460 does not bind 456 as both base and fixed 037 area reference")
    require(fit_report["inputs"].get("fixed_boundary_npz_sha256") == sha(GUARD_NPZ)
            and fit_report["inputs"].get("fixed_boundary_report_sha256") == sha(GUARD_REPORT),
            "fit-460 fixed guard inputs differ from retained refresh-457 bytes")

    base = np.load(BASE, allow_pickle=False)
    trial = np.load(TRIAL, allow_pickle=False)
    cand = np.load(CANDIDATE, allow_pickle=False)
    x = np.asarray(base["points_world_m"])
    y = np.asarray(cand["points_world_m"])
    require(x.dtype == np.float32 and y.dtype == np.float32 and x.shape == y.shape == (20889, 3),
            "base or selected output is not the pinned Float32 mesh")
    require(set(base.files) == set(cand.files), "candidate NPZ key set differs from 456 base")

    changed_keys = []
    for key in base.files:
        if key == "points_world_m":
            continue
        require(base[key].dtype == cand[key].dtype and base[key].shape == cand[key].shape
                and base[key].tobytes() == cand[key].tobytes(),
                "noncoordinate candidate array changed: " + key)
        changed_keys.append(key)
    require(len(changed_keys) == 169, "expected exactly 169 unchanged noncoordinate arrays")

    fit_support = np.asarray(config.get("fit_support_rows"), dtype=np.int64)
    fit_candidate_support = np.asarray(trial["local_witness_cluster_222_support_rows"],
                                      dtype=np.int64)
    require(np.array_equal(fit_support, fit_candidate_support),
            "461 configured support differs from fit-460 emitted support")
    require(len(fit_support) == 7127 and len(np.unique(fit_support)) == len(fit_support),
            "fit-460 support count or uniqueness mismatch")
    require(int(fit_report["support"]["support_nodes"]) == len(fit_support)
            and int(fit_report["support"]["fixed_anchor_rows_total"]) == 237,
            "fit-460 report support/anchor counts mismatch")
    anchors = np.asarray(base["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    require(len(anchors) == 237 and np.array_equal(x[anchors], y[anchors]),
            "one or more of the 237 typed anchors changed")
    changed = np.flatnonzero(np.any(x != y, axis=1))
    outside = np.setdiff1d(np.arange(len(x), dtype=np.int64), fit_support)
    require(np.isin(changed, fit_support).all() and np.array_equal(x[outside], y[outside]),
            "selected coordinates changed outside the fit-460 support")

    owner = load_owner()
    source_ref, source_receipt = owner.compose_source_reference(base, SOURCE)
    step_mm = np.linalg.norm(y.astype(np.float64) - x.astype(np.float64), axis=1) * 1000.0
    source_mm = np.linalg.norm(y.astype(np.float64) - source_ref, axis=1) * 1000.0
    area = owner.full_f32_area_gate(
        y, x.astype(np.float64) * 1000.0,
        np.asarray(base["area_reference_037_points_world_m_f64"], dtype=np.float64) * 1000.0,
        base["source_O_faces"], 0.1)
    require(area.get("pass") is True, "selected Float32 candidate failed fixed 037 area gate")
    require(float(source_mm.max(initial=0.0)) <= 12.0 + 1e-6,
            "selected candidate exceeds the 12 mm source displacement cap")

    current_point_path = PASS / "audit-130/summary.json"
    current_surface_path = PASS / "audit-159/report.json"
    base_point_path = BASE_RUN / "audit-130/summary.json"
    base_surface_path = BASE_RUN / "audit-159/report.json"
    current_point_report = read(current_point_path)
    current_surface_report = read(current_surface_path)
    base_point_report = read(base_point_path)
    base_surface_report = read(base_surface_path)
    require(sha(current_point_path) == state.get("point_summary_sha256"),
            "selected-result 130 summary hash mismatch")
    require(current_point_report.get("candidate_sha256") == candidate_sha
            and current_point_report.get("complete_surface_audit") is True
            and current_point_report.get("exact_PLC_surface_pairs") == 0,
            "selected 130 receipt does not bind candidate or has nonzero exact self pairs")
    require(current_surface_report.get("candidate_sha256") == candidate_sha,
            "selected 159 receipt does not bind candidate")
    require(base_point_report.get("candidate_sha256") == base_sha
            and base_surface_report.get("candidate_sha256") == base_sha,
            "456 baseline exact reports do not bind 456 candidate")
    current_points, base_points = point_set(current_point_report), point_set(base_point_report)
    current_surfaces, current_surface_count = surface_set(current_surface_report)
    base_surfaces, base_surface_count = surface_set(base_surface_report)
    point_added, point_removed = current_points - base_points, base_points - current_points
    surface_added = current_surfaces - base_surfaces
    surface_removed = base_surfaces - current_surfaces
    result = read(PASS / "result.json")
    pm, jm, dm = result["point_merit"], result["joint_merit"], result["depth_merit"]
    require((len(base_points), len(current_points)) ==
            (pm["before_inside_pairs"], pm["after_inside_pairs"]) == (321, 316),
            "461 point merit counts differ from retained exact 130 sets")
    require((base_surface_count, current_surface_count) ==
            (jm["before_surface_pairs"], jm["after_surface_pairs"]) == (3136, 3117),
            "461 surface merit counts differ from retained exact 159 sets")
    require(int(jm["typed_bank_count"]) == 185
            and (len(base_points) + base_surface_count,
                 len(current_points) + current_surface_count) ==
            (jm["before_total_pairs"], jm["after_total_pairs"]) == (3457, 3433),
            "461 joint merit totals differ from exact pair sets")
    require(not point_added and len(point_removed) == 5,
            "461 exact point identity delta differs from selected merit")
    require(len(surface_added) == 34 and len(surface_removed) == 53
            and dm.get("introduced_surface_pair_count") == len(surface_added),
            "461 exact surface identity delta differs from selected depth merit")

    projection = read(PREPROJECTION)
    seed_lp = projection.get("seed_cut_lp", {})
    fixed = seed_lp.get("fixed_guard_certificate", {})
    seed_scale = read(SEED_REPORT)
    seed_cache = np.load(SEED_CACHE, allow_pickle=False)
    seed_pairs = np.load(SEED_PAIRS, allow_pickle=False)
    seed_preflight = PREP / "report.json"
    require(seed_scale.get("status") ==
            "GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447"
            and sha(SEED_CACHE) == seed_scale.get("seed_row_cache_sha256")
            and sha(SEED_PAIRS) == seed_scale.get("seed_pair_sha256")
            and sha(seed_preflight) == config.get("self_pair_seed_report_sha256"),
            "459 seed cache or full-seed preflight pins mismatch")
    require(int(seed_scale.get("seed_pair_count", -1)) == len(seed_pairs) == 24712
            and int(seed_scale.get("seed_linear_row_count", -1)) == 139200
            and len(seed_cache["lower"]) == 139200
            and len(seed_cache["indptr"]) == 139201,
            "459 full seed cache row count mismatch")
    require(int(seed_lp.get("seed_rows_total", -1)) == 139200
            and int(seed_lp.get("seed_rows_checked", -1)) == 139200
            and int(seed_lp.get("seed_violated_rows", -1)) == 0
            and float(seed_lp.get("seed_max_violation_mm", float("inf"))) <= 1e-12,
            "459 full self-seed certificate failed")
    require(set(map(int, seed_lp.get("anchor_rows_forced_zero", []))) == set(map(int, anchors)),
            "459 preprojection zero-scale anchors differ from the 237 typed base anchors")
    guard_receipt = fixed.get("input_receipt", {})
    guard_metadata = read(GUARD_REPORT)
    guard_arrays = np.load(GUARD_NPZ, allow_pickle=False)
    require(fixed.get("passed") is True
            and int(fixed.get("row_count", -1)) == 13632
            and int(fixed.get("checked_rows", -1)) == 13632
            and int(fixed.get("violated_rows", -1)) == 0
            and float(fixed.get("max_violation_mm", float("inf"))) <= float(fixed["tolerance_mm"])
            and float(fixed.get("anchor_max_abs_scale", float("inf"))) == 0.0,
            "459 fixed-bone guard certificate failed")
    require(guard_receipt.get("base_candidate_sha256") == base_sha
            and guard_receipt.get("npz_sha256") == sha(GUARD_NPZ)
            and guard_receipt.get("report_sha256") == sha(GUARD_REPORT)
            and guard_metadata.get("base_candidate_sha256") == base_sha
            and guard_metadata.get("status") == "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT"
            and len(guard_arrays["skin_face_rows"]) * 6 == 13632
            and sha(GUARD_OWNER) == guard_receipt.get("313_owner_sha256"),
            "459 fixed-bone guard certificate input pins/count mismatch")

    paths = [STATE, INPUTS, CONFIG, PREPROJECTION, SEED_REPORT, SEED_CACHE,
             SEED_PAIRS, seed_preflight, GUARD_NPZ, GUARD_REPORT, BASE, TRIAL,
             FIT_REPORT, CANDIDATE, current_point_path, current_surface_path,
             base_point_path, base_surface_path, SOURCE, OWNER401, GUARD_OWNER, Path(__file__)]
    pins = {str(p): sha(p) for p in paths}
    receipt = {
        "status": "ROOT_INDEPENDENT_CONSTRUCTION_PASS",
        "classification": "construction-only independent verification; no physical/native admission",
        "candidate_path": str(CANDIDATE),
        "candidate_sha256": candidate_sha,
        "base456_path": str(BASE),
        "base456_sha256": base_sha,
        "fit460_candidate_sha256": trial_sha,
        "fit460_report_sha256": sha(FIT_REPORT),
        "fit460_support_rows": int(len(fit_support)),
        "fit460_support_sha256": array_sha(np.asarray(fit_support, dtype="<i8")),
        "outside_fit460_support_unchanged": True,
        "unchanged_noncoordinate_array_count": len(changed_keys),
        "anchor_count": int(len(anchors)),
        "anchors_bitwise_unchanged": True,
        "changed_node_count": int(len(changed)),
        "max_step_mm": float(step_mm.max(initial=0.0)),
        "max_source_displacement_mm": float(source_mm.max(initial=0.0)),
        "source_cap_mm": 12.0,
        "source_reference_receipt": source_receipt,
        "fixed_037_float32_area_gate": area,
        "exact_130": {
            "base_report_sha256": sha(base_point_path),
            "candidate_report_sha256": sha(current_point_path),
            "base_inside_pairs": len(base_points),
            "candidate_inside_pairs": len(current_points),
            "resolved_pairs": len(point_removed),
            "introduced_pairs": len(point_added),
            "candidate_exact_surface_pairs": int(current_point_report["exact_PLC_surface_pairs"]),
        },
        "exact_159": {
            "base_report_sha256": sha(base_surface_path),
            "candidate_report_sha256": sha(current_surface_path),
            "typed_banks": 185,
            "base_pairs": base_surface_count,
            "candidate_pairs": current_surface_count,
            "resolved_pairs": len(surface_removed),
            "introduced_pairs": len(surface_added),
        },
        "joint_pair_counts": {
            "base": len(base_points) + base_surface_count,
            "candidate": len(current_points) + current_surface_count,
            "candidate_reduced": (len(current_points) + current_surface_count
                                  < len(base_points) + base_surface_count),
        },
        "full_seed_certificate": {
            "seed_pair_count": int(seed_scale["seed_pair_count"]),
            "linear_rows": int(seed_scale["seed_linear_row_count"]),
            "rows_checked": int(seed_lp["seed_rows_checked"]),
            "violated_rows": int(seed_lp["seed_violated_rows"]),
            "max_violation_mm": float(seed_lp["seed_max_violation_mm"]),
            "seed_cache_sha256": sha(SEED_CACHE),
            "seed_pairs_sha256": sha(SEED_PAIRS),
            "preflight_report_sha256": sha(seed_preflight),
        },
        "fixed_guard_certificate": {
            "passed": bool(fixed["passed"]),
            "linear_rows": int(fixed["row_count"]),
            "checked_rows": int(fixed["checked_rows"]),
            "violated_rows": int(fixed["violated_rows"]),
            "max_violation_mm": float(fixed["max_violation_mm"]),
            "tolerance_mm": float(fixed["tolerance_mm"]),
            "anchor_max_abs_scale": float(fixed["anchor_max_abs_scale"]),
            "typed_fixed_pairs": int(guard_receipt["typed_pair_count"]),
            "guard_npz_sha256": sha(GUARD_NPZ),
            "guard_report_sha256": sha(GUARD_REPORT),
            "313_owner_sha256": guard_receipt["313_owner_sha256"],
        },
        "reported_depth_metrics": {
            "before_max_depth_m": pm["before_maximum_depth_m"],
            "after_max_depth_m": pm["after_maximum_depth_m"],
            "before_squared_depth_m2": pm["before_squared_depth_m2"],
            "after_squared_depth_m2": pm["after_squared_depth_m2"],
            "depth_merit_classification": dm.get("classification"),
        },
        "zero_clearance_achieved": False,
        "geometry_admitted": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "solver_or_geometry_audit_rerun": False,
        "verification_script": str(Path(__file__)),
        "verification_script_sha256": sha(Path(__file__)),
        "input_pins": pins,
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "candidate_sha256": candidate_sha,
        "support_rows": len(fit_support),
        "unchanged_noncoordinate_arrays": len(changed_keys),
        "anchors": len(anchors),
        "outside_support_unchanged": True,
        "area_gate_pass": area.get("pass"),
        "source_max_mm": float(source_mm.max(initial=0.0)),
        "point_pairs": [len(base_points), len(current_points)],
        "surface_pairs": [base_surface_count, current_surface_count],
        "seed_rows_checked": int(seed_lp["seed_rows_checked"]),
        "fixed_guard_rows_checked": int(fixed["checked_rows"]),
        "receipt_path": str(OUT),
        "receipt_sha256": sha(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
