#!/usr/bin/env python3
"""Independent construction receipt checker for existing Human candidate artifacts.

This checks saved exact audit identities, selected fit/backtrack reconstruction,
F32 area/source bounds, full self-seed alpha rows, and fixed-bone guard rows.
It does not run any geometry audit or solver.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def point_set(report):
    rows = report.get("inside_witnesses")
    require(isinstance(rows, list), "130 report has no inside_witnesses")
    result = {(int(r["query_row"]), int(r["bank_body"]), int(r["bank_stable_id"]))
              for r in rows}
    require(len(result) == len(rows), "130 report has duplicate typed inside identities")
    return result


def surface_set(report):
    require(report.get("status") == "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
            and report.get("complete") is True, "159 report is incomplete")
    banks = report.get("banks")
    require(isinstance(banks, list)
            and len(banks) == report.get("bank_count") == report.get("expected_bank_count") == 185,
            "159 report does not cover all 185 typed banks")
    result = set()
    count = 0
    for bank in banks:
        rows = (bank.get("result") or {}).get("triangle_pairs")
        if rows is None:
            continue
        require(int(bank["result"].get("count", -1)) == len(rows),
                "159 bank count disagrees with typed rows")
        count += len(rows)
        for face, tri in rows:
            pair = (int(face), int(bank["body"]), int(bank["stable_bone_id"]), int(tri))
            require(pair not in result, "159 has duplicate typed surface identity")
            result.add(pair)
    require(len(result) == count, "159 total disagrees with typed identities")
    return result, count


def load_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("verify493_area_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, type=Path, help="selected pass candidate.npz")
    ap.add_argument("--base", required=True, type=Path, help="pinned base candidate.npz")
    ap.add_argument("--fit-candidate", required=True, type=Path, help="fit proposal candidate.npz")
    ap.add_argument("--fit-report", required=True, type=Path, help="fit proposal report.json")
    ap.add_argument("--source-owner", required=True, type=Path, help="owner providing compose_source_reference/full_f32_area_gate")
    ap.add_argument("--source-reference", required=True, type=Path, help="source-reference endpoint candidate.npz")
    ap.add_argument("--prep-dir", type=Path, help="prep run directory; inferred from fit face_target_npz when omitted")
    ap.add_argument("--guard-npz", type=Path, help="fixed-bone guard NPZ; inferred from fit inputs when omitted")
    ap.add_argument("--guard-report", type=Path, help="fixed-bone guard report; inferred from fit inputs when omitted")
    ap.add_argument("--depth-npz", type=Path, help="point-depth rows; inferred from fit inputs when omitted")
    ap.add_argument("--depth-report", type=Path, help="365-compatible depth report; inferred from fit inputs when omitted")
    ap.add_argument("--seed-pairs", type=Path, help="full self-pair seed NPY; inferred from backtrack config when omitted")
    ap.add_argument("--output", type=Path, help="receipt path; defaults beside backtrack selected-result.json")
    args = ap.parse_args()

    candidate, base_path = args.candidate.resolve(), args.base.resolve()
    fit_candidate, fit_report_path = args.fit_candidate.resolve(), args.fit_report.resolve()
    pass_dir = candidate.parent
    back_dir, run_dir = pass_dir.parent, pass_dir.parent.parent
    state_path, result_path = back_dir / "selected-result.json", pass_dir / "result.json"
    config_path, launch_path = run_dir / "backtrack-config.json", run_dir / "launch.json"
    input_path = back_dir / "source-inputs.json"
    scales_path = pass_dir / "node-step-scales.npy"
    preprojection_path, seed_report_path, seed_cache_path = (
        run_dir / "preprojection.json", run_dir / "seed-scale-report.json", run_dir / "seed-rows.npz")
    prep_seed_report_fallback = None

    state, result, config, launch, source_inputs = map(
        read, (state_path, result_path, config_path, launch_path, input_path))
    fit_report = read(fit_report_path)
    fit_inputs = fit_report.get("inputs", {})
    prep_dir = (args.prep_dir.resolve() if args.prep_dir else
                Path(fit_inputs["face_target_npz"]).resolve().parent.parent)
    prep_report_path = prep_dir / "preparation-report.json"
    seed_preflight_path = prep_dir / "self-seeds/report.json"
    seed_pairs_path = (args.seed_pairs.resolve() if args.seed_pairs else
                       Path(config["self_pair_seed_path"]).resolve())
    guard_npz = (args.guard_npz.resolve() if args.guard_npz else
                 Path(fit_inputs["fixed_boundary_npz"]).resolve())
    guard_report_path = (args.guard_report.resolve() if args.guard_report else
                         Path(fit_inputs["fixed_boundary_report"]).resolve())
    depth_npz = (args.depth_npz.resolve() if args.depth_npz else
                 Path(fit_inputs["point_depth_rows_npz"]).resolve())
    depth_report_path = (args.depth_report.resolve() if args.depth_report else
                         Path(fit_inputs["point_depth_rows_report"]).resolve())
    guard_report = read(guard_report_path)
    guard_receipt = fit_report.get("fixed_boundary_receipt", {})
    owner313_path = Path(guard_receipt.get("owner_path") or
                         (guard_report.get("construction_scope") or {}).get("313_owner_path", ""))
    output = args.output.resolve() if args.output else back_dir / "root-independent-construction-verification-493.json"

    candidate_sha, base_sha, fit_sha = sha(candidate), sha(base_path), sha(fit_candidate)
    fit_report_sha = sha(fit_report_path)
    require(state.get("candidate_sha256") == candidate_sha
            and Path(state.get("candidate", "")).resolve() == candidate,
            "selected-result candidate path/hash mismatch")
    require(source_inputs.get("start_sha256") == base_sha
            and Path(source_inputs.get("start", "")).resolve() == base_path
            and source_inputs.get("trial_sha256") == fit_sha
            and Path(source_inputs.get("trial", "")).resolve() == fit_candidate,
            "backtrack source-inputs do not bind supplied base and fit")
    require(config.get("fit_candidate_sha256") == fit_sha
            and config.get("fit_report_sha256") == fit_report_sha
            and Path(config.get("trial", "")).resolve() == fit_candidate
            and Path(config.get("parent_fit", "")).resolve() == fit_report_path
            and Path((config.get("state") or {}).get("candidate", "")).resolve() == base_path,
            "backtrack config does not bind supplied base and fit")
    require(launch.get("base_sha256") == base_sha and launch.get("fit_sha256") == fit_sha
            and launch.get("native_admission") is False
            and launch.get("accepted_physical_steps") == 0,
            "backtrack launch pin/admission mismatch")
    require(fit_report.get("status") == "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED"
            and fit_report.get("native_admission") is False
            and fit_report.get("accepted_physical_steps") == 0,
            "fit report is not an unadmitted construction proposal")

    base, trial, cand = (np.load(p, allow_pickle=False) for p in (base_path, fit_candidate, candidate))
    x, z, y = (np.asarray(a["points_world_m"]) for a in (base, trial, cand))
    require(x.dtype == z.dtype == y.dtype == np.float32 and x.shape == z.shape == y.shape
            and x.ndim == 2 and x.shape[1] == 3, "input point arrays are not matching F32 meshes")
    require(set(base.files) == set(cand.files), "selected candidate NPZ key set differs from base")
    noncoordinate_keys = []
    for key in base.files:
        if key == "points_world_m":
            continue
        require(base[key].dtype == cand[key].dtype and base[key].shape == cand[key].shape
                and np.ascontiguousarray(base[key]).tobytes() == np.ascontiguousarray(cand[key]).tobytes(),
                "noncoordinate array changed: " + key)
        noncoordinate_keys.append(key)

    support = np.asarray(config.get("fit_support_rows"), dtype=np.int64)
    trial_support = np.asarray(trial["local_witness_cluster_222_support_rows"], dtype=np.int64)
    require(np.array_equal(support, trial_support) and len(np.unique(support)) == len(support),
            "fit support rows differ from pinned fit candidate")
    anchors = np.asarray(base["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    require(np.array_equal(x[anchors], y[anchors]), "typed anchors changed")
    changed = np.flatnonzero(np.any(x != y, axis=1))
    outside = np.setdiff1d(np.arange(len(x), dtype=np.int64), support)
    require(np.isin(changed, support).all() and np.array_equal(x[outside], y[outside]),
            "selected points changed outside fit support")
    scales = np.load(scales_path, allow_pickle=False)
    rebuilt = (x.astype(np.float64) + scales[:, None] *
               (z.astype(np.float64) - x.astype(np.float64))).astype(np.float32)
    require(scales.shape == (len(x),) and np.array_equal(rebuilt, y),
            "selected candidate does not reconstruct from saved per-node scales")
    require(array_sha(y) == result.get("emitted_points_sha256"),
            "emitted point bytes mismatch pass result")
    require(fit_report.get("candidate_sha256") == fit_sha
            and fit_report.get("inputs", {}).get("base_candidate_sha256") == base_sha
            and fit_report.get("inputs", {}).get("area_reference_sha256") == base_sha,
            "fit candidate/report or base/area pins mismatch")

    for key, path in [("fixed_boundary_npz_sha256", guard_npz),
                      ("fixed_boundary_report_sha256", guard_report_path),
                      ("point_depth_rows_npz_sha256", depth_npz),
                      ("point_depth_rows_report_sha256", depth_report_path),
                      ("self_pairs_sha256", seed_pairs_path)]:
        require(fit_inputs.get(key) == sha(path), "fit input hash mismatch: " + key)
    for artifact_key, report_key in [
        ("face_target_npz", "face_target_npz_sha256"),
        ("face_target_report", "face_target_report_sha256"),
        ("point_target_npz", "point_target_npz_sha256"),
        ("point_target_report", "point_target_report_sha256")]:
        path = Path(fit_inputs[artifact_key]).resolve()
        require(fit_inputs.get(report_key) == sha(path), "fit target hash mismatch: " + artifact_key)
    prep_report, seed_preflight = read(prep_report_path), read(seed_preflight_path)
    require(prep_report.get("candidate_sha256") == base_sha
            and prep_report.get("status") == "PASS_CONSTRUCTION_INPUTS_ONLY"
            and seed_preflight.get("status") == "PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY"
            and seed_preflight.get("output_pairs_sha256") == sha(seed_pairs_path),
            "prep inputs do not bind to supplied base/full self seeds")
    require(config.get("self_pair_seed_count") == len(np.load(seed_pairs_path, allow_pickle=False))
            and config.get("self_pair_seed_sha256") == sha(seed_pairs_path)
            and config.get("self_pair_seed_report_sha256") == sha(seed_preflight_path),
            "backtrack full-seed pins mismatch")

    owner = load_module(args.source_owner.resolve())
    source_reference = args.source_reference.resolve()
    source_ref, source_receipt = owner.compose_source_reference(base, source_reference)
    step_mm = np.linalg.norm(y.astype(np.float64) - x.astype(np.float64), axis=1) * 1000.0
    source_mm = np.linalg.norm(y.astype(np.float64) - source_ref, axis=1) * 1000.0
    area = owner.full_f32_area_gate(
        y, x.astype(np.float64) * 1000.0,
        np.asarray(base["area_reference_037_points_world_m_f64"], dtype=np.float64) * 1000.0,
        base["source_O_faces"], 0.1)
    require(area.get("pass") is True, "selected emitted-F32 candidate failed fixed-037 area gate")
    source_cap = float(fit_report.get("source_cap_mm", 12.0))
    require(float(source_mm.max(initial=0.0)) <= source_cap + 1e-6,
            "selected candidate exceeds source displacement cap")

    base_pass = base_path.parent
    base_point_path, base_surface_path = base_pass/"audit-130/summary.json", base_pass/"audit-159/report.json"
    point_path, surface_path = pass_dir/"audit-130/summary.json", pass_dir/"audit-159/report.json"
    base_point, base_surface = read(base_point_path), read(base_surface_path)
    point, surface = read(point_path), read(surface_path)
    require(point.get("candidate_sha256") == candidate_sha
            and point.get("complete_surface_audit") is True
            and int(point.get("exact_PLC_surface_pairs", -1)) == 0,
            "selected 130 exact self report mismatch/incomplete/nonzero")
    require(base_point.get("candidate_sha256") == base_sha
            and base_point.get("complete_surface_audit") is True
            and int(base_point.get("exact_PLC_surface_pairs", -1)) == 0,
            "base 130 report mismatch/incomplete/nonzero")
    require(surface.get("candidate_sha256") == candidate_sha
            and base_surface.get("candidate_sha256") == base_sha,
            "159 report candidate pins mismatch")
    base_points, current_points = point_set(base_point), point_set(point)
    base_surfaces, base_surface_count = surface_set(base_surface)
    current_surfaces, current_surface_count = surface_set(surface)
    point_added, point_removed = current_points-base_points, base_points-current_points
    surface_added, surface_removed = current_surfaces-base_surfaces, base_surfaces-current_surfaces
    pm, jm, dm = result["point_merit"], result["joint_merit"], result["depth_merit"]
    expected_point_added = {tuple(map(int, row)) for row in pm.get("introduced_typed_pairs", [])}
    expected_point_removed = {tuple(map(int, row)) for row in pm.get("removed_typed_pairs", [])}
    require((len(base_points), len(current_points)) ==
            (pm["before_inside_pairs"], pm["after_inside_pairs"])
            and point_added == expected_point_added and point_removed == expected_point_removed,
            "point merit counts/identity deltas do not match exact 130 reports")
    require((base_surface_count, current_surface_count) ==
            (jm["before_surface_pairs"], jm["after_surface_pairs"])
            and len(surface_added) == int(dm.get("introduced_surface_pair_count", -1))
            and (len(base_points)+base_surface_count, len(current_points)+current_surface_count) ==
            (jm["before_total_pairs"], jm["after_total_pairs"])
            and jm.get("typed_bank_count") == 185,
            "surface/joint merit counts do not match exact 159 reports")

    seed_scale, projection = read(seed_report_path), read(preprojection_path)
    seed_cache = np.load(seed_cache_path, allow_pickle=False)
    seed_pairs = np.load(seed_pairs_path, allow_pickle=False)
    seed_lp = projection.get("seed_cut_lp", {})
    fixed = seed_lp.get("fixed_guard_certificate", {})
    require(seed_scale.get("status") == "GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447"
            and int(seed_scale.get("seed_pair_count", -1)) == len(seed_pairs)
            and int(seed_scale.get("seed_linear_row_count", -1)) == len(seed_cache["lower"])
            and sha(seed_cache_path) == seed_scale.get("seed_row_cache_sha256")
            and sha(seed_pairs_path) == seed_scale.get("seed_pair_sha256")
            and int(seed_lp.get("seed_rows_total", -1)) == len(seed_cache["lower"])
            and int(seed_lp.get("seed_rows_checked", -1)) == len(seed_cache["lower"])
            and int(seed_lp.get("seed_violated_rows", -1)) == 0
            and float(seed_lp.get("seed_max_violation_mm", float("inf"))) <= 1e-12,
            "full self-seed rows/certificate mismatch")
    require(set(map(int, seed_lp.get("anchor_rows_forced_zero", []))) == set(map(int, anchors)),
            "full-seed certificate did not zero every typed anchor")
    guard_arrays = np.load(guard_npz, allow_pickle=False)
    require(fixed.get("passed") is True
            and int(fixed.get("row_count", -1)) == int(fixed.get("checked_rows", -2)) == 13950
            and int(fixed.get("violated_rows", -1)) == 0
            and float(fixed.get("max_violation_mm", float("inf"))) <= float(fixed["tolerance_mm"])
            and float(fixed.get("anchor_max_abs_scale", float("inf"))) == 0.0,
            "fixed-bone alpha guard certificate failed")
    guard_input = fixed.get("input_receipt", {})
    require(guard_input.get("base_candidate_sha256") == base_sha
            and guard_input.get("npz_sha256") == sha(guard_npz)
            and guard_input.get("report_sha256") == sha(guard_report_path)
            and guard_report.get("base_candidate_sha256") == base_sha
            and guard_report.get("status") == "COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT"
            and int(guard_report["construction_scope"]["retained_guard_pairs"]) == 2325
            and len(guard_arrays["skin_face_rows"]) * 6 == 13950
            and owner313_path.exists()
            and sha(owner313_path) == guard_input.get("313_owner_sha256"),
            "fixed-bone guard source/artifact pins mismatch")

    receipt_paths = [candidate, base_path, fit_candidate, fit_report_path, state_path, result_path,
        config_path, launch_path, input_path, scales_path, preprojection_path, seed_report_path,
        seed_cache_path, seed_pairs_path, prep_report_path, seed_preflight_path, guard_npz,
        guard_report_path, depth_npz, depth_report_path, source_reference, args.source_owner.resolve(),
        owner313_path, point_path, surface_path, base_point_path, base_surface_path]
    pins = {str(p): sha(p) for p in receipt_paths}
    receipt = {
        "status": "ROOT_INDEPENDENT_CONSTRUCTION_PASS",
        "classification": "construction-only independent verification; no physical/native admission",
        "candidate_path": str(candidate), "candidate_sha256": candidate_sha,
        "base_path": str(base_path), "base_sha256": base_sha,
        "fit_candidate_path": str(fit_candidate), "fit_candidate_sha256": fit_sha,
        "fit_report_path": str(fit_report_path), "fit_report_sha256": fit_report_sha,
        "fit_support_rows": int(len(support)), "fit_support_sha256": array_sha(np.asarray(support, dtype="<i8")),
        "outside_fit_support_unchanged": True,
        "unchanged_noncoordinate_array_count": len(noncoordinate_keys),
        "anchor_count": int(len(anchors)), "anchors_bitwise_unchanged": True,
        "changed_node_count": int(len(changed)),
        "max_step_mm": float(step_mm.max(initial=0.0)),
        "max_source_displacement_mm": float(source_mm.max(initial=0.0)),
        "source_cap_mm": source_cap, "source_reference_receipt": source_receipt,
        "fixed_037_float32_area_gate": area,
        "exact_130": {"base_report_sha256": sha(base_point_path), "candidate_report_sha256": sha(point_path),
                      "base_inside_pairs": len(base_points), "candidate_inside_pairs": len(current_points),
                      "resolved_pairs": len(point_removed), "introduced_pairs": len(point_added),
                      "candidate_exact_self_pairs": int(point["exact_PLC_surface_pairs"])},
        "exact_159": {"base_report_sha256": sha(base_surface_path), "candidate_report_sha256": sha(surface_path),
                      "typed_banks": 185, "base_pairs": base_surface_count, "candidate_pairs": current_surface_count,
                      "resolved_pairs": len(surface_removed), "introduced_pairs": len(surface_added)},
        "joint_pair_counts": {"base": len(base_points)+base_surface_count,
                              "candidate": len(current_points)+current_surface_count},
        "full_seed_certificate": {"seed_pairs": int(len(seed_pairs)), "linear_rows": int(len(seed_cache["lower"])),
                                  "rows_checked": int(seed_lp["seed_rows_checked"]),
                                  "violated_rows": int(seed_lp["seed_violated_rows"]),
                                  "max_violation_mm": float(seed_lp["seed_max_violation_mm"]),
                                  "seed_cache_sha256": sha(seed_cache_path), "seed_pairs_sha256": sha(seed_pairs_path)},
        "fixed_guard_certificate": {"typed_pairs": int(guard_input["typed_pair_count"]),
                                    "linear_rows": int(fixed["row_count"]),
                                    "checked_rows": int(fixed["checked_rows"]),
                                    "violated_rows": int(fixed["violated_rows"]),
                                    "max_violation_mm": float(fixed["max_violation_mm"]),
                                    "tolerance_mm": float(fixed["tolerance_mm"]),
                                    "anchor_max_abs_scale": float(fixed["anchor_max_abs_scale"]),
                                    "guard_npz_sha256": sha(guard_npz), "guard_report_sha256": sha(guard_report_path),
                                    "313_owner_sha256": sha(owner313_path)},
        "depth_merit_classification": dm.get("classification"),
        "zero_clearance_achieved": False,
        "edge_and_solid_clearance_certified": False,
        "geometry_admitted": False, "native_admission": False, "accepted_physical_steps": 0,
        "solver_or_geometry_audit_rerun": False,
        "input_pins": pins
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"candidate_sha256":candidate_sha,
        "fit_support_rows":len(support),"changed_nodes":len(changed),"anchors":len(anchors),
        "exact_130":[len(base_points),len(current_points)],"exact_159":[base_surface_count,current_surface_count],
        "seed_rows":int(seed_lp["seed_rows_checked"]),"guard_rows":int(fixed["checked_rows"]),
        "area_gate_pass":area.get("pass"),"source_max_mm":receipt["max_source_displacement_mm"],
        "receipt":str(output),"receipt_sha256":sha(output)},sort_keys=True))


if __name__ == "__main__":
    main()

