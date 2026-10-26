#!/usr/bin/env python3
"""Build an unadmitted failed-pose witness from recorded component alphas.

This deliberately ignores historical self-intersection fields in the 549
component-proposal report. Each chosen alpha must have passed the recorded
non-self construction gates and pass a fresh 408/549 arithmetic recheck.
The resulting pose is for a parent-owned exact 130 audit only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER408 = PROJECT / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER549 = PROJECT / "D/component-area-backtracking-549/component_area_backtracking549.py"
R0 = PROJECT / "full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
OWNER408_SHA = "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
OWNER549_SHA = "523b1aef89d92eef09bf63e9de2d4f3fa6bd938ef53a45915f9ea138f5b8f249"
R0_SHA = "70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
AREA_FLOOR = 0.1
SOURCE_CAP_MM = 12.0
EDGE_GRADIENT = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def import_pinned(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def recorded_nongeometry_pass(trial: dict, trust_mm: float) -> tuple[bool, list[str]]:
    """Gate only recorded area/trust/source/edge metrics; ignore self fields."""
    failures = []
    required = (
        "area_pass",
        "maximum_step_mm",
        "maximum_source_distance_mm",
        "minimum_edge_slack_mm",
    )
    missing = [k for k in required if k not in trial]
    if missing:
        return False, [f"missing_recorded_fields:{','.join(missing)}"]
    try:
        area_pass = bool(trial["area_pass"])
        step = float(trial["maximum_step_mm"])
        source = float(trial["maximum_source_distance_mm"])
        edge = float(trial["minimum_edge_slack_mm"])
    except (TypeError, ValueError, OverflowError):
        return False, ["invalid_recorded_metric"]
    if not np.isfinite([step, source, edge]).all():
        return False, ["nonfinite_recorded_metric"]
    if not area_pass:
        failures.append("recorded_area")
    if step > trust_mm + 1e-6:
        failures.append("recorded_trust")
    if source > SOURCE_CAP_MM + 1e-6:
        failures.append("recorded_source_cap")
    if edge < -1e-7:
        failures.append("recorded_edge_gradient")
    return not failures, failures


def emit_component(base: np.ndarray, delta_mm: np.ndarray,
                   nodes: np.ndarray, alpha: float) -> np.ndarray:
    """Use the frozen 549 Float32 emission order for one component."""
    if base.dtype != np.float32 or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base points must be Float32 (N,3)")
    if delta_mm.shape != base.shape or not np.isfinite(delta_mm).all():
        raise ValueError("raw displacement must be finite (N,3) millimeters")
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("component alpha outside [0,1]")
    g = np.asarray(nodes, dtype=np.int64)
    if g.ndim != 1 or len(np.unique(g)) != len(g) or np.any(g < 0) or np.any(g >= len(base)):
        raise ValueError("component node rows invalid")
    out = base.copy()
    if alpha:
        base_mm = base.astype(np.float64) * 1000.0
        out[g] = ((base_mm[g] + alpha * delta_mm[g]) / 1000.0).astype(np.float32)
    return out


def select_component_alpha(base: np.ndarray, delta_mm: np.ndarray, nodes: np.ndarray,
                           component_record: dict, trust_mm: float,
                           exact_arithmetic_gate: Callable[[np.ndarray], dict]) -> tuple[float, dict]:
    """Pick the largest *recorded* alpha that passes non-self gates and recheck."""
    trials = component_record.get("trials")
    if not isinstance(trials, list):
        raise ValueError("component trials must be a list")
    candidates = []
    rejected = []
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict) or "alpha" not in trial:
            raise ValueError("malformed recorded component trial")
        alpha = float(trial["alpha"])
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0 or alpha in seen:
            raise ValueError("invalid or duplicate recorded alpha")
        seen.add(alpha)
        metrics_ok, metric_failures = recorded_nongeometry_pass(trial, trust_mm)
        if not metrics_ok:
            rejected.append({"alpha": alpha, "reason": metric_failures})
            continue
        pose = emit_component(base, delta_mm, nodes, alpha)
        exact = exact_arithmetic_gate(pose)
        if not bool(exact.get("pass", False)):
            rejected.append({
                "alpha": alpha,
                "reason": ["fresh_408_arithmetic_recheck"],
                "fresh_arithmetic": exact,
            })
            continue
        candidates.append((alpha, exact, trial))
    if candidates:
        alpha, exact, selected_trial = max(candidates, key=lambda row: row[0])
    else:
        alpha, exact, selected_trial = 0.0, exact_arithmetic_gate(base), None
        if not exact.get("pass", False):
            raise ValueError("baseline fails fresh 408 arithmetic gate")
    return alpha, {
        "selected_alpha": alpha,
        "selected_recorded_trial": selected_trial,
        "fresh_component_only_arithmetic": exact,
        "rejected_recorded_trials": rejected,
        "recorded_trial_count": len(trials),
        "selection_ignored_fields": ["pass", "exact_self_pairs", "selected_face_count",
                                     "exact_incremental_reuse"],
        "self_intersection_audit_performed": False,
    }


def combine_components(base: np.ndarray, delta_mm: np.ndarray,
                       groups: list[np.ndarray], alphas: list[float]) -> np.ndarray:
    if len(groups) != len(alphas):
        raise ValueError("component/alpha count mismatch")
    seen = set()
    out = base.copy()
    base_mm = base.astype(np.float64) * 1000.0
    for nodes, alpha in zip(groups, alphas):
        g = np.asarray(nodes, dtype=np.int64)
        if any(int(i) in seen for i in g):
            raise ValueError("components overlap")
        seen.update(map(int, g))
        if alpha:
            out[g] = ((base_mm[g] + float(alpha) * delta_mm[g]) / 1000.0).astype(np.float32)
    return out


def payload_with_points(base_payload: dict[str, np.ndarray],
                        points: np.ndarray) -> dict[str, np.ndarray]:
    if points.dtype != np.float32 or points.shape != base_payload["points_world_m"].shape:
        raise ValueError("replacement point array has wrong dtype/shape")
    out = dict(base_payload)
    out["points_world_m"] = points
    for key in base_payload:
        if key != "points_world_m" and not same(out[key], base_payload[key]):
            raise AssertionError(f"nonpoint payload changed: {key}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="raw-direction-001 directory containing candidate.npz and report.json")
    ap.add_argument("--component-report", type=Path, required=True)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new output directory")
    ap.add_argument("--trust-mm", type=float, required=True)
    args = ap.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.trust_mm) or args.trust_mm <= 0:
        raise ValueError("trust-mm must be finite and positive")
    if sha(OWNER408) != OWNER408_SHA or sha(OWNER549) != OWNER549_SHA or sha(R0) != R0_SHA:
        raise ValueError("pinned arithmetic/source owner mismatch")

    component_report = json.loads(args.component_report.read_text())
    raw_report_path = args.raw_dir / "report.json"
    raw_candidate = args.raw_dir / "candidate.npz"
    raw_report = json.loads(raw_report_path.read_text())
    if component_report.get("status") != "CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
        raise ValueError("unexpected component-proposal status")
    if raw_report.get("status") != "RAW_LP_DIRECTION_FOR_EXISTING_545_QUALITY_PROJECTION":
        raise ValueError("unexpected raw-direction status")
    if component_report.get("native_admission") is not False or raw_report.get("native_admission") is not False:
        raise ValueError("input report claims native admission")
    if component_report.get("accepted_physical_steps") != 0 or raw_report.get("accepted_physical_steps") != 0:
        raise ValueError("input report claims accepted physical steps")
    if args.base != Path(component_report["base_candidate"]) or args.base != Path(raw_report["base_candidate"]):
        raise ValueError("base argument does not match both source reports")
    if sha(args.base) != component_report["base_sha256"] or sha(args.base) != raw_report["base_sha256"]:
        raise ValueError("base candidate hash mismatch")
    if Path(raw_report["candidate_path"]) != raw_candidate or sha(raw_candidate) != raw_report["candidate_sha256"]:
        raise ValueError("raw candidate path/hash mismatch")
    if raw_report.get("raw_delta_units") != "millimeters":
        raise ValueError("raw direction is not recorded in millimeters")
    if Path(component_report["candidate_path"]).name != "candidate.npz":
        raise ValueError("component proposal candidate path mismatch")
    component_candidate = Path(component_report["candidate_path"])
    if sha(component_candidate) != component_report["candidate_sha256"]:
        raise ValueError("component proposal candidate hash mismatch")
    if Path(component_report["base_candidate"]) != args.base:
        raise ValueError("component proposal base mismatch")
    if args.context != Path(next(
        (p for p in component_report["source_pins"] if p.endswith("/constraint-context.json")),
        "",
    )):
        raise ValueError("context argument differs from pinned component context")
    if sha(args.context) != component_report["source_pins"][str(args.context)]:
        raise ValueError("constraint-context pin mismatch")

    # Every source pin carried by 549 is rechecked; this binds the exact prepared
    # seed/guard inputs and the arithmetic/source owners used by the proposal.
    for pinned_path, pinned_sha in component_report["source_pins"].items():
        p = Path(pinned_path)
        if not p.is_file() or sha(p) != pinned_sha:
            raise ValueError(f"source pin mismatch: {p}")

    base = read_npz(args.base)
    raw = read_npz(raw_candidate)
    prior_component = read_npz(component_candidate)
    if set(base) != set(raw) or set(base) != set(prior_component):
        raise ValueError("candidate archive key mismatch")
    for key in base:
        if key == "points_world_m":
            continue
        if not same(base[key], raw[key]) or not same(base[key], prior_component[key]):
            raise ValueError(f"nonpoint input payload differs: {key}")
    x = base["points_world_m"]
    target = raw["points_world_m"]
    if x.dtype != np.float32 or target.dtype != np.float32 or x.shape != target.shape:
        raise ValueError("point arrays must be matching Float32 (N,3)")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite point array")
    ctx = json.loads(args.context.read_text())
    if ctx.get("candidate_sha256") != sha(args.base):
        raise ValueError("context is not bound to the base")
    free = np.asarray(raw_report["free_nodes"], dtype=np.int64)
    if not np.array_equal(free, np.asarray(component_report["free_nodes"], dtype=np.int64)):
        raise ValueError("raw and component free-node lists differ")
    if len(np.unique(free)) != len(free) or np.any(free < 0) or np.any(free >= len(x)):
        raise ValueError("invalid free-node list")
    anchors = np.asarray(base["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    if len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(x)):
        raise ValueError("invalid typed anchor rows")
    if np.intersect1d(free, anchors).size:
        raise ValueError("free-node set contains typed anchors")
    outside = np.setdiff1d(np.arange(len(x), dtype=np.int64), free)
    if not same(x[outside], target[outside]) or not same(x[anchors], target[anchors]):
        raise ValueError("raw direction changed outside-support or anchor points")

    # Confirm the saved component partition against the pinned 549 owner and
    # the exact context seed/guard inputs instead of trusting report node lists.
    q = import_pinned(OWNER549, "owner549_build_failed_pose554")
    m = import_pinned(OWNER408, "owner408_build_failed_pose554")
    faces = base["source_O_faces"]
    pair_path = Path(ctx["seed_pairs_path"])
    pairs = np.load(pair_path, allow_pickle=False)
    groups, graph_proof = q.components(len(x), faces, free, pairs)
    records = component_report.get("components")
    if not isinstance(records, list) or len(records) != len(groups):
        raise ValueError("component report count differs from rebuilt stencil graph")
    for i, (group, record) in enumerate(zip(groups, records)):
        nodes = np.asarray(record.get("nodes"), dtype=np.int64)
        if not np.array_equal(group, nodes):
            raise ValueError(f"component node set differs from owner graph at {i}")
    claimed = component_report.get("independence_proof", {})
    for key in ("all_stencils_single_mutable_owner", "complete_face_stencils",
                "complete_self_pair_stencils", "free_nodes", "components"):
        if claimed.get(key) != graph_proof.get(key):
            raise ValueError(f"component proof mismatch in {key}")

    guard_path = Path(ctx["guard_npz"])
    guard = read_npz(guard_path)
    guard_rows = np.asarray(guard["skin_face_rows"], dtype=np.int64)
    if guard_rows.ndim != 1 or np.any(guard_rows < 0) or np.any(guard_rows >= len(faces)):
        raise ValueError("fixed guard face rows invalid")
    owner = np.full(len(x), -1, dtype=np.int64)
    for ci, g in enumerate(groups):
        owner[g] = ci
    for face_row in guard_rows:
        mutable = owner[faces[int(face_row)]]
        mutable = mutable[mutable >= 0]
        if len(np.unique(mutable)) > 1:
            raise ValueError(f"fixed guard stencil crosses components at face {face_row}")
    if len(guard_rows) != claimed.get("fixed_bone_guard_face_stencils"):
        raise ValueError("fixed guard count does not match 549 component proof")

    bm = x.astype(np.float64) * 1000.0
    ref_mm = base["area_reference_037_points_world_m_f64"] * 1000.0
    EG, rhs, edge_meta = m.build_edge_gradient_constraints(bm, faces, free, EDGE_GRADIENT)
    with np.load(args.base, allow_pickle=False) as base_archive:
        source, source_receipts = m.compose_source_reference(base_archive, R0)
    delta_mm = (target.astype(np.float64) - x.astype(np.float64)) * 1000.0

    def arithmetic_gate(points: np.ndarray) -> dict:
        area = m.full_f32_area_gate(points, bm, ref_mm, faces, AREA_FLOOR)
        step = np.linalg.norm(points.astype(np.float64) - x.astype(np.float64), axis=1) * 1000.0
        source_distance = np.linalg.norm(points.astype(np.float64) - source, axis=1) * 1000.0
        edge_slack = np.asarray(EG @ ((points.astype(np.float64) * 1000.0 - bm)[free].ravel()) - rhs)
        max_step = float(step.max(initial=0.0))
        max_source = float(source_distance.max(initial=0.0))
        min_edge = float(edge_slack.min(initial=1e30))
        outside_unchanged = same(points[outside], x[outside])
        anchors_unchanged = same(points[anchors], x[anchors])
        checks = {
            "area_pass": bool(area["pass"]),
            "trust_pass": max_step <= args.trust_mm + 1e-6,
            "source_cap_pass": max_source <= SOURCE_CAP_MM + 1e-6,
            "edge_gradient_pass": min_edge >= -1e-7,
            "outside_support_unchanged": outside_unchanged,
            "anchors_unchanged": anchors_unchanged,
            "maximum_step_mm": max_step,
            "maximum_source_distance_mm": max_source,
            "minimum_edge_slack_mm": min_edge,
            "minimum_current_oriented_ratio": area["minimum_current_oriented_ratio"],
            "minimum_fixed_037_reference_magnitude_ratio": area["minimum_fixed_037_reference_magnitude_ratio"],
            "faces_checked": area["faces_checked"],
            "source_cap_mm": SOURCE_CAP_MM,
            "trust_mm": args.trust_mm,
            "area_floor": AREA_FLOOR,
            "edge_gradient_cap": EDGE_GRADIENT,
        }
        checks["pass"] = bool(all(checks[k] for k in (
            "area_pass", "trust_pass", "source_cap_pass", "edge_gradient_pass",
            "outside_support_unchanged", "anchors_unchanged",
        )))
        return checks

    if not arithmetic_gate(x)["pass"]:
        raise ValueError("base fails current arithmetic gates")
    chosen_alphas = []
    component_choices = []
    for group, record in zip(groups, records):
        alpha, choice = select_component_alpha(
            x, delta_mm, group, record, args.trust_mm, arithmetic_gate
        )
        chosen_alphas.append(alpha)
        component_choices.append({"component": int(record["component"]),
                                  "node_count": int(len(group)), **choice})
    combined = combine_components(x, delta_mm, groups, chosen_alphas)
    final_gate = arithmetic_gate(combined)
    if not final_gate["pass"]:
        raise ValueError(f"combined pose failed final arithmetic recheck: {final_gate}")

    if not same(combined[outside], x[outside]) or not same(combined[anchors], x[anchors]):
        raise AssertionError("combined pose changed a fixed point")
    payload = payload_with_points(base, combined)
    args.output.mkdir(parents=True)
    candidate_path = args.output / "candidate.npz"
    np.savez_compressed(candidate_path, **payload)
    pins = {
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "baseline": {"path": str(args.base), "sha256": sha(args.base)},
        "raw_candidate": {"path": str(raw_candidate), "sha256": sha(raw_candidate)},
        "raw_report": {"path": str(raw_report_path), "sha256": sha(raw_report_path)},
        "component_candidate": {"path": str(component_candidate), "sha256": sha(component_candidate)},
        "component_report": {"path": str(args.component_report), "sha256": sha(args.component_report)},
        "constraint_context": {"path": str(args.context), "sha256": sha(args.context)},
        "seed_pairs": {"path": str(pair_path), "sha256": sha(pair_path)},
        "fixed_guards": {"path": str(guard_path), "sha256": sha(guard_path)},
        "owner408": {"path": str(OWNER408), "sha256": sha(OWNER408)},
        "owner549": {"path": str(OWNER549), "sha256": sha(OWNER549)},
        "source_reference": {"path": str(R0), "sha256": sha(R0)},
    }
    report = {
        "schema": "numi.human.failed_pose_line_search_cuts554.v1",
        "status": "FAILED_WITNESS_PROPOSAL_FOR_PARENT_EXACT_AUDIT",
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha(candidate_path),
        "base_candidate": str(args.base),
        "base_sha256": sha(args.base),
        "raw_candidate": str(raw_candidate),
        "raw_candidate_sha256": sha(raw_candidate),
        "component_report": str(args.component_report),
        "component_report_sha256": sha(args.component_report),
        "context": str(args.context),
        "context_sha256": sha(args.context),
        "free_node_count": int(len(free)),
        "component_count": len(groups),
        "selected_alphas": chosen_alphas,
        "component_choices": component_choices,
        "independence_proof": graph_proof,
        "fixed_bone_guard_face_count": int(len(guard_rows)),
        "edge_constraint_metadata": edge_meta,
        "source_reference_receipts": source_receipts,
        "combined_arithmetic_recheck": final_gate,
        "all_nonpoint_payload_bitwise_unchanged": all(
            same(payload[k], base[k]) for k in base if k != "points_world_m"
        ),
        "all_points_outside_free_nodes_bitwise_unchanged": same(combined[outside], x[outside]),
        "typed_anchors_bitwise_unchanged": same(combined[anchors], x[anchors]),
        "exact_self_audit": "NOT_RUN_PARENT_MUST_RUN_COMPLETE_130",
        "bone_guard_exact_admission": "NOT_RUN",
        "native_admission": False,
        "accepted_physical_steps": 0,
        "scope": "retained recorded-alpha arithmetic-feasible failed-pose witness; self intersections intentionally not screened here",
        "source_pins": pins,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "candidate": str(candidate_path),
        "sha256": report["candidate_sha256"],
        "components": len(groups),
        "selected_alphas": chosen_alphas,
        "exact_self_audit": report["exact_self_audit"],
        "native_admission": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

