#!/usr/bin/env python3
"""Reproduce the bounded arithmetic diagnosis for rounds 010/011.

No geometry query, solver, full surface audit, or native check is launched.
The script reads retained reports/arrays, recomputes pair membership, local
component labels, and direction/area arithmetic, then writes report.json.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D" / "surface-clearance-546"
OUT = Path(__file__).resolve().parent / "report.json"
ROUNDS = {
    "round010": D / "round-010-full-plane-trust4",
    "round011": D / "round-011-full-plane-trust4",
}
BASE = D / "raw-lp-trial-005-full-plane" / "fit" / "candidate.npz"
TARGETS = D / "round-010-selected-2326" / "prepare" / "surface-targets" / "targets.npz"
TARGET_REPORT = D / "round-010-selected-2326" / "prepare" / "surface-targets" / "report.json"
PAIR_INPUT = ROUNDS["round011"] / "fit.sat-crossing-inputs-458.npz"
PLANE_OWNER = ROOT / "D" / "D" / "fixed-bone-outward-plane-458" / "fixed_bone_outward_plane_458.py"
PAIR_OWNER = ROOT / "barycentric-connected-clearance-170" / "connected-local-continuation-189-run001" / "round-005" / "proposal" / "round-001" / "fit" / "self_separation_026.py"
SURFACE_OBJECTIVE = D / "surface_objective_546.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon_pairs(a: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(x[0]), int(x[1])))) for x in np.asarray(a)}


def overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    sb = canon_pairs(b)
    return sum(tuple(sorted((int(x[0]), int(x[1])))) in sb for x in np.asarray(a))


def fit_summary(name: str, fit_dir: Path) -> dict:
    report_path = fit_dir / "fit" / "report.json"
    fit = json.loads(report_path.read_text())
    audit_dir = fit_dir / "first-check" / "audit-130"
    audit_summary = json.loads((audit_dir / "summary.json").read_text())
    audit_pairs = np.load(audit_dir / "surface-self-pairs.npy", allow_pickle=False)
    actual_seed_path = Path(fit["inputs"]["self_pairs"])
    actual_seeds = np.load(actual_seed_path, allow_pickle=False)
    refreshed_seed_path = fit_dir / "constraints" / "self-seeds" / "merged-self-pair-seeds.npy"
    refreshed_seeds = np.load(refreshed_seed_path, allow_pickle=False)
    assert sha(actual_seed_path) == fit["inputs"]["self_pairs_sha256"]
    assert fit["candidate_sha256"] == audit_summary["candidate_sha256"]
    assert int(audit_summary["exact_PLC_surface_pairs"]) == len(audit_pairs)
    return {
        "fit_status": fit["status"],
        "fit_candidate_sha256": fit["candidate_sha256"],
        "fit_base_candidate": fit["inputs"]["base_candidate"],
        "fit_base_candidate_sha256": fit["inputs"]["base_candidate_sha256"],
        "fit_self_pairs_input": str(actual_seed_path),
        "fit_self_pairs_input_sha256": sha(actual_seed_path),
        "fit_self_pairs_input_count": int(len(actual_seeds)),
        "fit_self_pair_linear_row_count_reported": int(fit["constraints"]["self_pair_rows"]),
        "exact_audit_summary_path": str(audit_dir / "summary.json"),
        "exact_audit_status": audit_summary["status"],
        "exact_audit_pair_count": int(len(audit_pairs)),
        "exact_audit_pairs_path": str(audit_dir / "surface-self-pairs.npy"),
        "exact_pairs_overlapping_actual_fit_input": int(overlap_count(audit_pairs, actual_seeds)),
        "exact_pairs_new_relative_to_actual_fit_input": int(len(audit_pairs) - overlap_count(audit_pairs, actual_seeds)),
        "postfit_refreshed_seed_path": str(refreshed_seed_path),
        "postfit_refreshed_seed_sha256": sha(refreshed_seed_path),
        "postfit_refreshed_seed_count": int(len(refreshed_seeds)),
        "exact_pairs_overlapping_postfit_refresh": int(overlap_count(audit_pairs, refreshed_seeds)),
        "candidate_and_exact_audit_match": fit["candidate_sha256"] == audit_summary["candidate_sha256"],
    }


def component_graph(fit: dict, base: dict, seed_path: Path) -> tuple[dict, dict]:
    faces = base["source_O_faces"].astype(np.int64, copy=False)
    free = set(map(int, base["local_witness_cluster_222_support_rows"]))
    free.difference_update(map(int, fit["support"]["fixed_anchor_rows_in_support"]))
    area_rows = base["local_witness_cluster_222_area_face_rows"].astype(np.int64, copy=False)
    seed_pairs = np.load(seed_path, allow_pickle=False)
    parent = np.arange(len(base["points_world_m"]), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        if a not in free or b not in free:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for face_row in area_rows:
        nodes = faces[int(face_row)]
        union(int(nodes[0]), int(nodes[1]))
        union(int(nodes[1]), int(nodes[2]))
        union(int(nodes[2]), int(nodes[0]))
    for fa, fb in seed_pairs:
        active = [int(n) for n in np.concatenate((faces[int(fa)], faces[int(fb)])) if int(n) in free]
        if active:
            for n in active[1:]:
                union(active[0], n)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for node in free:
        groups[find(node)].append(node)
    ordered = sorted(groups.values(), key=len, reverse=True)
    labels = {node: ci for ci, nodes in enumerate(ordered) for node in nodes}

    witness_faces = set()
    for iteration in fit["area_scp_iterations"]:
        for trial in iteration["exact_f32_area_backtrack"]["trials"]:
            gate = trial["area_gate"]
            witness_faces.update(map(int, gate.get("bad_fixed_037_reference_face_rows_first_64", [])))
            witness_faces.update(map(int, gate.get("bad_current_oriented_face_rows_first_64", [])))
    out_labels = {}
    for face_row in sorted(witness_faces):
        labs = sorted({labels[int(n)] for n in faces[face_row] if int(n) in labels})
        out_labels[str(face_row)] = labs
    return {
        "component_graph_method": "union active free nodes in every support-area triangle and every input self-seed triangle-pair stencil; fixed anchors excluded",
        "component_count": len(ordered),
        "component_free_node_counts_descending": [len(x) for x in ordered],
        "input_seed_count": int(len(seed_pairs)),
        "area_face_count": int(len(area_rows)),
        "witness_face_component_ids": out_labels,
    }, {"faces": faces, "labels": labels, "components": ordered}


def target_plane_summary(component_data: dict) -> dict:
    targets = np.load(TARGETS, allow_pickle=False)
    pair_input = np.load(PAIR_INPUT, allow_pickle=False)
    report = json.loads(TARGET_REPORT.read_text())
    fit_report = json.loads((ROUNDS["round011"] / "fit" / "report.json").read_text())
    plane_objective = fit_report["fixed_bone_outward_plane_objective_458"]
    tri = pair_input["fixed_triangles_world_mm"].astype(np.float64)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("degenerate fixed triangle in pinned pair input")
    normals /= lengths[:, None]
    proposal = targets["proposal_direction_world_unit"].astype(np.float64)
    dots = np.einsum("ij,ij->i", normals, proposal)
    banks = collections.Counter(zip(map(int, targets["bank_body"]), map(int, targets["bank_stable_bone_id"])))
    by_component: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    faces = component_data["faces"]
    labels = component_data["labels"]
    for i, face_row in enumerate(targets["face_rows"]):
        nodes = faces[int(face_row)]
        labs = {labels[int(n)] for n in nodes if int(n) in labels}
        if len(labs) == 1:
            ci = next(iter(labs))
            by_component[ci][(int(targets["bank_body"][i]), int(targets["bank_stable_bone_id"][i]))] += 1
    return {
        "pair_count": int(len(dots)),
        "target_npz_sha256": sha(TARGETS),
        "direction_definition": report["direction_method"],
        "plane_normals": "unit cross product of each ordered matched fixed-bone triangle",
        "bone_plane_normal_dot_certified_union_proposal_direction_quantiles": {
            str(q): float(np.quantile(dots, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "opposite_direction_count_dot_lt_0": int(np.count_nonzero(dots < 0.0)),
        "poor_alignment_count_dot_lt_0_5": int(np.count_nonzero(dots < 0.5)),
        "well_aligned_count_dot_ge_0_9": int(np.count_nonzero(dots >= 0.9)),
        "bank_pair_counts_descending": [
            {"body": k[0], "stable_bone_id": k[1], "pairs": v}
            for k, v in banks.most_common()
        ],
        "bank_normal_mismatch_summary": [
            {
                "body": key[0], "stable_bone_id": key[1], "pairs": int(mask.sum()),
                "dot_median": float(np.median(dots[mask])),
                "dot_p10": float(np.quantile(dots[mask], 0.1)),
                "opposite_count": int(np.count_nonzero(dots[mask] < 0.0)),
                "top_component_ids": [
                    {"component": int(ci), "count": int(by_component[ci][key])}
                    for ci in sorted(by_component, key=lambda c: by_component[c][key], reverse=True)
                    if by_component[ci][key]
                ][:3],
            }
            for key, _ in banks.most_common(12)
            for mask in [np.logical_and(targets["bank_body"] == key[0], targets["bank_stable_bone_id"] == key[1])]
        ],
        "hard_and_soft_row_semantics": {
            "objective": "three corner constraints per paired skin face against its matched bone triangle plane; only positive lower bounds are soft targets, nonpositive are hard nonregression rows",
            "positive_soft_rows": int(plane_objective["row_counts"]["positive_bone_plane_target_rows"]),
            "nonpositive_hard_rows": int(plane_objective["row_counts"]["nonpositive_corner_hard_rows"]),
            "point_objective_rows": int(plane_objective["row_counts"]["point_row_count"]),
            "matched_pairs": int(plane_objective["row_counts"]["bone_plane_target_pair_count"]),
            "each_plane_separates_complete_triangle_pair": True,
            "global_bank_or_global_surface_clearance_certified": False,
        },
        "plane_owner_pin": sha(PLANE_OWNER),
        "target_direction_limit": "per-pair proposal direction is a certified complete-bank union exit; the plane owner verifies outward winding but does not certify the individual triangle plane supports the whole possibly curved/nonconvex bank",
    }


def area_summary(fit: dict, graph: dict) -> tuple[dict, dict]:
    iterations = []
    witness_components = graph["witness_face_component_ids"]
    for it in fit["area_scp_iterations"]:
        trials = []
        for tr in it["exact_f32_area_backtrack"]["trials"]:
            ag = tr["area_gate"]
            ids = list(map(int, ag.get("bad_fixed_037_reference_face_rows_first_64", [])))
            cur = list(map(int, ag.get("bad_current_oriented_face_rows_first_64", [])))
            trials.append({
                "alpha": float(tr["alpha"]),
                "accepted": bool(tr["accepted"]),
                "reason": tr.get("reason"),
                "minimum_fixed_037_reference_magnitude_ratio": ag.get("minimum_fixed_037_reference_magnitude_ratio"),
                "bad_fixed_037_faces": ids,
                "bad_fixed_037_components": sorted({ci for f in ids for ci in witness_components.get(str(f), [])}),
                "bad_current_oriented_faces_first_12": cur[:12],
                "bad_current_oriented_components": sorted({ci for f in cur for ci in witness_components.get(str(f), [])}),
                "edge_gradient_pass": tr.get("edge_gradient_pass"),
                "minimum_edge_gradient_slack_mm": tr.get("minimum_edge_gradient_slack_mm"),
                "source_max_mm": tr.get("existing_emitted_gate", {}).get("maximum_source_displacement_mm"),
                "node_step_max_mm": tr.get("existing_emitted_gate", {}).get("maximum_node_step_mm"),
            })
        iterations.append({
            "iteration": int(it["iteration"]),
            "accepted_alpha": it["exact_f32_area_backtrack"].get("accepted_alpha"),
            "accepted_area_ratio": it["accepted_iterate_area_gate"].get("minimum_fixed_037_reference_magnitude_ratio"),
            "trials": trials,
        })
    last = iterations[-1]
    samples = []
    for tr in last["trials"]:
        if tr["alpha"] in (0.125, 0.25):
            samples.append((tr["alpha"], tr["minimum_fixed_037_reference_magnitude_ratio"]))
    alpha_estimate = None
    if len(samples) == 2:
        (a0, r0), (a1, r1) = sorted(samples)
        if r0 >= 0.1 and r1 < 0.1 and r1 != r0:
            alpha_estimate = float(a0 + (0.1 - r0) * (a1 - a0) / (r1 - r0))
    emit = fit["emitted_f32_checks"]
    return {
        "area_scp_iterations": iterations,
        "last_iteration_face_40715_linear_interpolated_area_floor_alpha_estimate": alpha_estimate,
        "alpha_estimate_is_not_a_f32_gate": True,
        "final_emitted_gate": {
            "status": fit["status"],
            "minimum_fixed_037_area_ratio": emit["minimum_fixed_037_reference_magnitude_ratio"],
            "minimum_current_oriented_ratio": emit["minimum_constrained_current_oriented_ratio"],
            "maximum_source_displacement_mm": emit["maximum_source_displacement_mm"],
            "source_cap_mm": fit["source_cap_mm"],
            "maximum_node_step_mm": emit["maximum_node_step_mm"],
            "trust_mm": fit["support"].get("trust_mm", 4.0),
            "minimum_edge_gradient_slack_mm": emit["edge_gradient_minimum_emitted_slack_mm"],
            "edge_gradient_pass": emit["edge_gradient_emission_pass"],
            "exact_global_self_audit": fit["exact_global_self_audit"],
        },
        "gate_conclusion": "for iteration 4, alpha .25 failed only fixed-037 face 40715 in component 0; edge/source/trust passed. At alpha .5 only face 40715 remained an area failure and edge/source/trust passed. At alpha 1 area/current-orientation failures crossed several components and edge failed; source/trust still passed.",
    }, {"area_witness_components": witness_components}


def main() -> None:
    fit_reports = {}
    self_membership = {}
    hashes: dict[str, str] = {}

    for name, fit_dir in ROUNDS.items():
        fit_reports[name] = fit_summary(name, fit_dir)
        exact = np.load(fit_dir / "first-check" / "audit-130" / "surface-self-pairs.npy", allow_pickle=False)
        used = np.load(fit_reports[name]["fit_self_pairs_input"], allow_pickle=False)
        post = np.load(fit_reports[name]["postfit_refreshed_seed_path"], allow_pickle=False)
        self_membership[name] = {
            "fit_input_overlap": fit_reports[name]["exact_pairs_overlapping_actual_fit_input"],
            "fit_input_new_pairs": fit_reports[name]["exact_pairs_new_relative_to_actual_fit_input"],
            "postfit_refresh_overlap": fit_reports[name]["exact_pairs_overlapping_postfit_refresh"],
            "fit_input_sha_matches_report": sha(Path(fit_reports[name]["fit_self_pairs_input"])) == json.loads((ROUNDS[name] / "fit" / "report.json").read_text())["inputs"]["self_pairs_sha256"],
            "postfit_refresh_pair_count": int(len(post)),
        }
    r11 = json.loads((ROUNDS["round011"] / "fit" / "report.json").read_text())
    base_path = Path(r11["inputs"]["base_candidate"])
    base = np.load(base_path, allow_pickle=False)
    graph, graph_data = component_graph(r11, base, Path(r11["inputs"]["self_pairs"]))
    area, _ = area_summary(r11, graph)
    direction = target_plane_summary(graph_data)

    input_paths = {
        "round010_fit_report": ROUNDS["round010"] / "fit" / "report.json",
        "round010_exact_summary": ROUNDS["round010"] / "first-check" / "audit-130" / "summary.json",
        "round010_exact_pairs": ROUNDS["round010"] / "first-check" / "audit-130" / "surface-self-pairs.npy",
        "round011_fit_report": ROUNDS["round011"] / "fit" / "report.json",
        "round011_exact_summary": ROUNDS["round011"] / "first-check" / "audit-130" / "summary.json",
        "round011_exact_pairs": ROUNDS["round011"] / "first-check" / "audit-130" / "surface-self-pairs.npy",
        "round010_selected_seed_pool": D / "round-010-selected-2326" / "prepare" / "constraints" / "self-seeds" / "merged-self-pair-seeds.npy",
        "round010_fit_seed_pool": Path(fit_reports["round010"]["fit_self_pairs_input"]),
        "round010_postfit_seed_pool": Path(fit_reports["round010"]["postfit_refreshed_seed_path"]),
        "round011_fit_seed_pool": Path(fit_reports["round011"]["fit_self_pairs_input"]),
        "round011_postfit_seed_pool": Path(fit_reports["round011"]["postfit_refreshed_seed_path"]),
        "base_candidate": base_path,
        "targets_npz": TARGETS,
        "targets_report": TARGET_REPORT,
        "plane_pair_input": PAIR_INPUT,
        "plane_owner": PLANE_OWNER,
        "pair_owner": PAIR_OWNER,
        "surface_objective_source": SURFACE_OBJECTIVE,
    }
    for key, path in input_paths.items():
        hashes[key] = sha(path)

    report = {
        "schema": "numi.human.surface_line_search_target_diagnosis546.v1",
        "scope": "read-only report parsing and narrow arithmetic; no solver, geometry query, full audit, or native run",
        "timings_measured": False,
        "physical_or_anatomical_admission": False,
        "inputs_sha256": hashes,
        "self_pair_chronology": {
            "round010": fit_reports["round010"],
            "round011": fit_reports["round011"],
            "interpretation": "113 round010 and 23 round011 exact self pairs were absent from the respective fit's actual seed input; each appears in its later refreshed pool. These are new post-fit witness cuts, not known violated SAT rows or Float32-only residuals. No 23/113 overlap is asserted.",
            "recomputed_026_row_margin_evaluation": "not applicable to the exact pairs because each pair set has zero overlap with the corresponding fit-time seed input",
            "pair_owner_sha256": sha(PAIR_OWNER),
        },
        "round011_area_and_component_diagnosis": {
            "candidate_sha256": r11["candidate_sha256"],
            "actual_fit_seed_count": fit_reports["round011"]["fit_self_pairs_input_count"],
            "support_area_and_seed_component_graph": graph,
            "area_gate_trials": area,
        },
        "target_plane_diagnosis": direction,
        "limits": [
            "The 0.1 area floor, 0.5 edge gradient cap, 12 mm source cap and 4 mm trust region are construction gates from this fit, not anatomy or physical tissue requirements.",
            "Bone triangle winding/positive-volume checks do not prove convexity or that each local triangle plane supports its entire bank.",
            "The per-pair plane rows guarantee separation only from the paired triangle when all corner inequalities hold; they do not certify the whole bone-bank union.",
            "Per-component alpha is a viable construction-only line-search change because hard nonlinear area/edge and existing seed stencils partition into the same exact 14 components here, but full F32 checks and complete exact surface/self audits must still be run after recombination.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(OUT),
        "round010_exact_pairs": fit_reports["round010"]["exact_audit_pair_count"],
        "round010_missing_from_fit_input": fit_reports["round010"]["exact_pairs_new_relative_to_actual_fit_input"],
        "round011_exact_pairs": fit_reports["round011"]["exact_audit_pair_count"],
        "round011_missing_from_fit_input": fit_reports["round011"]["exact_pairs_new_relative_to_actual_fit_input"],
        "round011_components": graph["component_free_node_counts_descending"],
        "area_alpha_estimate_only": area["last_iteration_face_40715_linear_interpolated_area_floor_alpha_estimate"],
        "bone_plane_dot_median": direction["bone_plane_normal_dot_certified_union_proposal_direction_quantiles"]["0.5"],
        "bone_plane_dot_negative": direction["opposite_direction_count_dot_lt_0"],
        "timings_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
