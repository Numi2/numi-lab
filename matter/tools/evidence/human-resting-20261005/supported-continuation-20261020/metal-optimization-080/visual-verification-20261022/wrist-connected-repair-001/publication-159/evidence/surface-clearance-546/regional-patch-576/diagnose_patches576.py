"""Bounded pure-array diagnostic of 571's first selected patches.

This replays 571's deterministic patch ordering and stops before Delaunay,
quality, exact self, typed-bank predicates, or any solver.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
N = ROOT / "D/surface-clearance-546"
STATE_DEFAULT = N / "connected-star-573/run-001/final/state.json"
CANDIDATE_OWNER = N / "regional-patch-571/regional_candidates571.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_order(faces, pair_report, limit):
    pairs_by_bank = {}
    for bank in pair_report["banks"]:
        key = (int(bank["body"]), int(bank["stable_bone_id"]))
        raw = (bank.get("result") or {}).get("triangle_pairs", [])
        pairs_by_bank[key] = set((int(p[0]), int(p[1])) for p in raw)
    if len(pairs_by_bank) != pair_report["expected_bank_count"]:
        raise ValueError("surface report does not cover expected typed banks")
    face_counts = {key: Counter(int(p[0]) for p in pairs) for key, pairs in pairs_by_bank.items()}
    targets = {int(p[0]) for pairs in pairs_by_bank.values() for p in pairs}
    nfaces = len(faces)
    if any(row < 0 or row >= nfaces for row in targets):
        raise ValueError("surface report has face rows outside candidate")
    incidence = defaultdict(list)
    neighbors = defaultdict(set)
    for row, tri in enumerate(faces):
        a, b, c = map(int, tri)
        for node in (a, b, c):
            incidence[node].append(row)
        for u, v in ((a, b), (b, c), (c, a)):
            neighbors[u].add(v)
            neighbors[v].add(u)
    vertex_scores = Counter()
    for row in targets:
        score = sum(face_counts[key].get(row, 0) for key in pairs_by_bank)
        for node in faces[row]:
            vertex_scores[int(node)] += score
    patches = {}
    seeds = sorted(vertex_scores, key=lambda node: (-vertex_scores[node], node))
    for seed in seeds:
        ball = {seed}
        for radius in range(1, 4):
            rows = tuple(sorted({row for node in ball for row in incidence[node]}))
            if 4 <= len(rows) <= 160:
                patches[rows] = (int(seed), radius)
            ball |= {neighbor for node in tuple(ball) for neighbor in neighbors[node]}
    ordered = sorted(
        patches,
        key=lambda rows: (
            -sum(face_counts[key].get(row, 0) for key in pairs_by_bank for row in rows),
            len(rows),
            rows,
        ),
    )
    return ordered[:limit], patches, pairs_by_bank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=STATE_DEFAULT)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-boundary", type=int, default=48)
    args = ap.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.limit < 1 or args.max_boundary < 3:
        raise ValueError("limit and max-boundary must be positive")

    state_path = args.state
    state = json.loads(state_path.read_text())
    for key in ("candidate", "surface_report", "point_summary"):
        if sha(state[key]) != state[key + "_sha256"]:
            raise ValueError(f"state input pin mismatch: {key}")
    candidate_path = Path(state["candidate"])
    report_path = Path(state["surface_report"])
    with np.load(candidate_path, allow_pickle=False) as z:
        points = np.asarray(z["points_world_m"])
        faces = np.asarray(z["source_O_faces"])
        reference = np.asarray(z["area_reference_037_points_world_m_f64"])
        face_kind = np.asarray(z["face_kind"])
    report = json.loads(report_path.read_text())
    if report.get("candidate_sha256") != state["candidate_sha256"] or report.get("complete") is not True:
        raise ValueError("surface report is not complete and matched")
    if len(report.get("banks", [])) != 185:
        raise ValueError("expected complete 185-bank report")
    if points.dtype != np.float32 or faces.shape != (len(faces), 3):
        raise ValueError("unexpected candidate emission arrays")
    expected_pairs = sum(int((b.get("result") or {}).get("count", 0)) for b in report["banks"])

    owner = load_module("regional_candidates571_diagnostic576", CANDIDATE_OWNER)
    _patch_owner, star = owner._owners()
    selected, patches, pairs = patch_order(faces, report, args.limit)
    counters = Counter()
    exact_first_failure = Counter()
    topology_failures = Counter()
    chart_failures = Counter()
    witnesses = defaultdict(list)
    current_boundary_self = 0
    reference_boundary_self = 0
    topology_disk = 0
    both_charts_valid = 0
    inside_both = 0
    outside_current_nodes = 0
    outside_reference_nodes = 0
    attempted_after_material_filter = 0

    for rank, rows_tuple in enumerate(selected):
        rows = list(rows_tuple)
        seed, radius = patches[rows_tuple]
        row_ids = np.asarray(rows, dtype=np.int64)
        kinds = np.unique(face_kind[row_ids])
        base_rec = {"rank": rank, "seed": seed, "radius": radius, "rows": rows, "face_count": len(rows)}
        if len(kinds) != 1:
            counters["mixed_material_precheck"] += 1
        else:
            attempted_after_material_filter += 1
        try:
            row_ids, old, boundary, vertices = owner._boundary_and_vertices(
                points, faces, rows, args.max_boundary
            )
            topology_disk += 1
            base_rec.update(
                boundary_count=len(boundary),
                interior_nodes=sorted(set(map(int, vertices)) - set(map(int, boundary))),
            )
        except Exception as exc:
            msg = str(exc)
            # A cap rejection can still be a valid larger disk; distinguish it
            # from malformed topology without changing the exact 571 message.
            if msg.startswith("boundary count ") and "exceeds max_boundary=" in msg:
                try:
                    _, _, _, _ = owner._boundary_and_vertices(points, faces, rows, 1_000_000)
                    topology_disk += 1
                except Exception:
                    pass
                counters["boundary_limit"] += 1
            counters["topology_" + msg] += 1
            topology_failures[msg] += 1
            if len(witnesses["topology:" + msg]) < 5:
                witnesses["topology:" + msg].append({**base_rec, "error": msg, "exception_type": type(exc).__name__})
            exact_first_failure[msg] += 1
            continue

        if len(kinds) != 1:
            exact_first_failure["mixed face_kind"] += 1
            continue
        try:
            current = owner._chart(points, boundary, vertices)
        except Exception as exc:
            msg = str(exc)
            counters["current_chart_" + msg] += 1
            chart_failures["current:" + msg] += 1
            if msg == "non-simple chart":
                current_boundary_self += 1
            exact_first_failure[msg] += 1
            if len(witnesses["current_chart:" + msg]) < 5:
                witnesses["current_chart:" + msg].append({**base_rec, "boundary_count": len(boundary), "error": msg})
            continue
        try:
            reference_chart = owner._chart(reference, boundary, vertices)
        except Exception as exc:
            msg = str(exc)
            counters["reference_chart_" + msg] += 1
            chart_failures["reference:" + msg] += 1
            if msg == "non-simple chart":
                reference_boundary_self += 1
            exact_first_failure[msg] += 1
            if len(witnesses["reference_chart:" + msg]) < 5:
                witnesses["reference_chart:" + msg].append({**base_rec, "boundary_count": len(boundary), "error": msg})
            continue
        both_charts_valid += 1
        current_poly, current_xy, current_eps = current
        ref_poly, ref_xy, ref_eps = reference_chart
        if not star.inside_polygon(current_poly[0], current_poly, current_eps):
            msg = "invalid current PCA polygon"
            exact_first_failure[msg] += 1
            counters["invalid_current_pca_polygon"] += 1
            continue
        if not star.inside_polygon(ref_poly[0], ref_poly, ref_eps):
            msg = "invalid reference PCA polygon"
            exact_first_failure[msg] += 1
            counters["invalid_reference_pca_polygon"] += 1
            continue

        interior = sorted(set(map(int, vertices)) - set(map(int, boundary)))
        current_out = [node for node in interior if not star.inside_polygon(current_xy[node], current_poly, current_eps)]
        reference_out = [node for node in interior if not star.inside_polygon(ref_xy[node], ref_poly, ref_eps)]
        outside_current_nodes += len(current_out)
        outside_reference_nodes += len(reference_out)
        if current_out or reference_out:
            counters["outside_current_interior_node"] += bool(current_out)
            counters["outside_reference_interior_node"] += bool(reference_out)
            exact_interior = set(map(int, vertices)) - set(map(int, boundary))
            for node in exact_interior:
                if not star.inside_polygon(current_xy[node], current_poly, current_eps):
                    msg = f"interior node {node} lies outside PCA boundary"
                    exact_first_failure[msg] += 1
                    break
                if not star.inside_polygon(ref_xy[node], ref_poly, ref_eps):
                    msg = f"interior node {node} lies outside PCA boundary"
                    exact_first_failure[msg] += 1
                    break
            if len(witnesses["interior_outside_pca"]) < 10:
                witnesses["interior_outside_pca"].append({
                    **base_rec, "current_outside_nodes": current_out,
                    "reference_outside_nodes": reference_out,
                })
            continue
        inside_both += 1
        counters["pre_delaunay_pca_pass"] += 1

    source_paths = [
        Path(__file__), CANDIDATE_OWNER, owner.PATCH570_PATH, owner.STAR569_PATH,
        state_path, candidate_path, report_path, Path(state["point_summary"]),
    ]
    pins = {str(p): sha(p) for p in source_paths}
    result = {
        "status": "PURE_PATCH_PREFLIGHT_DIAGNOSTIC_ONLY",
        "scope": "571 deterministic top-ranked first-N patch rows; stops before Delaunay and all quality/exact geometry/bank queries",
        "state_path": str(state_path),
        "state_sha256": sha(state_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": state["candidate_sha256"],
        "surface_report_path": str(report_path),
        "surface_report_sha256": state["surface_report_sha256"],
        "baseline_surface_pairs_from_report": expected_pairs,
        "selected_patch_count": len(selected),
        "mixed_material_patch_count": int(counters["mixed_material_precheck"]),
        "attempted_after_material_filter": attempted_after_material_filter,
        "actual_topological_disk_count": topology_disk,
        "current_and_reference_chart_valid_count": both_charts_valid,
        "interior_inside_both_pca_boundaries_count": inside_both,
        "current_boundary_self_intersection_chart_failures": current_boundary_self,
        "reference_boundary_self_intersection_chart_failures": reference_boundary_self,
        "interior_nodes_outside_current_pca_total": outside_current_nodes,
        "interior_nodes_outside_reference_pca_total": outside_reference_nodes,
        "topology_failure_messages": dict(topology_failures),
        "chart_failure_messages": dict(chart_failures),
        "exact_571_first_valueerror_counts": dict(exact_first_failure),
        "classification_counts": dict(counters),
        "witnesses_by_reason": dict(witnesses),
        "source_pins": pins,
        "elapsed_seconds": time.perf_counter() - start,
        "geometry_or_solver_executed": False,
    }
    out_path = args.output / "report.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "selected_patch_count", "actual_topological_disk_count",
        "current_and_reference_chart_valid_count",
        "interior_inside_both_pca_boundaries_count",
        "exact_571_first_valueerror_counts", "elapsed_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()

