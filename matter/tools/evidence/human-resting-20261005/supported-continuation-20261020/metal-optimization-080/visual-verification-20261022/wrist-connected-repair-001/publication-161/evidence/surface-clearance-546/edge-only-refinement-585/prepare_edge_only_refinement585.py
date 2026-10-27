#!/usr/bin/env python3
"""Pinned offline candidate preparation for the 585 edge-only refinement."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from edge_only_refinement585 import build_refined_arrays585, sha256_file


def bytes_sha256(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def _edge_key(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def choose_high_incidence_edges585(faces: np.ndarray, crossing_report: dict,
                                   max_edges: int | None):
    """Select a bounded pilot by crossed-face pair multiplicity, not edge proof.

    Scores are construction-selection priorities only: each edge inherits the
    number of exact bank-triangle witnesses attached to its incident crossed
    skin faces. This does not claim the geometric edge itself was intersected.
    """
    F = np.asarray(faces)
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu":
        raise ValueError("faces must be an integer Mx3 array")
    if crossing_report.get("complete") is not True or crossing_report.get("geometry_or_native_admission") is not False:
        raise ValueError("crossing report must be complete and diagnostic-only")
    if max_edges is not None and max_edges <= 0:
        raise ValueError("max_edges must be positive or omitted")
    face_pair_count = defaultdict(int)
    total_pairs = 0
    for bank in crossing_report.get("banks", []):
        result = bank.get("result")
        if result is None:
            continue
        pairs = result.get("triangle_pairs") or []
        if int(result.get("count", -1)) != len(pairs):
            raise ValueError("bank hit count does not match pair rows")
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("malformed surface-pair row")
            face = int(pair[0])
            if face < 0 or face >= len(F):
                raise ValueError("crossed skin face row is out of range")
            face_pair_count[face] += 1
            total_pairs += 1
    if not face_pair_count:
        raise ValueError("crossing report has no crossed skin faces")
    scores = defaultdict(int)
    crossed_face_incidence = defaultdict(int)
    for face, pair_count in face_pair_count.items():
        tri = F[face]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = _edge_key(a, b)
            scores[edge] += pair_count
            crossed_face_incidence[edge] += 1
    ranked = sorted(scores, key=lambda edge: (-scores[edge], -crossed_face_incidence[edge], edge))
    selected = ranked if max_edges is None else ranked[:max_edges]
    if not selected:
        raise ValueError("edge selection is empty")
    selection = {
        "mode": "all_edges_of_validated_crossed_faces" if max_edges is None else "bounded_high_pair_weighted_crossed_face_edge_incidence",
        "ranking_basis": "sum of exact bank triangle-pair witness counts on incident crossed skin faces; not a claim of edge-triangle intersection",
        "max_edges": max_edges,
        "candidate_edge_count": int(len(ranked)),
        "selected_edge_count": int(len(selected)),
        "excluded_candidate_edge_count": int(len(ranked) - len(selected)),
        "crossing_pair_count": int(total_pairs),
        "crossed_skin_face_count": int(len(face_pair_count)),
        "selected_edges_ranked": [
            {"edge": list(edge), "pair_weighted_face_incidence": int(scores[edge]),
             "crossed_face_incidence": int(crossed_face_incidence[edge])}
            for edge in selected
        ],
        "unselected_edge_rows_remain_unresolved": bool(len(selected) < len(ranked)),
    }
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2), selection


def prepare(candidate_path: Path, face_report_path: Path, output_candidate: Path,
            output_report: Path, *, expected_candidate_sha256: str,
            expected_face_report_sha256: str, expected_area_reference_sha256: str,
            max_edges: int | None = None):
    for path in (candidate_path, face_report_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    candidate_sha = sha256_file(candidate_path)
    face_sha = sha256_file(face_report_path)
    if candidate_sha != expected_candidate_sha256:
        raise ValueError(f"candidate pin mismatch: {candidate_sha}")
    if face_sha != expected_face_report_sha256:
        raise ValueError(f"face report pin mismatch: {face_sha}")
    output_candidate, output_report = Path(output_candidate), Path(output_report)
    if output_candidate.exists() or output_report.exists():
        raise FileExistsError("refuse to overwrite an existing 585 output")
    with np.load(candidate_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    crossing_report = json.loads(Path(face_report_path).read_text())
    if crossing_report.get("candidate_sha256") != candidate_sha:
        raise ValueError("crossing report is not bound to the candidate file")
    if "area_reference_037_points_world_m_f64" not in source or "area_reference_037_source_O_faces" not in source:
        raise ValueError("candidate lacks its matched current-topology 037 reference arrays")
    selected_edges, selection = choose_high_incidence_edges585(
        source["source_O_faces"], crossing_report, max_edges)
    arrays, build_report = build_refined_arrays585(
        source, crossing_report,
        source["area_reference_037_points_world_m_f64"],
        source["area_reference_037_source_O_faces"],
        candidate_sha256=candidate_sha,
        crossing_report_sha256=face_sha,
        area_reference_sha256=expected_area_reference_sha256,
        selected_edges_override=selected_edges,
        selection_policy=selection,
    )
    parent_count = len(source["points_world_m"])
    midpoint_rows = np.arange(parent_count, len(arrays["points_world_m"]), dtype="<i8")
    if not np.array_equal(np.asarray(arrays["refinement_node_kind"])[midpoint_rows], np.ones(len(midpoint_rows), dtype=np.int8)):
        raise AssertionError("new point suffix is not exclusively edge midpoint rows")
    edge_pairs = np.asarray(arrays["refinement_node_parent_edge_rows"])[midpoint_rows]
    if edge_pairs.shape != (len(midpoint_rows), 2) or np.any(edge_pairs < 0):
        raise AssertionError("midpoint node suffix lacks exact parent-edge identities")
    output_candidate.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_candidate.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    out_sha = sha256_file(output_candidate)
    report = dict(build_report)
    report.update({
        "input_candidate_path": str(candidate_path),
        "input_candidate_sha256": candidate_sha,
        "crossing_report_path": str(face_report_path),
        "crossing_report_sha256": face_sha,
        "area_reference_sha256": expected_area_reference_sha256,
        "output_candidate_path": str(output_candidate),
        "output_candidate_sha256": out_sha,
        "parent_prefix_node_count": int(parent_count),
        "input_points": int(parent_count),
        "input_points": int(parent_count),
        "parent_prefix_points_bytes_sha256": bytes_sha256(source["points_world_m"]),
        "edge_midpoint_rows": midpoint_rows.tolist(),
        "edge_midpoint_rows_count": int(len(midpoint_rows)),
        "edge_midpoint_rows_sha256": bytes_sha256(midpoint_rows),
        "edge_midpoint_rows_sha256_encoding": "sha256 of C-contiguous little-endian int64 row IDs as raw bytes",
        "edge_midpoint_parent_edges_sha256": bytes_sha256(edge_pairs),
        "added_nodes_are_only_edge_midpoints": True,
        "added_face_centroids": 0,
        "geometry_moved": False,
        "geometry_admitted": False,
        "containment_or_clearance_claim": False,
        "physical_steps": 0,
    })
    with output_report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--crossing-report", type=Path, required=True)
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-crossing-report-sha256", required=True)
    parser.add_argument("--expected-area-reference-sha256", required=True)
    parser.add_argument("--max-edges", type=int, default=None,
                        help="optional bounded pilot; omitted means all candidate edges")
    args = parser.parse_args()
    result = prepare(
        args.candidate, args.crossing_report, args.output_candidate, args.output_report,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_face_report_sha256=args.expected_crossing_report_sha256,
        expected_area_reference_sha256=args.expected_area_reference_sha256,
        max_edges=args.max_edges,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
