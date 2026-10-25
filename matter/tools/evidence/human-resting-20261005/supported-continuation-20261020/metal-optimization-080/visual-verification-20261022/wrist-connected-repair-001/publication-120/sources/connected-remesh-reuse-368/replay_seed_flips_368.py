#!/usr/bin/env python3
"""Replay the exact accepted 359 same-parent flip whitelist on new 362 coordinates.

This is a topology-construction proposal only. It rechecks each old allowed flip
against the new current and matched-037 geometry with the frozen 358 operator.
It does not carry over any self/bone witness result or physical admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
H = ROOT / "barycentric-connected-clearance-170"
FLIP_MODULE = ROOT / "D/connected-remesh-reuse-358/same_parent_flips_358.py"
TARGET_ROWS = ROOT / "D/connected-remesh-reuse-349/target-face-rows-322-344.json"

PINS = {
    "input_362": "814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3",
    "seed_input_2346": "2346d174df8b363bca2200c24589a36cf1ad6a4642a4b73a9a156a48e6826b25",
    "seed_output_359": "bc98b04d1035528c82c30aa6e929cd91159c6a17677ffcf9e3690d6866f76e20",
    "seed_flip_report": "57a99f494042e4768e73b29b0348d892a8fdad7767dd016123777a9e047f0e78",
    "target_rows": "fdf71a4c8a1d4186bc8aa3541bba0b09aefabd17607cfd167c8f3b0b1be64e7d",
}
INPUT_362 = H / "coherent-support-continuation-362/uniform-scale-recovery/backtrack/pass-001/candidate.npz"
SEED_INPUT_2346 = H / "joint-boundary-continuation-322/round-003/attempt-004/global-275/backtrack/pass-002/candidate.npz"
SEED_OUTPUT_359 = H / "same-parent-remesh-359/candidate.npz"
SEED_REPORT = H / "same-parent-remesh-359/flip-proposal-report.json"

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def load_flip_module(path: Path):
    spec = importlib.util.spec_from_file_location("same_parent_flips_358", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned flip helper: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and helper modules inspect sys.modules during import.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}

def face_csr_rows(ptr: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    if ptr.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(values) or np.any(np.diff(ptr) < 0):
        raise ValueError("invalid face-ancestry CSR")
    return [values[int(ptr[i]):int(ptr[i+1])] for i in range(len(ptr)-1)]

def _check_face_row_ancestry_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    direct = (
        "face_kind", "parent_coarse_face_row", "parent_fine_detail_face_row",
        "parent_source_face_marker", "parent_fine_subface_row",
        "refinement_face_parent_face_row", "refinement_face_parent_barycentric",
        "face_source_ancestry_indptr", "face_source_ancestry_face_rows",
        "face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows",
    )
    for key in direct:
        if key not in a or key not in b:
            raise ValueError(f"input lacks required topology ancestry array {key}")
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"362 input changed seed-topology ancestry array {key}")
    for ptr_key, val_key in (
        ("face_source_ancestry_indptr", "face_source_ancestry_face_rows"),
        ("face_fine_subface_ancestry_indptr", "face_fine_subface_ancestry_rows"),
    ):
        # Validate the CSR can be read rowwise; direct equality above pins bytes too.
        face_csr_rows(a[ptr_key], a[val_key])

def read_target_rows(path: Path) -> list[int]:
    obj = json.loads(path.read_text())
    rows = obj.get("face_rows") if isinstance(obj, dict) else obj
    if not isinstance(rows, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in rows):
        raise ValueError("target rows must be exact JSON integers")
    if len(rows) != len(set(rows)):
        raise ValueError("target rows contain duplicates")
    return rows

def validate_seed(
    input_data: dict[str, np.ndarray],
    seed_input: dict[str, np.ndarray],
    seed_output: dict[str, np.ndarray],
    seed_report: dict[str, Any],
    target_rows: list[int],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    f0 = np.asarray(seed_input["source_O_faces"])
    fi = np.asarray(input_data["source_O_faces"])
    fo = np.asarray(seed_output["source_O_faces"])
    if not (f0.shape == fi.shape == fo.shape):
        raise ValueError("seed and 362 face row domains differ")
    if not np.array_equal(f0, fi):
        raise ValueError("362 topology is not the exact 2346 seed-input topology")
    if not np.array_equal(fi, input_data["area_reference_037_source_O_faces"]):
        raise ValueError("362 current and matched-037 face connectivity differ")
    if not np.array_equal(f0, seed_input["area_reference_037_source_O_faces"]):
        raise ValueError("2346 current and matched-037 face connectivity differ")
    if not np.array_equal(fo, seed_output["area_reference_037_source_O_faces"]):
        raise ValueError("359 current and matched-037 face connectivity differ")
    if not np.array_equal(input_data["area_reference_037_points_world_m_f64"],
                          seed_input["area_reference_037_points_world_m_f64"]):
        raise ValueError("362 matched-037 reference points differ from the 2346 source frame")
    if not np.array_equal(input_data["refinement_face_parent_face_row"],
                          seed_input["refinement_face_parent_face_row"]):
        raise ValueError("362 immediate-parent face rows differ from 2346")
    _check_face_row_ancestry_equal(input_data, seed_input)
    changed = set(map(int, np.flatnonzero(np.any(f0 != fo, axis=1))))
    reported_changed = set(map(int, seed_report.get("changed_face_rows", [])))
    if changed != reported_changed:
        raise ValueError("359 report changed-face list does not equal candidate topology diff")
    if target_rows != seed_report.get("seed_face_rows"):
        raise ValueError("pinned target-face rows do not match 359 report")
    ops = seed_report.get("operations")
    if (not isinstance(ops, list) or len(ops) != seed_report.get("accepted_pair_count")
            or len(ops) != 106):
        raise ValueError("359 accepted operation list/count is malformed or not 106")
    edge_map: dict[tuple[int, int], set[int]] = {}
    for row, tri in enumerate(f0):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_map.setdefault((min(int(a), int(b)), max(int(a), int(b))), set()).add(row)
    allowed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    operation_rows: list[dict[str, Any]] = []
    op_changed: set[int] = set()
    for op in ops:
        edge = op.get("edge")
        rows = op.get("face_rows")
        new_faces = op.get("new_faces")
        if not (isinstance(edge, list) and len(edge) == 2 and isinstance(rows, list) and len(rows) == 2):
            raise ValueError("malformed seed operation")
        e = tuple(sorted(map(int, edge)))
        r = tuple(map(int, rows))
        if e in seen or set(r) != edge_map.get(e, set()):
            raise ValueError("seed operation edge/face pair does not match exact source incidence")
        if not set(r).issubset(changed):
            raise ValueError("seed operation does not correspond to a changed 359 face")
        if new_faces is None or np.asarray(new_faces).shape != (2, 3):
            raise ValueError("seed operation replacement faces malformed")
        if not np.array_equal(fo[np.asarray(r, dtype=np.int64)], np.asarray(new_faces, dtype=fo.dtype)):
            raise ValueError("359 candidate replacement triangles disagree with seed operation")
        seen.add(e)
        allowed.append(e)
        op_changed.update(r)
        operation_rows.append({"edge": list(e), "face_rows": list(r)})
    if op_changed != changed or len(changed) != 212:
        raise ValueError("accepted seed operations do not exactly cover the 212 changed rows")
    return allowed, operation_rows

def build(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = {
        "input_362": INPUT_362,
        "seed_input_2346": SEED_INPUT_2346,
        "seed_output_359": SEED_OUTPUT_359,
        "seed_flip_report": SEED_REPORT,
        "target_rows": TARGET_ROWS,
        "flip_helper_358": FLIP_MODULE,
    }
    got = {k: sha_file(v) for k, v in paths.items()}
    for key, expected in PINS.items():
        if got[key] != expected:
            raise ValueError(f"{key} SHA mismatch: {got[key]} != {expected}")
    data = load_npz(INPUT_362)
    old_input = load_npz(SEED_INPUT_2346)
    old_output = load_npz(SEED_OUTPUT_359)
    seed_report = json.loads(SEED_REPORT.read_text())
    if seed_report.get("input_candidate_sha256") != got["seed_input_2346"]:
        raise ValueError("359 report source candidate pin mismatch")
    if seed_report.get("output_candidate_sha256") != got["seed_output_359"]:
        raise ValueError("359 report output candidate pin mismatch")
    target_rows = read_target_rows(TARGET_ROWS)
    allowed_edges, old_ops = validate_seed(data, old_input, old_output, seed_report, target_rows)
    flip = load_flip_module(FLIP_MODULE)
    result, proposal = flip.propose_flip_batch(
        data,
        target_rows,
        max_flips=len(allowed_edges),
        min_angle_gain_deg=1.0e-8,
        min_normal_dot=0.0,
        allowed_edges=allowed_edges,
        excluded_face_rows=(),
    )
    # A changed point field is intentional; no coordinate or non-flip ancestry
    # may be silently replaced by the 359 candidate's old values.
    if not np.array_equal(result["points_world_m"], data["points_world_m"]):
        raise ValueError("flip helper changed the 362 current points")
    if not np.array_equal(result["area_reference_037_points_world_m_f64"],
                          data["area_reference_037_points_world_m_f64"]):
        raise ValueError("flip helper changed the matched-037 reference positions")
    if not np.array_equal(result["source_O_faces"], result["area_reference_037_source_O_faces"]):
        raise ValueError("output current and matched-037 face topology diverged")
    # Every accepted edge must be in the pinned 359 whitelist and every
    # accepted face pair must be the same row pair as the old seed operation.
    allowed_by_edge = {tuple(x["edge"]): tuple(x["face_rows"]) for x in old_ops}
    for op in proposal["operations"]:
        edge = tuple(sorted(map(int, op["edge"])))
        rows = tuple(map(int, op["face_rows"]))
        if edge not in allowed_by_edge or set(rows) != set(allowed_by_edge[edge]):
            raise ValueError("operator accepted a flip outside the exact 359 operation whitelist")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate.npz"
    with candidate_path.open("xb") as f:
        np.savez_compressed(f, **result)
    proposal.update({
        "status": "REVALIDATED_359_FLIP_SUBSET_ON_362_COORDINATES",
        "geometry_admitted": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "input_362_candidate": str(INPUT_362),
        "input_362_candidate_sha256": got["input_362"],
        "seed_359_operation_count": len(allowed_edges),
        "seed_359_report": str(SEED_REPORT),
        "seed_359_report_sha256": got["seed_flip_report"],
        "seed_359_candidate_sha256": got["seed_output_359"],
        "source_2346_candidate_sha256": got["seed_input_2346"],
        "target_face_rows_json": str(TARGET_ROWS),
        "target_face_rows_sha256": got["target_rows"],
        "allowed_edge_policy": "exact edges from the 106 accepted 359 operations only; re-evaluated on 362 current and matched-037 geometry; no replacement edges",
        "accepted_operations_may_be_fewer_than_106": True,
        "point_coordinates_bitwise_preserved_from_362": True,
        "matched_037_reference_points_bitwise_preserved_from_362": True,
        "all_185_typed_bank_audits_required": True,
        "exact_full_surface_self_audit_required": True,
        "source_cap_and_matched_037_area_audits_required": True,
        "input_array_hashes": {
            "points_world_m": sha_array(data["points_world_m"]),
            "source_O_faces": sha_array(data["source_O_faces"]),
            "matched_037_points": sha_array(data["area_reference_037_points_world_m_f64"]),
            "matched_037_faces": sha_array(data["area_reference_037_source_O_faces"]),
            "face_parent_rows": sha_array(data["refinement_face_parent_face_row"]),
            "face_parent_barycentric": sha_array(data["refinement_face_parent_barycentric"]),
        },
        "output_candidate": str(candidate_path),
        "output_candidate_sha256": sha_file(candidate_path),
        "output_arrays": {
            "points_world_m": sha_array(result["points_world_m"]),
            "source_O_faces": sha_array(result["source_O_faces"]),
            "area_reference_037_source_O_faces": sha_array(result["area_reference_037_source_O_faces"]),
            "refinement_face_parent_face_row": sha_array(result["refinement_face_parent_face_row"]),
            "refinement_face_parent_barycentric": sha_array(result["refinement_face_parent_barycentric"]),
            "face_source_ancestry_indptr": sha_array(result["face_source_ancestry_indptr"]),
            "face_source_ancestry_face_rows": sha_array(result["face_source_ancestry_face_rows"]),
            "face_fine_subface_ancestry_indptr": sha_array(result["face_fine_subface_ancestry_indptr"]),
            "face_fine_subface_ancestry_rows": sha_array(result["face_fine_subface_ancestry_rows"]),
        },
    })
    report_path = output_dir / "replay-report.json"
    report_path.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proposal

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    report = build(args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "accepted_seed_flips": report["accepted_pair_count"],
        "requested_seed_flips": report["seed_359_operation_count"],
        "changed_rows": len(report["changed_face_rows"]),
        "candidate": report["output_candidate"],
        "candidate_sha256": report["output_candidate_sha256"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()

