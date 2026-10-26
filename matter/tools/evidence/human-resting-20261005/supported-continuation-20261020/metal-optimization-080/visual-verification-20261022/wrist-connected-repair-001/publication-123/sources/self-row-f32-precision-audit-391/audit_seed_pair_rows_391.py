#!/usr/bin/env python3
"""Build one pinned 026 seed-row payload and audit its F32 margin sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

from row_f32_precision_audit import audit_pair_rows_f32_precision

ROOT = pathlib.Path("/Users/n/Work/numi-human-connected-surface-069")
PINNED_OWNER = ROOT / "barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
PINNED_OWNER_SHA256 = "c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3"
EXPECTED_CANDIDATE_SHA256 = "814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3"
EXPECTED_PAIRS_SHA256 = "6ff1d81071cb36b81969fcb03ba94e539915ba2e4ed0c6457e99f6e9e790bbe2"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_owner(path: pathlib.Path, expected_sha256: str):
    got = sha256_file(path)
    if got != expected_sha256:
        raise ValueError(f"026 owner SHA mismatch: {got}")
    spec = importlib.util.spec_from_file_location("pinned_self_separation_026", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned 026 owner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_pair_rows", None)):
        raise ValueError("pinned 026 owner lacks build_pair_rows")
    return module


def run(candidate_npz: pathlib.Path, pairs_npy: pathlib.Path, output_dir: pathlib.Path,
        *, owner_path: pathlib.Path = PINNED_OWNER, expected_candidate_sha256: str = EXPECTED_CANDIDATE_SHA256,
        expected_pairs_sha256: str = EXPECTED_PAIRS_SHA256, face_id: int | None = None,
        margin_fraction: float = 0.25, interval_radius_mm: float = 0.0,
        batch_size: int = 512) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    candidate_hash = sha256_file(candidate_npz)
    pairs_hash = sha256_file(pairs_npy)
    if expected_candidate_sha256 and candidate_hash != expected_candidate_sha256:
        raise ValueError(f"candidate SHA mismatch: {candidate_hash}")
    if expected_pairs_sha256 and pairs_hash != expected_pairs_sha256:
        raise ValueError(f"pair seed SHA mismatch: {pairs_hash}")
    owner = load_owner(owner_path, PINNED_OWNER_SHA256)

    with np.load(candidate_npz, allow_pickle=False) as z:
        if "points_world_m" not in z or "source_O_faces" not in z:
            raise ValueError("candidate lacks points_world_m/source_O_faces")
        points = np.asarray(z["points_world_m"])
        faces = np.asarray(z["source_O_faces"])
        if points.dtype != np.dtype(np.float32):
            raise ValueError("candidate points_world_m must remain exact float32")
        if faces.dtype.kind not in "iu":
            raise ValueError("candidate source_O_faces must be integer")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("candidate points have invalid shape/nonfinite values")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            raise ValueError("candidate faces have invalid shape")
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("candidate face index outside point table")
        faces = faces.astype(np.int64, copy=False)

    pairs = np.load(pairs_npy, allow_pickle=False)
    if pairs.dtype.kind not in "iu" or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair seed must be integer (P,2)")
    pairs = pairs.astype(np.int64, copy=False)
    pairs = np.sort(pairs, axis=1)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("pair seed contains self-pairs")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("pair seed contains duplicate unordered pairs")
    input_pair_count = int(len(pairs))
    if face_id is not None:
        pairs = pairs[np.any(pairs == int(face_id), axis=1)]
        if len(pairs) == 0:
            raise ValueError(f"no seed pair touches requested face {face_id}")

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    points_mm = points.astype(np.float64) * 1000.0
    rows = []
    records = []
    for start in range(0, len(pairs), int(batch_size)):
        stop = min(start + int(batch_size), len(pairs))
        batch_rows, batch_records = owner.build_pair_rows(
            points_mm, faces, pairs[start:stop], margin_fraction=margin_fraction)
        rows.extend(batch_rows)
        records.extend(batch_records)
        print(json.dumps({"pairs_built": stop, "pair_total": len(pairs),
                          "rows_built": len(rows)}, sort_keys=True), flush=True)
    if len(records) != len(pairs):
        raise ValueError(f"026 owner returned {len(records)} records for {len(pairs)} pairs")
    audit = audit_pair_rows_f32_precision(points, rows, records, interval_radius_mm=interval_radius_mm)

    output_dir.mkdir(parents=True, exist_ok=False)
    indptr = [0]
    node_rows = []
    coeff_rows = []
    for row in rows:
        node_rows.extend(np.asarray(row["nodes"], dtype=np.int64).tolist())
        coeff_rows.extend(np.asarray(row["coefficients"], dtype=np.float64).tolist())
        indptr.append(len(node_rows))
    row_path = output_dir / "merged-row-payload.npz"
    np.savez_compressed(
        row_path,
        row_indptr=np.asarray(indptr, dtype=np.int64),
        node_rows=np.asarray(node_rows, dtype=np.int64),
        coefficients_mm=np.asarray(coeff_rows, dtype=np.float64).reshape((-1, 3)),
        lower_mm=np.asarray([float(r["lower"]) for r in rows], dtype=np.float64),
        labels=np.asarray([str(r["label"]) for r in rows], dtype="U64"),
        pair_faces=np.asarray([r["faces"] for r in records], dtype=np.int64).reshape((-1, 2)),
        pair_kind=np.asarray([r["kind"] for r in records], dtype="U32"),
    )
    report = {
        "schema": "self_row_f32_precision_audit_391.v1",
        "diagnostic_only": True,
        "candidate_path": str(candidate_npz),
        "candidate_sha256": candidate_hash,
        "candidate_points_sha256": sha256_array(points),
        "candidate_faces_sha256": sha256_array(faces),
        "points_shape": list(points.shape),
        "faces_shape": list(faces.shape),
        "points_dtype": str(points.dtype),
        "seed_pairs_path": str(pairs_npy),
        "seed_pairs_sha256": pairs_hash,
        "seed_pair_count_total": input_pair_count,
        "selected_pair_count": int(len(pairs)),
        "face_filter": face_id,
        "margin_fraction": float(margin_fraction),
        "interval_radius_mm": float(interval_radius_mm),
        "pair_build_batch_size": int(batch_size),
        "owner_path": str(owner_path),
        "owner_sha256": sha256_file(owner_path),
        "audit_helper_path": str(pathlib.Path(__file__).with_name("row_f32_precision_audit.py")),
        "audit_helper_sha256": sha256_file(pathlib.Path(__file__).with_name("row_f32_precision_audit.py")),
        "rows_payload_path": str(row_path),
        "rows_payload_sha256": sha256_file(row_path),
        **audit,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, type=pathlib.Path)
    ap.add_argument("--pairs", required=True, type=pathlib.Path)
    ap.add_argument("--output-dir", required=True, type=pathlib.Path)
    ap.add_argument("--face-id", type=int, help="optional deterministic witness subset touching this face")
    ap.add_argument("--margin-fraction", type=float, default=0.25)
    ap.add_argument("--interval-radius-mm", type=float, default=0.0,
                    help="optional axiswise coordinate-radius sensitivity bound; not a geometry trust guarantee")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="pair rows per pinned-owner batch; deterministic, controls progress granularity only")
    ap.add_argument("--expected-candidate-sha256", default=EXPECTED_CANDIDATE_SHA256)
    ap.add_argument("--expected-pairs-sha256", default=EXPECTED_PAIRS_SHA256)
    ns = ap.parse_args(argv)
    report = run(ns.candidate, ns.pairs, ns.output_dir,
                 expected_candidate_sha256=ns.expected_candidate_sha256,
                 expected_pairs_sha256=ns.expected_pairs_sha256,
                 face_id=ns.face_id, margin_fraction=ns.margin_fraction,
                 interval_radius_mm=ns.interval_radius_mm, batch_size=ns.batch_size)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
