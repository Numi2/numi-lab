#!/usr/bin/env python3
"""Source-bound CPU check of the accepted-correction coarse-mode recipe.

This checks the vector algebra proposed by the Metal overlay against the
captured source-079 mesh/P and the N63 FGMRES correction control. It does not
run the native kernels and does not establish a solver-performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


DEFAULT_ROOT = Path("/Users/home/cardboard-implementation-20261014")
DEFAULT_RUN = Path("solver/sourcefit15-fg32-control1-newton63-capture-266/output")
DEFAULT_AUDIT = Path("analysis/control1-newton63-cached-operator-266")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def orthogonalize(candidate: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Match selector's two-pass modified Gram-Schmidt and 1e-10 norm gate."""
    vector = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
    original_norm2 = float(np.dot(vector, vector))
    if not np.isfinite(original_norm2) or original_norm2 <= 1.0e-24:
        return np.zeros_like(vector), 0.0, False
    for _ in range(2):
        for row in basis:
            projection = float(np.dot(row, vector))
            if not math.isfinite(projection):
                return np.zeros_like(vector), 0.0, False
            vector -= projection * row
    projected_norm2 = float(np.dot(vector, vector))
    if (not np.isfinite(projected_norm2) or
            projected_norm2 <= original_norm2 * 1.0e-10):
        return np.zeros_like(vector), 0.0, False
    projected_norm = math.sqrt(projected_norm2)
    if not math.isfinite(projected_norm) or projected_norm <= 0.0:
        return np.zeros_like(vector), 0.0, False
    return vector / projected_norm, projected_norm / math.sqrt(original_norm2), True


def orthonormalize_projected_basis(
    basis: np.ndarray,
    *,
    dependent_squared_ratio: float = 1.0e-10,
) -> tuple[np.ndarray, bool, int, bool]:
    """Two-pass MGS of candidate-masked basis rows.

    Exact zero/dependent rows are skipped. A finite nonzero row whose residual
    falls below the rank threshold is ambiguous in FP32 and rejects the whole
    recycle attempt instead of silently dropping a direction.
    """
    rows = np.asarray(basis, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        return np.zeros_like(rows), False, 0, False
    work = np.zeros_like(rows)
    rank = 0
    for index, source in enumerate(rows):
        raw_norm2 = float(np.dot(source, source))
        if not math.isfinite(raw_norm2):
            return np.zeros_like(rows), False, rank, False
        if raw_norm2 == 0.0:
            continue
        residual = source.copy()
        for _ in range(2):
            for prior in range(index):
                q = work[prior]
                projection = float(np.dot(q, residual))
                if not math.isfinite(projection):
                    return np.zeros_like(rows), False, rank, False
                residual -= projection * q
        residual_norm2 = float(np.dot(residual, residual))
        if not math.isfinite(residual_norm2):
            return np.zeros_like(rows), False, rank, False
        if residual_norm2 == 0.0:
            continue
        if residual_norm2 <= raw_norm2 * dependent_squared_ratio:
            return np.zeros_like(rows), False, rank, True
        work[index] = residual / math.sqrt(residual_norm2)
        rank += 1
    return work, True, rank, False


def orthogonalize_masked_candidate(
    candidate: np.ndarray,
    effective_basis: np.ndarray,
    *,
    dependent_squared_ratio: float = 1.0e-10,
    source_orthogonality_tolerance: float = 1.0e-4,
) -> tuple[np.ndarray, float, bool, dict]:
    """Match the guarded Metal selector after dynamic constraint projection."""
    vector = np.asarray(candidate, dtype=np.float64).reshape(-1).copy()
    rows = np.asarray(effective_basis, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != vector.size or not np.isfinite(vector).all():
        return np.zeros_like(vector), 0.0, False, {"reason": "invalid_input"}
    original_norm2 = float(np.dot(vector, vector))
    if not math.isfinite(original_norm2) or original_norm2 <= 1.0e-24:
        return np.zeros_like(vector), 0.0, False, {"reason": "zero_or_nonfinite_candidate"}
    orthogonal, valid, rank, ambiguous = orthonormalize_projected_basis(
        rows, dependent_squared_ratio=dependent_squared_ratio)
    if not valid:
        reason = "rank_ambiguous" if ambiguous else "invalid_projected_basis"
        return np.zeros_like(vector), 0.0, False, {"reason": reason, "rank": rank}
    for _ in range(2):
        for row in orthogonal:
            projection = float(np.dot(row, vector))
            if not math.isfinite(projection):
                return np.zeros_like(vector), 0.0, False, {"reason": "nonfinite_projection", "rank": rank}
            vector -= projection * row
    projected_norm2 = float(np.dot(vector, vector))
    if (not math.isfinite(projected_norm2) or
            projected_norm2 <= original_norm2 * 1.0e-10):
        return np.zeros_like(vector), 0.0, False, {"reason": "dependent_candidate", "rank": rank}
    projected_norm = math.sqrt(projected_norm2)
    if not math.isfinite(projected_norm) or projected_norm <= 0.0:
        return np.zeros_like(vector), 0.0, False, {"reason": "invalid_projected_norm", "rank": rank}
    vector /= projected_norm
    correlations = []
    for row in rows:
        row_norm2 = float(np.dot(row, row))
        if row_norm2 == 0.0:
            correlations.append(0.0)
            continue
        corr = abs(float(np.dot(row, vector))) / math.sqrt(row_norm2)
        correlations.append(corr)
    max_corr = max(correlations, default=0.0)
    if (not math.isfinite(max_corr) or
            max_corr > source_orthogonality_tolerance):
        return np.zeros_like(vector), projected_norm / math.sqrt(original_norm2), False, {
            "reason": "residual_not_orthogonal_to_original_rows",
            "rank": rank,
            "max_normalized_source_correlation": max_corr,
        }
    return vector, projected_norm / math.sqrt(original_norm2), True, {
        "reason": "accepted",
        "rank": rank,
        "max_normalized_source_correlation": max_corr,
    }


def make_y4_fallback(mesh: dict) -> np.ndarray:
    coordinates = np.asarray(mesh["nodes_m"], dtype=np.float64)
    fixed = set(int(node) for node in mesh["fixed_nodes"])
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum)
    order = sorted(range(3), key=lambda axis: -half[axis])
    x_axis, y_axis = order[:2]
    normal_axis = order[2]
    vector = np.zeros((coordinates.shape[0], 3), dtype=np.float64)
    for node, point in enumerate(coordinates):
        if node in fixed:
            continue
        x = (point[x_axis] - center[x_axis]) / half[x_axis]
        y = (point[y_axis] - center[y_axis]) / half[y_axis]
        z = point[normal_axis] - center[normal_axis]
        vector[node, normal_axis] = y ** 4
        vector[node, y_axis] = -z * (4.0 * y ** 3 / half[y_axis])
    return vector.reshape(-1)


def run(root: Path) -> dict:
    run_dir = root / DEFAULT_RUN
    audit_dir = root / DEFAULT_AUDIT
    condition_dir = run_dir / "rejected-conditioning-000002"
    mesh_path = run_dir / "mesh.json"
    p_path = condition_dir / "P.bin"
    correction_path = audit_dir / "offline-direct-correction.npz"
    vectors_path = audit_dir / "report.vectors.npz"
    manifest_path = condition_dir / "manifest.json"

    mesh = json.loads(mesh_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "numi.cardboard.regional-conditioning.v1":
        raise ValueError("unexpected source capture manifest schema")
    nodes = len(mesh["nodes_m"])
    modes = int(manifest["mode_count"])
    if nodes != int(manifest["fem_node_count"]):
        raise ValueError("mesh and capture manifest node counts differ")
    if modes != 26 or nodes != 882:
        raise ValueError(f"unexpected source-079 study dimensions: {modes} modes, {nodes} nodes")

    p = np.fromfile(p_path, dtype="<f4")
    expected = modes * nodes * 4
    if p.size != expected:
        raise ValueError(f"P.bin size mismatch: {p.size} float32 words, expected {expected}")
    p3 = p.reshape(modes, nodes, 4)[..., :3].astype(np.float64).reshape(modes, -1)
    if not np.isfinite(p3).all():
        raise ValueError("captured source basis contains nonfinite entries")

    fallback = make_y4_fallback(mesh)
    fallback_norm = float(np.linalg.norm(fallback))
    fallback_projected, fallback_ratio, fallback_independent = orthogonalize(
        fallback, p3)
    if not fallback_independent:
        raise ValueError("static y^4 fallback is not independent on captured source mesh")

    correction = np.load(correction_path)
    vectors = np.load(vectors_path)
    free_nodes = np.asarray(vectors["free_node_ids"], dtype=np.int64)
    raw_correction = np.asarray(correction["native"], dtype=np.float64)
    if raw_correction.size != free_nodes.size * 3:
        raise ValueError("captured N63 correction does not match free-node map")
    full_correction = np.zeros((nodes, 3), dtype=np.float64)
    full_correction[free_nodes] = raw_correction.reshape(-1, 3)
    fixed_nodes = set(int(node) for node in mesh["fixed_nodes"])
    if any(np.any(full_correction[node] != 0.0) for node in fixed_nodes):
        raise ValueError("mapped correction is nonzero on source fixed nodes")
    correction_norm = float(np.linalg.norm(full_correction))
    selected, selected_ratio, correction_independent = orthogonalize(
        full_correction, p3)
    if not correction_independent:
        raise ValueError("captured correction unexpectedly fails the MGS independence gate")
    max_orthogonality = max(
        abs(math.fsum(float(left) * float(right)
                      for left, right in zip(row, selected)))
        for row in p3)
    if max_orthogonality > 2.0e-7:
        raise ValueError(f"CPU MGS residual is too large: {max_orthogonality}")

    # Exercise every selector fallback predicate with stable small vectors.
    identity = np.eye(4, dtype=np.float64)[:3]
    _, _, zero_accepted = orthogonalize(np.zeros(4), identity)
    _, _, dependent_accepted = orthogonalize(np.array([1.0, 0.0, 0.0, 0.0]), identity)
    _, _, nonfinite_accepted = orthogonalize(np.array([np.nan, 0.0, 0.0, 0.0]), identity)
    independent_vec, _, independent_accepted = orthogonalize(
        np.array([1.0, 2.0, 3.0, 1.0]), identity)
    if zero_accepted or dependent_accepted or nonfinite_accepted or not independent_accepted:
        raise ValueError("CPU selector gate control cases did not match expected branches")
    if not np.isfinite(independent_vec).all():
        raise ValueError("independent control yielded nonfinite normalized mode")

    # Dynamic fixed-node masking can destroy the authored basis orthogonality.
    # Exercise the selector's reorthogonalized workspace and fail-closed rank
    # handling against an independently computed least-squares residual.
    masked_basis = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0, 1.0],
    ], dtype=np.float64)
    masked_basis[0] /= np.linalg.norm(masked_basis[0])
    masked_basis[1] /= np.linalg.norm(masked_basis[1])
    masked_basis[:, 1] = 0.0
    masked_dot = float(np.dot(masked_basis[0], masked_basis[1]))
    if abs(masked_dot) < 0.1:
        raise ValueError("dynamic-mask fixture did not produce a nonorthogonal basis")
    probe = np.array([0.2, -0.1, 0.7, 0.3], dtype=np.float64)
    projected, projected_ratio, projected_ok, projection_diag = \
        orthogonalize_masked_candidate(probe, masked_basis)
    q, _ = np.linalg.qr(masked_basis.T, mode="reduced")
    oracle_residual = probe - q @ (q.T @ probe)
    oracle_norm = float(np.linalg.norm(oracle_residual))
    if not projected_ok or not np.isfinite(projected).all():
        raise ValueError("dynamic-mask independent candidate failed the guarded MGS fixture")
    if np.linalg.norm(projected - oracle_residual / oracle_norm) > 2.0e-12:
        raise ValueError("guarded MGS differs from independent FP64 QR projection")
    dependent_candidate = masked_basis[0] + 3.0 * masked_basis[1]
    _, _, dependent_ok, dependent_diag = orthogonalize_masked_candidate(
        dependent_candidate, masked_basis)
    ambiguous_basis = np.array([[1.0, 0.0, 0.0], [1.0, 1.0e-6, 0.0]])
    _, _, ambiguous_ok, ambiguous_diag = orthogonalize_masked_candidate(
        np.array([0.0, 1.0, 0.0]), ambiguous_basis)
    if dependent_ok or ambiguous_ok or ambiguous_diag.get("reason") != "rank_ambiguous":
        raise ValueError("rank-deficient or rank-ambiguous candidate did not fail closed")
    # Environment-major workspace arithmetic: each environment receives a
    # disjoint M*N range, including the second environment in this fixture.
    environment_count, mode_count, node_count = 2, 3, 5
    workspace_size = environment_count * mode_count * node_count
    env0 = set(range(0, mode_count * node_count))
    env1 = set(range(mode_count * node_count, workspace_size))
    if env0 & env1 or len(env0) != mode_count * node_count or len(env1) != mode_count * node_count:
        raise ValueError("environment-major orthogonal scratch ranges overlap")

    return {
        "schema": "numi.cardboard.accepted-correction-recycle-cpu-check.v1",
        "verdict": "CPU vector-algebra controls pass; no native kernel or solver-performance qualification",
        "source_capture": {
            "root": str(root),
            "run_manifest_sha256": sha256(run_dir / "manifest.json"),
            "regional_manifest_sha256": sha256(manifest_path),
            "P_sha256": sha256(p_path),
            "mesh_sha256": sha256(mesh_path),
            "correction_npz_sha256": sha256(correction_path),
            "vectors_npz_sha256": sha256(vectors_path),
            "environment_count": int(manifest["environment_count"]),
            "captured_mode_count": modes,
            "node_count": nodes,
            "fixed_node_count": len(fixed_nodes),
            "interpretation": "N63 native FGMRES correction is an offline same-Newton control, not a captured preceding-Newton line-search increment",
        },
        "static_fallback": {
            "name": "source-basis orthogonalized bending y^4",
            "raw_norm": fallback_norm,
            "independent_projected_norm_ratio": fallback_ratio,
            "independent": fallback_independent,
        },
        "captured_correction_control": {
            "raw_norm": correction_norm,
            "projected_norm_ratio": selected_ratio,
            "independent": correction_independent,
            "max_abs_P_dot_selected": max_orthogonality,
            "is_previous_newton_increment": False,
        },
        "dynamic_masked_basis_controls": {
            "fixture_basis_dot_after_mask": masked_dot,
            "candidate_projected_norm_ratio": projected_ratio,
            "candidate_accepted": projected_ok,
            "candidate_max_normalized_source_correlation": projection_diag.get(
                "max_normalized_source_correlation"),
            "oracle_fp64_qr_direction_error": float(np.linalg.norm(
                projected - oracle_residual / oracle_norm)),
            "dependent_candidate_rejected": not dependent_ok,
            "dependent_candidate_reason": dependent_diag.get("reason"),
            "near_rank_ambiguous_basis_rejected": not ambiguous_ok,
            "near_rank_ambiguous_reason": ambiguous_diag.get("reason"),
            "environment_count": environment_count,
            "base_mode_count": mode_count,
            "node_count": node_count,
            "workspace_float4_count": workspace_size,
            "workspace_environment_ranges_disjoint": True,
            "interpretation": "synthetic CPU algebra and indexing controls; no Metal or solver qualification",
        },
        "predicate_controls": {
            "zero_rejected_to_fallback": not zero_accepted,
            "dependent_rejected_to_fallback": not dependent_accepted,
            "nonfinite_rejected_to_fallback": not nonfinite_accepted,
            "independent_candidate_accepted": independent_accepted,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.root)
    except Exception as exc:
        print(json.dumps({"schema": "numi.cardboard.accepted-correction-recycle-cpu-check.v1",
                          "verdict": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
