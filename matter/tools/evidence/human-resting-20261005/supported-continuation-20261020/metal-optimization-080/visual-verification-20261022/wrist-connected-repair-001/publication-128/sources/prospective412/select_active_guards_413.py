from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _pair_array(value, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with shape (N,2)")
    if a.size and (np.any(a < 0) or np.any(a[:, 0] >= a[:, 1])):
        raise ValueError(f"{name} must contain canonical nonnegative pairs (a < b)")
    return np.asarray(a, dtype=np.int64)


def _pair_keys(pairs: np.ndarray) -> np.ndarray:
    pairs = _pair_array(pairs, "pairs")
    if pairs.size and np.any(pairs > np.iinfo(np.uint32).max):
        raise ValueError("pair IDs exceed the stable packed-key domain")
    p = pairs.astype(np.uint64, copy=False)
    return (p[:, 0] << np.uint64(32)) | p[:, 1]


def select_threatened_pairs(
    *,
    row_indptr,
    row_node_ids,
    row_coefficients,
    row_lower_mm,
    row_pair_ids,
    predicted_delta_mm,
    mandatory_pair_ids,
    activity_margin_mm: float = 0.0,
    max_node_step_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select whole 026 pair groups whose linear rows the predicted step threatens.

    The prediction is evaluated from the selected guard-base state. Each node's
    proposal vector is radially clipped to the stated trust radius for this
    selector only. Mandatory historical/failed pair IDs are always retained;
    all other input pairs are either selected or explicitly returned deferred.
    This is a construction-constraint selector, not a collision certificate.
    """
    ptr = np.asarray(row_indptr)
    nodes = np.asarray(row_node_ids)
    coeff = np.asarray(row_coefficients)
    lower = np.asarray(row_lower_mm)
    row_pairs = _pair_array(row_pair_ids, "row_pair_ids")
    delta = np.asarray(predicted_delta_mm, dtype=np.float64)
    mandatory = _pair_array(mandatory_pair_ids, "mandatory_pair_ids")

    if ptr.ndim != 1 or not np.issubdtype(ptr.dtype, np.integer) or len(ptr) != len(lower) + 1:
        raise ValueError("row_indptr must be an integer vector with row_count+1 entries")
    if len(ptr) == 0 or ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(nodes):
        raise ValueError("row_indptr is not a valid CSR pointer")
    if nodes.ndim != 1 or not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("row_node_ids must be an integer vector")
    if coeff.shape != (len(nodes), 3) or not np.all(np.isfinite(coeff)):
        raise ValueError("row_coefficients must be finite with shape (nnz,3)")
    if lower.shape != (len(row_pairs),) or not np.all(np.isfinite(lower)):
        raise ValueError("row_lower_mm must be finite with shape (row_count,)")
    if delta.ndim != 2 or delta.shape[1] != 3 or not np.all(np.isfinite(delta)):
        raise ValueError("predicted_delta_mm must be a finite (node_count,3) array")
    if len(nodes) and (np.any(nodes < 0) or np.any(nodes >= len(delta))):
        raise ValueError("row node index is outside the predicted displacement array")
    if not np.isfinite(activity_margin_mm) or activity_margin_mm < 0:
        raise ValueError("activity_margin_mm must be finite and nonnegative")
    if not np.isfinite(max_node_step_mm) or max_node_step_mm <= 0:
        raise ValueError("max_node_step_mm must be finite and positive")
    lengths = np.diff(ptr).astype(np.int64, copy=False)
    if len(lengths) and np.any(lengths == 0):
        raise ValueError("empty sparse guard rows are not supported")

    mandatory = np.unique(mandatory, axis=0)
    all_pairs = np.unique(row_pairs, axis=0)
    all_keys = _pair_keys(all_pairs)
    mandatory_keys = _pair_keys(mandatory)
    if len(mandatory_keys) and not np.all(np.isin(mandatory_keys, all_keys, assume_unique=True)):
        missing = mandatory[~np.isin(mandatory_keys, all_keys, assume_unique=True)]
        raise ValueError(f"mandatory pair IDs are missing from row payload: {missing[:8].tolist()}")

    norms = np.linalg.norm(delta, axis=1)
    clipped_delta = delta.copy()
    clip = norms > max_node_step_mm
    if np.any(clip):
        clipped_delta[clip] *= (max_node_step_mm / norms[clip])[:, None]

    if len(row_pairs):
        term = np.einsum("ij,ij->i", coeff, clipped_delta[nodes], optimize=True)
        row_motion = np.add.reduceat(term, ptr[:-1].astype(np.int64, copy=False))
    else:
        row_motion = np.empty(0, dtype=np.float64)
    row_slack = row_motion - lower
    threatened_rows = row_slack < activity_margin_mm
    threatened_pairs = np.unique(row_pairs[threatened_rows], axis=0)
    selected_pairs = np.unique(np.concatenate((mandatory, threatened_pairs), axis=0), axis=0)
    selected_keys = _pair_keys(selected_pairs)
    selected_rows_mask = np.isin(_pair_keys(row_pairs), selected_keys, assume_unique=False)
    selected_rows = np.flatnonzero(selected_rows_mask).astype(np.int64)
    deferred_pairs = all_pairs[~np.isin(all_keys, selected_keys, assume_unique=False)]
    report = {
        "status": "COMPLETE_ACTIVE_SET_SELECTION",
        "semantics": "Predicted linearized 026-row threats plus mandatory IDs; deferred prospective IDs remain explicit. This is not an exact geometry or physical clearance certificate.",
        "row_count": int(len(row_pairs)),
        "unique_input_pair_count": int(len(all_pairs)),
        "mandatory_pair_count": int(len(mandatory)),
        "threatened_row_count": int(np.count_nonzero(threatened_rows)),
        "threatened_pair_count": int(len(threatened_pairs)),
        "selected_pair_count": int(len(selected_pairs)),
        "selected_row_count": int(len(selected_rows)),
        "deferred_pair_count": int(len(deferred_pairs)),
        "activity_margin_mm": float(activity_margin_mm),
        "max_node_step_mm": float(max_node_step_mm),
        "predicted_max_node_step_before_clip_mm": float(norms.max(initial=0.0)),
        "predicted_clipped_node_count": int(np.count_nonzero(clip)),
        "predicted_max_node_step_after_clip_mm": float(np.linalg.norm(clipped_delta, axis=1).max(initial=0.0)),
        "minimum_predicted_row_slack_mm": float(row_slack.min(initial=float("inf"))),
        "maximum_predicted_row_violation_mm": float(max(0.0, -row_slack.min(initial=0.0))),
        "mandatory_pair_ids_sha256": hashlib.sha256(mandatory.tobytes()).hexdigest(),
        "selected_pair_ids_sha256": hashlib.sha256(selected_pairs.tobytes()).hexdigest(),
        "deferred_pair_ids_sha256": hashlib.sha256(deferred_pairs.tobytes()).hexdigest(),
        "selected_row_indices_sha256": hashlib.sha256(selected_rows.tobytes()).hexdigest(),
        "zero_step_feasible_for_selected_rows": bool(np.all(-lower[selected_rows] >= -1e-12)),
    }
    return selected_pairs, deferred_pairs, selected_rows, row_slack, report


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_candidate(path: Path):
    sha = _sha(path)
    with np.load(path, allow_pickle=False) as z:
        points = np.asarray(z["points_world_m"])
        faces = np.asarray(z["source_O_faces"])
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{path}: points_world_m must be finite Float32 (N,3)")
    if not np.issubdtype(faces.dtype, np.integer) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: source_O_faces must be integer (F,3)")
    return points, faces, sha, hashlib.sha256(points.tobytes()).hexdigest(), hashlib.sha256(faces.tobytes()).hexdigest()


def _write_selected_rows(source_npz: Path, selected_rows: np.ndarray, output_npz: Path):
    with np.load(source_npz, allow_pickle=False) as z:
        ptr = np.asarray(z["row_indptr"])
        node_ids = np.asarray(z["row_node_ids"])
        coefficients = np.asarray(z["row_coefficients"])
        lower = np.asarray(z["row_lower_mm"])
        pair_ids = np.asarray(z["row_pair_ids"])
        lengths = np.diff(ptr)[selected_rows]
        new_ptr = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        spans = [np.arange(ptr[i], ptr[i + 1], dtype=np.int64) for i in selected_rows]
        nnz = np.concatenate(spans) if spans else np.empty(0, dtype=np.int64)
        np.savez_compressed(
            output_npz,
            row_indptr=new_ptr,
            row_node_ids=node_ids[nnz],
            row_coefficients=coefficients[nnz],
            row_lower_mm=lower[selected_rows],
            row_pair_ids=pair_ids[selected_rows],
            selected_source_row_indices=selected_rows,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard-base-candidate", required=True)
    ap.add_argument("--guard-report", required=True)
    ap.add_argument("--guard-rows", required=True)
    ap.add_argument("--motion-base-candidate", required=True)
    ap.add_argument("--motion-proposal-candidate", required=True)
    ap.add_argument("--mandatory-pair-ids", required=True, action="append")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--activity-margin-mm", type=float, default=0.0)
    ap.add_argument("--max-node-step-mm", type=float, default=1.0)
    ap.add_argument("--motion-scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    guard_base = Path(args.guard_base_candidate)
    guard_report_path = Path(args.guard_report)
    guard_rows_path = Path(args.guard_rows)
    motion_base = Path(args.motion_base_candidate)
    motion_proposal = Path(args.motion_proposal_candidate)
    outdir = Path(args.output_dir)
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite {outdir}")
    gr = json.loads(guard_report_path.read_text())
    expected_candidate_sha = gr.get("candidate_sha256")
    gp, gf, gsha, gpoints_sha, gfaces_sha = _load_candidate(guard_base)
    if expected_candidate_sha != gsha:
        raise ValueError("guard report candidate SHA does not match guard-base candidate")
    mbp, mbf, mbsha, mbpoints_sha, mbfaces_sha = _load_candidate(motion_base)
    mpp, mpf, mpsha, mppoints_sha, mpfaces_sha = _load_candidate(motion_proposal)
    if gp.shape != mbp.shape or gp.shape != mpp.shape or not np.array_equal(gf, mbf) or not np.array_equal(gf, mpf):
        raise ValueError("guard base, motion base, and proposal must share point shape and exact face topology")
    if not np.isfinite(args.motion_scale) or args.motion_scale <= 0:
        raise ValueError("motion_scale must be finite and positive")
    delta_mm = (mpp.astype(np.float64) - mbp.astype(np.float64)) * (1000.0 * args.motion_scale)
    mandatory_arrays = [np.load(p, allow_pickle=False) for p in args.mandatory_pair_ids]
    mandatory = np.unique(np.concatenate([_pair_array(a, "mandatory input") for a in mandatory_arrays], axis=0), axis=0)
    rows_sha = _sha(guard_rows_path)
    with np.load(guard_rows_path, allow_pickle=False) as z:
        result = select_threatened_pairs(
            row_indptr=z["row_indptr"], row_node_ids=z["row_node_ids"], row_coefficients=z["row_coefficients"],
            row_lower_mm=z["row_lower_mm"], row_pair_ids=z["row_pair_ids"],
            predicted_delta_mm=delta_mm, mandatory_pair_ids=mandatory,
            activity_margin_mm=args.activity_margin_mm, max_node_step_mm=args.max_node_step_mm,
        )
        selected_pairs, deferred_pairs, selected_rows, row_slack, summary = result
        outdir.mkdir(parents=True, exist_ok=False)
        np.save(outdir / "selected-pair-ids.npy", selected_pairs)
        np.save(outdir / "deferred-pair-ids.npy", deferred_pairs)
        np.save(outdir / "selected-row-indices.npy", selected_rows)
        np.save(outdir / "predicted-row-slack-mm.npy", row_slack)
        _write_selected_rows(guard_rows_path, selected_rows, outdir / "selected-guard-026-rows.npz")

    report = {
        **summary,
        "prediction_motion_scale": float(args.motion_scale),
        "inputs": {
            "guard_base_candidate": {"path": str(guard_base), "sha256": gsha, "points_sha256": gpoints_sha, "faces_sha256": gfaces_sha},
            "guard_report": {"path": str(guard_report_path), "sha256": _sha(guard_report_path)},
            "guard_rows": {"path": str(guard_rows_path), "sha256": rows_sha},
            "motion_base_candidate": {"path": str(motion_base), "sha256": mbsha, "points_sha256": mbpoints_sha},
            "motion_proposal_candidate": {"path": str(motion_proposal), "sha256": mpsha, "points_sha256": mppoints_sha},
            "mandatory_pair_inputs": [{"path": str(Path(p)), "sha256": _sha(Path(p))} for p in args.mandatory_pair_ids],
        },
        "outputs": {
            name: {"path": str(outdir / name), "sha256": _sha(outdir / name)}
            for name in ["selected-pair-ids.npy", "deferred-pair-ids.npy", "selected-row-indices.npy", "predicted-row-slack-mm.npy", "selected-guard-026-rows.npz"]
        },
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "unique_input_pair_count", "mandatory_pair_count", "threatened_pair_count", "selected_pair_count", "deferred_pair_count", "selected_row_count", "predicted_clipped_node_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
