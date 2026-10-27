#!/usr/bin/env python3
"""Create a construction-only connected-component field composition.

The operation reuses two already-complete matched 545/130/159 checks to
predict per-face bank-pair counts. It does not transfer their admission or
clearance certificate to the mixed candidate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np

from component_field_selection627 import (
    ValidationError, connected_changed_components, load_json, select_safe_components,
    sha256_file, validate_complete_evidence, validate_159_report,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-candidate", required=True)
    p.add_argument("--baseline-full-report", required=True)
    p.add_argument("--baseline-130-summary", required=True)
    p.add_argument("--baseline-159-report", required=True)
    p.add_argument("--baseline-arithmetic-report", required=True)
    p.add_argument("--target-candidate", required=True)
    p.add_argument("--target-full-report", required=True)
    p.add_argument("--target-130-summary", required=True)
    p.add_argument("--target-159-report", required=True)
    p.add_argument("--target-arithmetic-report", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    baseline_path, target_path = Path(a.baseline_candidate), Path(a.target_candidate)
    base_ev = validate_complete_evidence(baseline_path, a.baseline_full_report, a.baseline_130_summary, a.baseline_159_report, a.baseline_arithmetic_report)
    target_ev = validate_complete_evidence(target_path, a.target_full_report, a.target_130_summary, a.target_159_report, a.target_arithmetic_report)

    base = load_npz(baseline_path)
    target = load_npz(target_path)
    if set(base) != set(target):
        raise ValidationError("candidate NPZ key sets differ")
    for key in sorted(base):
        if key == "points_world_m":
            continue
        x, y = base[key], target[key]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes(order="C") != y.tobytes(order="C"):
            raise ValidationError(f"nonpoint payload differs: {key}")
    if "points_world_m" not in base or "source_O_faces" not in base:
        raise ValidationError("required geometry arrays missing")
    x, y, faces = base["points_world_m"], target["points_world_m"], base["source_O_faces"]
    if x.dtype != np.float32 or y.dtype != np.float32 or x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValidationError("candidate points are not matching finite F32 arrays")
    if faces.dtype.kind not in "iu" or faces.ndim != 2 or faces.shape[1] != 3 or (faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(x))):
        raise ValidationError("invalid topology")
    changed = np.flatnonzero(np.any(x.view(np.uint32) != y.view(np.uint32), axis=1)).astype(np.int64)
    if len(changed) == 0:
        raise ValidationError("no bitwise coordinate changes to select")
    anchors = base.get("typed_coarse_anchor_output_rows_018")
    if anchors is None or anchors.dtype.kind not in "iu" or anchors.shape != (237,) or len(np.unique(anchors)) != 237:
        raise ValidationError("typed anchor rows invalid")
    if target["typed_coarse_anchor_output_rows_018"].tobytes() != anchors.tobytes():
        raise ValidationError("anchor identities changed")
    if np.any(anchors < 0) or np.any(anchors >= len(x)):
        raise ValidationError("anchor row out of range")
    if x[anchors].tobytes() != y[anchors].tobytes():
        raise ValidationError("target moved a fixed typed anchor")

    baseline_banks = base_ev["banks"]
    target_banks = target_ev["banks"]
    selection = select_safe_components(faces, changed, baseline_banks, target_banks)
    selected = set(selection["selected_component_ids"])
    components, component_faces, _ = connected_changed_components(faces, changed)
    chosen_nodes = np.concatenate([nodes for ci, nodes in enumerate(components) if ci in selected]) if selected else np.empty(0, dtype=np.int64)
    chosen_nodes = np.unique(chosen_nodes)
    candidate = {k: np.array(v, copy=True) for k, v in base.items()}
    candidate["points_world_m"][chosen_nodes] = target["points_world_m"][chosen_nodes]
    if candidate["points_world_m"].dtype != np.float32 or not np.isfinite(candidate["points_world_m"]).all():
        raise ValidationError("composed F32 candidate invalid")
    if candidate["points_world_m"][anchors].tobytes() != x[anchors].tobytes():
        raise ValidationError("composed field changed anchors")
    for key in sorted(base):
        if key != "points_world_m" and candidate[key].tobytes(order="C") != base[key].tobytes(order="C"):
            raise ValidationError(f"composed candidate changed nonpoint payload: {key}")

    candidate_path = out / "candidate.npz"
    fd, tmp_name = tempfile.mkstemp(prefix=".candidate.", suffix=".npz", dir=out)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **candidate)
        os.replace(tmp_name, candidate_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    candidate_sha = sha256_file(candidate_path)

    report = {
        "schema": "connected-safe-component-field-selection-627-v1",
        "status": "CONSTRUCTION_ONLY_PREDICTED_COMPONENT_COMPOSITION",
        "native_admission": False,
        "geometry_admitted": False,
        "fresh_full_545_required": True,
        "exact_checks_transferred": False,
        "baseline": {
            "candidate_path": str(baseline_path.resolve()),
            "candidate_sha256": base_ev["candidate_sha256"],
            "full_report": str(Path(a.baseline_full_report).resolve()),
            "full_report_sha256": base_ev["full_report_sha256"],
            "130_summary": str(Path(a.baseline_130_summary).resolve()),
            "130_summary_sha256": base_ev["summary_sha256"],
            "159_report": str(Path(a.baseline_159_report).resolve()),
            "159_report_sha256": base_ev["surface_sha256"],
            "arithmetic_report": str(Path(a.baseline_arithmetic_report).resolve()),
            "arithmetic_report_sha256": base_ev["arithmetic_sha256"],
            "surface_pairs": int(base_ev["full_report"]["surface_pairs"]),
            "inside_pairs": 0, "self_pairs": 0, "complete_banks": 185,
        },
        "target": {
            "candidate_path": str(target_path.resolve()),
            "candidate_sha256": target_ev["candidate_sha256"],
            "full_report": str(Path(a.target_full_report).resolve()),
            "full_report_sha256": target_ev["full_report_sha256"],
            "130_summary": str(Path(a.target_130_summary).resolve()),
            "130_summary_sha256": target_ev["summary_sha256"],
            "159_report": str(Path(a.target_159_report).resolve()),
            "159_report_sha256": target_ev["surface_sha256"],
            "arithmetic_report": str(Path(a.target_arithmetic_report).resolve()),
            "arithmetic_report_sha256": target_ev["arithmetic_sha256"],
            "surface_pairs": int(target_ev["full_report"]["surface_pairs"]),
            "inside_pairs": 0, "self_pairs": 0, "complete_banks": 185,
        },
        "composed_candidate": str(candidate_path.resolve()),
        "composed_candidate_sha256": candidate_sha,
        "topology_sha256": hashlib.sha256(base["source_O_faces"].tobytes(order="C")).hexdigest(),
        "changed_node_count": int(len(changed)),
        "changed_node_rows": changed.tolist(),
        "selected_node_count": int(len(chosen_nodes)),
        "selected_node_rows": chosen_nodes.tolist(),
        "selected_component_ids": selection["selected_component_ids"],
        "component_count": len(selection["components"]),
        "components": selection["components"],
        "predicted_baseline_bank_counts": selection["baseline_bank_counts"],
        "predicted_composed_bank_counts": selection["predicted_bank_counts"],
        "predicted_baseline_surface_pairs": selection["baseline_total"],
        "predicted_composed_surface_pairs": selection["predicted_total"],
        "predicted_surface_pair_delta": selection["predicted_total_delta"],
        "prediction_limit": "Pair counts are composed from disjoint touched-face sets and matched exact 159 reports; this is not an exact audit of the mixed F32 candidate. Self-intersections may arise across component boundaries. Run a fresh full 545 check.",
        "input_npz_nonpoint_payload_bitwise_equal": True,
        "topology_and_metadata_bitwise_identical": True,
        "points_dtype": str(candidate["points_world_m"].dtype),
        "typed_anchor_rows": int(len(anchors)),
        "typed_anchors_bitwise_unchanged": True,
        "input_hashes": {
            "baseline_candidate": base_ev["candidate_sha256"],
            "target_candidate": target_ev["candidate_sha256"],
        },
    }
    report_path = out / "selection-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "report": str(report_path),
        "changed_nodes": len(changed),
        "selected_nodes": len(chosen_nodes),
        "components": len(selection["components"]),
        "selected_components": len(selected),
        "predicted_pairs": [selection["baseline_total"], selection["predicted_total"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
