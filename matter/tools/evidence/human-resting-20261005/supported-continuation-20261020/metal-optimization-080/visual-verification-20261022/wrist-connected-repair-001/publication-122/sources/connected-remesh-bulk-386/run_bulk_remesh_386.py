#!/usr/bin/env python3
"""Bounded 370 proposal / self-filter / 130+159 remesh pipeline from 381.

Construction only. Existing 370 scoring and 358 topology gates are reused.
Self-witnessed operations are removed as a fixed whitelist subset; no replacement
flips are generated. Every retained round must pass full 130 and 185-bank 159.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import traceback
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
H = ROOT / "barycentric-connected-clearance-170"
START_CANDIDATE = H / "same-parent-remesh-381/run-001/candidate.npz"
START_AUDIT_159 = H / "same-parent-remesh-381/run-001/audit-159/report.json"
START_AUDIT_130 = H / "same-parent-remesh-381/run-001/audit-130/summary.json"
SELECTOR370 = ROOT / "D/connected-remesh-reuse-370/select_bone_reducing_flips_370.py"
FLIP358 = ROOT / "D/connected-remesh-reuse-358/same_parent_flips_358.py"
AUDIT130 = ROOT / "audit_connected_outer_geometry_130.py"
AUDIT159 = ROOT / "audit_skin_bone_surfaces_159.py"
BANKHELPER = ROOT / "publication-093/canonical/matter/tools/human_skin_fat_bone_audit.py"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
SLOT = Path("/tmp/numi-human-geometry-slot.py")
EXPECTED = {
    "start_candidate": "8045605bc817a20a281359467c700a1135fead210372efddf8416704cbe51395",
    "start_report_159": "3f2b881889bee36c598416bc9db6ff400f8069877d28f21ba51663e76a944068",
    "selector370": "c8650b8c311ca8f386ebf1713b5fc285b7bded860cfa878192472704ac1336d5",
    "flip358": "a08e1f7e6392854f72ffab09af6f2b1f2afe885b1a84f586afe71160ff3ccbee",
    "audit130": "52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9",
    "audit159": "de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1",
    "bank_helper": "37c040a442f1fdf312745c1cc1fb6c543383f1cd0782db46c81ca1eadf0d8324",
}
EXPECTED_START_PAIR_COUNT = 3401
EXPECTED_INSIDE_NODES = 485
MAX_ROUNDS = 3
MAX_SELF_FILTER_PASSES = 3


class PipelineError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def write_npz(path: Path, data: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **data)
    return sha256_file(path)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(
        value, indent=2, sort_keys=True,
        default=lambda item: item.item() if hasattr(item, "item") else str(item),
    ) + "\n")


def _copy_verified(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    source_sha = sha256_file(source)
    if sha256_file(temporary) != source_sha:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"checkpoint copy hash mismatch: {source}")
    os.replace(temporary, destination)
    return source_sha


def persist_accepted_checkpoint(
    output_dir: Path, round_index: int, candidate_path: Path,
    audit130_summary_path: Path, audit130_pairs_path: Path,
    audit159_report_path: Path, strict_reduction: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically retain a candidate only after exact-self and full-159 decrease gates."""
    summary = json.loads(audit130_summary_path.read_text())
    report = json.loads(audit159_report_path.read_text())
    candidate_sha = sha256_file(candidate_path)
    if summary.get("candidate_sha256") != candidate_sha:
        raise PipelineError("checkpoint 130 candidate pin mismatch")
    if summary.get("exact_PLC_surface_pairs") != 0 or summary.get("complete_surface_audit") is not True:
        raise PipelineError("checkpoint candidate is not a complete exact-self pass")
    pairs = np.load(audit130_pairs_path, allow_pickle=False)
    if pairs.shape != (0, 2) or int(summary.get("exact_PLC_surface_pairs", -1)) != len(pairs):
        raise PipelineError("checkpoint exact-self pair payload is not empty and well formed")
    pair_counts(report)
    if report.get("candidate_sha256") != candidate_sha:
        raise PipelineError("checkpoint 159 candidate pin mismatch")
    reduction = strict_reduction.get("strict_reduction")
    if type(reduction) is not int or reduction <= 0:
        raise PipelineError("checkpoint requires positive strict 159 reduction")
    if strict_reduction.get("after_total", 1) >= strict_reduction.get("before_total", 0):
        raise PipelineError("checkpoint strict 159 totals do not decrease")

    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"round-{round_index:03d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary = checkpoint_root / f".round-{round_index:03d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    candidate_copy = temporary / "candidate.npz"
    _copy_verified(candidate_path, candidate_copy)
    summary_copy = temporary / "audit-130-summary.json"
    pairs_copy = temporary / "surface-self-pairs.npy"
    report_copy = temporary / "audit-159-report.json"
    _copy_verified(audit130_summary_path, summary_copy)
    _copy_verified(audit130_pairs_path, pairs_copy)
    _copy_verified(audit159_report_path, report_copy)
    record = {
        "status": "ACCEPTED_CHECKPOINT",
        "round": int(round_index),
        "candidate_sha256": candidate_sha,
        "audit130_summary_sha256": sha256_file(summary_copy),
        "self_pair_count": 0,
        "audit159_report_sha256": sha256_file(report_copy),
        "strict_159_reduction": dict(strict_reduction),
        "candidate": str(destination / "candidate.npz"),
    }
    (temporary / "checkpoint.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    record["checkpoint_json_sha256"] = sha256_file(destination / "checkpoint.json")
    return record


def write_terminal_receipt(output_dir: Path, status: str, report_path: Path | None = None) -> None:
    checkpoints = sorted((output_dir / "checkpoints").glob("round-*/checkpoint.json"))
    latest = None
    if checkpoints:
        latest_path = checkpoints[-1]
        latest = {
            "checkpoint_json": str(latest_path),
            "checkpoint_json_sha256": sha256_file(latest_path),
            "checkpoint": json.loads(latest_path.read_text()),
        }
    receipt = {"status": status, "latest_accepted_checkpoint": latest}
    if report_path is not None and report_path.exists():
        receipt["pipeline_report"] = str(report_path)
        receipt["pipeline_report_sha256"] = sha256_file(report_path)
    save_json(output_dir / "terminal-receipt.json", receipt)


def write_terminal_failure(output_dir: Path, exc: BaseException, candidate: Path, baseline: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted((output_dir / "checkpoints").glob("round-*/checkpoint.json"))
    latest = None
    if checkpoints:
        cp = checkpoints[-1]
        latest = {"checkpoint_json": str(cp), "checkpoint_json_sha256": sha256_file(cp),
                  "checkpoint": json.loads(cp.read_text())}
    excluded = {"terminal-failure.json"}
    artifact_index = [
        {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and "checkpoints" not in path.relative_to(output_dir).parts
        and path.name not in excluded and not path.name.endswith(".tmp")
    ]
    failure = {
        "status": "TERMINAL_FAILURE",
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate) if candidate.exists() else None,
        "baseline_159_report": str(baseline),
        "baseline_159_report_sha256": sha256_file(baseline) if baseline.exists() else None,
        "latest_accepted_checkpoint": latest,
        "failed_output_artifacts": artifact_index,
    }
    save_json(output_dir / "terminal-failure.json", failure)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"cannot import pinned owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def operation_signature(operation: Mapping[str, Any]) -> tuple:
    return (
        tuple(sorted(map(int, operation["edge"]))),
        tuple(map(int, operation["face_rows"])),
        tuple(tuple(map(int, tri)) for tri in operation["new_faces"]),
        int(operation["parent_row"]),
    )


def filter_implicated_operations(
    operations: Sequence[Mapping[str, Any]],
    witness_face_rows: Sequence[int],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Remove all operations touching a 130 witness face, without filling."""
    bad_rows = {int(row) for row in witness_face_rows}
    if any(row < 0 for row in bad_rows):
        raise PipelineError("self witness contains a negative face row")
    seen = set()
    kept, removed = [], []
    for operation in operations:
        signature = operation_signature(operation)
        if signature in seen:
            raise PipelineError("duplicate selected operation")
        seen.add(signature)
        rows = {int(row) for row in operation["face_rows"]}
        (removed if rows & bad_rows else kept).append(operation)
    return kept, removed


def replay_exact_whitelist(
    data: Mapping[str, np.ndarray],
    operations: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[int],
    flip_module: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild only the retained edge whitelist through pinned 358."""
    edges = [list(map(int, op["edge"])) for op in operations]
    normalized = [tuple(sorted(edge)) for edge in edges]
    if len(set(normalized)) != len(normalized):
        raise PipelineError("duplicate retained edge")
    if not operations:
        return {key: np.array(value, copy=True) for key, value in data.items()}, {
            "accepted_pair_count": 0, "operations": [], "allowed_edge_count": 0,
        }
    out, report = flip_module.propose_flip_batch(
        data, list(map(int, seed_rows)), max_flips=len(operations),
        min_angle_gain_deg=1.0e-8, min_normal_dot=0.0, allowed_edges=edges,
    )
    generated = report.get("operations")
    if not isinstance(generated, list):
        raise PipelineError("358 omitted exact operation identities")
    expected_signatures = {operation_signature(op) for op in operations}
    actual_signatures = {operation_signature(op) for op in generated}
    if len(generated) != len(expected_signatures) or actual_signatures != expected_signatures:
        raise PipelineError("358 failed exact whitelist replay; replacements are forbidden")
    if not np.array_equal(out["points_world_m"], data["points_world_m"]):
        raise PipelineError("358 changed point coordinates")
    if not np.array_equal(out["source_O_faces"], out["area_reference_037_source_O_faces"]):
        raise PipelineError("358 desynchronized current and 037 connectivity")
    return out, report


def pair_counts(report: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    if (report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or
            report.get("complete") is not True or report.get("bank_count") != 185 or
            report.get("expected_bank_count") != 185):
        raise PipelineError("159 report is not complete for all 185 typed banks")
    rows = report.get("banks")
    if not isinstance(rows, list) or len(rows) != 185:
        raise PipelineError("159 report has an incomplete bank inventory")
    counts: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (row.get("body"), row.get("stable_bone_id"))
        if (type(key[0]) is not int or type(key[1]) is not int or key in counts):
            raise PipelineError("159 typed bank identity is malformed or duplicated")
        result = row.get("result")
        if result is None:
            if row.get("skin_aabb_active") != 0 or row.get("bank_aabb_active") != 0:
                raise PipelineError(f"159 missing result lacks AABB proof for {key}")
            counts[key] = 0
        else:
            pairs = result.get("triangle_pairs")
            if not isinstance(pairs, list) or result.get("count") != len(pairs):
                raise PipelineError(f"159 count/pair rows disagree for {key}")
            counts[key] = len(pairs)
    if len(counts) != 185:
        raise PipelineError("159 typed bank inventory is incomplete")
    return counts


def require_strict_pair_decrease(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> dict[str, Any]:
    old, new = pair_counts(before), pair_counts(after)
    if set(old) != set(new):
        raise PipelineError("typed-bank identity domain changed")
    increases = [
        {"body": key[0], "stable_bone_id": key[1], "before": old[key], "after": new[key]}
        for key in sorted(old) if new[key] > old[key]
    ]
    reduction = sum(old.values()) - sum(new.values())
    if increases or reduction <= 0:
        raise PipelineError(
            f"159 strict decrease failed: reduction={reduction}, increases={increases[:8]}"
        )
    return {
        "before_total": sum(old.values()),
        "after_total": sum(new.values()),
        "strict_reduction": reduction,
        "typed_bank_increases": increases,
    }


def checked_130_summary(
    summary_path: Path, candidate_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    summary = json.loads(summary_path.read_text())
    if summary.get("candidate_sha256") != sha256_file(candidate_path):
        raise PipelineError("130 summary candidate hash mismatch")
    if summary.get("complete_surface_audit") is not True:
        raise PipelineError("130 surface self audit is incomplete")
    pairs = np.load(summary_path.parent / "surface-self-pairs.npy", allow_pickle=False)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or
            not np.issubdtype(pairs.dtype, np.integer) or
            int(summary.get("exact_PLC_surface_pairs", -1)) != len(pairs)):
        raise PipelineError("130 saved self-pair rows do not match summary")
    return summary, pairs.astype(np.int64, copy=False)


def run_130(candidate: Path, output: Path) -> tuple[dict[str, Any], np.ndarray]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(SLOT), str(PYTHON), str(AUDIT130), str(candidate), str(output)]
    with (output.parent / (output.name + ".log")).open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise PipelineError(f"130 failed ({result.returncode}); see {output.parent}/{output.name}.log")
    return checked_130_summary(output / "summary.json", candidate)


def run_159(candidate: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(SLOT), str(PYTHON), str(AUDIT159),
        str(candidate), str(output), "--max-seconds", "600",
    ]
    with (output.parent / (output.name + ".log")).open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise PipelineError(f"159 failed ({result.returncode}); see {output.parent}/{output.name}.log")
    report = json.loads((output / "report.json").read_text())
    pair_counts(report)
    if report.get("candidate_sha256") != sha256_file(candidate):
        raise PipelineError("159 report candidate hash mismatch")
    return report


def run_pipeline(
    candidate_path: Path,
    baseline_159_path: Path,
    output_dir: Path,
    expected_candidate_sha256: str,
    expected_159_sha256: str,
    max_rounds: int = 3,
    max_flips: int = 136,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= max_rounds <= MAX_ROUNDS:
        raise PipelineError("max_rounds must be between 1 and 3")
    if sha256_file(candidate_path) != expected_candidate_sha256:
        raise PipelineError("input candidate pin mismatch")
    if sha256_file(baseline_159_path) != expected_159_sha256:
        raise PipelineError("input 159 report pin mismatch")
    for name, path in (
        ("selector370", SELECTOR370), ("flip358", FLIP358),
        ("audit130", AUDIT130), ("audit159", AUDIT159), ("bank_helper", BANKHELPER),
    ):
        if sha256_file(path) != EXPECTED[name]:
            raise PipelineError(f"pinned owner source changed: {name}")

    baseline = json.loads(baseline_159_path.read_text())
    if baseline.get("candidate_sha256") != expected_candidate_sha256:
        raise PipelineError("initial 159 report is not for pinned candidate")
    if sum(pair_counts(baseline).values()) != EXPECTED_START_PAIR_COUNT:
        raise PipelineError("381 baseline 159 total is not 3401")
    start130 = json.loads(START_AUDIT_130.read_text())
    if (sha256_file(START_CANDIDATE) != EXPECTED["start_candidate"] or
            start130.get("candidate_sha256") != EXPECTED["start_candidate"] or
            start130.get("exact_PLC_surface_pairs") != 0 or
            start130.get("inside_nodes") != EXPECTED_INSIDE_NODES):
        raise PipelineError("381 initial exact self / inside receipt is not pinned")

    selector = import_module("select_bone_reducing_flips_370_for_383", SELECTOR370)
    flip, surface_owner, bank_helper, exact, runtime_hashes = selector.load_pinned_runtime()
    raw_inputs = bank_helper.read_bank_inputs(
        selector.BANK_DIR_DEFAULT, selector.EVIDENCE_ROOT_DEFAULT,
    )
    banks, bank_keys = selector.typed_banks_from_159_inputs(raw_inputs)
    bank_helper.summarize_surface_crossing_report(
        baseline, expected_banks=185, expected_bank_keys=bank_keys,
    )

    output_dir.mkdir(parents=True)
    current_path = candidate_path
    current_report_path = baseline_159_path
    current_report = baseline
    current_data = load_npz(candidate_path)
    history: list[dict[str, Any]] = []
    status = "MAX_ROUNDS_COMPLETED"

    for round_index in range(1, max_rounds + 1):
        rd = output_dir / f"round-{round_index:03d}"
        rd.mkdir()
        current_sha = sha256_file(current_path)
        report_sha = sha256_file(current_report_path)
        _pairs, target_rows, _counts = selector._pair_rows_and_counts(
            current_report, candidate_sha256=current_sha,
            face_count=len(current_data["source_O_faces"]),
            banks=banks, expected_bank_keys=bank_keys,
        )
        targets = sorted(target_rows)
        rec: dict[str, Any] = {
            "round": round_index, "input_candidate_sha256": current_sha,
            "input_159_sha256": report_sha,
            "input_pair_total": sum(pair_counts(current_report).values()),
            "input_target_face_count": len(targets),
        }
        if not targets:
            rec["status"] = "NO_159_TARGET_FACES"
            history.append(rec)
            status = "NO_PROGRESS"
            break

        proposal, proposal_report = flip.propose_flip_batch(
            current_data, targets, max_flips=max_flips,
            min_angle_gain_deg=1.0e-8, min_normal_dot=0.0,
        )
        proposed, selector_report = selector.filter_proposed_batch_370(
            current_data, proposal_report, current_report,
            candidate_sha256=current_sha, baseline_report_sha256=report_sha,
            banks=banks, expected_bank_keys=bank_keys,
            exact_predicate=exact, surface_records=surface_owner.records,
            flip_module=flip, max_flips=max_flips,
        )
        save_json(rd / "proposal-358-report.json", proposal_report)
        save_json(rd / "selector-370-report.json", selector_report)
        rec["proposal_358_count"] = len(proposal_report.get("operations", []))
        rec["selected_370_count"] = int(selector_report.get("strict_pair_reducing_flip_count", 0))
        rec["selector_status"] = selector_report.get("status")
        if not rec["selected_370_count"]:
            rec["status"] = "NO_SELECTED_PAIR_REDUCING_FLIPS"
            history.append(rec)
            status = "NO_PROGRESS"
            break

        proposed_path = rd / "proposal-370.npz"
        write_npz(proposed_path, proposed)
        proposed_130, self_pairs = run_130(proposed_path, rd / "proposal-130")
        rec["proposal_self_pairs"] = len(self_pairs)
        selected_ops = selector_report["final_flip_operator_report"]["operations"]
        seeds = selector_report["final_flip_operator_report"]["seed_face_rows"]
        if not isinstance(selected_ops, list) or not isinstance(seeds, list):
            raise PipelineError("370 selected operations or seed rows missing")
        working, working_path = proposed, proposed_path
        working_130, remaining_pairs = proposed_130, self_pairs
        working_130_path = rd / "proposal-130" / "summary.json"
        remaining_ops = list(selected_ops)
        removed_all: list[dict[str, Any]] = []

        for pass_index in range(1, MAX_SELF_FILTER_PASSES + 1):
            if len(remaining_pairs) == 0:
                break
            witness_rows = sorted(set(map(int, remaining_pairs.reshape(-1))))
            kept, removed = filter_implicated_operations(remaining_ops, witness_rows)
            if not removed:
                rec["self_filter_stop"] = "witness_rows_touch_no_remaining_operation"
                break
            removed_all.extend({
                "edge": list(map(int, op["edge"])),
                "face_rows": list(map(int, op["face_rows"])),
                "parent_row": int(op["parent_row"]),
            } for op in removed)
            rebuilt, replay_report = replay_exact_whitelist(current_data, kept, seeds, flip)
            subset_path = rd / f"self-safe-subset-{pass_index:02d}.npz"
            write_npz(subset_path, rebuilt)
            save_json(rd / f"self-safe-subset-{pass_index:02d}-358-report.json", replay_report)
            subset_130, subset_pairs = run_130(
                subset_path, rd / f"self-safe-subset-{pass_index:02d}-130",
            )
            rec.setdefault("self_filter_passes", []).append({
                "pass": pass_index, "removed_count": len(removed),
                "retained_count": len(kept), "self_pair_count": len(subset_pairs),
                "inside_nodes": subset_130.get("inside_nodes"),
            })
            working, working_path = rebuilt, subset_path
            working_130, remaining_pairs, remaining_ops = subset_130, subset_pairs, kept
            working_130_path = rd / f"self-safe-subset-{pass_index:02d}-130" / "summary.json"

        if len(remaining_pairs):
            rec["status"] = "STOPPED_SELF_FILTER_DID_NOT_CLEAR"
            rec["removed_self_implicated_operation_count"] = len(removed_all)
            history.append(rec)
            status = "SELF_FILTER_FAILED"
            break
        if working_130.get("inside_nodes") != EXPECTED_INSIDE_NODES:
            rec.update(status="STOPPED_INSIDE_COUNT_CHANGED",
                       inside_nodes=working_130.get("inside_nodes"))
            history.append(rec)
            status = "INSIDE_COUNT_CHANGED"
            break

        after_159_dir = rd / "accepted-159"
        after_159 = run_159(working_path, after_159_dir)
        reduction = require_strict_pair_decrease(current_report, after_159)
        rec.update({
            "status": "ROUND_ACCEPTED_STRICT_159_DECREASE",
            "final_self_pair_count": 0,
            "inside_nodes": working_130.get("inside_nodes"),
            "strict_159_decrease": reduction,
            "accepted_candidate": str(working_path),
            "accepted_candidate_sha256": sha256_file(working_path),
            "accepted_159_report": str(after_159_dir / "report.json"),
            "accepted_159_report_sha256": sha256_file(after_159_dir / "report.json"),
            "accepted_operation_count": len(remaining_ops),
            "removed_self_implicated_operations": removed_all,
            "accepted_operations": [
                {"edge": list(map(int, op["edge"])),
                 "face_rows": list(map(int, op["face_rows"])),
                 "parent_row": int(op["parent_row"])}
                for op in remaining_ops
            ],
        })
        checkpoint = persist_accepted_checkpoint(
            output_dir, round_index, working_path,
            working_130_path,
            working_130_path.parent / "surface-self-pairs.npy",
            after_159_dir / "report.json", reduction,
        )
        rec["accepted_checkpoint"] = checkpoint
        history.append(rec)
        current_path, current_report_path, current_report = (
            working_path, after_159_dir / "report.json", after_159,
        )
        current_data = working

    final_path = output_dir / "final-candidate.npz"
    shutil.copyfile(current_path, final_path)
    summary = {
        "status": status,
        "geometry_admitted": False, "native_admission": False,
        "accepted_physical_steps": 0,
        "starting_candidate_sha256": expected_candidate_sha256,
        "starting_159_report_sha256": expected_159_sha256,
        "pinned_sources": dict(EXPECTED),
        "runtime_source_hashes": runtime_hashes,
        "bank_input_pins": raw_inputs[8],
        "max_rounds": max_rounds, "max_flips_per_round": max_flips,
        "required_inside_nodes": EXPECTED_INSIDE_NODES,
        "rounds": history,
        "final_candidate": str(final_path),
        "final_candidate_sha256": sha256_file(final_path),
        "final_159_report": str(current_report_path),
        "final_159_report_sha256": sha256_file(current_report_path),
        "all_final_candidates_exact_self_audited": True,
        "all_pair_decrease_rounds_complete_185_bank": True,
        "limitations": "Fixed-coordinate topology proposal only. Full self and typed-bank audits do not qualify physical or native behavior.",
    }
    save_json(output_dir / "pipeline-386-report.json", summary)
    write_terminal_receipt(output_dir, status, output_dir / "pipeline-386-report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=START_CANDIDATE)
    parser.add_argument("--baseline-159-report", type=Path, default=START_AUDIT_159)
    parser.add_argument("--expected-candidate-sha256", default=EXPECTED["start_candidate"])
    parser.add_argument("--expected-report-159-sha256", default=EXPECTED["start_report_159"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-flips", type=int, default=136)
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.candidate, args.baseline_159_report, args.output_dir,
            args.expected_candidate_sha256, args.expected_report_159_sha256,
            args.max_rounds, args.max_flips,
        )
    except BaseException as exc:
        write_terminal_failure(args.output_dir, exc, args.candidate, args.baseline_159_report)
        raise
    print(json.dumps({
        "status": result["status"],
        "final_candidate_sha256": result["final_candidate_sha256"],
        "final_159_report_sha256": result["final_159_report_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
