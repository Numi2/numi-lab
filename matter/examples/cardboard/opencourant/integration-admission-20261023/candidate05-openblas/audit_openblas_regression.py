#!/usr/bin/env python3
"""Read-only identity audit for Candidate 05 OpenBLAS regression histories.

This script compares retained native time-history binaries directly. It does
not launch OpenCourant, Starter, or th_to_csv. Only the documented 24-byte
ASCII wall-clock field at byte offset 96 is normalized; all other bytes must
match exactly for a timestamp-only result.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUTPUT = HERE / "openblas-regression-audit.json"
OVERLAY_BASELINE_POST = HERE.parent / "post" / "overlay-04-direct"
OVERLAY_OPENBLAS_POST = HERE.parents[1] / "paper-predictor-20261023" / "post" / "openblas-overlay-04-direct"
TIMESTAMP_OFFSET = 96
TIMESTAMP_LENGTH = 24
TIMESTAMP_RE = re.compile(rb"^[A-Z][a-z]{2} [A-Z][a-z]{2} [ 0-9][0-9] [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_timestamp(data: bytes) -> str:
    value = data[TIMESTAMP_OFFSET : TIMESTAMP_OFFSET + TIMESTAMP_LENGTH]
    if len(value) != TIMESTAMP_LENGTH or not TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"unvalidated timestamp field at bytes {TIMESTAMP_OFFSET}:{TIMESTAMP_OFFSET + TIMESTAMP_LENGTH}: {value!r}")
    dt.datetime.strptime(value.decode("ascii"), "%a %b %d %H:%M:%S %Y")
    return value.decode("ascii")


def compare_bytes(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    a = reference_path.read_bytes()
    b = candidate_path.read_bytes()
    if len(a) != len(b):
        return {
            "status": "different_length",
            "reference": str(reference_path.relative_to(HERE)),
            "candidate": str(candidate_path.relative_to(HERE)),
            "reference_bytes": len(a),
            "candidate_bytes": len(b),
            "reference_sha256": sha256(a),
            "candidate_sha256": sha256(b),
        }

    time_a = read_timestamp(a)
    time_b = read_timestamp(b)
    start = TIMESTAMP_OFFSET
    end = start + TIMESTAMP_LENGTH
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    other_differences = [i for i in differing if not start <= i < end]
    normalized_a = a[:start] + bytes(TIMESTAMP_LENGTH) + a[end:]
    normalized_b = b[:start] + bytes(TIMESTAMP_LENGTH) + b[end:]
    return {
        "status": "timestamp_only" if not other_differences else "content_differs",
        "reference": str(reference_path.relative_to(HERE)),
        "candidate": str(candidate_path.relative_to(HERE)),
        "bytes": len(a),
        "reference_sha256": sha256(a),
        "candidate_sha256": sha256(b),
        "raw_identical": not differing,
        "reference_timestamp": time_a,
        "candidate_timestamp": time_b,
        "timestamp_byte_offset": start,
        "timestamp_bytes": TIMESTAMP_LENGTH,
        "timestamp_bytes_differ": sum(a[i] != b[i] for i in range(start, end)),
        "non_timestamp_difference_count": len(other_differences),
        "first_non_timestamp_difference_offsets": other_differences[:32],
        "normalized_reference_sha256": sha256(normalized_a),
        "normalized_candidate_sha256": sha256(normalized_b),
        "normalized_hashes_match": normalized_a == normalized_b,
    }


def run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {"run_receipt": None}
    receipt = json.loads(path.read_text())
    loaded = receipt.get("loaded_library_sha256", {})
    blas = {
        name: digest
        for name, digest in loaded.items()
        if any(token in name.casefold() for token in ("libblas", "liblapack", "openblas"))
    }
    input_decks = {
        name: record.get("sha256")
        for name, record in receipt.get("inputs", {}).items()
        if name.endswith(".rad")
    }
    return {
        "run_receipt": str(path.relative_to(HERE)),
        "case_id": receipt.get("case_id"),
        "case_sha256": receipt.get("case_sha256"),
        "case_sequence_complete": receipt.get("case_sequence_complete"),
        "execution_complete": receipt.get("execution_complete"),
        "runtime_profile": receipt.get("runtime_profile"),
        "engine_sha256": receipt.get("binary_sha256", {}).get("engine"),
        "starter_sha256": receipt.get("binary_sha256", {}).get("starter"),
        "elapsed_s": receipt.get("elapsed_s"),
        "environment": {
            key: receipt.get("environment", {}).get(key)
            for key in ("LD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "loaded_blas_libraries_sha256": blas,
        "input_rad_sha256": input_decks,
    }


def find_history(run_dir: Path, stage: str) -> Path:
    matches = sorted(p for p in run_dir.glob(f"*{stage}") if p.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one *{stage} under {run_dir}, found {len(matches)}")
    return matches[0]


def find_titles(run_dir: Path, stage: str) -> Path:
    history = find_history(run_dir, stage)
    titles = Path(str(history) + "_TITLES")
    if not titles.is_file():
        raise FileNotFoundError(titles)
    return titles


def compare_pair(label: str, candidate_dir: Path, reference_dir: Path, stages: tuple[str, ...]) -> dict[str, Any]:
    candidate_run = run_summary(candidate_dir)
    reference_run = run_summary(reference_dir)
    results: list[dict[str, Any]] = []
    for stage in stages:
        candidate_hist = find_history(candidate_dir, stage)
        reference_hist = find_history(reference_dir, stage)
        results.append({
            "history": compare_bytes(reference_hist, candidate_hist),
            "titles": {
                "reference": str(find_titles(reference_dir, stage).relative_to(HERE)),
                "candidate": str(find_titles(candidate_dir, stage).relative_to(HERE)),
                "byte_identical": find_titles(reference_dir, stage).read_bytes() == find_titles(candidate_dir, stage).read_bytes(),
                "sha256": sha256(find_titles(candidate_dir, stage).read_bytes()),
            },
        })
    return {
        "label": label,
        "candidate": candidate_run,
        "reference": reference_run,
        "case_sha256_equal": candidate_run.get("case_sha256") == reference_run.get("case_sha256"),
        "input_rad_sha256_equal": candidate_run.get("input_rad_sha256") == reference_run.get("input_rad_sha256"),
        "engine_sha256_equal": candidate_run.get("engine_sha256") == reference_run.get("engine_sha256"),
        "histories": results,
    }


def compare_csv(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Summarize existing CSVs only when retained binary histories differ."""
    with reference_path.open(newline="") as stream:
        ref_rows = list(csv.DictReader(stream))
    with candidate_path.open(newline="") as stream:
        cand_rows = list(csv.DictReader(stream))
    fields = sorted(set(ref_rows[0]) & set(cand_rows[0])) if ref_rows and cand_rows else []
    result: dict[str, Any] = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_rows": len(ref_rows),
        "candidate_rows": len(cand_rows),
        "shared_columns": fields,
        "columns": {},
    }
    for field in fields:
        diffs: list[float] = []
        paired = 0
        for ra, ca in zip(ref_rows, cand_rows):
            try:
                rv, cv = float(ra[field]), float(ca[field])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(rv) and math.isfinite(cv):
                diffs.append(cv - rv)
                paired += 1
        result["columns"][field] = {
            "paired_rows": paired,
            "max_abs_difference": max((abs(x) for x in diffs), default=0.0),
            "rms_difference": math.sqrt(sum(x * x for x in diffs) / len(diffs)) if diffs else 0.0,
        }
    return result


def _csv_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {field!r}")
    return value


def _history_output_receipt(run_dir: Path, stage: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text())
    outputs = run.get("outputs", {})
    candidates = [(name, value) for name, value in outputs.items() if name.endswith(stage) and not name.endswith("_TITLES")]
    if len(candidates) != 1:
        raise ValueError(f"expected one retained {stage} output in {run_dir}, found {len(candidates)}")
    name, record = candidates[0]
    history = run_dir / name
    if not history.is_file():
        raise FileNotFoundError(history)
    return run, history, record


def compare_existing_csv(reference_run_dir: Path, reference_post_dir: Path,
                         candidate_run_dir: Path, candidate_post_dir: Path,
                         stage: str) -> dict[str, Any]:
    """Compare pre-existing converted histories and bind them to native files."""
    reference_run, reference_history, reference_record = _history_output_receipt(reference_run_dir, stage)
    candidate_run, candidate_history, candidate_record = _history_output_receipt(candidate_run_dir, stage)
    reference_post = reference_post_dir / stage
    candidate_post = candidate_post_dir / stage
    reference_conversion = json.loads((reference_post / "conversion.json").read_text())
    candidate_conversion = json.loads((candidate_post / "conversion.json").read_text())
    reference_csv = reference_post / reference_conversion["output"]
    candidate_csv = candidate_post / candidate_conversion["output"]

    def provenance(run: dict[str, Any], history: Path, history_record: dict[str, Any],
                   post: Path, conversion: dict[str, Any], csv_path: Path) -> dict[str, Any]:
        titles = Path(str(history) + "_TITLES")
        checks = {
            "conversion_exit_zero": conversion.get("exit_code") == 0,
            "conversion_input_hash_matches_native_receipt": conversion.get("input_sha256") == history_record.get("sha256"),
            "conversion_input_hash_matches_native_file": conversion.get("input_sha256") == sha256(history.read_bytes()),
            "conversion_titles_hash_matches_native_file": conversion.get("titles_sha256") == sha256(titles.read_bytes()),
            "conversion_csv_hash_matches_file": conversion.get("output_sha256") == sha256(csv_path.read_bytes()),
            "run_completed": run.get("execution_complete") is True and run.get("case_sequence_complete") is True,
        }
        return {
            "run_receipt": str((history.parent / "run.json").relative_to(HERE)),
            "native_history": str(history.relative_to(HERE)),
            "native_history_sha256": sha256(history.read_bytes()),
            "native_history_sha256_from_receipt": history_record.get("sha256"),
            "conversion_receipt": str((post / "conversion.json").relative_to(HERE.parents[2])),
            "csv": str(csv_path),
            "csv_sha256": sha256(csv_path.read_bytes()),
            "converter_sha256": conversion.get("converter_sha256"),
            "checks": checks,
            "provenance_valid": all(checks.values()),
        }

    ref_provenance = provenance(reference_run, reference_history, reference_record, reference_post,
                                reference_conversion, reference_csv)
    cand_provenance = provenance(candidate_run, candidate_history, candidate_record, candidate_post,
                                 candidate_conversion, candidate_csv)
    with reference_csv.open(newline="") as stream:
        ref_rows = list(csv.DictReader(stream))
    with candidate_csv.open(newline="") as stream:
        cand_rows = list(csv.DictReader(stream))
    if not ref_rows or not cand_rows:
        raise ValueError(f"empty CSV history in stage {stage}")
    ref_fields = list(ref_rows[0])
    cand_fields = list(cand_rows[0])
    same_columns = ref_fields == cand_fields
    same_rows = len(ref_rows) == len(cand_rows)
    paired_count = min(len(ref_rows), len(cand_rows))
    time_diffs = [abs(_csv_float(a, "time") - _csv_float(b, "time")) for a, b in zip(ref_rows, cand_rows)]
    same_times = same_rows and all(d == 0.0 for d in time_diffs)

    fields: dict[str, Any] = {}
    for field in ref_fields:
        if field not in cand_rows[0]:
            continue
        differences = []
        reference_values = []
        candidate_values = []
        for reference_row, candidate_row in zip(ref_rows, cand_rows):
            reference_value = _csv_float(reference_row, field)
            candidate_value = _csv_float(candidate_row, field)
            reference_values.append(reference_value)
            candidate_values.append(candidate_value)
            differences.append(candidate_value - reference_value)
        scale = max(max((abs(value) for value in reference_values), default=0.0),
                    max((abs(value) for value in candidate_values), default=0.0))
        fields[field] = {
            "samples": paired_count,
            "changed_samples": sum(value != 0.0 for value in differences),
            "max_abs_difference": max((abs(value) for value in differences), default=0.0),
            "rms_difference": math.sqrt(sum(value * value for value in differences) / len(differences)),
            "max_abs_value_either_run": scale,
            "max_difference_over_peak": max((abs(value) for value in differences), default=0.0) / scale if scale else None,
        }

    def reaction_summary(rows: list[dict[str, str]], group_phrase: str, component: str) -> list[float]:
        headers = list(rows[0])
        selected = [header for header in headers if group_phrase in header and header.strip().endswith(component)]
        if not selected:
            raise ValueError(f"no reaction columns for {group_phrase!r} {component}")
        return [sum(_csv_float(row, header) for header in selected) for row in rows]

    reaction_results: dict[str, Any] = {}
    for group_name, phrase in (
        ("upper_driven", "upper driven outer-face reactions"),
        ("lower_anchored", "lower anchored outer-face reactions"),
    ):
        reaction_results[group_name] = {}
        for component in ("REACX", "REACY", "REACZ"):
            ref_total = reaction_summary(ref_rows, phrase, component)
            cand_total = reaction_summary(cand_rows, phrase, component)
            differences = [candidate - reference for reference, candidate in zip(ref_total, cand_total)]
            scale = max(max(map(abs, ref_total), default=0.0), max(map(abs, cand_total), default=0.0))
            reaction_results[group_name][component] = {
                "max_abs_resultant_either_run": scale,
                "max_abs_difference": max(map(abs, differences), default=0.0),
                "rms_difference": math.sqrt(sum(value * value for value in differences) / len(differences)),
                "max_difference_over_peak": max(map(abs, differences), default=0.0) / scale if scale else None,
            }

    def selected_group_difference(needles: tuple[str, ...], suffixes: tuple[str, ...] | None = None) -> dict[str, Any]:
        selected = [field for field in ref_fields if all(needle in field for needle in needles)
                    and (suffixes is None or field.strip().endswith(suffixes))]
        differences = []
        scale = 0.0
        for reference_row, candidate_row in zip(ref_rows, cand_rows):
            for field in selected:
                reference_value = _csv_float(reference_row, field)
                candidate_value = _csv_float(candidate_row, field)
                differences.append(candidate_value - reference_value)
                scale = max(scale, abs(reference_value), abs(candidate_value))
        return {
            "selected_columns": len(selected),
            "max_abs_value_either_run": scale,
            "max_abs_difference": max((abs(value) for value in differences), default=0.0),
            "rms_difference": math.sqrt(sum(value * value for value in differences) / len(differences)) if differences else 0.0,
            "max_difference_over_peak": max((abs(value) for value in differences), default=0.0) / scale if scale else None,
        }

    return {
        "stage": stage,
        "reference": ref_provenance,
        "candidate": cand_provenance,
        "provenance_valid": ref_provenance["provenance_valid"] and cand_provenance["provenance_valid"],
        "sample_count": {"reference": len(ref_rows), "candidate": len(cand_rows)},
        "column_count": {"reference": len(ref_fields), "candidate": len(cand_fields)},
        "columns_identical": same_columns,
        "sample_times_identical_at_csv_precision": same_times,
        "max_sample_time_difference_s": max(time_diffs, default=0.0),
        "matched_rows": paired_count,
        "reaction_resultants": reaction_results,
        "driven_boundary_displacements": selected_group_difference(
            ("upper driven outer-face reactions",), ("DX", "DY", "DZ")),
        "overlay_side_displacements": selected_group_difference(("overlay side-node motion",), ("DX", "DY", "DZ")),
        "cohesive_components_by_field": {
            component: selected_group_difference(("cohesive state and energy density",), (component,))
            for component in ("SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "IE")
        },
        "key_energy_fields": {name: fields[name] for name in (
            "INTERNAL ENERGY", "KINETIC ENERGY", "EXTERNAL WORK", "CONTACT ENERGY",
            "ELASTIC CONTACT ENERGY", "FRICTIONAL CONTACT ENERGY", "DAMPING CONTACT ENERGY ",
        ) if name in fields},
        "all_numeric_fields": fields,
    }


def main() -> None:
    comparisons = [
        compare_pair(
            "candidate05 OpenBLAS overlay versus candidate04 direct control",
            RUNS / "openblas-overlay-04-direct",
            RUNS / "overlay-04-direct",
            ("T01", "T02", "T03"),
        ),
        compare_pair(
            "candidate05 OpenBLAS staged birth versus candidate03 staged control",
            RUNS / "openblas-staged-birth-03",
            RUNS / "staged-bond-birth-03",
            ("T01", "T02", "T03"),
        ),
        compare_pair(
            "candidate05 OpenBLAS staged birth versus candidate04 paper regression",
            RUNS / "openblas-staged-birth-03",
            RUNS / "paper-predictor-04-adhesive-regression",
            ("T01", "T02", "T03"),
        ),
    ]

    ref_root = RUNS / "reference"
    openblas_root = RUNS / "openblas"
    paper07: list[dict[str, Any]] = []
    for reference_dir in sorted(ref_root.glob("paper-predictor-07-*")):
        key = reference_dir.name.removeprefix("paper-predictor-")
        candidate_dir = openblas_root / ("openblas-" + key)
        paper07.append(compare_pair(
            f"paper07 source-provider history {key}",
            candidate_dir,
            reference_dir,
            ("T01",),
        ))

    overlay_csv: list[dict[str, Any]] = []
    overlay_pair = comparisons[0]
    for stage, history_result in zip(("T01", "T02", "T03"), overlay_pair["histories"]):
        if history_result["history"].get("status") != "timestamp_only":
            overlay_csv.append(compare_existing_csv(
                RUNS / "overlay-04-direct", OVERLAY_BASELINE_POST,
                RUNS / "openblas-overlay-04-direct", OVERLAY_OPENBLAS_POST,
                stage,
            ))

    material_history_path = HERE.parents[1] / "paper-predictor-20261023" / "material-history-comparison-07-v-openblas.json"
    material_history = json.loads(material_history_path.read_text()) if material_history_path.is_file() else None

    report = {
        "schema": "cardboard.opencourant.openblas-regression-audit.v1",
        "method": "Direct comparison of full retained native T histories; only the validated 24-byte ASCII wall-clock field at byte offset 96 is normalized. No solver or converter is executed.",
        "timestamp_validation": "Second Fortran record wall-clock field; byte range [96,120); parsed against the native ASCII date format before masking.",
        "candidate05_runtime_identity": str((HERE / "setup/runtime-identity.json").relative_to(HERE)),
        "candidate05_provider_audit": str((HERE / "setup/provider-audit.json").relative_to(HERE)),
        "candidate05_setup": str((HERE / "setup/build-receipt.json").relative_to(HERE)),
        "adhesive_regression_prior_identity": json.loads((HERE / "adhesive-regression.json").read_text()),
        "solver_fixture_pairs": comparisons,
        "paper07_source_provider_pairs": paper07,
        "all_binary_histories_timestamp_only": all(
            history["history"].get("status") in ("timestamp_only",)
            and history["titles"]["byte_identical"]
            for pair in comparisons + paper07
            for history in pair["histories"]
        ),
        "csv_fallback": "Existing converted CSVs are compared only for overlay histories whose native T binaries differ outside the validated timestamp field. The audit verifies each conversion receipt against its native source, TITLES, converter output file, and run receipt; it does not launch the solver or converter.",
        "overlay_existing_csv_comparison": overlay_csv,
        "paper07_material_history_report": {
            "path": str(material_history_path.relative_to(HERE.parents[2])),
            "sha256": sha256(material_history_path.read_bytes()) if material_history_path.is_file() else None,
            "summary": material_history.get("summary") if material_history else None,
            "endpoint_sampling_semantics": material_history.get("endpoint_sampling_semantics") if material_history else None,
            "runtime_timing_diagnostic": material_history.get("runtime_timing_diagnostic") if material_history else None,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT.relative_to(HERE)}")
    print(f"solver fixture groups: {len(comparisons)}; paper07 provider groups: {len(paper07)}")
    print(f"all native histories timestamp-only: {report['all_binary_histories_timestamp_only']}")


if __name__ == "__main__":
    main()
