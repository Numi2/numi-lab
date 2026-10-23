#!/usr/bin/env python3
"""Measure the instrumented synthetic implicit-control histories.

This reader is deliberately case-specific. It verifies the generated /TH
cards, titled CSV columns, ramp load, native impulse semantics, converter
identity, and converter's finite-difference stencil. It refuses generic
``var NN`` columns and never re-differentiates converter-derived REACZ forces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


class MeasurementError(ValueError):
    """Raised when provenance, channels, or timebase cannot support a measure."""


# The stencil below is taken from this official source revision and is enabled
# only for the exact converter binary retained in evidence-20261023. The source
# and binary identities are deliberately recorded separately: the evidence
# does not attest that this binary was built from this source revision.
SUPPORTED_THTOCSV_BINARY_SHA256 = "f8176302da59caba3e373504804beb77715137b5e9e9ef79eaf3231f5e1a1502"
THTOCSV_SOURCE_COMMIT = "1b6bb9200ee7dca899e61c18bd982c3fb1c72fd1"
THTOCSV_SOURCE_SHA256 = "e680547aec4317db158313e5869f644046e8bbe73b8b0b68be18105c46687304"
THTOCSV_SOURCE_URL = (
    "https://github.com/OpenCourant/Tools/blob/"
    + THTOCSV_SOURCE_COMMIT
    + "/output_converters/th_to_csv/src/th_to_csv.c#L1297-L1334"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON receipt must contain an object: {path}")
    return value


def _verify_native_run(run: dict[str, Any]) -> None:
    if run.get("schema") != "cardboard.opencourant.run.v1":
        raise MeasurementError("run receipt is not cardboard.opencourant.run.v1")
    if run.get("execution_complete") is not True:
        raise MeasurementError("native run receipt does not certify execution_complete")
    commands = run.get("commands")
    if not isinstance(commands, list) or not commands:
        raise MeasurementError("native run receipt has no command records")
    for index, command in enumerate(commands):
        if (not isinstance(command, dict) or command.get("exit_code") != 0
                or command.get("normal_termination") is not True
                or command.get("error_termination") is True
                or command.get("timed_out") is True):
            raise MeasurementError(f"native command {index} did not terminate successfully")


def _history_cards(deck: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Read the generated /TH/NODE and /TH/BRIC cards needed by this control."""
    lines = deck.splitlines()
    cards: dict[tuple[str, int], dict[str, Any]] = {}
    index = 0
    while index < len(lines):
        match = re.fullmatch(r"\s*/TH/(NODE|BRIC)/(\d+)\s*", lines[index], re.I)
        if not match:
            index += 1
            continue
        kind, card_id = match.group(1).upper(), int(match.group(2))
        if index + 2 >= len(lines):
            raise MeasurementError(f"truncated /TH/{kind}/{card_id} card")
        title = lines[index + 1].strip()
        variables = tuple(token.upper() for token in lines[index + 2].split())
        if not title or not variables:
            raise MeasurementError(f"/TH/{kind}/{card_id} is missing title or variables")
        entities: list[int] = []
        cursor = index + 3
        while cursor < len(lines) and not lines[cursor].lstrip().startswith("/"):
            fields = lines[cursor].split()
            if fields:
                try:
                    entities.append(int(fields[0]))
                except ValueError:
                    raise MeasurementError(
                        f"invalid entity row in /TH/{kind}/{card_id}: {lines[cursor]!r}")
            cursor += 1
        key = (kind, card_id)
        if key in cards:
            raise MeasurementError(f"duplicate /TH/{kind}/{card_id} card")
        cards[key] = {"title": title, "variables": variables, "entities": tuple(entities)}
        index = cursor
    return cards


def _channel_spec(text: str) -> tuple[str, int, str]:
    match = re.search(r"/TH/(NODE|BRIC)/(\d+)\s+([A-Z0-9]+)\b", text, re.I)
    if not match:
        raise MeasurementError(f"cannot identify explicit /TH channel in metadata: {text!r}")
    return match.group(1).upper(), int(match.group(2)), match.group(3).upper()


def _verify_channel(cards: dict[tuple[str, int], dict[str, Any]], spec: str,
                    expected_kind: str, expected_variable: str) -> dict[str, Any]:
    kind, card_id, variable = _channel_spec(spec)
    if kind != expected_kind or variable != expected_variable:
        raise MeasurementError(f"unexpected declared channel in metadata: {spec!r}")
    card = cards.get((kind, card_id))
    if card is None or variable not in card["variables"]:
        raise MeasurementError(f"declared /TH/{kind}/{card_id} {variable} is absent from starter deck")
    if not card["entities"]:
        raise MeasurementError(f"declared /TH/{kind}/{card_id} has no entities")
    return {**card, "kind": kind, "id": card_id, "variable": variable}


def _card_payload(deck: str, card_name: str) -> list[list[str]]:
    """Return non-comment payload rows for one exact generated card."""
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.strip().casefold() == card_name.casefold()]
    if len(starts) != 1:
        raise MeasurementError(f"expected one {card_name} card in generated deck")
    start = starts[0]
    rows: list[list[str]] = []
    for line in lines[start + 2:]:  # card title is the first line after the keyword
        stripped = line.strip()
        if stripped.startswith("/"):
            break
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(stripped.split())
    return rows


def _verify_linear_ramp(deck: str, expected_end_s: float,
                        expected_total_force_N: float) -> None:
    """Verify this control's 0-to-1 linear function and four-node CLOAD."""
    try:
        function_rows = _card_payload(deck, "/FUNCT/1")
        if len(function_rows) != 2 or any(len(row) != 2 for row in function_rows):
            raise MeasurementError("/FUNCT/1 must contain exactly two time/value rows")
        points = [(float(row[0]), float(row[1])) for row in function_rows]
        load_rows = _card_payload(deck, "/CLOAD/1")
        if len(load_rows) != 1 or len(load_rows[0]) != 7:
            raise MeasurementError("/CLOAD/1 does not match the instrumented uniform-load layout")
        load = load_rows[0]
        group_id, function_id, per_node_force = int(load[4]), int(load[5]), float(load[6])
        direction = load[1].upper()
        group_rows = _card_payload(deck, f"/GRNOD/NODE/{group_id}")
        node_ids = [int(node) for row in group_rows for node in row]
    except (ValueError, IndexError) as exc:
        raise MeasurementError(f"cannot parse the instrumented ramp load: {exc}") from exc
    tolerance = max(1e-12, expected_end_s * 1e-9)
    if (points[0][0] != 0.0 or points[0][1] != 0.0
            or abs(points[1][0] - expected_end_s) > tolerance
            or points[1][1] != 1.0):
        raise MeasurementError("/FUNCT/1 is not the expected linear 0-to-1 ramp")
    if (direction != "Z" or function_id != 1 or sorted(node_ids) != [5, 6, 7, 8]
            or not math.isclose(per_node_force * len(node_ids), expected_total_force_N,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise MeasurementError("/CLOAD/1 does not apply the predicted total +Z force to top nodes 5-8")


def _converter_stencil(times: list[float], total_force_N: float,
                       end_time_s: float) -> list[dict[str, Any]]:
    """Expected ramp averages for the converter's forward/centered/backward differences."""
    if len(times) < 3:
        raise MeasurementError("converter force comparison needs at least three time samples")
    stencils: list[dict[str, Any]] = []
    last_index = len(times) - 1
    for index, sample_time in enumerate(times):
        if index == 0:
            left_index, right_index, method = 0, 1, "forward"
        elif index == last_index:
            left_index, right_index, method = last_index - 1, last_index, "backward"
        else:
            left_index, right_index, method = index - 1, index + 1, "centered"
        start_s, stop_s = times[left_index], times[right_index]
        average_N = total_force_N * (start_s + stop_s) / (2.0 * end_time_s)
        stencils.append({
            "sample_index": index,
            "sample_time_s": sample_time,
            "method": method,
            "window_start_s": start_s,
            "window_end_s": stop_s,
            "expected_force_average_N": average_N,
        })
    return stencils


def _matches_header(header: str, title: str, entity: int, variable: str) -> bool:
    """Match only a title, exact entity ID, and explicit terminal variable token."""
    normalized = " ".join(header.replace("_", " ").split())
    title_normalized = " ".join(title.replace("_", " ").split())
    return re.fullmatch(
        re.escape(title_normalized) + r"\s+" + str(entity) + r"\s+" +
        re.escape(variable) + r"\s*", normalized, re.I) is not None


def _find_columns(headers: list[str], card: dict[str, Any], variable: str) -> dict[int, int]:
    found: dict[int, int] = {}
    for entity in card["entities"]:
        matches = [index for index, header in enumerate(headers)
                   if _matches_header(header, card["title"], entity, variable)]
        if len(matches) != 1:
            raise MeasurementError(
                f"expected one titled {variable} column for entity {entity} under "
                f"/TH/{card['kind']}/{card['id']}; found {len(matches)}")
        found[entity] = matches[0]
    return found


def _read_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    try:
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            headers = next(reader)
            if not headers or headers[0].strip().casefold() != "time":
                raise MeasurementError("converted CSV must have a leading time column")
            if len(set(headers)) != len(headers):
                raise MeasurementError("converted CSV has duplicate column labels")
            rows: list[list[float]] = []
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise MeasurementError(f"CSV row {row_number} has {len(row)} values; expected {len(headers)}")
                values = [float(value) for value in row]
                if not all(math.isfinite(value) for value in values):
                    raise MeasurementError(f"CSV row {row_number} contains a non-finite value")
                rows.append(values)
    except (OSError, StopIteration, csv.Error, ValueError) as exc:
        if isinstance(exc, MeasurementError):
            raise
        raise MeasurementError(f"cannot parse converted CSV {path}: {exc}") from exc
    if len(rows) < 2:
        raise MeasurementError("converted CSV must contain at least two samples")
    return headers, rows


def _validate_timebase(rows: list[list[float]], end_time: float,
                       requested_interval: float) -> list[float]:
    times = [row[0] for row in rows]
    if abs(times[0]) > max(1e-12, requested_interval * 1e-6):
        raise MeasurementError(f"history starts at {times[0]:.9g} s, not at t=0")
    if abs(times[-1] - end_time) > max(1e-10, requested_interval * 0.5):
        raise MeasurementError(
            f"history ends at {times[-1]:.9g} s; expected {end_time:.9g} s")
    deltas = [right - left for left, right in zip(times, times[1:])]
    if any(delta <= 0 for delta in deltas):
        raise MeasurementError("history time values are not strictly increasing")
    # Some implicit T01s contain accepted solver steps only; /TFILE's requested
    # interval is not a promise that a sample was interpolated at each interval.
    # Use actual CSV times for any derivative or work calculation and retain the
    # requested-vs-observed cadence as metadata.
    return times


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / abs(target)


def _gate(value: float, target: float, maximum_relative_error: float) -> dict[str, Any]:
    error = _relative_error(value, target)
    return {"measured": value, "target": target,
            "relative_error": error,
            "maximum_relative_error": maximum_relative_error,
            "pass": error <= maximum_relative_error}


def _conversion_paths(run_dir: Path, receipt: dict[str, Any], csv_override: Path | None,
                      receipt_path: Path) -> tuple[Path, Path, str]:
    if receipt.get("schema") != "cardboard.opencourant.postprocess.v1":
        raise MeasurementError("conversion receipt is not cardboard.opencourant.postprocess.v1")
    if receipt.get("receipt_created_after_conversion") is not True:
        raise MeasurementError("conversion receipt does not attest post-conversion creation")
    if "exit_code" in receipt and receipt.get("exit_code") != 0:
        raise MeasurementError("converter did not exit successfully")
    output_name = receipt.get("output")
    if not isinstance(output_name, str) or Path(output_name).name != output_name:
        raise MeasurementError("conversion receipt output must be a run-local filename")
    csv_path = csv_override or (receipt_path.parent / output_name)
    input_relative = receipt.get("input_relative_path")
    if not isinstance(input_relative, str):
        raise MeasurementError("conversion receipt has no input_relative_path")
    # Existing receipts use paths relative to the evidence root (runs/<case>/...)
    # or run-local basenames. Resolve both without permitting parent traversal.
    relative = Path(input_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise MeasurementError("conversion receipt input path must be relative")
    candidates = [run_dir / relative.name, run_dir.parent.parent / relative]
    t01_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return csv_path, t01_path, output_name


def measure_run(run_dir: Path, *, csv_path: Path | None = None,
                conversion_receipt_path: Path | None = None) -> dict[str, Any]:
    """Collect titled displacement, reaction, and internal-energy measures.

    ``run_dir`` is the individual exported run folder containing ``case.json``,
    ``run.json``, the starter deck, and T01 outputs. A conversion receipt and
    CSV are required for a numeric result. This function never runs the solver
    or converter.
    """
    run_dir = run_dir.resolve()
    case_path, run_path = run_dir / "case.json", run_dir / "run.json"
    case, run = _read_json(case_path), _read_json(run_path)
    _verify_native_run(run)
    native = case.get("native_outputs")
    if not isinstance(native, dict):
        raise MeasurementError("case metadata lacks native_outputs")
    starter_name = case.get("starter")
    if not isinstance(starter_name, str) or Path(starter_name).name != starter_name:
        raise MeasurementError("case starter must be a run-local filename")
    try:
        deck = (run_dir / starter_name).read_text()
    except OSError as exc:
        raise MeasurementError(f"cannot read generated starter deck: {exc}") from exc
    cards = _history_cards(deck)
    force_text = str(native.get("time_history", {}).get("force_channel", ""))
    displacement_text = str(native.get("time_history", {}).get("displacement_channel", ""))
    energy_text = str(native.get("time_history", {}).get("energy_channel", ""))
    force_card = _verify_channel(cards, force_text, "NODE", "REACZ")
    displacement_card = _verify_channel(cards, displacement_text, "NODE", "DZ")
    energy_card = _verify_channel(cards, energy_text, "BRIC", "IE")
    if set(force_card["entities"]) != {1, 2, 3, 4}:
        raise MeasurementError("implicit-control reaction card must measure bottom nodes 1-4")

    receipt_path = conversion_receipt_path or (run_dir / "conversion.json")
    receipt = _read_json(receipt_path)
    csv_path, t01_path, output_name = _conversion_paths(run_dir, receipt, csv_path, receipt_path)
    for path, expected, label in (
            (t01_path, receipt.get("input_sha256"), "T01 input"),
            (csv_path, receipt.get("output_sha256"), "converted CSV")):
        if not path.is_file():
            raise MeasurementError(f"missing {label}: {path}")
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise MeasurementError(f"{label} SHA-256 does not match conversion receipt")
    titles_path = Path(str(t01_path) + "_TITLES")
    if not titles_path.is_file():
        raise MeasurementError("native T01_TITLES sidecar is missing")
    title_hash = _sha256(titles_path)
    run_outputs = run.get("outputs")
    if not isinstance(run_outputs, dict):
        raise MeasurementError("run receipt lacks output hashes for native histories")
    for path, label in ((t01_path, "T01"), (titles_path, "T01_TITLES")):
        record = next((value for name, value in run_outputs.items()
                       if Path(name).name == path.name), None)
        if not isinstance(record, dict) or record.get("sha256") != _sha256(path):
            raise MeasurementError(f"{label} SHA-256 does not match native run receipt")
    converter_sha = receipt.get("converter_sha256")
    if not isinstance(converter_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", converter_sha):
        raise MeasurementError("conversion receipt lacks a valid converter SHA-256")
    if converter_sha.casefold() != SUPPORTED_THTOCSV_BINARY_SHA256:
        raise MeasurementError("unsupported th_to_csv binary; the REACZ stencil is version-bound")
    argv = receipt.get("argv")
    if (not isinstance(argv, list) or len(argv) < 2 or not isinstance(argv[0], str)
            or "th_to_csv" not in Path(argv[0]).name.casefold()):
        raise MeasurementError("conversion receipt argv does not identify official th_to_csv")
    alias = receipt.get("input_alias")
    if alias is not None:
        if not isinstance(alias, str) or Path(alias).name != alias or Path(argv[-1]).name != alias:
            raise MeasurementError("converter input alias does not match its argv basename")
        alias_t01 = receipt_path.parent / alias
        alias_titles = Path(str(alias_t01) + "_TITLES")
        if (not alias_t01.is_file() or _sha256(alias_t01) != receipt["input_sha256"]
                or not alias_titles.is_file() or _sha256(alias_titles) != title_hash):
            raise MeasurementError("converter alias and T01_TITLES are not byte-identical to native outputs")
    elif Path(argv[-1]).name != t01_path.name:
        raise MeasurementError("converter argv does not name the attested T01 input")
    if receipt.get("titles_sha256") is not None and receipt["titles_sha256"] != title_hash:
        raise MeasurementError("conversion receipt TITLES hash does not match native output")
    if not receipt_path.is_file():
        raise MeasurementError("conversion receipt disappeared during measurement")

    headers, rows = _read_csv(csv_path)
    expected_end = float(case["expected_end_time_s"])
    interval = float(native["history_interval_s"])
    if not (math.isfinite(expected_end) and expected_end > 0 and math.isfinite(interval) and interval > 0):
        raise MeasurementError("case metadata has invalid history time bounds")
    times = _validate_timebase(rows, expected_end, interval)

    displacement_cols = _find_columns(headers, displacement_card, "DZ")
    energy_cols = _find_columns(headers, energy_card, "IE")
    if len(displacement_cols) != 4 or set(displacement_cols) != {5, 6, 7, 8}:
        raise MeasurementError("implicit-control displacement card must measure top nodes 5-8")
    if set(energy_cols) != {1}:
        raise MeasurementError("implicit-control energy card must measure brick 1")
    dz_series = [sum(row[displacement_cols[node]] for node in sorted(displacement_cols)) / 4
                 for row in rows]
    energy_series = [row[energy_cols[1]] for row in rows]
    prediction = case.get("prediction", {})
    gates = case.get("gates", {})
    disp_target = float(prediction["mean_upper_DZ_mm"])
    force_target = float(prediction["sum_bottom_reaction_magnitude_N"])
    total_load_N = float(prediction["total_applied_force_N"])
    energy_target = float(prediction["elastic_work_N_mm"])
    if not math.isclose(force_target, total_load_N, rel_tol=0.0, abs_tol=1e-12):
        raise MeasurementError("reaction endpoint prediction differs from prescribed ramp load")
    _verify_linear_ramp(deck, expected_end, total_load_N)
    time_history = native.get("time_history", {})
    force_semantics = str(time_history.get("force_semantics", ""))
    if "impulse in T01" not in force_semantics or "th_to_csv" not in force_semantics:
        raise MeasurementError("case metadata does not declare T01 impulse and th_to_csv force conversion")

    reacz_cols: dict[int, int] = {}
    for entity in force_card["entities"]:
        matches = [i for i, header in enumerate(headers)
                   if _matches_header(header, force_card["title"], entity, "REACZ")]
        if len(matches) > 1:
            raise MeasurementError(
                f"ambiguous titled REACZ force columns for bottom node {entity}")
        if matches:
            reacz_cols[entity] = matches[0]
    if reacz_cols and set(reacz_cols) != set(force_card["entities"]):
        raise MeasurementError("only part of the titled bottom REACZ channel is present")
    reaction_gate: dict[str, Any] | None = None
    endpoint_diagnostic: dict[str, Any] | None = None
    reaction_signed_N: list[float] | None = None
    reaction_magnitude_N: list[float] | None = None
    stencils: list[dict[str, Any]] | None = None
    reaction_tolerance = float(gates.get("relative_reaction_error_max", 0.01))
    if reacz_cols:
        reaction_signed_N = [sum(row[reacz_cols[node]] for node in sorted(reacz_cols))
                             for row in rows]
        reaction_magnitude_N = [abs(value) for value in reaction_signed_N]
        stencils = _converter_stencil(times, total_load_N, expected_end)
        reaction_sample_results = []
        for sample, measured in zip(stencils, reaction_signed_N):
            expected = sample["expected_force_average_N"]
            reaction_sample_results.append({
                **sample,
                "expected_signed_reaction_N": -expected,
                "measured_signed_reaction_N": measured,
                "measured_force_magnitude_N": abs(measured),
                "relative_error": _relative_error(measured, -expected),
            })
        worst_sample = max(reaction_sample_results, key=lambda sample: sample["relative_error"])
        reaction_gate = {
            "estimand": "whole converted reaction curve against the prescribed ramp averaged over each converter difference window",
            "maximum_sample_relative_error": worst_sample["relative_error"],
            "worst_sample_index": worst_sample["sample_index"],
            "worst_sample_time_s": worst_sample["sample_time_s"],
            "maximum_relative_error": reaction_tolerance,
            "pass": worst_sample["relative_error"] <= reaction_tolerance,
            "samples": reaction_sample_results,
        }
        endpoint_diagnostic = _gate(reaction_magnitude_N[-1], force_target, reaction_tolerance)
        endpoint_diagnostic.update({
            "gate_applicable": False,
            "status": "invalid_endpoint_interpretation_diagnostic_only",
            "reason": "the final REACZ CSV value is a backward-difference average over the final interval, not the 1 N endpoint load",
        })

    # Input work uses exact interval-average values of the verified linear
    # prescribed ramp. It does not integrate converter-smoothed reaction values.
    applied_force_interval_averages_N = [
        total_load_N * (left + right) / (2.0 * expected_end)
        for left, right in zip(times, times[1:])
    ]
    applied_work = sum(
        force_average * (dz_series[index + 1] - dz_series[index])
        for index, force_average in enumerate(applied_force_interval_averages_N)
    )
    reaction_trapezoid_work = (sum(
        0.5 * (reaction_magnitude_N[index] + reaction_magnitude_N[index + 1]) *
        (dz_series[index + 1] - dz_series[index])
        for index in range(len(reaction_magnitude_N) - 1)
    ) if reaction_magnitude_N is not None else None)

    disp_gate = _gate(dz_series[-1], disp_target,
                      float(gates.get("relative_displacement_error_max", 0.01)))
    energy_gate = _gate(energy_series[-1], energy_target, 0.01)
    reaction: dict[str, Any] = {
        "semantics": "T01 REACZ impulse converted to reaction force by the attested th_to_csv binary; CSV retains REACZ label",
        "source": "official_converter_finite_difference_of_T01_REACZ_impulse" if reacz_cols else None,
        "native_T01_REACZ_semantics": "impulse" if reacz_cols else None,
        "converted_force_stencil": "forward at first sample, centered at interior samples, backward at final sample" if reacz_cols else None,
        "target_endpoint_magnitude_N": force_target,
        "converter_force_signed_sum_series_N": reaction_signed_N,
        "converter_force_magnitude_series_N": reaction_magnitude_N,
        "final_signed_sum_N": reaction_signed_N[-1] if reaction_signed_N is not None else None,
        "final_magnitude_N": reaction_magnitude_N[-1] if reaction_magnitude_N is not None else None,
        "final_stencil_expected_average_N": stencils[-1]["expected_force_average_N"] if stencils else None,
        "gate": reaction_gate,
        "endpoint_target_diagnostic": endpoint_diagnostic,
        "status": "inconclusive" if reaction_gate is None else ("pass" if reaction_gate["pass"] else "fail"),
    }
    work_gate = _gate(applied_work, energy_target, 0.01)
    overall = "inconclusive"
    if reaction_gate is not None:
        overall = ("pass" if all((disp_gate["pass"], energy_gate["pass"],
                                   reaction_gate["pass"], work_gate["pass"])) else "fail")
    return {
        "schema": "cardboard.opencourant.measurement.v1",
        "case_id": case.get("id"),
        "physical_qualification": False,
        "interpretation": "synthetic implicit-control measurement; not material calibration",
        "overall_status": overall,
        "timebase": {"start_s": times[0], "end_s": times[-1], "sample_count": len(times),
                     "requested_interval_s": interval,
                     "minimum_observed_interval_s": min(b - a for a, b in zip(times, times[1:])),
                     "maximum_observed_interval_s": max(b - a for a, b in zip(times, times[1:])),
                     "sparse_vs_requested_interval": max(b - a for a, b in zip(times, times[1:])) > interval * 1.5,
                     "verified": True},
        "displacement": {"channel": displacement_text, "units": "mm",
                         "node_ids": sorted(displacement_cols),
                         "node_final_DZ_mm": {str(node): rows[-1][column]
                                               for node, column in sorted(displacement_cols.items())},
                         "mean_upper_DZ_mm": dz_series[-1], "gate": disp_gate},
        "reaction": reaction,
        "energy": {"channel": energy_text, "units": "N mm",
                   "brick_1_IE_final_N_mm": energy_series[-1], "gate": energy_gate},
        "external_work_crosscheck": {
            "method": "sum exact linear-ramp interval-average CLOAD times measured mean-upper-DZ increments",
            "applied_force_interval_average_N": applied_force_interval_averages_N,
            "value_N_mm": applied_work,
            "target_N_mm": energy_target,
            "gate": work_gate,
            "relative_difference_to_final_IE": _relative_error(applied_work, energy_series[-1]),
            "status": "pass" if work_gate["pass"] else "fail",
        },
        "reaction_work_quadrature_diagnostic": {
            "method": "trapezoid of converter-derived force values at their CSV-labeled times",
            "value_N_mm": reaction_trapezoid_work,
            "gate_applicable": False,
            "status": "inconclusive",
            "reason": "the converter force samples use different forward, centered, and backward impulse-difference windows; this time-label quadrature does not reconstruct reaction work",
        },
        "provenance": {
            "run_receipt": "run.json", "case_metadata": "case.json",
            "starter_deck": starter_name,
            "starter_sha256": _sha256(run_dir / starter_name),
            "t01": str(t01_path.relative_to(run_dir)) if t01_path.is_relative_to(run_dir) else str(t01_path),
            "t01_sha256": receipt["input_sha256"],
            "t01_titles": str(titles_path.relative_to(run_dir)) if titles_path.is_relative_to(run_dir) else str(titles_path),
            "t01_titles_sha256": title_hash,
            "converter_sha256": converter_sha,
            "converter_source": {
                "url": THTOCSV_SOURCE_URL,
                "commit": THTOCSV_SOURCE_COMMIT,
                "sha256": THTOCSV_SOURCE_SHA256,
                "algorithm_lines": "1297-1334",
                "binary_source_build_linkage_attested": False,
            },
            "conversion_argv": argv,
            "conversion_receipt": str(receipt_path),
            "converted_csv": output_name,
            "converted_csv_sha256": receipt["output_sha256"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="instrumented run folder with case.json and run.json")
    parser.add_argument("--csv", type=Path, help="converted th_to_csv output (default from conversion receipt)")
    parser.add_argument("--conversion-receipt", type=Path,
                        help="conversion receipt (default: RUN_DIR/conversion.json)")
    parser.add_argument("--output", type=Path, help="write JSON here (default: RUN_DIR/measurement.json)")
    args = parser.parse_args(argv)
    try:
        result = measure_run(args.run_dir, csv_path=args.csv,
                             conversion_receipt_path=args.conversion_receipt)
    except MeasurementError as exc:
        print(f"measurement unavailable: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        (args.run_dir / "measurement.json").write_text(serialized)
    print(serialized, end="")
    return 0 if result["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
