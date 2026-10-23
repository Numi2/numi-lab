#!/usr/bin/env python3
"""Measure the four retained direct-solver LAW112 source-fit coupons.

This is deliberately scoped to the one-element tensile-transfer study. It
checks the run/conversion receipts, complete finite T01 histories, achieved
face motion, nominal stress reconstructed from native Cauchy stress and the
measured face area vector, and work/energy consistency. Passing these checks
is an implementation/transfer result, not physical qualification of paper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import cardboard_opencourant_material as material


SCHEMA = "cardboard.opencourant.material-measurement.v1"
RUN_PREFIX = "paper-predictor-03-direct-law112-"
EXPECTED_CASES = {
    f"law112-{role}-{axis.lower()}-tensile-fit-transfer-implicit"
    for role in ("liner", "medium") for axis in ("MD", "CD")
}
STRESS_FIELDS = ("SX", "SY", "SZ", "SXY", "SYZ", "SXZ")
DISPLACEMENT_FIELDS = ("DX", "DY", "DZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _close(a: float, b: float, *, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def _column(header: list[str], group: str, field: str) -> str:
    hits = [name for name in header
            if group in name and (
                name.strip() == field if not group else name.rstrip().endswith(" " + field)
            )]
    if len(hits) != 1:
        raise ValueError(f"expected one {field} column in {group!r}; found {len(hits)}")
    return hits[0]


def _native_columns(header: list[str], group: int) -> dict[str, str]:
    prefix = f"native displacement and support reaction {group}"
    return {field: _column(header, prefix, field) for field in DISPLACEMENT_FIELDS}


def _stress_columns(header: list[str]) -> dict[str, str]:
    prefix = "native stress, strain and internal energ 1"
    return {field: _column(header, prefix, field) for field in STRESS_FIELDS}


def _parse_nodes(deck: Path) -> dict[int, tuple[float, float, float]]:
    lines = deck.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == "/NODE") + 1
    except StopIteration as exc:
        raise ValueError(f"missing /NODE card: {deck}") from exc
    nodes: dict[int, tuple[float, float, float]] = {}
    for line in lines[start:]:
        if line.strip().startswith("/"):
            break
        fields = line.split()
        if len(fields) >= 4:
            try:
                node_id = int(fields[0])
                xyz = tuple(_finite(v, f"node {node_id} coordinate") for v in fields[1:4])
            except (ValueError, TypeError):
                continue
            nodes[node_id] = xyz  # type: ignore[assignment]
    if len(nodes) != 8:
        raise ValueError(f"expected the eight nodes of the one-brick coupon; found {len(nodes)}")
    return nodes


def _parse_node_history_ids(deck: Path, channel: int) -> list[int]:
    lines = deck.read_text(encoding="utf-8", errors="strict").splitlines()
    marker = f"/TH/NODE/{channel}".upper()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == marker) + 1
    except StopIteration as exc:
        raise ValueError(f"missing {marker} card: {deck}") from exc
    ids: list[int] = []
    for line in lines[start:]:
        if line.strip().startswith("/"):
            break
        match = re.fullmatch(r"\s*(\d+)\s+\d+\s*", line)
        if match:
            ids.append(int(match.group(1)))
    if len(ids) != 8:
        raise ValueError(f"expected 8 nodes in {marker}; found {len(ids)}")
    return ids


def _parse_brick_nodes(deck: Path) -> list[int]:
    lines = deck.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().upper().startswith("/BRICK/")) + 1
    except StopIteration as exc:
        raise ValueError(f"missing /BRICK card: {deck}") from exc
    for line in lines[start:]:
        if line.strip().startswith("/"):
            break
        values = line.split()
        if len(values) >= 9:
            try:
                return [int(v) for v in values[1:9]]
            except ValueError:
                continue
    raise ValueError(f"could not read brick connectivity: {deck}")


def _card_body(deck: Path, marker: str) -> list[str]:
    lines = deck.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().upper() == marker.upper()) + 1
    except StopIteration as exc:
        raise ValueError(f"missing {marker} card: {deck}") from exc
    end = next((i for i in range(start, len(lines)) if lines[i].strip().startswith("/")), len(lines))
    return lines[start:end]


def _parse_drive(deck: Path, expected_axis: str, expected_loaded_nodes: list[int]) -> dict[str, Any]:
    """Read the narrow constant-rate /IMPVEL + /FUNCT schedule in these coupons."""
    impvel_cards = [line.strip() for line in deck.read_text(encoding="utf-8").splitlines()
                    if line.strip().upper().startswith("/IMPVEL/")]
    if len(impvel_cards) != 1:
        raise ValueError(f"expected one /IMPVEL card in {deck}")
    impvel = _card_body(deck, impvel_cards[0])
    if len(impvel) < 3:
        raise ValueError(f"incomplete /IMPVEL card in {deck}")
    id_fields = impvel[1].split()
    scale_fields = impvel[2].split()
    if len(id_fields) < 7 or len(scale_fields) < 4:
        raise ValueError(f"unsupported /IMPVEL field layout in {deck}")
    function_id = int(id_fields[0])
    direction = id_fields[1].upper()
    group_id = int(id_fields[4])
    if direction != {"MD": "X", "CD": "Y"}[expected_axis]:
        raise ValueError(f"/IMPVEL direction does not match material axis in {deck}")
    function_card = f"/FUNCT/{function_id}"
    function_body = _card_body(deck, function_card)
    knots: list[tuple[float, float]] = []
    for line in function_body[1:]:
        values = line.split()
        if len(values) >= 2:
            try:
                x, y = float(values[0]), float(values[1])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y):
                knots.append((x, y))
    if len(knots) < 2 or any(b[0] <= a[0] for a, b in zip(knots, knots[1:])):
        raise ValueError(f"unsupported /FUNCT/{function_id} curve in {deck}")
    scales = tuple(_finite(x, f"{deck.name} /IMPVEL scale/time") for x in scale_fields[:4])
    x_scale, velocity_scale, start_time, stop_time = scales
    if x_scale <= 0.0 or velocity_scale <= 0.0 or stop_time <= start_time:
        raise ValueError(f"unsupported /IMPVEL scales/time range in {deck}")
    group_lines = _card_body(deck, f"/GRNOD/NODE/{group_id}")
    members: list[int] = []
    for line in group_lines[1:]:
        for word in line.split():
            try:
                members.append(int(word))
            except ValueError:
                continue
    if set(members) != set(expected_loaded_nodes):
        raise ValueError(f"/IMPVEL group {group_id} does not match the loaded face in {deck}")
    return {
        "function_id": function_id,
        "direction": direction,
        "node_group_id": group_id,
        "abscissa_scale": x_scale,
        "velocity_scale_mm_s": velocity_scale,
        "active_start_s": start_time,
        "active_stop_s": stop_time,
        "function_knots": knots,
        "basis": "deck /IMPVEL field values and referenced /FUNCT points",
    }


def _interpolate_function(knots: list[tuple[float, float]], time: float) -> float:
    if time <= knots[0][0]:
        return knots[0][1]
    if time >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= time <= x1:
            f = (time - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    raise ValueError("function time falls outside its knots")


def _prescribed_displacement(drive: dict[str, Any], sample_time: float) -> float:
    """Integrate the referenced linearly interpolated velocity function to sample_time."""
    start = float(drive["active_start_s"])
    stop = min(sample_time, float(drive["active_stop_s"]))
    if stop <= start:
        return 0.0
    x_scale = float(drive["abscissa_scale"])
    knots = [(float(x), float(y)) for x, y in drive["function_knots"]]
    breakpoints = [start, stop]
    breakpoints.extend(x * x_scale for x, _ in knots if start < x * x_scale < stop)
    breakpoints.sort()
    scale = float(drive["velocity_scale_mm_s"])
    return sum(
        0.5 * scale * (
            _interpolate_function(knots, a / x_scale) +
            _interpolate_function(knots, b / x_scale)
        ) * (b - a)
        for a, b in zip(breakpoints, breakpoints[1:])
    )


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _face_area_vector(
    positions: dict[int, tuple[float, float, float]],
    reference: dict[int, tuple[float, float, float]],
    face_ids: list[int],
    axial_axis: int,
) -> tuple[float, float, float]:
    """Oriented area vector of a rectangular coupon face, ordered from reference geometry."""
    transverse = [axis for axis in range(3) if axis != axial_axis]
    lows = [min(reference[n][axis] for n in face_ids) for axis in transverse]
    highs = [max(reference[n][axis] for n in face_ids) for axis in transverse]
    corners = [
        (lows[0], lows[1]), (highs[0], lows[1]),
        (highs[0], highs[1]), (lows[0], highs[1]),
    ]
    ordered: list[int] = []
    for corner in corners:
        matches = [n for n in face_ids
                   if _close(reference[n][transverse[0]], corner[0])
                   and _close(reference[n][transverse[1]], corner[1])]
        if len(matches) != 1:
            raise ValueError("loaded face is not a four-corner rectangle")
        ordered.append(matches[0])
    p = [positions[n] for n in ordered]
    area_vector = tuple(0.5 * x for x in (
        _cross(_sub(p[1], p[0]), _sub(p[2], p[0]))[0] +
        _cross(_sub(p[2], p[0]), _sub(p[3], p[0]))[0],
        _cross(_sub(p[1], p[0]), _sub(p[2], p[0]))[1] +
        _cross(_sub(p[2], p[0]), _sub(p[3], p[0]))[1],
        _cross(_sub(p[1], p[0]), _sub(p[2], p[0]))[2] +
        _cross(_sub(p[2], p[0]), _sub(p[3], p[0]))[2],
    ))
    if area_vector[axial_axis] < 0.0:
        area_vector = tuple(-x for x in area_vector)  # type: ignore[assignment]
    if area_vector[axial_axis] <= 0.0:
        raise ValueError("loaded-face area vector has no positive axial component")
    return area_vector


def nominal_axial_stress(
    cauchy: tuple[float, float, float, float, float, float],
    current_area_vector: tuple[float, float, float],
    reference_area: float,
    axis: int,
) -> float:
    """Return the first-Piola axial component from Cauchy traction on a reference face."""
    sx, sy, sz, sxy, syz, sxz = cauchy
    sigma = ((sx, sxy, sxz), (sxy, sy, syz), (sxz, syz, sz))
    if reference_area <= 0.0:
        raise ValueError("reference_area must be positive")
    return sum(sigma[axis][j] * current_area_vector[j] for j in range(3)) / reference_area


def source_fit_at_engineering_strain(curve: dict[str, Any], strain: float) -> dict[str, float]:
    """Evaluate the published bilinear fit and H-pixel envelope at engineering strain."""
    E = _finite(curve["elastic_modulus_MPa"], "fit elastic modulus")
    anchor = curve["yield_anchor"]
    eps_y = _finite(anchor["strain_mm_per_mm"], "fit yield strain")
    S0 = _finite(anchor["stress_MPa"], "fit yield stress")
    H = _finite(curve["plastic_hardening_modulus_MPa"], "fit H")
    H_low, H_high = (_finite(x, "fit H pixel bound")
                     for x in curve["plastic_hardening_modulus_pixel_bound_MPa"])

    def stress_for_h(hardening: float) -> float:
        tangent = E * hardening / (E + hardening)
        return S0 if strain <= eps_y else S0 + tangent * (strain - eps_y)

    return {
        "central_MPa": stress_for_h(H),
        "pixel_bound_low_MPa": stress_for_h(H_low),
        "pixel_bound_high_MPa": stress_for_h(H_high),
    }


def _verify_hash_inventory(root: Path, inventory: dict[str, Any], name: str) -> bool:
    for rel, record in inventory.items():
        path = root / rel
        if not path.is_file():
            raise ValueError(f"{name} file missing: {rel}")
        expected = record.get("sha256")
        if expected and _sha256(path) != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {rel}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ValueError(f"{name} size mismatch: {rel}")
    return True


def _validate_history(times: list[float], expected_end_time: float) -> dict[str, Any]:
    if len(times) < 2:
        raise ValueError("T01 history needs at least two samples")
    if not _close(times[0], 0.0, rel=0.0, abs_=1e-10):
        raise ValueError(f"T01 history does not start at t=0: {times[0]}")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("T01 sample times are not strictly increasing")
    end_error = times[-1] - expected_end_time
    gaps = [b - a for a, b in zip(times, times[1:])]
    gap = abs(end_error)
    reaches_horizon = gap <= max(gaps) + 1.1e-6
    exact_endpoint = gap <= 1.1e-6
    if not reaches_horizon:
        raise ValueError(f"T01 history ends more than one accepted interval from requested endpoint: {times[-1]} vs {expected_end_time}")
    return {
        "sample_count": len(times),
        "first_time_s": times[0],
        "last_time_s": times[-1],
        "expected_end_time_s": expected_end_time,
        "endpoint_time_error_s": end_error,
        "maximum_sample_gap_s": max(gaps),
        "median_sample_gap_s": sorted(gaps)[len(gaps) // 2],
        "reaches_requested_horizon_within_one_sample_interval": reaches_horizon,
        "exact_requested_endpoint_frame": exact_endpoint,
        "accepted_steps_are_nondecreasing": True,
    }


def _read_history(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _vector_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _coupon_result(study_root: Path, run_dir: Path, case_suite_dir: str) -> dict[str, Any]:
    run = _json(run_dir / "run.json")
    case = _json(run_dir / "case.json")
    source_record = _json(run_dir / "source-record.json")
    source_case = source_record.get("case")
    if not isinstance(source_case, dict):
        raise ValueError(f"source record has no case metadata: {run_dir}")
    case_id = str(case["id"])
    if case_id not in EXPECTED_CASES or run.get("case_id") != case_id:
        raise ValueError(f"unexpected or mismatched case identity: {case_id}")
    if source_case.get("id") != case_id:
        raise ValueError(f"source record case identity mismatch: {case_id}")

    role = str(source_record["material_role"])
    axis_label = str(source_case["axis"])
    axis = {"MD": 0, "CD": 1}.get(axis_label)
    if axis is None:
        raise ValueError(f"unsupported coupon axis: {axis_label}")
    expected_component = ("SX", "SY")[axis]
    if source_case.get("stress_component") != expected_component:
        raise ValueError(f"coupon axis/stress component mismatch for {case_id}")

    # Compare the embedded case source record to the source module's hash-bound API.
    authoritative = material.source_record(role)
    branch = authoritative["hardening_transfer"]["branches"][axis_label]
    source_curve = source_case["source_curve"]
    for key in ("elastic_modulus_MPa", "plastic_hardening_modulus_MPa", "nominal_transfer_prediction_MPa"):
        if not _close(float(source_curve[key]), float(branch[key]), rel=1e-10, abs_=1e-10):
            raise ValueError(f"source-fit {key} drift for {case_id}")
    if source_curve["digitized_fit_endpoint"] != branch["digitized_fit_endpoint"]:
        raise ValueError(f"source-fit endpoint record drift for {case_id}")

    # The run inventory makes the execution evidence immutable and checks each
    # copied input/output before interpreting the measurement.
    _verify_hash_inventory(run_dir, run.get("inputs", {}), "run input")
    _verify_hash_inventory(run_dir, run.get("outputs", {}), "run output")
    if not run.get("execution_complete") or not run.get("case_sequence_complete"):
        raise ValueError(f"native run is incomplete: {case_id}")
    commands = run.get("commands", [])
    if len(commands) < 2 or not all(c.get("exit_code") == 0 and c.get("normal_termination")
                                    and not c.get("error_termination") for c in commands):
        raise ValueError(f"native starter/engine did not both terminate normally: {case_id}")

    case_dir = study_root / case_suite_dir / case_id
    if not case_dir.is_dir():
        raise ValueError(f"missing source case directory: {case_dir}")
    if _sha256(run_dir / "case.json") != run["case_sha256"]:
        raise ValueError(f"case manifest hash mismatch: {case_id}")
    input_rad = sorted(case_dir.glob("*_0000.rad"))
    if len(input_rad) != 1:
        raise ValueError(f"expected one starter deck for {case_id}")
    deck = input_rad[0]
    for source_path in (case_dir / "case.json", case_dir / "source-record.json", deck):
        inventory_record = run.get("inputs", {}).get(source_path.name)
        if not inventory_record or _sha256(source_path) != inventory_record.get("sha256"):
            raise ValueError(f"source case and executed input differ: {case_id}/{source_path.name}")
    reference_nodes = _parse_nodes(deck)
    drive = _parse_drive(deck, axis_label, [int(x) for x in source_case["loaded_face_nodes"]])
    node_history = _parse_node_history_ids(deck, int(source_case["history"]["node_channel"]))
    if set(node_history) != set(reference_nodes):
        raise ValueError(f"T01 displacement channel does not cover coupon nodes: {case_id}")
    fixed_ids = [int(x) for x in source_case["fixed_face_nodes"]]
    loaded_ids = [int(x) for x in source_case["loaded_face_nodes"]]
    brick = _parse_brick_nodes(deck)
    if not set(fixed_ids + loaded_ids).issubset(reference_nodes):
        raise ValueError(f"coupon boundary node references are invalid: {case_id}")

    run_name = run_dir.name
    post_dir = study_root / "post" / run_name / "T01"
    receipt = _json(post_dir / "conversion.json")
    csv_path = post_dir / str(receipt.get("output", "probeT01.csv"))
    t01_files = list(run_dir.glob("*T01"))
    titles_files = list(run_dir.glob("*T01_TITLES"))
    if len(t01_files) != 1 or len(titles_files) != 1:
        raise ValueError(f"expected one native T01 and titles file: {case_id}")
    if not receipt.get("receipt_created_after_conversion") or receipt.get("exit_code") != 0:
        raise ValueError(f"conversion receipt is incomplete: {case_id}")
    if _sha256(t01_files[0]) != receipt.get("input_sha256"):
        raise ValueError(f"converted T01 input hash mismatch: {case_id}")
    converted_input = post_dir / str(receipt.get("input_alias", "probeT01"))
    if not converted_input.is_file() or _sha256(converted_input) != receipt.get("input_sha256"):
        raise ValueError(f"postprocess T01 copy hash mismatch: {case_id}")
    if _sha256(csv_path) != receipt.get("output_sha256"):
        raise ValueError(f"converted CSV output hash mismatch: {case_id}")
    if _sha256(titles_files[0]) != receipt.get("titles_sha256"):
        raise ValueError(f"T01 titles hash mismatch: {case_id}")
    if _sha256(post_dir / "probeT01_TITLES") != receipt.get("titles_sha256"):
        raise ValueError(f"converted T01 titles hash mismatch: {case_id}")

    header, rows = _read_history(csv_path)
    if not rows:
        raise ValueError(f"empty converted T01 CSV: {case_id}")
    stress_cols = _stress_columns(header)
    time_col = _column(header, "", "time")
    energy_col = _column(header, "", "INTERNAL ENERGY")
    external_col = _column(header, "", "EXTERNAL WORK")
    kinetic_col = _column(header, "", "KINETIC ENERGY")
    plastic_col = _column(header, "", "PLASTIC WORK")
    time_step_col = _column(header, "", "TIME STEP")
    node_cols_by_id = {node_id: _native_columns(header, i + 1)
                       for i, node_id in enumerate(node_history)}

    required = [time_col, time_step_col, energy_col, external_col, kinetic_col, plastic_col,
                *stress_cols.values()]
    for cols in node_cols_by_id.values():
        required.extend(cols.values())
    for row_index, row in enumerate(rows):
        for col in required:
            _finite(row[col], f"{case_id} CSV row {row_index} {col.strip()}")
    times = [_finite(row[time_col], f"{case_id} sample time") for row in rows]
    coverage = _validate_history(times, float(case["expected_end_time_s"]))

    initial_face_area_vector = _face_area_vector(reference_nodes, reference_nodes, loaded_ids, axis)
    reference_area = _norm(initial_face_area_vector)
    brick_xyz = [reference_nodes[n] for n in brick]
    # Eight-node brick card order is 1-2-3-4 then 5-6-7-8. Its first three
    # independent edges define the affine one-brick coupon volume.
    edge1 = _sub(brick_xyz[1], brick_xyz[0])
    edge2 = _sub(brick_xyz[3], brick_xyz[0])
    edge3 = _sub(brick_xyz[4], brick_xyz[0])
    reference_volume = abs(_dot(edge1, _cross(edge2, edge3)))
    if reference_area <= 0.0 or reference_volume <= 0.0:
        raise ValueError(f"invalid reference geometry: {case_id}")
    expected_analysis_end = float(case["expected_end_time_s"])
    motion_window = source_case.get("imposed_motion_window", {})
    expected_active_stop = float(motion_window.get("active_stop_time_s", expected_analysis_end))
    if expected_active_stop < expected_analysis_end:
        raise ValueError(f"/IMPVEL stop precedes analysis end: {case_id}")
    if not _close(drive["active_stop_s"], expected_active_stop, rel=1e-10, abs_=1e-9):
        raise ValueError(f"/IMPVEL stop time does not match source motion-window record: {case_id}")

    axial_displacement_field = ("DX", "DY", "DZ")[axis]
    gauge_length = abs(
        sum(reference_nodes[n][axis] for n in loaded_ids) / len(loaded_ids)
        - sum(reference_nodes[n][axis] for n in fixed_ids) / len(fixed_ids)
    )
    if gauge_length <= 0.0:
        raise ValueError(f"coupon gauge length is not positive: {case_id}")

    strains: list[float] = []
    axial_displacements: list[float] = []
    cauchy_axial: list[float] = []
    nominal_axial: list[float] = []
    area_ratio: list[float] = []
    energies: list[float] = []
    external_work: list[float] = []
    for row in rows:
        positions: dict[int, tuple[float, float, float]] = {}
        for node_id in node_history:
            disp = tuple(_finite(row[node_cols_by_id[node_id][field]],
                                 f"{case_id} node {node_id} {field}")
                         for field in DISPLACEMENT_FIELDS)
            positions[node_id] = _vector_add(reference_nodes[node_id], disp)
        mean_loaded = sum(_finite(row[node_cols_by_id[n][axial_displacement_field]],
                                  f"{case_id} loaded face displacement") for n in loaded_ids) / len(loaded_ids)
        mean_fixed = sum(_finite(row[node_cols_by_id[n][axial_displacement_field]],
                                 f"{case_id} fixed face displacement") for n in fixed_ids) / len(fixed_ids)
        displacement = mean_loaded - mean_fixed
        strains.append(displacement / gauge_length)
        axial_displacements.append(displacement)
        area_vec = _face_area_vector(positions, reference_nodes, loaded_ids, axis)
        area_ratio.append(_norm(area_vec) / reference_area)
        cauchy = tuple(_finite(row[stress_cols[field]], f"{case_id} {field}")
                       for field in STRESS_FIELDS)
        cauchy_axial.append(cauchy[axis])
        nominal_axial.append(nominal_axial_stress(cauchy, area_vec, reference_area, axis))
        energies.append(_finite(row[energy_col], f"{case_id} internal energy"))
        external_work.append(_finite(row[external_col], f"{case_id} external work"))

    target_displacement = float(source_case["target_displacement_mm"])
    displacement_error = axial_displacements[-1] - target_displacement
    commanded_at_sample = _prescribed_displacement(drive, times[-1])
    commanded_at_requested_end = _prescribed_displacement(drive, float(case["expected_end_time_s"]))
    measured_command_error = axial_displacements[-1] - commanded_at_sample
    final_step_dt = _finite(rows[-1][time_step_col], f"{case_id} final time step")
    command_before_final_step = _prescribed_displacement(drive, max(0.0, times[-1] - final_step_dt))
    last_step_command_increment = commanded_at_sample - command_before_final_step
    target_stress = float(source_case["source_fit_endpoint_stress_MPa"])
    endpoint_nominal = nominal_axial[-1]
    endpoint_relative_error = (endpoint_nominal - target_stress) / target_stress
    realized_fit = source_fit_at_engineering_strain(source_curve, strains[-1])
    actual_strain_residual = (endpoint_nominal - realized_fit["central_MPa"]) / realized_fit["central_MPa"]
    in_digitization_interval = (
        realized_fit["pixel_bound_low_MPa"] <= endpoint_nominal <= realized_fit["pixel_bound_high_MPa"]
    )

    stress_work = reference_volume * sum(
        0.5 * (nominal_axial[i] + nominal_axial[i + 1]) * (strains[i + 1] - strains[i])
        for i in range(len(rows) - 1)
    )
    native_internal = energies[-1]
    native_external = external_work[-1]
    energy_balance_error = native_internal - native_external
    energy_balance_rel = abs(energy_balance_error) / max(abs(native_internal), abs(native_external), 1e-12)
    stress_work_rel = abs(stress_work - native_internal) / max(abs(native_internal), 1e-12)

    displacement_gate = case["gates"]["loaded_face_displacement_mm"]
    stress_gate = case["gates"]["final_axis_stress_MPa"]
    exact_endpoint_frame = coverage["exact_requested_endpoint_frame"]
    loading_endpoint_gate_pass = (
        exact_endpoint_frame
        and abs(measured_command_error) <= float(displacement_gate["absolute_error_max_mm"])
        and abs(commanded_at_requested_end - target_displacement)
        <= float(displacement_gate["absolute_error_max_mm"])
    )
    endpoint_source_gate_pass = (
        exact_endpoint_frame
        and abs(endpoint_relative_error) <= float(stress_gate["relative_error_max"])
    )
    return {
        "case_id": case_id,
        "material_role": role,
        "material_axis": axis_label,
        "native_run": {
            "execution_complete": bool(run["execution_complete"]),
            "case_sequence_complete": bool(run["case_sequence_complete"]),
            "normal_termination": True,
            "runtime_profile": run.get("runtime_profile"),
            "source_commit": run.get("source_commit"),
            "runtime_identity_sha256": run.get("runtime_identity_sha256"),
            "engine_sha256": run["binary_sha256"]["engine"],
            "starter_sha256": run["binary_sha256"]["starter"],
            "case_sha256": run["case_sha256"],
            "all_recorded_input_and_output_hashes_verified": True,
            "conversion": {
                "converter_sha256": receipt["converter_sha256"],
                "native_T01_sha256": receipt["input_sha256"],
                "csv_sha256": receipt["output_sha256"],
            },
        },
        "history": {
            **coverage,
            "full_time_coverage": coverage["reaches_requested_horizon_within_one_sample_interval"],
            "all_required_samples_finite": True,
            "native_outputs_are_sampled_at_accepted_history_times": True,
            "reaction_output_used_as_force": False,
        },
        "geometry": {
            "gauge_length_mm": gauge_length,
            "reference_face_area_mm2": reference_area,
            "reference_volume_mm3": reference_volume,
            "endpoint_current_to_reference_area_ratio": area_ratio[-1],
            "endpoint_loaded_face_displacement_mm": axial_displacements[-1],
            "prescribed_endpoint_displacement_mm": target_displacement,
            "displacement_error_mm": displacement_error,
            "displacement_error_fraction_of_target": displacement_error / target_displacement,
            "imposed_motion_from_native_deck": {
                "function_id": drive["function_id"],
                "direction": drive["direction"],
                "loaded_node_group_id": drive["node_group_id"],
                "abscissa_scale": drive["abscissa_scale"],
                "velocity_scale_mm_s": drive["velocity_scale_mm_s"],
                "active_start_s": drive["active_start_s"],
                "active_stop_s": drive["active_stop_s"],
                "analysis_end_s": expected_analysis_end,
                "active_after_analysis_end_s": drive["active_stop_s"] - expected_analysis_end,
                "function_knots": drive["function_knots"],
                "commanded_displacement_at_last_sample_mm": commanded_at_sample,
                "measured_minus_commanded_at_last_sample_mm": measured_command_error,
                "commanded_displacement_at_requested_end_mm": commanded_at_requested_end,
                "commanded_end_minus_source_target_mm": commanded_at_requested_end - target_displacement,
                "last_native_time_step_s": final_step_dt,
                "commanded_increment_over_last_native_step_mm": last_step_command_increment,
                "target_minus_measured_as_fraction_of_last_step_command": (
                    -displacement_error / last_step_command_increment
                    if abs(last_step_command_increment) > 1e-15 else None
                ),
                "source": "parsed from the executed /IMPVEL card and its referenced /FUNCT curve; displacement is its time integral",
            },
            "achieved_engineering_strain": strains[-1],
            "prescribed_engineering_strain": float(source_case["target_engineering_strain"]),
            "loading_endpoint_gate_pass": loading_endpoint_gate_pass,
            "loading_endpoint_gate_evaluable": exact_endpoint_frame,
        },
        "stress": {
            "component": source_case["stress_component"],
            "native_endpoint_cauchy_MPa": cauchy_axial[-1],
            "endpoint_nominal_first_piola_MPa": endpoint_nominal,
            "native_to_nominal_conversion": "P_axial = (sigma_Cauchy @ current_face_area_vector)[axial] / reference_area",
            "source_fit_target_endpoint_MPa": target_stress,
            "endpoint_relative_error_vs_source_target": endpoint_relative_error,
            "endpoint_source_gate_relative_error_max": float(stress_gate["relative_error_max"]),
            "endpoint_source_gate_pass": endpoint_source_gate_pass,
            "endpoint_source_threshold_is_backend_admission": False,
            "endpoint_source_error_percent_raw": 100.0 * endpoint_relative_error,
            "endpoint_source_numeric_gate_evaluable": exact_endpoint_frame,
            "endpoint_numeric_gate_note": "The case's 1% target-line threshold is retained as a numeric diagnostic. The source H pixel envelope is reported separately and is not a confidence interval.",
            "source_fit_at_achieved_engineering_strain_MPa": realized_fit["central_MPa"],
            "source_fit_visual_pixel_bound_at_achieved_strain_MPa": [
                realized_fit["pixel_bound_low_MPa"], realized_fit["pixel_bound_high_MPa"]
            ],
            "relative_error_vs_fit_at_achieved_strain": actual_strain_residual,
            "inside_source_visual_pixel_bound_at_achieved_strain": in_digitization_interval,
            "source_uncertainty_warning": "Pixel-readout bounds on a published fit line, not a confidence interval or specimen-level validation.",
        },
        "energy_and_work": {
            "native_global_internal_energy": native_internal,
            "native_global_external_work": native_external,
            "native_internal_minus_external": energy_balance_error,
            "native_energy_balance_relative_error": energy_balance_rel,
            "axial_nominal_stress_strain_trapezoid_work_Nmm": stress_work,
            "stress_strain_work_relative_error_vs_native_internal": stress_work_rel,
            "native_plastic_work": _finite(rows[-1][plastic_col], f"{case_id} plastic work"),
            "native_kinetic_energy": _finite(rows[-1][kinetic_col], f"{case_id} kinetic energy"),
            "work_quadrature_limit": "Axial one-brick trapezoid over sampled accepted states; transverse traction work is omitted and the sampling is not independently converged.",
        },
        "qualification": {
            "direct_solver_coupon_transfer_only": True,
            "physical_material_qualification": False,
            "unidentified_parameters": source_record["unidentified_parameters"],
            "compression_mapping": source_record["compression_mapping"],
            "limitations": source_record["limitations"],
        },
    }


def collect_study(study_root: Path, *, run_prefix: str = RUN_PREFIX,
                  case_suite_dir: str = "cases-material-transfer-06-direct") -> dict[str, Any]:
    study_root = study_root.resolve()
    if not run_prefix or Path(run_prefix).name != run_prefix:
        raise ValueError("run_prefix must be a non-empty directory-name prefix")
    if not case_suite_dir or Path(case_suite_dir).name != case_suite_dir:
        raise ValueError("case_suite_dir must be a non-empty directory name")
    runs_root = study_root / "runs"
    run_dirs = sorted(p for p in runs_root.glob(f"{run_prefix}*") if p.is_dir())
    found_ids = {str(_json(p / "case.json").get("id")) for p in run_dirs}
    if found_ids != EXPECTED_CASES:
        raise ValueError(f"expected exactly the four direct LAW112 coupons; got {sorted(found_ids)}")
    coupons = [_coupon_result(study_root, p, case_suite_dir) for p in run_dirs]
    summaries = {
        "four_direct_coupons_present": len(coupons) == 4,
        "all_native_runs_complete_and_normal": all(
            c["native_run"]["execution_complete"] and c["native_run"]["normal_termination"] for c in coupons
        ),
        "all_T01_histories_full_and_finite": all(
            c["history"]["full_time_coverage"] and c["history"]["all_required_samples_finite"] for c in coupons
        ),
        "exact_requested_endpoint_frames": sum(
            c["history"]["exact_requested_endpoint_frame"] for c in coupons
        ),
        "loaded_face_displacement_gates_passed": sum(
            c["geometry"]["loading_endpoint_gate_pass"] for c in coupons
        ),
        "nominal_source_endpoint_numeric_threshold_passed": sum(
            c["stress"]["endpoint_source_gate_pass"] for c in coupons
        ),
        "source_fit_endpoint_relative_error_percent_raw": {
            c["case_id"]: c["stress"]["endpoint_source_error_percent_raw"] for c in coupons
        },
        "achieved_strain_inside_visual_source_envelope": sum(
            c["stress"]["inside_source_visual_pixel_bound_at_achieved_strain"] for c in coupons
        ),
        "native_global_internal_external_energy_balance_passed": all(
            c["energy_and_work"]["native_energy_balance_relative_error"] <= 1e-10 for c in coupons
        ),
        "maximum_axial_trapezoid_work_relative_error": max(
            c["energy_and_work"]["stress_strain_work_relative_error_vs_native_internal"] for c in coupons
        ),
        "physical_material_qualification": False,
    }
    return {
        "schema": SCHEMA,
        "study": "LAW112 uniaxial source-fit transfer; four one-brick coupons",
        "source_data_class": "published bilinear fit lines and yield anchors; not specimen-level raw measurements",
        "source_module": {
            "path": "matter/tools/cardboard_opencourant_material.py",
            "sha256": _sha256(Path(material.__file__).resolve()),
            "source_commit": material.SOURCE_COMMIT,
            "source_hashes_sha256": material._source_hashes(),
        },
        "measurement_tool": {
            "path": "matter/tools/cardboard_opencourant_material_measure.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "selection": {"run_prefix": run_prefix, "case_suite_dir": case_suite_dir},
        "source_fit_threshold_interpretation": "The retained 1% endpoint threshold is a numeric transfer diagnostic only, not backend admission. Visual fit-line pixel bounds are reported separately and are not a statistical confidence interval.",
        "runtime_scope": "Four native direct-MUMPS implicit coupons on candidate04; this report does not qualify PCG or full-box use.",
        "overall_interpretation": "Measurement report only. A transfer coupon is not independent physical validation or a multiaxial material qualification.",
        "summary": summaries,
        "coupons": coupons,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True,
                        help="study directory containing runs/, post/, and the selected case suite")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON report output path")
    parser.add_argument("--run-prefix", default=RUN_PREFIX,
                        help="run directory prefix to collect (default: retained direct-MUMPS 03 suite)")
    parser.add_argument("--case-suite-dir", default="cases-material-transfer-06-direct",
                        help="case-source directory under the study root")
    args = parser.parse_args(argv)
    report = collect_study(args.study_dir, run_prefix=args.run_prefix,
                           case_suite_dir=args.case_suite_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(report['coupons'])} coupons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
