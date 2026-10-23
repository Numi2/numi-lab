#!/usr/bin/env python3
"""Measure retained LAW117 adhesion and cohesive-birth solver histories.

The reader binds every measurement to the exact native run inputs and output
history, the converted CSV/TITLES sidecars, and the retained converter hash.
LAW117 stress channels are measured directly against the measured separation;
T01 reaction channels remain converter-derived window-average forces and are
never treated as endpoint forces. This is a synthetic solver instrument, not
a paper or adhesive-material qualification.
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
    """Raised when retained provenance or explicit measurement channels fail."""


SUPPORTED_THTOCSV_BINARY_SHA256 = "f8176302da59caba3e373504804beb77715137b5e9e9ef79eaf3231f5e1a1502"
THTOCSV_SOURCE_COMMIT = "1b6bb9200ee7dca899e61c18bd982c3fb1c72fd1"
THTOCSV_SOURCE_URL = (
    "https://github.com/OpenCourant/Tools/blob/"
    + THTOCSV_SOURCE_COMMIT
    + "/output_converters/th_to_csv/src/th_to_csv.c#L1297-L1334"
)
LAW117_SOURCE_URL = (
    "https://github.com/OpenCourant/OpenRadioss/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/"
    "engine/source/materials/mat/mat117/sigeps117.F90#L150-L190"
)
LAW117_SOURCE_COMMIT = "cb760c206e346eff12161a2268b134ef9d48a4ef"
ACTIVE_OFF_VALUE = 1
INACTIVE_OFF_VALUE = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise MeasurementError(f"expected a JSON object in {path}")
    return result


def _finite(value: str, *, context: str) -> float | None:
    if not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise MeasurementError(f"invalid numeric value {value!r} in {context}") from exc
    return result if math.isfinite(result) else None


def _verify_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    case_path, receipt_path = run_dir / "case.json", run_dir / "run.json"
    case, receipt = _json(case_path), _json(receipt_path)
    if case.get("schema") not in {
        "cardboard.opencourant.adhesion.case.v1",
        "cardboard.opencourant.bond.v1",
    }:
        raise MeasurementError(f"unsupported case schema in {case_path}")
    if receipt.get("schema") != "cardboard.opencourant.run.v1":
        raise MeasurementError(f"unsupported run receipt in {receipt_path}")
    if receipt.get("execution_complete") is not True:
        raise MeasurementError(f"run {run_dir.name} is not execution_complete")
    if receipt.get("case_sha256") != _sha256(case_path):
        raise MeasurementError(f"run {run_dir.name} case.json hash does not match run receipt")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise MeasurementError(f"run {run_dir.name} has no input hash map")
    for relative, metadata in inputs.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            raise MeasurementError(f"malformed native input hash entry {relative!r}")
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise MeasurementError(f"native input hash mismatch or missing file: {path}")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or not commands:
        raise MeasurementError(f"run {run_dir.name} lacks native command records")
    for index, command in enumerate(commands):
        if (not isinstance(command, dict) or command.get("exit_code") != 0
                or command.get("normal_termination") is not True
                or command.get("error_termination") is True
                or command.get("timed_out") is True):
            raise MeasurementError(f"native command {index} did not terminate normally")
    expected_commit = receipt.get("source_commit")
    if not isinstance(expected_commit, str) or len(expected_commit) < 12:
        raise MeasurementError(f"run {run_dir.name} lacks a source commit identity")
    return case, receipt, _sha256(receipt_path)


def _read_cards(deck: str) -> dict[tuple[str, int], dict[str, Any]]:
    cards: dict[tuple[str, int], dict[str, Any]] = {}
    lines = deck.splitlines()
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
        entities: list[int] = []
        cursor = index + 3
        while cursor < len(lines) and not lines[cursor].lstrip().startswith("/"):
            fields = lines[cursor].split()
            if fields:
                try:
                    entities.append(int(fields[0]))
                except ValueError as exc:
                    raise MeasurementError(
                        f"invalid entity in /TH/{kind}/{card_id}: {lines[cursor]!r}") from exc
            cursor += 1
        key = (kind, card_id)
        if key in cards:
            raise MeasurementError(f"duplicate /TH/{kind}/{card_id} card")
        cards[key] = {"kind": kind, "id": card_id, "title": title,
                      "variables": variables, "entities": tuple(entities)}
        index = cursor
    return cards


def _column(headers: list[str], card: dict[str, Any], entity: int, variable: str) -> int:
    title = " ".join(card["title"].replace("_", " ").split())
    pattern = re.compile(
        r"^" + re.escape(title) + r"\s+" + str(entity) + r"\s+" +
        re.escape(variable) + r"\s*$", re.I)
    matches = [i for i, header in enumerate(headers)
               if pattern.fullmatch(" ".join(header.replace("_", " ").split()))]
    if len(matches) != 1:
        raise MeasurementError(
            f"expected one titled {variable} column for entity {entity} in /TH/"
            f"{card['kind']}/{card['id']}; found {len(matches)}")
    return matches[0]


def _parse_material(deck: str) -> dict[str, float]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines)
              if re.fullmatch(r"\s*/MAT/LAW117/\d+\s*", line, re.I)]
    if len(starts) != 1:
        raise MeasurementError("starter must contain exactly one /MAT/LAW117 card")
    start = starts[0]
    try:
        elastic = [float(x) for x in lines[start + 3].split()[:2]]
        peak = [float(x) for x in lines[start + 4].split()[2:4]]
        fracture = [float(x) for x in lines[start + 5].split()[:2]]
        failure = [float(x) for x in lines[start + 6].split()[2:4]]
    except (IndexError, ValueError) as exc:
        raise MeasurementError("cannot parse LAW117 stiffness/peak/fracture card") from exc
    if any(not math.isfinite(x) or x <= 0 for x in elastic + peak + fracture + failure):
        raise MeasurementError("LAW117 model card contains nonpositive parameters")
    return {
        "normal_stiffness_MPa_per_mm": elastic[0],
        "shear_stiffness_MPa_per_mm": elastic[1],
        "peak_normal_MPa": peak[0], "peak_shear_MPa": peak[1],
        "GIC_N_per_mm": fracture[0], "GIIC_N_per_mm": fracture[1],
        "failure_normal_mm": failure[0], "failure_shear_mm": failure[1],
    }


def _parse_nodes(deck: str) -> dict[int, tuple[float, float, float]]:
    lines = deck.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().upper() == "/NODE"]
    if len(starts) != 1:
        raise MeasurementError("starter must contain exactly one /NODE card")
    result: dict[int, tuple[float, float, float]] = {}
    for line in lines[starts[0] + 1:]:
        if line.lstrip().startswith("/"):
            break
        fields = line.split()
        if not fields:
            continue
        try:
            node_id = int(fields[0])
            xyz = tuple(float(v) for v in fields[1:4])
        except (ValueError, IndexError) as exc:
            raise MeasurementError(f"cannot parse node coordinates: {line!r}") from exc
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise MeasurementError(f"invalid node coordinates: {line!r}")
        result[node_id] = xyz
    return result


def _face_area(nodes: dict[int, tuple[float, float, float]], ids: list[int]) -> float:
    if len(ids) != 4 or any(node not in nodes for node in ids):
        raise MeasurementError("cohesive faces must declare four present nodes")
    p = [nodes[i] for i in ids]

    def triangle(a: tuple[float, ...], b: tuple[float, ...], c: tuple[float, ...]) -> float:
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        cross = (u[1] * v[2] - u[2] * v[1],
                 u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0])
        return 0.5 * math.sqrt(sum(x * x for x in cross))

    return triangle(p[0], p[1], p[2]) + triangle(p[0], p[2], p[3])


def _history_columns(deck_path: Path, headers: list[str], case: dict[str, Any]) -> dict[str, Any]:
    deck = deck_path.read_text(encoding="utf-8", errors="replace")
    cards = _read_cards(deck)
    is_bond = case.get("schema") == "cardboard.opencourant.bond.v1"
    primary_axis = case.get("normal_axis", "Z") if is_bond else None
    if primary_axis is not None and primary_axis not in {"X", "Y", "Z"}:
        raise MeasurementError(f"unsupported bond normal_axis {primary_axis!r}")
    axes = {"X", "Z"}
    if primary_axis is not None:
        axes.add(primary_axis)
    required_cards = {("NODE", 101), ("NODE", 102), ("BRIC", 103)}
    if is_bond:
        required_cards |= {("NODE", 104), ("BRIC", 105)}
    if not required_cards.issubset(cards):
        raise MeasurementError(f"starter history cards are missing: {sorted(required_cards - cards.keys())}")
    for key, variables in {
        ("NODE", 101): {"REAC" + axis for axis in axes},
        ("NODE", 102): {"D" + axis for axis in axes},
        # /TH/BRIC requests a STRESS tensor; the converter titles its expanded
        # components (including SZ and SXZ) as distinct CSV columns.
        ("BRIC", 103): {"OFF", "STRESS", "IE"},
    }.items():
        if not variables.issubset(set(cards[key]["variables"])):
            raise MeasurementError(f"/TH/{key[0]}/{key[1]} lacks {sorted(variables - set(cards[key]['variables']))}")
    fields: dict[str, Any] = {"reaction": {}, "top": {}, "cohesive": {}}
    for axis in sorted(axes):
        fields["reaction"][axis] = {
            node: _column(headers, cards[("NODE", 101)], node, "REAC" + axis)
            for node in cards[("NODE", 101)]["entities"]
        }
    for axis in sorted(axes):
        fields["top"][axis] = {
            node: _column(headers, cards[("NODE", 102)], node, "D" + axis)
            for node in cards[("NODE", 102)]["entities"]
        }
    for var in ("OFF", "SX", "SY", "SZ", "SXZ", "IE"):
        fields["cohesive"][var] = _column(headers, cards[("BRIC", 103)], 1, var)
    if is_bond:
        for key, variables in {
            ("NODE", 104): {"D" + primary_axis, "REAC" + primary_axis},
            ("BRIC", 105): {"STRESS", "IE"},
        }.items():
            if not variables.issubset(set(cards[key]["variables"])):
                raise MeasurementError(f"/TH/{key[0]}/{key[1]} lacks requested variables")
        fields["outer"] = {
            primary_axis: {node: _column(headers, cards[("NODE", 104)], node,
                                          "D" + primary_axis)
                           for node in cards[("NODE", 104)]["entities"]},
            "REA" + primary_axis: {
                node: _column(headers, cards[("NODE", 104)], node,
                              "REAC" + primary_axis)
                for node in cards[("NODE", 104)]["entities"]},
        }
        fields["solid"] = {
            "SX": _column(headers, cards[("BRIC", 105)], 2, "SX"),
            "SY": _column(headers, cards[("BRIC", 105)], 2, "SY"),
            "SZ": _column(headers, cards[("BRIC", 105)], 2, "SZ"),
            "IE": _column(headers, cards[("BRIC", 105)], 2, "IE"),
        }
        fields["primary_axis"] = primary_axis
    nodes = _parse_nodes(deck)
    top_ids = list(cards[("NODE", 102)]["entities"])
    bottom_ids = list(cards[("NODE", 101)]["entities"])
    area = _face_area(nodes, top_ids)
    if not math.isclose(area, _face_area(nodes, bottom_ids), rel_tol=1e-9, abs_tol=1e-12):
        raise MeasurementError("cohesive top and bottom face areas differ")
    fields["geometry"] = {"cohesive_area_mm2": area, "top_nodes": top_ids,
                           "bottom_nodes": bottom_ids}
    fields["material"] = _parse_material(deck)
    fields["cards"] = cards
    return fields


def _stage_records(run_dir: Path, post_dir: Path, case: dict[str, Any],
                   receipt: dict[str, Any]) -> list[dict[str, Any]]:
    engines = case.get("engines")
    if not isinstance(engines, list) or not engines:
        raise MeasurementError("case does not declare an ordered engine-stage list")
    records: list[dict[str, Any]] = []
    outputs = receipt.get("outputs", {})
    for stage_index, _ in enumerate(engines, start=1):
        stage_name = f"T{stage_index:02d}"
        directory = post_dir / stage_name
        conversion_path = directory / "conversion.json"
        csv_path = directory / "probeT01.csv"
        alias = directory / "probeT01"
        titles = directory / "probeT01_TITLES"
        if not all(path.is_file() for path in (conversion_path, csv_path, alias, titles)):
            raise MeasurementError(f"missing conversion artifact(s) for stage {stage_name} in {directory}")
        conversion = _json(conversion_path)
        if conversion.get("receipt_created_after_conversion") is not True:
            raise MeasurementError(f"{stage_name} conversion receipt does not certify post-conversion creation")
        if conversion.get("exit_code") != 0:
            raise MeasurementError(f"converter failed for stage {stage_name}")
        if conversion.get("converter_sha256") != SUPPORTED_THTOCSV_BINARY_SHA256:
            raise MeasurementError(f"unsupported th_to_csv binary for stage {stage_name}")
        if conversion.get("input_alias") != "probeT01" or conversion.get("output") != "probeT01.csv":
            raise MeasurementError(f"unexpected converter alias/output for {stage_name}")
        if _sha256(alias) != conversion.get("input_sha256"):
            raise MeasurementError(f"native history alias hash mismatch for {stage_name}")
        if _sha256(csv_path) != conversion.get("output_sha256"):
            raise MeasurementError(f"converted CSV hash mismatch for {stage_name}")
        if _sha256(titles) != conversion.get("titles_sha256"):
            raise MeasurementError(f"TITLES sidecar hash mismatch for {stage_name}")
        input_name = conversion.get("input_relative_path")
        title_name = input_name + "_TITLES" if isinstance(input_name, str) else None
        if not title_name or input_name not in outputs or title_name not in outputs:
            raise MeasurementError(f"run receipt does not retain the source histories for {stage_name}")
        if (outputs[input_name].get("sha256") != conversion.get("input_sha256")
                or outputs[title_name].get("sha256") != conversion.get("titles_sha256")):
            raise MeasurementError(f"converter inputs do not match the native run outputs for {stage_name}")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise MeasurementError(f"empty converted CSV: {csv_path}") from exc
            if not headers or headers[0].strip().casefold() != "time":
                raise MeasurementError("converted CSV needs a leading time column")
            if len(set(headers)) != len(headers):
                raise MeasurementError(f"duplicate column names in {csv_path}")
            card_map = _history_columns(run_dir / case["starter"], headers, case)
            rows = []
            for line, values in enumerate(reader, start=2):
                if len(values) != len(headers):
                    raise MeasurementError(f"{csv_path}:{line}: wrong number of values")
                parsed = [_finite(value, context=f"{csv_path}:{line}") for value in values]
                rows.append(parsed)
        if not rows:
            raise MeasurementError(f"converted history has no samples: {csv_path}")
        time_col = 0
        times = [row[time_col] for row in rows]
        if any(t is None for t in times):
            raise MeasurementError(f"converted history has a blank/nonfinite time in {csv_path}")
        time_values = [float(t) for t in times]
        if any(b <= a for a, b in zip(time_values, time_values[1:])):
            raise MeasurementError(f"converted timebase is not strictly increasing in {stage_name}")
        records.append({"stage": stage_name, "times": time_values, "rows": rows,
                        "columns": card_map, "headers": headers,
                        "conversion_sha256": _sha256(conversion_path),
                        "conversion": conversion, "csv_sha256": _sha256(csv_path),
                        "native_history_sha256": conversion["input_sha256"],
                        "titles_sha256": conversion["titles_sha256"]})
    return records


def _record_value(record: dict[str, Any], row_index: int, column: int) -> float | None:
    return record["rows"][row_index][column]


def _mean_entities(record: dict[str, Any], row_index: int,
                   columns: dict[int, int]) -> float | None:
    values = [_record_value(record, row_index, col) for col in columns.values()]
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def _sum_entities(record: dict[str, Any], row_index: int,
                  columns: dict[int, int]) -> float | None:
    values = [_record_value(record, row_index, col) for col in columns.values()]
    if not values or any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _series(record: dict[str, Any], group: str, key: str) -> list[float | None]:
    if group == "reaction":
        colmap = record["columns"][group][key]
        return [_sum_entities(record, i, colmap) for i in range(len(record["rows"]))]
    if group in ("top", "outer"):
        colmap = record["columns"][group][key]
        return [_mean_entities(record, i, colmap) for i in range(len(record["rows"]))]
    col = record["columns"][group][key]
    return [_record_value(record, i, col) for i in range(len(record["rows"]))]


def _stencil_indices(index: int, count: int) -> tuple[int, int, str]:
    if count < 3:
        raise MeasurementError("converter force window needs at least three samples")
    if index == 0:
        return 0, 1, "forward"
    if index == count - 1:
        return count - 2, count - 1, "backward"
    return index - 1, index + 1, "centered"


def _time_average(times: list[float], values: list[float | None], left: int,
                  right: int) -> float | None:
    if right <= left or any(values[i] is None for i in range(left, right + 1)):
        return None
    total = 0.0
    for i in range(left, right):
        total += 0.5 * (float(values[i]) + float(values[i + 1])) * (times[i + 1] - times[i])
    return total / (times[right] - times[left])


def _source_traction(separation_n: float, separation_s: float, history_max: float,
                     material: dict[str, float], area_mm2: float,
                     mode: str) -> tuple[float, float]:
    """Pure-mode LAW117 linear-evolution response from sigeps117.F90."""
    def damage(max_sep: float, peak: float, stiffness: float, fracture: float) -> float:
        initiation = peak / stiffness
        failure = 2.0 * fracture / peak
        if max_sep <= initiation or max_sep == 0:
            return 0.0
        return min(1.0, max(0.0, (failure / max_sep) *
                            ((max_sep - initiation) / max(failure - initiation, 1e-20))))

    kn = material["normal_stiffness_MPa_per_mm"]
    ks = material["shear_stiffness_MPa_per_mm"]
    if mode == "I":
        d = damage(history_max, material["peak_normal_MPa"], kn, material["GIC_N_per_mm"])
        normal = kn * separation_n if separation_n < 0 else (1.0 - d) * kn * separation_n
        return normal, 0.0
    if mode == "II":
        d = damage(history_max, material["peak_shear_MPa"], ks, material["GIIC_N_per_mm"])
        return 0.0, (1.0 - d) * ks * separation_s
    raise MeasurementError(f"source analytic traction supports pure modes only, got {mode!r}")


def _interpolated_path(records: list[dict[str, Any]], axis: str,
                       reference_time_s: float | None,
                       mode: str | None = None) -> tuple[list[dict[str, Any]], str]:
    mode = mode or ("II" if axis == "X" else "I")
    output: list[dict[str, Any]] = []
    ref: dict[str, float] | None = None
    if reference_time_s is not None:
        candidates = [(abs(t - reference_time_s), record, i)
                      for record in records
                      for i, t in enumerate(record["times"])]
        if not candidates:
            return [], mode
        _, record, i = min(candidates, key=lambda item: item[0])
        ref = {axis_name: (_series(record, "top", axis_name)[i] or 0.0)
               for axis_name in ("X", "Y", "Z")
               if axis_name in record["columns"]["top"]}
    for record in records:
        displacement = {axis_name: _series(record, "top", axis_name)
                        for axis_name in ("X", "Y", "Z")
                        if axis_name in record["columns"]["top"]}
        off = _series(record, "cohesive", "OFF")
        stress = {axis_name: _series(record, "cohesive", "S" + axis_name)
                  for axis_name in ("X", "Y", "Z")}
        sxz = _series(record, "cohesive", "SXZ")
        ie = _series(record, "cohesive", "IE")
        for i, time_s in enumerate(record["times"]):
            current = {axis_name: values[i] for axis_name, values in displacement.items()}
            if ref is not None:
                current = {axis_name: (None if value is None else value - ref[axis_name])
                           for axis_name, value in current.items()}
            point = {"time_s": time_s, "off": off[i], "sxz_MPa": sxz[i],
                     "ie_N_mm": ie[i], "stage": record["stage"], "index": i,
                     "record": record}
            point.update({"d" + axis_name.lower() + "_mm": value
                          for axis_name, value in current.items()})
            point.update({"s" + axis_name.lower() + "_MPa": values[i]
                          for axis_name, values in stress.items()})
            if axis == "combined":
                point["primary_separation_mm"] = None
                point["primary_traction_MPa"] = None
            else:
                primary_displacement = current.get(axis)
                primary_traction = (sxz[i] if mode == "II"
                                    else stress[axis][i])
                point["primary_separation_mm"] = primary_displacement
                point["primary_traction_MPa"] = primary_traction
            output.append(point)
    return output, mode


def _work(path: list[dict[str, Any]], area: float, *, activation_time_s: float | None,
          axis: str, mode: str = "I") -> dict[str, Any]:
    work_x = work_y = work_z = 0.0
    integrated_segments = 0
    for left, right in zip(path, path[1:]):
        if activation_time_s is not None and right["time_s"] < activation_time_s:
            continue
        if left["off"] != ACTIVE_OFF_VALUE or right["off"] != ACTIVE_OFF_VALUE:
            continue
        required = ("dx_mm", "sxz_MPa") if mode == "II" and axis == "X" else (
            ("dx_mm", "dz_mm", "sxz_MPa", "sz_MPa") if axis == "combined" else
            ("d" + axis.lower() + "_mm", "s" + axis.lower() + "_MPa"))
        if any(left[k] is None or right[k] is None for k in required):
            continue
        if mode == "II" and axis == "X":
            d_x = float(right["dx_mm"]) - float(left["dx_mm"])
            work_x += 0.5 * (float(left["sxz_MPa"]) + float(right["sxz_MPa"])) * area * d_x
        elif axis == "combined":
            d_x = float(right["dx_mm"]) - float(left["dx_mm"])
            d_z = float(right["dz_mm"]) - float(left["dz_mm"])
            work_x += 0.5 * (float(left["sxz_MPa"]) + float(right["sxz_MPa"])) * area * d_x
            work_z += 0.5 * (float(left["sz_MPa"]) + float(right["sz_MPa"])) * area * d_z
        elif axis == "X":
            d_x = float(right["dx_mm"]) - float(left["dx_mm"])
            work_x += 0.5 * (float(left["sx_MPa"]) + float(right["sx_MPa"])) * area * d_x
        elif axis == "Y":
            d_y = float(right["dy_mm"]) - float(left["dy_mm"])
            work_y += 0.5 * (float(left["sy_MPa"]) + float(right["sy_MPa"])) * area * d_y
        else:
            d_z = float(right["dz_mm"]) - float(left["dz_mm"])
            work_z += 0.5 * (float(left["sz_MPa"]) + float(right["sz_MPa"])) * area * d_z
        integrated_segments += 1
    if mode == "II" and axis == "X":
        selected, selected_work = "DX", work_x
    elif axis in {"X", "Y", "Z"}:
        selected, selected_work = "D" + axis, {"X": work_x, "Y": work_y, "Z": work_z}[axis]
    else:
        selected, selected_work = "combined", work_x + work_z
    return {"mode_i_work_N_mm": work_x + work_y + work_z if mode == "I" else work_z,
            "mode_ii_work_N_mm": work_x if mode == "II" else 0.0,
            "selected_work_N_mm": selected_work, "selected_separation": selected,
            "active_segments_integrated": integrated_segments,
            "method": "trapezoidal native cohesive stress times geometric face area times measured separation increment; only adjacent active samples"}


def _reaction_diagnostics(records: list[dict[str, Any]], area: float,
                          axis: str, mode: str = "I") -> dict[str, Any]:
    axis_name = axis if axis in {"X", "Y", "Z"} else "Z"
    reaction_samples = []
    absolute_residuals = []
    for record in records:
        times = record["times"]
        reaction = _series(record, "reaction", axis_name)
        stress_component = "SXZ" if mode == "II" and axis == "X" else "S" + axis_name
        stress = _series(record, "cohesive", stress_component)
        if len(times) < 3 or any(v is None for v in reaction):
            continue
        for i, time_s in enumerate(times):
            left, right, method = _stencil_indices(i, len(times))
            expected = _time_average(times, stress, left, right)
            measured_signed = reaction[i]
            if expected is None or measured_signed is None:
                continue
            measured = -float(measured_signed)  # lower-face support sign opposes positive cohesive traction
            expected_force = expected * area
            residual = measured - expected_force
            absolute_residuals.append(abs(residual))
            reaction_samples.append({"stage": record["stage"], "index": i,
                "time_s": time_s, "converter_window": method,
                "window_start_s": times[left], "window_end_s": times[right],
                "support_reaction_force_window_average_N": measured,
                "native_cohesive_traction_window_average_N": expected_force,
                "residual_N": residual})
    if not reaction_samples:
        return {"status": "unavailable", "reason": "fewer than three finite samples or reaction channels missing"}
    final_stress = _series(records[-1], "cohesive", stress_component)[-1]
    return {
        "status": "measured_diagnostic",
        "force_semantics": "T01 REAC channels are impulse differentiated by the pinned converter; each value is a source-stencil time-window average.",
        "converter_binary_sha256": SUPPORTED_THTOCSV_BINARY_SHA256,
        "converter_source_commit": THTOCSV_SOURCE_COMMIT,
        "converter_source_url": THTOCSV_SOURCE_URL,
        "samples": len(reaction_samples),
        "window_average_force_residual_max_abs_N": max(absolute_residuals),
        "window_average_force_residual_rms_N": math.sqrt(sum(x*x for x in absolute_residuals) / len(absolute_residuals)),
        "endpoint_force_vs_endpoint_traction_N": (
            None if final_stress is None
            else reaction_samples[-1]["support_reaction_force_window_average_N"] -
            (float(final_stress) * area)),
        "endpoint_comparison_note": "The final converted support reaction is a backward-window average; its endpoint subtraction from instantaneous traction is retained only as a diagnostic and is not a force gate.",
        "sample_examples": reaction_samples[:3] + reaction_samples[-1:],
    }


def _activation_info(case: dict[str, Any], path: list[dict[str, Any]],
                     coverage_ok: bool) -> dict[str, Any] | None:
    schema = case.get("schema")
    old_activation = case.get("parameters", {}).get("activation")
    is_activation = (bool(case.get("activation")) if schema == "cardboard.opencourant.bond.v1"
                    else isinstance(old_activation, dict))
    if not is_activation:
        return None
    if schema == "cardboard.opencourant.bond.v1":
        activation_time = case.get("prediction", {}).get("birth_time_s")
    else:
        activation_time = old_activation.get("activation_time_s")
    off_values = [p["off"] for p in path if p["off"] is not None]
    before = [p["off"] for p in path if activation_time is not None and p["time_s"] < activation_time]
    after = [p["off"] for p in path if activation_time is not None and p["time_s"] >= activation_time]
    dormant_before = bool(before) and all(value == INACTIVE_OFF_VALUE for value in before)
    active_after = bool(after) and any(value == ACTIVE_OFF_VALUE for value in after)
    if not coverage_ok:
        status = "coverage_failed"
    elif not active_after:
        status = "activation_not_observed"
    elif not dormant_before:
        status = "activation_prestate_not_dormant"
    else:
        status = "activation_observed"
    expected = old_activation or {}
    return {
        "status": status,
        "activation_time_s": activation_time,
        "OFF_semantics": {"active": ACTIVE_OFF_VALUE, "dormant_or_deleted": INACTIVE_OFF_VALUE},
        "sample_counts": {"before_activation": len(before), "at_or_after_activation": len(after)},
        "OFF_before_values": sorted({v for v in before if v is not None}),
        "OFF_after_values": sorted({v for v in after if v is not None}),
        "dormant_before_observed": dormant_before,
        "active_after_observed": active_after,
        "legacy_case_OFF_metadata_conflicts_with_native_semantics": (
            expected.get("expected_OFF_before") == ACTIVE_OFF_VALUE
            or expected.get("expected_OFF_after") == INACTIVE_OFF_VALUE),
        "OFF_observed_values": sorted({v for v in off_values if v is not None}),
    }


def _constitutive_diagnostics(path: list[dict[str, Any]], material: dict[str, float],
                              area_mm2: float, mode: str,
                              activation_time_s: float | None,
                              axis: str = "Z") -> dict[str, Any]:
    if mode not in {"I", "II"}:
        return {"status": "descriptive_only", "reason": "closed-form cross-check is implemented for pure mode I or II only"}
    separation_key = "primary_separation_mm"
    traction_key = "primary_traction_MPa"
    reference_separation = 0.0
    if activation_time_s is not None:
        candidates = [p for p in path if p["time_s"] >= activation_time_s]
        if candidates:
            reference_separation = float(candidates[0][separation_key] or 0.0)
    max_sep = 0.0
    expected_stress: list[float | None] = []
    observed: list[float | None] = []
    for point in path:
        separation = (None if point[separation_key] is None else
                      float(point[separation_key]) - reference_separation)
        sep = abs(separation or 0.0) if mode == "II" else max(separation or 0.0, 0.0)
        if activation_time_s is not None and point["time_s"] < activation_time_s:
            expected_stress.append(0.0)
            observed.append(point[traction_key])
            continue
        max_sep = max(max_sep, sep)
        normal, shear = _source_traction(separation or 0.0, separation or 0.0,
                                         max_sep, material, area_mm2, mode)
        expected_stress.append(shear if mode == "II" else normal)
        observed.append(point[traction_key])
    pairs = [(a, b) for a, b, p in zip(expected_stress, observed, path)
             if a is not None and b is not None and p["off"] == ACTIVE_OFF_VALUE
             and (activation_time_s is None or p["time_s"] >= activation_time_s)]
    if not pairs:
        return {"status": "unavailable", "reason": "no active finite stress/separation samples"}
    errors = [float(b) - float(a) for a, b in pairs]
    relative_floor = max(0.01, 0.01 * max(abs(float(a)) for a, _ in pairs))
    return {
        "status": "measured",
        "basis": "Pinned LAW117 linear displacement evolution: historical maximum equivalent separation, source damage expression, tensile/shear degradation and ISYM=0 normal compression branch.",
        "source_commit": LAW117_SOURCE_COMMIT,
        "source_url": LAW117_SOURCE_URL,
        "observed_traction_channel": "SXZ" if mode == "II" else "S" + axis,
        "measured_separation_component": "DX" if mode == "II" else "D" + axis,
        "samples_compared": len(pairs),
        "max_abs_stress_error_MPa": max(abs(e) for e in errors),
        "rms_stress_error_MPa": math.sqrt(sum(e*e for e in errors) / len(errors)),
        "max_relative_stress_error_reference_over_floor": (
            max(abs(float(b) - float(a)) / abs(float(a)) for a, b in pairs
                if abs(float(a)) >= relative_floor)
            if any(abs(float(a)) >= relative_floor for a, _ in pairs) else None),
        "relative_error_reference_floor_MPa": relative_floor,
        "max_historical_separation_mm": max_sep,
    }


def _crossing(path: list[dict[str, Any]], x_key: str, y_key: str, target: float,
              direction: int, *, after_time_s: float = -math.inf) -> dict[str, float] | None:
    for left, right in zip(path, path[1:]):
        if left["time_s"] < after_time_s or left[x_key] is None or right[x_key] is None:
            continue
        x0, x1 = float(left[x_key]), float(right[x_key])
        change = x1 - x0
        if (direction > 0 and change <= 0) or (direction < 0 and change >= 0):
            continue
        if min(x0, x1) <= target <= max(x0, x1):
            fraction = (target - x0) / change
            y0, y1 = left[y_key], right[y_key]
            if y0 is None or y1 is None:
                return None
            return {
                "time_s": left["time_s"] + fraction * (right["time_s"] - left["time_s"]),
                "separation_mm": target,
                "traction_MPa": float(y0) + fraction * (float(y1) - float(y0)),
            }
    return None


def _monotone_mode_work(max_separation_mm: float, material: dict[str, float],
                        area_mm2: float, mode: str) -> float:
    if mode == "I":
        stiffness = material["normal_stiffness_MPa_per_mm"]
        peak = material["peak_normal_MPa"]
        fracture = material["GIC_N_per_mm"]
        failure = material["failure_normal_mm"]
    else:
        stiffness = material["shear_stiffness_MPa_per_mm"]
        peak = material["peak_shear_MPa"]
        fracture = material["GIIC_N_per_mm"]
        failure = material["failure_shear_mm"]
    initiation = peak / stiffness
    separation = max(0.0, max_separation_mm)
    if separation <= initiation:
        return 0.5 * stiffness * separation * separation * area_mm2
    if separation >= failure:
        return fracture * area_mm2
    softening_work = peak / (failure - initiation) * (
        failure * (separation - initiation)
        - 0.5 * (separation * separation - initiation * initiation))
    return (0.5 * stiffness * initiation * initiation + softening_work) * area_mm2


def _path_behavior(case: dict[str, Any], path: list[dict[str, Any]],
                   material: dict[str, float], area: float, axis: str,
                   mode: str, work: dict[str, Any],
                   activation_time_s: float | None) -> dict[str, Any]:
    case_id = str(case.get("id", ""))
    x_key = "dx_mm" if axis == "X" else "dz_mm"
    y_key = "sxz_MPa" if axis == "X" else "sz_MPa"
    behavior: dict[str, Any] = {}
    if mode in {"I", "II"}:
        active_path = [p for p in path if p["off"] == ACTIVE_OFF_VALUE
                       and (activation_time_s is None or p["time_s"] >= activation_time_s)
                       and p["primary_separation_mm"] is not None]
        differences = [float(b[x_key]) - float(a[x_key])
                       for a, b in zip(active_path, active_path[1:])]
        if axis in {"X", "Y", "Z"}:
            differences = [float(b["primary_separation_mm"]) -
                           float(a["primary_separation_mm"])
                           for a, b in zip(active_path, active_path[1:])]
        monotone = bool(differences) and all(delta >= -1e-9 for delta in differences)
        max_separation = max((max(0.0, float(p["primary_separation_mm"]))
                              for p in active_path), default=0.0)
        behavior["monotone_loading_path"] = monotone
        if monotone:
            expected_work = _monotone_mode_work(max_separation, material, area, mode)
            behavior["source_curve_monotone_work_reference_N_mm"] = expected_work
            behavior["measured_work_minus_source_curve_reference_N_mm"] = (
                work["selected_work_N_mm"] - expected_work)
            behavior["source_curve_reference_gap_max_mm"] = max_separation

    if case_id == "law117-unload-reload":
        x_key, y_key = "dz_mm", "sz_MPa"
        # The first local peak is the transition from primary loading to unload.
        peak_time = None
        peak_sep = None
        for left, right in zip(path, path[1:]):
            if left[x_key] is None or right[x_key] is None:
                continue
            if float(right[x_key]) < float(left[x_key]):
                peak_time, peak_sep = left["time_s"], float(left[x_key])
                break
        target = 0.04
        first_load = _crossing(path, x_key, y_key, target, 1)
        unload = (_crossing(path, x_key, y_key, target, -1, after_time_s=peak_time)
                  if peak_time is not None else None)
        reload = (_crossing(path, x_key, y_key, target, 1,
                            after_time_s=unload["time_s"] if unload else math.inf))
        behavior["unload_reload_matched_gap"] = {
            "primary_loading_at_0.04_mm": first_load,
            "unloading_at_0.04_mm": unload,
            "reloading_at_0.04_mm": reload,
            "first_loading_peak_before_unload_mm": peak_sep,
            "unload_to_primary_traction_ratio_at_0.04_mm": (
                None if first_load is None or unload is None or abs(first_load["traction_MPa"]) < 1e-12
                else unload["traction_MPa"] / first_load["traction_MPa"]),
        }
    elif case_id == "law117-open-close-compression":
        last = path[-1] if path else {}
        dz = last.get("dz_mm")
        sz = last.get("sz_MPa")
        behavior["closure_compression"] = {
            "final_normal_separation_mm": dz,
            "final_normal_traction_MPa": sz,
            "ISYM0_undamaged_compression_reference_MPa": (
                None if dz is None else material["normal_stiffness_MPa_per_mm"] * min(0.0, float(dz))),
            "final_OFF": last.get("off"),
        }
    return behavior


def measure_run(run_dir: Path, post_root: Path) -> dict[str, Any]:
    run_dir, post_root = Path(run_dir), Path(post_root)
    case, receipt, receipt_sha = _verify_run(run_dir)
    if receipt.get("source_commit") != LAW117_SOURCE_COMMIT:
        raise MeasurementError(
            f"run source commit {receipt.get('source_commit')!r} does not match the LAW117 source basis")
    declared_case_id = case.get("id")
    if not isinstance(declared_case_id, str) or not declared_case_id:
        declared_case_id = run_dir.name
    post_dir = post_root / run_dir.name
    records = _stage_records(run_dir, post_dir, case, receipt)
    expected = float(case.get("expected_end_time_s", case.get("expected_end_time_s", 0.0)))
    if expected <= 0 or not math.isfinite(expected):
        raise MeasurementError("case has no positive expected_end_time_s")
    all_times = [t for record in records for t in record["times"]]
    start, end = min(all_times), max(all_times)
    sample_count = sum(len(record["times"]) for record in records)
    coverage_fraction = max(0.0, min(1.0, (end - start) / expected))
    coverage_ok = sample_count >= 3 and start <= max(1e-12, expected * 1e-6) and coverage_fraction >= 0.95
    is_bond = case.get("schema") == "cardboard.opencourant.bond.v1"
    # The staged generated inputs are byte-verified before parsing; use the first
    # stage card map for unit geometry and required channels.
    columns = records[0]["columns"]
    area = columns["geometry"]["cohesive_area_mm2"]
    material = columns["material"]
    birth = bool(case.get("activation")) if is_bond else isinstance(case.get("parameters", {}).get("activation"), dict)
    activation_time = None
    if birth:
        activation_time = (case.get("prediction", {}).get("birth_time_s") if is_bond
                           else case.get("parameters", {}).get("activation", {}).get("activation_time_s"))
    axis = "X" if case.get("id") == "law117-mode-ii-shear-failure" else "Z"
    if is_bond:
        axis = case.get("normal_axis", "Z")
        if axis not in {"X", "Y", "Z"}:
            raise MeasurementError(f"unsupported bond normal_axis {axis!r}")
        if axis not in records[0]["columns"]["top"] or axis not in records[0]["columns"]["reaction"]:
            raise MeasurementError(f"starter history lacks requested normal-axis {axis} channels")
    if case.get("id") == "law117-mixed-mode-opening":
        axis = "combined"
    mode = ("II" if not is_bond and case.get("id") == "law117-mode-ii-shear-failure"
            else "mixed" if axis == "combined" else "I")
    path, mode = _interpolated_path(records, "Z" if axis == "combined" else axis,
                                    activation_time, mode)
    if case.get("id") == "law117-mixed-mode-opening":
        mode = "mixed"
    activation = _activation_info(case, path, coverage_ok)
    case_status = "coverage_failed" if not coverage_ok else "measured"
    if activation and coverage_ok and activation["status"] != "activation_observed":
        case_status = activation["status"]
    energy = [p["ie_N_mm"] for p in path if p["ie_N_mm"] is not None]
    separation = [float(p["primary_separation_mm"]) for p in path
                  if p["primary_separation_mm"] is not None]
    traction = [float(p["primary_traction_MPa"]) for p in path
                if p["primary_traction_MPa"] is not None]
    raw_work = _work(path, area, activation_time_s=activation_time if birth else None,
                     axis=axis, mode=mode)
    force_diag = _reaction_diagnostics(records, area, axis, mode)
    analytical = _constitutive_diagnostics(path, material, area, mode,
                                           activation_time if birth else None, axis)
    path_behavior = _path_behavior(case, path, material, area, axis, mode,
                                   raw_work, activation_time if birth else None)

    off = [int(round(float(p["off"]))) for p in path if p["off"] is not None]
    transitions = sum(a != b for a, b in zip(off, off[1:]))
    result: dict[str, Any] = {
        "run_id": run_dir.name,
        "case_id": declared_case_id,
        "status": case_status,
        "coverage": {
            "sample_count": sample_count,
            "stage_sample_counts": {r["stage"]: len(r["times"]) for r in records},
            "start_time_s": start, "end_time_s": end,
            "expected_end_time_s": expected,
            "duration_fraction": coverage_fraction,
            "coverage_ok": coverage_ok,
            "minimum_samples": 3,
            "minimum_duration_fraction": 0.95,
        },
        "stages": [
            {
                "stage": record["stage"],
                "engine_deck": case["engines"][index],
                "engine_deck_sha256": _sha256(run_dir / case["engines"][index]),
                "implicit_cards_present": "/IMPL/" in (run_dir / case["engines"][index]).read_text(encoding="utf-8", errors="replace"),
                "start_time_s": record["times"][0],
                "end_time_s": record["times"][-1],
                "sample_count": len(record["times"]),
                "OFF_first": _series(record, "cohesive", "OFF")[0],
                "OFF_last": _series(record, "cohesive", "OFF")[-1],
                "cohesive_IE_first_N_mm": _series(record, "cohesive", "IE")[0],
                "cohesive_IE_last_N_mm": _series(record, "cohesive", "IE")[-1],
                "free_interface_DX_first_mm": _series(record, "top", "X")[0],
                "free_interface_DX_last_mm": _series(record, "top", "X")[-1],
                "free_interface_DZ_first_mm": _series(record, "top", "Z")[0],
                "free_interface_DZ_last_mm": _series(record, "top", "Z")[-1],
                "cohesive_SZ_first_MPa": _series(record, "cohesive", "SZ")[0],
                "cohesive_SZ_last_MPa": _series(record, "cohesive", "SZ")[-1],
                "cohesive_SXZ_first_MPa": _series(record, "cohesive", "SXZ")[0],
                "cohesive_SXZ_last_MPa": _series(record, "cohesive", "SXZ")[-1],
                **({
                    "primary_axis": axis,
                    "free_interface_primary_displacement_first_mm": _series(record, "top", axis)[0],
                    "free_interface_primary_displacement_last_mm": _series(record, "top", axis)[-1],
                    "cohesive_primary_traction_first_MPa": _series(record, "cohesive", "S" + axis)[0],
                    "cohesive_primary_traction_last_MPa": _series(record, "cohesive", "S" + axis)[-1],
                } if axis in {"X", "Y", "Z"} else {}),
                **({
                    "outer_drive_D" + axis + "_first_mm": _series(record, "outer", axis)[0],
                    "outer_drive_D" + axis + "_last_mm": _series(record, "outer", axis)[-1],
                } if is_bond else {}),
            }
            for index, record in enumerate(records)
        ],
        "source_bound_inputs": {
            "run_receipt_sha256": receipt_sha,
            "case_sha256": receipt.get("case_sha256"),
            "engine_source_commit": receipt.get("source_commit"),
            "engine_binary_sha256": receipt.get("binary_sha256"),
            "runtime_identity_sha256": receipt.get("runtime_identity_sha256"),
            "converter_binary_sha256": SUPPORTED_THTOCSV_BINARY_SHA256,
            "converter_source_commit": THTOCSV_SOURCE_COMMIT,
            "converter_source_url": THTOCSV_SOURCE_URL,
            "conversion_stages": [{"stage": r["stage"],
                                    "native_history_sha256": r["native_history_sha256"],
                                    "titles_sha256": r["titles_sha256"],
                                    "csv_sha256": r["csv_sha256"],
                                    "conversion_receipt_sha256": r["conversion_sha256"]}
                                   for r in records],
            "LAW117_source_commit_basis": LAW117_SOURCE_COMMIT,
            "LAW117_source_url": LAW117_SOURCE_URL,
            "note": "Runtime/source identities bind the run receipt; no source-to-binary build claim is inferred from hashes alone.",
        },
        "measurement_definition": {
            "OFF_semantics": {"active": ACTIVE_OFF_VALUE, "dormant_or_deleted": INACTIVE_OFF_VALUE},
            "area_mm2_from_starter_geometry": area,
            "bond_normal_axis_from_case": axis if is_bond else None,
            "primary_displacement_component": None if axis == "combined" else "D" + axis,
            "primary_traction_component": None if axis == "combined" else (
                "SXZ" if mode == "II" else "S" + axis),
            "material_from_starter": material,
            "traction_units": "MPa = N/mm^2; multiply by measured cohesive face area for N.",
            "work_units": "N mm, trapezoidal integral of native cohesive stress × starter-face area × measured interface separation increment.",
            "reaction_force_source": "Converted titled /TH/NODE reaction channels; impulse finite-difference windows are source-specific per stage.",
        },
        "observed": {
            "off_values": sorted(set(off)),
            "off_active_sample_count": sum(x == ACTIVE_OFF_VALUE for x in off),
            "off_dormant_or_deleted_sample_count": sum(x == INACTIVE_OFF_VALUE for x in off),
            "off_transitions": transitions,
            "separation_component": "DX+DZ" if axis == "combined" else "D" + axis,
            "separation_min_mm": min(separation) if separation else None,
            "separation_max_mm": max(separation) if separation else None,
            "traction_component": "SZ and SXZ" if axis == "combined" else (
                "SXZ" if mode == "II" else "S" + axis),
            "traction_min_MPa": min(traction) if traction else None,
            "traction_max_MPa": max(traction) if traction else None,
            "cohesive_IE_first_N_mm": energy[0] if energy else None,
            "cohesive_IE_final_N_mm": energy[-1] if energy else None,
            "cohesive_IE_max_N_mm": max(energy) if energy else None,
            "native_cohesive_work": raw_work,
            "path_behavior": path_behavior,
            "work_minus_final_cohesive_IE_N_mm": (
                None if not energy else raw_work["selected_work_N_mm"] - float(energy[-1])),
            "constitutive_crosscheck": analytical,
            "support_reaction_force_windows": force_diag,
        },
        "activation": activation,
        "limits": [
            "Synthetic solver instrumentation only; not a kraft, NSSC, corrugated-board, or adhesive qualification.",
            "The all-dormant original activation history cannot establish activation from one sample even when native execution terminated normally.",
            "T01 REAC force samples are finite-difference window averages; endpoint force subtraction is diagnostic only.",
            "Cohesive work uses sampled native stresses and actual interface displacement; output-time quadrature can miss unresolved path curvature between samples.",
            "No quasi-static or specimen claim follows without retained kinetic/internal/global-energy and mesh/time-step sensitivity evidence.",
        ],
    }
    if len(records) > 1:
        result["restart_handoffs"] = []
        for previous, following in zip(records, records[1:]):
            prev_index = len(previous["times"]) - 1
            next_index = 0
            handoff = {
                "from_stage": previous["stage"], "to_stage": following["stage"],
                "time_gap_s": following["times"][0] - previous["times"][-1],
                "cohesive_IE_last_before_N_mm": _series(previous, "cohesive", "IE")[prev_index],
                "cohesive_IE_first_after_N_mm": _series(following, "cohesive", "IE")[next_index],
                "OFF_before": _series(previous, "cohesive", "OFF")[prev_index],
                "OFF_after": _series(following, "cohesive", "OFF")[next_index],
                "cohesive_SZ_before_MPa": _series(previous, "cohesive", "SZ")[prev_index],
                "cohesive_SZ_after_MPa": _series(following, "cohesive", "SZ")[next_index],
                "free_interface_DZ_before_mm": _series(previous, "top", "Z")[prev_index],
                "free_interface_DZ_after_mm": _series(following, "top", "Z")[next_index],
                "primary_axis": axis,
                "free_interface_primary_displacement_before_mm": _series(previous, "top", axis)[prev_index],
                "free_interface_primary_displacement_after_mm": _series(following, "top", axis)[next_index],
                "cohesive_primary_traction_before_MPa": _series(previous, "cohesive", "S" + axis)[prev_index],
                "cohesive_primary_traction_after_MPa": _series(following, "cohesive", "S" + axis)[next_index],
            }
            if is_bond:
                handoff["outer_drive_D" + axis + "_before_mm"] = _series(previous, "outer", axis)[prev_index]
                handoff["outer_drive_D" + axis + "_after_mm"] = _series(following, "outer", axis)[next_index]
            result["restart_handoffs"].append(handoff)
    result["_restart_series"] = {
        "times": [p["time_s"] for p in path],
        "dz_mm": [p["dz_mm"] for p in path],
        "sz_MPa": [p["sz_MPa"] for p in path],
        "ie_N_mm": [p["ie_N_mm"] for p in path],
        "primary_separation_mm": [p["primary_separation_mm"] for p in path],
        "primary_traction_MPa": [p["primary_traction_MPa"] for p in path],
    }
    if is_bond:
        top_mean = _series(records[-1], "top", axis)[-1]
        outer_mean = _series(records[-1], "outer", axis)[-1]
        final_sz = _series(records[-1], "cohesive", "S" + axis)[-1]
        final_solid = _series(records[-1], "solid", "S" + axis)[-1]
        pred = case.get("prediction", {})
        result["bond_interface"] = {
            "free_interface_nodes": columns["geometry"]["top_nodes"],
            "outer_driven_nodes": sorted(records[-1]["columns"]["outer"][axis]),
            "normal_axis": axis,
            "primary_displacement_component": "D" + axis,
            "primary_traction_component": "S" + axis,
            "final_free_interface_displacement_mm": top_mean,
            "final_outer_drive_displacement_mm": outer_mean,
            "final_adherend_extension_mm": (None if top_mean is None or outer_mean is None else outer_mean - top_mean),
            "final_incremental_interface_separation_since_birth_mm": (
                None if not path else path[-1]["primary_separation_mm"]),
            "final_cohesive_traction_MPa": final_sz,
            "final_solid_adherend_traction_MPa": final_solid,
            "cohesive_face_area_mm2": area,
            "adherend_stiffness_N_per_mm_from_case_prediction": pred.get("adherend_stiffness_N_per_mm"),
            "cohesive_force_N_from_native_stress": None if final_sz is None else float(final_sz) * area,
            "predicted_birth_time_s": pred.get("birth_time_s"),
            "predicted_birth_hold_force_N": pred.get("birth_hold_force_N"),
            "predicted_final_force_N": pred.get("final_force_N"),
            "final_force_error_vs_prediction_N": (
                None if final_sz is None or pred.get("final_force_N") is None else
                float(final_sz) * area - float(pred["final_force_N"])),
        }
    return result


def measure_suite(runs_root: Path, post_root: Path,
                  case_ids: list[str] | None = None) -> dict[str, Any]:
    runs_root, post_root = Path(runs_root), Path(post_root)
    candidates = []
    supported_schemas = {"cardboard.opencourant.adhesion.case.v1",
                         "cardboard.opencourant.bond.v1"}
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir() or not (path / "case.json").is_file() or not (path / "run.json").is_file():
            continue
        try:
            schema = _json(path / "case.json").get("schema")
        except MeasurementError:
            continue
        if schema in supported_schemas:
            candidates.append(path)
    if case_ids:
        requested = set(case_ids)
        candidates = [path for path in candidates if path.name in requested]
        missing = requested - {path.name for path in candidates}
        if missing:
            raise MeasurementError(f"requested run directories not found: {sorted(missing)}")
    results, errors = [], []
    for run_dir in candidates:
        try:
            results.append(measure_run(run_dir, post_root))
        except MeasurementError as exc:
            errors.append({"case_id": run_dir.name, "status": "measurement_error", "error": str(exc)})
    by_name = {row["run_id"]: row for row in results}
    continuous = by_name.get("explicit-law117-mode-i-open-failure")
    restart = by_name.get("explicit-law117-mode-i-open-failure-restart")
    if continuous and restart:
        # Retain a direct restart-versus-continuous comparison on common times.
        # Interpolate only numeric observables; OFF is checked separately at each restart sample.
        a = continuous["_restart_series"]
        b = restart["_restart_series"]
        common_times = [t for t in b["times"] if a["times"][0] <= t <= a["times"][-1]]
        diffs = {key: [] for key in ("dz_mm", "sz_MPa", "ie_N_mm")}
        for time_s in common_times:
            for key in diffs:
                ref = _interp(a["times"], a[key], time_s)
                value = _interp(b["times"], b[key], time_s)
                if ref is not None and value is not None:
                    diffs[key].append(abs(float(value) - ref))
        restart["restart_vs_continuous"] = {
            key + "_max_abs_difference": max(values) if values else None
            for key, values in diffs.items()
        }
        restart["restart_vs_continuous"]["common_samples"] = len(common_times)
        restart["restart_vs_continuous"]["method"] = "linear interpolation of continuous mode-I history at split-run output times; no tolerance verdict encoded"
    for row in results:
        row.pop("_restart_series", None)
    return {
        "schema": "cardboard.opencourant.bond-measurement.v1",
        "measurement_status": "completed_with_errors" if errors else "completed",
        "suite_limits": [
            "Measurements and coverage are not admission or physical qualification verdicts.",
            "Source semantics are pinned by exact converter and engine-source revision identities, without asserting converter build provenance.",
        ],
        "cases": results,
        "errors": errors,
    }


def _interp(times: list[float], values: list[float | None], time_s: float) -> float | None:
    if not times or time_s < times[0] or time_s > times[-1]:
        return None
    for i in range(1, len(times)):
        if time_s <= times[i]:
            left, right = values[i - 1], values[i]
            if left is None or right is None:
                return None
            dt = times[i] - times[i - 1]
            fraction = 0.0 if dt == 0 else (time_s - times[i - 1]) / dt
            return float(left) + fraction * (float(right) - float(left))
    return values[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="retained native runs directory")
    parser.add_argument("--post", type=Path, required=True, help="converted CSV/TITLES directory")
    parser.add_argument("--case", action="append", dest="case_ids", help="measure a case directory (repeatable)")
    parser.add_argument("--output", type=Path, help="write JSON report here; stdout by default")
    args = parser.parse_args(argv)
    try:
        report = measure_suite(args.runs, args.post, args.case_ids)
    except MeasurementError as exc:
        print(json.dumps({"schema": "cardboard.opencourant.bond-measurement.v1",
                          "measurement_status": "measurement_error", "error": str(exc)}, indent=2))
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0 if report["measurement_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
