#!/usr/bin/env python3
"""Adjudicate source-bound native forward ensembles against published fixtures.

This program does not run a solver, fit a constitutive law, or infer probability
distributions. It checks complete fixture histories against source-derived
intervals and explicit transfer bounds. Missing evidence remains missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cardboard_source_data as source_data
from cardboard_shipping_peak import first_local_peak_index


TOOL = Path(__file__).resolve()
REPO = TOOL.parents[2]
DEFAULT_CATALOG = REPO / "matter/examples/cardboard/source-data/catalog.json"
DEFAULT_DATA = REPO / "matter/examples/cardboard/source-data"


class EnsembleError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EnsembleError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise EnsembleError(f"{name} must be finite")
    return result


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnsembleError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise EnsembleError(f"{what} must be a JSON object")
    return result


def _source_identity(source: dict[str, Any]) -> str | None:
    return source.get("external_original_sha256") or source.get("source_text_extract_sha256")


def load_source_package(catalog_path: Path = DEFAULT_CATALOG,
                        data_dir: Path = DEFAULT_DATA) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    catalog = source_data.load_catalog(catalog_path)
    errors = source_data.validate_catalog(catalog, data_dir)
    if errors:
        raise EnsembleError("published source-data package failed validation: " + "; ".join(errors))
    catalog_hash = sha256(catalog_path)
    sources = {str(row["id"]): row for row in catalog.get("sources", [])}
    datasets: dict[str, dict[str, Any]] = {}
    for row in catalog.get("datasets", []):
        dataset_id = str(row["id"])
        source_ids = row.get("source_ids", [])
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise EnsembleError(f"{dataset_id}: source identity is missing")
        hashes = [_source_identity(sources[source_id]) for source_id in source_ids]
        dataset_path = data_dir / row["file"]
        copy = dict(row)
        copy["resolved_path"] = str(dataset_path.resolve())
        copy["source_asset_sha256s"] = hashes
        datasets[dataset_id] = copy
    return catalog, catalog_hash, datasets


# These keys define indivisible published fixture conditions for current data.
# A plan cannot split a plotted fixture into individual points to calibrate on
# some points and validate on the remainder.
CONDITION_KEYS: dict[str, tuple[str, ...]] = {
    "johnson_popil_pin_force_displacement": ("series_id", "defect_id"),
    "johnson_popil_pin_summary": ("series_id", "pin_adhesion_strength_N_mm"),
    "popil_2012_ect_table2": ("series_id", "board_code"),
    "popil_2012_ect_rate_fit": ("series_id", "speed_mm_min"),
    "mrowczynski_2021_box_bct": ("grade", "box_condition"),
    "aduke_2024_fold_ratio": ("board_grade", "condition"),
    "conde_2012_hma_joint": ("adhesive", "geometry", "loading_mode"),
    "haj_ali_2009_yield_anchors": ("component", "basis_weight_g_m2", "axis", "loading"),
    "robertsson_2023_eq18_compression_branch": ("branch_case",),
    "robertsson_2023_identity_energy_benchmark": ("material_grade",),
}

# These published cases are analogues/holdouts in the current catalog. Their
# source identity cannot be upgraded to exact same-grade evidence by choosing
# matching caller-supplied strings. `source_grade` must agree with an actual
# grade/board-code column when the source publishes one; pin/scored-bend data
# have no exact grade field and therefore remain explicit bounded analogues.
PUBLISHED_ANALOGUE_DATASETS = {
    "johnson_popil_pin_force_displacement",
    "popil_2012_ect_table2",
    "popil_2012_ect_rate_fit",
    "mrowczynski_2021_box_bct",
    "nagasawa_2012_fig9_scored_bend_envelope",
}
SOURCE_GRADE_COLUMNS = {
    "popil_2012_ect_table2": "board_code",
    "mrowczynski_2021_box_bct": "grade",
}
NATIVE_PROTOCOL_IDS = {
    "johnson_popil_pin_force_displacement": "pin_adhesion_50x160_six_pin_v1",
    "popil_2012_ect_table2": "ect_T839_edge_compression_v1",
    "popil_2012_ect_rate_fit": "ect_edge_rate_response_v1",
    "mrowczynski_2021_box_bct": "shipping_bct_without_offset_first_local_peak_v1",
    "nagasawa_2012_fig9_scored_bend_envelope": "rotary_scored_bend_moment_per_width_v1",
}
BCT_CAPACITY_SCHEMA = "numi.cardboard.shipping-capacity-qualification.v2"
BCT_STAGE_REPORT_SCHEMA = "numi.cardboard.stage-resolution-analysis.v1"
BCT_CAPACITY_PROPERTIES = frozenset({
    "schema", "status", "board_recipe_id", "source_physics_fingerprint",
    "device_program_fingerprint", "compiled_world_fingerprint",
    "source_binding_sha256", "assembled_matter_checkpoint_sha256",
    "assembled_rigid_sidecar_sha256", "nominal_curve_sha256",
    "nominal_peak_sample_index", "nominal_accepted_sample_count",
    "nominal_terminal_stroke_fraction", "nominal_full_stroke_completed",
    "nominal_rate_sensitivity_passed", "mesh_status", "mesh_report_file",
    "mesh_report_sha256", "mesh_level_count", "mesh_all_cases_complete",
    "mesh_finest_pair_within_tolerance", "mesh_finest_peak_difference",
    "mesh_finest_curve_rms", "mesh_finest_work_difference",
    "mesh_coarse_middle_peak_difference", "mesh_coarse_middle_curve_rms",
    "mesh_coarse_middle_work_difference", "time_status",
    "time_report_file", "time_report_sha256", "time_level_count",
    "time_all_cases_complete", "time_finest_pair_within_tolerance",
    "time_finest_peak_difference", "time_finest_curve_rms",
    "time_finest_work_difference", "time_coarse_middle_peak_difference",
    "time_coarse_middle_curve_rms", "time_coarse_middle_work_difference",
})
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
PIN_VISIBLE_SERIES = ("low_adhesion_white_glue_lines", "A")
PIN_VISIBLE_SOURCE_WINDOW = {
    "start_load_N": 30.0,
    "end": "first_post_peak_return_to_start_load",
}
PIN_VISIBLE_NATIVE_WINDOW = {
    "phase": "relative_opening",
    "start_load_N": 30.0,
    "end": "first_post_peak_return_to_start_load",
    "required_phases": ["platen_approach", "relative_opening",
                         "platen_withdrawal", "released_hold"],
}


def _validate_source_transfer_binding(condition: dict[str, Any],
                                      dataset: dict[str, Any],
                                      rows: list[dict[str, str]],
                                      plan_target_grade_id: Any) -> None:
    dataset_id = str(dataset.get("id", ""))
    if dataset_id not in PUBLISHED_ANALOGUE_DATASETS:
        return
    cid = str(condition.get("condition_id", "<unknown>"))
    transfer = condition.get("transfer")
    if not isinstance(transfer, dict):
        raise EnsembleError(f"{cid}: published analogue needs an explicit transfer object")
    target_grade = str(plan_target_grade_id or "").strip()
    if not target_grade:
        raise EnsembleError(f"{cid}: plan target_grade_id is required for source-grade admission")
    if str(transfer.get("target_grade", "")).strip() != target_grade:
        raise EnsembleError(f"{cid}: transfer target_grade does not match plan target_grade_id")
    kind = transfer.get("kind")
    if kind == "same_grade":
        # The catalog marks every audited dataset as an analogue or holdout;
        # none has an independent identity binding to the current recipe.
        raise EnsembleError(
            f"{cid}: {dataset_id} is cataloged as {dataset.get('split')!r}; "
            "same_grade is unavailable without a source-bound material identity"
        )
    if kind != "bounded_analogue":
        raise EnsembleError(f"{cid}: audited source requires bounded_analogue transfer")
    source_grade = str(transfer.get("source_grade", "")).strip()
    if not source_grade:
        raise EnsembleError(f"{cid}: bounded analogue requires an explicit source_grade")
    source_column = SOURCE_GRADE_COLUMNS.get(dataset_id)
    if source_column:
        actual = {str(row.get(source_column, "")).strip() for row in rows}
        if len(actual) != 1 or "" in actual or source_grade not in actual:
            raise EnsembleError(
                f"{cid}: source_grade must exactly match the selected published "
                f"{source_column} value(s) {sorted(actual)}"
            )
    if not str(transfer.get("rationale", "")).strip():
        raise EnsembleError(f"{cid}: bounded analogue requires a written transfer rationale")
    expected_protocol = NATIVE_PROTOCOL_IDS[dataset_id]
    if condition.get("native_protocol_id") != expected_protocol:
        raise EnsembleError(
            f"{cid}: native_protocol_id must be {expected_protocol!r}; source and native "
            "conditions cannot be inferred from units or a generic completed-step count"
        )


def _validate_pin_metric_window(condition: dict[str, Any], spec: dict[str, Any],
                                metric_name: str) -> None:
    selector = condition.get("selector", {})
    series = (str(selector.get("series_id", "")),
              str(selector.get("defect_id", "")))
    if series != PIN_VISIBLE_SERIES:
        raise EnsembleError(
            f"{condition.get('condition_id')}/{metric_name}: only the low-adhesion "
            "slide-15 series A has a visible post-peak return; the higher-response "
            "series is censored and cannot supply a global peak"
        )
    source_window = spec.get("source_window")
    if not isinstance(source_window, dict) or any(
            source_window.get(key) != value
            for key, value in PIN_VISIBLE_SOURCE_WINDOW.items()):
        raise EnsembleError(
            f"{condition.get('condition_id')}/{metric_name}: pin source metrics "
            "must use the published visible 30 N through first post-peak return window"
        )
    native_window = spec.get("native_window")
    if not isinstance(native_window, dict) or any(
            native_window.get(key) != value
            for key, value in PIN_VISIBLE_NATIVE_WINDOW.items()):
        raise EnsembleError(
            f"{condition.get('condition_id')}/{metric_name}: pin prediction must use "
            "the accepted relative_opening 30 N window and all four completed phases"
        )


UNIT_DIMENSION = {
    "N": "force", "kN": "force", "mm": "length", "m": "length",
    "N/mm": "force_per_length", "N/m": "force_per_length",
    "N*mm": "work", "N*m": "work", "J": "work",
    "kN/m": "force_per_length", "dimensionless": "dimensionless",
    "category": "category", "MPa": "pressure", "MPa^-1": "inverse_pressure",
    "deg": "angle", "rad": "angle", "s": "time",
    "N*m_per_m": "moment_per_width",
}
SOURCE_COLUMN_UNITS = {
    "load_N": "N", "displacement_mm": "mm", "BCT_N": "N",
    "pin_adhesion_strength_N_mm": "N/mm", "pin_stiffness_N_mm": "N/mm",
    "pin_adhesion_work_plot_value": "unresolved", "ECT_kN_m": "kN/m",
    "speed_mm_min": "mm/min", "stress_MPa": "MPa", "strain": "dimensionless",
    "value": "dimensionless", "creased_to_uncreased_stiffness_ratio": "dimensionless",
    "angle_deg": "deg", "moment_envelope_min_Nm_per_m": "N*m_per_m",
    "moment_envelope_max_Nm_per_m": "N*m_per_m",
    "readout_uncertainty_angle_deg": "deg",
    "readout_uncertainty_moment_Nm_per_m": "N*m_per_m",
}


def _expected_derived_unit(kind: str, x_unit: str | None, y_unit: str) -> str:
    if kind == "peak":
        return y_unit
    if kind in {"secant", "peak_branch_secant"}:
        return {("N", "mm"): "N/mm", ("N", "m"): "N/m"}.get((y_unit, x_unit), "")
    if kind == "work":
        return {("N", "mm"): "N*mm", ("N", "m"): "N*m"}.get((y_unit, x_unit), "")
    return "category" if kind == "category" else ""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _pin_source_threshold_window(rows: list[dict[str, str]], x_column: str,
                                 y_column: str, x_uncertainty_column: str | None,
                                 y_uncertainty_column: str | None,
                                 threshold: float, end_rule: str,
                                 metric_name: str) -> list[dict[str, Any]]:
    """Select the visible pin-plot loading window without using its unknown x origin."""
    x = [_finite(row.get(x_column), x_column) for row in rows]
    y = [_finite(row.get(y_column), y_column) for row in rows]
    if len(rows) < 2 or any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise EnsembleError(f"{metric_name}: source pin trace must be strictly ordered by displacement")
    x_unc = [_uncertainty(row, x_uncertainty_column) for row in rows]
    y_unc = [_uncertainty(row, y_uncertainty_column) for row in rows]

    def boundary(left: int, right: int, target: float) -> dict[str, Any]:
        if left == right or y[right] == y[left]:
            fraction = 0.0
        else:
            fraction = (target - y[left]) / (y[right] - y[left])
        result: dict[str, Any] = dict(rows[right])
        result[x_column] = x[left] + fraction * (x[right] - x[left])
        result[y_column] = target
        # Use the larger adjacent pixel-readout bound at an interpolated crossing.
        if x_uncertainty_column:
            result[x_uncertainty_column] = max(x_unc[left], x_unc[right])
        if y_uncertainty_column:
            result[y_uncertainty_column] = max(y_unc[left], y_unc[right])
        return result

    start: dict[str, Any] | None = None
    start_index = -1
    for index, value in enumerate(y):
        if value < threshold:
            continue
        if index == 0:
            if value != threshold:
                raise EnsembleError(f"{metric_name}: source pin trace starts above its declared load threshold")
            start, start_index = dict(rows[index]), index
        elif value == threshold:
            start, start_index = dict(rows[index]), index
        else:
            start, start_index = boundary(index - 1, index, threshold), index
        break
    if start is None:
        raise EnsembleError(f"{metric_name}: source pin trace never reaches its load threshold")

    peak_index = max(range(start_index, len(y)), key=lambda index: y[index])
    if end_rule == "peak":
        end: dict[str, Any] = dict(rows[peak_index])
    elif end_rule == "first_post_peak_return_to_start_load":
        end = None
        for index in range(peak_index + 1, len(y)):
            if y[index] > threshold:
                continue
            if y[index] == threshold:
                end = dict(rows[index])
            else:
                end = boundary(index - 1, index, threshold)
            break
        if end is None:
            raise EnsembleError(f"{metric_name}: source pin trace lacks its post-peak return to threshold")
    else:
        raise EnsembleError(f"{metric_name}: unsupported source window end rule {end_rule!r}")

    start_x = _finite(start.get(x_column), x_column)
    end_x = _finite(end.get(x_column), x_column)
    selected: list[dict[str, Any]] = [start]
    selected.extend(dict(rows[index]) for index in range(start_index, len(rows))
                    if x[index] > start_x and x[index] < end_x)
    if end_x > start_x:
        selected.append(end)
    if len(selected) < 2:
        raise EnsembleError(f"{metric_name}: source threshold window contains fewer than two points")
    return selected


def _selected_rows(condition: dict[str, Any], dataset: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    dataset_id = str(condition.get("dataset_id", ""))
    keys = CONDITION_KEYS.get(dataset_id)
    selector = condition.get("selector")
    if not keys or not isinstance(selector, dict):
        raise EnsembleError(f"{condition.get('condition_id')}: unsupported or missing full-condition selector")
    if set(selector) != set(keys):
        raise EnsembleError(f"{condition.get('condition_id')}: selector must contain exactly whole-fixture keys {keys}")
    if "row_indices" in condition or "rows" in condition:
        raise EnsembleError(f"{condition.get('condition_id')}: row-level splitting is prohibited")
    all_rows = _read_csv_rows(Path(dataset["resolved_path"]))
    matches = [r for r in all_rows if all(r.get(k) == str(v) for k, v in selector.items())]
    if not matches:
        raise EnsembleError(f"{condition.get('condition_id')}: selector matched no source rows")
    group_keys = {tuple(r.get(k, "") for k in keys) for r in matches}
    if len(group_keys) != 1:
        raise EnsembleError(f"{condition.get('condition_id')}: selector accidentally combines multiple fixture conditions")
    return matches, canonical_sha256({"dataset_id": dataset_id, "selector": selector})


def _uncertainty(row: dict[str, str], column: str | None) -> float:
    return 0.0 if not column else max(0.0, _finite(row.get(column), column))


def _derive_source_metric(rows: list[dict[str, str]], spec: dict[str, Any], metric_name: str,
                          dataset_id: str) -> dict[str, Any]:
    kind = spec.get("kind")
    unit = spec.get("unit")
    if unit not in UNIT_DIMENSION:
        raise EnsembleError(f"{metric_name}: unknown or unresolved unit {unit!r}")
    if kind == "category":
        column = str(spec.get("column", ""))
        values = {str(row.get(column, "")).strip() for row in rows}
        if "" in values:
            raise EnsembleError(f"{metric_name}: source does not report categorical failure locus in {column}")
        if len(values) != 1:
            raise EnsembleError(f"{metric_name}: source condition has mixed failure categories")
        return {"kind": kind, "unit": unit, "value": next(iter(values)), "source_row_count": len(rows)}

    y_column = str(spec.get("y_column", ""))
    if not y_column:
        y_column = str(spec.get("column", ""))
    y_unit = spec.get("y_unit") or SOURCE_COLUMN_UNITS.get(y_column)
    if not y_unit or y_unit == "unresolved":
        raise EnsembleError(f"{metric_name}: source ordinate unit is missing or unresolved")
    published_y_unit = SOURCE_COLUMN_UNITS.get(y_column)
    if published_y_unit and published_y_unit != y_unit:
        raise EnsembleError(f"{metric_name}: declared y unit {y_unit!r} conflicts with source column {y_column} ({published_y_unit})")
    x_column = str(spec.get("x_column", ""))
    x_unit = spec.get("x_unit") or SOURCE_COLUMN_UNITS.get(x_column)
    if x_column:
        published_x_unit = SOURCE_COLUMN_UNITS.get(x_column)
        if not x_unit or (published_x_unit and published_x_unit != x_unit):
            raise EnsembleError(f"{metric_name}: x-axis unit is missing or conflicts with source column {x_column}")
    expected_unit = _expected_derived_unit(str(kind), x_unit, str(y_unit))
    if unit != expected_unit:
        raise EnsembleError(f"{metric_name}: unit {unit!r} is incompatible with {kind} of {y_unit!r} by {x_unit!r}")
    y_unc_col = spec.get("y_uncertainty_column") or spec.get("uncertainty_column")
    selected_rows: list[dict[str, Any]] = list(rows)
    x_column = str(spec.get("x_column", ""))
    x_unit = spec.get("x_unit") or SOURCE_COLUMN_UNITS.get(x_column)
    x_unc_col = spec.get("x_uncertainty_column")
    source_window = spec.get("source_window")
    if source_window is not None:
        if dataset_id != "johnson_popil_pin_force_displacement" or not isinstance(source_window, dict):
            raise EnsembleError(f"{metric_name}: source_window is supported only for the published pin trace")
        if x_column != "displacement_mm" or x_unit != "mm" or y_column != "load_N" or y_unit != "N":
            raise EnsembleError(f"{metric_name}: pin source window requires displacement_mm and load_N")
        threshold = _finite(source_window.get("start_load_N"), f"{metric_name} source window start load")
        selected_rows = _pin_source_threshold_window(
            rows, x_column, y_column, x_unc_col, y_unc_col, threshold,
            str(source_window.get("end", "")), metric_name)
    values = [_finite(row.get(y_column), y_column) for row in selected_rows]
    y_unc = [_uncertainty(row, y_unc_col) for row in selected_rows]
    if kind == "peak":
        lo = max(value - error for value, error in zip(values, y_unc))
        hi = max(value + error for value, error in zip(values, y_unc))
        return {"kind": kind, "unit": unit, "lower": lo, "upper": hi,
                "center": max(values), "source_row_count": len(selected_rows),
                "source_window": source_window,
                "source_scope": "visible_30N_threshold_window_not_full_specimen_curve"
                if dataset_id == "johnson_popil_pin_force_displacement" else "selected_complete_source_condition"}

    if not x_column:
        raise EnsembleError(f"{metric_name}: {kind} requires x_column")
    x = [_finite(row.get(x_column), x_column) for row in selected_rows]
    x_unc = [_uncertainty(row, x_unc_col) for row in selected_rows]
    if len(selected_rows) < 2 or any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise EnsembleError(f"{metric_name}: full fixture trace must contain >=2 strictly ordered x points")
    if kind in {"secant", "peak_branch_secant"}:
        end_index = max(range(len(values)), key=values.__getitem__) if kind == "peak_branch_secant" else len(values) - 1
        a = (values[end_index] - values[0]) / (x[end_index] - x[0])
        extrema = []
        for x0, y0, x1, y1 in itertools.product(
                (x[0] - x_unc[0], x[0] + x_unc[0]),
                (values[0] - y_unc[0], values[0] + y_unc[0]),
                (x[end_index] - x_unc[end_index], x[end_index] + x_unc[end_index]),
                (values[end_index] - y_unc[end_index], values[end_index] + y_unc[end_index])):
            denominator = x1 - x0
            if denominator > 0:
                extrema.append((y1 - y0) / denominator)
        if not extrema:
            raise EnsembleError(f"{metric_name}: uncertainty makes endpoint stiffness undefined")
        return {"kind": kind, "unit": unit, "lower": min(extrema), "upper": max(extrema),
                "center": a, "definition": "first-to-peak secant over the plotted loading branch; derived descriptor, not a source-reported stiffness" if kind == "peak_branch_secant" else "full-window endpoint secant; derived descriptor, not a source-reported stiffness",
                "source_row_count": len(selected_rows), "source_window": source_window,
                "source_scope": "visible_30N_threshold_window_not_full_specimen_curve"
                if dataset_id == "johnson_popil_pin_force_displacement" else "selected_complete_source_condition"}
    if kind == "work":
        lower = upper = center = 0.0
        for i in range(len(selected_rows) - 1):
            dx = x[i + 1] - x[i]
            dx_min = max(0.0, dx - x_unc[i] - x_unc[i + 1])
            dx_max = dx + x_unc[i] + x_unc[i + 1]
            ysum_min = values[i] - y_unc[i] + values[i + 1] - y_unc[i + 1]
            ysum_max = values[i] + y_unc[i] + values[i + 1] + y_unc[i + 1]
            if min(ysum_min, ysum_max) < 0:
                raise EnsembleError(f"{metric_name}: work bounds require nonnegative force ordinates")
            lower += 0.5 * ysum_min * dx_min
            upper += 0.5 * ysum_max * dx_max
            center += 0.5 * (values[i] + values[i + 1]) * dx
        return {"kind": kind, "unit": unit, "lower": lower, "upper": upper, "center": center,
                "definition": "signed trapezoid over the selected source trace window; not total test work",
                "source_row_count": len(selected_rows), "source_window": source_window,
                "source_scope": "visible_30N_threshold_window_not_full_specimen_curve"
                if dataset_id == "johnson_popil_pin_force_displacement" else "selected_complete_source_condition"}
    raise EnsembleError(f"{metric_name}: unsupported metric derivation {kind!r}")


def _transfer_interval(source_metric: dict[str, Any], transfer: dict[str, Any],
                       metric_name: str) -> tuple[float, float] | tuple[str, str]:
    if transfer.get("kind") not in {"same_grade", "bounded_analogue"}:
        raise EnsembleError(f"{metric_name}: explicit same-grade or bounded-analogue transfer is required")
    if transfer.get("kind") == "same_grade":
        if not transfer.get("source_grade") or transfer.get("source_grade") != transfer.get("target_grade"):
            raise EnsembleError(f"{metric_name}: same-grade transfer requires identical explicit source and target grade IDs")
    elif (not str(transfer.get("rationale", "")).strip() or not transfer.get("source_grade") or
          not transfer.get("target_grade")):
        raise EnsembleError(f"{metric_name}: bounded analogue requires grade IDs and a written transfer rationale")
    bounds = transfer.get("metric_scale_bounds", {}).get(metric_name)
    if source_metric["kind"] == "category":
        if transfer.get("kind") == "same_grade" and transfer.get("source_grade") == transfer.get("target_grade"):
            return source_metric["value"], source_metric["value"]
        mapping = transfer.get("category_map", {}).get(metric_name, {})
        values = mapping.get(source_metric["value"], [])
        if not values:
            raise EnsembleError(f"{metric_name}: analogue category mapping is missing")
        return str(values[0]), str(values[-1])
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise EnsembleError(f"{metric_name}: explicit published-data transfer scale bounds are required")
    low, high = (_finite(bounds[0], f"{metric_name} transfer lower"), _finite(bounds[1], f"{metric_name} transfer upper"))
    if low <= 0 or high < low:
        raise EnsembleError(f"{metric_name}: transfer bounds must be positive and ordered")
    if transfer.get("kind") == "bounded_analogue" and low == high:
        raise EnsembleError(f"{metric_name}: bounded analogue must include a nonzero transfer uncertainty range")
    return source_metric["lower"] * low, source_metric["upper"] * high


def _transfer_categories(source_metric: dict[str, Any], transfer: dict[str, Any],
                         metric_name: str) -> set[str]:
    if transfer.get("kind") not in {"same_grade", "bounded_analogue"}:
        raise EnsembleError(f"{metric_name}: explicit same-grade or bounded-analogue transfer is required")
    if transfer.get("kind") == "same_grade":
        if not transfer.get("source_grade") or transfer.get("source_grade") != transfer.get("target_grade"):
            raise EnsembleError(f"{metric_name}: same-grade transfer requires identical explicit source and target grade IDs")
        return {str(source_metric["value"])}
    if (not str(transfer.get("rationale", "")).strip() or not transfer.get("source_grade") or
            not transfer.get("target_grade")):
        raise EnsembleError(f"{metric_name}: bounded analogue requires grade IDs and a written transfer rationale")
    mapping = transfer.get("category_map", {}).get(metric_name, {})
    values = mapping.get(source_metric["value"], []) if isinstance(mapping, dict) else []
    if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
        raise EnsembleError(f"{metric_name}: analogue category mapping must list allowed category labels")
    return set(values)


def _native_metric_window(history: dict[str, Any], spec: dict[str, Any],
                          x_name: str, y_name: str, metric_name: str,
                          x_scale: float = 1.0) -> list[dict[str, Any]]:
    samples = history.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise EnsembleError(f"{metric_name}: native history needs at least two accepted samples")
    window = spec.get("native_window")
    if window is None:
        if any(sample.get("accepted") is False for sample in samples):
            raise EnsembleError(f"{metric_name}: rejected attempts cannot be included in native history samples")
        return samples
    if not isinstance(window, dict):
        raise EnsembleError(f"{metric_name}: native_window must be an object")
    required_phases = window.get("required_phases", [])
    if required_phases:
        completed = history.get("completed_phases")
        if (not isinstance(completed, list) or
                not set(required_phases).issubset({str(item) for item in completed})):
            raise EnsembleError(f"{metric_name}: native protocol phase completion is missing")
        # A phase-name list is a declaration, not evidence that the phase ran.
        # Require an accepted sample in each required phase so a loading prefix
        # cannot be relabeled as a complete approach/open/withdraw/release run.
        accepted_phase_samples = {
            str(sample.get("phase"))
            for sample in samples
            if isinstance(sample, dict) and sample.get("accepted") is True
        }
        absent_samples = sorted(set(required_phases) - accepted_phase_samples)
        if absent_samples:
            raise EnsembleError(
                f"{metric_name}: required protocol phases have no accepted samples: "
                + ", ".join(absent_samples))
    phase = window.get("phase")
    threshold = _finite(window.get("start_load_N"), f"{metric_name} native window start load")
    if not isinstance(phase, str) or not phase:
        raise EnsembleError(f"{metric_name}: native window requires an exact accepted phase name")
    phase_samples = []
    for sample in samples:
        if sample.get("phase") != phase:
            continue
        if not isinstance(sample.get("accepted"), bool):
            raise EnsembleError(f"{metric_name}: phase samples must label accepted/rejected status")
        if sample["accepted"]:
            phase_samples.append(sample)
    if len(phase_samples) < 2:
        raise EnsembleError(f"{metric_name}: native phase has fewer than two accepted points")
    x = [_finite(sample.get(x_name), x_name) * x_scale for sample in phase_samples]
    y = [_finite(sample.get(y_name), y_name) for sample in phase_samples]
    if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise EnsembleError(f"{metric_name}: accepted native phase is not strictly increasing in displacement")

    start = None
    for index, value in enumerate(y):
        if value < threshold:
            continue
        if index == 0:
            if value != threshold:
                raise EnsembleError(f"{metric_name}: phase begins above its declared source-load threshold")
            start = phase_samples[0]
        else:
            left_y = y[index - 1]
            if left_y > threshold:
                raise EnsembleError(f"{metric_name}: no first upward source-load crossing in native phase")
            if value == threshold:
                start = phase_samples[index]
            else:
                fraction = (threshold - left_y) / (value - left_y)
                start = dict(phase_samples[index])
                raw_left = _finite(phase_samples[index - 1].get(x_name), x_name)
                raw_right = _finite(phase_samples[index].get(x_name), x_name)
                start[x_name] = raw_left + fraction * (raw_right - raw_left)
                start[y_name] = threshold
        if start is not None:
            start_index = index
            break
    if start is None:
        raise EnsembleError(f"{metric_name}: native phase never reaches its source-load threshold")

    tail = phase_samples[start_index:]
    peak_local = max(range(len(tail)), key=lambda i: _finite(tail[i].get(y_name), y_name))
    peak_index = start_index + peak_local
    peak_sample = phase_samples[peak_index]
    end_rule = window.get("end")
    if end_rule == "peak":
        end = peak_sample
    elif end_rule == "first_post_peak_return_to_start_load":
        end = None
        for index in range(peak_index + 1, len(phase_samples)):
            if y[index] > threshold:
                continue
            previous_index = index - 1
            if y[index] == threshold:
                end = phase_samples[index]
            else:
                fraction = (threshold - y[previous_index]) / (y[index] - y[previous_index])
                end = dict(phase_samples[index])
                raw_left = _finite(phase_samples[previous_index].get(x_name), x_name)
                raw_right = _finite(phase_samples[index].get(x_name), x_name)
                end[x_name] = raw_left + fraction * (raw_right - raw_left)
                end[y_name] = threshold
            break
        if end is None:
            raise EnsembleError(f"{metric_name}: native phase lacks its required post-peak return to start load")
    elif end_rule == "phase_end":
        end = phase_samples[-1]
    else:
        raise EnsembleError(f"{metric_name}: unsupported native window end rule {end_rule!r}")

    end_x = _finite(end.get(x_name), x_name)
    selected = [start]
    selected.extend(sample for sample in phase_samples[start_index:]
                    if _finite(sample.get(x_name), x_name) > _finite(start.get(x_name), x_name) and
                    _finite(sample.get(x_name), x_name) < end_x)
    if end is not start:
        selected.append(end)
    if len(selected) < 2:
        raise EnsembleError(f"{metric_name}: native source-load window contains fewer than two points")
    return selected


def _derive_native_metric(history: dict[str, Any], spec: dict[str, Any], metric_name: str) -> dict[str, Any]:
    kind = spec.get("prediction_kind", spec.get("kind"))
    unit = spec.get("unit")
    if unit not in UNIT_DIMENSION:
        raise EnsembleError(f"{metric_name}: unknown or unresolved prediction unit {unit!r}")
    if kind == "category":
        value = history.get("outcomes", {}).get(str(spec.get("prediction_outcome", "")))
        if not isinstance(value, str) or not value:
            raise EnsembleError(f"{metric_name}: native outcome category is missing")
        return {"kind": kind, "unit": unit, "value": value}
    x_name = str(spec.get("prediction_x", spec.get("x_column", "")))
    y_name = str(spec.get("prediction_y", spec.get("y_column", spec.get("column", ""))))
    y_unit = history.get("units", {}).get(y_name)
    expected_x_unit = spec.get("x_unit")
    expected_y_unit = spec.get("y_unit")
    input_x_unit = spec.get("prediction_x_unit", expected_x_unit)
    x_scale = _finite(spec.get("prediction_x_scale", 1.0), f"{metric_name} prediction x scale")
    if x_scale <= 0.0:
        raise EnsembleError(f"{metric_name}: prediction x scale must be positive")
    if expected_y_unit not in UNIT_DIMENSION:
        raise EnsembleError(f"{metric_name}: source y unit declaration is required")
    if y_unit != expected_y_unit:
        raise EnsembleError(f"{metric_name}: incompatible y unit {y_unit!r}; expected {expected_y_unit!r}")
    if x_name:
        actual_x_unit = history.get("units", {}).get(x_name)
        if input_x_unit not in UNIT_DIMENSION or actual_x_unit != input_x_unit:
            raise EnsembleError(f"{metric_name}: incompatible native x unit {actual_x_unit!r}; expected {input_x_unit!r}")
    elif spec.get("native_window") is not None or kind != "peak":
        raise EnsembleError(f"{metric_name}: native x quantity is required for a window, stiffness, or work metric")
    samples = _native_metric_window(history, spec, x_name, y_name, metric_name, x_scale)
    y = [_finite(sample.get(y_name), y_name) for sample in samples]
    if kind == "peak":
        if unit != expected_y_unit:
            raise EnsembleError(f"{metric_name}: peak output unit {unit!r} does not match ordinate unit {expected_y_unit!r}")
        return {"kind": kind, "unit": unit, "value": max(y)}
    if expected_x_unit not in UNIT_DIMENSION:
        raise EnsembleError(f"{metric_name}: source x unit declaration is required")
    if input_x_unit != expected_x_unit and "prediction_x_scale" not in spec:
        raise EnsembleError(f"{metric_name}: converting native x units requires an explicit prediction_x_scale")
    x_unit = expected_x_unit
    expected_metric_unit = _expected_derived_unit(str(kind), expected_x_unit, str(expected_y_unit))
    if unit != expected_metric_unit:
        raise EnsembleError(f"{metric_name}: predicted unit {unit!r} is incompatible with {kind} of {expected_y_unit!r} by {expected_x_unit!r}")
    x = [_finite(sample.get(x_name), x_name) * x_scale for sample in samples]
    if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise EnsembleError(f"{metric_name}: native fixture trace is incomplete, unordered, or not a monotone loading window")
    if kind == "secant":
        endpoint = max(range(len(y)), key=y.__getitem__) if spec.get("native_secant_endpoint") == "peak" else len(y) - 1
        value = (y[endpoint] - y[0]) / (x[endpoint] - x[0])
        return {"kind": kind, "unit": unit, "value": value}
    if kind == "work":
        value = sum(0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]) for i in range(len(x) - 1))
        return {"kind": kind, "unit": unit, "value": value}
    raise EnsembleError(f"{metric_name}: unsupported metric derivation {kind!r}")


def _metric_bounds(source_row: dict[str, str], spec: dict[str, Any], metric_name: str) -> dict[str, Any]:
    # A printed unit conflict is never repaired by a caller-provided unit.
    status_column = spec.get("unit_status_column")
    if status_column:
        statuses = {str(source_row.get(status_column, "")) for source_row in [source_row]}
        if any(status and status != "resolved" for status in statuses):
            raise EnsembleError(f"{metric_name}: source marks this observable's unit unresolved")
    return {}


def _load_conditions(plan: dict[str, Any], datasets: dict[str, dict[str, Any]],
                     catalog_hash: str) -> tuple[dict[str, Any], list[str]]:
    conditions = plan.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise EnsembleError("plan requires whole-fixture conditions")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    calibration_ids: set[str] = set()
    validation_ids: set[str] = set()
    loaded: dict[str, Any] = {}
    blockers: list[str] = []
    for condition in conditions:
        cid = str(condition.get("condition_id", ""))
        if not cid or cid in seen_ids:
            raise EnsembleError("condition IDs must be present and unique")
        seen_ids.add(cid)
        role = condition.get("role")
        if role not in {"calibration", "validation", "holdout"}:
            raise EnsembleError(f"{cid}: role must be calibration, validation, or holdout")
        if role == "calibration": calibration_ids.add(cid)
        if role == "validation": validation_ids.add(cid)
        dataset_id = str(condition.get("dataset_id", ""))
        if dataset_id not in datasets:
            raise EnsembleError(f"{cid}: dataset {dataset_id!r} is not in the validated package")
        dataset = datasets[dataset_id]
        rows, group_signature = _selected_rows(condition, dataset)
        if group_signature in seen_groups:
            raise EnsembleError(f"{cid}: same published fixture group appears more than once across splits")
        seen_groups.add(group_signature)
        if condition.get("source_dataset_sha256") != dataset.get("sha256"):
            raise EnsembleError(f"{cid}: missing or mismatched published dataset SHA-256")
        source_ids = dataset.get("source_ids", [])
        if condition.get("source_id") not in source_ids:
            raise EnsembleError(f"{cid}: source_id does not match the dataset catalog")
        source = next(s for s in plan["_sources"] if s["id"] == condition.get("source_id"))
        source_hash = _source_identity(source)
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            raise EnsembleError(f"{cid}: publisher source asset/text SHA-256 is unavailable or malformed")
        if condition.get("source_asset_sha256") != source_hash:
            raise EnsembleError(f"{cid}: missing or mismatched source asset/text SHA-256")
        if condition.get("source_catalog_sha256") != catalog_hash:
            raise EnsembleError(f"{cid}: missing or mismatched source catalog SHA-256")
        _validate_source_transfer_binding(
            condition, dataset, rows, plan.get("target_grade_id"))
        split = str(dataset.get("split", "")).lower()
        family = str(condition.get("parameter_family", ""))
        allowlist = dataset.get("calibration_parameter_families")
        if allowlist is None and dataset_id == "haj_ali_2009_yield_anchors":
            allowlist = ["paper_yield_anchor"]
        if role == "calibration":
            if "holdout" in split or "validation" in split or "analogue" in split:
                blockers.append(f"{cid}: catalog split {dataset.get('split')!r} is holdout/analogue and cannot be calibrated")
            elif not family or family not in (allowlist or []):
                blockers.append(f"{cid}: catalog does not admit calibration of parameter family {family or '<missing>'!r}")
        metric_specs = condition.get("metrics", {})
        if not isinstance(metric_specs, dict) or not metric_specs:
            raise EnsembleError(f"{cid}: metric definitions are required")
        if dataset_id == "mrowczynski_2021_box_bct":
            if condition.get("selector", {}).get("box_condition") != "without_offset":
                raise EnsembleError(
                    f"{cid}: BCT full-peak comparison is unsupported for offset peaks; "
                    "the current capacity artifact qualifies only the complete nominal without-offset compression branch")
            for metric_name, spec in metric_specs.items():
                if spec.get("kind") == "unavailable":
                    continue
                if (spec.get("kind") != "peak" or spec.get("column") != "BCT_N" or
                        spec.get("prediction_x") != "stroke_m" or
                        spec.get("prediction_y") != "force_N" or
                        spec.get("native_window") is not None or
                        spec.get("x_unit") != "m" or
                        spec.get("prediction_x_unit", spec.get("x_unit")) != "m" or
                        spec.get("y_unit") != "N" or spec.get("unit") != "N"):
                    raise EnsembleError(
                        f"{cid}/{metric_name}: BCT requires a force peak from the complete nominal force-stroke curve in SI units")
        derived: dict[str, Any] = {}
        for name, spec in metric_specs.items():
            if spec.get("kind") == "unavailable":
                blockers.append(f"{cid}: {name} unavailable: {spec.get('reason', 'no published observable')}")
                continue
            if dataset_id == "johnson_popil_pin_force_displacement":
                _validate_pin_metric_window(condition, spec, name)
            if name in {"pin_work_summary", "pin_adhesion_work_plot_value"} or spec.get("column") == "pin_adhesion_work_plot_value":
                if dataset_id == "johnson_popil_pin_summary":
                    raise EnsembleError(f"{cid}: pin summary work is excluded because its N-m label conflicts with the slide-15 N-by-mm integral")
            # Current source package explicitly flags the slide-16 work unit as unresolved.
            if dataset_id == "johnson_popil_pin_summary" and any(
                    row.get("pin_adhesion_work_unit_status") == "unresolved_conflict_with_slide15_integral_units" for row in rows):
                if spec.get("column") == "pin_adhesion_work_plot_value" or name.lower().find("work") >= 0:
                    raise EnsembleError(f"{cid}: pin summary work unit is unresolved; do not convert N-m to N*mm")
            try:
                derived[name] = _derive_source_metric(rows, spec, name, dataset_id)
            except EnsembleError as exc:
                blockers.append(f"{cid}: {exc}")
        loaded[cid] = {"condition": condition, "dataset": dataset, "rows": rows,
                       "group_signature": group_signature, "observations": derived,
                       "source_asset_sha256": source_hash}
    overlap = calibration_ids & validation_ids
    if overlap:
        raise EnsembleError("condition IDs reused in calibration and validation: " + ", ".join(sorted(overlap)))
    return loaded, blockers


def _validate_candidate(candidate: dict[str, Any]) -> None:
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise EnsembleError("candidate_id is required")
    role = candidate.get("role")
    if role not in {"fit_candidate", "unbonded_control", "tied_control"}:
        raise EnsembleError(f"{candidate_id}: invalid role")
    if role == "fit_candidate":
        if candidate.get("origin") not in {"user_supplied", "sensitivity_grid"}:
            raise EnsembleError(f"{candidate_id}: parameter origin must be explicit; no default candidate grid is generated")
        params = candidate.get("parameters")
        if not isinstance(params, dict) or not params:
            raise EnsembleError(f"{candidate_id}: candidate parameter values are required")
        fitted = 0
        for name, p in params.items():
            if not isinstance(p, dict) or not p.get("family") or not p.get("unit") or not isinstance(p.get("fit"), bool):
                raise EnsembleError(f"{candidate_id}:{name}: value, family, unit and explicit fit flag are required")
            value = _finite(p.get("value"), f"{candidate_id}:{name}")
            if value < 0:
                raise EnsembleError(f"{candidate_id}:{name}: negative parameter value")
            fitted += int(p["fit"])
        if not fitted:
            raise EnsembleError(f"{candidate_id}: at least one parameter must be marked fit=true")
    else:
        if candidate.get("parameters", {}) not in ({}, None):
            raise EnsembleError(f"{candidate_id}: limiting controls must not enter parameter fitting")
        if candidate.get("control_definition") not in {"unbonded", "perfectly_tied"}:
            raise EnsembleError(f"{candidate_id}: explicit limiting-control definition required")
    body = {k: v for k, v in candidate.items() if k != "candidate_sha256"}
    if candidate.get("candidate_sha256") != canonical_sha256(body):
        raise EnsembleError(f"{candidate_id}: candidate config SHA-256 mismatch")


def _capacity_properties(path: Path) -> dict[str, str]:
    """Parse the producer's exact v2 key=value schema; never infer omitted gates."""
    if path.stat().st_size > 1024 * 1024:
        raise EnsembleError("BCT capacity qualification properties exceed the bounded 1 MiB input size")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EnsembleError(f"cannot read BCT capacity qualification: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EnsembleError(f"BCT capacity qualification line {line_number} is not key=value")
        key, value = line.split("=", 1)
        if key not in BCT_CAPACITY_PROPERTIES or not value or key in values:
            raise EnsembleError(f"BCT capacity qualification has unknown, empty, or duplicate field {key!r}")
        values[key] = value
    if set(values) != BCT_CAPACITY_PROPERTIES:
        missing = sorted(BCT_CAPACITY_PROPERTIES - set(values))
        raise EnsembleError("BCT capacity qualification is missing required v2 fields: " + ", ".join(missing))
    return values


def _capacity_int(values: dict[str, str], name: str) -> int:
    raw = values[name]
    if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
        raise EnsembleError(f"BCT capacity field {name} is not an unsigned decimal integer")
    return int(raw)


def _capacity_float(values: dict[str, str], name: str) -> float:
    return _finite(values[name], f"BCT capacity field {name}")


def _capacity_bool(values: dict[str, str], name: str) -> bool:
    if values[name] not in {"true", "false"}:
        raise EnsembleError(f"BCT capacity field {name} must be literal true or false")
    return values[name] == "true"


def _bct_artifact_path(run_dir: Path, artifact_hashes: dict[str, Any],
                       key: str) -> tuple[Path, str]:
    item = artifact_hashes.get(key)
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise EnsembleError(f"BCT full-peak admission requires the {key} artifact")
    path_text = item["path"]
    path = (run_dir / path_text).resolve()
    if path.parent != run_dir.resolve() or not path.is_file():
        raise EnsembleError(f"BCT artifact {key} is missing or escapes its native run directory")
    digest = item.get("sha256")
    if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None or sha256(path) != digest:
        raise EnsembleError(f"BCT artifact {key} has a missing or mismatched SHA-256")
    return path, digest


def _report_metric(report: dict[str, Any], pair_index: int,
                   key: str) -> float:
    pairs = report.get("pair_reports")
    if not isinstance(pairs, list) or len(pairs) != 2 or pair_index not in (0, 1):
        raise EnsembleError("BCT refinement report must retain both adjacent pairs")
    pair = pairs[pair_index]
    if (not isinstance(pair, dict) or
            pair.get("status") != "descriptive_metrics_available_with_unresolved_observables"):
        raise EnsembleError("BCT refinement report contains an incomplete or incompatible adjacent pair")
    metrics = pair.get("metrics", {}).get("compression_force_work", {})
    if (not isinstance(metrics, dict) or metrics.get("status") != "available" or
            metrics.get("comparison_is_complete_endpoint") is not True):
        raise EnsembleError("BCT refinement pair lacks complete accepted compression metrics")
    return _finite(metrics.get(key), f"BCT refinement {key}")


def _verify_bct_rate_history(path: Path, inside_height: float,
                            nominal_curve: list[tuple[float, float]]) -> None:
    """Recheck the producer's full nominal/half-speed accepted branches."""
    required = {"phase", "accepted", "stroke_m", "macro_observation_duration_s",
                "control_step_count", "control_step_dt_min_s", "control_step_dt_max_s",
                "platen_mean_force_y_N", "platen_reaction_samples"}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise EnsembleError("BCT compression ledger lacks accepted-branch and actual-timestep columns")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EnsembleError(f"cannot parse BCT compression branch ledger: {exc}") from exc

    branches = (
        ("nominal_12_5_mm_min", "nominal_12_5_mm_min_contact_origin",
         "nominal_12_5_mm_min_contact_approach",
         "nominal_12_5_mm_min_slow_compression", 12.5e-3 / 60.0),
        ("half_speed_6_25_mm_min", "half_speed_6_25_mm_min_contact_origin",
         "half_speed_6_25_mm_min_contact_approach",
         "half_speed_6_25_mm_min_slow_compression", 6.25e-3 / 60.0),
    )
    curves: list[list[tuple[float, float]]] = []
    selected_peaks: list[float] = []
    for name, origin_phase, approach_phase, phase, expected_speed in branches:
        origins = [row for row in rows if row.get("phase") == origin_phase]
        approaches = [row for row in rows if row.get("phase") == approach_phase]
        branch_rows = [row for row in rows if row.get("phase") == phase]
        if (len(origins) != 1 or not approaches or len(branch_rows) < 2 or
                any(row.get("accepted") != "1" for row in origins + approaches + branch_rows)):
            raise EnsembleError(f"BCT {name} branch is missing an accepted contact origin or complete loading rows")
        origin = origins[0]
        origin_stroke = _finite(origin.get("stroke_m"), f"{name} origin stroke")
        origin_force = _finite(origin.get("platen_mean_force_y_N"), f"{name} origin force")
        if origin_stroke != 0.0 or origin_force != 0.0:
            raise EnsembleError(f"BCT {name} does not start at the accepted zero-force stroke origin")
        curve = [(origin_stroke, origin_force)]
        curve_contacts = [0.0]
        crossing = [row for row in approaches
                    if _finite(row.get("platen_reaction_samples"), "BCT contact reaction sample count") > 0.0]
        if len(crossing) != 1:
            raise EnsembleError(f"BCT {name} must retain exactly one accepted first-contact stroke sample")
        crossing_row = crossing[0]
        crossing_stroke = _finite(crossing_row.get("stroke_m"), f"{name} contact crossing stroke")
        crossing_force = _finite(crossing_row.get("platen_mean_force_y_N"), f"{name} contact crossing force")
        crossing_duration = _finite(crossing_row.get("macro_observation_duration_s"), f"{name} contact duration")
        if crossing_stroke <= 0.0 or crossing_force < 0.0 or crossing_duration <= 0.0:
            raise EnsembleError(f"BCT {name} first-contact stroke sample is incomplete")
        crossing_contacts = _finite(crossing_row.get("platen_reaction_samples"),
                                     f"{name} contact sample count")
        if crossing_contacts <= 0.0:
            raise EnsembleError(f"BCT {name} first-contact record has no accepted contact samples")
        curve.append((crossing_stroke, crossing_force))
        curve_contacts.append(crossing_contacts)
        elapsed = crossing_duration
        for index, row in enumerate(branch_rows):
            stroke = _finite(row.get("stroke_m"), f"{name} row {index} stroke")
            force = _finite(row.get("platen_mean_force_y_N"), f"{name} row {index} force")
            contacts = _finite(row.get("platen_reaction_samples"), f"{name} row {index} contact samples")
            duration = _finite(row.get("macro_observation_duration_s"), f"{name} duration")
            dt_min = _finite(row.get("control_step_dt_min_s"), f"{name} minimum dt")
            dt_max = _finite(row.get("control_step_dt_max_s"), f"{name} maximum dt")
            count_text = row.get("control_step_count", "")
            if (stroke <= curve[-1][0] or force < 0.0 or contacts < 0.0 or duration <= 0.0 or
                    dt_min <= 0.0 or dt_max < dt_min or
                    re.fullmatch(r"[1-9][0-9]*", count_text) is None):
                raise EnsembleError(f"BCT {name} has a nonmonotone curve or invalid accepted timestep summary")
            count = int(count_text)
            allowance = max(1e-12, duration * 1e-9)
            if duration < count * dt_min - allowance or duration > count * dt_max + allowance:
                raise EnsembleError(f"BCT {name} macro duration is inconsistent with its accepted dt bounds/count")
            elapsed += duration
            curve.append((stroke, force))
            curve_contacts.append(contacts)
        selected_peak = first_local_peak_index(
            (point[0] for point in curve), (point[1] for point in curve), curve_contacts)
        if selected_peak is None:
            raise EnsembleError(f"BCT {name} branch has no qualified first local compression peak")
        selected_peaks.append(curve[selected_peak][1])
        final_fraction = curve[-1][0] / inside_height
        if not math.isclose(final_fraction, 0.70, rel_tol=0.0, abs_tol=1e-9):
            raise EnsembleError(f"BCT {name} accepted curve does not reach the complete 70-percent stroke endpoint")
        observed_speed = curve[-1][0] / elapsed
        if not math.isclose(observed_speed, expected_speed, rel_tol=1e-6, abs_tol=1e-12):
            raise EnsembleError(f"BCT {name} accepted duration does not match its declared physical compression rate")
        curves.append(curve)

    if len(curves[0]) != len(nominal_curve) or any(
            not math.isclose(a[0], b[0], rel_tol=0.0, abs_tol=1e-14) or
            not math.isclose(a[1], b[1], rel_tol=0.0, abs_tol=1e-10)
            for a, b in zip(curves[0], nominal_curve)):
        raise EnsembleError("BCT full nominal curve artifact does not match the accepted nominal branch ledger")
    nominal, half = curves
    common = min(nominal[-1][0], half[-1][0])
    knots = sorted({0.0, common, *(x for x, _ in nominal if 0.0 < x < common),
                    *(x for x, _ in half if 0.0 < x < common)})

    def at(curve: list[tuple[float, float]], x: float) -> float:
        if x <= curve[0][0]:
            return curve[0][1]
        for left, right in zip(curve, curve[1:]):
            if left[0] <= x <= right[0]:
                t = (x - left[0]) / (right[0] - left[0])
                return left[1] + t * (right[1] - left[1])
        return curve[-1][1]

    def integrate(values: list[float]) -> float:
        return sum(0.5 * (values[i] + values[i + 1]) * (knots[i + 1] - knots[i])
                   for i in range(len(knots) - 1))

    nominal_y = [at(nominal, x) for x in knots]
    half_y = [at(half, x) for x in knots]
    nominal_peak, half_peak = selected_peaks
    work_nominal, work_half = integrate(nominal_y), integrate(half_y)
    rms_nominal = math.sqrt(sum(value * value for value in nominal_y) / len(nominal_y))
    rms_delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(nominal_y, half_y)) / len(knots))
    if (nominal_peak <= 0.0 or work_nominal <= 0.0 or rms_nominal <= 0.0 or
            abs(half_peak - nominal_peak) / nominal_peak > 0.02 or
            abs(work_half - work_nominal) / work_nominal > 0.02 or
            rms_delta / rms_nominal > 0.05):
        raise EnsembleError("BCT accepted nominal/half-speed curves fail the producer's peak/work/full-curve sensitivity gates")


def _verify_bct_capacity_evidence(history: dict[str, Any], cert: dict[str, Any],
                                  manifest: dict[str, Any], receipt: dict[str, Any],
                                  run_dir: Path) -> None:
    """Fail closed unless BCT is tied to the full v2 capacity qualification.

    This is deliberately a narrow adapter contract. Generic expected_steps or a
    caller-selected protocol label cannot stand in for the producer's complete
    nominal curve, half-speed check, source/build identity, checkpoint, and
    three-level mesh/time evidence.
    """
    artifact_hashes = cert.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise EnsembleError("BCT full-peak validation is unsupported without the source-bound capacity artifact adapter")
    capacity_path, capacity_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_capacity_qualification")
    curve_path, curve_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_compression_nominal_curve")
    mesh_path, mesh_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_capacity_mesh_report")
    time_path, time_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_capacity_time_report")
    branch_path, branch_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_compression_branch_history")
    checkpoint_path, checkpoint_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_assembled_matter_checkpoint")
    rigid_path, rigid_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "shipping_assembled_rigid_sidecar")
    result_path, _ = _bct_artifact_path(run_dir, artifact_hashes, "native_result")
    _compiled_package_path, compiled_package_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "compiled_matter_package")
    _native_executable_path, native_executable_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "native_executable")
    _solver_source_path, solver_source_sha = _bct_artifact_path(
        run_dir, artifact_hashes, "solver_source")

    values = _capacity_properties(capacity_path)
    if values["schema"] != BCT_CAPACITY_SCHEMA or values["status"] != "qualified_for_numerical_capacity_only":
        raise EnsembleError("BCT full-peak validation requires a qualified shipping-capacity-qualification.v2 artifact")
    for key in ("source_binding_sha256", "assembled_matter_checkpoint_sha256",
                "assembled_rigid_sidecar_sha256", "nominal_curve_sha256",
                "mesh_report_sha256", "time_report_sha256"):
        if HEX_SHA256.fullmatch(values[key]) is None:
            raise EnsembleError(f"BCT capacity field {key} is not a lowercase SHA-256")
    if values["nominal_curve_sha256"] != curve_sha or values["mesh_report_sha256"] != mesh_sha or values["time_report_sha256"] != time_sha:
        raise EnsembleError("BCT capacity artifact does not bind the supplied nominal curve and mesh/time reports")
    if values["assembled_matter_checkpoint_sha256"] != checkpoint_sha or values["assembled_rigid_sidecar_sha256"] != rigid_sha:
        raise EnsembleError("BCT capacity artifact does not bind the supplied assembled Matter and rigid checkpoints")
    if values["mesh_report_file"] != mesh_path.name or values["time_report_file"] != time_path.name:
        raise EnsembleError("BCT capacity report filenames do not match the separately hashed report artifacts")
    if Path(values["mesh_report_file"]).name != values["mesh_report_file"] or Path(values["time_report_file"]).name != values["time_report_file"]:
        raise EnsembleError("BCT capacity report path must be a basename within the run directory")

    # The binding object is repeated in the result/certificate/history/manifest
    # so the v2 record cannot be attached to another run by copying its file.
    binding = cert.get("shipping_capacity_binding")
    if not isinstance(binding, dict) or any(
            other.get("shipping_capacity_binding") != binding
            for other in (history, manifest, receipt)):
        raise EnsembleError("BCT capacity binding must be identical in history, certificate, manifest, and receipt")
    expected_binding_fields = {
        "schema", "capacity_qualification_sha256", "board_recipe_id",
        "source_physics_fingerprint", "device_program_fingerprint",
        "compiled_world_fingerprint", "source_binding_sha256",
        "assembled_matter_checkpoint_sha256", "assembled_rigid_sidecar_sha256",
        "nominal_curve_sha256", "nominal_peak_sample_index",
        "nominal_accepted_sample_count", "nominal_terminal_stroke_fraction",
        "nominal_full_stroke_completed", "nominal_rate_sensitivity_passed",
        "inside_height_m", "mesh_report_sha256", "time_report_sha256",
        "compression_branch_history_sha256",
    }
    if set(binding) != expected_binding_fields or binding.get("schema") != "numi.cardboard.native-bct-capacity-binding.v1":
        raise EnsembleError("BCT capacity binding object does not match the strict v1 adapter schema")
    scalar_pairs = {
        "capacity_qualification_sha256": capacity_sha,
        "board_recipe_id": values["board_recipe_id"],
        "source_binding_sha256": values["source_binding_sha256"],
        "assembled_matter_checkpoint_sha256": values["assembled_matter_checkpoint_sha256"],
        "assembled_rigid_sidecar_sha256": values["assembled_rigid_sidecar_sha256"],
        "nominal_curve_sha256": values["nominal_curve_sha256"],
        "mesh_report_sha256": values["mesh_report_sha256"],
        "time_report_sha256": values["time_report_sha256"],
        "compression_branch_history_sha256": branch_sha,
    }
    for key, expected_value in scalar_pairs.items():
        if binding.get(key) != expected_value:
            raise EnsembleError(f"BCT capacity run binding mismatches {key}")
    for key in ("source_physics_fingerprint", "device_program_fingerprint", "compiled_world_fingerprint"):
        value = _capacity_int(values, key)
        if value <= 0 or binding.get(key) != value or manifest.get(key) != value:
            raise EnsembleError(f"BCT capacity source/build fingerprint mismatch: {key}")
    if _capacity_int(values, "device_program_fingerprint") != _capacity_int(values, "compiled_world_fingerprint"):
        raise EnsembleError("BCT source capacity artifact device and compiled-world fingerprints differ")
    if manifest.get("preset") != values["board_recipe_id"]:
        raise EnsembleError("BCT capacity board recipe does not match the executed native preset")
    for key, digest in (("compiled_matter_package_sha256", compiled_package_sha),
                        ("native_executable_sha256", native_executable_sha),
                        ("solver_source_sha256", solver_source_sha)):
        if manifest.get(key) != digest:
            raise EnsembleError(f"BCT native manifest does not bind its source/build artifact: {key}")
    if manifest.get("shipping_source_material_binding_sha256") != values["source_binding_sha256"]:
        raise EnsembleError("BCT capacity source-material binding does not match the executed manifest")
    if manifest.get("assembled_checkpoint_sha256_for_capacity_binding") != checkpoint_sha or manifest.get("assembled_rigid_sidecar_sha256_for_capacity_binding") != rigid_sha:
        raise EnsembleError("BCT capacity checkpoint identity does not match the executed manifest")

    integer_pairs = {
        "nominal_peak_sample_index": "nominal_peak_sample_index",
        "nominal_accepted_sample_count": "nominal_accepted_sample_count",
    }
    for key, property_key in integer_pairs.items():
        if not isinstance(binding.get(key), int) or isinstance(binding.get(key), bool) or binding[key] != _capacity_int(values, property_key):
            raise EnsembleError(f"BCT capacity binding mismatch: {key}")
    terminal = _capacity_float(values, "nominal_terminal_stroke_fraction")
    inside_height = _finite(binding.get("inside_height_m"), "BCT bound inside height")
    if inside_height <= 0.0 or not math.isclose(terminal, 0.70, rel_tol=0.0, abs_tol=1e-9):
        raise EnsembleError("BCT capacity nominal curve does not bind the required 70-percent inside-height endpoint")
    if (binding.get("nominal_full_stroke_completed") is not True or
            binding.get("nominal_rate_sensitivity_passed") is not True or
            _capacity_bool(values, "nominal_full_stroke_completed") is not True or
            _capacity_bool(values, "nominal_rate_sensitivity_passed") is not True):
        raise EnsembleError("BCT capacity artifact lacks full 70-percent stroke or passed half-speed sensitivity")
    if not math.isclose(_finite(binding.get("nominal_terminal_stroke_fraction"), "BCT binding stroke fraction"), terminal, rel_tol=0.0, abs_tol=1e-12):
        raise EnsembleError("BCT capacity terminal stroke fraction differs from the run binding")

    # Independently inspect the nominal curve and its sample-indexed native
    # history. A peak-only record or a correct expected_steps count is not a
    # complete loading trace.
    if curve_path.stat().st_size > 64 * 1024 * 1024:
        raise EnsembleError("BCT nominal compression curve exceeds the bounded 64 MiB artifact size")
    try:
        with curve_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["sample_index", "stroke_m", "force_N", "contact_samples"]:
                raise EnsembleError("BCT nominal curve has an unsupported column schema")
            curve_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EnsembleError(f"cannot parse BCT nominal curve: {exc}") from exc
    if len(curve_rows) != _capacity_int(values, "nominal_accepted_sample_count") or len(curve_rows) < 3:
        raise EnsembleError("BCT nominal curve sample count does not match the capacity artifact")
    forces: list[float] = []
    strokes: list[float] = []
    contacts_by_row: list[float] = []
    for index, row in enumerate(curve_rows):
        if row.get("sample_index") != str(index):
            raise EnsembleError("BCT nominal curve sample indices are not contiguous from zero")
        stroke = _finite(row.get("stroke_m"), "BCT nominal stroke")
        force = _finite(row.get("force_N"), "BCT nominal force")
        if stroke < 0.0 or force < 0.0 or re.fullmatch(r"0|[1-9][0-9]*", row.get("contact_samples", "")) is None:
            raise EnsembleError("BCT nominal curve contains invalid stroke, force, or contact sample count")
        if strokes and stroke <= strokes[-1]:
            raise EnsembleError("BCT nominal curve stroke must increase strictly")
        contacts = _finite(row.get("contact_samples"), "BCT nominal contact samples")
        if contacts < 0.0:
            raise EnsembleError("BCT nominal curve contains a negative contact sample count")
        strokes.append(stroke); forces.append(force); contacts_by_row.append(contacts)
    if not math.isclose(strokes[-1] / inside_height, 0.70, rel_tol=0.0, abs_tol=1e-9):
        raise EnsembleError("BCT nominal curve artifact does not end at 70 percent inside-height stroke")
    peak_index = first_local_peak_index(strokes, forces, contacts_by_row)
    if peak_index is None:
        raise EnsembleError("BCT nominal curve has no qualified first local compression peak")
    if peak_index != _capacity_int(values, "nominal_peak_sample_index"):
        raise EnsembleError("BCT capacity peak sample index is not the first qualified local peak in the bound nominal curve")
    samples = history.get("samples")
    if not isinstance(samples, list) or len(samples) != len(curve_rows):
        raise EnsembleError("BCT native history must contain the complete capacity curve sample-for-sample")
    for index, (sample, stroke, force) in enumerate(zip(samples, strokes, forces)):
        if (not isinstance(sample, dict) or sample.get("accepted") is not True or
                sample.get("phase") != "shipping_compression_nominal" or
                sample.get("sample_index") != index or
                _finite(sample.get("stroke_m"), "BCT history stroke") != stroke or
                _finite(sample.get("force_N"), "BCT history force") != force or
                _finite(sample.get("contact_samples"), "BCT history contact samples") != contacts_by_row[index]):
            raise EnsembleError("BCT native history differs from the accepted source-bound nominal compression curve")
    _verify_bct_rate_history(branch_path, inside_height, list(zip(strokes, forces)))

    # Verify the refinement reports themselves, not only booleans copied into
    # the properties file. The v2 producer intentionally says this is a
    # numerical-capacity gate, never a physical or material-grade qualification.
    result = _read_json(result_path, "BCT native result")
    for key in ("shipping_capacity_qualification_file_sha256",
                "nominal_compression_curve_sha256",
                "shipping_source_material_binding_sha256",
                "assembled_checkpoint_sha256_for_capacity_binding",
                "assembled_rigid_sidecar_sha256_for_capacity_binding"):
        expected = {
            "shipping_capacity_qualification_file_sha256": capacity_sha,
            "nominal_compression_curve_sha256": curve_sha,
            "shipping_source_material_binding_sha256": values["source_binding_sha256"],
            "assembled_checkpoint_sha256_for_capacity_binding": checkpoint_sha,
            "assembled_rigid_sidecar_sha256_for_capacity_binding": rigid_sha,
        }[key]
        if result.get(key) != expected:
            raise EnsembleError(f"BCT native result does not bind {key}")
    for key in ("source_physics_fingerprint", "device_program_fingerprint",
                "compiled_world_fingerprint"):
        if result.get(key) != _capacity_int(values, key):
            raise EnsembleError(f"BCT native result does not bind {key}")
    if result.get("shipping_capacity_binding") != binding:
        raise EnsembleError("BCT native result does not retain the capacity binding object")
    if (result.get("native_slow_compression_completed") is not True or
            result.get("native_half_speed_compression_completed") is not True or
            result.get("native_compression_rate_sensitivity_gate_passed") is not True or
            result.get("native_compression_numerical_capacity_qualification_passed") is not True or
            not math.isclose(_finite(result.get("compression_stroke_fraction_of_inside_height"), "BCT result stroke fraction"), 0.70, rel_tol=0.0, abs_tol=1e-9)):
        raise EnsembleError("BCT native result does not show completed full-stroke nominal and half-speed qualification")
    if (values["mesh_status"] != "qualified" or values["time_status"] != "qualified" or
            _capacity_int(values, "mesh_level_count") < 3 or _capacity_int(values, "time_level_count") < 3 or
            _capacity_bool(values, "mesh_all_cases_complete") is not True or
            _capacity_bool(values, "time_all_cases_complete") is not True or
            _capacity_bool(values, "mesh_finest_pair_within_tolerance") is not True or
            _capacity_bool(values, "time_finest_pair_within_tolerance") is not True):
        raise EnsembleError("BCT capacity qualification lacks completed passing three-level mesh and time studies")
    tolerance_by_prefix = {"mesh": (0.02, 0.05, 0.02), "time": (0.02, 0.05, 0.02)}
    for prefix, report_path, report_sha, axis in (
            ("mesh", mesh_path, mesh_sha, "mesh_nx"),
            ("time", time_path, time_sha, "time_step")):
        if report_sha != values[f"{prefix}_report_sha256"]:
            raise EnsembleError(f"BCT {prefix} report hash mismatch")
        report = _read_json(report_path, f"BCT {prefix} refinement report")
        if (report.get("schema") != BCT_STAGE_REPORT_SCHEMA or report.get("axis") != axis or
                report.get("status") != "descriptive_three_level_decrease_only" or
                report.get("convergence_claim") is not False):
            raise EnsembleError(f"BCT {prefix} report does not contain the required three-level descriptive study")
        levels = report.get("axis_levels")
        if not isinstance(levels, list) or len(levels) != 3:
            raise EnsembleError(f"BCT {prefix} report does not retain all three refinement levels")
        try:
            numbers = [float(item[0] if isinstance(item, list) and len(item) == 1 else item) for item in levels]
        except (TypeError, ValueError, IndexError) as exc:
            raise EnsembleError(f"BCT {prefix} report has malformed refinement levels") from exc
        if not all(math.isfinite(value) for value in numbers) or not (
                numbers[0] < numbers[1] < numbers[2] if prefix == "mesh" else
                numbers[0] > numbers[1] > numbers[2]):
            raise EnsembleError(f"BCT {prefix} levels are not strictly coarse-to-fine")
        fields = ("relative_peak_force_difference", "normalized_force_curve_rms",
                  "relative_work_difference")
        coarse_values = [_report_metric(report, 0, field) for field in fields]
        fine_values = [_report_metric(report, 1, field) for field in fields]
        if not all(fine < coarse for coarse, fine in zip(coarse_values, fine_values)):
            raise EnsembleError(f"BCT {prefix} adjacent refinement differences do not all decrease")
        limits = tolerance_by_prefix[prefix]
        if not all(value <= limit for value, limit in zip(fine_values, limits)):
            raise EnsembleError(f"BCT {prefix} finest pair exceeds peak/curve/work tolerances")
        for key, value in zip((f"{prefix}_coarse_middle_peak_difference",
                               f"{prefix}_coarse_middle_curve_rms",
                               f"{prefix}_coarse_middle_work_difference"), coarse_values):
            if not math.isclose(_capacity_float(values, key), value, rel_tol=0.0, abs_tol=1e-12):
                raise EnsembleError(f"BCT {prefix} coarse-middle property disagrees with its report")
        for key, value in zip((f"{prefix}_finest_peak_difference",
                               f"{prefix}_finest_curve_rms",
                               f"{prefix}_finest_work_difference"), fine_values):
            if not math.isclose(_capacity_float(values, key), value, rel_tol=0.0, abs_tol=1e-12):
                raise EnsembleError(f"BCT {prefix} finest-pair property disagrees with its report")


def _verify_certificate(record: dict[str, Any], base_dir: Path, expected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    history_path = (base_dir / str(record.get("history_path", ""))).resolve()
    certificate_path = (base_dir / str(record.get("certificate_path", ""))).resolve()
    if base_dir.resolve() not in history_path.parents or base_dir.resolve() not in certificate_path.parents:
        raise EnsembleError("native history/certificate path escapes the ensemble input directory")
    if not history_path.is_file() or not certificate_path.is_file():
        raise EnsembleError("native history or run certificate is missing")
    history_hash = sha256(history_path)
    certificate_hash = sha256(certificate_path)
    if history_hash != record.get("history_sha256"):
        raise EnsembleError(f"{history_path.name}: history SHA-256 mismatch")
    if certificate_hash != record.get("certificate_file_sha256"):
        raise EnsembleError(f"{certificate_path.name}: certificate file SHA-256 mismatch")
    history = _read_json(history_path, "native forward history")
    cert = _read_json(certificate_path, "native execution certificate")
    if history.get("schema") != "numi.cardboard.native_forward_history.v1":
        raise EnsembleError(f"{history_path.name}: unsupported native history schema")
    if cert.get("schema") != "numi.cardboard.native_execution_certificate.v1":
        raise EnsembleError(f"{certificate_path.name}: unsupported certificate schema")
    cert_body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    if cert.get("certificate_sha256") != canonical_sha256(cert_body):
        raise EnsembleError(f"{certificate_path.name}: certificate digest mismatch")
    required_match = ("candidate_id", "candidate_sha256", "condition_id", "specimen_id",
                      "native_protocol_id")
    for key in required_match:
        if history.get(key) != expected.get(key) or cert.get(key) != expected.get(key) or record.get(key) != expected.get(key):
            raise EnsembleError(f"{certificate_path.name}: {key} does not match ensemble case identity")
    if cert.get("history_sha256") != history_hash:
        raise EnsembleError(f"{certificate_path.name}: certificate does not bind the native history bytes")
    if cert.get("source_data_catalog_sha256") != expected["source_data_catalog_sha256"]:
        raise EnsembleError(f"{certificate_path.name}: missing/mismatched source catalog binding")
    if cert.get("source_dataset_sha256") != expected["source_dataset_sha256"]:
        raise EnsembleError(f"{certificate_path.name}: missing/mismatched source dataset binding")
    if cert.get("source_asset_sha256") != expected["source_asset_sha256"]:
        raise EnsembleError(f"{certificate_path.name}: missing/mismatched publisher asset/text binding")
    for key in ("source_data_catalog_sha256", "source_dataset_sha256", "source_asset_sha256"):
        if history.get(key) != expected[key]:
            raise EnsembleError(f"{history_path.name}: missing/mismatched {key} binding")
    if cert.get("execution_class") != "native_forward_prediction" or cert.get("backend") != "numi_matter_native_fem":
        raise EnsembleError(f"{certificate_path.name}: history is not certified as native forward FEM")
    if cert.get("terminal_status") != "completed" or cert.get("protocol_complete") is not True or cert.get("prefix_only") is not False:
        raise EnsembleError(f"{certificate_path.name}: completed prefixes and failed/incomplete histories cannot be adjudicated")
    expected_steps = cert.get("expected_steps")
    accepted_steps = cert.get("accepted_steps")
    if not isinstance(expected_steps, int) or expected_steps <= 0 or accepted_steps != expected_steps:
        raise EnsembleError(f"{certificate_path.name}: accepted step count does not cover the full protocol")
    if not isinstance(cert.get("physical_fem_steps"), int) or cert["physical_fem_steps"] <= 0:
        raise EnsembleError(f"{certificate_path.name}: zero-step or non-FEM fixture is not a forward prediction")
    samples = history.get("samples")
    if not isinstance(samples, list) or len(samples) < 2 or cert.get("sample_count") != len(samples):
        raise EnsembleError(f"{certificate_path.name}: sample count does not bind a complete history")
    artifact_hashes = cert.get("artifact_hashes")
    required_artifacts = {"native_input_manifest", "native_executable", "solver_source", "run_receipt"}
    if not isinstance(artifact_hashes, dict) or not required_artifacts.issubset(artifact_hashes):
        raise EnsembleError(f"{certificate_path.name}: required native source/build/receipt hashes are absent")
    for name in sorted(required_artifacts):
        artifact = artifact_hashes[name]
        if not isinstance(artifact, dict) or len(str(artifact.get("sha256", ""))) != 64 or not artifact.get("path"):
            raise EnsembleError(f"{certificate_path.name}: {name} artifact hash/path missing")
        path = (certificate_path.parent / artifact["path"]).resolve()
        if certificate_path.parent.resolve() not in path.parents or not path.is_file() or sha256(path) != artifact["sha256"]:
            raise EnsembleError(f"{certificate_path.name}: {name} artifact file missing or hash mismatch")
    manifest = _read_json(certificate_path.parent / artifact_hashes["native_input_manifest"]["path"], "native input manifest")
    for key in ("candidate_id", "candidate_sha256", "condition_id", "specimen_id",
                "native_protocol_id",
                "source_data_catalog_sha256", "source_dataset_sha256", "source_asset_sha256"):
        if manifest.get(key) != expected.get(key):
            raise EnsembleError(f"{certificate_path.name}: native input manifest {key} binding mismatch")
    if canonical_sha256(manifest.get("parameters", {})) != canonical_sha256(expected.get("parameters", {})):
        raise EnsembleError(f"{certificate_path.name}: executed parameter values differ from the candidate config")
    receipt = _read_json(certificate_path.parent / artifact_hashes["run_receipt"]["path"], "native run receipt")
    for key in ("candidate_id", "candidate_sha256", "condition_id", "specimen_id",
                "native_protocol_id",
                "terminal_status", "accepted_steps", "expected_steps", "physical_fem_steps"):
        if receipt.get(key) != cert.get(key):
            raise EnsembleError(f"{certificate_path.name}: native run receipt {key} mismatch")
    if receipt.get("backend") != "numi_matter_native_fem":
        raise EnsembleError(f"{certificate_path.name}: CPU/oracle receipt is not a native FEM execution")
    if expected.get("dataset_id") == "mrowczynski_2021_box_bct":
        try:
            _verify_bct_capacity_evidence(history, cert, manifest, receipt,
                                          certificate_path.parent)
        except EnsembleError as exc:
            raise EnsembleError(
                f"{certificate_path.name}: BCT full-peak validation is unsupported without a complete source-bound capacity artifact: {exc}") from exc
    return history, cert


def _interval_violation(value: float, low: float, high: float, scale: float) -> float:
    if low <= value <= high:
        return 0.0
    width = max(high - low, abs(scale) * 1e-12, 1e-12)
    return (low - value) / width if value < low else (value - high) / width


def adjudicate(plan_path: Path, output_path: Path | None = None,
               catalog_path: Path = DEFAULT_CATALOG, data_dir: Path = DEFAULT_DATA) -> dict[str, Any]:
    plan = _read_json(plan_path, "calibration ensemble plan")
    if plan.get("schema") != "numi.cardboard.calibration_ensemble.v1":
        raise EnsembleError("unsupported ensemble plan schema")
    catalog, catalog_hash, datasets = load_source_package(catalog_path, data_dir)
    plan["_sources"] = catalog.get("sources", [])
    if plan.get("source_data_catalog_sha256") != catalog_hash:
        raise EnsembleError("plan is not bound to the current source-data catalog hash")
    conditions, blockers = _load_conditions(plan, datasets, catalog_hash)
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise EnsembleError("plan requires explicit parameter candidates or limiting controls")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        _validate_candidate(candidate)
        cid = candidate["candidate_id"]
        if cid in candidate_by_id:
            raise EnsembleError(f"duplicate candidate_id {cid}")
        candidate_by_id[cid] = candidate
    calibration_condition_ids = [cid for cid, d in conditions.items() if d["condition"]["role"] == "calibration"]
    for condition_id in calibration_condition_ids:
        condition_data = conditions[condition_id]
        dataset = condition_data["dataset"]
        allowed = dataset.get("calibration_parameter_families")
        if allowed is None and dataset.get("id") == "haj_ali_2009_yield_anchors":
            allowed = ["paper_yield_anchor"]
        for candidate in candidates:
            if candidate.get("role") != "fit_candidate":
                continue
            families = {p.get("family") for p in candidate.get("parameters", {}).values()
                        if isinstance(p, dict) and p.get("fit") is True}
            if not families.issubset(set(allowed or [])):
                blockers.append(f"{condition_id}: candidate {candidate['candidate_id']} contains parameter families not admitted by dataset {dataset['id']}")
    scope_blocker_phrases = ("cannot be calibrated", "does not admit calibration",
                             "contains parameter families not admitted")
    calibration_scope_admitted = bool(calibration_condition_ids) and not any(
        blocker.startswith(tuple(f"{cid}:" for cid in calibration_condition_ids)) and
        any(phrase in blocker for phrase in scope_blocker_phrases)
        for blocker in blockers)
    histories = plan.get("native_histories")
    if not isinstance(histories, list) or not histories:
        raise EnsembleError("plan requires source-bound native forward histories")
    run_table: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    control_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for record in histories:
        candidate_id = str(record.get("candidate_id", ""))
        condition_id = str(record.get("condition_id", ""))
        if candidate_id not in candidate_by_id or condition_id not in conditions:
            raise EnsembleError("native history references unknown candidate or condition")
        key = (candidate_id, condition_id)
        if key in run_table:
            raise EnsembleError(f"duplicate native history for {key}")
        condition_data = conditions[condition_id]
        condition = condition_data["condition"]
        expected = {"candidate_id": candidate_id, "condition_id": condition_id,
                    "candidate_sha256": candidate_by_id[candidate_id]["candidate_sha256"],
                    "specimen_id": str(condition.get("specimen_id", "")),
                    "native_protocol_id": condition.get("native_protocol_id"),
                    "source_data_catalog_sha256": catalog_hash,
                    "source_dataset_sha256": condition_data["dataset"]["sha256"],
                    "source_asset_sha256": condition_data["source_asset_sha256"],
                    "dataset_id": condition_data["dataset"]["id"],
                    "target_recipe_id": plan.get("target_recipe_id"),
                    "parameters": candidate_by_id[candidate_id].get("parameters", {})}
        if not expected["specimen_id"]:
            raise EnsembleError(f"{condition_id}: specimen_id is required to prevent condition conflation")
        history, cert = _verify_certificate(record, plan_path.parent.resolve(), expected)
        run_table[key] = (history, cert)

    # Every enumerated case must have a complete whole-condition forward run.
    required_pairs = {(candidate_id, condition_id)
                      for candidate_id in candidate_by_id for condition_id in conditions}
    if set(run_table) != required_pairs:
        missing = sorted(required_pairs - set(run_table))
        extra = sorted(set(run_table) - required_pairs)
        raise EnsembleError(f"native histories do not cover the candidate x whole-condition matrix; missing={missing}, extra={extra}")

    metric_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for condition_id, data in conditions.items():
        condition = data["condition"]
        transfer = condition.get("transfer") or {}
        if condition["role"] in {"calibration", "validation"}:
            if transfer.get("kind") == "bounded_analogue" and not str(transfer.get("rationale", "")).strip():
                blockers.append(f"{condition_id}: bounded analogue requires a written transfer rationale")
            if transfer.get("kind") not in {"same_grade", "bounded_analogue"}:
                blockers.append(f"{condition_id}: explicit transfer class is required")
        for candidate_id, candidate in candidate_by_id.items():
            history, _ = run_table[(candidate_id, condition_id)]
            for metric_name, spec in condition["metrics"].items():
                observed = data["observations"].get(metric_name)
                if not observed:
                    continue
                predicted = _derive_native_metric(history, spec, metric_name)
                if UNIT_DIMENSION.get(predicted["unit"]) != UNIT_DIMENSION.get(observed["unit"]):
                    raise EnsembleError(f"{condition_id}/{metric_name}: incompatible source/prediction units {observed['unit']} and {predicted['unit']}")
                # Units must be identical until an explicit conversion is added;
                # same dimension alone must not silently alter a result.
                prediction_unit = spec.get("prediction_unit", observed["unit"])
                if prediction_unit != observed["unit"]:
                    raise EnsembleError(f"{condition_id}/{metric_name}: unit conversion is not implicit ({observed['unit']} vs {prediction_unit})")
                if observed["kind"] == "category":
                    allowed_categories = _transfer_categories(observed, transfer, metric_name)
                    feasible = predicted["value"] in allowed_categories
                    violation = 0.0 if feasible else 1.0
                else:
                    low, high = _transfer_interval(observed, transfer, metric_name)
                    value = predicted["value"]
                    feasible = low <= value <= high
                    violation = _interval_violation(value, low, high, max(abs(value), abs(high), 1.0))
                row = {"candidate_id": candidate_id, "condition_id": condition_id,
                       "role": condition["role"], "metric": metric_name,
                       "admissible_for_fit": condition["role"] != "calibration" or calibration_scope_admitted,
                       "observed": observed, "transfer_target": {"allowed_categories": sorted(allowed_categories)} if observed["kind"] == "category" else {"lower": low, "upper": high},
                       "predicted": predicted, "feasible": feasible, "normalized_interval_violation": violation}
                metric_rows[(candidate_id, condition_id, metric_name)] = row
                if candidate["role"] != "fit_candidate":
                    control_rows.append(row)
                elif condition["role"] == "validation":
                    validation_rows.append(row)
                elif condition["role"] == "calibration":
                    calibration_rows.append(row)

    validation_condition_ids = [cid for cid, d in conditions.items() if d["condition"]["role"] == "validation"]
    feasible_candidates: list[str] = []
    candidate_status: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        if candidate["role"] != "fit_candidate":
            continue
        rows = [r for r in calibration_rows if r["candidate_id"] == candidate_id]
        required_metric_count = sum(len(conditions[cid]["condition"]["metrics"]) for cid in calibration_condition_ids)
        complete = calibration_scope_admitted and len(rows) == required_metric_count and not any(
            c.startswith(tuple(f"{cid}:" for cid in calibration_condition_ids)) for c in blockers)
        accepted = complete and all(r["feasible"] for r in rows)
        if accepted:
            feasible_candidates.append(candidate_id)
        candidate_status.append({"candidate_id": candidate_id,
                                 "status": "feasible_against_all_calibration_fixtures" if accepted else
                                          "not_fit_no_admitted_calibration_fixture" if not calibration_scope_admitted else
                                          "infeasible_or_incomplete",
                                 "calibration_metric_count": len(rows),
                                 "all_metrics_feasible": bool(rows) and all(r["feasible"] for r in rows)})

    # Paired one-factor contrasts provide descriptive sensitivity ranges only.
    # Calibration contrasts remain the only sensitivity rows associated with
    # retained feasible sets. Held-out contrasts are reported separately and
    # never participate in candidate feasibility or parameter envelopes.
    sensitivity: list[dict[str, Any]] = []
    predictive_sensitivity: list[dict[str, Any]] = []
    fit_candidates = [c for c in candidates if c["role"] == "fit_candidate"]
    parameter_names = sorted({name for c in fit_candidates for name, value in c.get("parameters", {}).items()
                              if value.get("fit") is True})
    for parameter in parameter_names:
        for role, condition_ids in (
                ("calibration", calibration_condition_ids if calibration_scope_admitted else []),
                ("validation", [cid for cid, data in conditions.items()
                                if data["condition"]["role"] == "validation"]),
                ("holdout", [cid for cid, data in conditions.items()
                              if data["condition"]["role"] == "holdout"])):
            if not condition_ids:
                continue
            contrasts: list[dict[str, Any]] = []
            for a, b in itertools.combinations(fit_candidates, 2):
                pa, pb = a["parameters"], b["parameters"]
                if parameter not in pa or parameter not in pb:
                    continue
                if pa[parameter].get("fit") is not True or pb[parameter].get("fit") is not True:
                    continue
                if {k: v for k, v in pa.items() if k != parameter} != {k: v for k, v in pb.items() if k != parameter}:
                    continue
                va, vb = _finite(pa[parameter]["value"], parameter), _finite(pb[parameter]["value"], parameter)
                if va == vb:
                    continue
                for condition_id in condition_ids:
                    for metric_name in conditions[condition_id]["condition"]["metrics"]:
                        ra = metric_rows.get((a["candidate_id"], condition_id, metric_name))
                        rb = metric_rows.get((b["candidate_id"], condition_id, metric_name))
                        if (not ra or not rb or ra["predicted"]["unit"] != rb["predicted"]["unit"] or
                                not isinstance(ra["predicted"].get("value"), (int, float)) or
                                not isinstance(rb["predicted"].get("value"), (int, float))):
                            continue
                        delta = abs(float(rb["predicted"]["value"]) - float(ra["predicted"]["value"]))
                        target = ra["transfer_target"]
                        target_width = max(float(target["upper"]) - float(target["lower"]), 1e-12)
                        contrasts.append({"condition_id": condition_id,
                                         "condition_role": role,
                                         "metric": metric_name,
                                         "parameter_delta": abs(vb - va),
                                         "prediction_delta": delta,
                                         "normalized_prediction_delta": delta / target_width,
                                         "unit": ra["predicted"]["unit"],
                                         "normalization": "transfer-expanded target interval width"})
            sensitivity_row = {"parameter": parameter,
                               "condition_role": role,
                               "paired_contrast_count": len(contrasts),
                               "prediction_delta_range": {
                                   "minimum": min((c["prediction_delta"] for c in contrasts), default=None),
                                   "maximum": max((c["prediction_delta"] for c in contrasts), default=None)},
                               "normalized_prediction_delta_range": {
                                   "minimum": min((c["normalized_prediction_delta"] for c in contrasts), default=None),
                                   "maximum": max((c["normalized_prediction_delta"] for c in contrasts), default=None)},
                               "contrasts": contrasts,
                               "interpretation": "descriptive range over supplied paired candidates; not a confidence interval or parameter identification"}
            if role == "calibration":
                sensitivity.append(sensitivity_row)
            else:
                predictive_sensitivity.append(sensitivity_row)
    sensitivity.sort(key=lambda row: (row["normalized_prediction_delta_range"]["maximum"]
                                      if row["normalized_prediction_delta_range"]["maximum"] is not None else -1),
                     reverse=True)
    predictive_sensitivity.sort(key=lambda row: (
        row["condition_role"], row["parameter"],
        row["normalized_prediction_delta_range"]["maximum"]
        if row["normalized_prediction_delta_range"]["maximum"] is not None else -1))

    full_metrics_required = {name for data in conditions.values() for name in data["condition"].get("required_metrics", [])}
    metrics_present = {row["metric"] for row in metric_rows.values()}
    missing_metrics = sorted(full_metrics_required - metrics_present)
    if missing_metrics:
        blockers.extend(f"required fixture observable not available from supplied source/native histories: {name}" for name in missing_metrics)

    calibration_admitted = calibration_scope_admitted
    result = {
        "schema": "numi.cardboard.calibration_ensemble_result.v1",
        "source_data_catalog_sha256": catalog_hash,
        "plan_sha256": sha256(plan_path),
        "adjudication_method": "whole fixture interval feasibility over explicit source/digitization/transfer bounds; no pointwise train-test split",
        "calibration_status": "feasible_parameter_sets_retained" if calibration_admitted and feasible_candidates else
                              "no_feasible_parameter_set" if calibration_admitted else
                              "no_admitted_calibration_fixture",
        "feasible_candidate_ids": feasible_candidates,
        "feasible_parameter_sets": [
            {"candidate_id": cid, "parameters": candidate_by_id[cid]["parameters"]}
            for cid in feasible_candidates],
        "feasible_parameter_envelopes": {
            name: {"minimum": min(_finite(candidate_by_id[cid]["parameters"][name]["value"], name) for cid in feasible_candidates),
                  "maximum": max(_finite(candidate_by_id[cid]["parameters"][name]["value"], name) for cid in feasible_candidates),
                  "feasible_values": sorted({_finite(candidate_by_id[cid]["parameters"][name]["value"], name) for cid in feasible_candidates}),
                  "interpretation": "range across the supplied feasible ensemble; not a confidence interval"}
            for name in sorted({name for cid in feasible_candidates for name in candidate_by_id[cid]["parameters"]})
            if all(name in candidate_by_id[cid]["parameters"] for cid in feasible_candidates)},
        "parameter_identifiability": {
            name: {
                "status": "not_adjudicated_no_admitted_calibration_fixture" if not calibration_admitted else
                         "no_feasible_parameter_set_to_assess" if not feasible_candidates else
                         "nonunique_in_enumerated_feasible_set" if len({
                             _finite(candidate_by_id[cid]["parameters"][name]["value"], name)
                             for cid in feasible_candidates
                             if name in candidate_by_id[cid].get("parameters", {})}) > 1 else
                         "single_value_among_enumerated_candidates_only",
                "feasible_values": sorted({
                    _finite(candidate_by_id[cid]["parameters"][name]["value"], name)
                    for cid in feasible_candidates
                    if name in candidate_by_id[cid].get("parameters", {})}),
                "interpretation": "finite candidate grids cannot establish global parameter identification; a single surviving grid value is not proof of uniqueness",
            }
            for name in sorted({name for candidate in fit_candidates
                                for name, value in candidate.get("parameters", {}).items()
                                if value.get("fit") is True})},
        "candidate_status": candidate_status,
        "metric_comparisons": list(metric_rows.values()),
        "sensitivity_ranges": sensitivity,
        "heldout_predictive_sensitivity_ranges": predictive_sensitivity,
        "validation_condition_ids": validation_condition_ids,
        "validation_metrics": validation_rows,
        "limiting_controls": control_rows,
        "missing_required_metrics": missing_metrics,
        "blockers": sorted(set(blockers)),
        "interpretation_limits": [
            "Discrete candidate feasibility does not establish a unique constitutive calibration.",
            "Feasible parameter sets are not posterior distributions or confidence intervals.",
            "Sensitivity ranges describe only supplied one-factor paired candidates.",
            "Validation/holdout sensitivity is descriptive only and never changes calibration feasibility or feasible parameter sets.",
            "A finite candidate grid can expose non-uniqueness but cannot prove global parameter identification.",
            "Unbonded and perfectly tied controls are reported separately and never enter candidate ranking.",
            "Source analogue transfer is a user-declared bounded scenario, not evidence of grade equivalence.",
            "A native certificate is an integrity/provenance record, not a cryptographic attestation of truth.",
        ],
    }
    if output_path:
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="source-bound whole-fixture ensemble plan JSON")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = adjudicate(args.plan, args.output, args.catalog, args.data_dir)
    except EnsembleError as exc:
        print(f"adjudication rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"calibration_status": result["calibration_status"],
                      "feasible_candidate_ids": result["feasible_candidate_ids"],
                      "validation_metric_count": len(result["validation_metrics"]),
                      "limiting_control_metric_count": len(result["limiting_controls"]),
                      "blocker_count": len(result["blockers"]),
                      "output": str(args.output) if args.output else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
