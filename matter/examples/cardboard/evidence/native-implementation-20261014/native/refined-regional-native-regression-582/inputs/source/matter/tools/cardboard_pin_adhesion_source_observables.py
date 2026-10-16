#!/usr/bin/env python3
"""Extract bounded, source-bound observables from Johnson/Popil pin evidence.

This is a source-data instrument, not a constitutive fit. It reports only what
the retained 2016 presentation digitization can support and keeps the 2015
article as protocol context rather than a second, independent dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cardboard_source_data as source_data


TOOL = Path(__file__).resolve()
REPO = TOOL.parents[2]
DEFAULT_DATA = REPO / "matter/examples/cardboard/source-data"
DEFAULT_CATALOG = DEFAULT_DATA / "catalog.json"
FORCE_DATASET = "johnson_popil_pin_force_displacement"
SUMMARY_DATASET = "johnson_popil_pin_summary"
PROFILE_ID = "johnson_popil_pin_adhesion_engineering_prototype"
PLOT_SOURCE_ID = "johnson_popil_2016_author_presentation"
PROTOCOL_SOURCE_ID = "johnson_popil_2015_article"
WINDOW_THRESHOLD_N = 30.0


class PinObservableError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PinObservableError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise PinObservableError(f"{field} must be finite")
    return result


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise PinObservableError(f"cannot read {path}: {exc}") from exc


def _monotone_linear_extreme(coefficients: list[float], lows: list[float],
                             highs: list[float], maximize: bool) -> float:
    """Extremize c·x over box bounds plus x[0] <= ... <= x[n-1].

    Vertices of this bounded order polytope use one of the supplied interval
    endpoints. Dynamic programming over that finite endpoint set is exact for
    the closed, nondecreasing relaxation. If source intervals overlap, allowing
    equality produces a conservative infimum/supremum for strictly ordered
    plotted abscissae.
    """
    endpoints = sorted(set(lows + highs))
    previous: dict[float, float] = {}
    for i, coefficient in enumerate(coefficients):
        allowed = [v for v in endpoints if lows[i] <= v <= highs[i]]
        if not allowed:
            raise PinObservableError("empty x-axis uncertainty interval")
        current: dict[float, float] = {}
        for value in allowed:
            local = coefficient * value
            if i == 0:
                current[value] = local
                continue
            compatible = [score for prior, score in previous.items() if prior <= value]
            if compatible:
                current[value] = local + (max(compatible) if maximize else min(compatible))
        if not current:
            raise PinObservableError("digitization x-axis uncertainty intervals have no monotone realization")
        previous = current
    return max(previous.values()) if maximize else min(previous.values())


def _trapezoid_area(x: list[float], y: list[float]) -> float:
    return sum(0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
               for i in range(len(x) - 1))


def _area_enclosure(x: list[float], x_unc: list[float], y: list[float],
                    y_unc: list[float], threshold: float) -> dict[str, Any]:
    """Bound sampled-node trapezoid area under the pointwise pixel intervals."""
    if len(x) < 2:
        raise PinObservableError("pin source window needs at least two points")
    if y[0] != threshold or y[-1] != threshold:
        raise PinObservableError("source window endpoints must sit on its declared load threshold")
    y_low = [threshold] + [y[i] - y_unc[i] for i in range(1, len(y) - 1)] + [threshold]
    y_high = [threshold] + [y[i] + y_unc[i] for i in range(1, len(y) - 1)] + [threshold]
    if any(v < 0.0 for v in y_low):
        raise PinObservableError("pointwise load interval crosses zero; nonnegative-area bound unsupported")
    x_low = [x[i] - x_unc[i] for i in range(len(x))]
    x_high = [x[i] + x_unc[i] for i in range(len(x))]

    def extremum(loads: list[float], maximize: bool) -> float:
        coeff = [-0.5 * (loads[0] + loads[1])]
        coeff.extend(0.5 * (loads[i - 1] - loads[i + 1])
                     for i in range(1, len(loads) - 1))
        coeff.append(0.5 * (loads[-2] + loads[-1]))
        return _monotone_linear_extreme(coeff, x_low, x_high, maximize)

    lower = extremum(y_low, maximize=False)
    upper = extremum(y_high, maximize=True)
    estimate = _trapezoid_area(x, y)
    if lower > estimate or estimate > upper:
        raise PinObservableError("internal area interval failed to contain the centerline estimate")
    return {
        "estimate": estimate,
        "interval": [lower, upper],
        "unit": "N*mm",
        "interval_scope": "conservative enclosure from reported per-point x/y digitization bands; curve interpolation remains trapezoidal",
        "interpretation": "visible 30 N threshold-window area only; not the article or slide-16 pin-work quantity",
    }


def _curve(rows: list[dict[str, str]], dataset_id: str, series_id: str,
           threshold: float | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row.get("series_id") == series_id]
    if not selected:
        raise PinObservableError(f"{series_id}: no rows in {dataset_id}")
    if len({row.get("defect_id") for row in selected}) != 1:
        raise PinObservableError(f"{series_id}: mixed fixture-condition labels")
    x = [_number(row.get("displacement_mm"), "displacement_mm") for row in selected]
    y = [_number(row.get("load_N"), "load_N") for row in selected]
    xu = [_number(row.get("displacement_uncertainty_mm"), "displacement_uncertainty_mm") for row in selected]
    yu = [_number(row.get("load_uncertainty_N"), "load_uncertainty_N") for row in selected]
    if len(x) < 2 or any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise PinObservableError(f"{series_id}: source rows are not strictly ordered")
    if any(v < 0.0 for v in xu + yu):
        raise PinObservableError(f"{series_id}: negative digitization uncertainty")
    peak_index = max(range(len(y)), key=y.__getitem__)
    peak = {
        "sampled_centerline_estimate": y[peak_index],
        "digitization_interval": [y[peak_index] - yu[peak_index], y[peak_index] + yu[peak_index]],
        "x_mm_estimate": x[peak_index],
        "x_mm_digitization_interval": [x[peak_index] - xu[peak_index], x[peak_index] + xu[peak_index]],
        "unit": "N",
        "interpretation": "maximum among the digitized visible points, not a raw or replicate-resolved measurement",
    }
    result: dict[str, Any] = {
        "series_id": series_id,
        "defect_id": selected[0].get("defect_id"),
        "visible_point_count": len(selected),
        "visible_displacement_mm": [x[0], x[-1]],
        "visible_sampled_peak": peak,
        "complete_specimen_trace": False,
    }
    if threshold is not None:
        if y[0] != threshold:
            raise PinObservableError(f"{series_id}: first visible point does not equal the declared threshold")
        return_index = next((i for i in range(peak_index + 1, len(y)) if y[i] <= threshold), None)
        if return_index is None or return_index != len(y) - 1 or y[return_index] != threshold:
            raise PinObservableError(f"{series_id}: visible trace does not end at the first post-peak threshold return")
        if peak_index <= 0:
            raise PinObservableError(f"{series_id}: peak does not follow the threshold start")
        start_delta_max = (x[peak_index] + xu[peak_index]) - (x[0] - xu[0])
        start_delta_min = (x[peak_index] - xu[peak_index]) - (x[0] + xu[0])
        peak_load_low = y[peak_index] - yu[peak_index]
        peak_load_high = y[peak_index] + yu[peak_index]
        if start_delta_min <= 0.0:
            raise PinObservableError(f"{series_id}: peak/start x intervals overlap")
        result["visible_window"] = {
            "start_load_N": threshold,
            "end_rule": "first_post_peak_return_to_start_load",
            "start_x_mm_estimate": x[0],
            "end_x_mm_estimate": x[-1],
            "origin_free": True,
            "interval_scope": "digitization only; no replicate scatter or curve-shape uncertainty",
        }
        result["first_to_peak_visible_secant"] = {
            "estimate": (y[peak_index] - threshold) / (x[peak_index] - x[0]),
            "interval": [(peak_load_low - threshold) / start_delta_max,
                         (peak_load_high - threshold) / start_delta_min],
            "unit": "N/mm",
            "interpretation": "visible branch secant descriptor, not the initial fixture stiffness reported on slide 16",
        }
        result["visible_window_trapezoid_area"] = _area_enclosure(x, xu, y, yu, threshold)
        result["visible_threshold_window_complete"] = True
        result["window_end_is_visible_trace_end"] = True
    else:
        if y[0] != WINDOW_THRESHOLD_N:
            raise PinObservableError(f"{series_id}: visible high-response branch does not start at the declared threshold")
        last_delta_min = (x[-1] - xu[-1]) - (x[0] + xu[0])
        last_delta_max = (x[-1] + xu[-1]) - (x[0] - xu[0])
        if last_delta_min <= 0.0:
            raise PinObservableError(f"{series_id}: visible branch endpoint/start x intervals overlap")
        result["first_to_last_visible_branch_secant"] = {
            "estimate": (y[-1] - WINDOW_THRESHOLD_N) / (x[-1] - x[0]),
            "interval": [(y[-1] - yu[-1] - WINDOW_THRESHOLD_N) / last_delta_max,
                         (y[-1] + yu[-1] - WINDOW_THRESHOLD_N) / last_delta_min],
            "unit": "N/mm",
            "interpretation": "censored visible-branch secant; not initial stiffness or a complete curve metric",
        }
        result["visible_branch_terminal_load_lower_bound_N"] = y[-1] - yu[-1]
        result["global_peak"] = {"status": "unbounded_from_visible_trace; later branch is obscured"}
        result["full_curve_work"] = {"status": "unavailable; visible branch is truncated before peak/return"}
    return result


def analyze(catalog_path: Path = DEFAULT_CATALOG,
            data_dir: Path = DEFAULT_DATA) -> dict[str, Any]:
    try:
        catalog = source_data.load_catalog(catalog_path)
    except source_data.PackageError as exc:
        raise PinObservableError(str(exc)) from exc
    errors = source_data.validate_catalog(catalog, data_dir)
    if errors:
        raise PinObservableError("source package failed validation: " + "; ".join(errors))

    catalog_hash = _sha256(catalog_path)
    sources = {row["id"]: row for row in catalog.get("sources", [])}
    datasets = {row["id"]: row for row in catalog.get("datasets", [])}
    profiles = {row["id"]: row for row in catalog.get("fixture_profiles", [])}
    force = datasets.get(FORCE_DATASET)
    summary = datasets.get(SUMMARY_DATASET)
    profile_record = profiles.get(PROFILE_ID)
    if not force or not summary or not profile_record:
        raise PinObservableError("pin datasets/profile are absent from the source catalog")
    for dataset_id, dataset in ((FORCE_DATASET, force), (SUMMARY_DATASET, summary)):
        if dataset.get("source_ids") != [PLOT_SOURCE_ID]:
            raise PinObservableError(f"{dataset_id}: direct digitization source must be only the 2016 presentation")
        if dataset.get("protocol_source_id") != PROTOCOL_SOURCE_ID:
            raise PinObservableError(f"{dataset_id}: 2015 article protocol lineage changed")
        if dataset.get("split") != ("fixture_level_analogue_holdout_only" if dataset_id == FORCE_DATASET else "descriptive_analogue_no_fit"):
            raise PinObservableError(f"{dataset_id}: catalog split no longer matches the pin evidence contract")
    if PLOT_SOURCE_ID not in sources or PROTOCOL_SOURCE_ID not in sources:
        raise PinObservableError("direct plot or protocol source is missing")
    plot_source = sources[PLOT_SOURCE_ID]
    plot_hash = plot_source.get("external_original_sha256")
    protocol_source = sources[PROTOCOL_SOURCE_ID]
    protocol_hash = protocol_source.get("source_text_extract_sha256")
    if not plot_hash or not protocol_hash:
        raise PinObservableError("presentation PDF or article text extract is not hash-bound")

    force_path = (data_dir / force["file"]).resolve()
    summary_path = (data_dir / summary["file"]).resolve()
    profile_path = (data_dir / profile_record["file"]).resolve()
    if _sha256(force_path) != force.get("sha256") or _sha256(summary_path) != summary.get("sha256"):
        raise PinObservableError("pin digitization CSV SHA-256 does not match catalog")
    if _sha256(profile_path) != profile_record.get("sha256"):
        raise PinObservableError("engineering fixture profile SHA-256 does not match catalog")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("publication_protocol", {}).get("source_id") != PROTOCOL_SOURCE_ID or profile.get("digitized_observable", {}).get("direct_plot_source_id") != PLOT_SOURCE_ID:
        raise PinObservableError("engineering profile source lineage mismatch")
    if profile.get("engineering_prototype_geometry", {}).get("dimension_basis") != "engineering_prototype":
        raise PinObservableError("prototype geometry is not marked as engineering-only")

    force_rows = _rows(force_path)
    low = _curve(force_rows, FORCE_DATASET, "low_adhesion_white_glue_lines", WINDOW_THRESHOLD_N)
    high = _curve(force_rows, FORCE_DATASET, "applicator_roll_too_fast")

    summary_rows = _rows(summary_path)
    strength = [_number(row.get("pin_adhesion_strength_N_mm"), "pin_adhesion_strength_N_mm") for row in summary_rows]
    stiffness_rows = [row for row in summary_rows if row.get("pin_stiffness_N_mm", "").strip()]
    stiffness = [_number(row.get("pin_stiffness_N_mm"), "pin_stiffness_N_mm") for row in stiffness_rows]
    stiffness_unc = [_number(row.get("pin_stiffness_uncertainty_N_mm"), "pin_stiffness_uncertainty_N_mm") for row in stiffness_rows]
    work_values = [_number(row.get("pin_adhesion_work_plot_value"), "pin_adhesion_work_plot_value") for row in summary_rows]
    if not strength or not stiffness or not work_values:
        raise PinObservableError("slide-16 marker summary is incomplete")

    protocol = profile["publication_protocol"]
    engineering = profile["engineering_prototype_geometry"]
    output = {
        "schema": "numi.cardboard.pin_adhesion_source_observables.v1",
        "status": "source_observables_extracted_not_calibration",
        "source_bindings": {
            "catalog_sha256": catalog_hash,
            "force_dataset_id": FORCE_DATASET,
            "force_dataset_sha256": force["sha256"],
            "summary_dataset_id": SUMMARY_DATASET,
            "summary_dataset_sha256": summary["sha256"],
            "direct_plot_source_id": PLOT_SOURCE_ID,
            "direct_plot_source_sha256": plot_hash,
            "protocol_source_id": PROTOCOL_SOURCE_ID,
            "protocol_text_sha256": protocol_hash,
            "engineering_profile_id": PROFILE_ID,
            "engineering_profile_sha256": profile_record["sha256"],
        },
        "published_protocol": {
            "fixture": protocol.get("fixture"),
            "standard": protocol.get("standard"),
            "specimen_dimensions_mm": protocol.get("specimen_dimensions_mm"),
            "approach_rate_mm_per_min": protocol.get("platen_approach_rate_mm_per_min"),
            "pin_count": protocol.get("pin_count"),
            "upper_pin_ids": protocol.get("pin_series", {}).get("upper_platen"),
            "lower_pin_ids": protocol.get("pin_series", {}).get("lower_platen"),
            "selected_facing": protocol.get("pin_series", {}).get("selected_facing"),
            "omitted_fixture_geometry": protocol.get("unreported_fixture_geometry"),
            "specimen_orientation_relative_to_MD_CD": protocol.get("dimension_orientation_relative_to_MD_CD"),
        },
        "published_grade_context": {
            "liner_basis_weight_g_m2": 205,
            "liner_grade": "unbleached kraft",
            "medium_basis_weight_g_m2": 122,
            "medium_grade": "regular NSSC",
            "adhesive_grade_or_cure": "not specified in this pin protocol record",
            "transfer_to_other_board": "analogue only; no geometry/grade transfer interval is published",
        },
        "presentation_curve_observables": {
            "low_adhesion_white_glue_lines": low,
            "applicator_roll_too_fast": high,
            "partial_plot_axis_origin": "unknown; low-trace visible threshold metrics are origin-free, raw offsets are not comparable",
            "failure_locus": {"status": "unavailable_in_source", "reason": "no condition-level observed pin/paper/core failure labels"},
        },
        "slide16_summary_markers": {
            "strength_center_range_N_per_mm": [min(strength), max(strength)],
            "strength_digitization_uncertainty_reported_range_N_per_mm": [0.02, 0.03],
            "strength_outer_digitization_range_N_per_mm": [min(strength) - 0.03, max(strength) + 0.03],
            "stiffness_center_range_N_per_mm": [min(stiffness), max(stiffness)],
            "stiffness_outer_digitization_range_N_per_mm": [min(v - u for v, u in zip(stiffness, stiffness_unc)), max(v + u for v, u in zip(stiffness, stiffness_unc))],
            "work_axis_centers": [min(work_values), max(work_values)],
            "work_axis_unit": "unresolved; printed N-m conflicts with integrating slide-15 N by mm",
            "specimen_or_defect_mapping": "unavailable; do not join these summary points to slide-15 curves or article Figure 5",
            "statistical_interpretation": "digitized marker centers only; visible error-bar meaning is unspecified",
        },
        "article_narrative_summary": {
            "reported_pin_strength_range_N_per_mm": [0.17, 0.8],
            "source_id": PROTOCOL_SOURCE_ID,
            "source_text_sha256": protocol_hash,
            "specimen_or_defect_mapping": "unavailable; article Figure 5 values are not point-matched to presentation slide 16",
            "relation_to_slide16": "reported ranges differ; preserve separately and do not infer replicate identity",
        },
        "engineering_profile": {
            "dimension_basis": engineering.get("dimension_basis"),
            "chosen_geometry": engineering,
            "source_equivalent": False,
            "published_nuisance_bounds_available": False,
        },
        "identifiability": {
            "whole_fixture_visible_window_peak": "bounded analogue descriptor for the low slide-15 curve only; not a source-grade matched test",
            "whole_fixture_initial_stiffness": "not condition-identifiable; slide-16 points lack specimen IDs and slide-15 displacement origin is absent",
            "visible_window_secant_and_area": "extractable for the low curve under the stated threshold window; area is N*mm and is not slide-16 work",
            "local_starch_traction_strength": "not identifiable from whole-fixture force/displacement alone",
            "GIc_GIIc_or_mixed_mode_law": "not identifiable without local/opening partitions, matched grade, and condition-level failure locus",
            "failure_locus": "not observed; native damage categories cannot be fitted as if they were source outcomes",
            "nuisance_geometry_transfer": "unbounded by publication; 1 mm pin, 8 mm pitch, registration, overhang, and 5 mm caliper in profile are engineering choices only",
            "calibration_decision": "blocked; retain user-supplied feasible parameter sets and report sensitivity only",
        },
        "native_prediction": {
            "provided": False,
            "note": "This source-side report does not admit or synthesize a native history. Use cardboard_calibration_ensemble.py with a complete hash-bound native certificate when available.",
        },
        "limitations": [
            "Presentation curve points are sparse visual estimates, not raw measurements or replicates.",
            "The article is protocol context only; article figures are not pixel-matched to presentation points.",
            "The low-curve peak is the maximum sampled visible centerline point, not guaranteed the specimen's global maximum.",
            "Trapezoidal area is a visible-window descriptor; reported pointwise pixel bounds do not cover unknown curve shape between samples.",
            "The high-response curve is truncated before its peak and threshold return.",
            "No source-bounded nuisance geometry or grade-transfer interval exists for current shipping constituents.",
        ],
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.catalog, args.data_dir)
    except (PinObservableError, source_data.PackageError, OSError, json.JSONDecodeError) as exc:
        print(f"cardboard_pin_adhesion_source_observables: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
