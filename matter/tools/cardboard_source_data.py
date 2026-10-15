#!/usr/bin/env python3
"""Validate and analyze the published-data-only cardboard source package.

This utility deliberately does not infer cohesive-zone parameters from pin
adhesion, ECT, or unrelated adhesive studies. Digitization is a coordinate
conversion with provenance, not a claim that plotted pixels are raw data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


TOOL = Path(__file__).resolve()
REPO = TOOL.parents[2]
DATA = REPO / "matter" / "examples" / "cardboard" / "source-data"
CATALOG = DATA / "catalog.json"


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read catalog {path}: {exc}") from exc
    if catalog.get("schema") != "numi.cardboard.source_data_catalog.v1":
        raise PackageError("unsupported source-data catalog schema")
    return catalog


def validate_catalog(catalog: dict[str, Any], data_dir: Path = DATA) -> list[str]:
    errors: list[str] = []
    sources = {row.get("id"): row for row in catalog.get("sources", [])}
    datasets = catalog.get("datasets", [])
    ids = [row.get("id") for row in datasets]
    if len(ids) != len(set(ids)):
        errors.append("duplicate dataset id")
    for row in datasets:
        dataset_id = str(row.get("id", "<missing-id>"))
        filename = row.get("file")
        if not filename:
            errors.append(f"{dataset_id}: missing data file")
            continue
        path = (data_dir / filename).resolve()
        if data_dir.resolve() not in path.parents:
            errors.append(f"{dataset_id}: data file escapes package directory")
            continue
        if not path.is_file():
            errors.append(f"{dataset_id}: missing {filename}")
            continue
        if sha256(path) != row.get("sha256"):
            errors.append(f"{dataset_id}: SHA256 mismatch for {filename}")
            continue
        for source_id in row.get("source_ids", []):
            if source_id not in sources:
                errors.append(f"{dataset_id}: unknown source id {source_id}")
        required = ("class", "split", "specimen", "condition", "failure",
                    "source_location", "units", "limits", "calibration_scope")
        for field in required:
            if field not in row:
                errors.append(f"{dataset_id}: missing metadata field {field}")
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except (OSError, csv.Error) as exc:
            errors.append(f"{dataset_id}: CSV read error: {exc}")
            continue
        if not rows:
            errors.append(f"{dataset_id}: CSV contains no rows")
        if dataset_id.startswith("johnson_popil"):
            presentation_id = "johnson_popil_2016_author_presentation"
            article_id = "johnson_popil_2015_article"
            if row.get("source_ids") != [presentation_id]:
                errors.append(
                    f"{dataset_id}: direct plot source must be the 2016 presentation only; "
                    "article protocol context cannot be merged into pixel provenance"
                )
            if row.get("protocol_source_id") != article_id:
                errors.append(f"{dataset_id}: article protocol source must remain separately identified")
            lineage = row.get("source_lineage") or {}
            if lineage.get("direct_plot_source_id") != presentation_id or \
                    lineage.get("protocol_source_id") != article_id or \
                    "never count the article and presentation as independent datasets" not in \
                    lineage.get("independence", ""):
                errors.append(f"{dataset_id}: article/presentation source lineage is missing or merged")
            if not row.get("published_protocol") and dataset_id == "johnson_popil_pin_force_displacement":
                errors.append(f"{dataset_id}: published article protocol fields are missing")
            for index, point in enumerate(rows, start=2):
                if point.get("observation_class") != "digitized_estimate":
                    errors.append(f"{dataset_id}:{index}: pin plot point must remain a digitized estimate")
                if point.get("source_id") != presentation_id:
                    errors.append(f"{dataset_id}:{index}: CSV source_id must bind to the presentation pixels")
            if "manual_digitization" not in row.get("class", ""):
                errors.append(f"{dataset_id}: plotted pin data class must declare manual digitization")
        if dataset_id == "johnson_popil_pin_summary":
            for index, point in enumerate(rows, start=2):
                if point.get("pin_adhesion_work_unit_as_printed") != "N-m" or point.get("pin_adhesion_work_unit_status") != "unresolved_conflict_with_slide15_integral_units":
                    errors.append(f"{dataset_id}:{index}: preserve the printed work unit and unresolved dimensional conflict")
        if dataset_id == "haj_ali_2009_yield_anchors":
            allowed = {"published_tensile_yield_anchor", "published_model_input_60pct_assumption"}
            for index, point in enumerate(rows, start=2):
                if point.get("observation_class") not in allowed:
                    errors.append(f"{dataset_id}:{index}: unexpected curve or measurement classification")
                if point.get("loading") == "compression" and point.get("observation_class") != "published_model_input_60pct_assumption":
                    errors.append(f"{dataset_id}:{index}: compression point mislabeled as measurement")
        if dataset_id == "haj_ali_2009_bilinear_tension_fit_nodes":
            required_roles = {
                "axis_origin",
                "published_Table_1_tensile_yield_anchor",
                "digitized_bilinear_fit_endpoint",
            }
            series = {}
            for index, point in enumerate(rows, start=2):
                series.setdefault(point.get("series_id"), []).append(point)
                if point.get("observation_class") != "digitized_source_bilinear_fit_estimate_not_raw_measurement":
                    errors.append(f"{dataset_id}:{index}: preserve source-fit estimate classification")
                if point.get("curve_label") != "Bilinear tension":
                    errors.append(f"{dataset_id}:{index}: only the plotted bilinear tension fit is admitted")
            if len(series) != 4 or any({p.get("point_role") for p in points} != required_roles or len(points) != 3
                                       for points in series.values()):
                errors.append(f"{dataset_id}: expected three linked nodes for each of four published curves")
            if any("compression" in str(p.get("series_id", "")).lower() for p in rows):
                errors.append(f"{dataset_id}: compression response is not digitized from these fits")
        if dataset_id == "popil_2012_ect_rate_fit":
            for index, point in enumerate(rows, start=2):
                if point.get("observation_class") != "published_equation_evaluation":
                    errors.append(f"{dataset_id}:{index}: equation evaluations must not be labeled raw observations")
        if dataset_id == "impact_data_boundary":
            for index, point in enumerate(rows, start=2):
                if point.get("availability") == "source_states_experimental_data_confidential" and point.get("requested_observable") != "raw_force_time_or_acceleration_time_curves":
                    errors.append(f"{dataset_id}:{index}: confidential raw-data boundary was altered")
        if dataset_id == "robertsson_2023_eq18_compression_branch":
            for index, point in enumerate(rows, start=2):
                x = float(point["I13"])
                e6 = float(point["E6_MPa"])
                e7 = float(point["E7"])
                if point["branch_case"] == "exact_identity_tension_branch":
                    printed = authored = 0.0
                else:
                    exponential = math.exp(-e7 * (x * x - 1.0)) / e7
                    printed = e6 * (x * x + exponential - 2.0)
                    authored = e6 * (x * x + exponential - 1.0 - 1.0 / e7)
                if not math.isclose(float(point["paper_absolute_eq18_zd_term_MPa"]), printed, rel_tol=0.0, abs_tol=2e-12):
                    errors.append(f"{dataset_id}:{index}: paper_absolute Eq. 18 evaluation mismatch")
                if not math.isclose(float(point["recentered_builder_zd_term_MPa"]), authored, rel_tol=0.0, abs_tol=2e-12):
                    errors.append(f"{dataset_id}:{index}: recentered builder evaluation mismatch")
                if not math.isclose(float(point["recentered_minus_paper_absolute_MPa"]), authored - printed, rel_tol=0.0, abs_tol=2e-12):
                    errors.append(f"{dataset_id}:{index}: reported absolute-energy offset mismatch")
        if dataset_id == "robertsson_2023_strength_hardening":
            expected = {
                "1": (20.4, 12.2, 690.0, "logarithmic"),
                "2": (11.7, 5.10, 435.0, "logarithmic"),
                "3": (9.91, 6.63, 334.0, "logarithmic"),
                "4": (22.0, None, None, "none_reported"),
                "5": (16.7, None, None, "none_reported"),
                "6": (9.91, 6.63, 334.0, "logarithmic"),
                "7": (1.0, 49.0, None, "linear"),
                "8": (0.40, None, None, "none_reported"),
                "9": (2.51, None, None, "none_reported"),
                "10": (2.51, None, None, "none_reported"),
                "11": (2.51, None, None, "none_reported"),
                "12": (2.51, None, None, "none_reported"),
            }
            if len(rows) != 12 or {point.get("subsurface_nu") for point in rows} != set(expected):
                errors.append(f"{dataset_id}: expected exactly the twelve Table C2 modes")
            for index, point in enumerate(rows, start=2):
                values = expected.get(point.get("subsurface_nu", ""))
                if not values:
                    continue
                k0, amplitude, rate, law = values
                if not math.isclose(float(point["initial_yield_K0_MPa"]), k0,
                                    rel_tol=0.0, abs_tol=1e-12):
                    errors.append(f"{dataset_id}:{index}: Table C2 initial strength mismatch")
                if point.get("hardening_law") != law:
                    errors.append(f"{dataset_id}:{index}: Table C2 hardening law mismatch")
                for column, expected_value in (("hardening_a_MPa", amplitude),
                                               ("hardening_b", rate)):
                    value = point.get(column, "").strip()
                    if expected_value is None:
                        if value:
                            errors.append(f"{dataset_id}:{index}: source dash must remain blank in {column}")
                    elif not math.isclose(float(value), expected_value,
                                          rel_tol=0.0, abs_tol=1e-12):
                        errors.append(f"{dataset_id}:{index}: Table C2 {column} mismatch")
        if dataset_id == "nagasawa_2012_fig9_scored_bend_envelope":
            errors.extend(validate_nagasawa_fig9_envelope(rows, row, sources))
        digitization = row.get("digitization") or {}
        render = digitization.get("render") or {}
        render_path = render.get("path")
        if render_path and Path(render_path).is_file():
            if sha256(Path(render_path)) != render.get("sha256"):
                errors.append(f"{dataset_id}: archived plot render hash mismatch")
        if render:
            expected_pdf_hashes = {
                sources[source_id].get("external_original_sha256")
                for source_id in row.get("source_ids", [])
                if source_id in sources and sources[source_id].get("external_original_sha256")
            }
            if expected_pdf_hashes and render.get("source_pdf_sha256") not in expected_pdf_hashes:
                errors.append(f"{dataset_id}: plot render is not bound to an associated source PDF")
    ident = catalog.get("cohesive_parameter_identifiability", {})
    if ident.get("identified_parameters"):
        errors.append("cohesive parameters are marked identified despite the registered source limitations")
    for key in ("mode-I fracture energy GIc", "mode-II fracture energy GIIc"):
        if key not in ident.get("unidentified_parameters", []):
            errors.append(f"cohesive identifiability record omits {key}")
    if ident.get("fail_closed") is not True:
        errors.append("cohesive identifiability must fail closed")
    for source in catalog.get("sources", []):
        external = source.get("external_original_pdf")
        expected = source.get("external_original_sha256")
        if external and expected:
            path = Path(external)
            # Archived originals may be absent on another host. A present file
            # must match; catalog CSV validation remains portable.
            if path.exists() and sha256(path) != expected:
                errors.append(f"source {source.get('id')}: external PDF hash mismatch")
        text_extract = source.get("source_text_extract")
        text_hash = source.get("source_text_extract_sha256")
        if text_extract:
            path = (data_dir / text_extract).resolve()
            if data_dir.resolve() not in path.parents:
                errors.append(f"source {source.get('id')}: source text extract escapes package directory")
            elif not path.is_file():
                errors.append(f"source {source.get('id')}: missing source text extract {text_extract}")
            elif not text_hash or sha256(path) != text_hash:
                errors.append(f"source {source.get('id')}: source text extract SHA256 mismatch")
        external_text = source.get("external_text_extract")
        external_text_hash = source.get("external_text_extract_sha256")
        if external_text and Path(external_text).exists():
            if not external_text_hash or sha256(Path(external_text)) != external_text_hash:
                errors.append(f"source {source.get('id')}: external source text extract hash mismatch")
        for asset_key, label in (("embedded_figure_image", "embedded figure image"),
                                 ("page_render", "page render"),
                                 ("access_audit", "access audit")):
            asset = source.get(asset_key) or {}
            asset_path = asset.get("path")
            asset_hash = asset.get("sha256")
            if asset_path and Path(asset_path).is_file():
                if not asset_hash or sha256(Path(asset_path)) != asset_hash:
                    errors.append(f"source {source.get('id')}: external {label} hash mismatch")
    article = sources.get("johnson_popil_2015_article", {})
    if article.get("source_text_extract") != "johnson_popil_2015_pin_protocol_record.md":
        errors.append("Johnson-Popil article protocol record is not registered as the separate article extract")
    if not article.get("source_text_extract_sha256"):
        errors.append("Johnson-Popil article protocol extract must be SHA-bound")
    conflict_ids = {row.get("id") for row in catalog.get("source_conflicts", [])}
    required_conflicts = {
        "johnson_popil_article_vs_presentation_figure_identity",
        "johnson_popil_pin_work_unit_conflict",
        "johnson_popil_strength_range_across_artifacts",
    }
    if not required_conflicts.issubset(conflict_ids):
        errors.append("Johnson-Popil article/presentation figure, work-unit, or strength-range conflict record is missing")
    for profile in catalog.get("fixture_profiles", []):
        profile_name = str(profile.get("file", ""))
        profile_path = (data_dir / profile_name).resolve()
        if not profile_name or data_dir.resolve() not in profile_path.parents:
            errors.append("pin fixture profile path escapes source-data package")
        elif not profile_path.is_file():
            errors.append(f"pin fixture profile is missing: {profile_name}")
        elif sha256(profile_path) != profile.get("sha256"):
            errors.append(f"pin fixture profile SHA256 mismatch: {profile_name}")
        if profile.get("source_exact_geometry") is not False:
            errors.append("pin fixture prototype must not claim source-exact geometry")
        if profile.get("direct_protocol_source_id") != "johnson_popil_2015_article" or \
                profile.get("direct_plot_source_id") != "johnson_popil_2016_author_presentation":
            errors.append("pin fixture profile must separate article protocol and presentation plot sources")
    return errors


def validate_nagasawa_fig9_envelope(rows: list[dict[str, str]], dataset: dict[str, Any],
                                    sources: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Check group identity, axis conversion and readout bounds for Fig. 9."""
    errors: list[str] = []
    prefix = "nagasawa_2012_fig9_scored_bend_envelope"
    if len(rows) != 90:
        errors.append(f"{prefix}: expected 90 grouped angle samples, not {len(rows)}")
        return errors
    if any(key in rows[0] for key in ("trace_id", "sample_id", "specimen_id")):
        errors.append(f"{prefix}: individual trace/specimen identities must stay unresolved")
    conditions = {point.get("condition_id") for point in rows}
    groups = {point.get("condition_group") for point in rows}
    if conditions != {"nagasawa_2012_fig9_cP3p0_cC0p7_V2p92"} or len(groups) != 1:
        errors.append(f"{prefix}: samples must remain one source condition group")
    if dataset.get("split") != "condition_group_holdout_no_point_split":
        errors.append(f"{prefix}: all angle samples must share a condition-group split")
    curve_identity = dataset.get("curve_identity", {})
    if (curve_identity.get("reported_trace_count") != 10
            or curve_identity.get("individual_trace_mapping") != "unresolved_overlap"
            or curve_identity.get("stored_representation") != "pooled_outer_envelope_not_replicates"):
        errors.append(f"{prefix}: source curve identity must remain a pooled, unresolved envelope")
    if any(key in dataset for key in ("confidence_interval", "replicate_distribution", "individual_curves")):
        errors.append(f"{prefix}: do not infer replicate statistics from the plotted envelope")
    ratio_note = dataset.get("score_gap_ratio_semantics", {})
    if ratio_note.get("nagasawa_article_sweep_0p09_to_0p40") != "main_creaser_roll_gap_over_measured_board_caliper":
        errors.append(f"{prefix}: 0.09-0.40 must remain the main roll-gap/caliper ratio")
    if ratio_note.get("not_punch_indentation_ratio") is not True:
        errors.append(f"{prefix}: do not relabel roll-gap/caliper as punch indentation")
    digitization = dataset.get("digitization", {})
    axes = digitization.get("axis_mapping", {})
    image_hash = (digitization.get("render") or {}).get("sha256")
    pdf_hash = (digitization.get("render") or {}).get("source_pdf_sha256")
    expected_readout_angle = 2.0 * 100.0 / (616.0 - 117.0)
    expected_readout_moment = 2.0 * 5.0 / (324.0 - 21.0)
    expected_pixels = list(range(1, 91))
    for index, point in enumerate(rows, start=1):
        line = index + 1
        try:
            angle = float(point["angle_deg"])
            moment_low = float(point["moment_envelope_min_Nm_per_m"])
            moment_high = float(point["moment_envelope_max_Nm_per_m"])
            x_pixel = int(point["sample_x_pixel"])
            low_y_pixel = int(point["lower_envelope_y_pixel"])
            high_y_pixel = int(point["upper_envelope_y_pixel"])
            ux = float(point["readout_uncertainty_angle_deg"])
            uy = float(point["readout_uncertainty_moment_Nm_per_m"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}:{line}: malformed pixel or coordinate value")
            continue
        if angle != float(expected_pixels[index - 1]):
            errors.append(f"{prefix}:{line}: angles must be ordered one-degree group samples from 1 through 90")
        expected_x = round(117.0 + angle * (616.0 - 117.0) / 100.0)
        if x_pixel != expected_x:
            errors.append(f"{prefix}:{line}: sample x pixel does not match the declared angle mapping")
        expected_low = 0.0 if point.get("lower_envelope_censored_at_zero_axis") == "true" else 5.0 - (low_y_pixel - 21.0) * 5.0 / 303.0
        expected_high = 5.0 - (high_y_pixel - 21.0) * 5.0 / 303.0
        if not math.isclose(moment_low, expected_low, rel_tol=0.0, abs_tol=0.0000011):
            errors.append(f"{prefix}:{line}: low moment does not match its pixel and censor flag")
        if not math.isclose(moment_high, expected_high, rel_tol=0.0, abs_tol=0.0000011):
            errors.append(f"{prefix}:{line}: high moment does not match its pixel")
        if moment_low < 0.0 or moment_high < moment_low or moment_high > 5.0:
            errors.append(f"{prefix}:{line}: invalid bending-moment envelope")
        if not math.isclose(ux, expected_readout_angle, rel_tol=0.0, abs_tol=0.0000011):
            errors.append(f"{prefix}:{line}: angle readout uncertainty must equal the two-pixel bound")
        if not math.isclose(uy, expected_readout_moment, rel_tol=0.0, abs_tol=0.0000011):
            errors.append(f"{prefix}:{line}: moment readout uncertainty must equal the two-pixel bound")
        if point.get("observation_class") != "digitized_group_envelope_estimate":
            errors.append(f"{prefix}:{line}: plotted envelope must remain a digitized estimate")
        if point.get("source_pdf_sha256") != pdf_hash or point.get("embedded_image_sha256") != image_hash:
            errors.append(f"{prefix}:{line}: CSV source hashes disagree with catalog render binding")
    if axes != {
        "x_pixel_min": 117, "x_pixel_max": 616, "x_min_deg": 0, "x_max_deg": 100,
        "x_formula": "angle_deg = (x_pixel - 117) * 100 / (616 - 117)",
        "y_pixel_bottom": 324, "y_pixel_top": 21, "y_min_Nm_per_m": 0, "y_max_Nm_per_m": 5,
        "y_formula": "moment_Nm_per_m = (324 - y_pixel) * 5 / (324 - 21)",
    }:
        errors.append(f"{prefix}: declared axis mapping differs from the digitization transform")
    line_width = digitization.get("line_width_handling", "")
    if "outer edges" not in line_width or "two-pixel" not in line_width:
        errors.append(f"{prefix}: line-width and edge-readout handling must be documented")
    source_ids = dataset.get("source_ids", [])
    registered_audit_hashes = {
        (sources[source_id].get("access_audit") or {}).get("sha256")
        for source_id in source_ids if sources and source_id in sources
    }
    if len(registered_audit_hashes) != 1 or digitization.get("source_access_audit_sha256") not in registered_audit_hashes:
        errors.append(f"{prefix}: dataset must bind to the registered source-access audit hash")
    return errors


def package_audit(catalog: dict[str, Any]) -> dict[str, Any]:
    conflicts = {str(row.get("id")): row
                 for row in catalog.get("source_conflicts", [])}
    pin_dataset = next((row for row in catalog.get("datasets", [])
                        if row.get("id") == "johnson_popil_pin_force_displacement"), {})
    pin_profile = next((row for row in catalog.get("fixture_profiles", [])
                        if row.get("id") == "johnson_popil_pin_adhesion_engineering_prototype"), {})
    return {
        "schema": "numi.cardboard.source_data_audit.v1",
        "catalog_version": catalog.get("version"),
        "dataset_count": len(catalog.get("datasets", [])),
        "source_count": len(catalog.get("sources", [])),
        "pin_direct_plot_source_ids": pin_dataset.get("source_ids", []),
        "pin_protocol_context_source_id": pin_dataset.get("protocol_source_id"),
        "pin_article_presentation_figure_identity_status": conflicts.get(
            "johnson_popil_article_vs_presentation_figure_identity", {}).get("status"),
        "pin_article_presentation_strength_range_conflict": conflicts.get(
            "johnson_popil_strength_range_across_artifacts", {}),
        "pin_summary_work_unit_conflict_status": conflicts.get(
            "johnson_popil_pin_work_unit_conflict", {}).get("status"),
        "pin_engineering_fixture_profile": {
            "file": pin_profile.get("file"),
            "sha256": pin_profile.get("sha256"),
            "source_exact_geometry": pin_profile.get("source_exact_geometry"),
        },
        "haj_ali_full_hardening_curve_available": False,
        "haj_ali_curve_reason": "the author-posted page-4 render exposes the four source-labeled bilinear tension-fit lines, now retained as bounded digitized fit estimates; raw experimental marker curves and compression curves remain unavailable",
        "haj_ali_bilinear_tension_fit_estimate_available": True,
        "haj_ali_bilinear_tension_fit_reason": "four fitted tension lines are represented by Table 1 yield anchors plus one pixel-read endpoint each; they are source-model fit estimates, not raw experimental traces or specimen replicates",
        "robertsson_strength_hardening_table_available": True,
        "robertsson_strength_hardening_scope": "Table C2 and Eq. 27 source parameters for separate single-ply softwood grade; not current Kraft/NSSC calibration",
        "robertsson_figure4_experimental_curve_available": False,
        "robertsson_figure4_curve_reason": "Wiley Figure 4 is graphical with no numeric curve table; the publisher PowerPoint endpoint returned unsupported application/vnd.ms-powerpoint, the headed-browser article attempt returned HTTP 403, and the in-app browser blocked direct endpoint access, so no source pixels were captured",
        "robertsson_eq18_energy_discrepancy": {
            "resolved": False,
            "recentered_minus_paper_absolute_compression_constant_MPa": 0.3332608695652174,
            "conjugate_force_affected": True,
            "available_builder_conventions": ["paper_absolute", "recentered"],
            "implicit_default": None,
            "historical_regression_entry_point": "makeLegacyRecenteredMaterial",
            "required_source_verification_entry_point": "makePublishedReferenceMaterial(paper_absolute)",
            "full_selected_energy_scales_damage_potential": True,
            "source_extract_sha256": next((s.get("source_text_extract_sha256") for s in catalog.get("sources", []) if s.get("id") == "robertsson_2023"), None),
        },
        "cohesive_calibration_identified": bool(catalog.get("cohesive_parameter_identifiability", {}).get("identified_parameters")),
        "cohesive_parameters_unidentified": catalog.get("cohesive_parameter_identifiability", {}).get("unidentified_parameters", []),
        "impact_rate_calibration_available": False,
        "impact_rate_reason": "Bernad 2023 reports confidential raw data; other found impact source is a different padded package without raw synchronized histories",
    }


def select_public_pdf(source: dict[str, Any], url_index: int = 0) -> str:
    access = source.get("access_state", "").lower()
    if not any(word in access for word in ("open", "public")):
        raise PackageError("source is not registered as openly/publicly accessible; do not bypass an access gate")
    urls = [u for u in source.get("urls", []) if urlparse(u).scheme == "https" and urlparse(u).path.lower().endswith(".pdf")]
    if url_index < 0 or url_index >= len(urls):
        raise PackageError("no registered public PDF at that URL index")
    return urls[url_index]


def acquire_pdf(source: dict[str, Any], output: Path, repo_root: Path = REPO, url_index: int = 0) -> dict[str, Any]:
    """Download only a catalog-registered, explicitly public PDF to an external path."""
    url = select_public_pdf(source, url_index)
    output = output.expanduser().resolve()
    if output == repo_root.resolve() or repo_root.resolve() in output.parents:
        raise PackageError("original PDFs must be stored outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        digest = sha256(output)
        expected = source.get("external_original_sha256")
        if expected and digest == expected:
            return {"status": "already_present_hash_matches", "path": str(output), "sha256": digest, "url": url}
        raise PackageError("output exists; refusing to overwrite an unverified archive file")
    temporary = output.with_name(output.name + ".part")
    request = Request(url, headers={"User-Agent": "Numi-Cardboard-Published-Source-Archive/1.0"})
    total = 0
    try:
        with urlopen(request, timeout=30) as response, temporary.open("wb") as target:
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                raise PackageError(f"registered URL did not return a PDF content type: {content_type or 'missing'}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > 100 * 1024 * 1024:
                    raise PackageError("PDF exceeds 100 MiB acquisition limit")
                target.write(block)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    try:
        with temporary.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise PackageError("download did not begin with a PDF signature")
        digest = sha256(temporary)
        expected = source.get("external_original_sha256")
        if expected and digest != expected:
            raise PackageError(f"download hash differs from registered original: expected {expected}, got {digest}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"status": "downloaded", "path": str(output), "sha256": digest, "url": url}


def pixel_to_units(x_px: float, y_px: float, axes: dict[str, Any]) -> tuple[float, float]:
    """Map image coordinates (origin top-left) to linear chart units."""
    x0, x1 = float(axes["x_pixel_min"]), float(axes["x_pixel_max"])
    y0, y1 = float(axes["y_pixel_min"]), float(axes["y_pixel_max"])
    xmin, xmax = float(axes["x_min"]), float(axes["x_max"])
    ymin, ymax = float(axes["y_min"]), float(axes["y_max"])
    values = (x0, x1, y0, y1, xmin, xmax, ymin, ymax, float(x_px), float(y_px))
    if not all(math.isfinite(value) for value in values):
        raise PackageError("pixel coordinates and axis bounds must be finite")
    if not (x1 > x0 and y1 > y0 and xmax > xmin and ymax > ymin):
        raise PackageError("pixel bounds must increase from minimum to maximum")
    x = xmin + (x_px - x0) * (xmax - xmin) / (x1 - x0)
    # y pixel bounds are image top and image bottom; physical y increases upward.
    y = ymax - (y_px - y0) * (ymax - ymin) / (y1 - y0)
    if not (xmin <= x <= xmax):
        raise PackageError("point falls outside calibrated x-axis bounds")
    if not (ymin <= y <= ymax):
        raise PackageError("point falls outside calibrated y-axis bounds")
    return x, y


def digitize_request(request_path: Path, output_path: Path, catalog: dict[str, Any]) -> None:
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read digitization request: {exc}") from exc
    sources = {row["id"]: row for row in catalog.get("sources", [])}
    source_id = request.get("source_id")
    if source_id not in sources:
        raise PackageError("digitization source_id is not registered")
    registered_sha = sources[source_id].get("external_original_sha256")
    if not registered_sha or request.get("source_original_sha256") != registered_sha:
        raise PackageError("digitization request does not bind to a retained, hash-verified original source")
    page = request.get("page")
    page_sha = request.get("page_render_sha256")
    expected_pdf_hash = sources[source_id].get("external_original_sha256")
    allowed_renders = [
        dataset.get("digitization", {}).get("render", {})
        for dataset in catalog.get("datasets", [])
        if source_id in dataset.get("source_ids", [])
        and dataset.get("digitization", {}).get("render")
    ]
    if not any(render.get("page") == page
               and render.get("sha256") == page_sha
               and render.get("source_pdf_sha256") == expected_pdf_hash
               for render in allowed_renders):
        raise PackageError("digitization request does not bind to a registered rendered page SHA and page number")
    axes = request.get("axes", {})
    point_rows = request.get("points", [])
    if len(point_rows) < 2:
        raise PackageError("at least two manually selected source-plot points are required")
    x_unit, y_unit = axes.get("x_unit"), axes.get("y_unit")
    if not x_unit or not y_unit or not page or not request.get("series_id"):
        raise PackageError("page, series_id, and both axis units must be stated")
    ux, uy = request.get("uncertainty", {}).get("x"), request.get("uncertainty", {}).get("y")
    try:
        ux_value, uy_value = float(ux), float(uy)
    except (TypeError, ValueError) as exc:
        raise PackageError("positive finite visual digitization uncertainties are required") from exc
    if not (math.isfinite(ux_value) and math.isfinite(uy_value) and ux_value > 0 and uy_value > 0):
        raise PackageError("positive visual digitization uncertainties are required")
    output_path = output_path.expanduser().resolve()
    request_path = request_path.expanduser().resolve()
    if output_path == request_path:
        raise PackageError("digitized CSV output cannot overwrite its request file")
    if output_path.exists():
        raise PackageError("digitized CSV output already exists; refusing to overwrite evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_id", "series_id", "x", "y", "x_unit", "y_unit", "x_uncertainty", "y_uncertainty", "observation_class", "source_page", "source_original_sha256", "source_render_sha256"]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in point_rows:
            try:
                x, y = pixel_to_units(float(point["x_px"]), float(point["y_px"]), axes)
            except (KeyError, TypeError, ValueError) as exc:
                output_path.unlink(missing_ok=True)
                raise PackageError(f"invalid digitized point: {exc}") from exc
            writer.writerow({"source_id": source_id, "series_id": request["series_id"], "x": f"{x:.8g}", "y": f"{y:.8g}", "x_unit": x_unit, "y_unit": y_unit, "x_uncertainty": f"{ux_value:.8g}", "y_uncertainty": f"{uy_value:.8g}", "observation_class": "digitized_estimate", "source_page": page, "source_original_sha256": registered_sha, "source_render_sha256": page_sha})


def linear_fit(rows: list[dict[str, str]], x_key: str, y_key: str) -> dict[str, float]:
    values = [(float(r[x_key]), float(r[y_key])) for r in rows if r.get(x_key, "").strip() and r.get(y_key, "").strip()]
    if len(values) < 3:
        raise PackageError("descriptive fit requires at least three digitized plot markers")
    xs, ys = zip(*values)
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        raise PackageError("fit x values have no spread")
    slope = sum((x - xbar) * (y - ybar) for x, y in values) / sxx
    intercept = ybar - slope * xbar
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in values)
    total = sum((y - ybar) ** 2 for y in ys)
    return {"n": len(values), "slope": slope, "intercept": intercept, "r_squared_descriptive": 1.0 - residual / total if total else 1.0}


def fit_pin_summary(catalog: dict[str, Any], outcome: str) -> dict[str, Any]:
    row = next(d for d in catalog["datasets"] if d["id"] == "johnson_popil_pin_summary")
    path = DATA / row["file"]
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    key = "pin_adhesion_work_plot_value" if outcome == "work" else "pin_stiffness_N_mm"
    filtered = [r for r in data if r.get(key, "").strip()]
    fit = linear_fit(filtered, "pin_adhesion_strength_N_mm", key)
    y_label = "pin adhesion work (source axis unit unresolved; slide prints N-m)" if outcome == "work" else "pin adhesion stiffness (N/mm)"
    return {"status": "descriptive_digitized_fit_only", "source_dataset": row["id"], "x": "pin adhesion strength (N/mm)", "y": y_label, "fit": fit, "confidence_interval": None, "prediction_admission": "not admitted for material/cohesive calibration; no raw replicate points or covariance are available", "input_points_are_digitized_estimates": True}


def cohesive_status(catalog: dict[str, Any]) -> int:
    ident = catalog["cohesive_parameter_identifiability"]
    missing = ident["unidentified_parameters"]
    print(json.dumps({"status": "not_identified", "identified_parameters": ident["identified_parameters"], "unidentified_parameters": missing, "reason": "the available pin, HMA and bulk-paper sources do not supply a same-grade internal-starch traction-separation calibration"}, indent=2))
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check catalog metadata, source links, CSV hashes and fail-closed rules")
    sub.add_parser("audit", help="report which published calibration targets remain unavailable")
    sub.add_parser("check-cohesive", help="fail closed when a local starch cohesive law is not identified")
    acquire_parser = sub.add_parser("acquire-pdf", help="download a catalog-registered public PDF to an external archive")
    acquire_parser.add_argument("source_id")
    acquire_parser.add_argument("--output", type=Path, required=True)
    acquire_parser.add_argument("--url-index", type=int, default=0)
    fit_parser = sub.add_parser("fit-pin-summary", help="descriptive OLS of digitized author-summary markers; no confidence interval")
    fit_parser.add_argument("--outcome", choices=("work", "stiffness"), required=True)
    digitize_parser = sub.add_parser("digitize", help="map manually selected image pixels from a hash-bound source plot")
    digitize_parser.add_argument("request", type=Path)
    digitize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog()
        if args.command == "validate":
            errors = validate_catalog(catalog)
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"valid: {len(catalog['datasets'])} datasets; source CSV hashes match; unidentifiable interface parameters remain fail-closed")
            return 0
        if args.command == "audit":
            print(json.dumps(package_audit(catalog), indent=2))
            return 0
        if args.command == "check-cohesive":
            return cohesive_status(catalog)
        if args.command == "acquire-pdf":
            source = next((s for s in catalog.get("sources", []) if s.get("id") == args.source_id), None)
            if source is None:
                raise PackageError(f"unknown source id {args.source_id}")
            print(json.dumps(acquire_pdf(source, args.output, url_index=args.url_index), indent=2))
            return 0
        if args.command == "fit-pin-summary":
            print(json.dumps(fit_pin_summary(catalog, args.outcome), indent=2))
            return 0
        if args.command == "digitize":
            digitize_request(args.request, args.output, catalog)
            print(f"wrote digitized estimates to {args.output}")
            return 0
    except (PackageError, KeyError, ValueError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
