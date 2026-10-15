#!/usr/bin/env python3
"""CPU contract tests; temporary certificate fixtures are not native evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


TOOL_PATH = Path(__file__).with_name("cardboard_calibration_ensemble.py")
SPEC = importlib.util.spec_from_file_location("cardboard_calibration_ensemble", TOOL_PATH)
assert SPEC and SPEC.loader
ensemble = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ensemble)
DATA = TOOL_PATH.parents[1] / "examples/cardboard/source-data"
CATALOG_PATH = DATA / "catalog.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_candidate(candidate_id: str, role: str = "fit_candidate", value: float = 1.234) -> dict[str, Any]:
    if role == "fit_candidate":
        row = {"candidate_id": candidate_id, "role": role, "origin": "sensitivity_grid",
               "parameters": {"GIc": {"value": value, "unit": "J/m2", "family": "mode_I_cohesive", "fit": True}}}
    else:
        control = "unbonded" if role == "unbonded_control" else "perfectly_tied"
        row = {"candidate_id": candidate_id, "role": role,
               "control_definition": control, "parameters": {}}
    row["candidate_sha256"] = canonical(row)
    return row


class CalibrationEnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.catalog_hash, self.datasets = ensemble.load_source_package(CATALOG_PATH, DATA)
        self.source = next(s for s in self.catalog["sources"] if s["id"] == "johnson_popil_2016_author_presentation")
        self.dataset = self.datasets["johnson_popil_pin_force_displacement"]
        self.selector = {"series_id": "low_adhesion_white_glue_lines", "defect_id": "A"}
        with Path(self.dataset["resolved_path"]).open(newline="", encoding="utf-8") as stream:
            self.observed_rows = [r for r in csv.DictReader(stream)
                                  if all(r.get(k) == v for k, v in self.selector.items())]
        self.assertGreaterEqual(len(self.observed_rows), 2)

    def make_plan(self, base: Path, *, role: str = "validation", prefix: bool = False,
                  bad_unit: bool = False, duplicate_group: bool = False,
                  missing_source_hash: bool = False,
                  backend: str = "numi_matter_native_fem") -> Path:
        condition = {
            "condition_id": "pin-A",
            "specimen_id": "test-only-contract-fixture-pin-A",
            "dataset_id": "johnson_popil_pin_force_displacement",
            "source_id": "johnson_popil_2016_author_presentation",
            "source_dataset_sha256": self.dataset["sha256"],
            "source_asset_sha256": "" if missing_source_hash else self.source["external_original_sha256"],
            "source_catalog_sha256": self.catalog_hash,
            "role": role,
            "parameter_family": "mode_I_cohesive",
            "selector": self.selector,
            "transfer": {"kind": "bounded_analogue", "source_grade": "presentation_board_205_122",
                         "target_grade": "literature2009_205_126",
                         "rationale": "test-only declared transfer interval; not a fitted material assumption",
                         "metric_scale_bounds": {"peak_load_N": [0.8, 1.2],
                                                  "trace_secant_stiffness_N_per_mm": [0.8, 1.2],
                                                  "trace_window_work_N_mm": [0.8, 1.2]}},
            "metrics": {
                "peak_load_N": {"kind": "peak", "column": "load_N", "unit": "N", "y_unit": "N",
                                "uncertainty_column": "load_uncertainty_N"},
                "trace_secant_stiffness_N_per_mm": {"kind": "secant", "x_column": "displacement_mm",
                                "y_column": "load_N", "x_unit": "mm", "y_unit": "N", "unit": "N/mm",
                                "x_uncertainty_column": "displacement_uncertainty_mm",
                                "y_uncertainty_column": "load_uncertainty_N"},
                "trace_window_work_N_mm": {"kind": "work", "x_column": "displacement_mm",
                                "y_column": "load_N", "x_unit": "mm", "y_unit": "N", "unit": "N*mm",
                                "x_uncertainty_column": "displacement_uncertainty_mm",
                                "y_uncertainty_column": "load_uncertainty_N"},
                "failure_locus": {"kind": "unavailable", "reason": "presentation did not publish a condition-level failure-locus label"},
            },
            "required_metrics": ["peak_load_N", "trace_secant_stiffness_N_per_mm",
                                 "trace_window_work_N_mm", "failure_locus"],
        }
        conditions = [condition]
        if duplicate_group:
            second = dict(condition)
            second["condition_id"] = "pin-A-validation"
            second["role"] = "validation" if role == "calibration" else "calibration"
            conditions.append(second)
        candidates = [make_candidate("grid-001"), make_candidate("grid-002", value=2.468),
                      make_candidate("unbonded", "unbonded_control"),
                      make_candidate("tied", "tied_control")]
        plan = {"schema": "numi.cardboard.calibration_ensemble.v1",
                "source_data_catalog_sha256": self.catalog_hash,
                "target_recipe_id": "literature2009_cross_source_recipe",
                "target_grade_id": "literature2009_205_126_kraft_NSSC",
                "conditions": conditions, "candidates": candidates,
                "native_histories": []}
        run_root = base / "runs"
        for candidate in candidates:
            for cond in conditions:
                rows = self.observed_rows
                samples = [{"step": i, "displacement_mm": float(row["displacement_mm"]),
                            "load_N": float(row["load_N"]) * (1.1 if candidate["candidate_id"] == "grid-002" else 1.0)}
                           for i, row in enumerate(rows)]
                history = {"schema": "numi.cardboard.native_forward_history.v1",
                           "candidate_id": candidate["candidate_id"], "condition_id": cond["condition_id"],
                           "candidate_sha256": candidate["candidate_sha256"],
                           "specimen_id": cond["specimen_id"],
                           "source_data_catalog_sha256": self.catalog_hash,
                           "source_dataset_sha256": self.dataset["sha256"],
                           "source_asset_sha256": self.source["external_original_sha256"],
                           "units": {"displacement_mm": "mm", "load_N": "kN" if bad_unit and candidate["candidate_id"] == "grid-001" else "N"},
                           "samples": samples,
                           "outcomes": {}}
                relative_dir = Path("runs") / candidate["candidate_id"] / cond["condition_id"]
                case_dir = base / relative_dir
                history_path = case_dir / "history.json"
                write_json(history_path, history)
                expected_steps = 10
                accepted_steps = 5 if prefix else expected_steps
                artifacts = {}
                manifest = {"schema": "numi.cardboard.native_input_manifest.v1",
                            "candidate_id": candidate["candidate_id"],
                            "candidate_sha256": candidate["candidate_sha256"],
                            "condition_id": cond["condition_id"], "specimen_id": cond["specimen_id"],
                            "source_data_catalog_sha256": self.catalog_hash,
                            "source_dataset_sha256": self.dataset["sha256"],
                            "source_asset_sha256": self.source["external_original_sha256"],
                            "parameters": candidate.get("parameters", {})}
                artifact_specs = {
                    "native_input_manifest": ("native_input_manifest.json", manifest),
                    "native_executable": ("native_executable.bin", b"test executable bytes"),
                    "solver_source": ("solver_source.txt", b"test source bytes"),
                    "run_receipt": ("run_receipt.json", {"candidate_id": candidate["candidate_id"],
                        "candidate_sha256": candidate["candidate_sha256"],
                        "condition_id": cond["condition_id"], "specimen_id": cond["specimen_id"],
                        "terminal_status": "completed", "accepted_steps": accepted_steps,
                        "expected_steps": expected_steps, "physical_fem_steps": 10,
                        "backend": backend}),
                }
                for name, (filename, content) in artifact_specs.items():
                    artifact = case_dir / filename
                    if isinstance(content, dict):
                        write_json(artifact, content)
                    else:
                        artifact.write_bytes(content)
                    artifacts[name] = {"path": artifact.name, "sha256": digest(artifact)}
                cert = {"schema": "numi.cardboard.native_execution_certificate.v1",
                        "candidate_id": candidate["candidate_id"], "condition_id": cond["condition_id"],
                        "candidate_sha256": candidate["candidate_sha256"],
                        "specimen_id": cond["specimen_id"],
                        "execution_class": "native_forward_prediction", "backend": backend,
                        "terminal_status": "completed", "protocol_complete": not prefix, "prefix_only": prefix,
                        "expected_steps": expected_steps, "accepted_steps": accepted_steps,
                        "physical_fem_steps": 10, "sample_count": len(samples),
                        "history_sha256": digest(history_path),
                        "source_data_catalog_sha256": self.catalog_hash,
                        "source_dataset_sha256": self.dataset["sha256"],
                        "source_asset_sha256": self.source["external_original_sha256"],
                        "artifact_hashes": artifacts}
                cert["certificate_sha256"] = canonical(cert)
                cert_path = case_dir / "certificate.json"
                write_json(cert_path, cert)
                plan["native_histories"].append({
                    "candidate_id": candidate["candidate_id"], "condition_id": cond["condition_id"],
                    "candidate_sha256": candidate["candidate_sha256"],
                    "specimen_id": cond["specimen_id"],
                    "history_path": str(relative_dir / "history.json"), "history_sha256": digest(history_path),
                    "certificate_path": str(relative_dir / "certificate.json"),
                    "certificate_file_sha256": digest(cert_path),
                })
        plan_path = base / "plan.json"
        write_json(plan_path, plan)
        return plan_path

    def test_whole_validation_fixture_is_bound_and_controls_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory))
            result = ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)
        self.assertEqual(result["calibration_status"], "no_admitted_calibration_fixture")
        self.assertEqual(result["feasible_candidate_ids"], [])
        self.assertEqual(len(result["validation_metrics"]), 6)
        self.assertEqual(len(result["limiting_controls"]), 6)
        self.assertEqual(result["parameter_identifiability"]["GIc"]["status"],
                         "not_adjudicated_no_admitted_calibration_fixture")
        heldout = result["heldout_predictive_sensitivity_ranges"]
        self.assertEqual(len(heldout), 1)
        self.assertEqual(heldout[0]["condition_role"], "validation")
        self.assertEqual(heldout[0]["paired_contrast_count"], 3)
        self.assertEqual(result["feasible_candidate_ids"], [])
        self.assertIn("failure_locus", result["missing_required_metrics"])
        self.assertTrue(any("holdout/analogue" in reason for reason in result["blockers"]) is False)

    def test_feasible_sets_and_sensitivity_require_admitted_whole_calibration_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog_copy = json.loads(json.dumps(self.catalog))
            source_dataset = next(d for d in catalog_copy["datasets"]
                                  if d["id"] == "johnson_popil_pin_force_displacement")
            source_dataset["split"] = "test_only_calibration_fixture"
            source_dataset["calibration_parameter_families"] = ["mode_I_cohesive"]
            catalog_path = base / "catalog.json"
            write_json(catalog_path, catalog_copy)
            plan_path = self.make_plan(base)
            plan = json.loads(plan_path.read_text())
            plan["source_data_catalog_sha256"] = digest(catalog_path)
            condition = plan["conditions"][0]
            condition["role"] = "calibration"
            condition["source_catalog_sha256"] = digest(catalog_path)
            condition["metrics"].pop("failure_locus")
            condition["required_metrics"] = ["peak_load_N", "trace_secant_stiffness_N_per_mm", "trace_window_work_N_mm"]
            for record in plan["native_histories"]:
                cert_path = base / record["certificate_path"]
                cert = json.loads(cert_path.read_text())
                cert["source_data_catalog_sha256"] = digest(catalog_path)
                history_path = base / record["history_path"]
                history = json.loads(history_path.read_text())
                history["source_data_catalog_sha256"] = digest(catalog_path)
                write_json(history_path, history)
                record["history_sha256"] = digest(history_path)
                cert["history_sha256"] = record["history_sha256"]
                manifest_info = cert["artifact_hashes"]["native_input_manifest"]
                manifest_path = cert_path.parent / manifest_info["path"]
                manifest = json.loads(manifest_path.read_text())
                manifest["source_data_catalog_sha256"] = digest(catalog_path)
                write_json(manifest_path, manifest)
                manifest_info["sha256"] = digest(manifest_path)
                cert.pop("certificate_sha256")
                cert["certificate_sha256"] = canonical(cert)
                write_json(cert_path, cert)
                record["certificate_file_sha256"] = digest(cert_path)
            write_json(plan_path, plan)
            result = ensemble.adjudicate(plan_path, catalog_path=catalog_path, data_dir=DATA)
        self.assertEqual(result["calibration_status"], "feasible_parameter_sets_retained")
        self.assertEqual(set(result["feasible_candidate_ids"]), {"grid-001", "grid-002"})
        self.assertEqual(result["feasible_parameter_envelopes"]["GIc"]["feasible_values"], [1.234, 2.468])
        self.assertEqual(result["parameter_identifiability"]["GIc"]["status"],
                         "nonunique_in_enumerated_feasible_set")
        self.assertEqual(result["sensitivity_ranges"][0]["parameter"], "GIc")
        self.assertEqual(result["sensitivity_ranges"][0]["paired_contrast_count"], 3)
        self.assertIn("not a confidence interval", result["feasible_parameter_envelopes"]["GIc"]["interpretation"])

    def test_completed_prefix_rejected_even_when_history_file_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory), prefix=True)
            with self.assertRaisesRegex(ensemble.EnsembleError, "completed prefixes"):
                ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)

    def test_explicit_cpu_oracle_backend_cannot_enter_native_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory), backend="python_cpu_oracle")
            with self.assertRaisesRegex(ensemble.EnsembleError, "not certified as native forward FEM"):
                ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)

    def test_units_cannot_be_relabelled_to_make_a_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory), bad_unit=True)
            with self.assertRaisesRegex(ensemble.EnsembleError, "incompatible y unit"):
                ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)

    def test_same_published_condition_cannot_be_reused_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory), duplicate_group=True)
            with self.assertRaisesRegex(ensemble.EnsembleError, "same published fixture group"):
                ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)

    def test_missing_publisher_source_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_path = self.make_plan(Path(directory), missing_source_hash=True)
            with self.assertRaisesRegex(ensemble.EnsembleError, "source asset/text SHA-256"):
                ensemble.adjudicate(plan_path, catalog_path=CATALOG_PATH, data_dir=DATA)

    def test_pin_protocol_article_cannot_replace_direct_presentation_source(self) -> None:
        _, catalog_hash, datasets = ensemble.load_source_package(CATALOG_PATH, DATA)
        source = next(s for s in self.catalog["sources"]
                      if s["id"] == "johnson_popil_2015_article")
        condition = {"condition_id": "pin-A", "specimen_id": "sim-pin-A",
                     "dataset_id": "johnson_popil_pin_force_displacement",
                     "source_id": source["id"],
                     "source_dataset_sha256": datasets["johnson_popil_pin_force_displacement"]["sha256"],
                     "source_asset_sha256": source["source_text_extract_sha256"],
                     "source_catalog_sha256": catalog_hash,
                     "role": "validation", "selector": self.selector,
                     "metrics": {"peak_load_N": {"kind": "peak", "column": "load_N",
                                                    "unit": "N", "y_unit": "N"}}}
        with self.assertRaisesRegex(ensemble.EnsembleError,
                                    "source_id does not match the dataset catalog"):
            ensemble._load_conditions({"conditions": [condition],
                                       "_sources": self.catalog["sources"]},
                                      datasets, catalog_hash)

    def test_catalog_source_without_retained_hash_cannot_be_used(self) -> None:
        _, catalog_hash, datasets = ensemble.load_source_package(CATALOG_PATH, DATA)
        condition = {"condition_id": "haj-yield", "specimen_id": "test-only",
                     "dataset_id": "haj_ali_2009_yield_anchors", "source_id": "haj_ali_2009",
                     "source_dataset_sha256": datasets["haj_ali_2009_yield_anchors"]["sha256"],
                     "source_asset_sha256": None, "source_catalog_sha256": catalog_hash,
                     "role": "calibration", "parameter_family": "paper_yield_anchor",
                     "selector": {"component": "liner", "basis_weight_g_m2": "205", "axis": "MD",
                                  "loading": "tension"},
                     "metrics": {"yield": {"kind": "peak", "column": "stress_MPa", "unit": "MPa"}}}
        plan = {"conditions": [condition], "_sources": self.catalog["sources"]}
        with self.assertRaisesRegex(ensemble.EnsembleError, "publisher source asset/text SHA-256 is unavailable"):
            ensemble._load_conditions(plan, datasets, catalog_hash)

    def test_pin_summary_work_unit_conflict_is_never_converted(self) -> None:
        _, catalog_hash, datasets = ensemble.load_source_package(CATALOG_PATH, DATA)
        source = next(s for s in self.catalog["sources"] if s["id"] == "johnson_popil_2016_author_presentation")
        condition = {"condition_id": "pin-summary-013", "specimen_id": "test-only", "role": "validation",
                     "dataset_id": "johnson_popil_pin_summary", "source_id": source["id"],
                     "source_dataset_sha256": datasets["johnson_popil_pin_summary"]["sha256"],
                     "source_asset_sha256": source["external_original_sha256"],
                     "source_catalog_sha256": catalog_hash,
                     "selector": {"series_id": "summary_work_stiffness_vs_strength", "pin_adhesion_strength_N_mm": "0.13"},
                     "metrics": {"work": {"kind": "peak", "column": "pin_adhesion_work_plot_value", "unit": "N*mm"}}}
        package = dict(self.catalog)
        plan = {"_sources": package["sources"]}
        with self.assertRaisesRegex(ensemble.EnsembleError, "conflicts with the slide-15|unit is unresolved"):
            ensemble._load_conditions({"conditions": [condition], **plan}, datasets, catalog_hash)

    def test_bounded_analogue_failure_locus_uses_explicit_category_set(self) -> None:
        source = {"kind": "category", "unit": "category", "value": "edge-tear"}
        transfer = {"kind": "bounded_analogue", "source_grade": "published-grade",
                    "target_grade": "target-grade", "rationale": "explicit category transfer set",
                    "category_map": {"failure_locus": {"edge-tear": ["edge-tear", "liner-delamination"]}}}
        allowed = ensemble._transfer_categories(source, transfer, "failure_locus")
        self.assertEqual(allowed, {"edge-tear", "liner-delamination"})
        self.assertNotIn("flute-crush", allowed)

    def test_pin_native_force_threshold_window_uses_accepted_relative_opening_only(self) -> None:
        history = {
            "completed_phases": ["approach", "relative_opening", "platen_withdrawal", "released"],
            "units": {"relative_opening_mm": "mm", "load_N": "N"},
            "samples": [
                {"phase": "approach", "accepted": True, "relative_opening_mm": 0.0, "load_N": 0.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 0.0, "load_N": 0.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 2.0, "load_N": 60.0},
                # A rejected attempt with a very large force must not enter the trace.
                {"phase": "relative_opening", "accepted": False, "relative_opening_mm": 2.5, "load_N": 900.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 3.0, "load_N": 100.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 4.0, "load_N": 60.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 6.0, "load_N": 0.0},
                {"phase": "platen_withdrawal", "accepted": True, "relative_opening_mm": 6.0, "load_N": 500.0},
                {"phase": "released", "accepted": True, "relative_opening_mm": 6.0, "load_N": 1000.0},
            ],
        }
        window = {"phase": "relative_opening", "start_load_N": 30.0,
                  "end": "first_post_peak_return_to_start_load",
                  "required_phases": ["approach", "relative_opening", "platen_withdrawal", "released"]}
        peak = ensemble._derive_native_metric(history, {
            "kind": "peak", "unit": "N", "y_unit": "N", "prediction_y": "load_N",
            "prediction_x": "relative_opening_mm", "x_unit": "mm", "native_window": window}, "peak")
        secant = ensemble._derive_native_metric(history, {
            "kind": "secant", "unit": "N/mm", "y_unit": "N", "x_unit": "mm",
            "prediction_y": "load_N", "prediction_x": "relative_opening_mm",
            "native_secant_endpoint": "peak", "native_window": window}, "secant")
        work = ensemble._derive_native_metric(history, {
            "kind": "work", "unit": "N*mm", "y_unit": "N", "x_unit": "mm",
            "prediction_y": "load_N", "prediction_x": "relative_opening_mm",
            "native_window": window}, "work")
        self.assertEqual(peak["value"], 100.0)
        self.assertAlmostEqual(secant["value"], 35.0)
        self.assertAlmostEqual(work["value"], 250.0)

    def test_pin_native_threshold_window_rejects_incomplete_return_and_protocol(self) -> None:
        history = {"completed_phases": ["relative_opening"],
                   "units": {"relative_opening_mm": "mm", "load_N": "N"},
                   "samples": [
                       {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 0.0, "load_N": 0.0},
                       {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 1.0, "load_N": 40.0},
                       {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 2.0, "load_N": 60.0},
                   ]}
        spec = {"kind": "work", "unit": "N*mm", "y_unit": "N", "x_unit": "mm",
                "prediction_y": "load_N", "prediction_x": "relative_opening_mm",
                "native_window": {"phase": "relative_opening", "start_load_N": 30.0,
                                  "end": "first_post_peak_return_to_start_load",
                                  "required_phases": ["approach", "relative_opening", "platen_withdrawal", "released"]}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "phase completion is missing"):
            ensemble._derive_native_metric(history, spec, "work")
        history["completed_phases"] = ["approach", "relative_opening", "platen_withdrawal", "released"]
        with self.assertRaisesRegex(ensemble.EnsembleError, "lacks its required post-peak return"):
            ensemble._derive_native_metric(history, spec, "work")

    def test_published_pin_threshold_window_and_native_metre_input_are_comparable(self) -> None:
        source_rows = ensemble._read_csv_rows(
            ensemble.DEFAULT_DATA / "johnson_popil_2016_pin_force_displacement_digitized.csv")
        source_rows = [row for row in source_rows
                       if row["series_id"] == "low_adhesion_white_glue_lines" and row["defect_id"] == "A"]
        source_window = {"start_load_N": 30.0, "end": "first_post_peak_return_to_start_load"}
        source_spec = {
            "kind": "peak_branch_secant", "unit": "N/mm", "x_column": "displacement_mm",
            "x_unit": "mm", "y_column": "load_N", "y_unit": "N",
            "x_uncertainty_column": "displacement_uncertainty_mm",
            "y_uncertainty_column": "load_uncertainty_N", "source_window": source_window,
        }
        source_secant = ensemble._derive_source_metric(
            source_rows, source_spec, "pin_loading_stiffness", "johnson_popil_pin_force_displacement")
        source_work = ensemble._derive_source_metric(
            source_rows, {**source_spec, "kind": "work", "unit": "N*mm"},
            "pin_plot_window_area", "johnson_popil_pin_force_displacement")

        source_trace = [(11.6, 30), (12.2, 30), (12.6, 32), (12.8, 46),
                        (13.0, 67), (13.1, 78), (13.2, 63), (13.35, 39), (13.5, 30)]
        history = {
            "completed_phases": ["platen_approach", "relative_opening", "platen_withdrawal", "released_hold"],
            "units": {"relativeOpeningM": "m", "instrumentLoadMagnitudeN": "N"},
            "samples": [
                {"phase": "platen_approach", "accepted": True, "acceptedStep": 1,
                 "elapsedS": 0.0, "commandedRelativeSpeedMPerS": 0.0,
                 "relativeOpeningM": 0.0, "instrumentLoadMagnitudeN": 0.0},
                *[{"phase": "relative_opening", "accepted": True, "acceptedStep": i + 2,
                   "elapsedS": i * 0.01, "commandedRelativeSpeedMPerS": 0.0125,
                   "relativeOpeningM": (x - 11.1) * 1e-3, "instrumentLoadMagnitudeN": float(load)}
                  for i, (x, load) in enumerate(source_trace)],
                {"phase": "platen_withdrawal", "accepted": True, "acceptedStep": 11,
                 "elapsedS": 0.1, "commandedRelativeSpeedMPerS": 0.0125,
                 "relativeOpeningM": 0.0024, "instrumentLoadMagnitudeN": 0.0},
                {"phase": "released_hold", "accepted": True, "acceptedStep": 12,
                 "elapsedS": 0.11, "commandedRelativeSpeedMPerS": 0.0,
                 "relativeOpeningM": 0.0024, "instrumentLoadMagnitudeN": 0.0},
            ],
        }
        native_window = {"phase": "relative_opening", "start_load_N": 30.0,
                         "end": "first_post_peak_return_to_start_load",
                         "required_phases": ["platen_approach", "relative_opening",
                                             "platen_withdrawal", "released_hold"]}
        native_spec = {
            "kind": "peak_branch_secant", "prediction_kind": "secant",
            "unit": "N/mm", "x_unit": "mm", "prediction_x_unit": "m",
            "prediction_x_scale": 1000.0, "y_unit": "N",
            "prediction_x": "relativeOpeningM", "prediction_y": "instrumentLoadMagnitudeN",
            "native_secant_endpoint": "peak", "native_window": native_window,
        }
        native_secant = ensemble._derive_native_metric(history, native_spec, "pin_loading_stiffness")
        native_work = ensemble._derive_native_metric(
            history, {**native_spec, "kind": "work", "prediction_kind": "work", "unit": "N*mm"},
            "pin_plot_window_area")
        self.assertAlmostEqual(source_secant["center"], native_secant["value"], places=10)
        self.assertAlmostEqual(source_work["center"], native_work["value"], places=10)
        self.assertAlmostEqual(source_secant["center"], 32.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
