#!/usr/bin/env python3
"""CPU contract tests; temporary certificate fixtures are not native evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


TOOL_PATH = Path(__file__).with_name("cardboard_calibration_ensemble.py")
REPO_ROOT = Path(os.environ.get("NUMI_CARDBOARD_REPO_ROOT",
                               "/Users/home/numi-cardboard-20261006")).resolve()
sys.path.insert(0, str(REPO_ROOT / "matter/tools"))
SPEC = importlib.util.spec_from_file_location("cardboard_calibration_ensemble", TOOL_PATH)
assert SPEC and SPEC.loader
ensemble = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ensemble)
DATA = REPO_ROOT / "matter/examples/cardboard/source-data"
CATALOG_PATH = DATA / "catalog.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bct_capacity_fixture(base: Path, *, complete: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Synthetic artifact graph for parser tests only; never native evidence."""
    base.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "shipping-assembled.matter": b"test-only assembled Matter checkpoint",
        "shipping-assembled.scene": b"test-only rigid scene sidecar",
        "compiled.nmatterpack": b"test-only compiled package",
        "native-executable.bin": b"test-only executable",
        "solver-source.txt": b"test-only source",
    }
    for name, contents in checkpoints.items():
        (base / name).write_bytes(contents)
    curve_path = base / "shipping-compression-nominal-curve.csv"
    curve_path.write_text(
        "sample_index,stroke_m,force_N,contact_samples\n"
        "0,0,0,0\n1,0.00025,40,2\n"
        "2,0.07,100,4\n3,0.14,80,4\n", encoding="utf-8")
    curve_sha = digest(curve_path)
    source_binding = "a" * 64
    checkpoint_sha = digest(base / "shipping-assembled.matter")
    rigid_sha = digest(base / "shipping-assembled.scene")
    mesh_values = [(0.01, 0.03, 0.01), (0.005, 0.01, 0.005)]
    time_values = [(0.015, 0.04, 0.015), (0.008, 0.02, 0.01)]

    def report(axis: str, levels: list[Any], values: list[tuple[float, float, float]]) -> dict[str, Any]:
        fields = ("relative_peak_force_difference", "normalized_force_curve_rms",
                  "relative_work_difference")
        pairs = []
        for values_at_pair in values:
            pairs.append({
                "status": "descriptive_metrics_available_with_unresolved_observables",
                "metrics": {"compression_force_work": {
                    "status": "available", "comparison_is_complete_endpoint": True,
                    **dict(zip(fields, values_at_pair)),
                }},
            })
        return {"schema": "numi.cardboard.stage-resolution-analysis.v1",
                "axis": axis, "status": "descriptive_three_level_decrease_only",
                "convergence_claim": False, "axis_levels": levels,
                "pair_reports": pairs}

    mesh_report = report("mesh_nx", [[1], [2], [3]], mesh_values)
    time_report = report("time_step", [0.01, 0.005, 0.0025], time_values)
    write_json(base / "mesh-study.json", mesh_report)
    write_json(base / "time-study.json", time_report)
    mesh_sha = digest(base / "mesh-study.json")
    time_sha = digest(base / "time-study.json")
    capacity_fields = {
        "schema": "numi.cardboard.shipping-capacity-qualification.v2",
        "status": "qualified_for_numerical_capacity_only" if complete else "not_qualified",
        "board_recipe_id": "literature2009",
        "source_physics_fingerprint": "111", "device_program_fingerprint": "222",
        "compiled_world_fingerprint": "222", "source_binding_sha256": source_binding,
        "assembled_matter_checkpoint_sha256": checkpoint_sha,
        "assembled_rigid_sidecar_sha256": rigid_sha,
        "nominal_curve_sha256": curve_sha, "nominal_peak_sample_index": "2",
        "nominal_accepted_sample_count": "4", "nominal_terminal_stroke_fraction": "0.7",
        "nominal_full_stroke_completed": "true" if complete else "false",
        "nominal_rate_sensitivity_passed": "true",
        "mesh_status": "qualified", "mesh_report_file": "mesh-study.json",
        "mesh_report_sha256": mesh_sha, "mesh_level_count": "3",
        "mesh_all_cases_complete": "true", "mesh_finest_pair_within_tolerance": "true",
        "mesh_finest_peak_difference": "0.005", "mesh_finest_curve_rms": "0.01",
        "mesh_finest_work_difference": "0.005", "mesh_coarse_middle_peak_difference": "0.01",
        "mesh_coarse_middle_curve_rms": "0.03", "mesh_coarse_middle_work_difference": "0.01",
        "time_status": "qualified", "time_report_file": "time-study.json",
        "time_report_sha256": time_sha, "time_level_count": "3",
        "time_all_cases_complete": "true", "time_finest_pair_within_tolerance": "true",
        "time_finest_peak_difference": "0.008", "time_finest_curve_rms": "0.02",
        "time_finest_work_difference": "0.01", "time_coarse_middle_peak_difference": "0.015",
        "time_coarse_middle_curve_rms": "0.04", "time_coarse_middle_work_difference": "0.015",
    }
    capacity_path = base / "shipping-capacity-qualification.properties"
    capacity_path.write_text("# synthetic parser fixture\n" + "".join(
        f"{key}={capacity_fields[key]}\n" for key in sorted(capacity_fields)), encoding="utf-8")
    capacity_sha = digest(capacity_path)
    branch_path = base / "shipping-compression.csv"
    branch_fields = ["phase", "accepted", "stroke_m", "macro_observation_duration_s",
                     "control_step_count", "control_step_dt_min_s", "control_step_dt_max_s",
                     "platen_reaction_samples", "platen_mean_force_y_N"]
    branch_rows = []
    for prefix, rate in (("nominal_12_5_mm_min", 12.5e-3 / 60.0),
                         ("half_speed_6_25_mm_min", 6.25e-3 / 60.0)):
        branch_rows.extend([
            {"phase": f"{prefix}_contact_origin", "accepted": "1", "stroke_m": "0",
             "macro_observation_duration_s": "0", "control_step_count": "0",
             "control_step_dt_min_s": "0", "control_step_dt_max_s": "0",
             "platen_reaction_samples": "0", "platen_mean_force_y_N": "0"},
            {"phase": f"{prefix}_contact_approach", "accepted": "1", "stroke_m": "0.00025",
             "macro_observation_duration_s": str(0.00025 / rate),
             "control_step_count": str(round((0.00025 / rate) / 0.0005)),
             "control_step_dt_min_s": "0.0005", "control_step_dt_max_s": "0.0005",
             "platen_reaction_samples": "2", "platen_mean_force_y_N": "40"},
        ])
        previous = 0.00025
        for stroke, force in ((0.07, 100.0), (0.14, 80.0)):
            duration = (stroke - previous) / rate
            branch_rows.append({
                "phase": f"{prefix}_slow_compression", "accepted": "1",
                "stroke_m": str(stroke), "macro_observation_duration_s": str(duration),
                "control_step_count": str(round(duration / 0.0005)),
                "control_step_dt_min_s": "0.0005", "control_step_dt_max_s": "0.0005",
                "platen_reaction_samples": "4", "platen_mean_force_y_N": str(force),
            })
            previous = stroke
    with branch_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=branch_fields)
        writer.writeheader(); writer.writerows(branch_rows)
    branch_sha = digest(branch_path)
    binding = {
        "schema": "numi.cardboard.native-bct-capacity-binding.v1",
        "capacity_qualification_sha256": capacity_sha,
        "board_recipe_id": "literature2009", "source_physics_fingerprint": 111,
        "device_program_fingerprint": 222, "compiled_world_fingerprint": 222,
        "source_binding_sha256": source_binding,
        "assembled_matter_checkpoint_sha256": checkpoint_sha,
        "assembled_rigid_sidecar_sha256": rigid_sha,
        "nominal_curve_sha256": curve_sha, "nominal_peak_sample_index": 2,
        "nominal_accepted_sample_count": 4,
        "nominal_terminal_stroke_fraction": 0.7,
        "nominal_full_stroke_completed": True, "nominal_rate_sensitivity_passed": True,
        "inside_height_m": 0.2, "mesh_report_sha256": mesh_sha, "time_report_sha256": time_sha,
        "compression_branch_history_sha256": branch_sha,
    }
    artifact_files = {
        "shipping_capacity_qualification": capacity_path,
        "shipping_compression_nominal_curve": curve_path,
        "shipping_capacity_mesh_report": base / "mesh-study.json",
        "shipping_capacity_time_report": base / "time-study.json",
        "shipping_compression_branch_history": branch_path,
        "shipping_assembled_matter_checkpoint": base / "shipping-assembled.matter",
        "shipping_assembled_rigid_sidecar": base / "shipping-assembled.scene",
        "compiled_matter_package": base / "compiled.nmatterpack",
        "native_executable": base / "native-executable.bin",
        "solver_source": base / "solver-source.txt",
    }
    native_result = {
        "source_physics_fingerprint": 111,
        "device_program_fingerprint": 222,
        "compiled_world_fingerprint": 222,
        "shipping_capacity_qualification_file_sha256": capacity_sha,
        "nominal_compression_curve_sha256": curve_sha,
        "shipping_source_material_binding_sha256": source_binding,
        "assembled_checkpoint_sha256_for_capacity_binding": checkpoint_sha,
        "assembled_rigid_sidecar_sha256_for_capacity_binding": rigid_sha,
        "native_slow_compression_completed": True,
        "native_half_speed_compression_completed": True,
        "native_compression_rate_sensitivity_gate_passed": True,
        "native_compression_numerical_capacity_qualification_passed": True,
        "compression_stroke_fraction_of_inside_height": 0.7,
        "shipping_capacity_binding": binding,
    }
    write_json(base / "result.json", native_result)
    artifact_files["native_result"] = base / "result.json"
    artifact_hashes = {key: {"path": path.name, "sha256": digest(path)}
                      for key, path in artifact_files.items()}
    history = {
        "shipping_capacity_binding": binding,
        "samples": [
            {"sample_index": i, "phase": "shipping_compression_nominal", "accepted": True,
             "stroke_m": stroke, "force_N": force}
            for i, (stroke, force) in enumerate(((0.0, 0.0), (0.00025, 40.0),
                                                  (0.07, 100.0), (0.14, 80.0)))
        ],
    }
    manifest = {
        "preset": "literature2009", "source_physics_fingerprint": 111,
        "device_program_fingerprint": 222, "compiled_world_fingerprint": 222,
        "shipping_source_material_binding_sha256": source_binding,
        "assembled_checkpoint_sha256_for_capacity_binding": checkpoint_sha,
        "assembled_rigid_sidecar_sha256_for_capacity_binding": rigid_sha,
        "compiled_matter_package_sha256": digest(base / "compiled.nmatterpack"),
        "native_executable_sha256": digest(base / "native-executable.bin"),
        "solver_source_sha256": digest(base / "solver-source.txt"),
        "shipping_capacity_binding": binding,
    }
    receipt = {"shipping_capacity_binding": binding}
    cert = {"artifact_hashes": artifact_hashes, "shipping_capacity_binding": binding}
    return history, cert, manifest, receipt, capacity_path


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
            "native_protocol_id": "pin_adhesion_50x160_six_pin_v1",
            "parameter_family": "mode_I_cohesive",
            "selector": self.selector,
            "transfer": {"kind": "bounded_analogue", "source_grade": "presentation_board_205_122",
                         "target_grade": "literature2009_205_126_kraft_NSSC",
                         "rationale": "test-only declared transfer interval; not a fitted material assumption",
                         "metric_scale_bounds": {"peak_load_N": [0.8, 1.2],
                                                  "trace_secant_stiffness_N_per_mm": [0.8, 1.2],
                                                  "trace_window_work_N_mm": [0.8, 1.2]}},
            "metrics": {
                "peak_load_N": {"kind": "peak", "column": "load_N", "unit": "N", "y_unit": "N",
                                "x_column": "displacement_mm", "x_unit": "mm",
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
        for metric in condition["metrics"].values():
            if metric.get("kind") == "unavailable":
                continue
            metric["source_window"] = {
                "start_load_N": 30.0,
                "end": "first_post_peak_return_to_start_load",
            }
            metric["native_window"] = {
                "phase": "relative_opening",
                "start_load_N": 30.0,
                "end": "first_post_peak_return_to_start_load",
                "required_phases": ["platen_approach", "relative_opening",
                                    "platen_withdrawal", "released_hold"],
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
                samples = [{"step": 0, "phase": "platen_approach", "accepted": True,
                            "displacement_mm": float(rows[0]["displacement_mm"]) - 0.2,
                            "load_N": 0.0},
                           {"step": 1, "phase": "relative_opening", "accepted": True,
                            "displacement_mm": float(rows[0]["displacement_mm"]) - 0.1,
                            "load_N": 0.0}]
                samples.extend({"step": i + 2, "phase": "relative_opening", "accepted": True,
                                "displacement_mm": float(row["displacement_mm"]),
                                "load_N": float(row["load_N"]) * (1.1 if candidate["candidate_id"] == "grid-002" else 1.0)}
                               for i, row in enumerate(rows))
                samples.append({"step": len(samples), "phase": "relative_opening", "accepted": True,
                                "displacement_mm": float(rows[-1]["displacement_mm"]) + 0.1,
                                "load_N": 0.0})
                samples.extend([
                    {"step": len(samples), "phase": "platen_withdrawal", "accepted": True,
                     "displacement_mm": float(rows[-1]["displacement_mm"]) + 0.2,
                     "load_N": 0.0},
                    {"step": len(samples) + 1, "phase": "released_hold", "accepted": True,
                     "displacement_mm": float(rows[-1]["displacement_mm"]) + 0.2,
                     "load_N": 0.0},
                ])
                history = {"schema": "numi.cardboard.native_forward_history.v1",
                           "candidate_id": candidate["candidate_id"], "condition_id": cond["condition_id"],
                           "candidate_sha256": candidate["candidate_sha256"],
                           "specimen_id": cond["specimen_id"],
                           "source_data_catalog_sha256": self.catalog_hash,
                           "source_dataset_sha256": self.dataset["sha256"],
                           "source_asset_sha256": self.source["external_original_sha256"],
                           "native_protocol_id": cond["native_protocol_id"],
                           "completed_phases": ["platen_approach", "relative_opening",
                                                "platen_withdrawal", "released_hold"],
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
                            "native_protocol_id": cond["native_protocol_id"],
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
                    "native_protocol_id": cond["native_protocol_id"],
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
                        "native_protocol_id": cond["native_protocol_id"],
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
                    "native_protocol_id": cond["native_protocol_id"],
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
            "completed_phases": ["platen_approach", "relative_opening",
                                 "platen_withdrawal", "released_hold"],
            "units": {"relative_opening_mm": "mm", "load_N": "N"},
            "samples": [
                {"phase": "platen_approach", "accepted": True, "relative_opening_mm": 0.0, "load_N": 0.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 0.0, "load_N": 0.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 2.0, "load_N": 60.0},
                # A rejected attempt with a very large force must not enter the trace.
                {"phase": "relative_opening", "accepted": False, "relative_opening_mm": 2.5, "load_N": 900.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 3.0, "load_N": 100.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 4.0, "load_N": 60.0},
                {"phase": "relative_opening", "accepted": True, "relative_opening_mm": 6.0, "load_N": 0.0},
                {"phase": "platen_withdrawal", "accepted": True, "relative_opening_mm": 6.0, "load_N": 500.0},
                {"phase": "released_hold", "accepted": True, "relative_opening_mm": 6.0, "load_N": 1000.0},
            ],
        }
        window = {"phase": "relative_opening", "start_load_N": 30.0,
                  "end": "first_post_peak_return_to_start_load",
                  "required_phases": ["platen_approach", "relative_opening",
                                      "platen_withdrawal", "released_hold"]}
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
                                  "required_phases": ["platen_approach", "relative_opening",
                                                      "platen_withdrawal", "released_hold"]}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "phase completion is missing"):
            ensemble._derive_native_metric(history, spec, "work")

    def test_pin_phase_names_cannot_claim_unobserved_withdrawal_or_release(self) -> None:
        history = {
            "completed_phases": ["platen_approach", "relative_opening",
                                 "platen_withdrawal", "released_hold"],
            "units": {"relativeOpeningM": "m", "instrumentLoadMagnitudeN": "N"},
            "samples": [
                {"phase": "relative_opening", "accepted": True,
                 "relativeOpeningM": 0.0, "instrumentLoadMagnitudeN": 0.0},
                {"phase": "relative_opening", "accepted": True,
                 "relativeOpeningM": 0.001, "instrumentLoadMagnitudeN": 60.0},
            ],
        }
        spec = {
            "kind": "peak", "unit": "N", "y_unit": "N", "x_unit": "m",
            "prediction_y": "instrumentLoadMagnitudeN",
            "prediction_x": "relativeOpeningM",
            "native_window": {
                "phase": "relative_opening", "start_load_N": 30.0,
                "end": "first_post_peak_return_to_start_load",
                "required_phases": ["platen_approach", "relative_opening",
                                    "platen_withdrawal", "released_hold"],
            },
        }
        with self.assertRaisesRegex(ensemble.EnsembleError,
                                    "required protocol phases have no accepted samples"):
            ensemble._derive_native_metric(history, spec, "pin_peak")
        history["samples"].extend([
            {"phase": "platen_approach", "accepted": True,
             "relativeOpeningM": 0.0, "instrumentLoadMagnitudeN": 0.0},
            {"phase": "platen_withdrawal", "accepted": True,
             "relativeOpeningM": 0.001, "instrumentLoadMagnitudeN": 0.0},
            {"phase": "released_hold", "accepted": True,
             "relativeOpeningM": 0.001, "instrumentLoadMagnitudeN": 0.0},
        ])
        with self.assertRaisesRegex(ensemble.EnsembleError, "lacks its required post-peak return"):
            ensemble._derive_native_metric(history, spec, "work")

    def test_published_pin_threshold_window_and_native_metre_input_are_comparable(self) -> None:
        source_rows = ensemble._read_csv_rows(
            DATA / "johnson_popil_2016_pin_force_displacement_digitized.csv")
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
        self.assertEqual(source_secant["source_scope"],
                         "visible_30N_threshold_window_not_full_specimen_curve")

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

    def test_same_grade_cannot_be_claimed_for_published_pin_analogue(self) -> None:
        condition = {"condition_id": "pin-same-grade", "selector": self.selector,
                     "transfer": {"kind": "same_grade",
                                  "source_grade": "pin-board", "target_grade": "pin-board"}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "same_grade is unavailable"):
            ensemble._validate_source_transfer_binding(
                condition, self.dataset, self.observed_rows, "pin-board")

    def test_bc780_cannot_be_relabelled_as_same_grade_by_matching_strings(self) -> None:
        dataset = self.datasets["mrowczynski_2021_box_bct"]
        with Path(dataset["resolved_path"]).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row["grade"] == "BC-780" and row["box_condition"] == "without_offset"]
        condition = {"condition_id": "bct-false-same-grade",
                     "selector": {"grade": "BC-780", "box_condition": "without_offset"},
                     "transfer": {"kind": "same_grade", "source_grade": "BC-780",
                                  "target_grade": "BC-780"}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "same_grade is unavailable"):
            ensemble._validate_source_transfer_binding(condition, dataset, rows, "BC-780")

    def test_bct_bounded_analogue_grade_is_bound_to_selected_published_row(self) -> None:
        dataset = self.datasets["mrowczynski_2021_box_bct"]
        with Path(dataset["resolved_path"]).open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row["grade"] == "BC-780" and row["box_condition"] == "without_offset"]
        condition = {"condition_id": "bct-wrong-source-grade",
                     "selector": {"grade": "BC-780", "box_condition": "without_offset"},
                     "transfer": {"kind": "bounded_analogue", "source_grade": "EB-965",
                                  "target_grade": "literature2009", "rationale": "test-only"}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "exactly match the selected published grade"):
            ensemble._validate_source_transfer_binding(condition, dataset, rows, "literature2009")

    def test_bct_complete_capacity_artifact_binds_curve_build_checkpoint_and_refinements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, _ = make_bct_capacity_fixture(base)
            ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_bct_expected_steps_or_protocol_id_cannot_replace_capacity_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, _ = make_bct_capacity_fixture(base)
            cert["expected_steps"] = 4
            cert["accepted_steps"] = 4
            cert["native_protocol_id"] = "shipping_bct_without_offset_first_local_peak_v1"
            cert["artifact_hashes"].pop("shipping_capacity_qualification")
            with self.assertRaisesRegex(ensemble.EnsembleError, "requires the shipping_capacity_qualification artifact"):
                ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_bct_half_speed_flag_does_not_replace_accepted_half_speed_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, _ = make_bct_capacity_fixture(base)
            branch_path = base / cert["artifact_hashes"]["shipping_compression_branch_history"]["path"]
            fields, rows = ensemble._read_csv_rows(branch_path), None
            # Keep the nominal branch and half-speed contact, but remove its
            # measured slow-compression rows while leaving all declaration flags true.
            with branch_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames
                rows = list(reader)
            assert names is not None and rows is not None
            rows = [row for row in rows if row["phase"] != "half_speed_6_25_mm_min_slow_compression"]
            with branch_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader(); writer.writerows(rows)
            branch_sha = digest(branch_path)
            binding = dict(history["shipping_capacity_binding"])
            binding["compression_branch_history_sha256"] = branch_sha
            for obj in (history, cert, manifest, receipt):
                obj["shipping_capacity_binding"] = binding
            cert["artifact_hashes"]["shipping_compression_branch_history"]["sha256"] = branch_sha
            result_path = base / cert["artifact_hashes"]["native_result"]["path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["shipping_capacity_binding"] = binding
            write_json(result_path, result)
            cert["artifact_hashes"]["native_result"]["sha256"] = digest(result_path)
            with self.assertRaisesRegex(ensemble.EnsembleError, "complete loading rows"):
                ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_bct_capacity_source_build_fingerprint_must_match_executed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, _ = make_bct_capacity_fixture(base)
            manifest["source_physics_fingerprint"] = 999
            with self.assertRaisesRegex(ensemble.EnsembleError, "source/build fingerprint mismatch"):
                ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_bct_partial_nominal_peak_is_not_admitted_by_capacity_properties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, capacity_path = make_bct_capacity_fixture(base)
            values = ensemble._capacity_properties(capacity_path)
            values["nominal_full_stroke_completed"] = "false"
            capacity_path.write_text("".join(f"{key}={values[key]}\n" for key in sorted(values)), encoding="utf-8")
            capacity_hash = digest(capacity_path)
            binding = dict(history["shipping_capacity_binding"])
            binding["capacity_qualification_sha256"] = capacity_hash
            history["shipping_capacity_binding"] = binding
            cert["shipping_capacity_binding"] = binding
            manifest["shipping_capacity_binding"] = binding
            receipt["shipping_capacity_binding"] = binding
            cert["artifact_hashes"]["shipping_capacity_qualification"]["sha256"] = capacity_hash
            # The nominal run result binds the actual qualification file too.
            result_path = base / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["shipping_capacity_qualification_file_sha256"] = capacity_hash
            write_json(result_path, result)
            cert["artifact_hashes"]["native_result"]["sha256"] = digest(result_path)
            with self.assertRaisesRegex(ensemble.EnsembleError, "lacks full 70-percent stroke"):
                ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_bct_curve_endpoint_or_sample_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            history, cert, manifest, receipt, _ = make_bct_capacity_fixture(base)
            history["samples"].pop()
            with self.assertRaisesRegex(ensemble.EnsembleError, "complete capacity curve sample-for-sample"):
                ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, base)

    def test_censored_pin_high_curve_cannot_supply_a_full_peak(self) -> None:
        condition = {"condition_id": "pin-high-censored",
                     "selector": {"series_id": "applicator_roll_too_fast", "defect_id": "F"}}
        with self.assertRaisesRegex(ensemble.EnsembleError, "censored and cannot supply a global peak"):
            ensemble._validate_pin_metric_window(
                condition, {"kind": "peak", "column": "load_N"}, "peak_load")

    def test_pin_peak_without_visible_window_is_rejected(self) -> None:
        condition = {"condition_id": "pin-low-unwindowed", "selector": self.selector}
        with self.assertRaisesRegex(ensemble.EnsembleError, "must use the published visible 30 N"):
            ensemble._validate_pin_metric_window(
                condition, {"kind": "peak", "column": "load_N"}, "peak_load")

    def test_scored_bend_moment_per_width_cannot_be_relabelled_as_force(self) -> None:
        rows = ensemble._read_csv_rows(
            DATA / "nagasawa_2012_fig9_scored_bend_envelope.csv")
        rows = [row for row in rows
                if row["condition_id"] == "nagasawa_2012_fig9_cP3p0_cC0p7_V2p92"]
        with self.assertRaisesRegex(ensemble.EnsembleError, "conflicts with source column"):
            ensemble._derive_source_metric(
                rows, {"kind": "peak", "column": "moment_envelope_max_Nm_per_m",
                       "unit": "N", "y_unit": "N"},
                "scored_bend_peak", "nagasawa_2012_fig9_scored_bend_envelope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
