#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PRODUCER_PATH = TOOLS / "bct_capacity_history_producer.py"
ENSEMBLE_PATH = TOOLS / "cardboard_calibration_ensemble.py"
REPO = TOOLS.parents[1]
CATALOG = REPO / "matter/examples/cardboard/source-data/catalog.json"
DATA_DIR = CATALOG.parent
sys.path.insert(0, str(TOOLS))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer = load_module(PRODUCER_PATH, "bct_capacity_history_producer")
ensemble = load_module(ENSEMBLE_PATH, "cardboard_calibration_ensemble_for_producer_test")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def write_capacity_reports(directory: Path) -> tuple[Path, Path, str, str]:
    def report(axis: str, levels: list[float]):
        return {
            "schema": "numi.cardboard.stage-resolution-analysis.v1",
            "axis": axis,
            "status": "descriptive_three_level_decrease_only",
            "convergence_claim": False,
            "axis_levels": levels,
            "pair_reports": [
                {"status": "descriptive_metrics_available_with_unresolved_observables",
                 "metrics": {"compression_force_work": {
                     "status": "available", "comparison_is_complete_endpoint": True,
                     "relative_peak_force_difference": 0.03,
                     "normalized_force_curve_rms": 0.07,
                     "relative_work_difference": 0.03}}},
                {"status": "descriptive_metrics_available_with_unresolved_observables",
                 "metrics": {"compression_force_work": {
                     "status": "available", "comparison_is_complete_endpoint": True,
                     "relative_peak_force_difference": 0.01,
                     "normalized_force_curve_rms": 0.03,
                     "relative_work_difference": 0.01}}},
            ],
        }
    mesh_path = directory / "mesh-study.json"
    time_path = directory / "time-study.json"
    write_json(mesh_path, report("mesh_nx", [2, 4, 8]))
    write_json(time_path, report("time_step", [0.002, 0.001, 0.0005]))
    return mesh_path, time_path, digest(mesh_path), digest(time_path)


def make_run(base: Path):
    run = base / "native-run"
    out = run / "output"
    out.mkdir(parents=True)
    catalog, catalog_sha, datasets = ensemble.load_source_package(CATALOG, DATA_DIR)
    dataset = datasets["mrowczynski_2021_box_bct"]
    source = next(row for row in catalog["sources"] if row["id"] == "mrowczynski_2021")
    candidate = {
        "candidate_id": "synthetic-contract-only",
        "role": "fit_candidate",
        "origin": "user_supplied",
        "parameters": {"score_depth_ratio": {
            "value": 0.25, "unit": "dimensionless", "family": "tooling", "fit": True}},
    }
    candidate["candidate_sha256"] = producer.canonical_sha256(candidate)
    condition = {
        "condition_id": "BC780-without-offset-test-only",
        "specimen_id": "synthetic-contract-only-BC780",
        "dataset_id": "mrowczynski_2021_box_bct",
        "source_id": "mrowczynski_2021",
        "selector": {"grade": "BC-780", "box_condition": "without_offset"},
        "source_catalog_sha256": catalog_sha,
        "source_dataset_sha256": dataset["sha256"],
        "source_asset_sha256": source["external_original_sha256"],
        "native_protocol_id": producer.PROTOCOL_ID,
        "role": "validation",
        "metrics": {"capacity": {"kind": "peak", "column": "BCT_N", "unit": "N", "y_unit": "N"}},
        "transfer": {"kind": "bounded_analogue", "source_grade": "BC-780",
                     "target_grade": "literature2009_205_126_kraft_NSSC",
                     "rationale": "synthetic contract test only",
                     "metric_scale_bounds": {"capacity": [1.0, 1.0]}},
    }
    plan = {
        "schema": "numi.cardboard.calibration_ensemble.v1",
        "source_data_catalog_sha256": catalog_sha,
        "target_recipe_id": "literature2009_cross_source_recipe",
        "target_grade_id": "literature2009_205_126_kraft_NSSC",
        "conditions": [condition], "candidates": [candidate], "native_histories": [],
    }
    plan_path = base / "plan.json"
    write_json(plan_path, plan)
    bindings_path = base / "parameter-bindings.json"
    write_json(bindings_path, {"score_depth_ratio": {
        "kind": "manifest_json_pointer",
        "pointer": "/shipping_score_command/selected_depth_ratio_of_caliper",
        "argv_flag": "--shipping-score-depth-ratio"}})
    case_path = base / "case-binding.json"
    producer.prepare_case_binding(plan_path, candidate["candidate_id"],
                                  condition["condition_id"],
                                  json.loads(bindings_path.read_text()), case_path)

    exe = base / "native-executable"
    exe.write_bytes(b"synthetic test only; never native evidence")
    qualified, mesh_sha, time_sha = base / "shipping-capacity-qualification.v2", None, None
    mesh_path, time_path, mesh_sha, time_sha = write_capacity_reports(base)
    checkpoint = out / "checkpoints" / "assembled.matter"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic checkpoint test only")
    rigid = out / "checkpoints" / "assembled.scene"
    rigid.write_bytes(b"synthetic rigid sidecar test only")
    source_binding = hashlib.sha256(b"test source binding").hexdigest()
    checkpoint_sha, rigid_sha = digest(checkpoint), digest(rigid)
    write_json(out / "shipping-rigid-accepted-checkpoint.json", {
        "schema": "numi.cardboard.shipping-rigid-checkpoint.v1",
        "fem_checkpoint_file": checkpoint.name,
        "fem_checkpoint_sha256": checkpoint_sha,
        "paired_scene_sidecar_file": rigid.name,
        "paired_scene_sidecar_sha256": rigid_sha,
        "scene_bodies": [{"index": 0, "synthetic_test_only": True}],
    })
    curve_path = out / "shipping-compression-nominal-curve.csv"
    curve_rows = [(0.0, 0.0, 0), (0.00025, 40.0, 4), (0.07, 100.0, 4), (0.14, 80.0, 4)]
    with curve_path.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["sample_index", "stroke_m", "force_N", "contact_samples"])
        writer.writerows([[i, x, y, n] for i, (x, y, n) in enumerate(curve_rows)])
    curve_sha = digest(curve_path)
    branch_path = out / "shipping-compression.csv"
    fields = ["phase", "accepted", "stroke_m", "macro_observation_duration_s",
              "control_step_count", "control_step_dt_min_s", "control_step_dt_max_s",
              "platen_mean_force_y_N", "platen_reaction_samples"]
    branch_rows = []
    for prefix, rate, crossing_duration in (("nominal_12_5_mm_min", 12.5e-3 / 60, 1.2),
                                              ("half_speed_6_25_mm_min", 6.25e-3 / 60, 2.4)):
        branch_rows.extend([
            {"phase": f"{prefix}_contact_origin", "accepted": "1", "stroke_m": "0",
             "macro_observation_duration_s": "0", "control_step_count": "0",
             "control_step_dt_min_s": "0", "control_step_dt_max_s": "0",
             "platen_mean_force_y_N": "0", "platen_reaction_samples": "0"},
            {"phase": f"{prefix}_contact_approach", "accepted": "1", "stroke_m": "0.00025",
             "macro_observation_duration_s": str(crossing_duration),
             "control_step_count": str(round(crossing_duration / 0.0005)),
             "control_step_dt_min_s": "0.0005", "control_step_dt_max_s": "0.0005",
             "platen_mean_force_y_N": "40", "platen_reaction_samples": "4"},
        ])
        prev = 0.00025
        for stroke, force in ((0.07, 100.0), (0.14, 80.0)):
            duration = (stroke - prev) / rate
            branch_rows.append({"phase": f"{prefix}_slow_compression", "accepted": "1",
                "stroke_m": str(stroke), "macro_observation_duration_s": str(duration),
                "control_step_count": str(round(duration / 0.0005)),
                "control_step_dt_min_s": "0.0005", "control_step_dt_max_s": "0.0005",
                "platen_mean_force_y_N": str(force), "platen_reaction_samples": "4"})
            prev = stroke
    with branch_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(branch_rows)
    branch_sha = digest(branch_path)
    props = {
        "schema": "numi.cardboard.shipping-capacity-qualification.v2",
        "status": "qualified_for_numerical_capacity_only",
        "board_recipe_id": "literature2009",
        "source_physics_fingerprint": "111", "device_program_fingerprint": "222",
        "compiled_world_fingerprint": "222", "source_binding_sha256": source_binding,
        "assembled_matter_checkpoint_sha256": checkpoint_sha,
        "assembled_rigid_sidecar_sha256": rigid_sha, "nominal_curve_sha256": curve_sha,
        "nominal_peak_sample_index": "2", "nominal_accepted_sample_count": "4",
        "nominal_terminal_stroke_fraction": "0.7", "nominal_full_stroke_completed": "true",
        "nominal_rate_sensitivity_passed": "true", "mesh_status": "qualified",
        "mesh_report_file": mesh_path.name, "mesh_report_sha256": mesh_sha,
        "mesh_level_count": "3", "mesh_all_cases_complete": "true",
        "mesh_finest_pair_within_tolerance": "true", "mesh_finest_peak_difference": "0.01",
        "mesh_finest_curve_rms": "0.03", "mesh_finest_work_difference": "0.01",
        "mesh_coarse_middle_peak_difference": "0.03", "mesh_coarse_middle_curve_rms": "0.07",
        "mesh_coarse_middle_work_difference": "0.03", "time_status": "qualified",
        "time_report_file": time_path.name, "time_report_sha256": time_sha,
        "time_level_count": "3", "time_all_cases_complete": "true",
        "time_finest_pair_within_tolerance": "true", "time_finest_peak_difference": "0.01",
        "time_finest_curve_rms": "0.03", "time_finest_work_difference": "0.01",
        "time_coarse_middle_peak_difference": "0.03", "time_coarse_middle_curve_rms": "0.07",
        "time_coarse_middle_work_difference": "0.03",
    }
    qualified.write_text("".join(f"{key}={value}\n" for key, value in props.items()))
    binding = {
        "schema": producer.BINDING_SCHEMA,
        "capacity_qualification_sha256": digest(qualified),
        "board_recipe_id": "literature2009",
        "source_physics_fingerprint": 111, "device_program_fingerprint": 222,
        "compiled_world_fingerprint": 222, "source_binding_sha256": source_binding,
        "assembled_matter_checkpoint_sha256": checkpoint_sha,
        "assembled_rigid_sidecar_sha256": rigid_sha, "nominal_curve_sha256": curve_sha,
        "nominal_peak_sample_index": 2, "nominal_accepted_sample_count": 4,
        "nominal_terminal_stroke_fraction": 0.7, "nominal_full_stroke_completed": True,
        "nominal_rate_sensitivity_passed": True, "inside_height_m": 0.2,
        "mesh_report_sha256": mesh_sha, "time_report_sha256": time_sha,
        "compression_branch_history_sha256": branch_sha,
    }
    result = {
        "status": "completed_native_compression_only_unqualified",
        "native_steps_attempted": 12, "native_steps_accepted": 12,
        "source_physics_fingerprint": 111, "device_program_fingerprint": 222,
        "compiled_world_fingerprint": 222,
        "shipping_capacity_qualification_file": str(qualified),
        "shipping_capacity_qualification_file_sha256": digest(qualified),
        "shipping_source_material_binding_sha256": source_binding,
        "assembled_checkpoint_sha256_for_capacity_binding": checkpoint_sha,
        "assembled_rigid_sidecar_sha256_for_capacity_binding": rigid_sha,
        "nominal_compression_curve_sha256": curve_sha,
        "nominal_compression_curve_file": curve_path.name,
        "native_slow_compression_completed": True,
        "native_half_speed_compression_completed": True,
        "native_compression_rate_sensitivity_gate_passed": True,
        "native_compression_numerical_capacity_qualification_passed": True,
        "compression_stroke_fraction_of_inside_height": 0.7,
        "shipping_capacity_binding": binding,
    }
    write_json(out / "box-native-result.json", result)
    native_manifest = {
        "schema": "numi.cardboard.explicit-strip.v1", "preset": "literature2009",
        "compiled_world_fingerprint": 222,
        "shipping_score_command": {"selected_depth_ratio_of_caliper": 0.25},
    }
    write_json(out / "manifest.json", native_manifest)
    (out / "compiled.nmatterpack").write_bytes(b"synthetic compiled package test only")
    prediction = {
        "schema": "numi.cardboard.native-prediction.v1",
        "native_protocol_id": condition["native_protocol_id"],
        "calibration_case_binding": {
            "schema": "numi.cardboard.source-bound-calibration-execution.v1",
            "plan_sha256": digest(plan_path),
            "case_binding_sha256": digest(case_path),
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "parameter_bindings_sha256": producer.canonical_sha256(json.loads(case_path.read_text())["parameter_bindings"]),
            "capacity_qualification_sha256": digest(qualified),
            "mesh_report_sha256": mesh_sha,
            "time_report_sha256": time_sha,
            "external_input_sha256": {
                str(path.resolve()): digest(path)
                for path in (plan_path, case_path, qualified, mesh_path, time_path)
            },
            "pre_run_parameter_checks": [{
                "name": "score_depth_ratio", "candidate_value": 0.25,
                "manifest_pointer": "/shipping_score_command/selected_depth_ratio_of_caliper",
                "argv_flag": "--shipping-score-depth-ratio", "kind": "argv_scalar",
                "argv_value": 0.25}],
        },
        "external_inputs": [str(plan_path), str(case_path), str(qualified), str(mesh_path), str(time_path)],
        "inputs_sha256": {
            f"external/{plan_path.name}": digest(plan_path),
            f"external/{case_path.name}": digest(case_path),
            f"external/{qualified.name}": digest(qualified),
            f"external/{mesh_path.name}": digest(mesh_path),
            f"external/{time_path.name}": digest(time_path),
            exe.name: digest(exe),
        },
        "argv": [str(exe), "--shipping-box", "--shipping-score-depth-ratio", "0.25",
                 "--shipping-capacity-qualification",
                 str(qualified), "--output", str(out)],
        "source_snapshot": str(base), "runtime_dir": str(base),
        "source_binding_limitation": "synthetic contract fixture only",
    }
    external_copy_root = run / "inputs" / "external"
    external_copy_root.mkdir(parents=True)
    for external_path in (plan_path, case_path, qualified, mesh_path, time_path):
        shutil.copyfile(external_path, external_copy_root / external_path.name)
    write_json(run / "prediction.json", prediction)
    write_json(run / "receipt.json", {
        "exit_code": 0, "timed_out": False, "wall_seconds": 1.0,
        "source_inputs_unchanged": True,
        "native_protocol_id": condition["native_protocol_id"],
        "calibration_plan_sha256": digest(plan_path),
        "calibration_case_binding_sha256": digest(case_path),
        "calibration_candidate_id": candidate["candidate_id"],
        "calibration_candidate_sha256": candidate["candidate_sha256"],
        "calibration_condition_id": condition["condition_id"],
        "calibration_parameter_validation": {
            "status": "passed", "manifest_sha256": digest(out / "manifest.json"),
            "native_protocol_id": condition["native_protocol_id"],
            "protocol_manifest_field": None,
            "protocol_validation": "pre_run_plan_bound",
            "parameter_checks": [{
                "name": "score_depth_ratio",
                "pointer": "/shipping_score_command/selected_depth_ratio_of_caliper",
                "expected": 0.25, "observed": 0.25, "matches": True}],
        },
    })
    (run / "stdout.log").write_text("synthetic test only\n")
    (run / "stderr.log").write_text("")
    return run, plan_path, case_path, candidate, condition, binding


class BCTProducerTests(unittest.TestCase):
    def test_synthetic_complete_contract_wraps_without_mutating_native_run(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run, plan_path, case_path, candidate, condition, binding = make_run(base)
            result_path = run / "output/box-native-result.json"
            before = digest(result_path)
            package = plan_path.parent / "native_histories" / candidate["candidate_id"] / condition["condition_id"]
            record_path = producer.finalize_run(run, plan_path, case_path, package, CATALOG, DATA_DIR)
            self.assertEqual(before, digest(result_path))
            record = json.loads(record_path.read_text())
            self.assertEqual(record["candidate_sha256"], candidate["candidate_sha256"])
            history = json.loads((package / "history.json").read_text())
            cert = json.loads((package / "certificate.json").read_text())
            manifest = json.loads((package / "native_input_manifest.json").read_text())
            receipt = json.loads((package / "run_receipt.json").read_text())
            self.assertEqual(history["shipping_capacity_binding"], binding)
            self.assertEqual(cert["shipping_capacity_binding"], binding)
            self.assertEqual(manifest["shipping_capacity_binding"], binding)
            self.assertEqual(receipt["shipping_capacity_binding"], binding)
            ensemble._verify_bct_capacity_evidence(history, cert, manifest, receipt, package)
            expected = {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "condition_id": condition["condition_id"],
                "specimen_id": condition["specimen_id"],
                "native_protocol_id": condition["native_protocol_id"],
                "source_data_catalog_sha256": json.loads(plan_path.read_text())["source_data_catalog_sha256"],
                "source_dataset_sha256": condition["source_dataset_sha256"],
                "source_asset_sha256": condition["source_asset_sha256"],
                "dataset_id": condition["dataset_id"],
                "target_recipe_id": "literature2009_cross_source_recipe",
                "parameters": candidate["parameters"],
            }
            ensemble._verify_certificate(record, plan_path.parent, expected)

    def test_parameter_mismatch_fails_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run, plan_path, case_path, candidate, condition, _ = make_run(base)
            manifest_path = run / "output/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["shipping_score_command"]["selected_depth_ratio_of_caliper"] = 0.3
            write_json(manifest_path, manifest)
            package = plan_path.parent / "must-not-exist"
            with self.assertRaisesRegex(producer.ProducerError, "raw harness receipt does not prove"):
                producer.finalize_run(run, plan_path, case_path, package, CATALOG, DATA_DIR)
            self.assertFalse(package.exists())

    def test_unsealed_case_binding_fails_and_keeps_raw_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run, plan_path, case_path, candidate, condition, _ = make_run(base)
            prediction_path = run / "prediction.json"
            prediction = json.loads(prediction_path.read_text())
            prediction["external_inputs"].remove(str(case_path))
            prediction["inputs_sha256"].pop(f"external/{case_path.name}")
            write_json(prediction_path, prediction)
            raw_hash = digest(run / "output/box-native-result.json")
            package = plan_path.parent / "must-not-exist"
            with self.assertRaisesRegex(producer.ProducerError, "absent from prediction.external_inputs"):
                producer.finalize_run(run, plan_path, case_path, package, CATALOG, DATA_DIR)
            self.assertFalse(package.exists())
            self.assertEqual(raw_hash, digest(run / "output/box-native-result.json"))

    def test_checkpoint_pair_locator_mismatch_fails_without_rewriting_native_run(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            run, plan_path, case_path, candidate, condition, _ = make_run(base)
            metadata_path = run / "output/shipping-rigid-accepted-checkpoint.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["fem_checkpoint_sha256"] = "0" * 64
            write_json(metadata_path, metadata)
            before = digest(run / "output/box-native-result.json")
            package = plan_path.parent / "must-not-exist"
            with self.assertRaisesRegex(producer.ProducerError,
                                        "does not bind the qualified Matter checkpoint"):
                producer.finalize_run(run, plan_path, case_path, package, CATALOG, DATA_DIR)
            self.assertFalse(package.exists())
            self.assertEqual(before, digest(run / "output/box-native-result.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
