#!/usr/bin/env python3
"""Synthetic CSV/JSON contracts only; these are not native cardboard evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "cardboard_stage_resolution_study", HERE / "cardboard_stage_resolution_study.py")
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fake_box_run(root: Path, name: str, *, nx: int = 8, dt: float = 0.1,
                 score_depth: float = 0.25,
                 liner_hash: str = "liner-a", failed: bool = False,
                 old_step_schema: bool = False, force_scale: float = 1.0,
                 fully_failed_fraction: float = 0.25) -> Path:
    path = root / name
    path.mkdir()
    materials = {}
    for role, token in (("liner", liner_hash), ("medium", "medium-a"), ("glue", "glue-a")):
        source = path / f"{role}.nmatter"
        payload = f"synthetic-{role}:{token}\n".encode()
        source.write_bytes(payload)
        materials[f"{role}_path"] = str(source)
        materials[f"{role}_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "numi.cardboard.explicit-strip.v1",
        "specimen": "fefco0201_flat_blank",
        "preset": "literature2009",
        "geometry": {"source_dimensions": "synthetic fixture", "length_m": 1.025,
                     "width_m": 0.4, "total_height_m": 0.00421, "pitch_m": 0.0079,
                     "liner_thickness_m": 0.000277, "medium_normal_thickness_m": 0.000191,
                     "medium_centerline_height_m": 0.003335, "medium_present": True,
                     "glue_minimum_gap_m": 0.0001, "bond_width_m": 0.0008,
                     "upper_glue_gap_m": 0.00003, "upper_bond_width_m": 0.0006,
                     "contact_tie_assumption": "finite bridge"},
        "mesh": {"nx_per_pitch": nx, "ny": 1, "thickness_slices": 1},
        "materials": materials,
        "tooling": {"mode": "two rails and rounded punch", "punch_radius_m": 0.001,
                    "channel_width_m": 0.006, "support_rail_width_m": 0.01,
                    "die_to_channel_clearance_calipers": 0.25,
                    "initial_nose_clearance_m": 0.0001, "initial_anvil_clearance_m": 0.0001},
        "shipping_score_command": {"enabled": True, "selected_depth_ratio_of_caliper": score_depth,
                                   "tool_speed_m_per_s": 0.05,
                                   "source_command_sha256": f"score-{score_depth}"},
        "solver": {"dt_s": dt, "relative_residual_tolerance": 1e-6,
                   "volume_tolerance": 1e-4, "pressure_tolerance": 1e-4,
                   "deformable_self_contact": True, "contact_slop_m": 1e-6,
                   "deformation_gradient": "persistent_reference_displacement",
                   "local_material_newton_iterations": 16, "newton_iteration_budget": 64,
                   "fgmres_restart": 10, "fgmres_iteration_budget": 1024,
                   "fgmres_preconditioner": "scalar_diagonal", "line_search_steps": 8,
                   "maximum_rate_exponent": 0},
    }
    write_json(path / "manifest.json", manifest)
    result = {"status": "complete", "compiled_world_fingerprint": 7}
    write_json(path / "result.json", result)
    native_status = "failed_native" if failed else "completed_native_gravity_transfer_compression_and_drop_unqualified"
    native = {
        "schema": "numi.cardboard.fefco0201-native-execution.v1", "status": native_status,
        "compiled_world_fingerprint": 7,
        "native_steps_attempted": 2 * round(0.3 / dt),
        "native_steps_accepted": 2 * round(0.3 / dt) - (1 if failed else 0),
        "source_physics_fingerprint": 8, "device_program_fingerprint": 7,
        "score_line_count": 12, "score_tool_and_anvil_withdrawn_before_folding": True,
        "shipping_score_approach_steps_accepted": 4, "shipping_score_retract_steps_accepted": 4,
        "released_assembly_settled": True, "released_clamp_zero_active_gate_passed": True,
        "released_assembly_temporary_tool_contact_free": True,
        "released_assembly_simulation_process_complete": True,
        "native_slow_compression_completed": True, "native_half_speed_compression_completed": True,
        "native_compression_rate_sensitivity_gate_passed": True,
        "compression_stroke_m": 0.7,
        "compression_stroke_fraction_of_inside_height": 0.70,
        "compression_peak_sample_index": 1,
        "shipping_source_material_binding_sha256": "a" * 64,
        "assembled_checkpoint_sha256_for_capacity_binding": "b" * 64,
        "assembled_rigid_sidecar_sha256_for_capacity_binding": "c" * 64,
        "compression_rate_sensitivity": {"passed": True, "comparison_evaluated": True},
        "springback_observation_file": "box-score-springback.csv",
        "springback_observation_rows": 12,
        "springback_observable_rows": 12,
        "released_settle_steps_accepted": 1,
        "remaining_active_temporary_clamp_node_count": 0,
        "shape_fit_coverage_minimum": 0.75,
        "shape_fit_bound_is_published_tolerance": False,
        "native_gravity_transfer_all_cases_completed": True,
        "native_finite_mass_drop_completed_cases": 6, "native_finite_mass_drop_planned_cases": 6,
        "shipping_through_stage": "complete",
    }
    write_json(path / "box-native-result.json", native)
    fields = ["native_step", "stage", "action_index", "accepted", "status_env0", "status_env1",
              "fgmres_env0", "certificate_valid_env0", "certificate_residual_env0",
              "certificate_correction_env0", "certificate_volume_env0", "certificate_pressure_env0",
              "simulation_time_s", "dt_s"]
    if old_step_schema:
        fields = ["native_step", "stage", "action_index", "accepted", "status_env0"]
    step_rows = []
    steps_per_branch = round(0.3 / dt)
    native_step_count = 2 * steps_per_branch
    for i in range(native_step_count):
        branch = "nominal" if i < steps_per_branch else "half_speed"
        branch_local = i if branch == "nominal" else i - steps_per_branch
        row = {"native_step": i,
               "stage": f"slow_compression_{branch}",
               "action_index": 0,
               "accepted": 0 if failed and i == native_step_count - 1 else 1,
               "status_env0": 10 if failed and i == native_step_count - 1 else 0}
        if not old_step_schema:
            row.update({"status_env1": "NA", "fgmres_env0": 4, "certificate_valid_env0": 1,
                        "certificate_residual_env0": 1e-8, "certificate_correction_env0": 1e-8,
                        "certificate_volume_env0": 1e-6, "certificate_pressure_env0": 1e-6,
                        "simulation_time_s": (i + 1) * dt, "dt_s": dt})
        step_rows.append(row)
    write_csv(path / "box-native-steps.csv", fields, step_rows)
    action_fields = ["action_index", "status", "accepted_native_steps"]
    action_rows = [{"action_index": i, "status": "completed_from_accepted_state", "accepted_native_steps": 1}
                   for i in range(15)]
    write_csv(path / "box-actions.csv", action_fields, action_rows)
    angle_fields = ["native_step", "stage", "action_index", "environment", "step_accepted",
                    "score_line_index", "fit_valid", "fold_turn_deg"]
    write_csv(path / "box-score-angles.csv", angle_fields, [])
    force_fields = ["native_step", "phase", "accepted", "stroke_m", "dt_s",
                    "macro_observation_duration_s", "control_step_count",
                    "control_step_dt_min_s", "control_step_dt_max_s",
                    "platen_mean_force_y_N", "cumulative_platen_work_J"]
    force_rows = []
    for branch, scale in (("nominal", 1.0), ("half_speed", 1.01)):
        branch_offset = 0 if branch == "nominal" else steps_per_branch
        for i, (stroke, force) in enumerate(((0.0, 0.0), (0.35, 10.0), (0.7, 8.0))):
            summary = {
                "native_step": branch_offset + steps_per_branch - 1,
                "phase": f"{branch}_slow_compression" if i == 2 else f"{branch}_contact_origin",
                "accepted": 1,
                "stroke_m": stroke, "dt_s": dt,
                "macro_observation_duration_s": 0.3 if i == 2 else "",
                "control_step_count": steps_per_branch if i == 2 else "",
                "control_step_dt_min_s": dt if i == 2 else "",
                "control_step_dt_max_s": dt if i == 2 else "",
                "platen_mean_force_y_N": force * scale * force_scale,
                "cumulative_platen_work_J": stroke * force * scale * force_scale,
            }
            force_rows.append(summary)
    write_csv(path / "shipping-compression.csv", force_fields, force_rows)
    nominal_path = path / "shipping-compression-nominal-curve.csv"
    write_csv(nominal_path,
              ["sample_index", "stroke_m", "force_N", "contact_samples"],
              [{"sample_index": 0, "stroke_m": 0.0, "force_N": 0.0, "contact_samples": 0.0},
               {"sample_index": 1, "stroke_m": 0.35,
                "force_N": 10.0 * force_scale, "contact_samples": 1.0},
               {"sample_index": 2, "stroke_m": 0.7,
                "force_N": 8.0 * force_scale, "contact_samples": 1.0}])
    native["nominal_compression_curve_sha256"] = hashlib.sha256(
        nominal_path.read_bytes()).hexdigest()
    write_json(path / "box-native-result.json", native)
    hist_fields = ["native_step", "simulation_time_s", "dt_s", "stage", "accepted",
                   "paper_status", "paper_area_weighted_damage_fraction", "paper_reference_plane_area_m2",
                   "starch_status", "starch_area_weighted_cohesive_damage_fraction", "starch_active_reference_area_m2",
                   "starch_fully_failed_status", "starch_fully_failed_cohesive_area_m2",
                   "starch_fully_failed_cohesive_area_fraction", "starch_fully_failed_faces",
                   "hma_status", "hma_area_weighted_cohesive_damage_fraction", "hma_active_reference_area_m2",
                   "hma_fully_failed_status", "hma_fully_failed_cohesive_area_m2",
                   "hma_fully_failed_cohesive_area_fraction", "hma_fully_failed_faces"]
    hist = [{"native_step": native_step_count - 1, "simulation_time_s": native_step_count * dt, "dt_s": dt, "stage": "final_accepted_state",
             "accepted": 1, "paper_status": "available_model_area_weighted_damage_index",
             "paper_area_weighted_damage_fraction": 0.1, "paper_reference_plane_area_m2": 0.5,
             "starch_status": "available_area_weighted_cohesive_degradation_index",
             "starch_area_weighted_cohesive_damage_fraction": 0.2, "starch_active_reference_area_m2": 0.1,
             "starch_fully_failed_status": "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area",
             "starch_fully_failed_cohesive_area_m2": 0.1 * fully_failed_fraction,
             "starch_fully_failed_cohesive_area_fraction": fully_failed_fraction,
             "starch_fully_failed_faces": 2,
             "hma_status": "not_enabled", "hma_area_weighted_cohesive_damage_fraction": "",
             "hma_active_reference_area_m2": "", "hma_fully_failed_status": "not_enabled",
             "hma_fully_failed_cohesive_area_m2": "", "hma_fully_failed_cohesive_area_fraction": "",
             "hma_fully_failed_faces": 0}]
    write_csv(path / "shipping-history-observables.csv", hist_fields, hist)

    reaction_fields = ["native_step", "time_s", "stage", "assembly_action_index",
                       "score_line_index", "environment", "step_accepted", "reaction_valid",
                       "active_grip_node_count", "total_moment_x_Nm", "total_moment_y_Nm",
                       "total_moment_z_Nm"]
    reaction_rows = []
    for step, time_s, moment_scale in ((0, dt, 1.0),
                                       (native_step_count - 1, native_step_count * dt, 2.0)):
        reaction_rows.append({"native_step": step, "time_s": time_s,
                              "stage": "fixture_grip", "assembly_action_index": 0,
                              "score_line_index": 0, "environment": 0,
                              "step_accepted": 1, "reaction_valid": 1,
                              "active_grip_node_count": 4,
                              "total_moment_x_Nm": 0.0,
                              "total_moment_y_Nm": moment_scale * force_scale,
                              "total_moment_z_Nm": 0.0})
    write_csv(path / "box-grip-reactions.csv", reaction_fields, reaction_rows)

    springback_fields = ["environment", "score_line_index", "held_native_step",
                         "held_time_s", "released_native_step", "released_time_s",
                         "released_settled", "held_turn_deg", "released_turn_deg",
                         "springback_turn_deg", "held_fit_rms_m", "released_fit_rms_m",
                         "fit_bound_m", "held_fit_coverage", "released_fit_coverage",
                         "sample_count", "held_fit_pass", "released_fit_pass",
                         "springback_observable", "reason"]
    springback_rows = [{
        "environment": 0, "score_line_index": score,
        "held_native_step": 0, "held_time_s": dt,
        "released_native_step": native_step_count - 1,
        "released_time_s": native_step_count * dt, "released_settled": 1,
        "held_turn_deg": 90.0,
        "released_turn_deg": 90.0 + 0.1 * force_scale,
        "springback_turn_deg": 0.1 * force_scale,
        "held_fit_rms_m": 0.0001, "released_fit_rms_m": 0.0001,
        "fit_bound_m": 0.004, "held_fit_coverage": 0.9,
        "released_fit_coverage": 0.9, "sample_count": 20,
        "held_fit_pass": 1, "released_fit_pass": 1,
        "springback_observable": 1, "reason": "synthetic_fixture",
    } for score in range(12)]
    write_csv(path / "box-score-springback.csv", springback_fields, springback_rows)
    return path


def add_impact_history(path: Path, *, dt: float, force_scale: float = 1.0,
                       rejected: bool = False) -> None:
    native_path = path / "box-native-result.json"
    native = json.loads(native_path.read_text())
    native["shipping_impact_timestep_levels_s"] = [dt]
    write_json(native_path, native)
    fields = ["case_id", "time_s", "dt_s", "accepted", "body_y_m",
              "matter_mean_force_y_n", "rigid_mechanical_energy_relative_j"]
    rows = []
    horizon = 0.001
    step_count = round(horizon / dt) + 1
    for case_index, case_id in enumerate(("centered_0.5g", "quarter_1.0g")):
        for step in range(step_count):
            time_s = step * dt
            rows.append({"case_id": case_id, "time_s": time_s, "dt_s": dt,
                         "accepted": 0 if rejected and step == step_count - 1 else 1,
                         "body_y_m": 0.1 - 0.5 * time_s * time_s,
                         "matter_mean_force_y_n": force_scale * (10 + case_index + 1000 * time_s),
                         "rigid_mechanical_energy_relative_j": 1 + case_index + 250 * time_s})
    write_csv(path / "shipping-impact-dynamics.csv", fields, rows)


class CardboardStageResolutionStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_failed_prefix_is_never_baseline(self) -> None:
        failed = fake_box_run(self.root, "failed", failed=True)
        run = study.load_run(failed)
        self.assertEqual(run.admission["status"], "failed_prefix")
        plan = study.make_plan(failed)
        self.assertEqual(plan["status"], "reference_not_admitted")

    def test_old_step_csv_missing_certificate_is_inconclusive(self) -> None:
        run_dir = fake_box_run(self.root, "old", old_step_schema=True)
        run = study.load_run(run_dir)
        self.assertEqual(run.admission["status"], "inconclusive_missing_certificates")
        self.assertTrue(any("certificate_valid_env0" in reason for reason in run.admission["reasons"]))

    def test_preflight_is_not_a_physics_baseline(self) -> None:
        run_dir = fake_box_run(self.root, "preflight")
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["status"] = "native_preflight_only_no_physics_step"
        write_json(result_path, result)
        native_path = run_dir / "box-native-result.json"
        native = json.loads(native_path.read_text())
        native["status"] = "preflight_only"
        write_json(native_path, native)
        run = study.load_run(run_dir)
        self.assertEqual(run.admission["status"], "preflight_only")

    def test_full_native_pair_reports_available_force_and_missing_physical_observables(self) -> None:
        left = fake_box_run(self.root, "left", nx=8)
        right = fake_box_run(self.root, "right", nx=16, force_scale=1.005)
        report = study.compare_runs(left, right, "mesh_nx", "all")
        self.assertEqual(report["status"], "descriptive_metrics_available_with_unresolved_observables")
        force = report["metrics"]["compression_force_work"]
        self.assertLess(force["relative_peak_force_difference"], 0.01)
        moments = report["metrics"]["accepted_grip_moment_curve"]
        self.assertEqual(moments["status"], "available")
        self.assertIn("not divided by board width", moments["load_normalization"])
        self.assertEqual(report["metrics"]["released_springback"]["status"], "available")
        damage = report["metrics"]["damage_and_debond"]
        self.assertIsNone(damage["geometric_paper_crack_fraction_difference"])
        self.assertEqual(damage["left"]["area_fraction_status"], "inconclusive_not_exported_by_probe")
        self.assertEqual(damage["status"], "capacity_loss_area_available_geometric_open_area_unexported")
        self.assertAlmostEqual(
            damage["fully_failed_cohesive_capacity_area"]["starch"]["absolute_difference"], 0.0)

    def test_fully_failed_cohesive_area_is_separate_from_geometric_open_area(self) -> None:
        left = fake_box_run(self.root, "left", fully_failed_fraction=0.2)
        right = fake_box_run(self.root, "right", nx=16, fully_failed_fraction=0.35)
        report = study.compare_runs(left, right, "mesh_nx", "all")
        damage = report["metrics"]["damage_and_debond"]
        starch = damage["fully_failed_cohesive_capacity_area"]["starch"]
        self.assertAlmostEqual(starch["absolute_difference"], 0.15)
        self.assertIsNone(damage["geometric_starch_debond_fraction_difference"])

    def test_fully_failed_area_rejects_inconsistent_active_area_denominator(self) -> None:
        run_dir = fake_box_run(self.root, "inconsistent")
        history_path = run_dir / "shipping-history-observables.csv"
        fields, rows = study._rows(history_path)
        rows[0]["starch_fully_failed_cohesive_area_m2"] = "0.2"
        rows[0]["starch_fully_failed_cohesive_area_fraction"] = "0.5"
        write_csv(history_path, fields, rows)
        run = study.load_run(run_dir)
        history = study._history_metrics(run)
        self.assertEqual(history["fully_failed_cohesive_capacity_area"]["starch"]["status"],
                         "inconclusive_inconsistent_area_accounting")
        self.assertEqual(history["fully_failed_cohesive_capacity_area_status"],
                         "inconclusive_missing_or_invalid_capacity_area")

    def test_capacity_fraction_pair_requires_matching_activated_reference_area(self) -> None:
        left = fake_box_run(self.root, "left")
        right = fake_box_run(self.root, "right", nx=16)
        history_path = right / "shipping-history-observables.csv"
        fields, rows = study._rows(history_path)
        rows[0]["starch_active_reference_area_m2"] = "0.09"
        rows[0]["starch_fully_failed_cohesive_area_m2"] = "0.0225"
        write_csv(history_path, fields, rows)
        report = study.compare_runs(left, right, "mesh_nx", "all")
        damage = report["metrics"]["damage_and_debond"]
        self.assertEqual(damage["status"], "incompatible_activated_reference_area")
        self.assertEqual(damage["fully_failed_cohesive_capacity_area"]["starch"]["status"],
                         "incompatible_activated_reference_area")
        self.assertIsNone(damage["fully_failed_cohesive_capacity_area"]["starch"]["absolute_difference"])

    def test_material_change_outside_declared_axis_rejects_pair(self) -> None:
        left = fake_box_run(self.root, "left")
        right = fake_box_run(self.root, "right", nx=16, liner_hash="different-liner")
        report = study.compare_runs(left, right, "mesh_nx", "all")
        self.assertEqual(report["status"], "incompatible_physical_identity")

    def test_source_digest_mismatch_prevents_candidate_admission(self) -> None:
        run_dir = fake_box_run(self.root, "tampered")
        (run_dir / "liner.nmatter").write_text("changed after manifest seal\n", encoding="utf-8")
        run = study.load_run(run_dir)
        self.assertEqual(run.admission["status"], "inconclusive_source_binding")
        self.assertTrue(any("digest_mismatch:liner" in reason for reason in run.admission["reasons"]))

    def test_triplet_reports_decreasing_available_metrics_without_claiming_convergence(self) -> None:
        coarse = fake_box_run(self.root, "coarse", nx=8, force_scale=1.0)
        middle = fake_box_run(self.root, "middle", nx=16, force_scale=1.004)
        fine = fake_box_run(self.root, "fine", nx=32, force_scale=1.005)
        report = study.compare_triplet(coarse, middle, fine, "mesh_nx", "compression")
        self.assertEqual(report["status"], "descriptive_three_level_decrease_only")
        self.assertFalse(report["convergence_claim"])
        self.assertIn("geometric paper crack area fraction", report["unresolved_observables"])

    def test_triplet_rejects_sensitivity_axis_as_convergence(self) -> None:
        shallow = fake_box_run(self.root, "shallow", score_depth=0.1)
        middle = fake_box_run(self.root, "middle", score_depth=0.25)
        deep = fake_box_run(self.root, "deep", score_depth=0.4)
        report = study.compare_triplet(shallow, middle, deep, "score_depth", "scoring")
        self.assertEqual(report["status"], "incompatible_pair")
        self.assertFalse(report["convergence_claim"])

    def test_capacity_evidence_requires_complete_mesh_and_time_triplets(self) -> None:
        mesh_coarse = fake_box_run(self.root, "mesh-coarse", nx=8, force_scale=1.03)
        reference = fake_box_run(self.root, "shared-middle", nx=16, dt=0.15,
                                 force_scale=1.01)
        mesh_fine = fake_box_run(self.root, "mesh-fine", nx=32, force_scale=1.0)
        time_coarse = fake_box_run(self.root, "time-coarse", nx=16, dt=0.3,
                                   force_scale=1.03)
        time_fine = fake_box_run(self.root, "time-fine", nx=16, dt=0.075,
                                 force_scale=1.0)
        evidence_path = self.root / "capacity" / "capacity.properties"
        report = study.create_capacity_qualification(
            reference, mesh_coarse, reference, mesh_fine,
            time_coarse, reference, time_fine, evidence_path)
        self.assertEqual(report["status"], "qualified_for_numerical_capacity_only")
        self.assertTrue(evidence_path.is_file())
        self.assertTrue((evidence_path.parent / report["mesh_report_file"]).is_file())
        self.assertTrue((evidence_path.parent / report["time_report_file"]).is_file())
        self.assertEqual(report["scope"],
                         "complete accepted 70-percent shipping compression curve and first local peak only")

    def test_capacity_evidence_rejects_half_speed_only_or_prefix_cases(self) -> None:
        mesh_coarse = fake_box_run(self.root, "mesh-coarse", nx=8, force_scale=1.03)
        reference = fake_box_run(self.root, "shared-middle", nx=16, dt=0.15,
                                 force_scale=1.01)
        mesh_fine = fake_box_run(self.root, "mesh-fine", nx=32, force_scale=1.0)
        time_coarse = fake_box_run(self.root, "time-coarse", nx=16, dt=0.3,
                                   force_scale=1.03)
        time_fine = fake_box_run(self.root, "time-fine", nx=16, dt=0.075,
                                 force_scale=1.0, failed=True)
        evidence_path = self.root / "capacity" / "capacity.properties"
        report = study.create_capacity_qualification(
            reference, mesh_coarse, reference, mesh_fine,
            time_coarse, reference, time_fine, evidence_path)
        self.assertEqual(report["status"], "not_qualified")
        self.assertFalse(evidence_path.exists())
        self.assertTrue(any("level_2" in reason
                            for reason in report["time_study"]["reasons"]))

    def test_capacity_reference_must_be_shared_middle_identity(self) -> None:
        a = fake_box_run(self.root, "a", nx=8)
        b = fake_box_run(self.root, "b", nx=16, dt=0.15)
        c = fake_box_run(self.root, "c", nx=32)
        d = fake_box_run(self.root, "d", nx=16, dt=0.3)
        e = fake_box_run(self.root, "e", nx=16, dt=0.075)
        with self.assertRaises(study.StageStudyError):
            study.create_capacity_qualification(
                a, a, b, c, d, b, e, self.root / "qualification.properties")

    def test_capacity_triplets_reject_adjacent_differences_that_grow(self) -> None:
        mesh_coarse = fake_box_run(self.root, "mesh-coarse", nx=8, force_scale=1.01)
        reference = fake_box_run(self.root, "shared-middle", nx=16, dt=0.15,
                                 force_scale=1.02)
        mesh_fine = fake_box_run(self.root, "mesh-fine", nx=32, force_scale=1.0)
        time_coarse = fake_box_run(self.root, "time-coarse", nx=16, dt=0.3,
                                   force_scale=1.01)
        time_fine = fake_box_run(self.root, "time-fine", nx=16, dt=0.075,
                                 force_scale=1.0)
        evidence_path = self.root / "capacity-monotone" / "capacity.properties"
        report = study.create_capacity_qualification(
            reference, mesh_coarse, reference, mesh_fine,
            time_coarse, reference, time_fine, evidence_path)
        self.assertEqual(report["status"], "not_qualified")
        self.assertFalse(evidence_path.exists())
        self.assertFalse(report["mesh_study"]["adjacent_pair_differences_decrease"])
        self.assertIn("adjacent_pair_metric_differences_do_not_decrease",
                      report["mesh_study"]["reasons"])

    def test_impact_pair_compares_only_complete_accepted_body_histories(self) -> None:
        coarse = fake_box_run(self.root, "coarse")
        fine = fake_box_run(self.root, "fine", nx=16)
        add_impact_history(coarse, dt=0.0005)
        add_impact_history(fine, dt=0.00025, force_scale=1.001)
        report = study.compare_runs(coarse, fine, "impact_time_step", "impact")
        self.assertEqual(report["status"], "descriptive_metrics_available_with_unresolved_observables")
        impact = report["metrics"]["impact_dynamics"]
        self.assertEqual(impact["status"], "available")
        self.assertEqual(impact["case_count"], 2)
        self.assertIn("moment_curve", impact)

    def test_rejected_impact_sample_blocks_impact_comparison(self) -> None:
        coarse = fake_box_run(self.root, "coarse")
        fine = fake_box_run(self.root, "fine", nx=16)
        add_impact_history(coarse, dt=0.0005, rejected=True)
        add_impact_history(fine, dt=0.00025)
        report = study.compare_runs(coarse, fine, "impact_time_step", "impact")
        self.assertEqual(report["metrics"]["impact_dynamics"]["status"],
                         "inconclusive_missing_or_invalid_impact_history")

    def test_score_depth_pair_only_changes_declared_depth(self) -> None:
        shallow = fake_box_run(self.root, "shallow", score_depth=0.1)
        deep = fake_box_run(self.root, "deep", score_depth=0.4)
        report = study.compare_runs(shallow, deep, "score_depth", "scoring")
        self.assertEqual(report["status"], "descriptive_metrics_available_with_unresolved_observables")
        self.assertEqual(report["left"]["axis_value"], 0.1)
        self.assertEqual(report["right"]["axis_value"], 0.4)

    def test_same_axis_value_is_not_a_refinement_pair(self) -> None:
        left = fake_box_run(self.root, "left", nx=8)
        right = fake_box_run(self.root, "right", nx=8)
        report = study.compare_runs(left, right, "mesh_nx", "all")
        self.assertEqual(report["status"], "incompatible_pair")

    def test_compression_rate_is_within_run_and_never_time_refinement(self) -> None:
        run = fake_box_run(self.root, "run")
        report = study.analyze_compression_rate(run)
        self.assertEqual(report["status"], "descriptive_rate_metrics_available")
        self.assertFalse(report["metrics"]["compression_force_work"]["is_time_refinement"])

    def test_plan_binds_files_and_declares_missing_observables(self) -> None:
        run = fake_box_run(self.root, "run", failed=True)
        plan = study.make_plan(run)
        self.assertEqual(plan["status"], "reference_not_admitted")
        self.assertEqual(len(plan["plan_sha256"]), 64)
        self.assertIn("moment_curve", plan["required_native_observables"])
        self.assertIn("geometric-debond", plan["required_native_observables"]["debond"])

    def test_timestep_admission_uses_actual_accepted_native_substeps(self) -> None:
        coarse = fake_box_run(self.root, "coarse", dt=0.1)
        fine = fake_box_run(self.root, "fine", dt=0.05)
        report = study._time_step_refinement_admission(
            study.load_run(coarse), study.load_run(fine), 0.1, 0.05)
        self.assertEqual(report["status"], "available")
        for branch in ("nominal", "half_speed"):
            self.assertEqual(report["coarse"][branch]["accepted_native_substep_count"], 3)
            self.assertEqual(report["fine"][branch]["accepted_native_substep_count"], 6)
            self.assertAlmostEqual(report["coarse"][branch]["accepted_duration_s"], 0.3)
            self.assertAlmostEqual(report["fine"][branch]["accepted_duration_s"], 0.3)

    def test_timestep_label_change_without_actual_refinement_is_rejected(self) -> None:
        coarse = fake_box_run(self.root, "coarse", dt=0.1)
        mislabeled = fake_box_run(self.root, "mislabeled", dt=0.1)
        manifest_path = mislabeled / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["solver"]["dt_s"] = 0.05
        write_json(manifest_path, manifest)
        report = study.compare_runs(coarse, mislabeled, "time_step", "compression")
        self.assertEqual(report["status"], "incompatible_physical_schedule")
        refinement = report["metrics"]["actual_compression_timestep_refinement"]
        self.assertIn("compression_branch_max_dt_did_not_decrease:nominal",
                      refinement["reasons"])
        self.assertIn("compression_branch_substep_count_did_not_increase:half_speed",
                      refinement["reasons"])

    def test_timestep_admission_rejects_missing_or_forged_actual_dt_summary(self) -> None:
        fine = fake_box_run(self.root, "fine", dt=0.05)
        path = fine / "shipping-compression.csv"
        fields, rows = study._rows(path)
        rows[-1]["control_step_dt_max_s"] = "0.01"
        write_csv(path, fields, rows)
        stats, reasons = study._compression_branch_step_stats(study.load_run(fine))
        self.assertTrue(any(reason.startswith(
            "invalid_actual_compression_branch_record:half_speed") for reason in reasons))
        self.assertIn("nominal", stats)
        self.assertNotIn("half_speed", stats)

        coarse = fake_box_run(self.root, "coarse", dt=0.1)
        step_path = coarse / "box-native-steps.csv"
        step_fields, step_rows = study._rows(step_path)
        del step_rows[0]["dt_s"]
        write_csv(step_path, step_fields, step_rows)
        _, reasons = study._compression_branch_step_stats(study.load_run(coarse))
        self.assertTrue(any("missing_actual_compression_dt:nominal" in reason
                            for reason in reasons))

    def test_triplet_uses_accepted_moment_and_springback_distances(self) -> None:
        coarse = fake_box_run(self.root, "coarse", nx=8, force_scale=1.03)
        middle = fake_box_run(self.root, "middle", nx=16, force_scale=1.01)
        fine = fake_box_run(self.root, "fine", nx=32, force_scale=1.0)
        report = study.compare_triplet(coarse, middle, fine, "mesh_nx", "all")
        metrics = report["pair_metric_differences"]
        self.assertIn("grip_moment.normalized_total_moment_vector_curve_rms", metrics)
        self.assertIn("released_springback.rms_springback_difference_deg", metrics)

    def test_invalid_accepted_moment_or_springback_fit_blocks_observable(self) -> None:
        right = fake_box_run(self.root, "right", nx=16)
        reaction_path = right / "box-grip-reactions.csv"
        fields, rows = study._rows(reaction_path)
        rows[0]["reaction_valid"] = "0"
        write_csv(reaction_path, fields, rows)
        self.assertNotEqual(study._grip_moment_series(study.load_run(right))[1]["status"],
                            "available")

        springback_path = right / "box-score-springback.csv"
        fields, rows = study._rows(springback_path)
        rows[0]["released_fit_coverage"] = "0.2"
        write_csv(springback_path, fields, rows)
        self.assertNotEqual(study._springback_observations(study.load_run(right))[1]["status"],
                            "available")


if __name__ == "__main__":
    unittest.main()
