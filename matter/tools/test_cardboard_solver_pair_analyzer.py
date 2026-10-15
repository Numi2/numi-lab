#!/usr/bin/env python3
"""CPU failure-injection tests for the source-bound pair-aware analyzer."""
from __future__ import annotations

import tempfile
import unittest
import hashlib
from pathlib import Path

import cardboard_solver_pair_analyzer as analyzer


def parameters(offset: float = 0.0) -> list[float]:
    # Positive-definite scales and a valid Hill quadratic; intentionally not
    # either packaged grade's material constants.
    return [700.0 + offset, 4.8e9 + offset, 2.1e9, 2.0e7, 3.8e8, 2.0e5,
            2.0e5, 1.2e9, 1.7e8, 1.4e8, 2.3e7,
            6.3, 3.0, -1.9, 4.56e5, 4.56e5, 59.5, 2.0e-6, 1.0e-8, 10.0]


def state_record(f11: float, epsilon: float | None = None) -> dict:
    f = [f11, 0, 0, 0, 1, 0, 0, 0, 1]
    state = [0.0] * 13
    record = {
        "F_local": f,
        "F_local_low": [1.0e-8] + [0.0] * 8,
        "P_local_Pa": [0.0] * 9,
        "projected_state": state,
        "projected_state_low": [0.0] * 13,
        "projection": [1.0e-8, 2.0e-8, 3.0, 1.0],
        "hill_q_lambda_a_b": [0.8, 0.0, 0.0, 0.2],
    }
    if epsilon is not None:
        record["epsilon"] = epsilon
    return record


def minimal_capture() -> tuple[dict, list[dict]]:
    metadata = {
        "index": 4,
        "name": "hajali2009_liner_hill_ideal",
        "parameter_offset": 0,
        "parameter_names": list(analyzer.PAPER_PARAMETER_NAMES),
    }
    f_direction = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
    center = state_record(1.0)
    center.update({
        "accepted_state": [0.0] * 13,
        "F_direction_local_per_s": f_direction,
        "tangent_local_Pa_per_s": [1.0] * 9,
    })
    sides = [state_record(1.001, 0.01), state_record(.999, 0.01),
             state_record(1.002, 0.02), state_record(.998, 0.02)]
    tet = {
        "record": "tetrahedron", "environment": 0, "tet_id": 7,
        "material_index": 4, "material_state_count": 13,
        "material_parameter_count": 20, "center": center, "sides": sides,
    }
    header = {
        "record": "header", "schema": analyzer.CAPTURE_SCHEMA,
        "root": 1, "newton_iteration": 19, "environment_count": 1,
        "node_count_per_environment": 0, "tetrahedron_count_per_environment": 1,
        "F_local_word_encoding": "high_float32_plus_low_float32_for_compensated_material_input",
        "materials": [metadata], "parameters_by_environment": [parameters()],
        "dt_seconds_by_environment": [1.0e-4],
        "production_inputs_bitwise_unchanged": True,
    }
    return header, [tet]


def minimal_binding(capture_path: Path) -> tuple[dict, dict]:
    snapshot = capture_path.parent / "source-snapshot"
    hashes = {}
    for relative in analyzer.REQUIRED_SOURCE_BINDINGS:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test source {relative}\n", encoding="utf-8")
        hashes[f"source/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    prediction = {
        "schema": "numi.cardboard.native-prediction.v1",
        "executed_build_verified": True,
        "inputs_sha256": hashes,
        "source_snapshot": str(snapshot),
        "argv": ["probe", *analyzer.REQUIRED_TRANSFORMS],
        "environment": {
            "NM_FEM_OPERATOR_FD_SELECTOR": "1:19",
            "NM_FEM_OPERATOR_FD_CAPTURE_PATH": str(capture_path.resolve()),
        },
    }
    receipt = {
        "schema": "numi.cardboard.native-run-receipt.v2",
        "execution_started": True,
        "source_inputs_unchanged": True,
        "timed_out": False,
        "exit_code": 2,
    }
    return prediction, receipt


class PairAnalyzerTests(unittest.TestCase):
    def test_all_four_native_builder_transforms_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.jsonl"
            capture.touch()
            header, _ = minimal_capture()
            prediction, receipt = minimal_binding(capture)
            analyzer.validate_run_binding(capture, header, prediction, receipt)
            for missing in analyzer.REQUIRED_TRANSFORMS:
                broken = dict(prediction)
                broken["argv"] = [flag for flag in prediction["argv"] if flag != missing]
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "required builder transform"):
                    analyzer.validate_run_binding(capture, header, broken, receipt)

    def test_capture_selector_and_path_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.jsonl"
            capture.touch()
            header, _ = minimal_capture()
            prediction, receipt = minimal_binding(capture)
            prediction["environment"]["NM_FEM_OPERATOR_FD_SELECTOR"] = "1:0"
            with self.assertRaisesRegex(ValueError, "selector"):
                analyzer.validate_run_binding(capture, header, prediction, receipt)

    def test_complete_capture_can_be_audited_after_timeout_without_claiming_full_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.jsonl"
            capture.touch()
            header, _ = minimal_capture()
            prediction, receipt = minimal_binding(capture)
            receipt["timed_out"] = True
            receipt["exit_code"] = -15
            binding = analyzer.validate_run_binding(capture, header, prediction, receipt)
            self.assertTrue(binding["receipt_timed_out"])
            self.assertFalse(analyzer.full_history_complete(binding))

    def test_retained_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture.jsonl"
            capture.touch()
            header, _ = minimal_capture()
            prediction, receipt = minimal_binding(capture)
            changed = Path(prediction["source_snapshot"]) / analyzer.REQUIRED_SOURCE_BINDINGS[0]
            changed.write_text("changed source\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                analyzer.validate_run_binding(capture, header, prediction, receipt)

    def test_parameter_names_are_verified_but_values_are_not_defaults(self):
        header, tets = minimal_capture()
        changed = parameters(offset=123.0)
        header["parameters_by_environment"] = [changed]
        descriptor = analyzer.material_descriptors(header)[4]
        row = analyzer.paper_parameter_row(header, descriptor, 0)
        self.assertEqual(row[0], analyzer.f32(changed[0]))
        self.assertNotEqual(row[0], 766.0)
        descriptor["parameter_names"][0] = "wrong_density_slot"
        with self.assertRaisesRegex(ValueError, "names/order"):
            analyzer.paper_parameter_row(header, descriptor, 0)

    def test_parameters_are_selected_per_environment(self):
        header, _ = minimal_capture()
        second = parameters(offset=500.0)
        header["environment_count"] = 2
        header["parameters_by_environment"] = [parameters(), second]
        descriptor = analyzer.material_descriptors(header)[4]
        self.assertNotEqual(analyzer.paper_parameter_row(header, descriptor, 0),
                            analyzer.paper_parameter_row(header, descriptor, 1))

    def test_high_low_words_are_preserved_in_replay_rows(self):
        header, tets = minimal_capture()
        rows, _, _ = analyzer.make_replay_rows(header, tets, 1.0e-6)
        fields = rows[0].split()
        self.assertEqual(fields[0], "e0_t7_s0")
        # F high[0] and low[0] are adjacent inputs, not a rounded-away sum.
        self.assertEqual(float(fields[2]), analyzer.f32(1.0))
        self.assertEqual(float(fields[11]), analyzer.f32(1.0e-8))
        # Projected candidate starts after F[18], direction[9], accepted[13].
        candidate_high_start = 2 + 18 + 9 + 13
        candidate_low_start = candidate_high_start + 13
        self.assertEqual(len(fields), 2 + 18 + 9 + 13 + 26 + 20)
        self.assertEqual(float(fields[candidate_low_start]), 0.0)

    def test_missing_candidate_low_word_fails_closed(self):
        header, tets = minimal_capture()
        del tets[0]["center"]["projected_state_low"]
        with self.assertRaisesRegex(ValueError, "high or low word"):
            analyzer.make_replay_rows(header, tets, 1.0e-6)

    def test_missing_deformation_low_word_fails_closed(self):
        header, tets = minimal_capture()
        del tets[0]["center"]["F_local_low"]
        with self.assertRaisesRegex(ValueError, "high or low word"):
            analyzer.make_replay_rows(header, tets, 1.0e-6)

    def test_material_ranges_cannot_overlap(self):
        header, _ = minimal_capture()
        second = dict(header["materials"][0])
        second["index"] = 5
        header["materials"].append(second)
        with self.assertRaisesRegex(ValueError, "overlap"):
            analyzer.material_descriptors(header)

    def test_near_yield_and_crossing_are_distinct_from_smooth_branch(self):
        band = 1.0e-6
        self.assertEqual(analyzer.classify_trial_q(1.0, band), "near_yield_unresolved")
        self.assertEqual(analyzer.classify_trial_q(0.999, band), "elastic_trial")
        self.assertEqual(analyzer.classify_trial_q(1.001, band), "plastic_trial")

    def test_zero_reference_metric_never_becomes_a_pass(self):
        result = analyzer.metric([0.0, 0.0], [0.0, 0.0])
        self.assertIsNone(result["relative_error"])
        self.assertFalse(result["nonzero_reference"])
        self.assertEqual(result["verdict"], "descriptive_only_no_measured_noise_floor")

    def test_inconclusive_gate_has_nonzero_exit_and_replay_failure_is_distinct(self):
        inconclusive = {"strict_gate": "inconclusive_no_measured_noise_floor",
                        "replay": {"returncode": 0, "invalid_rows": 0, "output_errors": []}}
        self.assertEqual(analyzer.analysis_exit_code(inconclusive), 3)
        replay_failure = {"strict_gate": "inconclusive_no_measured_noise_floor",
                          "replay": {"returncode": 2, "invalid_rows": 1, "output_errors": []}}
        self.assertEqual(analyzer.analysis_exit_code(replay_failure), 4)
        qualified_gate = {"strict_gate": "qualified",
                          "replay": {"returncode": 0, "invalid_rows": 0, "output_errors": []}}
        self.assertEqual(analyzer.analysis_exit_code(qualified_gate), 0)

    def test_candidate_and_reprojected_tangents_remain_separate(self):
        header, tets = minimal_capture()
        rows, sources, metadata = analyzer.make_replay_rows(header, tets, 1.0e-6)
        replay_row = {
            "valid": True,
            "oracle_reprojected_state": [0.0] * 13,
            "oracle_reprojected_P_Pa": [1.0] * 9,
            "oracle_candidate_P_Pa": [2.0] * 9,
            "oracle_tangent_at_captured_candidate_Pa_per_s": [3.0] * 9,
            "oracle_tangent_at_reprojected_state_Pa_per_s": [4.0] * 9,
            "candidate_residual_unscaled_inf": 1.0e-9,
            "candidate_residual_scaled_inf": 2.0e-8,
            "candidate_residual_scaled_rows": [0.0] * 13,
            "source_vm_compensated_scaled_residual_inf": 2.1e-8,
            "source_vm_compensated_scaled_residual_rows": [0.0] * 13,
            "elastic_trial_q": 0.5,
            "candidate_q": 0.8,
        }
        replay = {row.split()[0]: replay_row for row in rows}
        audit = analyzer.make_tet_audit(tets, sources, replay, metadata, 1.0e-6)[0]
        candidate = audit["native_tangent_vs_oracle_at_captured_candidate"]
        projected = audit["native_tangent_vs_oracle_at_reprojected_reference"]
        self.assertEqual(candidate["reference_scale"], 3.0)
        self.assertEqual(projected["reference_scale"], 4.0)
        self.assertIn("native_centered_FD_vs_oracle_tangent_at_captured_candidate",
                      audit["finite_difference_pairs"][0])
        self.assertIn("oracle_reprojected_centered_FD_vs_oracle_tangent_at_reprojected_reference",
                      audit["finite_difference_pairs"][0])
        self.assertEqual(audit["gate_verdict"], "inconclusive_no_measured_noise_floor")

    def test_pair_crossing_is_reported_and_excluded_from_smooth_gate(self):
        header, tets = minimal_capture()
        rows, sources, metadata = analyzer.make_replay_rows(header, tets, 1.0e-6)
        replay = {}
        for row in rows:
            key = row.split()[0]
            side = int(key.rsplit("_s", 1)[1])
            q = 0.9995 if side == 0 else (1.0001 if side in (1, 3) else 0.9999)
            replay[key] = {
                "valid": True,
                "oracle_reprojected_state": [0.0] * 13,
                "oracle_reprojected_P_Pa": [1.0] * 9,
                "oracle_candidate_P_Pa": [2.0] * 9,
                "oracle_tangent_at_captured_candidate_Pa_per_s": [3.0] * 9,
                "oracle_tangent_at_reprojected_state_Pa_per_s": [4.0] * 9,
                "candidate_residual_unscaled_inf": 1.0e-9,
                "candidate_residual_scaled_inf": 2.0e-8,
                "candidate_residual_scaled_rows": [0.0] * 13,
                "source_vm_compensated_scaled_residual_inf": 2.1e-8,
                "source_vm_compensated_scaled_residual_rows": [0.0] * 13,
                "elastic_trial_q": q,
                "candidate_q": q,
            }
        audit = analyzer.make_tet_audit(tets, sources, replay, metadata, 1.0e-6)[0]
        resolved = [pair for pair in audit["finite_difference_pairs"] if pair["status"] == "pair_resolved"]
        self.assertTrue(resolved)
        self.assertTrue(all(pair["branch_crossing"] for pair in resolved))
        self.assertTrue(all(not pair["smooth_branch_comparison_eligible"] for pair in resolved))

    def test_near_yield_pair_stays_inconclusive(self):
        header, tets = minimal_capture()
        rows, sources, metadata = analyzer.make_replay_rows(header, tets, 1.0e-6)
        replay = {}
        for row in rows:
            key = row.split()[0]
            replay[key] = {
                "valid": True,
                "oracle_reprojected_state": [0.0] * 13,
                "oracle_reprojected_P_Pa": [1.0] * 9,
                "oracle_candidate_P_Pa": [2.0] * 9,
                "oracle_tangent_at_captured_candidate_Pa_per_s": [3.0] * 9,
                "oracle_tangent_at_reprojected_state_Pa_per_s": [4.0] * 9,
                "candidate_residual_unscaled_inf": 1.0e-9,
                "candidate_residual_scaled_inf": 2.0e-8,
                "candidate_residual_scaled_rows": [0.0] * 13,
                "source_vm_compensated_scaled_residual_inf": 2.1e-8,
                "source_vm_compensated_scaled_residual_rows": [0.0] * 13,
                "elastic_trial_q": 1.0,
                "candidate_q": 1.0,
            }
        audit = analyzer.make_tet_audit(tets, sources, replay, metadata, 1.0e-6)[0]
        resolved = [pair for pair in audit["finite_difference_pairs"] if pair["status"] == "pair_resolved"]
        self.assertTrue(resolved)
        self.assertTrue(all(pair["near_yield_unresolved"] for pair in resolved))
        self.assertTrue(all(not pair["smooth_branch_comparison_eligible"] for pair in resolved))

    def test_zero_direction_sides_are_not_misreported_as_a_pair(self):
        header, tets = minimal_capture()
        tets[0]["center"]["F_direction_local_per_s"] = [0.0] * 9
        rows, sources, metadata = analyzer.make_replay_rows(header, tets, 1.0e-6)
        replay = {}
        for row in rows:
            replay[row.split()[0]] = {
                "valid": True,
                "oracle_reprojected_state": [0.0] * 13,
                "oracle_reprojected_P_Pa": [1.0] * 9,
                "oracle_candidate_P_Pa": [2.0] * 9,
                "oracle_tangent_at_captured_candidate_Pa_per_s": [3.0] * 9,
                "oracle_tangent_at_reprojected_state_Pa_per_s": [4.0] * 9,
                "candidate_residual_unscaled_inf": 1.0e-9,
                "candidate_residual_scaled_inf": 2.0e-8,
                "candidate_residual_scaled_rows": [0.0] * 13,
                "source_vm_compensated_scaled_residual_inf": 2.1e-8,
                "source_vm_compensated_scaled_residual_rows": [0.0] * 13,
                "elastic_trial_q": 0.5,
                "candidate_q": 0.8,
            }
        audit = analyzer.make_tet_audit(tets, sources, replay, metadata, 1.0e-6)[0]
        self.assertTrue(audit["finite_difference_pairs"])
        for pair in audit["finite_difference_pairs"]:
            self.assertNotEqual(pair["status"], "pair_resolved")
            self.assertEqual(pair["unresolved_reason"], "captured local deformation direction is zero")


if __name__ == "__main__":
    unittest.main()
