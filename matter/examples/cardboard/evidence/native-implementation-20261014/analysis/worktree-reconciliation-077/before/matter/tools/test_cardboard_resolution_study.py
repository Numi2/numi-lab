#!/usr/bin/env python3
"""Synthetic contract tests only; these fixtures are not native or physical evidence."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
conv_spec = importlib.util.spec_from_file_location("cardboard_convergence", HERE / "cardboard_convergence.py")
assert conv_spec and conv_spec.loader
cc = importlib.util.module_from_spec(conv_spec)
sys.modules[conv_spec.name] = cc
conv_spec.loader.exec_module(cc)

fixture_spec = importlib.util.spec_from_file_location("cardboard_convergence_fixtures", HERE / "test_cardboard_convergence.py")
assert fixture_spec and fixture_spec.loader
fixtures = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixtures
fixture_spec.loader.exec_module(fixtures)

study_spec = importlib.util.spec_from_file_location("cardboard_resolution_study", HERE / "cardboard_resolution_study.py")
assert study_spec and study_spec.loader
study = importlib.util.module_from_spec(study_spec)
sys.modules[study_spec.name] = study
study_spec.loader.exec_module(study)


class CardboardResolutionStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _release_run(self, name: str, *, nx: int = 2, moment_scale: float = 1.0,
                     accepted_flag: int = 1) -> Path:
        return fixtures.fake_run(
            self.root, name, nx=nx, dt=0.01, steps=4, hold=2, unload=4, relax=2,
            release=2, phase_angle=1.0, moment_scale=moment_scale,
            accepted_flag=accepted_flag,
        )

    def test_declared_mesh_axes_and_combined_point(self) -> None:
        cases = {case["id"]: case["mesh"] for case in study._spatial_cases()}
        self.assertEqual(cases, {
            "nx16-ny2-t1": [16, 2, 1],
            "nx32-ny2-t1": [32, 2, 1],
            "nx8-ny4-t1": [8, 4, 1],
            "nx8-ny8-t1": [8, 8, 1],
            "nx8-ny2-t2": [8, 2, 2],
            "nx8-ny2-t4": [8, 2, 4],
            "nx32-ny8-t4": [32, 8, 4],
        })

    def test_plan_binds_inputs_and_freezes_full_case_matrix_without_launching(self) -> None:
        baseline = self._release_run("baseline-plan", nx=8)
        manifest_path = baseline / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["preset"] = "literature2009"
        manifest["mesh"]["ny"] = 2
        manifest["tooling"] = {"enabled": False}
        manifest["solver"]["fgmres_preconditioner"] = "scalar_diagonal"
        manifest["solver"]["deformation_gradient"] = "absolute_position"
        for path_key, digest_key, role in (
            ("liner_path", "liner_sha256", "liner"),
            ("medium_path", "medium_sha256", "medium"),
            ("glue_path", "glue_sha256", "glue"),
        ):
            material = self.root / f"{role}.nmatter"
            material.write_text(f"synthetic {role} source\n")
            manifest["materials"][path_key] = str(material)
            manifest["materials"][digest_key] = hashlib.sha256(material.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        probe = self.root / "probe"
        probe.write_text("synthetic executable binding; never launched by this test\n")
        probe.chmod(probe.stat().st_mode | 0o111)
        metallib = self.root / "cardboard.metallib"
        metallib.write_bytes(b"synthetic metallib binding\n")
        extra_source = self.root / "extra-source.mm"
        extra_source.write_text("synthetic source binding\n")

        plan = study.make_plan(
            baseline_dir=baseline, probe=probe, metallib=metallib,
            run_root=self.root / "study-run-root", timeout_seconds=123.0,
            slower_load_factor=2, max_free_force_residual_n=0.5,
            source_paths=[extra_source],
        )
        self.assertEqual(plan["plan_sha256"], study._plan_digest(plan))
        self.assertEqual(len(plan["cases"]), 10)
        self.assertEqual(plan["prediction"]["max_free_force_residual_n"], 0.5)
        self.assertEqual(plan["execution"]["parallelism"], 1)
        self.assertIn({"path": str(extra_source.resolve()),
                       "sha256": hashlib.sha256(extra_source.read_bytes()).hexdigest()},
                      plan["execution"]["source_files"])
        time_cases = [case for case in plan["cases"] if case["kind"] == "temporal"]
        self.assertEqual([case["time_refinement_factor"] for case in time_cases], [2, 4])
        for case in time_cases:
            self.assertAlmostEqual(case["relative_residual_tolerance"] / case["dt_s"],
                                   manifest["solver"]["relative_residual_tolerance"] /
                                   manifest["solver"]["dt_s"])
        self.assertEqual(plan["cases"][-1]["kind"], "load_rate")
        self.assertFalse((self.root / "study-run-root").exists())
        extra_source.write_text("modified source must invalidate the sealed plan\n")
        with self.assertRaises(study.StudyError):
            study._verify_bindings(plan)

    def test_time_refinement_preserves_each_phase_duration_and_speed(self) -> None:
        base = {"loading": 16, "hold": 8, "unloading": 16, "relaxation": 32, "release": 8, "dt_s": 0.000025}
        for factor in (2, 4):
            counts = study._time_counts(base, factor)
            self.assertEqual(set(counts), set(study.PHASE_ORDER))
            for phase in study.PHASE_ORDER:
                self.assertAlmostEqual((base["dt_s"] / factor) * counts[phase], base["dt_s"] * base[phase])
            base_loading_rate = 0.25 / (base["dt_s"] * base["loading"])
            refined_loading_rate = 0.25 / ((base["dt_s"] / factor) * counts["loading"])
            self.assertAlmostEqual(base_loading_rate, refined_loading_rate)

    def test_native_command_preserves_measured_solver_and_projection_policy(self) -> None:
        baseline = self._release_run("current-policy", nx=8)
        run = cc._load_run(baseline)
        run.manifest["tooling"] = {"enabled": False}
        for role in ("liner", "medium", "glue"):
            run.manifest["materials"][role + "_path"] = str(self.root / (role + ".nmatter"))
        solver = run.manifest["solver"]
        solver.update({"fgmres_preconditioner": "regional_galerkin_block_jacobi",
                       "fgmres_restart": 16, "line_search_steps": 24,
                       "fgmres_algorithmic_tangent_cache": True,
                       "paper_compensated_arithmetic": True,
                       "safeguarded_paper_hill_initializer": True})
        run.manifest["paper_projection_residual_scaling"] = {"enabled": True}
        run.manifest["paper_hill_accurate_projection"] = {"enabled": True}
        common = study._native_base_arguments(run)
        for flag in ("--cache-fem-algorithmic-tangent", "--scaled-paper-residuals",
                     "--compensated-paper-arithmetic", "--accurate-paper-projection",
                     "--safeguarded-paper-initializer"):
            self.assertIn(flag, common)
        for flag, value in (("--fgmres-restart", "16"), ("--line-search-steps", "24"),
                            ("--fgmres-preconditioner", "regional-galerkin-block")):
            self.assertEqual(common[common.index(flag) + 1], value)
        case = {"mesh": [32, 8, 4], "dt_s": .005,
                "relative_residual_tolerance": 2.5e-7,
                "phase_steps": {phase: 8 for phase in study.PHASE_ORDER}}
        argv = study._native_args(case, common, self.root / "output", self.root / "shaders")
        self.assertEqual(argv.count("--residual-tolerance"), 1)
        self.assertEqual(float(argv[argv.index("--residual-tolerance") + 1]), 2.5e-7)
        del solver["safeguarded_paper_hill_initializer"]
        with self.assertRaisesRegex(study.StudyError, "explicit arithmetic/initializer"):
            study._native_base_arguments(run)

    def test_source15_command_preserves_the_effective_compiled_policy(self) -> None:
        run = cc._load_run(self._release_run("source15-policy", nx=8))
        run.manifest["tooling"] = {"enabled": False}
        for role in ("liner", "medium", "glue"):
            run.manifest["materials"][role + "_path"] = str(self.root / (role + ".nmatter"))
        run.manifest["solver"].update({"fgmres_preconditioner": "scalar_diagonal",
                                      "paper_compensated_arithmetic": True,
                                      "safeguarded_paper_hill_initializer": False})
        run.manifest["compensated_source_paper_projection"] = {
            "enabled": True, "source_material_indices": [0, 1],
            "source_map_sha256": run.manifest["materials"]["regional_map_sha256"],
            "compiled_world_fingerprint": run.manifest["compiled_world_fingerprint"],
            "effective_compiled_flags": {
                role + suffix: value for role in ("liner", "medium")
                for suffix, value in (("_flags", 0x50),
                                      ("_generic_compensated_implicit_projection", True),
                                      ("_compensated_paper_arithmetic", True))}}
        args = study._native_base_arguments(run)
        self.assertIn("--compensated-source-paper-projection", args)
        self.assertIn("--compensated-paper-arithmetic", args)
        run.manifest["compensated_source_paper_projection"]["effective_compiled_flags"]["liner_flags"] = 0
        with self.assertRaisesRegex(study.StudyError, "compiled liner"):
            study._native_base_arguments(run)

    def test_admitted_pair_has_force_work_springback_and_force_residual_observables(self) -> None:
        coarse = self._release_run("coarse", nx=2)
        fine = self._release_run("fine", nx=4, moment_scale=1.01)
        admission = cc.analyze_pair(coarse, fine, "spatial", list(study.PHASE_ORDER))
        self.assertEqual(admission["status"], "analyzed_descriptive")
        run_a = cc._load_run(coarse)
        run_b = cc._load_run(fine)
        measured = study._pair_metrics(run_a, run_b, "spatial")
        gates = study._gate_pair(measured)
        self.assertIn("peak_force_relative", measured)
        self.assertIn("absolute_moment_path_work_J", measured["raw_summary"]["baseline"])
        self.assertEqual(gates["observed_free_force_residuals"]["status"], "inconclusive_no_absolute_bound_declared")
        self.assertFalse(gates["observed_free_force_residuals"]["absolute_force_error_bound"])
        self.assertEqual(gates["paper_damage_area_fraction_difference"]["status"], "not_enabled")
        measured["max_free_force_residual_limit_N"] = 0.30
        self.assertEqual(study._gate_pair(measured)["observed_free_force_residuals"]["status"], "pass")
        self.assertFalse(study._gate_pair(measured)["observed_free_force_residuals"]["absolute_force_error_bound"])
        measured["max_free_force_residual_limit_N"] = 0.20
        self.assertEqual(study._gate_pair(measured)["observed_free_force_residuals"]["status"],
                         "exceeds_declared_absolute_bound")

    def test_slower_load_is_separate_and_uses_normalized_phase_samples(self) -> None:
        base = fixtures.fake_run(self.root, "load-base", nx=2, dt=0.01, steps=4,
                                 hold=2, unload=4, relax=2, release=2)
        slower = fixtures.fake_run(self.root, "load-slower", nx=2, dt=0.01, steps=8,
                                   hold=2, unload=4, relax=2, release=2)
        admission = cc.analyze_pair(base, slower, "load_rate", list(study.PHASE_ORDER))
        self.assertEqual(admission["status"], "analyzed_descriptive")
        metrics = study._pair_metrics(cc._load_run(base), cc._load_run(slower), "load_rate")
        self.assertEqual(metrics["normalized_force_curve_rms"], 0.0)
        self.assertEqual(metrics["normalized_moment_curve_rms"], 0.0)

    def test_analyzer_rejects_a_prefix_or_failed_certificate(self) -> None:
        good = self._release_run("good")
        failed = self._release_run("failed", nx=4)
        csv_path = failed / "observations.csv"
        with csv_path.open(newline="") as source:
            reader = csv.DictReader(source)
            fields, rows = reader.fieldnames, list(reader)
        assert fields is not None
        rows[0]["status_code"] = "10"
        rows[0]["step_accepted"] = "0"
        with csv_path.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report = cc.analyze_pair(good, failed, "spatial", list(study.PHASE_ORDER))
        self.assertEqual(report["status"], "failed_run")
        self.assertEqual(report["reasons"][0]["code"], "step_rejected")

    def test_missing_enabled_damage_area_is_inconclusive(self) -> None:
        base = self._release_run("area-base")
        cand = self._release_run("area-candidate", nx=4)
        for path, enable in ((base, True), (cand, True)):
            result_path = path / "result.json"
            result = json.loads(result_path.read_text())
            result["resolution_features"] = {
                "paper_damage_area_enabled": enable,
                "glue_debond_area_enabled": False,
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        run_a, run_b = cc._load_run(base), cc._load_run(cand)
        metrics = study._pair_metrics(run_a, run_b, "spatial")
        gates = study._gate_pair(metrics)
        self.assertEqual(gates["paper_damage_area_fraction_difference"]["status"], "inconclusive_missing_metric")
        self.assertEqual(study._damage_areas(run_a).get("missing"), "damage_and_debond_area_fractions")

    def test_triple_gate_requires_decreasing_differences(self) -> None:
        coarse_middle = {
            "peak_force_relative": .04, "peak_moment_relative": .03,
            "path_work_relative": .04, "normalized_force_curve_rms": .08,
            "normalized_moment_curve_rms": .07, "springback_angle_difference_deg": 1.2,
            "paper_damage_area_fraction_difference": None,
            "debond_area_fraction_difference": None,
            "damage_debond": {"baseline": {"paper_enabled": False, "debond_enabled": False},
                              "candidate": {"paper_enabled": False, "debond_enabled": False}},
        }
        middle_fine = dict(coarse_middle)
        middle_fine.update({
            "peak_force_relative": .01, "peak_moment_relative": .01,
            "path_work_relative": .01, "normalized_force_curve_rms": .03,
            "normalized_moment_curve_rms": .02, "springback_angle_difference_deg": .4,
        })
        passed = study._gate_triple(coarse_middle, middle_fine)
        self.assertTrue(all(item["status"] in ("pass", "not_enabled") for item in passed.values()))
        middle_fine["normalized_force_curve_rms"] = .09
        failed = study._gate_triple(coarse_middle, middle_fine)
        self.assertEqual(failed["normalized_force_curve_rms"]["status"], "difference_not_decreasing")


if __name__ == "__main__":
    unittest.main()
