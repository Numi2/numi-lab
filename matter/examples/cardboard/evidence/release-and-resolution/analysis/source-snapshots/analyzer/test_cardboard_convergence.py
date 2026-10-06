#!/usr/bin/env python3
"""Synthetic reader tests for cardboard_convergence.py; no Matter or GPU runs."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).with_name("cardboard_convergence.py")
SPEC = importlib.util.spec_from_file_location("cardboard_convergence", TOOL)
assert SPEC and SPEC.loader
cc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cc
SPEC.loader.exec_module(cc)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def fake_run(root: Path, name: str, *, nx: int = 2, dt: float = 1.0,
             steps: int = 1, hold: int = 1, unload: int = 1, relax: int = 1,
             release: int = 0, phase_angle: float = 1.0, material_digest: str = "liner-v1",
             geometry_length: float = 1.0, plastic_liner: float = 0.0,
             plastic_medium: float = 3e-6, moment_scale: float = 1.0,
             raw_status: int = 0, accepted_flag: int = 1,
             release_constraint_error: bool = False, omit_step_accepted: bool = False) -> Path:
    path = root / name
    path.mkdir()
    phases = (steps, hold, unload, relax, release)
    total = sum(phases)
    geo = {
        "source_dimensions": "synthetic only", "initial_condition": "stress-free synthetic mesh",
        "equation": "synthetic tetrahedra", "length_m": geometry_length, "width_m": 1.0,
        "total_height_m": 1.0, "pitch_m": 1.0, "liner_thickness_m": 0.1,
        "medium_normal_thickness_m": 0.1, "medium_centerline_height_m": 0.4,
        "medium_present": True, "glue_minimum_gap_m": 0.01, "bond_width_m": 0.1,
        "upper_glue_gap_m": 0.01, "upper_bond_width_m": 0.1, "contact_tie_assumption": "test only",
    }
    cells = [
        {"index": 0, "name": "hajali2009_liner_hill_ideal", "tetrahedra": 1,
         "volume_m3": 1.0/6.0, "mass_kg": 1.0/6.0},
        {"index": 1, "name": "hajali2009_medium_hill_ideal", "tetrahedra": 1,
         "volume_m3": 2.0/6.0, "mass_kg": 2.0/6.0},
        {"index": 2, "name": "starch2007_finite_glue_elastic", "tetrahedra": 1,
         "volume_m3": 0.5/6.0, "mass_kg": 1.5/6.0},
    ]
    solver = {
        "backend": "implicit nonlinear Matter FEM on Apple Metal", "deformable_self_contact": True,
        "contact_slop_m": 0.001, "dt_s": dt, "bend_angle_deg": phase_angle,
        "steps": steps, "hold_steps": hold, "unload_steps": unload, "relax_steps": relax,
        "release_steps": release,
        "release_policy": "right grip free; left grip remains fixed; physical state preserved",
        "local_material_newton_iterations": 16, "newton_iteration_budget": 14,
        "fgmres_restart": 10, "fgmres_iteration_budget": 32, "line_search_steps": 8,
        "relative_residual_tolerance": 1e-4, "volume_tolerance": 1e-4,
        "pressure_tolerance": 1e-4, "transport_tolerance": 1e-4,
        "maximum_rate_exponent": 0, "runtime_execution": True,
    }
    material = {"liner_sha256": material_digest, "medium_sha256": "medium-v1", "glue_sha256": "glue-v1",
                "regional_map_sha256": "region-v1", "frame_map_sha256": "frames-v1", "cells": cells}
    manifest = {
        "schema": cc.MANIFEST_SCHEMA, "owner": "Numi Matter FEM runtime", "preset": "test-only",
        "geometry": geo, "glue_footprints": [{"upper": False, "center_x_m": .5, "requested_width_m": .1,
                                                "actual_width_m": .1, "cross_section_area_m2": .01}],
        "mesh": {"nodes": 12, "tetrahedra": 3, "fixed_grip_nodes": 6, "nx_per_pitch": nx,
                 "ny": 1, "thickness_slices": 1, "free_nodes": 6,
                 "min_extent_m": [0, 0, 0], "max_extent_m": [1, 1, 1],
                 "shared_node_total_mass_relative_error": 0},
        "materials": material, "solver": solver, "compiled_world_fingerprint": 1234,
        "evidence_boundary": "synthetic reader fixture; no physical claim",
    }
    write_json(path / "manifest.json", manifest)
    write_json(path / "result.json", {
        "schema": cc.RESULT_SCHEMA, "status": "completed", "accepted_steps": total,
        "compiled_world_fingerprint": 1234, "nodes": 12, "tetrahedra": 3,
        "material_map_sha256": "region-v1", "frame_map_sha256": "frames-v1",
        "failure": "", "physical_validation": False,
    })
    # Each material has a separate right tetrahedron with reference volumes
    # 1/6, 2/6, and 0.5/6, making the aggregate RMS exercise volume weighting.
    points = []
    tets = []
    for material_index, scale in enumerate((1.0, 2.0, 0.5)):
        start = len(points)
        # Stretch only x to create the requested determinant without degeneracy.
        points.extend([[0, 0, 0], [scale, 0, 0], [0, 1, 0], [0, 0, 1]])
        tets.append([start, start+1, start+2, start+3])
    # The states hold the six named plastic strains followed by scratch values.
    state_names = ["ep11", "ep22", "ep33", "ep23", "ep13", "ep12", "sbar11", "sbar22",
                   "sbar33", "sbar23", "sbar13", "sbar12", "accumulated_multiplier"]
    mesh = {
        "schema": cc.MESH_SCHEMA, "units": "metres", "geometry_equation": "synthetic",
        "medium_present": True, "material_densities_kg_m3": [1.0, 1.0, 3.0],
        "nodes_m": points, "tetrahedra": tets, "material_indices": [0, 1, 2],
        "material_frames_xyzw": [[0, 0, 0, 1]] * 3,
        "fixed_nodes": [0, 1, 2, 3, 4, 5], "left_grip_nodes": [0, 1, 2],
        "right_grip_nodes": [3, 4, 5],
    }
    write_json(path / "mesh.json", mesh)
    headers = ["step", "time_s", "environment", "arm", "target_angle_deg", "status_code",
               "reaction_x_N", "reaction_y_N", "reaction_z_N", "reaction_moment_y_Nm", "phase",
               "max_free_displacement_m", "max_free_speed_m_s", "kinetic_energy_J", "certificate_residual",
               "certificate_volume", "certificate_pressure", "certificate_raw_accepted_flag"]
    if not omit_step_accepted:
        headers.append("step_accepted")
    headers += ["right_grip_constrained", "measured_right_grip_angle_deg", "current_fixed_nodes"]
    rows = []
    phase_counts = dict(zip(cc.PHASE_ORDER, phases))
    cursor = 0
    schedule = []
    for phase in cc.PHASE_ORDER:
        for local in range(phase_counts[phase]):
            if phase == "loading":
                angle = phase_angle * (local + 1) / max(steps, 1)
            elif phase == "hold":
                angle = phase_angle
            elif phase == "unloading":
                angle = phase_angle * (1 - (local + 1) / max(unload, 1))
            elif phase == "release":
                angle = 0.0
            else:
                angle = 0.0
            schedule.append((cursor, phase, angle))
            cursor += 1
    phase_names = {"liner": "hajali2009_liner_hill_ideal", "medium": "hajali2009_medium_hill_ideal",
                   "glue": "starch2007_finite_glue_elastic"}
    paper_state_names = [
        {"index": 0, "name": phase_names["liner"], "state_names": state_names},
        {"index": 1, "name": phase_names["medium"], "state_names": state_names},
        {"index": 2, "name": phase_names["glue"], "state_names": []},
    ]
    end_steps = []
    cursor = 0
    for phase in cc.PHASE_ORDER:
        cursor += phase_counts[phase]
        if phase_counts[phase]:
            end_steps.append((phase, cursor))
    for step, phase, angle in schedule:
        for env in (0, 1):
            row = {
                "step": step, "time_s": (step + 1) * dt, "environment": env,
                "arm": "bent" if env == 0 else "held_reference",
                "target_angle_deg": angle if env == 0 else 0.0,
                "status_code": raw_status,
                "reaction_x_N": -.1 * moment_scale * angle if env == 0 else 0.0,
                "reaction_y_N": 0.0, "reaction_z_N": -.2 * angle if env == 0 else 0.0,
                "reaction_moment_y_Nm": -moment_scale * angle if env == 0 else 0.0,
                "phase": phase, "max_free_displacement_m": angle * 1e-4,
                "max_free_speed_m_s": .01 * angle, "kinetic_energy_J": 1e-9 * (step + 1),
                "certificate_residual": 1e-5, "certificate_volume": 0.0,
                "certificate_pressure": 0.0, "certificate_raw_accepted_flag": accepted_flag,
                "right_grip_constrained": 0 if phase == "release" else 1,
                "measured_right_grip_angle_deg": (0.25 if env == 0 and phase == "release" else (angle if env == 0 else 0.0)),
                "current_fixed_nodes": 3 if phase == "release" else 6,
            }
            if release_constraint_error and phase == "release":
                row["right_grip_constrained"] = 1
            if not omit_step_accepted:
                row["step_accepted"] = 1
            rows.append(row)
    with (path / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    state_phases = {phase: end for phase, end in end_steps}
    for phase, end in state_phases.items():
        for env, arm in ((0, "bent"), (1, "held_reference")):
            amount = plastic_liner if env == 0 else 0.0
            medium_amount = plastic_medium if env == 0 else 0.0
            tet_states = [
                [amount, 0, 0, 0, 0, 0] + [0] * 7,
                [medium_amount, 0, 0, 0, 0, 0] + [0] * 7,
                [],
            ]
            state = {
                "schema": cc.STATE_SCHEMA, "environment": env, "status_code": 0,
                "solver_certificate_raw_accepted_flag": 1,
                "materials": paper_state_names,
                "tetrahedra": [{"index": i, "material_index": i, "state": tet_states[i]} for i in range(3)],
            }
            write_json(path / f"material_state_{end:06d}_{arm}.json", state)
    return path


class CardboardConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_spatial_summary_volume_weighted_plastic_and_curves(self) -> None:
        a = fake_run(self.root, "coarse", nx=2, plastic_liner=0, plastic_medium=3e-6)
        b = fake_run(self.root, "fine", nx=4, plastic_liner=0, plastic_medium=3e-6, moment_scale=1.05)
        result = cc.analyze_pair(a, b, "spatial", ["loading", "relaxation"], .10)
        self.assertEqual(result["status"], "analyzed_descriptive")
        self.assertFalse(result["physical_validation"])
        self.assertEqual(result["convergence_verdict"].split(";")[0], "not_assessed")
        total = result["baseline_run"]["phase_endpoints"]["relaxation"]["bent"]["plastic_strain"]["paper_total"]
        self.assertAlmostEqual(total["reference_volume_m3"], .5, places=10)
        # The medium occupies twice the liner reference volume, so the total
        # RMS is sqrt((2/3) * ep^2), not an unweighted region average.
        self.assertAlmostEqual(total["volume_rms_frobenius_ep"], math.sqrt(2 / 3) * 3e-6, places=12)
        self.assertIn("reaction_moment_y_Nm", result["reaction_dynamics_and_energy_curves"]["loading"]["bent"])
        screen = result["endpoint_resolution_screen"]
        self.assertEqual(screen["status"], "within_declared_thresholds")
        self.assertAlmostEqual(screen["metrics_by_phase"]["relaxation"]["plastic_volume_rms_frobenius_ep"]["relative_difference_with_floor"], 0.0)

    def test_failed_status_not_overridden_by_raw_accepted_flag(self) -> None:
        a = fake_run(self.root, "failed", raw_status=10, accepted_flag=1)
        b = fake_run(self.root, "control", nx=4)
        result = cc.analyze_pair(a, b, "spatial")
        self.assertEqual(result["status"], "failed_run")
        self.assertEqual(result["reasons"][0]["code"], "step_rejected")
        self.assertNotIn("reaction_dynamics_and_energy_curves", result)

    def test_raw_flag_without_step_acceptance_is_inconclusive(self) -> None:
        a = fake_run(self.root, "raw-only", omit_step_accepted=True)
        b = fake_run(self.root, "fine", nx=4)
        # The only remaining acceptance field is intentionally removed to
        # emulate a legacy raw-accepted-only export.
        csv_path = a / "observations.csv"
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.pop("certificate_accepted", None)
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "step_accepted"])
            writer.writeheader()
            writer.writerows(rows)
        result = cc.analyze_pair(a, b, "spatial")
        self.assertEqual(result["status"], "inconclusive_evidence")
        self.assertEqual(result["reasons"][0]["code"], "missing_step_acceptance")

    def test_material_geometry_duration_and_sampling_mismatches_reject(self) -> None:
        baseline = fake_run(self.root, "base")
        material = fake_run(self.root, "material", nx=4, material_digest="other")
        result = cc.analyze_pair(baseline, material, "spatial")
        self.assertEqual(result["status"], "incompatible_pair")
        self.assertEqual(result["reasons"][0]["code"], "material_source_mismatch")
        geometry = fake_run(self.root, "geometry", nx=4, geometry_length=1.1)
        self.assertEqual(cc.analyze_pair(baseline, geometry, "spatial")["reasons"][0]["code"], "geometry_mismatch")
        wrong_time = fake_run(self.root, "duration", nx=4, dt=2.0)
        self.assertEqual(cc.analyze_pair(baseline, wrong_time, "spatial")["reasons"][0]["code"], "timestep_mismatch")
        no_nest = fake_run(self.root, "no-nest", nx=2, dt=.75, steps=4, hold=4, unload=4, relax=4)
        coarse = fake_run(self.root, "coarse-time", nx=2, dt=1.0, steps=3, hold=3, unload=3, relax=3)
        self.assertEqual(cc.analyze_pair(coarse, no_nest, "temporal")["reasons"][0]["code"], "incommensurate_time_sampling")
        wrong_duration = fake_run(self.root, "wrong-duration", nx=2, dt=.5, steps=5, hold=5, unload=5, relax=5)
        self.assertEqual(cc.analyze_pair(coarse, wrong_duration, "temporal")["reasons"][0]["code"], "phase_duration_mismatch")

    def test_enabled_tool_loading_is_explicitly_unsupported(self) -> None:
        baseline = fake_run(self.root, "baseline")
        explicit_bend = fake_run(self.root, "explicit-bend", nx=4)
        manifest_path = explicit_bend / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tooling"] = {"enabled": False}
        write_json(manifest_path, manifest)
        self.assertEqual(cc.analyze_pair(baseline, explicit_bend, "spatial")["status"], "analyzed_descriptive")
        tool_loaded = fake_run(self.root, "tool-loaded", nx=4)
        manifest_path = tool_loaded / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tooling"] = {"enabled": True, "mode": "crease", "indentation_m": 0.001}
        write_json(manifest_path, manifest)
        result = cc.analyze_pair(baseline, tool_loaded, "spatial")
        self.assertEqual(result["status"], "inconclusive_evidence")
        self.assertEqual(result["reasons"][0]["code"], "unsupported_loading_mode")

    def test_nonfinite_endpoint_thresholds_reject(self) -> None:
        baseline = fake_run(self.root, "baseline")
        candidate = fake_run(self.root, "candidate", nx=4)
        for tolerance, plastic_floor, moment_floor in (
            (math.nan, 1e-8, 1e-6), (.10, math.nan, 1e-6), (.10, 1e-8, math.inf),
        ):
            with self.subTest(tolerance=tolerance, plastic_floor=plastic_floor, moment_floor=moment_floor):
                result = cc.analyze_pair(baseline, candidate, "spatial", ["loading"],
                                         tolerance, plastic_floor, moment_floor)
                self.assertEqual(result["status"], "inconclusive_evidence")
                self.assertEqual(result["reasons"][0]["code"], "invalid_screen_threshold")

    def test_cli_output_refuses_to_overwrite_existing_report(self) -> None:
        baseline = fake_run(self.root, "baseline")
        candidate = fake_run(self.root, "candidate", nx=4)
        output = self.root / "report.json"
        command = [sys.executable, str(TOOL), "--kind", "spatial", "--baseline", str(baseline),
                   "--candidate", str(candidate), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(second.returncode, 2)
        self.assertIn("refusing to overwrite", second.stderr)
        self.assertEqual(output.read_bytes(), original)

    def test_temporal_nested_samples_and_load_rate_are_explicit_axes(self) -> None:
        coarse = fake_run(self.root, "coarse", nx=2, dt=1.0, steps=2, hold=2, unload=2, relax=2)
        fine = fake_run(self.root, "fine", nx=2, dt=.5, steps=4, hold=4, unload=4, relax=4)
        temporal = cc.analyze_pair(coarse, fine, "temporal")
        self.assertEqual(temporal["status"], "analyzed_descriptive")
        slower = fake_run(self.root, "slower", nx=2, dt=1.0, steps=4, hold=2, unload=2, relax=2)
        rate = cc.analyze_pair(coarse, slower, "load_rate")
        self.assertEqual(rate["status"], "analyzed_descriptive")
        self.assertTrue(rate["comparison_contract"]["duration_difference_is_declared_axis"])

    def test_free_release_is_measured_and_not_confused_with_clamped_relaxation(self) -> None:
        clamped = fake_run(self.root, "clamped", release=0)
        released = fake_run(self.root, "released", release=2, steps=1, hold=1, unload=0, relax=0)
        missing = cc.analyze_pair(clamped, fake_run(self.root, "clamped2", nx=4), "spatial", ["release"])
        self.assertEqual(missing["status"], "inconclusive_evidence")
        self.assertEqual(missing["reasons"][0]["code"], "inconclusive_missing_free_release")
        # Compare two release runs at different time resolutions but the same
        # physical duration and verify the reader checks the actual boundary.
        fine_release = fake_run(self.root, "released-fine", nx=2, release=4, dt=.5,
                                steps=2, hold=2, unload=0, relax=0)
        result = cc.analyze_pair(released, fine_release, "temporal", ["release"])
        self.assertEqual(result["status"], "analyzed_descriptive")
        response = result["baseline_run"]["free_release_response"]
        self.assertTrue(response["boundary_verified"])
        self.assertEqual(response["pre_release_phase"], "hold")
        self.assertAlmostEqual(response["right_grip_angle_change_deg"], -.75)
        self.assertAlmostEqual(response["after_release_endpoint"]["measured_right_grip_angle_deg"], .25)
        bad = fake_run(self.root, "release-still-clamped", release=2, steps=1, hold=1,
                       unload=0, relax=0, release_constraint_error=True)
        rejected = cc.analyze_pair(released, bad, "temporal", ["release"])
        self.assertEqual(rejected["status"], "inconclusive_evidence")
        self.assertEqual(rejected["reasons"][0]["code"], "release_boundary_mismatch")

    def test_single_run_release_report_binds_observation_and_state_inputs(self) -> None:
        released = fake_run(self.root, "released", release=2, steps=1, hold=1, unload=0, relax=0)
        report = cc.analyze_run(released, ["hold", "release"])
        self.assertEqual(report["status"], "analyzed_descriptive")
        self.assertFalse(report["physical_validation"])
        self.assertEqual(len(report["analyzer_sha256"]), 64)
        evidence = report["run_evidence"]
        hashes = evidence["input_sha256"]
        self.assertIn("observations.csv", hashes)
        self.assertIn("material_state_000004_bent.json", hashes)
        self.assertIn("material_state_000004_held_reference.json", hashes)
        self.assertTrue(evidence["free_release_response"]["boundary_verified"])
        self.assertEqual(len(report["reaction_dynamics_and_energy_curves"]["release"]["bent"]), 2)
        missing = cc.analyze_run(fake_run(self.root, "clamped"), ["release"])
        self.assertEqual(missing["status"], "inconclusive_evidence")
        self.assertEqual(missing["reasons"][0]["code"], "inconclusive_missing_free_release")

    def test_unknown_schema_and_missing_endpoint_are_inconclusive(self) -> None:
        a = fake_run(self.root, "a")
        b = fake_run(self.root, "b", nx=4)
        m = json.loads((b / "manifest.json").read_text())
        m["schema"] = "future.unknown"
        write_json(b / "manifest.json", m)
        self.assertEqual(cc.analyze_pair(a, b, "spatial")["reasons"][0]["code"], "unsupported_manifest_schema")
        c = fake_run(self.root, "c", nx=4)
        (c / "material_state_000004_bent.json").unlink()
        result = cc.analyze_pair(a, c, "spatial", ["relaxation"])
        self.assertEqual(result["status"], "inconclusive_evidence")
        self.assertEqual(result["reasons"][0]["code"], "unreadable_json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
