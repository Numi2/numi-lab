"""Arithmetic/admission regression tests, not physiological qualification."""
import math
import csv
import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from metalrobo.science_notebook import validate as validate_plan
from metalrobo.science_notebook import validate_calibration
from metalrobo.science_notebook import register as register_plan

import resting_intervention_study as adapter
from resting_intervention_study import (complete_breath_metrics, positive_linear_area,
                                       native_scene_command, native_scene_summary, native_body_trace_consistency,
                                       observation, TRACE_COLUMNS)


class AcceptedBreathWindowTests(unittest.TestCase):
    @staticmethod
    def trace():
        rows = []
        for index in range(501):
            # Four-second sine-flow cycle. Exact-zero samples precede the
            # owner's strictly-positive inspiration transition.
            phase = index % 80
            flow = 0.0 if phase in (0, 40) else 500 * math.sin(2 * math.pi * phase / 80)
            breaths = max(0, (index - 1) // 80)
            rows.append({'time_s': index * .05, 'airflow_ml_s': flow, 'breaths': breaths})
        return rows

    def test_complete_breath_ventilation_is_independent_of_partial_edges(self):
        rows = self.trace()
        first = complete_breath_metrics(rows, .2, 19.8)
        shifted = complete_breath_metrics(rows, 3.5, 21.7)
        self.assertTrue(first['available'] and shifted['available'])
        self.assertEqual(first['window_s'], [4.0, 16.0])
        self.assertEqual(shifted['window_s'], [4.0, 20.0])
        self.assertEqual(first['complete_breath_count'], 3)
        self.assertEqual(shifted['complete_breath_count'], 4)
        expected_l_min = (2000 / math.pi) * 60 / (4 * 1000)
        self.assertAlmostEqual(first['inspiratory_minute_ventilation_L_min'], expected_l_min, delta=.006)
        self.assertAlmostEqual(first['inspiratory_minute_ventilation_L_min'],
                               shifted['inspiratory_minute_ventilation_L_min'], places=12)
        self.assertEqual(first['respiratory_rate_per_min'], 15)

    def test_partial_breath_window_is_unavailable_not_zero_ventilation(self):
        value = complete_breath_metrics(self.trace(), 1, 3)
        self.assertFalse(value['available'])
        self.assertNotIn('inspiratory_minute_ventilation_L_min', value)

    def test_skipped_accepted_breath_transition_is_rejected(self):
        rows = self.trace()
        rows[100]['breaths'] += 2
        with self.assertRaisesRegex(ValueError, 'does not resolve'):
            complete_breath_metrics(rows, 0, 20)

    def test_positive_flow_area_includes_only_the_correct_triangle(self):
        self.assertEqual(positive_linear_area(-2, 2, 4), 2)
        self.assertEqual(positive_linear_area(2, -2, 4), 2)
        self.assertEqual(positive_linear_area(-2, -1, 4), 0)

    def test_gpu_event_ledger_resolves_events_missed_by_sampled_flow(self):
        # Coarse samples are all expiratory; positive excursions happened
        # between them. The second retained event skips an additional event.
        rows = []
        for count, sample_time, event_time, volume in (
                (0, 0.0, 0.0, 0.0), (1, 5.2, 5.0, 500.0),
                (3, 15.2, 15.0, 1500.0), (4, 20.2, 20.0, 2000.0)):
            rows.append({"time_s": sample_time, "airflow_ml_s": -10.0, "breaths": count,
                         "last_inspiration_step": event_time * 500,
                         "last_inspiration_time_s": event_time,
                         "last_inspiration_volume_accum_ml": volume,
                         "inspired_volume_accum_ml": volume,
                         "last_complete_breath_inspired_ml": 500.0})
        measured = complete_breath_metrics(rows, 1, 21)
        self.assertEqual(measured["complete_breath_count"], 3)
        self.assertEqual(measured["events_between_retained_samples"], 1)
        self.assertEqual(measured["respiratory_rate_per_min"], 12)
        self.assertEqual(measured["inspiratory_minute_ventilation_L_min"], 6)
        rows[2]["last_inspiration_time_s"] = 25.0
        with self.assertRaisesRegex(ValueError, "invalid accepted breath event ledger"):
            complete_breath_metrics(rows, 1, 21)


class RestingReferenceComparisonTests(unittest.TestCase):
    @staticmethod
    def trace():
        rows = []
        for breath in AcceptedBreathWindowTests.trace():
            row = {key: 0.0 for key in TRACE_COLUMNS}
            row.update(breath)
            row.update(PaCO2_mmhg=50.0, PaO2_mmhg=102.0, SaO2=.98,
                       aorta_mmhg=90.0, pulmonary_artery_mmhg=16.0,
                       last_lv_stroke_ml=70.0, tidal_ml=500.0,
                       aortic_ejected_ml=80 * row["time_s"],
                       pulmonary_ejected_ml=80 * row["time_s"],
                       complete_filling_ejection_cycles=math.floor(row["time_s"] / .8))
            rows.append(row)
        return rows

    def test_reference_outliers_are_reported_without_numerical_rejection(self):
        result = adapter.resting_reference_comparison(self.trace(), 1, 21)
        comparisons = result["general_adult_resting_reference_comparisons"]
        self.assertFalse(comparisons["mean_PaCO2_mmhg"]["within_reference_bounds"])
        self.assertFalse(comparisons["mean_PaO2_mmhg"]["within_reference_bounds"])
        self.assertEqual(comparisons["aortic_output_L_min"]["measured"], 4.8)
        self.assertEqual(comparisons["complete_breath_rate_per_min"]["measured"], 15)
        self.assertNotIn("reference_bounds", result["supine_male_cohort_context"]["values"]["tidal_volume_L"])

    def test_unresolved_complete_cycles_are_unavailable_not_zero(self):
        rows = self.trace()
        for row in rows:
            row["breaths"] = row["complete_filling_ejection_cycles"] = 0
        result = adapter.resting_reference_comparison(rows, 1, 21)
        self.assertNotIn("complete_breath_rate_per_min", result["general_adult_resting_reference_comparisons"])
        self.assertNotIn("complete_heartbeat_rate_per_min", result["general_adult_resting_reference_comparisons"])
        self.assertIsNone(result["supine_male_cohort_context"]["values"]["minute_ventilation_L_min"]["measured"])


class NativeRespiratoryMechanicalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parameters = self.root / 'parameters.json'
        self.config = {'frc_m3':.0025,'diaphragm_area_m2':.02,'rib_effective_area_m2':.06,
                       'airway_resistance_pa_s_per_m3':100000.,'lung_compliance_m3_per_pa':.000002,
                       'rest_pleural_pressure_pa':-500.}
        self.parameters.write_text(json.dumps(self.config))
        self.trace = self.root / 'mechanical.csv'
        self.rows = []
        for i in range(128):
            pressure = -20. if i%2 else 20.
            self.rows.append({'time_s':i*.1,'lung_volume_ml':2800.,'airflow_ml_s':-pressure*10,
                'alveolar_pa':pressure,'pleural_pa':-650.+pressure,'diaphragm_mm':10.,'rib_mm':5/3,
                'diaphragm_excitation':.1,'intercostal_excitation':.1,
                'diaphragm_activation':.09,'intercostal_activation':.09})

    def run_check(self):
        with self.trace.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.rows[0]);writer.writeheader();writer.writerows(self.rows)
        return adapter.native_respiration_trace_consistency(self.trace,self.parameters,{'test':[0.,12.8]})['respiratory_mechanics']

    def test_known_unit_conversion_and_both_flow_directions(self):
        result=self.run_check()
        self.assertLess(max(result['maximum_absolute_identity_residuals'].values()),1e-9)
        self.assertEqual(result['windows']['test']['ranges']['airflow_ml_s'],[-200.,200.])
        self.assertAlmostEqual(result['windows']['test']['means']['diaphragm_activation'],.09)

    def test_wrong_units_stale_area_or_invalid_activation_are_rejected(self):
        for key,value in [('diaphragm_mm',.01),('alveolar_pa',200.),('pleural_pa',-6.3),
                          ('diaphragm_activation',1.01)]:
            before=self.rows[63][key]
            with self.subTest(key=key):
                self.rows[63][key]=value
                with self.assertRaises(ValueError):self.run_check()
                self.rows[63][key]=before
        self.config['diaphragm_area_m2']=.021
        self.parameters.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,'volume_decomposition'):self.run_check()


class NativeSceneBindingTests(unittest.TestCase):
    def invocation(self):
        return {"argv": ["/build/bin/numi-human-native", "/rigid", "/myo", "/bones", "/old-output",
                         "--persistent-metal-stand", "--resting-scene", "/network", "/respiration",
                         "--vascular-dense45", "--resting-anatomy-receipt", "/anatomy",
                         "--skin-payload", "/skin", "--tendon-payload", "/tendon",
                         "--muscle-step-count", "64", "--muscle-step-seconds", ".001",
                         "--resting-movie", "/old-output/native-viewer.mov"],
                "asset_sha256": {p: "bound" for p in ("/build/bin/numi-human-native", "/rigid", "/myo",
                    "/bones", "/network", "/respiration", "/anatomy", "/skin", "/tendon",
                    "/build/lib/libmetalrobo.dylib", "/build/shaders/MetalRobo.metallib",
                    "/build/shaders/MetalRoboHyperPolicy.metallib", "/build/shaders/NumiNeuron.metallib",
                    "/build/matter/shaders/HumanRespiration.metallib", "/build/matter/shaders/NumiMatter.metallib",
                    "/build/matter/shaders/NumiMatterPhysicalStateDigest.metallib")}}

    def args(self):
        return Namespace(steps=155000, dt=.002, arm="treatment", start_s=120., end_s=180., scale=.5)

    def test_rebinding_changes_only_output_timing_and_declared_intervention(self):
        original = self.invocation()
        command = native_scene_command(original, Path("/new"), self.args())
        self.assertEqual(command[4], "/new")
        self.assertEqual(command[command.index("--muscle-step-count") + 1], "155000")
        self.assertEqual(command[-4:], ["--resting-drive-intervention", "120.0", "180.0", "0.5"])
        self.assertEqual(original["argv"][4], "/old-output")

    def test_unbound_consumed_anatomy_and_preintervened_reference_are_rejected(self):
        original = self.invocation()
        del original["asset_sha256"]["/anatomy"]
        with self.assertRaisesRegex(ValueError, "unbound file"):
            native_scene_command(original, Path("/new"), self.args())
        original = self.invocation()
        original["argv"].extend(["--resting-drive-intervention", "1", "2", ".5"])
        with self.assertRaisesRegex(ValueError, "no intervention"):
            native_scene_command(original, Path("/new"), self.args())

    def test_implicit_runtime_dependencies_must_be_preregistered(self):
        for path in ("/build/lib/libmetalrobo.dylib", "/build/shaders/MetalRobo.metallib",
                     "/build/shaders/MetalRoboHyperPolicy.metallib", "/build/shaders/NumiNeuron.metallib",
                     "/build/matter/shaders/HumanRespiration.metallib", "/build/matter/shaders/NumiMatter.metallib",
                     "/build/matter/shaders/NumiMatterPhysicalStateDigest.metallib"):
            with self.subTest(path=path):
                original = self.invocation()
                del original["asset_sha256"][path]
                with self.assertRaisesRegex(ValueError, "unbound runtime dependency"):
                    native_scene_command(original, Path("/new"), self.args())

    def test_incomplete_and_assisted_native_logs_are_rejected(self):
        log = ('runtime=Numi Matter runtime initialized with eligible dense45 vascular solve '
               'device=Apple M4 Pro world_fingerprint=123 timestep_s=.001\n'
               'resting_body_source_fingerprint=456 coupled_program_fingerprint=789\n'
               'stand_terminal_state={"step_count":32,"root_assistance":false,"q":[0],"v":[0]}\n'
               'resting_integrated_body=completed simulated_s=.032 wall_s=1 real_time_factor=.032 '
               'physiology_body_clock=matched root_assistance=false presentation_qualification=pending\n')
        result = native_scene_summary(log)
        self.assertEqual(result['accepted_steps'], 32)
        self.assertEqual(result['world_fingerprint'], '123')
        self.assertEqual(result['coupled_program_fingerprint'], '789')
        with self.assertRaisesRegex(ValueError, 'program identity'):
            native_scene_summary(log.replace('coupled_program_fingerprint=789', 'missing_program=789'))
        with self.assertRaisesRegex(ValueError, 'complete'):
            native_scene_summary(log.replace('physiology_body_clock=matched', 'physiology_body_clock=failed'))
        with self.assertRaisesRegex(ValueError, 'root assistance'):
            native_scene_summary(log.replace('"root_assistance":false', '"root_assistance":true'))

    def test_intermediate_assistance_cannot_be_hidden_by_an_unassisted_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "body.csv"
            header = "step,time_s,root_assistance_n,root_assistance_nm,peak_penetration_m\n"
            rows = "32,.064,0,0,.00001\n64,.128,0,0,.00002\n96,.192,0,0,.00001\n"
            trace.write_text(header + rows)
            result = native_body_trace_consistency(trace, 96, .002)
            self.assertFalse(result['root_assistance_observed'])
            self.assertEqual(result['maximum_contact_penetration_m'], .00002)
            trace.write_text(header + rows.replace("64,.128,0,0", "64,.128,.001,0"))
            with self.assertRaisesRegex(ValueError, 'root assistance'):
                native_body_trace_consistency(trace, 96, .002)
            trace.write_text(header + rows.replace("64,.128,0,0,.00002\n", ""))
            with self.assertRaisesRegex(ValueError, 'skipped'):
                native_body_trace_consistency(trace, 96, .002)


class CommonCardiacSurfaceTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "surfaces.csv"
        fixture = Path(__file__).parent / "fixtures/resting-common-geometry-native-64-steps.csv"
        with fixture.open() as stream:
            reader = csv.DictReader(stream)
            self.columns = reader.fieldnames
            self.rows = list(reader)

    def check_trace(self):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        return adapter.native_surface_trace_consistency(self.path, 64, .002)

    def test_captured_common_coordinates_replace_only_inapplicable_legacy_fields(self):
        result = self.check_trace()
        self.assertEqual(result["geometry_mode"], "common_seven_coordinate_v1")
        self.assertEqual(result["common_cardiac_geometry"]["coordinate_count"], 7)
        self.assertFalse(result["common_cardiac_geometry"]["legacy_coordinates_and_closure_applicable"])
        material = result["ventricular_material"]
        self.assertTrue(material["bound"])
        self.assertFalse(material["legacy_wall_bound"])
        self.assertGreater(material["target_ml"], 0)
        self.assertIsNone(material["closure_range_mm"])
        self.assertEqual(result["whole_mesh_area_audit"]["triangles_checked_per_frame"], 4910160)
        self.assertFalse(result["whole_body_interfaces_qualified"])

    def test_active_common_geometry_failures_are_never_treated_as_nan_sentinels(self):
        cases = [
            ("geometry_mode", "unknown"),
            ("geometry_mode", "legacy_ventricular_wall_v2"),
            ("common_coordinate_solver_status", "5"),
            ("common_coordinate_solver_iterations", "-1"),
            ("common_coordinate_domain_box", "4294967295"),
            ("common_coordinate_normalized_residual", "nan"),
            ("common_coordinate_normalized_residual", "0.0001"),
            ("ventricular_closure_mm_applicable", "1"),
            ("ventricular_wall_bound", "1"),
            ("ventricular_material_ml", "nan"),
            ("ventricular_material_ml", "165"),
            ("ventricular_material_target_ml", "165"),
            ("ventricular_material_status", "2"),
            ("functional_geometry_status", "2"),
            ("q_ra", "inf"),
            ("ventricular_closure_mm", "0"),
            ("mesh_zero_area_triangles", "1"),
            ("mesh_nonfinite_area_triangles", "1"),
            ("mesh_triangles_checked", "0"),
            ("mesh_triangles_checked", "4910159"),
        ]
        cases.extend(("common_coordinate_" + key, "nan") for key in
                     ("RA", "RV", "LA", "LV", "RA_material", "ventricular_material", "LA_material"))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                before = self.rows[1][key]
                self.rows[1][key] = value
                with self.assertRaises(ValueError):
                    self.check_trace()
                self.rows[1][key] = before

    def test_incomplete_common_or_mesh_diagnostics_are_rejected(self):
        for key in ("common_coordinate_LA_material", "ventricular_material_status",
                    "ventricular_closure_mm_applicable", "mesh_triangles_checked"):
            with self.subTest(key=key):
                index = self.columns.index(key)
                self.columns.remove(key)
                values = [row.pop(key) for row in self.rows]
                with self.assertRaises(ValueError):
                    self.check_trace()
                self.columns.insert(index, key)
                for row, value in zip(self.rows, values):
                    row[key] = value


class NativeSurfaceTraceTests(unittest.TestCase):
    def test_ventricular_material_cannot_hide_an_intermediate_volume_or_binding_failure(self):
        columns = ("step,time_s,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,"
                   "max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,"
                   "la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml,"
                   "ventricular_wall_bound,ventricular_material_ml,ventricular_material_target_ml,ventricular_closure_mm\n")
        rows = [f"{step},{step*.002},0,0,0,0,0,0,0,0,40,120,50,120,0,0,2500,1,164,164,{closure}\n"
                for step, closure in ((0, -1), (31, .5), (63, 2))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surfaces.csv"
            path.write_text(columns + ''.join(rows))
            wall = adapter.native_surface_trace_consistency(path, 64, .002)['ventricular_material']
            self.assertTrue(wall['bound'])
            self.assertEqual(wall['closure_range_mm'], [-1, 2])
            self.assertFalse(wall['gpu_degenerate_triangle_check_recorded'])
            status_columns = columns.rstrip('\n') + ',ventricular_material_status\n'
            status_rows = [row.rstrip('\n') + ',0\n' for row in rows]
            path.write_text(status_columns + ''.join(status_rows))
            self.assertTrue(adapter.native_surface_trace_consistency(path, 64, .002)
                            ['ventricular_material']['gpu_degenerate_triangle_check_recorded'])
            status_rows[1] = status_rows[1][:-2] + '2\n'
            path.write_text(status_columns + ''.join(status_rows))
            with self.assertRaisesRegex(ValueError, 'degenerate ventricular triangle'):
                adapter.native_surface_trace_consistency(path, 64, .002)
            functional_columns = columns.rstrip('\n') + ',functional_geometry_status\n'
            functional_rows = [row.rstrip('\n') + ',0\n' for row in rows]
            path.write_text(functional_columns + ''.join(functional_rows))
            adapter.native_surface_trace_consistency(path, 64, .002)
            functional_rows[1] = functional_rows[1][:-2] + '2\n'
            path.write_text(functional_columns + ''.join(functional_rows))
            with self.assertRaisesRegex(ValueError, 'invalid functional triangles'):
                adapter.native_surface_trace_consistency(path, 64, .002)
            for header, middle in (
                (columns.replace('ventricular_closure_mm', 'missing_closure'), rows[1]),
                (columns, rows[1].replace(',1,164,164,', ',1,165,164,')),
                (columns, rows[1].replace(',1,164,164,', ',1,165,165,')),
                (columns, rows[1].replace(',1,164,164,', ',0,0,0,')),
                (columns, rows[1].replace(',164,164,', ',nan,164,')),
            ):
                path.write_text(header + rows[0] + middle + rows[2])
                with self.assertRaises(ValueError):
                    adapter.native_surface_trace_consistency(path, 64, .002)
            path.write_text(columns + ''.join(row[:row.rfind(',1,164,164,')] + ',0,0,0,0\n' for row in rows))
            self.assertFalse(adapter.native_surface_trace_consistency(path, 64, .002)['ventricular_material']['bound'])

    def test_body_observation_rejects_incomplete_nonfinite_or_changing_mass(self):
        columns = ("step,time_s,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,"
                   "max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,"
                   "la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml,"
                   "body_com_x_m,body_com_y_m,body_com_z_m,represented_body_mass_kg\n")
        rows = [f"{step},{step*.002},0,0,0,0,0,0,0,0,40,120,50,120,0,0,2500,{x},-.7,.1,72\n"
                for step, x in ((0, 0), (31, .01), (63, .02))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surfaces.csv"
            path.write_text(columns + ''.join(rows))
            body = adapter.native_surface_trace_consistency(path, 64, .002)['body_center_of_mass']
            self.assertEqual(body['represented_mass_kg'], 72)
            self.assertEqual(body['displacement_m'], [.02, 0, 0])
            for header, invalid in (
                (columns.replace('body_com_x_m', 'missing_com_x'), rows),
                (columns, [rows[0], rows[1].replace(',0.01,', ',nan,'), rows[2]]),
                (columns, [rows[0], rows[1].replace(',72\n', ',73\n'), rows[2]]),
                (columns, [row.replace(',72\n', ',0\n') for row in rows]),
            ):
                path.write_text(header + ''.join(invalid))
                with self.assertRaises(ValueError):
                    adapter.native_surface_trace_consistency(path, 64, .002)

    def test_displayed_clock_and_intermediate_invalid_geometry_are_not_hidden(self):
        columns = ("step,time_s,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,"
                   "max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,"
                   "la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml\n")
        rows = [f"{step},{step * .002},-.0001,0,0,.000001,0,0,0,0,40,120,50,120,0,0,2500\n"
                for step in (0, 31, 63, 95)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surfaces.csv"
            path.write_text(columns + ''.join(rows))
            result = adapter.native_surface_trace_consistency(path, 96, .002)
            self.assertEqual(result['displayed_accepted_frames'], 4)
            self.assertEqual(result['minimum_full_skin_bed_gap_m'], -.0001)
            self.assertFalse(result['whole_body_interfaces_qualified'])
            cases = (
                (rows[:2] + rows[3:], 'skipped'),
                (rows[:-1], 'final displayed'),
                ([row.replace('63,0.126', '63,0.128') for row in rows], 'clock differs'),
                ([row.replace('63,0.126,-.0001,0,0', '63,0.126,-.0001,1,0') for row in rows], 'invalid skin'),
                ([row.replace('63,0.126,-.0001,0,0,.000001', '63,0.126,-.0001,0,0,.001') for row in rows], 'tolerance'),
            )
            for invalid, message in cases:
                with self.subTest(message=message):
                    path.write_text(columns + ''.join(invalid))
                    with self.assertRaisesRegex(ValueError, message):
                        adapter.native_surface_trace_consistency(path, 96, .002)

    def test_partial_final_segment_is_checked_at_its_actual_displayed_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surfaces.csv"
            path.write_text("step,time_s,min_skin_bed_gap_m,vertices_below_1mm,nonfinite_skin_vertices,"
                            "max_functional_volume_relative_error,q_ra,q_rv,q_la,q_lv,ra_target_ml,rv_target_ml,"
                            "la_target_ml,lv_target_ml,diaphragm_swept_ml,rib_swept_ml,lung_target_ml\n" +
                            ''.join(f"{step},{step*.002},0,0,0,0,0,0,0,0,40,120,50,120,0,0,2500\n"
                                    for step in (0, 31, 34)))
            self.assertEqual(adapter.native_surface_trace_consistency(path, 35, .002)['displayed_accepted_frames'], 3)


class NativeV2PlanPreparationTests(unittest.TestCase):
    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def test_native_plan_uses_exact_pair_ids_and_prespecified_windows_without_registering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_names = ("bin/numi-human-native", "rigid.nhrigid", "muscle.nhmyo", "bones.nhbones",
                           "network.json", "respiration.json", "anatomy.json", "skin.nhskin",
                           "tendon.nhtendon", "tissue.nhtissue", "lib/libmetalrobo.dylib",
                           "shaders/MetalRobo.metallib", "shaders/MetalRoboHyperPolicy.metallib",
                           "shaders/NumiNeuron.metallib", "matter/shaders/HumanRespiration.metallib",
                           "matter/shaders/NumiMatter.metallib", "matter/shaders/NumiMatterPhysicalStateDigest.metallib")
            asset_hashes = {}
            for index, name in enumerate(asset_names):
                path = root / "assets" / name
                asset_hashes[str(path)] = self.write(path, f"test asset {index}".encode())
            source_path = root / "source" / "human.cpp"
            source_hashes = {str(source_path): self.write(source_path, b"frozen source fixture")}
            source_hash_path = root / "source-hashes.json"
            source_hash_path.write_text(json.dumps(source_hashes), encoding="utf-8")
            source_revisions_path = root / "source-revisions.json"
            source_revisions = {
                "numi-lab": {"revision": "a" * 40, "diff_sha256": "b" * 64},
                "numilab-human": {"revision": "c" * 40, "diff_sha256": "d" * 64},
                "numi-brain": {"revision": "e" * 40, "diff_sha256": "f" * 64},
            }
            source_revisions_path.write_text(json.dumps(source_revisions), encoding="utf-8")
            invocation_path = root / "invocation.json"
            invocation = {
                "argv": [str(root / "assets" / "bin" / "numi-human-native"),
                         str(root / "assets" / "rigid.nhrigid"), str(root / "assets" / "muscle.nhmyo"),
                         str(root / "assets" / "bones.nhbones"), str(root / "old-output"),
                         "--persistent-metal-stand", "--muscle-step-seconds", ".002",
                         "--muscle-step-count", "3000", "--support-contact-payload", str(root / "assets" / "skin.nhskin"),
                         "--joint-equality-payload", str(root / "assets" / "bones.nhbones"),
                         "--tendon-payload", str(root / "assets" / "tendon.nhtendon"),
                         "--resting-scene", str(root / "assets" / "network.json"), str(root / "assets" / "respiration.json"),
                         "--vascular-dense45", "--resting-anatomy-receipt", str(root / "assets" / "anatomy.json"),
                         "--skin-payload", str(root / "assets" / "skin.nhskin"),
                         "--soft-tissue-payload", str(root / "assets" / "tissue.nhtissue"),
                         "--resting-movie", str(root / "old-output" / "native-viewer.mov")],
                "asset_sha256": asset_hashes,
            }
            invocation_path.write_text(json.dumps(invocation), encoding="utf-8")
            fixture = root / "accepted-fixture.csv"
            fixture.write_text("calibration fixture reserved for the owner parser", encoding="utf-8")
            output = root / "registration-draft"
            repository = root / "owner"
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "-c", "user.name=Numi Test", "-c",
                            "user.email=numi-test@example.invalid", "commit", "-q", "--allow-empty", "-m",
                            "Temporary registration test"], check=True)
            args = Namespace(repository=str(repository), directory=str(output), invocation=str(invocation_path),
                             source_hashes=str(source_hash_path), source_revisions=str(source_revisions_path),
                             parser_fixture=str(fixture),
                             world_fingerprint="123456", control_program_fingerprint="456789",
                             treatment_program_fingerprint="987654", device="Apple M4 Pro",
                             steps=160000, dt=.002, start_s=60., end_s=100., scale=.5, window_s=12.)
            calibration = {"schema": "numi.science.calibration.v1", "status": "passed",
                           "checks": [{"id": "parser_fixture_only", "passed": True}],
                           "scope": "test double only", "observed_units": []}
            with patch.object(adapter, "known_parser_calibration", return_value=calibration):
                plan_path = adapter.prepare_native(args)
            plan = json.loads(plan_path.read_text())
            identity = json.loads((output / "native-build-identity.json").read_text())
            validate_plan(plan, live=False)
            calibration_report = json.loads((output / "calibration.json").read_text())
            artifact_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for path in plan["artifacts"]}
            validate_calibration(plan, artifact_hashes, calibration_report)
            self.assertEqual(set(calibration_report["bindings"]), set(plan["instrument"]["artifacts"]))
            fixture_key = str(fixture.resolve())
            broken = dict(calibration_report, bindings={fixture_key: artifact_hashes[fixture_key]})
            with self.assertRaisesRegex(ValueError, "does not bind the exact instrument"):
                validate_calibration(plan, artifact_hashes, broken)
            self.assertEqual(plan["schema"], "numi.science.plan.v2")
            self.assertEqual(plan["prediction"], {"estimand": "paired_difference_mean", "minimum": 0., "maximum": 40.})
            self.assertEqual(plan["design"]["pre_dose_window_s"], [48., 60.])
            self.assertEqual(plan["design"]["dose_window_s"], [88., 100.])
            self.assertEqual(plan["design"]["recovery_window_s"], [308., 320.])
            recovery = plan["secondary_predictions"]["late_recovery_equivalence"]
            self.assertEqual(recovery["comparison"], "treatment versus unchanged control, both averaged over the final 12 s")
            self.assertEqual(plan["secondary_predictions"]["late_recovery_equivalence"]["PaCO2_absolute_difference_max_mmhg"], 1.)
            self.assertEqual(plan["secondary_predictions"]["late_recovery_equivalence"]["inspiratory_minute_ventilation_relative_difference_max_fraction"], .1)
            self.assertEqual(plan["secondary_predictions"]["late_recovery_equivalence"]["PaO2_absolute_difference_max_mmhg"], 5.)
            control, treatment = plan["trials"]
            self.assertEqual(control["unit"], treatment["unit"])
            self.assertIn("--program-fingerprint", control["argv"])
            self.assertEqual(control["argv"][control["argv"].index("--program-fingerprint") + 1], "456789")
            self.assertEqual(treatment["argv"][treatment["argv"].index("--program-fingerprint") + 1], "987654")
            self.assertNotEqual(identity["coupled_program_fingerprint_by_arm"]["control"],
                                identity["coupled_program_fingerprint_by_arm"]["treatment"])
            self.assertEqual(identity["source_revisions"], source_revisions)
            self.assertTrue(all("whole_body_anatomy_qualified" not in condition.get("path", [])
                                for condition in plan["validity"]))
            self.assertFalse(any(["whole_body_anatomy_qualified"] in pair for pair in plan["paired_equal"]))
            self.assertEqual(len(plan["trials"]), 2)
            self.assertFalse((output / "study").exists())
            self.assertFalse((output / "registration").exists())
            # Exercise the real registration boundary; validation alone omits
            # exact calibration bindings and the Git owner identity lookup.
            self.assertEqual(plan["repository"], str(repository.resolve()))
            register_plan(plan_path, root / "test-study")

    def test_native_plan_rejects_same_arm_program_identity_and_stale_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            source_hashes = {str(source): self.write(source, b"source")}
            invocation_path = root / "invocation.json"
            executable = root / "bin" / "numi-human-native"
            binding = self.write(executable, b"native executable")
            invocation_path.write_text(json.dumps({"argv": [str(executable), "/rigid", "/myo", "/bones", "/out",
                                                       "--persistent-metal-stand", "--resting-scene", "/network", "/resp",
                                                       "--vascular-dense45", "--resting-anatomy-receipt", "/anatomy",
                                                       "--skin-payload", "/skin", "--tendon-payload", "/tendon",
                                                       "--muscle-step-count", "1", "--muscle-step-seconds", ".002",
                                                       "--resting-movie", "/out/movie"],
                                                   "asset_sha256": {str(executable): binding}}), encoding="utf-8")
            fixture = root / "fixture.csv"
            fixture.write_text("fixture", encoding="utf-8")
            source_hash_path = root / "sources.json"
            source_hash_path.write_text(json.dumps(source_hashes), encoding="utf-8")
            source_revisions_path = root / "source-revisions.json"
            source_revision_data = {
                "numi-lab": {"revision": "a" * 40, "diff_sha256": "b" * 64},
                "numilab-human": {"revision": "c" * 40, "diff_sha256": "d" * 64},
                "numi-brain": {"revision": "e" * 40, "diff_sha256": "f" * 64},
            }
            source_revisions_path.write_text(json.dumps(source_revision_data), encoding="utf-8")
            args = Namespace(directory=str(root / "out"), invocation=str(invocation_path),
                             source_hashes=str(source_hash_path), source_revisions=str(source_revisions_path),
                             parser_fixture=str(fixture),
                             world_fingerprint="123", control_program_fingerprint="456",
                             treatment_program_fingerprint="456", device="Apple M4 Pro",
                             steps=160000, dt=.002, start_s=60., end_s=100., scale=.5, window_s=30.)
            source_revisions_path.write_text(json.dumps({"numi-lab": source_revision_data["numi-lab"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must include numi-lab, numilab-human, and numi-brain"):
                adapter.native_plan_components(args, json.loads(invocation_path.read_text()),
                                               source_hashes, Path(adapter.__file__).resolve())
            source_revisions_path.write_text(json.dumps(source_revision_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct coupled-program"):
                adapter.native_plan_components(args, json.loads(invocation_path.read_text()),
                                               source_hashes, Path(adapter.__file__).resolve())
            args.treatment_program_fingerprint = "789"
            executable.write_bytes(b"changed native executable")
            with self.assertRaisesRegex(ValueError, "asset differs"):
                adapter.native_plan_components(args, json.loads(invocation_path.read_text()),
                                               source_hashes, Path(adapter.__file__).resolve())

    def test_observation_uses_final_accepted_window_for_recovery(self):
        # Distinct pre/dose/recovery plateaus catch accidentally reusing the
        # dose interval. This is parser arithmetic, not simulated physiology.
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'trace.csv'
            with trace.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=TRACE_COLUMNS)
                writer.writeheader()
                for index in range(2001):
                    t = index * .016
                    row = dict.fromkeys(TRACE_COLUMNS, 0.)
                    row.update(time_s=t, PaCO2_mmhg=40. if t < 12 else 44. if t < 24 else 41.)
                    writer.writerow(row)
            args = Namespace(steps=16000, dt=.002, window_s=8., start_s=12., end_s=24.,
                             unit_id='arithmetic-only', arm='treatment', scale=.5)
            native = dict(accepted_steps=16000, simulated_s=32., device='Apple M4 Pro',
                          world_fingerprint='123', vascular_dense45=True, brain_control=True,
                          real_time_factor=.1)
            result = observation(args, trace, native, 'fixture')
            self.assertEqual(result['primary_delta_PaCO2_mmhg'], 4.)
            self.assertEqual(result['recovery_window_s'], [24., 32.])
            self.assertEqual(result['PaCO2_recovery_mean_mmhg'], 41.)
            self.assertEqual(result['dose_to_recovery_PaCO2_change_mmhg'], -3.)



class Native310sPreparationTests(unittest.TestCase):
    @staticmethod
    def invocation():
        environment = dict(adapter.NATIVE_310S_REQUIRED_ENVIRONMENT)
        environment.update(adapter.NATIVE_310S_DISABLED_EXPERIMENTS)
        return {
            "qualification": "native execution receipt; physiological and anatomical acceptance require separate audits",
            "machine": "arm64", "system": "Darwin-26.6-arm64",
            "environment": environment,
            "argv": ["/native/build/bin/numi-human-native", "--muscle-step-seconds", "0.002",
                     "--muscle-step-count", "1000", "--resting-movie", "/preflight/native-viewer.mov"],
            "asset_sha256": {},
        }

    @staticmethod
    def write_surface_fixture(path, steps, dt, *, include_whole_mesh=True):
        columns = ["step", "time_s", "min_skin_bed_gap_m", "vertices_below_1mm",
                   "nonfinite_skin_vertices", "max_functional_volume_relative_error",
                   "q_ra", "q_rv", "q_la", "q_lv", "ra_target_ml", "rv_target_ml",
                   "la_target_ml", "lv_target_ml", "diaphragm_swept_ml", "rib_swept_ml",
                   "lung_target_ml", "functional_geometry_status"]
        if include_whole_mesh:
            columns += ["mesh_zero_area_triangles", "mesh_nonfinite_area_triangles",
                        "mesh_triangles_checked"]
        frame_interval = round(0.064 / dt)
        selected = [0, *[step for step in range(frame_interval - 1, steps, frame_interval) if step != 0]]
        if selected[-1] != steps - 1:
            selected.append(steps - 1)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for step in selected:
                row = {"step": step, "time_s": step * dt, "min_skin_bed_gap_m": 0.001,
                       "vertices_below_1mm": 0, "nonfinite_skin_vertices": 0,
                       "max_functional_volume_relative_error": 0, "q_ra": 1, "q_rv": 1,
                       "q_la": 1, "q_lv": 1, "ra_target_ml": 1, "rv_target_ml": 1,
                       "la_target_ml": 1, "lv_target_ml": 1, "diaphragm_swept_ml": 0,
                       "rib_swept_ml": 0, "lung_target_ml": 1, "functional_geometry_status": 0}
                if include_whole_mesh:
                    row.update(mesh_zero_area_triangles=0, mesh_nonfinite_area_triangles=0,
                               mesh_triangles_checked=100)
                writer.writerow(row)

    def reference_fixture(self, root):
        segment_dir = root / "segment8"
        segment_dir.mkdir()
        segment_path = segment_dir / "execution.json"
        segment = {
            "exit_code": 0,
            "environment": {"NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
                            "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "8",
                            "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT": "0"},
            "run_configuration": {"accepted_com_audit_segment_steps": 8,
                                  "full_accepted_q_audit": False,
                                  "requested_dt_s": "0.008", "requested_step_count": 250,
                                  "presentation_period_s": 0.064, "transaction_probe": True,
                                  "root_assistance_requested": False},
        }
        segment_path.write_text(json.dumps(segment), encoding="utf-8")
        for name in ("resting-coupled.csv", "resting-com-support-impulses.csv",
                     "resting-com-momentum-diagnostic.csv"):
            (segment_dir / name).write_text("retained segment reference\n", encoding="utf-8")
        self.write_surface_fixture(segment_dir / "resting-surface-audit.csv", 250, 0.008)

        q_dir = root / "full-q"
        q_dir.mkdir()
        q_path = q_dir / "execution.json"
        q = {
            "exit_code": 0,
            "environment": {"NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
                            "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "1",
                            "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT": "1"},
            "run_configuration": {"accepted_com_audit_segment_steps": 1,
                                  "full_accepted_q_audit": True,
                                  "requested_dt_s": "0.002", "requested_step_count": 1000,
                                  "presentation_period_s": 0.064, "transaction_probe": True,
                                  "root_assistance_requested": False},
        }
        q_path.write_text(json.dumps(q), encoding="utf-8")
        (q_dir / "resting-coupled.csv").write_text("retained 2ms physiology\n", encoding="utf-8")
        q_csv = q_dir / "resting-com-q-integration.csv"
        q_csv.write_text("step\n" + "".join(f"{step}\n" for step in range(1000)), encoding="utf-8")
        self.write_surface_fixture(q_dir / "resting-surface-audit.csv", 1000, 0.002)
        runtime_path = root / "runtime-correctness.json"
        runtime_path.write_text(json.dumps({
            "evidence": {"integrated-final-runtime-2ms-check-801/execution.json": {
                "path": str(q_path.resolve()), "sha256": hashlib.sha256(q_path.read_bytes()).hexdigest()}},
            "integrated_2ms_check": {"accepted_steps": 1000},
        }), encoding="utf-8")
        return segment_path, q_path, runtime_path

    def test_310s_plan_requires_exact_2ms_gpu_path_and_diagnostic_only_coarsening(self):
        invocation = self.invocation()
        adapter.validate_native_310s_invocation(invocation)
        for key, value, expected_error in (
            ("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT", "1", "Q_INTEGRATION_AUDIT=0"),
            ("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS", "1", "SEGMENT_STEPS=8"),
            ("NUMI_HUMAN_STAND_CONTACT_WARMSTART", "1", "CONTACT_WARMSTART=0"),
            ("NUMI_HUMAN_RESIDENT_PHYSICS_PILOT", "8", "single-Human scene"),
        ):
            bad = dict(invocation)
            bad["environment"] = dict(invocation["environment"], **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, expected_error):
                adapter.validate_native_310s_invocation(bad)
        bad = dict(invocation)
        bad["argv"] = list(invocation["argv"])
        bad["argv"][bad["argv"].index("--muscle-step-seconds") + 1] = "0.008"
        with self.assertRaisesRegex(ValueError, "real 2 ms native run"):
            adapter.validate_native_310s_invocation(bad)

    def test_surface_audit_rejects_missing_whole_mesh_area_columns_when_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.csv"
            self.write_surface_fixture(path, 64, 0.002, include_whole_mesh=False)
            with self.assertRaisesRegex(ValueError, "required whole-mesh area audit"):
                adapter.native_surface_trace_consistency(path, 64, 0.002, require_whole_mesh=True)

    def test_310s_plan_pins_segment8_and_full_q_references_with_distinct_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segment, full_q, runtime = self.reference_fixture(root)
            manifest, artifacts = adapter.native_310s_reference_manifest(Namespace(
                segment8_reference=str(segment), full_q_reference=str(full_q),
                runtime_correctness_reference=str(runtime)))
            self.assertEqual(manifest["segment8_schedule_reference"]["execution_sha256"],
                             hashlib.sha256(segment.read_bytes()).hexdigest())
            self.assertEqual(manifest["full_q_2ms_reference"]["full_q_audit_rows"], 1000)
            self.assertEqual(manifest["full_q_2ms_reference"]["displayed_surface_frames"], 33)
            self.assertEqual(manifest["full_q_2ms_reference"]["whole_mesh_triangles_checked_per_frame"], 100)
            self.assertEqual(manifest["segment8_schedule_reference"]["displayed_surface_frames"], 33)
            self.assertIn("does not qualify 8 ms temporal accuracy",
                          manifest["segment8_schedule_reference"]["scope"])
            self.assertIn("not 310 s endurance",
                          manifest["full_q_2ms_reference"]["scope"])
            self.assertIn(str(segment.resolve()), artifacts)
            self.assertIn(str((full_q.parent / "resting-com-q-integration.csv").resolve()), artifacts)
            invalid = json.loads(segment.read_text())
            invalid["run_configuration"]["requested_dt_s"] = "0.002"
            segment.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "segment-8 reference no longer matches"):
                adapter.native_310s_reference_manifest(Namespace(
                    segment8_reference=str(segment), full_q_reference=str(full_q),
                    runtime_correctness_reference=str(runtime)))

    def test_310s_plan_wrapper_fixes_physics_horizon_and_only_prepares_v2_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invocation = self.invocation()
            invocation_path = root / "invocation.json"
            invocation_path.write_text(json.dumps(invocation), encoding="utf-8")
            (root / "run-metadata.json").write_text(json.dumps({
                "exit_code": 0, "loaded_metal_runtime": {"verified": True},
                "source_files_changed_during_run": [], "argv": invocation["argv"],
                "asset_sha256": invocation["asset_sha256"],
                "environment": invocation["environment"],
            }), encoding="utf-8")
            (root / "native.log").write_text(
                "runtime=device=Apple M4 Pro world_fingerprint=123 eligible dense45 vascular solve\n"
                "resting_body_source_fingerprint=111 coupled_program_fingerprint=456\n"
                'stand_terminal_state={"root_assistance":false,"step_count":1000,"q":[],"v":[]}\n'
                "resting_integrated_body=completed simulated_s=2 wall_s=20 real_time_factor=0.1 "
                "physiology_body_clock=matched root_assistance=false\n", encoding="utf-8")
            source_hashes = root / "source-hashes.json"
            source_hashes.write_text("{}", encoding="utf-8")
            source_revisions = root / "source-revisions.json"
            source_revisions.write_text("{}", encoding="utf-8")
            segment, full_q, runtime = self.reference_fixture(root)
            args = Namespace(
                invocation=str(invocation_path), source_hashes=str(source_hashes),
                source_revisions=str(source_revisions), directory=str(root / "plan"),
                segment8_reference=str(segment), full_q_reference=str(full_q),
                runtime_correctness_reference=str(runtime), repository=str(root),
                parser_fixture=str(root / "parser.csv"), world_fingerprint="123",
                control_program_fingerprint="456", treatment_program_fingerprint="789",
                device="Apple M4 Pro")
            identity = {"configuration": {}}
            calibration = {"schema": "numi.science.calibration.v1"}
            plan = {"schema": "numi.science.plan.v2", "model": {}, "artifacts": [],
                    "design": {}, "limitations": "", "validity": []}
            with patch.object(adapter, "native_plan_components",
                              return_value=(identity, calibration, plan)) as build:
                plan_path = adapter.prepare_native_310s(args)
            called = build.call_args.args[0]
            self.assertEqual(called.steps, 155000)
            self.assertEqual(called.dt, 0.002)
            self.assertEqual(called.start_s, 60.0)
            self.assertEqual(called.end_s, 100.0)
            self.assertEqual(called.scale, 0.5)
            self.assertEqual(called.window_s, 30.0)
            emitted = json.loads(plan_path.read_text())
            emitted_identity = json.loads((root / "plan" / "native-build-identity.json").read_text())
            schedule = emitted_identity["configuration"]["audit_schedule"]
            self.assertEqual(schedule["accepted_com_momentum_audit"]["segment_steps"], 8)
            self.assertFalse(schedule["accepted_q_integration_audit"]["enabled"])
            self.assertTrue(schedule["presented_surface_geometry_audit"]["enabled"])
            self.assertEqual(schedule["presented_surface_geometry_audit"]["cadence_s"], 0.064)
            self.assertEqual(schedule["physical_timestep_s"], 0.002)
            self.assertEqual(schedule["presentation_period_s"], 0.064)
            self.assertIn("each accepted 2 ms native root",
                          emitted_identity["configuration"]["physical_step_contract"])
            self.assertEqual(emitted["schema"], "numi.science.plan.v2")
            self.assertIn({"path": ["timestep_s"], "equals": 0.002}, emitted["validity"])
            self.assertIn("do not require a minimum", emitted["limitations"])
            self.assertFalse((root / "plan" / "study").exists())


class NativeFailureEvidenceIsolationTests(unittest.TestCase):
    def test_failed_arm_cannot_overwrite_preflight_failure_evidence(self):
        # Exercise an actual failing child and its environment without launching
        # GPU physics. Both frozen and inherited diagnostic paths must be local
        # to the new trial, while the original receipt stays unchanged.
        import os
        import sys
        key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
        for inherited in (False, True):
            with self.subTest(inherited=inherited), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                prior = root / "preflight-failure.json"
                prior.write_text("retained preflight evidence")
                bound = root / "frozen-input"
                bound.write_text("unchanged input")
                invocation = {"argv": [], "asset_sha256": {
                    str(bound): hashlib.sha256(bound.read_bytes()).hexdigest()},
                    "environment": {} if inherited else {key: str(prior)}}
                reference = root / "reference.json"
                reference.write_text(json.dumps(invocation))
                reference_before = reference.read_bytes()
                output = root / "scene"
                args = Namespace(invocation=str(reference), output=str(output),
                                 steps=160000, dt=.002, start_s=60., end_s=100.,
                                 scale=.5, window_s=30.)
                child = [sys.executable, "-c",
                         "import os,pathlib,sys; "
                         "pathlib.Path(os.environ['" + key + "']).write_text('native failure fixture'); "
                         "sys.exit(3)"]
                cwd = Path.cwd()
                try:
                    os.chdir(root)
                    with patch.dict(os.environ, {key: str(prior)}), \
                         patch.object(adapter, "native_scene_command", return_value=child):
                        with self.assertRaisesRegex(ValueError, "native scene failed with status 3"):
                            adapter.execute_native_scene_arm(args)
                finally:
                    os.chdir(cwd)
                self.assertEqual(prior.read_text(), "retained preflight evidence")
                self.assertEqual(reference.read_bytes(), reference_before)
                self.assertEqual((output / "common-field-failure.json").read_text(),
                                 "native failure fixture")
                recorded = json.loads((output / "invocation.json").read_text())
                self.assertEqual(recorded["environment"][key],
                                 str(output / "common-field-failure.json"))

if __name__ == '__main__':
    unittest.main()
