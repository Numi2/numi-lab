"""Arithmetic/admission regression tests, not physiological qualification."""
import math
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from metalrobo.science_notebook import validate as validate_plan

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


class NativeSceneBindingTests(unittest.TestCase):
    def invocation(self):
        return {"argv": ["/build/numi-human-native", "/rigid", "/myo", "/bones", "/old-output",
                         "--persistent-metal-stand", "--resting-scene", "/network", "/respiration",
                         "--vascular-dense45", "--resting-anatomy-receipt", "/anatomy",
                         "--skin-payload", "/skin", "--tendon-payload", "/tendon",
                         "--muscle-step-count", "64", "--muscle-step-seconds", ".001",
                         "--resting-movie", "/old-output/native-viewer.mov"],
                "asset_sha256": {p: "bound" for p in ("/build/numi-human-native", "/rigid", "/myo",
                    "/bones", "/network", "/respiration", "/anatomy", "/skin", "/tendon")}}

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


class NativeV2PlanPreparationTests(unittest.TestCase):
    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def test_native_plan_uses_exact_pair_ids_and_prespecified_windows_without_registering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_names = ("numi-human-native", "rigid.nhrigid", "muscle.nhmyo", "bones.nhbones",
                           "network.json", "respiration.json", "anatomy.json", "skin.nhskin",
                           "tendon.nhtendon", "tissue.nhtissue")
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
                "argv": [str(root / "assets" / "numi-human-native"),
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
            args = Namespace(directory=str(output), invocation=str(invocation_path),
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

    def test_native_plan_rejects_same_arm_program_identity_and_stale_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.cpp"
            source_hashes = {str(source): self.write(source, b"source")}
            invocation_path = root / "invocation.json"
            executable = root / "numi-human-native"
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


if __name__ == '__main__':
    unittest.main()
