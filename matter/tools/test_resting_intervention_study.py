"""Arithmetic/admission regression tests, not physiological qualification."""
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from resting_intervention_study import (complete_breath_metrics, positive_linear_area,
                                       native_scene_command, native_scene_summary, native_body_trace_consistency)


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
               'stand_terminal_state={"step_count":32,"root_assistance":false,"q":[0],"v":[0]}\n'
               'resting_integrated_body=completed simulated_s=.032 wall_s=1 real_time_factor=.032 '
               'physiology_body_clock=matched root_assistance=false presentation_qualification=pending\n')
        result = native_scene_summary(log)
        self.assertEqual(result['accepted_steps'], 32)
        self.assertEqual(result['world_fingerprint'], '123')
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


if __name__ == '__main__':
    unittest.main()
