#!/usr/bin/env python3
"""CPU-only synthetic/admission checks for the source143 convergence study."""
from __future__ import annotations
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

B = Path('/Users/home/cardboard-implementation-20261014')
SRC = B / 'snapshots/source-cgls-primary-policy-143/matter/tools'
PACKAGE = B / 'analysis/cardboard-convergence-tool-overlay-145-001'
FAILED_RUNS = [
    B / 'solver/bounded-cgls64-exact-state-450/output',
    B / 'solver/bounded-cgls64-capture-off-parity-452/output',
]

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Existing source143 synthetic run builder is used only to provide fully marked
# synthetic reader inputs; it makes no claim about physical trajectories.
fixtures = load_module('source143_convergence_fixtures', SRC / 'test_cardboard_convergence.py')

class ToolOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpmod = tempfile.TemporaryDirectory()
        moddir = Path(cls.tmpmod.name)
        shutil.copy2(SRC / 'cardboard_convergence.py', moddir / 'cardboard_convergence.py')
        shutil.copy2(PACKAGE / 'overlay/cardboard_resolution_study.py', moddir / 'cardboard_resolution_study.py')
        cls.study = load_module('cardboard_resolution_study_overlay145', moddir / 'cardboard_resolution_study.py')
        cls.cc = cls.study.cc

    @classmethod
    def tearDownClass(cls):
        cls.tmpmod.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_matrix_phases_time_refinement_and_separate_rate_axis(self):
        s = self.study
        self.assertEqual(s.PHASE_ORDER, ('loading', 'hold', 'unloading', 'relaxation', 'release'))
        meshes = {c['id']: c['mesh'] for c in s._spatial_cases()}
        self.assertEqual(meshes, {
            'nx16-ny2-t1': [16, 2, 1], 'nx32-ny2-t1': [32, 2, 1],
            'nx8-ny4-t1': [8, 4, 1], 'nx8-ny8-t1': [8, 8, 1],
            'nx8-ny2-t2': [8, 2, 2], 'nx8-ny2-t4': [8, 2, 4],
            'nx32-ny8-t4': [32, 8, 4],
        })
        protocol = {'loading': 8, 'hold': 4, 'unloading': 8, 'relaxation': 16,
                    'release': 4, 'dt_s': .00005}
        for factor in (2, 4):
            counts = s._time_counts(protocol, factor)
            for phase in s.PHASE_ORDER:
                self.assertEqual(protocol['dt_s'] * protocol[phase],
                                 (protocol['dt_s'] / factor) * counts[phase])
        rate_pairs = s._case_pairs(True)
        self.assertEqual(rate_pairs[-1]['group'], 'slower_load')
        self.assertEqual(rate_pairs[-1]['kind'], 'load_rate')

    def test_actual_native_450_452_failed_runs_cannot_pass_from_accepted_prefix(self):
        for run in FAILED_RUNS:
            result = self.cc.analyze_run(run)
            self.assertEqual(result['status'], 'failed_run', (run, result))
            self.assertEqual(result['reasons'][0]['code'], 'native_run_failed')
            self.assertFalse(result['physical_validation'])
            raw = json.loads((run / 'result.json').read_text())
            self.assertEqual(raw['status'], 'failed')
            self.assertEqual(raw['accepted_steps'], 1)

    def test_completed_status_with_short_accepted_prefix_is_rejected(self):
        run = fixtures.fake_run(self.root, 'short-prefix', steps=2, hold=1, unload=1,
                                relax=1, release=1)
        result_path = run / 'result.json'
        result = json.loads(result_path.read_text())
        result['accepted_steps'] = 1
        result_path.write_text(json.dumps(result))
        report = self.cc.analyze_run(run)
        self.assertEqual(report['status'], 'failed_run')
        self.assertEqual(report['reasons'][0]['code'], 'accepted_step_count_mismatch')
        self.assertFalse(report['physical_validation'])

    def test_missing_required_baseline_phase_is_refused(self):
        run = fixtures.fake_run(self.root, 'missing-relax-and-release', nx=8, steps=2,
                                hold=1, unload=1, relax=0, release=0)
        manifest_path = run / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['mesh']['ny'] = 2
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(self.study.StudyError, 'positive loading, hold, withdrawal, relaxation, and release'):
            self.study._baseline_contract(run)

    def test_shifted_phase_duration_is_refused(self):
        baseline = fixtures.fake_run(self.root, 'time-base', nx=8, dt=.01,
                                     steps=2, hold=2, unload=2, relax=2, release=2)
        shifted = fixtures.fake_run(self.root, 'time-shifted', nx=8, dt=.005,
                                    steps=4, hold=3, unload=4, relax=4, release=4)
        result = self.cc.analyze_pair(baseline, shifted, 'temporal', list(self.study.PHASE_ORDER),
                                      force_matched_tolerance=True)
        self.assertEqual(result['status'], 'incompatible_pair')
        self.assertEqual(result['reasons'][0]['code'], 'phase_duration_mismatch')
        self.assertFalse(result['physical_validation'])

    def test_declared_nonincreasing_rule_accepts_equality_but_rejects_worsening(self):
        common = {
            'peak_force_relative': .01, 'peak_moment_relative': .01, 'path_work_relative': .01,
            'normalized_force_curve_rms': .03, 'normalized_moment_curve_rms': .02,
            'springback_angle_difference_deg': .4, 'signed_springback_angle_difference_deg': .4,
            'paper_damage_area_fraction_difference': .01, 'debond_area_fraction_difference': .02,
            'damage_debond': {'baseline': {'paper_enabled': True, 'debond_enabled': True},
                              'candidate': {'paper_enabled': True, 'debond_enabled': True}},
            'area_feature_enablement_matches': True,
        }
        equal = self.study._gate_triple(dict(common), dict(common))
        self.assertTrue(all(x['status'] == 'pass' for x in equal.values()), equal)
        worsened = dict(common)
        worsened['normalized_force_curve_rms'] = .031
        rejected = self.study._gate_triple(dict(common), worsened)
        self.assertEqual(rejected['normalized_force_curve_rms']['status'], 'difference_not_decreasing')

    def test_complete_synthetic_curve_metrics_are_screened_without_physical_claim(self):
        base = fixtures.fake_run(self.root, 'synthetic-coarse', nx=8, steps=2, hold=2,
                                 unload=2, relax=2, release=2, moment_scale=1.0)
        fine = fixtures.fake_run(self.root, 'synthetic-fine', nx=16, steps=2, hold=2,
                                 unload=2, relax=2, release=2, moment_scale=1.01)
        for path, damage, debond in ((base, .20, .10), (fine, .21, .11)):
            result_path = path / 'result.json'
            result = json.loads(result_path.read_text())
            result['resolution_features'] = {
                'paper_damage_area_enabled': True,
                'glue_debond_area_enabled': True,
            }
            result['damage_and_debond_area_fractions'] = {
                'paper_damage_area_fraction': damage,
                'debond_area_fraction': debond,
            }
            result_path.write_text(json.dumps(result))
        admission = self.cc.analyze_pair(base, fine, 'spatial', list(self.study.PHASE_ORDER))
        self.assertEqual(admission['status'], 'analyzed_descriptive')
        self.assertFalse(admission['physical_validation'])
        metrics = self.study._pair_metrics(self.cc._load_run(base), self.cc._load_run(fine), 'spatial')
        metrics['max_free_force_residual_limit_N'] = .30
        gates = self.study._gate_pair(metrics)
        for key in ('peak_force_relative', 'peak_moment_relative', 'path_work_relative',
                    'normalized_force_curve_rms', 'normalized_moment_curve_rms',
                    'signed_springback_angle_difference_deg',
                    'paper_damage_area_fraction_difference', 'debond_area_fraction_difference'):
            self.assertEqual(gates[key]['status'], 'pass', (key, metrics[key], gates[key]))
        self.assertEqual(gates['observed_free_force_residuals']['status'], 'pass')
        self.assertLessEqual(metrics['peak_force_relative'], .02)
        self.assertLessEqual(metrics['peak_moment_relative'], .02)
        self.assertLessEqual(metrics['path_work_relative'], .02)
        self.assertLessEqual(metrics['normalized_force_curve_rms'], .05)
        self.assertLessEqual(metrics['normalized_moment_curve_rms'], .05)
        self.assertLessEqual(metrics['signed_springback_angle_difference_deg'], 1.0)
        self.assertLessEqual(metrics['paper_damage_area_fraction_difference'], .02)
        self.assertLessEqual(metrics['debond_area_fraction_difference'], .02)

if __name__ == '__main__':
    unittest.main(verbosity=2)
