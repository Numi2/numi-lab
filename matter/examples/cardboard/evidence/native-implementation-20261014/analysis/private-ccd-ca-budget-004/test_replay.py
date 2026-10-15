import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('private_ccd_replay', HERE / 'replay.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

CAPTURES = [
    replay.BASE / 'native/center-action-feasibility-bracket-333/private-trial.coupled.json',
    replay.BASE / 'native/center-action-feasibility-bracket-334/private-trial.coupled.json',
]


class CapturedNearMissTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = [replay.capture_feature(path) for path in CAPTURES]

    def test_capture_hashes_and_feature_lineage(self):
        for (cap, item, *_) in self.loaded:
            self.assertEqual(cap.manifest['root'], 1201)
            self.assertEqual(cap.manifest['iteration'], 12)
            self.assertEqual(item['candidate'], [88, 127, 331, 1])
            self.assertEqual(item['primitive_nodes'], [[13, 14, 17], [28, 32, 31]])
            self.assertEqual(item['edge_node_ids'], [[13, 14], [28, 32]])
            self.assertEqual(cap.manifest['selected_phase'], 'both')

    def test_preceding_vt_features_and_captured_ee_budget(self):
        report = replay.build_report(CAPTURES)
        for capture_report in report['captures']:
            # The production narrowphase checks six VT orientations before the
            # nine EE pairs. Confirm all six VT checks certify in the FP32
            # source-equation replay before EE(0,0) reaches its iteration cap.
            for mix_name in ('difference_form_fp32', 'weighted_form_fp32'):
                checks = capture_report['vertex_triangle_source_order'][mix_name]
                self.assertEqual(len(checks), 6)
                self.assertTrue(all(row['certified'] for row in checks))
                self.assertEqual([(row['direction'], row['vertex_index']) for row in checks],
                                 [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])
        for _, _, sa, fa, sb, fb, thickness in self.loaded:
            for mixfn in (replay.mix_difference, replay.mix_weighted):
                at16 = replay.ca_result('ee', sa[:2], fa[:2], sb[:2], fb[:2], thickness, mixfn, 16)
                at17 = replay.ca_result('ee', sa[:2], fa[:2], sb[:2], fb[:2], thickness, mixfn, 17)
                self.assertTrue(at17['certified'])
                self.assertFalse(at17['hit'])
                self.assertEqual(at16['effective_thickness_m'], at17['effective_thickness_m'])
                if mixfn is replay.mix_difference:
                    self.assertFalse(at16['certified'])
                    self.assertEqual(at16['reason'], 'iteration_budget_exhausted')
                    self.assertEqual(at17['reason'], 'conservative_step_passes_endpoint')
                else:
                    self.assertTrue(at16['certified'])
                    self.assertEqual(at16['reason'], 'conservative_step_passes_endpoint')

    def test_independent_fp64_endpoint_clearance_is_positive(self):
        for _, _, sa, fa, sb, fb, thickness in self.loaded:
            effective = replay.ca_result('ee', sa[:2], fa[:2], sb[:2], fb[:2],
                                         thickness, replay.mix_difference, 17)['effective_thickness_m']
            end_distance = replay.closest_segments64(fa[0], fa[1], fb[0], fb[1])[2]
            self.assertGreater(end_distance, effective)
            self.assertLess(end_distance-effective, 2.0e-9)

    def test_true_hit_controls_are_detected_without_threshold_changes(self):
        _, _, sa, fa, sb, fb, thickness = self.loaded[0]
        for shift, budget, expected_time in ((60e-9, 17, 0.0), (50e-9, 32, None)):
            shifted_start = [v.copy() for v in sb]
            shifted_finish = [v.copy() for v in fb]
            for point in shifted_start + shifted_finish:
                point[2] = replay.F(point[2]-replay.F(shift))
            for mixfn in (replay.mix_difference, replay.mix_weighted):
                result = replay.ca_result('ee', sa[:2], fa[:2], shifted_start,
                                          shifted_finish, thickness, mixfn, budget)
                self.assertTrue(result['certified'])
                self.assertTrue(result['hit'])
                if expected_time is not None:
                    self.assertEqual(result['time'], expected_time)
            if shift == 50e-9:
                # At the candidate's 17-step cap this longer swept hit remains
                # unresolved and must stay fail-closed, never become a miss.
                for mixfn in (replay.mix_difference, replay.mix_weighted):
                    bounded = replay.ca_result('ee', sa[:2], fa[:2], shifted_start,
                                               shifted_finish, thickness, mixfn, 17)
                    self.assertFalse(bounded['certified'])
                    self.assertFalse(bounded['hit'])

    def test_patch_changes_only_edge_edge_budget(self):
        frozen = replay.BASE / 'snapshots/source-center-action-twoenv-restart-099/matter/src/metal/contact.metalinc'
        source = frozen.read_text()
        candidate = (HERE / 'contact.candidate.metalinc').read_text()
        self.assertEqual(replay.sha(frozen), '3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132')
        self.assertEqual(source.count('for (uint iteration = 0u; iteration < 16u; ++iteration) {'), 2)
        self.assertEqual(candidate.count('for (uint iteration = 0u; iteration < 16u; ++iteration) {'), 1)
        self.assertEqual(candidate.count('for (uint iteration = 0u; iteration < 17u; ++iteration) {'), 1)
        patch = (HERE / 'contact-16-to-17.patch').read_text()
        self.assertEqual(patch.count('-    for (uint iteration = 0u; iteration < 16u; ++iteration) {'), 1)
        self.assertEqual(patch.count('+    for (uint iteration = 0u; iteration < 17u; ++iteration) {'), 1)
        self.assertEqual(replay.sha(HERE / 'contact.candidate.metalinc'), 'eee5d4def83e3d8fd5a80a6cb4272b977a05e334cab1b9021402a18bbe324d8d')

    def test_patch_applies_to_exact_frozen_source(self):
        frozen = replay.BASE / 'snapshots/source-center-action-twoenv-restart-099/matter/src/metal/contact.metalinc'
        with tempfile.TemporaryDirectory(prefix='ccd-budget-overlay-') as temp:
            target = Path(temp) / 'matter/src/metal/contact.metalinc'
            target.parent.mkdir(parents=True)
            shutil.copyfile(frozen, target)
            result = subprocess.run(['patch', '-p1', '--batch'], cwd=temp,
                input=(HERE / 'contact-16-to-17.patch').read_text(), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(replay.sha(target), replay.sha(HERE / 'contact.candidate.metalinc'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
