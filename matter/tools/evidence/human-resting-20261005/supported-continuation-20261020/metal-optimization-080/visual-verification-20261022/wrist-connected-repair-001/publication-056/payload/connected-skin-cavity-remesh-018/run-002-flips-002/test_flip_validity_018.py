import hashlib, json, pathlib, unittest
RUN = pathlib.Path(__file__).resolve().parent
class FlipValidityReceipt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality = json.loads((RUN / 'flip-report-quality-only-as-run.json').read_text())
        cls.audit = json.loads((RUN / 'flip-validity-audit.json').read_text())
        cls.geom = RUN / 'local-flip-geometry.npz'
    def test_quality_only_candidates_are_not_admitted(self):
        self.assertTrue(self.quality['candidate_unadmitted'])
        self.assertEqual(self.audit['status'], 'both_quality_passing_2to3_trials_rejected_as_nonconforming_retriangulations')
        self.assertEqual(self.audit['candidate_npz_sha256'], hashlib.sha256(self.geom.read_bytes()).hexdigest())
    def test_both_2to3_candidates_fail_same_domain_overlap(self):
        self.assertEqual(len(self.audit['trials']), 2)
        for row in self.audit['trials']:
            self.assertGreaterEqual(min(self.quality['trials'][0]['new_q'] + self.quality['trials'][1]['new_q']), 1e-5)
            self.assertTrue(all(abs(v - 1.0) < 1e-8 for v in row['old_cell_coverage']))
            self.assertTrue(any(v < 1.0 - 1e-8 for v in row['new_child_overlap_coverage']))
            self.assertGreater(abs(row['relative_patch_volume_delta']), 1e-3)
    def test_no_eligible_3to2_star(self):
        self.assertEqual(self.audit['eligible_3to2_edge_count'], 0)
        self.assertFalse(any(e['eligible_3to2'] for e in self.audit['bad_tet_edge_stars']))
    def test_full_ancestry_and_mass_are_pending(self):
        self.assertIn('pending', self.quality['complete_original_parent_ancestry'])
        self.assertIn('pending', self.quality['full_material_mass_ledger'])
if __name__ == '__main__': unittest.main(verbosity=2)
