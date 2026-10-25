import json
from pathlib import Path
import unittest
P=Path('/Users/n/Work/numi-human-connected-surface-069/digit-source-diagnosis-110/batch-146-curved-source-midpoints-quality-114.json')
class Batch146Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(P.read_text())
    def test_unique_population(self):
        rows=self.report['edges']
        self.assertEqual(len(rows),146)
        self.assertEqual(len({tuple(r['edge_output_rows']) for r in rows}),146)
        self.assertEqual(self.report['counts']['crossing_source_edges'],145)
        self.assertEqual(self.report['counts']['quality_only_source_edges'],1)
    def test_all_have_oriented_reference_and_current_proposals(self):
        for row in self.report['edges']:
            self.assertTrue(row['source_normal_proposal_eligible'])
            for k in ('reference_outward_normal','current037_outward_normal'):
                n=row[k]
                self.assertAlmostEqual(sum(x*x for x in n),1.0,places=8)
            self.assertIn('normal_scalar_source_proposal_reference',row)
            self.assertIn('normal_scalar_source_proposal_current037',row)
    def test_quality_edge_is_real_two_face_edge_and_source_anchored(self):
        row=self.report['quality_edge_source_proposal']
        self.assertEqual(row['edge_output_rows'],[344,2332])
        self.assertEqual(row['parent_source_O_face_rows'],[716,717])
        self.assertEqual(row['edge_face_incidence'],2)
        anchors=row['endpoint_source_face_anchors']
        self.assertEqual(anchors[1]['source_face_vertex_rows'],[3775,3776,3777])
        self.assertAlmostEqual(sum(anchors[1]['barycentric']),1.0,places=12)
        self.assertGreater(anchors[1]['residual_m'],0.001)
        self.assertLess(anchors[1]['residual_m'],0.002)
    def test_quality_depths_are_existing_and_positive(self):
        row=self.report['quality_edge_source_proposal']
        for key in ('source_skin_depth_reference_endpoint_m','source_fat_depth_reference_endpoint_m',
                    'source_skin_depth_current037_endpoint_m','source_fat_depth_current037_endpoint_m'):
            self.assertEqual(len(row[key]),2)
            self.assertTrue(all(x>0 for x in row[key]))
        self.assertEqual(row['normal_scalar_source_proposal_reference']['depths_are_interpolated_existing_scalar_fields'],True)
    def test_two_bad_signed_endpoint_depths_are_preserved(self):
        bad=[r for r in self.report['edges'] if r.get('scalar_depth_validity','').startswith('unresolved')]
        self.assertEqual(len(bad),2)
        self.assertEqual({tuple(r['edge_output_rows']) for r in bad},{(310,364),(2236,2272)})
if __name__=='__main__': unittest.main()
