from __future__ import annotations
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
HERE=Path(__file__).resolve().parent;SRC=HERE/'filter_self_safe_followup_543.py'
BASE=HERE/'run-003-filtered/candidate.npz';FULL=HERE/'run-004-followup/candidate.npz'
FULL_REPORT=HERE/'run-004-followup/report.json';AUDIT=HERE/'check-004-followup/audit-130/summary.json'
PAIRS=HERE/'check-004-followup/audit-130/surface-self-pairs.npy';SAVED=HERE/'run-005-filtered/candidate.npz'
sp=importlib.util.spec_from_file_location('filter_543_test',SRC);mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
class FilterFollowup543Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(BASE,allow_pickle=False) as z:cls.base={k:z[k] for k in z.files}
  with np.load(FULL,allow_pickle=False) as z:cls.full={k:z[k] for k in z.files}
  cls.report=json.loads(FULL_REPORT.read_text());cls.audit=json.loads(AUDIT.read_text());cls.pairs=np.load(PAIRS,allow_pickle=False)
  cls.replay=load('replay405_filter543_test',D/'local-two-parent-retriangulation-405/cross_seam_search_405.py')
  cls.owner=load('replay534_filter543_test',D/'connected-quality-patch-534/replay_surviving_flips_534.py')
  cls.builder=load('followup_builder_filter543_test',HERE/'repair_focused_clearance_followup.py')
  cls.out,cls.meta=mod.filter_batch(cls.base,cls.full,cls.report,cls.audit,cls.pairs,cls.replay,cls.owner,cls.builder)
  with np.load(SAVED,allow_pickle=False) as z:cls.saved={k:z[k] for k in z.files}
 def test_pinned_inputs_and_audit_binding(self):
  for p,key in [(BASE,'base_sha256'),(FULL,'full_candidate_sha256'),(FULL_REPORT,'full_report_sha256'),(AUDIT,'self_audit_sha256'),(PAIRS,'self_pairs_sha256')]:self.assertEqual(mod.sha(p),mod.PINS[key])
  self.assertEqual(self.audit['candidate_sha256'],mod.PINS['full_candidate_sha256']);self.assertTrue(self.audit['complete_surface_audit'])
  self.assertEqual(self.pairs.shape,(102,2))
 def test_all_witnessed_ops_removed(self):
  self.assertEqual(self.meta['full_batch_selected_operations'],76)
  self.assertEqual(self.meta['removed_operation_count'],23);self.assertEqual(self.meta['retained_operation_count'],53)
  bad=set(map(int,self.pairs.ravel()))
  for op in self.meta['removed_operations']:self.assertTrue(set(op['face_rows'])&bad)
  for i in self.meta['retained_operation_indices']:self.assertFalse(set(self.report['selected_flips'][i]['face_rows'])&bad)
 def test_filtered_candidate_exactly_matches_saved_replay(self):
  for k in self.out:self.assertTrue(mod.same(self.out[k],self.saved[k]),k)
  self.assertTrue(mod.same(self.out['points_world_m'],self.base['points_world_m']))
  self.assertTrue(self.meta['all_self_witness_faces_excluded_from_retained_operations'])
  self.assertTrue(self.meta['outside_surviving_operation_rows_bitwise_unchanged'])
  self.assertTrue(self.meta['outside_surviving_operation_ancestry_unchanged'])
 def test_scope_anchor_boundary_and_nonadmission(self):
  self.assertEqual(self.meta['typed_anchor_count'],237);self.assertTrue(self.meta['typed_anchor_coordinates_bitwise_unchanged'])
  self.assertTrue(self.meta['boundary_edge_set_unchanged']);self.assertTrue(self.meta['fixed037_connectivity_synchronized'])
  self.assertFalse(self.meta['geometry_admitted']);self.assertFalse(self.meta['native_admission']);self.assertEqual(self.meta['accepted_physical_steps'],0)
 def test_wrong_witness_shape_and_manifest_drift_fail_closed(self):
  with self.assertRaises(ValueError):mod.filter_batch(self.base,self.full,self.report,self.audit,np.zeros((102,3),dtype=np.int64),self.replay,self.owner,self.builder)
  bad=dict(self.report);bad['selected_flips']=bad['selected_flips'][:-1]
  with self.assertRaises(ValueError):mod.filter_batch(self.base,self.full,bad,self.audit,self.pairs,self.replay,self.owner,self.builder)
if __name__=='__main__':unittest.main()

