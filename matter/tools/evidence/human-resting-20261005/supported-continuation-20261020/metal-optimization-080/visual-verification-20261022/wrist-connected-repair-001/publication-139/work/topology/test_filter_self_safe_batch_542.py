from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
HERE=Path(__file__).resolve().parent
FILTER=HERE/'filter_self_safe_batch_542.py'
BASE=D/'connected-quality-patch-538/run-001/candidate.npz'
FULL=HERE/'run-002-full/candidate.npz'
FULL_REPORT=HERE/'run-002-full/report.json'
AUDIT=HERE/'check-002-full/audit-130/summary.json'
PAIRS=HERE/'check-002-full/audit-130/surface-self-pairs.npy'
SAVED=HERE/'run-003-filtered/candidate.npz'
spec=importlib.util.spec_from_file_location('filter_self_safe_batch_542_test',FILTER)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

class Filter542Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(BASE,allow_pickle=False) as z:cls.base={k:z[k] for k in z.files}
  with np.load(FULL,allow_pickle=False) as z:cls.full={k:z[k] for k in z.files}
  cls.report=json.loads(FULL_REPORT.read_text());cls.audit=json.loads(AUDIT.read_text())
  cls.pairs=np.load(PAIRS,allow_pickle=False)
  cls.owner=load('owner534_test',D/'connected-quality-patch-534/replay_surviving_flips_534.py')
  cls.replay=load('replay405_test',D/'local-two-parent-retriangulation-405/cross_seam_search_405.py')
  cls.focus=load('builder541_test',D/'focused-clearance-541/topology/repair_focused_clearance_541.py')
  cls.out,cls.meta=mod.filter_batch(cls.base,cls.full,cls.report,cls.audit,cls.pairs,cls.replay,cls.owner,cls.focus)
  with np.load(SAVED,allow_pickle=False) as z:cls.saved={k:z[k] for k in z.files}

 def test_exact_audit_and_pair_table_are_bound_to_full_candidate(self):
  self.assertEqual(self.audit['candidate_sha256'],mod.PINS['full_candidate_sha256'])
  self.assertTrue(self.audit['complete_surface_audit'])
  self.assertEqual(self.pairs.shape,(72,2))
  self.assertEqual(mod.sha(PAIRS),mod.PINS['self_pairs_sha256'])

 def test_all_implicated_ops_removed_and_remaining_manifest_matches(self):
  self.assertEqual(self.meta['full_batch_selected_operations'],90)
  self.assertEqual(self.meta['removed_operation_count'],19)
  self.assertEqual(self.meta['retained_operation_count'],71)
  bad=set(map(int,self.pairs.ravel()))
  for op in self.meta['removed_operations']:self.assertTrue(set(op['face_rows'])&bad)
  for i in self.meta['retained_operation_indices']:
   self.assertFalse(set(self.report['selected_flips'][i]['face_rows'])&bad)

 def test_replay_matches_saved_candidate_and_keeps_base_coordinates(self):
  for k in self.out:self.assertTrue(mod.same(self.out[k],self.saved[k]),k)
  self.assertTrue(mod.same(self.out['points_world_m'],self.base['points_world_m']))
  self.assertTrue(self.meta['fixed037_connectivity_synchronized'])
  self.assertTrue(self.meta['boundary_edge_set_unchanged'])
  self.assertEqual(self.meta['typed_anchor_count'],237)
  self.assertTrue(self.meta['typed_anchor_coordinates_bitwise_unchanged'])

 def test_nonadmission_and_scope(self):
  self.assertFalse(self.meta['geometry_admitted'])
  self.assertFalse(self.meta['native_admission'])
  self.assertEqual(self.meta['accepted_physical_steps'],0)
  self.assertTrue(self.meta['all_self_witness_faces_excluded_from_retained_operations'])
  self.assertEqual(self.meta['exact_self_surface_and_185_bank_audits'],'NOT_RUN_PENDING_ROOT_SLOT')

 def test_malformed_self_pair_table_fails_closed(self):
  with self.assertRaises(ValueError):
   mod.filter_batch(self.base,self.full,self.report,self.audit,np.zeros((72,3),dtype=np.int64),self.replay,self.owner,self.focus)

 def test_operation_manifest_drift_fails_closed(self):
  broken=dict(self.report);broken['selected_flips']=broken['selected_flips'][:-1]
  with self.assertRaises(ValueError):
   mod.filter_batch(self.base,self.full,broken,self.audit,self.pairs,self.replay,self.owner,self.focus)

if __name__=='__main__':unittest.main()

