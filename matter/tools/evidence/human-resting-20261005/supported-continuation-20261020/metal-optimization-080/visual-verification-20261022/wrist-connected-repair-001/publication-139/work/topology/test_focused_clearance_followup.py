from __future__ import annotations
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
HERE=Path(__file__).resolve().parent
SRC=HERE/'repair_focused_clearance_followup.py'
CAND=HERE/'run-003-filtered/candidate.npz'
AUDIT=HERE/'check-003-filtered/audit-159/report.json'
SAVED=HERE/'run-004-followup/candidate.npz'
spec=importlib.util.spec_from_file_location('focused_clearance_followup_test',SRC)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class FollowupBuilderTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(CAND,allow_pickle=False) as z:cls.base={k:z[k] for k in z.files}
  cls.report=json.loads(AUDIT.read_text())
  cls.out,cls.meta=mod.build(cls.base,cls.report,max_flips=0)
  cls.one,cls.one_meta=mod.build(cls.base,cls.report,max_flips=1)
  with np.load(SAVED,allow_pickle=False) as z:cls.saved={k:z[k] for k in z.files}

 def test_current_candidate_and_report_are_exactly_bound(self):
  self.assertEqual(mod.sha_file(CAND),'8dd096a2fcb8f9b24571fb4c1ffcab0bd3e1bd1c37537916255bd832e4265e65')
  self.assertEqual(mod.sha_file(AUDIT),'d0f2ca650d5287b3d3cbe054175dd2c55e7c60ad54f06ff0459dfdc134a0b1de')
  self.assertEqual(self.report['candidate_sha256'],mod.sha_file(CAND))
  self.assertTrue(self.report['complete'])
  self.assertEqual(self.report['bank_count'],185)

 def test_dynamic_unique_focus_count_table(self):
  focus,counts=mod.focused_crossing_rows(self.report,len(self.base['source_O_faces']))
  expected={(109,114):502,(115,118):370,(65,99):350,(43,16):349,(124,123):293,(59,95):291}
  self.assertEqual(counts,expected)
  self.assertEqual(sum(counts.values()),2155)
  self.assertEqual(len(focus),1198)
  self.assertEqual(len(mod.edge_one_ring(self.base['source_O_faces'],focus)),1911)

 def test_inconsistent_pair_count_and_duplicate_focus_bank_fail_closed(self):
  bad=json.loads(json.dumps(self.report))
  b=next(x for x in bad['banks'] if (x['body'],x['stable_bone_id'])==(109,114))
  b['result']['count']+=1
  with self.assertRaises(ValueError):mod.focused_crossing_rows(bad,len(self.base['source_O_faces']))
  bad=json.loads(json.dumps(self.report)); duplicate=next(x for x in bad['banks'] if (x['body'],x['stable_bone_id'])==(109,114))
  bad['banks'].append(duplicate)
  with self.assertRaises(ValueError):mod.focused_crossing_rows(bad,len(self.base['source_O_faces']))

 def test_zero_is_unlimited_and_positive_limit_is_respected(self):
  self.assertEqual(self.meta['selected_flip_count'],76)
  self.assertEqual(self.one_meta['selected_flip_count'],1)
  self.assertGreater(self.meta['selected_flip_count'],self.one_meta['selected_flip_count'])

 def test_scope_anchors_coordinates_and_topology_invariants(self):
  focus,_=mod.focused_crossing_rows(self.report,len(self.base['source_O_faces']))
  scope=mod.edge_one_ring(self.base['source_O_faces'],focus)
  changed=set(map(int,np.flatnonzero(np.any(self.out['source_O_faces']!=self.base['source_O_faces'],axis=1))))
  self.assertTrue(changed<=scope)
  self.assertTrue(mod.identical(self.base['points_world_m'],self.out['points_world_m']))
  anchors=self.base['typed_coarse_anchor_output_rows_018']
  self.assertEqual(len(anchors),237)
  self.assertTrue(mod.identical(self.base['points_world_m'][anchors],self.out['points_world_m'][anchors]))
  _,b0,n0=mod.topology(self.base['source_O_faces']);_,b1,n1=mod.topology(self.out['source_O_faces'])
  self.assertFalse(n0);self.assertFalse(n1);self.assertEqual(b0,b1)
  self.assertTrue(np.array_equal(self.out['source_O_faces'],self.out['area_reference_037_source_O_faces']))

 def test_candidate_matches_saved_run_and_is_unqualified(self):
  for k in self.out:self.assertTrue(mod.identical(self.out[k],self.saved[k]),k)
  self.assertFalse(self.meta['geometry_admitted']);self.assertFalse(self.meta['native_admission'])
  self.assertEqual(self.meta['accepted_physical_steps'],0)
  self.assertEqual(self.meta['exact_self_surface_and_185_bank_audits'],'NOT_RUN_PENDING_ROOT_SLOT')
  self.assertTrue(self.meta['outside_focus_faces_bitwise_unchanged'])
  self.assertTrue(self.meta['outside_focus_ancestry_rows_unchanged'])

if __name__=='__main__':unittest.main()

