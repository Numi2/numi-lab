from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
HERE=Path(__file__).resolve().parent
SRC=HERE/'repair_focused_clearance_541.py'
CAND=ROOT/'D/connected-quality-patch-538/run-001/candidate.npz'
AUDIT=ROOT/'D/connected-quality-patch-538/root-exact-run001/audit-159/report.json'
OUT=HERE/'run-002-full/candidate.npz'
spec=importlib.util.spec_from_file_location('repair_focused_clearance_541_test',SRC)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class FocusedClearance541Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(CAND,allow_pickle=False) as z: cls.base={k:z[k] for k in z.files}
  cls.report=json.loads(AUDIT.read_text())
  cls.out,cls.meta=mod.build(cls.base,cls.report,max_flips=172)
  with np.load(OUT,allow_pickle=False) as z: cls.saved={k:z[k] for k in z.files}

 def test_input_candidate_and_complete_audit_are_pinned(self):
  self.assertEqual(mod.sha_file(CAND),mod.EXPECTED_CANDIDATE_SHA)
  self.assertEqual(mod.sha_file(AUDIT),mod.EXPECTED_AUDIT_SHA)
  self.assertEqual(self.report['candidate_sha256'],mod.EXPECTED_CANDIDATE_SHA)
  self.assertTrue(self.report['complete'])
  self.assertEqual(self.report['bank_count'],185)

 def test_exact_focus_banks_and_one_edge_ring(self):
  focus,counts=mod.focused_crossing_rows(self.report,len(self.base['source_O_faces']))
  self.assertEqual(counts,mod.FOCUS_BANK_PAIR_COUNTS)
  self.assertEqual(sum(counts.values()),2209)
  self.assertEqual(len(focus),1213)
  scope=mod.edge_one_ring(self.base['source_O_faces'],focus)
  self.assertEqual(len(scope),1925)
  self.assertTrue(focus<=scope)

 def test_full_batch_is_disjoint_and_inside_declared_scope(self):
  scope=mod.edge_one_ring(self.base['source_O_faces'],mod.focused_crossing_rows(self.report,len(self.base['source_O_faces']))[0])
  self.assertEqual(self.meta['candidate_edge_count'],172)
  self.assertEqual(self.meta['selected_flip_count'],90)
  touched=set()
  nodes=set()
  for op in self.meta['selected_flips']:
   rows=set(map(int,op['face_rows']))
   self.assertEqual(len(rows),2)
   self.assertTrue(rows<=scope)
   self.assertTrue(rows&touched==set())
   qnodes=set(map(int,self.base['source_O_faces'][list(rows)].ravel()))
   self.assertFalse(qnodes&nodes)
   touched|=rows;nodes|=qnodes
  changed=set(map(int,np.flatnonzero(np.any(self.out['source_O_faces']!=self.base['source_O_faces'],axis=1))))
  self.assertEqual(changed,touched)
  self.assertEqual(changed,set(self.meta['changed_face_rows']))

 def test_outside_faces_ancestry_points_and_anchors_are_unchanged(self):
  self.assertTrue(mod.identical(self.base['points_world_m'],self.out['points_world_m']))
  self.assertTrue(self.meta['outside_focus_faces_bitwise_unchanged'])
  self.assertTrue(self.meta['outside_focus_ancestry_rows_unchanged'])
  self.assertEqual(self.meta['typed_anchor_count'],237)
  anchors=self.base['typed_coarse_anchor_output_rows_018']
  self.assertTrue(mod.identical(self.base['points_world_m'][anchors],self.out['points_world_m'][anchors]))
  for k in self.base:
   if np.asarray(self.base[k]).ndim and np.asarray(self.base[k]).shape[0]==len(self.base['source_O_faces']):
    changed=set(self.meta['changed_face_rows']);mask=np.ones(len(self.base['source_O_faces']),dtype=bool);mask[list(changed)]=False
    self.assertTrue(np.array_equal(self.base[k][mask],self.out[k][mask],equal_nan=True),k)

 def test_fixed037_topology_manifold_and_boundary(self):
  self.assertTrue(np.array_equal(self.out['source_O_faces'],self.out['area_reference_037_source_O_faces']))
  _,b0,n0=mod.topology(self.base['source_O_faces']);_,b1,n1=mod.topology(self.out['source_O_faces'])
  self.assertFalse(n0);self.assertFalse(n1);self.assertEqual(b0,b1)

 def test_every_flip_keeps_material_source_compatibility_and_quality_guard(self):
  src=mod.csr_rows(self.base['face_source_ancestry_indptr'],self.base['face_source_ancestry_face_rows'],len(self.base['source_O_faces']))
  sub=mod.csr_rows(self.base['face_fine_subface_ancestry_indptr'],self.base['face_fine_subface_ancestry_rows'],len(self.base['source_O_faces']))
  detail=mod.csr_rows(self.base['face_fine_detail_ancestry_indptr'],self.base['face_fine_detail_ancestry_rows'],len(self.base['source_O_faces']))
  for op in self.meta['selected_flips']:
   i,j=op['face_rows']
   self.assertTrue(mod.compatible(self.base,i,j,src,sub,detail))
   self.assertEqual(int(self.base['face_kind'][i]),int(self.base['face_kind'][j]))
   self.assertGreaterEqual(op['quality']['after']['minimum_area_ratio'],mod.MIN_AREA_RATIO)
   self.assertGreater(op['quality']['after']['minimum_normal_cosine'],0.)
   self.assertGreaterEqual(op['quality']['after']['minimum_angle_deg'],mod.MIN_PATCH_ANGLE_DEG)

 def test_replay_output_and_nonadmission_receipt(self):
  for k in self.out:
   self.assertTrue(mod.identical(self.out[k],self.saved[k]),k)
  self.assertEqual(self.meta['status'],'FOCUSED_LOCAL_CONSTRUCTION_CANDIDATE_PENDING_ROOT_EXACT_AUDIT')
  self.assertFalse(self.meta['geometry_admitted'])
  self.assertFalse(self.meta['native_admission'])
  self.assertEqual(self.meta['accepted_physical_steps'],0)
  self.assertEqual(self.meta['exact_self_surface_and_185_bank_audits'],'NOT_RUN_PENDING_ROOT_SLOT')

 def test_incomplete_or_missing_focus_bank_fails_closed(self):
  bad=dict(self.report);bad['complete']=False
  with self.assertRaises(ValueError): mod.focused_crossing_rows(bad,len(self.base['source_O_faces']))
  bad=json.loads(json.dumps(self.report));bad['banks']=[b for b in bad['banks'] if not (b['body']==109 and b['stable_bone_id']==114)]
  with self.assertRaises(ValueError): mod.focused_crossing_rows(bad,len(self.base['source_O_faces']))

if __name__=='__main__': unittest.main()

