from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np

SRC=Path(__file__).with_name('repair_eligible_patches_531.py')
CAND=Path('/Users/n/Work/numi-human-connected-surface-069/D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz')
AUDIT=Path('/Users/n/Work/numi-human-connected-surface-069/D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/audit-159/report.json')
spec=importlib.util.spec_from_file_location('repair_eligible_patches_531_test',SRC);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class EligiblePatch531Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with np.load(CAND,allow_pickle=False) as z:cls.base={k:z[k] for k in z.files}
  cls.report=json.loads(AUDIT.read_text())
  cls.out,cls.receipt=mod.build(cls.base,cls.report,max_flips=24)
  cls.q=mod.face_quality(cls.base['points_world_m'].astype(float)*1000,cls.base['area_reference_037_points_world_m_f64']*1000,cls.base['source_O_faces'],cls.base['area_reference_037_source_O_faces'])
  cls.cross=mod.crossing_face_rows(cls.report,len(cls.base['source_O_faces']))
  cls.seeds,cls.low,cls.sharp=mod.seed_face_rows(cls.base,cls.q,cls.cross)

 def test_exact_witnesses_and_low_area_drive_seedset(self):
  self.assertTrue(self.cross <= self.seeds)
  self.assertIn(40714,self.low)
  self.assertIn(40714,self.seeds)
  # A negative normal alone must not create a seed.
  ix=np.flatnonzero((self.q['normal_cos']<=0)&(self.q['area_ratio']>=mod.LOW_AREA_SEED_RATIO)&(self.q['min_angle_deg']>=mod.SHARP_SEED_ANGLE_DEG))
  normal_only=next((int(i) for i in ix if int(i) not in self.cross),None)
  self.assertIsNotNone(normal_only)
  self.assertNotIn(normal_only,self.seeds)

 def test_generated_batch_is_fixed_coordinate_and_manifold(self):
  self.assertGreater(self.receipt['selected_flip_count'],0)
  self.assertTrue(mod.identical(self.base['points_world_m'],self.out['points_world_m']))
  self.assertTrue(np.array_equal(self.out['source_O_faces'],self.out['area_reference_037_source_O_faces']))
  _,b0,n0=mod.topology(self.base['source_O_faces']);_,b1,n1=mod.topology(self.out['source_O_faces'])
  self.assertFalse(n0);self.assertFalse(n1);self.assertEqual(b0,b1)
  self.assertEqual(len(self.base['source_O_faces']),len(self.out['source_O_faces']))

 def test_flip_eligibility_keeps_source_and_material_lineage(self):
  for op in self.receipt['selected_flips']:
   rows=op['face_rows']
   self.assertEqual(op['material_kind'],int(self.base['face_kind'][rows[0]]))
   self.assertEqual(int(self.base['face_kind'][rows[0]]),int(self.base['face_kind'][rows[1]]))
   self.assertEqual(int(self.base['parent_fine_detail_face_row'][rows[0]]),int(self.base['parent_fine_detail_face_row'][rows[1]]))
   self.assertEqual(int(self.base['parent_fine_subface_row'][rows[0]]),int(self.base['parent_fine_subface_row'][rows[1]]))
   self.assertEqual(self.base['parent_source_face_marker'][rows[0]],self.base['parent_source_face_marker'][rows[1]])
   self.assertEqual(self.base['parent_coarse_face_row'][rows[0]],self.base['parent_coarse_face_row'][rows[1]])
   src=mod.csr_rows(self.base['face_source_ancestry_indptr'],self.base['face_source_ancestry_face_rows'],len(self.base['source_O_faces']))
   sub=mod.csr_rows(self.base['face_fine_subface_ancestry_indptr'],self.base['face_fine_subface_ancestry_rows'],len(self.base['source_O_faces']))
   parent=mod.csr_rows(self.base['refinement_face_parent_ancestry_indptr'],self.base['refinement_face_parent_ancestry_face_rows'],len(self.base['source_O_faces']))
   self.assertEqual(op['source_ancestry_union'],sorted(set(src[rows[0]])|set(src[rows[1]])))
   self.assertEqual(op['subface_ancestry_union'],sorted(set(sub[rows[0]])|set(sub[rows[1]])))
   self.assertEqual(op['immediate_parent_union'],sorted(set(parent[rows[0]])|set(parent[rows[1]])))
   self.assertTrue(op['immediate_parent_union'])

 def test_area_orientation_and_angle_bounds(self):
  q=mod.face_quality(self.out['points_world_m'].astype(float)*1000,self.out['area_reference_037_points_world_m_f64']*1000,self.out['source_O_faces'],self.out['area_reference_037_source_O_faces'])
  for op in self.receipt['selected_flips']:
   for row in op['face_rows']:
    self.assertGreaterEqual(float(q['area_ratio'][row]),mod.MIN_AREA_RATIO)
    self.assertGreater(float(q['normal_cos'][row]),0.)
    self.assertGreaterEqual(float(q['min_angle_deg'][row]),mod.MIN_PATCH_ANGLE_DEG)
   self.assertGreaterEqual(op['quality']['after']['minimum_angle_deg']-op['quality']['before']['minimum_angle_deg'],-mod.MAX_MIN_ANGLE_LOSS_DEG-1e-9)

 def test_noncoordinate_diff_is_declared_and_not_admitted(self):
  changed=[k for k in self.base if not mod.identical(self.base[k],self.out[k])]
  added=set(self.out)-set(self.base)
  self.assertFalse(set(changed)-mod.ALLOWED_CHANGED_KEYS,changed)
  self.assertFalse(added-mod.ALLOWED_CHANGED_KEYS,added)
  self.assertFalse(self.receipt['geometry_admitted'])
  self.assertEqual(self.receipt['accepted_physical_steps'],0)
  self.assertEqual(self.receipt['exact_full_self_surface_and_185_bank_audits'],'NOT_RUN_PENDING_ROOT_SLOT')

if __name__=='__main__':unittest.main()
