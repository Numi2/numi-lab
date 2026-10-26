from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).with_name('replay_surviving_flips_534.py')
D=Path('/Users/n/Work/numi-human-connected-surface-069/D')
BASE=D/'finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz'
ALL=D/'connected-quality-patch-531/run-002/candidate.npz'
REPORT=D/'connected-quality-patch-531/run-002/report.json'
PAIRS=D/'connected-quality-patch-531/root-exact-run002/audit-130/surface-self-pairs.npy'
REPLAY=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
spec=importlib.util.spec_from_file_location('replay_surviving_flips_534_test',SRC);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class Replay534Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pins=mod.verify_pins(BASE,ALL,REPORT,D/'connected-quality-patch-531/root-exact-run002/audit-130/summary.json',PAIRS,REPLAY)
  cls.ops=json.loads(REPORT.read_text())['selected_flips']
  cls.pairs=np.load(PAIRS,allow_pickle=False)
 def test_manifest_matches_actual_candidate_arrays(self):
  with np.load(BASE,allow_pickle=False) as z:base={k:z[k] for k in z.files}
  with np.load(ALL,allow_pickle=False) as z:full={k:z[k] for k in z.files}
  receipt=mod.validate_operation_manifest(base,full,json.loads(REPORT.read_text()))
  self.assertEqual(receipt['operation_count'],286)
  self.assertEqual(receipt['operation_face_rows'],572)

 def test_removes_entire_operations_touching_either_self_pair_face(self):
  kept,removed=mod.select_remaining(self.ops,self.pairs.ravel())
  self.assertEqual((len(kept),len(removed)),(264,22))
  bad=set(map(int,self.pairs.ravel()))
  self.assertFalse(any(bad & set(op['face_rows']) for _,op in kept))
  self.assertTrue(all(set(x['face_rows']) & bad for x in removed))
 def test_input_pins_match_actual_failed_audit(self):
  self.assertEqual(self.pins['531 candidate']['sha256'],'32ec169a84ce4ed22915c0ffb9ea110cac6d7962c682bad9e6d025b0b03f5e4e')
  self.assertEqual(self.pins['self-pair witness']['sha256'],'129927157f3a00a39d521048cb94d9ff24f4cb5baa87822450f6330f22f9416d')
 def test_batch_replay_matches_405_sequential_semantics_on_three_real_ops(self):
  with np.load(BASE,allow_pickle=False) as z:base={k:z[k] for k in z.files}
  replay=mod.loadmod('cross_seam_search_405_test534',REPLAY)
  ops=[op for _,op in mod.select_remaining(self.ops,self.pairs.ravel())[0][:3]]
  batch=mod.apply_ops_batch(base,ops,replay)
  seq=base
  for op in ops:seq=replay.apply_op(seq,op)
  self.assertEqual(set(batch),set(seq))
  for k in batch:self.assertTrue(mod.identical(batch[k],seq[k]),k)
 def test_all_remaining_batch_preserves_scope_and_lineage(self):
  with np.load(BASE,allow_pickle=False) as z:base={k:z[k] for k in z.files}
  with np.load(ALL,allow_pickle=False) as z:full={k:z[k] for k in z.files}
  replay=mod.loadmod('cross_seam_search_405_test534_all',REPLAY)
  kept,removed=mod.select_remaining(self.ops,self.pairs.ravel())
  out,receipt=mod.build(base,json.loads(REPORT.read_text()),self.pairs,replay)
  self.assertEqual(receipt['remaining_operation_count'],264)
  self.assertEqual(receipt['removed_operation_count'],22)
  self.assertTrue(mod.identical(base['points_world_m'],out['points_world_m']))
  self.assertTrue(np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']))
  self.assertEqual(set(map(int,receipt['changed_face_rows'])),{r for _,op in kept for r in op['face_rows']})
  self.assertFalse({int(x) for x in self.pairs.ravel()} & set(receipt['changed_face_rows']))
  for key in base:
   if key not in mod.ALLOWED_CHANGED_KEYS:self.assertTrue(mod.identical(base[key],out[key]),key)
  changed=set(receipt['changed_face_rows'])
  replay=mod.loadmod('cross_seam_search_405_csr534',REPLAY)
  csr_pairs=[('face_source_ancestry_indptr','face_source_ancestry_face_rows'),
   ('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),
   ('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows')]
  for ptr,vals in csr_pairs:
   old_rows=replay.csr_rows(base[ptr],base[vals],len(base['source_O_faces']))
   new_rows=replay.csr_rows(out[ptr],out[vals],len(out['source_O_faces']))
   for row in set(range(len(old_rows)))-changed:self.assertEqual(old_rows[row],new_rows[row],f'{ptr} row {row}')
  old_detail=replay.detail_csr(base,{9570,9577,10226})
  new_detail=replay.detail_csr(out,{9570,9577,10226})
  for row in set(range(len(old_detail)))-changed:self.assertEqual(old_detail[row],new_detail[row],f'detail row {row}')
  self.assertFalse(receipt['geometry_admitted'])
 def test_hash_mismatch_fails_closed(self):
  with self.assertRaises(ValueError):
   mod.verify_pins(BASE,ALL,REPORT,D/'connected-quality-patch-531/root-exact-run002/audit-130/summary.json',PAIRS,D/'missing-owner.py')

if __name__=='__main__':unittest.main()

