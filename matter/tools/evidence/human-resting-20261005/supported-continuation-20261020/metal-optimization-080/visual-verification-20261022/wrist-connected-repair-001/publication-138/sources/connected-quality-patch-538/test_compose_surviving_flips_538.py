from __future__ import annotations
import importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).with_name('compose_surviving_flips_538.py')
D=Path('/Users/n/Work/numi-human-connected-surface-069/D')
spec=importlib.util.spec_from_file_location('compose_surviving_flips_538_test',SRC);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
class Composition538Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.pins=mod.require_inputs()
  cls.helper=mod.loadmod('replay_surviving_flips_534_for_538_test',mod.HELPER)
  cls.replay=mod.loadmod('cross_seam_search_405_for_538_test',mod.REPLAY)
  cls.base=mod.load_npz(mod.BASE);cls.full=mod.load_npz(mod.C531);cls.coord=mod.load_npz(mod.COORD)
  cls.report=json.loads(mod.R531.read_text())
  cls.p1=np.load(mod.P531,allow_pickle=False);cls.p2=np.load(mod.P536,allow_pickle=False)
  cls.pair_union=np.unique(np.sort(np.vstack((cls.p1,cls.p2)),axis=1),axis=0)
 def test_union_removes_all_incident_operations(self):
  kept,removed=self.helper.select_remaining(self.report['selected_flips'],self.pair_union.ravel())
  self.assertEqual((len(kept),len(removed)),(263,23))
  bad=set(map(int,self.pair_union.ravel()))
  self.assertFalse(any(bad & set(op['face_rows']) for _,op in kept))
  self.assertTrue(all(bad & set(x['face_rows']) for x in removed))
 def test_pins_and_exact_pair_counts_are_bound(self):
  self.assertEqual(self.pins['536_coordinate_candidate']['sha256'],'e880f1c96e8bb540f1918673fecd07e9dccfea54ebe5cefc91fc14a77307edc5')
  self.assertEqual(len(self.p1),85);self.assertEqual(len(self.p2),4)
  self.assertEqual(len(self.pair_union),89)
 def test_batch_replay_matches_405_sequential_semantics_on_three_real_operations(self):
  kept,_=self.helper.select_remaining(self.report['selected_flips'],self.pair_union.ravel())
  ops=[op for _,op in kept[:3]]
  batch=self.helper.apply_ops_batch(self.base,ops,self.replay)
  sequential=self.base
  for op in ops:sequential=self.replay.apply_op(sequential,op)
  for k in batch:self.assertTrue(self.helper.identical(batch[k],sequential[k]),k)
 def test_composition_preserves_only_requested_coordinate_branch(self):
  out,meta=mod.compose(self.base,self.full,self.report,self.p1,self.p2,self.coord,self.helper,self.replay)
  self.assertEqual(meta['retained_flip_count'],263)
  self.assertTrue(self.helper.identical(out['points_world_m'],self.coord['points_world_m']))
  self.assertTrue(np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']))
  self.assertFalse(meta['geometry_admitted'])
  changed=set(meta['changed_face_rows'])
  for key in self.base:
   if key not in self.helper.ALLOWED_CHANGED_KEYS and key!='points_world_m':
    self.assertTrue(self.helper.identical(self.base[key],out[key]),key)
  self.assertFalse({int(x) for x in self.pair_union.ravel()} & changed)
 def test_wrong_source_pin_fails_closed(self):
  old=mod.PINS['536_coordinate_candidate'];mod.PINS['536_coordinate_candidate']=('bad',old[1])
  try:
   with self.assertRaises(ValueError):mod.require_inputs()
  finally:mod.PINS['536_coordinate_candidate']=old
if __name__=='__main__':unittest.main()

