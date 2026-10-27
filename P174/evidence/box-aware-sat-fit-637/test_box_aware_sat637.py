import importlib.util, sys, tempfile, unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import box_aware_sat637 as sat637
import run_box_aware_sat637 as runner

def load_objective():
 p=sat637.OBJ458;s=importlib.util.spec_from_file_location("obj458_test637",p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class SatSelectorTests(unittest.TestCase):
 def setUp(self):
  self.skin=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  self.bone=np.array([[.2,.2,-.2],[.8,.2,.2],[.2,.8,.2]])
  self.nodes=np.array([10,11,12],dtype=np.int64)
  self.support=self.nodes.copy()
 def test_selects_individually_feasible_axis(self):
  lo=np.tile([-3.,-3.,-3.],3);hi=-lo
  p=sat637.choose_pair_sat_plane_with_bounds(self.skin,self.bone,self.nodes,self.support,lo,hi)
  self.assertTrue(p["box_feasible"]);self.assertEqual(p["axis_strategy"],"finite_pair_sat_box_aware_soft_target_637")
  self.assertEqual(p["normal_world_unit"].shape,(3,));self.assertLessEqual(p["box_max_unmet_mm"],2e-9)
  self.assertFalse(p["whole_bank_clearance_certificate"])
 def test_zero_box_uses_least_unmet_sat_not_infinite_fallback(self):
  zeros=np.zeros(9);p=sat637.choose_pair_sat_plane_with_bounds(self.skin,self.bone,self.nodes,self.support,zeros,zeros)
  self.assertFalse(p["box_feasible"])
  axes=sat637.sat._signed_sat_axes(self.skin,self.bone)
  vals=[]
  for a in axes:
   n=np.asarray(a["normal_world_unit"]);rhs=float(np.max(self.bone@n))+sat637.sat._float32_world_m_emission_margin_mm(self.skin,self.bone,n)-self.skin@n
   vals.append(float(np.maximum(rhs,0.).max()))
  self.assertAlmostEqual(p["box_max_unmet_mm"],min(vals),places=10)
  self.assertNotEqual(p["axis_strategy"],"bone_face_458_compatible")
  self.assertGreater(p["candidate_axis_count"],1)
 def test_rejects_bad_bounds_and_pair_identity(self):
  with self.assertRaises(ValueError):sat637.choose_pair_sat_plane_with_bounds(self.skin,self.bone,[10,11,13],self.support,np.zeros(9),np.zeros(9))
  with self.assertRaises(ValueError):sat637.choose_pair_sat_plane_with_bounds(self.skin,self.bone,self.nodes,self.support,np.zeros(8),np.zeros(8))
  with self.assertRaises(ValueError):sat637.choose_pair_sat_plane_with_bounds(self.skin,self.bone,self.nodes,self.support,np.ones(9),np.ones(9))

class BuilderIntegrationTests(unittest.TestCase):
 def setUp(self):
  self.objective=load_objective()
  self.points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  self.faces=np.array([[0,1,2]],dtype=np.int64)
  self.bone=np.array([[.2,.2,-.2],[.8,.2,.2],[.2,.8,.2]])
  self.bones={(1,2,3):self.bone}
  self.pairs=[{"face_row":0,"body":1,"bone_id":2,"triangle_id":3}]
  class Owner:pass
  self.owner=Owner()
  def old_bounds(points,source,support,anchors,trust,cap):
   return np.repeat(-1.,3*len(support)),np.repeat(1.,3*len(support)),{"maximum_component_bound_mm":1.0}
  self.owner.source_safe_trust_bounds=old_bounds
 def test_458_builder_keeps_soft_rows_and_selected_plane_metadata(self):
  st=sat637.install(self.owner,self.objective,{},source_ref_mm=self.points.copy(),base_sha="a"*64,source_sha="b"*64)
  rows=self.objective.build_fixed_bone_outward_plane_rows(self.points,self.faces,np.arange(3),[],self.pairs,self.bones,total_weight=1.)
  self.assertEqual(len(st["selected"]),1)
  self.assertEqual(rows["diagnostics"]["input_pair_count"],1)
  self.assertAlmostEqual(float(rows["weights"].sum()),1.0,places=12)
  self.assertEqual(rows["metadata"][0]["plane_selection"],"finite_pair_sat_box_aware_soft_target_637")
  self.assertFalse(rows["metadata"][0]["axis_individually_box_feasible"] is None)
  np.testing.assert_allclose(rows["lower"],st["selected"][0]["corner_lower_mm"],rtol=0,atol=1e-12)
  lo,hi,report=self.owner.source_safe_trust_bounds(self.points,self.points,np.arange(3),[],4.,12.)
  self.assertTrue(report["old408_cube_contained"])
  self.assertTrue(np.all(lo<=-1.) and np.all(hi>=1.))
 def test_458_invocation_identity_mismatch_fails_closed(self):
  st=sat637.install(self.owner,self.objective,{},source_ref_mm=self.points.copy(),base_sha="c"*64,source_sha="d"*64)
  class Changing(dict):
   calls=0
   def __getitem__(self,k):
    self.calls+=1
    return self.bone if self.calls==1 else self.bone+np.array([0.1,0.,0.])
  changing=Changing(self.bones);changing.bone=self.bone
  with self.assertRaisesRegex(ValueError,"fixed bone triangle differs"):
   self.objective.build_fixed_bone_outward_plane_rows(self.points,self.faces,np.arange(3),[],self.pairs,changing,total_weight=1.)
 def test_support_order_must_match_selector_bounds(self):
  sat637.install(self.owner,self.objective,{},source_ref_mm=self.points.copy(),base_sha="e"*64,source_sha="f"*64)
  self.objective.build_fixed_bone_outward_plane_rows(self.points,self.faces,np.arange(3),[],self.pairs,self.bones,total_weight=1.)
  with self.assertRaisesRegex(ValueError,"bounds/support order"):
   self.owner.source_safe_trust_bounds(self.points,self.points,np.array([1,0,2]),[],4.,12.)

class LauncherTests(unittest.TestCase):
 def test_frozen603_patch_is_exactly_one_insertion(self):
  text=runner.FROZEN.read_text()
  patched=runner.patch_source(text)
  self.assertEqual(patched.count(runner.INJECT),1)
  self.assertEqual(patched.count(runner.NEEDLE),1)
  with self.assertRaises(ValueError):runner.patch_source(text.replace(runner.NEEDLE,""))
 def test_launch_receipt_does_not_create_forbidden_output_directory(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/"fit"
   receipt=runner.write_launch_receipt(out,{"x":1})
   self.assertFalse(out.exists());self.assertTrue(receipt.is_file())
 def test_output_receipt_roundtrips_json_and_npz(self):
  state={"selected":[],"base_candidate_sha256":"0"*64,"source_base_sha256":"1"*64,"source_reference_sha256":"2"*64,
   "trust_mm":4.,"source_cap_mm":12.,"soft_row_count":0,"soft_pair_count":0,"soft_weight_mass":0.,"corner_guard_count":0}
  with tempfile.TemporaryDirectory() as d:
   report,npz=sat637.write_result(state,d)
   self.assertTrue(report.is_file());self.assertTrue(npz.is_file())
   import json
   parsed=json.loads(report.read_text());self.assertEqual(parsed["pair_count"],0);self.assertEqual(parsed["individually_box_infeasible_count"],0)
   with np.load(npz,allow_pickle=False) as data:self.assertEqual(data["pair_identity"].shape,(0,4))

if __name__=="__main__":unittest.main()
