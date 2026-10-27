import importlib.util,unittest,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location("r614",Path(__file__).with_name("restrict_refined_field614.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def payload(self):
  p={"points_world_m":np.arange(720,dtype=np.float32).reshape(240,3),"typed_coarse_anchor_output_rows_018":np.arange(237),"area_reference_037_points_world_m_f64":np.arange(720,dtype=np.float64).reshape(240,3),"other":np.array([42])}
  r={k:v.copy()for k,v in p.items()};r["points_world_m"]=np.vstack([p["points_world_m"],np.zeros((1,3),np.float32)]);r["area_reference_037_points_world_m_f64"]=np.vstack([p["area_reference_037_points_world_m_f64"],np.zeros((1,3))])
  r.update(refinement_input_candidate_sha256=np.array("parent"),immediate_refinement_prolongation_indptr=np.arange(242),immediate_refinement_prolongation_node_rows=np.arange(241),immediate_refinement_prolongation_weights=np.ones(241))
  t={k:v.copy()for k,v in r.items()};t["points_world_m"][239,0]+=1
  return p,r,t
 def test_preserves_parent_payload(self):
  p,r,t=self.payload();z=m.restrict(p,r,t,"parent");self.assertEqual(set(z),set(p));self.assertTrue(m.same(z["other"],p["other"]));self.assertEqual(z["points_world_m"][239,0],t["points_world_m"][239,0])
 def test_parent_pin(self):
  p,r,t=self.payload()
  with self.assertRaises(ValueError):m.restrict(p,r,t,"wrong")
 def test_anchor(self):
  p,r,t=self.payload();t["points_world_m"][1,0]+=1
  with self.assertRaises(ValueError):m.restrict(p,r,t,"parent")
 def test_identity(self):
  p,r,t=self.payload();r["immediate_refinement_prolongation_node_rows"][3]=4
  with self.assertRaises(ValueError):m.restrict(p,r,t,"parent")
 def test_prefix(self):
  p,r,t=self.payload();r["points_world_m"][239,0]+=1
  with self.assertRaises(ValueError):m.restrict(p,r,t,"parent")
if __name__=="__main__":unittest.main()
