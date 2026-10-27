import unittest,numpy as np
from select_components631_v2 import labels_for_changes,component_deltas,bank_pairs,PRED
class Tests(unittest.TestCase):
 def test_zero_bridge_separates(self):
  x=np.zeros((5,3),np.float32);y=x.copy();y[[0,1,3,4],0]=1
  lab,owner=labels_for_changes(x,y,np.array([[0,1,2],[2,3,4]]))
  self.assertEqual(lab[0],lab[1]);self.assertEqual(lab[3],lab[4]);self.assertNotEqual(lab[0],lab[3]);self.assertEqual(lab[2],-1)
 def test_changed_edge_connects(self):
  x=np.zeros((3,3),np.float32);y=x.copy();y[[0,2],0]=1
  lab,_=labels_for_changes(x,y,np.array([[0,1,2]]));self.assertEqual(lab[0],lab[2])
 def test_nonfinite_and_dtype(self):
  x=np.zeros((3,3),np.float32);y=x.copy();y[0,0]=np.nan
  with self.assertRaises(ValueError):labels_for_changes(x,y,np.array([[0,1,2]]))
  with self.assertRaises(ValueError):labels_for_changes(x.astype(float),x,np.array([[0,1,2]]))
 def test_bankwise_selection(self):
  b={(1,1):{(0,0),(1,0)},(2,2):set()};t={(1,1):{(1,0)},(2,2):{(1,2)}}
  ds,keep=component_deltas(np.array([5,6]),b,t)
  self.assertEqual(keep,{5});self.assertEqual(ds[5][(1,1)],-1)
 def test_unchanged_face_mismatch(self):
  with self.assertRaises(ValueError):component_deltas(np.array([-1]),{(1,1):set()},{(1,1):{(0,0)}})
 def test_empty_no_selection(self):
  ds,keep=component_deltas(np.array([-1]),{(1,1):set()},{(1,1):set()});self.assertEqual(ds,{});self.assertFalse(keep)
 def test_report_inventory(self):
  d={"candidate_sha256":"h","complete":True,"bank_count":185,"expected_bank_count":185,"exact_predicate_sha256":PRED,"banks":[{"body":j,"stable_bone_id":j,"result":{"count":0,"triangle_pairs":[]}}for j in range(185)]}
  self.assertEqual(len(bank_pairs(d,"h",2)),185)
  d["banks"][1]=d["banks"][0]
  with self.assertRaises(ValueError):bank_pairs(d,"h",2)
 def test_empty_aabb_result(self):
  d={"candidate_sha256":"h","complete":True,"bank_count":185,"expected_bank_count":185,"exact_predicate_sha256":PRED,"banks":[{"body":j,"stable_bone_id":j,"skin_aabb_active":0,"bank_aabb_active":0,"result":None}for j in range(185)]}
  self.assertEqual(len(bank_pairs(d,"h",2)),185)
  d["banks"][0]["skin_aabb_active"]=1
  with self.assertRaises(ValueError):bank_pairs(d,"h",2)
if __name__=="__main__":unittest.main()

