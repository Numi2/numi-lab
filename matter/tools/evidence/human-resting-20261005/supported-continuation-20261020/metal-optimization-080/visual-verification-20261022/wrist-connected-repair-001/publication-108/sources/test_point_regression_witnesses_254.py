import importlib.util,unittest,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location("w",Path(__file__).with_name("point-regression-witnesses-254.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class WitnessTests(unittest.TestCase):
 def test_exact_incidence_and_duplicate_bank_witness(self):
  F=np.array([[0,1,2],[1,3,2],[4,5,6]])
  after=dict(inside_node_bank_pairs=2,inside_witnesses=[dict(output_node=1,classification="inside")]*2)
  rows,fs=m.regressed_point_faces(dict(inside_node_bank_pairs=0),after,F,7)
  np.testing.assert_array_equal(rows,[1]);np.testing.assert_array_equal(fs,[0,1])
 def test_incomplete_invalid_or_nonzero_reference_rejected(self):
  F=np.array([[0,1,2]])
  for before,after in [(1,dict(inside_node_bank_pairs=1,inside_witnesses=[dict(output_node=1,classification="inside")])),(0,dict(inside_node_bank_pairs=2,inside_witnesses=[dict(output_node=1,classification="inside")])),(0,dict(inside_node_bank_pairs=1,inside_witnesses=[dict(output_node=8,classification="inside")])),(0,dict(inside_node_bank_pairs=1,inside_witnesses=[dict(output_node=1,classification="outside")]))]:
   with self.assertRaises(ValueError):m.regressed_point_faces(dict(inside_node_bank_pairs=before),after,F,3)
if __name__=="__main__":unittest.main()
