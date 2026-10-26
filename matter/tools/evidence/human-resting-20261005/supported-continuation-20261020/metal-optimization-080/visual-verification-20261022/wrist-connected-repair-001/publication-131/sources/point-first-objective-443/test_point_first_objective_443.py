import unittest,importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
spec=importlib.util.spec_from_file_location("p443",Path(__file__).with_name("point_first_objective_443.py"));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_keeps_point_rows_and_normalizes_mass(self):
  C=sparse.eye(3,format="csr");h=np.array([1.,2.,3.]);w=np.array([.25,.25,.5]);meta=[{"kind":"point","node":0},{"kind":"point","node":1},{"kind":"face"}]
  out=m.replacement_for(lambda*a:(C,h,w,meta))(None,None,None,None,{"point_rows":[0,1]},None,None)
  self.assertTrue(np.array_equal(out[0].toarray(),C[:2].toarray()));self.assertTrue(np.array_equal(out[1],h[:2]));self.assertTrue(np.array_equal(out[2],[.5,.5]));self.assertEqual([x["node"]for x in out[3]],[0,1])
 def test_no_points_fails_closed(self):
  with self.assertRaises(ValueError):m.replacement_for(lambda*a:(sparse.eye(1),np.ones(1),np.ones(1),[{"kind":"face"}]))(None,None,None,None,{"point_rows":[]},None,None)
if __name__=="__main__":unittest.main()
