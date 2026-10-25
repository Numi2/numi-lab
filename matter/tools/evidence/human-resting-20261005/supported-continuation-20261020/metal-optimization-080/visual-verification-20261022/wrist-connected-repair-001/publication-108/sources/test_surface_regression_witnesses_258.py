import importlib.util,unittest,copy,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location("w",Path(__file__).with_name("surface-regression-witnesses-258.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SurfaceTests(unittest.TestCase):
 def report(self,pairs):
  b=[dict(body=i,stable_bone_id=i,skin_aabb_active=0,result=None)for i in range(185)]
  b[0].update(skin_aabb_active=1,result=dict(count=len(pairs),triangle_pairs=pairs))
  return dict(complete=True,bank_count=185,expected_bank_count=185,surface_face_count=4,surface_point_count=7,banks=b)
 def test_exact_introduced_faces_only(self):
  before=self.report([[0,2],[1,4]]);after=self.report([[1,4],[2,8],[2,9]])
  fs,n=m.introduced_surface_faces(before,after,4);np.testing.assert_array_equal(fs,[2]);self.assertEqual(n,2)
 def test_no_introduced_witness_returns_empty(self):
  a=self.report([[0,2]]);fs,n=m.introduced_surface_faces(a,a,4);self.assertEqual(len(fs),0);self.assertEqual(n,0)
 def test_missing_duplicate_bad_range_and_unexplained_skip_rejected(self):
  changes=[lambda r:r.update(complete=False),lambda r:r["banks"].pop(),lambda r:r["banks"].__setitem__(1,r["banks"][0]),lambda r:r["banks"][0]["result"].update(triangle_pairs=[[4,1]]),lambda r:r["banks"][1].update(skin_aabb_active=2)]
  for f in changes:
   r=self.report([[0,2]]);f(r)
   with self.assertRaises(ValueError):m.typed_pairs(r,4)
if __name__=="__main__":unittest.main()
