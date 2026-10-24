import importlib.util,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location("emission_geom",Path(__file__).resolve().parents[1]/"tools/human_skin_patch_geometry.py")
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class Float32WorldEmissionTests(unittest.TestCase):
 def test_rigid_world_translation_can_collapse_a_locally_valid_triangle(self):
  local=np.array([[0,0,0],[1e-8,0,0],[0,1e-3,0]],np.float32)
  g.validate_float32_surface_emission(local,local,[[0,1,2]])
  world=local.astype(float)+np.array([1.,-1.,.1])
  with self.assertRaisesRegex(ValueError,"rows 0"):
   g.validate_float32_surface_emission(world,world.astype(np.float32),[[0,1,2]])
 def test_reversal_is_rejected_without_removing_faces(self):
  x=np.array([[0,0,0],[1,0,0],[0,1,0]],float);y=x[[0,2,1]].astype(np.float32)
  with self.assertRaisesRegex(ValueError,"reverses"):
   g.validate_float32_surface_emission(x,y,[[0,1,2]])
 def test_valid_final_frame_and_empty_faces(self):
  x=np.array([[1,2,3],[1.001,2,3],[1,2.001,3]],float)
  g.validate_float32_surface_emission(x,x.astype(np.float32),[[0,1,2]])
  g.validate_float32_surface_emission(x,x.astype(np.float32),np.empty((0,3),np.int64))
 def test_invalid_inputs_fail(self):
  x=np.array([[0,0,0],[1,0,0],[0,1,0]],float);y=x.astype(np.float32)
  for xx,yy,f in [(x,x,[[0,1,2]]),(x,y,[[0,1,3]]),(x,y,[[0.,1.,2.]]),(x,y,[[-1,1,2]]),(x,y,[[0,0,2]]),(x*np.nan,y,[[0,1,2]])]:
   with self.assertRaises(ValueError):g.validate_float32_surface_emission(xx,yy,f)
if __name__=="__main__":unittest.main()
