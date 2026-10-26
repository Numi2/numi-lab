import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import f32_area_gate_409_next as gate

class AreaGateTests(unittest.TestCase):
 def fixture(self):
  p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)*np.float32(.01)
  f=np.asarray([[0,1,2],[0,3,1]],np.int32)
  return p,f,p.astype(np.float64)
 def test_identical_emitted_geometry_passes_full_gate(self):
  p,f,r=self.fixture(); q=gate.f32_area_floor(p,p,r,f,floor=.1)
  self.assertTrue(q["pass"]); self.assertEqual(q["scope"],"all faces")
  self.assertAlmostEqual(q["minimum_fixed_037_reference_magnitude_ratio"],1.)
 def test_selected_face_gate_detects_reference_magnitude_collapse(self):
  p,f,r=self.fixture(); t=p.copy(); t[1]=np.asarray([.0001,0,0],np.float32)
  q=gate.f32_area_floor(t,p,r,f,[0],floor=.1)
  self.assertFalse(q["pass"]); self.assertEqual(q["bad_fixed_037_reference_face_rows"],[0])
  self.assertEqual(q["face_rows"],[0])
 def test_selected_face_gate_detects_orientation_reversal(self):
  p,f,r=self.fixture(); t=p.copy(); t[1]=np.asarray([-.01,0,0],np.float32)
  q=gate.f32_area_floor(t,p,r,f,[0],floor=.1)
  self.assertFalse(q["pass"]); self.assertEqual(q["bad_current_oriented_face_rows"],[0])
 def test_preserves_floor_tolerance_semantics(self):
  p,f,r=self.fixture(); t=p.copy(); t[1]=np.asarray([.001,0,0],np.float32)
  q=gate.f32_area_floor(t,p,r,f,[0],floor=.1)
  self.assertTrue(q["pass"])
 def test_rejects_wrong_dtype_empty_or_invalid_rows(self):
  p,f,r=self.fixture()
  with self.assertRaises(ValueError): gate.f32_area_floor(p.astype(np.float64),p,r,f)
  with self.assertRaises(ValueError): gate.f32_area_floor(p,p,r,f,[])
  with self.assertRaises(ValueError): gate.f32_area_floor(p,p,r,f,[0,0])

if __name__=="__main__": unittest.main()
