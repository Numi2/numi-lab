import importlib.util,unittest,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location("proposal250",Path(__file__).with_name("propose_joint_clearance_witnesses_274.py"))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class DirectionTests(unittest.TestCase):
 def fixture(self):
  return dict(nearest_boundary_direction_world_unit=np.array([[-1.,0,0]]),proposal_direction_world_unit=np.array([[1.,0,0]]),source_surface_normal_world_unit=np.array([[1.,0,0]]),source_normal_valid=np.array([True]))
 def test_legacy_nearest_preserved(self):
  np.testing.assert_array_equal(m.select_face_target_directions(self.fixture(),{}),[[-1,0,0]])
 def test_outward_uses_proposal_not_opposite_diagnostic(self):
  np.testing.assert_array_equal(m.select_face_target_directions(self.fixture(),{"direction_mode":"source-outward"}),[[1,0,0]])
 def test_missing_unvalidated_nonunit_mismatched_and_unknown_fail_closed(self):
  for field,value in [("source_normal_valid",[False]),("proposal_direction_world_unit",[[-1.,0,0]]),("source_surface_normal_world_unit",[]),("proposal_direction_world_unit",[[float("nan"),0,0]])]:
   f=self.fixture();f[field]=value
   with self.assertRaises(ValueError):m.select_face_target_directions(f,{"direction_mode":"source-outward"})
  f=self.fixture();f["proposal_direction_world_unit"]=[[2.,0,0]];f["source_surface_normal_world_unit"]=[[2.,0,0]]
  with self.assertRaises(ValueError):m.select_face_target_directions(f,{"direction_mode":"source-outward"})
  with self.assertRaises(ValueError):m.select_face_target_directions(self.fixture(),{"direction_mode":"arbitrary"})
 def test_outward_nearest_accepts_only_nonnegative_source_hemisphere_direction(self):
  f=self.fixture();f["proposal_direction_world_unit"]=[[0.6,0.8,0.0]];f["source_surface_normal_world_unit"]=[[1.,0.,0.]]
  got=m.select_face_target_directions(f,{"direction_mode":"outward-nearest"})
  np.testing.assert_array_equal(got,[[0.6,0.8,0.0]])
  f=self.fixture();f["proposal_direction_world_unit"]=[[-1e-13,1.,0.]];f["source_surface_normal_world_unit"]=[[1.,0.,0.]]
  with self.assertRaises(ValueError):m.select_face_target_directions(f,{"direction_mode":"outward-nearest"})

 def test_outward_nearest_requires_valid_unit_normal_and_unit_proposal(self):
  cases=[]
  f=self.fixture();f["source_normal_valid"]=[False];cases.append(f)
  f=self.fixture();f["source_surface_normal_world_unit"]=[[0.,0.,0.]];cases.append(f)
  f=self.fixture();f["proposal_direction_world_unit"]=[[2.,0.,0.]];cases.append(f)
  f=self.fixture();f["source_surface_normal_world_unit"]=[[2.,0.,0.]];cases.append(f)
  for f in cases:
   with self.assertRaises(ValueError):m.select_face_target_directions(f,{"direction_mode":"outward-nearest"})

 def test_source_outward_and_default_nearest_contracts_remain_unchanged(self):
  np.testing.assert_array_equal(m.select_face_target_directions(self.fixture(),{}),[[-1,0,0]])
  np.testing.assert_array_equal(m.select_face_target_directions(self.fixture(),{"direction_mode":"source-outward"}),[[1,0,0]])

if __name__=="__main__":unittest.main()
