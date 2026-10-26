import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069"); D=ROOT/"D"; VAR=D/"D/soft-target-nonregression-470"
BASE=D/"constraint-project-backtrack-456/run-001/backtrack-373/pass-006/candidate.npz"
PREP=D/"connected-continuation-prep-457/run-001"; FACE_NPZ=PREP/"surface-targets-335/targets.npz"; FACE_REPORT=PREP/"surface-targets-335/report.json"
POINT_NPZ=PREP/"point-targets-390/nearest-union-exits.npz"; POINT_REPORT=PREP/"point-targets-390/nearest-union-exits.json"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
class Integration470(unittest.TestCase):
 def test_matched_rows_guards_and_owner_solver_install(self):
  wrapper=load("wrapper470_integration",VAR/"run_grouped_fit_470.py"); runner,solver=wrapper.load_dependencies()
  owner=load("owner408_470_integration",runner.OWNER); objective=load("objective458_470_integration",runner.HELPER)
  crossing=runner.prepare_crossing_inputs(BASE,BASE.parent/"audit-159/report.json")
  with np.load(BASE,allow_pickle=False) as z:
   points=np.asarray(z["points_world_m"],dtype=np.float32); anchors=np.asarray(z["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
  pt,ft=owner.load_clearance_target_inputs(POINT_NPZ,POINT_REPORT,FACE_NPZ,FACE_REPORT,BASE,points,crossing["faces"])
  bary=owner.load_module("barycentric_owner_470_test",owner.DEFAULT_BARY_OWNER)
  support=np.unique(np.concatenate((pt["point_rows"],ft["face_nodes"].ravel()))).astype(np.int64); pmm=points.astype(np.float64)*1000.
  original=owner.build_clearance_rows; ec,eh,ew,em=original(pmm,crossing["faces"],support,anchors,pt,ft,bary); n=len(pt["point_rows"])
  state=wrapper.install_variant(owner,objective,crossing,runner=runner,solver=solver)
  self.assertTrue(getattr(owner.solve_sparse_l1_clearance_in_basis,"__soft_target_no_regression_470__"))
  C,h,w,meta=owner.build_clearance_rows(pmm,crossing["faces"],support,anchors,pt,ft,bary)
  np.testing.assert_array_equal(C[:n].indptr,ec[:n].indptr); np.testing.assert_array_equal(C[:n].indices,ec[:n].indices); np.testing.assert_array_equal(C[:n].data,ec[:n].data)
  np.testing.assert_array_equal(h[:n],eh[:n]); np.testing.assert_array_equal(w[:n],ew[:n]); self.assertEqual(meta[:n],em[:n])
  self.assertEqual(state["objective"]["all_current_crossing_pairs_loaded"],3136)
  self.assertEqual(state["objective"]["three_corner_lower_bounds_per_pair"],9408)
  self.assertEqual(state["objective"]["nonpositive_corner_hard_rows"],4518)
  self.assertEqual(state["objective"]["positive_bone_plane_target_rows"],4890)
  self.assertEqual(len(owner.combine_barrier_rows([],[])),4518)
  self.assertEqual(state["solver_470"]["stage1_and_stage2_no_regression_rows_enforced"],0)
if __name__=="__main__": unittest.main()
