import importlib.util,sys,tempfile,unittest,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
DRIVER=ROOT/"D/local-two-parent-retriangulation-408/exact_global_incremental_flip_search_409.py"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
 sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
driver=load("exact_global_incremental_flip_search_409",DRIVER)
class Search409Tests(unittest.TestCase):
 def test_zero_accept_smoke_runs_single_complete_baseline_and_preserves_402_arrays(self):
  xy=[(0.,0.),(.4,0.),(.4,.2),(0.,.6),(.8,0.),(.8,.2)]
  points=np.asarray([(x,y,0.) for x,y in xy],np.float32)
  faces=np.asarray([[1,3,0],[3,1,2],[1,4,2],[4,5,2]],np.int64)
  n=len(faces); bary=np.broadcast_to(np.eye(3),(n,3,3)).copy()
  c={"points_world_m":points,"source_O_faces":faces,
     "area_reference_037_points_world_m_f64":points.astype(np.float64),
     "area_reference_037_source_O_faces":faces.copy(),"face_kind":np.ones(n,np.int8),
     "parent_fine_detail_face_row":np.asarray([10,10,20,20],np.int32),
     "parent_fine_subface_row":np.asarray([30,30,40,40],np.int32),
     "parent_coarse_face_row":np.full(n,5,np.int32),"parent_source_face_marker":np.asarray([70,71,80,81],np.int32),
     "refinement_face_parent_face_row":np.asarray([100,101,200,201],np.int64),
     "refinement_face_parent_barycentric":bary,
     "face_source_ancestry_indptr":np.arange(n+1,dtype=np.int64),
     "face_source_ancestry_face_rows":np.asarray([500,500,600,600],np.int32),
     "face_fine_subface_ancestry_indptr":np.arange(n+1,dtype=np.int64),
     "face_fine_subface_ancestry_rows":np.asarray([30,30,40,40],np.int32),
     "coarse_boundary_anchor_rows":np.asarray([0,5],np.int64),
     "typed_coarse_anchor_output_rows_018":np.asarray([0,5],np.int64),
     "typed_coarse_anchor_source_ids_018":np.asarray([100,105],np.int64)}
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);inp=td/"candidate.npz";np.savez_compressed(inp,**c)
   report=driver.run(inp,[10,20],[0,1,2,3],td/"out",max_accept=0)
   self.assertTrue(report["baseline_audit_complete"])
   self.assertEqual(report["full_surface_face_count"],n)
   self.assertEqual(report["initial_global_exact_pair_count"],0)
   self.assertEqual(report["final_global_exact_pair_count"],0)
   self.assertTrue(report["points_bitwise_unchanged"])
   self.assertTrue(report["matched_037_connectivity_synchronized"])
   self.assertTrue((td/"out"/"selected"/"candidate.npz").is_file())
 def test_search_commits_only_strict_global_reduction_and_pins_step(self):
  sys.path.insert(0,str(ROOT/"D/local-two-parent-retriangulation-402"))
  import two_parent_patch_flips_402 as patch_owner
  points=np.asarray([
   (0,0,0),(1,0,0),(0,1,0),(.25,-.2,-1),(.25,.8,1),(.25,.8,-1),
   (10,0,0),(11,0,0),(10,1,0),(10.25,-.2,-1),(10.25,.8,1),(10.25,.8,-1),
   (20,0,0),(21,0,0),(20,1,0),(30,0,0),(31,0,0),(30,1,0),
   (40,0,0),(41,0,0),(40,1,0)
  ],np.float32)
  faces=np.asarray([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]],np.int64)
  n=len(faces); bary=np.broadcast_to(np.eye(3),(n,3,3)).copy()
  c={"points_world_m":points,"source_O_faces":faces,
     "area_reference_037_points_world_m_f64":points.astype(np.float64),
     "area_reference_037_source_O_faces":faces.copy(),"face_kind":np.ones(n,np.int8),
     "parent_fine_detail_face_row":np.asarray([10,10,20,20,30],np.int32),
     "parent_fine_subface_row":np.asarray([30,30,40,40,50],np.int32),
     "parent_coarse_face_row":np.full(n,5,np.int32),"parent_source_face_marker":np.arange(70,75,dtype=np.int32),
     "refinement_face_parent_face_row":np.arange(100,105,dtype=np.int64),
     "refinement_face_parent_barycentric":bary,
     "face_source_ancestry_indptr":np.arange(n+1,dtype=np.int64),
     "face_source_ancestry_face_rows":np.arange(500,505,dtype=np.int32),
     "face_fine_subface_ancestry_indptr":np.arange(n+1,dtype=np.int64),
     "face_fine_subface_ancestry_rows":np.asarray([30,30,40,40,50],np.int32)}
  original=patch_owner.propose_patch_flips
  def proposed(candidate,*args,**kwargs):
   out={k:np.array(v,copy=True) for k,v in candidate.items()}
   out["source_O_faces"][0]=[15,16,17];out["source_O_faces"][1]=[18,19,20]
   out["area_reference_037_source_O_faces"][[0,1]]=out["source_O_faces"][[0,1]]
   return out,{"accepted_flip_count":1,"operations":[{"edge":[0,1],"angle_gain_deg":1.0,"face_rows":[0,1]}]}
  patch_owner.propose_patch_flips=proposed
  try:
   with tempfile.TemporaryDirectory() as td:
    td=Path(td);inp=td/"candidate.npz";np.savez_compressed(inp,**c)
    report=driver.run(inp,[10],[0],td/"out",max_accept=1)
    self.assertEqual(report["initial_global_exact_pair_count"],2)
    self.assertEqual(report["final_global_exact_pair_count"],1)
    self.assertEqual(report["accepted_flip_count"],1)
    self.assertTrue(report["accepted_trials_no_new_pairs"])
    self.assertTrue(report["accepted_trials_strict_pair_reduction"])
    step=td/"out"/"step-01"/"candidate.npz"
    self.assertTrue(step.is_file())
    self.assertEqual(report["accepted_steps"][0]["candidate_sha256"],hashlib.sha256(step.read_bytes()).hexdigest())
    self.assertEqual(report["accepted_steps"][0]["changed_face_rows"],[0,1])
    self.assertTrue((td/"out"/"progress.json").is_file())
    self.assertEqual(json.loads((td/"out"/"progress.json").read_text())["current_global_exact_pair_count"],1)
  finally:
   patch_owner.propose_patch_flips=original
if __name__=="__main__":unittest.main(verbosity=2)
