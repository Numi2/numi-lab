import importlib.util, hashlib, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
HELPER=ROOT/"D/local-two-parent-retriangulation-408/incremental_exact_flip_check_408.py"
SEARCH=ROOT/"D/local-two-parent-retriangulation-402/exact_local_flip_search_402.py"
PATCH=ROOT/"D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py"
PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load("incremental_exact_flip_check_408",HELPER)
search=load("exact_local_flip_search_402",SEARCH)
patch=load("two_parent_patch_flips_402",PATCH)
owner=search.exact_owner()
# Correct the known frozen predicate location explicitly for this fixture.
if not PRED.exists():
 PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
class PatchIntegrationTests(unittest.TestCase):
 def test_402_two_face_flip_is_incrementally_checked_without_moving_anchors(self):
  xy=[(0.,0.),(.4,0.),(.4,.2),(0.,.6),(.8,0.),(.8,.2)]
  p=np.asarray([(x,y,0.) for x,y in xy],np.float32)
  f=np.asarray([[1,3,0],[3,1,2],[1,4,2],[4,5,2]],np.int64)
  n=len(f);bary=np.broadcast_to(np.eye(3),(n,3,3)).copy()
  c={"points_world_m":p.copy(),"source_O_faces":f.copy(),
   "area_reference_037_points_world_m_f64":p.astype(np.float64),
   "area_reference_037_source_O_faces":f.copy(),"face_kind":np.ones(n,np.int8),
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
  anchors={k:c[k].copy() for k in ("coarse_boundary_anchor_rows","typed_coarse_anchor_output_rows_018","typed_coarse_anchor_source_ids_018")}
  out,flip=patch.propose_patch_flips(c,[10,20],[0,1,2,3],max_flips=2,max_rounds=1)
  self.assertEqual(flip["accepted_flip_count"],1)
  self.assertTrue(np.array_equal(out["points_world_m"],p))
  for k,v in anchors.items():self.assertTrue(np.array_equal(out[k],v))
  rec,_=owner.float32_surface_records(p,f,source_seam_groups=())
  full=owner._audit_self_pairs_touching_faces(rec,set(range(n)))
  pairs=np.asarray(full["triangle_pairs"],dtype=np.int64).reshape(-1,2);cs="c"*64;ph=hashlib.sha256(PRED.read_bytes()).hexdigest()
  report={"complete_surface_audit":True,"candidate_sha256":cs,"surface_predicate_sha256":ph,"faces":n,"exact_PLC_surface_pairs":len(pairs)}
  state=m.prepare(owner,p,f,pairs,audit_report=report,candidate_sha256=cs,predicate_sha256=ph,expected_predicate_sha256=ph)
  changed=np.flatnonzero(np.any(out["source_O_faces"]!=f,axis=1))
  self.assertEqual(len(changed),2)
  exact=m.evaluate(state,p,out["source_O_faces"],changed_face_rows=changed)
  self.assertTrue(exact.report["complete_surface_audit"])
  self.assertTrue(exact.report["no_new_pairs"])
  self.assertEqual(exact.report["changed_face_count"] if "changed_face_count" in exact.report else len(exact.changed),2)
if __name__=="__main__":unittest.main(verbosity=2)
