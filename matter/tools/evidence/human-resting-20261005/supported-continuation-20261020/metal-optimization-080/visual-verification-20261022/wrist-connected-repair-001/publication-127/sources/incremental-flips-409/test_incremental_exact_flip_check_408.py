import importlib.util,hashlib,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
M=ROOT/"D/local-two-parent-retriangulation-408/incremental_exact_flip_check_408.py"
S=ROOT/"D/local-two-parent-retriangulation-402/exact_local_flip_search_402.py"
P=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
m=load("incremental_exact_flip_check_408",M);o=load("exact_local_flip_search_402",S).exact_owner();ph=hashlib.sha256(P.read_bytes()).hexdigest()
def fixture(cross=True):
 p=np.asarray([(0,0,0),(1,0,0),(0,1,0),(.25,-.2,-1),(.25,.8,1),(.25,.8,-1),(10,0,0),(11,0,0),(10,1,0),(10.25,-.2,-1),(10.25,.8,1),(10.25,.8,-1),(20,0,0),(21,0,0),(20,1,0)],np.float32)
 f=np.asarray([[0,1,2],[3,4,5],[6,7,8],[9,10,11]],np.int64)
 if not cross:f[0]=[12,13,14]
 return p,f
def state(p,f):
 r,_=o.float32_surface_records(p,f,source_seam_groups=());q=o._audit_self_pairs_touching_faces(r,set(range(len(f))))
 pairs=np.asarray(q["triangle_pairs"],np.int64).reshape(-1,2);sha="a"*64
 report={"complete_surface_audit":True,"candidate_sha256":sha,"surface_predicate_sha256":ph,"faces":len(f),"exact_PLC_surface_pairs":len(pairs)}
 return m.prepare(o,p,f,pairs,audit_report=report,candidate_sha256=sha,predicate_sha256=ph,expected_predicate_sha256=ph),pairs
def fixture_sequence():
 p=np.asarray([
  (0,0,0),(1,0,0),(0,1,0),(.25,-.2,-1),(.25,.8,1),(.25,.8,-1),
  (10,0,0),(11,0,0),(10,1,0),(10.25,-.2,-1),(10.25,.8,1),(10.25,.8,-1),
  (20,0,0),(21,0,0),(20,1,0),(30,0,0),(31,0,0),(30,1,0),
  (40,0,0),(41,0,0),(40,1,0),(20.25,-.2,-1),(20.25,.8,1),(20.25,.8,-1)
 ],np.float32)
 f=np.asarray([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]],np.int64)
 return p,f
class T(unittest.TestCase):
 def test_empty_pair_schema_is_narrow(self):
  self.assertEqual(m._pairs([],4),frozenset())
  self.assertEqual(m._pairs(np.empty((0,2),dtype=np.int64),4),frozenset())
  with self.assertRaisesRegex(ValueError,"integer Nx2"):
   m._pairs(np.empty((0,3),dtype=np.int64),4)
 def test_reuse_and_remove(self):
  p,f=fixture();s,ps=state(p,f);self.assertEqual({tuple(x) for x in ps},{(0,1),(2,3)})
  t=f.copy();t[0]=[12,13,14];r=m.evaluate(s,p,t,changed_face_rows=[0]);self.assertEqual(r.pairs,{(2,3)})
  self.assertEqual(r.report["unchanged_baseline_pairs_reused"],1);self.assertEqual(r.report["removed_pairs"],[[0,1]])
  n=m.commit(s,r);self.assertTrue(np.array_equal(n.faces,t))
 def test_new_pair_is_found(self):
  p,f=fixture(False);s,_=state(p,f);t=f.copy();t[0]=[0,1,2];r=m.evaluate(s,p,t)
  self.assertEqual(r.report["introduced_pairs"],[[0,1]])
  with self.assertRaisesRegex(ValueError,"introduces"):m.commit(s,r)
 def test_changed_rows_and_fixed_points(self):
  p,f=fixture();s,_=state(p,f);t=f.copy();t[0]=[12,13,14]
  with self.assertRaisesRegex(ValueError,"changed rows"):m.evaluate(s,p,t,changed_face_rows=[1])
  x=p.copy();x[0,0]=np.nextafter(x[0,0],np.float32(1))
  with self.assertRaisesRegex(ValueError,"bitwise-identical"):m.evaluate(s,x,t)
 def test_receipt_fail_closed(self):
  p,f=fixture();r,_=o.float32_surface_records(p,f,source_seam_groups=());q=o._audit_self_pairs_touching_faces(r,set(range(len(f))))
  ps=np.asarray(q["triangle_pairs"],np.int64).reshape(-1,2);sha="a"*64
  report={"complete_surface_audit":False,"candidate_sha256":sha,"surface_predicate_sha256":ph,"faces":len(f),"exact_PLC_surface_pairs":len(ps)}
  with self.assertRaisesRegex(ValueError,"incomplete"):m.prepare(o,p,f,ps,audit_report=report,candidate_sha256=sha,predicate_sha256=ph,expected_predicate_sha256=ph)
  report["complete_surface_audit"]=True
  with self.assertRaisesRegex(ValueError,"candidate pin"):m.prepare(o,p,f,ps,audit_report=report,candidate_sha256="b"*64,predicate_sha256=ph,expected_predicate_sha256=ph)
 def test_trial_is_bound_to_baseline(self):
  p,f=fixture();s,_=state(p,f);s2,_=state(p,f);t=f.copy();t[0]=[12,13,14]
  with self.assertRaisesRegex(ValueError,"another baseline"):m.commit(s2,m.evaluate(s,p,t))
 def test_full_baseline_prepared_once_and_sequential_trials_match_full_predicate(self):
  p,f=fixture_sequence()
  state,receipt,base_audit=m.prepare_full(o,p,f,candidate_sha256="d"*64,predicate_sha256=ph,expected_predicate_sha256=ph)
  self.assertEqual(receipt["complete_surface_face_count"],len(f))
  self.assertEqual(state.pairs,{(0,1),(2,3)})
  t1=f.copy();t1[0]=[15,16,17]
  trial1=m.evaluate(state,p,t1,changed_face_rows=[0])
  full1,_=o.float32_surface_records(p,t1,source_seam_groups=())
  exact1=o._audit_self_pairs_touching_faces(full1,set(range(len(f))))
  self.assertEqual(trial1.pairs,{tuple(x) for x in exact1["triangle_pairs"]})
  self.assertEqual(trial1.report["introduced_pairs"],[])
  state1=m.commit(state,trial1)
  t2=t1.copy();t2[2]=[18,19,20]
  trial2=m.evaluate(state1,p,t2,changed_face_rows=[2])
  full2,_=o.float32_surface_records(p,t2,source_seam_groups=())
  exact2=o._audit_self_pairs_touching_faces(full2,set(range(len(f))))
  self.assertEqual(trial2.pairs,{tuple(x) for x in exact2["triangle_pairs"]})
  self.assertEqual(trial2.pairs,frozenset())
  state2=m.commit(state1,trial2)
  self.assertEqual(state2.pairs,frozenset())
 def test_changed_face_is_checked_against_disjoint_nearby_third_face(self):
  p,f=fixture_sequence();state,_,_=m.prepare_full(o,p,f,candidate_sha256="e"*64,predicate_sha256=ph,expected_predicate_sha256=ph)
  t=f.copy();t[0]=[21,22,23]
  trial=m.evaluate(state,p,t,changed_face_rows=[0])
  full,_=o.float32_surface_records(p,t,source_seam_groups=())
  exact=o._audit_self_pairs_touching_faces(full,set(range(len(f))))
  expected={tuple(map(int,x)) for x in exact["triangle_pairs"]}
  self.assertEqual(trial.pairs,expected)
  self.assertIn((0,4),trial.pairs)
  self.assertIn([0,4],trial.report["introduced_pairs"])
 def test_empty_full_baseline(self):
  p=np.asarray([(0,0,0),(1,0,0),(0,1,0)],np.float32)
  f=np.asarray([[0,1,2]],np.int64)
  state,receipt,audit=m.prepare_full(o,p,f,candidate_sha256="f"*64,predicate_sha256=ph,expected_predicate_sha256=ph)
  self.assertEqual(state.pairs,frozenset())
  self.assertEqual(receipt["exact_PLC_surface_pairs"],0)
 def test_shared_vertex_contact_is_reconciled_with_full_predicate(self):
  p=np.asarray([(0,0,0),(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)],np.float32)
  f=np.asarray([[0,1,2],[0,3,4]],np.int64)
  state,receipt,audit=m.prepare_full(o,p,f,candidate_sha256="1"*64,predicate_sha256=ph,expected_predicate_sha256=ph)
  self.assertEqual(state.pairs,{tuple(map(int,x)) for x in audit["triangle_pairs"]})
  t=f.copy();t[1]=[0,4,3]
  trial=m.evaluate(state,p,t,changed_face_rows=[1])
  full,_=o.float32_surface_records(p,t,source_seam_groups=())
  exact=o._audit_self_pairs_touching_faces(full,set(range(len(f))))
  self.assertEqual(trial.pairs,{tuple(map(int,x)) for x in exact["triangle_pairs"]})
if __name__=="__main__":unittest.main(verbosity=2)
