import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import prospective_self_guards_409_next as g

class LinearKeyAABB:
 def float32_point_lattice_key(self,p): return tuple(float(np.float32(x)) for x in p)
 def _query_surface_aabb(self,queries,index):
  for q in queries:
   for r in index.records:
    if all(q[2][k]>=r[1][k] and r[2][k]>=q[1][k] for k in range(3)):
     yield q,r

def state_fixture():
 p=np.asarray([
  [0,0,0],[1,0,0],[0,1,0],             # face 0
  [0,0,.0004],[1,0,.0004],[0,1,.0004], # face 1, within radius
  [.5,-1,0],                             # face 2 shares edge 0-1
  [-1,0,0],[0,0,-1],                     # face 3 shares vertex 0
  [10,10,10],[11,10,10],[10,11,10],      # face 4 distant
 ],dtype=np.float32)
 f=np.asarray([[0,1,2],[3,4,5],[1,0,6],[0,7,8],[9,10,11]],np.int32)
 own=LinearKeyAABB(); records=[]
 for i,triids in enumerate(f):
  tri=tuple(own.float32_point_lattice_key(p[j]) for j in triids)
  lo=tuple(min(v[k] for v in tri) for k in range(3)); hi=tuple(max(v[k] for v in tri) for k in range(3))
  records.append((tri,lo,hi,i,tuple(map(int,triids))))
 idx=SimpleNamespace(records=tuple(records))
 return SimpleNamespace(owner=own,points=p,faces=f,records=tuple(records),index=idx,candidate_sha256="a"*64,predicate_sha256="b"*64)

class GuardRowsTests(unittest.TestCase):
 def test_expanded_aabb_near_pairs_union_seed_and_adjacency_policy(self):
  st=state_fixture()
  out=g.enumerate_prospective_pairs(st,.0005,[[0,4]])
  pairs={tuple(x) for x in out["pair_ids"].tolist()}
  near={tuple(x) for x in out["prospective_pair_ids"].tolist()}
  self.assertIn((0,1),near)     # .4 mm gap in z
  self.assertNotIn((0,2),near)  # shared edge is not prospective
  self.assertIn((0,3),near)     # shared vertex remains a possible fold
  self.assertIn((0,4),pairs)    # original guard is never dropped
  self.assertIn((0,4),{tuple(x) for x in out["seed_only_pair_ids"].tolist()})
  self.assertEqual(out["report"]["union_pair_count"],len(pairs))
  self.assertFalse(out["report"]["exact_collision_test_performed"])
 def test_pair_generation_is_deterministic_and_typed_empty(self):
  st=state_fixture()
  a=g.enumerate_prospective_pairs(st,.0005,[])
  b=g.enumerate_prospective_pairs(st,.0005,np.empty((0,2),np.int64))
  np.testing.assert_array_equal(a["pair_ids"],b["pair_ids"])
  self.assertEqual(a["pair_ids"].shape[1],2)
  far=state_fixture()
  far.points[:]=np.float32(0)
  # no use of mutated state: empty-mesh fixture verifies canonical empty output
  own=LinearKeyAABB(); empty=SimpleNamespace(owner=own,points=np.empty((0,3),np.float32),
      faces=np.empty((0,3),np.int32),records=(),index=SimpleNamespace(records=()),
      candidate_sha256="c"*64,predicate_sha256="d"*64)
  z=g.enumerate_prospective_pairs(empty,.0005,[])
  self.assertEqual(z["pair_ids"].shape,(0,2))
  self.assertEqual(z["report"]["union_pair_count"],0)
 def test_pair_inputs_and_radius_fail_closed(self):
  st=state_fixture()
  for r in (-1,float("nan"),float("inf")):
   with self.assertRaises(ValueError): g.enumerate_prospective_pairs(st,r,[])
  for bad in ([[0,0]],[[0,99]],[[1.0,2.0]],np.empty((0,3),np.int32)):
   with self.assertRaises(ValueError): g.enumerate_prospective_pairs(st,.0005,bad)
 def test_guard_rows_keep_incompatible_pair_explicit(self):
  class Owner:
   def build_pair_rows(self,p,f,pairs,*,margin_fraction):
    i,j=map(int,pairs[0])
    if (i,j)==(1,2): raise ValueError("nonseparable fixture")
    return ([{"nodes":np.asarray([0,1]),"coefficients":np.asarray([[1.,0,0],[-1.,0,0]]),"lower":-.1}],
            [{"faces":[i,j],"kind":"fixture"}])
  p=np.zeros((6,3),np.float64); f=np.asarray([[0,1,2],[3,4,5],[0,4,5]],np.int32)
  rows,recs,rep=g.build_guard_rows(p,f,[[0,1],[1,2]],Owner())
  self.assertEqual(len(rows),1); self.assertEqual(len(recs),1)
  self.assertEqual(rep["compatible_pair_ids"],[[0,1]])
  self.assertEqual(rep["incompatible_pairs"][0]["pair"],[1,2])
  self.assertEqual(rep["input_pair_count"],2); self.assertTrue(rep["all_pair_ids_preserved"])
 def test_guard_rows_reject_positive_baseline_rhs_and_retain_reason(self):
  class Owner:
   def build_pair_rows(self,p,f,pairs,*,margin_fraction):
    i,j=map(int,pairs[0])
    return ([{"nodes":np.asarray([0]),"coefficients":np.asarray([[1.,0,0]]),"lower":1e-3}],
            [{"faces":[i,j]}])
  p=np.zeros((6,3),np.float64); f=np.asarray([[0,1,2],[3,4,5]],np.int32)
  rows,recs,rep=g.build_guard_rows(p,f,[[0,1]],Owner())
  self.assertEqual(rows,[]); self.assertEqual(rep["incompatible_pair_count"],1)
  self.assertIn("zero-feasible",rep["incompatible_pairs"][0]["reason"])

if __name__=="__main__": unittest.main()
