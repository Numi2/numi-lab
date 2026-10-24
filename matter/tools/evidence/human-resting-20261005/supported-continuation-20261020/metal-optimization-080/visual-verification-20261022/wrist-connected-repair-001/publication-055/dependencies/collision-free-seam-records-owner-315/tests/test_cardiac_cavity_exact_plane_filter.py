import itertools
import random
import unittest
from fractions import Fraction
from numilab_human import cardiac_cavity_intersections as m

def reference(pairs,same):
 result=[];allowed=0
 for (tri,lo,hi,i,ids),(other,lower,upper,j,other_ids) in pairs:
  points=m.triangle_intersection_points(tri,other)
  if not points:continue
  shared=set(ids)&set(other_ids) if same else set()
  common={tri[ids.index(k)] for k in shared}
  if same and len(shared) in (1,2) and all(m._allowed_shared_point(p,common) for p in points):allowed+=1
  else:result.append([i,j])
 return {'triangle_pairs':sorted(result),'count':len(result),'aabb_candidate_pairs':len(pairs),'allowed_shared_vertex_or_edge_pairs':allowed}

def row(tri,ids,i):
 return (tuple(map(tuple,tri)),tuple(min(p[k] for p in tri) for k in range(3)),tuple(max(p[k] for p in tri) for k in range(3)),i,tuple(ids))
class ExactPlaneFilterTests(unittest.TestCase):
 def parity(self,a,b,ids=(0,1,2),other_ids=(0,1,3)):
  pairs=[(row(a,ids,0),row(b,other_ids,1))]
  for same in (False,True):self.assertEqual(m._audit_record_pairs(pairs,same_surface=same),reference(pairs,same))
 def test_shared_edge_planes_and_coplanar_overlap(self):
  a=((0,0,0),(4,0,0),(0,4,0))
  for third in ((0,1,2),(0,-4,0),(1,1,0),(0,4,0),(8,8,0)):
   self.parity(a,(a[0],a[1],third))
 def test_shared_vertex_separation_and_nonlocal_crossings(self):
  a=((0,0,0),(4,0,0),(0,4,0))
  for b in (((0,0,0),(1,1,1),(2,1,2)),((0,0,0),(1,1,-1),(1,1,1)),((0,0,0),(-2,0,0),(0,-2,0))):
   self.parity(a,b,other_ids=(0,3,4))
 def test_duplicates_disjoint_ids_and_inconsistent_shared_positions(self):
  a=((0,0,0),(4,0,0),(0,4,0))
  self.parity(a,a,other_ids=(0,1,2));self.parity(a,a,other_ids=(3,4,5))
  self.parity(a,((1,0,0),(4,0,0),(1,2,1)))
 def test_exact_nearly_coplanar_huge_translation_and_rational(self):
  for scale,offset in ((1,0),(1<<149,1<<180),(Fraction(1,1<<512),Fraction(3,7))):
   a=((0,0,0),(4,0,0),(0,4,0));b=((0,0,0),(4,0,0),(1,1,Fraction(1,1<<96)))
   tr=lambda x:tuple(tuple(offset+scale*v for v in p) for p in x)
   self.parity(tr(a),tr(b));self.parity(tr(a),tr(b),other_ids=(3,4,5))
 def test_random_indexed_records_match_full_predicate(self):
  rng=random.Random(264)
  for _ in range(3):
   vertices=[tuple(rng.randint(-5,5) for _ in range(3)) for _ in range(18)]
   records=[]
   while len(records)<22:
    ids=tuple(rng.sample(range(len(vertices)),3));tri=tuple(vertices[k] for k in ids)
    if not any(m._cross(m._sub(tri[1],tri[0]),m._sub(tri[2],tri[0]))):continue
    records.append(row(tri,ids,len(records)))
   pairs=list(itertools.combinations(records,2))
   for same in (False,True):self.assertEqual(m._audit_record_pairs(pairs,same_surface=same),reference(pairs,same))
 def test_float_inputs_retain_exact_original_path(self):
  pairs=[(row(((0.,0.,0.),(.1,0.,0.),(0.,.1,0.)),(0,1,2),0),row(((0.,0.,0.),(.1,0.,0.),(0.,0.,.1)),(0,1,3),1))]
  with self.assertRaises(m.HumanImportError):reference(pairs,True)
  with self.assertRaises(m.HumanImportError):m._audit_record_pairs(pairs,same_surface=True)
 def test_degenerate_remains_rejected(self):
  with self.assertRaises(Exception):self.parity(((0,0,0),(1,0,0),(2,0,0)),((0,0,0),(1,0,0),(0,1,0)))
if __name__=='__main__':unittest.main()
