import tempfile,csv,hashlib,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"tools"))
from human_skin_fat_reference import *
class Tests(unittest.TestCase):
 def make(self,root):
  rows=[dict(row=i,ox=x,oy=y,oz=.02,ix=x,iy=y,iz=.01,dx=x,dy=y,dz=0,skin_thickness_m=.01,fat_thickness_m=.01) for i,(x,y) in enumerate([(0,0),(1,0),(0,1),(1,1)])]
  tables={"nodes.csv":rows,"faces.csv":[dict(face=0,a=0,b=1,c=2,parent_surface_face=17,stable_source_face=91),dict(face=1,a=1,b=3,c=2,parent_surface_face=42,stable_source_face=17)],"embedding.csv":[dict(row=i,source_vertex=i+100) for i in range(4)],"tetrahedra.csv":[dict(tet=0,a=0,b=1,c=2,d=4,material=1)]}
  for name,rs in tables.items():
   with (root/name).open("w",newline="") as h:
    w=csv.DictWriter(h,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
  return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in tables}
 def test_identity_domains_and_matched_rows(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);h=self.make(p);r=load_reference(p,h)
   self.assertEqual(r.by_parent_surface(ParentSurfaceIds(np.array([17]))).values.tolist(),[0])
   self.assertEqual(r.by_detailed_source_face(DetailedSourceFaceIds(np.array([17]))).values.tolist(),[1])
   with self.assertRaises(TypeError):r.by_parent_surface(DetailedSourceFaceIds(np.array([17])))
   with self.assertRaises(ValueError):r.by_parent_surface(ParentSurfaceIds(np.array([0])))
   with self.assertRaises(ValueError):r.take_faces(SurfaceFaceRows(np.array([0]),"wrong"))
   self.assertEqual(r.take_faces(r.by_parent_surface(ParentSurfaceIds(np.array([42])))).tolist(),[[1,3,2]])
   self.assertFalse(r.outer.flags.writeable)
 def test_oriented_bijection_handles_row_permutation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);r=load_reference(p,self.make(p))
   f=np.roll(r.faces[::-1],1,axis=1)
   self.assertEqual(r.oriented_bijection(f,r.detailed_source_face_ids[::-1]).tolist(),[1,0])
   with self.assertRaises(ValueError):r.oriented_bijection(f[:,::-1],r.detailed_source_face_ids[::-1])
   with self.assertRaises(ValueError):r.oriented_bijection(f,r.detailed_source_face_ids)
 def test_unsigned_wrap_and_fractional_corners_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);r=load_reference(p,self.make(p))
   overflow=np.array([np.iinfo(np.uint64).max],dtype=np.uint64)
   with self.assertRaises(ValueError):r.take_faces(SurfaceFaceRows(overflow,r.hashes["faces.csv"]))
   with self.assertRaises(ValueError):r.by_parent_surface(ParentSurfaceIds(overflow))
   invalid=r.faces.astype(float);invalid[0,1]+=.25
   with self.assertRaises(ValueError):r.oriented_bijection(invalid,r.detailed_source_face_ids)
   with self.assertRaises(ValueError):r.oriented_bijection(r.faces,overflow)
 def test_hash_and_count_mismatches(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);h=self.make(p)
   with self.assertRaises(ValueError):load_reference(p,{"nodes.csv":h["nodes.csv"]})
   (p/"faces.csv").write_text((p/"faces.csv").read_text()+"\n")
   with self.assertRaisesRegex(ValueError,"SHA-256"):load_reference(p,h)
if __name__=="__main__":unittest.main()
