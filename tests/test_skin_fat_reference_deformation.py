
from pathlib import Path
import importlib.util,sys,unittest,gzip,tempfile,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"apps"))
import simplify_human_skin_fat_boundary as m
class DeformationScreen(unittest.TestCase):
 def setUp(self):
  self.p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],np.float32)
  self.t=np.array([[0,1,2,3],[1,2,3,4]],np.int64)
  self.u=self.p.astype(float)*.001
 def gate(self,u=None):return m.ReferenceDeformationGate(self.p,self.t,self.u if u is None else u,np.array([4,4]),{"test":True})
 def test_identity(self):
  g=self.gate();rows=np.arange(2);live=np.ones(2,bool);v=g.propose(rows,self.t,live);self.assertIsNotNone(v);g.commit(rows,live,v)
  np.testing.assert_array_equal(g.base[4],g.current[4])
 def test_rejected_proposal_does_not_mutate(self):
  u=self.u.copy();u[4]=[.1,.1,.1];g=self.gate(u);n=g.norm.copy();v=g.vol.copy()
  trial=self.t.copy();trial[0]=[0,1,2,4]
  self.assertIsNone(g.propose(np.arange(2),trial,np.ones(2,bool)))
  np.testing.assert_array_equal(n,g.norm);np.testing.assert_array_equal(v,g.vol)
 def test_rest_field_exactly_zero(self):
  g=self.gate(np.zeros_like(self.p));self.assertIsNotNone(g.propose(np.arange(2),self.t,np.ones(2,bool)));self.assertEqual(g.base[4].max(),0)
 def test_invalid_reference_deformation_rejected(self):
  u=self.u.copy();u[3]=[0,0,-2]
  with self.assertRaisesRegex(ValueError,"inverts"):self.gate(u)
 def test_degenerate_proposal_rejected(self):
  g=self.gate();trial=self.t.copy();trial[0]=[0,1,2,2]
  self.assertIsNone(g.propose(np.arange(2),trial,np.ones(2,bool)))
 def test_reference_loader_hash_rejects(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.gz";p.write_bytes(b"abc")
   with self.assertRaisesRegex(ValueError,"SHA"):m.load_reference_deformation(p,"0"*64,self.p,self.t,np.full((2,3),4))

class EndpointBinding(unittest.TestCase):
 def payload(self):
  p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);t=np.array([[0,1,2,3]]);labels=np.array([[15,4,9]])
  owner="BodyParts3D whole-body skin-fat bonded shell"
  lines=["resting_FEM_endpoint_capture=begin version=2 coordinate_unit=m strain_unit=1"]
  for i,v in enumerate(p):
   fields=[f"resting_FEM_node,object={owner}",f"global={100+i}","reference_mode=1"]
   for j,a in enumerate("xyz"):fields.extend([f"rest_{a}={v[j]}",f"reference_u_hi_{a}={float(v[j])*.001}",f"reference_u_lo_{a}=0"])
   lines.append(",".join(fields))
  lines.extend([f"resting_FEM_tet,object={owner},global=0,n0=100,n1=101,n2=102,n3=103,J=1,anatomical_region_id=4,anatomical_region_source=registered_skin_fat_donor_body","resting_FEM_endpoint_capture=end","resting_integrated_body=completed steps=1"])
  return p,t,labels,"\n".join(lines)+"\n"
 def load(self,edit=lambda s:s):
  p,t,l,s=self.payload()
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"endpoint.gz"
   with gzip.open(path,"wt") as f:f.write(edit(s))
   h=hashlib.sha256(path.read_bytes()).hexdigest()
   return m.load_reference_deformation(path,h,p,t,l)
 def test_exact_geometry_donor_identity_and_low_words(self):
  u,r=self.load();self.assertEqual(u.shape,(4,3));self.assertEqual(r["exact_source_cells"],1);self.assertIn("geometric_donor_body",r["native_capture_region_semantics"])
 def test_wrong_donor_identity(self):
  with self.assertRaisesRegex(ValueError,"regions"):self.load(lambda s:s.replace("anatomical_region_id=4","anatomical_region_id=9"))
 def test_wrong_coordinates(self):
  with self.assertRaisesRegex(ValueError,"coordinates"):self.load(lambda s:s.replace("rest_x=1.0","rest_x=1.25"))
 def test_wrong_connectivity(self):
  with self.assertRaisesRegex(ValueError,"cell order"):self.load(lambda s:s.replace("n3=103","n3=102"))
 def test_incomplete_capture(self):
  with self.assertRaisesRegex(ValueError,"incomplete"):self.load(lambda s:s.replace("resting_FEM_endpoint_capture=end","interrupted"))
 def test_nonfinite_displacement(self):
  with self.assertRaisesRegex(ValueError,"nonfinite"):self.load(lambda s:s.replace("reference_u_hi_x=0.0","reference_u_hi_x=nan",1))

if __name__=="__main__":unittest.main()
