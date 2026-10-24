import importlib.util
from pathlib import Path
import struct
import unittest
import numpy as np
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rb255',root/'tools/rebind_human_skin_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('fixture255',Path(__file__).parent/'test_human_skin_reference_rebind.py');fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)
class RefinementComposition(unittest.TestCase):
 def refine(self,base,a,b,edit=None):
  n=base['nv'];vertices=np.vstack((base['verts'],base['verts'][a].copy()));vertices[n,:3]=(base['verts'][a,:3]+base['verts'][b,:3])/2
  if edit is not None:vertices[edit,2]+=.001
  weights=np.vstack((base['weights'],(base['weights'][a]+base['weights'][b])/2))
  faces=[];parents=[]
  for parent,tri in enumerate(base['inds']):
   for k in range(3):
    if {int(tri[k]),int(tri[(k+1)%3])}=={a,b}:
     u,v,w=int(tri[k]),int(tri[(k+1)%3]),int(tri[(k+2)%3]);faces.extend(((u,n,w),(n,v,w)));parents.extend((parent,parent));break
   else:faces.append(tri);parents.append(parent)
  header=bytearray(base['raw'][:60]);struct.pack_into('<2I',header,16,n+1,3*len(faces))
  raw=bytes(header)+base['binds'].tobytes()+vertices.astype('<f4').tobytes()+np.asarray(faces,dtype='<u4').tobytes()+weights.astype('<f4').tobytes()
  return m.decode(raw),np.array(parents),np.array([[a,b]]),np.array([[.5,.5]])
 def test_disjoint_refinements_and_original_row_edits(self):
  base=fx.fixture();a=self.refine(base,0,1,0);b=self.refine(base,1,3,3)
  raw,report=m.merge_edge_refinement_repairs(base,[a,b]);out=m.decode(raw)
  self.assertEqual(out['nv'],7);self.assertEqual(len(out['inds']),4)
  self.assertEqual(report['face_parent_source_rows'],[0,0,1,1]);self.assertEqual(report['changed_original_rows'],[0,3])
  np.testing.assert_array_equal(out['verts'][[1,2,4]],base['verts'][[1,2,4]])
  np.testing.assert_array_equal(out['weights'][:5],base['weights']);self.assertFalse(report['physical_admission'])
 def test_empty_and_shared_original_edge_are_valid(self):
  base=fx.fixture();self.assertEqual(m.merge_edge_refinement_repairs(base,[])[0],base['raw'])
  out=m.decode(m.merge_edge_refinement_repairs(base,[self.refine(base,1,2)])[0]);self.assertEqual(len(out['inds']),4)
 def test_overlap_and_conflicting_old_edits_fail(self):
  base=fx.fixture()
  with self.assertRaisesRegex(m.RebindError,'overlapping'):
   m.merge_edge_refinement_repairs(base,[self.refine(base,0,1),self.refine(base,0,2)])
  with self.assertRaisesRegex(m.RebindError,'conflicting'):
   a=self.refine(base,0,1,0);b=list(self.refine(base,1,3,0));raw=bytearray(b[0]['raw']);struct.pack_into('<f',raw,b[0]['vertoff']+8,.002);b[0]=m.decode(bytes(raw));m.merge_edge_refinement_repairs(base,[a,tuple(b)])
 def test_bad_parent_ancestry_and_winding_fail(self):
  base=fx.fixture();repair=self.refine(base,0,1)
  bad=list(repair);bad[1]=np.array([0,1,1])
  with self.assertRaises(m.RebindError):m.merge_edge_refinement_repairs(base,[tuple(bad)])
  bad=list(repair);raw=bytearray(bad[0]['raw']);face=bad[0]['inds'][0][::-1];raw[bad[0]['indexoff']:bad[0]['indexoff']+12]=face.astype('<u4').tobytes();bad[0]=m.decode(bytes(raw))
  with self.assertRaises(m.RebindError):m.merge_edge_refinement_repairs(base,[tuple(bad)])
 def test_bad_edge_weights_or_tjunction_fail(self):
  base=fx.fixture();repair=self.refine(base,1,2);bad=list(repair);bad[2]=np.array([[0,4]])
  with self.assertRaises(m.RebindError):m.merge_edge_refinement_repairs(base,[tuple(bad)])
  bad=list(repair);bad[3]=np.array([[.4,.5]])
  with self.assertRaises(m.RebindError):m.merge_edge_refinement_repairs(base,[tuple(bad)])
  repair=list(self.refine(base,0,1));repair[2]=np.array([[1,2]])
  with self.assertRaisesRegex(m.RebindError,'T junction'):m.merge_edge_refinement_repairs(base,[tuple(repair)])
 def test_changed_binding_source_and_missing_parent_fail(self):
  base=fx.fixture();repair=self.refine(base,0,1)
  for offset,value in ((28,1),(base['vertoff']+40,0)):
   bad=list(repair);raw=bytearray(bad[0]['raw']);struct.pack_into('<I',raw,offset,value);bad[0]=m.decode(bytes(raw))
   with self.assertRaises(m.RebindError):m.merge_edge_refinement_repairs(base,[tuple(bad)])
if __name__=='__main__':unittest.main()
