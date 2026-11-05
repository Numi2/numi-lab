import ast,unittest
from pathlib import Path
import numpy as np
from numi_human_tet_face_incidence import CompactTetFaceMap
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
P=H/'current-role-p1-support-938-1025-active-source-map/current_fine_217_owner_938_v3_active_source_rows.py'
tree=ast.parse(P.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tetrahedron_boundary_faces')
namespace={'np':np,'Any':object};exec(compile(ast.Module(body=[f],type_ignores=[]),str(P),'exec'),namespace)
original=namespace[f.name]
class TestCompactFaces(unittest.TestCase):
 def compare(self,p,t,m=None):
  mesh=dict(points=np.asarray(p,np.float32),tetrahedra=np.asarray(t,np.int64).reshape(-1,4),element_ids=np.arange(len(t))*7+3,material_indices=np.asarray(m if m is not None else [15]*len(t)))
  a=original(mesh);b=CompactTetFaceMap(mesh)
  self.assertEqual(len(a),len(b));self.assertEqual(set(a),set(b))
  self.assertEqual(a,{k:list(v) for k,v in b.items()})
  for key,values in a.items():
   self.assertEqual(values,list(b[key]));self.assertEqual(values,list(b.get(key)))
  self.assertEqual(b.get((900,901,902),[]),[])
 def test_empty(self):self.compare(np.empty((0,3)),[])
 def test_single(self):self.compare([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2,3]])
 def test_material_interface(self):self.compare([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],[[0,1,2,3],[0,2,1,4]],[15,16])
 def test_random_and_reversed(self):
  rng=np.random.default_rng(320)
  for i in range(32):
   p=rng.normal(size=(40,3));t=np.asarray([rng.choice(40,4,replace=False) for _ in range(8)])
   self.compare(p,t)
 def test_degenerate(self):self.compare([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[[0,1,2,3]])
 def test_nonmanifold_witness(self):
  mesh=dict(points=np.eye(7,3,dtype=np.float32),tetrahedra=np.array([[0,1,2,3],[0,2,1,4],[0,1,2,5],[0,1,2,6]]),element_ids=np.arange(4),material_indices=np.full(4,15))
  with self.assertRaises(ValueError) as a:original(mesh)
  with self.assertRaises(ValueError) as b:CompactTetFaceMap(mesh)
  self.assertEqual(str(a.exception),str(b.exception))
if __name__=='__main__':unittest.main()
