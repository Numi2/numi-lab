import ast
from pathlib import Path
import unittest
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170')
SOURCE=H/'run_local_witness_backtracking_287.py'

def load_preflight_functions():
    tree=ast.parse(SOURCE.read_text(),filename=str(SOURCE))
    names={'require_pinned_sha','validate_area_reference','assert_fixed_prefix_equal'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    ns={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns

class Preflight285Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn=load_preflight_functions()
        cls.path=H/'connected-refined-278/candidate.npz'
        cls.z=np.load(cls.path,allow_pickle=False)
        cls.points=cls.z['points_world_m']
        cls.faces=cls.z['source_O_faces']
        cls.ref=cls.z['area_reference_037_points_world_m_f64']
        cls.reffaces=cls.z['area_reference_037_source_O_faces']

    def test_actual_278_matched_reference_and_fixed_prefix(self):
        current_cross,den,areas=self.fn['validate_area_reference'](
            self.points.astype(float),self.faces,self.ref,self.reffaces)
        self.assertEqual(current_cross.shape,(41733,3))
        self.assertTrue(np.isfinite(den).all())
        self.assertTrue((areas>0).all())
        self.fn['assert_fixed_prefix_equal'](self.points,self.points,18430)

    def test_wrong_reference_hash_rejected(self):
        self.fn['require_pinned_sha']('a'*64,'a'*64,'matched area')
        with self.assertRaises(AssertionError):
            self.fn['require_pinned_sha']('a'*64,'b'*64,'matched area')

    def test_mismatched_reference_topology_rejected(self):
        bad=self.reffaces.copy()
        bad[0,0]=(bad[0,0]+1)%len(self.points)
        with self.assertRaises(AssertionError):
            self.fn['validate_area_reference'](self.points.astype(float),self.faces,self.ref,bad)

    def test_changed_fixed_prefix_rejected(self):
        trial=self.points.copy()
        trial[7,0]=np.nextafter(trial[7,0],np.float32(np.inf),dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.fn['assert_fixed_prefix_equal'](trial,self.points,18430)

    def test_degenerate_reference_face_rejected(self):
        bad=self.ref.copy()
        _,b,c=map(int,self.faces[0])
        bad[c]=bad[b]
        with self.assertRaises(AssertionError):
            self.fn['validate_area_reference'](self.points.astype(float),self.faces,bad,self.reffaces)

    def test_degenerate_current_face_rejected(self):
        bad=self.points.astype(float)
        bad[self.faces[0,2]]=bad[self.faces[0,1]]
        with self.assertRaises(AssertionError):
            self.fn['validate_area_reference'](bad,self.faces,self.ref,self.reffaces)

if __name__=='__main__':
    unittest.main()
