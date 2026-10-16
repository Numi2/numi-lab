import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cardboard_bilinear_capture_layers as layer

class CaptureLayerContracts(unittest.TestCase):
    def test_nonfinite_report_fields_retain_explicit_locations_and_do_not_change_gates(self):
        import json
        raw={'nested':[1.25,float('nan'),{'a/b':float('inf'),'~key':-float('inf')}],
             'all_rows_at_relative_gate':False,'failed_rows':3,'optional':None}
        clean,missing=layer.finite_report(raw)
        self.assertEqual(clean['nested'],[1.25,None,{'a/b':None,'~key':None}])
        self.assertEqual([r['path'] for r in missing],['$/nested/1','$/nested/2/a~1b','$/nested/2/~0key'])
        self.assertEqual([r['value'] for r in missing],['nan','inf','-inf'])
        self.assertFalse(clean['all_rows_at_relative_gate'])
        self.assertEqual(clean['failed_rows'],3)
        self.assertIsNone(clean['optional'])
        json.dumps(clean,allow_nan=False)

    def test_zero_reference_stays_inconclusive(self):
        result,*_=layer.comparison(np.zeros((1,9)),np.zeros((1,9)),1e-3)
        self.assertIsNone(result['max_relative_l2'])
        self.assertFalse(result['all_rows_at_relative_gate'])
        self.assertEqual(result['zero_reference_rows'],1)

    def test_invalid_reference_row_does_not_disappear_from_gate(self):
        result,*_=layer.comparison(np.ones((2,3)),np.array([[1,1,1],[np.nan,1,1]]),1e-3)
        self.assertEqual(result['finite_rows'],1)
        self.assertFalse(result['all_rows_at_relative_gate'])

    def test_empty_branch_side_is_inconclusive(self):
        result,*_=layer.comparison(np.empty((0,3,3)),np.empty((0,3,3)))
        self.assertEqual(result['rows'],0)
        self.assertIsNone(result['max_relative_l2'])

    def test_uniform_stress_gather_cancels_shared_face(self):
        # Two equal positive-volume tetrahedra on opposite sides of z=0.
        # Their common-face traction cancels at all three shared nodes.
        positions=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],float)
        ids=np.array([[0,1,2,3],[0,2,1,4]])
        inv=np.stack([np.linalg.inv((positions[t[1:]]-positions[t[0]]).T) for t in ids])
        stress=np.array([np.diag([0,0,7.])]*2)
        force=layer.forces(stress,inv,np.full(2,1/6))
        np.testing.assert_allclose(force.sum(axis=1),0,atol=1e-15)
        gathered=layer.gather(force,ids,5)
        np.testing.assert_allclose(gathered[:3],0,atol=1e-15)
        np.testing.assert_allclose(gathered[3:],[[0,0,-7/6],[0,0,7/6]],atol=1e-15)

    def test_identity_rejects_other_environment_hole_and_bad_fingerprint(self):
        h={'environment_count':2,'node_count_per_environment':1,'tetrahedron_count_per_environment':1,'root':20,'newton_iteration':63,'world_fingerprint_hex':'a','execution_fingerprint_hex':'b','cached_algorithmic_tangent_capture':{'matrix_layout':'row-major 9x9, local material frame','derivative_contract':'dP_local/dF_local'}}
        b={'prediction_selector':'20:63','operator_fd_iteration':{'world_fingerprint':10,'execution_fingerprint':11}}
        n=[{'environment':e,'node_id':0} for e in (0,1)];t=[{'environment':e,'tet_id':0} for e in (0,1)]
        self.assertEqual(layer.validate_identity(h,n,t,b),(20,63))
        with self.assertRaises(ValueError):layer.validate_identity(h,n,t[:1],b)
        wrong=copy.deepcopy(h);wrong['execution_fingerprint_hex']='c'
        with self.assertRaises(ValueError):layer.validate_identity(wrong,n,t,b)
        with self.assertRaises(ValueError):layer.validate_identity(h,n,[t[0],t[0]],b)

    def test_epsilon_work_bound_precedes_io(self):
        for powers in ([],[1],[2,2],[-1,2],[1,41],list(range(17))):
            with self.assertRaisesRegex(ValueError,'epsilon grid'):
                layer.audit(Path('/nonexistent-input'),Path('/nonexistent-output'),epsilon_powers=powers)

if __name__=='__main__':unittest.main()
