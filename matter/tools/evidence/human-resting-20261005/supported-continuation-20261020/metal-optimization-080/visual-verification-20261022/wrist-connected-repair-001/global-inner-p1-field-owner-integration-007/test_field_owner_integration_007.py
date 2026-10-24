import unittest
import importlib.util
import sys
from pathlib import Path
import numpy as np
from field_owner_integration_007 import (
    map_output_tets_to_base_tets,
    initialize_field_on_mesh,
    fixed_zero_capacity_report,
    compose_parent_csrs,
)


class FieldOwnerIntegration007Tests(unittest.TestCase):
    def setUp(self):
        self.Xb=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        self.Tb=np.array([[0,1,2,3]],dtype=np.int64)
        self.phi=np.array([0.0,1.0,0.5,0.25],dtype=np.float64)
        self.query=np.array([0.25,0.25,0.25],dtype=np.float32)
        self.X=np.vstack([self.Xb,self.query]).astype(np.float32)
        self.T=np.array([[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]],dtype=np.int64)

    def test_newpoint_initializer_is_containing_parent_p1_value(self):
        off,rows=map_output_tets_to_base_tets(
            np.array([0,1,2,3,4]),np.array([0,0,0,0]),np.array([0]),len(self.T),len(self.Tb))
        val,chosen,bary=initialize_field_on_mesh(
            self.X,self.T,self.Xb,self.Tb,self.phi,np.array([0]),off,rows)
        self.assertAlmostEqual(val[4],0.4375,places=14)
        self.assertEqual(chosen[4],0)
        self.assertTrue(np.allclose(bary[4],[0.25,0.25,0.25,0.25],atol=1e-14))
        self.assertEqual(val[0],0.0)

    def test_mesh_domain_mismatch_rejected(self):
        off,rows=map_output_tets_to_base_tets(
            np.array([0,1,2,3,4]),np.array([0,0,0,0]),np.array([0]),len(self.T),len(self.Tb))
        bad=np.array(self.X,copy=True);bad[1,0]=np.nextafter(bad[1,0],np.float32(2.0))
        with self.assertRaisesRegex(ValueError,'prefix exactly'):
            initialize_field_on_mesh(bad,self.T,self.Xb,self.Tb,self.phi,[0],off,rows)
        with self.assertRaisesRegex(ValueError,'base initializer'):
            initialize_field_on_mesh(self.X,self.T,self.Xb,self.Tb,self.phi[:-1],[0],off,rows)

    def test_tet_domain_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'does not match output mesh domain'):
            map_output_tets_to_base_tets(np.array([0,1]),np.array([0]),np.array([0]),2,1)
        with self.assertRaisesRegex(ValueError,'intermediate-to-base'):
            map_output_tets_to_base_tets(np.array([0,1]),np.array([0]),np.array([1]),1,1)
        off,rows=map_output_tets_to_base_tets(
            np.array([0,1,2,3,4]),np.array([0,0,0,0]),np.array([0]),len(self.T),len(self.Tb))
        with self.assertRaisesRegex(ValueError,'base-parent tet CSR'):
            initialize_field_on_mesh(self.X,self.T[:-1],self.Xb,self.Tb,self.phi,[0],off,rows)

    def test_multilevel_parent_ancestry_composes_and_rejects_domain_mismatch(self):
        off,rows=compose_parent_csrs(
            np.array([0,2,3]),np.array([0,1,1]),
            np.array([0,2,3]),np.array([2,0,1]),
            child_count=2,intermediate_count=2,base_count=3)
        self.assertEqual(off.tolist(),[0,3,4])
        self.assertEqual(rows.tolist(),[0,1,2,1])
        with self.assertRaisesRegex(ValueError,'intermediate tet domain'):
            compose_parent_csrs(np.array([0,1]),np.array([2]),np.array([0,1,1]),np.array([0]),child_count=1,intermediate_count=2,base_count=1)
        with self.assertRaisesRegex(ValueError,'parent ancestry CSR'):
            compose_parent_csrs(np.array([0,1]),np.array([0]),np.array([0,2]),np.array([0]),child_count=1,intermediate_count=1,base_count=1)

    def test_402_solution_is_independent_of_bounded_initializer(self):
        solver_path=(Path(__file__).resolve().parent.parent /
                     'global-inner-p1-field-402-absolute-variable-integration' /
                     'absolute_field_402.py')
        spec=importlib.util.spec_from_file_location('absolute_field_402_initializer_test',solver_path)
        self.assertIsNotNone(spec)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        T=np.array([[0,1,2,3]],dtype=np.int64)
        ids=np.array([0],dtype=np.int64)
        B=np.array([[0.0,0.5,0.5,0.0]],dtype=np.float64)
        fixed=np.array([0],dtype=np.int64)
        fv=np.array([0.0],dtype=np.float64)
        costs=np.ones(4,dtype=np.float64)
        zero=np.zeros(4,dtype=np.float64)
        other_bounded=np.array([0.0,0.9,0.1,0.7],dtype=np.float64)
        a,ra=module.solve_absolute_obstacle_field(zero,T,ids,B,fixed,fv,costs,0.4)
        b,rb=module.solve_absolute_obstacle_field(other_bounded,T,ids,B,fixed,fv,costs,0.4)
        self.assertTrue(ra['pass'],ra)
        self.assertTrue(rb['pass'],rb)
        self.assertEqual(ra['initializer_role'],'diagnostic_only_not_bound_or_objective')
        self.assertEqual(rb['initializer_role'],'diagnostic_only_not_bound_or_objective')
        np.testing.assert_array_equal(a,b)
        self.assertGreaterEqual(float(B[0]@a[T[0]]),0.4-1e-12)
        self.assertEqual(a[0],0.0)

    def test_fixed_o_capacity_does_not_relax_zero_values(self):
        T=np.array([[0,1,2,3]],dtype=np.int64)
        B=np.array([[0.5,0.5,0.0,0.0]],dtype=np.float64)
        cap=fixed_zero_capacity_report(T,[(0,B)],np.array([0,1]),4)
        self.assertEqual(cap[0]['max_free_capacity'],0.0)
        cap2=fixed_zero_capacity_report(T,[(0,B)],np.array([0]),4)
        self.assertEqual(cap2[0]['max_free_capacity'],0.5)

if __name__=='__main__': unittest.main(verbosity=2)
