import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from absolute_field_402 import solve_absolute_obstacle_field, reactivate_skipped_tets, certify_skipped_tets

class AbsoluteFieldTests(unittest.TestCase):
    def setUp(self):
        self.T=np.asarray([[0,1,2,3],[1,4,2,3]],np.int64)
        self.cost=np.ones(5)
    def test_weighted_minimum_can_lower_initializer(self):
        p=np.full(5,.9)
        B=np.asarray([[.5,.5,0,0]])
        x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[0],[0.0],self.cost,.2)
        self.assertIsNotNone(x);self.assertTrue(r["pass"])
        self.assertLess(x[1],p[1]);self.assertAlmostEqual(x[1],.4,places=8)
        self.assertGreaterEqual(float(B[0]@x[self.T[0]]),.2-1e-8)
        self.assertFalse(r["initializer_as_lower_bound"] if "initializer_as_lower_bound" in r else False)
        self.assertEqual(r["initializer_role"],"diagnostic_only_not_bound_or_objective")
    def test_fixed_support_conflict(self):
        p=np.zeros(5);B=np.asarray([[.2,.8,0,0]])
        x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[0,1],[0,0],self.cost,.1)
        self.assertIsNone(x);self.assertEqual(r["status"],"fixed_support_conflict");self.assertFalse(r["pass"])
    def test_noninteger_identity_rows_are_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        with self.assertRaisesRegex(ValueError,"constraint_tet_ids"):
            solve_absolute_obstacle_field(p,self.T,[0.5],B,[0],[0.0],self.cost,.2)

    def test_max_value_validation_precedes_initializer_bounds(self):
        for bad in (float("nan"),float("inf"),0.0,-1.0):
            with self.subTest(max_value=bad), self.assertRaisesRegex(ValueError,"max_value"):
                solve_absolute_obstacle_field(np.asarray([.5]),np.asarray([[0,0,0,0]],np.int64),[],np.empty((0,4)),[],[],np.ones(1),.2,max_value=bad)
    def test_corrupted_success_vector_is_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        fake=SimpleNamespace(success=True,message="mocked",status=0,nit=1,x=np.asarray([np.nan,0,0,0]))
        with patch("absolute_field_402.linprog",return_value=fake):
            x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[],[],self.cost,.5)
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["status"],"solver_output_validation_failed")
    def test_finite_but_constraint_violating_success_is_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        fake=SimpleNamespace(success=True,message="mocked",status=0,nit=1,x=np.zeros(5))
        with patch("absolute_field_402.linprog",return_value=fake):
            x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[],[],self.cost,.5)
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["failure"],"postsolve_field_or_constraint_check")
    def test_reactivation_rejects_nonfinite_inputs(self):
        for field,threshold in (([0,.9,.9,.9,.9],np.nan),([0,np.inf,.9,.9,.9],.5)):
            with self.subTest(field=field,threshold=threshold),self.assertRaises(ValueError):
                reactivate_skipped_tets(self.T,np.asarray(field),[0],threshold)
            with self.subTest(certify=True,field=field,threshold=threshold),self.assertRaises(ValueError):
                certify_skipped_tets(self.T,np.asarray(field),[0],threshold)
    def test_reactivation_finds_skipped_cell_after_decrease(self):
        p=np.asarray([0,.9,.9,.9,.9]);p[4]=.1
        self.assertEqual(reactivate_skipped_tets(self.T,p,[0],.5).tolist(),[1])
        self.assertFalse(certify_skipped_tets(self.T,p,[0],.5)["pass"])
    def test_solver_preserves_explicit_fixed_values(self):
        p=np.full(5,.9);B=np.asarray([[.5,.5,0,0]])
        x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[0],[0.0],self.cost,.2)
        self.assertEqual(float(x[0]),0.0);self.assertTrue(r["fixed_values_exact"])

if __name__=="__main__":unittest.main()
