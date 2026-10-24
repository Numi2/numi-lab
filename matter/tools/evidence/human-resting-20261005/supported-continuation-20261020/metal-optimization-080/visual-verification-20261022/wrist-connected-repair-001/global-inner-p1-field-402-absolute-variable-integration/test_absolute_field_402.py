import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib, json
import numpy as np
from absolute_field_402 import solve_absolute_obstacle_field, reactivate_skipped_tets, certify_skipped_tets

HERE=Path(__file__).resolve().parent
class AbsoluteFieldTests(unittest.TestCase):
    def setUp(self):
        self.T=np.asarray([[0,1,2,3],[1,4,2,3]],np.int64)
        self.cost=np.ones(5)
    def test_current_producer_uses_398_and_no_legacy_constraint_parts(self):
        source=(HERE/"build_global_p1_402.py").read_text()
        self.assertIn("local-quality-cavities-398/candidate-with-transfer.npz",source)
        self.assertIn("p1-interior-face-support-395/refined-background-unsolved-initializer.npz",source)
        self.assertIn("global-inner-p1-field-399-398-refresh/run-002/constraint-parts",source)
        for legacy in ("global-inner-p1-field-296b-allbank-exclusion", "global-inner-p1-field-310-all689-resume", "global-inner-p1-field-311-all689-resume"):
            self.assertNotIn(legacy,source)
    def test_weighted_minimum_can_lower_initializer(self):
        p=np.full(5,.9);B=np.asarray([[.5,.5,0,0]])
        x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[0],[0.0],self.cost,.2)
        self.assertIsNotNone(x);self.assertTrue(r["pass"])
        self.assertLess(x[1],p[1]);self.assertAlmostEqual(x[1],.4,places=8)
        self.assertGreaterEqual(float(B[0]@x[self.T[0]]),.2-1e-8)
        self.assertEqual(r["initializer_role"],"diagnostic_only_not_bound_or_objective")
    def test_actual_fixture_fixed_O_conflict(self):
        q=np.load(HERE/"retained-399-fixed-O-witness-398.npz",allow_pickle=False)
        T=q["tetrahedra_prefix"];n=int(q["node_count"]);ti=int(q["constraint_tet_id"])
        self.assertEqual(ti,88190);self.assertEqual(T[ti].tolist(),[9451,9471,9473,27269])
        fixed=q["fixed_outer_node_ids"];support=q["support_node_ids"]
        self.assertTrue(set(map(int,support)).issubset(set(map(int,fixed))))
        B=q["constraint_barycentric"];p=np.full(n,.5);cost=np.ones(n)
        x,r=solve_absolute_obstacle_field(p,T,[ti],B,fixed,np.zeros(len(fixed)),cost,float(q["target"]))
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["status"],"fixed_support_conflict")
        self.assertEqual(r["conflict_tet_ids"],[ti]);self.assertEqual(r["minimum_capacity"],0.0)
    def test_actual_399_partial_witness_is_pinned_to_398_and_conflicts(self):
        root=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
        g398=root/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
        i395=root/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-interior-face-support-395/refined-background-unsolved-initializer.npz"
        c399=root/"global-inner-p1-field-399-398-refresh/run-002/global-field-cache.npz"
        r399=root/"global-inner-p1-field-399-398-refresh/run-002/partial-fixed-boundary-conflict-classification.json"
        d399=root/"global-inner-p1-field-399-398-refresh/run-002"
        a=np.load(g398,allow_pickle=False);i=np.load(i395,allow_pickle=False);c=np.load(c399,allow_pickle=False)
        self.assertTrue(np.array_equal(a["points_world_m"],i["points_world_m"]))
        self.assertTrue(np.array_equal(a["points_world_m"],c["points_world_m"]))
        self.assertTrue(np.array_equal(a["tetrahedra"],c["tetrahedra"]))
        self.assertEqual(hashlib.sha256(i395.read_bytes()).hexdigest(),str(c["initializer_sha256"]))
        meta=json.loads(r399.read_text());w=meta["first_witness"]
        self.assertEqual(meta["background_sha256"],hashlib.sha256(g398.read_bytes()).hexdigest())
        self.assertEqual(meta["initializer_sha256"],hashlib.sha256(i395.read_bytes()).hexdigest())
        self.assertEqual((w["tet_row"],w["flat_bank_face_row"],w["bank_id"]),(88190,16346,4))
        q=np.load(d399/"constraint-parts/part-00000.npz",allow_pickle=False)
        rows=np.flatnonzero((q["tet_ids"]==w["tet_row"])&(q["face_ids"]==w["flat_bank_face_row"])&(q["bank_ids"]==w["bank_id"]))
        self.assertEqual(len(rows),1)
        row=int(rows[0]);bary=q["polygon_barycentric"][q["poly_offsets"][row]:q["poly_offsets"][row+1]]
        wb=np.asarray(w["barycentric"],np.float64)
        self.assertTrue(np.any(np.all(bary==wb,axis=1)))
        T=np.asarray(a["tetrahedra"],np.int64);n=len(a["points_world_m"]);ti=int(w["tet_row"])
        fixed=np.asarray(a["physical_outer_node_ids"],np.int64)
        self.assertEqual(T[ti].tolist(),[9451,9471,9473,27269])
        self.assertTrue(set(w["support_nodes"]).issubset(set(map(int,fixed))))
        point=wb@np.asarray(a["points_world_m"],np.float64)[T[ti]]
        np.testing.assert_allclose(point,np.asarray(w["world_m"]),rtol=0,atol=2e-8)
        x,r=solve_absolute_obstacle_field(np.asarray(i["phi_initializer_unsolved"]),T,[ti],wb[None,:],fixed,np.zeros(len(fixed)),np.ones(n),0.341)
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["status"],"fixed_support_conflict")
        self.assertEqual(r["minimum_capacity"],0.0)
    def test_all_fixed_satisfied_skips_empty_lp(self):
        T=np.asarray([[0,1,2,3]],np.int64);p=np.zeros(4);B=np.asarray([[.25,.25,.25,.25]])
        fixed=np.arange(4,dtype=np.int64);fv=np.full(4,.5)
        with patch("absolute_field_402.linprog",side_effect=AssertionError("LP should not run")):
            x,r=solve_absolute_obstacle_field(p,T,[0],B,fixed,fv,np.ones(4),.5)
        self.assertTrue(r["pass"]);np.testing.assert_array_equal(x,fv)
    def test_fixed_support_conflict(self):
        p=np.zeros(5);B=np.asarray([[.2,.8,0,0]])
        x,r=solve_absolute_obstacle_field(p,self.T,[0],B,[0,1],[0,0],self.cost,.1)
        self.assertIsNone(x);self.assertEqual(r["status"],"fixed_support_conflict");self.assertFalse(r["pass"])
    def test_max_value_validation_precedes_initializer_bounds(self):
        for bad in (float("nan"),float("inf"),0.0,-1.0):
            with self.subTest(max_value=bad),self.assertRaisesRegex(ValueError,"max_value"):
                solve_absolute_obstacle_field(np.asarray([.5]),np.asarray([[0,0,0,0]],np.int64),[],np.empty((0,4)),[],[],np.ones(1),.2,max_value=bad)
    def test_corrupted_success_vector_is_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        fake=SimpleNamespace(success=True,message="mocked",status=0,nit=1,x=np.full(5,np.nan))
        with patch("absolute_field_402.linprog",return_value=fake):
            x,r=solve_absolute_obstacle_field(p,self.T,[0],B,np.empty(0,dtype=np.int64),[],self.cost,.5)
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["status"],"solver_output_validation_failed")
    def test_finite_but_constraint_violating_success_is_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        fake=SimpleNamespace(success=True,message="mocked",status=0,nit=1,x=np.zeros(5))
        with patch("absolute_field_402.linprog",return_value=fake):
            x,r=solve_absolute_obstacle_field(p,self.T,[0],B,np.empty(0,dtype=np.int64),[],self.cost,.5)
        self.assertIsNone(x);self.assertFalse(r["pass"]);self.assertEqual(r["failure"],"postsolve_field_or_constraint_check")
    def test_noninteger_identity_rows_are_rejected(self):
        p=np.zeros(5);B=np.asarray([[.5,.5,0,0]])
        with self.assertRaisesRegex(ValueError,"constraint_tet_ids"):
            solve_absolute_obstacle_field(p,self.T,[.5],B,[0],[0.0],self.cost,.2)
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
