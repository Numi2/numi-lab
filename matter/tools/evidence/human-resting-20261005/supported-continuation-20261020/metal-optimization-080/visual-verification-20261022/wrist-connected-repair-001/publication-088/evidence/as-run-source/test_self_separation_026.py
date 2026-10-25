import unittest
import numpy as np
from self_separation_026 import build_pair_rows, project_linear_rows, relax_pair_rows_for_projection

class PairConstraintTests(unittest.TestCase):
    def test_shared_vertex_wedge_rows_are_feasible_and_project_crossing_step(self):
        # Two triangles share only the origin and occupy opposite sides of an x=0 plane.
        pts=np.array([[0,0,0],[1,0,0],[1,1,0],[0,-1,0],[-1,-1,0]],dtype=float)
        faces=np.array([[0,1,2],[0,3,4]],dtype=int)
        rows,meta=build_pair_rows(pts,faces,[(0,1)])
        self.assertEqual(len(rows),4); self.assertEqual(meta[0]['kind'],'common_vertex_wedge')
        d=np.zeros_like(pts); d[[1,2],0]=-2.0; d[[3,4],0]=2.0
        out,report=project_linear_rows(d,rows,max_sweeps=1000,tolerance_mm=1e-9)
        self.assertTrue(report['converged'],report)
        for r in rows:
            self.assertGreaterEqual(sum(float(np.dot(c,out[i])) for i,c in zip(r['nodes'],r['coefficients'])),float(r['lower'])-1e-8)

    def test_disjoint_triangles_use_moving_separating_midplane(self):
        pts=np.array([[0,0,0],[1,0,0],[0,1,0],[1.001,0,0],[2,0,0],[1.001,1,0]],float)
        faces=np.array([[0,1,2],[3,4,5]],int)
        rows,meta=build_pair_rows(pts,faces,[(0,1)])
        self.assertEqual(len(rows),6); self.assertEqual(meta[0]['kind'],'moving_sat_midplane')
        self.assertGreater(meta[0]['closest_distance_mm'],0)
        d=np.zeros_like(pts); d[3:,0]=-1.0
        out,report=project_linear_rows(d,rows,max_sweeps=1000,tolerance_mm=1e-9)
        self.assertTrue(report['converged'],report)
        self.assertLessEqual(np.max(np.abs(out.sum(axis=0))),100.0)
        for r in rows:
            val=sum(float(np.dot(c,out[i])) for i,c in zip(r['nodes'],r['coefficients']))
            self.assertGreaterEqual(val,float(r['lower'])-1e-8)

    def test_projection_allowance_preserves_positive_margin_and_relaxes_rhs(self):
        pts=np.array([[0,0,0],[1,0,0],[1,1,0],[0,-1,0],[-1,-1,0]],float)
        faces=np.array([[0,1,2],[0,3,4]],int)
        rows,meta=build_pair_rows(pts,faces,[(0,1)])
        old=np.array([r['lower'] for r in rows])
        adjusted=relax_pair_rows_for_projection(rows,meta,1.5e-5)
        self.assertTrue(all(r['effective_preserved_margin_mm']>0 for r in adjusted))
        np.testing.assert_allclose(np.array([r['lower'] for r in rows]),old-1.5e-5,rtol=0,atol=1e-15)
        self.assertTrue(all(r['projection_relaxation_mm']==1.5e-5 for r in adjusted))
        rows2,meta2=build_pair_rows(pts,faces,[(0,1)])
        with self.assertRaises(ValueError): relax_pair_rows_for_projection(rows2,meta2,meta2[0]['preserved_margin_mm'])

    def test_pair_ids_validate_and_baseline_rows_feasible(self):
        pts=np.array([[0,0,0],[1,0,0],[1,1,0],[0,-1,0],[-1,-1,0]],float)
        faces=np.array([[0,1,2],[0,3,4]],int)
        rows,_=build_pair_rows(pts,faces,[(0,1)])
        projected,report=project_linear_rows(np.zeros_like(pts),rows)
        self.assertTrue(report['converged']); np.testing.assert_allclose(projected,0,atol=1e-12)
        with self.assertRaises(ValueError): build_pair_rows(pts,faces,[(0,8)])

if __name__=='__main__': unittest.main()

