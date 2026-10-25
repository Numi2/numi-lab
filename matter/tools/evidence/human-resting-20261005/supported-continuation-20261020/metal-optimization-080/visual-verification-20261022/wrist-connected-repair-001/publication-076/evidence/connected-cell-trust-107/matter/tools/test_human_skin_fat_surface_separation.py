import unittest
import numpy as np
from scipy import sparse
from human_skin_fat_surface_separation import SurfaceSeparationError, build_pair_inequalities, _vertex_constraint_row

def emit_positions(origin, weights, transfer_origin, points):
    return origin.astype(np.float64) + weights @ (points.astype(np.float64)-transfer_origin.astype(np.float64))

class SurfaceSeparationTests(unittest.TestCase):
    def test_shared_identity_weak_separator_and_baseline_chain_rule(self):
        detail=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]])
        ia=np.array([0,1,2]);ib=np.array([0,3,4]);w=sparse.eye(5,format="csr");x0=detail.copy()
        xc=x0.copy();xc[3]=[.2,0.,0.];xc[4]=[0.,.2,0.];dc=emit_positions(detail,w,x0,xc)
        M,t,r=build_pair_inequalities(ia,ib,detail,w,x0,x0,candidate_detail_positions=dc,pair_id="shared")
        self.assertEqual(M.shape,(8,15));self.assertEqual(r["shared_vertex_ids"],[0]);self.assertEqual(r["margin_m"],0.)
        self.assertGreater(r["candidate_margin_violation_m"],0.)
        omitted=[p for p in r["vertex_pair_provenance"] if p["status"]=="omitted_shared_identity_zero_row"]
        self.assertEqual(len(omitted),1);self.assertEqual(omitted[0]["target"],0.)
        self.assertGreaterEqual(float(np.min(M@x0.reshape(-1)-t)),-1e-12)
    def test_nonshared_positive_margin_and_mapped_chain_rule(self):
        detail=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        ia=np.array([0,1,2]);ib=np.array([3,4,5]);w=sparse.eye(6,format="csr");x0=detail.copy()
        xc=x0.copy();xc[3:,2]=.4;dc=emit_positions(detail,w,x0,xc)
        M,t,r=build_pair_inequalities(ia,ib,detail,w,x0,x0,candidate_detail_positions=dc,pair_id=7)
        self.assertEqual(M.shape,(9,18));self.assertAlmostEqual(r["margin_m"],.5,places=12)
        self.assertGreater(r["candidate_margin_violation_m"],0.)
        base_slack=np.asarray(M@x0.reshape(-1)-t);candidate_slack=np.asarray(M@xc.reshape(-1)-t)
        self.assertGreaterEqual(float(base_slack.min()),-1e-12)
        normal=np.asarray(r["normal"])
        provenance=[p for p in r["vertex_pair_provenance"] if p["status"]=="constraint"]
        for k,p in enumerate(provenance):
            direct=float(np.dot(normal,dc[p["vertex_b"]]-dc[p["vertex_a"]])-p["margin_m"])
            self.assertAlmostEqual(float(candidate_slack[k]),direct,places=12)
    def test_fixed_zero_transfer_infeasible_margin_fails(self):
        fixed=sparse.csr_matrix([[1.,0.]])
        detail=np.array([[0.,0.,0.],[.5,0.,0.]]);origin=np.zeros((2,3))
        with self.assertRaisesRegex(SurfaceSeparationError,"fixed zero-transfer pair"):
            _vertex_constraint_row(0,1,np.array([1.,0.,0.]),.75,detail,origin,origin,fixed,fixed)
    def test_fixed_zero_transfer_feasible_row_is_skipped(self):
        fixed=sparse.csr_matrix([[1.,0.]])
        detail=np.array([[0.,0.,0.],[.5,0.,0.]]);origin=np.zeros((2,3))
        row,target,p=_vertex_constraint_row(0,1,np.array([1.,0.,0.]),.25,detail,origin,origin,fixed,fixed)
        self.assertIsNone(row);self.assertIsNone(target);self.assertEqual(p["status"],"omitted_fixed_transfer_zero_row")
        self.assertGreaterEqual(p["baseline_gap_m"],p["margin_m"])
    def test_finite_shape_and_face_domain_validation(self):
        detail=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        ia=np.array([0,1,2]);ib=np.array([3,4,5]);w=sparse.eye(6,format="csr");x=detail.copy()
        bad=detail.copy();bad[0,0]=np.nan
        with self.assertRaises(SurfaceSeparationError):build_pair_inequalities(ia,ib,bad,w,x,x)
        with self.assertRaises(SurfaceSeparationError):build_pair_inequalities(np.array([0,1,9]),ib,detail,w,x,x)
        with self.assertRaises(SurfaceSeparationError):build_pair_inequalities(np.array([0.,1.,2.]),ib,detail,w,x,x)
        with self.assertRaises(SurfaceSeparationError):build_pair_inequalities(ia,ib,detail,w[:,:5],x,x)

    def test_intersecting_baseline_triangles_have_no_separator(self):
        a=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        b=np.array([[.2,-.2,-1.],[.2,1.,-1.],[.2,.2,1.]])
        ids_a=np.array([0,1,2]);ids_b=np.array([3,4,5])
        w=sparse.eye(6,format="csr")
        with self.assertRaisesRegex(SurfaceSeparationError,"no valid baseline SAT separator"):
            build_pair_inequalities(ids_a,ids_b,np.vstack([a,b]),w,np.vstack([a,b]),np.vstack([a,b]))

    def test_three_pair_constraint_composition_preserves_baseline(self):
        detail=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],
                         [0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        w=sparse.eye(len(detail),format="csr");x=detail.copy()
        faces=[(np.array([0,1,2]),np.array([0,3,4])),
               (np.array([0,1,2]),np.array([5,6,7])),
               (np.array([3,4,2]),np.array([5,6,7]))]
        rows=[];targets=[]
        for pair,(ia,ib) in enumerate(faces):
            M,t,_=build_pair_inequalities(ia,ib,detail,w,x,x,pair_id=pair)
            rows.append(M);targets.extend(t)
        M=sparse.vstack(rows,format="csr");targets=np.asarray(targets)
        self.assertEqual(M.shape[1],3*len(x))
        self.assertGreater(M.shape[0],0)
        self.assertGreaterEqual(float(np.min(M@x.reshape(-1)-targets)),-1e-12)

if __name__=="__main__":unittest.main()

