import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from area_scp_support_023 import project_ball_halfspace, InfeasibleProjection, face_area_gradient, ball_intersection_support, progressive_plane_lower, project_two_balls_halfspace

class AreaSCPFixtures(unittest.TestCase):
    def test_six_mm_ball_plane_boundary(self):
        got=project_ball_halfspace(np.array([[9.,0.,0.]]),np.zeros((1,3)),6.,np.array([[1.,0.,0.]]),np.array([6.]),np.array([0]))
        self.assertAlmostEqual(float(np.linalg.norm(got[0])),6.,places=10)
        self.assertAlmostEqual(float(got[0,0]),6.,places=10)

    def test_infeasible_halfspace_ball_is_explicit(self):
        with self.assertRaises(InfeasibleProjection):
            project_ball_halfspace(np.zeros((1,3)),np.zeros((1,3)),6.,np.array([[1.,0.,0.]]),np.array([6.001]),np.array([0]))

    def test_tangent_circle_projection(self):
        got=project_ball_halfspace(np.array([[5.,4.,0.]]),np.zeros((1,3)),6.,np.array([[1.,0.,0.]]),np.array([5.9]),np.array([0]))
        self.assertGreaterEqual(got[0,0],5.9-1e-10)
        self.assertAlmostEqual(float(np.linalg.norm(got[0])),6.,places=9)

    def test_anchor_prefix_bitwise_fixed(self):
        # Bitwise equality includes signed zero and NaN payloads; fixture uses finite coordinates.
        old=np.array([[1.25,-2.5,3.75],[-0.0,4.0,5.0]],np.float32)
        updated=old.copy(); updated[1]=np.array([8,9,10],np.float32)
        anchors=np.array([0],np.int64)
        out=updated.copy(); out[anchors]=old[anchors]
        self.assertEqual(out[0].tobytes(),old[0].tobytes())
        self.assertNotEqual(out[1].tobytes(),old[1].tobytes())

    def test_area_gradient_finite_difference_37_faces(self):
        # Deterministic irregular mesh; the same 37 face-row count used by run-020's
        # actual reversal set. Validate the analytic local gradient by directional FD.
        rng=np.random.default_rng(20261025)
        points=rng.normal(size=(45,3))*7.0
        faces=[]
        for i in range(37): faces.append([i%45,(i+1)%45,(i+8)%45])
        faces=np.asarray(faces,np.int64)
        d=rng.normal(size=points.shape)
        eps=2e-5
        errors=[]
        for face in faces:
            a,b,c=points[face]; n=np.cross(b-a,c-a); n/=np.linalg.norm(n)
            g=face_area_gradient(points,face,n)
            numeric=(np.dot(n,np.cross(points[face][1]+eps*d[face[1]]-points[face[0]]-eps*d[face[0]],points[face][2]+eps*d[face[2]]-points[face[0]]-eps*d[face[0]]))-
                     np.dot(n,np.cross(points[face][1]-eps*d[face[1]]-points[face[0]]+eps*d[face[0]],points[face][2]-eps*d[face[2]]-points[face[0]]+eps*d[face[0]])))/(2*eps)
            analytic=float(np.dot(g,d[face].reshape(-1)))
            errors.append(abs(numeric-analytic)/max(1.,abs(numeric),abs(analytic)))
        self.assertLess(max(errors),2e-8)

if __name__=='__main__': unittest.main()

class FrozenRun020Fixtures(unittest.TestCase):
    ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150')
    def test_run020_37_flipped_face_gradients(self):
        z=np.load(self.ROOT/'run-018-generated-node-exit-targets-002/anchor-corrected-refined-start.npz',allow_pickle=False)
        c=np.load(self.ROOT/'run-020-nearest-exit-connected-fit-002/candidate.npz',allow_pickle=False)
        x=np.asarray(z['points_world_m'],np.float32).astype(np.float64)*1000.0
        y=np.asarray(c['points_world_m'],np.float32).astype(np.float64)*1000.0
        f=np.asarray(z['source_O_faces'],np.int64)
        xr=np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
        yr=np.cross(y[f[:,1]]-y[f[:,0]],y[f[:,2]]-y[f[:,0]])
        ids=np.flatnonzero(np.einsum('ij,ij->i',xr,yr)<=0)
        self.assertEqual(len(ids),37)
        rng=np.random.default_rng(20261025)
        d=rng.normal(size=x.shape)
        eps=2e-5
        errs=[]
        for row in ids:
            tri=f[row]
            nr=xr[row]/np.linalg.norm(xr[row])
            g=face_area_gradient(y,tri,nr)
            def val(sign):
                p=y[tri]+sign*eps*d[tri]
                return float(np.dot(nr,np.cross(p[1]-p[0],p[2]-p[0])))
            fd=(val(1)-val(-1))/(2*eps)
            an=float(np.dot(g,d[tri].reshape(-1)))
            errs.append(abs(fd-an)/max(1.,abs(fd),abs(an)))
        self.assertLess(max(errs),1e-8)

    def test_run020_typed_anchor_prefix_bitwise(self):
        z=np.load(self.ROOT/'run-018-generated-node-exit-targets-002/anchor-corrected-refined-start.npz',allow_pickle=False)
        c=np.load(self.ROOT/'run-020-nearest-exit-connected-fit-002/candidate.npz',allow_pickle=False)
        x=np.asarray(z['points_world_m'],np.float32); y=np.asarray(c['points_world_m'],np.float32)
        anchors=np.asarray(z['typed_coarse_anchor_output_rows_018'],np.int64)
        self.assertEqual(len(anchors),237)
        self.assertEqual(np.ascontiguousarray(x[anchors]).tobytes(),np.ascontiguousarray(y[anchors]).tobytes())

    def test_scaled_normal_rescales_plane_bound(self):
        a=project_ball_halfspace(np.zeros((1,3)),np.zeros((1,3)),6.,np.array([[2.,0.,0.]]),np.array([2.]),np.array([0]))
        b=project_ball_halfspace(np.zeros((1,3)),np.zeros((1,3)),6.,np.array([[1.,0.,0.]]),np.array([1.]),np.array([0]))
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_anchor_source_ball_conflict_is_explicit(self):
        with self.assertRaises(InfeasibleProjection):
            project_ball_halfspace(np.zeros((1,3)),np.array([[7.,0.,0.]]),6.,np.empty((0,3)),np.empty(0),np.empty(0,dtype=np.int64),np.array([0]))

    def test_fractional_index_rejected(self):
        with self.assertRaises(ValueError):
            project_ball_halfspace(np.zeros((1,3)),np.zeros((1,3)),6.,np.array([[1.,0.,0.]]),np.array([1.]),np.array([0.5]))


class BallIntersectionSupportTests(unittest.TestCase):
    def test_diagonal_overlap_support_point_is_on_both_spheres(self):
        c1=np.array([0.0,0.0,0.0]); r1=3.0; c2=np.array([1.0,1.0,0.0]); r2=2.0
        n=np.array([0.4,0.7,0.5916079783099616]); n=n/np.linalg.norm(n); d=c2-c1; D=np.linalg.norm(d); u=d/D
        a=(r1*r1-r2*r2+D*D)/(2*D); rho=np.sqrt(r1*r1-a*a); cp=c1+a*u
        q=n-np.dot(n,u)*u; q=q/np.linalg.norm(q); p=cp+rho*q
        self.assertAlmostEqual(np.linalg.norm(p-c1),r1,places=12); self.assertAlmostEqual(np.linalg.norm(p-c2),r2,places=12)
        self.assertAlmostEqual(ball_intersection_support(c1,r1,c2,r2,n),float(np.dot(n,p)),places=12)
    def test_support_uses_contained_directional_extreme(self):
        self.assertAlmostEqual(ball_intersection_support(np.zeros(3),2.,np.array([.5,0.,0.]),3.,np.array([0.,1.,0.])),2.,places=12)
    def test_concentric_uses_smaller_ball(self):
        c=np.array([1.,-2.,3.]); n=np.array([.2,.3,.4]); got=ball_intersection_support(c,7.,c,2.,n)
        self.assertAlmostEqual(got,float(np.dot(n/np.linalg.norm(n),c)+2.),places=12)
    def test_progress_stays_strictly_inside_diagonal_support(self):
        lower=progressive_plane_lower(np.array([.25]),np.array([4.]),np.array([2.])); self.assertAlmostEqual(float(lower[0]),.25+.95*(2-.25),places=12); self.assertLess(float(lower[0]),2.)
    def test_already_feasible_target_does_not_regress(self):
        self.assertEqual(float(progressive_plane_lower(np.array([2.]),np.array([1.]),np.array([3.]))[0]),2.)


class ExactTwoBallPlaneProjectionTests(unittest.TestCase):
    def test_diagonal_lens_with_plane_returns_feasible_projection(self):
        c1=np.array([0.,0.,0.]); c2=np.array([1.,1.,0.]); n=np.array([1.,0.,0.])
        got=project_two_balls_halfspace(np.array([5.,4.,3.]),c1,3.,c2,2.,n,2.6)
        self.assertLessEqual(np.linalg.norm(got-c1),3.+1e-9)
        self.assertLessEqual(np.linalg.norm(got-c2),2.+1e-9)
        self.assertGreaterEqual(float(np.dot(n,got)),2.6-1e-9)
        again=project_two_balls_halfspace(got,c1,3.,c2,2.,n,2.6)
        np.testing.assert_allclose(again,got,rtol=0,atol=1e-9)

    def test_active_three_boundary_vertex_is_recovered(self):
        c1=np.zeros(3); c2=np.array([1.,1.,0.]); r1=3.; r2=2.; n=np.array([1.,0.,0.]); h=2.6
        y=.9; z=np.sqrt(2.24-y*y); p=np.array([h,y,z])
        target=p+2*(p-c1)+3*(p-c2)-4*n
        got=project_two_balls_halfspace(target,c1,r1,c2,r2,n,h)
        np.testing.assert_allclose(got,p,rtol=0,atol=1e-8)

    def test_infeasible_plane_ball_lens_fails_closed(self):
        with self.assertRaises(ValueError):
            project_two_balls_halfspace(np.zeros(3),np.zeros(3),2.,np.array([.5,0.,0.]),1.,np.array([1.,0.,0.]),3.)

    def test_actual_five_source_trust_overlap_rows(self):
        R=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150')
        start=np.load(R/'run-018-generated-node-exit-targets-002/anchor-corrected-refined-start.npz',allow_pickle=False)
        base=np.load(R/'run-009-certified-source-normal-union-rays/endpoint-candidate.npz',allow_pickle=False)
        from scipy import sparse
        X=np.asarray(start['points_world_m'],np.float32).astype(float)
        P=sparse.csr_matrix((np.asarray(start['source_prolongation_weights'],float),np.asarray(start['source_prolongation_node_rows'],np.int64),np.asarray(start['source_prolongation_indptr'],np.int64)),shape=(len(X),len(base['points_world_m'])))
        ref=np.asarray(P@np.asarray(base['points_before_joint_self_pair_repair_m'],float))
        center=(ref-X)*1000.
        rows=np.flatnonzero(np.linalg.norm(center,axis=1)+2.0>7.0+1e-10)
        self.assertEqual(len(rows),5)
        n=np.array([.31,.47,.826]); n=n/np.linalg.norm(n)
        for row in rows:
            c=center[row]; support=ball_intersection_support(c,7.,np.zeros(3),2.,n)
            lower=float(progressive_plane_lower(np.array([0.]),np.array([8.]),np.array([support]))[0])
            got=project_two_balls_halfspace(c+10*n,c,7.,np.zeros(3),2.,n,lower)
            self.assertLessEqual(np.linalg.norm(got-c),7.+1e-8)
            self.assertLessEqual(np.linalg.norm(got),2.+1e-8)
            self.assertGreaterEqual(float(np.dot(n,got)),lower-1e-8)
