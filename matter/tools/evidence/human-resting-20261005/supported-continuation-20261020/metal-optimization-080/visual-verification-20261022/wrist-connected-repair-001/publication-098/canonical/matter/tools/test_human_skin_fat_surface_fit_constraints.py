import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from human_skin_fat_surface_fit_constraints import project_ball_halfspace, InfeasibleProjection, face_area_gradient, ball_intersection_support, progressive_plane_lower, project_two_balls_halfspace

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


class FrozenRun020Fixtures(unittest.TestCase):
    FIXTURE=Path(__file__).with_name('fixtures')/'human_skin_fat_surface_fit_constraints.npz'
    def test_retained_37_flipped_face_gradients(self):
        z=np.load(self.FIXTURE,allow_pickle=False)
        tri=z['trial_triangles_mm'];normal=z['reference_normals'];direction=z['gradient_directions']
        self.assertEqual(len(tri),37)
        self.assertTrue(np.all(np.einsum('ij,ij->i',normal,np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))<=0))
        for x,n,d in zip(tri,normal,direction):
            g=face_area_gradient(x,np.arange(3),n)
            def value(sign):
                p=x+sign*2e-5*d
                return np.dot(n,np.cross(p[1]-p[0],p[2]-p[0]))
            fd=(value(1)-value(-1))/(4e-5)
            an=float(g@d.reshape(-1))
            self.assertLess(abs(fd-an)/max(1.,abs(fd),abs(an)),1e-8)

    def test_retained_typed_anchors_preserved_bitwise(self):
        z=np.load(self.FIXTURE,allow_pickle=False)
        self.assertEqual(len(z['anchor_output_rows']),237)
        self.assertEqual(z['anchor_start_f32'].tobytes(),z['anchor_trial_f32'].tobytes())

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
        z=np.load(FrozenRun020Fixtures.FIXTURE,allow_pickle=False)
        center=z['overlap_source_centers_mm']
        self.assertEqual(len(center),5)
        rows=np.arange(5)
        n=np.array([.31,.47,.826]); n=n/np.linalg.norm(n)
        for row in rows:
            c=center[row]; support=ball_intersection_support(c,7.,np.zeros(3),2.,n)
            lower=float(progressive_plane_lower(np.array([0.]),np.array([8.]),np.array([support]))[0])
            got=project_two_balls_halfspace(c+10*n,c,7.,np.zeros(3),2.,n,lower)
            self.assertLessEqual(np.linalg.norm(got-c),7.+1e-8)
            self.assertLessEqual(np.linalg.norm(got),2.+1e-8)
            self.assertGreaterEqual(float(np.dot(n,got)),lower-1e-8)

    def test_projection_variational_inequality_on_feasible_samples(self):
        rng=np.random.default_rng(951)
        for _ in range(24):
            c1=rng.normal(size=3)*.3
            c2=rng.normal(size=3)*.3
            n=rng.normal(size=3);n/=np.linalg.norm(n)
            r1=1.0;r2=1.1;h=-.1
            v=rng.normal(size=3)*3
            p=project_two_balls_halfspace(v,c1,r1,c2,r2,n,h)
            samples=rng.normal(size=(700,3))*.8
            good=(np.linalg.norm(samples-c1,axis=1)<=r1)&(np.linalg.norm(samples-c2,axis=1)<=r2)&((samples@n)>=h)
            self.assertTrue(good.any())
            self.assertLessEqual(float(np.max((samples[good]-p)@(v-p))),1e-8)

if __name__=='__main__': unittest.main()
