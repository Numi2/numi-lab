#!/usr/bin/env python3
"""Closed-form qualification and retained counterexamples for wave32 updates."""
import math
import unittest


def project(p, mu, maximum=0):
    normal=max(p[0],0)
    if maximum>0:
        normal=min(normal,maximum)
    limits=(mu[0]*normal,mu[1]*normal)
    if not all(x>0 for x in limits):
        return [normal,0.,0.]
    length=math.hypot(p[1]/limits[0],p[2]/limits[1])
    return [normal,p[1]/max(1,length),p[2]/max(1,length)]


def diagonal_update(impulse, velocity, diagonal, mu, maximum=0):
    rhs=[-velocity[i]-diagonal[i]*impulse[i] for i in range(3)]
    normal=max(impulse[0]+rhs[0]/max(diagonal[0],1e-10),0.)
    if maximum>0:
        normal=min(normal,maximum)
    return project([normal,impulse[1]+rhs[1]/diagonal[1],
                    impulse[2]+rhs[2]/diagonal[2]],mu,maximum)


class Wave32AnalyticQualification(unittest.TestCase):
    def test_frictionless_and_interior_stick(self):
        for mu, velocity in [((0.,0.),[-2.,.1,.2]),((.5,.5),[-2.,.1,.2])]:
            p=diagonal_update([0.,0.,0.],velocity,[1.,2.,1.],mu)
            self.assertAlmostEqual(p[0],2)
            if mu[0]==0:
                self.assertEqual(p[1:],[0,0])
            else:
                self.assertAlmostEqual(velocity[1]+2*p[1],0)
                self.assertAlmostEqual(velocity[2]+p[2],0)
                self.assertLess(math.hypot(p[1],p[2]),mu[0]*p[0])

    def test_normal_cap_has_distinct_complementarity(self):
        p=diagonal_update([0.,0.,0.],[-2.,0.,0.],[1.,1.,1.],(0.,0.),1.)
        self.assertEqual(p,[1.,0.,0.])
        # A nonzero normal residual at the upper impulse bound is expected.
        self.assertEqual(-2+p[0],-1.)

    def test_preexisting_one_axis_projection_discards_allowed_friction(self):
        self.assertEqual(project([2.,.3,.2],(.5,0)),[2.,0.,0.])
        # The u-axis impulse .3 is admissible and its loss is not a property
        # of a degenerate ellipse, which becomes a line segment.

    def test_preexisting_anisotropic_projection_has_wrong_sliding_direction(self):
        mu=(.8,.2); n=2.; incoming=(3.,4.); t=[0.,0.]
        for _ in range(300):
            proposal=[t[i]-(incoming[i]+t[i]) for i in range(2)]
            p=project([n,*proposal],mu);t=p[1:]
        slip=[incoming[i]+t[i] for i in range(2)]
        scale=math.hypot(mu[0]*slip[0],mu[1]*slip[1])
        oracle=[-n*mu[i]**2*slip[i]/scale for i in range(2)]
        self.assertGreater(math.hypot(t[0]-oracle[0],t[1]-oracle[1]),1.)

    def test_distributed_unrelaxed_identical_rows_oscillate(self):
        # W=[[1,1],[1,1]], rhs=(1,1), initial impulses=(0,0).
        p=[0.,0.]; observed=[]
        for _ in range(4):
            rhs=1-sum(p)
            p=[max(p[0]+rhs,0),max(p[1]+rhs,0)]
            observed.append(tuple(p))
        self.assertEqual(observed,[(1.,1.),(0.,0.),(1.,1.),(0.,0.)])

    def test_candidate_floor_disagrees_with_factor_regularization(self):
        normal_response=1e-12; scale=1.; normal_rhs=1.
        candidate=normal_rhs/max(normal_response,1e-10)
        regularized_inverse=normal_rhs/(normal_response+.01*scale)
        self.assertGreater(candidate/regularized_inverse,1e7)
        self.assertLess(normal_response*candidate,.02)


if __name__=="__main__":
    unittest.main()
