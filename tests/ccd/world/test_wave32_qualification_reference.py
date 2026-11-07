#!/usr/bin/env python3
"""Closed-form qualification and retained counterexamples for wave32 updates."""
import math
import random
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


def inverse3(matrix):
    a=[list(row)+[float(i==j) for j in range(3)] for i,row in enumerate(matrix)]
    for k in range(3):
        i=max(range(k,3),key=lambda j:abs(a[j][k]))
        a[k],a[i]=a[i],a[k]
        pivot=a[k][k]
        a[k]=[v/pivot for v in a[k]]
        for i in range(3):
            if i!=k:
                factor=a[i][k]
                a[i]=[v-factor*w for v,w in zip(a[i],a[k])]
    return [row[3:] for row in a]


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

    def test_helper_behavior_for_unvalidated_one_axis(self):
        self.assertEqual(project([2.,.3,.2],(.5,0)),[2.,0.,0.])
        # This is not an accepted ConstraintIR cone: validation requires both
        # coefficients to be positive or both to be zero.

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

    def test_candidate_002_floor_disagrees_with_factor_regularization(self):
        normal_response=1e-12; scale=1.; normal_rhs=1.
        candidate=normal_rhs/max(normal_response,1e-10)
        regularized_inverse=normal_rhs/(normal_response+.01*scale)
        self.assertGreater(candidate/regularized_inverse,1e7)
        self.assertLess(normal_response*candidate,.02)

    def test_conditional_update_matches_regularized_operator(self):
        # Independently solve the 2x2 conditional tangent equations directly,
        # then compare the inverse-block Schur identity used by the shader.
        rng=random.Random(7192)
        cases=[[[1e-12,0,0],[0,1,0],[0,0,1]]]
        for _ in range(100):
            a=[[rng.uniform(-2,2) for _ in range(3)] for _ in range(3)]
            cases.append([[sum(a[k][i]*a[k][j] for k in range(3))
                           for j in range(3)] for i in range(3)])
        for w in cases:
            scale=max(abs(v) for row in w for v in row)
            m=[[(w[i][j]+w[j][i])*.5+(.01*scale if i==j else 0.)
                for j in range(3)] for i in range(3)]
            inv=inverse3(m); rhs=[.4,-.7,1.1]
            dn=rhs[0]/m[0][0]
            t=[rhs[i]-m[i][0]*dn for i in (1,2)]
            det=m[1][1]*m[2][2]-m[1][2]*m[2][1]
            direct=[(m[2][2]*t[0]-m[1][2]*t[1])/det,
                    (-m[2][1]*t[0]+m[1][1]*t[1])/det]
            schur=[sum((inv[i][j]-inv[i][0]*inv[0][j]/inv[0][0])*rhs[j]
                       for j in (1,2))+inv[i][0]*dn/inv[0][0]
                   for i in (1,2)]
            for x,y in zip(direct,schur):
                self.assertAlmostEqual(x,y,delta=1e-8*max(1,abs(x),abs(y)))
            self.assertGreater(m[0][0],0)
        # The singular-scale example now uses the regularized response.
        self.assertAlmostEqual(1/(1e-12+.01),100,places=7)


if __name__=="__main__":
    unittest.main()
