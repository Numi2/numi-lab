#!/usr/bin/env python3
"""Independent FP64, nonnegative two-row impulse QP reference; friction excluded."""
import math

def best_pair(a,b,c,gi,gj,old_i,old_j):
    # Minimize 1/2 d^T W d + g^T d, with old+d >= 0.
    assert a>0 and c>0
    lo_i,lo_j=-old_i,-old_j
    choices=[(0.,0.),(lo_i,lo_j)]
    if lo_j<=0: choices.append((max(lo_i,-gi/a),0.))
    if lo_i<=0: choices.append((0.,max(lo_j,-gj/c)))
    choices.append((lo_i,max(lo_j,-(gj+b*lo_i)/c)))
    choices.append((max(lo_i,-(gi+b*lo_j)/a),lo_j))
    det=a*c-b*b
    if det > 1.e-12*a*c:
        di=(b*gj-c*gi)/det
        dj=(b*gi-a*gj)/det
        if di>=lo_i and dj>=lo_j: choices.append((di,dj))
    def objective(z):
        x,y=z
        return .5*(a*x*x+2*b*x*y+c*y*y)+gi*x+gj*y
    return min(choices,key=objective)

# Centered 257-contact fixture. Both endpoint impulses stop the body and
# cancel torque; all intermediate rows consequently have zero velocity.
x=6.4
a=1+.001*x*x;b=1-.001*x*x
di,dj=best_pair(a,b,a,-1.,-1.,0.,0.)
assert abs(di-.5)<1e-12 and abs(dj-.5)<1e-12
assert max(abs(-1+(1+.001*t*(-x))*di+(1+.001*t*x)*dj)
           for t in [(i-128)*.05 for i in range(257)]) < 1.e-12
# Rank-deficient identical rows must still make valid progress via a bound.
di,dj=best_pair(1.,1.,1.,-1.,-1.,0.,0.)
assert min(di,dj)>=0 and abs(di+dj-1.)<1.e-12
# A unilateral boundary controls a case whose unconstrained answer is negative.
di,dj=best_pair(1.,.5,1.,-1.,.1,0.,0.)
assert abs(di-1.)<1.e-12 and dj==0.
assert all(math.isfinite(z) for z in (di,dj))
print("pair-normal-fp64=ok; independent normal model only, no friction")
