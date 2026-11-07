#!/usr/bin/env python3
import json, math, random
# Solve min .5*(t-p)^T H (t-p), subject to sum((t_i/L_i)^2)<=1.
# H is a symmetric positive-definite tangent response, L strictly positive.
def solve(H,p,L):
    A=H[0]*L[0]*L[0]; B=H[1]*L[0]*L[1]; C=H[2]*L[1]*L[1]
    g=(L[0]*(H[0]*p[0]+H[1]*p[1]),L[1]*(H[1]*p[0]+H[2]*p[1]))
    def u(tau):
        det=(A+tau)*(C+tau)-B*B
        return (((C+tau)*g[0]-B*g[1])/det,((A+tau)*g[1]-B*g[0])/det)
    if math.hypot(p[0]/L[0],p[1]/L[1])<=1:return p,0.
    lo=0.; hi=math.hypot(*g)
    for _ in range(120):
        mid=(lo+hi)/2; q=u(mid)
        if math.hypot(*q)>1:lo=mid
        else:hi=mid
    q=u(hi);return (L[0]*q[0],L[1]*q[1]),hi

def metrics(H,p,L,t,tau):
    q=(t[0]/L[0],t[1]/L[1]); d=(t[0]-p[0],t[1]-p[1]);
    k=(L[0]*(H[0]*d[0]+H[1]*d[1])+tau*q[0],L[1]*(H[1]*d[0]+H[2]*d[1])+tau*q[1])
    return math.hypot(*q), math.hypot(*k)
H=(1.,0.,1.); p=(-3.,-4.); L=(1.6,.4)
t,tau=solve(H,p,L);norm,kkt=metrics(H,p,L,t,tau); slip=(3+t[0],4+t[1]); scale=math.hypot(.8*slip[0],.2*slip[1]); oracle=(-2*.8**2*slip[0]/scale,-2*.2**2*slip[1]/scale)
assert math.hypot(t[0]-oracle[0],t[1]-oracle[1])<1e-12
rng=random.Random(7192);maxk=0.;maxbound=0.
for _ in range(1000):
    aa,bb,cc,dd=[rng.uniform(-2,2) for _ in range(4)]
    H=(aa*aa+cc*cc+.05,aa*bb+cc*dd,bb*bb+dd*dd+.05)
    p=(rng.uniform(-5,5),rng.uniform(-5,5)); L=(10**rng.uniform(-2,.3),10**rng.uniform(-2,.3))
    t,tau=solve(H,p,L);n,k=metrics(H,p,L,t,tau);maxk=max(maxk,k);maxbound=max(maxbound,n-1)
    assert n<=1+1e-12 and k<1e-10
print(json.dumps({'case':'positive anisotropic mu=(0.8,0.2), normal=2, incoming=(3,4), unit tangent operator','impulse':solve((1,0,1),(-3,-4),(1.6,.4))[0],'post_slip':slip,'max_dissipation_oracle':oracle,'kkt_residual':kkt,'norm':norm,'random_cases':1000,'seed':7192,'max_kkt_residual':maxk,'max_cone_excess':maxbound},indent=2))
