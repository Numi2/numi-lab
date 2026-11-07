#!/usr/bin/env python3
"""Independent FP64 priority-row examples; this is not a Metal or friction test."""
import math

def dot(a, b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def solve(normals, positions, rhs, limit):
    n=len(normals)
    moments=[cross(r,q) for r,q in zip(positions,normals)]
    w=[[dot(normals[i],normals[j])+10*dot(moments[i],moments[j]) for j in range(n)] for i in range(n)]
    impulse=[0.]*n
    trace=[]
    for _ in range(limit):
        g=[-rhs[i]+sum(w[i][j]*impulse[j] for j in range(n)) for i in range(n)]
        v=[abs(g[i]) if impulse[i]>1.e-12 else max(0.,-g[i]) for i in range(n)]
        k=max(range(n),key=lambda i:v[i])
        trace.append(v[k])
        if v[k]<1.e-8: break
        impulse[k]=max(0.,impulse[k]-g[k]/w[k][k])
    return trace,impulse

n=257
trace,j=solve([(0,1,0)]*n,[(.05*i,0,0) for i in range(n)],[.08175]*n,4)
assert trace == [.08175,0.] and abs(j[0]-.08175)<1e-15 and sum(j[1:])==0
trace,j=solve([(0,1,0)]*n,[(0,0,0)]*n,[1.]*n,4)
assert trace==[1.,0.] and sum(j)==1.
trace,j=solve([(1,0,0),(0,1,0),(0,0,1)],[(0,0,0)]*3,[.01]*3,5)
assert trace==[.01,.01,.01,0.]
trace,j=solve([(1,0,0),(.5,math.sqrt(.75),0)],[(0,0,0)]*2,[1.,1.],4)
assert trace==[1.,.5,.25,.125] # fixed pivot caps are not convergence guarantees
print("priority-normal-fp64=ok")
