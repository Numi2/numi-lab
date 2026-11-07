#!/usr/bin/env python3
"""Independent FP64 samples check the full-interval derivative criterion."""
import math
import random
import unittest

from contact_frame_diagnostic import rotate, dot


def mul(a, b):
    products = [x*y for x in a for y in b]
    return min(products), max(products)


def sub(a, b):
    return a[0]-b[1], a[1]-b[0]


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def hull(*a):
    return min(x[0] for x in a), max(x[1] for x in a)


def scale(a, factor):
    return mul(a, (factor, factor))


def polynomial(q0, q1, v, n, t):
    q = tuple((1-t)*x+t*y for x,y in zip(q0,q1))
    d = dot(q,q)
    return dot(rotate(q,v),n)*d, d


def coeff(q0,q1,v,n):
    a=polynomial(q0,q1,v,n,0)
    b=polynomial(q0,q1,v,n,.5)
    c=polynomial(q0,q1,v,n,1)
    return [[(a[k],a[k]),(2*b[k]-(a[k]+c[k])/2,)*2,(c[k],c[k])] for k in (0,1)]


def derivative_lower(r,d,center_delta,q0,q1):
    rr=hull(*r);dd=hull(*d)
    rp=hull(scale(sub(r[1],r[0]),2),scale(sub(r[2],r[1]),2))
    dp=hull(scale(sub(d[1],d[0]),2),scale(sub(d[2],d[1]),2))
    k=(dot(q0,q1),math.sqrt(dot(q0,q0)*dot(q1,q1)))
    return add(sub(mul(rp,dd),mul(rr,dp)),mul(mul((center_delta,)*2,k),dd))[0]


def physical_time(q0,q1,a):
    n=math.sqrt(dot(q0,q0)); c=dot(q0,q1)/n
    sine=math.sqrt(sum((q1[i]-q0[i]*c/n)**2 for i in range(4)))
    angle=math.atan2(sine,c)
    return math.atan2(a*sine,(1-a)*n+a*c)/angle if angle>1e-6 else a*c/((1-a)*n+a*c)


class MonotonicReferenceTests(unittest.TestCase):
    def test_bound_encloses_unused_rotations_and_corner_motion(self):
        rng=random.Random(738271)
        proven=0
        for case in range(1000):
            q0=[rng.uniform(-1,1) for _ in range(4)]
            q1=[rng.uniform(-1,1) for _ in range(4)]
            q0=[x/math.sqrt(dot(q0,q0)) for x in q0]
            q1=[x/math.sqrt(dot(q1,q1)) for x in q1]
            if dot(q0,q1)<0:q1=[-x for x in q1]
            v=[rng.uniform(-.3,.3) for _ in range(3)]
            n=[rng.uniform(-1,1) for _ in range(3)]
            n=[x/math.sqrt(dot(n,n)) for x in n]
            delta=rng.uniform(-2,2)
            r,d=coeff(q0,q1,v,n)
            if derivative_lower(r,d,delta,q0,q1)<=1e-10:
                continue
            proven+=1
            values=[]
            for i in range(201):
                a=i/200
                rq,dq=polynomial(q0,q1,v,n,a)
                values.append(rq/dq+delta*physical_time(q0,q1,a))
            self.assertTrue(all(values[i+1]>=values[i]-1e-12 for i in range(200)),
                            msg="invalid interval certificate")
        self.assertGreater(proven,30)


if __name__ == "__main__":
    unittest.main()
