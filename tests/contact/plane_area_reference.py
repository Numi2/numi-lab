#!/usr/bin/env python3
"""FP64 exact unilateral pressure-gap reference, no Numi solver."""
import math
WIDTH=1e-4
K=1e16                  # N/m^5, held fixed across integration and meshes
AREA=0.02**2            # m^2, reference bottom face
WEIGHT=.0088*9.81       # N
def pressure(g):
    d=WIDTH; o=g*g-d*d
    first=-K*(2*o*math.log(g*g/(d*d))+o*o/(g*g))
    return -2*g*first
def energy(g):
    d=WIDTH;o=g*g-d*d
    return -K*o*o*math.log(g*g/(d*d))
lo,hi=.1*WIDTH,WIDTH
for _ in range(100):
    mid=(lo+hi)/2
    if pressure(mid)>WEIGHT/AREA:lo=mid
    else:hi=mid
gap=(lo+hi)/2
epsilon=1e-9*WIDTH
finite_difference=-(energy(gap+epsilon)-energy(gap-epsilon))/(2*epsilon)
assert abs(finite_difference-pressure(gap))/pressure(gap)<1e-5
assert abs(AREA*pressure(gap)-WEIGHT)<1e-12
print("analytical uniform-gap law: k N/m^5",K,"area m^2",AREA,"load N",WEIGHT,"gap m",gap,"pressure Pa",pressure(gap))
print("This is a pressure-gap law check, not an independent FEM impact solver.")
