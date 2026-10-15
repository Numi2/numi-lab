#!/usr/bin/env python3
"""Independent arithmetic check of the shared-face ownership regression fixture."""
from itertools import combinations
import math

upper = ((0.,0.,0.), (1.,0.,0.), (0.,1.,0.), (0.,0.,1.))
lower = ((0.,0.,0.), (1.,0.,0.), (0.,1.,0.), (0.,0.,-1.))
face_u = tuple(upper[i] for i in (0, 1, 2))
face_l = tuple(lower[i] for i in (0, 1, 2))
assert sorted(face_u) == sorted(face_l)

def det(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(sum(x*x for x in a))
for tet in (upper,lower):
    volume = abs(det(sub(tet[1],tet[0]),sub(tet[2],tet[0]),sub(tet[3],tet[0])))/6
    area = norm(cross(sub(tet[1],tet[0]),sub(tet[2],tet[0])))/2
    length = volume/area
    assert math.isclose(volume,1/6,rel_tol=0,abs_tol=1e-15)
    assert math.isclose(area,1/2,rel_tol=0,abs_tol=1e-15)
    assert math.isclose(length,1/3,rel_tol=0,abs_tol=1e-15)

Gc=100.0
per_element=Gc*0.5
assert per_element == 50.0
assert per_element+per_element == 2*Gc*0.5
assert per_element+per_element != Gc*0.5

# The admitted control plane x=1/4 cuts each tet internally; it is not the
# shared z=0 face. Its plane values on that shared triangle are not all zero.
shared_z = [p[2] for p in face_u]
interior_x = [p[0]-0.25 for p in face_u]
assert all(z == 0.0 for z in shared_z)
assert not all(x == 0.0 for x in interior_x)
print("fixture arithmetic passed: shared face area=0.5, V/A=1/3, per-element Gc*A=50 J, duplicated sum=100 J")
