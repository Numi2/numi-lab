"""Independent exact-binary witness check; offline asset audit only."""
from fractions import Fraction
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def exact_point(p):
    return tuple(Fraction.from_float(float(x)) for x in p)
def classify_segment_triangle(a,b,triangle):
    a,b=map(exact_point,(a,b));v0,v1,v2=map(exact_point,triangle)
    d=sub(b,a);e1=sub(v1,v0);e2=sub(v2,v0);normal=cross(e1,e2)
    if not any(normal):return {"classification":"invalid_degenerate_triangle"}
    p=cross(d,e2);det=dot(e1,p);tv=sub(a,v0)
    if not det:
        return {"classification":"coplanar_requires_2d_test" if dot(normal,tv)==0 else "parallel_disjoint"}
    un=dot(tv,p);q=cross(tv,e1);vn=dot(d,q);tn=dot(e2,q)
    if det<0:det,un,vn,tn=(-det,-un,-vn,-tn)
    if un<0 or vn<0 or un+vn>det or tn<0 or tn>det:return {"classification":"disjoint"}
    boundary=un==0 or vn==0 or un+vn==det or tn==0 or tn==det
    return dict(classification="boundary_hit" if boundary else "proper_interior_hit",
                t=float(tn/det),barycentric=[float((det-un-vn)/det),float(un/det),float(vn/det)],
                arithmetic="exact rational signs over supplied binary floating-point endpoints/vertices")
def selftest():
    tri=((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
    assert classify_segment_triangle((.25,.25,-1),(.25,.25,1),tri)["classification"]=="proper_interior_hit"
    assert classify_segment_triangle((2.,2.,-1),(2.,2.,1),tri)["classification"]=="disjoint"
    assert classify_segment_triangle((0.,0.,-1),(0.,0.,1),tri)["classification"]=="boundary_hit"
    assert classify_segment_triangle((.25,.25,0),(.75,.25,0),tri)["classification"]=="coplanar_requires_2d_test"
    assert classify_segment_triangle((.25,.25,1),(.75,.25,1),tri)["classification"]=="parallel_disjoint"
    assert classify_segment_triangle((0,0,-1),(0,0,1),((0,0,0),(0,0,0),(1,0,0)))["classification"]=="invalid_degenerate_triangle"
    # Tiny translated triangle: classification uses signs, not a global epsilon.
    z=2.**-400;s=2.**-410;v=((z,z,z),(z+s,z,z),(z,z+s,z))
    assert classify_segment_triangle((z+s/4,z+s/4,z-s),(z+s/4,z+s/4,z+s),v)["classification"]=="proper_interior_hit"
    # Segment reversal must preserve hit and reverse its parameter.
    r1=classify_segment_triangle((.25,.25,-1),(.25,.25,3),tri)
    r2=classify_segment_triangle((.25,.25,3),(.25,.25,-1),tri)
    assert r1["t"]==.25 and r2["t"]==.75
    return {"status":"PASS","checks":8,"scope":"Offline exact witness signs only; no whole-scene clearance or physical admission."}
if __name__=="__main__":
    import json
    print(json.dumps(selftest()))

