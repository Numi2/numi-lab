"""Independent FP64 piecewise-quadratic translating sphere/AABB reference."""
import math
import numpy as np
HALF=np.array([.03,.025,.025],dtype=np.float32).astype(float)
RADIUS=float(np.float32(.008));TOL=float(np.float32(1e-5))
DT=float(np.float32(.000625))
def rotate(q,p):
    return p+2*np.cross(q[:3],np.cross(q[:3],p)+q[3]*p)
def local_segment(c,f):
    q=c[0,2].astype(float);q/=np.linalg.norm(q);q[:3]*=-1
    return rotate(q,c[1,1,:3].astype(float)-c[0,1,:3].astype(float)),rotate(q,(f[1,1,:3].astype(float)-f[0,1,:3].astype(float))-(c[1,1,:3].astype(float)-c[0,1,:3].astype(float)))
def cast(p,d,h=HALF,r=RADIUS,tol=TOL):
    p=np.array(p,dtype=float);d=np.array(d,dtype=float);h=np.array(h,dtype=float)
    cuts=[0.,1.]
    for i in range(3):
        if d[i]!=0:
            cuts.extend(t for side in (-h[i],h[i]) if 0<(t:=(side-p[i])/d[i])<1)
    cuts=sorted(set(cuts));first=None;minimum=float('inf');minimum_alpha=0.;expanded=r+tol
    for lo,hi in zip(cuts,cuts[1:]):
        mid=p+((lo+hi)/2)*d;at=p+lo*d
        mask=np.abs(mid)>h
        b=np.where(mask,at-np.sign(mid)*h,0.);v=np.where(mask,d,0.)
        aa=float(v@v);bb=float(b@v);cc=float(b@b)-expanded**2
        u=min(hi-lo,max(0.,-bb/aa)) if aa else 0.
        gap=float(np.linalg.norm(b+u*v))-r
        if gap<minimum:minimum=gap;minimum_alpha=lo+u
        hit=0. if cc<=0 else None
        disc=bb*bb-aa*cc
        if hit is None and aa>0 and bb<0 and disc>=0:
            hit=cc/(-bb+math.sqrt(disc))
        if hit is not None and 0<=hit<=hi-lo+1e-14:
            first=lo+hit if first is None else min(first,lo+hit)
    return {'first_alpha':first,'minimum_alpha':minimum_alpha,'minimum_unsigned_box_distance_minus_radius_m':minimum,
        'start_gap_m':float(np.linalg.norm(np.maximum(np.abs(p)-h,0.)))-r}
def calibrate():
    h=np.array([.03,.025,.025]);r=.008;tol=1e-5
    tests=[([.1,0,0],[-.2,0,0],(.1-h[0]-r-tol)/.2),
       (h+[.01,.01,0],[-.02,-.02,0],(.01-(r+tol)/math.sqrt(2))/.02),
       (h+[.01,.01,.01],[-.02,-.02,-.02],(.01-(r+tol)/math.sqrt(3))/.02)]
    errors=[abs(cast(p,d,h,r,tol)['first_alpha']-t) for p,d,t in tests]
    assert max(errors)<1e-14
    assert cast([h[0]+r+20e-6,0,0],[0,.01,0],h,r,tol)['first_alpha'] is None
    assert cast([h[0]+r-.001,0,0],[0,0,0],h,r,tol)['first_alpha']==0
    return {'passed':True,'known_first_contact_alpha_errors':errors,'null_and_initial_overlap_known_values':2,
       'scope':'Continuous FP64 ideal linear translation, normalized fixed box quaternion. Float32 native interpolation is compared with a declared 0.5 micrometre ambiguity band; no rotating-box reference.'}
