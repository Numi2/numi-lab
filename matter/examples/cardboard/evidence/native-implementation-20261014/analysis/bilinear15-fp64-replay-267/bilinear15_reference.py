"""Independent, batched FP64 15-state Hill/tensile-hardening equations.
No production compiler, bytecode, local projector, or derivative programs.
State order: ep(6), normalized second Piola(6), multiplier, MD/CD histories.
"""
import numpy as np

def stiffness(p):
    c=np.zeros(p.shape[:-1]+(6,6),dtype=p.dtype)
    c[...,0,0]=p[...,1];c[...,1,1]=p[...,2];c[...,2,2]=p[...,3]
    for i,j,k in [(0,1,4),(0,2,5),(1,2,6)]:c[...,i,j]=c[...,j,i]=p[...,k]
    c[...,3,3]=2*p[...,9];c[...,4,4]=2*p[...,8];c[...,5,5]=2*p[...,7]
    return c/p[...,10,None,None]

def strain(f):
    e=(np.einsum('...ki,...kj->...ij',f,f)-np.eye(3))/2
    return e[...,(0,1,2,1,0,0),(0,1,2,2,2,1)]

def hill(p,s):
    f,g,h,l,m,n=[p[...,i] for i in range(11,17)]
    a,b,c,d,e,v=np.moveaxis(s,-1,0)
    q=np.sqrt(f*(b-c)**2+g*(c-a)**2+h*(a-b)**2+2*l*d*d+2*m*e*e+2*n*v*v+p[...,25]**2)
    flow=np.stack([g*(a-c)+h*(a-b),f*(b-c)+h*(b-a),f*(c-b)+g*(c-a),l*d,m*e,n*v],axis=-1)/q[...,None]
    return q,flow

def residual(p,f,old,z):
    s=z[...,6:12];q,flow=hill(p,s);dl=z[...,12]-old[...,12]
    r=np.empty_like(z,dtype=np.result_type(p,f,old,z))
    r[...,:6]=z[...,:6]-old[...,:6]-dl[...,None]*flow
    r[...,6:12]=s-np.einsum('...ij,...j->...i',stiffness(p),strain(f)-z[...,:6])
    # The common nondimensionalization cancels algebraically. Evaluate the
    # physical positive-stress weights directly in FP64.
    pos=(s[...,:3]+np.sqrt(s[...,:3]**2+p[...,23,None]**2))/2
    w=pos**2/(np.sum(pos**2,axis=-1)+p[...,25]**2)[...,None]
    r[...,13:15]=z[...,13:15]-old[...,13:15]-dl[...,None]*w[...,:2]
    y=1+p[...,17]*p[...,19]*p[...,21]*z[...,13]/p[...,10]+p[...,18]*p[...,20]*p[...,22]*z[...,14]/p[...,10]-q
    root=np.sqrt(dl*dl+y*y+p[...,26]**2)
    # Rationalize FB near elastic or plastic complementarity to retain tiny
    # residuals. Complex-step derivatives use the same analytic expression.
    den=root+dl+y
    r[...,12]=np.where(np.real(den)>1e-14,(p[...,26]**2-2*dl*y)/den,root-dl-y)
    return r

def stress(p,f,z):
    s=p[...,10,None]*np.einsum('...ij,...j->...i',stiffness(p),strain(f)-z[...,:6])
    second=np.empty(f.shape,dtype=np.result_type(f,z))
    second[...,0,0]=s[...,0];second[...,1,1]=s[...,1];second[...,2,2]=s[...,2]
    second[...,1,2]=second[...,2,1]=s[...,3]
    second[...,0,2]=second[...,2,0]=s[...,4]
    second[...,0,1]=second[...,1,0]=s[...,5]
    return f@second

def jacobian(p,f,old,z):
    j=np.empty(z.shape+(15,),dtype=float)
    for col in range(15):
        q=z.astype(complex);q[...,col]+=1e-25j
        j[...,col]=np.imag(residual(p,f,old,q))/1e-25
    return j

def project(p,f,old,maxiter=80):
    z=old.copy();z[...,6:12]=np.einsum('...ij,...j->...i',stiffness(p),strain(f)-old[...,:6])
    counts=np.zeros(z.shape[:-1],dtype=int);scale=np.array([1e4]*6+[1]*6+[1e4]*3)
    for iteration in range(maxiter):
        r=residual(p,f,old,z);norm=np.max(np.abs(r)*scale,axis=-1); active=norm>2e-10
        if not active.any():return z,counts,norm
        ids=np.flatnonzero(active);pa=p[ids];fa=f[ids];oa=old[ids];za=z[ids]
        j=jacobian(pa,fa,oa,za);d=np.linalg.solve(j,-r[ids,...,None])[...,0]
        alpha=np.ones(len(ids));pending=np.ones(len(ids),dtype=bool);trial=za.copy()
        for bt in range(30):
            ti=za+alpha[:,None]*d;tn=np.max(np.abs(residual(pa,fa,oa,ti))*scale,axis=-1)
            ok=pending&np.isfinite(tn)&(tn<norm[ids]);trial[ok]=ti[ok];pending[ok]=False
            if not pending.any():break
            alpha[pending]*=.5
        if pending.any():raise RuntimeError(f'FP64 return map stalled iteration{iteration} rows{ids[pending][:10]} norm{norm[ids[pending]][:10]}')
        z[ids]=trial;counts[ids]+=1
    raise RuntimeError(f'FP64 projection iteration ceiling; worst{norm.max()}')

def tangent(p,f,old,z,h):
    rf=np.imag(residual(p,f.astype(complex)+1e-25j*h,old,z))/1e-25
    dz=np.linalg.solve(jacobian(p,f,old,z),-rf[...,None])[...,0]
    return np.imag(stress(p,f.astype(complex)+1e-25j*h,z.astype(complex)+1e-25j*dz))/1e-25
