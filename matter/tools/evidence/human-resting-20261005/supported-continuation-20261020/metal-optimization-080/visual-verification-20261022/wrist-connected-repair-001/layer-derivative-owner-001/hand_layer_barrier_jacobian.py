"""Exact piecewise derivatives for native signed-prism quality constraints.
Preparation only. Coordinates are millimetres; ratios are scale independent.
DOF maps hold x/y/z column starts, or -1 for fixed vertices.
"""
import numpy as np

def prism_ratio_jacobian(outer, inner, faces, outer_dof, inner_dof, ndof, floor=1e-5):
    faces=np.asarray(faces,np.int64)
    sf=np.sort(faces,axis=1); a,b,c=sf.T
    odd=((faces[:,0]>faces[:,1]).astype(int)+(faces[:,0]>faces[:,2]).astype(int)+(faces[:,1]>faces[:,2]).astype(int))%2
    sign=np.column_stack((np.where(odd,1.,-1.),np.where(odd,-1.,1.),np.where(odd,1.,-1.)))
    points=np.stack((np.stack((outer[a],outer[b],outer[c],inner[c]),1),np.stack((outer[a],outer[b],inner[b],inner[c]),1),np.stack((outer[a],inner[a],inner[b],inner[c]),1)),1)
    maps=np.stack((np.column_stack((outer_dof[a],outer_dof[b],outer_dof[c],inner_dof[c])),np.column_stack((outer_dof[a],outer_dof[b],inner_dof[b],inner_dof[c])),np.column_stack((outer_dof[a],inner_dof[a],inner_dof[b],inner_dof[c]))),1)
    p=points.reshape(-1,4,3); maps=maps.reshape(-1,4); sign=sign.ravel(); n=len(p)
    u=p[:,1]-p[:,0];v=p[:,2]-p[:,0];w=p[:,3]-p[:,0]
    det=np.einsum('ij,ij->i',u,np.cross(v,w))
    grad=np.zeros_like(p);grad[:,1]=np.cross(v,w);grad[:,2]=np.cross(w,u);grad[:,3]=np.cross(u,v);grad[:,0]=-grad[:,1:].sum(1)
    pairs=np.array(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)))
    edges=p[:,pairs[:,0]]-p[:,pairs[:,1]];lens=np.einsum('nij,nij->ni',edges,edges)
    win=np.argmax(lens,axis=1);idx=np.arange(n);l2=lens[idx,win];safe=np.maximum(l2,1e-16)
    gl2=np.zeros_like(p);ev=edges[idx,win]*2;gl2[idx,pairs[win,0]]=ev;gl2[idx,pairs[win,1]]=-ev
    gl2[l2<1e-16]=0
    grad=sign[:,None,None]*(grad/safe[:,None,None]**1.5-1.5*det[:,None,None]*gl2/safe[:,None,None]**2.5)
    vals=sign*det/safe**1.5-floor
    jac=np.zeros((n,ndof));rr=np.arange(n)
    for corner in range(4):
        good=maps[:,corner]>=0
        for axis in range(3):
            np.add.at(jac,(rr[good],maps[good,corner]+axis),grad[good,corner,axis])
    return vals,jac

def width_constraints_jacobian(outer,inner,deep,rows,normal,slo,shi,flo,fhi,available,inner_dof,deep_dof,ndof):
    rows=np.asarray(rows,np.int64);o=outer[rows];i=inner[rows];d=deep[rows];N=normal[rows]
    si=o-i;fi=i-d;ti=o-d
    sn=np.linalg.norm(si,axis=1);fn=np.linalg.norm(fi,axis=1);tn=np.linalg.norm(ti,axis=1)
    su=si/np.maximum(sn[:,None],1e-30);fu=fi/np.maximum(fn[:,None],1e-30);tu=ti/np.maximum(tn[:,None],1e-30)
    vals=np.concatenate((sn-slo[rows],shi[rows]-sn,fn-flo[rows],fhi[rows]-fn,available[rows]-tn,np.einsum('ij,ij->i',si,N)-.05*sn,np.einsum('ij,ij->i',fi,N)-.05*fn))
    jac=np.zeros((7*len(rows),ndof));ii=inner_dof[rows];dd=deep_dof[rows];rr=np.arange(len(rows));m=len(rows)
    def add(block,cols,grad):
        good=cols>=0
        for axis in range(3):np.add.at(jac,(rr[good]+block*m,cols[good]+axis),grad[good,axis])
    add(0,ii,-su);add(1,ii,su);add(2,ii,fu);add(2,dd,-fu);add(3,ii,-fu);add(3,dd,fu);add(4,dd,tu);add(5,ii,-N+.05*su);add(6,ii,N-.05*fu);add(6,dd,-N+.05*fu)
    return vals,jac
