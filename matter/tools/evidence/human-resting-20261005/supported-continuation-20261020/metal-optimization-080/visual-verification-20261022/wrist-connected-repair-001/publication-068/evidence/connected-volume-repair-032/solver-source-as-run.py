"""Connected static tissue-mesh authoring; never a per-step physics solver.

Sparse coupled clearance steps retain the original reference, all cell gates and
anchored outside vertices. Linearized feasibility is not anatomical admission.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

def cell_values_gradients(points, reference, tetrahedra):
    X=np.asarray(points,dtype=np.float64);R=np.asarray(reference,dtype=np.float64);T=np.asarray(tetrahedra)
    if X.ndim!=2 or X.shape[1]!=3 or R.shape!=X.shape or T.ndim!=2 or T.shape[1]!=4 or T.dtype.kind not in 'iu':
        raise ValueError('invalid point/reference/tetrahedron arrays')
    if not np.isfinite(X).all() or not np.isfinite(R).all() or T.min()<0 or T.max()>=len(X):raise ValueError('invalid finite/index domain')
    V=X[T];U=R[T]
    a,b,c=V[:,1]-V[:,0],V[:,2]-V[:,0],V[:,3]-V[:,0]
    det=np.einsum('ij,ij->i',a,np.cross(b,c))
    ar,br,cr=U[:,1]-U[:,0],U[:,2]-U[:,0],U[:,3]-U[:,0]
    den=np.einsum('ij,ij->i',ar,np.cross(br,cr))
    if np.any(den<=0):raise ValueError('reference cell is not positively oriented')
    g1,g2,g3=np.cross(b,c),np.cross(c,a),np.cross(a,b)
    G=np.stack((-(g1+g2+g3),g1,g2,g3),axis=1)
    pairs=np.asarray([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    edge=V[:,pairs[:,0]]-V[:,pairs[:,1]]
    lengths=np.sum(edge*edge,axis=2);which=np.argmax(lengths,axis=1)
    L=np.sqrt(np.maximum(lengths[np.arange(len(T)),which],1e-30))
    q=det/L**3;Q=G/L[:,None,None]**3
    d=edge[np.arange(len(T)),which];term=(-3*det/L**5)[:,None]*d
    np.add.at(Q,(np.arange(len(T)),pairs[which,0]),term)
    np.add.at(Q,(np.arange(len(T)),pairs[which,1]),-term)
    return det/den,G/den[:,None,None],q,Q

def sparse_cell_jacobian(gradients,tetrahedra,free_nodes,node_count):
    F=np.asarray(free_nodes,dtype=np.int64);T=np.asarray(tetrahedra,dtype=np.int64)
    if F.ndim!=1 or len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=node_count):raise ValueError('invalid free-node IDs')
    lookup=np.full(node_count,-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    cols=lookup[T];mask=cols>=0
    rr=np.broadcast_to(np.arange(len(T))[:,None,None],gradients.shape)
    cc=3*cols[:,:,None]+np.arange(3)[None,None,:]
    valid=np.broadcast_to(mask[:,:,None],gradients.shape)
    return sparse.coo_matrix((gradients[valid],(rr[valid],cc[valid])),shape=(len(T),3*len(F))).tocsr()

def coupled_clearance_step(points,reference,tetrahedra,free_nodes,contact_matrix,contact_targets,
                           *, step_limit_mm=2.,q_min=1e-5,J_bounds=(.35,2.5),time_limit_s=90.,
                           lower_linear_matrix=None,lower_linear_targets=None):
    """One bounded sparse LP and hard nonlinear Float32 backtracking transaction.

    Contact matrix maps absolute xyz metres to normal separation metres. Each row
    may address a vertex or a barycentric point on an edge/face. Slack is explicit
    and reported; the caller must refresh exact collision witnesses.
    """
    X=np.asarray(points);R=np.asarray(reference);T=np.asarray(tetrahedra,dtype=np.int64);F=np.asarray(free_nodes,dtype=np.int64)
    if X.dtype!=np.float32:raise ValueError('actual emitted Float32 input required')
    J,G,q,Q=cell_values_gradients(X,R,T)
    if np.min(q)<q_min or np.min(J)<J_bounds[0] or np.max(J)>J_bounds[1]:raise ValueError('starting mesh fails physical cell gates')
    if step_limit_mm<=0 or time_limit_s<=0:raise ValueError('invalid bound')
    C=sparse.csr_matrix(contact_matrix,dtype=float);target=np.asarray(contact_targets,float)
    if C.shape[1]!=3*len(X) or target.shape!=(C.shape[0],) or not np.isfinite(C.data).all() or not np.isfinite(target).all():raise ValueError('invalid contact rows')
    nd=3*len(F);nc=len(target);dcols=(3*F[:,None]+np.arange(3)).ravel()
    A=C[:,dcols];rhs=1000*(target-C@X.astype(float).ravel())
    active=np.unique(np.flatnonzero(np.isin(T,F).any(axis=1)))
    AJ=sparse_cell_jacobian(G[active],T[active],F,len(X))*.001
    AQ=sparse_cell_jacobian(Q[active],T[active],F,len(X))*.001
    z=lambda n:sparse.csr_matrix((n,nc+1))
    rows=[sparse.hstack([-A,-sparse.eye(nc),sparse.csr_matrix((nc,1))],format='csr')]
    bs=[-rhs]
    lo,hi=J_bounds
    # Leave interior buffer to make full nonlinear/F32 steps less fragile.
    jl=max(lo+.02,lo);jh=min(hi-.02,hi)
    rows += [sparse.hstack([AJ,z(len(active))]),sparse.hstack([-AJ,z(len(active))]),sparse.hstack([-AQ,z(len(active))])]
    bs += [jh-J[active],J[active]-jl,q[active]-q_min*1.02]
    I=sparse.eye(nd,format='csr')
    rows += [sparse.hstack([I,sparse.csr_matrix((nd,nc)),-np.ones((nd,1))]),
             sparse.hstack([-I,sparse.csr_matrix((nd,nc)),-np.ones((nd,1))])]
    bs += [np.zeros(nd),np.zeros(nd)]
    if lower_linear_matrix is not None:
        L=sparse.csr_matrix(lower_linear_matrix);lt=np.asarray(lower_linear_targets)
        if L.shape[1]!=3*len(X) or lt.shape!=(L.shape[0],):raise ValueError('invalid layer constraint')
        rows.append(sparse.hstack([-L[:,dcols],z(L.shape[0])]))
        bs.append(1000*(L@X.astype(float).ravel()-lt))
    objective=np.r_[np.zeros(nd),np.ones(nc),.01]
    result=linprog(objective,A_ub=sparse.vstack(rows,format='csr'),b_ub=np.concatenate(bs),
                   bounds=[(-step_limit_mm,step_limit_mm)]*nd+[(0,None)]*nc+[(0,step_limit_mm)],
                   method='highs',options={'time_limit':time_limit_s})
    report={'lp_success':bool(result.success),'lp_status':int(result.status),'lp_message':result.message,
            'free_nodes':len(F),'touched_cells':len(active),'contact_rows':nc,'starting_contact_deficit_l1_mm':float(np.maximum(rhs,0).sum()),
            'reference_unchanged':True,'geometry_admitted':False,'accepted_physical_steps':0}
    if not result.success:return X.copy(),report
    delta=result.x[:nd].reshape(-1,3)*.001
    original=np.maximum(target-C@X.astype(float).ravel(),0).sum()
    report['lp_predicted_contact_deficit_l1_mm']=float(result.x[nd:nd+nc].sum())
    best=X.copy()
    for k in range(15):
        scale=2.**-k;Y=X.copy();Y[F]=X[F].astype(float)+scale*delta
        jj,_,qq,_=cell_values_gradients(Y,R,T)
        linear_pass=True
        if lower_linear_matrix is not None:linear_pass=bool(np.min(L@Y.astype(float).ravel()-lt)>=-1e-9)
        deficit=np.maximum(target-C@Y.astype(float).ravel(),0).sum()
        if np.min(qq)>=q_min and np.min(jj)>=lo and np.max(jj)<=hi and linear_pass and deficit<original-1e-12:
            best=Y;report.update({'step_applied':True,'line_fraction':scale,'emitted_q_min':float(qq.min()),'emitted_J_min':float(jj.min()),'emitted_J_max':float(jj.max()),'contact_deficit_l1_mm':float(deficit*1000),'maximum_step_mm':float(np.linalg.norm((Y-X).astype(float),axis=1).max()*1000)})
            break
    else:report['step_applied']=False
    fixed=np.setdiff1d(np.arange(len(X)),F);assert np.array_equal(best[fixed].view(np.uint32),X[fixed].view(np.uint32))
    return best,report

