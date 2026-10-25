"""Connected static tissue-mesh authoring; never a per-step physics solver.

Sparse coupled clearance steps retain the original reference, all cell gates and
anchored outside vertices. Linearized feasibility is not anatomical admission.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

def cell_values_gradients(points, reference, tetrahedra):
    """Return J, dJ, signed det/longest-edge-cubed quality, and its gradient.

    The quality floor applies to this dimensionless signed-volume measure;
    mean-ratio, radius-edge and dihedral metrics are not interchangeable.
    Pass the emitted Float32 positions when checking emitted mesh acceptance.
    """
    rawX=np.asarray(points);rawR=np.asarray(reference);rawT=np.asarray(tetrahedra)
    if rawT.dtype.kind not in 'iu' or rawT.ndim!=2 or rawT.shape[1]!=4 or len(rawT)==0:
        raise ValueError('tetrahedra must be a nonempty integer Mx4 table')
    if rawX.dtype.kind not in 'fiu' or rawR.dtype.kind not in 'fiu':
        raise ValueError('points and reference must be numeric arrays')
    X=rawX.astype(np.float64,copy=False);R=rawR.astype(np.float64,copy=False);T=rawT.astype(np.int64,copy=False)
    if X.ndim!=2 or X.shape[1]!=3 or len(X)==0 or R.shape!=X.shape:
        raise ValueError('invalid point/reference array shapes')
    if not np.isfinite(X).all() or not np.isfinite(R).all() or np.any(T<0) or np.any(T>=len(X)):
        raise ValueError('invalid finite/index domain')
    if np.any(np.diff(np.sort(T,axis=1),axis=1)==0):raise ValueError('tetrahedra may not repeat node IDs')
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
    rawF=np.asarray(free_nodes);rawT=np.asarray(tetrahedra);G=np.asarray(gradients,dtype=np.float64)
    if rawF.dtype.kind not in 'iu' or rawF.ndim!=1 or len(rawF)==0:raise ValueError('free-node IDs must be a nonempty integer vector')
    if rawT.dtype.kind not in 'iu' or rawT.ndim!=2 or rawT.shape[1]!=4 or len(rawT)==0:raise ValueError('tetrahedra must be a nonempty integer Mx4 table')
    if not isinstance(node_count,(int,np.integer)) or isinstance(node_count,(bool,np.bool_)) or node_count<=0:raise ValueError('node_count must be a positive integer')
    F=rawF.astype(np.int64,copy=False);T=rawT.astype(np.int64,copy=False)
    if G.shape!=(len(T),4,3) or not np.isfinite(G).all():raise ValueError('gradients must be finite Mx4x3')
    if np.any(T<0) or np.any(T>=node_count) or np.any(np.diff(np.sort(T,axis=1),axis=1)==0):raise ValueError('invalid tetrahedron node IDs')
    if len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=node_count):raise ValueError('invalid free-node IDs')
    lookup=np.full(node_count,-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    cols=lookup[T];mask=cols>=0
    rr=np.broadcast_to(np.arange(len(T))[:,None,None],gradients.shape)
    cc=3*cols[:,:,None]+np.arange(3)[None,None,:]
    valid=np.broadcast_to(mask[:,:,None],gradients.shape)
    return sparse.coo_matrix((gradients[valid],(rr[valid],cc[valid])),shape=(len(T),3*len(F))).tocsr()

def boundary_face_values_gradients(points, origin, faces):
    """Signed projected double area, relative to the fixed authoring surface."""
    X=np.asarray(points,dtype=float);O=np.asarray(origin,dtype=float);raw=np.asarray(faces)
    if X.ndim!=2 or X.shape[1]!=3 or O.shape!=X.shape or not np.isfinite(X).all() or not np.isfinite(O).all():
        raise ValueError('invalid boundary point/origin coordinates')
    if raw.dtype.kind not in 'iu' or raw.ndim!=2 or raw.shape[1]!=3 or len(raw)==0:
        raise ValueError('boundary faces must be nonempty integer Mx3')
    T=raw.astype(np.int64,copy=False)
    if np.any(T<0) or np.any(T>=len(X)) or np.any(np.diff(np.sort(T,axis=1),axis=1)==0):
        raise ValueError('invalid boundary face indices')
    V=X[T];U=O[T]
    b=V[:,1]-V[:,0];c=V[:,2]-V[:,0]
    original=np.cross(U[:,1]-U[:,0],U[:,2]-U[:,0])
    den=np.einsum('ij,ij->i',original,original)
    if np.any(den<=0):raise ValueError('degenerate authoring boundary face')
    current=np.cross(b,c);n=original/den[:,None]
    value=np.einsum('ij,ij->i',current,n)
    g1=np.cross(c,n);g2=np.cross(n,b)
    G=np.stack((-(g1+g2),g1,g2),axis=1)
    ratio=np.linalg.norm(current,axis=1)/np.sqrt(den)
    cosine=value/np.maximum(ratio,np.finfo(float).tiny)
    return value,G,ratio,cosine

def boundary_area_cosine_values_gradients(points, origin, faces):
    """Return true area ratios/cosines and their per-vertex gradients.

    Gradients are with respect to the three current vertex coordinates in
    metres. The area ratio is ||cross|| / ||reference_cross||; cosine compares
    the current oriented area vector with the fixed construction-origin normal.
    """
    area, cosine = boundary_face_values_gradients(points, origin, faces)[2:]
    X=np.asarray(points,dtype=np.float64);O=np.asarray(origin,dtype=np.float64)
    F=np.asarray(faces,dtype=np.int64)
    V=X[F];U=O[F]
    e1=V[:,1]-V[:,0];e2=V[:,2]-V[:,0]
    cross=np.cross(e1,e2)
    ref_cross=np.cross(U[:,1]-U[:,0],U[:,2]-U[:,0])
    cross_norm=np.linalg.norm(cross,axis=1)
    ref_norm=np.linalg.norm(ref_cross,axis=1)
    if np.any(cross_norm<=np.finfo(np.float64).tiny) or np.any(ref_norm<=np.finfo(np.float64).tiny):
        raise ValueError('degenerate current or origin boundary face')
    unit_ref=ref_cross/ref_norm[:,None]
    unit_current=cross/cross_norm[:,None]
    grad_cross_area=cross/(cross_norm*ref_norm)[:,None]
    grad_cross_cos=(unit_ref-cosine[:,None]*unit_current)/cross_norm[:,None]

    def vertex_gradients(gc):
        g1=np.cross(e2,gc)
        g2=np.cross(gc,e1)
        return np.stack((-(g1+g2),g1,g2),axis=1)

    area_grad=vertex_gradients(grad_cross_area)
    cosine_grad=vertex_gradients(grad_cross_cos)
    if not np.isfinite(area_grad).all() or not np.isfinite(cosine_grad).all():
        raise ValueError('nonfinite boundary shape derivative')
    return area,area_grad,cosine,cosine_grad

def sparse_face_jacobian(gradients,faces,free_nodes,node_count):
    F=np.asarray(free_nodes,dtype=np.int64);T=np.asarray(faces,dtype=np.int64)
    lookup=np.full(node_count,-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    cols=lookup[T];valid=np.broadcast_to((cols>=0)[:,:,None],gradients.shape)
    rr=np.broadcast_to(np.arange(len(T))[:,None,None],gradients.shape)
    cc=3*cols[:,:,None]+np.arange(3)[None,None,:]
    return sparse.coo_matrix((gradients[valid],(rr[valid],cc[valid])),shape=(len(T),3*len(F))).tocsr()

def coupled_clearance_step(points,reference,tetrahedra,free_nodes,contact_matrix,contact_targets,
                           *, step_limit_mm=2.,q_min=1e-5,J_bounds=(.35,2.5),time_limit_s=90.,
                           lower_linear_matrix=None,lower_linear_targets=None,max_incremental_gradient=.12,
                           construction_origin=None,cumulative_displacement_weight=.01,
                           boundary_faces=None,boundary_area_bounds=(.25,4.),boundary_normal_cosine_min=.25,linear_method="highs",embedded_surface=None):
    """One bounded sparse LP and hard nonlinear Float32 backtracking transaction.

    Contact matrix maps absolute xyz metres to normal separation metres. Each row
    may address a vertex or a barycentric point on an edge/face. Slack is explicit
    and reported; the caller must refresh exact collision witnesses.
    """
    if linear_method not in ("highs","highs-ipm","highs-ds"):raise ValueError("unsupported sparse linear method")
    X=np.asarray(points);R=np.asarray(reference);rawT=np.asarray(tetrahedra);rawF=np.asarray(free_nodes)
    if X.dtype!=np.float32:raise ValueError('actual emitted Float32 input required')
    if rawT.dtype.kind not in 'iu' or rawT.ndim!=2 or rawT.shape[1]!=4 or len(rawT)==0:raise ValueError('tetrahedra must be a nonempty integer Mx4 table')
    if rawF.dtype.kind not in 'iu' or rawF.ndim!=1 or len(rawF)==0:raise ValueError('free_nodes must be a nonempty integer vector')
    T=rawT.astype(np.int64,copy=False);F=rawF.astype(np.int64,copy=False)
    if R.dtype.kind not in 'fiu' or not np.isfinite(R).all():raise ValueError('reference coordinates must be finite numeric values')
    if len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=len(X)):raise ValueError('invalid free-node IDs')
    try: lo,hi=map(float,J_bounds);q_min=float(q_min);step_limit_mm=float(step_limit_mm);time_limit_s=float(time_limit_s)
    except Exception as exc:raise ValueError('J/q/step/time bounds must be numeric') from exc
    if not np.isfinite([lo,hi,q_min,step_limit_mm,time_limit_s]).all() or lo<=0 or hi<=lo or q_min<=0 or step_limit_mm<=0 or time_limit_s<=0:raise ValueError('invalid finite J/q/step/time bounds')
    if max_incremental_gradient is not None:
        try:max_incremental_gradient=float(max_incremental_gradient)
        except Exception as exc:raise ValueError('displacement-gradient bound must be numeric or None') from exc
        if not np.isfinite(max_incremental_gradient) or max_incremental_gradient<=0:raise ValueError('invalid finite displacement-gradient bound')
    J,G,q,Q=cell_values_gradients(X,R,T)
    if np.min(q)<q_min or np.min(J)<lo or np.max(J)>hi:raise ValueError('starting mesh fails physical cell gates')
    C=sparse.csr_matrix(contact_matrix,dtype=float);target=np.asarray(contact_targets,float)
    if C.shape[0]==0 or C.nnz==0 or np.any(np.diff(C.indptr)==0) or C.shape[1]!=3*len(X) or target.shape!=(C.shape[0],) or not np.isfinite(C.data).all() or not np.isfinite(target).all():raise ValueError('invalid or empty contact rows')
    O=None if construction_origin is None else np.asarray(construction_origin,dtype=float)
    if O is not None and (O.shape!=X.shape or not np.isfinite(O).all()):
        raise ValueError('construction origin must match finite point coordinates')
    cumulative_displacement_weight=float(cumulative_displacement_weight)
    if not np.isfinite(cumulative_displacement_weight) or cumulative_displacement_weight<=0:
        raise ValueError('cumulative displacement weight must be positive and finite')
    BF=None
    if boundary_faces is not None:
        if O is None:raise ValueError('boundary gates require a fixed construction origin')
        amin,amax=map(float,boundary_area_bounds);cmin=float(boundary_normal_cosine_min)
        if not np.isfinite([amin,amax,cmin]).all() or amin<=0 or amax<=amin or cmin<=0 or cmin>1:
            raise ValueError('invalid boundary area/normal bounds')
        BF=np.asarray(boundary_faces,dtype=np.int64)
        ar,area_grad,ac,cosine_grad=boundary_area_cosine_values_gradients(X,O,BF)
        if np.min(ar)<amin or np.max(ar)>amax or np.min(ac)<cmin:
            raise ValueError('starting boundary fails fixed-origin area/normal gates')
    if embedded_surface is not None:
        if O is None or embedded_surface.coarse_origin.shape!=X.shape or not np.array_equal(embedded_surface.coarse_origin,O.astype(np.float32)):
            raise ValueError('mapped skin must use the identical fixed construction origin')
        if not embedded_surface.passes(X):raise ValueError('starting mesh fails mapped detailed-skin gates')
    nd=3*len(F);nc=len(target);dcols=(3*F[:,None]+np.arange(3)).ravel()
    A=C[:,dcols];rhs=1000*(target-C@X.astype(float).ravel())
    active=np.unique(np.flatnonzero(np.isin(T,F).any(axis=1)))
    AJ=sparse_cell_jacobian(G[active],T[active],F,len(X))*.001
    AQ=sparse_cell_jacobian(Q[active],T[active],F,len(X))*.001
    z=lambda n:sparse.csr_matrix((n,nc+1+nd))
    rows=[sparse.hstack([-A,-sparse.eye(nc),sparse.csr_matrix((nc,1+nd))],format='csr')]
    bs=[-rhs]
    lo,hi=J_bounds
    # Leave interior buffer to make full nonlinear/F32 steps less fragile.
    jl=max(lo+.02,lo);jh=min(hi-.02,hi)
    rows += [sparse.hstack([AJ,z(len(active))]),sparse.hstack([-AJ,z(len(active))]),sparse.hstack([-AQ,z(len(active))])]
    bs += [jh-J[active],J[active]-jl,q[active]-q_min*1.02]
    I=sparse.eye(nd,format='csr')
    rows += [sparse.hstack([I,sparse.csr_matrix((nd,nc)),-np.ones((nd,1)),sparse.csr_matrix((nd,nd))]),
             sparse.hstack([-I,sparse.csr_matrix((nd,nc)),-np.ones((nd,1)),sparse.csr_matrix((nd,nd))])]
    bs += [np.zeros(nd),np.zeros(nd)]
    if lower_linear_matrix is not None:
        L=sparse.csr_matrix(lower_linear_matrix);lt=np.asarray(lower_linear_targets,dtype=np.float64)
        if L.shape[0]==0 or L.shape[1]!=3*len(X) or lt.shape!=(L.shape[0],) or np.any(np.diff(L.indptr)==0) or not np.isfinite(L.data).all() or not np.isfinite(lt).all():raise ValueError('invalid or empty layer constraint')
        rows.append(sparse.hstack([-L[:,dcols],z(L.shape[0])]))
        bs.append(1000*(L@X.astype(float).ravel()-lt))
    rows += [sparse.hstack([I,sparse.csr_matrix((nd,nc+1)),-I]),sparse.hstack([-I,sparse.csr_matrix((nd,nc+1)),-I])]
    # The same auxiliary variables penalize displacement from one immutable
    # construction origin, rather than repeatedly rewarding a small new step.
    anchor_offset=np.zeros(nd) if O is None else 1000*(X[F].astype(float)-O[F]).ravel()
    bs += [-anchor_offset,anchor_offset]
    if BF is not None:
        # Linearize the actual norm-based area ratio and normal cosine. These
        # are hard LP rows; the unchanged nonlinear Float32 gates below remain
        # authoritative after each backtracked trial.
        AA=sparse_face_jacobian(area_grad,BF,F,len(X))*.001
        AC=sparse_face_jacobian(cosine_grad,BF,F,len(X))*.001
        rows += [sparse.hstack([-AA,z(len(BF))]),sparse.hstack([AA,z(len(BF))]),
                 sparse.hstack([-AC,z(len(BF))])]
        bs += [ar-amin,amax-ar,ac-cmin]
    if embedded_surface is not None:
        ev,EJ=embedded_surface.projected_area_linearization(X,F)
        emin,emax=embedded_surface.area_bounds
        EJ=.001*EJ
        rows += [sparse.hstack([-EJ,z(len(ev))]),sparse.hstack([EJ,z(len(ev))])]
        bs += [ev-emin,emax-ev]
    # Limit the actual P1 displacement gradient, instead of letting slender
    # cells force a tiny whole-body line search after unrelated nodal jumps.
    if max_incremental_gradient is not None:
        if max_incremental_gradient<=0:raise ValueError('invalid displacement-gradient bound')
        U=R.astype(float)[T[active]]
        dm=np.stack([U[:,k]-U[:,0]for k in [1,2,3]],axis=2)
        inverse=np.linalg.inv(dm)
        shape_grad=np.concatenate((-inverse.sum(axis=1)[:,None,:],inverse),axis=1)
        lookup=np.full(len(X),-1,dtype=np.int64);lookup[F]=np.arange(len(F))
        columns=lookup[T[active]];rr=[];cc=[];vv=[]
        for a in range(3):
            for b in range(3):
                for k in range(4):
                    ok=columns[:,k]>=0;cells=np.flatnonzero(ok)
                    rr.extend((9*cells+3*a+b).tolist())
                    cc.extend((3*columns[ok,k]+a).tolist())
                    vv.extend((.001*shape_grad[ok,k,b]).tolist())
        DG=sparse.coo_matrix((vv,(rr,cc)),shape=(9*len(active),nd)).tocsr()
        rows += [sparse.hstack([DG,z(DG.shape[0])]),sparse.hstack([-DG,z(DG.shape[0])])]
        bs += [np.full(DG.shape[0],max_incremental_gradient),np.full(DG.shape[0],max_incremental_gradient)]
    objective=np.r_[np.zeros(nd),np.ones(nc),.01,np.full(nd,cumulative_displacement_weight)]
    result=linprog(objective,A_ub=sparse.vstack(rows,format='csr'),b_ub=np.concatenate(bs),
                   bounds=[(-step_limit_mm,step_limit_mm)]*nd+[(0,None)]*nc+[(0,step_limit_mm)]+[(0,None if O is not None else step_limit_mm)]*nd,
                   method=linear_method,options={'time_limit':time_limit_s})
    report={'linear_method':linear_method,'lp_success':bool(result.success),'lp_status':int(result.status),'lp_message':result.message,
            'free_nodes':len(F),'touched_cells':len(active),'contact_rows':nc,'starting_contact_deficit_l1_mm':float(np.maximum(rhs,0).sum()),
            'reference_unchanged':True,'geometry_admitted':False,'accepted_physical_steps':0}
    if embedded_surface is not None:report.update(embedded_surface.report(X))
    if not result.success:return X.copy(),report
    delta=result.x[:nd].reshape(-1,3)*.001
    original=np.maximum(target-C@X.astype(float).ravel(),0).sum()
    report['lp_predicted_contact_deficit_l1_mm']=float(result.x[nd:nd+nc].sum())
    if BF is not None:
        report.update({
            'starting_boundary_area_ratio_min':float(ar.min()),
            'starting_boundary_area_ratio_max':float(ar.max()),
            'starting_boundary_normal_cosine_min':float(ac.min()),
            'lp_linearized_boundary_area_ratio_min':float(np.min(ar+AA@result.x[:nd])),
            'lp_linearized_boundary_area_ratio_max':float(np.max(ar+AA@result.x[:nd])),
            'lp_linearized_boundary_normal_cosine_min':float(np.min(ac+AC@result.x[:nd])),
        })
    # Cells, collision rows and layer constraints induce exact connected solve
    # components. Independent components may take different line-search lengths;
    # no cell or constraint is allowed to straddle two components.
    lookup=np.full(len(X),-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    er=[];ec=[]
    for nodes in T:
        ix=lookup[nodes];ix=ix[ix>=0]
        for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    for B in [C]+([L]if lower_linear_matrix is not None else []):
        for row in range(B.shape[0]):
            ix=np.unique(lookup[B.indices[B.indptr[row]:B.indptr[row+1]]//3]);ix=ix[ix>=0]
            for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    if BF is not None:
        for face in BF:
            ix=lookup[face];ix=ix[ix>=0]
            for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    if embedded_surface is not None:
        S=embedded_surface.support
        for row in range(S.shape[0]):
            ix=lookup[S.indices[S.indptr[row]:S.indptr[row+1]]]
            ix=ix[ix>=0]
            for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    graph=sparse.coo_matrix((np.ones(len(er)),(er,ec)),shape=(len(F),len(F))).tocsr()
    count,labels=connected_components(graph,directed=False)
    best=X.copy();components=[]
    for label in range(count):
        local=np.flatnonzero(labels==label);nodes=F[local]
        original=np.maximum(target-C@best.astype(float).ravel(),0).sum()
        accepted=False;attempts=[]
        for k in range(15):
            scale=2.**-k;Y=best.copy();Y[nodes]=X[nodes].astype(float)+scale*delta[local]
            jj,_,qq,_=cell_values_gradients(Y,R,T)
            linear_pass=True
            if lower_linear_matrix is not None:linear_pass=bool(np.min(L@Y.astype(float).ravel()-lt)>=-1e-9)
            surface_pass=True
            trial_area_min=trial_area_max=trial_cosine_min=None
            if BF is not None:
                _,_,area,cosine=boundary_face_values_gradients(Y,O,BF)
                trial_area_min=float(np.min(area));trial_area_max=float(np.max(area))
                trial_cosine_min=float(np.min(cosine))
                surface_pass=bool(trial_area_min>=amin and trial_area_max<=amax and trial_cosine_min>=cmin)
            deficit=np.maximum(target-C@Y.astype(float).ravel(),0).sum()
            first_failed = (
                'cell_quality' if not np.min(qq)>=q_min else
                'cell_J' if not (np.min(jj)>=lo and np.max(jj)<=hi) else
                'hard_linear' if not linear_pass else
                'coarse_surface' if not surface_pass else
                'mapped_skin' if embedded_surface is not None and not embedded_surface.passes(Y) else
                'contact_progress' if not deficit<original-1e-12 else None)
            attempts.append({'fraction':float(scale),'first_failed_gate':first_failed,
                'q_min':float(qq.min()),'q_witness_row':int(qq.argmin()),
                'J_min':float(jj.min()),'J_max':float(jj.max()),
                'hard_linear_min_slack_m':None if lower_linear_matrix is None else float(np.min(L@Y.astype(float).ravel()-lt)),
                'hard_linear_witness_row':None if lower_linear_matrix is None else int(np.argmin(L@Y.astype(float).ravel()-lt)),
                'contact_deficit_l1_mm':float(deficit*1000),
                'boundary_area_ratio_min':trial_area_min,
                'boundary_area_ratio_max':trial_area_max,
                'boundary_normal_cosine_min':trial_cosine_min})
            if first_failed is None:
                best=Y;accepted=True;break
        components.append({'nodes':len(nodes),'applied':accepted,'line_fraction':scale if accepted else 0.,'attempts':attempts})
    jj,_,qq,_=cell_values_gradients(best,R,T)
    report.update({'step_applied':bool(np.any(best.view(np.uint32)!=X.view(np.uint32))),'components':components,
        'emitted_q_min':float(qq.min()),'emitted_J_min':float(jj.min()),'emitted_J_max':float(jj.max()),
        'contact_deficit_l1_mm':float(np.maximum(target-C@best.astype(float).ravel(),0).sum()*1000),
        'maximum_step_mm':float(np.linalg.norm((best-X).astype(float),axis=1).max()*1000)})
    if O is not None:
        report['maximum_cumulative_displacement_mm']=float(np.linalg.norm(best.astype(float)-O,axis=1).max()*1000)
        report['cumulative_displacement_l1_mm']=float(np.abs(best.astype(float)-O).sum()*1000)
    if BF is not None:
        _,_,area,cosine=boundary_face_values_gradients(best,O,BF)
        report.update({'boundary_faces':len(BF),'boundary_area_ratio_min':float(area.min()),
                       'boundary_area_ratio_max':float(area.max()),'boundary_normal_cosine_min':float(cosine.min())})
    if embedded_surface is not None:report.update(embedded_surface.report(best))
    fixed=np.setdiff1d(np.arange(len(X)),F);assert np.array_equal(best[fixed].view(np.uint32),X[fixed].view(np.uint32))
    return best,report


def sparse_displacement_gradient(reference, tetrahedra, free_nodes):
    """P1 gradient of an xyz displacement in millimetres, in reference SI units."""
    R=np.asarray(reference,dtype=float);T=np.asarray(tetrahedra,dtype=np.int64);F=np.asarray(free_nodes,dtype=np.int64)
    U=R[T];dm=np.stack([U[:,k]-U[:,0]for k in [1,2,3]],axis=2)
    inverse=np.linalg.inv(dm)
    shape=np.concatenate((-inverse.sum(axis=1)[:,None,:],inverse),axis=1)
    lookup=np.full(len(R),-1,dtype=np.int64);lookup[F]=np.arange(len(F));cols=lookup[T]
    # Axes are cell, displacement component, derivative component, vertex.
    rr=np.broadcast_to((9*np.arange(len(T))[:,None,None,None]+
                        3*np.arange(3)[None,:,None,None]+np.arange(3)[None,None,:,None]),(len(T),3,3,4))
    cc=np.broadcast_to(3*cols[:,None,None,:]+np.arange(3)[None,:,None,None],rr.shape)
    vv=np.broadcast_to(.001*shape.transpose(0,2,1)[:,None,:,:],rr.shape)
    valid=np.broadcast_to(cols[:,None,None,:]>=0,rr.shape)
    return sparse.coo_matrix((vv[valid],(rr[valid],cc[valid])),shape=(9*len(T),3*len(F))).tocsr()

def coupled_clearance_least_squares_step(points,reference,tetrahedra,free_nodes,contact_matrix,contact_targets,
        *,construction_origin,boundary_faces,lower_linear_matrix=None,lower_linear_targets=None,
        step_limit_mm=2.,q_min=1e-5,J_bounds=(.35,2.5),boundary_area_bounds=(.5,2.),
        boundary_normal_cosine_min=.5,cumulative_displacement_weight=.002,
        gradient_penalty_mm=2.,max_incremental_gradient=.3,iteration_limit=400,active_set_passes=0,active_set_penalty=100.,embedded_surface=None):
    """Bounded sparse authoring direction with exact emitted-geometry admission.

    This is a least-squares direction, not a linear feasibility certificate.
    Current reference/material state is never changed. Cell, boundary and layer
    constraints and any supplied detailed-skin map are mandatory nonlinear
    backtracking gates; failed directions
    leave all coordinates unchanged. No simulation state is advanced.
    """
    from scipy.sparse.linalg import lsmr
    X=np.asarray(points);R=np.asarray(reference);rawT=np.asarray(tetrahedra);rawF=np.asarray(free_nodes)
    if X.dtype!=np.float32:raise ValueError('actual emitted Float32 input required')
    if rawT.dtype.kind not in 'iu' or rawF.dtype.kind not in 'iu' or rawF.ndim!=1 or len(rawF)==0:
        raise ValueError('integer tetrahedra/free-node domain required')
    T=rawT.astype(np.int64,copy=False);F=rawF.astype(np.int64,copy=False)
    if len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=len(X)):raise ValueError('invalid free-node IDs')
    J,G,quality,Q=cell_values_gradients(X,R,T)
    O=np.asarray(construction_origin,dtype=float)
    if O.shape!=X.shape or not np.isfinite(O).all():raise ValueError('invalid construction origin')
    lo,hi=map(float,J_bounds);amin,amax=map(float,boundary_area_bounds);cmin=float(boundary_normal_cosine_min)
    vals=[lo,hi,amin,amax,cmin,step_limit_mm,q_min,cumulative_displacement_weight,gradient_penalty_mm,max_incremental_gradient]
    if not np.isfinite(vals).all() or lo<=0 or hi<=lo or amin<=0 or amax<=amin or not 0<cmin<=1 or min(vals[5:])<=0:
        raise ValueError('invalid finite clearance bounds')
    if not isinstance(iteration_limit,(int,np.integer)) or isinstance(iteration_limit,(bool,np.bool_)) or iteration_limit<=0:
        raise ValueError('invalid least-squares iteration limit')
    if not isinstance(active_set_passes,(int,np.integer)) or isinstance(active_set_passes,(bool,np.bool_)) or active_set_passes<0 or active_set_passes>12:
        raise ValueError('active-set pass budget must be an integer in [0,12]')
    if not np.isfinite(active_set_penalty) or active_set_penalty<=0:raise ValueError('invalid active-set penalty')
    _,_,area,cosine=boundary_face_values_gradients(X,O,boundary_faces);BF=np.asarray(boundary_faces,dtype=np.int64)
    if J.min()<lo or J.max()>hi or quality.min()<q_min or area.min()<amin or area.max()>amax or cosine.min()<cmin:
        raise ValueError('starting mesh fails exact cell/boundary gates')
    C=sparse.csr_matrix(contact_matrix,dtype=float);target=np.asarray(contact_targets,dtype=float)
    if C.shape[1]!=3*len(X) or not len(target) or target.shape!=(C.shape[0],) or np.any(np.diff(C.indptr)==0) or not np.isfinite(C.data).all() or not np.isfinite(target).all():
        raise ValueError('invalid contact rows')
    L=None
    if lower_linear_matrix is not None:
        L=sparse.csr_matrix(lower_linear_matrix,dtype=float);lt=np.asarray(lower_linear_targets,dtype=float)
        if L.shape[1]!=3*len(X) or L.shape[0]==0 or lt.shape!=(L.shape[0],) or np.any(np.diff(L.indptr)==0) or not np.isfinite(L.data).all() or not np.isfinite(lt).all():
            raise ValueError('invalid layer rows')
        if np.min(L@X.astype(float).ravel()-lt)<-1e-9:raise ValueError('starting mesh fails layer gates')
    if embedded_surface is not None:
        if embedded_surface.coarse_origin.shape!=X.shape or not np.array_equal(embedded_surface.coarse_origin,O.astype(np.float32)):
            raise ValueError('mapped skin must use the identical fixed construction origin')
        if not embedded_surface.passes(X):raise ValueError('starting mesh fails mapped detailed-skin gates')
    rhs=1000*(target-C@X.astype(float).ravel());hot=rhs>1e-8
    report={'algorithm':'sparse_regularized_least_squares_exact_geometry_backtracking','free_nodes':len(F),
            'contact_rows':len(rhs),'active_contact_rows':int(hot.sum()),'starting_contact_deficit_l1_mm':float(np.maximum(rhs,0).sum()),
            'accepted_physical_steps':0,'geometry_admitted':False,'reference_unchanged':True}
    if embedded_surface is not None:report.update(embedded_surface.report(X))
    if not np.any(hot):report['step_applied']=False;return X.copy(),report
    dcols=(3*F[:,None]+np.arange(3)).ravel()
    active=np.flatnonzero(np.isin(T,F).any(axis=1))
    DG=sparse_displacement_gradient(R,T[active],F)
    ridge=float(cumulative_displacement_weight)**.5;offset=1000*(X[F].astype(float)-O[F]).ravel()
    A=sparse.vstack([C[hot][:,dcols],ridge*sparse.eye(3*len(F)),float(gradient_penalty_mm)*DG],format='csr')
    b=np.r_[rhs[hot],-ridge*offset,np.zeros(DG.shape[0])]
    scale=1/np.sqrt(np.maximum(np.asarray(A.power(2).sum(axis=0)).ravel(),1e-30))
    solve=lsmr(A@sparse.diags(scale),b,atol=1e-6,btol=1e-6,maxiter=int(iteration_limit))
    delta=(scale*solve[0]).reshape(-1,3)
    active_history=[]
    if active_set_passes:
        # Fit all near-active connected constraints into the same sparse solve.
        # This avoids scaling the whole body because one node exceeded a local
        # step/gradient bound. Exact nonlinear gates below remain authoritative.
        I=sparse.eye(3*len(F),format='csr')
        AJ=sparse_cell_jacobian(G[active],T[active],F,len(X))*.001
        AQ=sparse_cell_jacobian(Q[active],T[active],F,len(X))*.001
        av,ag,_,_=boundary_face_values_gradients(X,O,BF)
        AF=sparse_face_jacobian(ag,BF,F,len(X))*.001
        H=sparse.vstack([I,-I,DG,-DG,AJ,-AJ,-AQ,AF,-AF]+([]if L is None else[-L[:,dcols]]),format='csr')
        h=np.r_[np.full(I.shape[0],step_limit_mm),np.full(I.shape[0],step_limit_mm),
            np.full(DG.shape[0],max_incremental_gradient),np.full(DG.shape[0],max_incremental_gradient),
            hi-J[active],J[active]-lo,quality[active]-q_min,amax-av,av-amin,
            []if L is None else 1000*(L@X.astype(float).ravel()-lt)]
        if embedded_surface is not None:
            ev,EJ=embedded_surface.projected_area_linearization(X,F)
            emin,emax=embedded_surface.area_bounds
            H=sparse.vstack([H,.001*EJ,-.001*EJ],format='csr')
            h=np.r_[h,emax-ev,ev-emin]
        norm=np.sqrt(np.asarray(H.power(2).sum(axis=1)).ravel())
        keep=norm>1e-20
        if np.any((~keep)&(h<-1e-9)):raise ValueError('fixed authoring constraints are infeasible')
        H=sparse.diags(1/norm[keep])@H[keep];h=h[keep]/norm[keep]
        def merit(v):
            e=A@v-b;violation=np.maximum(H@v-h,0)
            return .5*float(e@e)+.5*float(active_set_penalty)**2*float(violation@violation)
        for apass in range(active_set_passes):
            current=delta.ravel();violation=np.asarray(H@current-h).ravel()
            selected=violation>1e-7
            if not selected.any():break
            AA=sparse.vstack([A,float(active_set_penalty)*H[selected]],format='csr')
            bb=np.r_[b,float(active_set_penalty)*h[selected]]
            col=1/np.sqrt(np.maximum(np.asarray(AA.power(2).sum(axis=0)).ravel(),1e-30))
            solve2=lsmr(AA@sparse.diags(col),bb,atol=1e-6,btol=1e-6,maxiter=int(iteration_limit))
            proposal=col*solve2[0]
            gradient=A.T@(A@current-b)+float(active_set_penalty)**2*(H[selected].T@violation[selected])
            direction=proposal-current;gd=float(gradient@direction)
            if not np.isfinite(gd) or gd>=0:
                direction=-(col*col)*gradient;gd=float(gradient@direction)
            before=merit(current);fraction=1.;accepted=False
            for attempt in range(20):
                candidate=current+fraction*direction;after=merit(candidate)
                if np.isfinite(after) and after<=before+1e-4*fraction*gd:
                    delta=candidate.reshape(-1,3);accepted=True;break
                fraction*=.5
            active_history.append({'pass':apass,'selected_rows':int(selected.sum()),
                'max_normalized_violation_before_mm':float(violation.max()),
                'max_normalized_violation_after_mm':float(np.max(H@delta.ravel()-h)),
                'merit_before':before,'merit_after':merit(delta.ravel()),
                'merit_line_fraction':fraction if accepted else 0.,
                'linear_stop':int(solve2[1]),'linear_iterations':int(solve2[2])})
            if not accepted:break
    report['active_set_history']=active_history
    if not np.isfinite(delta).all():raise ValueError('nonfinite sparse authoring direction')
    report.update({'linear_stop':int(solve[1]),'linear_iterations':int(solve[2]),'linear_residual_norm':float(solve[3]),
                   'estimated_condition':float(solve[6]),'touched_cells':len(active)})
    lookup=np.full(len(X),-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    er=[];ec=[]
    for nodes in T:
        ix=lookup[nodes];ix=ix[ix>=0]
        for j in ix[1:]:er.extend([ix[0],j]);ec.extend([j,ix[0]])
    # Surface-face gates couple otherwise disconnected volume components too.
    for face in BF:
        ix=lookup[face];ix=ix[ix>=0]
        for j in ix[1:]:er.extend([ix[0],j]);ec.extend([j,ix[0]])
    for matrix in [C]+([]if L is None else[L]):
        for row in range(matrix.shape[0]):
            ix=np.unique(lookup[matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]//3]);ix=ix[ix>=0]
            for j in ix[1:]:er.extend([ix[0],j]);ec.extend([j,ix[0]])
    if embedded_surface is not None:
        S=embedded_surface.support
        for row in range(S.shape[0]):
            ix=lookup[S.indices[S.indptr[row]:S.indptr[row+1]]];ix=ix[ix>=0]
            for j in ix[1:]:er.extend([ix[0],j]);ec.extend([j,ix[0]])
    graph=sparse.coo_matrix((np.ones(len(er)),(er,ec)),shape=(len(F),len(F))).tocsr()
    count,labels=connected_components(graph,directed=False)
    best=X.copy();components=[]
    for label in range(count):
        local=np.flatnonzero(labels==label);nodes=F[local];dd=np.zeros_like(delta);dd[local]=delta[local]
        maxstep=float(np.abs(dd).max());maxgrad=float(np.abs(DG@dd.ravel()).max(initial=0))
        fraction=min(1.,float(step_limit_mm)/max(maxstep,1e-30),float(max_incremental_gradient)/max(maxgrad,1e-30))
        initial=np.maximum(target-C@best.astype(float).ravel(),0).sum();accepted=False;attempts=[]
        for backtrack in range(16):
            alpha=fraction*2.**-backtrack
            Y=best.copy();Y[nodes]=X[nodes].astype(float)+.001*alpha*delta[local]
            jj,_,qq,_=cell_values_gradients(Y,R,T);_,_,ar,co=boundary_face_values_gradients(Y,O,BF)
            layer_ok=L is None or np.min(L@Y.astype(float).ravel()-lt)>=-1e-9
            deficit=np.maximum(target-C@Y.astype(float).ravel(),0).sum()
            first_failed = (
                'cell_J' if not (jj.min()>=lo and jj.max()<=hi) else
                'cell_quality' if not qq.min()>=q_min else
                'coarse_surface' if not (ar.min()>=amin and ar.max()<=amax and co.min()>=cmin) else
                'hard_linear' if not layer_ok else
                'mapped_skin' if embedded_surface is not None and not embedded_surface.passes(Y) else
                'contact_progress' if not deficit<initial-1e-12 else None)
            attempts.append({'fraction':float(alpha),'first_failed_gate':first_failed,
                'q_min':float(qq.min()),'q_witness_row':int(qq.argmin()),
                'J_min':float(jj.min()),'J_max':float(jj.max()),
                'hard_linear_min_slack_m':None if L is None else float(np.min(L@Y.astype(float).ravel()-lt)),
                'hard_linear_witness_row':None if L is None else int(np.argmin(L@Y.astype(float).ravel()-lt)),
                'contact_deficit_l1_mm':float(deficit*1000)})
            if first_failed is None:
                best=Y;accepted=True;break
        components.append({'nodes':len(nodes),'applied':accepted,'line_fraction':float(alpha if accepted else 0),
                           'direction_max_incremental_gradient':maxgrad,'attempts':attempts})
    jj,_,qq,_=cell_values_gradients(best,R,T);_,_,ar,co=boundary_face_values_gradients(best,O,BF)
    fixed=np.setdiff1d(np.arange(len(X)),F)
    assert np.array_equal(best[fixed].view(np.uint32),X[fixed].view(np.uint32))
    report.update({'step_applied':bool(np.any(best.view(np.uint32)!=X.view(np.uint32))),'components':components,
      'contact_deficit_l1_mm':float(np.maximum(target-C@best.astype(float).ravel(),0).sum()*1000),
      'emitted_J_min':float(jj.min()),'emitted_J_max':float(jj.max()),'emitted_q_min':float(qq.min()),
      'boundary_area_ratio_min':float(ar.min()),'boundary_area_ratio_max':float(ar.max()),'boundary_normal_cosine_min':float(co.min()),
      'maximum_step_mm':float(np.linalg.norm(best.astype(float)-X,axis=1).max()*1000),
      'maximum_cumulative_displacement_mm':float(np.linalg.norm(best.astype(float)-O,axis=1).max()*1000)})
    if embedded_surface is not None:report.update(embedded_surface.report(best))
    return best,report
