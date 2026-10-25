"""Connected static tissue-mesh authoring; never a per-step physics solver.

Sparse coupled clearance steps retain the original reference, all cell gates and
anchored outside vertices. Linearized feasibility is not anatomical admission.
"""
from __future__ import annotations
import time
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

def sufficient_surface_gradient_caps(current_area_ratio,current_normal_cosine,area_bounds,normal_cosine_min,user_limit):
    """Sufficient componentwise caps for one triangle's incremental P1 gradient.

    Let eps bound the spectral norm of the incremental 3-D map on the triangle
    tangent plane. Area ratios change by at most (1±eps)^2. In an orthonormal
    tangent basis, the cross-product perturbation is bounded by 2eps+eps^2;
    requiring that to be at most sin(allowed angular margin) bounds normal
    rotation. The returned eps includes a 0.95 numerical safety factor. Float32
    nonlinear area/normal gates remain authoritative after backtracking.
    """
    r=np.asarray(current_area_ratio,dtype=np.float64)
    c=np.asarray(current_normal_cosine,dtype=np.float64)
    try:lo,hi=map(float,area_bounds);cmin=float(normal_cosine_min);limit=float(user_limit)
    except Exception as exc:raise ValueError('invalid surface gradient cap inputs') from exc
    if (r.ndim!=1 or c.shape!=r.shape or not len(r) or
        not np.isfinite(r).all() or not np.isfinite(c).all() or
        not np.isfinite([lo,hi,cmin,limit]).all() or
        np.any(r<=0.) or lo<=0. or hi<=lo or cmin<=0. or cmin>1. or
        not 0.<limit<1. or np.any(r<lo) or np.any(r>hi) or np.any(c<cmin) or np.any(c>1.+1e-10)):
        raise ValueError('invalid surface gradient cap inputs or current triangle outside gates')
    c=np.clip(c,-1.,1.)
    eps_area_lo=1.-np.sqrt(lo/r)
    eps_area_hi=np.sqrt(hi/r)-1.
    allowed=np.arccos(cmin)
    current=np.arccos(c)
    margin=np.maximum(allowed-current,0.)
    eps_normal=np.sqrt(1.+np.sin(margin))-1.
    return .95*np.maximum(0.,np.minimum.reduce((np.full_like(r,limit),eps_area_lo,eps_area_hi,eps_normal)))


def sparse_triangle_displacement_gradient(points,faces,free_nodes,node_count,vertex_map=None):
    """Nine-row P1 displacement-gradient Jacobian per triangle through a vertex map.

    LP displacement variables are millimetres; points and the returned
    gradients are SI. With no vertex_map, triangle vertices are mechanics nodes.
    Rows are face-major, then output component, then spatial derivative.
    """
    X=np.asarray(points,dtype=np.float64);rawF=np.asarray(free_nodes);rawT=np.asarray(faces)
    if X.ndim!=2 or X.shape[1]!=3 or not len(X) or not np.isfinite(X).all():
        raise ValueError('triangle points must be finite Nx3')
    if rawT.dtype.kind not in 'iu' or rawT.ndim!=2 or rawT.shape[1]!=3 or not len(rawT):
        raise ValueError('triangle faces must be a nonempty integer Mx3 table')
    if rawF.dtype.kind not in 'iu' or rawF.ndim!=1 or len(np.unique(rawF))!=len(rawF):
        raise ValueError('free nodes must be a unique integer vector')
    if not isinstance(node_count,(int,np.integer)) or isinstance(node_count,(bool,np.bool_)) or node_count<=0:
        raise ValueError('node_count must be a positive integer')
    F=rawF.astype(np.int64,copy=False);T=rawT.astype(np.int64,copy=False)
    if np.any(F<0) or np.any(F>=node_count) or np.any(T<0) or np.any(T>=len(X)) or np.any(np.diff(np.sort(T,axis=1),axis=1)==0):
        raise ValueError('invalid triangle/free-node index')
    if vertex_map is None:
        if len(X)!=node_count:raise ValueError('identity vertex map requires mechanics-sized points')
        W=sparse.eye(node_count,format='csr')
    else:
        W=sparse.csr_matrix(vertex_map,dtype=np.float64,copy=True)
        W.sum_duplicates();W.eliminate_zeros();W.sort_indices()
        if W.shape!=(len(X),node_count) or not np.isfinite(W.data).all() or np.any(W.data<0.):
            raise ValueError('invalid nonnegative vertex displacement map')
        sums=np.asarray(W.sum(axis=1)).ravel();active=np.diff(W.indptr)>0
        if np.any(np.abs(sums[active]-1.)>1e-6):raise ValueError('active vertex-map rows must sum to one')
    V=X[T]
    E=np.stack((V[:,1]-V[:,0],V[:,2]-V[:,0]),axis=2)
    try:pinv=np.linalg.pinv(E)
    except np.linalg.LinAlgError as exc:raise ValueError('triangle P1 pseudoinverse failed') from exc
    shape=np.stack((-pinv[:,0,:]-pinv[:,1,:],pinv[:,0,:],pinv[:,1,:]),axis=1)
    if not np.isfinite(shape).all():raise ValueError('nonfinite triangle P1 shape gradients')
    rows=[];cols=[];vals=[];face_rows=np.arange(len(T),dtype=np.int64)
    for b in range(3):
        H=sparse.csr_matrix((len(T),node_count),dtype=np.float64)
        for k in range(3):
            H=H+sparse.diags(shape[:,k,b],format='csr')@W[T[:,k],:]
        H=H[:,F].tocoo()
        for a in range(3):
            rows.extend((9*H.row+3*a+b).tolist())
            cols.extend((3*H.col+a).tolist())
            vals.extend((.001*H.data).tolist())
    return sparse.coo_matrix((vals,(rows,cols)),shape=(9*len(T),3*len(F))).tocsr()


def sparse_face_jacobian(gradients,faces,free_nodes,node_count):
    F=np.asarray(free_nodes,dtype=np.int64);T=np.asarray(faces,dtype=np.int64)
    lookup=np.full(node_count,-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    cols=lookup[T];valid=np.broadcast_to((cols>=0)[:,:,None],gradients.shape)
    rr=np.broadcast_to(np.arange(len(T))[:,None,None],gradients.shape)
    cc=3*cols[:,:,None]+np.arange(3)[None,None,:]
    return sparse.coo_matrix((gradients[valid],(rr[valid],cc[valid])),shape=(len(T),3*len(F))).tocsr()

def sufficient_cell_gradient_caps(current_quality,current_J,q_min,J_bounds,user_limit):
    """Per-cell componentwise ||grad(u)||_inf caps sufficient for current q/J.

    For any 3x3 incremental gradient with each entry bounded by g, its spectral
    norm is at most its Frobenius norm, 3g. Thus singular values of I+grad(u)
    lie in [1-3g,1+3g]; the determinant and longest-edge bounds give the
    sufficient q and J limits below. Caps are relative to the current geometry,
    in exact arithmetic. Emitted Float32 nonlinear gates remain authoritative.
    """
    q=np.asarray(current_quality,dtype=np.float64);J=np.asarray(current_J,dtype=np.float64)
    try:
        qmin=float(q_min);lo,hi=map(float,J_bounds);limit=float(user_limit)
    except Exception as exc:
        raise ValueError("invalid cell gradient cap inputs") from exc
    if (q.ndim!=1 or J.shape!=q.shape or not len(q) or not np.isfinite(q).all() or
        not np.isfinite(J).all() or np.any(q<=0.) or np.any(J<=0.) or
        not np.isfinite([qmin,lo,hi,limit]).all() or qmin<=0. or lo<=0. or
        hi<=lo or not 0.<limit<1.):
        raise ValueError("invalid cell gradient cap inputs")
    if np.any(q<qmin) or np.any(J<lo) or np.any(J>hi):
        raise ValueError("current cells must already satisfy q/J gates")
    ratio_q=np.cbrt(qmin/q)
    ratio_lo=np.cbrt(lo/J)
    # Divide cube roots instead of forming hi/J, which could overflow for
    # otherwise finite extreme-but-valid diagnostic inputs.
    ratio_hi=np.cbrt(hi)/np.cbrt(J)
    q_cap=(1.-ratio_q)/(3.*(1.+ratio_q))
    lo_cap=(1.-ratio_lo)/3.
    hi_cap=(ratio_hi-1.)/3.
    return .95*np.maximum(0.,np.minimum.reduce((np.full_like(q,limit),q_cap,lo_cap,hi_cap)))


def coupled_clearance_step(points,reference,tetrahedra,free_nodes,contact_matrix,contact_targets,
                           *, step_limit_mm=2.,q_min=1e-5,J_bounds=(.35,2.5),time_limit_s=90.,
                           lower_linear_matrix=None,lower_linear_targets=None,max_incremental_gradient=.12,
                           construction_origin=None,cumulative_displacement_weight=.01,
                           boundary_faces=None,boundary_area_bounds=(.25,4.),boundary_normal_cosine_min=.25,linear_method="highs",embedded_surface=None,dihedral_active_fraction=0.,dihedral_constraint_passes=0,cell_witness_gradient_limit=None,surface_witness_gradient_limit=None,backtracking_domain_labels=None):
    """One bounded sparse LP and hard nonlinear Float32 backtracking transaction.

    Contact matrix maps absolute xyz metres to normal separation metres. Each row
    may address a vertex or a barycentric point on an edge/face. Slack is explicit
    and reported; the caller must refresh exact collision witnesses.
    """
    if linear_method not in ("highs","highs-ipm","highs-ds"):raise ValueError("unsupported sparse linear method")
    try:dihedral_active_fraction=float(dihedral_active_fraction)
    except Exception as exc:raise ValueError('dihedral active fraction must be numeric') from exc
    if not np.isfinite(dihedral_active_fraction) or not 0.<=dihedral_active_fraction<1.:
        raise ValueError('dihedral active fraction must lie in [0,1)')
    if isinstance(dihedral_constraint_passes,(bool,np.bool_)) or not isinstance(dihedral_constraint_passes,(int,np.integer)) or not 0<=dihedral_constraint_passes<=8:
        raise ValueError('dihedral constraint passes must be an integer in [0,8]')
    if cell_witness_gradient_limit is not None:
        cell_witness_gradient_limit=float(cell_witness_gradient_limit)
        if not np.isfinite(cell_witness_gradient_limit) or not 0.<cell_witness_gradient_limit<1. or not dihedral_constraint_passes:
            raise ValueError('cell witness gradient limit requires (0,1) and refinement passes')
    if surface_witness_gradient_limit is not None:
        surface_witness_gradient_limit=float(surface_witness_gradient_limit)
        if (not np.isfinite(surface_witness_gradient_limit) or
            not 0.<surface_witness_gradient_limit<1. or not dihedral_constraint_passes or
            (boundary_faces is None and embedded_surface is None)):
            raise ValueError('surface witness gradient limit requires (0,1), a surface, and refinement passes')
    can_generate=(cell_witness_gradient_limit is not None or
                  surface_witness_gradient_limit is not None or
                  (embedded_surface is not None and embedded_surface.max_dihedral_change_degrees is not None))
    if dihedral_constraint_passes and not can_generate:
        raise ValueError('constraint-generation passes require an enabled geometric witness generator')
    X=np.asarray(points);R=np.asarray(reference);rawT=np.asarray(tetrahedra);rawF=np.asarray(free_nodes)
    if X.dtype!=np.float32:raise ValueError('actual emitted Float32 input required')
    if rawT.dtype.kind not in 'iu' or rawT.ndim!=2 or rawT.shape[1]!=4 or len(rawT)==0:raise ValueError('tetrahedra must be a nonempty integer Mx4 table')
    if rawF.dtype.kind not in 'iu' or rawF.ndim!=1 or len(rawF)==0:raise ValueError('free_nodes must be a nonempty integer vector')
    T=rawT.astype(np.int64,copy=False);F=rawF.astype(np.int64,copy=False)
    if R.dtype.kind not in 'fiu' or not np.isfinite(R).all():raise ValueError('reference coordinates must be finite numeric values')
    if len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=len(X)):raise ValueError('invalid free-node IDs')
    domain_labels=None
    if backtracking_domain_labels is not None:
        raw_domain_labels=np.asarray(backtracking_domain_labels)
        if (raw_domain_labels.dtype.kind not in 'iu' or raw_domain_labels.ndim!=1 or
            raw_domain_labels.shape!=(len(X),) or np.any(raw_domain_labels<0)):
            raise ValueError('backtracking domain labels must be a nonnegative integer[N] vector')
        domain_labels=raw_domain_labels.astype(np.int64,copy=False)
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
    embedded_lp_info=None
    embedded_dihedral_lp_info=None
    if embedded_surface is not None:
        # Linearize only detailed faces touched by a free coarse node. The
        # unchanged nonlinear Float32 gate below still checks every face.
        surface_face_ids=embedded_surface.linearization_face_ids(F)
        if len(surface_face_ids):
            ids,ea,eaJ,ec,ecJ=embedded_surface.area_cosine_linearization(
                X,F,face_ids=surface_face_ids)
            emin,emax=embedded_surface.area_bounds
            eaJ=.001*eaJ;ecJ=.001*ecJ
            rows += [sparse.hstack([-eaJ,z(len(ea))]),sparse.hstack([eaJ,z(len(ea))]),
                     sparse.hstack([-ecJ,z(len(ec))])]
            bs += [ea-emin,emax-ea,ec-embedded_surface.normal_cosine_min]
            embedded_lp_info={'face_ids':ids,'area':ea,'area_j_mm':eaJ,
                              'cosine':ec,'cosine_j_mm':ecJ}
        else:
            embedded_lp_info={'face_ids':surface_face_ids}
        if embedded_surface.max_dihedral_change_degrees is not None:
            selected_edges=None
            if dihedral_active_fraction>0.:
                all_edges,all_delta=embedded_surface.dihedral_change_metrics(X)
                active_limit=dihedral_active_fraction*np.deg2rad(embedded_surface.max_dihedral_change_degrees)
                selected_edges=all_edges[np.abs(all_delta)>=active_limit]
            edge_ids,dihedral_delta,dihedral_J=embedded_surface.dihedral_change_linearization(X,F,edge_ids=selected_edges)
            if len(edge_ids):
                dihedral_J_mm=.001*dihedral_J
                limit=np.deg2rad(embedded_surface.max_dihedral_change_degrees)
                rows += [sparse.hstack([dihedral_J_mm,z(len(edge_ids))]),
                         sparse.hstack([-dihedral_J_mm,z(len(edge_ids))])]
                bs += [limit-dihedral_delta,limit+dihedral_delta]
                embedded_dihedral_lp_info={'edge_ids':edge_ids,'delta':dihedral_delta,
                    'jacobian_mm':dihedral_J_mm,'limit':limit}
            else:
                embedded_dihedral_lp_info={'edge_ids':edge_ids,'delta':dihedral_delta,
                    'jacobian_mm':dihedral_J,'limit':np.deg2rad(embedded_surface.max_dihedral_change_degrees)}
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
    bounds=[(-step_limit_mm,step_limit_mm)]*nd+[(0,None)]*nc+[(0,step_limit_mm)]+[(0,None if O is not None else step_limit_mm)]*nd
    # Constraint generation fits *all* connected tissues together. A proposed
    # crease adds its starting-state derivative before another solve; it never
    # admits the proposed surface or replaces the full Float32 postcheck.
    # One shared time budget covers all LP calls and witness preparation.
    lp_started=time.monotonic();lp_history=[];result=None;lp_stop_reason='generation_limit'
    pending_dihedral_info=embedded_dihedral_lp_info
    cell_cap_ids=np.empty(0,dtype=np.int64);solved_cell_cap_ids=cell_cap_ids.copy()
    cell_cap_values={}
    solved_cell_cap_values=np.empty(0,dtype=np.float64)
    surface_cap_values={}
    solved_surface_cap_values={}
    surface_caps_not_added_at_budget=[]
    for generation in range(dihedral_constraint_passes+1):
        remaining=time_limit_s-(time.monotonic()-lp_started)
        if result is not None and remaining<=.001:
            lp_stop_reason='time_budget_exhausted_before_refinement';break
        trial_result=linprog(objective,A_ub=sparse.vstack(rows,format='csr'),b_ub=np.concatenate(bs),
            bounds=bounds,method=linear_method,options={'time_limit':max(.001,remaining)})
        entry={'generation':generation,'success':bool(trial_result.success),
            'status':int(trial_result.status),'message':str(trial_result.message),
            'constraint_rows':int(sum(len(b) for b in bs)),
            'cumulative_elapsed_s':float(time.monotonic()-lp_started)}
        lp_history.append(entry)
        if not trial_result.success:
            if result is None:result=trial_result
            lp_stop_reason='initial_lp_failed' if not result.success else 'refinement_lp_failed'
            # Keep the previous solved direction; normal full backtracking
            # still certifies it. Failed refinement is explicitly retained.
            break
        result=trial_result;embedded_dihedral_lp_info=pending_dihedral_info
        solved_cell_cap_ids=cell_cap_ids.copy()
        solved_cell_cap_values=np.asarray(
            [cell_cap_values[int(row)] for row in solved_cell_cap_ids],dtype=np.float64)
        solved_surface_cap_values=dict(surface_cap_values)
        if not dihedral_constraint_passes:
            lp_stop_reason='constraint_generation_disabled';break
        proposed=X.copy()
        proposed[F]=X[F].astype(float)+.001*result.x[:nd].reshape(-1,3)
        all_edges=np.empty(0,dtype=np.int64);all_delta=np.empty(0,dtype=np.float64)
        limit=None;violated=np.empty(0,dtype=np.int64)
        if embedded_surface is not None and embedded_surface.max_dihedral_change_degrees is not None:
            all_edges,all_delta=embedded_surface.dihedral_change_metrics(proposed)
            limit=np.deg2rad(embedded_surface.max_dihedral_change_degrees)
            violated=all_edges[np.abs(all_delta)>limit]
        entry['proposed_dihedral_violations']=int(len(violated))
        proposed_bad_cells=np.empty(0,dtype=np.int64)
        if cell_witness_gradient_limit is not None:
            proposed_J,_,proposed_q,_=cell_values_gradients(proposed,R,T)
            proposed_bad_cells=np.intersect1d(active,np.flatnonzero(
                (proposed_q<q_min*1.02)|(proposed_J<lo+.02)|(proposed_J>hi-.02)))
            entry['proposed_cell_witness_count']=int(len(proposed_bad_cells))
            entry['proposed_cell_q_min']=float(proposed_q.min())
            entry['proposed_cell_J_min']=float(proposed_J.min())
            entry['proposed_cell_J_max']=float(proposed_J.max())

        new_surface_caps=[]
        proposed_surface_witnesses=[]
        if surface_witness_gradient_limit is not None:
            if BF is not None:
                _,_,pa,pc=boundary_face_values_gradients(proposed,O,BF)
                coarse_bad=np.flatnonzero((pa<amin)|(pa>amax)|(pc<cmin))
                for face_id in coarse_bad:
                    key=('coarse_boundary',int(face_id))
                    proposed_surface_witnesses.append({'surface':key[0],'face_id':key[1]})
                    if key not in surface_cap_values:
                        eps=sufficient_surface_gradient_caps(
                            ar[face_id:face_id+1],ac[face_id:face_id+1],
                            (amin,amax),cmin,surface_witness_gradient_limit)[0]
                        new_surface_caps.append((key,float(eps)))
            if embedded_surface is not None:
                _,_,da,dc=embedded_surface.metrics(proposed)
                _,_,ca,cc=embedded_surface.metrics(X)
                detail_bad=np.flatnonzero((da<embedded_surface.area_bounds[0])|
                    (da>embedded_surface.area_bounds[1])|
                    (dc<embedded_surface.normal_cosine_min))
                for face_id in detail_bad:
                    key=('mapped_skin',int(face_id))
                    proposed_surface_witnesses.append({'surface':key[0],'face_id':key[1]})
                    if key not in surface_cap_values:
                        eps=sufficient_surface_gradient_caps(
                            ca[face_id:face_id+1],cc[face_id:face_id+1],
                            embedded_surface.area_bounds,embedded_surface.normal_cosine_min,
                            surface_witness_gradient_limit)[0]
                        new_surface_caps.append((key,float(eps)))
        entry['proposed_surface_violation_count']=int(len(proposed_surface_witnesses))
        entry['proposed_surface_violations']=proposed_surface_witnesses
        new_cells=np.setdiff1d(proposed_bad_cells,cell_cap_ids)
        already_edges=(np.empty(0,dtype=np.int64) if embedded_dihedral_lp_info is None
                       else embedded_dihedral_lp_info['edge_ids'])
        selected=np.setdiff1d(violated,already_edges)
        edge_ids=np.empty(0,dtype=np.int64);delta_edges=np.empty(0);J_edges=sparse.csr_matrix((0,nd))
        if len(selected):
            edge_ids,delta_edges,J_edges=embedded_surface.dihedral_change_linearization(X,F,edge_ids=selected)
        entry['new_dihedral_rows']=int(len(edge_ids))
        entry['new_cell_gradient_caps']=int(len(new_cells))
        entry['new_surface_gradient_caps']=int(len(new_surface_caps))
        if generation==dihedral_constraint_passes:
            entry['new_cell_caps_not_added_at_budget']=int(len(new_cells))
            entry['new_dihedral_edges_not_added_at_budget']=int(len(selected))
            surface_caps_not_added_at_budget=[
                {'surface':key[0],'face_id':key[1],'epsilon_cap':eps,'componentwise_g':eps/3.}
                for key,eps in new_surface_caps]
            entry['new_surface_caps_not_added_at_budget']=surface_caps_not_added_at_budget
            entry['cumulative_elapsed_s']=float(time.monotonic()-lp_started)
            lp_stop_reason='generation_limit';break
        if not len(edge_ids) and not len(new_cells) and not len(new_surface_caps):
            entry['cumulative_elapsed_s']=float(time.monotonic()-lp_started)
            lp_stop_reason='no_new_geometric_witnesses';break
        if len(edge_ids):
            Jmm=.001*J_edges
            rows += [sparse.hstack([Jmm,z(len(edge_ids))]),sparse.hstack([-Jmm,z(len(edge_ids))])]
            bs += [limit-delta_edges,limit+delta_edges]
            old_info=embedded_dihedral_lp_info
            if old_info is None:
                pending_dihedral_info={'edge_ids':edge_ids,'delta':delta_edges,
                    'jacobian_mm':Jmm,'limit':limit}
            else:
                pending_dihedral_info={'edge_ids':np.r_[old_info['edge_ids'],edge_ids],
                    'delta':np.r_[old_info['delta'],delta_edges],
                    'jacobian_mm':sparse.vstack([old_info['jacobian_mm'],Jmm],format='csr'),'limit':limit}
        if len(new_cells):
            new_caps=sufficient_cell_gradient_caps(
                q[new_cells],J[new_cells],q_min,(lo,hi),cell_witness_gradient_limit)
            # Bound the incremental gradient relative to the actual current
            # emitted geometry, not the undeformed reference configuration.
            DG=sparse_displacement_gradient(X,T[new_cells],F)
            rows += [sparse.hstack([DG,z(DG.shape[0])]),sparse.hstack([-DG,z(DG.shape[0])])]
            row_caps=np.repeat(new_caps,9)
            bs += [row_caps,row_caps]
            cell_cap_values.update({int(row):float(cap) for row,cap in zip(new_cells,new_caps)})
            cell_cap_ids=np.union1d(cell_cap_ids,new_cells)
            entry['new_cell_gradient_cap_min']=float(new_caps.min())
            entry['new_cell_gradient_cap_max']=float(new_caps.max())
        for key,eps in new_surface_caps:
            face_id=key[1]
            if key[0]=='coarse_boundary':
                DG=sparse_triangle_displacement_gradient(X,BF[face_id:face_id+1],F,len(X))
            else:
                _,DG=embedded_surface.triangle_gradient_linearization(X,F,[face_id])
            if DG.shape!=(9,nd) or DG.nnz==0:
                raise ValueError('violating surface face has no free triangle-gradient support')
            g=eps/3.
            rows += [sparse.hstack([DG,z(DG.shape[0])]),sparse.hstack([-DG,z(DG.shape[0])])]
            caps=np.full(9,g,dtype=np.float64)
            bs += [caps,caps]
            surface_cap_values[key]={'epsilon_cap':eps,'componentwise_g':g}
        if new_surface_caps:
            entry['new_surface_gradient_cap_min_g']=float(min(eps/3. for _,eps in new_surface_caps))
            entry['new_surface_gradient_cap_max_g']=float(max(eps/3. for _,eps in new_surface_caps))
        entry['cumulative_elapsed_s']=float(time.monotonic()-lp_started)
    lp_elapsed=float(time.monotonic()-lp_started)
    def surface_cap_records(values):
        return [{'surface':key[0],'face_id':int(key[1]),
                 'epsilon_cap':float(value['epsilon_cap']),
                 'componentwise_g':float(value['componentwise_g'])}
                for key,value in sorted(values.items(),key=lambda item:(item[0][0],item[0][1]))]
    pending_surface_caps={key:value for key,value in surface_cap_values.items()
                          if key not in solved_surface_cap_values}
    report={'linear_method':linear_method,'lp_success':bool(result.success),'lp_status':int(result.status),'lp_message':result.message,
            'free_nodes':len(F),'touched_cells':len(active),'contact_rows':nc,'starting_contact_deficit_l1_mm':float(np.maximum(rhs,0).sum()),
            'reference_unchanged':True,'geometry_admitted':False,'accepted_physical_steps':0}
    report.update({'lp_cell_witness_gradient_limit':cell_witness_gradient_limit,
                   'lp_cell_gradient_cap_count':int(len(solved_cell_cap_ids)),
                   'lp_cell_gradient_cap_row_count':int(9*len(solved_cell_cap_ids)),
                   'lp_cell_gradient_cap_tetrahedron_rows':solved_cell_cap_ids.tolist(),
                   'lp_cell_gradient_cap_values':solved_cell_cap_values.tolist(),
                   'lp_cell_gradient_cap_min':float(solved_cell_cap_values.min()) if len(solved_cell_cap_values) else None,
                   'lp_cell_gradient_cap_max':float(solved_cell_cap_values.max()) if len(solved_cell_cap_values) else None,
                   'cell_gradient_caps_are_current_geometry_sufficient_bounds':cell_witness_gradient_limit is not None,
                   'cell_gradient_caps_are_componentwise_infinity_bounds':True,
                   'cell_gradient_caps_require_exact_float32_q_J_postcheck':True,
                   'cell_gradient_caps_are_authoring_only':True,
                   'lp_surface_witness_gradient_limit':surface_witness_gradient_limit,
                   'lp_surface_gradient_cap_solved_count':int(len(solved_surface_cap_values)),
                   'lp_surface_gradient_cap_solved_component_count':int(9*len(solved_surface_cap_values)),
                   'lp_surface_gradient_cap_solved_inequality_row_count':int(18*len(solved_surface_cap_values)),
                   'lp_surface_gradient_cap_solved':surface_cap_records(solved_surface_cap_values),
                   'lp_surface_gradient_cap_pending_count':int(len(pending_surface_caps)),
                   'lp_surface_gradient_cap_pending':surface_cap_records(pending_surface_caps),
                   'lp_surface_gradient_cap_not_added_at_budget':surface_caps_not_added_at_budget,
                   'surface_gradient_caps_are_current_triangle_sufficient_bounds':surface_witness_gradient_limit is not None,
                   'surface_gradient_caps_are_componentwise_infinity_bounds':True,
                   'surface_gradient_caps_require_exact_float32_area_normal_postcheck':True,
                   'surface_gradient_caps_are_authoring_only':True,
                   'lp_calls':len(lp_history),'lp_elapsed_s':lp_elapsed,
                   'lp_constraint_generation':lp_history,
                   'dihedral_constraint_passes_budget':dihedral_constraint_passes,
                   'lp_refinement_failure_retained':bool(result.success and any(not h['success'] for h in lp_history)),
                   'lp_constraint_generation_stop_reason':lp_stop_reason,
                   'lp_refinement_budget_exhausted':lp_stop_reason=='time_budget_exhausted_before_refinement'})
    if embedded_surface is not None:
        report.update(embedded_surface.report(X))
        if embedded_lp_info is not None:
            ids=embedded_lp_info['face_ids']
            report['mapped_skin_linearization_face_count']=int(len(ids))
            report['mapped_skin_linearization_face_ids']=ids.tolist()
        if embedded_dihedral_lp_info is not None:
            report['mapped_skin_dihedral_active_fraction']=dihedral_active_fraction
            report['mapped_skin_dihedral_postcheck_scope']='all configured edges, regardless of LP row selection'
            report['mapped_skin_dihedral_linearization_edge_count']=int(len(embedded_dihedral_lp_info['edge_ids']))
            report['mapped_skin_dihedral_linearization_edge_ids']=embedded_dihedral_lp_info['edge_ids'].tolist()
    if not result.success:return X.copy(),report
    delta=result.x[:nd].reshape(-1,3)*.001
    original=np.maximum(target-C@X.astype(float).ravel(),0).sum()
    report['lp_predicted_contact_deficit_l1_mm']=float(result.x[nd:nd+nc].sum())
    if embedded_lp_info is not None and 'area' in embedded_lp_info:
        earea=embedded_lp_info['area']+embedded_lp_info['area_j_mm']@result.x[:nd]
        ecos=embedded_lp_info['cosine']+embedded_lp_info['cosine_j_mm']@result.x[:nd]
        report.update({'lp_linearized_mapped_skin_area_ratio_min':float(np.min(earea)),
                       'lp_linearized_mapped_skin_area_ratio_max':float(np.max(earea)),
                       'lp_linearized_mapped_skin_normal_cosine_min':float(np.min(ecos))})
    if embedded_dihedral_lp_info is not None and len(embedded_dihedral_lp_info['edge_ids']):
        linearized=(embedded_dihedral_lp_info['delta']+
                    embedded_dihedral_lp_info['jacobian_mm']@result.x[:nd])
        report.update({
            'lp_linearized_mapped_skin_dihedral_abs_change_max_degrees':float(np.degrees(np.abs(linearized).max())),
            'lp_linearized_mapped_skin_dihedral_violations':int(np.count_nonzero(np.abs(linearized)>embedded_dihedral_lp_info['limit']+1e-8))})
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
        S=embedded_surface.coupling_support
        for row in range(S.shape[0]):
            ix=lookup[S.indices[S.indptr[row]:S.indptr[row+1]]]
            ix=ix[ix>=0]
            for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    if embedded_dihedral_lp_info is not None:
        DJ=embedded_dihedral_lp_info['jacobian_mm'].tocsr()
        for row in range(DJ.shape[0]):
            ix=np.unique(DJ.indices[DJ.indptr[row]:DJ.indptr[row+1]]//3)
            if len(ix)>1:
                for j in ix[1:]:er.append(ix[0]);ec.append(j);er.append(j);ec.append(ix[0])
    graph=sparse.coo_matrix((np.ones(len(er)),(er,ec)),shape=(len(F),len(F))).tocsr()
    count,labels=connected_components(graph,directed=False)
    best=X.copy();components=[]
    for label in range(count):
        local=np.flatnonzero(labels==label);nodes=F[local]
        component_start=best.copy()
        original=np.maximum(target-C@best.astype(float).ravel(),0).sum()
        accepted=False;attempts=[];initial_full_failed=None
        for k in range(15):
            scale=2.**-k;Y=best.copy();Y[nodes]=X[nodes].astype(float)+scale*delta[local]
            jj,_,qq,_=cell_values_gradients(Y,R,T)
            linear_pass=True
            if lower_linear_matrix is not None:linear_pass=bool(np.min(L@Y.astype(float).ravel()-lt)>=-1e-9)
            surface_pass=True
            trial_area_min=trial_area_max=trial_cosine_min=None
            trial_dihedral_max=None
            dihedral_pass=True
            if embedded_surface is not None and embedded_surface.max_dihedral_change_degrees is not None:
                _,trial_delta=embedded_surface.dihedral_change_metrics(Y)
                trial_dihedral_max=float(np.degrees(np.abs(trial_delta).max())) if len(trial_delta) else 0.
                dihedral_pass=embedded_surface.dihedral_passes(Y)
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
                'mapped_skin_dihedral' if not dihedral_pass else
                'mapped_skin' if embedded_surface is not None and not embedded_surface.passes(Y) else
                'contact_progress' if not deficit<original-1e-12 else None)
            attempt={'fraction':float(scale),'first_failed_gate':first_failed,
                'q_min':float(qq.min()),'q_witness_row':int(qq.argmin()),
                'J_min':float(jj.min()),'J_max':float(jj.max()),
                'hard_linear_min_slack_m':None if lower_linear_matrix is None else float(np.min(L@Y.astype(float).ravel()-lt)),
                'hard_linear_witness_row':None if lower_linear_matrix is None else int(np.argmin(L@Y.astype(float).ravel()-lt)),
                'contact_deficit_l1_mm':float(deficit*1000),
                'boundary_area_ratio_min':trial_area_min,
                'boundary_area_ratio_max':trial_area_max,
                'boundary_normal_cosine_min':trial_cosine_min}
            if trial_dihedral_max is not None:
                attempt['mapped_skin_dihedral_abs_change_max_degrees']=trial_dihedral_max
            attempts.append(attempt)
            if k==0:initial_full_failed=first_failed
            if first_failed is None:
                best=Y;accepted=True;break
        component_report={'nodes':len(nodes),'applied':accepted,'line_fraction':scale if accepted else 0.,'attempts':attempts}
        if domain_labels is not None:
            component_report['initial_full_step_first_failed_gate']=initial_full_failed
        components.append(component_report)
    domain_attempts=[];domain_accepted=False;domain_candidate=X.copy();domain_selected=False
    component_full_alpha1_failed=bool(domain_labels is not None and any(
        c.get('initial_full_step_first_failed_gate') is not None for c in components))
    if component_full_alpha1_failed:
        # Try each declared construction domain across the complete node set,
        # from the original input plus previously accepted domain updates.
        # Labels do not alter the LP or its coupling graph; every global gate
        # and the global C deficit is rechecked after each local proposal.
        for domain_id in np.unique(domain_labels[F]):
            domain_nodes=F[domain_labels[F]==domain_id];domain_local=lookup[domain_nodes]
            before=float(np.maximum(target-C@domain_candidate.astype(float).ravel(),0).sum())
            domain_applied=False;domain_trials=[]
            for dk in range(15):
                domain_scale=2.**-dk;Y=domain_candidate.copy()
                Y[domain_nodes]=X[domain_nodes].astype(float)+domain_scale*delta[domain_local]
                djj,_,dqq,_=cell_values_gradients(Y,R,T)
                dlinear=True
                if lower_linear_matrix is not None:dlinear=bool(np.min(L@Y.astype(float).ravel()-lt)>=-1e-9)
                dsurface=True;darea_min=darea_max=dcos_min=None;ddihedral_max=None;ddihedral=True
                if embedded_surface is not None and embedded_surface.max_dihedral_change_degrees is not None:
                    _,ddelta=embedded_surface.dihedral_change_metrics(Y)
                    ddihedral_max=float(np.degrees(np.abs(ddelta).max())) if len(ddelta) else 0.
                    ddihedral=embedded_surface.dihedral_passes(Y)
                if BF is not None:
                    _,_,darea,dcos=boundary_face_values_gradients(Y,O,BF)
                    darea_min=float(np.min(darea));darea_max=float(np.max(darea));dcos_min=float(np.min(dcos))
                    dsurface=bool(darea_min>=amin and darea_max<=amax and dcos_min>=cmin)
                ddeficit=float(np.maximum(target-C@Y.astype(float).ravel(),0).sum())
                dfail=(
                    'cell_quality' if not np.min(dqq)>=q_min else
                    'cell_J' if not (np.min(djj)>=lo and np.max(djj)<=hi) else
                    'hard_linear' if not dlinear else
                    'coarse_surface' if not dsurface else
                    'mapped_skin_dihedral' if not ddihedral else
                    'mapped_skin' if embedded_surface is not None and not embedded_surface.passes(Y) else
                    'contact_progress' if not ddeficit<before-1e-12 else None)
                dtrial={'fraction':float(domain_scale),'first_failed_gate':dfail,
                    'q_min':float(dqq.min()),'q_witness_row':int(dqq.argmin()),
                    'J_min':float(djj.min()),'J_max':float(djj.max()),
                    'hard_linear_min_slack_m':None if lower_linear_matrix is None else float(np.min(L@Y.astype(float).ravel()-lt)),
                    'hard_linear_witness_row':None if lower_linear_matrix is None else int(np.argmin(L@Y.astype(float).ravel()-lt)),
                    'contact_deficit_l1_mm':float(ddeficit*1000),
                    'boundary_area_ratio_min':darea_min,'boundary_area_ratio_max':darea_max,
                    'boundary_normal_cosine_min':dcos_min}
                if ddihedral_max is not None:dtrial['mapped_skin_dihedral_abs_change_max_degrees']=ddihedral_max
                domain_trials.append(dtrial)
                if dfail is None:
                    domain_candidate=Y;domain_applied=True;domain_accepted=True;break
            domain_attempts.append({'domain_id':int(domain_id),'nodes':int(len(domain_nodes)),
                'applied':domain_applied,'line_fraction':float(domain_scale if domain_applied else 0.),
                'attempts':domain_trials})
        full_deficit=float(np.maximum(target-C@best.astype(float).ravel(),0).sum())
        candidate_deficit=float(np.maximum(target-C@domain_candidate.astype(float).ravel(),0).sum())
        if domain_accepted and candidate_deficit<full_deficit-1e-12:
            best=domain_candidate;domain_selected=True
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
    if domain_labels is not None:
        final_hard_linear=True
        final_linear_min=None
        if lower_linear_matrix is not None:
            final_linear_min=float(np.min(L@best.astype(float).ravel()-lt))
            final_hard_linear=final_linear_min>=-1e-9
        final_coarse=True
        if BF is not None:
            final_coarse=bool(report['boundary_area_ratio_min']>=amin and
                report['boundary_area_ratio_max']<=amax and report['boundary_normal_cosine_min']>=cmin)
        final_dihedral=True
        if embedded_surface is not None and embedded_surface.max_dihedral_change_degrees is not None:
            final_dihedral=bool(embedded_surface.dihedral_passes(best))
        final_detail=True if embedded_surface is None else bool(embedded_surface.passes(best))
        final_hard=bool(qq.min()>=q_min and jj.min()>=lo and jj.max()<=hi and
            final_hard_linear and final_coarse and final_dihedral and final_detail)
        final_deficit=float(np.maximum(target-C@best.astype(float).ravel(),0).sum())
        report.update({'construction_backtracking_domain_labels_enabled':True,
            'construction_backtracking_domain_label_count':int(len(np.unique(domain_labels[F]))),
            'construction_backtracking_partition_is_not_physical_ownership':True,
            'construction_backtracking_full_component_alpha1_failed':component_full_alpha1_failed,
            'construction_backtracking_domain_fallback_attempted':component_full_alpha1_failed,
            'construction_backtracking_domain_fallback_candidate_valid':domain_accepted,
            'construction_backtracking_domain_fallback_accepted':domain_selected,
            'construction_backtracking_selected_path':'domain_partition' if domain_selected else 'full_components',
            'construction_backtracking_domain_attempts':domain_attempts,
            'construction_backtracking_global_final_hard_gates_pass':final_hard,
            'construction_backtracking_global_final_contact_deficit_reduced':bool(final_deficit<
                np.maximum(target-C@X.astype(float).ravel(),0).sum()-1e-12),
            'construction_backtracking_global_final_contact_deficit_l1_mm':float(final_deficit*1000),
            'construction_backtracking_global_final_hard_linear_min_slack_m':final_linear_min})
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
    if embedded_surface is not None:
        report.update(embedded_surface.report(X))
        report['mapped_skin_linearization_face_count']=0
        report['mapped_skin_linearization_face_ids']=[]
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
    embedded_lp_info=None
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
            surface_face_ids=embedded_surface.linearization_face_ids(F)
            if len(surface_face_ids):
                ids,ea,eaJ,ec,ecJ=embedded_surface.area_cosine_linearization(
                    X,F,face_ids=surface_face_ids)
                emin,emax=embedded_surface.area_bounds
                eaJ=.001*eaJ;ecJ=.001*ecJ
                H=sparse.vstack([H,eaJ,-eaJ,-ecJ],format='csr')
                h=np.r_[h,emax-ea,ea-emin,ec-embedded_surface.normal_cosine_min]
                embedded_lp_info={'face_ids':ids}
            else:
                embedded_lp_info={'face_ids':surface_face_ids}
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
    if embedded_surface is not None and embedded_lp_info is not None:
        report['mapped_skin_linearization_face_count']=int(len(embedded_lp_info['face_ids']))
        report['mapped_skin_linearization_face_ids']=embedded_lp_info['face_ids'].tolist()
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
        S=embedded_surface.coupling_support
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
