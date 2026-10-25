"""Detailed skin constraints composed with its existing coarse displacement map.

Offline connected-volume authoring only. No constitutive law, physical state,
rendering owner, or runtime format is changed.
"""
from __future__ import annotations
import numpy as np

from scipy import sparse

def _interior_edge_topology(faces):
    """Return deterministic 094-style two-face edges and their oriented 4-vertex stars.

    Edge IDs index lexicographically sorted undirected edges after retaining only
    edges with exactly two incident faces. Direction comes from the first
    incidence in source-face traversal order, matching regional-distortion-094.
    """
    F=np.asarray(faces,dtype=np.int64)
    directed=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]],axis=0)
    edge_face=np.tile(np.arange(len(F),dtype=np.int64),3)
    undirected=np.sort(directed,axis=1)
    unique_edges,inv,counts=np.unique(undirected,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inv,kind="stable")
    starts=np.r_[0,np.cumsum(counts[:-1])]
    manifold_ids=np.flatnonzero(counts==2)
    i0=order[starts[manifold_ids]]
    i1=order[starts[manifold_ids]+1]
    f0=edge_face[i0];f1=edge_face[i1]
    edge_dir=directed[i0]
    opposite=np.empty((len(manifold_ids),2),dtype=np.int64)
    for k,(a,b) in enumerate(edge_dir):
        for j,face_id in enumerate((f0[k],f1[k])):
            tri=F[face_id]
            rem=tri[(tri!=a)&(tri!=b)]
            if rem.shape!=(1,):
                raise ValueError("invalid manifold edge incidence")
            opposite[k,j]=rem[0]
    vertices=np.column_stack((edge_dir,opposite))
    reverse=np.all(directed[i1]==edge_dir[:,::-1],axis=1)
    return unique_edges[manifold_ids],vertices,np.column_stack((f0,f1)),reverse

def _wrap_radians(delta):
    d=np.asarray(delta,dtype=np.float64)
    return (d+np.pi)%(2*np.pi)-np.pi

def _skew(v):
    x,y,z=v[:,0],v[:,1],v[:,2]
    out=np.zeros((len(v),3,3),dtype=np.float64)
    out[:,0,1]=-z;out[:,0,2]=y
    out[:,1,0]=z;out[:,1,2]=-x
    out[:,2,0]=-y;out[:,2,1]=x
    return out

def _normal_coordinate_gradients(v0,v1,v2,normal_gradient):
    """Gradient of a scalar against the unit triangle normal."""
    e1=v1-v0;e2=v2-v0
    cross=np.cross(e1,e2);length=np.linalg.norm(cross,axis=1)
    if np.any(length<=np.finfo(np.float64).tiny):
        raise ValueError("degenerate triangle in dihedral derivative")
    n=cross/length[:,None]
    q=(normal_gradient-n*np.einsum("ij,ij->i",n,normal_gradient)[:,None])/length[:,None]
    matrices=np.stack((_skew(e2-e1),-_skew(e2),_skew(e1)),axis=1)
    return np.einsum("mkji,mj->mki",matrices,q)

def _dihedral_values_gradients(points, edge_vertices):
    """094 signed dihedral and analytic xyz derivative for each oriented edge.

    edge_vertices rows are [start,end,opposite_face0,opposite_face1]; face0 uses
    [start,end,opp0] and face1 uses [end,start,opp1].
    """
    X=np.asarray(points,dtype=np.float64);rawV=np.asarray(edge_vertices)
    if rawV.dtype.kind not in 'iu' or rawV.ndim!=2 or rawV.shape[1]!=4:
        raise ValueError("invalid oriented dihedral edge vertices")
    V=rawV.astype(np.int64,copy=False)
    if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():
        raise ValueError("finite Nx3 points required")
    if V.dtype.kind not in "iu" or V.ndim!=2 or V.shape[1]!=4 or np.any(V<0) or np.any(V>=len(X)):
        raise ValueError("invalid oriented dihedral edge vertices")
    if len(V)==0:
        return np.empty(0),np.empty((0,4,3))
    if np.any(np.diff(np.sort(V,axis=1),axis=1)==0):
        raise ValueError("dihedral edge star must contain four distinct vertices")
    p=X[V]
    start,end,o0,o1=p[:,0],p[:,1],p[:,2],p[:,3]
    edge=end-start;length=np.linalg.norm(edge,axis=1)
    if np.any(length<=np.finfo(np.float64).tiny):
        raise ValueError("zero-length dihedral edge")
    u=edge/length[:,None]
    c0=np.cross(end-start,o0-start)
    c1=np.cross(start-end,o1-end)
    l0=np.linalg.norm(c0,axis=1);l1=np.linalg.norm(c1,axis=1)
    if np.any(l0<=np.finfo(np.float64).tiny) or np.any(l1<=np.finfo(np.float64).tiny):
        raise ValueError("degenerate incident face in dihedral")
    n0=c0/l0[:,None];n1=c1/l1[:,None]
    A=np.einsum("ij,ij->i",u,np.cross(n0,n1))
    B=np.clip(np.einsum("ij,ij->i",n0,n1),-1.,1.)
    den=A*A+B*B
    if np.any(den<=np.finfo(np.float64).tiny):
        raise ValueError("undefined signed dihedral")
    angle=np.arctan2(A,B)
    gu=(B[:,None]*np.cross(n0,n1))/den[:,None]
    gn0=(B[:,None]*np.cross(n1,u)-A[:,None]*n1)/den[:,None]
    gn1=(B[:,None]*np.cross(u,n0)-A[:,None]*n0)/den[:,None]
    P=(np.eye(3)[None,:,:]-u[:,:,None]*u[:,None,:])/length[:,None,None]
    edge_end=np.einsum("mji,mj->mi",P,gu)
    grad=np.zeros((len(V),4,3),dtype=np.float64)
    grad[:,0]-=edge_end
    grad[:,1]+=edge_end
    g0=_normal_coordinate_gradients(start,end,o0,gn0)
    g1=_normal_coordinate_gradients(end,start,o1,gn1)
    grad[:,0]+=g0[:,0];grad[:,1]+=g0[:,1];grad[:,2]+=g0[:,2]
    grad[:,1]+=g1[:,0];grad[:,0]+=g1[:,1];grad[:,3]+=g1[:,2]
    if not np.isfinite(angle).all() or not np.isfinite(grad).all():
        raise ValueError("nonfinite signed dihedral or derivative")
    return angle,grad

def _sparse_vertex_jacobian(gradients,vertex_ids,free_nodes,node_count):
    G=np.asarray(gradients,dtype=np.float64);V=np.asarray(vertex_ids,dtype=np.int64)
    F=np.asarray(free_nodes,dtype=np.int64)
    if V.ndim!=2 or G.shape!=(len(V),V.shape[1],3):
        raise ValueError("invalid vertex derivative table")
    lookup=np.full(node_count,-1,dtype=np.int64);lookup[F]=np.arange(len(F))
    cols=lookup[V];valid=np.broadcast_to((cols>=0)[:,:,None],G.shape)
    rr=np.broadcast_to(np.arange(len(V))[:,None,None],G.shape)
    cc=3*cols[:,:,None]+np.arange(3)[None,None,:]
    return sparse.coo_matrix((G[valid],(rr[valid],cc[valid])),
                             shape=(len(V),3*len(F))).tocsr()

class MappedSkinSurface:
    def __init__(self, coarse_origin, detail_origin, faces, displacement_map,
                 *, area_bounds=(.5,2.), normal_cosine_min=.5,
                 max_dihedral_change_degrees=None, dihedral_edge_ids=None):
        C=np.asarray(coarse_origin);D=np.asarray(detail_origin);F=np.asarray(faces)
        if C.dtype!=np.float32 or D.dtype!=np.float32:
            raise ValueError('emitted Float32 coarse/detail origins required')
        if C.ndim!=2 or D.ndim!=2 or C.shape[1]!=3 or D.shape[1]!=3 or not len(C) or not len(D) or not np.isfinite(C).all() or not np.isfinite(D).all():
            raise ValueError('finite nonempty Nx3 origins required')
        if F.dtype.kind not in 'iu' or F.ndim!=2 or F.shape[1]!=3 or not len(F) or np.any(F<0) or np.any(F>=len(D)) or np.any(np.diff(np.sort(F,axis=1),axis=1)==0):
            raise ValueError('invalid detailed skin faces')
        W=sparse.csr_matrix(displacement_map,dtype=np.float64,copy=True)
        W.sum_duplicates();W.eliminate_zeros();W.sort_indices()
        if W.shape!=(len(D),len(C)) or not np.isfinite(W.data).all() or np.any(W.data<0):
            raise ValueError('invalid nonnegative displacement map')
        sums=np.asarray(W.sum(axis=1)).ravel();active=np.diff(W.indptr)>0
        if np.any(np.abs(sums[active]-1)>1e-6):
            raise ValueError('active displacement weights must sum to one')
        lo,hi=map(float,area_bounds);co=float(normal_cosine_min)
        if not np.isfinite([lo,hi,co]).all() or lo<=0 or hi<=lo or not 0<co<=1:
            raise ValueError('invalid detailed skin geometry gates')
        self.coarse_origin=C.copy();self.detail_origin=D.copy()
        self.faces=F.astype(np.int64,copy=True);self.weights=W
        self.area_bounds=(lo,hi);self.normal_cosine_min=co
        self.max_dihedral_change_degrees=None
        self.dihedral_edge_ids=None
        self._dihedral_vertices=None
        self._dihedral_support=None
        self._dihedral_reference=None
        if max_dihedral_change_degrees is not None:
            if isinstance(max_dihedral_change_degrees,(bool,np.bool_)):
                raise ValueError('dihedral bound must be degrees in [0,180]')
            bound=float(max_dihedral_change_degrees)
            if not np.isfinite(bound) or bound<0 or bound>180:
                raise ValueError('dihedral bound must be degrees in [0,180]')
            _,all_vertices,_,opposite=_interior_edge_topology(self.faces)
            if dihedral_edge_ids is None:
                ids=np.arange(len(all_vertices),dtype=np.int64)
            else:
                raw_ids=np.asarray(dihedral_edge_ids)
                if raw_ids.ndim!=1 or (len(raw_ids) and raw_ids.dtype.kind not in 'iu') or len(np.unique(raw_ids))!=len(raw_ids):
                    raise ValueError('dihedral edge IDs must be a unique integer vector')
                ids=raw_ids.astype(np.int64,copy=False)
                if np.any(ids<0) or np.any(ids>=len(all_vertices)):
                    raise ValueError('dihedral edge ID out of range')
                ids=np.sort(ids)
            if len(ids) and not np.all(opposite[ids]):
                raise ValueError('selected dihedral edges require opposite incident face winding')
            self.max_dihedral_change_degrees=bound
            self.dihedral_edge_ids=ids.copy()
            self._dihedral_vertices=all_vertices[ids].copy()
            self._dihedral_reference=_dihedral_values_gradients(
                self.detail_origin,self._dihedral_vertices)[0]
            support=sparse.csr_matrix((len(ids),len(C)),dtype=np.float64)
            for k in range(4):
                support=support+W[self._dihedral_vertices[:,k]]
            support=support.tocsr();support.sum_duplicates();support.eliminate_zeros();support.sort_indices()
            self._dihedral_support=support
            self.dihedral_edge_ids.flags.writeable=False
            self._dihedral_vertices.flags.writeable=False
            self._dihedral_reference.flags.writeable=False
            for array in (support.data,support.indices,support.indptr):array.flags.writeable=False
        elif dihedral_edge_ids is not None:
            raise ValueError('dihedral_edge_ids requires an enabled dihedral bound')
        for a in (self.coarse_origin,self.detail_origin,self.faces):a.flags.writeable=False
        # Import lazily to avoid a circular module dependency.
        self.metrics(C)  # Reject degenerate source faces before optimization.
        self.support=(W[self.faces[:,0]]+W[self.faces[:,1]]+W[self.faces[:,2]]).tocsr()
        self._vector_map=sparse.kron(W,sparse.eye(3,format='csr'),format='csr')
        for matrix in (self.weights,self.support,self._vector_map):
            for array in (matrix.data,matrix.indices,matrix.indptr):array.flags.writeable=False

    def positions(self, points):
        X=np.asarray(points)
        if X.shape!=self.coarse_origin.shape or X.dtype.kind not in 'fiu' or not np.isfinite(X).all():
            raise ValueError('invalid mapped-skin mechanical coordinates')
        # This is the existing double-precision CPU reference followed by F32
        # emission. Native shader accumulation parity is a separate gate.
        return (self.detail_origin.astype(np.float64)+
                self.weights@(X.astype(np.float64)-self.coarse_origin)).astype(np.float32)

    def metrics(self, points):
        from human_skin_fat_connected_repair import boundary_face_values_gradients
        return boundary_face_values_gradients(self.positions(points),self.detail_origin,self.faces)

    def dihedral_change_metrics(self, points):
        if self.max_dihedral_change_degrees is None:
            return np.empty(0,dtype=np.int64),np.empty(0,dtype=np.float64)
        if len(self.dihedral_edge_ids)==0:
            return self.dihedral_edge_ids,np.empty(0,dtype=np.float64)
        Y=self.positions(points)
        angle,_=_dihedral_values_gradients(Y,self._dihedral_vertices)
        return self.dihedral_edge_ids,_wrap_radians(angle-self._dihedral_reference)

    def dihedral_passes(self, points):
        if self.max_dihedral_change_degrees is None:
            return True
        _,delta=self.dihedral_change_metrics(points)
        return bool(np.all(np.abs(delta)<=np.deg2rad(self.max_dihedral_change_degrees)))

    def passes(self, points):
        _,_,area,cos=self.metrics(points);lo,hi=self.area_bounds
        return bool(area.min()>=lo and area.max()<=hi and
                    cos.min()>=self.normal_cosine_min and self.dihedral_passes(points))

    def _validated_free_nodes(self, free_nodes):
        F=np.asarray(free_nodes)
        if F.dtype.kind not in 'iu' or F.ndim!=1 or not len(F) or len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=len(self.coarse_origin)):
            raise ValueError('invalid free nodes for mapped skin')
        return F.astype(np.int64,copy=False)

    def _validated_face_ids(self, face_ids):
        ids=np.asarray(face_ids)
        if ids.dtype.kind not in 'iu' or ids.ndim!=1 or np.any(ids<0) or np.any(ids>=len(self.faces)) or len(np.unique(ids))!=len(ids):
            raise ValueError('invalid mapped-skin linearization face IDs')
        return ids.astype(np.int64,copy=False)

    def linearization_face_ids(self, free_nodes):
        """Faces whose mapped vertices depend on at least one free coarse node."""
        F=self._validated_free_nodes(free_nodes)
        touched=self.support[:,F].tocsr()
        return np.flatnonzero(np.diff(touched.indptr)>0).astype(np.int64)

    def dihedral_linearization_edge_ids(self, free_nodes):
        """Configured interior edges whose four mapped vertices touch free nodes."""
        if self.max_dihedral_change_degrees is None or not len(self.dihedral_edge_ids):
            return np.empty(0,dtype=np.int64)
        F=self._validated_free_nodes(free_nodes)
        touched=self._dihedral_support[:,F].tocsr()
        return self.dihedral_edge_ids[np.flatnonzero(np.diff(touched.indptr)>0)]

    def dihedral_change_linearization(self, points, free_nodes, edge_ids=None):
        """Wrapped signed-angle changes and analytic sparse Jacobian through W.

        Jacobian columns are coarse xyz coordinates in metres. Rows are omitted
        safely when every configured edge is inactive on the supplied free set.
        """
        F=self._validated_free_nodes(free_nodes)
        touching=self.dihedral_linearization_edge_ids(F)
        if edge_ids is None:
            edge_ids=touching
        else:
            supplied=np.asarray(edge_ids)
            if supplied.dtype.kind not in 'iu' or supplied.ndim!=1 or len(np.unique(supplied))!=len(supplied) or not np.isin(supplied,self.dihedral_edge_ids).all():
                raise ValueError('invalid mapped-skin linearization edge IDs')
            edge_ids=np.intersect1d(supplied,touching)
        if not len(edge_ids):
            return edge_ids,np.empty(0,dtype=np.float64),sparse.csr_matrix((0,3*len(F)))
        local=np.searchsorted(self.dihedral_edge_ids,edge_ids)
        vertices=self._dihedral_vertices[local]
        Y=self.positions(points)
        angle,grad=_dihedral_values_gradients(Y,vertices)
        delta=_wrap_radians(angle-self._dihedral_reference[local])
        detailJ=_sparse_vertex_jacobian(grad,vertices,np.arange(len(self.detail_origin)),len(self.detail_origin))
        cols=(3*F[:,None]+np.arange(3)).ravel()
        J=(detailJ@self._vector_map[:,cols]).tocsr()
        return edge_ids,delta,J

    @property
    def coupling_support(self):
        """All nonlinear surface constraints, independent of selected LP rows.

        Adjacent faces may share only fixed coarse nodes; their crease still
        couples the opposite free vertices. Never infer this graph from a
        zero-valued or active-only derivative.
        """
        if self.max_dihedral_change_degrees is None:
            return self.support
        return sparse.vstack([self.support,self._dihedral_support],format='csr')

    def _compose_face_jacobian(self, gradients, face_ids, free_nodes):
        from human_skin_fat_connected_repair import sparse_face_jacobian
        ids=np.asarray(face_ids)
        if ids.dtype.kind not in 'iu' or ids.ndim!=1 or np.any(ids<0) or np.any(ids>=len(self.faces)) or len(np.unique(ids))!=len(ids):
            raise ValueError('invalid mapped-skin linearization face IDs')
        F=self._validated_free_nodes(free_nodes)
        if gradients.shape!=(len(ids),3,3) or not np.isfinite(gradients).all():
            raise ValueError('mapped-skin face derivative shape or values are invalid')
        detailJ=sparse_face_jacobian(gradients,self.faces[ids],np.arange(len(self.detail_origin)),len(self.detail_origin))
        cols=(3*F[:,None]+np.arange(3)).ravel()
        return (detailJ@self._vector_map[:,cols]).tocsr()

    def projected_area_linearization(self, points, free_nodes, face_ids=None):
        """Projected-area Jacobian; omitted face_ids preserves the all-face API."""
        from human_skin_fat_connected_repair import boundary_face_values_gradients
        F=self._validated_free_nodes(free_nodes)
        ids=np.arange(len(self.faces),dtype=np.int64) if face_ids is None else self._validated_face_ids(face_ids)
        if not len(ids):
            return np.empty(0,dtype=np.float64),sparse.csr_matrix((0,3*len(F)))
        Y=self.positions(points)
        value,grad,_,_=boundary_face_values_gradients(Y,self.detail_origin,self.faces[ids])
        return value,self._compose_face_jacobian(grad,ids,F)

    def area_cosine_linearization(self, points, free_nodes, face_ids=None):
        """True area-ratio and oriented-normal-cosine Jacobians through W."""
        from human_skin_fat_connected_repair import boundary_area_cosine_values_gradients
        F=self._validated_free_nodes(free_nodes)
        ids=np.arange(len(self.faces),dtype=np.int64) if face_ids is None else self._validated_face_ids(face_ids)
        if not len(ids):
            empty=np.empty(0,dtype=np.float64)
            zero=sparse.csr_matrix((0,3*len(F)))
            return ids,empty,zero,empty.copy(),zero.copy()
        Y=self.positions(points)
        area,area_grad,cosine,cosine_grad=boundary_area_cosine_values_gradients(
            Y,self.detail_origin,self.faces[ids])
        areaJ=self._compose_face_jacobian(area_grad,ids,F)
        cosineJ=self._compose_face_jacobian(cosine_grad,ids,F)
        return ids.astype(np.int64,copy=False),area,areaJ,cosine,cosineJ

    def report(self, points):
        _,_,area,cos=self.metrics(points)
        report=dict(mapped_skin_faces=len(self.faces),
                    mapped_skin_active_vertices=int(np.count_nonzero(np.diff(self.weights.indptr))),
                    mapped_skin_area_ratio_min=float(area.min()),
                    mapped_skin_area_ratio_max=float(area.max()),
                    mapped_skin_normal_cosine_min=float(cos.min()),
                    mapped_skin_nonpositive_normal_faces=int(np.count_nonzero(cos<=0)),
                    mapped_skin_geometry_pass=self.passes(points),
                    mapped_skin_native_shader_parity_qualified=False)
        if self.max_dihedral_change_degrees is not None:
            _,delta=self.dihedral_change_metrics(points)
            absolute=np.abs(delta);limit=np.deg2rad(self.max_dihedral_change_degrees)
            report.update({
                'mapped_skin_dihedral_constraint_enabled':True,
                'mapped_skin_dihedral_edge_count':int(len(delta)),
                'mapped_skin_dihedral_change_bound_degrees':float(self.max_dihedral_change_degrees),
                'mapped_skin_dihedral_abs_change_max_degrees':float(np.degrees(absolute.max())) if len(delta) else 0.,
                'mapped_skin_dihedral_abs_change_p95_degrees':float(np.degrees(np.quantile(absolute,.95))) if len(delta) else 0.,
                'mapped_skin_dihedral_violations':int(np.count_nonzero(absolute>limit)),
                'mapped_skin_dihedral_geometry_pass':bool(np.all(absolute<=limit))})
        return report
