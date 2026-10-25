"""Detailed skin constraints composed with its existing coarse displacement map.

Offline connected-volume authoring only. No constitutive law, physical state,
rendering owner, or runtime format is changed.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse

class MappedSkinSurface:
    def __init__(self, coarse_origin, detail_origin, faces, displacement_map,
                 *, area_bounds=(.5,2.), normal_cosine_min=.5):
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

    def passes(self, points):
        _,_,area,cos=self.metrics(points);lo,hi=self.area_bounds
        return bool(area.min()>=lo and area.max()<=hi and cos.min()>=self.normal_cosine_min)

    def projected_area_linearization(self, points, free_nodes):
        from human_skin_fat_connected_repair import sparse_face_jacobian
        F=np.asarray(free_nodes)
        if F.dtype.kind not in 'iu' or F.ndim!=1 or not len(F) or len(np.unique(F))!=len(F) or np.any(F<0) or np.any(F>=len(self.coarse_origin)):
            raise ValueError('invalid free nodes for mapped skin')
        value,grad,_,_=self.metrics(points)
        detailJ=sparse_face_jacobian(grad,self.faces,np.arange(len(self.detail_origin)),len(self.detail_origin))
        cols=(3*F[:,None]+np.arange(3)).ravel()
        return value,(detailJ@self._vector_map[:,cols]).tocsr()

    def report(self, points):
        _,_,area,cos=self.metrics(points)
        return dict(mapped_skin_faces=len(self.faces),
                    mapped_skin_active_vertices=int(np.count_nonzero(np.diff(self.weights.indptr))),
                    mapped_skin_area_ratio_min=float(area.min()),
                    mapped_skin_area_ratio_max=float(area.max()),
                    mapped_skin_normal_cosine_min=float(cos.min()),
                    mapped_skin_nonpositive_normal_faces=int(np.count_nonzero(cos<=0)),
                    mapped_skin_geometry_pass=self.passes(points),
                    mapped_skin_native_shader_parity_qualified=False)
