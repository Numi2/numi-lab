"""Compact initialization-only tetrahedron-face incidence for the existing 217 compiler.

All faces and both incident cells remain represented. Records are materialized
on access; no topology, orientation, material or nonmanifold gate is omitted.
"""
from collections.abc import Mapping, Sequence
import numpy as np

_FACE_DTYPE=np.dtype([("a","<u4"),("b","<u4"),("c","<u4")])
_LOCAL=np.asarray(((0,1,2,3),(0,3,1,2),(0,2,3,1),(1,3,2,0)),np.int64)

class _FaceRows(Sequence):
    __slots__=("table","first","last")
    def __init__(self,table,first,last):self.table=table;self.first=first;self.last=last
    def __len__(self):return self.last-self.first
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        entry=int(self.table.order[self.first+index]);cell,lane=divmod(entry,4)
        tet=self.table.tets[cell];a,b,c,opposite=_LOCAL[lane]
        face=[int(tet[a]),int(tet[b]),int(tet[c])]
        pa,pb,pc=self.table.points[face];po=self.table.points[int(tet[opposite])]
        if float(np.dot(np.cross(pb-pa,pc-pa),po-pa))>0.0:face[1],face[2]=face[2],face[1]
        return {"element_row":cell,"element_id":int(self.table.elements[cell]),
                "oriented_nodes":tuple(face),"material_index":int(self.table.materials[cell])}

class CompactTetFaceMap(Mapping):
    def __init__(self,mesh):
        self.points=np.asarray(mesh["points"],dtype=np.float32).astype(np.float64)
        self.tets=np.asarray(mesh["tetrahedra"],dtype=np.int64)
        self.elements=mesh["element_ids"];self.materials=mesh["material_indices"]
        if self.tets.size and (self.tets.min()<0 or self.tets.max()>=len(self.points) or self.tets.max()>np.iinfo(np.uint32).max):
            raise ValueError("tetrahedron face incidence node index outside UInt32 point domain")
        faces=np.sort(self.tets[:,_LOCAL[:,:3]].reshape(-1,3),axis=1).astype("<u4")
        raw=np.ascontiguousarray(faces).view(_FACE_DTYPE).reshape(-1)
        order=np.argsort(raw,kind="stable")
        ordered=raw[order]
        self.starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1].astype(np.int64) if len(ordered) else np.empty(0,np.int64)
        self.ends=np.r_[self.starts[1:],len(ordered)].astype(np.int64) if len(ordered) else np.empty(0,np.int64)
        self.keys=ordered[self.starts]
        self.order=order.astype(np.uint32) if len(order)<=np.iinfo(np.uint32).max else order
        bad=np.flatnonzero(self.ends-self.starts>2)
        if len(bad):
            # Match the old insertion-order witness, rather than a sort-dependent one.
            badrow=int(bad[np.argmin(self.order[self.starts[bad]])])
            key=tuple(map(int,self.keys[badrow]))
            raise ValueError(f"TetGen mesh has a nonmanifold tetrahedron face {key}")
    def __len__(self):return len(self.keys)
    def __iter__(self):
        for key in self.keys:yield tuple(map(int,key))
    def __getitem__(self,key):
        if not isinstance(key,tuple) or len(key)!=3 or any(not isinstance(x,(int,np.integer)) or x<0 or x>np.iinfo(np.uint32).max for x in key):
            raise KeyError(key)
        query=np.asarray(key,dtype=_FACE_DTYPE)
        row=int(np.searchsorted(self.keys,query))
        if row==len(self.keys) or self.keys[row]!=query:raise KeyError(key)
        return _FaceRows(self,int(self.starts[row]),int(self.ends[row]))
    def items(self):
        for row,key in enumerate(self.keys):
            yield tuple(map(int,key)),_FaceRows(self,int(self.starts[row]),int(self.ends[row]))
