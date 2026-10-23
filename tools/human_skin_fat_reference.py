"""Pinned skin/fat table identities for anatomical patch preparation.

Surface row ordinals, parent-surface identities and detailed-source identities
are distinct domains. These helpers do not certify contact or tissue mechanics.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import numpy as np

INVALID = 0xFFFFFFFF

def _ids(values):
    v=np.asarray(values)
    if v.ndim!=1 or v.dtype.kind not in "iu" or np.any(v<0) or np.any(v>np.iinfo(np.int64).max):
        raise ValueError("one-dimensional nonnegative integer identities required")
    return v.astype(np.int64,copy=True)

@dataclass(frozen=True)
class ParentSurfaceIds:
    values: np.ndarray

@dataclass(frozen=True)
class DetailedSourceFaceIds:
    values: np.ndarray

@dataclass(frozen=True)
class SurfaceFaceRows:
    values: np.ndarray
    table_sha256: str

def _oriented(tri):
    a,b,c=map(int,tri)
    return min((a,b,c),(b,c,a),(c,a,b))

@dataclass(frozen=True)
class SkinFatReference:
    outer: np.ndarray
    inner: np.ndarray
    deep: np.ndarray
    faces: np.ndarray
    parent_surface_ids: np.ndarray
    detailed_source_face_ids: np.ndarray
    direct_source_vertices: np.ndarray
    tetrahedra: np.ndarray
    materials: np.ndarray
    hashes: dict

    def _select(self, wanted, domain, kind):
        if not isinstance(wanted,kind):raise TypeError("wrong anatomical identity domain")
        values=_ids(wanted.values)
        missing=np.setdiff1d(values,domain)
        if len(missing):raise ValueError("unknown anatomical identities: "+str(missing[:8].tolist()))
        return SurfaceFaceRows(np.flatnonzero(np.isin(domain,values)),self.hashes["faces.csv"])

    def by_parent_surface(self, ids):
        return self._select(ids,self.parent_surface_ids,ParentSurfaceIds)

    def by_detailed_source_face(self, ids):
        return self._select(ids,self.detailed_source_face_ids,DetailedSourceFaceIds)

    def take_faces(self, rows):
        if not isinstance(rows,SurfaceFaceRows) or rows.table_sha256!=self.hashes["faces.csv"]:
            raise ValueError("surface rows belong to another table/build")
        ix=_ids(rows.values)
        if np.any(ix>=len(self.faces)):raise ValueError("surface row out of range")
        return self.faces[ix]

    def oriented_bijection(self, other_faces, other_source_ids):
        f=np.asarray(other_faces);s=_ids(other_source_ids)
        if f.shape!=self.faces.shape or len(s)!=len(f):raise ValueError("face inventory differs")
        if f.dtype.kind not in "iu" or np.any(f<0) or np.any(f>=len(self.outer)):
            raise ValueError("face corners must be in-domain integer identities")
        other={_oriented(t):i for i,t in enumerate(f)}
        if len(other)!=len(f):raise ValueError("duplicate oriented face identity")
        own=[_oriented(t) for t in self.faces]
        if len(set(own))!=len(own) or set(own)!=set(other):raise ValueError("oriented topology differs")
        rows=np.asarray([other[t] for t in own],dtype=np.int64)
        if not np.array_equal(s[rows],self.detailed_source_face_ids):
            raise ValueError("detailed source ancestry differs")
        return rows

def load_reference(directory, expected_hashes):
    directory=Path(directory)
    required=("nodes.csv","faces.csv","embedding.csv","tetrahedra.csv")
    if set(expected_hashes)!=set(required):raise ValueError("pin every reference table hash")
    tables={}
    for name in required:
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_hashes[name]:
            raise ValueError("reference table SHA-256 mismatch: "+name)
        tables[name]=list(csv.DictReader(raw.decode().splitlines()))
    def ordinal(rows,key):
        if [int(r[key]) for r in rows]!=list(range(len(rows))):
            raise ValueError("noncanonical row ordinals: "+key)
    ns,fs,es,ts=[tables[x] for x in required]
    ordinal(ns,"row");ordinal(fs,"face");ordinal(es,"row");ordinal(ts,"tet")
    if not ns or not fs or not ts or len(es)!=len(ns):raise ValueError("empty or inconsistent reference inventory")
    layers=[np.asarray([[float(r[k]) for k in keys] for r in ns]) for keys in
            (("ox","oy","oz"),("ix","iy","iz"),("dx","dy","dz"))]
    if any(not np.isfinite(x).all() for x in layers):raise ValueError("nonfinite reference position")
    for a,b,column in [(layers[0],layers[1],"skin_thickness_m"),(layers[1],layers[2],"fat_thickness_m")]:
        width=np.asarray([float(r[column]) for r in ns])
        if not np.isfinite(width).all() or np.any(width<=0) or not np.allclose(np.linalg.norm(a-b,axis=1),width,rtol=2e-5,atol=2e-7):
            raise ValueError("reference thickness disagrees with layer positions")
    faces=np.asarray([[int(r[k]) for k in ("a","b","c")] for r in fs],dtype=np.int64)
    tets=np.asarray([[int(r[k]) for k in ("a","b","c","d")] for r in ts],dtype=np.int64)
    if np.any(faces<0) or np.any(faces>=len(ns)) or np.any(tets<0) or np.any(tets>=3*len(ns)):
        raise ValueError("reference connectivity exceeds its domain")
    if any(len(set(t))!=3 for t in faces) or any(len(set(t))!=4 for t in tets):
        raise ValueError("repeated reference corner")
    parents=np.asarray([int(r["parent_surface_face"]) for r in fs],dtype=np.int64)
    source=np.asarray([int(r["stable_source_face"]) for r in fs],dtype=np.int64)
    direct=np.asarray([int(r["source_vertex"]) for r in es],dtype=np.int64)
    materials=np.asarray([int(r["material"]) for r in ts],dtype=np.int64)
    for x in [*layers,faces,tets,parents,source,direct,materials]:x.flags.writeable=False
    return SkinFatReference(*layers,faces,parents,source,direct,tets,materials,dict(expected_hashes))
