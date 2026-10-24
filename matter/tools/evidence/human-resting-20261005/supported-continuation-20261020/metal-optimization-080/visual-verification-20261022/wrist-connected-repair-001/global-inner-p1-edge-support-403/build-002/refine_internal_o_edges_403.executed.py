#!/usr/bin/env python3
"""Conformingly bisect all internal tetra edges with two physical-O endpoints.

The input shell coordinates/faces and physical O node IDs remain unchanged.
New edge midpoints are ordinary volume nodes with averaged source ancestry,
not physical-shell nodes or zero constraints. This emits topology/ancestry only;
no P1 field solve or physical/native admission is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable

import numpy as np

BASE = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
INPUT = BASE / 'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz'
PARENT_MAP = BASE / 'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/material-parent-014-conservative-map.npz'
EXPECTED_INPUT_SHA256 = '8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890'
EXPECTED_PARENT_MAP_SHA256 = 'e4ac929347673e55082c8bf2b4fa239617c18ac297195c18fae60ce8602bf38d'
EDGE_PAIRS = tuple(itertools.combinations(range(4), 2))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signed_det(points: list[np.ndarray] | np.ndarray, tet: Iterable[int]) -> float:
    ids = tuple(map(int, tet))
    q = np.asarray([points[i] for i in ids], dtype=np.float64)
    return float(np.linalg.det(np.stack((q[1]-q[0], q[2]-q[0], q[3]-q[0]), axis=0)))


def orient_positive(points: list[np.ndarray], tet: Iterable[int]) -> tuple[int, int, int, int]:
    out = list(map(int, tet))
    d = signed_det(points, out)
    if not np.isfinite(d) or d == 0.0:
        raise ValueError(f'zero/nonfinite child determinant: tet={out} det={d}')
    if d < 0.0:
        out[0], out[1] = out[1], out[0]
        d = -d
    if d <= 0.0:
        raise ValueError(f'nonpositive oriented child determinant: tet={out} det={d}')
    return tuple(out)


def tet_edges(tet: Iterable[int]) -> tuple[tuple[int, int], ...]:
    t = tuple(map(int, tet))
    return tuple((min(t[i], t[j]), max(t[i], t[j])) for i, j in EDGE_PAIRS)


def find_internal_outer_edges(tets: np.ndarray, outer_node_ids: np.ndarray, shell_faces: np.ndarray) -> tuple[list[tuple[int, int]], dict]:
    tets = np.asarray(tets, dtype=np.int64)
    n = max(int(tets.max()) + 1, int(np.max(outer_node_ids)) + 1)
    is_outer = np.zeros(n, dtype=np.bool_)
    is_outer[np.asarray(outer_node_ids, dtype=np.int64)] = True
    pairs = np.concatenate([tets[:, [i, j]] for i, j in EDGE_PAIRS], axis=0)
    pairs.sort(axis=1)
    keep = is_outer[pairs[:, 0]] & is_outer[pairs[:, 1]]
    outer_edges = np.unique(pairs[keep], axis=0)
    sf = np.asarray(shell_faces, dtype=np.int64)
    shell_edges = np.concatenate([sf[:, [0, 1]], sf[:, [0, 2]], sf[:, [1, 2]]], axis=0)
    shell_edges.sort(axis=1)
    shell_edges = np.unique(shell_edges, axis=0)
    shell_set = {tuple(map(int, e)) for e in shell_edges}
    selected = [tuple(map(int, e)) for e in outer_edges if tuple(map(int, e)) not in shell_set]
    return selected, {'outer_endpoint_edges_in_tets': int(len(outer_edges)), 'shell_edge_count': int(len(shell_edges)), 'internal_outer_edge_count': int(len(selected))}


def oriented_boundary_faces(tets: np.ndarray) -> np.ndarray:
    t = np.asarray(tets, dtype=np.int64)
    faces = np.concatenate((t[:, [1, 2, 3]], t[:, [0, 3, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 1]]), axis=0)
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError(f'nonmanifold tetra facet incidence: max={int(counts.max())}')
    return faces[first[counts == 1]]


def canonical_oriented_faces(faces: np.ndarray) -> list[tuple[int, int, int]]:
    out=[]
    for row in np.asarray(faces, dtype=np.int64):
        k=int(np.argmin(row))
        out.append(tuple(int(row[(k+i)%3]) for i in range(3)))
    return out


def compare_oriented_boundary(expected: np.ndarray, actual: np.ndarray) -> dict:
    e=canonical_oriented_faces(expected)
    a=canonical_oriented_faces(actual)
    em={tuple(sorted(x)):x for x in e}
    am={tuple(sorted(x)):x for x in a}
    return {'face_count_expected':len(e),'face_count_actual':len(a),'exact_unoriented_set':set(em)==set(am),'oriented_mismatch_count':sum(em[k]!=am.get(k) for k in em) + len(set(am)-set(em)),'missing_count':len(set(em)-set(am)),'extra_count':len(set(am)-set(em))}


def split_edges_globally(points_f32: np.ndarray, tets: np.ndarray, marked_edges: Iterable[tuple[int,int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split every current cell incident to each original edge, in sorted order.

    A dynamic edge-to-active-tetra map propagates each split through both sides
    of every shared face. Multiple marked edges in one parent are handled by
    updating incidence after every local split, avoiding a whole-mesh rescan.
    Returns points, active tets, each output tet's original-row parent, and the
    input endpoint pair for each appended midpoint node.
    """
    X=np.asarray(points_f32)
    T=np.asarray(tets,dtype=np.int64)
    if X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
        raise ValueError('points must be finite Nx3 Float32')
    if T.ndim != 2 or T.shape[1] != 4 or np.min(T)<0 or np.max(T)>=len(X):
        raise ValueError('invalid tetrahedron table')
    marked=sorted(set((min(int(a),int(b)),max(int(a),int(b))) for a,b in marked_edges))
    if any(a==b for a,b in marked): raise ValueError('zero-length topology edge')
    marks=set(marked)
    records=[tuple(map(int,row)) for row in T]
    active=bytearray([1])*len(records)
    parent=list(range(len(records)))
    points=list(X)
    incidence={edge:set() for edge in marked}
    for ti,tet in enumerate(records):
        for edge in tet_edges(tet):
            if edge in marks: incidence[edge].add(ti)
    absent=[edge for edge in marked if not incidence[edge]]
    if absent: raise ValueError(f'marked edges absent from tetrahedra: {absent[:8]}')
    edge_midpoints=[]
    for edge_index,edge in enumerate(marked):
        incident=sorted(incidence[edge])
        if not incident: raise ValueError(f'marked edge lost before split: {edge}')
        a,b=edge
        midpoint=np.asarray((np.asarray(points[a],dtype=np.float64)+np.asarray(points[b],dtype=np.float64))*0.5,dtype=np.float32)
        if np.array_equal(midpoint,points[a]) or np.array_equal(midpoint,points[b]):
            raise ValueError(f'Float32 midpoint collapsed to endpoint: edge={edge}')
        midrow=len(points)
        points.append(midpoint)
        edge_midpoints.append(edge)
        for oldrow in incident:
            if not active[oldrow]: raise ValueError(f'inactive tet in edge incidence: edge={edge} tet={oldrow}')
            old=records[oldrow]
            others=[v for v in old if v not in edge]
            if len(others)!=2: raise ValueError(f'edge/tet incidence malformed: edge={edge} tet={oldrow}')
            # Remove parent references for every still-pending marked edge.
            for e in tet_edges(old):
                if e in marks: incidence[e].discard(oldrow)
            active[oldrow]=0
            c,d=others
            children=(orient_positive(points,(a,midrow,c,d)),orient_positive(points,(midrow,b,c,d)))
            for child in children:
                newrow=len(records)
                records.append(child); parent.append(parent[oldrow]); active.append(1)
                for e in tet_edges(child):
                    if e in marks: incidence[e].add(newrow)
        if incidence[edge]:
            raise ValueError(f'edge not fully removed after conforming split: {edge} has {len(incidence[edge])} incident cells')
        if (edge_index+1)%1000==0:
            pass
    live=[i for i,on in enumerate(active) if on]
    outT=np.asarray([records[i] for i in live],dtype=np.int64)
    outP=np.asarray([parent[i] for i in live],dtype=np.int64)
    outX=np.asarray(points,dtype=np.float32)
    return outX,outT,outP,np.asarray(edge_midpoints,dtype=np.int64)


def tet_faces(tet: Iterable[int]) -> tuple[tuple[int,int,int], ...]:
    t=tuple(map(int,tet))
    return tuple(tuple(sorted((t[i],t[j],t[k]))) for i,j,k in ((1,2,3),(0,2,3),(0,1,3),(0,1,2)))


def find_internal_all_o_faces(tets: np.ndarray, outer_node_ids: np.ndarray) -> tuple[list[tuple[int,int,int]], dict]:
    T=np.asarray(tets,dtype=np.int64)
    is_o=np.zeros(max(int(T.max()),int(np.max(outer_node_ids)))+1,dtype=np.bool_)
    is_o[np.asarray(outer_node_ids,dtype=np.int64)]=True
    faces=np.concatenate((T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]),axis=0)
    keys=np.sort(faces,axis=1)
    unique,counts=np.unique(keys,axis=0,return_counts=True)
    selected=unique[(counts==2)&is_o[unique].all(axis=1)]
    return [tuple(map(int,f)) for f in selected],{'internal_all_o_face_count':len(selected),'internal_face_count':int(np.count_nonzero(counts==2))}


def split_internal_o_faces_globally(points_f32: np.ndarray, tets: np.ndarray, parent_rows: np.ndarray, marked_faces: Iterable[tuple[int,int,int]]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Stellar-split every selected shared face while propagating incidence.

    Each all-O internal triangle receives one free Float32 centroid and both
    incident tetrahedra are split into three cells. Existing boundary facets
    remain exact; neighboring selected faces are tracked dynamically when a
    cell has already been replaced by a prior face split.
    """
    X=np.asarray(points_f32)
    T=np.asarray(tets,dtype=np.int64)
    parents=np.asarray(parent_rows,dtype=np.int64)
    if X.dtype!=np.float32 or X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():
        raise ValueError('points must be finite Nx3 Float32')
    if T.ndim!=2 or T.shape[1]!=4 or parents.shape!=(len(T),):
        raise ValueError('invalid tetrahedron or parent table')
    marked=sorted(set(tuple(sorted(map(int,f))) for f in marked_faces))
    marks=set(marked)
    records=[tuple(map(int,row)) for row in T]
    active=bytearray([1])*len(records)
    out_parent=parents.tolist()
    points=list(X)
    incidence={face:set() for face in marked}
    for ti,tet in enumerate(records):
        for face in tet_faces(tet):
            if face in marks: incidence[face].add(ti)
    absent=[face for face in marked if len(incidence[face])!=2]
    if absent: raise ValueError(f'selected face does not have exactly two incident cells: {[(f,len(incidence[f])) for f in absent[:8]]}')
    centroid_rows=[]
    for face in marked:
        incident=sorted(incidence[face])
        if len(incident)!=2: raise ValueError(f'selected face incidence changed unexpectedly: {face} {incident}')
        a,b,c=face
        centroid=np.asarray((X[a].astype(np.float64)+X[b].astype(np.float64)+X[c].astype(np.float64))/3.0,dtype=np.float32)
        if any(np.array_equal(centroid,X[v]) for v in face):
            raise ValueError(f'Float32 face centroid collapsed to vertex: face={face}')
        mid=len(points);points.append(centroid);centroid_rows.append(mid)
        for oldrow in incident:
            if not active[oldrow]: raise ValueError(f'inactive tet in face incidence: face={face} tet={oldrow}')
            old=records[oldrow]
            opposite=[v for v in old if v not in face]
            if len(opposite)!=1: raise ValueError(f'face/tet incidence malformed: face={face} tet={oldrow}')
            d=opposite[0]
            for f in tet_faces(old):
                if f in marks: incidence[f].discard(oldrow)
            active[oldrow]=0
            children=(orient_positive(points,(a,b,mid,d)),orient_positive(points,(b,c,mid,d)),orient_positive(points,(c,a,mid,d)))
            for child in children:
                newrow=len(records)
                records.append(child);out_parent.append(int(out_parent[oldrow]));active.append(1)
                for f in tet_faces(child):
                    if f in marks: incidence[f].add(newrow)
        if incidence[face]: raise ValueError(f'all incident cells were not split for face {face}: {sorted(incidence[face])[:8]}')
    live=[i for i,on in enumerate(active) if on]
    outT=np.asarray([records[i] for i in live],dtype=np.int64)
    outP=np.asarray([out_parent[i] for i in live],dtype=np.int64)
    outX=np.asarray(points,dtype=np.float32)
    return outX,outT,outP,np.asarray(centroid_rows,dtype=np.int64)


def append_weighted_source_csr(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, support_vertices: np.ndarray, support_weights: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    ptr=np.asarray(indptr,dtype=np.int64);ids=np.asarray(indices,dtype=np.int64);w=np.asarray(weights,dtype=np.float64)
    vertices=np.asarray(support_vertices,dtype=np.int64);coeff=np.asarray(support_weights,dtype=np.float64)
    if vertices.ndim!=2 or coeff.shape!=vertices.shape or ptr.ndim!=1 or ptr[0]!=0 or ptr[-1]!=len(ids) or len(ids)!=len(w):
        raise ValueError('invalid support ancestry CSR or interpolation arrays')
    if not np.isfinite(coeff).all() or np.any(coeff<0) or np.any(np.abs(coeff.sum(axis=1)-1.0)>2e-12):
        raise ValueError('support interpolation weights must be finite nonnegative and sum to one')
    new_ids=ids.tolist();new_w=w.tolist();new_ptr=ptr.tolist();errors=[]
    for row,weights_row in zip(vertices,coeff):
        accum={}
        for node,alpha in zip(row,weights_row):
            lo,hi=int(ptr[int(node)]),int(ptr[int(node)+1])
            for sid,sw in zip(ids[lo:hi],w[lo:hi]):
                accum[int(sid)]=accum.get(int(sid),0.0)+float(alpha)*float(sw)
        keys=sorted(accum);vals=[accum[k] for k in keys]
        if not keys or not np.isfinite(vals).all() or min(vals)<0: raise ValueError('empty/invalid support source ancestry')
        total=float(sum(vals));errors.append(abs(total-1.0))
        if abs(total-1.0)>2e-12: raise ValueError(f'support source ancestry does not sum to one: {total}')
        new_ids.extend(keys);new_w.extend(vals);new_ptr.append(len(new_ids))
    return np.asarray(new_ptr,np.int64),np.asarray(new_ids,np.int64),np.asarray(new_w,np.float64),{'new_rows':len(vertices),'max_abs_weight_sum_error':float(max(errors)) if errors else 0.0,'new_source_nonzeros':len(new_ids)-len(ids)}


def append_midpoint_source_csr(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, edge_endpoints: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    ptr=np.asarray(indptr,dtype=np.int64); ids=np.asarray(indices,dtype=np.int64); w=np.asarray(weights,dtype=np.float64)
    if ptr.ndim!=1 or ids.ndim!=1 or w.shape!=ids.shape or ptr[0]!=0 or ptr[-1]!=len(ids) or len(ptr)==0 or np.any(np.diff(ptr)<0):
        raise ValueError('invalid input source ancestry CSR')
    out_ids=ids.tolist();out_w=w.tolist();out_ptr=ptr.tolist()
    row_sums=[]
    for a,b in np.asarray(edge_endpoints,dtype=np.int64):
        accum={}
        for node in (int(a),int(b)):
            lo,hi=int(ptr[node]),int(ptr[node+1])
            for sid,sw in zip(ids[lo:hi],w[lo:hi]):
                accum[int(sid)]=accum.get(int(sid),0.0)+0.5*float(sw)
        keys=sorted(accum)
        vals=[accum[k] for k in keys]
        if not keys or not np.isfinite(vals).all() or min(vals)<0:
            raise ValueError(f'empty/invalid midpoint source ancestry for edge {(int(a),int(b))}')
        s=float(sum(vals))
        if abs(s-1.0)>2e-12:
            raise ValueError(f'midpoint source ancestry does not sum to one for edge {(int(a),int(b))}: {s}')
        out_ids.extend(keys);out_w.extend(vals);out_ptr.append(len(out_ids));row_sums.append(s)
    return np.asarray(out_ptr,np.int64),np.asarray(out_ids,np.int64),np.asarray(out_w,np.float64),{'new_rows':len(edge_endpoints),'max_abs_weight_sum_error':float(max(abs(x-1.0) for x in row_sums)) if row_sums else 0.0,'new_source_nonzeros':len(out_ids)-len(ids)}


def tetra_metrics(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    P=np.asarray(points,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
    V=np.linalg.det(np.stack((P[:,1]-P[:,0],P[:,2]-P[:,0],P[:,3]-P[:,0]),axis=2))
    maxedge=np.zeros(len(P),dtype=np.float64)
    for i,j in EDGE_PAIRS:
        maxedge=np.maximum(maxedge,np.linalg.norm(P[:,i]-P[:,j],axis=1))
    q=V/np.maximum(maxedge,1e-300)**3
    return V,q


def conservative_child_fractions(child_volumes: np.ndarray, parent_rows: np.ndarray, parent_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize measured positive child volumes within each input parent."""
    volumes=np.asarray(child_volumes,dtype=np.float64)
    parents=np.asarray(parent_rows,dtype=np.int64)
    if volumes.ndim!=1 or parents.shape!=volumes.shape or np.any(parents<0) or np.any(parents>=parent_count):
        raise ValueError("malformed child volumes or parent rows")
    if not np.isfinite(volumes).all() or np.any(volumes<=0):
        raise ValueError("child volumes must be finite and positive")
    totals=np.bincount(parents,weights=volumes,minlength=parent_count)
    if np.any(totals<=0):
        raise ValueError("one or more parents have no child volume")
    fractions=volumes/totals[parents]
    closure=np.bincount(parents,weights=fractions,minlength=parent_count)
    return fractions,totals,closure


def internal_o_edge_and_cell_counts(tets: np.ndarray, original_o_ids: np.ndarray, shell_faces: np.ndarray) -> dict:
    T=np.asarray(tets,dtype=np.int64)
    maxn=int(T.max())+1
    is_o=np.zeros(maxn,dtype=np.bool_);is_o[np.asarray(original_o_ids,dtype=np.int64)]=True
    pairs=np.concatenate([T[:,[i,j]] for i,j in EDGE_PAIRS],axis=0);pairs.sort(axis=1)
    pairs=pairs[is_o[pairs[:,0]]&is_o[pairs[:,1]]]
    uniq,counts=np.unique(pairs,axis=0,return_counts=True)
    sf=np.asarray(shell_faces,dtype=np.int64)
    se=np.concatenate([sf[:,[0,1]],sf[:,[0,2]],sf[:,[1,2]]],axis=0);se.sort(axis=1);se=np.unique(se,axis=0)
    shellset={tuple(map(int,x)) for x in se}
    residual_edges=[tuple(map(int,x)) for x in uniq if tuple(map(int,x)) not in shellset]
    faces=np.concatenate((T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]),axis=0)
    keys=np.sort(faces,axis=1)
    _,first,fc=np.unique(keys,axis=0,return_index=True,return_counts=True)
    boundary=fc[first] if False else None
    # NumPy unique returns first indices aligned to unique keys; use counts directly.
    face_keys,face_first,face_counts=np.unique(keys,axis=0,return_index=True,return_counts=True)
    internal=face_keys[face_counts==2]
    all_o_internal_faces=int(np.count_nonzero(is_o[internal].all(axis=1)))
    all_o_tets=int(np.count_nonzero(is_o[T].all(axis=1)))
    return {'remaining_internal_original_o_o_edge_count':len(residual_edges),'remaining_internal_original_o_o_edges_first16':[list(e) for e in residual_edges[:16]],'remaining_internal_all_o_face_count':all_o_internal_faces,'remaining_all_o_tet_count':all_o_tets,'original_o_o_edge_incidence_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(counts,return_counts=True))}}


def build(input_path: Path, output_dir: Path, expected_input_sha: str, parent_map_path: Path, expected_parent_map_sha: str) -> dict:
    t0=time.monotonic()
    input_path=input_path.resolve();parent_map_path=parent_map_path.resolve();output_dir=output_dir.resolve()
    if sha256(input_path)!=expected_input_sha: raise ValueError(f'398 input SHA mismatch: {sha256(input_path)} != {expected_input_sha}')
    if sha256(parent_map_path)!=expected_parent_map_sha: raise ValueError(f'398 parent-map SHA mismatch: {sha256(parent_map_path)} != {expected_parent_map_sha}')
    output_dir.mkdir(parents=True,exist_ok=True)
    candidate=output_dir/'candidate.npz';report_path=output_dir/'refinement-report.json'
    if candidate.exists() or report_path.exists(): raise FileExistsError(f'refusing to overwrite existing 403 output: {output_dir}')
    z=np.load(input_path,allow_pickle=False)
    required={'points_world_m','tetrahedra','region_attributes','physical_outer_node_ids','physical_shell_faces','physical_outer_faces','construction_closure_faces','node_source_indptr','node_source_indices','node_source_weights'}
    missing=required-set(z.files)
    if missing: raise ValueError(f'398 missing required arrays: {sorted(missing)}')
    X=z['points_world_m'];T=z['tetrahedra'].astype(np.int64);attrs=z['region_attributes'].astype(np.int64)
    O=z['physical_outer_node_ids'].astype(np.int64);shell=z['physical_shell_faces'].astype(np.int64)
    if X.dtype!=np.float32 or X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all(): raise ValueError('398 points are not finite Float32 Nx3')
    if T.shape!=(len(attrs),4) or np.min(T)<0 or np.max(T)>=len(X): raise ValueError('398 tetra/material arrays malformed')
    if np.any(np.linalg.det((X[T][:,1:].astype(np.float64)-X[T][:,:1].astype(np.float64)).transpose(0,2,1))<=0):
        raise ValueError('398 contains nonpositive tetrahedra before refinement')
    # Prove the declared shell equals the full current tetra boundary and has
    # the same orientation before treating any edge as interior support.
    b0=oriented_boundary_faces(T);boundary0=compare_oriented_boundary(shell,b0)
    if not boundary0['exact_unoriented_set'] or boundary0['oriented_mismatch_count']:
        raise ValueError(f'398 shell/boundary preflight failed: {boundary0}')
    edges,edge_summary=find_internal_outer_edges(T,O,shell)
    # Explicitly ensure none of these edge keys lie on any actual exterior face.
    b_edges={tuple(sorted((int(f[i]),int(f[j])))) for f in b0 for i,j in ((0,1),(0,2),(1,2))}
    on_boundary=[e for e in edges if e in b_edges]
    if on_boundary: raise ValueError(f'internal O edge selector included actual boundary edges: {on_boundary[:8]}')
    # Per-input-edge lengths and marked-edge multi-incidence statistics.
    E=np.asarray(edges,dtype=np.int64)
    edge_lengths=np.linalg.norm(X[E[:,0]].astype(np.float64)-X[E[:,1]].astype(np.float64),axis=1)
    edge_set=set(edges)
    marked_per_tet=np.zeros(len(T),dtype=np.int8)
    for i,tet in enumerate(T):
        marked_per_tet[i]=sum(e in edge_set for e in tet_edges(tet))
    initialV,initialQ=tetra_metrics(X,T)
    X2,T2,parent398,edge_endpoints=split_edges_globally(X,T,edges)
    # New edge midpoint node IDs are allocated in the sorted edge order.
    midpoint_rows=np.arange(len(X),len(X)+len(edges),dtype=np.int64)
    if len(edge_endpoints)!=len(edges) or not np.array_equal(edge_endpoints,E): raise AssertionError('midpoint/edge allocation order mismatch')
    ptr,srcids,srcweights,lineage_edge=append_midpoint_source_csr(z['node_source_indptr'],z['node_source_indices'],z['node_source_weights'],E)
    face_support_faces,face_summary=find_internal_all_o_faces(T2,O)
    X3,T3,parent398,face_centroid_rows=split_internal_o_faces_globally(X2,T2,parent398,face_support_faces)
    face_vertices=np.asarray(face_support_faces,dtype=np.int64).reshape((-1,3))
    face_weights=np.full(face_vertices.shape,1.0/3.0,dtype=np.float64)
    ptr,srcids,srcweights,lineage_face=append_weighted_source_csr(ptr,srcids,srcweights,face_vertices,face_weights)
    lineage={'edge_midpoints':lineage_edge,'face_centroids':lineage_face}
    # The original 398 coordinate/source prefix is immutable.
    if not np.array_equal(X3[:len(X)],X): raise AssertionError('398 Float32 coordinate prefix changed')
    if not np.array_equal(ptr[:len(z['node_source_indptr'])],z['node_source_indptr']): raise AssertionError('398 source CSR pointer prefix changed')
    if not np.array_equal(srcids[:len(z['node_source_indices'])],z['node_source_indices']) or not np.array_equal(srcweights[:len(z['node_source_weights'])],z['node_source_weights']): raise AssertionError('398 source CSR prefix changed')
    V2,Q2=tetra_metrics(X3,T3)
    if not np.isfinite(V2).all() or np.any(V2<=0): raise ValueError(f'nonpositive/nonfinite 403 children: {int((~np.isfinite(V2)|(V2<=0)).sum())}')
    oldvol=initialV/6.0;newvol=V2/6.0
    frac,parent_volume,parent_sum=conservative_child_fractions(newvol,parent398,len(T))
    parent_sum_error=float(np.max(np.abs(parent_sum-1.0)))
    raw_parent_ratio=parent_volume/oldvol
    max_parent_volume_rel_error=float(np.max(np.abs(raw_parent_ratio-1.0)))
    raw_child_fraction=newvol/oldvol[parent398]
    global_volume_delta=float(newvol.sum()-oldvol.sum())
    global_volume_rel=abs(global_volume_delta)/float(oldvol.sum())
    parent_child_count=np.bincount(parent398,minlength=len(T))
    new_support_child_mask=parent_child_count[parent398]>1
    b1=oriented_boundary_faces(T3);boundary=compare_oriented_boundary(shell,b1)
    shell_arrays_unchanged=np.array_equal(z['physical_shell_faces'],shell) and np.array_equal(z['physical_outer_faces'],z['physical_outer_faces']) and np.array_equal(z['construction_closure_faces'],z['construction_closure_faces']) and np.array_equal(O,z['physical_outer_node_ids'])
    support=internal_o_edge_and_cell_counts(T3,O,shell)
    # Float32 midpoint displacement from the exact mathematical edge midpoint
    exact=(X[E[:,0]].astype(np.float64)+X[E[:,1]].astype(np.float64))*0.5
    actual=X3[midpoint_rows].astype(np.float64)
    delta=actual-exact
    off=np.linalg.norm(np.cross(actual-X[E[:,0]].astype(np.float64),X[E[:,1]].astype(np.float64)-X[E[:,0]].astype(np.float64)),axis=1)/np.maximum(edge_lengths,1e-300)
    metrics={'input_tet_count':len(T),'output_tet_count':len(T3),'input_node_count':len(X),'output_node_count':len(X3),'inserted_midpoint_count':len(midpoint_rows),'internal_o_o_edges_selected':len(edges),'internal_all_o_faces_support_split':len(face_support_faces),'input_residual_all_o_face_summary':face_summary,'face_centroid_count':len(face_centroid_rows),'face_support_parent_faces':face_vertices.tolist(),'face_support_centroid_node_rows':face_centroid_rows.tolist(),'all_selected_edges_split':True,'selected_edge_length_mm':{'min':float(edge_lengths.min()*1000),'median':float(np.median(edge_lengths)*1000),'max':float(edge_lengths.max()*1000)},'selected_edge_incident_tet_count_total':int(marked_per_tet.sum()),'touched_input_tets':int(np.count_nonzero(marked_per_tet)),'touched_input_tets_with_multiple_marked_edges':int(np.count_nonzero(marked_per_tet>1)),'multi_marked_edges_per_parent_distribution':{str(int(k)):int(v) for k,v in zip(*np.unique(marked_per_tet[marked_per_tet>0],return_counts=True))},'midpoint_max_rounding_displacement_m':float(np.linalg.norm(delta,axis=1).max()),'midpoint_max_off_edge_distance_m':float(off.max()),'tetra_positive_finite':bool(np.isfinite(V2).all() and np.all(V2>0)),'input_min_normalized_determinant':float(initialQ.min()),'output_min_normalized_determinant':float(Q2.min()),'output_below_1e_5_count':int(np.count_nonzero(Q2<1e-5)),'output_below_1e_5_impacted_by_new_support_count':int(np.count_nonzero((Q2<1e-5)&new_support_child_mask)),'input_total_volume_m3':float(oldvol.sum()),'output_total_volume_m3':float(newvol.sum()),'global_volume_delta_m3':global_volume_delta,'global_volume_relative_error':global_volume_rel,'max_abs_raw_398_parent_volume_ratio_error':max_parent_volume_rel_error,'max_398_parent_conservative_fraction_closure_error':parent_sum_error,'398_parent_conservative_fraction_min_max':[float(frac.min()),float(frac.max())],'398_parent_raw_child_volume_ratio_min_max':[float(raw_child_fraction.min()),float(raw_child_fraction.max())],'all_398_parents_covered':bool(np.all(parent_sum>0)),'boundary':boundary,'shell_and_typed_surface_arrays_unchanged':False,'node_source_csr':lineage,'source_prefix_byte_exact':True,'physical_outer_node_ids_unchanged':True,'edge_support_residuals':support,'field_transfer':'No P1 field values are emitted for new midpoint rows. They are free volume support DOFs, not zero-valued physical O constraints; a subsequent field solve/reinitialization is required.','parent_transfer_method':'each 403 child maps to one input 398 tet; conservative weights normalize measured child volumes within each parent. Raw child-volume/398-parent-volume ratios and parent volume drift are reported separately. Existing 398-to-014 map remains a separate owner input and is composed using conservative weights. This is topology ancestry, not a recomputed geometric overlap certificate.','native_admission':False}
    # Compose conservative 398->014 fractions by distributing each parent's
    # inherited row according to actual 403 child volume fraction. This retains
    # the existing parent-column totals, while its meaning stays conservative
    # inheritance rather than a fresh geometric-overlap measurement.
    with np.load(parent_map_path,allow_pickle=False) as pm:
        po=pm['overlap_child_offsets'].astype(np.int64);p014=pm['overlap_parent_original_input_tet_row'].astype(np.int64);pf=pm['overlap_parent_conservative_fraction'].astype(np.float64)
    if len(po)!=len(T)+1 or po[0]!=0 or po[-1]!=len(p014) or len(pf)!=len(p014): raise ValueError('398->014 conservative map does not index every 398 tet')
    offsets=[0];parent014=[];frac014=[]
    for child,parent_row,fraction in zip(range(len(T3)),parent398,frac):
        lo,hi=int(po[parent_row]),int(po[parent_row+1])
        parent014.extend(p014[lo:hi].tolist())
        frac014.extend((pf[lo:hi]*fraction).tolist())
        offsets.append(len(parent014))
    parent014=np.asarray(parent014,np.int64);frac014=np.asarray(frac014,np.float64);offsets=np.asarray(offsets,np.int64)
    source_014_count=int(p014.max())+1
    sums014=np.bincount(parent014,weights=frac014,minlength=source_014_count)
    transfer_col_error=float(np.max(np.abs(sums014-1.0)))
    metrics['inherited_398_to_014_conservative_map']={'source_map_sha256':sha256(parent_map_path),'source_map_path':str(parent_map_path),'entry_count':len(parent014),'child_csr_offsets_shape':list(offsets.shape),'parent_column_count':source_014_count,'max_parent_column_sum_error':transfer_col_error,'method':'scale each 398 conservative parent contribution by measured 403 child/398 parent volume fraction','fresh_geometric_overlap_recomputed':False}
    # Preserve only arrays whose meaning is unchanged. Old 398-local overlap
    # arrays are intentionally not copied as if they indexed the refined mesh.
    out={k:z[k] for k in ('physical_outer_faces','physical_shell_faces','construction_closure_faces','physical_outer_node_ids')}
    out.update(points_world_m=X3,tetrahedra=T3,region_attributes=attrs[parent398],node_source_indptr=ptr,node_source_indices=srcids,node_source_weights=srcweights,child_source_tetrahedron_rows=parent398,parent_398_tet_row=parent398,child_volume_fraction_of_parent=frac,parent_398_conservative_fraction=frac,child_raw_volume_ratio_to_398_parent=raw_child_fraction,edge_support_parent_endpoints=E,edge_support_midpoint_node_rows=midpoint_rows,face_support_parent_vertices=face_vertices,face_support_centroid_node_rows=face_centroid_rows,material_parent_child_offsets=offsets,material_parent_014_tet_rows=parent014,material_parent_014_conservative_fraction=frac014)
    report={'schema':'numi.global-inner-p1-internal-o-edge-support-403.report.v1','status':'edge-support-refinement-candidate-unadmitted','input_path':str(input_path),'input_sha256':sha256(input_path),'parent_398_to_014_map_path':str(parent_map_path),'parent_398_to_014_map_sha256':sha256(parent_map_path),'output_dir':str(output_dir),'candidate_path':str(candidate),'child_parent_alias':'parent_398_tet_row (identical to child_source_tetrahedron_rows)','builder_path':str(Path(__file__).resolve()),'builder_sha256':sha256(Path(__file__).resolve()),'algorithm':'deterministic sequential globally-conforming tetra edge bisection for all internal physical-O/O edges, followed by stellar centroid splits of every residual internal all-physical-O face; dynamic edge/face-to-cell incidence updates handle multiple selected supports in shared cells','input_shell_faces_sha256':hashlib.sha256(np.asarray(shell,dtype=np.int64).tobytes()).hexdigest(),'input_tetrahedra_sha256':hashlib.sha256(np.asarray(T,dtype=np.int64).tobytes()).hexdigest(),'input_node_count':len(X),'input_tet_count':len(T),'selected_edge_summary':edge_summary,'geometry_and_ancestry':metrics,'geometry_gate_pass':bool(metrics['tetra_positive_finite'] and boundary['exact_unoriented_set'] and boundary['oriented_mismatch_count']==0 and parent_sum_error<2e-12 and support['remaining_internal_original_o_o_edge_count']==0 and support['remaining_internal_all_o_face_count']==0 and support['remaining_all_o_tet_count']==0),'raw_volume_retention_gate_pass':bool(global_volume_rel<1e-6 and max_parent_volume_rel_error<1e-4),'quality_gate_pass':bool(Q2.min()>=1e-5),'support_gate_pass':bool(support['remaining_internal_original_o_o_edge_count']==0 and support['remaining_internal_all_o_face_count']==0 and support['remaining_all_o_tet_count']==0),'source_398_overlap_arrays':'Not copied into standard per-child arrays because their 398-local row semantics do not index 403 children; the pinned map is retained separately and a clearly labeled conservative parent-volume composition is emitted.','elapsed_build_s':time.monotonic()-t0,'native_admission':False}
    # Emit candidate after computing and recording all gates, even if only the
    # ordinary quality floor fails; edge support is the required invariant.
    with tempfile.NamedTemporaryFile(prefix='.candidate-403-',suffix='.tmp',dir=output_dir,delete=False) as tmp:
        tmpname=Path(tmp.name)
        np.savez_compressed(tmp,**out)
    surface_keys=('physical_outer_faces','physical_shell_faces','construction_closure_faces','physical_outer_node_ids')
    with np.load(tmpname,allow_pickle=False) as emitted:
        surface_bytes_exact=all(emitted[k].dtype==z[k].dtype and emitted[k].shape==z[k].shape and emitted[k].tobytes()==z[k].tobytes() for k in surface_keys)
        point_prefix_exact=emitted['points_world_m'][:len(X)].dtype==X.dtype and emitted['points_world_m'][:len(X)].tobytes()==X.tobytes()
        source_prefix_exact=(np.array_equal(emitted['node_source_indptr'][:len(z['node_source_indptr'])],z['node_source_indptr']) and emitted['node_source_indices'][:len(z['node_source_indices'])].tobytes()==z['node_source_indices'].tobytes() and emitted['node_source_weights'][:len(z['node_source_weights'])].tobytes()==z['node_source_weights'].tobytes())
        alias_exact=np.array_equal(emitted['parent_398_tet_row'],emitted['child_source_tetrahedron_rows'])
    if not (surface_bytes_exact and point_prefix_exact and source_prefix_exact and alias_exact):
        tmpname.unlink(missing_ok=True)
        raise ValueError(f'403 serialized candidate verification failed: surfaces={surface_bytes_exact} point_prefix={point_prefix_exact} source_prefix={source_prefix_exact} parent_alias={alias_exact}')
    metrics['shell_and_typed_surface_arrays_unchanged']=surface_bytes_exact
    metrics['source_prefix_byte_exact']=source_prefix_exact
    metrics['parent_398_row_alias_exact']=alias_exact
    os.replace(tmpname,candidate)
    report['candidate_sha256']=sha256(candidate)
    report['candidate_bytes']=candidate.stat().st_size
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=INPUT)
    ap.add_argument('--input-sha256',default=EXPECTED_INPUT_SHA256)
    ap.add_argument('--parent-map',type=Path,default=PARENT_MAP)
    ap.add_argument('--parent-map-sha256',default=EXPECTED_PARENT_MAP_SHA256)
    ap.add_argument('--output-dir',type=Path,default=BASE/'global-inner-p1-edge-support-403')
    args=ap.parse_args()
    result=build(args.input,args.output_dir,args.input_sha256,args.parent_map,args.parent_map_sha256)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['support_gate_pass'] and result['geometry_gate_pass'] else 2

if __name__=='__main__':
    raise SystemExit(main())
