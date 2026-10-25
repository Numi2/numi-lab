"""Registered geometric surface anchors, independent of displacement weights.

A one-hot displacement row carries no position coincidence guarantee:
x_detail = x_detail_rest + W (x_mechanical - x_mechanical_rest).
Only compiler-exported source IDs/barycentrics establish source correspondence.
"""
from __future__ import annotations
import numpy as np

def registered_surface_anchors(detail_points, detail_faces, coarse_points, rows, *,
                               detail_source_vertex_ids, direct_tolerance_m=None):
    P=np.asarray(detail_points);F=np.asarray(detail_faces);C=np.asarray(coarse_points)
    if P.dtype!=np.float32 or P.ndim!=2 or P.shape[1]!=3 or not np.isfinite(P).all():
        raise ValueError("detail_points must be finite Nx3 Float32")
    if C.dtype!=np.float32 or C.ndim!=2 or C.shape[1]!=3 or not np.isfinite(C).all():
        raise ValueError("coarse_points must be finite Mx3 Float32 in the same registered frame")
    if F.dtype.kind not in'iu' or F.ndim!=2 or F.shape[1]!=3 or F.min(initial=0)<0 or F.max(initial=0)>=len(P):
        raise ValueError("invalid detailed source faces")
    if direct_tolerance_m is not None and (not np.isfinite(direct_tolerance_m) or direct_tolerance_m<=0):
        raise ValueError("direct_tolerance_m must be positive")
    source_ids=np.asarray(detail_source_vertex_ids)
    if source_ids.dtype.kind not in'iu'or source_ids.shape!=(len(P),)or np.any(source_ids<0)or len(np.unique(source_ids))!=len(P):
        raise ValueError("explicit unique source vertex IDs must match detailed point rows")
    id_to_row={int(v):i for i,v in enumerate(source_ids)}
    records=list(rows)
    if len(records)!=len(C):raise ValueError("one explicit source record per coarse row is required")
    triangles={tuple(sorted(map(int,f)))for f in F}
    result=[];points=np.empty((len(C),3),np.float64);seen=set()
    for r in records:
        i=int(r['coarse_row'])
        if i in seen or not 0<=i<len(C):raise ValueError("duplicate/out-of-range coarse row")
        seen.add(i);direct=int(r['source_direct_id'])
        if direct in id_to_row:
            ids=[direct];weights=[1.];kind='source_direct_id'
            drows=[id_to_row[direct]];q=P[drows[0]].astype(float)
            residual=float(np.linalg.norm(q-C[i].astype(float)))
            if direct_tolerance_m is not None and residual>direct_tolerance_m:
                raise ValueError(f"direct registered anchor {i}/{direct} differs by {residual} m")
        elif direct==4294967295:
            ids=[int(r[f'source_face_v{j}'])for j in range(3)]
            weights=[float(r[f'source_bary{j}'])for j in range(3)]
            if any(v not in id_to_row for v in ids):raise ValueError(f"row {i}: unknown source face ID")
            drows=[id_to_row[v]for v in ids]
            if len(set(ids))!=3 or tuple(sorted(drows))not in triangles:
                raise ValueError(f"row {i}: barycentric ancestry is not a detailed source face")
            w=np.asarray(weights,float)
            if not np.isfinite(w).all()or np.any(w<0)or np.any(w>1)or abs(float(w.sum())-1)>1e-6:
                raise ValueError(f"row {i}: invalid source barycentrics")
            # Preserve exported values exactly. Do not renormalize or replace the
            # registered source point with the possibly offset mechanical point.
            q=w@P[drows].astype(float);kind='source_face_barycentric'
            residual=float(np.linalg.norm(q-C[i].astype(float)))
        else:raise ValueError(f"row {i}: invalid source_direct_id; no proximity fallback")
        points[i]=q
        result.append({'coarse_row':i,'kind':kind,'source_vertex_ids':ids,
                       'detail_vertex_rows':drows,'source_weights':weights,'source_point_world_m':q.tolist(),
                       'coarse_to_source_offset_m':(q-C[i].astype(float)).tolist(),
                       'coincidence_residual_m':residual})
    result.sort(key=lambda r:r['coarse_row'])
    return points,result


class AnchorResolutionError(ValueError):
    pass

def resolve_typed_coarse_anchors(anchor_source_ids, node_kind, node_source_row, *, coarse_kind=0):
    """Resolve coarse source identities to unique current output rows.

    Source IDs must never be used as row indices after compaction/refinement.
    Noncoarse nodes can carry the same numeric source ID and are ignored.
    """
    anchors=np.asarray(anchor_source_ids)
    kinds=np.asarray(node_kind)
    sources=np.asarray(node_source_row)
    if anchors.ndim!=1 or kinds.ndim!=1 or sources.ndim!=1 or kinds.shape!=sources.shape:
        raise AnchorResolutionError("anchor IDs and typed node provenance must be 1D and aligned")
    if anchors.dtype.kind not in "iu" or kinds.dtype.kind not in "iu" or sources.dtype.kind not in "iu":
        raise AnchorResolutionError("anchor IDs, node_kind and node_source_row must be integer arrays")
    if np.any(anchors<0) or np.any(sources[kinds==int(coarse_kind)]<0):
        raise AnchorResolutionError("coarse source IDs must be nonnegative")
    if any(a.dtype.kind=="u" and np.any(a>np.iinfo(np.int64).max)for a in (anchors,kinds,sources)):
        raise AnchorResolutionError("typed identity exceeds signed 64-bit domain")
    anchor_ids=anchors.astype(np.int64,copy=False)
    if len(np.unique(anchor_ids))!=len(anchor_ids):
        raise AnchorResolutionError("coarse anchor source IDs are duplicated")
    rows=np.flatnonzero(kinds==int(coarse_kind))
    src=sources[rows].astype(np.int64,copy=False)
    output=np.empty(len(anchor_ids),dtype=np.int64)
    for j,sid in enumerate(anchor_ids):
        match=rows[src==sid]
        if len(match)==0:
            raise AnchorResolutionError(f"missing coarse output row for source ID {int(sid)}")
        if len(match)!=1:
            raise AnchorResolutionError(f"ambiguous coarse source ID {int(sid)}: output rows {match.tolist()}")
        output[j]=int(match[0])
    if len(np.unique(output))!=len(output):
        raise AnchorResolutionError("distinct source anchor IDs resolved to duplicate output rows")
    return output
