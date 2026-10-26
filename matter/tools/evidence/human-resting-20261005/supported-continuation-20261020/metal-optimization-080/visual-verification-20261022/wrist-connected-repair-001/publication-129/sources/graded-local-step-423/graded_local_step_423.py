"""Witness-local scale damping followed by edge-cost min-plus grading.

Returns proposal multipliers only. All candidate geometry must pass the
existing exact gates after every emitted step.
"""
from __future__ import annotations
import hashlib
import heapq
import importlib.util
import os
from pathlib import Path
import numpy as np

LOCAL_HELPER = Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-witness-backtracking-183/final-a644/local_witness_backtracking.py")
LOCAL_SHA256 = "a644de24e2f23e90807eb75578309618d89c1a3f9d95d4f6a6f045f4d935c31a"
_CONTEXT_CACHE = None


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _edges_from_faces(faces):
    f=np.asarray(faces,dtype=np.int64)
    e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[0,2]]),axis=0)
    e.sort(axis=1)
    return np.unique(e,axis=0)


def load_context_423():
    global _CONTEXT_CACHE
    path=os.environ.get("NUMI_GRADED_LOCAL_STEP_CONTEXT_423")
    expected=os.environ.get("NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423")
    kappa_text=os.environ.get("NUMI_GRADED_LOCAL_STEP_KAPPA_423")
    if not path or not expected or not kappa_text:
        raise ValueError("423 graded context must be explicitly pinned by the validated 419 wrapper")
    kappa=float(kappa_text)
    if not np.isfinite(kappa) or not 0<kappa<=1:
        raise ValueError("423 kappa must be finite and in (0,1]")
    key=(path,expected,kappa)
    if _CONTEXT_CACHE is not None and _CONTEXT_CACHE[0]==key:
        return _CONTEXT_CACHE[1]
    p=Path(path)
    if not p.is_file() or _sha(p)!=expected:
        raise ValueError("423 context path/hash mismatch")
    with np.load(p,allow_pickle=False) as z:
        required={"faces","edges","edge_scale_budget","points_baseline_m","proposal_displacement_m","anchors"}
        if not required.issubset(z.files):
            raise ValueError("423 context missing required arrays")
        c={k:np.asarray(z[k]).copy() for k in required}
    f=c["faces"];e=c["edges"];b=c["edge_scale_budget"]
    x=c["points_baseline_m"];d=c["proposal_displacement_m"];a=c["anchors"]
    if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or len(f)==0:
        raise ValueError("423 context faces invalid")
    n=len(x)
    if x.shape!=(n,3) or n==0 or not np.isfinite(x).all() or d.shape!=(n,3) or not np.isfinite(d).all():
        raise ValueError("423 context point/displacement arrays invalid")
    if np.any(f<0) or np.any(f>=n): raise ValueError("423 context face index out of range")
    expected_edges=_edges_from_faces(f)
    if e.shape!=expected_edges.shape or e.dtype.kind not in "iu" or not np.array_equal(e,expected_edges):
        raise ValueError("423 context edges do not exactly match sorted unique face edges")
    if b.shape!=(len(e),) or not np.isfinite(b).all() or np.any(b<=0) or np.any(b>1):
        raise ValueError("423 edge scale budget invalid")
    if a.ndim!=1 or a.dtype.kind not in "iu" or np.any(a<0) or np.any(a>=n) or len(np.unique(a))!=len(a):
        raise ValueError("423 anchor rows invalid")
    edge_len=np.linalg.norm(x[e[:,0]]-x[e[:,1]],axis=1)
    amp=np.maximum(np.linalg.norm(d[e[:,0]],axis=1),np.linalg.norm(d[e[:,1]],axis=1))
    expected_budget=np.ones(len(e),dtype=np.float64)
    moving=amp>0
    expected_budget[moving]=np.minimum(1.0,kappa*edge_len[moving]/amp[moving])
    if not np.allclose(b,expected_budget,rtol=2e-12,atol=2e-15):
        raise ValueError("423 edge budgets do not match kappa, edge lengths, and displacement amplitudes")
    adjacency=[[] for _ in range(n)]
    for (u,v),cost in zip(e,b):
        u=int(u);v=int(v);cost=float(cost)
        adjacency[u].append((v,cost));adjacency[v].append((u,cost))
    context={"faces":f.astype(np.int64,copy=False),"edges":e.astype(np.int64,copy=False),
             "budget":b.astype(np.float64,copy=False),"anchors":a.astype(np.int64,copy=False),
             "adjacency":adjacency,"kappa":kappa,"context_sha256":expected}
    _CONTEXT_CACHE=(key,context)
    return context


def reduce_witness_node_scales(faces, step_scales, bad_face_rows, self_pair_face_rows, *, anchors=(), zero_below=1.0/128.0):
    """Apply pinned 183 witness reduction, then min-plus edge-cost grading."""
    c=load_context_423()
    f=np.asarray(faces)
    s=np.asarray(step_scales,dtype=np.float64)
    a=np.asarray(anchors,dtype=np.int64).reshape(-1)
    if f.dtype.kind not in "iu" or f.shape!=c["faces"].shape or not np.array_equal(f,c["faces"]):
        raise ValueError("runtime faces differ from 423 context topology")
    if s.shape!=(len(c["adjacency"]),) or not np.isfinite(s).all() or np.any(s<0) or np.any(s>1):
        raise ValueError("runtime scales differ from 423 context node domain")
    if not np.array_equal(np.sort(a),np.sort(c["anchors"])):
        raise ValueError("runtime anchors differ from 423 context pins")
    if not LOCAL_HELPER.is_file() or _sha(LOCAL_HELPER)!=LOCAL_SHA256:
        raise ValueError("pinned 183 local reducer changed")
    spec=importlib.util.spec_from_file_location("pinned_local_witness_backtracking_183",LOCAL_HELPER)
    if spec is None or spec.loader is None: raise ValueError("cannot import pinned 183 local reducer")
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    bad=np.asarray(bad_face_rows,dtype=np.int64).reshape(-1)
    pairs=np.asarray(self_pair_face_rows,dtype=np.int64)
    if pairs.size==0:pairs=np.empty((0,2),dtype=np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2:raise ValueError("self-pair rows must have shape (K,2)")
    if np.any(bad<0) or np.any(bad>=len(f)) or (pairs.size and (np.any(pairs<0) or np.any(pairs>=len(f)))):
        raise ValueError("witness face row out of range")
    witness_faces=np.unique(np.r_[bad,pairs.ravel()]).astype(np.int64)
    if len(witness_faces)==0:raise ValueError("423 requires exact failure witnesses")
    witness_nodes=np.unique(f[witness_faces].ravel()).astype(np.int64)
    local,local_receipt=owner.reduce_witness_node_scales(
        f,s,bad,pairs,anchors=a,zero_below=zero_below)
    local=np.asarray(local,dtype=np.float64)
    # Multi-source min-plus closure starts from the current reduced scale
    # field but excludes fixed anchors as propagation sources. Disconnected
    # components with no reduced source remain bitwise unchanged.
    anchor_mask=np.zeros(len(local),dtype=bool);anchor_mask[a]=True
    sources=np.flatnonzero(~anchor_mask)
    dist=np.full(len(local),np.inf,dtype=np.float64)
    dist[sources]=local[sources]
    heap=[(float(dist[i]),int(i)) for i in sources]
    heapq.heapify(heap)
    while heap:
        du,u=heapq.heappop(heap)
        if du!=dist[u]:continue
        for v,cost in c["adjacency"][u]:
            alt=du+cost
            if alt<dist[v]:
                dist[v]=alt;heapq.heappush(heap,(alt,v))
    out=np.minimum(local,dist)
    out[a]=0.0
    if out.shape!=s.shape or not np.isfinite(out).all() or np.any(out<0) or np.any(out>s):
        raise ValueError("423 graded scales are malformed or increase motion")
    changed=np.flatnonzero(out<local)
    nonanchor_edge=~(anchor_mask[c["edges"][:,0]]|anchor_mask[c["edges"][:,1]])
    diff=np.abs(out[c["edges"][:,0]]-out[c["edges"][:,1]])
    max_excess=float(np.maximum(diff[nonanchor_edge]-c["budget"][nonanchor_edge],0).max(initial=0.0))
    if max_excess>2e-12:
        raise AssertionError("423 output violates a nonanchor edge scale budget")
    untouched=np.ones(len(out),dtype=bool);untouched[changed]=False
    report=dict(status="GRADED_LOCAL_WITNESS_SCALES_ONLY",construction_only=True,
        geometry_certified=False,policy="LOCAL_183_THEN_EDGE_COST_MIN_PLUS_423",
        source_local_helper_sha256=LOCAL_SHA256,context_sha256=c["context_sha256"],
        kappa=c["kappa"],witness_face_rows=witness_faces.tolist(),
        witness_node_count=int(len(witness_nodes)),propagated_node_count=int(len(changed)),
        propagated_node_rows=changed.tolist(),anchor_rows_forced_zero=a.tolist(),
        min_edge_scale_budget=float(c["budget"].min()),max_edge_scale_budget=float(c["budget"].max()),
        max_edge_budget_excess_nonanchor_edges=max_excess,
        exempt_anchor_incident_edge_count=int(np.count_nonzero(~nonanchor_edge)),
        unaffected_scales_bitwise_unchanged=bool(np.array_equal(out[untouched],local[untouched])),
        local_receipt=local_receipt,native_admission=False,accepted_physical_steps=0)
    return out,report
