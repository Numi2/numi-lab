"""Opt-in 426 scale LP with a cached, pinned 026 self-seed cutting plane.

The full baseline seed rows are checked on every call. Violated rows are added
monotonically to the alpha LP until all cached rows pass. Exact emitted-geometry
audits remain mandatory. Opt in by selecting this helper and setting:
  NUMI_GRADED_SELF_SEED_PATH_431
  NUMI_GRADED_SELF_SEED_SHA256_431
  NUMI_GRADED_SELF_SEED_PREFLIGHT_PATH_431
  NUMI_GRADED_SELF_SEED_PREFLIGHT_SHA256_431
  NUMI_GRADED_SELF_SEED_CACHE_431   (writable .npz path)
  NUMI_GRADED_SELF_SEED_REPORT_431  (writable .json path)
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069")
CUTS426=BASE/"D/graded-cut-step-426/graded_cut_step_426.py"
CUTS426_SHA256="36eae51a16923a0a636518cb05aefea141cf8dd63a67210504575a91302d41ed"
SAT_SHA256="c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3"
MARGIN_FRACTION=0.25
ROW_TOL_MM=2e-8
LP_TOL=1e-8
MAX_CUT_ROUNDS=32

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def _load_owner():
    if _sha(CUTS426)!=CUTS426_SHA256: raise ValueError("pinned 426 helper changed")
    spec=importlib.util.spec_from_file_location("graded_cut_step_426_pinned_for_431",CUTS426)
    if spec is None or spec.loader is None: raise ValueError("cannot load pinned 426 helper")
    owner=importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    if owner.sat is None or _sha(owner.SAT)!=SAT_SHA256: raise ValueError("pinned 026 SAT owner changed")
    return owner

cuts=_load_owner()
_FAILED_PAIRS=set()
_SEED_CACHE=None
_PROJECTED_CACHE=None

def _require_env(name):
    value=os.environ.get(name)
    if not value: raise ValueError(f"431 requires {name}")
    return value

def _seed_inputs(faces, context):
    seed_path=Path(_require_env("NUMI_GRADED_SELF_SEED_PATH_431"))
    expected=_require_env("NUMI_GRADED_SELF_SEED_SHA256_431").lower()
    preflight_path=Path(_require_env("NUMI_GRADED_SELF_SEED_PREFLIGHT_PATH_431"))
    preflight_sha=_require_env("NUMI_GRADED_SELF_SEED_PREFLIGHT_SHA256_431").lower()
    cache_path=Path(_require_env("NUMI_GRADED_SELF_SEED_CACHE_431"))
    report_path=Path(_require_env("NUMI_GRADED_SELF_SEED_REPORT_431"))
    if not seed_path.is_file() or _sha(seed_path)!=expected:
        raise ValueError("431 self-seed path/hash mismatch")
    if not preflight_path.is_file() or _sha(preflight_path)!=preflight_sha:
        raise ValueError("431 self-seed preflight path/hash mismatch")
    preflight=json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("schema")!="numi.human.continuation-self-seeds.v1" or
        preflight.get("status")!="PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY" or
        preflight.get("zero_step_feasible") is not True or
        preflight.get("output_pairs_path")!=str(seed_path) or
        preflight.get("output_pairs_sha256")!=expected or
        int(preflight.get("merged_unique_pair_count",-1))!=len(np.load(seed_path,allow_pickle=False)) or
        preflight.get("pair_owner_sha256")!=SAT_SHA256):
        raise ValueError("431 seed preflight receipt does not authenticate the selected full seed cache")
    if cache_path.suffix!=".npz" or report_path.suffix!=".json":
        raise ValueError("431 cache/report paths must end in .npz/.json")
    if not cache_path.parent.is_dir() or not report_path.parent.is_dir():
        raise ValueError("431 cache/report parent directories must already exist")
    pairs=np.load(seed_path,allow_pickle=False)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.dtype.kind not in "iu" or len(pairs)==0:
        raise ValueError("431 self-seed array must be nonempty integer (K,2)")
    pairs=np.sort(pairs.astype(np.int64,copy=False),axis=1)
    if np.any(pairs[:,0]<0) or np.any(pairs[:,1]>=len(faces)) or np.any(pairs[:,0]==pairs[:,1]):
        raise ValueError("431 self-seed face row out of range")
    pairs=np.unique(pairs,axis=0)
    faces=np.asarray(faces,dtype=np.int64)
    face_sha=hashlib.sha256(np.ascontiguousarray(faces,dtype="<i8").tobytes()).hexdigest()
    with np.load(os.environ["NUMI_GRADED_LOCAL_STEP_CONTEXT_423"],allow_pickle=False) as z:
        points_mm=np.asarray(z["points_baseline_m"],dtype=np.float64)*1000.0
        context_faces=np.asarray(z["faces"],dtype=np.int64)
    if points_mm.shape!=(len(context["adjacency"]),3) or not np.array_equal(faces,context_faces):
        raise ValueError("431 baseline or topology differs from pinned 423 context")
    points_sha=hashlib.sha256(np.ascontiguousarray(points_mm,dtype="<f8").tobytes()).hexdigest()
    meta={"schema":"numi-graded-self-seed-row-cache-431-v1","seed_sha256":expected,
          "seed_path":str(seed_path),"seed_pair_count":int(len(pairs)),
          "seed_preflight_path":str(preflight_path),"seed_preflight_sha256":preflight_sha,
          "context_sha256":context["context_sha256"],"faces_sha256":face_sha,
          "baseline_points_sha256":points_sha,"sat_owner_sha256":SAT_SHA256,
          "margin_fraction":MARGIN_FRACTION}
    return pairs,points_mm,cache_path,report_path,meta

def _pack_rows(rows):
    lens=np.fromiter((len(r["nodes"]) for r in rows),dtype=np.int64,count=len(rows))
    ptr=np.empty(len(rows)+1,dtype=np.int64); ptr[0]=0; np.cumsum(lens,out=ptr[1:])
    nodes=np.concatenate([np.asarray(r["nodes"],dtype=np.int64) for r in rows])
    coeff=np.concatenate([np.asarray(r["coefficients"],dtype=np.float64) for r in rows],axis=0)
    lower=np.fromiter((float(r["lower"]) for r in rows),dtype=np.float64,count=len(rows))
    return {"indptr":ptr,"nodes":nodes,"coefficients":coeff,"lower":lower}

def _read_cache(path, meta):
    with np.load(path,allow_pickle=False) as z:
        saved=json.loads(str(z["metadata_json"].item()))
        if saved!=meta: raise ValueError("431 row-cache pin mismatch")
        out={k:np.asarray(z[k]).copy() for k in ("indptr","nodes","coefficients","lower")}
    _validate_packed(out,meta)
    return out

def _validate_packed(packed,meta):
    ptr=packed["indptr"]; nodes=packed["nodes"]; coeff=packed["coefficients"]; lower=packed["lower"]
    if ptr.ndim!=1 or len(ptr)!=len(lower)+1 or ptr[0]!=0 or ptr[-1]!=len(nodes):
        raise ValueError("431 packed seed rows have malformed indptr")
    if np.any(np.diff(ptr)<1) or coeff.shape!=(len(nodes),3) or not np.isfinite(coeff).all() or not np.isfinite(lower).all():
        raise ValueError("431 packed seed rows are malformed")
    if nodes.dtype.kind not in "iu" or np.any(nodes<0): raise ValueError("431 seed row node out of range")
    if len(lower)!=0 and float(lower.max())>ROW_TOL_MM: raise ValueError("431 seed rows are infeasible at zero step")
    if int(meta["seed_pair_count"])<=0: raise ValueError("431 seed metadata has no pairs")

def _load_seed_rows(faces,context):
    global _SEED_CACHE
    pairs,points_mm,path,report_path,meta=_seed_inputs(faces,context)
    key=json.dumps(meta,sort_keys=True,separators=(",",":"))
    if _SEED_CACHE is not None and _SEED_CACHE[0]==key:
        return pairs,_SEED_CACHE[1],path,report_path,meta
    if path.exists():
        packed=_read_cache(path,meta)
    else:
        rows,_records=cuts.sat.build_pair_rows(points_mm,faces,pairs,margin_fraction=MARGIN_FRACTION)
        packed=_pack_rows(rows)
        _validate_packed(packed,meta)
        temp=path.with_name(path.name+f".tmp-{os.getpid()}")
        try:
            with temp.open("wb") as f:
                np.savez(f,metadata_json=np.asarray(json.dumps(meta,sort_keys=True,separators=(",",":"))),**packed)
                f.flush(); os.fsync(f.fileno())
            os.replace(temp,path)
        finally:
            if temp.exists(): temp.unlink()
    _SEED_CACHE=(key,packed)
    return pairs,packed,path,report_path,meta

def _row_matrices(pair_rows, displacement_mm, n):
    rr=[];cc=[];vv=[];lo=[]
    for row in pair_rows:
        nodes=np.asarray(row["nodes"],dtype=np.int64)
        projected=np.einsum("ij,ij->i",np.asarray(row["coefficients"],float),displacement_mm[nodes])
        mx=float(np.max(np.abs(projected),initial=0.0)); bound=-float(row["lower"])
        if mx<1e-16:
            if bound < -1e-10: raise ValueError("infeasible constant failed-pair row")
            continue
        rid=len(lo); rr.extend([rid]*len(nodes));cc.extend(nodes.tolist());vv.extend((-projected/mx).tolist());lo.append(bound/mx)
    A=coo_matrix((vv,(rr,cc)),shape=(len(lo),n)).tocsr()
    return A,np.asarray(lo,dtype=np.float64)

def _project_seed_rows(packed, displacement_mm, n, key):
    global _PROJECTED_CACHE
    delta_sha=hashlib.sha256(np.ascontiguousarray(displacement_mm,dtype="<f8").tobytes()).hexdigest()
    cache_key=(key,delta_sha,n)
    if _PROJECTED_CACHE is not None and _PROJECTED_CACHE[0]==cache_key:
        return _PROJECTED_CACHE[1:]
    ptr=packed["indptr"]; nodes=packed["nodes"]; coeff=packed["coefficients"]; lower=packed["lower"]
    row_ids=np.repeat(np.arange(len(lower),dtype=np.int64),np.diff(ptr))
    projected=np.einsum("ij,ij->i",coeff,displacement_mm[nodes])
    raw=coo_matrix((projected,(row_ids,nodes)),shape=(len(lower),n)).tocsr()
    maxima=np.asarray(np.abs(raw).max(axis=1).toarray()).reshape(-1)
    valid=np.flatnonzero(maxima>=1e-16)
    const=np.flatnonzero(maxima<1e-16)
    if len(const) and np.any(lower[const]>ROW_TOL_MM): raise ValueError("infeasible constant seed row")
    inv=1.0/maxima[valid]
    # Geometry requires projected @ alpha >= lower; scipy uses A_ub @ alpha <= b_ub.
    norm=(-raw[valid].multiply(inv[:,None])).tocsr()
    norm_lower=-lower[valid]*inv
    result=(raw,lower,valid,norm,norm_lower)
    _PROJECTED_CACHE=(cache_key,*result)
    return result

def _solve_with_seed_cuts(faces, upper, anchors, edges, budgets, displacement_mm, pair_rows, packed, cache_key):
    n=len(upper); d=np.asarray(displacement_mm,dtype=np.float64)
    pair_A,pair_b=_row_matrices(pair_rows,d,n)
    mask=np.zeros(n,dtype=bool); mask[np.asarray(anchors,dtype=np.int64)]=True
    edge_rr=[];edge_cc=[];edge_vv=[];edge_b=[]
    for (u,v),b in zip(np.asarray(edges,dtype=np.int64),np.asarray(budgets,dtype=float)):
        if b>=1 or mask[u] or mask[v]: continue
        r=len(edge_b);edge_rr.extend((r,r));edge_cc.extend((int(u),int(v)));edge_vv.extend((1.0,-1.0));edge_b.append(float(b))
        r=len(edge_b);edge_rr.extend((r,r));edge_cc.extend((int(u),int(v)));edge_vv.extend((-1.0,1.0));edge_b.append(float(b))
    edge_A=coo_matrix((edge_vv,(edge_rr,edge_cc)),shape=(len(edge_b),n)).tocsr()
    base_A=vstack((pair_A,edge_A),format="csr");base_b=np.r_[pair_b,np.asarray(edge_b,dtype=float)]
    raw_seed,lower,valid,seed_A,seed_b=_project_seed_rows(packed,d,n,cache_key)
    amp=np.linalg.norm(d,axis=1);w=np.maximum(amp,1e-8);w/=max(float(w.max(initial=0.0)),1e-8)
    ub=np.asarray(upper,dtype=float).copy();ub[np.asarray(anchors,dtype=np.int64)]=0.0
    active=np.zeros(len(valid),dtype=bool); additions=[]; result=None; A=None; b=None; x=None
    for cut_round in range(1,MAX_CUT_ROUNDS+1):
        active_ids=np.flatnonzero(active)
        A=vstack((base_A,seed_A[active_ids]),format="csr") if len(active_ids) else base_A
        b=np.r_[base_b,seed_b[active_ids]] if len(active_ids) else base_b
        result=linprog(-w,A_ub=A,b_ub=b,bounds=np.column_stack((np.zeros(n),ub)),method="highs",
            options={"time_limit":30.,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        if not result.success: raise ValueError("431 seed-cut alpha LP failed: "+result.message)
        x=np.minimum(ub,np.maximum(0.0,result.x))
        violation=lower-np.asarray(raw_seed@x).reshape(-1)
        bad=np.flatnonzero(violation>ROW_TOL_MM)
        # valid is in original row order; all non-constant violated rows map directly.
        bad_valid=np.searchsorted(valid,bad)
        bad_valid=bad_valid[(bad_valid<len(valid)) & (valid[np.minimum(bad_valid,len(valid)-1)]==bad)] if len(valid) else np.empty(0,dtype=np.int64)
        new=bad_valid[~active[bad_valid]] if len(bad_valid) else np.empty(0,dtype=np.int64)
        additions.append(int(len(new)))
        if len(new)==0: break
        active[new]=True
    else:
        raise ValueError("431 seed-cut alpha LP did not close all seed violations")
    assert result is not None and x is not None and A is not None and b is not None
    lp_res=float(np.max(np.asarray(A@x-b).reshape(-1),initial=0.0))
    seed_v=lower-np.asarray(raw_seed@x).reshape(-1)
    max_seed=float(np.maximum(seed_v,0.0).max(initial=0.0)); bad_seed=int(np.count_nonzero(seed_v>ROW_TOL_MM))
    if lp_res>LP_TOL or max_seed>ROW_TOL_MM or bad_seed:
        raise ValueError("431 final alpha LP/seed certificate failed")
    proof={"lp_success":True,"lp_iterations":int(result.nit),"lp_rows":int(A.shape[0]),"lp_variables":int(n),
           "normalized_max_row_violation":lp_res,"retained_weighted_scale":float(w@x),
           "possible_weighted_scale":float(w@ub),"seed_cut_rounds":int(len(additions)),
           "seed_rows_added_per_round":additions,"seed_rows_active":int(active.sum()),
           "seed_rows_total":int(len(lower)),"seed_rows_checked":int(len(lower)),
           "seed_max_violation_mm":max_seed,"seed_violated_rows":bad_seed,
           "anchor_rows_forced_zero":np.asarray(anchors,dtype=np.int64).tolist()}
    return x,proof

def _write_report(path, report):
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    try:
        with tmp.open("w",encoding="utf-8") as f:
            json.dump(report,f,sort_keys=True,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

def reduce_witness_node_scales(faces,step_scales,bad_face_rows,self_pair_face_rows,*,anchors=(),zero_below=1/128):
    """426 reduction plus cached seed-row cuts for every independently scaled vertex."""
    context=cuts.g.load_context_423()
    faces=np.asarray(faces,dtype=np.int64)
    upper,report=cuts.g.reduce_witness_node_scales(faces,step_scales,bad_face_rows,self_pair_face_rows,
                                                    anchors=anchors,zero_below=zero_below)
    pairs_now=np.asarray(self_pair_face_rows,dtype=np.int64)
    if pairs_now.size==0: pairs_now=np.empty((0,2),dtype=np.int64)
    if pairs_now.ndim!=2 or pairs_now.shape[1]!=2: raise ValueError("431 failed-pair ids must have shape (K,2)")
    before=len(_FAILED_PAIRS)
    _FAILED_PAIRS.update(tuple(sorted(map(int,p))) for p in pairs_now)
    points_path=Path(os.environ["NUMI_GRADED_LOCAL_STEP_CONTEXT_423"])
    with np.load(points_path,allow_pickle=False) as z:
        baseline=np.asarray(z["points_baseline_m"],dtype=np.float64)*1000.0
        displacement=np.asarray(z["proposal_displacement_m"],dtype=np.float64)*1000.0
    pair_ids=np.asarray(sorted(_FAILED_PAIRS),dtype=np.int64).reshape(-1,2)
    failed_rows,failed_records=cuts.sat.build_pair_rows(baseline,faces,pair_ids,margin_fraction=MARGIN_FRACTION) if len(pair_ids) else ([],[])
    seeds,packed,cache_path,report_path,meta=_load_seed_rows(faces,context)
    key=json.dumps(meta,sort_keys=True,separators=(",",":"))
    out,proof=_solve_with_seed_cuts(faces,upper,anchors,context["edges"],context["budget"],displacement,failed_rows,packed,key)
    report.update(status="GRADED_426_WITH_PINNED_SELF_SEED_ALPHA_CUTS_431",policy="423_PLUS_026_FAILED_AND_FULL_SEED_SAT_ALPHA_CUTS",
        seed_pair_path=meta["seed_path"],seed_pair_sha256=meta["seed_sha256"],seed_pair_count=meta["seed_pair_count"],
        seed_row_cache_path=str(cache_path),seed_row_cache_sha256=_sha(cache_path),seed_linear_row_count=int(len(packed["lower"])),
        baseline_points_sha256=meta["baseline_points_sha256"],faces_sha256=meta["faces_sha256"],sat_owner_sha256=SAT_SHA256,
        failed_pair_count=int(len(pair_ids)),failed_pair_rows=int(len(failed_rows)),new_failed_pairs_this_call=int(len(_FAILED_PAIRS)-before),
        failed_pair_records=failed_records,seed_cut_lp=proof,native_admission=False,accepted_physical_steps=0)
    _write_report(report_path,report)
    return out,report
