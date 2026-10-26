#!/usr/bin/env python3
"""Conservative swept-face-AABB pair preseed for a bounded surface trial.

This only proposes candidate face pairs for later linearized constraints and
exact audits. It is not a self-intersection test or geometry acceptance gate.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def canonical_pairs(rows, face_count: int) -> np.ndarray:
    a=np.asarray(rows)
    if a.size == 0:
        a=np.empty((0,2),dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype,np.integer):
        raise ValueError("pair rows must be an integer Nx2 array")
    a=np.asarray(a,dtype=np.int64)
    if np.any(a<0) or np.any(a>=face_count) or np.any(a[:,0]==a[:,1]):
        raise ValueError("pair rows contain invalid face IDs")
    a=np.sort(a,axis=1)
    if len(a):
        a=np.unique(a,axis=0)
    return a.astype(np.int64,copy=False)


def _validate_mesh(points, faces):
    p=np.asarray(points)
    f=np.asarray(faces)
    if p.dtype != np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError("point arrays must be finite emitted Float32 Nx3")
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer):
        raise ValueError("faces must be an integer Mx3 array")
    f=np.asarray(f,dtype=np.int64)
    if np.any(f<0) or np.any(f>=len(p)):
        raise ValueError("face index outside point array")
    if np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2])):
        raise ValueError("degenerate index face")
    return p,f


def _shared_counts(fa, fb):
    # Chunking bounds temporary memory for large sweep bins.
    return np.any(fa[:,:,None] == fb[:,None,:],axis=1).sum(axis=1)


def swept_face_aabb_candidates(base_points_f32, face_rows, trial_points_f32, *,
                               scale: float=2.0, padding_m: float=1e-7,
                               include_pairs=(), exclude_shared_vertices: int=2):
    """Return conservative pairs whose swept component AABBs overlap.

    For each trial, its displacement from the same base is scaled about the
    base; each vertex moves linearly to that scaled endpoint. The union AABB of
    the base and scaled endpoint contains the full vertex trajectory and hence
    the moving triangle at every intermediate parameter. Pairs sharing at
    least ``exclude_shared_vertices`` vertices are omitted as ordinary
    topological adjacency; caller-supplied pairs are always retained.
    """
    base,faces=_validate_mesh(base_points_f32,face_rows)
    trials=list(trial_points_f32) if isinstance(trial_points_f32,(list,tuple)) else [trial_points_f32]
    if not trials: raise ValueError("at least one trial endpoint is required")
    if not np.isfinite(scale) or scale < 1.0: raise ValueError("scale must be finite and >=1")
    if not np.isfinite(padding_m) or padding_m < 0: raise ValueError("padding must be finite and nonnegative")
    if not isinstance(exclude_shared_vertices,int) or exclude_shared_vertices not in (1,2,3):
        raise ValueError("exclude_shared_vertices must be 1, 2, or 3")
    lo=np.full((len(faces),3),np.inf,dtype=np.float64)
    hi=np.full((len(faces),3),-np.inf,dtype=np.float64)
    p64=base.astype(np.float64)
    for raw in trials:
        trial=np.asarray(raw)
        if trial.dtype != np.float32 or trial.shape != base.shape or not np.isfinite(trial).all():
            raise ValueError("each trial must be finite Float32 and match base point shape")
        # Scaling is about the exact base geometry; F64 arithmetic only expands
        # the broadphase bounds, it does not change emitted trial coordinates.
        end=p64+scale*(trial.astype(np.float64)-p64)
        tri0=p64[faces]; tri1=end[faces]
        lo=np.minimum(lo,np.minimum(tri0.min(axis=1),tri1.min(axis=1)))
        hi=np.maximum(hi,np.maximum(tri0.max(axis=1),tri1.max(axis=1)))
    lo-=padding_m; hi+=padding_m

    # Pick the sweep axis with the smallest interval-overlap workload estimate.
    workload=[]
    order_by_axis=[]
    stop_by_axis=[]
    for axis in range(3):
        order=np.lexsort((np.arange(len(faces)),lo[:,axis]))
        starts=lo[order,axis]
        stops=np.searchsorted(starts,hi[order,axis],side="right")
        cost=int(np.sum(np.maximum(stops-np.arange(len(faces))-1,0),dtype=np.int64))
        workload.append(cost); order_by_axis.append(order); stop_by_axis.append(stops)
    axis=int(np.argmin(workload)); order=order_by_axis[axis]; stops=stop_by_axis[axis]
    other=[d for d in range(3) if d!=axis]
    found=[]
    # Sweep one start interval at a time; vectorize candidate checks and process
    # in small chunks to keep memory bounded even in dense cross-sections.
    chunk=100_000
    for pos,fi0 in enumerate(order):
        stop=int(stops[pos])
        if stop<=pos+1: continue
        candidates=order[pos+1:stop]
        for begin in range(0,len(candidates),chunk):
            js=candidates[begin:begin+chunk]
            mask=(lo[js,other[0]]<=hi[fi0,other[0]])&(hi[js,other[0]]>=lo[fi0,other[0]])
            mask &= (lo[js,other[1]]<=hi[fi0,other[1]])&(hi[js,other[1]]>=lo[fi0,other[1]])
            js=js[mask]
            if len(js)==0: continue
            shared=_shared_counts(np.broadcast_to(faces[fi0],(len(js),3)),faces[js])
            js=js[shared<exclude_shared_vertices]
            if len(js):
                pair=np.column_stack((np.minimum(fi0,js),np.maximum(fi0,js))).astype(np.int64)
                found.append(pair)
    swept=np.vstack(found) if found else np.empty((0,2),dtype=np.int64)
    swept=canonical_pairs(swept,len(faces))
    seeds=canonical_pairs(include_pairs,len(faces))
    result=canonical_pairs(np.vstack((swept,seeds)),len(faces))
    return result,{"face_count":int(len(faces)),"trial_count":len(trials),
        "raw_displacement_scale":float(scale),"aabb_padding_m":float(padding_m),
        "excluded_shared_vertex_threshold":int(exclude_shared_vertices),
        "sweep_axis":axis,"estimated_axis_interval_workloads":workload,
        "swept_aabb_pair_count":int(len(swept)),"seed_pair_count":int(len(seeds)),
        "union_pair_count":int(len(result)),
        "scope":"conservative construction broadphase only; not an exact intersection test or acceptance gate"}


def classify_pair_rows(points_f32, faces, pair_rows, owner_module, *, max_seconds=1200.0, batch_size=64):
    """Classify 026 row eligibility within a bounded owner-call budget.

    Candidate pairs are chunked, then each failed chunk is bisected until the
    exact rejected pair is isolated. Unvisited rows on timeout remain explicit
    as unclassified and cannot be presented as a complete solver seed.
    """
    points_mm=np.asarray(points_f32,dtype=np.float64)*1000.0
    pairs=canonical_pairs(pair_rows,len(faces))
    max_seconds=float(max_seconds)
    if not np.isfinite(max_seconds) or max_seconds<=0: raise ValueError("max_seconds must be finite and positive")
    if not isinstance(batch_size,int) or batch_size<1: raise ValueError("batch_size must be positive")
    deadline=time.monotonic()+max_seconds
    eligible=[]; rejected=[]; unclassified=[]; row_count=0; calls=0
    def visit(chunk):
        nonlocal row_count,calls
        if len(chunk)==0:return
        if time.monotonic()>=deadline:
            unclassified.extend(tuple(map(int,x)) for x in chunk);return
        if len(chunk)>batch_size:
            for start in range(0,len(chunk),batch_size):
                visit(chunk[start:start+batch_size])
                if time.monotonic()>=deadline:
                    # Remaining chunks have not been sent to the owner.
                    unclassified.extend(tuple(map(int,x)) for x in chunk[start+batch_size:])
                    break
            return
        calls+=1
        pair_ids=[tuple(map(int,x)) for x in chunk]
        try:
            rows,records=owner_module.build_pair_rows(points_mm,faces,pair_ids)
            if len(records)!=len(pair_ids):
                raise ValueError(f"owner returned {len(records)} records for {len(pair_ids)} pairs")
            eligible.extend(pair_ids);row_count+=len(rows)
        except Exception as exc:
            if len(chunk)==1:
                rejected.append({"face_rows":pair_ids[0],"exception_type":type(exc).__name__,"reason":str(exc)})
                return
            mid=len(chunk)//2
            visit(chunk[:mid]);visit(chunk[mid:])
    visit(pairs)
    eligible.sort();rejected.sort(key=lambda r:tuple(r["face_rows"]))
    unclassified.sort()
    return {"eligible_pair_count":len(eligible),"eligible_pairs":eligible,
            "rejected_pair_count":len(rejected),"rejected_pairs":rejected,
            "unclassified_pair_count":len(unclassified),"unclassified_pairs":unclassified,
            "generated_linear_row_count":row_count,"owner_call_count":calls,
            "classification_time_budget_seconds":max_seconds,"classification_completed":not unclassified,
            "rejected_pairs_retained_in_broadphase_output":True}


def load_owner(path: Path, expected_sha: str):
    actual=sha256(path)
    if actual!=expected_sha: raise ValueError(f"pair owner SHA mismatch: {actual}")
    spec=importlib.util.spec_from_file_location("self_separation_026_350",path)
    if spec is None or spec.loader is None: raise ValueError("cannot load pair owner")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not callable(getattr(module,"build_pair_rows",None)): raise ValueError("pinned owner lacks build_pair_rows")
    return module


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-npz",required=True,type=Path)
    ap.add_argument("--raw-trial-npz",required=True,type=Path,action="append")
    ap.add_argument("--seed-pairs",required=True,type=Path,action="append")
    ap.add_argument("--witness-pairs",type=Path,action="append",default=[])
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--scale",type=float,default=2.0)
    ap.add_argument("--padding-m",type=float,default=1e-7)
    ap.add_argument("--pair-owner",required=True,type=Path)
    ap.add_argument("--pair-owner-sha256",required=True)
    ap.add_argument("--max-candidate-pairs",type=int,default=500000)
    ap.add_argument("--classification-timeout-seconds",type=float,default=1200.0)
    args=ap.parse_args(argv)
    if args.max_candidate_pairs<1: raise ValueError("max candidate pair cap must be positive")
    if not np.isfinite(args.classification_timeout_seconds) or args.classification_timeout_seconds<=0:
        raise ValueError("classification timeout must be finite and positive")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with np.load(args.base_npz,allow_pickle=False) as z:
        base=np.asarray(z["points_world_m"])
        faces=np.asarray(z["source_O_faces"])
    trials=[];pins={"base_candidate":sha256(args.base_npz)}
    for i,p in enumerate(args.raw_trial_npz):
        with np.load(p,allow_pickle=False) as z:
            if "points_world_m" not in z.files or "source_O_faces" not in z.files: raise ValueError(f"trial {p} lacks required arrays")
            t=np.asarray(z["points_world_m"]);f=np.asarray(z["source_O_faces"])
            if f.dtype!=faces.dtype or f.shape!=faces.shape or f.tobytes()!=faces.tobytes(): raise ValueError(f"trial topology bytes differ: {p}")
            trials.append(t)
        pins[f"raw_trial_{i}"]=sha256(p)
    seed_parts=[]; witness_parts=[]
    for i,p in enumerate(args.seed_pairs):
        seed_parts.append(np.load(p,allow_pickle=False))
        pins[f"seed_pairs_{i}"]=sha256(p)
    for i,p in enumerate(args.witness_pairs):
        witness_parts.append(np.load(p,allow_pickle=False))
        pins[f"witness_pairs_{i}"]=sha256(p)
    seeds=np.vstack(seed_parts) if seed_parts else np.empty((0,2),dtype=np.int64)
    witnesses=np.vstack(witness_parts) if witness_parts else np.empty((0,2),dtype=np.int64)
    pairs,report=swept_face_aabb_candidates(base,faces,trials,scale=args.scale,
        padding_m=args.padding_m,include_pairs=np.vstack((seeds,witnesses)))
    owner=load_owner(args.pair_owner,args.pair_owner_sha256)
    if len(pairs)>args.max_candidate_pairs:
        classified={"eligible_pair_count":0,"eligible_pairs":[],"rejected_pair_count":0,"rejected_pairs":[],
                    "unclassified_pair_count":len(pairs),"unclassified_pairs":[tuple(map(int,x)) for x in pairs],
                    "generated_linear_row_count":0,"owner_call_count":0,
                    "classification_time_budget_seconds":args.classification_timeout_seconds,
                    "classification_completed":False,"rejected_pairs_retained_in_broadphase_output":True,
                    "classification_stop_reason":"candidate_pair_cap"}
    else:
        classified=classify_pair_rows(base,faces,pairs,owner,max_seconds=args.classification_timeout_seconds)
    eligible=np.asarray(classified.pop("eligible_pairs"),dtype=np.int64).reshape((-1,2))
    rejected=np.asarray([r["face_rows"] for r in classified["rejected_pairs"]],dtype=np.int64).reshape((-1,2))
    unclassified=np.asarray(classified.pop("unclassified_pairs"),dtype=np.int64).reshape((-1,2))
    witness_set={tuple(map(int,row)) for row in canonical_pairs(witnesses,len(faces))}
    rejected_set={tuple(map(int,row)) for row in rejected}
    rejected_witness=sorted(witness_set & rejected_set)
    unclassified_set={tuple(map(int,row)) for row in unclassified}
    unclassified_witness=sorted(witness_set & unclassified_set)
    if rejected_witness:
        report_status="BLOCKED_EXACT_WITNESS_NOT_ROW_ELIGIBLE"
    elif unclassified_witness or len(unclassified):
        report_status="BLOCKED_CLASSIFICATION_INCOMPLETE"
    elif len(rejected):
        report_status="ELIGIBLE_WITH_EXPLICIT_PROSPECTIVE_REJECTIONS"
    else:
        report_status="PAIR_ROWS_ELIGIBLE"
    potential_path=args.output_dir/"self-pairs-potential.npy"
    eligible_path=args.output_dir/"self-pairs-input.npy"
    rejected_path=args.output_dir/"self-pairs-rejected.npy"
    unclassified_path=args.output_dir/"self-pairs-unclassified.npy"
    np.save(potential_path,pairs,allow_pickle=False)
    np.save(eligible_path,eligible,allow_pickle=False)
    np.save(rejected_path,rejected,allow_pickle=False)
    np.save(unclassified_path,unclassified,allow_pickle=False)
    report.update({"status":report_status,"input_pins":pins,
                   "pair_owner_path":str(args.pair_owner),"pair_owner_sha256":args.pair_owner_sha256,
                   "potential_pair_array_sha256":sha256(potential_path),
                   "eligible_pair_array_sha256":sha256(eligible_path),
                   "rejected_pair_array_sha256":sha256(rejected_path),
                   "unclassified_pair_array_sha256":sha256(unclassified_path),
                   "potential_pair_array_shape":list(pairs.shape),
                   "eligible_pair_array_shape":list(eligible.shape),
                   "rejected_pair_array_shape":list(rejected.shape),
                   "unclassified_pair_array_shape":list(unclassified.shape),
                   "max_candidate_pairs":args.max_candidate_pairs,
                   "pair_row_eligibility":classified,
                   "rejected_exact_witness_pairs":[list(x) for x in rejected_witness],
                   "all_exact_witnesses_retained_in_potential_output":witness_set.issubset({tuple(map(int,x)) for x in pairs}),
                   "exact_intersection_audit":"not run by this tool",
                   "acceptance":"none; emitted pairs only seed optional linearized constraints; exact emitted-F32 audit remains mandatory"})
    (args.output_dir/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report_status,"potential":len(pairs),"eligible":len(eligible),
                      "rejected":len(rejected),"unclassified":len(unclassified),
                      "output":str(args.output_dir),"sweep_axis":report["sweep_axis"]},sort_keys=True))

if __name__=="__main__": main()
