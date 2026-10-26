from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def pairkeys(p):
    p=np.asarray(p,dtype=np.uint64)
    return (p[:,0]<<np.uint64(32))|p[:,1]
def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    assert spec.loader is not None;spec.loader.exec_module(mod);return mod

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--base-report",required=True)
    ap.add_argument("--base-rows",required=True)
    ap.add_argument("--required-pair-ids",required=True,action="append")
    ap.add_argument("--row-owner",required=True)
    ap.add_argument("--output-dir",required=True)
    a=ap.parse_args(argv)
    candidate=Path(a.candidate); base_report=Path(a.base_report); base_rows=Path(a.base_rows); owner_path=Path(a.row_owner); out=Path(a.output_dir)
    if out.exists(): raise FileExistsError(out)
    br=json.loads(base_report.read_text())
    candidate_sha=sha(candidate)
    if br.get("candidate_sha256")!=candidate_sha: raise ValueError("base report/candidate SHA mismatch")
    owner_sha=sha(owner_path)
    with np.load(candidate,allow_pickle=False) as z:
        points=np.asarray(z["points_world_m"]);faces=np.asarray(z["source_O_faces"])
    if points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all(): raise ValueError("candidate points must be finite F32 Nx3")
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer): raise ValueError("candidate faces must be integer Fx3")
    required=[]
    required_inputs=[]
    for p in a.required_pair_ids:
        arr=np.load(p,allow_pickle=False)
        if arr.ndim!=2 or arr.shape[1]!=2 or not np.issubdtype(arr.dtype,np.integer): raise ValueError("required pair IDs must be integer Nx2")
        if arr.size and (np.any(arr<0) or np.any(arr[:,0]>=arr[:,1])): raise ValueError("required pairs must be canonical a<b")
        required.append(np.asarray(arr,dtype=np.int64))
        required_inputs.append({"path":p,"sha256":sha(p),"count":len(arr)})
    required=np.unique(np.concatenate(required,axis=0),axis=0)
    with np.load(base_rows,allow_pickle=False) as z:
        ptr=np.asarray(z["row_indptr"],dtype=np.int64); nodes=np.asarray(z["row_node_ids"]); coeff=np.asarray(z["row_coefficients"])
        lower=np.asarray(z["row_lower_mm"]); rowpairs=np.asarray(z["row_pair_ids"])
    if len(ptr)!=len(lower)+1 or ptr[0]!=0 or ptr[-1]!=len(nodes) or len(rowpairs)!=len(lower) or coeff.shape!=(len(nodes),3): raise ValueError("malformed base CSR rows")
    if not np.isfinite(coeff).all() or not np.isfinite(lower).all() or (len(nodes) and (nodes.min()<0 or nodes.max()>=len(points))): raise ValueError("invalid base row contents")
    available=np.unique(rowpairs,axis=0); missing=required[~np.isin(pairkeys(required),pairkeys(available))]
    owner=loadmod(owner_path,"self_separation_026_augment413")
    extra_ptr=[0]; extra_nodes=[]; extra_coeff=[]; extra_lower=[]; extra_pairs=[]; extra_records=[]
    points_mm=points.astype(np.float64)*1000.0
    for pair in missing:
        rows,recs=owner.build_pair_rows(points_mm,faces,pair.reshape(1,2),margin_fraction=0.25)
        if len(recs)!=1 or list(map(int,recs[0].get("faces",())))!=list(map(int,pair)): raise ValueError(f"owner record mismatch for {pair.tolist()}")
        if not rows: raise ValueError(f"owner returned no guard rows for {pair.tolist()}")
        for r in rows:
            rn=np.asarray(r.get("nodes"));rc=np.asarray(r.get("coefficients"),dtype=np.float64);rl=float(r.get("lower"))
            if rn.ndim!=1 or not np.issubdtype(rn.dtype,np.integer) or len(rn)==0 or rc.shape!=(len(rn),3): raise ValueError(f"malformed row for {pair.tolist()}")
            if rn.min()<0 or rn.max()>=len(points) or not np.isfinite(rc).all() or not np.isfinite(rl): raise ValueError(f"invalid row for {pair.tolist()}")
            if rl>2e-8: raise ValueError(f"mandatory pair row not zero-step feasible: {pair.tolist()} rhs={rl}")
            extra_nodes.append(rn.astype(np.int32));extra_coeff.append(rc);extra_lower.append(rl);extra_pairs.append(pair.astype(np.int32));extra_ptr.append(extra_ptr[-1]+len(rn))
        extra_records.append({"pair":pair.astype(int).tolist(),"row_count":len(rows),"owner_record":recs[0]})
    eptr=np.asarray(extra_ptr,dtype=np.int64)
    eni=np.concatenate(extra_nodes) if extra_nodes else np.empty(0,np.int32)
    eco=np.concatenate(extra_coeff,axis=0) if extra_coeff else np.empty((0,3),np.float64)
    elo=np.asarray(extra_lower,dtype=np.float64)
    epa=np.asarray(extra_pairs,dtype=np.int32).reshape((-1,2))
    merged_ptr=np.concatenate((ptr,ptr[-1]+eptr[1:]))
    merged_nodes=np.concatenate((nodes,eni))
    merged_coeff=np.concatenate((coeff,eco),axis=0)
    merged_lower=np.concatenate((lower,elo))
    merged_pairs=np.concatenate((rowpairs,epa),axis=0)
    if np.any(merged_lower>2e-8): raise ValueError("augmented rows are not zero-step feasible")
    out.mkdir(parents=True)
    np.save(out/"augmented-mandatory-pair-ids.npy",missing,allow_pickle=False)
    np.savez_compressed(out/"guard-026-rows-augmented.npz",row_indptr=merged_ptr,row_node_ids=merged_nodes,row_coefficients=merged_coeff,row_lower_mm=merged_lower,row_pair_ids=merged_pairs)
    report={
        "schema":"numi.human.prospective-self-guards-413.mandatory-row-augmentation.v1",
        "status":"COMPLETE_ZERO_STEP_FEASIBLE",
        "semantics":"Only required pair rows absent from the frozen 410 guard payload were appended, rebuilt at exact selected-410 F32 coordinates. No geometry or physical clearance claim.",
        "candidate_path":str(candidate),"candidate_sha256":candidate_sha,"candidate_points_sha256":arrsha(points),"candidate_faces_sha256":arrsha(faces),"point_count":len(points),"face_count":len(faces),
        "base_guard_report_path":str(base_report),"base_guard_report_sha256":sha(base_report),"base_guard_rows_path":str(base_rows),"base_guard_rows_sha256":sha(base_rows),
        "row_owner_path":str(owner_path),"row_owner_sha256":owner_sha,"required_pair_inputs":required_inputs,
        "required_pair_count":len(required),"base_pair_count":len(available),"missing_required_pair_count":len(missing),"extra_guard_row_count":len(elo),"extra_coefficient_nnz":len(eni),
        "augmented_pair_count":len(np.unique(merged_pairs,axis=0)),"augmented_row_count":len(merged_lower),"augmented_zero_step_max_rhs_mm":float(merged_lower.max(initial=-float("inf"))),
        "added_pair_records":extra_records,
        "outputs":{"extra_pair_ids":{"path":str(out/"augmented-mandatory-pair-ids.npy"),"sha256":sha(out/"augmented-mandatory-pair-ids.npy")},"rows":{"path":str(out/"guard-026-rows-augmented.npz"),"sha256":sha(out/"guard-026-rows-augmented.npz")}},
    }
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ["status","required_pair_count","missing_required_pair_count","extra_guard_row_count","augmented_row_count","augmented_zero_step_max_rhs_mm"]},sort_keys=True))
if __name__=="__main__":main()
