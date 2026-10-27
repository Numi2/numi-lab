#!/usr/bin/env python3
"""Same-topology exact self-seed merge. Preparation only; never fits or audits."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
DEFAULT_REFRESH=ROOT/"D/surface-clearance-546/refresh_surface_constraints546.py"
DEFAULT_PAIR=ROOT/"publication-101/evidence/204/source/fit/self_separation_026.py"
DEFAULT_PACK=ROOT/"D/focused-clearance-541/constraints/refresh_guards_541.py"
PIN={"refresh546":"a9586a81d92f117040e774c980b6028168432386b2e9babd96a7b7f725175d09",
     "owner026":"c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3",
     "pack541":"293e4da49a10f07420d1da7c512823dcb182dca246e411c3775d18866b27a241"}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def read_json(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise ValueError(f"expected JSON object: {p}")
    return x
def loadnpz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def loadmod(name,p):
    s=importlib.util.spec_from_file_location(name,p)
    if s is None or s.loader is None:raise ImportError(f"cannot load owner: {p}")
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def canonical_pairs(pairs,nfaces,label):
    q=np.asarray(pairs)
    if q.ndim!=2 or q.shape[1]!=2 or q.dtype.kind not in "iu":raise ValueError(f"{label} must be integer (K,2)")
    q=np.asarray(q,dtype=np.int64).copy()
    if len(q) and (q.min()<0 or q.max()>=nfaces):raise ValueError(f"{label} has out-of-range face index")
    if np.any(q[:,0]==q[:,1]):raise ValueError(f"{label} has identical face rows")
    q.sort(axis=1);u=np.unique(q,axis=0)
    if len(u)!=len(q):raise ValueError(f"{label} contains duplicate pairs")
    return u
def exact_face_remap(owner,source_faces,dest_faces,pairs,label):
    a=np.asarray(source_faces);b=np.asarray(dest_faces)
    if a.ndim!=2 or a.shape[1]!=3 or b.shape!=a.shape:raise ValueError(f"{label} face shape mismatch")
    if not np.array_equal(a,b):raise ValueError(f"{label} topology differs; identity remap invalid")
    q=canonical_pairs(pairs,len(a),label);m=owner.map_seed_pairs_partial(a,b,q)
    if np.any(m["source_status_code"]!=0) or not np.array_equal(m["mapped_pairs"],q):
        raise ValueError(f"{label} did not identity-map uniquely")
    return m
def merge_seed_pairs(old,new):
    a=np.asarray(old,dtype=np.int64).reshape(-1,2);b=np.asarray(new,dtype=np.int64).reshape(-1,2)
    overlap=sorted({tuple(map(int,x)) for x in a}&{tuple(map(int,x)) for x in b})
    merged=np.unique(np.vstack((a,b)),axis=0) if len(a)+len(b) else np.empty((0,2),np.int64)
    if len(merged)!=len(a)+len(b)-len(overlap):raise ValueError("merged pairs are not the set union")
    return merged,overlap
def arg_value_index(argv,flag):
    hits=[i for i,x in enumerate(argv) if x==flag]
    if len(hits)!=1 or hits[0]+1>=len(argv):raise ValueError(f"argv needs exactly one {flag} value")
    return hits[0]+1

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for flag in ("base-candidate","current-seed-pairs","current-seed-report","failed-candidate","failed-audit130","failed-self-pairs","fit-argv","output"):
        ap.add_argument("--"+flag,required=True)
    ap.add_argument("--refresh-owner",default=str(DEFAULT_REFRESH));ap.add_argument("--pair-owner",default=str(DEFAULT_PAIR));ap.add_argument("--pack-owner",default=str(DEFAULT_PACK))
    c=ap.parse_args(argv)
    base,oldpairs,oldreport,failed,audit,newpairs,fitargv,out=map(Path,[c.base_candidate,c.current_seed_pairs,c.current_seed_report,c.failed_candidate,c.failed_audit130,c.failed_self_pairs,c.fit_argv,c.output])
    if out.exists():raise FileExistsError(f"output already exists: {out}")
    inputs=[base,oldpairs,oldreport,failed,audit,newpairs,fitargv]
    if any(not p.is_file() for p in inputs):raise FileNotFoundError(next(p for p in inputs if not p.is_file()))
    refresh,pair,pack=map(Path,[c.refresh_owner,c.pair_owner,c.pack_owner])
    owner_hashes={"refresh546":sha(refresh),"owner026":sha(pair),"pack541":sha(pack)}
    if owner_hashes!=PIN:raise ValueError(f"pinned owners changed: {owner_hashes}")
    base_sha,old_sha,failed_sha=sha(base),sha(oldpairs),sha(failed)
    cr=read_json(oldreport)
    if cr.get("candidate_sha256")!=base_sha:raise ValueError("current seed report candidate mismatch")
    if cr.get("output_pairs_sha256")!=old_sha or Path(cr.get("output_pairs_path","")).resolve()!=oldpairs.resolve():raise ValueError("current seed report pair path/hash mismatch")
    if cr.get("current_exact_self_pair_count",0)!=0:raise ValueError("current base lacks zero-self seed-report status")
    ar=read_json(audit)
    if ar.get("candidate_sha256")!=failed_sha:raise ValueError("failed audit candidate mismatch")
    if ar.get("complete_surface_audit") is not True:raise ValueError("failed audit incomplete")
    exact_count=ar.get("exact_PLC_surface_pairs")
    if not isinstance(exact_count,int) or exact_count<0:raise ValueError("failed audit exact pair count invalid")
    base_npz,failed_npz=loadnpz(base),loadnpz(failed)
    f=np.asarray(base_npz.get("source_O_faces"));ff=np.asarray(failed_npz.get("source_O_faces"));p=np.asarray(base_npz.get("points_world_m"))
    if f.ndim!=2 or f.shape[1]!=3 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError("base geometry arrays invalid")
    old=canonical_pairs(np.load(oldpairs,allow_pickle=False),len(f),"current seeds")
    new=canonical_pairs(np.load(newpairs,allow_pickle=False),len(ff),"failed witnesses")
    if len(new)!=exact_count:raise ValueError(f"exact audit says {exact_count} pairs, file has {len(new)}")
    owner=loadmod("refresh546_seedmerge609",refresh)
    oldmap=exact_face_remap(owner,f,f,old,"current seeds")
    newmap=exact_face_remap(owner,ff,f,new,"failed witnesses")
    merged,overlap=merge_seed_pairs(oldmap["mapped_pairs"],newmap["mapped_pairs"])
    args=json.loads(fitargv.read_text())
    if not isinstance(args,list) or not args or not all(isinstance(x,str) for x in args):raise ValueError("fit argv must be string array")
    bi=arg_value_index(args,"--base-candidate");si=arg_value_index(args,"--self-pairs")
    if Path(args[bi]).resolve()!=base.resolve() or Path(args[si]).resolve()!=oldpairs.resolve():raise ValueError("fit argv does not match supplied base/seeds")
    pairmod=loadmod("selfseparation026_seedmerge609",pair)
    rows,records=pairmod.build_pair_rows(p.astype(np.float64)*1000.0,f,merged,margin_fraction=.25)
    rhs=max((float(r["lower"]) for r in rows),default=0.0)
    if not np.isfinite(rhs) or rhs>2e-8:raise ValueError(f"026 zero-step certificate fails: {rhs}")
    packmod=loadmod("refresh541_pack_seedmerge609",pack);packed=packmod.pack(rows)
    out.mkdir(parents=True,exist_ok=False);sd=out/"self-seeds";sd.mkdir()
    mergedpath=sd/"merged-self-pair-seeds.npy";np.save(mergedpath,merged)
    rowpath=sd/"merged-self-pair-rows.npz";np.savez_compressed(rowpath,**packed)
    # Write exact mapping payloads using the pinned owner result.
    # Recompute mappings here to keep variable names distinct from artifact paths.
    old_mapping=owner.map_seed_pairs_partial(f,f,old)
    new_mapping=owner.map_seed_pairs_partial(ff,f,new)
    oldmap_path=sd/"current-seed-remap.npz";np.savez_compressed(oldmap_path,**old_mapping["payload"])
    newmap_path=sd/"failed-trial-remap.npz";np.savez_compressed(newmap_path,**new_mapping["payload"])
    argsout=list(args);argsout[arg_value_index(argsout,"--self-pairs")-1]=str(mergedpath);argsout[arg_value_index(argsout,"--output")-1]=str(out/"fit")
    fitout=out/"fit-argv.json";write_json(fitout,argsout)
    source_inputs=inputs+[refresh,pair,pack]
    sourcepins={str(x):sha(x) for x in source_inputs};sourcepins[str(Path(__file__))]=sha(__file__)
    if any(sha(x)!=sourcepins[str(x)] for x in source_inputs):raise RuntimeError("input changed during preparation")
    report={"schema":"numi.human.continuation-self-seeds.v1","status":"PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY","mapping_status":"PASS_PARTIAL_EXACT_FACE_REMAP_AND_ZERO_STEP_026_CERTIFICATE","candidate_path":str(base),"candidate_sha256":base_sha,"current_exact_self_pair_count":0,"zero_step_feasible":True,"zero_step_max_rhs_mm":rhs,"rebuilt_linear_row_count":len(rows),"merged_unique_pair_count":len(merged),"native_admission":False,"accepted_physical_steps":0,"construction_only":True,
      "current_seed_source":{"pairs_path":str(oldpairs),"pairs_sha256":old_sha,"source_pair_count":len(old),"mapped_unique_pair_count":len(old_mapping["mapped_pairs"]),"dropped_pair_count":int(np.count_nonzero(old_mapping["source_status_code"]!=0)),"mapping_payload_path":str(oldmap_path),"mapping_payload_sha256":sha(oldmap_path)},
      "failed_trial_source":{"candidate":str(failed),"candidate_sha256":failed_sha,"audit130_path":str(audit),"audit130_sha256":sha(audit),"pairs_path":str(newpairs),"pairs_sha256":sha(newpairs),"source_pair_count":len(new),"mapped_unique_pair_count":len(new_mapping["mapped_pairs"]),"dropped_pair_count":int(np.count_nonzero(new_mapping["source_status_code"]!=0)),"exact_audit_pair_count":exact_count,"mapping_payload_path":str(newmap_path),"mapping_payload_sha256":sha(newmap_path)},
      "deduplication":{"overlap_count":len(overlap),"overlap_pairs":[list(x) for x in overlap],"merged_count_equals_union":len(merged)==len(old)+len(new)-len(overlap)},
      "overlap_pair_diagnostics":[{"face_rows":list(map(int,r["faces"])),"kind":r.get("kind"),"shared_vertex":r.get("shared_vertex"),"minimum_vertex_side_gap_mm":r.get("minimum_vertex_side_gap_mm"),"preserved_margin_mm":r.get("preserved_margin_mm")} for r in records if tuple(sorted(map(int,r["faces"]))) in set(overlap)],
      "output_pairs_path":str(mergedpath),"output_pairs_sha256":sha(mergedpath),"output_rows_path":str(rowpath),"output_rows_sha256":sha(rowpath),"fit_argv_path":str(fitout),"fit_argv_sha256":sha(fitout),"source_pins":sourcepins,"limitation":"Seed refresh and zero-step 026 certificate only; no fit or exact post-fit audit was run."}
    reportpath=sd/"seed-preflight.json";write_json(reportpath,report)
    write_json(out/"fit-plan.json",{"status":"PREPARED_NOT_LAUNCHED","fit_argv_path":str(fitout),"fit_argv_sha256":sha(fitout),"seed_preflight_path":str(reportpath),"seed_preflight_sha256":sha(reportpath),"source_pins":sourcepins,"native_admission":False,"accepted_physical_steps":0})
    print(json.dumps({"status":report["status"],"merged_pairs":len(merged),"overlap_pairs":len(overlap),"rows":len(rows),"zero_step_max_rhs_mm":rhs,"output":str(out),"fit_argv":str(fitout),"fit_status":"PREPARED_NOT_LAUNCHED"},sort_keys=True))
if __name__=="__main__":main()
