"""Inert 545/315 batch selector; output still requires complete 130/159 checks."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

W=Path(__file__).resolve().parent
EVID=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PRED=EVID/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PRED_SHA="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_rows(rows, targets, npoints, anchors):
    rows=np.asarray(rows); targets=np.asarray(targets); anchors=np.asarray(anchors,dtype=np.int64)
    if rows.ndim!=1 or rows.dtype.kind not in "iu": raise ValueError("query_rows must be integer vector")
    rows=rows.astype(np.int64,copy=False)
    if targets.shape!=(len(rows),3) or targets.dtype!=np.float32 or not np.isfinite(targets).all(): raise ValueError("invalid Float32 target endpoints")
    if np.any(rows<0) or np.any(rows>=npoints): raise ValueError("target row outside candidate")
    if len(np.unique(rows))!=len(rows): raise ValueError("duplicate or conflicting target rows")
    if np.intersect1d(rows,anchors).size: raise ValueError("target attempts to move a typed anchor")
    return rows

def proposal(base,rows,targets,keep):
    y=np.asarray(base).copy(); rows=np.asarray(rows,dtype=np.int64); keep=np.asarray(keep,dtype=bool)
    y[rows[keep]]=np.asarray(targets,dtype=np.float32)[keep]
    moved=np.zeros(len(y),dtype=bool); moved[rows[keep]]=True
    if not np.array_equal(y[~moved],np.asarray(base)[~moved]): raise AssertionError("non-target coordinate changed")
    return y

def arithmetic_witnesses(y,x,F,ref,src,trust_mm=1.,source_cap_mm=12.):
    y=np.asarray(y);x=np.asarray(x);F=np.asarray(F,dtype=np.int64);ref=np.asarray(ref,dtype=float);src=np.asarray(src,dtype=float)
    if y.dtype!=np.float32 or x.dtype!=np.float32 or y.shape!=x.shape or not np.isfinite(y).all(): raise ValueError("invalid emitted candidate")
    if F.ndim!=2 or F.shape[1]!=3 or F.min()<0 or F.max()>=len(x): raise ValueError("invalid faces")
    xm=x.astype(float)*1000; ym=y.astype(float)*1000; rm=ref*1000; sm=src*1000
    cb=np.cross(xm[F[:,1]]-xm[F[:,0]],xm[F[:,2]]-xm[F[:,0]])
    cy=np.cross(ym[F[:,1]]-ym[F[:,0]],ym[F[:,2]]-ym[F[:,0]])
    cr=np.cross(rm[F[:,1]]-rm[F[:,0]],rm[F[:,2]]-rm[F[:,0]])
    den=np.einsum("ij,ij->i",cb,cb); mag=np.linalg.norm(cr,axis=1)
    if np.any(den<=0) or np.any(mag<=0): raise ValueError("degenerate face")
    badf=np.flatnonzero((np.einsum("ij,ij->i",cy,cb)/den<.1)|(np.linalg.norm(cy,axis=1)/mag<.1))
    E=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
    L=np.linalg.norm(xm[E[:,0]]-xm[E[:,1]],axis=1)
    if np.any(L<=0): raise ValueError("degenerate edge")
    d=ym-xm; ed=np.max(np.abs(d[E[:,0]]-d[E[:,1]]),axis=1)
    bade=np.flatnonzero(ed>.5*L+1e-8)
    step=np.linalg.norm(d,axis=1); sd=np.linalg.norm(ym-sm,axis=1)
    bt=np.flatnonzero(step>trust_mm+1e-6); bs=np.flatnonzero(sd>source_cap_mm+1e-6)
    return {"bad_area_faces":badf,"bad_edge_rows":bade,"bad_trust_nodes":bt,"bad_source_nodes":bs,"bad_bound_nodes":np.union1d(bt,bs).astype(np.int64)}

def contributors(rows,F,E,w,self_pairs):
    nodes=set(); record={k:[int(v) for v in w.get(k,())] for k in ("bad_area_faces","bad_edge_rows","bad_bound_nodes")}
    record["self_pairs"]=[]
    for fi in record["bad_area_faces"]:
        if not 0<=fi<len(F): raise ValueError("area witness face out of range")
        nodes.update(map(int,F[fi]))
    for ei in record["bad_edge_rows"]:
        if not 0<=ei<len(E): raise ValueError("edge witness row out of range")
        nodes.update(map(int,E[ei]))
    nodes.update(record["bad_bound_nodes"]); active=set(map(int,rows)); drop=active.intersection(nodes)
    for pair in self_pairs:
        if len(pair)!=2: raise ValueError("invalid self-pair witness")
        a,b=map(int,pair)
        if a==b or min(a,b)<0 or max(a,b)>=len(F): raise ValueError("self-pair face out of range")
        record["self_pairs"].append([a,b]); drop.update(active.intersection(map(int,F[a]))); drop.update(active.intersection(map(int,F[b])))
    return drop,record

class BatchRoundsExhausted(RuntimeError):
    def __init__(self,history,remaining): super().__init__("unresolved after four witness-removal rounds"); self.history=history; self.remaining=[int(v) for v in remaining]

def select_batch(base,F,rows,targets,anchors,arithmetic_check,self_check,max_rounds=4):
    x=np.asarray(base);F=np.asarray(F,dtype=np.int64);rows=validate_rows(rows,targets,len(x),anchors);targets=np.asarray(targets,dtype=np.float32)
    if F.ndim!=2 or F.shape[1]!=3 or F.min()<0 or F.max()>=len(x): raise ValueError("invalid faces")
    if not 1<=max_rounds<=4: raise ValueError("max_rounds must be 1..4")
    E=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
    keep=np.ones(len(rows),bool);hist=[]
    for nround in range(1,max_rounds+1):
        y=proposal(x,rows,targets,keep); active=rows[keep]; touched=np.flatnonzero(np.isin(F,active).any(axis=1))
        w=arithmetic_check(y); bad=any(len(w.get(k,())) for k in ("bad_area_faces","bad_edge_rows","bad_bound_nodes"))
        pairs=[] if bad or not len(active) else list(self_check(y,touched))
        if not bad and not pairs:
            return {"status":"BATCH_ENDPOINTS_PENDING_FULL_130_159_CHECKS","points":y,"retained":rows[keep].copy(),"removed":rows[~keep].copy(),"history":hist,"rounds":nround}
        dropnodes,wr=contributors(rows[keep],F,E,w,pairs); mask=keep&np.isin(rows,list(dropnodes)); removed=rows[mask]
        if not len(removed): raise RuntimeError("bad witness cannot be attributed to a remaining endpoint")
        hist.append({"round":nround,"witnesses":wr,"removed_node_rows":removed.tolist(),"remaining_operations":int(keep.sum()-mask.sum())});keep[mask]=False
    raise BatchRoundsExhausted(hist,rows[keep])

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def validate_baseline(path):
    st=json.loads(Path(path).read_text())
    for k in ("candidate","point_summary","surface_report"):
        if sha(st[k])!=st[k+"_sha256"]: raise ValueError("baseline pin mismatch: "+k)
    p=json.loads(Path(st["point_summary"]).read_text());s=json.loads(Path(st["surface_report"]).read_text());h=st["candidate_sha256"]
    if p.get("candidate_sha256")!=h or p.get("complete_surface_audit") is not True or p.get("exact_PLC_surface_pairs")!=0: raise ValueError("baseline is not complete zero-self")
    if s.get("candidate_sha256")!=h or s.get("complete") is not True: raise ValueError("baseline 159 receipt stale/incomplete")
    return st,p,s

def validate_targets(report_path,npz_path,base_sha):
    r=json.loads(Path(report_path).read_text()); zpath=Path(npz_path)
    if r.get("status")!="PASS_FIRST_UNION_EXIT_TARGETS_ONLY" or r.get("inputs",{}).get("candidate_sha256")!=base_sha: raise ValueError("target report status/baseline mismatch")
    o=r.get("outputs",{})
    if o.get("npz_sha256")!=sha(zpath) or Path(o.get("npz","")).resolve()!=zpath.resolve(): raise ValueError("target NPZ/report hash or path mismatch")
    n=r.get("query_rows")
    if type(n)is not int or n<1 or r.get("passing_targets")!=n or r.get("all_bank_ray_results")!=n or r.get("physical_steps")!=0: raise ValueError("target report does not certify all endpoints")
    z=np.load(zpath,allow_pickle=False); rows=z["query_rows"]; T=z["target_world_f32_m"]
    if len(rows)!=n or T.shape!=(n,3) or T.dtype!=np.float32: raise ValueError("target array/report shape mismatch")
    for k in ("pass_mask","target_exactly_outside_all185"):
        if z[k].shape!=(n,) or z[k].dtype!=np.bool_ or not z[k].all(): raise ValueError("target array lacks all185 certification: "+k)
    items=r.get("targets")
    if not isinstance(items,list) or len(items)!=n: raise ValueError("report row details missing")
    for i,item in enumerate(items):
        if item.get("query_row")!=int(rows[i]) or item.get("status")!="PASS_FIRST_UNION_EXIT": raise ValueError("target report row identity/status mismatch")
        if not np.array_equal(np.asarray(item.get("target_world_f32_m"),dtype=np.float32),T[i]): raise ValueError("report endpoint differs from Float32 target array")
        if item.get("target_exactly_outside_all185") is not True or item.get("exact_endpoint_bank_count")!=185 or item.get("end_outside_all185_by_event_parity") is not True: raise ValueError("target report row lacks complete all185 certificate")
        sel=item.get("selection",{})
        if sel.get("target_exactly_outside_all_banks") is not True or sel.get("classified_bank_count")!=185: raise ValueError("target selection row lacks complete all185 certificate")
    return r,z,rows,T

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--target-report",type=Path,required=True);ap.add_argument("--targets",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--trust-mm",type=float,default=1.);ap.add_argument("--source-cap-mm",type=float,default=12.);a=ap.parse_args()
    start=time.monotonic();st,base130,base159=validate_baseline(a.state);tr,t,rows,T=validate_targets(a.target_report,a.targets,st["candidate_sha256"])
    bpath=Path(st["candidate"]);z=np.load(bpath,allow_pickle=False);x=z["points_world_m"];F=z["source_O_faces"];anchors=z["typed_coarse_anchor_output_rows_018"];rows=validate_rows(rows,T,len(x),anchors)
    checks=loadmod(Path("/Users/numi/Work/numi-human-connected-surface-069/D/contained-clearance-545/run_candidate_checks545.py"),"checks545_batch546");owner=checks.mod(checks.OWNER)
    with np.load(bpath,allow_pickle=False) as bz: src,levels=owner.compose_source_reference(bz,checks.R0)
    if sha(PRED)!=PRED_SHA: raise ValueError("pinned 315 owner changed")
    sys.path.insert(0,str(EVID/"incremental-exact-audit-owner-253/src"));exact=loadmod(PRED,"numilab_human._batch_endpoint546")
    ar=lambda y:arithmetic_witnesses(y,x,F,z["area_reference_037_points_world_m_f64"],src,a.trust_mm,a.source_cap_mm)
    def selfcheck(y,touched):
        records,_=exact.float32_surface_records(y,F,source_seam_groups=())
        return exact._audit_self_pairs_touching_faces(records,set(map(int,touched)))["triangle_pairs"]
    try: result=select_batch(x,F,rows,T,anchors,ar,selfcheck)
    except BatchRoundsExhausted as e:
        a.out.mkdir(parents=True,exist_ok=False);(a.out/"report.json").write_text(json.dumps({"status":"UNRESOLVED_WITHIN_FOUR_ROUNDS","baseline_state":str(a.state),"baseline_state_sha256":sha(a.state),"history":e.history,"remaining_node_rows":e.remaining,"base_sha256":st["candidate_sha256"],"target_sha256":sha(a.targets),"source_sha256":sha(__file__)},indent=2)+"\n");raise
    a.out.mkdir(parents=True,exist_ok=False);payload={k:z[k] for k in z.files};payload["points_world_m"]=result["points"];np.savez_compressed(a.out/"candidate.npz",**payload)
    report={"status":result["status"],"baseline_state":str(a.state),"baseline_state_sha256":sha(a.state),"candidate_path":str(a.out/"candidate.npz"),"candidate_sha256":sha(a.out/"candidate.npz"),"base_candidate":str(bpath),"base_sha256":st["candidate_sha256"],"baseline_130_sha256":sha(st["point_summary"]),"baseline_159_sha256":sha(st["surface_report"]),"target_report":str(a.target_report),"target_report_sha256":sha(a.target_report),"target_npz":str(a.targets),"target_sha256":sha(a.targets),"source_sha256":sha(__file__),"545_check_owner_sha256":sha(Path("/Users/numi/Work/numi-human-connected-surface-069/D/contained-clearance-545/run_candidate_checks545.py")),"315_owner_sha256":PRED_SHA,"source_reference_levels":levels,"trust_mm":a.trust_mm,"source_cap_mm":a.source_cap_mm,"retained_node_rows":result["retained"].tolist(),"removed_node_rows":result["removed"].tolist(),"retained_operations":len(result["retained"]),"removed_operations":len(result["removed"]),"removal_rounds":result["rounds"],"history":result["history"],"non_target_coordinates_bitwise_unchanged":True,"construction_only":True,"native_admission":False,"accepted_physical_steps":0,"seconds":time.monotonic()-start,"next_required_check":"complete exact audits 130 and 159"}
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:report[k] for k in ("status","retained_operations","removed_operations","candidate_sha256","seconds")}))

if __name__=="__main__": main()
