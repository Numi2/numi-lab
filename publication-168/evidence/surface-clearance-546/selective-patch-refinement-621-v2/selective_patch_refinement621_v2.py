#!/usr/bin/env python3
"""621 v2: rerun the pinned patch selector after excluding only certified infeasible midpoint directions."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546"
V1=D/"selective-patch-refinement-621/selective_patch_refinement621.py"
V1_SHA="574be9d48eb23cb72243614da3fa8c2a799403b17ec3d13c9e3521dd33aca966"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def edge(x):
 a,b=map(int,x); return (a,b) if a<b else (b,a)
def select_remaining_edges(selected,rows,parent_edges,prefix,infeasible):
 n=int(prefix); r=np.asarray(rows); e=np.asarray(parent_edges)
 if r.ndim!=1 or r.dtype.kind not in "iu" or e.shape!=(len(r),2) or e.dtype.kind not in "iu": raise ValueError("midpoint mapping schema invalid")
 if len(set(map(int,r)))!=len(r) or len(set(map(edge,e)))!=len(e): raise ValueError("duplicate midpoint mapping")
 if not np.array_equal(np.sort(r.astype(np.int64)),np.arange(n,n+len(r))): raise ValueError("midpoint rows not contiguous appended suffix")
 sel=[edge(x) for x in selected]
 if not sel or len(set(sel))!=len(sel) or any(a<0 or b<=a for a,b in sel): raise ValueError("invalid selected edges")
 by={edge(x):int(q) for x,q in zip(e,r)}
 if set(by)!=set(sel): raise ValueError("selected edges do not match midpoint ancestry")
 bad=[int(q) for q in infeasible]
 if len(set(bad))!=len(bad): raise ValueError("duplicate infeasible row")
 if any(q<n or q>=n+len(r) for q in bad): raise ValueError("infeasible witness is not an appended midpoint row")
 excluded={edge(e[q-n]) for q in bad}
 if not excluded<=set(sel): raise ValueError("infeasible parent edge outside selected patch")
 keep=[list(x) for x in sel if x not in excluded]
 if not keep: raise ValueError("infeasible rows remove entire patch")
 return keep,[{"query_row":q,"parent_edge":list(edge(e[q-n]))} for q in sorted(bad)]

def validate_cone(cone,cone_path,ref,refpath,failed,fc,fsha,point):
 if cone.get("status")!="JOINT_POINT_DIRECTION_FEASIBILITY_DIAGNOSTIC": raise ValueError("unexpected cone status")
 if Path(cone.get("candidate","")).resolve()!=fc.resolve() or cone.get("candidate_sha256")!=fsha: raise ValueError("cone candidate pin mismatch")
 if Path(cone.get("refinement_report","")).resolve()!=refpath.resolve() or cone.get("refinement_report_sha256")!=sha(refpath): raise ValueError("cone refinement pin mismatch")
 if cone.get("minimum_cosine")!=0.05 or cone.get("all185_endpoints_certified") is not False or cone.get("native_admission") is not False: raise ValueError("cone diagnostic has wrong contract")
 normal=Path(cone.get("normal_audit","")).resolve()
 if not normal.is_file() or sha(normal)!=cone.get("normal_audit_sha256"): raise ValueError("cone normal-audit pin mismatch")
 pins=cone.get("source_pins")
 if not isinstance(pins,dict) or not pins: raise ValueError("cone source pins missing")
 for path,want in pins.items():
  if not Path(path).is_file() or sha(path)!=want: raise ValueError("cone source pin changed: "+str(path))
 ws=point.get("inside_witnesses",[]); expect={int(x["output_node"]) for x in ws}
 rec=cone.get("rows")
 if len(expect)!=int(point.get("inside_nodes",-1)) or not isinstance(rec,list) or len(rec)!=len(expect) or int(cone.get("target_row_count",-1))!=len(expect): raise ValueError("cone records do not cover point witnesses")
 ids=[int(x["query_row"]) for x in rec]
 if len(ids)!=len(set(ids)) or set(ids)!=expect: raise ValueError("cone rows differ from candidate-bound inside rows")
 bad=[int(x) for x in cone.get("infeasible_rows",[])]
 if len(bad)!=len(set(bad)) or not set(bad)<=expect: raise ValueError("invalid infeasible row list")
 b=set(bad)
 if any(x.get("status")!=("INFEASIBLE_OR_NUMERICALLY_UNRESOLVED" if int(x["query_row"]) in b else "FEASIBLE") for x in rec): raise ValueError("cone status/list mismatch")
 if int(cone.get("feasible_count",-1))!=len(expect)-len(b) or int(cone.get("infeasible_count",-1))!=len(b): raise ValueError("cone counts mismatch")
 if failed.get("candidate_sha256")!=fsha or failed.get("refinement_report_sha256")!=sha(refpath): raise ValueError("failed state pins mismatch")
 if ref.get("input_candidate_sha256")!=failed.get("original_baseline_candidate_sha256"): raise ValueError("failed refinement baseline mismatch")
 return sorted(b)

def load_v1():
 if sha(V1)!=V1_SHA: raise ValueError("621 v1 source hash mismatch")
 s=importlib.util.spec_from_file_location("621_v1_pinned",V1)
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def validate_failed_receipts(failed,ref,point_path,point,state_path,check_path,refpath,candidate,parent,parent_sha):
 if Path(failed.get("candidate","")).resolve()!=candidate.resolve() or failed.get("candidate_sha256")!=sha(candidate): raise ValueError("failed candidate pin mismatch")
 if Path(failed.get("point_summary","")).resolve()!=point_path.resolve() or failed.get("point_summary_sha256")!=sha(point_path): raise ValueError("failed point-summary pin mismatch")
 if point.get("candidate_sha256")!=sha(candidate): raise ValueError("point summary candidate hash mismatch")
 if Path(failed.get("refinement_report","")).resolve()!=refpath.resolve() or failed.get("refinement_report_sha256")!=sha(refpath): raise ValueError("failed refinement report pin mismatch")
 if Path(ref.get("input_candidate_path","")).resolve()!=parent.resolve() or ref.get("input_candidate_sha256")!=parent_sha: raise ValueError("refinement input candidate pin mismatch")
 if Path(ref.get("output_candidate_path","")).resolve()!=candidate.resolve() or ref.get("output_candidate_sha256")!=sha(candidate): raise ValueError("refinement output candidate pin mismatch")
 if Path(ref.get("input_state_path","")).resolve()!=state_path.resolve() or ref.get("input_state_sha256")!=sha(state_path): raise ValueError("refinement input state pin mismatch")
 if Path(ref.get("input_construction_report_path","")).resolve()!=check_path.resolve() or ref.get("input_construction_report_sha256")!=sha(check_path): raise ValueError("refinement input check-report pin mismatch")
 return True

def prepare(state_path,check_path,cone_path,failed_state_path,failed_report_path,outdir,max_edges=256):
 m=load_v1()
 state_path,check_path,cone_path,failed_state_path,failed_report_path=map(lambda p:Path(p).resolve(),(state_path,check_path,cone_path,failed_state_path,failed_report_path))
 outdir=Path(outdir)
 if outdir.exists(): raise FileExistsError(str(outdir))
 st=json.loads(state_path.read_text()); parent=Path(st["candidate"]).resolve(); ph=sha(parent)
 if st.get("candidate_sha256")!=ph: raise ValueError("strict parent candidate pin mismatch")
 fs=json.loads(failed_state_path.read_text()); fc=Path(fs["candidate"]).resolve(); fh=sha(fc)
 ref=json.loads(failed_report_path.read_text()); cone=json.loads(cone_path.read_text()); point_path=Path(fs["point_summary"]).resolve(); point=json.loads(point_path.read_text())
 validate_failed_receipts(fs,ref,point_path,point,state_path,check_path,failed_report_path,fc,parent,ph)
 bad=validate_cone(cone,cone_path,ref,failed_report_path,fs,fc,fh,point)
 if fs.get("original_baseline_candidate_sha256")!=ph: raise ValueError("failed refinement is not rooted at supplied strict parent")
 n=int(ref["parent_prefix_node_count"]); rows=np.asarray(ref["edge_midpoint_rows"],dtype=np.int64)
 with np.load(parent,allow_pickle=False) as p,np.load(fc,allow_pickle=False) as c:
  xp=np.asarray(p["points_world_m"]); xc=np.asarray(c["points_world_m"])
  if len(xp)!=n or not m.same(xp,xc[:n]) or not np.array_equal(rows,np.arange(n,len(xc))): raise ValueError("parent prefix or appended suffix mismatch")
  er=np.asarray(c["refinement_node_parent_edge_rows"]); ew=np.asarray(c["refinement_node_parent_edge_weights"])
  if er.shape!=(len(xc),2) or ew.shape!=(len(xc),2): raise ValueError("edge ancestry shape mismatch")
  if "refinement_node_kind" not in c or not np.all(np.asarray(c["refinement_node_kind"])[rows]==1): raise ValueError("suffix contains a non-midpoint node")
  rh=hashlib.sha256(np.ascontiguousarray(rows,dtype="<i8").view(np.uint8).tobytes()).hexdigest()
  if ref.get("edge_midpoint_rows_sha256")!=rh: raise ValueError("midpoint row hash mismatch")
  pe=np.sort(er[rows].astype(np.int64),axis=1)
  if np.any(pe[:,0]<0) or np.any(pe[:,1]>=n) or not np.all(ew[rows]==.5): raise ValueError("invalid midpoint edge ancestry")
  mid=((xp[pe[:,0]].astype(np.float64)+xp[pe[:,1]].astype(np.float64))*.5).astype(np.float32)
  if not m.same(mid,xc[rows]): raise ValueError("midpoint coordinates differ from F32 affine edge averages")
  if "source_prolongation_indptr" in c and "source_prolongation_indptr" in p:
   cp,cc,cv=(np.asarray(c[k]) for k in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights"))
   pp,pc,pv=(np.asarray(p[k]) for k in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights"))
   if not m.same(cp[:n+1],pp) or not m.same(cc[:pp[-1]],pc) or not m.same(cv[:pp[-1]],pv): raise ValueError("Q parent prefix changed")
   for row,(u,v) in zip(rows,pe):
    d={}
    for q in (int(u),int(v)):
     for col,val in zip(pc[pp[q]:pp[q+1]],pv[pp[q]:pp[q+1]]): d[int(col)]=d.get(int(col),0.)+.5*float(val)
    it=sorted(d.items())
    if not np.array_equal(cc[cp[row]:cp[row+1]],np.asarray([x for x,_ in it],dtype=cc.dtype)) or not np.allclose(cv[cp[row]:cp[row+1]],[x for _,x in it],rtol=0,atol=2e-15): raise ValueError("Q midpoint row is not parent-edge average")
 original=m.select_patch; picked={}
 def select(faces,report,cap,point_count=None):
  sel=original(faces,report,cap,point_count); saved=ref.get("selected_edge_subset_policy",{}).get("selected_edges")
  old=[list(edge(x)) for x in sel["selected_edges"]]
  if saved is None or sorted(map(tuple,old))!=sorted(tuple(map(int,x)) for x in saved): raise ValueError("failed patch differs from deterministic selector")
  keep,ledger=select_remaining_edges(old,rows,pe,n,bad)
  sel=dict(sel);sel["selected_edges"]=keep;sel["selected_edge_count"]=len(keep);picked.update(keep=keep,ledger=ledger);return sel
 m.select_patch=select
 try:r=m.prepare(state_path,check_path,outdir,max_edges)
 finally:m.select_patch=original
 rp=Path(r["output_report_path"]); report=json.loads(rp.read_text())
 report.update({"adapter":"selective_patch_refinement621_v2","v1_source_path":str(V1),"v1_source_sha256":V1_SHA,
  "cone_report_path":str(cone_path),"cone_report_sha256":sha(cone_path),"failed_refinement_report_path":str(failed_report_path),
  "failed_refinement_report_sha256":sha(failed_report_path),"failed_refinement_candidate_path":str(fc),"failed_refinement_candidate_sha256":fh,
  "infeasible_midpoint_rows":bad,"excluded_parent_edges":picked["ledger"],"excluded_parent_edge_count":len(picked["ledger"]),
  "remaining_selected_edge_count":len(picked["keep"]),"exclusion_is_construction_only":True,"native_admission":False,
  "accepted_physical_steps":0,"requires_fresh_130_159_545_checks":True})
 rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n");r.update(report);return r

def main(argv=None):
 p=argparse.ArgumentParser()
 for x in ("state","construction-report","cone-feasibility-report","failed-state","failed-refinement-report","out"):p.add_argument("--"+x,type=Path,required=True)
 p.add_argument("--max-edges",type=int,default=256);a=p.parse_args(argv)
 r=prepare(a.state,a.construction_report,a.cone_feasibility_report,a.failed_state,a.failed_refinement_report,a.out,a.max_edges)
 print(json.dumps({"status":r.get("status"),"output_candidate_sha256":r.get("output_candidate_sha256"),"excluded_parent_edge_count":r.get("excluded_parent_edge_count"),"native_admission":False},sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
