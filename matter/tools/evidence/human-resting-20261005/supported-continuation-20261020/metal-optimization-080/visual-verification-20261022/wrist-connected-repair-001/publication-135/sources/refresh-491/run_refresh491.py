#!/usr/bin/env python3
"""Prepare or sequentially run an input refresh from selected 130/159 receipts."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
TEMPLATES=D/"refresh-476"
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
DEPTH=D/"point-depth-refresh-413/refresh_point_depth_rows.py"
GUARDS=D/"fixed-bone-guard-refresh-413/refresh_fixed_bone_guards.py"
PREP=D/"prepare_connected_continuation_inputs_417.py"

EXTRA_SOURCES=[
 Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/source-outward-proposal-300/build_outward_nearest_union_targets_335.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/D/coherent-point-targets-390/build_coherent_point_targets_390.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/publication-113/sources/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"),
 Path("/Users/numi/Work/numi-human-connected-surface-069/publication-101/evidence/204/source/fit/self_separation_026.py"),
]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_json(p):
 return json.loads(Path(p).read_text())
def selected(path):
 p=Path(path).resolve(); x=read_json(p)
 def ref(key):
  q=Path(x[key]); return q if q.is_absolute() else (p.parent/q).resolve()
 candidate=ref("candidate"); point=ref("point_summary"); surface=ref("surface_report")
 ch=sha(candidate)
 if x.get("candidate_sha256")!=ch: raise ValueError(f"selected candidate hash mismatch: {p}")
 if x.get("point_summary_sha256")!=sha(point): raise ValueError(f"selected 130 summary hash mismatch: {p}")
 a=read_json(point); s=read_json(surface)
 if a.get("candidate_sha256")!=ch or a.get("complete_surface_audit") is not True: raise ValueError("selected 130 audit is incomplete or not candidate-bound")
 if s.get("candidate_sha256")!=ch or s.get("complete") is not True or int(s.get("bank_count",-1))!=185: raise ValueError("selected 159 report is incomplete or not candidate-bound")
 query=point.parent/"query"; receipt=query/"query-receipt.json"
 if not receipt.is_file(): raise FileNotFoundError(receipt)
 return {"selected_json":p,"candidate":candidate,"candidate_sha256":ch,"point_summary":point,"point_summary_sha256":sha(point),"surface_report":surface,"surface_report_sha256":sha(surface),"query_dir":query,"query_receipt":receipt,"query_receipt_sha256":sha(receipt),"audit130":a,"audit159":s}
def flag(argv,name):
 if argv.count(name)!=1: raise ValueError(f"template must contain {name} exactly once")
 return argv.index(name)
def set_flags(argv,values):
 out=list(argv)
 for name,value in values.items():
  i=flag(out,name)
  if i+1>=len(out): raise ValueError(f"missing template value after {name}")
  out[i+1]=str(value)
 return out
def owner_command(template,owner):
 a=read_json(template); i=a.index(str(owner))
 # Drop the old external slot launcher; owner scripts manage their own children.
 return [a[0],str(owner),*a[i+1:]]
def load_prior(prior_prep,prior_guard,previous):
 prep=Path(prior_prep).resolve(); guard=Path(prior_guard).resolve()
 pr=read_json(prep/"preparation-report.json")
 seeds=pr["self_seeds"]; seed_report=Path(seeds["report_path"]); seed_pairs=Path(seeds["path"])
 for p,k in ((seed_report,"report_sha256"),(seed_pairs,"sha256")):
  if sha(p)!=seeds[k]: raise ValueError(f"prior prep seed pin mismatch: {p}")
 gr=read_json(guard/"report.json")
 if gr.get("base_candidate_sha256")!=previous["candidate_sha256"]: raise ValueError("prior guard report is not bound to the previous-base selected candidate")
 if Path(gr.get("base_candidate_path","")).resolve()!=previous["candidate"].resolve(): raise ValueError("prior guard path binding differs from previous base candidate")
 gnpz=Path(gr["output_npz"]); gnpz=gnpz if gnpz.is_absolute() else (guard/gnpz).resolve()
 if sha(gnpz)!=gr.get("output_npz_sha256"): raise ValueError("prior guard NPZ hash mismatch")
 return {"prep_report":prep/"preparation-report.json","prep_report_sha256":sha(prep/"preparation-report.json"),"seed_report":seed_report,"seed_report_sha256":sha(seed_report),"seed_pairs":seed_pairs,"seed_pairs_sha256":sha(seed_pairs),"guard_report":guard/"report.json","guard_report_sha256":sha(guard/"report.json"),"guard_npz":gnpz,"guard_npz_sha256":sha(gnpz)}
def file_pins(cur,prev,pri,failed_path,failed):
 paths=[Path(__file__),DEPTH,GUARDS,PREP,*EXTRA_SOURCES,*sorted(TEMPLATES.glob("*-argv.json")),failed_path]
 for item in (cur,prev):
  paths.extend(item[k] for k in ("selected_json","candidate","point_summary","surface_report","query_receipt"))
 paths.extend(Path(pri[k]) for k in ("prep_report","seed_report","seed_pairs","guard_report","guard_npz"))
 for row in failed["trials"]:
  paths.extend(Path(row[k]) for k in ("candidate_path","audit130_path","pairs_path"))
 return {str(Path(x).resolve()):sha(x) for x in paths}
def recheck_pins(pins):
 for raw,expected in pins.items():
  path=Path(raw)
  if not path.is_file() or sha(path)!=expected: raise RuntimeError(f"491 pinned input/source changed during run: {path}")
def failed_trials(dirs,face_count):
 trials=[];excluded=[]
 seen=set()
 for raw in dirs:
  root=Path(raw).resolve()
  if not root.is_dir(): raise FileNotFoundError(root)
  for audit in sorted(root.rglob("audit-130/summary.json")):
   d=read_json(audit)
   n=int(d.get("exact_PLC_surface_pairs",-1))
   if d.get("complete_surface_audit") is not True:
    excluded.append({"audit130_path":str(audit),"reason":"incomplete audit"}); continue
   if n<=0:
    excluded.append({"audit130_path":str(audit),"reason":"zero exact self pairs"}); continue
   candidate=audit.parent.parent/"candidate.npz"; pairs=audit.parent/"surface-self-pairs.npy"
   if not candidate.is_file() or not pairs.is_file(): raise ValueError(f"audited positive trial missing candidate/pair payload: {audit}")
   ch=sha(candidate); ah=sha(audit); ph=sha(pairs)
   if d.get("candidate_sha256")!=ch: raise ValueError(f"failed audit candidate binding mismatch: {audit}")
   arr=np.load(pairs,allow_pickle=False)
   if arr.ndim!=2 or arr.shape!=(n,2) or np.any(arr<0) or np.any(arr>=face_count) or np.any(arr[:,0]>=arr[:,1]):
    raise ValueError(f"failed trial exact pair payload/count invalid: {pairs}")
   if len(np.unique(arr,axis=0))!=n: raise ValueError(f"failed trial pair payload has duplicates: {pairs}")
   key=(ch,ah,ph)
   if key in seen: continue
   seen.add(key)
   trials.append({"candidate_path":str(candidate),"candidate_sha256":ch,"audit130_path":str(audit),"audit130_sha256":ah,"pairs_path":str(pairs),"pairs_sha256":ph,"exact_self_pair_count":n,"source_backtrack_dir":str(root)})
 return {"trials":trials,"excluded":excluded}
def build_plan(current_selected,previous_selected,prior_prep,prior_guard,outroot,failed_dirs,execute=False):
 cur=selected(current_selected); prev=selected(previous_selected); pri=load_prior(prior_prep,prior_guard,prev)
 if cur["candidate"].resolve()==prev["candidate"].resolve(): raise ValueError("current and previous selected candidates must differ")
 with np.load(cur["candidate"],allow_pickle=False) as z: face_count=len(z["source_O_faces"])
 failed=failed_trials(failed_dirs,face_count)
 out=Path(outroot).resolve()
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True)
 depth_out=out/"point-depth-refresh491/run-001"; guard_out=out/"fixed-bone-guard-refresh491/run-001"; prep_out=out/"connected-continuation-prep491/run-001"
 failed_path=out/"failed-trials.json"; failed_path.write_text(json.dumps(failed,indent=2,sort_keys=True)+"\n")
 d=owner_command(TEMPLATES/"depth-argv.json",DEPTH)
 d=set_flags(d,{"--candidate":cur["candidate"],"--point-summary":cur["point_summary"],"--surface-report":cur["surface_report"],"--query-dir":cur["query_dir"],"--output-dir":depth_out,"--expected-candidate-sha256":cur["candidate_sha256"],"--expected-point-summary-sha256":cur["point_summary_sha256"],"--expected-surface-report-sha256":cur["surface_report_sha256"],"--expected-query-receipt-sha256":cur["query_receipt_sha256"]})
 if d[d.index("--expected-owner-sha256")+1]!=sha(D/"point-depth-barriers-357/point_depth_barriers_357.py"): raise ValueError("476 depth CLI owner hash is stale")
 g=owner_command(TEMPLATES/"guards-argv.json",GUARDS)
 g=set_flags(g,{"--previous-candidate":prev["candidate"],"--previous-surface-report":prev["surface_report"],"--prior-guard-report":pri["guard_report"],"--prior-guard-npz":pri["guard_npz"],"--candidate":cur["candidate"],"--surface-report":cur["surface_report"],"--output-dir":guard_out})
 p=owner_command(TEMPLATES/"prep-argv.json",PREP)
 p=set_flags(p,{"--candidate":cur["candidate"],"--candidate-sha256":cur["candidate_sha256"],"--audit130":cur["point_summary"],"--audit130-sha256":cur["point_summary_sha256"],"--audit159":cur["surface_report"],"--audit159-sha256":cur["surface_report_sha256"],"--prior-seed-report":pri["seed_report"],"--prior-seed-report-sha256":pri["seed_report_sha256"],"--prior-seed-pairs":pri["seed_pairs"],"--prior-seed-pairs-sha256":pri["seed_pairs_sha256"],"--failed-trials-manifest":failed_path,"--failed-trials-manifest-sha256":sha(failed_path),"--point-depth-npz":depth_out/"point-depth-rows.npz","--point-depth-report":depth_out/"365-compatible-report.json","--point-depth-npz-sha256":"PENDING_AFTER_DEPTH","--point-depth-report-sha256":"PENDING_AFTER_DEPTH","--query-dir":cur["query_dir"],"--output-root":prep_out})
 # Output hashes are unavailable until depth executes; fill them in between stages.
 commands={"depth":d,"guards":g,"prep":p}
 for name,argv in commands.items(): (out/f"{name}-argv.json").write_text(json.dumps(argv,indent=2)+"\n")
 pins={"runner":sha(Path(__file__)),"depth_owner":sha(DEPTH),"guard_owner":sha(GUARDS),"prep_owner":sha(PREP),"templates":{x.name:sha(x) for x in sorted(TEMPLATES.glob("*-argv.json"))},"current":{k:v for k,v in cur.items() if k not in ("audit130","audit159")},"previous":{"selected_json":str(prev["selected_json"]),"candidate":str(prev["candidate"]),"candidate_sha256":prev["candidate_sha256"],"point_summary":str(prev["point_summary"]),"surface_report":str(prev["surface_report"])},"prior":pri,"failed_manifest_sha256":sha(failed_path),"file_sha256":file_pins(cur,prev,pri,failed_path,failed)}
 (out/"source-pins.json").write_text(json.dumps(pins,indent=2,sort_keys=True,default=str)+"\n")
 receipt={"schema":"numi.refresh491.plan.v1","status":"PLAN_READY","execute_requested":bool(execute),"commands":{k:str(out/f"{k}-argv.json") for k in commands},"failed_manifest":str(failed_path),"failed_trials_included":len(failed["trials"]),"failed_trials_excluded":len(failed["excluded"]),"outputs":{"point_depth":str(depth_out),"fixed_bone_guards":str(guard_out),"connected_prep":str(prep_out)},"source_pins":str(out/"source-pins.json"),"role_note":"After prep, roles are read from preparation-report.json; no global path substitution."}
 (out/"plan.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 if execute:
  recheck_pins(pins["file_sha256"])
  results=[]
  for name in ("depth","guards","prep"):
   argv=json.loads((out/f"{name}-argv.json").read_text())
   if name=="prep":
    depth_report=depth_out/"365-compatible-report.json"; depth_npz=depth_out/"point-depth-rows.npz"
    argv=set_flags(argv,{"--point-depth-npz-sha256":sha(depth_npz),"--point-depth-report-sha256":sha(depth_report)})
    (out/"prep-argv.json").write_text(json.dumps(argv,indent=2)+"\n")
   started=time.monotonic()
   with (out/f"{name}.log").open("wb") as stream: rc=subprocess.run(argv,cwd=D,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
   stage={"name":name,"argv":argv,"returncode":rc,"seconds":time.monotonic()-started,"log":str(out/f"{name}.log")}
   results.append(stage); (out/f"{name}-execution.json").write_text(json.dumps(stage,indent=2)+"\n")
   recheck_pins(pins["file_sha256"])
   if rc: (out/"execution.json").write_text(json.dumps({"status":"FAILED","stages":results},indent=2)+"\n"); raise RuntimeError(f"{name} failed; stopped before next stage")
  prep_report=read_json(prep_out/"preparation-report.json")
  if prep_report.get("candidate_sha256")!=cur["candidate_sha256"]: raise RuntimeError("491 prep output is not bound to current selected candidate")
  role_paths={"self_seed_report":prep_report["self_seeds"]["report_path"],"self_seed_pairs":prep_report["self_seeds"]["path"],"surface_targets_report":prep_report["335"]["report_path"],"surface_targets_npz":prep_report["335"]["npz_path"],"point_targets_report":prep_report["390"]["report_path"],"point_targets_npz":prep_report["390"]["npz_path"]}
  roles={k:{"path":str(v),"sha256":sha(v)} for k,v in role_paths.items()}
  recheck_pins(pins["file_sha256"])
  (out/"output-roles.json").write_text(json.dumps({"schema":"numi.refresh491.output-roles.v1","from_preparation_report":str(prep_out/"preparation-report.json"),"roles":roles},indent=2,sort_keys=True)+"\n")
  (out/"execution.json").write_text(json.dumps({"status":"PASS","stages":results,"output_roles":"output-roles.json"},indent=2)+"\n")
 return receipt
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--current-selected",required=True); ap.add_argument("--previous-selected",required=True)
 ap.add_argument("--prior-prep-dir",required=True); ap.add_argument("--prior-guard-dir",required=True)
 ap.add_argument("--outroot",required=True); ap.add_argument("--failed-backtrack-dir",action="append",required=True)
 ap.add_argument("--execute",action="store_true",help="run the three refresh owners sequentially; default writes a reviewed plan only")
 a=ap.parse_args()
 print(json.dumps(build_plan(a.current_selected,a.previous_selected,a.prior_prep_dir,a.prior_guard_dir,a.outroot,a.failed_backtrack_dir,a.execute),indent=2,sort_keys=True))
if __name__=="__main__": main()
