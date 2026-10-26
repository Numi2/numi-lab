from pathlib import Path
import json,hashlib,subprocess,time,os,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R490=D/"D/D/component-lp-decomposition-490/run_matched_component_fit_490.py";R491=D/"D/refresh-491/run_refresh491.py";CHECK=D/"D/checked-proposal-459/run_checked_proposal_459.py"
assert sha(R490)=="7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"
assert sha(CHECK)=="f1c0c9828757b2760186da40e12beba0bde9ac5185d9b1fe8521eeb16ebe5072"
assert sha(R491)==sys.argv[1]
ROOT=D/"D/refreshed-connected-continuation-492/run-001";ROOT.mkdir(parents=True,exist_ok=False)
pins={str(p):sha(p)for p in [R490,R491,CHECK,Path(__file__)]}
current=D/"D/combined-backtrack-487/run-001/backtrack-373/selected-result.json"
previous=D/"D/outward-plane-backtrack-461/run-001/backtrack-373/selected-result.json"
prep=D/"D/connected-continuation-prep476/run-001";guard=D/"D/fixed-bone-guard-refresh476/run-001"
failed=[D/"D/component-backtrack-479/run-001/backtrack-373",D/"D/soft-target-backtrack-482/run-002/backtrack-373",current.parent]
started=time.monotonic();history=[]
def dump(p,v):p.write_text(json.dumps(v,indent=2)+"\n")
def unchanged():
 for p,h in pins.items():
  if sha(p)!=h:raise RuntimeError("source changed: "+p)
def run(label,a,out,slot=False,env=None):
 unchanged();out.mkdir(parents=True,exist_ok=True);dump(out/(label+"-argv.json"),a)
 t=time.monotonic()
 with(out/(label+".log")).open("x")as f:
  r=subprocess.run([a[0],"/tmp/numi-human-geometry-slot.py",*a]if slot else a,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,**(env or {})})
 unchanged();v=dict(returncode=r.returncode,seconds=time.monotonic()-t,source_pins_unchanged=True,native_admission=False,accepted_physical_steps=0);dump(out/(label+"-execution.json"),v)
 if r.returncode:raise RuntimeError(label+" failed")
 return v
try:
 for iteration in range(1,5):
  if time.monotonic()-started>1200:break
  out=ROOT/f"iteration-{iteration:03}";out.mkdir()
  s=json.loads(current.read_text());cur=Path(s["candidate"]);audit=Path(s["surface_report"])
  refresh=out/"refresh"
  a=[PY,str(R491),"--current-selected",str(current),"--previous-selected",str(previous),"--prior-prep-dir",str(prep),"--prior-guard-dir",str(guard),"--outroot",str(refresh),"--execute"]
  for p in failed:a.extend(["--failed-backtrack-dir",str(p)])
  run("refresh",a,out)
  plan=json.loads((refresh/"plan.json").read_text());prepnext=Path(plan["outputs"]["connected_prep"]);guardnext=Path(plan["outputs"]["fixed_bone_guards"]);depth=Path(plan["outputs"]["point_depth"])
  pr=json.loads((prepnext/"preparation-report.json").read_text())
  cfg=json.loads((D/"D/D/component-lp-decomposition-490/matched_inputs_476.json").read_text())
  paths=dict(candidate=cur,audit159=audit,face_target_npz=Path(pr["335"]["npz_path"]),point_target_npz=Path(pr["390"]["npz_path"]))
  cfg["inputs"]={k:dict(path=str(p),sha256=sha(p))for k,p in paths.items()};config=out/"matched-inputs.json";dump(config,cfg)
  fit=out/"fit"
  a=json.loads((D/"D/combined-fit-486/run-001-launch/argv.json").read_text());a[1]=str(R490)
  updates={"--base-candidate":cur,"--raw-trial":cur,"--area-reference-npz":cur,"--self-pairs":pr["self_seeds"]["path"],"--point-target-npz":paths["point_target_npz"],"--point-target-report":pr["390"]["report_path"],"--face-target-npz":paths["face_target_npz"],"--face-target-report":pr["335"]["report_path"],"--point-depth-rows-npz":depth/"point-depth-rows.npz","--point-depth-rows-report":depth/"365-compatible-report.json","--fixed-boundary-npz":guardnext/"introduced-skin-bone-barriers.npz","--fixed-boundary-report":guardnext/"report.json","--output":fit}
  for k,v in updates.items():a[a.index(k)+1]=str(v)
  a.extend(["--matched-inputs-json-490",str(config),"--audit159-490",str(audit)])
  fitreceipt=run("fit",a,out,True)
  fr=json.loads((fit/"report.json").read_text())
  if fr.get("status")!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
   history.append(dict(iteration=iteration,status="NO_PROPOSAL",fit=fitreceipt));break
  check=out/"check"
  run("check",[PY,str(CHECK),str(current),str(fit),str(prepnext),str(guardnext),str(check)],out)
  result=check/"backtrack-373/selected-result.json"
  if not result.is_file():
   history.append(dict(iteration=iteration,status="NO_CHECKED_PROGRESS"));break
  selected=json.loads(result.read_text());merit=selected["joint_merit"]
  if not merit["construction_progress"]:raise RuntimeError("selection did not pass existing joint merit")
  history.append(dict(iteration=iteration,status="CONSTRUCTION_PROGRESS_ONLY",selected_result=str(result),candidate_sha256=selected["candidate_sha256"],joint_merit=merit))
  dump(ROOT/"history.json",history)
  previous,current=current,result;prep,guard=prepnext,guardnext;failed.append(result.parent)
  if merit["after_total_pairs"]==0:break
 unchanged()
 dump(ROOT/"execution.json",dict(status="BOUNDED_ROUND_COMPLETE",seconds=time.monotonic()-started,history=history,current_selected=str(current),source_pins=pins,native_admission=False,accepted_physical_steps=0))
except BaseException as e:
 dump(ROOT/"execution.json",dict(status="FAILED_STOPPED",error=repr(e),seconds=time.monotonic()-started,history=history,current_selected=str(current),source_pins=pins,native_admission=False,accepted_physical_steps=0));raise
