"""Continue matched construction fits with strict improvement and witness rollback.
No native state is promoted. Stop on no progress; all heavy work uses existing owners.
"""
from pathlib import Path
import argparse,importlib.util,json,sys,shutil,time
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("stages581",N/"continue_surface_round546.py");m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
def metrics(path):
 s=m.read_json(path)
 for k in("candidate","point_summary","surface_report"):
  if m.sha(s[k])!=s[k+"_sha256"]:raise ValueError("strict-state pin mismatch "+k)
 h=m.read_json(s["point_summary"]);r=m.read_json(s["surface_report"])
 if h.get("complete_surface_audit")is not True or h.get("candidate_sha256")!=s["candidate_sha256"]or h.get("inside_node_bank_pairs")!=0 or h.get("exact_PLC_surface_pairs")!=0:raise ValueError("strict complete zero point/self state required")
 if r.get("candidate_sha256")!=s["candidate_sha256"]or r.get("complete")is not True or r.get("bank_count")!=185 or r.get("expected_bank_count")!=185:raise ValueError("strict full185 bank state required")
 if s.get("native_admission")is not False or s.get("accepted_physical_steps")!=0:raise ValueError("construction-only state required")
 return sum(int((b.get("result")or{}).get("count",0))for b in r["banks"])
def main():
 p=argparse.ArgumentParser();p.add_argument("--state",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--rounds",type=int,default=3);a=p.parse_args()
 if not 1<=a.rounds<=6:raise ValueError("bounded 1..6 rounds required")
 O=a.out;O.mkdir(parents=True,exist_ok=False);S=a.state;before=metrics(S);history=[]
 for step in range(1,a.rounds+1):
  if before==0:break
  if shutil.disk_usage(O).free<512*1024**2:raise RuntimeError("512MiB output reserve reached; preserve and recover verified redundant outputs before resuming")
  T=O/f"round-{step:03d}";owner=N/"matched-connected-fit-575/run_matched_fit575.py"
  m.run_stage([sys.executable,owner,"--state",S,"--out",T],O/f"round-{step:03d}-stage",slot=False,source_paths=(Path(__file__),owner,S),timeout=1500)
  candidates=[]
  for path in(T/"local-retention/final/state.json",T/"retained-regions/final/state.json"):
   if path.is_file():candidates.append((metrics(path),path))
  best_count,best_state=min(candidates,key=lambda z:z[0])if candidates else(before,S)
  # Restore exactly new contained witnesses only when the global field clears more surfaces.
  P=T/"local-retention";check=P/"full-check/report.json"
  if check.is_file():
   r=m.read_json(check)
   if r.get("status")=="COMPLETE_EXACT_CONSTRUCTION_CHECKS"and r.get("self_pairs")==0 and 0<int(r.get("contained_pairs",0))<=32 and int(r["surface_pairs"])<best_count:
    target=T/"witness-restored-target";owner=N/"witness-restore-578/restore_witness_points578_v2.py"
    m.run_stage([sys.executable,owner,"--state",best_state,"--target",P/"candidate.npz","--point-summary",P/"full-check/audit-130/summary.json","--out",target],T/"witness-target-stage",slot=False,source_paths=(owner,best_state,P/"candidate.npz",P/"full-check/audit-130/summary.json"),timeout=30)
    owner=N/"local-witness-retention-567/run_target567.py";out=T/"witness-restored-check"
    m.run_stage([sys.executable,owner,"--state",best_state,"--target",target/"candidate.npz","--out",out],T/"witness-orchestrator-stage",slot=False,source_paths=(owner,best_state,target/"candidate.npz"),timeout=900)
    if(out/"final/state.json").is_file():
     count=metrics(out/"final/state.json")
     if count<best_count:best_count,best_state=count,out/"final/state.json"
  row={"round":step,"input_state":str(S),"input_state_sha256":m.sha(S),"before_surface_pairs":before,"best_state":str(best_state),"best_state_sha256":m.sha(best_state),"after_surface_pairs":best_count,"strict_progress":best_count<before,"contained_pairs":0,"self_pairs":0,"native_admission":False,"accepted_physical_steps":0}
  history.append(row);m.write_json(O/"progress.json",{"status":"CONTINUING_CONSTRUCTION"if best_count<before else"STOPPED_NO_STRICT_PROGRESS","rounds":history,"best_state":str(best_state),"surface_pairs":best_count,"native_admission":False,"accepted_physical_steps":0})
  print(json.dumps(row,sort_keys=True),flush=True)
  if best_count>=before:break
  S,before=best_state,best_count
 m.write_json(O/"checkpoint.json",{"state":str(S),"state_sha256":m.sha(S),"surface_pairs":before,"complete_surface_clearance":before==0,"native_admission":False,"accepted_physical_steps":0})
if __name__=="__main__":main()

