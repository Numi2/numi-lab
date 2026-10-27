"""Bounded connected-fit/topology continuation using existing checked owners.
Offline construction only. Every adopted state has complete 130/159 checks.
"""
from pathlib import Path
import argparse,importlib.util,json,shutil,sys,time
N=Path(__file__).resolve().parent.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load("stages584",N/"continue_surface_round546.py")
state_owner=load("strict581for584",N/"continued-connected-fit-581/continue_matched581.py")
def best_state(paths, baseline):
 count=state_owner.metrics(baseline);chosen=baseline
 for p in paths:
  if not p.is_file():continue
  c=state_owner.metrics(p)
  if c<count:count,chosen=c,p
 return count,chosen
def final_states(out):
 return sorted(out.rglob("final/state.json"))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--state",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 p.add_argument("--fit-owner",type=Path,required=True);p.add_argument("--rounds",type=int,default=4)
 p.add_argument("--trust-mm",type=float,default=2.);p.add_argument("--fit-edge-gradient",type=float,default=.5)
 p.add_argument("--topology-seconds",type=float,default=90.);a=p.parse_args()
 if not 1<=a.rounds<=4:raise ValueError("at most four connected rounds")
 if not 0<a.trust_mm<=4 or not 0<a.fit_edge_gradient<=.5:raise ValueError("construction proposals must remain within current 545 limits")
 if not 15<=a.topology_seconds<=160:raise ValueError("bounded topology search")
 O=a.out;O.mkdir(parents=True,exist_ok=False);S=a.state;initial=state_owner.metrics(S)
 history=[];start=time.monotonic();reason="ROUND_BUDGET"
 sources=(Path(__file__),Path(__file__).with_name("run_capped584_v2.py"),a.fit_owner,N/"connected-topology-580/run_connected_round580.py",N/"continued-connected-fit-581/continue_matched581.py",N/"continue_surface_round546.py")
 source_pins={str(z):m.sha(z)for z in sources}
 def save(state,why):
  c=state_owner.metrics(state)
  m.write_json(O/"checkpoint.json",{"state":str(state),"state_sha256":m.sha(state),"surface_pairs":c,"initial_surface_pairs":initial,"complete_surface_clearance":c==0,"stop_reason":why,"source_pins":source_pins,"seconds":time.monotonic()-start,"native_admission":False,"accepted_physical_steps":0})
 def phase(argv,stem,paths,minimum=30):
  remaining=1200-(time.monotonic()-start)
  if remaining<minimum:return False
  cap=min(remaining,900.)
  wrapped=[sys.executable,Path(__file__).with_name("run_capped584_v2.py"),"--seconds",str(cap),"--",*argv]
  try:m.run_stage(wrapped,stem,slot=False,source_paths=(*sources,*paths),timeout=cap+12)
  except RuntimeError:
   execution=m.read_json(stem.with_suffix(".execution.json"))
   if execution.get("returncode")==124:return False
   raise
  return True
 save(S,"RUNNING")
 for i in range(1,a.rounds+1):
  before=state_owner.metrics(S)
  if before==0:reason="ZERO_SURFACE_PAIRS";break
  if time.monotonic()-start>=1200:reason="TWENTY_MINUTE_METHOD_BUDGET";break
  if shutil.disk_usage(O).free<512*1024**2:reason="OUTPUT_RESERVE_REACHED";break
  R=O/f"round-{i:03d}";R.mkdir()
  fit=R/"emitted-fit"
  argv=[sys.executable,a.fit_owner,"--state",S,"--out",fit,"--trust-mm",str(a.trust_mm),"--fit-edge-gradient",str(a.fit_edge_gradient),"--retention-edge-gradient",".5","--target-policy","emitted"]
  if not phase(argv,R/"emitted-fit-orchestrator",(S,),minimum=90):reason="TWENTY_MINUTE_METHOD_BUDGET";break
  count,best=best_state(final_states(fit),S)
  save(best,"EMITTED_FIT_CHECKED")
  if count:
   replay=R/"raw-replay"
   argv=[sys.executable,a.fit_owner,"--state",S,"--out",replay,"--fit-from",fit,"--prepared-from",fit,"--trust-mm",str(a.trust_mm),"--fit-edge-gradient",str(a.fit_edge_gradient),"--retention-edge-gradient",".5","--target-policy","raw-last"]
   if not phase(argv,R/"raw-replay-orchestrator",(S,),minimum=90):
    S=best;reason="TWENTY_MINUTE_METHOD_BUDGET";break
   count,best=best_state(final_states(replay),best)
   save(best,"RAW_REPLAY_CHECKED")
  topology=R/"connected-topology"
  if count:
   owner=N/"connected-topology-580/run_connected_round580.py"
   argv=[sys.executable,owner,"--state",best,"--out",topology,"--helper",N/"same-count-star-payload-568/same_count_patch_payload568.py","--seconds",str(a.topology_seconds)]
   if not phase(argv,R/"topology-orchestrator",(best,),minimum=a.topology_seconds+60):
    S=best;reason="TWENTY_MINUTE_METHOD_BUDGET";break
   count,best=best_state(final_states(topology),best)
  row={"round":i,"input_state":str(S),"input_state_sha256":m.sha(S),"before_surface_pairs":before,"best_state":str(best),"best_state_sha256":m.sha(best),"after_surface_pairs":count,"strict_progress":count<before,"contained_pairs":0,"self_pairs":0,"native_admission":False,"accepted_physical_steps":0}
  history.append(row)
  m.write_json(O/"progress.json",{"rounds":history,"best_state":str(best),"surface_pairs":count,"native_admission":False,"accepted_physical_steps":0})
  print(json.dumps(row,sort_keys=True),flush=True)
  if count>=before:reason="NO_STRICT_PROGRESS_CHANGE_METHOD";break
  S=best
  m.write_json(O/"checkpoint.json",{"state":str(S),"state_sha256":m.sha(S),"surface_pairs":count,"complete_surface_clearance":count==0,"native_admission":False,"accepted_physical_steps":0})
 for path,h in source_pins.items():
  if m.sha(path)!=h:raise ValueError("owner changed during continuation")
 count=state_owner.metrics(S)
 m.write_json(O/"checkpoint.json",{"state":str(S),"state_sha256":m.sha(S),"surface_pairs":count,"initial_surface_pairs":initial,"complete_surface_clearance":count==0,"stop_reason":reason,"source_pins":source_pins,"seconds":time.monotonic()-start,"native_admission":False,"accepted_physical_steps":0})
if __name__=="__main__":main()
