"""Bounded replay of one saved field from a checked current state; original owners retain all gates."""
from pathlib import Path
import argparse,importlib.util,json,sys,shutil
N=Path(__file__).resolve().parent.parent
def main():
 ap=argparse.ArgumentParser()
 for name in ("state","target","out"):ap.add_argument("--"+name,type=Path,required=True)
 a=ap.parse_args();hp=N/"continue_surface_round546.py";sp=importlib.util.spec_from_file_location("stages615",hp);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 s=m.read_json(a.state)
 for key in ("candidate","point_summary","surface_report"):
  if m.sha(s[key])!=s[key+"_sha256"]:raise ValueError("state pin "+key)
 h=N/"matched-connected-fit-583/run_matched_fit583_v2.py";sp=importlib.util.spec_from_file_location("payload615",h);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
 v._validate_target_payload(Path(s["candidate"]),a.target,allow_fit_receipt_updates=True)
 if shutil.disk_usage(N).free<195*1024**2:raise RuntimeError("insufficient scratch reserve")
 a.out.mkdir(parents=True,exist_ok=False)
 owner=N/"local-witness-retention-582/retain_local_witnesses582.py"
 m.run_stage([sys.executable,str(owner),"--state",str(a.state),"--target",str(a.target),"--out",str(a.out/"retention"),"--trust-mm","4","--max-edge-gradient",".5"],a.out/"retention-stage",slot=True,source_paths=(Path(__file__),hp,h,owner,a.state,a.target,Path(s["candidate"])),timeout=600)
 check=N.parent/"contained-clearance-545/run_candidate_checks545.py";candidate=a.out/"retention/candidate.npz";out=a.out/"full-check"
 m.run_stage([sys.executable,str(check),"--candidate",str(candidate),"--baseline-state",str(a.state),"--trust-mm","4","--output-dir",str(out)],a.out/"check-stage",slot=False,source_paths=(check,candidate,a.state),timeout=600)
 r=m.read_json(out/"report.json")
 result={k:r.get(k)for k in ("status","surface_pairs","contained_pairs","self_pairs","complete_banks","arithmetic_pass","construction_progress")}
 result.update(state=str(a.state),target=str(a.target),native_admission=False,accepted_physical_steps=0)
 if r.get("arithmetic_pass") and r.get("construction_progress") and r.get("contained_pairs")==0 and r.get("self_pairs")==0 and r.get("surface_delta",0)<0:
  f=N/"finalize_existing_checks546.py"
  m.run_stage([sys.executable,str(f),"--candidate",str(candidate),"--baseline-state",str(a.state),"--point-summary",str(out/"audit-130/summary.json"),"--surface-report",str(out/"audit-159/report.json"),"--out",str(a.out/"final"),"--trust-mm","4"],a.out/"finalize-stage",slot=False,source_paths=(f,candidate,a.state,out/"audit-130/summary.json",out/"audit-159/report.json"),timeout=60)
  result["passing_state"]=str(a.out/"final/state.json")
 m.write_json(a.out/"result.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
