"""Thin in-memory launcher; pass the frozen 603 fit arguments unchanged."""
from pathlib import Path
import hashlib, json, sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
FROZEN=D/"D/surface-clearance-546/component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
EXPECTED="38c3dfa9ddd961250111c87f56d6184075bde43be07c46e09d239f057b06d692"
NEEDLE="                result = install(owner, objective, crossing)\n"
INJECT=NEEDLE+"                from box_aware_sat637 import install_from_cli\n                install_from_cli(owner, objective, crossing)\n"
DETAIL="458 crossed-pair plane normals/targets and their nonpositive guards are rerouted to finite-pair SAT axes; 408 per-node component boxes are replaced by 636-v2 asymmetric boxes contained within the same trust/source balls. All other 408/603 constraints and final exact checks remain active."
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def patch_source(source):
 if source.count(NEEDLE)!=1:raise ValueError("603 install hook is not unique")
 return source.replace(NEEDLE,INJECT,1)
def arg(flag):
 if sys.argv.count(flag)!=1:raise ValueError(f"expected one {flag}")
 i=sys.argv.index(flag)
 if i+1>=len(sys.argv):raise ValueError(f"missing {flag} value")
 return sys.argv[i+1]
def write_launch_receipt(output_path,launch):
 out=Path(output_path)
 if out.exists():raise FileExistsError("fit output must remain absent for frozen 408 startup")
 path=out.with_name(out.name+"-box-aware-sat-637.launch.json")
 path.write_text(json.dumps(launch,indent=2,sort_keys=True)+"\n")
 return path
def _correct_claims(obj,route):
 if isinstance(obj,dict):
  if "objective_contract" in obj:
   obj["objective_contract"]="458 positive corner rows remain soft; 637 changes per-pair finite SAT target planes and their nonpositive-corner guards; 636-v2 replaces the 408 component box inside unchanged trust/source balls."
  if obj.get("lp_equations_and_gate_sources_unchanged") is True:obj["lp_equations_and_gate_sources_unchanged"]=False
  for value in obj.values():_correct_claims(value,route)
 elif isinstance(obj,list):
  for value in obj:_correct_claims(value,route)
 return obj
def _annotate_raw(rawdir,route):
 summary_path=Path(rawdir)/"report.json"
 if not summary_path.is_file():return None
 summary=json.loads(summary_path.read_text())
 for rec in summary.get("solver_returns",[]):
  p=Path(rec.get("proposal_report_path",""))
  if p.is_file():
   r=_correct_claims(json.loads(p.read_text()),route)
   r["pair_sat_box_routing_637"]=route
   p.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
   rec["proposal_report_sha256"]=sha(p)
 summary=_correct_claims(summary,route)
 summary["pair_sat_box_routing_637"]=route
 summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
 return summary_path
def _annotate(state,out):
 import box_aware_sat637 as h
 sat_report,selected_npz=h.write_result(state,out)
 route={"report_path":str(sat_report),"report_sha256":sha(sat_report),"selected_rows_path":str(selected_npz),
  "selected_rows_sha256":sha(selected_npz),"soft_objective_preserved":True,"positive_rows_hardened":False,
  "directional_bounds636_replaced_408_cube":True,"lp_equations_and_gate_sources_unchanged":False,
  "exact_changes":DETAIL,"native_admission":False,"accepted_physical_steps":0}
 fit=Path(out)/"report.json"
 rawdir=Path(out)/"raw-joint-capture-563"
 raw_summary=_annotate_raw(rawdir,route)
 if fit.is_file():
  report=_correct_claims(json.loads(fit.read_text()),route)
  report["pair_sat_box_routing_637"]=route
  partial=report.get("full_coordinate_partial_timeout564")
  if isinstance(partial,dict):
   partial["box_aware_sat_637_report_sha256"]=sha(sat_report)
   partial["box_aware_sat_637_exact_changes"]=DETAIL
   if raw_summary is not None:partial["raw_joint_capture_summary_sha256"]=sha(raw_summary)
  fit.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
 if sha(FROZEN)!=EXPECTED:raise RuntimeError("frozen 603 runner pin mismatch")
 for p in (HERE,FROZEN.parent):
  if str(p) not in sys.path:sys.path.insert(0,str(p))
 patched=patch_source(FROZEN.read_text());out=Path(arg("--output")).resolve()
 launch={"schema":"numi.human.box_aware_sat_launch_637.v1","frozen_603_path":str(FROZEN),
  "frozen_603_sha256":EXPECTED,"adapter_path":str(Path(__file__).resolve()),"adapter_sha256":sha(__file__),
  "helper_path":str(HERE/"box_aware_sat637.py"),"helper_sha256":sha(HERE/"box_aware_sat637.py"),
  "in_memory_patch_sha256":hashlib.sha256(patched.encode()).hexdigest(),"insertion_count":1,
  "argv":sys.argv[1:],"native_admission":False}
 write_launch_receipt(out,launch)
 ns={"__name__":"_frozen603_runtime_637","__file__":str(FROZEN)}
 try:
  exec(compile(patched,str(FROZEN),"exec"),ns)
  ns["main"]()
 finally:
  import box_aware_sat637 as h
  if h.INSTALLS:
   try:_annotate(h.INSTALLS[-1],out)
   except Exception as exc:
    path=out.with_name(out.name+"-box-aware-sat-637-annotation-failure.json")
    path.write_text(json.dumps({"error_type":type(exc).__name__,"error":str(exc),"native_admission":False},indent=2)+"\n")
if __name__=="__main__":main()
