"""Consolidate fresh arithmetic with already completed, hash-matched exact owners."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D");C=D/"contained-clearance-545/run_candidate_checks545.py"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
 a=argparse.ArgumentParser()
 for k in ["candidate","baseline-state","point-summary","surface-report","out"]:a.add_argument("--"+k,required=True,type=Path)
 a=a.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 s=importlib.util.spec_from_file_location("checks545_for546",C);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 paths=[a.candidate,a.baseline_state,a.point_summary,a.surface_report,a.point_summary.parent/"surface-self-pairs.npy",C,Path(__file__)]
 pins={str(p):sha(p)for p in paths};base=read(a.baseline_state)
 argv=[sys.executable,str(C),"--candidate",str(a.candidate),"--baseline-state",str(a.baseline_state),"--output-dir",str(a.out/"arithmetic"),"--arithmetic-only"]
 with(a.out/"arithmetic.log").open("x")as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 report=read(a.out/"arithmetic/report.json")
 assert r.returncode==0 and report["status"]=="ARITHMETIC_PASS",report
 h=sha(a.candidate);p=read(a.point_summary);v=read(a.surface_report);z=np.load(a.candidate,allow_pickle=False)
 assert p["candidate_sha256"]==h and p["complete_surface_audit"]is True and p["exact_PLC_surface_pairs"]==0
 assert p["points"]==len(z["points_world_m"]) and p["faces"]==len(z["source_O_faces"])
 assert len(np.load(a.point_summary.parent/"surface-self-pairs.npy"))==0
 n,counts=m.validate_banks(v,h);bn,bc=m.validate_banks(read(base["surface_report"]),base["candidate_sha256"])
 assert p["inside_node_bank_pairs"]==0,"zero containment must be preserved"
 assert n<bn,"not a strict surface reduction"
 assert all(sha(k)==v for k,v in pins.items())
 report.update(status="COMPLETE_EXACT_CONSTRUCTION_CHECKS",self_pairs=0,contained_pairs=0,surface_pairs=n,surface_delta=n-bn,contained_delta=0,complete_banks=185,construction_progress=True,exact_audits_reused_without_reexecution=True,exact_reuse_pins=pins,per_bank_deltas={k:counts[k]-bc[k]for k in counts if counts[k]!=bc[k]},arithmetic_argv=argv,native_admission=False,accepted_physical_steps=0)
 write(a.out/"report.json",report)
 st=dict(candidate=str(a.candidate),point_summary=str(a.point_summary),surface_report=str(a.surface_report))
 st.update({k+"_sha256":sha(v)for k,v in list(st.items())});st.update(native_admission=False,accepted_physical_steps=0)
 write(a.out/"state.json",st)
 print(json.dumps(dict(surface_pairs=n,contained_pairs=0,self_pairs=0,state=str(a.out/"state.json"),candidate_sha256=h)))
if __name__=="__main__":main()
