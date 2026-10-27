"""Prepare independent current-topology guards and face targets ahead of point exits."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
def main():
 p=N/"continue_surface_round546.py";s=importlib.util.spec_from_file_location("stages_geometry604",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 out=N/"refined-fit-604/preparation-603-001";out.mkdir(parents=True,exist_ok=False)
 state=N/"connected-refinement-603/baseline-599-affine-filtered-002/baseline-check-001/baseline-state.json"
 st=m.read_json(state);base=Path(st["candidate"]);summary=Path(st["point_summary"]);surface=Path(st["surface_report"]);query=summary.parent/"query/query-receipt.json"
 parent=N/"refined-fit-604/parent-context-599-001/constraint-context.json";ctx=m.read_json(parent);refresh=N/"refresh_surface_constraints546.py"
 for k in ("candidate","point_summary","surface_report"):
  if m.sha(st[k])!=st[k+"_sha256"]:raise ValueError("state pin "+k)
 argv=[m.PYTHON,refresh,"--candidate",base,"--audit130",summary,"--audit159",surface,
 "--query-dir",query.parent,"--ancestry-parent-candidate",ctx["candidate"],
 "--prior-candidate",ctx["candidate"],"--prior-surface-report",ctx["surface_report"],
 "--prior-guard-candidate",ctx["candidate"],"--prior-guard-npz",ctx["guard_npz"],
 "--prior-guard-report",ctx["guard_report"],"--prior-seed-pairs",ctx["seed_pairs_path"],
 "--prior-seed-report",ctx["seed_report"],"--failed-seed-candidate",base,
 "--failed-seed-audit130",summary,"--failed-seed-pairs",summary.parent/"surface-self-pairs.npy",
 "--output-dir",out/"constraints","--point-continuation"]
 pins=(Path(__file__),refresh,state,parent,base,summary,surface,query,Path(ctx["candidate"]),Path(ctx["surface_report"]),Path(ctx["guard_npz"]),Path(ctx["guard_report"]),Path(ctx["seed_pairs_path"]),Path(ctx["seed_report"]))
 m.run_stage(argv,out/"refresh-stage",slot=True,source_paths=pins,timeout=600)
 target=out/"joint-face-targets";builder=N/"source-current-targets-597/build_joint_outward_targets597_v4.py"
 m.run_stage([m.PYTHON,builder,"--candidate",base,"--surface-report",surface,"--query-receipt",query,"--output-dir",target,"--clearance-m",".00005","--direction-mode","source-outward"],out/"joint-faces-stage",slot=True,source_paths=(builder,builder.with_name("angular_joint_direction597.py"),state,base,surface,query),timeout=600)
 verifier=Path(__file__).with_name("verify_joint_targets604.py")
 recovery=target/"recovery-verification.json"
 m.run_stage([m.PYTHON,verifier,"--folder",target,"--state",state,"--output",recovery],out/"face-verification-stage",slot=False,source_paths=(verifier,target/"targets.npz",target/"report.json",target/"target-records.jsonl"),timeout=60)
 compat=N/"connected-fit-598/target-compatibility-001/compatibility_report598_v2.py";cr=out/"face-fit-compatibility.json"
 m.run_stage([m.PYTHON,compat,"--report",target/"report.json","--npz",target/"targets.npz","--records",target/"target-records.jsonl","--recovery",recovery,"--out",cr],out/"face-compat-stage",slot=False,source_paths=(compat,recovery,target/"targets.npz",target/"report.json",target/"target-records.jsonl"),timeout=60)
 m.write_json(out/"prepared.json",{"status":"CURRENT_TOPOLOGY_GUARDS_AND_FACE_TARGETS_PREPARED","state":str(state),"state_sha256":m.sha(state),"parent_context":str(parent),"parent_context_sha256":m.sha(parent),"refresh_report":str(out/"constraints/refresh-report.json"),"refresh_report_sha256":m.sha(out/"constraints/refresh-report.json"),"face_targets":str(target),"face_compatibility":str(cr),"face_compatibility_sha256":m.sha(cr),"native_admission":False,"accepted_physical_steps":0})
 print(json.dumps({"status":"CURRENT_TOPOLOGY_GUARDS_AND_FACE_TARGETS_PREPARED","out":str(out)}))
if __name__=="__main__":main()
