"""One-shot preparation of the checked 603/606 connected reconstruction fit."""
from pathlib import Path
import importlib.util,json,sys,numpy as np
N=Path(__file__).resolve().parent.parent
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def main():
 m=load("stages607",N/"continue_surface_round546.py");h=load("joint595_for607",N/"joint-refined-595/joint_refined595.py")
 out=N/"refined-fit-604/matched-607-002";out.mkdir(parents=True,exist_ok=False)
 state=N/"connected-refinement-603/baseline-599-affine-filtered-002/baseline-check-001/baseline-state.json"
 st=m.read_json(state);candidate=Path(st["candidate"]);point=Path(st["point_summary"]);surface=Path(st["surface_report"])
 parent=N/"refined-fit-604/parent-context-599-001/constraint-context.json";prior=m.read_json(parent)
 G=N/"refined-fit-604/preparation-603-001";g=m.read_json(G/"prepared.json");refresh=G/"constraints/refresh-report.json";r=m.read_json(refresh)
 pt=N/"source-depth-targets-606/targets-603-filtered-001/nearest-union-exits.npz";pr=pt.with_name("nearest-union-exits.lineage-v3.json")
 dn=N/"connected-refinement-603/depth-413-002/point-depth-rows.npz";dr=dn.with_name("365-compatible-report.json")
 fn=G/"joint-face-targets/targets.npz";fr=G/"face-fit-compatibility.json"
 for k in ("candidate","point_summary","surface_report"):
  if m.sha(st[k])!=st[k+"_sha256"]:raise ValueError("state pin "+k)
 for path,key in ((refresh,"refresh_report_sha256"),(parent,"parent_context_sha256"),(fr,"face_compatibility_sha256")):
  if m.sha(path)!=g[key]:raise ValueError("geometry preparation pin "+key)
 if r["candidate_sha256"]!=m.sha(candidate) or g["state_sha256"]!=m.sha(state):raise ValueError("prepared topology differs")
 validation=h.validate_joint_refined_inputs(state=st,point_summary=m.read_json(point),surface_report=m.read_json(surface),arrays=arrays(candidate),candidate_sha256=m.sha(candidate),point_target_report=m.read_json(pr),point_targets=arrays(pt),point_target_sha256=m.sha(pt),point_depth_report=m.read_json(dr),depth_npz=arrays(dn),depth_sha256=m.sha(dn),prior_context=prior,prior_candidate_sha256=m.sha(prior["candidate"]),ancestry_parent_sha256=m.sha(prior["candidate"]))
 seed=Path(r["seed_pairs_path"]);guard=Path(r["fixed_guard_npz_path"]);gr=Path(r["fixed_guard_report_path"])
 for key in ("seed_pairs","fixed_guard_npz","fixed_guard_report"):
  if m.sha(r[key+"_path"])!=r[key+"_sha256"]:raise ValueError("fresh guard pin "+key)
 template=N/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json";mt=template.with_name("matched-inputs.json")
 matched=out/"matched-inputs.json"
 m.matched_inputs_from_template(mt,matched,{"candidate":candidate,"audit159":surface,"face_target_npz":fn,"point_target_npz":pt})
 argv=h.fit_argv_from_template(json.loads(template.read_text()),candidate=candidate,audit159=surface,point_target_npz=pt,point_target_report=pr,face_target_npz=fn,face_target_report=fr,point_depth_npz=dn,point_depth_report=dr,self_pairs=seed,fixed_guard_npz=guard,fixed_guard_report=gr,matched_inputs=matched,output=out/"fit",trust_mm=4)
 entry=N/"component-scp-area-backtrack-603/frozen-v1/run_joint_full_coordinate_fit603.py"
 if argv[1]!=str(N/"partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"):raise ValueError("unexpected template entry")
 argv[1]=str(entry)
 for flag in argv:
  if flag.startswith("--")and argv.count(flag)!=1:raise ValueError("duplicate flag "+flag)
 m.write_json(out/"fit-argv.json",argv)
 pins={str(p):m.sha(p)for p in [Path(__file__),Path(h.__file__),state,candidate,point,surface,parent,G/"prepared.json",refresh,pt,pr,dn,dr,fn,fr,seed,guard,gr,template,mt,matched,entry,entry.with_name("component_scp_backtracking603.py"),out/"fit-argv.json"]}
 m.write_json(out/"prepared.json",{"status":"MATCHED_REFINED_POINT_AND_FACE_TARGETS_PREPARED_UNADMITTED","validation":validation,"source_pins":pins,"state":str(state),"candidate_sha256":m.sha(candidate),"point_rows":422,"face_rows":2629,"all_goal_rows_retained":True,"fit_argv":str(out/"fit-argv.json"),"fit_not_run":True,"native_admission":False,"accepted_physical_steps":0})
 print(json.dumps({"status":"MATCHED_REFINED_POINT_AND_FACE_TARGETS_PREPARED_UNADMITTED","out":str(out)}))
if __name__=="__main__":main()
