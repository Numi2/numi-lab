"""Opt-in SAT target adapter reusing the pinned458 preflight and408 fitter."""
from pathlib import Path
import importlib.util,hashlib,json,sys,types
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
WRAPPER=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
SAT=Path(__file__).with_name("triangle_sat_rows_466.py")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def install_sat(owner,original_objective,crossing):
 ids={(int(p["face_row"]),int(p["body"]),int(p["bone_id"]),int(p["triangle_id"]))for p in crossing["pairs"]}
 def rows(*args,**kw):
  kw["separating_axis_pairs"]=ids
  return sat.build_fixed_bone_outward_plane_rows(*args,**kw)
 proxy=types.SimpleNamespace(build_fixed_bone_outward_plane_rows=rows)
 return original_install(owner,proxy,crossing)
if __name__=="__main__":
 assert sha(WRAPPER)=="08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
 assert sha(SAT)=="cf4df8cd8a3f2773f0992a5f9a1d9b899d64db781b0a58cc2ed618285fa11cf3"
 pins={str(p):sha(p)for p in(WRAPPER,SAT,Path(__file__))}
 wrapper=load("pinned_fit458_for_sat466",WRAPPER);sat=load("sat_target466",SAT)
 original_install=wrapper.install_objective;wrapper.install_objective=install_sat
 wrapper.main()
 out=Path(sys.argv[sys.argv.index("--output")+1]);p=out/"report.json";report=json.loads(p.read_text())
 report["fixed_bone_outward_plane_objective_458"]["name"]="466 SAT minimum maximum corner deficit, outward hemisphere filter;458 input validation and408 constraints"
 report["separating_axis_objective_466"]=dict(pins=pins,all_current_crossing_pairs_selected=True,selection="deterministic min max positive deficit over bone normal, skin normal and nine edge-cross axes; outward hemisphere",same_hard_constraints=True,retained_458_nonpositive_corner_guards=True,native_admission=False,accepted_physical_steps=0)
 p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 assert all(sha(k)==v for k,v in pins.items())
