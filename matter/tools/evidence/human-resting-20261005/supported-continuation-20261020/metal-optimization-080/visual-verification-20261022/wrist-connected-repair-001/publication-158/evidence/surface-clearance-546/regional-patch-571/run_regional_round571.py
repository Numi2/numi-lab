"""Pinned star-search execution followed by the existing physical-construction gates."""
from pathlib import Path
import importlib.util,json,sys,argparse
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("stage569",N/"continue_surface_round546.py");m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument("--helper",required=True,type=Path);p.add_argument("--state",type=Path,default=N/"connected-star-573/run-001/final/state.json");p.add_argument("--out",required=True,type=Path);p.add_argument("--seconds",default="160");a=p.parse_args()
O=a.out;S=a.state
if O.exists():raise FileExistsError(O)
O.mkdir(parents=True)
folder=Path(__file__).resolve().parent
m.run_stage([sys.executable,"-m","pytest",folder/"test_regional_candidates571.py",N/"changed-triangle-quality-572/test_changed_triangle_masks572.py","-q"],O/"unit-tests",slot=False,source_paths=(folder/"regional_candidates571.py",folder/"test_regional_candidates571.py",N/"connected-star-569/star_candidates569.py",N/"changed-triangle-quality-572/changed_triangle_masks572.py",N/"changed-triangle-quality-572/test_changed_triangle_masks572.py"),timeout=30)
owner=folder/"search_regional571.py"
m.run_stage([sys.executable,owner,"--state",S,"--payload-helper",a.helper,"--output",O/"search","--seconds",a.seconds,"--rounds","6","--max-centers","1000","--max-degree","48","--max-candidates-per-center","128"],O/"search-stage",slot=True,source_paths=(Path(__file__),owner,folder/"regional_candidates571.py",N/"connected-star-569/star_candidates569.py",N/"changed-triangle-quality-572/changed_triangle_masks572.py",N/"same-count-star-payload-568/cached_counter569_v2.py",a.helper,S),timeout=300)
r=json.loads((O/"search/report.json").read_text());print(json.dumps({k:r.get(k)for k in("status","before_pairs","after_pairs_composed","reduction","exact_triangle_calls","seconds")}),flush=True)
if not r["accepted_operations"]:raise SystemExit(0)
lineage=N/"affine-ancestry-refresh-566/rebuild_affine_rows566.py"
m.run_stage([sys.executable,lineage,"--candidate",O/"search/candidate.npz","--out",O/"lineage"],O/"lineage-stage",slot=True,source_paths=(lineage,),timeout=60)
check=N.parent/"contained-clearance-545/run_candidate_checks545.py"
m.run_stage([sys.executable,check,"--candidate",O/"lineage/candidate.npz","--baseline-state",S,"--trust-mm","4","--output-dir",O/"full-check"],O/"full-check-stage",slot=False,source_paths=(check,),timeout=600)
r=json.loads((O/"full-check/report.json").read_text());print(json.dumps({k:r.get(k)for k in("status","surface_pairs","contained_pairs","self_pairs","construction_progress")}),flush=True)
if r.get("construction_progress")and r.get("contained_pairs")==0 and r.get("self_pairs")==0:
 final=N/"finalize_existing_checks546.py"
 m.run_stage([sys.executable,final,"--candidate",O/"lineage/candidate.npz","--baseline-state",S,"--point-summary",O/"full-check/audit-130/summary.json","--surface-report",O/"full-check/audit-159/report.json","--out",O/"final","--trust-mm","4"],O/"final-stage",slot=False,source_paths=(final,),timeout=60)

