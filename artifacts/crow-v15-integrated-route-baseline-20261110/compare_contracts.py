from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-integrated-route-baseline-20261110";B=R/"build/crow-v15-integrated-route-baseline-20261110";cmd=json.loads((A/"contract-bindings.json").read_text())["argv"]
for flag in ("--policy-pack","--base-policy-pack"):
 i=cmd.index(flag);del cmd[i:i+2]
rows={}
for mode,lib,visual in [("prior",R/"build/crow-v15-grip-search-20261109/contract-debug-envelope",R/"assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json"),("integrated",B/"contract-debug-integrated",R/"artifacts/crow-v15-visual-binding-fix-20261110/crow-navigation.toe-body.visual-observation.json")]:
 c=cmd.copy();c[c.index("--visual-observation-config")+1]=str(visual)
 env={**os.environ,"DYLD_LIBRARY_PATH":str(lib),"CROW_CONTRACT_DIAGNOSTIC":"1","PYTHONDONTWRITEBYTECODE":"1"};p=subprocess.run(c,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/(mode+"-contract.stdout")).write_text(p.stdout);(A/(mode+"-contract.stderr")).write_text(p.stderr)
 if "CROW_CONTRACT layout" not in p.stderr:raise RuntimeError(mode+" "+p.stderr[-500:])
 rows[mode]=[x for x in p.stderr.splitlines() if x.startswith("CROW_CONTRACT")];print(mode,*rows[mode],sep="\n")
(A/"contract-comparison.json").write_text(json.dumps({**rows,"scope":"compiled table signatures only; corrected visual inputs and mechanics differ"},indent=2)+"\n")
