from pathlib import Path
import json,os,re,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-corvid-toe-contact-admission-20261112";B=R/"build/crow-v15-corvid-toe-contact-admission-20261112";cmd=json.loads((A/"candidate-preflight-20261127/command.json").read_text());out={}
for name in ("control","high"):
 c=cmd.copy()
 for key in ("--state-trace","--crow-replay-pack"):
  c[c.index(key)+1]=str(B/("layout-"+name+"-"+key[2:]))
 env={**os.environ,"DYLD_LIBRARY_PATH":str(B/"layout-audit"),"CROW_LAYOUT_AUDIT_IGNORE_WORLD":"1","CROW_LAYOUT_AUDIT_IGNORE_RADII":"1","CROW_LAYOUT_AUDIT_PRINT_RADII":"1","CROW_EXPERIMENT_GAIT_SMOOTH":"1"};env.pop("CROW_EXPERIMENT_CORVID_TOES",None)
 if name=="high":env["CROW_EXPERIMENT_CORVID_TOES"]="1"
 p=subprocess.run(c,cwd=R,env=env,text=True,capture_output=True,timeout=120);(A/("layout-"+name+".stdout")).write_text(p.stdout);(A/("layout-"+name+".stderr")).write_text(p.stderr);m=re.search(r"task=\(world=(\d+),task=(\d+),observation=(\d+),action=(\d+)\)",p.stderr);assert p.returncode!=0 and m,p.stderr
 out[name]=list(map(int,m.groups()))
out["same_other_observation_action_hashes"]=out["control"][2:]==out["high"][2:];out["different_world_hashes"]=out["control"][0]!=out["high"][0];out["scope"]="Diagnostic-only host omits the world fingerprint and contact-member radii from the observation/action hashes while retaining all other original hash inputs; changed radii remain meaningful model data. Actor contract failure is expected. Equality corroborates unchanged compiled observation/action layout inputs, not equivalence of physical trajectories or an approval to promote the changed toe geometry world. Normal host retains full world-bound fingerprints."
(A/"layout-audit.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if not out["same_other_observation_action_hashes"] or not out["different_world_hashes"]:raise SystemExit("layout equivalence not established")
