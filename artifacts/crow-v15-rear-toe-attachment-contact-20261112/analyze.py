from pathlib import Path
import importlib.util,gzip,json,collections,math
p=Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112")
def mod(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=mod("original","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");h=mod("ratio",p/"audit-ratio.py");ground=next(x["ground_z_m"] for x in json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())["rows"] if x["seed"]==20261127)
paths={"control":p/"control-20261127","candidate":p/"candidate-20261127"};out={}
for name,path in paths.items():
 a=h if name=="candidate" else g;rows=a.analyze(path,ground);sub={};caps=collections.Counter();bad=[]
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if "CROW_SUBSTEP_ERROR" in l:bad.append(l.strip())
   if l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if 201<=t<=300:
     assert (e,t,k) not in sub;sub[e,t,k]=list(map(float,z[6:]));
     if int(z[4])>int(z[5]):bad.append("overflow")
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if 201<=t<=300 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1
 assert len(sub)==12800
 for e,r in enumerate(rows):
  r["both_zero_substeps"]=sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for t in range(201,301) for k in range(16));r["nonfoot_intervals"]=[t for t in range(201,301) if sum(sub[e,t,k][4] for k in range(16))>1e-7];r["toe_caps"]=caps[e]
 episode=json.loads((path/"run.stdout").read_text())["first_episode_navigation"]
 out[name]={"rows":rows,"episode":episode,"errors":bad,"low_toe_count":sum(sum(r["sole_clear_but_toes_below_clearance_intervals"]) for r in rows)}
out["scope"]="Development seed, 2 s walking after 4 s preparation. Candidate uses model-specific native FK and collision extents, validated against lane-5 replay. Geometry differs from control, so trajectories are not expected to match; results are exploratory and not biological qualification."
(p/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for n in paths:
 print(n,"low-toe",out[n]["low_toe_count"],"errors",len(out[n]["errors"]))
 for r in out[n]["rows"]:print(r["lane"],"travel",round(r["forward_root_displacement_m"],4),"bouts",[len(z) for z in r["support_qualified_swing_bouts"]],"zero",r["both_zero_substeps"],"caps",r["toe_caps"],"nonfoot",len(r["nonfoot_intervals"]))
