from pathlib import Path
import collections,gzip,importlib.util,json,math

def loadmodule(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=loadmodule("gait","artifacts/crow-v15-whole-gait-audit-20261111/audit.py")
c=loadmodule("angles","artifacts/crow-v15-corvid-stride-audit-20261111/analyze.py")
A=Path("artifacts/crow-v15-delayed-ankle-gait-screen-20261111")
old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text())

def evidence(p):
 end={};sub={};caps=collections.Counter();bad=[];nonfoot=collections.defaultdict(float)
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if "CROW_SUBSTEP_ERROR" in line:bad.append(line.strip())
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
    if t<=199:
     if (e,t) in end:raise ValueError("duplicate prefix state")
     end[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   if line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);required,capacity=map(int,z[4:6])
    if required>capacity:bad.append("overflow %d %d %d"%(e,t,k))
    if t<=199:
     if (e,t,k) in sub:raise ValueError("duplicate prefix contact")
     sub[e,t,k]=z[4:]
    if 201<=t<=300:nonfoot[e,t]+=float(z[10])
   if line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1
 return end,sub,caps,bad,nonfoot

results=[]
for seed in (20261126,20261127):
 p=A/("autonomous-"+str(seed));ref=a.BASE[seed]
 if json.loads((p/"run-result.json").read_text())["exit_code"]!=0:raise ValueError("candidate execution failed")
 reference=next(r for r in old["rows"] if r["seed"]==seed);ground=reference["ground_z_m"]
 rows=a.analyze(p,ground);sup,sub,body,com=a.load(p);pe,ps,pc,pbad,pnon=evidence(p);re,rs,rc,rbad,rnon=evidence(ref)
 for e in range(8):
  ed=[t for t in range(1,200) if (e,t) not in pe or (e,t) not in re or pe[e,t]!=re[e,t]]
  sd=[(t,k) for t in range(1,200) for k in range(16) if (e,t,k) not in ps or (e,t,k) not in rs or ps[e,t,k]!=rs[e,t,k]]
  per=[]
  for si,side in enumerate(("left","right")):
   bouts=rows[e]["support_qualified_swing_bouts"][si];interior=[(lo,hi) for lo,hi in bouts if lo>201 and hi<300];evaluated=[]
   for lo,hi in interior:
    angles=[c.calc(body,sup[e,t]["q"],e,t,side) for t in range(lo,hi+1)];k=lo+min(range(len(angles)),key=lambda n:angles[n][1]);v=lo+min(range(len(angles)),key=lambda n:angles[n][2]);evaluated.append({"bout":[lo,hi],"knee_min_step":k,"ankle_min_step":v,"ankle_after_knee":v>k})
   per.append({"side":side,"all_bouts":bouts,"interior":evaluated,"phase_pass":bool(evaluated) and sum(z["ankle_after_knee"] for z in evaluated)>len(evaluated)/2})
  refrow=reference["baseline"][e];travel=rows[e]["forward_root_displacement_m"];rtravel=refrow["forward_root_displacement_m"]
  non=[t for t in range(201,301) if pnon[e,t]>1e-7]
  row={"seed":seed,"lane":e,"prefix_state_mismatches":len(ed),"prefix_contact_mismatches":len(sd),"first_state_mismatch":ed[:1],"first_contact_mismatch":sd[:1],"candidate_travel_m":travel,"reference_travel_m":rtravel,"candidate_toe_cap_substeps":pc[e],"reference_toe_cap_substeps":rc[e],"candidate_nonfoot_intervals":non,"sides":per,"pass_except_phase":not ed and not sd and travel>0 and travel>=.9*rtravel and pc[e]<=rc[e] and not non and all(len(z["all_bouts"])>=2 for z in per) and not pbad}
  results.append(row)
 print(seed,"candidate errors",pbad,"reference errors",rbad)
phase=sum(z["phase_pass"] for x in results for z in x["sides"]);out={"rows":results,"phase_sides_passing":phase,"all_lane_gates_pass":all(x["pass_except_phase"] for x in results),"all_gates_pass":phase>=24 and all(x["pass_except_phase"] for x in results)}
(A/"comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
for x in results:print(x["seed"],x["lane"],"phase",[z["phase_pass"] for z in x["sides"]],"swings",[len(z["all_bouts"]) for z in x["sides"]],"travel",x["candidate_travel_m"],"reference",x["reference_travel_m"],"cap",x["candidate_toe_cap_substeps"],x["reference_toe_cap_substeps"],"lane_pass",x["pass_except_phase"])
