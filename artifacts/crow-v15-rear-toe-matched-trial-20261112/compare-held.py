from pathlib import Path
import collections,gzip,importlib.util,json,math,sys,statistics
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=mod("a","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");A=Path("artifacts/crow-v15-rear-toe-matched-trial-20261112");variant="candidate-vs-held-control-20261127";C=A/"candidate-20261127-rebound-preload";R=A/"candidate-20261127-rebound-control"
old=json.loads(Path("artifacts/crow-v15-whole-gait-audit-20261111/audit.json").read_text());ref=next(z for z in old["rows"] if z["seed"]==20261127);crow=a.analyze(C,ref["ground_z_m"]); matched_ref=a.analyze(R,ref["ground_z_m"])
def parse(p):
 st={};sub={};q={};caps=collections.Counter();bad=[]
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if "CROW_SUBSTEP_ERROR" in l:bad.append(l.strip())
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"];
    if 1<=t<=300:st[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in sub
    if int(z[4])>int(z[5]):bad.append("overflow")
    if 1<=t<=300:sub[e,t,k]=list(map(float,z[6:]))
   elif l.startswith("CROW_Q_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in q;q[e,t,k]=list(map(float,z[4:]));assert len(q[e,t,k])==41
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e]+=1
 assert len(st)==2400 and len(sub)==38400 and len(q)==12800
 assert max(abs(x-y) for e in range(8) for t in range(201,301) for x,y in zip(q[e,t,15],st[e,t][0]))<1e-6
 return st,sub,q,caps,bad
c=parse(C);r=parse(R);assert json.loads((C/"run-result.json").read_text())["exit_code"]==0
assert json.loads((R/"run-result.json").read_text())["exit_code"]==0
post_divergence=sum(c[0][e,t]!=r[0][e,t] for e in range(8) for t in range(201,301))
assert post_divergence > 0, "experimental switch produced no recorded walking difference"
rows=[]
for e in range(8):
 sd=[t for t in range(1,200) if c[0][e,t]!=r[0][e,t]]
 cd=[(t,k) for t in range(1,200) for k in range(16) if c[1][e,t,k]!=r[1][e,t,k]]
 def un(sub):return sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for t in range(201,301) for k in range(16))
 def non(sub):return [t for t in range(201,301) if sum(sub[e,t,k][4] for k in range(16))>1e-7]
 row={"lane":e,"prefix_state_mismatches":sd[:3],"prefix_contact_mismatches":cd[:3],"candidate_unloaded_substeps":un(c[1]),"reference_unloaded_substeps":un(r[1]),"candidate_nonfoot_intervals":non(c[1]),"candidate_caps":c[3][e],"reference_caps":r[3][e],"candidate_swing_counts":[len(v) for v in crow[e]["support_qualified_swing_bouts"]],"candidate_travel_m":crow[e]["forward_root_displacement_m"],"reference_travel_m":matched_ref[e]["forward_root_displacement_m"]}
 row["whole_body_metrics"]={k:crow[e][k] for k in ("root_height_span_m","whole_body_com_height_span_m","root_roll_span_rad","root_pitch_span_rad","joint_spans_rad","head_world_axis_spans_m","head_relative_to_torso_axis_spans_m","foot_loaded_fraction_window","both_loaded_fraction_window","neither_loaded_fraction_window","sole_clear_but_toes_below_clearance_intervals","foot_COM_motion_while_loaded_proxy_m","stride_metrics_status")}
 row["pass"]=not sd and not cd and not row["candidate_nonfoot_intervals"] and row["candidate_caps"]<=row["reference_caps"] and min(row["candidate_swing_counts"])>=2 and row["candidate_travel_m"]>=.9*row["reference_travel_m"]
 rows.append(row)
def meanacc(q,j):
 acc=[]
 for e in range(8):
  for t in range(202,301):
   x=q[e,t,0][j];y=q[e,t-1,15][j];z=q[e,t-1,14][j];acc.append(abs(x-2*y+z)/(.00125**2))
 return sum(acc)/len(acc)
def meanwithin(q,j):
 vals=[]
 for e in range(8):
  for t in range(202,301):
   for k in range(1,16):
    def v(n):return q[e,t+n//16,n%16][j]
    vals.append(abs(v(k)-2*v(k-1)+v(k-2))/(.00125**2))
 return sum(vals)/len(vals)
acc=[]
for name,j in (("left_knee",15),("left_ankle",16),("right_knee",18),("right_ankle",19)):
 x=meanacc(c[2],j);y=meanacc(r[2],j);acc.append({"joint":name,"candidate_mean":x,"reference_mean":y,"fraction":x/y,"candidate_within_mean":meanwithin(c[2],j),"reference_within_mean":meanwithin(r[2],j),"pass":x<=.8*y})
cu=sum(x["candidate_unloaded_substeps"] for x in rows);ru=sum(x["reference_unloaded_substeps"] for x in rows)
out={"post_walk_divergent_state_samples":post_divergence,"rows":rows,"boundary_acceleration":acc,"candidate_unloaded_substeps":cu,"reference_unloaded_substeps":ru,"candidate_errors":c[4],"reference_errors":r[4],"all_gates_pass":all(x["pass"] for x in rows) and statistics.median(x["candidate_travel_m"] for x in rows)>=.9*statistics.median(x["reference_travel_m"] for x in rows) and cu<=ru and not c[4] and not r[4],"limits":"Stance continuity begins with walking; pre-gait exact parity is required. Acceleration values are descriptive only. Development seed only, 2s walking window. q second differences are sampled acceleration proxies, not actual torque or biological validation."}
(A/(variant+"-comparison.json")).write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
