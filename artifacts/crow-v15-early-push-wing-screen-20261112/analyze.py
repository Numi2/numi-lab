from pathlib import Path
import gzip,importlib.util,json,math,os,subprocess
R=Path.cwd();A=Path("artifacts/crow-v15-early-push-wing-screen-20261112");ref=Path("artifacts/crow-v15-moderate-walk-flight-integration-20261112/candidate-20261127-rebound-flexed")
def read(p,observe):
 states={};contacts={};coords={};caps={};errors=[]
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for l in f:
   if "CROW_SUBSTEP_ERROR" in l:errors.append(l.strip())
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=370:assert (e,t) not in states;states[e,t]=[x[k] for k in ("q","v","policy_action","contacts")]
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);
    if int(z[4])>int(z[5]):errors.append("contact overflow")
    if 1<=t<=370:assert (e,t,k) not in contacts;contacts[e,t,k]=z[4:]
   elif l.startswith("CROW_Q_SUBSTEP ") and observe:
    z=l.split();t,k,e=map(int,z[1:4]);q=list(map(float,z[4:]));assert (e,t,k) not in coords and len(q)==41 and all(math.isfinite(x) for x in q);coords[e,t,k]=q
   elif l.startswith("CROW_TOE_V_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4]);caps[e,t,k]=max(abs(float(v)) for v in z[4:])
 assert len(states)==8*370 and len(contacts)==8*370*16 and not errors,(len(states),len(contacts),errors[:3])
 if observe:assert len(coords)==8*31*16
 return states,contacts,coords,caps
reference=read(ref,False);runs={}
for label in ("baseline","early","earlyinv"):
 p=(Path("artifacts/crow-v15-contact-timed-wing-screen-20261112/baseline-20261127-rebound") if label=="baseline" else A/(label+"-20261127-rebound"));assert json.loads((p/"run-result.json").read_text())["exit_code"]==0;runs[label]=read(p,True)
control=runs["baseline"];mismatch=[sum(control[0][e,t]!=reference[0][e,t] for e in range(8) for t in range(1,371)),sum(control[1][e,t,k]!=reference[1][e,t,k] for e in range(8) for t in range(1,371) for k in range(16))];assert mismatch==[0,0],mismatch
s=importlib.util.spec_from_file_location("audit","artifacts/crow-v15-whole-gait-audit-20261111/audit.py");a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
out={"baseline_endpoint_and_contact_mismatches":mismatch,"arms":{},"limits":"Used development seed. Modeled wing-center motion is not feather shape or wing force. Ground vertical impulse includes passive and aerodynamic contributions; it is not isolated muscle impulse. Same-substep ordering cannot be resolved."}
for label,(st,co,qs,caps) in runs.items():
 err=max(abs(x-y) for (e,t,k),q in qs.items() if k==15 for x,y in zip(q,st[e,t][0]));assert err<1e-6,err
 inp="".join(f"{e} {t} {k} "+" ".join(map(str,v))+"\n" for (e,t,k),v in sorted(qs.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")};p=subprocess.run(["build/crow-v15-takeoff-substep-wing-observer-20261112/kinematics"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
 for line in p.stdout.splitlines():
  z=line.split();body[int(z[1]),int(z[2]),int(z[3]),z[4]]=list(map(float,z[5:]))
 assert len(body)==len(qs)*3
 with gzip.open((Path("artifacts/crow-v15-contact-timed-wing-screen-20261112/baseline-20261127-rebound") if label=="baseline" else A/(label+"-20261127-rebound"))/"replay.json.gz","rt") as f:rp=json.load(f)["payload"]
 assert rp["environment"]==5;frames={x["step"]+1:x for x in rp["frames"] if 334<=x["step"]<=364};assert len(frames)==31
 poserr=roterr=0.
 for t,f in frames.items():
  for n in ("crow_body","crow_left_wing","crow_right_wing"):
   x=body[5,t,15,n];i=rp["body_names"].index(n)*13;y=f["body_states"][i:i+7];poserr=max(poserr,math.dist(x[:3],y[:3]));roterr=max(roterr,min(math.dist(x[3:7],y[3:7]),math.dist(x[3:7],[-v for v in y[3:7]])))
 assert poserr<1e-5 and roterr<1e-5
 def z(e,t,k,side):
  q=qs[e,t,k];p=body[e,t,k,"crow_"+side+"_wing"];return a.rotate_inv(q[3:7],[p[i]-q[i] for i in range(3)])[2]
 rows=[]
 for e in range(8):
  loaded=[(t,k) for t in range(336,366) for k in range(16) if sum(map(float,co[e,t,k][4:6]))>1e-7];last=loaded[-1] if loaded else None
  samples=[co[e,t,k] for t in range(340,366) for k in range(16) if last and (t,k)<=last];imp=[sum(float(v[2+i]) for v in samples) for i in (0,1)];weight=.45006*9.81*.00125*len(samples)
  prior=next((t for t in range(1,371) if st[e,t]!=control[0][e,t]),None)
  rows.append({"lane":e,"first_endpoint_difference":prior,"last_loaded_substep_340_365":last,"loaded_window_duration_ms_from_step340":len(samples)*1.25,"foot_upward_impulse_Ns":imp,"weight_impulse_Ns":weight,"toe_speed_caps_340_370":sum(v>=11.99 for (lane,t,k),v in caps.items() if lane==e and 340<=t<=370),"wing_z_change_before_last_loaded_m":{side:z(e,*last,side)-z(e,339,15,side) for side in ("left","right")} if last and last[0]>=340 else None})
 out["arms"][label]={"endpoint_substep_q_error":err,"replay_fk_position_error_m":poserr,"replay_fk_orientation_error":roterr,"rows":rows}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
