from pathlib import Path
import collections,gzip,json,math,os,subprocess
R=Path.cwd();A=R/"artifacts/crow-v15-per-lane-reset-clearance-diagnostic-20261111";P=A/"autonomous-20261126-sourcecheck";B=A/"autonomous-20261126";initial={};observed={};reference={}
def read(p,out,source):
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1])
    if x["step"]<=4:out["state",x["lane"],x["step"]]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP ") or line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=4:out[z[0],e,t,k]=z[4:]
   elif source and line.startswith("CROW_QV_SOURCE "):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=4:initial[e,t,k]=list(map(float,z[4:]))
read(P,observed,True);read(B,reference,False);differences=[k for k in set(observed)|set(reference) if observed.get(k)!=reference.get(k)]
continuity=[]
for e in range(8):
 for t in (3,4):
  v=initial[e,t,0];prev=observed["state",e,t-1];continuity.extend(abs(a-b) for a,b in zip(v,prev["q"]+prev["v"]))
text="".join(str(e)+" 1 "+" ".join(map(str,initial[e,1,0]))+"\n" for e in range(8))
env={**os.environ,"DYLD_LIBRARY_PATH":str(R/"build/crow-v15-hip-roll-journey-integration-20261111")}
p=subprocess.run([str(R/"build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics")],input=text,capture_output=True,text=True,check=True,env=env);poses=collections.defaultdict(list)
for line in p.stdout.splitlines():
 z=line.split()
 if z[0]!="body":continue
 e,t=int(z[1]),int(z[2]);name=z[3];pose=list(map(float,z[4:]))
 if "toe" not in name and not name.endswith("_foot"):continue
 half=[.01,.01725,.0092] if name.endswith("_foot") else [.009,.0025,.0092];x,y,z,w=pose[3:7];row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));gap=pose[2]-sum(abs(a)*b for a,b in zip(row,half));poses[e].append((gap,name))
out={"compared_records":[len(observed),len(reference)],"differing_records":len(differences),"first_difference":str(differences[:1]),"source_samples":len(initial),"expected_source_samples":8*4*16,"previous_endpoint_max_abs_error":max(continuity),"lanes":[{"lane":e,"initial_root_m":initial[e,1,0][:3],"minimum_foot_toe_gap_m":min(poses[e])[0],"lowest_shape":min(poses[e])[1],"toe_v_max":max(abs(x) for x in initial[e,1,0][41+22:41+38])} for e in range(8)],"scope":"Read-only first source state from v15 with hip roll, compared with the matching seed-specific corrected-reset run. Negative separation denotes initial overlap; geometry is estimated, and overlap alone does not prove causation."}
(A/"source-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
