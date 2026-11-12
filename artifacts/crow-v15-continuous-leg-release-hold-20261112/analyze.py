from pathlib import Path
import collections,gzip,hashlib,json,math,sys
A=Path("artifacts/crow-v15-continuous-leg-release-hold-20261112");B=Path("artifacts/crow-v15-sustained-crouch-replication-20261112")
def read(d):
 meta=json.loads((d/"run.stdout").read_text())["first_episode_navigation"];h={(e,k):hashlib.sha256() for e in range(8) for k in ("state","contact")};cnt=collections.Counter();stats=collections.defaultdict(collections.Counter);state={};caps=collections.defaultdict(list);errors=[]
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split();
   if not z:continue
   if z[0]=="CROW_SUPPORT_SAMPLE":
    x=json.loads(line.split(" ",1)[1]);t,e=x["step"],x["lane"];key="state"
    if t<=meta["steps_by_environment"][e]:state[e,t]=x
   elif z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);key="contact"
    if t<=meta["steps_by_environment"][e] and 201<=t<=625:
     s=stats[e];s["contact"]+=1;s["overflow"]+=int(int(z[4])>int(z[5]));s["nonfoot"]+=int(float(z[10])>1e-7);
     if 501<=t<=515:s["transition_coverage"]+=1;s["transition_unloaded"]+=int(float(z[8])+float(z[9])<=1e-7)
     if 549<=t<=556:s["launch_normal_impulse_ns"]+=float(z[8])+float(z[9])
     if 621<=t<=625:s["post_push_coverage"]+=1;s["post_push_loaded"]+=int(float(z[8])+float(z[9])>1e-7)
   elif z[0]=="CROW_TOE_V_SUBSTEP":
    t,k,e=map(int,z[1:4]);
    if t<=meta["steps_by_environment"][e] and 201<=t<=625:
     stats[e]["toe_coverage"]+=1
     if max(abs(float(v)) for v in z[4:])>=11.99:caps[e].append([t,k])
    continue
   elif z[0]=="CROW_SUBSTEP_ERROR":errors.append(line.strip());continue
   else:continue
   if t<=485 and t<=meta["steps_by_environment"][e]:h[e,key].update(line.encode());cnt[e,key]+=1
 return {"meta":meta,"hashes":{str(e)+"-"+k:{"n":cnt[e,k],"sha":h[e,k].hexdigest()} for e in range(8) for k in ("state","contact")},"stats":stats,"state":state,"caps":caps,"errors":errors}
a=read(B/("candidate-"+sys.argv[1]+"-rebound-hip"));b=read(A/("candidate-"+sys.argv[1]+"-rebound-hip"));rows=[]
for e in range(8):
 s=b["state"];end=b["meta"]["steps_by_environment"][e];r={"lane":e,"arm":[.20,.20,.40,.40,.60,.60,.80,.80][e],"prefix_equal":{k:a["hashes"][str(e)+"-"+k]==b["hashes"][str(e)+"-"+k] for k in ("state","contact")},"prefix_record_counts":{k:b["hashes"][str(e)+"-"+k]["n"] for k in ("state","contact")},"end":end,"failure":b["meta"]["physical_failure_by_environment"][e],"termination_reason":b["meta"]["termination_reason_by_environment"][e],"stats":dict(b["stats"][e]),"caps":b["caps"][e],"z_vz_samples":{str(t):[s[e,t]["q"][2],s[e,t]["v"][2]] for t in (500,515,530,548,556,570,575,600,625) if (e,t) in s}};rows.append(r)
r={"rows":rows,"errors":b["errors"],"scope":"Exploratory used-seed handoff. Total normal impulse includes gravity/aerodynamics; no leg-only work claim. Body/wing caps unobserved."};(A/("analysis-"+sys.argv[1]+".json")).write_text(json.dumps(r,indent=2)+"\n")
for x in rows:print(x)
print("errors",r["errors"])
