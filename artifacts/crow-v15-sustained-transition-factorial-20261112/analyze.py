from pathlib import Path
import collections,gzip,hashlib,json,math
A=Path("artifacts/crow-v15-sustained-transition-factorial-20261112");B=Path("artifacts/crow-v15-sustained-contact-gait-20261112")
def read(d):
 meta=json.loads((d/"run.stdout").read_text())["first_episode_navigation"];h={(e,k):hashlib.sha256() for e in range(8) for k in ("state","contact")};cnt=collections.Counter();stat=collections.defaultdict(lambda:collections.Counter());state={};caps=collections.defaultdict(list);errors=[]
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split();
   if not z:continue
   if z[0]=="CROW_SUPPORT_SAMPLE":
    x=json.loads(line.split(" ",1)[1]);t,e=x["step"],x["lane"];key="state"
    if t<=meta["steps_by_environment"][e]:state[e,t]=x
   elif z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);key="contact"
    if t<=meta["steps_by_environment"][e] and 201<=t<=500:
     stat[e]["contact"]+=1
     if int(z[4])>int(z[5]):stat[e]["overflow"]+=1
     if t>=301 and float(z[8])+float(z[9])<=1e-7:stat[e]["post6_unloaded"]+=1
     if float(z[10])>1e-7:stat[e]["nonfoot"]+=1
   elif z[0]=="CROW_TOE_V_SUBSTEP":
    t,k,e=map(int,z[1:4]);
    if t<=meta["steps_by_environment"][e] and 201<=t<=500:
     stat[e]["toe_coverage"]+=1
     if max(abs(float(v)) for v in z[4:])>=11.99:caps[e].append([t,k])
    continue
   elif z[0]=="CROW_SUBSTEP_ERROR":errors.append(line.strip());continue
   else:continue
   if t<=300 and t<=meta["steps_by_environment"][e]:h[e,key].update(line.encode());cnt[e,key]+=1
 return {"meta":meta,"hashes":{str(e)+"-"+k:{"n":cnt[e,k],"sha":h[e,k].hexdigest()} for e in range(8) for k in ("state","contact")},"stat":stat,"state":state,"caps":caps,"errors":errors}
a=read(B/"candidate-20261126-rebound-hip");b=read(A/"candidate-20261126-rebound-hip");rows=[]
for e in range(8):
 x,y=a["stat"][e],b["stat"][e];matches={k:a["hashes"][str(e)+"-"+k]==b["hashes"][str(e)+"-"+k] for k in ("state","contact")};s=b["state"];dx=s[e,500]["q"][0]-s[e,200]["q"][0] if (e,500) in s and (e,200) in s else None
 rows.append({"lane":e,"arm":["original","original","crouch","crouch","feedback","feedback","both","both"][e],"prefix_match":matches,"prefix_record_counts":{k:b["hashes"][str(e)+"-"+k]["n"] for k in matches},"baseline":dict(x),"candidate":dict(y),"baseline_caps":a["caps"][e],"caps":b["caps"][e],"dx_m":dx,"failure":b["meta"]["physical_failure_by_environment"][e],"end":b["meta"]["steps_by_environment"][e]})
r={"rows":rows,"errors":{"baseline":a["errors"],"candidate":b["errors"]},"scope":"Used-seed exploratory transition factorial with matched same-lane source trajectories. Contact proxy is not material slip. Physical or biological qualification does not follow."};(A/"analysis.json").write_text(json.dumps(r,indent=2)+"\n")
for x in rows: print(x)
print("errors",r["errors"])
