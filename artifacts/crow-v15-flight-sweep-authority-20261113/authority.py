from pathlib import Path
import gzip,json,math,collections
A=Path("artifacts/crow-v15-flight-sweep-authority-20261113");P=A/"push-wing-20261126-rebound-sweep";Q=Path("artifacts/crow-v15-overlap-obstacle-free-20261113/push-wing-20261126-rebound-clear")
data=[]
for d in (Q,P):
 state={};contact=collections.defaultdict(lambda:[0.,0.,0])
 with gzip.open(d/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);t=x["step"];e=x["lane"]
    if e>=2 and 475<=t<=525:state[e,t]=x
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if e>=2 and 475<=t<=525:
     c=contact[e,t];c[0]+=sum(map(float,z[8:10]));c[1]+=float(z[10]);c[2]+=1
 data.append((state,contact))
audit=json.loads((A/"sensor-feedback-audit.json").read_text());identical=bool(audit["mask_valid"] and audit["policy_pack_bytes_identical"]);rows=[]
def pitch(q):
 x,y,z,w=q[3:7];return math.asin(max(-1.,min(1.,2*(w*y-z*x))))
for e in range(2,8):
 missing=[];delta=[];stops=[];imp=[]
 for t in range(475,526):
  if any((e,t) not in s or c[e,t][2]!=16 for s,c in data):missing.append(t);continue
  old,new=(s[e,t] for s,c in data);delta.append(pitch(new["q"])-pitch(old["q"]))
  for s,c in data:
   imp.append(c[e,t][:2]);stops += [abs(s[e,t]["q"][7]),abs(s[e,t]["q"][8])]
 parity=next(x for x in audit["rows"] if x["lane"]==e)["reference_parity_before_first_change"]
 clean=not missing and bool(imp) and all(max(x)<=1e-7 for x in imp)
 d500=None
 if all((e,500) in s for s,c in data):d500=pitch(data[1][0][e,500]["q"])-pitch(data[0][0][e,500]["q"])
 mean=sum(delta)/len(delta) if len(delta)==51 else None
 rows.append({"lane":e,"mean_pitch_difference_475_525_rad":mean,"pitch_difference_at_500_rad":d500,"missing_ticks":missing,"both_unloaded_and_no_nonfoot_impulse":clean,"matched_preintervention":parity,"max_sampled_abs_sweep_position_rad":max(stops) if stops else None})
pairs={}
for e in (2,4,6):
 r=[x for x in rows if x["lane"] in (e,e+1)];v=[x["mean_pitch_difference_475_525_rad"] for x in r]
 pairs[str(e)+"-"+str(e+1)]=bool(identical and len(r)==2 and all(x["matched_preintervention"] and x["both_unloaded_and_no_nonfoot_impulse"] for x in r) and all(x is not None and abs(x)>=.05 for x in v) and v[0]*v[1]>0)
out={"mask_and_policy_valid":identical,"rows":rows,"paired_pitch_authority_indication":pairs,"scope":"Matched used-seed identification only; no biological or flight qualification."}
(A/"authority.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
