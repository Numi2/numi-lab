from pathlib import Path
import collections,gzip,hashlib,json,sys
A=Path("artifacts/crow-v15-integrated-leg-push-control-20261112");B=Path("artifacts/crow-v15-toe-bypass-paired-control-20261112");seed=sys.argv[1];name="candidate-"+seed+"-rebound-hip"
def read(d):
 h={};n=collections.Counter();states={};contact={}
 with gzip.open(d/name/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);t,e=x["step"],x["lane"];k="state";states[e,t]=x
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,sub,e=map(int,z[1:4]);k="contact";contact[e,t,sub]=[float(z[i]) for i in (6,7,8,9)]
   else:continue
   if t<=625:
    key=e,t,k
    if key not in h:h[key]=hashlib.sha256()
    h[key].update(line.encode());n[key]+=1
 return {k:(v.hexdigest(),n[k]) for k,v in h.items()},states,contact
(ha,sa,ca),(hb,sb,cb)=read(A),read(B);rows=[]
for e in range(8):
 x={"lane":e}
 for kind in ("state","contact"):
  start=1 if kind=="state" else 201;expected=1 if kind=="state" else 16
  missing=[t for t in range(start,626) if ha.get((e,t,kind),(None,0))[1]!=expected or hb.get((e,t,kind),(None,0))[1]!=expected]
  diffs=[t for t in range(start,626) if ha.get((e,t,kind))!=hb.get((e,t,kind))]
  x[kind+"_incomplete_steps"]=missing;x[kind+"_first_difference"]=min(diffs) if diffs else None
 for label,s,c in (("crouch_hold",sa,ca),("extension",sb,cb)):
  windows={}
  for lo,hi in ((549,556),(549,560),(549,575),(549,600)):
   v=[c[e,t,k] for t in range(lo,hi+1) for k in range(16) if (e,t,k) in c]
   windows[f"{lo}_{hi}"]={"count":len(v),"vertical_normal_ns":sum(z[0]+z[1] for z in v),"normal_ns":sum(z[2]+z[3] for z in v),"unloaded_substeps":sum(z[2]+z[3]<=1e-7 for z in v)}
  x[label]={"windows":windows,"root":{str(t):{"z_m":s[e,t]["q"][2],"vz_m_s":s[e,t]["v"][2]} for t in (548,556,560,575,600,625) if (e,t) in s}}
 rows.append(x)
(A/("paired-"+seed+".json")).write_text(json.dumps(rows,indent=2)+"\n")
for x in rows:
 print(x["lane"],"first state/contact",x["state_first_difference"],x["contact_first_difference"],"missing",len(x["state_incomplete_steps"]),len(x["contact_incomplete_steps"]),"normal 549–556 extension/hold",round(x["extension"]["windows"]["549_556"]["vertical_normal_ns"],4),round(x["crouch_hold"]["windows"]["549_556"]["vertical_normal_ns"],4),"vz 556 extension/hold",x["extension"]["root"].get("556",{}).get("vz_m_s"),x["crouch_hold"]["root"].get("556",{}).get("vz_m_s"))
