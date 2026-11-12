from pathlib import Path
import collections,gzip,hashlib,json,statistics,sys
A=Path(sys.argv[1]);B=Path("artifacts/crow-v15-uniform-hip-pitch-walk-launch-20261112");name="candidate-20261127-rebound-hip"
def read(p):
 h={(e,k):hashlib.sha256() for e in range(8) for k in ("state","contact","allstate","allcontact")};n=collections.Counter();stats=collections.defaultdict(collections.Counter);errors=[]
 with gzip.open(p/name/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split();
   if not z:continue
   if z[0]=="CROW_SUPPORT_SAMPLE":
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"];kind="state"
   elif z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);kind="contact"
    if 201<=t<=300:
     stats[e]["contacts"]+=1;stats[e]["unloaded"]+=int(float(z[8])+float(z[9])<=1e-7);stats[e]["nonfoot"]+=int(float(z[10])>1e-7);stats[e]["overflow"]+=int(int(z[4])>int(z[5]))
   elif z[0]=="CROW_TOE_V_SUBSTEP":
    t,k,e=map(int,z[1:4]);
    if 201<=t<=300:stats[e]["toe_coverage"]+=1;stats[e]["toe_caps"]+=int(max(abs(float(v)) for v in z[4:])>=11.99)
    continue
   elif z[0]=="CROW_SUBSTEP_ERROR":errors.append(line.strip());continue
   else:continue
   if t<=300:
    allkind="all"+kind;h[e,allkind].update(line.encode());n[e,allkind]+=1
    if t<=200:h[e,kind].update(line.encode());n[e,kind]+=1
 return {"hashes":{(e,k):(n[e,k],h[e,k].hexdigest()) for e,k in h},"stats":stats,"errors":errors}
a=read(A);b=read(B);vel=json.loads((A/"substep-contact-speed-fine.json").read_text())["rows"];base=json.loads((B/"substep-contact-speed-fine.json").read_text())["rows"];toes=json.loads((A/"toe-contact-sequence.json").read_text())["rows"];disp=json.loads((A/"displacement-fine.json").read_text())["rows"];st=json.loads((A/"stance-hallux-digit3.json").read_text())["records"] if (A/"stance-hallux-digit3.json").exists() else [];rows=[]
for e in range(8):
 ratios={s:next(x for x in vel if x["lane"]==e and x["side"]==s)["integral_of_mean_contact_speed_m"]/next(x for x in base if x["lane"]==e and x["side"]==s)["integral_of_mean_contact_speed_m"] for s in ("left","right")};ankle={}
 for s in ("left","right"):
  vals=[x["ankle_late_minus_mid_deg"] for x in st if x["lane"]==e and x["side"]==s];ankle[s]=statistics.median(vals) if vals else None
 t=[x for x in toes if x["lane"]==e];r={"lane":e,"prewalk_equal":{k:a["hashes"][e,k]==b["hashes"][e,k] for k in ("state","contact")},"full_equal":{k:a["hashes"][e,"all"+k]==b["hashes"][e,"all"+k] for k in ("state","contact")},"candidate_counts":dict(a["stats"][e]),"loaded_contact_speed_ratio":ratios,"late_ankle_by_side_deg":ankle,"toe_strict_last":sum(x["front3_strictly_last"] is True for x in t),"front3_unobserved":sum(x["onsets"]["front3"] is None for x in t),"dx_m":disp[e]["candidate"]["dx_m"]};rows.append(r)
r={"rows":rows,"errors":{"candidate":a["errors"],"uniform":b["errors"]},"scope":"Candidate vs same-lane uniform-gait reference. Contact-point integral is not material slip; starts are correlated."};(A/"matched-evaluation.json").write_text(json.dumps(r,indent=2)+"\n")
for x in rows:print(x)
print("errors",r["errors"])
