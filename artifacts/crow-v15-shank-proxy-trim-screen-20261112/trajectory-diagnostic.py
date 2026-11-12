from pathlib import Path
import gzip,json,statistics,math,collections
p=Path("artifacts/crow-v15-shank-proxy-trim-screen-20261112");paths={"trimmed_all_contacts":p/"candidate-20261127","pair_excluded":Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112/candidate-20261127")};out={}
for name,path in paths.items():
 rows=[];first={};seen=collections.Counter()
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if not 1<=t<=100:continue
   cs=[]
   for c in x["contacts"]:
    a,b=c["a_name"],c["b_name"]
    if c["impulses"][0]>1e-7 and "shank" in a+" "+b and "rear_toe" in a+" "+b:
     part="proximal" if "proximal" in a+" "+b else "distal";cs.append(part);first.setdefault((e,part),t);seen[part]+=1
   if t in (1,25,50,75,100):
    q=x["q"];a,b,c,d=q[3:7];pitch=math.asin(max(-1,min(1,2*(d*b-c*a))));rows.append({"lane":e,"step":t,"root_x_m":q[0],"root_z_m":q[2],"pitch_rad":pitch,"rear_angles_rad":[q[29],q[30],q[37],q[38]],"self_contact_parts":cs})
 out[name]={"first_self_contact_step":{f"{e}_{part}":t for (e,part),t in first.items()},"endpoint_pair_contacts_first_100":dict(seen),"snapshots":rows}
out["scope"]="First 100 accepted endpoint samples, no causal identification. Modified-shank arm and pair-exclusion arm have distinct collision models and diverged trajectories. Joint values are model angles, not biological measurements.";(p/"trajectory-diagnostic.json").write_text(json.dumps(out,indent=2)+"\n")
for k in paths:
 v=out[k];print(k,"contacts",v["endpoint_pair_contacts_first_100"],"first",v["first_self_contact_step"])
 for t in (1,25,50,75,100):
  rs=[r for r in v["snapshots"] if r["step"]==t];print(t,"pitch median",statistics.median(r["pitch_rad"] for r in rs),"rear angles medians",[statistics.median(r["rear_angles_rad"][i] for r in rs) for i in range(4)])
