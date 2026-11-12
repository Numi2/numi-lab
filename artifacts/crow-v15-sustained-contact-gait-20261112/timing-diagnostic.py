from pathlib import Path
import collections,gzip,json,math
A=Path("artifacts/crow-v15-sustained-contact-gait-20261112");out={}
for variant in ("control","hip"):
 D=A/("candidate-20261126-rebound-"+variant);rows=collections.defaultdict(lambda:collections.defaultdict(lambda:{"unloaded":0,"toe_caps":[],"peak_tilt":0.0,"min_z":1e9,"max_z":-1e9}))
 with gzip.open(D/"run.stderr.gz","rt") as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=="CROW_TWO_TOE_SUBSTEP":
    t,k,e=map(int,z[1:4]);
    if 201<=t<=500 and float(z[8])+float(z[9])<=1e-7:rows[e][(t-201)//50+4]["unloaded"]+=1
   elif z[0]=="CROW_TOE_V_SUBSTEP":
    t,k,e=map(int,z[1:4]);
    if 201<=t<=500:
     for i,x in enumerate(z[4:]):
      if abs(float(x))>=11.99:rows[e][(t-201)//50+4]["toe_caps"].append({"step":t,"substep":k,"toe_velocity_index":i,"value":float(x)})
   elif z[0]=="CROW_SUPPORT_SAMPLE":
    x=json.loads(line.split(" ",1)[1]);t,e=x["step"],x["lane"];
    if 201<=t<=500:
     q=x["q"];r=rows[e][(t-201)//50+4];r["peak_tilt"]=max(r["peak_tilt"],math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2)))));r["min_z"]=min(r["min_z"],q[2]);r["max_z"]=max(r["max_z"],q[2])
 out[variant]={e:dict(v) for e,v in rows.items()}
(A/"timing-diagnostic.json").write_text(json.dumps(out,indent=2)+"\n")
for v in out:
 print(v)
 for e,r in out[v].items(): print(e,[(k,x["unloaded"],x["toe_caps"],round(x["peak_tilt"],3)) for k,x in r.items()])
