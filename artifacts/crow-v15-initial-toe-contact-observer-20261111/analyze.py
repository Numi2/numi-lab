from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-initial-toe-contact-observer-20261111");cur=A/"full/run.stderr.gz";ref=Path("artifacts/crow-v15-slow-fold-speed-compensation-20261111/full/run.stderr.gz")
def read(p):
 samples={};caps=[];contact=collections.defaultdict(list)
 with gzip.open(p,"rt") as f:
  for line in f:
   if line.startswith("CROW_FLIGHT_SAMPLE "):
    d=json.loads(line.split(" ",1)[1]);
    if d["lane"]==4 and d["step"]<=120:samples[d["step"]]=d
   elif line.startswith("CROW_TOE_V_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));
    if e==4 and t<=120 and max(map(abs,v))>=11.99:caps.append([t,k,max(range(len(v)),key=lambda i:abs(v[i]))])
   elif line.startswith("CROW_INITIAL_GROUND "):
    z=line.split();t,k,e,a,b=map(int,z[1:6]);n,sep,x,y,zpos,tu,tv=map(float,z[6:13]);contact[t,k].append({"bird_body":b if a==35 else a,"normal_impulse_ns":n,"separation_m":sep,"position_m":[x,y,zpos],"tangent_impulses_ns":[tu,tv]})
 return samples,caps,contact
s,c,contacts=read(cur);rs,rc,_=read(ref);dif=[t for t in sorted(set(s)|set(rs)) if t not in s or t not in rs or any(s[t][k]!=rs[t][k] for k in ("q","v","policy_action"))]
hits=[]
for t,k,j in c:
 near=contacts.get((t,k),[]);hits.append({"step":t,"substep":k,"toe_dof_index":j,"ground_contact_count":len(near),"total_normal_impulse_ns":sum(x["normal_impulse_ns"] for x in near),"contacts_on_affected_digit":[x for x in near if x["bird_body"] in ([25,26] if j<8 else [33,34])]})
out={"lane":4,"sample_steps":sorted(s),"first_sample_mismatch":dif[:1],"caps_match_reference":c==rc,"caps":hits,"scope":"Read-only ground-contact observer on one used initial condition. Endpoint parity at the listed sample steps and identical cap timing do not prove substep equality. Impulse and solver separation are observed contact values, not validation against a physical bird."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
