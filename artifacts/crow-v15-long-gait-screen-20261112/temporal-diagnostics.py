from pathlib import Path
import gzip,json,collections,statistics
p=Path("artifacts/crow-v15-long-gait-screen-20261112")
paths={"sham":p/"sham-20261127-long","candidate":p/"candidate-20261127-rebound-long"}
allarms={}
for name,path in paths.items():
 q={};contacts={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(line.split(" ",1)[1]);e,t=z["lane"],z["step"]
    if 200<=t<=500:q[e,t]=z["q"]
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 201<=t<=500:
     v=list(map(float,z[6:]));contacts[e,t,k]=v
 assert len(q)==8*301 and len(contacts)==8*300*16
 rows=[]
 for e in range(8):
  sections=[];flags=[]
  for start in (201,301,401):
   end=start+99; displacement=q[e,end][0]-q[e,start-1][0]
   no=sum(contacts[e,t,k][2]<=1e-7 and contacts[e,t,k][3]<=1e-7 for t in range(start,end+1) for k in range(16))
   sections.append({"steps":[start,end],"root_displacement_m":displacement,"mean_root_speed_m_s":displacement/2,"unsupported_substeps":no})
  for t in range(201,501):
   for k in range(16):
    x=contacts[e,t,k];flags.append(x[2]<=1e-7 and x[3]<=1e-7)
  bouts=[];start=None
  for i,x in enumerate(flags+[False]):
   if x and start is None:start=i
   if not x and start is not None:bouts.append({"first_step":201+start//16,"length_substeps":i-start,"duration_s":(i-start)*.00125});start=None
  rows.append({"lane":e,"two_second_sections":sections,"unsupported_bouts":len(bouts),"longest_unsupported_s":max((b["duration_s"] for b in bouts),default=0),"first_unsupported_step":bouts[0]["first_step"] if bouts else None})
 allarms[name]=rows
out={"arms":allarms,"scope":"Post-hoc diagnostic, ground-contact normal impulse <=1e-7 on both feet is called unsupported; threshold can include short contact fluctuations, not proof of flight or a fall. Root x displacement per 2 seconds is not crow measured speed."}
(p/"temporal-diagnostics.json").write_text(json.dumps(out,indent=2)+"\n")
for n,rows in allarms.items():
 print(n)
 for r in rows:print(r["lane"],"speeds",[round(s["mean_root_speed_m_s"],4) for s in r["two_second_sections"]],"unsupported",[s["unsupported_substeps"] for s in r["two_second_sections"]],"first",r["first_unsupported_step"],"longest ms",round(1000*r["longest_unsupported_s"],2))
