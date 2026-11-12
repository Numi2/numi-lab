from pathlib import Path
import collections,gzip,json
A=Path("artifacts/crow-v15-gait-support-transition-audit-20261112");B=Path("artifacts/crow-v15-matched-gait-second-seed-20261112");out={}
for name,path in (("sham",B/"sham-20261127-dev"),("candidate",B/"candidate-20261127-rebound")):
 states={};phase={};sub={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);t=x["step"]
    if x["lane"]==2 and 201<=t<=300:states[t]=x
   elif line.startswith("CROW_PHASE_SAMPLE "):
    z=line.split();t,e,w=map(int,z[1:4])
    if e==2 and 201<=t<=300:phase[t]=w
   elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if e==2 and 201<=t<=300:sub[t,k]=list(map(float,z[6:]))
 assert len(states)==100 and len(phase)==100 and len(sub)==1600
 offs=[(t,k) for (t,k),v in sorted(sub.items()) if v[2]<=1e-7 and v[3]<=1e-7]
 longest=cur=0
 for t in range(201,301):
  for k in range(16):
   cur=cur+1 if sub[t,k][2]<=1e-7 and sub[t,k][3]<=1e-7 else 0;longest=max(longest,cur)
 timeline=[]
 for t in range(216,228):
  x=states[t];v=x["v"];q=x["q"];data=[sub[t,k] for k in range(16)];timeline.append({"step":t,"last_status_support_bits":(phase[t]>>20)&7,"root_x_m":q[0],"root_z_m":q[2],"root_vx_m_s":v[0],"root_vz_m_s":v[2],"left_normal_impulse_ns":sum(r[2] for r in data),"right_normal_impulse_ns":sum(r[3] for r in data),"both_unloaded_substeps":sum(r[2]<=1e-7 and r[3]<=1e-7 for r in data)})
 out[name]={"lane":2,"all_window_unloaded_substeps":len(offs),"first_unloaded_substep":offs[0] if offs else None,"longest_consecutive_unloaded_substeps":longest,"min_vz_216_227_m_s":min(r["root_vz_m_s"] for r in timeline),"max_vz_216_227_m_s":max(r["root_vz_m_s"] for r in timeline),"timeline":timeline}
out["scope"]="Accepted endpoint phase and state observations with all contact substeps, one used development lane; phase bits indicate control flags, not a direct measure of force. Temporal association does not prove a cause."
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
