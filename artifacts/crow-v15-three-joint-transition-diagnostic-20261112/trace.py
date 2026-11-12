from pathlib import Path
import gzip,json,math,collections
P=Path("artifacts/crow-v15-three-joint-transition-diagnostic-20261112");D=Path("artifacts/crow-v15-three-joint-lift-screen-20261112");out={}
for label,name in (("sham","sham-20261127-paired"),("candidate","candidate-20261127-rebound")):
 states={};sub={}
 with gzip.open(D/name/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    z=json.loads(l.split(" ",1)[1]);e=z["lane"];t=z["step"]
    if e==5 and (260<=t<=270 or 285<=t<=295):states[t]=z
   elif l.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=l.split();t,k,e=map(int,z[1:4])
    if e==5 and (260<=t<=270 or 285<=t<=295):sub[t,k]=list(map(float,z[6:]))
 assert len(states)==22 and len(sub)==22*16
 rows=[]
 for t,z in sorted(states.items()):
  q=z["q"];v=z["v"];norm=[sum(sub[t,k][2+i] for k in range(16)) for i in range(2)];zero=sum(sub[t,k][2]<=1e-7 and sub[t,k][3]<=1e-7 for k in range(16))
  rows.append({"step":t,"root_z_m":q[2],"root_v_z_m_s":v[2],"root_pitch_rad":math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3])))),"left_carrier":math.sin(2*math.pi*(t*.02+.005)/.5),"left_right_normal_impulses":norm,"zero_normal_substeps":zero,"joints_rad":q[14:20]})
 out[label]=rows
out["scope"]="Retained lane 5 endpoint state and interval normal impulse; cannot directly infer forces at the endpoint or root COM velocity without coordinate verification.";(P/"trace.json").write_text(json.dumps(out,indent=2)+"\n")
for t in (263,264,265,266,267,288,289,290,291,292):
 print("step",t)
 for label in ("sham","candidate"):
  z=next(r for r in out[label] if r["step"]==t);print(label,"z",round(z["root_z_m"],4),"v_z",round(z["root_v_z_m_s"],4),"pitch",round(z["root_pitch_rad"],3),"norm",[round(x,4) for x in z["left_right_normal_impulses"]],"zero",z["zero_normal_substeps"])
