from pathlib import Path
import gzip,json,math,statistics
p=Path("artifacts/crow-v15-coupled-knee-ankle-screen-20261112")
def load(name):
 out={}
 with gzip.open(p/name/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_TWO_TOE_SUBSTEP "):
    z=line.split();t,k,e=map(int,z[1:4])
    if 201<=t<=300:
     v=list(map(float,z[6:]));out[e,t,k]=(v[2],v[3])
 assert len(out)==12800
 return out
out={}
for label,name in (("sham","sham-20261127-paired"),("candidate","candidate-20261127-rebound")):
 d=load(name);rows=[]
 for side,sgn in ((0,1),(1,-1)):
  samples=[]
  for e in range(8):
   for t in range(201,301):
    carrier=math.sin(2*math.pi*(t*.02+.005)/.5)*sgn
    if carrier>.7:
     for k in range(16):samples.append(d[e,t,k])
  rows.append({"side":"left" if side==0 else "right","samples":len(samples),"own_normal_positive_fraction":sum(x[side]>1e-7 for x in samples)/len(samples),"opposite_normal_positive_fraction":sum(x[1-side]>1e-7 for x in samples)/len(samples),"both_normal_positive_fraction":sum(x[side]>1e-7 and x[1-side]>1e-7 for x in samples)/len(samples),"both_zero_fraction":sum(x[side]<=1e-7 and x[1-side]<=1e-7 for x in samples)/len(samples)})
 out[label]=rows
out["scope"]="Descriptive, using interval command phase and each substep normal impulse; support gate is an accepted-state flag rather than these exact substep impulses. Not proof of cause."
(p/"load-phase.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
