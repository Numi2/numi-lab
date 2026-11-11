from pathlib import Path
import gzip,json,math
A=Path("artifacts/crow-v15-support-load-gait-screen-20261111");base={20261126:Path("artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126"),20261127:Path("artifacts/crow-v15-walk-seed27-reference-20261111/autonomous-20261127")}
def evaluate(p):
 b={i:[] for i in range(8)};last=set()
 with gzip.open(p/"run.stderr.gz","rt") as f:
  for line in f:
   if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
   z=json.loads(line.split(" ",1)[1]);e=z["lane"];t=z["step"]
   if 201<=t<=300:b[e].append(z["q"])
   if t==300:last.add(e)
   if len(last)==8:break
 assert all(len(v)==100 for v in b.values())
 out={}
 for e,qs in b.items():
  angles=[]
  for q in qs:
   x,y,z,w=q[3:7];angles.append((math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))))
  out[e]={"roll_span_rad":max(a[0] for a in angles)-min(a[0] for a in angles),"pitch_span_rad":max(a[1] for a in angles)-min(a[1] for a in angles),"height_span_m":max(q[2] for q in qs)-min(q[2] for q in qs),"minimum_height_m":min(q[2] for q in qs),"hip_spans_rad":[max(q[i] for q in qs)-min(q[i] for q in qs) for i in (14,17)],"ankle_spans_rad":[max(q[i] for q in qs)-min(q[i] for q in qs) for i in (16,19)]}
 return out
rows=[]
for seed in (20261126,20261127):
 x=evaluate(base[seed]);y=evaluate(A/f"autonomous-{seed}")
 for e in range(8):rows.append({"seed":seed,"lane":e,"baseline":x[e],"candidate":y[e]})
(A/"body-comparison.json").write_text(json.dumps({"rows":rows,"scope":"Rigid root rotation and displacement from control endpoints only; no flexible torso geometry or biological validation."},indent=2)+"\n")
