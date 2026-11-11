from pathlib import Path
import gzip,json,subprocess
A=Path(__file__).parent
with gzip.open("artifacts/crow-v15-two-segment-gait-lift-screen-20261111/stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1])
   if x["lane"]==1 and x["step"]==101:break
 else:raise RuntimeError("missing retained pose")
q=x["q"];v=x["v"];b=Path("build/crow-v15-two-segment-gait-geometry-20261111");lines=[]
for i in range(7,23):
 for sign in (-1,1):
  z=q.copy();z[i]+=.1*sign;lines.append(f"{i} {sign} "+" ".join(map(str,z+v)))
p=b/"sensitivity.txt";p.write_text("\n".join(lines)+"\n");output=subprocess.check_output([str(b/"probe"),str(p)],text=True);r={}
for line in output.splitlines():
 a=line.split()
 if a[2] in ("crow_left_foot","crow_right_foot"):r[int(a[0]),int(a[1]),a[2]]=tuple(map(float,a[3:6]))
rows=[]
for i in range(7,23):
 for name in ("crow_left_foot","crow_right_foot"):
  plus=r[i,1,name];minus=r[i,-1,name];dx=(plus[0]-minus[0])/.2;dz=(plus[2]-minus[2])/.2
  if abs(dx)+abs(dz)>1e-5:rows.append({"q_index":i,"body":name,"dx_per_rad":dx,"dz_per_rad":dz})
report={"source":"retained lift screen lane 1 control interval 101","method":"independent native FP64 forward kinematics; central difference +/-0.1 rad at fixed root and other joints","rows":rows,"limitation":"static local sensitivity; not a simulated step or evidence of walking"}
(A/"kinematic-sensitivity.json").write_text(json.dumps(report,indent=2)+"\n")
