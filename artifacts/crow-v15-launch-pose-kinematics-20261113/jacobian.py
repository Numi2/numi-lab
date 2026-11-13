from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();p=R/"artifacts/crow-v15-mirrored-leg-isolation-20261113/push-wing-20261126-rebound-mirrored/run.stderr.gz";d=R/"artifacts/crow-v15-launch-pose-kinematics-20261113";exe=R/"build/crow-v15-launch-pose-kinematics-20261113/kinematics";lib=R/"build/crow-v15-mirrored-leg-isolation-20261113";x=None
with gzip.open(p,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   a=json.loads(line.split(" ",1)[1]);
   if a["lane"]==2 and a["step"]==370:x=a;break
assert x
cases=[("reference",[]),("hip+",[(14,.05),(17,.05)]),("knee+",[(15,.05),(18,.05)]),("ankle+",[(16,.05),(19,.05)])];inp=[]
for i,(_,changes) in enumerate(cases):
 q=x["q"][:]
 for j,v in changes:q[j]+=v
 inp.append(str(i)+" 0 "+" ".join(map(str,q+x["v"])))
r=subprocess.run([str(exe)],input="\n".join(inp)+"\n",capture_output=True,text=True,check=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(lib)});data={}
for line in r.stdout.splitlines():
 z=line.split();i=int(z[1]);v=list(map(float,z[3:]));com=v[1:4];body={str(k):v[7+3*j:10+3*j] for j,k in enumerate((11,15,19,20,25,26,27,28,33,34))};data[cases[i][0]]={"com":com,"body":body}
base=data["reference"];out={}
for name in ("hip+","knee+","ankle+"):
 v=data[name];out[name]={str(k):[(v["body"][str(k)][j]-v["com"][j]-(base["body"][str(k)][j]-base["com"][j]))/.05 for j in (0,2)] for k in (11,15,20,26,28,34)}
(d/"jacobian.json").write_text(json.dumps({"derivative_dx_dz_per_radian":out,"scope":"Finite forward differences on unrelaxed fixed-root crouch state. Body COM locations do not determine collider contact, support, momentum or physical viability."},indent=2)+"\n");print(json.dumps(out,indent=2))
