from pathlib import Path
import gzip,json,os,subprocess,hashlib
R=Path.cwd();p=R/"artifacts/crow-v15-mirrored-leg-isolation-20261113/push-wing-20261126-rebound-mirrored/run.stderr.gz";d=R/"artifacts/crow-v15-launch-pose-kinematics-20261113";exe=R/"build/crow-v15-launch-pose-kinematics-20261113/kinematics";lib=R/"build/crow-v15-mirrored-leg-isolation-20261113";x=None
with gzip.open(p,"rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   a=json.loads(line.split(" ",1)[1]);
   if a["lane"]==2 and a["step"]==370:x=a;break
assert x is not None
values=[(0,0.),(1,-.10),(2,.10)];inp=[]
for i,delta in values:
 q=x["q"][:];q[14]+=delta;q[17]+=delta;inp.append(str(i)+" 0 "+" ".join(map(str,q+x["v"])))
r=subprocess.run([str(exe)],input="\n".join(inp)+"\n",capture_output=True,text=True,check=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(lib)});rows=[]
for l in r.stdout.splitlines():
 z=l.split();i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==37;com=v[1:4];feet={str(body):v[7+3*j:10+3*j] for j,body in enumerate((11,15,19,20,25,26,27,28,33,34))};rows.append({"delta_bilateral_hip_rad":values[i][1],"com_xyz_m":com,"body_com_xyz_m":feet,"left_foot_minus_com_x_m":feet["11"][0]-com[0],"right_foot_minus_com_x_m":feet["15"][0]-com[0]})
out={"rows":rows,"scope":"Offline FK sensitivity at one retained crouched state, manually perturbed coordinates, no dynamics or contact solving. Body COMs are not collider contact points. Used-seed research model; not biological calibration.","source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"kinematics_sha256":hashlib.sha256(exe.read_bytes()).hexdigest()};(d/"analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(rows,indent=2))
