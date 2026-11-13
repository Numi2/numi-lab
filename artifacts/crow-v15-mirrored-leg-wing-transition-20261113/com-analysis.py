from pathlib import Path
import gzip,json,os,subprocess,hashlib,sys
A=Path("artifacts/crow-v15-mirrored-leg-wing-transition-20261113");P=A/sys.argv[1];g=json.loads((P/"gate.json").read_text());wanted={(r["lane"],t) for r in g["rows"] for t in (r["release"],450,500) if t is not None};states={}
with gzip.open(P/"run.stderr.gz","rt") as f:
 for l in f:
  if l.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(l.split(" ",1)[1]);k=x["lane"],x["step"]
   if k in wanted:states[k]=x
exe=Path("build/crow-v15-mirrored-leg-isolation-20261113/kinematics");lib=exe.parent;keys=sorted(states);inp="\n".join(str(i)+" 0 "+" ".join(map(str,states[k]["q"]+states[k]["v"])) for i,k in enumerate(keys))+"\n";r=subprocess.run([str(exe)],input=inp,text=True,capture_output=True,check=True,env={**os.environ,"DYLD_LIBRARY_PATH":str(lib)});data={}
for l in r.stdout.splitlines():
 z=l.split();data[keys[int(z[1])]]=list(map(float,z[3:]))
rows=[]
for r in g["rows"]:
 e=r["lane"];t=r["release"];a=data.get((e,t));b=data.get((e,500));rows.append({"lane":e,"release":t,"release_com_vx_m_s":a[4] if a else None,"release_com_vz_m_s":a[6] if a else None,"com_x_displacement_release_to_500_m":b[1]-a[1] if a and b else None,"com_vx_at_500_m_s":b[4] if b else None})
out={"rows":rows,"kinematics_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"scope":"Posthoc whole-body COM for the physical transition; released COM is not an isolated leg contribution while wings are active."};(P/"com-analysis.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
