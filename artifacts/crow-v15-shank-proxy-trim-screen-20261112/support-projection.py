from pathlib import Path
import gzip,json,math,os,statistics,subprocess
p=Path("artifacts/crow-v15-shank-proxy-trim-screen-20261112");paths={"trimmed":p/"candidate-20261127","pair_excluded":Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112/candidate-20261127")};out={}
for name,path in paths.items():
 states={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for l in f:
   if l.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(l.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if t in (1,25,50,75,100):states[e,t]=x
 assert len(states)==40
 inp="".join(f"{e} {t} "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(states.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};q=subprocess.run(["build/crow-v15-rear-toe-attachment-contact-20261112/kinematics-velocity"],input=inp,text=True,capture_output=True,check=True,env=env);com={}
 for line in q.stdout.splitlines():
  z=line.split()
  if z[0]=="com":com[int(z[1]),int(z[2])]=list(map(float,z[4:7]))
 rows=[]
 for (e,t),x in sorted(states.items()):
  pts=[]
  for c in x["contacts"]:
   if c["impulses"][0]<=1e-7:continue
   a,b=c["a_name"],c["b_name"];n=a if b=="locomotion_ground" else b if a=="locomotion_ground" else ""
   if "_foot" in n or "_toe" in n:pts.append(c["point"])
  cx=com[e,t][0];lo=min((v[0] for v in pts),default=None);hi=max((v[0] for v in pts),default=None)
  rows.append({"lane":e,"step":t,"com_x_m":cx,"min_foot_contact_x_m":lo,"max_foot_contact_x_m":hi,"com_minus_rear_contact_m":cx-lo if lo is not None else None,"foot_contact_count":len(pts)})
 out[name]={"rows":rows}
out["scope"]="Native whole-body COM versus minimum world-x accepted foot/toe contact points at selected endpoints. This is a one-dimensional support proxy, not a full contact polygon, center of pressure, friction cone, or proof of instability; forces are dynamic and contact/pose timestamps may differ. Pair-excluded trajectory is diagnostic and cannot establish physical realism.";(p/"support-projection.json").write_text(json.dumps(out,indent=2)+"\n")
for n in paths:
 print(n)
 for t in (1,25,50,75,100):
  v=[r["com_minus_rear_contact_m"] for r in out[n]["rows"] if r["step"]==t and r["com_minus_rear_contact_m"] is not None];print(t,"median_com_minus_rear_contact_m",(statistics.median(v) if v else None),"negative",sum(x<0 for x in v),"samples",len(v))
