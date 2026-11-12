from pathlib import Path
import gzip,json,math,os,subprocess,statistics,collections
p=Path("artifacts/crow-v15-shank-toe-geometry-audit-20261112");paths={"reattached":Path("artifacts/crow-v15-rear-toe-attachment-contact-20261112/candidate-20261127"),"pair_excluded_diagnostic":Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112/candidate-20261127")}
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def axes(q):
 x,y,z,w=q;return [(1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)),(2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)),(2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y))]
def sat(a,b,ha,hb):
 A=axes(a[3:7]);B=axes(b[3:7]);d=[b[i]-a[i] for i in range(3)];ls=A+B+[cross(u,v) for u in A for v in B];best=(-1e30,None)
 for L in ls:
  n=math.sqrt(dot(L,L))
  if n<1e-8:continue
  L=[x/n for x in L];gap=abs(dot(d,L))-sum(ha[i]*abs(dot(A[i],L))+hb[i]*abs(dot(B[i],L)) for i in range(3))
  if gap>best[0]:best=(gap,L)
 return best
out={}
for label,path in paths.items():
 state={}
 with gzip.open(path/"run.stderr.gz","rt") as f:
  for line in f:
   if line.startswith("CROW_SUPPORT_SAMPLE "):
    x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
    if 1<=t<=100:state[e,t]=x
 assert len(state)==800
 inp="".join(f"{e} {t} "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(state.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};proc=subprocess.run(["build/crow-v15-rear-toe-attachment-contact-20261112/kinematics-velocity"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
 for line in proc.stdout.splitlines():
  z=line.split()
  if z[0]=="body":body[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
 rows=[]
 for (e,t),x in sorted(state.items()):
  for side in ("left","right"):
   sn="crow_"+side+"_shank";tn="crow_"+side+"_rear_toe_proximal";gap,axis=sat(body[e,t,sn],body[e,t,tn],(.01265,.01035,.031625),(.009,.0025,.0092));react=[c for c in x["contacts"] if {c["a_name"],c["b_name"]}=={sn,tn} and c["impulses"][0]>1e-7];rows.append({"lane":e,"step":t,"side":side,"box_separation_m":gap,"sat_axis":axis,"accepted_contacts":len(react),"reported_min_separation_m":min((c["separation"] for c in react),default=None)})
 gaps=sorted(x["box_separation_m"] for x in rows);out[label]={"count":len(rows),"overlap_count":sum(x<0 for x in gaps),"max_overlap_m":-gaps[0],"median_separation_m":statistics.median(gaps),"max_separation_m":gaps[-1],"rows":rows}
out["scope"]="Separating-axis test on shank and proximal rear-toe boxes at accepted endpoints, using native FK, identity local shape rotations and dimensions from the model source. Negative is geometric overlap, positive separation; solver contact margins can produce accepted contact at positive separation. Numerical box overlap is not direct tissue anatomy or proof that narrowing a collider is valid. Only first 100 steps of each development arm."
(p/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
for k in paths:print(k,{n:v for n,v in out[k].items() if n!="rows"})
