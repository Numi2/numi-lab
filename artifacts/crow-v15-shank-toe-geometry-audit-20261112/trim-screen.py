from pathlib import Path
import gzip,json,math,os,subprocess,statistics,collections
p=Path("artifacts/crow-v15-shank-toe-geometry-audit-20261112");path=Path("artifacts/crow-v15-rear-toe-self-contact-control-20261112/candidate-20261127")
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def axes(q):
 x,y,z,w=q;return [(1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)),(2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)),(2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y))]
def sat(a,b,ha,hb,offset):
 A=axes(a[3:7]);B=axes(b[3:7]);d=[b[i]-(a[i]+offset*A[2][i]) for i in range(3)];best=-1e30
 for L in A+B+[cross(u,v) for u in A for v in B]:
  n=math.sqrt(dot(L,L))
  if n<1e-8:continue
  L=[x/n for x in L];gap=abs(dot(d,L))-sum(ha[i]*abs(dot(A[i],L))+hb[i]*abs(dot(B[i],L)) for i in range(3));best=max(best,gap)
 return best
state={}
with gzip.open(path/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);e,t=x["lane"],x["step"]
   if 1<=t<=300:state[e,t]=x
assert len(state)==2400
inp="".join(f"{e} {t} "+" ".join(map(str,x["q"]+x["v"]))+"\n" for (e,t),x in sorted(state.items()));env={**os.environ,"DYLD_LIBRARY_PATH":str(Path("build/crow-v15-hip-roll-journey-integration-20261111").resolve())};proc=subprocess.run(["build/crow-v15-rear-toe-attachment-contact-20261112/kinematics-velocity"],input=inp,text=True,capture_output=True,env=env,check=True);body={}
for line in proc.stdout.splitlines():
 z=line.split()
 if z[0]=="body":body[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
results={}
for mm in (0,2,4,6,8,10):
 trim=mm*.001;out=collections.defaultdict(list);worst={}
 for (e,t) in state:
  for side in ("left","right"):
   sh=body[e,t,"crow_"+side+"_shank"]
   for digit in ("rear_toe","front_toe_1","front_toe_2","front_toe_3"):
    for seg in ("proximal","distal"):
     toe=body[e,t,"crow_"+side+"_"+digit+"_"+seg];gap=sat(sh,toe,(.01265,.01035,.031625-trim/2),(.009,.0025,.0092),trim/2);key=digit+"_"+seg;out[key].append(gap)
     if key not in worst or gap<worst[key]["gap_m"]:worst[key]={"lane":e,"step":t,"side":side,"gap_m":gap}
 results[str(mm)]={key:{"min_separation_m":min(vals),"median_separation_m":statistics.median(vals),"overlap_count":sum(x<0 for x in vals),"within_1mm_count":sum(x<.001 for x in vals),"count":len(vals),"worst":worst[key]} for key,vals in out.items()}
result={"trials":results,"scope":"Counterfactual box SAT evaluated against fixed kinematics from the two-exclusion diagnostic, not rerun dynamics. Each integer is mm removed from lower shank box only, its upper end held fixed. All 8 toes compared per side across 300 endpoints, not continuous substeps. This geometry is an engineering screen, not measured crow morphology; contact margin, perch grasp and flight are not qualified."};(p/"trim-screen.json").write_text(json.dumps(result,indent=2)+"\n")
for mm in results:print(mm,{k:(round(v["min_separation_m"]*1000,3),v["overlap_count"],v["within_1mm_count"]) for k,v in results[mm].items()})
