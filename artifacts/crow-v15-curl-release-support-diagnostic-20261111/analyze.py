from pathlib import Path
import collections,gzip,json
p=Path("artifacts/crow-v15-autonomous-curl-release-control-20261111/autonomous-20261126")
a=Path("artifacts/crow-v15-curl-release-support-diagnostic-20261111")
sums=collections.defaultdict(lambda:[0.]*7);cov=collections.Counter()
with gzip.open(p/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4])
   if e==5 and 3600<=t<=3800:cov[t,k]+=1
  elif line.startswith("CROW_SCENE_CONTACT "):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if e!=5 or not 3600<=t<=3800 or 40 not in (ba,bb):continue
   body=bb if ba==40 else ba;sgn=1 if ba==40 else -1
   normal,t1,t2=map(float,z[6:9]);nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]);up=sgn*(nz*normal+tz*t1+(nx*ty-ny*tx)*t2)
   side=0 if body==11 or 19<=body<=26 else 1 if body==15 or 27<=body<=34 else -1
   v=sums[t]
   if side<0:v[6]+=normal;continue
   v[side]+=up
   if body>=19:
    digit=((body-19)%8)//2;v[2+side*2+(1 if digit==3 else 0)]+=sgn*nx*normal
threshold=.5*.45006*9.81*.02
rows=[]
for t in range(3600,3801):
 left,right,lf,lr,rf,rr,other=sums[t]
 rows.append({"step":t,"left_up":left,"right_up":right,"left_front_x":lf,"left_rear_x":lr,"right_front_x":rf,"right_rear_x":rr,"nonfoot_normal":other,
 "halfweight":left+right>=threshold,"bilateral_opposition":left>0 and right>0 and left+right>=threshold and lf>1e-6 and lr< -1e-6 and rf>1e-6 and rr< -1e-6,"complete_substeps":all(cov[t,k]==1 for k in range(16))})
last=rows[-29:]
out={"lane":5,"threshold_Ns":threshold,"last_29":{"halfweight_intervals":sum(x["halfweight"] for x in last),"bilateral_opposition_intervals":sum(x["bilateral_opposition"] for x in last),"nonfoot_contact_intervals":sum(x["nonfoot_normal"]>1e-7 for x in last),"missing_substeps":sum(not x["complete_substeps"] for x in last),"rows":last},"scope":"Summed substep contact impulses against estimated perch geometry; sign opposition is not biological validation or proof of continuously stable support. One used development seed."}
(a/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out["last_29"].items() if k!="rows"},indent=2))
