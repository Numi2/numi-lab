from pathlib import Path
from collections import defaultdict
import math,json
D=Path('artifacts/crow-v15-staged-toe-clearance-diagnostic-20261111');m=defaultdict(list)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def axes(q):
 x,y,z,w=q;M=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];return [[M[i][j] for i in range(3)] for j in range(3)]
def gap(obj,half):
 P=axes(obj[3:]);Q=[[1,0,0],[0,1,0],[0,0,1]];d=[-obj[0],-obj[1],-.055-obj[2]];out=[]
 for a in P+Q+[cross(x,y) for x in P for y in Q]:
  norm=math.sqrt(dot(a,a))
  if norm<1e-8:continue
  u=[x/norm for x in a];out.append(abs(dot(d,u))-sum(abs(dot(x,u))*h for x,h in zip(P,half))-sum(abs(dot(x,u))*h for x,h in zip(Q,[.025,.72,.055])))
 return max(out)
for line in (D/'poses.txt').read_text().splitlines():
 a=line.split();t=int(a[2]);n=a[3];v=list(map(float,a[4:]));h=[.01,.01725,.0092] if n=='crow_left_foot' else [.009,.0025,.0092];m[t].append({'body':n,'gap_m':gap(v,h)})
rows=[{'step_zero_based':t,'nearest_shapes':sorted(v,key=lambda x:x['gap_m'])[:4]} for t,v in sorted(m.items())];out={'source':'retained lane 4 poses from crow-v15-hip-roll-staged-release-20261111','rows':rows,'limitations':'Box separating-axis distance lower bound at five endpoint samples, not substep penetration or an exact curved biological claw shape.'};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
