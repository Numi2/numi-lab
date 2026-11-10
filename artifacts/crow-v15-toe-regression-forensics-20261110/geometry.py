import gzip,json,math
from pathlib import Path
A=Path("artifacts/crow-v15-toe-regression-forensics-20261110");x=json.load(gzip.open("artifacts/crow-v15-guide-midheight-20261110/development-reserved-final-j.json.gz","rt"))["payload"]
def axes(q):
 a,b,c,d=q;return [[1-2*(b*b+c*c),2*(a*b+c*d),2*(a*c-b*d)],[2*(a*b-c*d),1-2*(a*a+c*c),2*(b*c+a*d)],[2*(a*c+b*d),2*(b*c-a*d),1-2*(a*a+b*b)]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def separation(toe,perch):
 u=axes(toe[3:7]);v=axes(perch[3:7]);delta=[toe[i]-perch[i] for i in range(3)];best=-1e9
 for w in u+v+[cross(a,b) for a in u for b in v]:
  n=math.sqrt(dot(w,w))
  if n<1e-8:continue
  w=[z/n for z in w];gap=abs(dot(delta,w))-sum(h*abs(dot(k,w)) for h,k in zip((.018,.0025,.0092),u))-sum(h*abs(dot(k,w)) for h,k in zip((.025,.72,.055),v));best=max(best,gap)
 return best
rows=[]
for t in range(4389,4398):
 b=x["frames"][t-1]["body_states"];perch=b[390:403];g=[separation(b[13*i:13*i+13],perch) for i in range(17,25)];rows.append({"step":t,"minimum_separating_axis_gap_m":min(g),"front_toe_minimum_m":min(g[:3]+g[4:7])})
(A/"geometry.json").write_text(json.dumps(rows,indent=2)+"\n")
for z in rows:print(z["step"],"minimum separating-axis gap mm",round(z["minimum_separating_axis_gap_m"]*1000,3))
