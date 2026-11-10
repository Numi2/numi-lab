import math,json,struct
from pathlib import Path
A=Path('artifacts/crow-v15-hybrid-ccd-failure-20261110');d=json.loads((A/'sweep.json').read_text());a,b=d['shapes'];raw=(A/'sweep.bin').read_bytes();gpu=[struct.unpack_from('<4f',raw,128+16*i) for i in range(257)]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def nlerp(p,q,t):
 if dot(p,q)<0:q=[-x for x in q]
 v=[x*(1-t)+y*t for x,y in zip(p,q)];n=math.sqrt(dot(v,v));return [x/n for x in v]
def axes(q):
 a,b,c,d=q;return [[1-2*(b*b+c*c),2*(a*b+c*d),2*(a*c-b*d)],[2*(a*b-c*d),1-2*(a*a+c*c),2*(b*c+a*d)],[2*(a*c+b*d),2*(b*c-a*d),1-2*(a*a+b*b)]]
def pose(s,t):
 p=s['start']['center_radius'][:3];q=s['end']['center_radius'][:3];return [x*(1-t)+y*t for x,y in zip(p,q)],axes(nlerp(s['start']['rotation'],s['end']['rotation'],t))
def gap(t):
 ca,u=pose(a,t);cb,v=pose(b,t);delta=[x-y for x,y in zip(ca,cb)];best=-1e9
 for w in u+v+[cross(x,y) for x in u for y in v]:
  n=math.sqrt(dot(w,w))
  if n<1e-10:continue
  w=[x/n for x in w];g=abs(dot(delta,w))-sum(h*abs(dot(k,w)) for h,k in zip(a['dimensions'][:3],u))-sum(h*abs(dot(k,w)) for h,k in zip(b['dimensions'][:3],v));best=max(best,g)
 return best
rows=[{'index':i,'alpha':g[0],'gpu_signed_distance':g[1],'sat_axis_gap':gap(g[0]),'gpu_status':int(g[2]),'gpu_ok':bool(g[3])} for i,g in enumerate(gpu)];diff=[r for r in rows if (r['gpu_signed_distance']<=0)!=(r['sat_axis_gap']<=0)];res={'rows':rows,'sign_disagreements':len(diff),'first_sign_disagreement':diff[0] if diff else None,'first_sat_overlap':next((r for r in rows if r['sat_axis_gap']<=0),None),'first_gpu_overlap':next((r for r in rows if r['gpu_signed_distance']<=0),None),'note':'SAT axis gap is a separating-axis lower bound and not generally Euclidean distance; compare overlap signs only.'};(A/'sat.json').write_text(json.dumps(res,indent=2)+'\n');print({k:v for k,v in res.items() if k!='rows'})
