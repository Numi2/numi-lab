import json,math,pathlib
import numpy as np
p=pathlib.Path(__file__).parent;x=json.loads((p/'analysis.json').read_text());out=[]
def mul(a,b):return np.polynomial.polynomial.polymul(a,b)
def bern(a):return [float(a[0]),float(a[0]+a[1]/2),float(sum(a))]
def projection(q,v):
 u=q[:3];return float((float(q@q)*v+2*(q[3]*np.cross(u,v)+np.cross(u,np.cross(u,v)) ))[1])
for s in x['scales']:
 a=s['capture'];q0=np.array(a['q0']);q1=np.array(a['q1']);delta=q1-q0
 d=np.array([q0@q0,2*(q0@delta),delta@delta]);displacement=a['y1']-a['y0'];dot=float(q0@q1);guarded=a['y0']-a['plane_y']-a['guard_m']-a['tolerance_m'];assert displacement>=0 and dot>=0
 rows=[]
 for i in range(8):
  v=np.array(a['extents'])*np.array([1 if i&1 else -1,1 if i&2 else -1,1 if i&4 else -1]);r0=projection(q0,v);r2=projection(delta,v);r1=projection(q1,v)-r0-r2;r=np.array([r0,r1,r2])
  gap=bern(r+guarded*d)
  deriv=np.array([r1*d[0]-r0*d[1],2*(r2*d[0]-r0*d[2]),r2*d[1]-r1*d[2]])+displacement*dot*d
  derivative=bern(deriv);start=gap[0]
  rows.append({'corner':i,'gap_bernstein':gap,'derivative_lower_bernstein':derivative,'positive_gap_bound':min(gap)>0,'monotone_from_positive_start':start>0 and min(derivative)>=0})
 out.append({'scale':s['scale'],'dot':dot,'center_displacement':displacement,'all_corners_screen_positive':all(z['positive_gap_bound'] or z['monotone_from_positive_start'] for z in rows),'corners':rows})
(p/'monotonic.json').write_text(json.dumps({'schema':'numi.ccd.monotonic-screen.v1','results':out,'limitation':'Offline FP64 feasibility only; no outward-rounding or production safety proof.'},indent=2)+'\n');print(json.dumps(out,indent=2))
