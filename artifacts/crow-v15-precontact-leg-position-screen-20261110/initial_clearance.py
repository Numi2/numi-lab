from pathlib import Path
import gzip,json,numpy as np
D=Path('artifacts/crow-v15-precontact-leg-position-screen-20261110')
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(a,A,ha,b,B,hb):
 axes=[*A.T,*B.T]+[np.cross(A[:,i],B[:,j]) for i in range(3) for j in range(3)];vals=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;vals.append(abs((b-a)@u)-sum(abs(A.T@u)*ha)-sum(abs(B.T@u)*hb))
 return float(max(vals))
shapes={};grounds={}
with gzip.open(D/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='groundpose':grounds[(int(v[1]),int(v[2]))]=np.array(list(map(float,v[3:])))
  elif v[0]=='initialshape':
   i,a=int(v[1]),int(v[2]);name=v[3];typ=int(v[4]);z=np.array(list(map(float,v[5:])));assert typ==2 and len(z)==17
   R=rot(z[3:7]);center=z[:3]+R@z[7:10];A=R@rot(z[10:14]);shapes.setdefault((i,a),[]).append((name,center,A,z[14:17]))
out=[]
for key in sorted(shapes):
 g=grounds[key];B=rot(g[3:]);b=g[:3]+B@np.array([0,0,-.055]);rows=[(name,gap(pos,A,h,b,B,np.array([.025,.72,.055]))) for name,pos,A,h in shapes[key]]
 out.append({'case':key[0],'arm':key[1],'shape_count':len(rows),'minimum_m':min(z[1] for z in rows),'minimum_toe_m':min(z[1] for z in rows if '_toe' in z[0]),'minimum_body':min(rows,key=lambda z:z[1])[0],'all_box_gaps_m':dict(rows)})
assert len(out)==4 and all(x['shape_count']==len(shapes[(0,4)]) for x in out)
(D/'initial-clearance.json').write_text(json.dumps(out,indent=2)+'\n')
for a in (4,5):print('arm',a,'minimum all shape gap m',min(x['minimum_m'] for x in out if x['arm']==a),'minimum toe gap m',min(x['minimum_toe_m'] for x in out if x['arm']==a))
print('all four pose sets separated:',all(x['minimum_m']>0 for x in out))
