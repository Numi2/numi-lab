from pathlib import Path
import gzip,json,sys,numpy as np
p=Path(sys.argv[1]); release=int(sys.argv[2]); output=Path(sys.argv[3]);
with gzip.open(p,'rt') as f:frames=json.load(f)['payload']['frames']
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def sat(a,qa,ha,b,qb,hb):
 A=rot(qa);B=rot(qb);d=b-a;axes=[*A.T,*B.T]+[np.cross(A[:,i],B[:,j]) for i in range(3) for j in range(3)];gaps=[]
 for axis in axes:
  n=np.linalg.norm(axis)
  if n<1e-8:continue
  u=axis/n;gaps.append(abs(np.dot(d,u))-np.sum(np.abs(A.T@u)*ha)-np.sum(np.abs(B.T@u)*hb))
 return float(max(gaps))
def calc(frame):
 body=np.asarray(frame['body_states'],dtype=float).reshape(-1,13);p=body[30];g=[]
 for i in range(17,25):
  t=body[i];g.append(sat(t[:3],t[3:7],[.018,.0025,.0092],p[:3],p[3:7],[.025,.72,.055]))
 return g
result={}
for label,lo,hi in [('before',release-100,release-1),('early_after',release+20,release+119),('late_after',release+150,release+249)]:
 rows=[calc(z) for z in frames if lo<=z['step']<=hi];x=np.array(rows);result[label]={'frames':len(rows),'all_eight_strictly_separated_frames':int(np.sum(np.all(x>1e-5,axis=1))) if len(rows) else 0,'minimum_separation_axis_gap_m':float(np.min(x)) if len(rows) else None,'median_per_toe_gap_m':np.median(x,axis=0).tolist() if len(rows) else None}
r={'replay':str(p),'release_step':release,'windows':result,'method':'SAT separating-axis maximum gap for 8 toe OBBs versus perch OBB from accepted body transforms; positive gap establishes separation at that endpoint. Shape half-extents from original v15 research host. No inference about substeps.'};output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
