from pathlib import Path
import gzip,json,math,sys,numpy as np
A=Path('artifacts/crow-v15-retry-climb-20261110');name=sys.argv[1];source=Path(sys.argv[2]) if len(sys.argv)>2 else A;state={};support={};release=None;last=-1
with gzip.open(source/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_TASK_STATE '):
   z=line.split();t=int(z[2]);
   if t<last:break
   last=t;state[t]={'left':float(z[5]),'right':float(z[6]),'phase':float(z[7])}
   if release is None and float(z[7])>=2.5:release=t
  elif line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);support[x['step']]=x
if release is None:raise SystemExit('No release phase')
with gzip.open(source/(name+'.json.gz'),'rt') as f:frames=json.load(f)['payload']['frames']
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def sat(a,qa,b,qb):
 aa=rot(qa);bb=rot(qb);d=b-a;axes=[*aa.T,*bb.T]+[np.cross(aa[:,i],bb[:,j]) for i in range(3) for j in range(3)];g=[]
 for u in axes:
  n=np.linalg.norm(u)
  if n<1e-8:continue
  u=u/n;g.append(abs(np.dot(d,u))-np.sum(np.abs(aa.T@u)*[.018,.0025,.0092])-np.sum(np.abs(bb.T@u)*[.025,.72,.055]))
 return max(g)
def longest(seq):
 m=n=0
 for x in seq:
  n=n+1 if x else 0;m=max(m,n)
 return m
rows=[]
for z in frames:
 t=z['step']
 if not(release<=t<=release+160) or (t+1) not in support or t not in state:continue
 body=np.asarray(z['body_states']).reshape(-1,13);perch=body[30];gap=[]
 for i in range(17,25):toe=body[i];gap.append(sat(toe[:3],toe[3:7],perch[:3],perch[3:7]))
 impulses={'left':0.,'right':0.};upward={'left':0.,'right':0.}
 for c in support[t+1]['contacts']:
  a,b=c['a_name'],c['b_name'];n=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
  if n:
   for side in impulses:
    if '_'+side+'_' in n and ('_foot' in n or '_toe' in n):
     upward[side]+=max((-1. if n==a else 1.)*c['normal'][2]*c['impulses'][0],0.)
     if '_toe' in n:impulses[side]+=abs(c['impulses'][0])
 rows.append({'step':t,'gap_left':min(gap[:4]),'gap_right':min(gap[4:]),'impulses':impulses,'upward':upward,'loads':state[t]})
result={'release_step':release,'samples':len(rows),'sides':{}}
for side,other in [('left','right'),('right','left')]:
 window=[x for x in rows if (release+10<=x['step']<=release+60 if side=='left' else release+70<=x['step']<=release+120)];good=[x['gap_'+side]>.001 and x['impulses'][side]<=1e-8 and x['loads'][other]>.1 and x['upward'][other]>.00005 for x in window];result['sides'][side]={'window_samples':len(window),'qualifying_endpoints':int(sum(good)),'longest_consecutive_endpoints':longest(good),'max_min_toe_gap_m':float(max(x['gap_'+side] for x in window)) if window else None,'first_qualifying_step':next((x['step'] for x,g in zip(window,good) if g),None)}
result['limitations']='Accepted endpoints with support trace sample t+1 aligned to replay/task-state t; strict 1 mm separating-axis gap and at most 1e-8 Ns toe impulse on the lifted side; opposite load is measured at accepted task state and requires upward contact impulse above 0.00005 N s at the same endpoint. Windows follow the commanded left and right lift intervals; does not prove clearance at every substep.';(A/(name+'-step-audit.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
