from pathlib import Path
import collections,gzip,json,numpy as np
A=Path('artifacts/crow-v15-supported-slip-geometry-20261111'); B=Path('artifacts/crow-v15-v1-pause-aligned-descent-20261111')
with gzip.open(B/'route.json.gz','rt') as f:r=json.load(f)['payload'];frames={x['step']:x for x in r['frames']}
names=r['body_names'];stride=r['body_state_stride'];toe=[n for n in names if '_toe' in n];perch=names.index('crow_course_perch'); S={};imp=collections.defaultdict(lambda:[0.0]*21);task={}
with gzip.open(B/'route.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   v=json.loads(line.split(' ',1)[1]);S[v['step']]=v
  elif line.startswith('CROW_PERCH_SUBSTEP '):
   v=line.split();t=int(v[1]);lane=int(v[3]);
   if lane==0:
    for i,x in enumerate(map(float,v[6:])):imp[t][i]+=x
  elif line.startswith('CROW_TASK_STATE '):
   v=line.split()
   if int(v[1])==0:task[int(v[2])]=v[3:]
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(a,b):
 P=rot(a[3:7]);Q=rot(b[3:7]);delta=np.array(b[:3])-np.array(a[:3]);axes=[*P.T,*Q.T]+[np.cross(P[:,i],Q[:,j]) for i in range(3) for j in range(3)];g=[]
 for axis in axes:
  n=np.linalg.norm(axis)
  if n>1e-8:
   u=axis/n;g.append(abs(delta@u)-sum(abs(P.T@u)*[.018,.0025,.0092])-sum(abs(Q.T@u)*[.025,.72,.055]))
 return float(max(g))
out=[]
for t in [2137,2140,2150,2160,2170,2180,2190,2200,2210,2220,2230,2240]:
 pre=frames[t-1];end=frames[t];s=S[t];b=end['body_states'];p=b[perch*stride:perch*stride+7];j=[]
 for i,n in enumerate(toe):
  x=b[names.index(n)*stride:names.index(n)*stride+7];j.append({'name':n,'center_m':x[:3],'dx_from_perch_m':x[0]-p[0],'obb_gap_m':gap(x,p),'normal_impulse_Ns':imp[t][5+i],'signed_x_impulse_Ns':imp[t][13+i]})
 out.append({'step':t,'perch_center_m':p[:3],'root_q':s['q'][:7],'root_v':s['v'][:6],'previous_frame_q_max_error':max(abs(float(x)-float(y)) for x,y in zip(pre['q'],s['q'])),'phase_raw':task.get(t),'foot_up_impulse_Ns':imp[t][:2],'toe':j})
(A/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:
 print(x['step'],'root',*[round(v,4) for v in x['root_q'][:3]],'v',*[round(v,4) for v in x['root_v'][:3]],'p',*[round(v,4) for v in x['perch_center_m']], 'phase', x['phase_raw'], 'foot',*[round(v,4) for v in x['foot_up_impulse_Ns']]);
 for z in x['toe']:
  if 'left' in z['name']: print(' ',z['name'],'dx',round(z['dx_from_perch_m'],4),'gap',round(z['obb_gap_m'],4),'N',round(z['normal_impulse_Ns'],4),'Nx',round(z['signed_x_impulse_Ns'],4))
