import copy,json,math
from collections import Counter
from pathlib import Path
b=json.loads(Path('/Users/numi/mixed-slip-validation-20261108/control.json').read_text())['cases']
c=json.loads(Path('control-with-sweeps.json').read_text())['cases'];t=json.loads(Path('candidate-trace.json').read_text())['cases'];fails=json.loads(Path('candidate-2.json').read_text())['cases']
assert len(b)==len(c)==len(t)==len(fails)==60
counts=Counter();bad=[];max_v=0.;max_rod=0.;max_imp=0.;angle=0.;max_dot=0.;max_velerr=0.
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
for base,plain,traced,limited in zip(b,c,t,fails):
 key=(traced['mu'],traced['initial_vn'],traced['dir'])
 x=copy.deepcopy(plain);y=copy.deepcopy(traced)
 for v in (x,y):
  for row in v['rows']:row.pop('post_relative')
 if x!=y:bad.append(['trace_changes_output',key])
 if not traced['accepted'] or traced['status'] or not traced['replay'] or not traced['mixed_island'] or traced['normal_residual']>.002 or traced['candidate_momentum_error_uninterpretable_on_rejection']>2e-5:bad.append(['invalid_adaptive',key])
 if not limited['accepted']:
  counts['rejected_cap2']+=1
  if limited['status']!=11 or not limited['restored'] or not limited['replay']:bad.append(['bad_rollback',key])
 else:counts['accepted_cap2']+=1
 counts['sweeps_'+str(traced['performed_sweeps'])]+=1
 for row in traced['rows']:
  n,u,v=row['impulse'];vu,vv=row['post_relative'][1:];mag=math.hypot(u,v);speed=math.hypot(vu,vv);mu=traced['mu'];limit=mu*n
  if n < -1e-8 or mag>limit+1e-7:bad.append(['cone',key])
  tangdot=u*vu+v*vv;max_dot=max(max_dot,tangdot)
  if tangdot>1e-6:bad.append(['slip_work_sign',key])
  if mu>0 and speed>1e-3 and mag>1e-5:
   theta=math.acos(max(-1.,min(1.,-tangdot/(mag*speed))));angle=max(angle,theta)
   if theta>.01 or abs(mag-limit)/max(limit,1e-30)>.01:bad.append(['sliding',key,theta,speed,mag,limit])
  if not row['flags']&8 and row['body_a']==0 and row['body_b']==1:
   state=traced['final_v'];w=cross(state[3:6],row['anchor_b']);point=[state[i]+w[i] for i in range(3)];nvec=row['normal'];tan=row['tangent'];vvec=cross(nvec,tan)
   err=max(abs(dot(point,d)-value) for d,value in zip((nvec,tan,vvec),row['post_relative']));max_velerr=max(max_velerr,err)
   if err>1e-5:bad.append(['rigid_velocity',key,err])
 # Compare the adaptive two-request arm to same physical case at fixed 64 sweeps.
 if traced['iterations']==2:
  ref=next(a for a in b if a['iterations']==64 and (a['mu'],a['initial_vn'],a['dir'])==key)
  max_v=max(max_v,max(abs(x-y) for x,y in zip(traced['final_v'],ref['final_v'])))
  max_rod=max(max_rod,max(abs(x-y) for u,v in zip(traced['rod_v'],ref['rod_v']) for x,y in zip(u,v)))
  max_imp=max(max_imp,max(abs(x-y) for r1,r2 in zip(traced['rows'],ref['rows']) for x,y in zip(r1['impulse'],r2['impulse'])))
out={'cases':len(c),'bad_count':len(bad),'bad':bad,'counts':dict(counts),'max_early_vs_64_generalized_velocity_m_s':max_v,'max_early_vs_64_rod_velocity_m_s':max_rod,'max_early_vs_64_row_impulse_Ns':max_imp,'max_sliding_angle_rad':angle,'max_positive_final_tangent_work_J':max_dot,'max_independent_sphere_velocity_error_m_s':max_velerr}
Path('analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
