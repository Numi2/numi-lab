from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-front-grip-floor-20261110')
def pct(x,p):
 x=sorted(x);return x[min(len(x)-1,math.ceil(p*len(x))-1)] if x else None
def runs(seq):
 out=[];k=0
 for v in seq+[False]:
  if v:k+=1
  elif k:out.append(k);k=0
 return out
result={}
for name in ('j-grip-original','i-grip-original'):
 gait=[];state=[];last=-1;nonfinite=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_GAIT '):
    x=json.loads(line.split(' ',1)[1]);gait.append(x)
   elif line.startswith('CROW_TASK_STATE '):
    v=line.split();t=int(v[2]);
    if t<last:break
    last=t
    if any(not math.isfinite(float(v[k])) for k in [4,5,6,7,8,9]):nonfinite.append(t)
    state.append({'step':t,'packed':int(v[3]),'left':float(v[5]),'right':float(v[6]),'phase':float(v[7])})
 xs=sorted([x for x in gait if 100<=x['step']<=600],key=lambda x:x['step']);sides=[]
 for i in [0,1]:
  stance=[x['sides'][i] for x in xs if x['sides'][i]['impulse_Ns']>.001];swing=[x['sides'][i]['impulse_Ns']<=.001 and x['sides'][1-i]['impulse_Ns']>.001 and x['sides'][i]['clearance_m']>=.002 for x in xs];sides.append({'loaded_endpoints':len(stance),'slip_p95_m_s':pct([y['mean_slip_m_s'] for y in stance],.95),'clear_swing_runs_at_least_3':sum(x>=3 for x in runs(swing))})
 event=next((x['step'] for x in state if x['phase']>=2.5),None);after=[x for x in state if event is not None and x['step']>=event];latch=next((x['step'] for x in state if x['step']<670 and x['packed']&4096),None)
 doc=json.loads((A/(name+'-analysis.json')).read_text());route=json.loads((A/'route-diagnostics.json').read_text())[name];r={'walking_samples':len(xs),'walking_forward_m':xs[-1]['root'][0]-xs[0]['root'][0] if xs else None,'walking_nonfoot_endpoints':sum(x['nonfoot_impulse']>1e-7 for x in xs),'walking_unsupported_endpoints':sum(all(y['impulse_Ns']<=.001 for y in x['sides']) for x in xs),'sides':sides,'leg_push_latch_step':latch,'nav5_step':doc['nav5_step'],'first_failure':route['first_fatal_contact'],'min_obstacle_clearance':route['minimum_accepted_step_obstacle_clearance'],'nonfoot_perch_contact_samples':doc['nonfoot_contact_samples'],'flight_toe_max_rad_after_25_free_samples':doc['max_flight_toe_abs_after_25_impulse_free'],'first_release_step':event,'post_release_samples':len(after),'post_release_bilateral_loaded':sum(x['left']>.1 and x['right']>.1 and x['left']+x['right']>2.2 for x in after),'nonfinite_task_samples':nonfinite,'solver_failed_steps':doc['failed_environment_steps'],'limits':'Accepted control endpoints; not all physics substeps. Release opens joints but does not prove toes lifted clear. Takeoff latch by itself does not prove causal impulse; matched no-push controls are archived separately.'};result[name]=r
(A/'full-sequence-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
