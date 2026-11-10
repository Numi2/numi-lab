from pathlib import Path
import json,gzip,math
A=Path('artifacts/crow-v15-gait-stance-refinement-20261110')
def pct(x,p):
 x=sorted(x);return x[min(len(x)-1,math.ceil(p*len(x))-1)] if x else None
def longest(x):
 best=cur=0
 for v in x:cur=cur+1 if v else 0;best=max(best,cur)
 return best
out={}
for name in ('baseline','stance020','stance015','stance010'):
 E=Path('artifacts/crow-v15-gait-kinematics-20261110/gait.stderr.gz') if name=='baseline' else A/(name+'.stderr.gz');O=Path('artifacts/crow-v15-gait-kinematics-20261110/gait.stdout') if name=='baseline' else A/(name+'.stdout')
 with gzip.open(E,'rt') as f: data=[json.loads(l.split(' ',1)[1]) for l in f if l.startswith('CROW_GAIT ')]
 doc=json.loads(O.read_text());first=doc['first_episode_navigation'];rows=[]
 for lane in range(8):
  xs=sorted((x for x in data if x['lane']==lane and 100<=x['step']<=200),key=lambda x:x['step']);sides=[]
  for side in (0,1):
   stance=[x['sides'][side] for x in xs if x['sides'][side]['impulse_Ns']>.001];swing=[x['sides'][side]['impulse_Ns']<=.001 and x['sides'][1-side]['impulse_Ns']>.001 and x['sides'][side]['clearance_m']>=.002 for x in xs];sides.append({'loaded_endpoints':len(stance),'slip_p95_m_s':pct([x['mean_slip_m_s'] for x in stance],.95),'peak_contact_slip_m_s':max((x['max_slip_m_s'] for x in stance),default=None),'clear_swing_max_consecutive_steps':longest(swing),'max_clearance_m':max((x['sides'][side]['clearance_m'] for x in xs),default=None)})
  forward=xs[-1]['root'][0]-xs[0]['root'][0] if xs else None;nonfoot=sum(x['nonfoot_impulse']>1e-7 for x in xs);unsupported=sum(all(y['impulse_Ns']<=.001 for y in x['sides']) for x in xs);valid=len(xs)==101 and [x['step'] for x in xs]==list(range(100,201));passed=(valid and not first['physical_failure_by_environment'][lane] and first['steps_by_environment'][lane]==400 and doc['failed_environment_steps']==0 and forward>=.02 and not nonfoot and not unsupported and all(x['slip_p95_m_s'] is not None and x['slip_p95_m_s']<=.05 and x['clear_swing_max_consecutive_steps']>=3 for x in sides));rows.append({'lane':lane,'forward_m':forward,'nonfoot_endpoints':nonfoot,'unsupported_endpoints':unsupported,'physical_failure':first['physical_failure_by_environment'][lane],'episode_steps':first['steps_by_environment'][lane],'sides':sides,'pass':passed})
 out[name]={'lanes':rows,'passing_lanes':sum(x['pass'] for x in rows),'failed_environment_steps':doc['failed_environment_steps']};r=rows[0];print(name,'pass',out[name]['passing_lanes'],'forward',r['forward_m'],'slip',[x['slip_p95_m_s'] for x in r['sides']],'clear',[x['clear_swing_max_consecutive_steps'] for x in r['sides']],'fail',r['physical_failure'])
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
