from pathlib import Path
import json,gzip,math,statistics
A=Path('artifacts/crow-v15-gait-kinematics-20261110')
def read(path,prefix):
 with gzip.open(path,'rt') as f:return [json.loads(x.split(' ',1)[1]) for x in f if x.startswith(prefix+' ')]
gait=read(A/'gait.stderr.gz','CROW_GAIT');got=read(A/'gait.stderr.gz','CROW_FLIGHT_SAMPLE');ref=read(Path('artifacts/crow-v15-damping-launch-conditional-20261110/20261126-push-trigger.stderr.gz'),'CROW_FLIGHT_SAMPLE');support=read(A/'gait.stderr.gz','CROW_SUPPORT_SAMPLE');a={(x['lane'],x['step']):x for x in got};b={(x['lane'],x['step']):x for x in ref};common=set(a)&set(b);delta=max((abs(x-y) for k in common for n in ('q','v') for x,y in zip(a[k][n],b[k][n])),default=None)
support_map={(x['lane'],x['step']):x for x in support};impulse_error=0
for s in gait:
 expected=[0.,0.];raw=support_map[s['lane'],s['step']]
 for c in raw['contacts']:
  name=c['a_name'] if c['b_name']=='locomotion_ground' else c['b_name'] if c['a_name']=='locomotion_ground' else ''
  for side,prefix in enumerate(('crow_left_','crow_right_')):
   if name==prefix+'foot' or name.startswith(prefix) and '_toe' in name:expected[side]+=max(0,c['impulses'][0])
 impulse_error=max(impulse_error,*[abs(expected[i]-s['sides'][i]['impulse_Ns']) for i in (0,1)])
def pct(x,p):
 x=sorted(x);return x[min(len(x)-1,math.ceil(p*len(x))-1)] if x else None
def longest(x):
 best=cur=0
 for v in x:cur=cur+1 if v else 0;best=max(best,cur)
 return best
rows=[]
for lane in range(8):
 xs=sorted((x for x in gait if x['lane']==lane),key=lambda x:x['step']);sides=[]
 for side in (0,1):
  stance=[x['sides'][side] for x in xs if x['sides'][side]['impulse_Ns']>.001];swing=[x['sides'][side]['impulse_Ns']<=.001 and x['sides'][1-side]['impulse_Ns']>.001 and x['sides'][side]['clearance_m']>=.002 for x in xs]
  sides.append({'loaded_endpoints':len(stance),'stance_mean_slip_p50_m_s':pct([x['mean_slip_m_s'] for x in stance],.5),'stance_mean_slip_p95_m_s':pct([x['mean_slip_m_s'] for x in stance],.95),'stance_peak_contact_slip_m_s':max((x['max_slip_m_s'] for x in stance),default=None),'swing_clearance_2mm_endpoints':sum(swing),'longest_contiguous_clear_swing_steps':longest(swing),'max_clearance_m':max(x['sides'][side]['clearance_m'] for x in xs)})
 rows.append({'lane':lane,'steps':len(xs),'forward_displacement_m':xs[-1]['root'][0]-xs[0]['root'][0], 'root_height_min_m':min(x['root'][2] for x in xs),'root_height_max_m':max(x['root'][2] for x in xs),'nonfoot_positive_endpoints':sum(x['nonfoot_impulse']>1e-7 for x in xs),'neither_foot_loaded_endpoints':sum(all(y['impulse_Ns']<=.001 for y in x['sides']) for x in xs),'sides':sides})
out={'observer_matched_qv_samples':len(common),'observer_max_abs_qv_difference':delta,'contact_impulse_max_disagreement_Ns':impulse_error,'samples':len(gait),'rows':rows,'scope':'short previously exposed run; accepted endpoint contact-point velocity estimate, not continuous contact trajectory or long-run gait qualification'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('parity',len(common),delta,'impulse difference',impulse_error,'lanes',len(rows));print(json.dumps(rows[0],indent=2))
if len(common)!=32 or delta!=0 or len(gait)!=808 or impulse_error>1e-6:raise SystemExit('observer validity failed')
