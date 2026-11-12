from pathlib import Path
import gzip,json,math,collections
R=Path('artifacts');A=R/'crow-v15-flight-propulsion-response-20261112';B=R/'crow-v15-slow-push-uniform-dev-20261112';p='candidate-20261127-rebound-flexed/run.stderr.gz'
def read(path):
 st={};co=collections.defaultdict(list)
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);st[x['lane'],x['step']]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);co[e].append((t,k,float(z[8])+float(z[9]),float(z[10])))
 return st,co
s,c=read(B/p);t,d=read(A/p);diff=[];counts=collections.Counter()
for e in range(8):
 for i in range(1,451):
  if (e,i) not in s or (e,i) not in t:diff.append([e,i,'missing']);continue
  for k in ('q','v','policy_action','contacts'):
   if s[e,i][k]!=t[e,i][k]:diff.append([e,i,k])
  counts['state_samples']+=1
 ac=[z for z in c[e] if z[0]<=450];bc=[z for z in d[e] if z[0]<=450];counts['contact_substeps']+=len(ac)
 if ac!=bc:diff.append([e,'contact'])
print('pre-intervention parity',dict(counts),'differences',len(diff));assert not diff,diff[:10]
meta=json.loads((A/'candidate-20261127-rebound-flexed/run.stdout').read_text())['first_episode_navigation'];
def att(q):
 x,y,z,w=q[3:7];return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
rows=[];arms=['control','control','sweep +','sweep -','pronation +','pronation -','tail +','tail -']
for e in range(8):
 end=min(800,meta['steps_by_environment'][e]);v=t[e,end];base=s[e,end];rows.append({'lane':e,'arm':arms[e],'first_episode_steps':end,'termination':meta['termination_reason_by_environment'][e],'waypoints':meta['maximum_waypoints_by_environment'][e],'position_m':v['q'][:3],'velocity_m_s':v['v'][:3],'attitude_roll_pitch_yaw_rad':att(v['q']),'matched_position_delta_m':[v['q'][k]-base['q'][k] for k in range(3)],'matched_velocity_delta_m_s':[v['v'][k]-base['v'][k] for k in range(3)]})
(A/'comparison.json').write_text(json.dumps({'pre_intervention_parity':dict(counts),'rows':rows,'scope':'One lane per signed input on a used layout and seed. Model responses are not independent biological validation or navigation qualification.'},indent=2)+'\n')
for r in rows:print(r['lane'],r['arm'],'end',r['first_episode_steps'],'reason',r['termination'],'wp',r['waypoints'],'pos',*[round(x,3) for x in r['position_m']],'vel',*[round(x,3) for x in r['velocity_m_s']],'rpy',*[round(x,3) for x in r['attitude_roll_pitch_yaw_rad']],'delta x',round(r['matched_position_delta_m'][0],3))
