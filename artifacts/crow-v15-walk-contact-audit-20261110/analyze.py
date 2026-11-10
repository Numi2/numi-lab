from pathlib import Path
import gzip,json,math,statistics
R=Path.cwd();A=R/'artifacts/crow-v15-walk-contact-audit-20261110';src=R/'artifacts/crow-v15-toe-launch-control-20261110';lanes=[{} for _ in range(8)]
with gzip.open(src/'control.stderr.gz','rt') as f:
 for l in f:
  if not l.startswith('CROW_SUPPORT_SAMPLE '):continue
  d=json.loads(l.split(' ',1)[1]);t=d['step']
  if not 100<=t<=200:continue
  force=[0.,0.]
  for c in d['contacts']:
   a,b=c['a_name'],c['b_name'];obj=b if a=='locomotion_ground' else a if b=='locomotion_ground' else ''
   for side,ind in (('left',0),('right',1)):
    if obj.startswith('crow_'+side+'_') and ('foot' in obj or '_toe' in obj):force[ind]+=max(0.,c['impulses'][0])
  lanes[d['lane']][t]=force
rows=[]
for i,lane in enumerate(lanes):
 assert set(lane)==set(range(100,201));v=list(lane.values());threshold=.001
 rows.append({'lane':i,'both_loaded':sum(l>threshold and r>threshold for l,r in v),'left_only_loaded':sum(l>threshold and r<=threshold for l,r in v),'right_only_loaded':sum(r>threshold and l<=threshold for l,r in v),'neither_loaded':sum(l<=threshold and r<=threshold for l,r in v)})
with gzip.open(src/'control.replay.json.gz','rt') as f:p=json.load(f)['payload']
frames=[x for x in p['frames'] if 100<=x['step']<=200];assert len(frames)==101
feet={}
for name in ('crow_left_foot','crow_right_foot'):
 idx=p['body_names'].index(name)*13;arr=[x['body_states'][idx:idx+13] for x in frames];speeds=[math.hypot(x[7],x[8]) for x in arr]
 feet[name]={'origin_height_min_m':min(x[2] for x in arr),'origin_height_max_m':max(x[2] for x in arr),'origin_planar_speed_median_m_s':statistics.median(speeds),'origin_planar_speed_max_m_s':max(speeds)}
out={'source':'crow-v15-toe-launch-control-20261110/control','window':'steps 100..200 inclusive, 20ms endpoints','normal_impulse_threshold_Ns':.001,'lanes':rows,'lane_0_foot_origin_kinematics':feet,'limits':'Foot-body-origin speed is not contact-point slip; contact samples do not establish continuous swing clearance, natural gait or long-distance stability.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
