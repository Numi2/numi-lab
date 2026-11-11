from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-propulsive-launch-controls-20261111');P=Path('artifacts/crow-v15-propulsive-leg-push-20261111')
def read(path):
 sup={};sub={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=270:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=270:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
base,bs=read(P/'hn50-an30.stderr.gz');out={}
for name,path in [('candidate',P/'hn50-an30.stderr.gz'),('wing-suppressed',A/'wing-suppressed.stderr.gz'),('held-crouch',A/'held-crouch.stderr.gz')]:
 sup,sub=read(path);rows=[]
 for lane in range(8):
  def ints(t):
   assert all((lane,t,k) in sub for k in range(16)),(name,lane,t)
   return [sum(sub[lane,t,k][i] for k in range(16)) for i in range(7)]
  diffs=[t for t in range(1,249) if (lane,t) not in sup or (lane,t) not in base or any(sup[lane,t][f]!=base[lane,t][f] for f in ('q','v'))]
  first=next((t for t in range(249,271) if sum(ints(t)[2:4])<=1e-7),None);rising=first is not None and first<270 and all(sup[lane,t]['v'][2]>.05 and sum(ints(t)[2:4])<=1e-7 for t in (first,first+1))
  recontact=next((t for t in range(first+1,271) if sum(ints(t)[2:4])+ints(t)[6]>1e-7),None) if first is not None else None
  end=recontact-1 if recontact else 270;root0=sup[lane,247]['q'][2];rise=max(sup[lane,t]['q'][2]-root0 for t in range(249,end+1));vz=max(sup[lane,t]['v'][2] for t in range(249,271));other=max(ints(t)[6] for t in range(249,271));rows.append({'lane':lane,'first_qv_difference_before_push':min(diffs) if diffs else None,'first_unload_step':first,'first_unload_vertical_speed_m_s':sup[lane,first]['v'][2] if first is not None else None,'two_unloaded_rising_intervals':rising,'first_recontact_step':recontact,'maximum_pre_recontact_height_gain_m':rise,'maximum_vertical_speed_m_s':vz,'maximum_nonfoot_impulse_Ns':other})
 data=json.loads(((P if name=='candidate' else A)/(('hn50-an30' if name=='candidate' else name)+'.stdout')).read_text());out[name]={'lanes':rows,'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count']}
w=out['wing-suppressed'];h=out['held-crouch'];out['leg_power_control_pass']=all(not r['first_qv_difference_before_push'] and r['two_unloaded_rising_intervals'] and r['maximum_pre_recontact_height_gain_m']>.02 and r['maximum_nonfoot_impulse_Ns']<=1e-7 for r in w['lanes']) and all(not r['first_qv_difference_before_push'] and not r['two_unloaded_rising_intervals'] for r in h['lanes']) and not w['failed_environment_steps'] and not w['termination_count'] and not h['failed_environment_steps'] and not h['termination_count']
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
