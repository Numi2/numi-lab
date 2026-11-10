from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-toe-launch-control-20261110');data={}
for arm in ('control','push'):
 lanes=[{} for _ in range(8)]
 with gzip.open(A/(arm+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);lanes[d['lane']][d['step']]=d
 doc=json.loads((A/(arm+'.stdout')).read_text());data[arm]=(lanes,doc)
assert all(set(range(230,271)) <= set(lane) for arm in data for lane in data[arm][0]),'missing accepted contact samples'
parity=[]
for i,(a,b) in enumerate(zip(data['control'][0],data['push'][0])):
 ds=[max(abs(x-y) for k in ('q','v') for x,y in zip(a[t][k],b[t][k])) for t in range(1,247) if t in a and t in b];assert len(ds)==246
 parity.append(max(ds));assert parity[-1]<1e-6,('preintervention mismatch',i,parity[-1])
reports={}
for arm,(lanes,doc) in data.items():
 nav=doc['first_episode_navigation'];rows=[]
 for i,lane in enumerate(lanes):
  end=min(400,nav['steps_by_environment'][i]+1);nonfoot=[];loaded={}
  for step,s in lane.items():
   if step>end:continue
   on=[]
   for c in s['contacts']:
    a,b=c['a_name'],c['b_name'];imp=c['impulses'][0]
    if 'locomotion_ground' not in (a,b) or imp<=1e-7:continue
    bird=b if a=='locomotion_ground' else a
    if bird in ('crow_left_foot','crow_right_foot') or '_toe' in bird:on.append(bird)
    else:nonfoot.append({'step':step,'body':bird,'impulse_Ns':imp})
   loaded[step]=on
  pairs=[t for t in range(249,270) if t+1<=end and not loaded.get(t,['missing']) and not loaded.get(t+1,['missing']) and lane[t]['v'][2]>0 and lane[t+1]['v'][2]>0]
  eligible=[t for t in range(249,271) if t<=end and t in lane];root=lane[1]['q'];last=lane[end] if end in lane else lane[max(t for t in lane if t<=end)];airborne=bool(end==400 and all(not loaded.get(t,['missing']) for t in range(350,401)) and last['q'][2]>root[2]+.15 and not nav['physical_failure_by_environment'][i]);row={'lane':i,'first_episode_end':end,'native_physical_failure':nav['physical_failure_by_environment'][i],'unloaded_upward_pairs':pairs,'max_upward_speed_m_s':max((lane[t]['v'][2] for t in eligible),default=None),'nonfoot_ground_contacts_before_end':nonfoot,'start_root_position_m':root[:3],'end_root_position_m':last['q'][:3],'end_root_velocity_m_s':last['v'][:3],'airborne_last_second_and_15cm_higher':airborne};row['leg_hop_gate']=bool(pairs and not any(x['step']<=270 for x in nonfoot));rows.append(row)
 reports[arm]=rows
result={'preintervention_max_qv_difference_by_lane':parity,'arms':reports,'failed_environment_steps':{k:d[1]['failed_environment_steps'] for k,d in data.items()},'scope':'eight perturbed starts, development course; flap suppressed only; no biological qualification'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for arm,rows in reports.items():print(arm,'hop',sum(x['leg_hop_gate'] for x in rows),'of 8; airborne at 8s',sum(x['airborne_last_second_and_15cm_higher'] for x in rows),'first failures',sum(x['native_physical_failure'] for x in rows));print([(r['lane'],r['first_episode_end'],r['unloaded_upward_pairs'],r['max_upward_speed_m_s'],r['end_root_position_m'][2]) for r in rows])
