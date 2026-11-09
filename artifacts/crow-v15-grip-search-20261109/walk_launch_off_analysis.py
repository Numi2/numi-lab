from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-grip-search-20261109');names=['walk-launch-wings-off-control','walk-launch-wings-off-push'];data={}
for name in names:
 lanes=[{} for _ in range(8)]
 for line in gzip.open(A/(name+'.stderr.gz'),'rt'):
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   s=json.loads(line.split(' ',1)[1]);lanes[s['lane']][s['step']]=s
 doc=json.load(open(A/(name+'.stdout')));data[name]=(lanes,doc)
assert all(set(range(230,271))<=set(lane) for name in names for lane in data[name][0]),'missing contact samples'
pre=[]
for a,b in zip(data[names[0]][0],data[names[1]][0]):
 x,y=a[180],b[180];pre.append(max(abs(i-j) for key in ('q','v') for i,j in zip(x[key],y[key])))
assert max(pre)<1e-6,'preintervention mismatch'
reports={}
for name in names:
 lanes,doc=data[name];nav=doc['first_episode_navigation'];assert doc['failed_environment_steps']==0
 rows=[]
 for i,lane in enumerate(lanes):
  flags={}; nonfoot=[]
  for step in range(230,271):
   s=lane[step];loaded=set()
   for c in s['contacts']:
    aa,bb=c['a_name'],c['b_name']
    if 'locomotion_ground' not in (aa,bb) or c['impulses'][0]<=0:continue
    foot=next((x for x in (aa,bb) if x in ('crow_left_foot','crow_right_foot')),None)
    if foot and (-1 if aa==foot else 1)*c['normal'][2]*c['impulses'][0]>0:loaded.add(foot)
    elif not foot:nonfoot.append(step)
   flags[step]=loaded
  pairs=[step for step in range(230,270) if not flags[step] and not flags[step+1] and lane[step]['v'][2]>0]
  row={'lane':i,'initial_q_at_first_observation':lane[min(lane)]['q'][:7],'max_vz_window_m_s':max(lane[t]['v'][2] for t in range(230,271)),'unloaded_upward_pair_first_steps':pairs,'nonfoot_ground_contact_steps':sorted(set(nonfoot)),'first_episode_physical_failure':nav['physical_failure_by_environment'][i]}
  row['exploratory_endpoint_gate_pass']=bool(pairs and not nonfoot and not row['first_episode_physical_failure']);rows.append(row)
 reports[name]=rows
selected=json.load(gzip.open(A/(names[1]+'.replay.json.gz'),'rt'))['payload'];frames={f['step']:f for f in selected['frames']};clear=[]
for s in reports[names[1]][1]['unloaded_upward_pair_first_steps']:
 pair=[]
 for step in [s,s+1]:
  f=frames[step-1];sample=data[names[1]][0][1][step];assert max(abs(x-y) for k in ('q','v') for x,y in zip(sample[k],f[k]))<1e-6
  feet=[]
  for nm in ['crow_left_foot','crow_right_foot']:
   i=selected['body_names'].index(nm)*13;bs=f['body_states'][i:i+13];x,y,z,w=bs[3:7];r20=2*(x*z-w*y);r21=2*(y*z+w*x);r22=1-2*(x*x+y*y);feet.append(bs[2]-(abs(r20)*.05175+abs(r21)*.01725+abs(r22)*.0092))
  pair.append({'trace_step':step,'replay_frame':step-1,'foot_box_bottom_above_ground_m':feet,'root_vz_m_s':f['v'][2]})
 clear.append(pair)
report={'preintervention_max_abs_qv_difference_by_lane':pre,'records_per_lane_window':41,'arms':reports,'selected_lane_1_geometry':clear,'scope':'exploratory endpoint measurements; selected-lane clearance sampled only; perturbations are at initialization, not at launch; walk-to-launch development screen; no autonomous route, articulated toes or grip'}
(A/'walk-launch-wings-off-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for n in names:
 print(n,'endpoint_success',sum(r['exploratory_endpoint_gate_pass'] for r in reports[n]),'of',len(reports[n]));print([(r['lane'],r['unloaded_upward_pair_first_steps'],round(r['max_vz_window_m_s'],3)) for r in reports[n]])
print('selected lane geometry',clear)
