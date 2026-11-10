from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-causal-launch-20261110');P=Path('artifacts/crow-v15-toe-launch-control-20261110');data={}
for arm in ('push-trigger','push-timer','control-trigger'):
 lanes=[{} for _ in range(8)];status=[[] for _ in range(8)]
 with gzip.open(A/(arm+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);lanes[d['lane']][d['step']]=d
   elif line.startswith('CROW_TASK_STATE '):
    _,lane,step,packed,nav,l,r=line.split();status[int(lane)].append({'step':int(step),'packed':int(packed),'left_support_N':float(l),'right_support_N':float(r)})
 doc=json.loads((A/(arm+'.stdout')).read_text());data[arm]=(lanes,status,doc)
reference=[{} for _ in range(8)]
with gzip.open(P/'push.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   d=json.loads(line.split(' ',1)[1]);reference[d['lane']][d['step']]=d
parity=[]
for i in range(8):
 for t in range(1,247):
  base=reference[i][t]
  for arm in data:
   d=data[arm][0][i][t];diff=max(abs(x-y) for k in ('q','v') for x,y in zip(base[k],d[k]));assert diff<1e-6,(i,t,arm,diff)
 parity.append(True)
result={}
for arm,(lanes,status,doc) in data.items():
 nav=doc['first_episode_navigation'];rows=[]
 for i,lane in enumerate(lanes):
  end=min(400,nav['steps_by_environment'][i]+1);latches=[s for s in status[i] if s['step']<270 and s['packed']&4096];first=latches[0] if latches else None
  if first:assert first['left_support_N']<0.05 and first['right_support_N']<0.05
  nofoot={};bad=[]
  for t,s in lane.items():
   if t>end:continue
   ground=[]
   for c in s['contacts']:
    a,b=c['a_name'],c['b_name'];imp=c['impulses'][0]
    if 'locomotion_ground' not in (a,b) or imp<=1e-7:continue
    bird=b if a=='locomotion_ground' else a;ground.append(bird)
    if bird not in ('crow_left_foot','crow_right_foot') and '_toe' not in bird:bad.append({'step':t,'body':bird,'impulse_Ns':imp})
   nofoot[t]=len(ground)==0
  initial=lane[1]['q'];last=lane[end] if end in lane else lane[max(t for t in lane if t<=end)];flight=bool(end==400 and all(nofoot.get(t,False) for t in range(350,401)) and last['q'][2]>initial[2]+.15 and not nav['physical_failure_by_environment'][i]);rows.append({'lane':i,'first_latch':first,'first_episode_end':end,'native_physical_failure':nav['physical_failure_by_environment'][i],'nonfoot_ground_contacts':bad,'initial_z_m':initial[2],'final_z_m':last['q'][2],'final_forward_m':last['q'][0]-initial[0],'final_second_unloaded_and_15cm_higher':flight,'posthoc_continuously_ground_clear_after_latch':bool(first and end==400 and all(nofoot.get(t,False) for t in range(first['step']+1,401)))})
 result[arm]=rows
out={'matched_preextension_to_prior_run_by_lane':parity,'arms':result,'failed_environment_steps':{k:v[2]['failed_environment_steps'] for k,v in data.items()},'scope':'experimental v15 toe body; 8 perturbed starts, development course, passive aerodynamics active; not full-route or biological qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,rows in result.items():print(arm,'early_latch',sum(x['first_latch'] is not None for x in rows),'of 8; airborne final second',sum(x['final_second_unloaded_and_15cm_higher'] for x in rows),'failure',sum(x['native_physical_failure'] for x in rows));print([(x['lane'],x['first_latch'],round(x['final_z_m'],3)) for x in rows])
