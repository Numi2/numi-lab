from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-reset-once-ablation-20261114');P=A/'turn-20261126-rebound-resetonce';R=Path('artifacts/crow-v15-upward-trigger-stroke-20261114');ref={(s['lane'],s['step']):s for s in json.load(gzip.open(R/'push-window.json.gz','rt'))['state']}; states={};sub=collections.defaultdict(list);ground=collections.defaultdict(list);errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);states[x['lane'],x['step']]=x;continue
  a=line.split()
  if not a:continue
  if 'OBSERVER_ERROR' in a[0] or a[0]=='CROW_SUBSTEP_ERROR':errors.append(line.strip());continue
  if a[0] in ('CROW_TWO_TOE_SUBSTEP','CROW_GROUND_CONTACT'):
   t,k,e=map(int,a[1:4])
   if a[0]=='CROW_TWO_TOE_SUBSTEP':
    if int(a[4])>int(a[5]):errors.append('contact capacity exceeded '+str((e,t,k)))
    sub[e,t].append((k,list(map(float,a[6:]))))
   else:ground[e].append((t,k,int(a[4]),int(a[5])))
stdout=json.loads((P/'run.stdout').read_text());meta=stdout['first_episode_navigation']; rows=[]
for e in range(8):
 x=states[e,1];y=ref[e,1];diff={k:max(abs(a-b) for a,b in zip(x[k],y[k])) for k in ('q','v')}
 row={'lane':e,'first_observer_state_difference':diff,'first_episode_steps':meta['steps_by_environment'][e],'physical_failure':meta['physical_failure_by_environment'][e]}
 if e in (6,7):
  first=next((t for t in range(371,501) if len(sub[e,t])==16 and sum(sum(v[2:4]) for _,v in sub[e,t])<=1e-7),None)
  after=next((t for t in range((first or 500)+1,501) if sum(sum(v[2:4]) for _,v in sub[e,t])>1e-7),None)
  row.update({'first_unloaded_interval':first,'first_later_foot_load':after,'first_later_ground_contact':next((g for g in ground[e] if first and g[0]>first),None),'unique_complete_foot_substeps':all(sorted(k for k,_ in sub[e,t])==list(range(16)) for t in range(348,501)),'finite_state_coverage':sum((e,t) in states and all(math.isfinite(z) for z in states[e,t]['q']+states[e,t]['v']) for t in range(1,501))})
 rows.append(row)
o={'scope':'same-seed one-versus-two reset blocks, first recorded state is not independently pre-step','rows':rows,'teacher_off':stdout['birdflow_journey_teacher'] is False,'errors':errors};(A/'analysis.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
