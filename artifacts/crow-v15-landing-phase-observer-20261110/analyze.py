from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-landing-phase-observer-20261110');P=Path('artifacts/crow-v15-ramped-perch-approach-20261110');original=json.loads((P/'development-reserved-final-j.stdout').read_text())['first_episode_navigation'];new=json.loads((A/'observer.stdout').read_text())['first_episode_navigation']
def read(p,states=False):
 sample={};phase={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);k=(x['lane'],x['step'])
    if x['step']<=original['steps_by_environment'][x['lane']]:sample[k]=(x['q'],x['v'])
   elif states and l.startswith('CROW_TASK_STATE '):
    v=l.split();lane=int(v[1]);step=int(v[2])
    if step<=original['steps_by_environment'][lane] and float(v[4])>=5:phase[(lane,step)]={'phase':float(v[7]),'height_command':float(v[8]),'foot_loads':[float(v[5]),float(v[6])],'forward_speed':float(v[9])}
 return sample,phase
x,_=read(P/'development-reserved-final-j.stderr.gz');y,ph=read(A/'observer.stderr.gz',True);ks=set(x)|set(y);diff=[k for k in sorted(ks) if x.get(k)!=y.get(k)];out=[]
for lane in range(8):
 a=[(k[1],v) for k,v in ph.items() if k[0]==lane];a.sort();phase2=[(step,v) for step,v in a if v['phase']>=1.5];both=[(step,v) for step,v in a if min(v['foot_loads'])>0.05];out.append({'lane':lane,'first_phase2':phase2[0] if phase2 else None,'first_bilateral_load':both[0] if both else None,'last':a[-1] if a else None,'first_episode_termination':original['steps_by_environment'][lane]})
r={'same_sample_keys':set(x)==set(y),'sample_count_original':len(x),'sample_count_observer':len(y),'different_samples':diff[:20],'sample_parity_exact':not diff,'same_termination_steps':original['steps_by_environment']==new['steps_by_environment'],'same_termination_reasons':original['termination_reason_by_environment']==new['termination_reason_by_environment'],'lanes':out};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
