from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-front-grip-probe-20261110')
runs=[('control',Path('artifacts/crow-v15-approach-i-early-recovery-20261110/development-i-approach-early-recovery.stderr.gz')),('probe',A/'development-i-front-grip.stderr.gz')]
out={}
for name,p in runs:
 nav=0;last=0;prev=None;run=best=front=opposed=total=0;tail=None;events=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_TASK_STATE '):
    v=l.split();t=int(v[2]);
    if t<last:break
    last=t;nav=float(v[4]);tail={'step':t,'nav':nav,'left_load_n':float(v[5]),'right_load_n':float(v[6]),'phase':float(v[7])}
   elif l.startswith('CROW_SUPPORT_SAMPLE ') and nav>=5:
    x=json.loads(l.split(' ',1)[1]);t=x['step'];d={s:{'front':0.,'rear':0.} for s in ('left','right')}
    for c in x['contacts']:
     a,b=c['a_name'],c['b_name'];body=a if b=='crow_course_perch' else b if a=='crow_course_perch' else None
     if body is None:continue
     side=next((s for s in d if '_'+s+'_' in body),None);sign=-1 if body==a else 1;nx=-sign*c['normal'][0]
     if side and 'front_toe' in body and nx<-.5:d[side]['front']+=c['impulses'][0]
     if side and 'rear_toe' in body and nx>.5:d[side]['rear']+=c['impulses'][0]
    total+=1;bothfront=all(d[s]['front']>.00005 for s in d);bothop=all(d[s][kind]>.00005 for s in d for kind in ('front','rear'));front+=bothfront;opposed+=bothop;run=(run+1 if prev is not None and t==prev+1 else 1) if bothop else 0;best=max(best,run);prev=t
    if t in (4000,4200,4400,4420,4440,4460,4470,4500,5000,6000,7000):events.append({'step':t,'impulses':d,'root':x['q'][:3],'v':x['v'][:3],'toe_angles':x['q'][23:31]})
 out[name]={'support_contact_samples_after_waypoint5':total,'bilateral_front_impulse_samples':front,'simultaneous_bilateral_opposed_samples':opposed,'longest_consecutive_opposed_samples':best,'last_task_state':tail,'events':events}
out['limits']='Counts use accepted-step contact samples; missing samples break consecutive runs. They do not establish all-substep uninterrupted opposing forces or biological realism.'
(A/'grip-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
