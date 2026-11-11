from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-v1-slow-descent-20261111');B=Path('artifacts/crow-v15-v1-teacher-launch-parity-20261111');j=json.loads((A/'route.stdout').read_text())
def read(p):
 z={};clear=[];fatal=None;first={};submax=0
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);z[d['step']]=d
    if d['step']<=2053:
     for c in d['contacts']:
      names=c['a_name']+' '+c['b_name']
      if 'perch' in names and c['impulses'][0]>1e-7:
       key='foot_or_toe' if 'foot' in names or 'toe' in names else 'other_body'
       if key not in first:first[key]={'step':d['step'],'contact':c,'root_xyz':d['q'][:3]}
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_CONTACT_TRACE ') and fatal is None:fatal=json.loads(l.split(' ',1)[1])
   elif l.startswith('CROW_WALK_SUBSTEP '):submax=max(submax,int(l.split()[1]))
 return z,clear,fatal,first,submax
a,clear,fatal,first,submax=read(A/'route.stderr.gz');b,_,_,_,_=read(B/'route.stderr.gz');diff=[t for t in range(1,1306) if t not in a or t not in b or any(a[t][k]!=b[t][k] for k in ('q','v'))];out={'first_episode':j['first_episode_navigation'],'failed_environment_steps':j['failed_environment_steps'],'foot_support_step_count':j['foot_support_step_count'],'q_v_differences_through_1305':diff,'first_accepted_perch_contacts':first,'first_fatal_contact':fatal,'clearance_samples':clear,'last_ground_only_substep_trace_step':submax,'limitation':'substep observer only captured ground contacts through tick 505; no full-substep perch evidence exists in this run; post-failure reset excluded'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
