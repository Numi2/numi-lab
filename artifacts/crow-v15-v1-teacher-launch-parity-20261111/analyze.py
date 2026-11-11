from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-v1-teacher-launch-parity-20261111');B=Path('artifacts/crow-v15-v1-route-diagnostic-20261111');j=json.loads((A/'route.stdout').read_text())
def read(p):
 z={};clear=[];first=None
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);z[d['step']]=d
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_CONTACT_TRACE ') and first is None:first=json.loads(l.split(' ',1)[1])
 return z,clear,first
a,clear,first=read(A/'route.stderr.gz');b,_,_=read(B/'route.stderr.gz');diff=[t for t in range(1,351) if t not in a or t not in b or any(a[t][k]!=b[t][k] for k in ('q','v'))];out={'first_episode':j['first_episode_navigation'],'failed_environment_steps':j['failed_environment_steps'],'foot_support_step_count':j['foot_support_step_count'],'q_v_differences_first_350':diff,'first_fatal_contact':first,'clearance_samples':clear,'terminal_q':a[max(a)]['q'][:7],'scope':'one privileged development J route, not autonomous and no landing'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
