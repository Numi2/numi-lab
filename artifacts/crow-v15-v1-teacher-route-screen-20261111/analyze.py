from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-v1-teacher-route-screen-20261111');B=Path('artifacts/crow-v15-v1-route-diagnostic-20261111');j=json.loads((A/'route.stdout').read_text())
def read(p):
 with gzip.open(p,'rt') as f:return [[float(v) for v in line.split()] for line in f if not line.startswith('#')]
a=read(A/'route.tsv.gz');b=read(B/'route.tsv.gz');diff=[{'step':int(x[0]),'max_q_difference':max(abs(i-k) for i,k in zip(x[1:32],y[1:32]))} for x,y in zip(a[:350],b[:350]) if x[:32]!=y[:32]];clear=[];first=None
with gzip.open(A/'route.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(line.split(' ',1)[1]))
  if line.startswith('CROW_CONTACT_TRACE ') and first is None:first=json.loads(line.split(' ',1)[1])
out={'first_episode':j['first_episode_navigation'],'termination_count':j['termination_count'],'failed_environment_steps':j['failed_environment_steps'],'foot_support_step_count':j['foot_support_step_count'],'pre_handoff_q_difference_count':len(diff),'first_q_difference':diff[0] if diff else None,'maximum_pre_handoff_q_difference':max((x['max_q_difference'] for x in diff),default=0),'first_fatal_contact':first,'clearance_samples':clear,'terminal_root_xyz_m':a[-1][1:4],'scope':'one privileged development J route; not autonomous, no landing; parity comparison is q only'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
