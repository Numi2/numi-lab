from pathlib import Path
import json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-batch-isolation-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-batch-isolation-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111';base=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-visual-bindings-20261111/rebound-command.json').read_text());rows=[]
for n in (1,2,4,8,16,32,80):
 cmd=list(base);cmd[cmd.index('--envs')+1]=str(n);cmd[cmd.index('--state-trace')+1]=str(B/(str(n)+'.tsv'));(A/(str(n)+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');p=subprocess.run(cmd,cwd=R,env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=180);(A/(str(n)+'.stdout')).write_text(p.stdout);(A/(str(n)+'.stderr')).write_text(p.stderr);row={'batch':n,'return_code':p.returncode,'stderr_tail':p.stderr[-2200:]}
 if p.returncode==0:
  x=json.loads(p.stdout);row.update({'failed_environment_steps':x['failed_environment_steps'],'first_episode':x['first_episode_navigation']})
 rows.append(row);print('batch',n,'exit',p.returncode,p.stderr[-500:],flush=True)
(A/'analysis.json').write_text(json.dumps({'cases':rows,'scope':'two-step batch isolation, not flight qualification'},indent=2)+'\n')
