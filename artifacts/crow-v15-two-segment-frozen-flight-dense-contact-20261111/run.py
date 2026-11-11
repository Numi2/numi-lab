from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-dense-contact-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-dense-contact-20261111';base=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-early-screen-20261111/command.json').read_text());rows=[]
for n in (3,200):
 cmd=list(base);cmd[cmd.index('--steps')+1]=str(n);cmd[cmd.index('--state-trace')+1]=str(B/(str(n)+'.tsv'));(A/(str(n)+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n');p=subprocess.run(cmd,cwd=R,env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=900);(A/(str(n)+'.stdout')).write_text(p.stdout)
 with (A/(str(n)+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 samples=[json.loads(l.split(' ',1)[1]) for l in p.stderr.splitlines() if l.startswith('CROW_FLIGHT_SAMPLE ')];r={'steps':n,'exit':p.returncode,'observed_events':len(samples),'stderr_tail':p.stderr[-1800:]}
 if p.returncode==0:
  j=json.loads(p.stdout);f=j['first_episode_navigation'];r.update({'failed_environment_steps':j['failed_environment_steps'],'physical_failure_count':f['physical_failure_count'],'steps_by_environment':f['steps_by_environment']})
 rows.append(r);print('steps',n,'exit',p.returncode,'events',len(samples),p.stderr[-400:],flush=True)
 if p.returncode or r.get('failed_environment_steps') or r.get('physical_failure_count'):break
(A/'analysis.json').write_text(json.dumps({'cases':rows,'scope':'dense contact alternative diagnostic, no flight qualification'},indent=2)+'\n')
