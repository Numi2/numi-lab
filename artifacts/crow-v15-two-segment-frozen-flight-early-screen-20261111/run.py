from pathlib import Path
import gzip,hashlib,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-frozen-flight-early-screen-20261111';B=R/'build/crow-v15-two-segment-frozen-flight-early-screen-20261111';L=R/'build/crow-v15-two-segment-frozen-flight-contact-map-20261111';cmd=json.loads((R/'artifacts/crow-v15-two-segment-frozen-flight-visual-bindings-20261111/rebound-command.json').read_text())
for k,v in [('--envs','80'),('--steps','200'),('--seed','20261120'),('--state-trace',str(B/'trace.tsv'))]:cmd[cmd.index(k)+1]=v
(A/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'inputs.json').write_text(json.dumps({'host_sha256':h(L/'libmetalrobo.dylib'),'metallib_sha256':h(L/'tracking.metallib'),'policy_sha256':h(L/'two-segment-zero.policypack'),'frozen_controller_sha256':h(R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal')},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compress):
 with path.open('wb') as raw:
  if compress:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):raw.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,A/'run.stdout',False)),threading.Thread(target=pump,args=(p.stderr,A/'run.stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
report={'process_exit_code':rc,'scope':'80 development lanes, early 200 ticks, obstacle-free flight admission'}
if rc==0:
 data=json.loads((A/'run.stdout').read_text());first=data['first_episode_navigation'];report.update({'failed_environment_steps':data['failed_environment_steps'],'physical_failure_count':first['physical_failure_count'],'steps_by_environment':first['steps_by_environment'],'termination_reasons':first['termination_reason_by_environment'],'early_admission_pass':data['failed_environment_steps']==0 and first['physical_failure_count']==0 and len(first['steps_by_environment'])==80 and all(v==200 for v in first['steps_by_environment'])})
if (B/'trace.tsv').exists():
 with (B/'trace.tsv').open('rb') as f,(A/'trace.tsv.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if rc==0 else 1)
