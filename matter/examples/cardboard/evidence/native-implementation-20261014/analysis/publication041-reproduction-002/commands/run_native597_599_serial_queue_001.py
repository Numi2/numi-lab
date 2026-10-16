from pathlib import Path
import os,json,time,subprocess,datetime,hashlib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native597-599-serialized-queue-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
scripts=['run_native597_freebody_trial_summary_001.py','run_native598_contact_compactor_fixture_fixed_001.py','run_native599_primary_cache_action_diagnostic_001.py']
spec={'schema':'numi.cardboard.serialized-native-queue.v1','wait_for':'solver/strict-state-source240-control19-final-newton-diagnostic-596/receipt.json','scripts':{n:sha(q/n) for n in scripts},'gpu_workers':1,'independent_checks_continue_after_recorded_failures':True,'launched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'queue-spec.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'queue':'waiting_for596','pid':os.getpid(),'scripts':scripts}),flush=True)
deadline=time.monotonic()+7500
while not (q/spec['wait_for']).exists():
 if time.monotonic()>deadline:
  (d/'timeout.json').write_text(json.dumps({'reason':'596 receipt unavailable after7500s; no queued GPU run launched'},indent=2)+'\n');raise SystemExit(124)
 time.sleep(2)
results=[]
for n in scripts:
 assert sha(q/n)==spec['scripts'][n]
 workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')]
 assert not workers,workers
 start=time.monotonic();print(json.dumps({'starting':n,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
 with (d/(n+'.log')).open('xb') as out:
  p=subprocess.run(['/usr/bin/python3',str(q/n)],stdout=out,stderr=subprocess.STDOUT,timeout=1100)
 row={'script':n,'exit_code':p.returncode,'wall_seconds':time.monotonic()-start,'log_sha256':sha(d/(n+'.log'))};results.append(row)
 (d/'progress.json').write_text(json.dumps({'results':results},indent=2)+'\n');print(json.dumps(row),flush=True)
(d/'report.json').write_text(json.dumps({'schema':'numi.cardboard.serialized-native-queue-result.v1','results':results,'all_scripts_zero':all(x['exit_code']==0 for x in results),'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
print(json.dumps({'queue_complete':True,'results':results}),flush=True)
