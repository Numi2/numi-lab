from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,threading
R=Path.cwd();A=R/'artifacts/crow-v15-sustained-disturbance-20261114';B=R/'build/crow-v15-sustained-disturbance-20261114';C=B;M=R/'build/crow-v15-mirrored-pronation-20261113';J=R/'artifacts/crow-v15-hip-roll-journey-integration-20261111';L=R/'build/crow-v15-sustained-disturbance-20261114';name=sys.argv[1];assert name in ('control','gust');seed=int(sys.argv[2]) if len(sys.argv)>2 else 20261126;assert seed in (20261126,20261127);tag=sys.argv[3] if len(sys.argv)>3 else '';substeps=int(sys.argv[4]) if len(sys.argv)>4 else 16;assert substeps == 16, "release threshold is calibrated only for this 16-substep diagnostic";assert all(c.isalnum() or c=='-' for c in tag);D=A/(name+'-'+str(seed)+(('-'+tag) if tag else ''));D.mkdir(exist_ok=False);cmd=json.loads((J/'rebound-command.json').read_text());
for k,v in [('--seed',str(seed)),('--steps','1800'),('--physics-substeps',str(substeps)),('--metallib',str(C/(name+'.metallib'))),('--state-trace',str(B/(D.name+'-trace.tsv'))),('--crow-replay-pack',str(B/(D.name+'-replay.json')))]:cmd[cmd.index(k)+1]=v
if substeps==32:
 for option,label in (('--policy-pack','landing'),('--base-policy-pack','route')):
  candidate=B/('phys32-'+label+'.policypack')
  if candidate.exists():cmd[cmd.index(option)+1]=str(candidate)
cmd[cmd.index('--base-policy-pack')+1]=str(R/'build/crow-v15-localized-route-student-imitation-20261111/route-imitation.policypack')
cmd[cmd.index("--state-trace-environment")+1]="5"  # retain the prior release lane
cmd.remove('--birdflow-journey-teacher')
idx=cmd.index('--birdflow-journey-student-authority'); del cmd[idx:idx+2]
if tag.startswith('rebound'):
 for option,label in (('--policy-pack','landing'),('--base-policy-pack','route')):
  cmd[cmd.index(option)+1]=str(M/(label+'-candidate.policypack'))
cmd[0]=str(R/'build/crow-v15-replay-writer-roundtrip-20261111/metalrobo_task_rollout')
(D/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(D/'inputs.json').write_text(json.dumps({'host_sha256':h(L/'libmetalrobo.dylib'),'metallib_sha256':h(Path(cmd[cmd.index('--metallib')+1])),'frozen_flight_sha256':h(R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal'),'landing_policy_sha256':h(Path(cmd[cmd.index('--policy-pack')+1])),'route_policy_sha256':h(Path(cmd[cmd.index('--base-policy-pack')+1]))},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_AERO_WRENCH_TRACE':'1','CROW_SUBSTEP_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_EXPERIMENT_GAIT_SMOOTH',None);env.update({'CROW_EXPERIMENT_GAIT_SMOOTH':'1'} if True else {});p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compress):
 with path.open('wb') as raw:
  if compress:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):raw.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,D/'run.stdout',False)),threading.Thread(target=pump,args=(p.stderr,D/'run.stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
for kind in ('trace.tsv','replay.json'):
 path=B/(D.name+'-'+kind)
 if path.exists():
  with path.open('rb') as f,(D/(kind+'.gz')).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(D/'run-result.json').write_text(json.dumps({'exit_code':rc},indent=2)+'\n');print(name,'exit',rc,flush=True)

if rc != 0: raise SystemExit(rc)
