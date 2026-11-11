from pathlib import Path
import gzip,json,os,subprocess,threading,sys
R=Path.cwd();A=R/'artifacts/crow-v15-teacher-support-dev-comparison-20261111';B=R/'build/crow-v15-teacher-support-dev-comparison-20261111';L=R/'build/crow-v15-two-segment-walk-frozen-preflight-20261111';name=sys.argv[1];assert name in ('original','separated');out=A/name;out.mkdir(exist_ok=True);bb=B/name;bb.mkdir(exist_ok=True);cmd=json.loads((R/'artifacts/crow-v15-two-segment-contact-flight-handoff-20261111/command.json').read_text());cmd[cmd.index('--steps')+1]='1300';cmd[cmd.index('--metallib')+1]=str(R/('build/crow-v15-two-segment-contact-flight-handoff-20261111/tracking.metallib' if name=='original' else 'build/crow-v15-teacher-support-complete-20261111/tracking.metallib'));cmd[cmd.index('--state-trace')+1]=str(bb/'trace.tsv');cmd[cmd.index('--crow-replay-pack')+1]=str(bb/'replay.json');cmd.extend(['--birdflow-journey-teacher','--birdflow-journey-student-authority','0']);(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path,compress):
 with path.open('wb') as f:
  if compress:
   with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
    for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  else:
   for b in iter(lambda:src.read(1<<16),b''):f.write(b)
ts=[threading.Thread(target=pump,args=(p.stdout,out/'stdout',False)),threading.Thread(target=pump,args=(p.stderr,out/'stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
for source,dest in [(bb/'trace.tsv',out/'trace.tsv.gz'),(bb/'replay.json',out/'replay.json.gz')]:
 if source.exists():
  with source.open('rb') as f,dest.open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(out/'run-result.json').write_text(json.dumps({'exit':rc},indent=2)+'\n');print(name,'exit',rc,flush=True)
