from pathlib import Path
import gzip,json,os,subprocess,threading
R=Path.cwd();A=R/'artifacts/crow-v15-two-segment-leg-takeoff-comparison-20261111';B=R/'build/crow-v15-two-segment-leg-takeoff-comparison-20261111';L=R/'build/crow-v15-two-segment-walk-frozen-preflight-20261111'
for arm in ('push','control'):
 cmd=json.loads((R/'artifacts/crow-v15-two-segment-walk-screen-20261111/command.json').read_text());cmd[cmd.index('--steps')+1]='450';cmd[cmd.index('--metallib')+1]=str(B/(arm+'.metallib'));cmd[cmd.index('--state-trace')+1]=str(B/(arm+'.trace.tsv'));cmd[cmd.index('--crow-replay-pack')+1]=str(B/(arm+'.replay.json'));(A/(arm+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_SUBSTEP_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CONTACT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 def pump(src,path,zipped):
  with path.open('wb') as f:
   if zipped:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:
     for b in iter(lambda:src.read(1<<16),b''):z.write(b)
   else:
    for b in iter(lambda:src.read(1<<16),b''):f.write(b)
 ts=[threading.Thread(target=pump,args=(p.stdout,A/(arm+'.stdout'),False)),threading.Thread(target=pump,args=(p.stderr,A/(arm+'.stderr.gz'),True))]
 for t in ts:t.start()
 rc=p.wait()
 for t in ts:t.join()
 for source,dest in [(B/(arm+'.trace.tsv'),A/(arm+'.trace.tsv.gz')),(B/(arm+'.replay.json'),A/(arm+'.replay.json.gz'))]:
  if source.exists():
   with source.open('rb') as f,dest.open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
     for b in iter(lambda:f.read(1<<16),b''):z.write(b)
 (A/(arm+'.result.json')).write_text(json.dumps({'exit':rc},indent=2)+'\n');print(arm,'exit',rc,flush=True)
 if rc:break
