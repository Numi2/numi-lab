from pathlib import Path
import gzip,os,subprocess,threading,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-supported-return-gate-20261114';B=R/'build/crow-v15-supported-return-gate-20261114';P=R/'build/crow-v15-perch-toe-substep-velocity-20261114';W=R/'build/crow-v15-wide-contact-isolation-20261111';assert hashlib.sha256((B/'probe').read_bytes()).hexdigest() == (A/'prebuild-observed.sha256').read_text().split()[0], 'prebuilt probe changed'
(A/'run-observed-input-hashes.json').write_text(json.dumps({str(x.relative_to(R)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (A/'probe.cpp',B/'probe',P/'libmetalrobo.dylib',W/'MetalRobo.metallib')},indent=2)+'\n')
assert not (A/'trace.log.gz').exists() and not (A/'substeps.log.gz').exists(), 'output already exists'
p=subprocess.Popen([str(B/'probe')],cwd=R,env={**os.environ,'CROW_SUBSTEP_TRACE':'1'},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def pump(src,path):
 with path.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:
   for chunk in iter(lambda:src.read(1<<16),b''):out.write(chunk)
ts=[threading.Thread(target=pump,args=(p.stdout,A/'trace.log.gz')),threading.Thread(target=pump,args=(p.stderr,A/'substeps.log.gz'))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
print('exit',rc,flush=True);raise SystemExit(rc)
