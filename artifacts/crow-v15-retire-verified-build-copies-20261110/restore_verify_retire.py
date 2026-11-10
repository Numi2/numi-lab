from pathlib import Path
import gzip,hashlib,io,json,subprocess,tarfile
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110')
folders=('crow-v15-pitch-toe-stiffness-screen-20261110','crow-v15-rear-toe-contact-regulation-20261110','crow-v15-rear-toe-hold-regulation-20261110','crow-v15-bilateral-perturbation-screen-20261110','crow-v15-precontact-leg-position-screen-20261110','crow-v15-toe-preparation-timing-screen-20261110','crow-v15-foot-position-feedback-screen-20261110','crow-v15-foot-support-substep-audit-20261110','crow-v15-foot-placement-balance-gain-screen-20261110','crow-v15-long-perch-support-screen-20261110','crow-v15-long-pitch-balance-screen-20261110')
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True)
rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor','3e74433a',rev],check=True)
def show(path):return subprocess.check_output(['git','-C',str(C),'show',rev+':'+path])
def sha(b):return hashlib.sha256(b).hexdigest()
retired=[]
for folder in folders:
 artifact='artifacts/'+folder;index=json.loads(show(artifact+'/runtime-index.json'));remote=show(artifact+'/runtime.tar.gz')
 if sha(remote)!=sha((R/artifact/'runtime.tar.gz').read_bytes()):raise SystemExit('remote archive mismatch '+folder)
 with tarfile.open(fileobj=io.BytesIO(remote),mode='r:gz') as tar:
  for item in index:
   path=item['path'];prefix='build/'+folder+'/'
   if not path.startswith(prefix) or Path(path).name not in ('probe','libmetalrobo.dylib'):continue
   restored=tar.extractfile(path)
   if restored is None:raise SystemExit('missing member '+path)
   data=restored.read();local=(R/path).read_bytes()
   if sha(data)!=item['sha256'] or sha(local)!=item['sha256'] or len(data)!=item['bytes']:raise SystemExit('restore mismatch '+path)
   retired.append({'path':path,'sha256':item['sha256'],'bytes':len(data),'archive':artifact+'/runtime.tar.gz'})
if len(retired)!=2*len(folders):raise SystemExit('unexpected member count')
for item in retired:(R/item['path']).unlink()
(R/'artifacts/crow-v15-retire-verified-build-copies-20261110/retired.json').write_text(json.dumps({'remote_commit':rev,'retired':retired},indent=2)+'\n')
print('Verified remote restoration of',len(retired),'build copies; retired',sum(x['bytes'] for x in retired),'bytes')
