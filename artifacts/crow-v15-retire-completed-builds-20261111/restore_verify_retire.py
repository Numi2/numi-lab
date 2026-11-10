from pathlib import Path
import gzip,hashlib,io,json,subprocess,tarfile
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110')
folders=('crow-v15-ankle-authority-screen-20261111', 'crow-v15-com-balance-screen-20261111', 'crow-v15-matched-balance-grip-screen-20261111', 'crow-v15-perch-control-response-20261111', 'crow-v15-perch-dynamics-audit-20261111', 'crow-v15-rear-grip-reacquisition-20261111', 'crow-v15-sustained-grip-screen-20261111')
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True)
rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor','56145bd8',rev],check=True)
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
(R/'artifacts/crow-v15-retire-completed-builds-20261111/retired.json').write_text(json.dumps({'remote_commit':rev,'retired':retired},indent=2)+'\n')
print('Verified remote restoration of',len(retired),'build copies; retired',sum(x['bytes'] for x in retired),'bytes')
