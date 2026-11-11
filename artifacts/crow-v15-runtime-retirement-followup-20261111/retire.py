from pathlib import Path
import hashlib,io,json,subprocess,tarfile
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');folders=('crow-v15-regrip-horizontal-force-audit-20261111','crow-v15-passive-contact-refinement-20261111','crow-v15-regrip-grid-force-audit-20261111')
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip();subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor','c28c3fff',rev],check=True)
def get(path):return subprocess.check_output(['git','-C',str(C),'show',rev+':'+path])
def sha(b):return hashlib.sha256(b).hexdigest()
items=[]
for folder in folders:
 base='artifacts/'+folder+'/';index=json.loads(get(base+'runtime-index.json'));payload=get(base+'runtime.tar.gz')
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
  chosen=[x for x in index if x['path'] in ('build/'+folder+'/probe','build/'+folder+'/libmetalrobo.dylib')]
  if len(chosen)!=2:raise SystemExit('runtime index incomplete '+folder)
  for x in chosen:
   p=R/x['path'];f=t.extractfile(x['path']);assert f is not None;archived=f.read();local=p.read_bytes();assert sha(archived)==x['sha256']==sha(local) and len(local)==x['bytes'];h=subprocess.run(['/usr/sbin/lsof','-t',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   if h.returncode not in (0,1) or h.stdout.strip():raise SystemExit('runtime is in use or cannot be checked: '+str(p))
   items.append({'path':x['path'],'archive':base+'runtime.tar.gz','sha256':x['sha256'],'bytes':len(local)})
for x in items:(R/x['path']).unlink()
out={'remote_commit':rev,'verification':'fetched independently, restored tar members and compared sha256 and size against runtime index and local copies; checked open handles','retired':items};(R/'artifacts/crow-v15-runtime-retirement-followup-20261111/retired.json').write_text(json.dumps(out,indent=2)+'\n');print('verified and retired',len(items),'task-owned binaries;',sum(x['bytes'] for x in items),'bytes; archived runtimes remain in Git')
