from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd(); C=Path('/tmp/crow-restore-check-20261110')
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True)
rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
if not rev.startswith('db90f62d'): raise SystemExit('unexpected remote revision '+rev)
def sha(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
items=[]
for folder,names,exts in [('crow-v15-full-substep-sensitivity-20261110',('j-original-8-rebound',),('json','tsv'))]:
 for name in names:
  for ext in exts:
   raw=f'build/crow-v15-long-perch-host-20261110/{name}.{ext}'; arc=f'artifacts/{folder}/{name}.{ext}.gz'
   p=subprocess.Popen(['git','-C',str(C),'show',rev+':'+arc],stdout=subprocess.PIPE)
   with gzip.GzipFile(fileobj=p.stdout,mode='rb') as f: a=sha(f)
   if p.wait(): raise SystemExit('archive retrieval failed '+arc)
   with (R/raw).open('rb') as f: b=sha(f)
   if a!=b: raise SystemExit('restoration mismatch '+arc)
   items.append({'raw':raw,'archive':arc,'sha256':b,'bytes':(R/raw).stat().st_size})
for x in items: (R/x['raw']).unlink()
(R/'artifacts/crow-v15-restoration-substeps-20261110/retired.json').write_text(json.dumps({'remote_commit':rev,'retired':items},indent=2)+'\n')
print('restored and verified',len(items),'raw duplicates; removed',sum(x['bytes'] for x in items),'bytes')
