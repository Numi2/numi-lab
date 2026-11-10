from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');branch='agent/american-crow-rl-20261107';subprocess.run(['git','-C',str(C),'fetch','--quiet','origin',branch],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
if not rev.startswith('b8938ba7'):raise SystemExit('unexpected remote revision '+rev)
def sha(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
items=[]
for directory,name in [('crow-v15-hybrid-route-probe-20261110','development-j-hybrid-sat-capacity'),('crow-v15-hybrid-approach-20261110','development-j-approach')]:
 for ext in ('json','tsv'):
  raw=R/'build/crow-v15-hybrid-route-probe-20261110'/(name+'.'+ext);arc='artifacts/'+directory+'/'+name+'.'+ext+'.gz';p=subprocess.Popen(['git','-C',str(C),'show',rev+':'+arc],stdout=subprocess.PIPE)
  with gzip.GzipFile(fileobj=p.stdout,mode='rb') as f:a=sha(f)
  if p.wait():raise SystemExit('archive retrieval failed')
  with raw.open('rb') as f:b=sha(f)
  if a!=b:raise SystemExit('restoration mismatch '+arc)
  items.append({'raw':str(raw.relative_to(R)),'archive':arc,'sha256':b,'bytes':raw.stat().st_size})
for x in items:(R/x['raw']).unlink()
(R/'artifacts/crow-v15-hybrid-archive-check-20261110/retired.json').write_text(json.dumps({'remote_commit':rev,'retired':items},indent=2)+'\n');print('restored and verified',len(items),'raw duplicates; removed',sum(x['bytes'] for x in items),'bytes')
