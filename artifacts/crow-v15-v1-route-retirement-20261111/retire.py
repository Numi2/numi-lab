from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');specs={'crow-v15-v1-route-diagnostic-20261111': ('route',)}
subprocess.run(['git','-C',str(C),'fetch','--quiet','origin','agent/american-crow-rl-20261107'],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip();subprocess.run(['git','-C',str(C),'merge-base','--is-ancestor','e73b844d',rev],check=True)
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
items=[]
for folder,names in specs.items():
 for name in names:
  for ext,arc in [('json','json.gz'),('tsv','tsv.gz')]:
   raw=R/'build'/folder/(name+'.'+ext);archive='artifacts/'+folder+'/'+name+'.'+arc
   if not raw.is_file():raise SystemExit('missing local file '+str(raw))
   p=subprocess.Popen(['git','-C',str(C),'show',rev+':'+archive],stdout=subprocess.PIPE)
   with gzip.GzipFile(fileobj=p.stdout,mode='rb') as f:remote=digest(f)
   if p.wait():raise SystemExit('restore failed '+archive)
   with raw.open('rb') as f:local=digest(f)
   if local!=remote:raise SystemExit('hash mismatch '+str(raw))
   q=subprocess.run(['/usr/sbin/lsof','-t',str(raw)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   if q.returncode not in (0,1) or q.stdout.strip():raise SystemExit('in use or cannot inspect '+str(raw))
   items.append({'path':str(raw.relative_to(R)),'archive':archive,'sha256':local,'bytes':raw.stat().st_size})
for x in items:(R/x['path']).unlink()
(D:=R/'artifacts/crow-v15-v1-route-retirement-20261111'/'retired.json').write_text(json.dumps({'remote_commit':rev,'verification':'independent fetched Git restore, gzip decode, SHA-256 and open-handle checks','retired':items},indent=2)+'\n');print('verified and retired',len(items),'duplicate raw files,',sum(x['bytes'] for x in items),'bytes')
