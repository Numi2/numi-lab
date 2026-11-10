from pathlib import Path
import gzip,hashlib,json,subprocess
R=Path.cwd();C=Path('/tmp/crow-restore-check-20261110');expected='04f1d5c4';branch='agent/american-crow-rl-20261107';subprocess.run(['git','-C',str(C),'fetch','--quiet','origin',branch],check=True);rev=subprocess.check_output(['git','-C',str(C),'rev-parse','FETCH_HEAD'],text=True).strip()
if not rev.startswith(expected):raise SystemExit('unexpected remote tip '+rev)
def sha(f):
 h=hashlib.sha256()
 for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
d='crow-v15-foot-alignment-gate-20261110';retired=[]
for label in ['gate','zero_bias']:
 for ext in ['json','tsv']:
  name='development-reserved-final-j-'+label+'.'+ext;raw=R/'build'/d/name;arc='artifacts/'+d+'/'+name+'.gz';p=subprocess.Popen(['git','-C',str(C),'show',rev+':'+arc],stdout=subprocess.PIPE)
  with gzip.GzipFile(fileobj=p.stdout,mode='rb') as f:a=sha(f)
  if p.wait():raise SystemExit('archive read failed')
  with raw.open('rb') as f:b=sha(f)
  if a!=b:raise SystemExit('mismatch '+name)
  retired.append({'raw':str(raw.relative_to(R)),'archive':arc,'sha256':b,'bytes':raw.stat().st_size})
for x in retired:(R/x['raw']).unlink()
(R/'artifacts/crow-v15-foot-gate-archive-check-20261110/retired.json').write_text(json.dumps({'remote_commit':rev,'retired':retired},indent=2)+'\n');print('restored and matched',len(retired),'raw duplicates; retired',sum(x['bytes'] for x in retired),'bytes')
