from pathlib import Path
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision=subprocess.check_output(['git','rev-parse','08d11d5a'],text=True).strip()
token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for batch in ['a1','a2','a3']:
 rel=f'artifacts/crow-production-gate-dagger-20261109/{batch}/rollout.bin.gz'
 manifest=json.loads((R/f'artifacts/crow-production-gate-dagger-20261109/{batch}/manifest.json').read_text())
 req=urllib.request.Request(f'https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{rel}',headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
 h=hashlib.sha256();size=0
 with urllib.request.urlopen(req,timeout=180) as response:
  with gzip.GzipFile(fileobj=response) as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
 assert h.hexdigest()==manifest['raw_pack_sha256'] and size==manifest['raw_pack_bytes']
 raw=R/f'build/crow-production-gate-dagger-20261109/{batch}.rollout'
 assert raw.is_file() and raw.stat().st_size==size
 g=hashlib.sha256()
 with raw.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):g.update(block)
 assert g.hexdigest()==h.hexdigest()
 p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True)
 assert p.returncode==1 and not p.stdout,(raw,p.returncode,p.stdout,p.stderr)
 os.remove(raw)
 rows.append({'path':rel,'raw_bytes':size,'raw_sha256':h.hexdigest(),'removed_raw_copy':str(raw.relative_to(R))})
 print(batch,'restored and verified',flush=True)
(R/'artifacts/crow-production-gate-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'method':'independent GitHub HTTPS decompression and SHA-256, local hash and open-handle check before deletion','rows':rows,'total_retired_bytes':sum(x['raw_bytes'] for x in rows)},indent=2)+'\n')
