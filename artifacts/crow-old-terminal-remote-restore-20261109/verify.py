from pathlib import Path
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision='51cdc203';token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
kinds=['crow-goal-memory-demonstrations-20261109','crow-goal-memory-terminal-dagger-20261109','crow-deterministic-terminal-dagger-20261109','crow-full-student-dagger-20261109']
for kind in kinds:
 for course in ['training','held-out-a','held-out-b']:
  rel=f'artifacts/{kind}/{course}/rollout.bin.gz';raw=R/f'build/{kind}/{course}.rollout'
  m=json.loads((R/f'artifacts/{kind}/{course}/manifest.json').read_text())
  req=urllib.request.Request(f'https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{rel}',headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
  h=hashlib.sha256();size=0
  with urllib.request.urlopen(req,timeout=180) as response:
   with gzip.GzipFile(fileobj=response) as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
  g=hashlib.sha256()
  with raw.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):g.update(block)
  assert g.hexdigest()==h.hexdigest()==m['raw_pack_sha256'] and raw.stat().st_size==size
  p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
  os.remove(raw)
  rows.append({'path':rel,'raw_sha256':h.hexdigest(),'raw_bytes':size,'removed_raw_copy':str(raw.relative_to(R))})
  (R/'artifacts/crow-old-terminal-remote-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'rows':rows,'retired_bytes':sum(x['raw_bytes'] for x in rows),'method':'independent GitHub HTTPS decompression, SHA-256 and open-handle check before deletion'},indent=2)+'\n')
  print(kind,course,'verified and retired',flush=True)
