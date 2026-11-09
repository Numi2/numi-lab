from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision='d3d3638d';token=subprocess.check_output(['gh','auth','token'],text=True).strip()
items=json.loads((R/'artifacts/crow-landing-trace-archive-20261109/manifest.json').read_text())['rows']
def verify(x):
 req=urllib.request.Request(f"https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{x['archive_path']}",headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
 h=hashlib.sha256();size=0
 with urllib.request.urlopen(req,timeout=120) as response:
  with gzip.GzipFile(fileobj=response) as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
 p=R/x['raw_path'];assert h.hexdigest()==x['sha256'] and p.stat().st_size==size==x['bytes']
 g=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):g.update(b)
 assert g.hexdigest()==h.hexdigest()
 f=subprocess.run(['lsof','--',str(p)],capture_output=True,text=True);assert f.returncode==1 and not f.stdout
 os.remove(p)
 return {'archive_path':x['archive_path'],'sha256':x['sha256'],'raw_bytes':size,'removed_raw_copy':x['raw_path']}
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(verify,x) for x in items]):
  rows.append(future.result())
  (R/'artifacts/crow-landing-trace-remote-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'rows':rows,'retired_bytes':sum(x['raw_bytes'] for x in rows),'method':'independent GitHub HTTPS decompression, SHA-256 and open-handle check before deletion'},indent=2)+'\n')
  if len(rows)%10==0:print(len(rows),'traces verified and retired',flush=True)
print('done',len(rows),sum(x['raw_bytes'] for x in rows),flush=True)
