from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request
R=Path.cwd(); revision=subprocess.check_output(['git','rev-parse','903352e5'],text=True).strip()
token=subprocess.check_output(['gh','auth','token'],text=True).strip()
rows=[]
for kind in ['gate-full-student-dagger','gate-second-student-dagger']:
 for course in ['training','held-out-a','held-out-b']:
  rel=f'artifacts/crow-{kind}-20261109/{course}/rollout.bin.gz'
  manifest=json.loads((R/f'artifacts/crow-{kind}-20261109/{course}/manifest.json').read_text())
  req=urllib.request.Request(f'https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{rel}',headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
  h=hashlib.sha256();size=0
  with urllib.request.urlopen(req,timeout=180) as response:
   with gzip.GzipFile(fileobj=response) as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
  ok=h.hexdigest()==manifest['raw_pack_sha256'] and size==manifest['raw_pack_bytes']
  rows.append({'path':rel,'raw_bytes':size,'raw_sha256':h.hexdigest(),'matches_manifest':ok})
  print(kind,course,ok,flush=True)
  if not ok:raise RuntimeError('Remote restoration mismatch')
(R/'artifacts/crow-gate-remote-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'method':'independent HTTPS download, gzip decompression and SHA-256 verification; no downloaded copy retained','rows':rows},indent=2)+'\n')
