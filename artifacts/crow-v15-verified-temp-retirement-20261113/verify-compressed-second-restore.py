import hashlib,json,subprocess,urllib.request
from pathlib import Path
commit=Path('/tmp/crow-continuation-commit').read_text().strip();token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for artifact,tag in [('crow-v15-front-toe-preload-screen-20261113','toepreload')]:
 for arm in ('push-wing','hold-wing'):
  rel=f'artifacts/{artifact}/{arm}-20261126-rebound-{tag}/run.stderr.gz';path=f'/Users/n/Developer/numi-crow-rl-20261107/{rel}'
  req=urllib.request.Request(f'https://api.github.com/repos/Numi2/numi-lab/contents/{rel}?ref={commit}',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verify'})
  h=hashlib.sha256()
  with urllib.request.urlopen(req,timeout=180) as resp:
   for b in iter(lambda:resp.read(1<<20),b''):h.update(b)
  cur=subprocess.check_output(['ssh','-4','-o','HostName=192.168.1.141','macmini','shasum','-a','256',path],text=True).split()[0];restored=h.hexdigest()
  row={'host':'macmini','path':path,'archive_path':rel,'commit':commit,'sha256':cur,'restored_sha256':restored,'match':cur==restored};rows.append(row)
  print(artifact,arm,'MATCH' if cur==restored else 'MISMATCH',flush=True)
  if cur!=restored:raise SystemExit('archive verification failed')
Path('/tmp/crow-compressed-second-verified.json').write_text(json.dumps(rows,indent=2)+'\n')
