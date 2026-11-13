import gzip,hashlib,json,subprocess,urllib.request
from pathlib import Path
commit=Path('/tmp/crow-continuation-commit').read_text().strip();token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for artifact,tag in [('crow-v15-pronation-phase-screen-20261113','phase'),('crow-v15-preload-foot-placement-20261113','preload')]:
 for arm in ('push-wing','hold-wing'):
  run=f'{arm}-20261126-rebound-{tag}'
  for kind in ('trace.tsv','replay.json'):
   rel=f'artifacts/{artifact}/{run}/{kind}.gz'
   remote=f'/Users/n/Developer/numi-crow-rl-20261107/build/{artifact}/{run}-{kind}'
   req=urllib.request.Request(f'https://api.github.com/repos/Numi2/numi-lab/contents/{rel}?ref={commit}',headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verify'})
   h=hashlib.sha256()
   with urllib.request.urlopen(req,timeout=120) as resp,gzip.GzipFile(fileobj=resp,mode='rb') as data:
    for b in iter(lambda:data.read(1<<20),b''):h.update(b)
   r=subprocess.run(['ssh','-4','-o','HostName=192.168.1.141','macmini','shasum','-a','256',remote],text=True,capture_output=True,check=True)
   current=r.stdout.split()[0];archived=h.hexdigest();rows.append({'host':'macmini','path':remote,'archive_path':rel,'commit':commit,'sha256':current,'restored_sha256':archived,'match':current==archived})
   print(artifact,run,kind,'MATCH' if current==archived else 'MISMATCH',flush=True)
   if current!=archived:raise SystemExit('restore verification failed')
Path('/tmp/crow-retirement-second-verified.json').write_text(json.dumps(rows,indent=2)+'\n')
