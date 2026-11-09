from pathlib import Path
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision='095e8e25';token=subprocess.check_output(['gh','auth','token'],text=True).strip();items=[]
d=json.loads((R/'artifacts/crow-conditional-landing-native-20261109/trace-manifest.json').read_text())
for x in d['rows']:
 items.append({'raw':f"build/crow-conditional-landing-native-20261109/{x['name']}",'archive':f"artifacts/crow-conditional-landing-native-20261109/{x['name']}.gz",'sha256':x['sha256'],'bytes':x['bytes']})
d=json.loads((R/'artifacts/crow-unused-layout-confirmation-round4-20261109/summary.json').read_text())
for x in d['rows']:
 items.append({'raw':f"build/crow-unused-layout-confirmation-round4-20261109/{x['layout']}-lane1.tsv",'archive':f"artifacts/crow-unused-layout-confirmation-round4-20261109/{x['layout']}/lane1.tsv.gz",'sha256':x['trace_sha256'],'bytes':x['trace_bytes']})
rows=[]
for x in items:
 req=urllib.request.Request(f"https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{x['archive']}",headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
 h=hashlib.sha256();size=0
 with urllib.request.urlopen(req,timeout=120) as response:
  with gzip.GzipFile(fileobj=response) as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
 p=R/x['raw'];g=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):g.update(b)
 assert h.hexdigest()==g.hexdigest()==x['sha256'] and p.stat().st_size==size==x['bytes']
 z=subprocess.run(['lsof','--',str(p)],capture_output=True,text=True);assert z.returncode==1 and not z.stdout
 os.remove(p);rows.append({'archive_path':x['archive'],'sha256':x['sha256'],'raw_bytes':size,'removed_raw_copy':x['raw']})
 (R/'artifacts/crow-final-remote-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'rows':rows,'retired_bytes':sum(x['raw_bytes'] for x in rows),'method':'independent GitHub HTTPS decompression, hash and open-handle check before deletion'},indent=2)+'\n')
print('verified and retired',len(rows),'traces',sum(x['raw_bytes'] for x in rows),'bytes',flush=True)
