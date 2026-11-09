from pathlib import Path
import hashlib,json,subprocess,os
R=Path.cwd();rows=[]
groups=[
 ('crow-mixed-stage3-dagger-20261109',['training','held-out-a','held-out-b'],'artifacts/crow-stage3-remote-restore-20261109/results.json'),
 ('crow-expanded-dagger-20261109',['layout-c','layout-d'],'artifacts/crow-expanded-remote-restore-20261109/results.json'),
 ('crow-expanded-gate-dagger-20261109',['layout-c','layout-d'],'artifacts/crow-expanded-remote-restore-20261109/results.json')]
for directory,batches,proof in groups:
 evidence=json.loads((R/proof).read_text());verified={z['path']:z['raw_sha256'] for z in evidence['rows']}
 for batch in batches:
  rel=f'artifacts/{directory}/{batch}/rollout.bin.gz';raw=R/f'build/{directory}/{batch}.rollout'
  manifest=json.loads((R/f'artifacts/{directory}/{batch}/manifest.json').read_text())
  assert verified[rel]==manifest['raw_pack_sha256'];h=hashlib.sha256()
  with raw.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  assert h.hexdigest()==verified[rel]
  p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
  size=raw.stat().st_size;os.remove(raw)
  rows.append({'path':str(raw.relative_to(R)),'sha256':h.hexdigest(),'bytes':size,'independent_restore_proof':proof})
proof='artifacts/crow-third-remote-restore-20261109/results.json';evidence=json.loads((R/proof).read_text());verified={z['path']:z['raw_sha256'] for z in evidence['rows']}
for schedule in ['early','late','terminal']:
 for batch in ['layout-e','layout-f']:
  rel=f'artifacts/crow-third-dagger-20261109/{schedule}/{batch}/rollout.bin.gz';raw=R/f'build/crow-third-dagger-20261109/{schedule}/{batch}.rollout'
  manifest=json.loads((R/f'artifacts/crow-third-dagger-20261109/{schedule}/{batch}/manifest.json').read_text())
  assert verified[rel]==manifest['raw_pack_sha256'];h=hashlib.sha256()
  with raw.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  assert h.hexdigest()==verified[rel]
  p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
  size=raw.stat().st_size;os.remove(raw)
  rows.append({'path':str(raw.relative_to(R)),'sha256':h.hexdigest(),'bytes':size,'independent_restore_proof':proof})
print('retired bytes',sum(x['bytes'] for x in rows),flush=True)
(R/'artifacts/crow-fourth-temp-input-retirement-20261109/results.json').write_text(json.dumps({'rows':rows,'retired_bytes':sum(x['bytes'] for x in rows),'method':'compared with independently restored GitHub artifact, checked for open handles'},indent=2)+'\n')
