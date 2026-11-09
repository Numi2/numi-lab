from pathlib import Path
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision='b52f9147';rel='artifacts/crow-four-expert-native-integration-20261109/shadow.rollout.gz'
token=subprocess.check_output(['gh','auth','token'],text=True).strip()
req=urllib.request.Request(f'https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{rel}',headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
h=hashlib.sha256();size=0
with urllib.request.urlopen(req,timeout=180) as response:
 with gzip.GzipFile(fileobj=response) as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
raw=R/'build/crow-four-expert-native-integration-20261109/shadow.rollout'
expected=json.loads((R/'artifacts/crow-four-expert-native-integration-20261109/teacher-shadow-results.json').read_text())['raw_rollout_sha256']
g=hashlib.sha256()
with raw.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):g.update(block)
assert g.hexdigest()==h.hexdigest()==expected and raw.stat().st_size==size
p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
os.remove(raw)
out={'remote_revision':revision,'path':rel,'raw_sha256':expected,'raw_bytes':size,'removed_raw_copy':str(raw.relative_to(R)),'method':'independent GitHub HTTPS decompression, hash and open-handle check before deletion'}
(R/'artifacts/crow-four-shadow-remote-restore-20261109/results.json').write_text(json.dumps(out,indent=2)+'\n')
print('verified and retired',size,'bytes',flush=True)
