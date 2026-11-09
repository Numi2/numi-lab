from pathlib import Path
import gzip,hashlib,json,os,subprocess,urllib.request
R=Path.cwd();revision=subprocess.check_output(['git','rev-parse','aebe1f36'],text=True).strip()
token=subprocess.check_output(['gh','auth','token'],text=True).strip()
rel='artifacts/crow-dual-expert-native-integration-20261109/shadow.rollout.gz'
expected=json.loads((R/'artifacts/crow-dual-expert-native-integration-20261109/teacher-shadow-results.json').read_text())['raw_rollout_sha256']
req=urllib.request.Request(f'https://raw.githubusercontent.com/Numi2/numi-lab/{revision}/{rel}',headers={'Authorization':'Bearer '+token,'Accept':'application/octet-stream'})
h=hashlib.sha256();size=0
with urllib.request.urlopen(req,timeout=180) as response:
 with gzip.GzipFile(fileobj=response) as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
assert h.hexdigest()==expected
raw=R/'build/crow-dual-expert-native-integration-20261109/shadow.rollout'
g=hashlib.sha256()
with raw.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):g.update(block)
assert g.hexdigest()==expected and raw.stat().st_size==size
p=subprocess.run(['lsof','--',str(raw)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
os.remove(raw)
(R/'artifacts/crow-shadow-remote-restore-20261109/results.json').write_text(json.dumps({'remote_revision':revision,'raw_sha256':expected,'retired_bytes':size,'method':'independent GitHub HTTPS decompression, SHA-256 and open-handle check before deleting temporary raw copy'},indent=2)+'\n')
print('verified remote shadow rollout and retired',size,'bytes',flush=True)
