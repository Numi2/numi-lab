from pathlib import Path
import argparse,gzip,hashlib,json,shutil
from trial_capture_reader import Capture
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
for name,v in m['artifacts'].items():
 data=(b/name).read_bytes()
 assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'],name
raw=gzip.decompress((b/'private-trial.coupled.bin.gz').read_bytes())
assert len(raw)==m['raw_payload_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_payload_sha256']
a.directory.mkdir(exist_ok=False,parents=True)
with (a.directory/'private-trial.coupled.bin').open('xb') as f:f.write(raw)
with (a.directory/'private-trial.coupled.json').open('xb') as f:f.write((b/'private-trial.coupled.json').read_bytes())
c=Capture(a.directory/'private-trial.coupled.json')
print(json.dumps(c.summary(),indent=2,allow_nan=False))
