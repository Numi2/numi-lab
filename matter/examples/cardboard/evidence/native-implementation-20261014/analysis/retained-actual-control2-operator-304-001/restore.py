from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
data=(b/m['compressed_name']).read_bytes()
assert hashlib.sha256(data).hexdigest()==m['compressed_sha256']
raw=gzip.decompress(data)
assert len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256']
records=[json.loads(line) for line in raw.splitlines()]
assert records[0]['root']==2 and records[0]['newton_iteration']==63
with a.destination.open('xb') as f:f.write(raw)
print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=len(records))))
