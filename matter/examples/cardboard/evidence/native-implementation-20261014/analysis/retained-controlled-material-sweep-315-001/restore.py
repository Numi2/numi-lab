from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
a.directory.mkdir(exist_ok=False,parents=True)
for name,v in m['files'].items():
 data=(b/v['compressed_name']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==v['compressed_sha256']
 raw=gzip.decompress(data)
 assert len(raw)==v['raw_bytes'] and hashlib.sha256(raw).hexdigest()==v['raw_sha256']
 rows=[json.loads(line) for line in raw.splitlines()]
 assert rows[0]['rows']==rows[-1]['rows']==len(rows)-2==1536
 with (a.directory/name).open('xb') as f:f.write(raw)
print(json.dumps(dict(restored_files=len(m['files']),samples_per_policy=1536)))
