#!/usr/bin/env python3
from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('output_directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent
def reject(s):raise ValueError(s)
m=json.loads((b/'package.json').read_text(),parse_constant=reject)
a.output_directory.mkdir(parents=True,exist_ok=False)
for item in m['entries']:
    relative=Path(item['restored_relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid restore path')
    packed=(b/item['compressed_filename']).read_bytes()
    if hashlib.sha256(packed).hexdigest()!=item['compressed_sha256']:raise ValueError('packed hash mismatch')
    raw=gzip.decompress(packed)
    if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('restored hash mismatch')
    target=a.output_directory/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
print(json.dumps(dict(restored_files=len(m['entries']),all_exact_hashes_verified=True,
    physical_validation=False,accepted_checkpoint=False)))
