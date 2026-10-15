#!/usr/bin/env python3
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('output_directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
for name,d in m['files'].items():
 q=b/name
 if hashlib.sha256(q.read_bytes()).hexdigest()!=d['sha256']:raise ValueError('package input changed: '+name)
a.output_directory.mkdir(parents=True,exist_ok=False)
raw=gzip.decompress((b/m['compressed_payload']).read_bytes())
if hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:raise ValueError('payload hash mismatch')
(a.output_directory/m['payload_filename']).write_bytes(raw)
shutil.copy2(b/m['manifest_filename'],a.output_directory/m['manifest_filename'])
import importlib.util
s=importlib.util.spec_from_file_location('coverage',b/'coupled_contact_coverage.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
print(json.dumps(v.validate_capture(a.output_directory/m['manifest_filename']),indent=2))
