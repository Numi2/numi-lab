#!/usr/bin/env python3
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,shutil
p=argparse.ArgumentParser();p.add_argument('output_directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent
def reject(token):raise ValueError('nonstandard JSON constant: '+token)
m=json.loads((b/'package.json').read_text(),parse_constant=reject)
for name,d in m['files'].items():
    if hashlib.sha256((b/name).read_bytes()).hexdigest()!=d['sha256']:
        raise ValueError('package input changed: '+name)
a.output_directory.mkdir(parents=True,exist_ok=False)
raw=gzip.decompress((b/m['compressed_payload']).read_bytes())
if hashlib.sha256(raw).hexdigest()!=m['payload_sha256']:raise ValueError('payload hash mismatch')
(a.output_directory/m['payload_filename']).write_bytes(raw)
manifest=a.output_directory/m['manifest_filename']
shutil.copy2(b/m['manifest_filename'],manifest)
def module(name,file):
    s=importlib.util.spec_from_file_location(name,b/file);v=importlib.util.module_from_spec(s)
    s.loader.exec_module(v);return v
strict=module('strict_capture','coupled_capture_reader.py').Capture(manifest)
assert strict.payload_path.resolve()==(a.output_directory/m['payload_filename']).resolve()
coverage=module('coverage','coupled_contact_coverage.py').validate_capture(manifest)
print(json.dumps(dict(strict=strict.summary(),coverage=coverage,
    restored_adjacent_payload_verified=True),indent=2,allow_nan=False))
