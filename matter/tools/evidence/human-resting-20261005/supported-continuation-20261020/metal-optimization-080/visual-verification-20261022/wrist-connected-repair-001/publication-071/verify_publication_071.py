#!/usr/bin/env python3
"""Verify the retained source and canonical bytes; does not admit geometry."""
from pathlib import Path
import argparse,gzip,hashlib,json
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--canonical-root',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent
 m=json.loads((root/'source-artifacts.json').read_text())
 for r in m['files']:
  f=(root/r['path']).resolve();assert f.is_relative_to(root)
  b=f.read_bytes();assert sha(b)==r['stored_sha256'],str(f)
  if r['gzip']:b=gzip.decompress(b)
  assert len(b)==r['uncompressed_size_bytes'] and sha(b)==r['uncompressed_sha256'],str(f)
 c=json.loads((root/'canonical-manifest.json').read_text())
 for r in c['canonical_paths']:
  f=(a.canonical_root/r['path']).resolve();assert f.is_relative_to(a.canonical_root.resolve())
  assert sha(f.read_bytes())==r['sha256'],str(f)
 print(json.dumps({'status':'PASS_RETAINED_BYTES','canonical_files':len(c['canonical_paths']),'evidence_files':len(m['files']),'accepted_physical_steps':0,'geometry_admitted':False}))
if __name__=='__main__':main()
