from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
P=Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((P/'publication-manifest-065.json').read_text())
listing=P/'payload-files-065.json';assert sha(listing)==m['payload_file_list_sha256']
rows=json.loads(listing.read_text())['files']
for r in rows:
 f=P/r['path'];assert f.stat().st_size==r['size_bytes']and sha(f)==r['sha256'],r['path']
for r in m['canonical_paths']:
 f=a.source_root/r['path'];assert f.stat().st_size==r['size_bytes']and sha(f)==r['sha256'],r['path']
assert m['accepted_physical_steps']==0 and m['native_or_physical_admission']is False
assert '100% tests passed out of 3' in(P/'root-multiobject-ctest.log').read_text()
assert m['legacy_v19_sha256']=='f838b00b5a2515e17bc055390c8ffa94a737b3b422bfd5e9d9bfe1c82f02f42e'
print(json.dumps({'status':'PASS_HASHES_10_CANONICAL_FILES_3_FOCUSED_TESTS','payload_files':len(rows),'canonical_files':len(m['canonical_paths']),'physical_qualification':False}))

