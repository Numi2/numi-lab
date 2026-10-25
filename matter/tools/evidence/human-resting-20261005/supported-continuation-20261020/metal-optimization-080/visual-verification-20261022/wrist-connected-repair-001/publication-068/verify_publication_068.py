from pathlib import Path
import hashlib,json,gzip,argparse
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
P=Path(__file__).resolve().parent
for r in json.loads((P/'source-artifacts.json').read_text())['files']:
 b=(P/r['path']).read_bytes();b=gzip.decompress(b)if r['gzip']else b
 assert len(b)==r['uncompressed_size_bytes']and hashlib.sha256(b).hexdigest()==r['uncompressed_sha256'],r['path']
m=json.loads((P/'canonical-manifest.json').read_text())
for r in m['canonical_paths']:
 b=(a.source_root/r['path']).read_bytes()
 assert len(b)==r['size_bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
r=json.loads((P/'evidence/root-connected-041-review.json').read_text())
assert r['nonpositive_cells']==0 and r['quality_below_1e_5']==0 and r['boundary_edges_not_incidence2']==0
assert r['edge_bank_crossing_groups']==437 and r['contained_nodes']==15
assert r['accepted_physical_steps']==0 and not r['anatomy_qualified'] and not r['physical_qualification']
print('PASS original evidence hashes, canonical files, static gates and failed-clearance labels')
