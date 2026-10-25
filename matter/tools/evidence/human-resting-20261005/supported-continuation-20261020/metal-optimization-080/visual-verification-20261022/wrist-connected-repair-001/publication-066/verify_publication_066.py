from pathlib import Path
import argparse,json,hashlib,gzip
ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path);args=ap.parse_args()
P=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
m=json.loads((P/'publication-manifest-066.json').read_text());listing=P/'payload-files-066.json'
assert sha(listing.read_bytes())==m['payload_file_list_sha256']
rows=json.loads(listing.read_text())['files']
assert len(rows)==m['payload_file_count'] and sum(r['size_bytes'] for r in rows)==m['payload_bytes']
for r in rows:
 p=P/r['path'];assert p.stat().st_size==r['size_bytes'] and sha(p.read_bytes())==r['sha256'],r['path']
for r in json.loads((P/'source-artifacts.json').read_text())['files']:
 b=(P/r['path']).read_bytes();b=gzip.decompress(b)if r['gzip']else b
 assert len(b)==r['uncompressed_size_bytes'] and sha(b)==r['uncompressed_sha256'],r['path']
if args.source_root:
 for r in m['canonical_paths']:
  p=args.source_root/r['path'];assert sha(p.read_bytes())==r['sha256'],r['path']
C=json.loads((P/'evidence/route-audit-counter-fix-030/comparison.json').read_text())
for row in C['cases']:
 b=gzip.decompress((P/'evidence/route-audit-counter-fix-030'/row['case']/'tet-edge-report.json.gz').read_bytes())
 assert sha(b)==row['report_sha256']
 r=json.loads(b);w=r['witnesses']
 assert r['counts']['proper_hits']==sum(x['classification']=='proper_interior_hit'for x in w)
 assert r['counts']['boundary_hits']==sum(x['classification']=='boundary_hit'for x in w)
 assert r['edge_bank_groups_with_hits']==len({(x['edge_index'],x['body'],x['bone_stable_id'])for x in w})
 assert r['edges_with_hits']==len({x['edge_index']for x in w})
expected=[2363,1868,1787];assert [r['edge_bank_groups_with_hits']for r in C['cases']]==expected
Q=json.loads((P/'evidence/root-independent-static-query-028.json').read_text())
assert not Q['inside_witnesses']and Q['ambiguous_banks']==0 and Q['invalid_banks']==0
assert Q['accepted_physical_steps']==0 and not Q['physical_qualification']
LP=json.loads((P/'evidence/o12-outward-repair-028-same-entry-halfspace-lp-002/report.json').read_text())
assert LP['lp']['success']and not LP['lp']['emitted_f32_quality_pass_1e-5']and LP['lp']['emitted_f32_min_global_q']<0
review=json.loads((P/'root-review.json').read_text());assert not review['geometry_candidate_admitted'] and not review['physical_qualification']
print(json.dumps({'status':'PASS','files':len(rows),'restored_gzip_artifacts':sum(r['gzip']for r in json.loads((P/'source-artifacts.json').read_text())['files']),'canonical_source_verified':bool(args.source_root),'physical_qualification':False}))

