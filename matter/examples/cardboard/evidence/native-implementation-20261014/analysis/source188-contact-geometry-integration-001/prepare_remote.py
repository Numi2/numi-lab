from pathlib import Path
import hashlib,json,shutil,datetime
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'snapshots/source-portable-paper-reference-185';S=B/'snapshots/source-scaled-contact-geometry-188';O=B/'analysis/source188-contact-geometry-integration-001'
expected={'matter/CMakeLists.txt': '47564b029c911a1949caf10f60551550df292c2968df25082118fb4f66b18569', 'matter/src/metal/contact.metalinc': 'a102749ef13ac17276f2e3b014f362ccfa7967327340bbf4677be4b6e2cdb048', 'matter/tools/cardboard_full_contact_armijo_check.mm': 'b2e70293dc5f1c26ae12d43929217427d76319e575a27900af271ce91883a096', 'matter/tools/cardboard_contact_geometry_check.metal': '29fb8764e3b521a7d4234e15da249acbf15b5790e29a139bd2e2b1878054057f', 'matter/tools/cardboard_contact_geometry_check.mm': '31f1942625ee8d37cee6b5bd304c71026f33fd1e57280c8a2ce231cb9473f987'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((P/'frozen-source.json').read_text());support=json.loads((P/'configure-support-source.json').read_text())
for n,h in f['source_sha256'].items():assert sha(P/n)==h,n
for n,h in support['files'].items():assert sha(P/n)==h,n
assert not S.exists();shutil.copytree(P,S)
for n,h in expected.items():
 p=O/'inbox'/n;assert sha(p)==h,n
 shutil.copy2(p,S/n);f['source_sha256'][n]=h
f['base_snapshot']=str(P);f['scope']='Source185 plus scale-correct production contact feature helpers and native analytical regression; no tolerance relaxation.'
(S/'frozen-source.json').write_text(json.dumps(f,indent=2)+'\n')
assert len(f['source_sha256'])==814
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
for n,h in support['files'].items():assert sha(S/n)==h,n
receipt={'frozen_sha256':sha(S/'frozen-source.json'),'selected_files':814,'support_files':len(support['files']),'all_transferred_and_verified':True,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected':expected}
(O/'complete-transfer-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
