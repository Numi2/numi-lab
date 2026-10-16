from pathlib import Path
import gzip,hashlib,json,subprocess,datetime
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032';P='matter/examples/cardboard/evidence/native-implementation-20261014/';D=R/P
O=Q/'analysis/publication032-independent-byte-audit-003';assert not O.exists();O.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
original=git('show','HEAD:'+P+'summary.json');old=json.loads(original)
history=(D/'summary-reference031.json').read_bytes();assert history==original
s=json.loads((D/'summary.json').read_text());assert len(old['records'])==1607 and s['records'][:1607]==old['records']
b=json.loads((D/'publication-binding-reference032.json').read_text())
assert sha((D/'summary.json').read_bytes())==b['summary_sha256']
assert sha((D/'publication-binding-reference031.json').read_bytes())==b['prior_binding_sha256']
assert not b['physical_validation'] and len(s['verdicts'])==6 and all(v['status'] not in ('pass','qualified','complete') for v in s['verdicts'].values())
new=s['records'][1607:];assert [r['case'] for r in new]==b['cases']
checked=0;rawbytes=0;expected=set(b['changed_source_files'])
for rec in new:
 for f in rec['files']:
  key='repository_copy' if 'repository_copy' in f else 'repository_compressed_copy'
  p=D/f[key];payload=p.read_bytes();data=payload
  if key=='repository_compressed_copy':
   assert sha(payload)==f['compressed_sha256'] and len(payload)==f['compressed_bytes'];data=gzip.decompress(payload)
  assert len(data)==f['bytes'] and sha(data)==f['sha256'],p
  assert Path(f['external_path']).read_bytes()==data,f['external_path']
  checked+=1;rawbytes+=len(data);expected.add(P+f[key])
assert checked==b['copied_files']==604
S=Q/'snapshots/source-scaled-contact-geometry-188-buildfix002';frozen=(S/'frozen-source.json').read_bytes()
assert sha(frozen)==b['source188_frozen_sha256']
sources=json.loads(frozen)['source_sha256'];assert len(sources)==814
for n,h in sources.items():assert sha((R/n).read_bytes())==h,n
buildpath=Q/'build-source188-002/source-build.json';assert sha(buildpath.read_bytes())==b['source188_host_build_sha256']
build=json.loads(buildpath.read_text());assert build['success'] and build['source_sha256']==sources
for n,h in build['artifacts_sha256'].items():assert sha(Path(n).read_bytes())==h,n
for n,h in build['build_metadata_sha256'].items():assert sha((Q/'build-source188-002'/n).read_bytes())==h,n
cpu=json.loads((Q/'analysis/source188-native-validation-001/cpu-checks/receipt.json').read_text())
assert cpu['exit_code']==0,cpu
geom=json.loads((Q/'native/contact-scaled-geometry-504/receipt.json').read_text());assert geom['exit_code']==0
fixture=json.loads((Q/'native/contact-scaled-geometry-fixture-505/receipt.json').read_text());assert fixture['exit_code']!=0
expected.update(P+n for n in ['summary.json','summary-reference031.json','publication-binding-reference032.json'])
pathlist=(Q/'publication032-staged-paths.nul').read_bytes();paths={x.decode() for x in pathlist.split(b'\0') if x}
assert len(paths)==625 and paths==expected
actual={x.decode() for x in git('diff','--name-only','-z').split(b'\0') if x}|{x.decode() for x in git('ls-files','--others','-z').split(b'\0') if x}
assert paths==actual,(paths-actual,actual-paths)
assert not git('diff','--cached','--name-only','-z')
assert not git('diff','--check')
result={'schema':'numi.cardboard.publication032-independent-byte-audit.v1','verdict':'PASS','historical_records_unchanged':1607,'new_cases':len(new),'payloads_verified':checked,'original_payload_bytes':rawbytes,'source_files_verified':len(sources),'summary_sha256':sha((D/'summary.json').read_bytes()),'binding_sha256':sha((D/'publication-binding-reference032.json').read_bytes()),'staged_pathlist_sha256':sha(pathlist),'staged_paths':len(paths),'source188_frozen_sha256':sha(frozen),'build_artifacts_verified':len(build['artifacts_sha256']),'native_geometry_exit_code':geom['exit_code'],'loaded_contact_exit_code':fixture['exit_code'],'full_protocol_qualification':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'report.json').write_text(json.dumps(result,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
