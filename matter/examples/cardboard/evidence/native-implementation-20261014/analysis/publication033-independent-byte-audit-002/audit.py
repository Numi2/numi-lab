from pathlib import Path
import gzip,hashlib,json,subprocess,datetime
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');R=Q/'checkout-publication032';P='matter/examples/cardboard/evidence/native-implementation-20261014/';D=R/P
O=Q/'analysis/publication033-independent-byte-audit-002';assert not O.exists();O.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
assert git('rev-parse','HEAD').decode().strip()=='877cd6e69657c9a7b4e5032bcd475705bcc06c85'
original=git('show','HEAD:'+P+'summary.json');old=json.loads(original)
assert (D/'summary-reference032.json').read_bytes()==original
s=json.loads((D/'summary.json').read_text());assert len(old['records'])==1669 and s['records'][:1669]==old['records']
b=json.loads((D/'publication-binding-reference033.json').read_text())
assert sha((D/'summary.json').read_bytes())==b['summary_sha256']
assert sha((D/'publication-binding-reference032.json').read_bytes())==b['prior_binding_sha256']
assert not b['physical_validation'] and len(s['verdicts'])==6 and all(v['status'] not in ('pass','qualified','complete') for v in s['verdicts'].values())
new=s['records'][1669:];assert [r['case'] for r in new]==b['cases']
checked=rawbytes=0;expected=set(b['changed_source_files'])
for rec in new:
 for f in rec['files']:
  key='repository_copy' if 'repository_copy' in f else 'repository_compressed_copy'
  p=D/f[key];payload=p.read_bytes();data=payload
  if key=='repository_compressed_copy':
   assert sha(payload)==f['compressed_sha256'] and len(payload)==f['compressed_bytes'];data=gzip.decompress(payload)
  assert len(data)==f['bytes'] and sha(data)==f['sha256'],p
  assert Path(f['external_path']).read_bytes()==data,f['external_path']
  checked+=1;rawbytes+=len(data);expected.add(P+f[key])
assert checked==b['copied_files']
S=Q/'snapshots/source-contact-damped-memory-198-001';frozen=(S/'frozen-source.json').read_bytes()
assert sha(frozen)==b['source198_frozen_sha256']
sources=json.loads(frozen)['source_sha256'];assert len(sources)==816
for n,h in sources.items():assert sha((R/n).read_bytes())==sha((S/n).read_bytes())==h,n
buildpath=Q/'build-source198-001/source-build.json';assert sha(buildpath.read_bytes())==b['source198_host_build_sha256']
build=json.loads(buildpath.read_text());assert build['success'] and build['source_sha256']==sources
for n,h in build['artifacts_sha256'].items():assert sha(Path(n).read_bytes())==h,n
for n,h in build['build_metadata_sha256'].items():assert sha((Q/'build-source198-001'/n).read_bytes())==h,n
for c in ['cpu-checks','solver-environment','damped-cgls-oracle']:
 receipt=json.loads((Q/'analysis/source198-cpu-validation-001'/c/'receipt.json').read_text());assert receipt['exit_code']==0,(c,receipt)
statuses={}
for number,case in [('526','integrated198-contact-regional-526'),('527','integrated198-contact-restart16-527')]:
 root=Q/'native'/case
 tx=[json.loads((root/'baseline'/('transaction-%06d.json'%i)).read_text()) for i in range(3)]
 assert [x['accepted'] for x in tx]==[True,True,False]
 assert tx[1]['after']['solver_certificates'][0]['nonlinear'][0]<=1e-7
 failed=tx[2];code=failed['after']['statuses'][0]['code'];statuses[number]=code
 assert code==(1 if number=='526' else 10)
 if number=='526':assert failed['after']['statuses'][0]['diagnostics']==[-3,-3,-3,-3]
 for n,h in failed['before']['accepted_state_vector_sha256'].items():assert failed['after']['accepted_state_vector_sha256'][n]==h
 assert failed['before']['control_step']==failed['after']['control_step']==2
 assert failed['after']['solver_certificates'][0]['validity'][3]==0
 a={str(p.relative_to(root/'baseline')):p.read_bytes() for p in (root/'baseline').rglob('*') if p.is_file()}
 z={str(p.relative_to(root/'aliased')):p.read_bytes() for p in (root/'aliased').rglob('*') if p.is_file()}
 assert len(a)==13 and a==z
helpers=Q/'native/damped-cgls-kernel-replay-523'
assert (helpers/'O0/stdout.json').read_bytes()==(helpers/'O3/stdout.json').read_bytes()
h=json.loads((helpers/'O3/stdout.json').read_text());assert len(h['cases'])==8
assert all(c['error_bits']==0 and c['iterations']==c['requested_cap'] for c in h['cases'] if c['mu_kg']>0)
assert all(c['status']=='breakdown' and c['error_bits']==32 for c in h['cases'] if c['mu_kg']==0)
expected.update(P+n for n in ['summary.json','summary-reference032.json','publication-binding-reference033.json'])
expected.update(['matter/examples/cardboard/solver-milestone-source198.md','matter/examples/cardboard/implementation-20261014.md'])
pathlist=(Q/'publication033-staged-paths.nul').read_bytes();paths={x.decode() for x in pathlist.split(b'\0') if x}
assert paths==expected
actual={x.decode() for x in git('diff','--name-only','-z').split(b'\0') if x}|{x.decode() for x in git('ls-files','--others','-z').split(b'\0') if x}
assert paths==actual,(paths-actual,actual-paths)
assert not git('diff','--cached','--name-only','-z')
assert not git('diff','--check')
result={'schema':'numi.cardboard.publication033-independent-byte-audit.v1','verdict':'PASS','historical_records_unchanged':1669,'new_cases':len(new),'payloads_verified':checked,'original_payload_bytes':rawbytes,'source_files_verified':len(sources),'summary_sha256':sha((D/'summary.json').read_bytes()),'binding_sha256':sha((D/'publication-binding-reference033.json').read_bytes()),'staged_pathlist_sha256':sha(pathlist),'staged_paths':len(paths),'source198_frozen_sha256':sha(frozen),'build_artifacts_verified':len(build['artifacts_sha256']),'rejected_step_statuses':statuses,'full_protocol_qualification':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/'report.json').write_text(json.dumps(result,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
