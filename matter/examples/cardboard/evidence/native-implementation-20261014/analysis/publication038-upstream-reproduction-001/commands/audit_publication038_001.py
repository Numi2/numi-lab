from pathlib import Path
import json,hashlib,gzip,subprocess,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';a=q/'analysis/publication038-candidate-001';d=q/'analysis/publication038-root-audit-001';d.mkdir(exist_ok=False);e=Path('matter/examples/cardboard/evidence/native-implementation-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *v:subprocess.check_output(['git',*v],cwd=r)
meta=json.loads((a/'report.json').read_text());parent=meta['parent'];assert git('rev-parse','HEAD').decode().strip()==parent
assert not git('diff','--cached','--name-only')
allowed={x.decode() for x in (a/'stage-paths.nul').read_bytes().split(b'\0') if x};assert len(allowed)==4713
summary=json.loads((r/e/'summary.json').read_text());binding=json.loads((r/e/'publication-binding-reference038.json').read_text())
assert sha(r/e/'summary.json')==meta['summary_sha256']==binding['summary_sha256']
assert sha(r/e/'publication-binding-reference038.json')==meta['binding_sha256']
oldbytes=git('show',parent+':'+str(e/'summary.json'));old=json.loads(oldbytes)
assert (r/e/'summary-reference037.json').read_bytes()==oldbytes
assert len(old['records'])==1932 and len(summary['records'])==1997 and summary['records'][:1932]==old['records']
for k in old['verdicts']:
 assert summary['verdicts'][k]['status']==old['verdicts'][k]['status']
 assert summary['qualification'][k]==summary['verdicts'][k]['evidence']
payloads=[];original=stored=0
for rec in summary['records'][1932:]:
 assert '555' not in rec['case'] and 'source220' not in rec['case'],rec['case']
 for f in rec['files']:
  rel=str(e/f.get('repository_copy',f.get('repository_compressed_copy')));p=r/rel;buf=p.read_bytes()
  if 'repository_compressed_copy' in f:
   assert len(buf)==f['compressed_bytes'] and hashlib.sha256(buf).hexdigest()==f['compressed_sha256'],rel
   raw=gzip.decompress(buf)
  else:raw=buf
  assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],rel
  payloads.append(rel);stored+=len(buf);original+=len(raw)
assert len(payloads)==len(set(payloads))==4705
source=q/'snapshots/source-capacity-gates-checkpoint-proof-222-001';f=json.loads((source/'frozen-source.json').read_text());assert sha(source/'frozen-source.json')==meta['source_frozen_sha256']
for n,h in f['source_sha256'].items():assert sha(r/n)==h,n
changed=[]
for n,h in f['source_sha256'].items():
 if hashlib.sha256(git('show',parent+':'+n)).hexdigest()!=h:changed.append(n)
assert sorted(changed)==binding['source222']['changed_sources']==meta['source_changed']
known=set(payloads)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference037.json','publication038-case-selection.json','publication-binding-reference038.json']}
assert known==allowed
def zset(*v):return {x.decode() for x in git(*v).split(b'\0') if x}
actual=zset('diff','--name-only','-z')|zset('ls-files','--others','--exclude-standard','-z')|zset('ls-files','--others','--ignored','--exclude-standard','-z')
assert actual==allowed,{'extra':sorted(actual-allowed),'missing':sorted(allowed-actual)}
check=subprocess.run(['git','diff','--check','--',*changed],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(d/'diffcheck.stdout').write_bytes(check.stdout);(d/'diffcheck.stderr').write_bytes(check.stderr);assert check.returncode==0
assert sha(q/'analysis/source217-independent-review-001/report.json')=='676dc59fcba3e207f129057f849b8ea5fa6cea9758e2d18e99a0b68c3330a580'
assert sha(q/'analysis/source221-owner-review-001/report.json')=='f4dd33cf2b6c70b239769618c2c7bf7a5369ec32d0334d81dfd5057f30880919'
assert json.loads((q/'analysis/source222-runner-proof-cpu-tests-001/report.json').read_text())['all_pass']
assert json.loads((q/'analysis/source222-composition-build-001/report.json').read_text())['capacity_tests_exit_code']==0
report={'schema':'numi.cardboard.publication038-root-audit.v1','verdict':'PASS_WITH_INVALID_541_FD_EVIDENCE_WITHDRAWN','parent':parent,'paths':len(allowed),'payloads':len(payloads),'historical_records_exact':1932,'new_records':65,'original_bytes':original,'stored_bytes':stored,'source_count':len(f['source_sha256']),'source_changed':sorted(changed),'source_frozen_sha256':meta['source_frozen_sha256'],'summary_sha256':meta['summary_sha256'],'binding_sha256':meta['binding_sha256'],'all_six_statuses_open':True,'active555_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'stage-paths.nul').write_bytes((a/'stage-paths.nul').read_bytes());print(json.dumps(report),flush=True)
