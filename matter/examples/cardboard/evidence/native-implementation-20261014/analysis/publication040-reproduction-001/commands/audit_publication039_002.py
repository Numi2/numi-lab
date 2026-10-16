from pathlib import Path
import json,hashlib,gzip,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';a=q/'analysis/publication039-candidate-001';d=q/'analysis/publication039-root-audit-002';d.mkdir(exist_ok=False);e=Path('matter/examples/cardboard/evidence/native-implementation-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
git=lambda *v:subprocess.check_output(['git',*v],cwd=r)
meta=json.loads((a/'report.json').read_text());parent=meta['parent'];assert git('rev-parse','HEAD').decode().strip()==parent
assert not git('diff','--cached','--name-only')
allowed={x.decode() for x in (a/'stage-paths.nul').read_bytes().split(b'\0') if x};assert len(allowed)==4883
summary=json.loads((r/e/'summary.json').read_text());binding=json.loads((r/e/'publication-binding-reference039.json').read_text())
assert sha(r/e/'summary.json')==meta['summary_sha256']==binding['summary_sha256']
assert sha(r/e/'publication-binding-reference039.json')==meta['binding_sha256']
oldbytes=git('show',parent+':'+str(e/'summary.json'));old=json.loads(oldbytes)
assert (r/e/'summary-reference038.json').read_bytes()==oldbytes
assert len(old['records'])==2002 and len(summary['records'])==2062 and summary['records'][:2002]==old['records']
for k in old['verdicts']:
 assert summary['verdicts'][k]['status']==old['verdicts'][k]['status']
 assert summary['qualification'][k]==summary['verdicts'][k]['evidence']
payloads=[];original=stored=0
for rec in summary['records'][2002:]:
 for f in rec['files']:
  rel=str(e/f.get('repository_copy',f.get('repository_compressed_copy')));p=r/rel;buf=p.read_bytes()
  if 'repository_compressed_copy' in f:
   assert len(buf)==f['compressed_bytes'] and hashlib.sha256(buf).hexdigest()==f['compressed_sha256'],rel
   raw=gzip.decompress(buf)
  else:raw=buf
  assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],rel
  payloads.append(rel);stored+=len(buf);original+=len(raw)
assert len(payloads)==len(set(payloads))==4863
source=q/'snapshots/source231-upstream-compatibility-234-001';f=json.loads((source/'frozen-source.json').read_text());assert sha(source/'frozen-source.json')==meta['source_frozen_sha256']
assert len(f['source_sha256'])==832
for n,h in f['source_sha256'].items():assert sha(r/n)==h,n
changed=[]
for n,h in f['source_sha256'].items():
 prev=subprocess.run(['git','show',parent+':'+n],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if prev.returncode or hashlib.sha256(prev.stdout).hexdigest()!=h:changed.append(n)
assert sorted(changed)==binding['source234']['changed_sources']==meta['source_changed']
known=set(payloads)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference038.json','publication039-case-selection.json','publication-binding-reference039.json']}
assert known==allowed
def zset(*v):return {x.decode() for x in git(*v).split(b'\0') if x}
actual=zset('diff','--name-only','-z')|zset('ls-files','--others','--exclude-standard','-z')|zset('ls-files','--others','--ignored','--exclude-standard','-z')
assert actual==allowed,{'extra':sorted(actual-allowed),'missing':sorted(allowed-actual)}
check=subprocess.run(['git','diff','--check','--',*changed],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(d/'diffcheck.stdout').write_bytes(check.stdout);(d/'diffcheck.stderr').write_bytes(check.stderr);assert check.returncode==0
for n in ['apps/NumiHumanRestingIntercostalRibAttachment.hpp','tools/test_intercostal_rib_motion_attachment.mm']:assert (r/n).read_bytes()==git('show',parent+':'+n)
for n,h in [
('analysis/native571-composed-production-parity-001/report.json','387cb41eee92f3ef529e72c3db6f43b29505e287f87d5d8edd17ac81fca10dc5'),
('analysis/native573-composed-production-parity-001/report.json','aced7f90a46ba73e84e71303271d4fb6a5122d28dc0baa2f48300041862b7872'),
('analysis/native555-production-suffix-parity-001/report.json','9e6d9f202daa7e7bea76cde01d2f8c40693d09750a1831b78f7eb090e940f417'),
('analysis/native565-production-suffix-parity-001/report.json','393b79379353e07b423f1a2f36840cf8565ce4f7610f4a5f594c960d13e15528'),
('analysis/native563-strict-local-projection-adjudication-001/report.json','c17b4bee59a8c063a7e7ed49d8d6a39ff720861ae2b68247ab7724a013af833b')]:assert sha(q/n)==h,n
compat=json.loads((q/'analysis/source234-upstream-compatibility-001/report.json').read_text())
assert compat['all_pass'] and compat['all_source231_cardboard_bytes_unchanged'] and compat['all_inputs_unchanged'] and compat['parent']==parent
assert sha(q/'build-source234-upstream-compatibility-001/source-build.json')==compat['source_build_sha256']==binding['source234']['source_build_sha256']
for n in ['source-strict-local-projection-upstream-232','source-first-failure-dispatch-233-001','source-contact-sample-pages-235-001']:
 assert not any(n in rec['case'] for rec in summary['records'][2002:]),n
report={'schema':'numi.cardboard.publication039-root-audit.v1','verdict':'PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN','parent':parent,'paths':len(allowed),'payloads':len(payloads),'historical_records_exact':2002,'new_records':60,'original_bytes':original,'stored_bytes':stored,'source_count':len(f['source_sha256']),'source_changed':sorted(changed),'source_frozen_sha256':meta['source_frozen_sha256'],'summary_sha256':meta['summary_sha256'],'binding_sha256':meta['binding_sha256'],'all_six_statuses_open':True,'active232_233_235_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'stage-paths.nul').write_bytes((a/'stage-paths.nul').read_bytes());print(json.dumps(report),flush=True)
