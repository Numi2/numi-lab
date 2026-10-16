from pathlib import Path
import json,hashlib,gzip,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';a=q/'analysis/publication041-candidate-001';d=q/'analysis/publication041-root-audit-001';d.mkdir(exist_ok=False);e=Path('matter/examples/cardboard/evidence/native-implementation-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();git=lambda *v:subprocess.check_output(['git',*v],cwd=r)
meta=json.loads((a/'report.json').read_text());parent=meta['parent'];assert git('rev-parse','HEAD').decode().strip()==parent;assert not git('diff','--cached','--name-only')
allowed={x.decode() for x in (a/'stage-paths.nul').read_bytes().split(b'\0') if x};summary=json.loads((r/e/'summary.json').read_text());binding=json.loads((r/e/'publication-binding-reference041.json').read_text())
assert sha(r/e/'summary.json')==meta['summary_sha256']==binding['summary_sha256'];assert sha(r/e/'publication-binding-reference041.json')==meta['binding_sha256']
oldbytes=git('show',parent+':'+str(e/'summary.json'));old=json.loads(oldbytes);assert (r/e/'summary-reference040.json').read_bytes()==oldbytes;assert len(old['records'])==2144 and len(summary['records'])==2144+meta['new_records'] and summary['records'][:2144]==old['records']
for k in old['verdicts']:assert summary['verdicts'][k]['status']==old['verdicts'][k]['status'] and summary['qualification'][k]==summary['verdicts'][k]['evidence']
payloads=[];original=stored=0;case_set=set()
for rec in summary['records'][2144:]:
 assert rec['case'] not in case_set;case_set.add(rec['case']);assert not any(x in rec['case'] for x in ['native602','diagnostic-602','source255','source257']);assert rec['physical_box_qualification'] is False
 for f in rec['files']:
  rel=str(e/f.get('repository_copy',f.get('repository_compressed_copy')));p=r/rel;buf=p.read_bytes()
  if 'repository_compressed_copy' in f:assert len(buf)==f['compressed_bytes'] and hashlib.sha256(buf).hexdigest()==f['compressed_sha256'],rel;raw=gzip.decompress(buf)
  else:raw=buf
  assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'],rel
  payloads.append(rel);stored+=len(buf);original+=len(raw)
assert len(payloads)==len(set(payloads))==meta['payloads']
source=q/'snapshots/source-integrated-measured-coupon-contact-254-002';f=json.loads((source/'frozen-source.json').read_text());support=json.loads((source/'configure-support-source.json').read_text())['files'];assert sha(source/'frozen-source.json')==meta['source_frozen_sha256'] and len(f['source_sha256'])==849
for n,h in f['source_sha256'].items():assert sha(r/n)==h,n
support_exception=q/'analysis/publication041-reproduction-support-001/canonical-support'
for n,h in support.items():
 p=r/n
 if not p.is_file() or sha(p)!=h:p=support_exception/n
 assert p.is_file() and sha(p)==h,n
changed=[]
for n,h in f['source_sha256'].items():
 prev=subprocess.run(['git','show',parent+':'+n],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if prev.returncode or hashlib.sha256(prev.stdout).hexdigest()!=h:changed.append(n)
assert sorted(changed)==binding['source254']['changed_sources']==meta['source_changed'];assert all(n.startswith('matter/') or n=='CMakeLists.txt' for n in changed)
known=set(payloads)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference040.json','publication041-case-selection.json','publication-binding-reference041.json']};assert known==allowed
def zset(*v):return {x.decode() for x in git(*v).split(b'\0') if x}
actual=zset('diff','--name-only','-z')|zset('ls-files','--others','--exclude-standard','-z')|zset('ls-files','--others','--ignored','--exclude-standard','-z');assert actual==allowed,{'extra':sorted(actual-allowed),'missing':sorted(allowed-actual)}
check=subprocess.run(['git','diff','--check','--',*changed],cwd=r,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/'diffcheck.stdout').write_bytes(check.stdout);(d/'diffcheck.stderr').write_bytes(check.stderr);assert check.returncode==0
for n in ['apps/NumiHumanRestingVisual.hpp','apps/NumiHumanRestingIntercostalRibAttachment.hpp','tools/test_intercostal_rib_motion_attachment.mm']:assert (r/n).read_bytes()==git('show',parent+':'+n)
cpu1=json.loads((q/'analysis/source254-build-001/report.json').read_text());cpu2=json.loads((q/'analysis/source254-build-002/report.json').read_text());assert cpu1['all_pass'] and cpu2['all_pass'] and cpu2['all_inputs_unchanged'];assert sha(q/'build-source254-integrated-002/source-build.json')==binding['source254']['source_build_sha256']==cpu2['source_build_sha256']
pkg=q/'analysis/publication041-reproduction-002';rep=json.loads((pkg/'report.json').read_text());canonical={**support,**f['source_sha256']};variant_checks=[]
for row in rep['variants']:
 v=pkg/'source-variants'/row['name']
 for n,h in row['actual_input_sha256'].items():
  if n in row['delta_sha256']:assert sha(v/'delta'/n)==h==row['delta_sha256'][n],(row['name'],n)
  else:assert canonical[n]==h,(row['name'],n)
 if not row['declared_binding_valid']:assert row['declared_mismatches']
 variant_checks.append({'name':row['name'],'actual_inputs_reconstructible':len(row['actual_input_sha256']),'declared_binding_valid':row['declared_binding_valid'],'invalid_drafts_not_qualified':True})
for n,b in rep['files'].items():assert sha(pkg/n)==b['sha256'] and (pkg/n).stat().st_size==b['bytes']
report={'schema':'numi.cardboard.publication041-root-audit.v1','verdict':'PASS_WITH_SIX_FULL_SEQUENCE_AXES_OPEN','parent':parent,'paths':len(allowed),'payloads':len(payloads),'historical_records_exact':2144,'new_records':meta['new_records'],'original_bytes':original,'stored_bytes':stored,'source_count':len(f['source_sha256']),'configure_support_count':len(support),'source_changed':sorted(changed),'source_frozen_sha256':meta['source_frozen_sha256'],'summary_sha256':meta['summary_sha256'],'binding_sha256':meta['binding_sha256'],'variant_checks':variant_checks,'all_six_statuses_open':True,'canonical_native_trajectory_pending':True,'active602_255_257_inventory_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');(d/'stage-paths.nul').write_bytes((a/'stage-paths.nul').read_bytes());print(json.dumps(report),flush=True)
