from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/publication041-reproduction-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=q/'snapshots/source-integrated-measured-coupon-contact-254-002'
cf=json.loads((canon/'frozen-source.json').read_text())['source_sha256'];cs=json.loads((canon/'configure-support-source.json').read_text())['files'];canonical={**cs,**cf}
files={}
def put(p,n):
 p=Path(p);z=d/n;assert p.is_file() and not p.is_symlink(),p;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z);assert sha(z)==sha(p);files[str(n)]={'external_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
ids=[235,244,246,247,248,249,250,252,253,254,256]
variants=[]
for s in sorted((q/'snapshots').iterdir()):
 if not any(('-'+str(i)+'-' in s.name or s.name.endswith('-'+str(i))) for i in ids) or not (s/'frozen-source.json').is_file():continue
 f=json.loads((s/'frozen-source.json').read_text())['source_sha256'];support=json.loads((s/'configure-support-source.json').read_text())['files']
 names=set(f)|set(support)
 # Earlier235 drafts retained their parent selection map; retain every final235 declared file actually present as well.
 if '-235-' in s.name:
  final=json.loads((q/'snapshots/source-shipping-contact-sample-pages-235-007/frozen-source.json').read_text())['source_sha256']
  names.update(n for n in final if (s/n).is_file())
 actual={n:sha(s/n) for n in sorted(names) if (s/n).is_file()}
 invalid=[n for n,h in {**support,**f}.items() if actual.get(n)!=h]
 put(s/'frozen-source.json',Path('source-variants')/s.name/'frozen-source.json');put(s/'configure-support-source.json',Path('source-variants')/s.name/'configure-support-source.json')
 delta={}
 for n,h in actual.items():
  if canonical.get(n)!=h:put(s/n,Path('source-variants')/s.name/'delta'/n);delta[n]=h
 variants.append({'name':s.name,'selected_files':len(f),'support_files':len(support),'delta_files':len(delta),'delta_sha256':delta,'actual_input_sha256':actual,'declared_binding_valid':not invalid,'declared_mismatches':invalid,'scope':'Verified immutable binding' if not invalid else 'Preserved failed/draft source state; declared parent map is not a valid binding for these actual bytes and must not be reused as a qualified frozen source.'})
for p in sorted(q.glob('*.py')):
 if any(x in p.name for x in ['585','588','589','590','591','592','593','594','595','596','597','598','599','600','601']+[f'source{i}' for i in ids]+['publication040']):
  put(p,Path('commands')/p.name)
build_roots=[]
for p in sorted(q.iterdir()):
 if p.is_dir() and p.name.startswith('build-source') and any(('source'+str(i)+'-') in p.name for i in ids):build_roots.append(p)
for a in ['analysis/source235-sample-compactor-007/build','analysis/source246-fullbox-force-accuracy-001','analysis/source247-contact-sample-integration-001/build','analysis/source247-contact-sample-integration-001/build-002','analysis/source249-contact-compaction-fixture-001/build-001']:build_roots.append(q/a)
builds=[]
for b in build_roots:
 p=b/'source-build.json'
 if not p.is_file():continue
 label=str(b.relative_to(q)).replace('/','__');j=json.loads(p.read_text());put(p,Path('builds')/label/'source-build.json')
 for a in sorted(b.glob('provenance-command-*.log')):put(a,Path('builds')/label/a.name)
 arts=[];mismatch=[]
 for n,h in j.get('artifacts_sha256',{}).items():
  a=Path(n)
  if not a.is_file() or sha(a)!=h:
   mismatch.append({'path':n,'declared_sha256':h,'actual_sha256':sha(a) if a.is_file() else None});continue
  put(a,Path('builds')/label/a.relative_to(b));arts.append(n)
 builds.append({'path':str(b.relative_to(q)),'success':j.get('success'),'source_build_sha256':sha(p),'archived_artifacts':arts,'unavailable_or_changed_artifacts':mismatch})
readme="""Reference041 reproduction inputs

Canonical254-002 composes the corrected cached-operator observer, checked production contact-sample page consumers, volume-weighted damage summaries, measured force-accuracy checks, first-local-peak capacity identity, fail-closed contact memory preflight and full-body scene observers. The concurrent b18cf2bb Human update is retained. Combined native evolution remains a separate gate.

Each source variant records actual input hashes and every differing byte relative to canonical254-002. Verify the reconstructed hashes before using it. Earlier draft235/250/252 states whose copied parent map does not bind the edited files are explicitly invalid drafts; their bytes and failed evidence are preserved, not promoted to valid frozen sources.

Earlier native555/565/596 FD observers execute the uncached evaluator even though the production FGMRES path uses cached material matrices. Additive scope errata withdraw any cached-kernel interpretation of those action numbers. Independent FP64/90-digit derivative results and production-history parity remain evidence for their narrower stated scopes. Corrected native599 uses the actual cached mode and preserves32 small-regression production files; its reconstructed cached action relative difference is1.25749e-7. Native602 actual failing-state replay is active and excluded at this cutoff.

Full contact601 captures exact trial geometry and preserves12 transaction files, while the fixture still fails mortar complement CCD certification. Its payload cannot identify the individual failed complement feature. New best-direction and mortar-feature trace candidates255/257 are active and excluded.

All six full-sequence qualification axes remain open. No complete score/fold/assemble/checkpoint, qualified capacity, gravity crush or impact result is declared.
"""
(d/'README.md').write_text(readme)
report={'schema':'numi.cardboard.publication041-reproduction.v1','canonical_source':str(canon),'canonical_frozen_sha256':sha(canon/'frozen-source.json'),'files':files,'variants':variants,'builds':builds,'closed_inputs_only':True,'excluded_active':['native602 actual failing-state corrected cache observer','source255 best-of Newton/CGLS','source257 mortar feature trace','source250 complete runtime allocation inventory'],'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n')
print(json.dumps({'files':len(files),'variants':len(variants),'invalid_draft_variants':[x['name'] for x in variants if not x['declared_binding_valid']],'builds':len(builds),'report_sha256':sha(d/'report.json')}))
