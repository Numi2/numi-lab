from pathlib import Path
import json,hashlib,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/publication042-reproduction-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=q/'snapshots/source-integrated-globalization-contact-memory-259-003'
cf=json.loads((canon/'frozen-source.json').read_text())['source_sha256'];cs=json.loads((canon/'configure-support-source.json').read_text())['files'];canonical={**cs,**cf}
files={}
def put(p,n):
 p=Path(p);z=d/n;assert p.is_file() and not p.is_symlink(),p;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z);assert sha(z)==sha(p);files[str(n)]={'external_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
ids=[255,257,258,259,263]
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
 if any(x in p.name for x in [str(i) for i in range(602,613)]+[f'source{i}' for i in ids]+['publication041','publication042']):
  put(p,Path('commands')/p.name)
build_roots=[]
for p in sorted(q.iterdir()):
 if p.is_dir() and p.name.startswith('build-source') and any(('source'+str(i)+'-') in p.name for i in ids):build_roots.append(p)

builds=[]
for b in build_roots:
 p=b/'source-build.json'
 label=str(b.relative_to(q)).replace('/','__')
 for a in sorted(b.iterdir()):
  if a.is_file() and (a.name in ['CMakeCache.txt','compile_commands.json'] or a.suffix in ['.json','.log']):
   put(a,Path('builds')/label/a.name)
 if not p.is_file():
  builds.append({'path':str(b.relative_to(q)),'success':False,'source_build_sha256':None,'scope':'No completed source-build record; top-level logs/metadata retained as an incomplete or preflight-failed attempt, never a successful build.'})
  continue
 j=json.loads(p.read_text())
 arts=[];mismatch=[]
 for n,h in j.get('artifacts_sha256',{}).items():
  a=Path(n)
  if not a.is_file() or sha(a)!=h:
   mismatch.append({'path':n,'declared_sha256':h,'actual_sha256':sha(a) if a.is_file() else None});continue
  rel=a.relative_to(b) if a.is_relative_to(b) else Path('external-artifacts')/a.relative_to(q)
  put(a,Path('builds')/label/rel);arts.append(n)
 source_bindings=j.get('source_sha256',{})
 available={n:{h} for n,h in canonical.items()}
 for variant in variants:
  for n,h in variant['actual_input_sha256'].items():available.setdefault(n,set()).add(h)
 missing_source=[{'path':n,'sha256':h} for n,h in source_bindings.items() if h not in available.get(n,set())]
 builds.append({'path':str(b.relative_to(q)),'success':j.get('success'),'source_build_sha256':sha(p),'archived_artifacts':arts,'unavailable_or_changed_artifacts':mismatch,'source_binding_reconstruction_complete':not missing_source,'unavailable_historical_source_bytes':missing_source,'scope':'Artifact and failed-attempt preservation; only source bindings with all exact source bytes reconstructed may be replay-qualified.'})
readme="""Reference042 reproduction inputs

Canonical259-003 composes transactional opt-in Newton/CGLS comparison, actual first-uncertified mortar CCD query capture and exact contact incidence/transition memory planning, repairs five missing Runtime pipeline registrations exposed by native608, and preserves the concurrent420b4f86 Human update.

Each source variant is reconstructed from canonical259-003 plus recorded exact delta bytes. Invalid intermediate source maps remain explicitly invalid. Completed source-bound build artifacts and failed/interrupted build records are retained without treating compilation as physical evidence.

Native602 measures the actual cached primary operator at the strict failing state; all production suffix bytes match585/596. The native-versus-exact-captured-frame CPU cached action difference is0.0328127%, below the1% descriptive comparison gate; no empirical FP32 noise-floor or complete solver qualification is inferred. Native603 completes the unchanged canonical254 tiny seven-phase history. Native604/605 preserve observer-off/on transactions and capture the actual unresolved right vertex-triangle query; native610 captures the distinct query at the smallest rejected fraction. Neither uncertified query is a clear certificate.

Native606 passes414 assertions on actual production selection kernels, with zero physical transactions. Native607 preserves32 default-off production files. Native608 SIGSEGV before first accepted step is retained with the OS crash report; the263 registration fix requires subsequent fresh integrated native evidence. Native611/612, if completed and present in the archive, supply those separately bounded histories. The long fresh native609 strict60-control coupon is excluded while active; its immutable command may be archived as a continuation command, never as a completed result.

All six full-sequence qualification axes remain open. No complete scored/folded/bonded/released shipping box, qualified capacity, gravity crush or impact is declared. Whole-runtime shipping allocation inventory is not bound by the generic252-slot planner.
"""

(d/'README.md').write_text(readme)
report={'schema':'numi.cardboard.publication042-reproduction.v1','canonical_source':str(canon),'canonical_frozen_sha256':sha(canon/'frozen-source.json'),'files':files,'variants':variants,'builds':builds,'closed_inputs_only':True,'excluded_active':['native609 strict full coupon while active','source260 conservative CCD repair','source261 full-runtime compaction parity','source262 material evidence correction'],'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n')
print(json.dumps({'files':len(files),'variants':len(variants),'invalid_draft_variants':[x['name'] for x in variants if not x['declared_binding_valid']],'builds':len(builds),'report_sha256':sha(d/'report.json')}))
