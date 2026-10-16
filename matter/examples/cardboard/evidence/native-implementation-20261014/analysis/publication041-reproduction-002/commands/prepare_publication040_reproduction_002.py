from pathlib import Path
import hashlib,json,shutil,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
d=q/'analysis/publication040-reproduction-002'
d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={}
def put(p,n):
 p=Path(p);z=d/n
 assert p.is_file() and not p.is_symlink(),p
 z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,z)
 assert sha(p)==sha(z)
 files[str(n)]={'external_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
canon=q/'snapshots/source-strict-refined-contact-observers-241-004'
cf=json.loads((canon/'frozen-source.json').read_text())['source_sha256']
cs=json.loads((canon/'configure-support-source.json').read_text())['files']
for p in sorted(q.glob('*.py')):
 if any(x in p.name for x in ['555','578','579','580','581','582','583','584','586','587','source232','source233','source236','source237','source238','source239','source240','source241','source242','source243','publication039']):
  put(p,Path('commands')/p.name)
variants=[]
for s in sorted((q/'snapshots').iterdir()):
 if not any(('-'+str(i)) in s.name for i in [232,233,236,237,238,239,240,241,242,243]):continue
 if not (s/'frozen-source.json').is_file():continue
 f=json.loads((s/'frozen-source.json').read_text())['source_sha256'];support=json.loads((s/'configure-support-source.json').read_text())['files']
 if s.name=='source-deformable-contact-trial-trace-236-001':
  overlay=q/'analysis/deformable-contact-trial-alpha-trace-236-001/source236-overlay.json'
  f=json.loads(overlay.read_text())['candidate_source_bindings']
  put(overlay,Path('source-variants')/s.name/'authoritative-overlay.json')
 if s.name=='source-deformable-contact-trial-trace-236-002':
  overlay=q/'analysis/deformable-contact-trial-alpha-trace-236-002/source236-overlay.json'
  build=q/'analysis/deformable-contact-trial-alpha-trace-236-002/build-source236-001/source-build.json'
  f=json.loads(build.read_text())['source_sha256']
  put(overlay,Path('source-variants')/s.name/'authoritative-overlay.json')
  put(build,Path('source-variants')/s.name/'authoritative-source-build.json')
 assert all(sha(s/n)==h for n,h in f.items()),s
 assert all(sha(s/n)==h for n,h in support.items()),s
 put(s/'frozen-source.json',Path('source-variants')/s.name/'frozen-source.json')
 put(s/'configure-support-source.json',Path('source-variants')/s.name/'configure-support-source.json')
 delta={}
 for n,h in dict(support,**f).items():
  canonical_h=cf.get(n,cs.get(n))
  if canonical_h!=h:
   put(s/n,Path('source-variants')/s.name/'delta'/n);delta[n]=h
 variants.append({'name':s.name,'selected_files':len(f),'support_files':len(support),'delta_files':len(delta),'delta_sha256':delta,'authoritative_selected_sha256':f})
build_names=[
 'build-source232-executed-bound-003',
 'build-source233-first-failure-001',
 'analysis/deformable-contact-trial-alpha-trace-236-001/build-source236-003',
 'analysis/deformable-contact-trial-alpha-trace-236-002/build-source236-001',
 'build-source237-strict-main-001',
 'build-source238-refined-coupon-001',
 'build-source240-refined-regional-graph-003',
 'build-source240-refined-native-components-004',
 'build-source240-refined-graph-cmake-004',
 'build-source241-integrated-observers-001',
 'build-source243-native-reporting-001']
builds=[]
for name in build_names:
 b=q/name;f=b/'source-build.json'
 if not f.exists():
  builds.append({'path':name,'not_found':True});continue
 j=json.loads(f.read_text());put(f,Path('builds')/b.name/'source-build.json')
 for p in sorted(b.glob('provenance-command-*.log')):put(p,Path('builds')/b.name/p.name)
 artifacts=j.get('artifacts_sha256',{})
 for n,h in artifacts.items():
  p=Path(n);assert sha(p)==h,p;put(p,Path('builds')/b.name/p.relative_to(b))
 builds.append({'path':name,'manifest_sha256':sha(f),'success':j.get('success'),'artifacts':len(artifacts)})
readme="""Reference040 reproduction inputs

Canonical source241-004 composes stricter local projected-state certification, checked 32768-node pure FEM residual/CGLS arenas, actual refined graph layouts, respected shipping resolution controls, corrected large-node diagnostic labels and contact observer fixes. The full integrated build and eight CPU checks pass. A complete nonlinear or shipping-box solve is not implied.

Each source variant includes its exact selected and configure-support maps plus every file differing from canonical241. To reconstruct a variant, start with canonical selected/support files, overwrite the recorded delta, then verify every declared hash before building. Existing absolute command paths are provenance; executing under a new root requires a fresh source-bound command and receipt.

High precision capture555 replay distinguishes source derivative accuracy from finite-difference endpoint representation/cancellation. The original audit omitted192 glue elements; corrected full-recipe mass/stiffness/action comparisons and additive factor-two errata preserve the prior results. Native empirical finite-difference noise and complete nonlinear convergence remain separate open gates.

Native578 is a full fresh strict-state run that failed control19 after64 Newton iterations. All failed launches, malformed observer JSON and corrected repeats are preserved. Corrected contact observer586/587 histories are byte-identical, but the legacy per-feature activation line search fails; no contact qualification is inferred.

Source242 separately adds reference-volume damage mean and threshold-fraction resolution gates; its final build and33 Python tests pass. It is a component candidate, not yet canonical241 or physical material calibration. Native585 and activation/paging/force-accuracy work still active at this package cutoff are excluded.

All six full-sequence qualification axes remain open: full solver, mesh/time convergence, material/bond calibration, analogue validation, fixture-free physical assembly, crush/impact prediction.
"""
(d/'README.md').write_text(readme)
report={'schema':'numi.cardboard.publication040-reproduction.v1','canonical_source':str(canon),'canonical_frozen_sha256':sha(canon/'frozen-source.json'),'files':files,'variants':variants,'builds':builds,'closed_inputs_only':True,'excluded_active':['native585','source235 production contact paging','source244 activation/body rollback','source246 full-box force accuracy'],'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}},indent=2)+'\n')
print(json.dumps({'files':len(files),'variants':len(variants),'builds':builds,'report_sha256':sha(d/'report.json')}))
