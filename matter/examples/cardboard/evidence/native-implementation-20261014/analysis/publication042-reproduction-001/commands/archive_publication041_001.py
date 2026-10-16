from pathlib import Path
import json,hashlib,gzip,shutil,os,stat,subprocess,datetime,re
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';s=q/'snapshots/source-integrated-measured-coupon-contact-254-002';d=q/'analysis/publication041-candidate-001';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();enc=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
parent='b18cf2bb62ed492ab307ece760b4ec108fd37f69'
assert git('rev-parse','HEAD').decode().strip()==parent and git('branch','--show-current').decode().strip()=='main'
assert not git('status','--porcelain=v1','--untracked-files=all')
cpu1=json.loads((q/'analysis/source254-build-001/report.json').read_text());cpu2=json.loads((q/'analysis/source254-build-002/report.json').read_text());assert cpu1['all_pass'] and cpu2['all_pass'] and cpu2['all_inputs_unchanged']
f=json.loads((s/'frozen-source.json').read_text());support=json.loads((s/'configure-support-source.json').read_text())['files'];assert sha(s/'frozen-source.json')=='6b35572e45d27c534d8671dbad22106720542c13e7b8bfafca66c918862afbb1'
assert all(sha(s/n)==h for n,h in {**support,**f['source_sha256']}.items())
changed=[n for n,h in f['source_sha256'].items() if not (r/n).is_file() or sha(r/n)!=h]
assert changed and all(n.startswith('matter/') or n=='CMakeLists.txt' for n in changed),changed
d.mkdir(exist_ok=False);oldbytes=(dest/'summary.json').read_bytes();summary=json.loads(oldbytes);oldcount=len(summary['records']);assert oldcount==2144;oldcases={a['case'] for a in summary['records']};cases=set()
source_ids=[235,244,246,247,248,249,250,252,253,254,256]
for p in (q/'analysis').iterdir():
 if not p.is_dir():continue
 n=p.name
 selected=any(n.startswith('source'+str(i)+'-') for i in source_ids)
 selected=selected or any(n.startswith('native'+str(i)+'-') for i in [585]+list(range(588,602)))
 selected=selected or ('native555' in n and '-253-' in n)
 selected=selected or n in ['publication041-reproduction-002','publication041-reproduction-failure-001','publication041-reproduction-support-001','publication040-root-audit-002','publication040-candidate-001']
 # The complete allocation inventory is still active and must not enter this cutoff.
 if selected and not ('allocation-inventory' in n or 'runtime-inventory' in n):cases.add(str(p.relative_to(q)))
for kind in ['native','solver']:
 for p in (q/kind).iterdir():
  if p.is_dir() and any(p.name.endswith('-'+str(n)) for n in [585]+list(range(588,602))) and (p/'receipt.json').exists():cases.add(str(p.relative_to(q)))
cases-=oldcases
assert all(not any(x in c for x in ['native602','diagnostic-602','source255','source257']) for c in cases)
files=[];records=[]
for case in sorted(cases):
 root=q/case;fs=[];excluded=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  skip=[n for n in dirs if n in ['__pycache__','.git','CMakeFiles'] or (n.startswith('build') and not case.startswith('analysis/publication041-reproduction'))]
  excluded.extend(str((Path(current)/n).relative_to(root)) for n in skip);dirs[:]=sorted(n for n in dirs if n not in skip)
  for name in sorted(names):
   p=Path(current)/name;assert p.is_file() and not p.is_symlink(),p
   rel=str(p.relative_to(root));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();stored=Path(case)/rel;gz=len(raw)>262144 and p.suffix!='.gz';data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
   if gz:stored=stored.with_name(stored.name+'.gz')
   target=dest/stored;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
   assert (gzip.decompress(target.read_bytes()) if gz else target.read_bytes())==raw
   item={'name':rel,'bytes':len(raw),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
   if gz:item.update(repository_compressed_copy=str(stored),compressed_bytes=len(data),compressed_sha256=sha(target),restore='gunzip; restore recorded mode; verify byte count and SHA-256')
   else:item.update(repository_copy=str(stored),restore='copy exact bytes and restore recorded mode')
   fs.append(item);files.append(str(e/stored))
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs}
 if excluded:rec.update(excluded_build_workspaces=excluded,build_artifact_reproduction='Exact completed artifact copies and provenance are in publication041-reproduction-002; failed/draft source bindings are explicitly marked invalid.')
 records.append(rec)
summary['records'].extend(records);assert summary['records'][:oldcount]==json.loads(oldbytes)['records'];assert not (dest/'summary-reference040.json').exists();(dest/'summary-reference040.json').write_bytes(oldbytes)
summary['milestone']='Canonical254 composes production cache observer repair, validated contact-sample compactor fixture, volume-weighted damage and measured force-accuracy convergence gates, first-local-peak capacity identity, fail-closed rigid-contact memory planning and complete host scene observers. Eight target build and13 CPU check commands pass; canonical native trajectory pending. Strict128-Newton coupon585/596 still fails control19. Additive observer-scope errata distinguish earlier uncached action audits from actual cached production application; corrected tiny regression599 preserves32 files. Full private contact601 preserves12 transactions and localizes first failure to mortar complement CCD certification. Shipping memory inventory/admission and all six full-sequence axes remain open.'
evidence={
'solver_correctness':'Additive cache-scope errata correct039/040: native555/565/596 diagnostic actions used the uncached evaluator, despite production FGMRES using cached matrices. The previous1.19724e-5 native-versus-MP number is uncached action evidence only; .000597272 was CPU cached reconstruction versus uncached native, not cached-kernel roundoff. Independent FP64/90-digit derivative and observer-history parity evidence retain their narrower scope. Corrected observer599 selects actual primary cache mode1, preserves32 small-regression production files and matches exact captured-frame cached CPU reconstruction to1.25749e-7 relative; no empirical noise floor or full failing-state cached qualification is inferred. Actual596 N127 FP64 local projection/stress agreement is strong, while centered FP64 derivative/noise completeness remains open. Canonical254 build/CPU checks pass; full solver correctness is unqualified.',
'numerical_convergence':'Fresh strict128-Newton585 and byte-identical suffix replay596 still fail control19 at5.4423458e-7 versus unchanged5e-7. Accepted controls0-18 do not constitute a complete history. Final captured matrix condition estimate1.061e7 and FP64 direct solve defect2.45e-18 are offline results; even ideal FP64 Newton gives tiny residual-descent fractions at this state. Offline damped least-squares directions suggest a separately active native best-of experiment, not a solver certificate. Source254 incorporates actual free-DOF residual/dt force bounds, volume-weighted constituent damage mean/fraction, first-local-peak identity and their curve/work/springback/damage/debond gates. Actual27 refined mesh layouts remain component evidence; complete mesh/time/slower-loading histories are unqualified.',
'component_calibration':'Volume-weighted damage and first-local-peak source/fixture contracts are now composed into the driver and resolution tooling. Physical recipe/equations and source-grade uncertainty are unchanged. The published-data package, recovered hardening and analogue response machinery do not uniquely identify kraft/NSSC fracture/shear/compression or local starch/HMA fracture parameters. Mesh/orientation energy objectivity and full native calibration/feasible-set forward ensembles remain open. No exact-recipe calibration is declared.',
'analogue_validation':'Close-grade scored-bend, ECT, pin-fixture, HMA and slow-rate evidence retain their specimen/test-condition split and transfer limits. Solver checks, fixture parity, CPU gates and independent captured-state algebra do not validate a complete corrugated-board response. Exact-recipe fracture/debond and this bare-box impact remain unavailable, not closed.',
'physical_assembly_in_simulation':'Source244/252 add default-off activation full-residual selection and host scene evidence through1199/1200/1201 and failure1202, with unavailable exception readbacks explicit. Native590/591 still fail Newton7 of1202; accepted Matter history hashes/cursor remain at1201. Native597 summary and601 full-private capture preserve12 transaction files and identify the first failure as self-narrowphase mortar complement certification at primitive pair16/99 for all8 trial fractions; a specific VT/EE query is not yet identified. Actual production contact-compactor598 passes E2/P7 finite component parity and overflow rejection, with zero physical transactions. Source250 adds exact11 contact subarena checks but explicitly lacks complete runtime/staging/transient coverage; shipping caps are retained. No measured under16GiB shipping allocation or scored/folded/bonded/released300x200x200mm checkpoint exists.',
'crush_impact_prediction':'First-local-peak identity now excludes later densification maxima and correctly maps the native full saved curve, including its zero origin. This is an implementation/CPU contract change, not a measured capacity. No numerically qualified complete native compression curve or frozenFpeak exists, so capacity-derived masses, gravity crushing and100mm impacts remain unqualified. Momentum/energy accounting, rate-envelope sensitivity, survival/collapse and impact refinement await the assembled checkpoint.'
}
for k,v in evidence.items():summary['verdicts'][k]['evidence']=v;summary['qualification'][k]=v
summary.setdefault('historical_summary_snapshots',{})['reference040']={'path':'summary-reference040.json','sha256':hashlib.sha256(oldbytes).hexdigest()};(dest/'summary.json').write_bytes(enc(summary))
for n in changed:z=r/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/n,z)
selection={'schema':'numi.cardboard.publication041-case-selection.v1','cases':sorted(cases),'excluded_active':['native602','source255','source257','source250 full runtime allocation inventory'],'closed_attempts_only':True}
(dest/'publication041-case-selection.json').write_bytes(enc(selection))
binding={'schema':'numi.cardboard.reference041-publication-binding.v1','parent':parent,'source254':{'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_files':len(f['source_sha256']),'configure_support_files':len(support),'configure_support_sha256':sha(s/'configure-support-source.json'),'source_build_sha256':cpu2['source_build_sha256'],'changed_sources':sorted(changed),'native_trajectory_pending':True,'physical_operator_equations_changed_by_composition':False},'historical_records_preserved':oldcount,'new_records':len(records),'summary_sha256':sha(dest/'summary.json'),'case_selection_sha256':sha(dest/'publication041-case-selection.json'),'all_six_full_sequence_axes_open':True,'cache_scope_erratum_sha256':sha(q/'analysis/native555-primary-cache-action-scope-erratum-253-001/report.json'),'corrected599_parity_sha256':sha(q/'analysis/native599-corrected-cache-observer-parity-002/report.json'),'native596_598_adjudication_sha256':sha(q/'analysis/native596-598-root-adjudication-001/report.json'),'native601_adjudication_sha256':sha(q/'analysis/native601-full-private-trial-root-adjudication-001/report.json'),'all_archived_failures_remain_failures':True}
(dest/'publication-binding-reference041.json').write_bytes(enc(binding))
allowed=set(files)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference040.json','publication041-case-selection.json','publication-binding-reference041.json']}
(d/'stage-paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(allowed)))
report={'schema':'numi.cardboard.publication041-candidate.v1','parent':parent,'paths':len(allowed),'payloads':len(files),'new_records':len(records),'historical_records_preserved':oldcount,'source_changed':sorted(changed),'source_frozen_sha256':sha(s/'frozen-source.json'),'summary_sha256':sha(dest/'summary.json'),'binding_sha256':sha(dest/'publication-binding-reference041.json'),'all_six_statuses_open':True,'canonical_native_trajectory_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_bytes(enc(report));print(json.dumps(report),flush=True)
