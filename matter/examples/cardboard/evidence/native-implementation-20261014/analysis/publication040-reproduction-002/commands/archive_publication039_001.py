from pathlib import Path
import json,hashlib,gzip,shutil,os,stat,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';s=q/'snapshots/source231-upstream-compatibility-234-001';d=q/'analysis/publication039-candidate-001';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();enc=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
parent='938e30343dd1cc313e222c8d2e8c11f5d486754d'
assert git('rev-parse','HEAD').decode().strip()==parent and git('branch','--show-current').decode().strip()=='main';assert not git('status','--porcelain=v1','--untracked-files=all')
for p in ['analysis/source234-upstream-compatibility-001/report.json','analysis/source231-upstream-compatibility-001/report.json','analysis/source230-composition-build-001/report.json']:
 assert json.loads((q/p).read_text())['all_pass'],p
for n in [571,573]:
 report=json.loads((q/f'analysis/native{n}-composed-production-parity-001/report.json').read_text());assert report['all_compared_files_byte_identical'] and report['complete_controls']==7
for n in [555,565]:assert json.loads((q/f'analysis/native{n}-production-suffix-parity-001/report.json').read_text())['verdict']=='PASS_EXACT_RESUMED_SUFFIX'
assert sha(q/'analysis/native563-strict-local-projection-adjudication-001/report.json')=='c17b4bee59a8c063a7e7ed49d8d6a39ff720861ae2b68247ab7724a013af833b'
f=json.loads((s/'frozen-source.json').read_text());assert all(sha(s/n)==h for n,h in f['source_sha256'].items())
changed=[n for n,h in f['source_sha256'].items() if not (r/n).exists() or sha(r/n)!=h];assert changed and all(n.startswith('matter/') or n=='CMakeLists.txt' for n in changed),changed
d.mkdir(exist_ok=False);oldbytes=(dest/'summary.json').read_bytes();summary=json.loads(oldbytes);oldcount=len(summary['records']);assert oldcount==2002;assert not (dest/'summary-reference038.json').exists();oldcases={a['case'] for a in summary['records']}
cases=set()
for pat in ['*215*','source216-*','source225-*','source226-*','source228-*','source229-*','source230-*','source231-*','source234-*','source-local-projection-state-certification-220-*','native563-strict-local-projection-adjudication-001','native565-*','native571-*','native573-*']:
 cases.update(str(p.relative_to(q)) for p in (q/'analysis').glob(pat) if p.is_dir())
cases.update(['analysis/native555-command-001','analysis/native555-production-suffix-parity-001','analysis/native555-cached-operator-001','analysis/native555-tight-projection-achievability-004','analysis/first-failure-dispatch-capture-overlay-224-001','analysis/source-production-cohesive-pages-227-001','analysis/publication039-reproduction-001','analysis/publication039-reproduction-extension-002'])
cases.update(str(p.relative_to(q)) for p in (q/'native').iterdir() if p.is_dir() and any(p.name.endswith('-'+str(n)) for n in range(556,575)))
cases.update(['solver/control19-newton3-resume-diagnostic-555','solver/control19-newton3-small-epsilon-resume-diagnostic-564','solver/control19-newton3-small-epsilon-resume-diagnostic-565','solver/composed230-seven-phase-regression-571','solver/composed231-seven-phase-regression-573'])
cases-=oldcases
assert all(not any(x in c for x in ['source232','source233','source235','native572']) for c in cases)
for case in cases:assert (q/case).is_dir(),case
for case in cases:
 if case.startswith(('solver/','native/')):assert (q/case/'receipt.json').is_file(),case
files=[];records=[]
for case in sorted(cases):
 root=q/case;fs=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=sorted(n for n in dirs if n not in ['__pycache__','.git','CMakeFiles'])
  for name in sorted(names):
   p=Path(current)/name;assert p.is_file() and not p.is_symlink(),p
   rel=str(p.relative_to(root));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();stored=Path(case)/rel;gz=len(raw)>262144 and p.suffix!='.gz';data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
   if gz:stored=stored.with_name(stored.name+'.gz')
   target=dest/stored;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
   assert (gzip.decompress(target.read_bytes()) if gz else target.read_bytes())==raw
   item={'name':rel,'bytes':len(raw),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
   if gz:item.update({'repository_compressed_copy':str(stored),'compressed_bytes':len(data),'compressed_sha256':sha(target),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
   else:item.update({'repository_copy':str(stored),'restore':'copy exact bytes and restore recorded mode'})
   fs.append(item);files.append(str(e/stored))
 records.append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs})
summary['records'].extend(records);assert summary['records'][:oldcount]==json.loads(oldbytes)['records'];(dest/'summary-reference038.json').write_bytes(oldbytes)
summary['milestone']='Canonical234 preserves current main Human changes and integrates checked pure-FEM8192 workspaces, prescribed BOX/capsule/freeBOX admission and bounded epsilon replay. Source230/231 complete the tiny seven-control all-phase regression with all32production files exactly matching210. Actual19/Newton3 diagnostic555/565 retains the original19 failure with full suffix parity. Fine FD and stricter FP64 projection remain inconclusive. Candidate220 strict local projected-state correction passes all10 targeted tangent rows in563 but is not canonical234 or a full coupon result. Cohesive paging227 first kernel helper574 fails; recorder567 has no usable sidecar. All six full-sequence qualification axes remain open.'
evidence={
'solver_correctness':'Canonical234 composes the checked pure8192 workspaces and exact prescribed-tool admission; source230 component568/569/570 passes and source231 full tiny seven-control573 regression is byte-identical to210 across32production files. Actual555/56519/Newton3 captures preserve complete production suffix parity, and cached material/element/mass assembly comparisons remain bounded. Fine native finite differences and tighter FP64 endpoint gates remain inconclusive; no resolved global derivative certificate. Targeted candidate220/563 repairs one local projected-state stopping defect,10/10selected tangent rows below1e-3, but is not a completed nonlinear solve. Recorder567 produced no usable sidecar, so its terminal index32 does not identify a dispatch.',
'numerical_convergence':'Original545 and exact diagnostic555/565 accept controls0-18 then fail19 under unchanged5e-7 outer tolerance, strict active-RHS inner forcing and restart16. Two complete .01degree seven-control regressions are bounded software regressions. The target60-control coupon, independent8/16/32 by2/4/8 by1/2/4 mesh and dt/dt2/dt4 histories remain unqualified; no accepted prefix is promoted.',
'component_calibration':'Source563 independent FP64 adjudication shows candidate strict projected-state correction passes all10 targeted native material tangent rows with max3.0948e-5, versus original p23+error1.1800e-3. This is numerical constitutive-update evidence, zero FEM transactions, not source-grade calibration. Enabled damage/debond observable gates remain. Exact kraft/NSSC fracture, starch/HMA local identifiability and impact rate data remain open.',
'analogue_validation':'Published nearby source grades and separate ECT, scored-bend and slow-rate series remain analogue inputs. No new complete native analogue validation or exact kraft/NSSC fracture/impact calibration is established.',
'physical_assembly_in_simulation':'Pure-FEM8192 scope and freeBOX4096 scope remain bounded; the explicit shipping mesh528444nodes/1477728tetrahedra still exceeds current admitted solver/contact paths.225 ordinal page checks pass but227 production cohesive helper574 fails, dense contact samples remain unpaged, and16GiB whole-process peak is unmeasured. Contact activation fails in566/567 with equal production snapshots; observer567 cannot attribute the failing dispatch. No physically scored/folded/bonded, fixture-free300x200x200mm checkpoint.',
'crush_impact_prediction':'No qualified complete native compression curve or frozenFpeak, capacity-derived heavy mass, gravity crush or drop result. Existing capacity gates retain enabled missing/unconverged paper damage and bond-area observables as blockers. Strict local numerical repair and tiny regression do not validate crush or impact.'
}
for k,v in evidence.items():summary['verdicts'][k]['evidence']=v;summary['qualification'][k]=v
summary.setdefault('historical_summary_snapshots',{})['reference038']={'path':'summary-reference038.json','sha256':hashlib.sha256(oldbytes).hexdigest()};(dest/'summary.json').write_bytes(enc(summary))
for n in changed:
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/n,p)
selection={'schema':'numi.cardboard.publication039-case-selection.v1','cases':sorted(cases),'excluded_active':['source232 strict policy full-run integration','source233 recorder repair','source235 contact paging','native572 full strict coupon not run at cutoff'],'closed_attempts_only':True};(dest/'publication039-case-selection.json').write_bytes(enc(selection))
binding={'schema':'numi.cardboard.reference039-publication-binding.v1','parent':parent,'source234':{'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_files':len(f['source_sha256']),'source_build_sha256':sha(q/'build-source234-upstream-compatibility-001/source-build.json'),'changed_sources':sorted(changed),'runtime_admission_and_workspace_changed':True,'physical_operator_equations_changed':False,'upstream_human_files_preserved':True},'historical_records_preserved':oldcount,'new_records':len(records),'summary_sha256':sha(dest/'summary.json'),'case_selection_sha256':sha(dest/'publication039-case-selection.json'),'all_six_full_sequence_axes_open':True,'complete_tiny_regression_reports':{'source230':sha(q/'analysis/native571-composed-production-parity-001/report.json'),'source231':sha(q/'analysis/native573-composed-production-parity-001/report.json')},'diagnostic_suffix_parity_reports':{'native555':sha(q/'analysis/native555-production-suffix-parity-001/report.json'),'native565':sha(q/'analysis/native565-production-suffix-parity-001/report.json')},'strict_local_candidate_only_report':sha(q/'analysis/native563-strict-local-projection-adjudication-001/report.json')}
(dest/'publication-binding-reference039.json').write_bytes(enc(binding))
allowed=set(files)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference038.json','publication039-case-selection.json','publication-binding-reference039.json']}
(d/'stage-paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(allowed)))
report={'schema':'numi.cardboard.publication039-candidate.v1','parent':parent,'paths':len(allowed),'payloads':len(files),'new_records':len(records),'historical_records_preserved':oldcount,'source_changed':sorted(changed),'source_frozen_sha256':sha(s/'frozen-source.json'),'summary_sha256':sha(dest/'summary.json'),'binding_sha256':sha(dest/'publication-binding-reference039.json'),'all_six_statuses_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_bytes(enc(report));print(json.dumps(report),flush=True)
