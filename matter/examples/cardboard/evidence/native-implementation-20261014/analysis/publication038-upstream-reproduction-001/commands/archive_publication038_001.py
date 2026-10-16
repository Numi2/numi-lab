from pathlib import Path
import json,hashlib,gzip,shutil,os,stat,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';s=q/'snapshots/source-capacity-gates-checkpoint-proof-222-001';d=q/'analysis/publication038-candidate-001';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
enc=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
git=lambda *args:subprocess.check_output(['git',*args],cwd=r)
parent='990ea59792d57c9989281c96377434e403cd8c2e'
assert git('rev-parse','HEAD').decode().strip()==parent and git('branch','--show-current').decode().strip()=='main'
assert not git('status','--porcelain=v1','--untracked-files=all')
assert sha(s/'frozen-source.json')=='698a3b1106e6997aa251a8090894adcb8b2850085b70a83f681ccd6fd6f383b6'
assert sha(q/'analysis/reference038-evidence-errata-002/report.json')=='8796954c6865150a3624865ed43757c29fa7069382c066a0decd4d6293809fd3'
for name in ['source217-independent-review-001','source221-owner-review-001']:
 assert (q/'analysis'/name/'report.json').is_file(),name
d.mkdir(exist_ok=False)
oldbytes=(dest/'summary.json').read_bytes();assert hashlib.sha256(oldbytes).hexdigest()=='918715b6309c33c3ac4dba903f64872df0f7e6fbcf89e2bfd48b342efa1010aa'
summary=json.loads(oldbytes);oldcount=len(summary['records']);assert oldcount==1932
assert not (dest/'summary-reference037.json').exists();(dest/'summary-reference037.json').write_bytes(oldbytes)
oldcases={v['case'] for v in summary['records']}
cases=set()
for pattern in ['source212*','source213*','source214*','source217*','source219*','source221*','source222*','native547*','native549*','native550*','native551*','native552*','native553*','native554*','reference038-evidence-errata-*']:
 cases.update(str(p.relative_to(q)) for p in (q/'analysis').glob(pattern) if p.is_dir())
cases.update(['analysis/native545-first-failure-adjudication-001','analysis/fresh-source210-strict-inner-full-coupon-545-command-001','analysis/native553-proof-repair-delta-001','analysis/publication038-reproduction-001','analysis/source217-independent-review-001','analysis/source221-owner-review-001','solver/fresh-source210-strict-inner-full-coupon-545','native/true-residual-full-contact-546','native/critical23-power-sweep-547','native/integrated-true-residual-full-contact-548','native/integrated-contact-baseline-549','native/integrated-contact-newton64-550','solver/control19-newton3-resume-diagnostic-551','solver/control19-newton3-resume-diagnostic-552','solver/control19-newton3-resume-diagnostic-553','solver/control19-newton3-resume-diagnostic-554'])
cases-=oldcases
assert all('555' not in x and '220' not in x and '215' not in x and '216' not in x for x in cases)
files=[];records=[]
for case in sorted(cases):
 root=q/case;assert root.is_dir(),case
 fs=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=sorted(n for n in dirs if n not in ['__pycache__','.git','CMakeFiles'])
  for name in sorted(names):
   p=Path(current)/name;assert p.is_file() and not p.is_symlink(),p
   rel=str(p.relative_to(root));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();stored=Path(case)/rel
   gz=len(raw)>262144 and p.suffix!='.gz'
   data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
   if gz:stored=stored.with_name(stored.name+'.gz')
   target=dest/stored;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
   restored=gzip.decompress(target.read_bytes()) if gz else target.read_bytes()
   assert restored==raw
   item={'name':rel,'bytes':len(raw),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
   if gz:item.update({'repository_compressed_copy':str(stored),'compressed_bytes':len(data),'compressed_sha256':sha(target),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
   else:item.update({'repository_copy':str(stored),'restore':'copy exact bytes and restore recorded mode'})
   fs.append(item);files.append(str(e/stored))
 records.append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs})
summary['records'].extend(records)
assert summary['records'][:oldcount]==json.loads(oldbytes)['records']
summary['milestone']='Source222 composes capacity damage/debond gates and source-runner checkpoint-proof repair on source210, without changing the physical operator or material law. Source222 fresh build,31 capacity tests,12 CPU runner-proof cases and the existing checkpoint checker pass. Native545 fails control19 after19 accepted controls. The old541 finite-difference endpoint comparisons are invalid from an aliased input builder and are withdrawn; corrected547 remains bounded and incomplete. Integrated contact activation fails with and without true-residual mode. Active555 and candidate220/215/216 are excluded without outcomes. All six full-sequence qualification axes remain open.'
evidence={
'solver_correctness':'Native543 operator and strict-inner helper evidence remains bounded. Native547 corrected1472-row replay has23 center tangents within1e-3 (max2.9614e-4), but one endpoint reaches1.1800e-3 and only13/23 directions have adjacent strict-FP64 and native resolved FD windows. The old541 FD counts are INVALID_INPUTS and withdrawn. Source221/222 runner-proof admission passes12 CPU cases, not restore parity. Integrated548/549/550 fail during initial bond activation; trueR off reproduces the failure. No completed nonlinear solve is qualified.',
'numerical_convergence':'Native545 uses unchanged5e-7 outer tolerance, strict active-RHS inner forcing, restart16 and64 Newton iterations. It accepts controls0-18 in both environments, then fails control19 at residual1.0195885806751903e-6. Complete60-control coupon and mesh/time convergence remain unqualified. Active555 diagnostic replay is excluded at archive cutoff.',
'component_calibration':'Source217/222 requires accepted irreversible D==1 quadrature reference-area debond comparisons and blocks capacity when enabled paper damage lacks a mesh-objective area fraction.31 tests and independent review pass; this is software admission evidence. Source214 orientation/energy checker remains a separate synthetic fixture. Native547 local projection discrepancy and source-grade damage, starch/HMA identifiability and rate calibration remain open.',
'analogue_validation':'Published source grades and separate series remain the analogue evidence. No new full-board analogue validation pass or exact kraft/NSSC fracture calibration is established.',
'physical_assembly_in_simulation':'The shipping mesh remains528444nodes,1477728tetrahedra and260835cohesivefaces. Bounded native contact/solver arenas still reject the full shipping world. Integrated fixture bond activation fails; no fixture-free physically scored/folded/bonded300x200x200mm checkpoint or measured whole-process16GiB peak is qualified.',
'crush_impact_prediction':'No qualified native compression curve, frozenFpeak, capacity-derived mass, gravity-crush or drop-impact result. Source222 prevents force-only capacity promotion while enabled damage/debond observables are missing or fail convergence.'
}
for k,v in evidence.items():summary['verdicts'][k]['evidence']=v;summary['qualification'][k]=v
summary.setdefault('historical_summary_snapshots',{})['reference037']={'path':'summary-reference037.json','sha256':hashlib.sha256(oldbytes).hexdigest()}
summary.setdefault('additive_errata',[]).append({'reference':'reference038','case':'analysis/reference038-evidence-errata-002','report_sha256':'8796954c6865150a3624865ed43757c29fa7069382c066a0decd4d6293809fd3','old541_finite_difference_inputs':'INVALID_INPUTS; derived FD counts withdrawn; raw artifacts retained'})
(dest/'summary.json').write_bytes(enc(summary))
f=json.loads((s/'frozen-source.json').read_text());changed=[]
for n,h in f['source_sha256'].items():
 assert sha(s/n)==h,n
 current=r/n;assert current.is_file(),n
 if sha(current)!=h:shutil.copy2(s/n,current);changed.append(n)
assert sorted(changed)==['matter/tools/cardboard_probe.mm','matter/tools/cardboard_stage_resolution_study.md','matter/tools/cardboard_stage_resolution_study.py','matter/tools/test_cardboard_stage_resolution_study.py']
selection={'schema':'numi.cardboard.publication038-case-selection.v1','cases':sorted(cases),'active_or_experimental_excluded':['solver/control19-newton3-resume-diagnostic-555','source215','source216','source220'],'closed_attempts_only':True}
(dest/'publication038-case-selection.json').write_bytes(enc(selection))
binding={'schema':'numi.cardboard.reference038-publication-binding.v1','parent':parent,'summary_sha256':sha(dest/'summary.json'),'case_selection_sha256':sha(dest/'publication038-case-selection.json'),'source222':{'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_files':len(f['source_sha256']),'source_build_sha256':'be8655b5cb7f20919029077bd65908ad7574f67abfcbd89d77fac898f8aa171b','changed_sources':sorted(changed),'physical_operator_or_material_law_changed':False},'historical_records_preserved':oldcount,'new_records':len(records),'cases':sorted(cases),'all_six_full_sequence_axes_open':True,'erratum_report_sha256':'8796954c6865150a3624865ed43757c29fa7069382c066a0decd4d6293809fd3'}
(dest/'publication-binding-reference038.json').write_bytes(enc(binding))
allowed=set(files)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference037.json','publication038-case-selection.json','publication-binding-reference038.json']}
(d/'stage-paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(allowed)))
report={'schema':'numi.cardboard.publication038-candidate.v1','parent':parent,'paths':len(allowed),'payloads':len(files),'new_records':len(records),'historical_records_preserved':oldcount,'source_changed':sorted(changed),'source_frozen_sha256':sha(s/'frozen-source.json'),'summary_sha256':sha(dest/'summary.json'),'binding_sha256':sha(dest/'publication-binding-reference038.json'),'case_selection_sha256':sha(dest/'publication038-case-selection.json'),'all_six_statuses_open':True,'active555_excluded':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_bytes(enc(report));print(json.dumps(report),flush=True)
