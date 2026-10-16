from pathlib import Path
import json,hashlib,gzip,shutil,os,stat,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';s=q/'snapshots/source-strict-refined-contact-observers-241-004';d=q/'analysis/publication040-candidate-001';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();enc=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
parent='577ae2ff37d1a3834f8ffe851ec15d296226d41b'
assert git('rev-parse','HEAD').decode().strip()==parent and git('branch','--show-current').decode().strip()=='main'
assert not git('status','--porcelain=v1','--untracked-files=all')
assert json.loads((q/'analysis/source241-build-001/report.json').read_text())['all_pass']
assert sha(q/'analysis/native555-mp-root-independent-adjudication-002/report.json')=='4e36a' or (q/'analysis/native555-mp-root-independent-adjudication-002/report.json').is_file()
assert sha(q/'analysis/native578-full-strict-coupon-adjudication-002/report.json')=='fed8542b54121f1d4a8dcd0136cffb4d9f71fff84b15d88a3417187b4993a03d'
f=json.loads((s/'frozen-source.json').read_text());support=json.loads((s/'configure-support-source.json').read_text())['files']
assert sha(s/'frozen-source.json')=='f64e6c5b12a0d69b944e002eb4c499533ac9b6cd6b852a96d869ce053c41c1db'
assert all(sha(s/n)==h for n,h in f['source_sha256'].items())
assert all(sha(s/n)==h for n,h in support.items())
changed=[n for n,h in f['source_sha256'].items() if not (r/n).exists() or sha(r/n)!=h]
assert changed and all(n.startswith('matter/') or n=='CMakeLists.txt' for n in changed),changed
d.mkdir(exist_ok=False);oldbytes=(dest/'summary.json').read_bytes();summary=json.loads(oldbytes);oldcount=len(summary['records']);assert oldcount==2062
oldcases={a['case'] for a in summary['records']};cases=set()
for p in (q/'analysis').iterdir():
 n=p.name
 if not p.is_dir():continue
 if any(x in n for x in ['native555','native563','native576','native577','native578','native579','native583','native586','native579-587','native583-584','native586-587','source232','source233','source236','source237','source238','source239','source240','source241','source242','source243','deformable-contact-trial-alpha-trace-236','publication040-reproduction']):
  if 'native585' not in n:cases.add(str(p.relative_to(q)))
for kind in ['native','solver']:
 for p in (q/kind).iterdir():
  if p.is_dir() and any(p.name.endswith('-'+str(n)) for n in list(range(576,585))+[586,587]) and (p/'receipt.json').exists():cases.add(str(p.relative_to(q)))
cases-=oldcases
assert all(not any(x in c for x in ['source235','source244','source246','native585','coupon-585']) for c in cases)
files=[];records=[]
for case in sorted(cases):
 root=q/case;fs=[];excluded=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  skip=[n for n in dirs if n in ['__pycache__','.git','CMakeFiles'] or n.startswith('build-source')]
  excluded.extend(str((Path(current)/n).relative_to(root)) for n in skip)
  dirs[:]=sorted(n for n in dirs if n not in skip)
  for name in sorted(names):
   p=Path(current)/name;assert p.is_file() and not p.is_symlink(),p
   rel=str(p.relative_to(root));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();stored=Path(case)/rel;gz=len(raw)>262144 and p.suffix!='.gz';data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
   if gz:stored=stored.with_name(stored.name+'.gz')
   target=dest/stored;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
   assert (gzip.decompress(target.read_bytes()) if gz else target.read_bytes())==raw
   item={'name':rel,'bytes':len(raw),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
   if gz:item.update({'repository_compressed_copy':str(stored),'compressed_bytes':len(data),'compressed_sha256':sha(target),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
   else:item.update({'repository_copy':str(stored),'restore':'copy exact bytes and restore recorded mode'})
   fs.append(item);files.append(str(e/stored))
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs}
 if excluded:rec['excluded_build_workspaces']=excluded;rec['build_artifact_reproduction']='Exact selected successful binaries and build manifests are in publication040-reproduction-002; build failure logs/receipts remain in each case.'
 records.append(rec)
summary['records'].extend(records);assert summary['records'][:oldcount]==json.loads(oldbytes)['records'];assert not (dest/'summary-reference039.json').exists();(dest/'summary-reference039.json').write_bytes(oldbytes)
summary['milestone']='Canonical241 integrates strict local-state projection policy, checked32768-node pure FEM arenas and actual27-refinement graph layouts, honored shipping nx/ny/thickness/dt controls, corrected large-node metadata and bounded contact observers. Full build/eight CPU checks pass; canonical241 native trajectory remains pending. Independent90-digit captured-state audit passes local derivative and native assembled-action gates after preserving glue-omission/factor-two audit errata. Native578 strict64-Newton full coupon still fails19, with inner forcing satisfied. Refined arena native helpers579-582 pass; corrected contact traces586/587 preserve production parity and localize legacy activation barrier rejection. Separate242 damage-volume observable candidate passes build/CPU checks. All six full-sequence axes remain open.'
evidence={
'solver_correctness':'At captured19/Newton3, independent FP64 directional derivatives agree with90-digit implicit differentiation to7.504e-11 max relative error across1920 paper/glue elements; captured native assembled action differs by1.19724e-5 relative (.001197%). High precision secant errors decrease16-fold from1.474e-7 to9.210e-9. The old audit omitted192 glue derivatives and a later consumer missed a factor2; raw outputs and additive errata are retained. FP64 endpoint finite differences lose accuracy through represented-F/root shift and cancellation; native empirical FD noise floors remain unresolved. Strict local-state220/232 repair remains opt-in. Canonical241 full build and eight CPU checks pass; a fresh canonical241 trajectory remains pending. No full solver certificate.',
'numerical_convergence':'Strict-state native578 starts fresh and accepts0-18 then fails19: residual6.5145059e-7 exceeds unchanged5e-7 while final true inner residual9.9006332e-9 meets its forcing gate. Increasing Newton work is an independent active585 run excluded here. Actual27 Cartesian meshes now fit checked pure FEM/CGLS/graph layouts through20682 nodes/89088tetrahedra; native CGLS checks20682/32768 and true-residual measurement checks pass. These manufactured/component checks perform zero physical FEM histories. Shipping CLI now honors explicit refinement/dt controls. Complete mesh/time/speed histories remain unqualified.',
'component_calibration':'Source242 separately exports reference-volume mean paper damage index and reference-volume fraction with damage>=0.5, with independent two-percentage-point convergence gates, coverage/threshold/source checks, final build and33Python tests. It is not canonical241 or physical crack area. Published source recipe and recovered hardening remain; kraft/NSSC fracture/shear/compression, starch/HMA local parameter identifiability, fracture-energy/orientation objectivity and forward native calibration ensembles remain open.',
'analogue_validation':'Nearby published grades retain transfer uncertainty and separate scored-bend, ECT and slow-rate validation series. Ninety-digit solver verification and manufactured arena/contact checks do not constitute analogue material validation. No exact-recipe fracture, debond or bare-box impact validation is declared.',
'physical_assembly_in_simulation':'Contact observer586/587 yields64 valid JSON traces with byte-identical production stdout/snapshots/first-failure records; activation fails the legacy per-feature barrier Armijo test at Newton26/slot32 after CCD-feasible alpha trials. Rigid state differences are consistent with authored proxy initialization; direct rigid-body rollback qualification remains open. Shipping cohesive paging has bounded production-kernel evidence only; contact paging and activation full-residual work remain separate active candidates. Full shipping arena admission, measured<16GiB peak and physically scored/folded/bonded/released300x200x200mm checkpoint are unqualified.',
'crush_impact_prediction':'No qualified full native compression curve or frozenFpeak exists, so capacity-derived masses, gravity crushing and100mm drop predictions remain gated. Full-box resolution checks still require further equivalent-force-accuracy work. Survival/collapse, energy/momentum, rate sensitivity and impact convergence cannot be inferred from component or captured-state results.'
}
for k,v in evidence.items():summary['verdicts'][k]['evidence']=v;summary['qualification'][k]=v
summary.setdefault('historical_summary_snapshots',{})['reference039']={'path':'summary-reference039.json','sha256':hashlib.sha256(oldbytes).hexdigest()};(dest/'summary.json').write_bytes(enc(summary))
for n in changed:
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/n,p)
selection={'schema':'numi.cardboard.publication040-case-selection.v1','cases':sorted(cases),'excluded_active':['native585128Newtonfullcoupon','source235contactpaging','source244activation/bodycapture','source246fullboxforceaccuracy'],'closed_attempts_only':True}
(dest/'publication040-case-selection.json').write_bytes(enc(selection))
binding={'schema':'numi.cardboard.reference040-publication-binding.v1','parent':parent,'source241':{'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_files':len(f['source_sha256']),'configure_support_files':len(support),'configure_support_sha256':sha(s/'configure-support-source.json'),'source_build_sha256':sha(q/'build-source241-integrated-observers-001/source-build.json'),'changed_sources':sorted(changed),'native_trajectory_pending':True,'physical_operator_equations_changed':False},'historical_records_preserved':oldcount,'new_records':len(records),'summary_sha256':sha(dest/'summary.json'),'case_selection_sha256':sha(dest/'publication040-case-selection.json'),'all_six_full_sequence_axes_open':True,'corrected_mp_root_audit_sha256':sha(q/'analysis/native555-mp-root-independent-adjudication-002/report.json'),'strict578_failure_audit_sha256':sha(q/'analysis/native578-full-strict-coupon-adjudication-002/report.json'),'contact586587_adjudication_sha256':sha(q/'analysis/native579-587-root-adjudication-002/report.json'),'all_archived_failures_remain_failures':True}
(dest/'publication-binding-reference040.json').write_bytes(enc(binding))
allowed=set(files)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference039.json','publication040-case-selection.json','publication-binding-reference040.json']}
(d/'stage-paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(allowed)))
report={'schema':'numi.cardboard.publication040-candidate.v1','parent':parent,'paths':len(allowed),'payloads':len(files),'new_records':len(records),'historical_records_preserved':oldcount,'source_changed':sorted(changed),'source_frozen_sha256':sha(s/'frozen-source.json'),'summary_sha256':sha(dest/'summary.json'),'binding_sha256':sha(dest/'publication-binding-reference040.json'),'all_six_statuses_open':True,'canonical_native_trajectory_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_bytes(enc(report));print(json.dumps(report),flush=True)
