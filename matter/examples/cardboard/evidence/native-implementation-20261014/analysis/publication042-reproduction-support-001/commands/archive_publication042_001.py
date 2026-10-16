from pathlib import Path
import json,hashlib,gzip,shutil,os,stat,subprocess,datetime,re
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';s=q/'snapshots/source-integrated-globalization-contact-memory-259-003';d=q/'analysis/publication042-candidate-001';e=Path('matter/examples/cardboard/evidence/native-implementation-20261014');dest=r/e
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();enc=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();git=lambda *a:subprocess.check_output(['git',*a],cwd=r)
parent='420b4f866decdfaed1f5a81ef102d3cb6efbb442'
assert git('rev-parse','HEAD').decode().strip()==parent and git('branch','--show-current').decode().strip()=='main'
assert not git('status','--porcelain=v1','--untracked-files=all')
cpu2=json.loads((q/'analysis/source259-build-003/report.json').read_text());assert cpu2['all_pass'] and cpu2['all_inputs_unchanged']
for n in ['native611-integrated259-default-off-history-adjudication-001','native612-best-of-seven-phase-root-adjudication-002']:
 assert json.loads((q/'analysis'/n/'report.json').read_text())['all_six_full_sequence_axes_open'] if n.startswith('native612') else True
assert json.loads((q/'solver/integrated259-default-off-seven-phase-611/receipt.json').read_text())['exit_code']==0
assert json.loads((q/'solver/integrated259-best-of-seven-phase-612/receipt.json').read_text())['exit_code']==0

f=json.loads((s/'frozen-source.json').read_text());support=json.loads((s/'configure-support-source.json').read_text())['files'];assert sha(s/'frozen-source.json')=='663a2bf205f2439200077d997804c429c33635d8606462f7a3f700a2d29a2743'
assert all(sha(s/n)==h for n,h in {**support,**f['source_sha256']}.items())
changed=[n for n,h in f['source_sha256'].items() if not (r/n).is_file() or sha(r/n)!=h]
assert changed and all(n.startswith('matter/') or n=='CMakeLists.txt' for n in changed),changed
d.mkdir(exist_ok=False);oldbytes=(dest/'summary.json').read_bytes();summary=json.loads(oldbytes);oldcount=len(summary['records']);assert oldcount==2234;oldcases={a['case'] for a in summary['records']};cases=set()
source_ids=[255,257,258,259,263]
for p in (q/'analysis').iterdir():
 if not p.is_dir():continue
 n=p.name
 selected=any(n.startswith('source'+str(i)+'-') for i in source_ids)
 selected=selected or any(n.startswith('native'+str(i)+'-') for i in [602,603,604,605,606,607,608,610,611,612])
 selected=selected or n.startswith('native602-605-') or n.startswith('native606-607-') or n.startswith('native604-605-') or n=='native605-610-independent-replay-root-audit-001' or n=='source260-605-610-strict-replay-and-json-correction-001'
 selected=selected or n in ['publication042-reproduction-001','publication042-reproduction-support-001','publication042-upstream-001','publication041-root-audit-001','publication041-candidate-001']
 # The complete allocation inventory is still active and must not enter this cutoff.
 if selected and not ('allocation-inventory' in n or 'runtime-inventory' in n):cases.add(str(p.relative_to(q)))
for kind in ['native','solver']:
 for p in (q/kind).iterdir():
  if p.is_dir() and any(p.name.endswith('-'+str(n)) for n in [602,603,604,605,606,607,608,610,611,612]) and (p/'receipt.json').exists():cases.add(str(p.relative_to(q)))
cases-=oldcases
assert all(not any(x in c for x in ['native609','strict-full-coupon-609','source261','source262','source265']) for c in cases)
files=[];records=[]
for case in sorted(cases):
 root=q/case;fs=[];excluded=[]
 for current,dirs,names in os.walk(root,followlinks=False):
  skip=[n for n in dirs if n in ['__pycache__','.git','CMakeFiles'] or (n.startswith('build') and not case.startswith('analysis/publication042-reproduction'))]
  excluded.extend(str((Path(current)/n).relative_to(root)) for n in skip);dirs[:]=sorted(n for n in dirs if n not in skip)
  for name in sorted(names):
   p=Path(current)/name;assert p.is_file() and not p.is_symlink(),p
   rel=str(p.relative_to(root));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();stored=Path(case)/rel;gz=len(raw)>262234 and p.suffix!='.gz';data=gzip.compress(raw,compresslevel=6,mtime=0) if gz else raw
   if gz:stored=stored.with_name(stored.name+'.gz')
   target=dest/stored;assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);mode=stat.S_IMODE(p.stat().st_mode);target.chmod(0o644 if gz else mode)
   assert (gzip.decompress(target.read_bytes()) if gz else target.read_bytes())==raw
   item={'name':rel,'bytes':len(raw),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
   if gz:item.update(repository_compressed_copy=str(stored),compressed_bytes=len(data),compressed_sha256=sha(target),restore='gunzip; restore recorded mode; verify byte count and SHA-256')
   else:item.update(repository_copy=str(stored),restore='copy exact bytes and restore recorded mode')
   fs.append(item);files.append(str(e/stored))
 rec={'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs}
 if excluded:rec.update(excluded_build_workspaces=excluded,build_artifact_reproduction='Exact completed artifact copies and provenance are in publication042-reproduction-001; failed/draft source bindings are explicitly marked invalid.')
 records.append(rec)
summary['records'].extend(records);assert summary['records'][:oldcount]==json.loads(oldbytes)['records'];assert not (dest/'summary-reference041.json').exists();(dest/'summary-reference041.json').write_bytes(oldbytes)
summary['milestone']='Integrated259 repairs best-of CGLS runtime registration, preserves default-off seven-stage histories, and completes the enabled tiny seven-stage coupon with unchanged physical certificates. The actual cached failing-state action is captured and independently reconstructed. Exact mortar query captures locate two uncertified crossings; conservative CCD repair remains active. Contact arena planning now includes17 buffers, but complete whole-runtime allocation admission and shipping execution remain open. All six full-sequence qualification axes remain open.'
evidence={
'solver_correctness':'Actual failing-state cached production action602 matches exact captured-frame CPU reconstruction with relative error3.28127e-4; this is below1% but lacks an empirical noise-floor/full smooth derivative qualification. Canonical259 composes255 best-of globalization,257 exact mortar query observer and258 memory plan; source263 fixes five missing Runtime pipeline registrations after enabled608 SIGSEGV. Ten targets and14 CPU check commands pass. Default-off611 reproduces32 prior production files exactly. Enabled612 completes all7 tiny control stages/14 environment rows, with11 strictly lower CGLS trials,10 retained Newton trials, exact selected alpha and final residual at most4.0135958557e-7. Its selected_residual trace field is mislabeled minJ and excluded; final outcome0 comparisons are unavailable, not zero measured winning residuals. Full solver qualification remains open.',
'numerical_convergence':'Original strict585 and diagnostic602 remain failed at control19. Fresh integrated259 strict60-control coupon609 is active and excluded from completed evidence. Tiny7-control success is not mesh/time/slower-loading convergence. Full longitudinal8/16/32, width2/4/8, thickness1/2/4 and dt refinements, decreasing differences, complete withdrawal/release/relaxation histories and force/work/curve/springback/damage/debond gates remain unqualified. Physical tolerances and restart16 are unchanged.',
'component_calibration':'Kraft/NSSC/starch recipe and physical constitutive equations remain unchanged. Published source-grade hardening, volume-weighted damage and feasible-set analogue machinery retain their evidence limits. Whole-fixture pin adhesion does not uniquely identify local starch mixed-mode parameters; kraft/NSSC fracture/shear/compression and mesh/orientation energy checks, complete native calibration and feasible-set forward ensembles remain open. Exact-recipe calibration is not declared.',
'analogue_validation':'Close-grade scored bend, ECT, pin fixture, HMA and slow-rate data retain source specimen/condition splits and transfer uncertainty. Local algebra, component fixtures and tiny solver histories do not validate a complete corrugated-board response. Exact-recipe fracture/debond and bare-box impact evidence remain unavailable.',
'physical_assembly_in_simulation':'Full-contact fixtures604/605/610 preserve12 production transaction files and still fail1202 Newton7. New384-byte captures identify actual right-side VT mortar complement queries for pair16/99. Exact rational endpoint signs prove threshold crossings for605 and610, and CPU source-order FP32 distance at captured native time matches native bits. Full recurrence parity is inconclusive for610 and native hit/cert remain false. Conservative CCD repair260 is active. Source258 accounts17 contact/incidence buffers and generic old/replacement/upload/readback transitions, but wholeRuntimeAllocationInventoryComplete remains false; no verified under16GiB full shipping allocation exists. Full-runtime compact parity261 and shipping proxy/damage evidence262 remain active. No scored/folded/bonded/released300x200x200mm checkpoint exists.',
'crush_impact_prediction':'First-local-peak capacity identity and slow/gravity/impact protocol remain implemented but no numerically qualified complete native compression curve or frozen Fpeak exists. The assembled checkpoint, capacity-derived masses, body-backed gravity crushing,100mm centered/offset impacts, momentum/energy accounting, rate-envelope sensitivity and impact refinement remain unqualified. Survival, buckling, debonding and collapse remain outcomes.'
}

for k,v in evidence.items():summary['verdicts'][k]['evidence']=v;summary['qualification'][k]=v
summary.setdefault('historical_summary_snapshots',{})['reference041']={'path':'summary-reference041.json','sha256':hashlib.sha256(oldbytes).hexdigest()};(dest/'summary.json').write_bytes(enc(summary))
for n in changed:z=r/n;z.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/n,z)
selection={'schema':'numi.cardboard.publication042-case-selection.v1','cases':sorted(cases),'excluded_active':['native609','source260 conservative CCD implementation','source261','source262','source265'],'closed_attempts_only':True}
(dest/'publication042-case-selection.json').write_bytes(enc(selection))
binding={'schema':'numi.cardboard.reference042-publication-binding.v1','parent':parent,'source259':{'snapshot':str(s),'frozen_source_sha256':sha(s/'frozen-source.json'),'source_files':len(f['source_sha256']),'configure_support_files':len(support),'configure_support_sha256':sha(s/'configure-support-source.json'),'source_build_sha256':cpu2['source_build_sha256'],'changed_sources':sorted(changed),'tiny_native_trajectory_qualified':True,'full_target_trajectory_qualified':False,'physical_operator_equations_changed_by_composition':False},'historical_records_preserved':oldcount,'new_records':len(records),'summary_sha256':sha(dest/'summary.json'),'case_selection_sha256':sha(dest/'publication042-case-selection.json'),'all_six_full_sequence_axes_open':True,'actual_cached602_adjudication_sha256':sha(q/'analysis/native602-corrected-actual-cache-adjudication-002/operator.json'),'native608_crash_sha256':sha(q/'analysis/native608-pipeline-crash-root-adjudication-001/report.json'),'native611_adjudication_sha256':sha(q/'analysis/native611-integrated259-default-off-history-adjudication-001/report.json'),'native612_adjudication_sha256':sha(q/'analysis/native612-best-of-seven-phase-root-adjudication-002/report.json'),'native605_610_recomputed_replay_sha256':sha(q/'analysis/native605-610-independent-replay-root-audit-001/report.json'),'all_archived_failures_remain_failures':True,'historical_source_reconstruction_gaps_explicit':True}

(dest/'publication-binding-reference042.json').write_bytes(enc(binding))
allowed=set(files)|set(changed)|{str(e/n) for n in ['summary.json','summary-reference041.json','publication042-case-selection.json','publication-binding-reference042.json']}
(d/'stage-paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in sorted(allowed)))
report={'schema':'numi.cardboard.publication042-candidate.v1','parent':parent,'paths':len(allowed),'payloads':len(files),'new_records':len(records),'historical_records_preserved':oldcount,'source_changed':sorted(changed),'source_frozen_sha256':sha(s/'frozen-source.json'),'summary_sha256':sha(dest/'summary.json'),'binding_sha256':sha(dest/'publication-binding-reference042.json'),'all_six_statuses_open':True,'canonical_tiny_native_trajectory_qualified':True,'full_target_trajectory_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_bytes(enc(report));print(json.dumps(report),flush=True)
