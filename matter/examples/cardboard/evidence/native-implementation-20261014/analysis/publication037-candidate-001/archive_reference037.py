from pathlib import Path
import datetime, gzip, hashlib, json, os, stat, subprocess, tempfile

Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
C=Q/'checkout-publication032'
REL=Path('matter/examples/cardboard')
EVIDREL=REL/'evidence/native-implementation-20261014'
D=C/EVIDREL
SNAP210=Q/'snapshots/source-composed-strict-periodic-freebox-210-buildfix-003'
SNAP206=Q/'snapshots/source-stable-green-reference-206-buildfix-002'
O=Q/'analysis/publication037-payload-preflight-001'
P=Q/'analysis/publication037-candidate-001'
EXPECTED_HEAD='76b3fdbd822bd3179c1937c4ebea7a17ce131cb1'
OLD_SUMMARY_SHA='b16415655a44a86f83d4f81d06c260a5c95ab2f79281f4e3923ed726cbc6614c'
OLD_BINDING_SHA='42242bd1f2d14bf020bf656bc7496c713a725eaea3908b5bb5f1b4b7522a084f'
SOURCE210_SHA='477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931'
SOURCE210_BUILD_SHA='75f9a256c4413c0a32e877f2dfcc168633c7662b391690dd4b83dbd4819f8e93'
SOURCE206_SHA='846aa41eab31797886190cc50df512a6af2d49c981244e1448cc1b024c913340'
MAX_COMPRESSED_BYTES=200_000_000
CASES=[
'analysis/source204-snapshot-freeze-001',
'analysis/source204-topology-frame-export-001',
'analysis/source204-topology-frame-export-002',
'analysis/source204-topology-frame-export-003',
'analysis/source204-topology-frame-export-004',
'analysis/source204-topology-frame-export-full-suite-005',
'analysis/source205-cohesive-quadrature-area-001',
'analysis/source205-independent-owner-review-001',
'analysis/source205-true-residual-operator-audit-001',
'analysis/source207-free-box-true-residual-001',
'analysis/source207-free-box-true-residual-buildfix-002',
'analysis/source208-periodic-accepted-checkpoint-001',
'analysis/source209-strict-inner-forcing-001',
'analysis/source209-strict-inner-forcing-buildfix-002',
'analysis/source209-strict-inner-forcing-owner-review-001',
'analysis/source-composed-strict-periodic-freebox-210-composition-001',
'analysis/source-composed-strict-periodic-freebox-210-buildfix-002-composition-001',
'analysis/source-composed-strict-periodic-freebox-210-buildfix-003-composition-001',
'analysis/source210-combined-build-001',
'analysis/source210-runtime-fgmres-review-001',
'analysis/source210-publication-binding-001',
'analysis/native539-predecessor-readiness-001',
'analysis/native539-first-failure-adjudication-001',
'analysis/native539-source203-cauchy-all-fallback-command-001',
'solver/fresh-source203-cauchy-all-fallback-coupon-539',
'native/strict-inner-forcing-helper-540',
'analysis/native541-center-input-001',
'analysis/native541-center-parity-001',
'analysis/native541-generic-package-batch-instrument-001',
'native/captured-centers-541',
'native/free-box-true-residual-542',
'native/combined-freebox-strict-inner-543',
'analysis/source206-stable-green-reference-001',
'analysis/source206-stable-green-reference-buildfix-002',
'analysis/source206-reference-binding-001',
'analysis/native531-stable-green-strain-reference-017-001',
'analysis/native531-stable-green-strain-invariants-018-001',
'analysis/native531-stable-green-strain-invariants-018-002',
'analysis/native531-stable-green-strain-invariants-018-003',
'analysis/native531-stable-green-strain-overlay-019-001',
'analysis/native544-periodic-observer-parity-001',
'analysis/native544-periodic-observer-parity-002',
'analysis/source210-periodic-observer-off-544-command-001',
'analysis/source210-periodic-observer-on-544-command-001',
'solver/source210-periodic-observer-off-544',
'solver/source210-periodic-observer-on-544',
'analysis/publication037-candidate-001'
]
ACTIVE_EXCLUDED=[
'analysis/native545-readiness-001',
'analysis/fresh-source210-strict-inner-full-coupon-545-command-001',
'solver/fresh-source210-strict-inner-full-coupon-545'
]
OTHER_EXCLUDED=['other mutable work products and active native runs']
EXPECTED_CHANGED=[
'matter/CMakeLists.txt','matter/src/metal/fgmres.metalinc','matter/src/runtime.mm',
'matter/tools/cardboard_cached_operator_audit.py','matter/tools/cardboard_cohesive_capacity_area.hpp',
'matter/tools/cardboard_cohesive_capacity_area_check.cpp','matter/tools/cardboard_coupled_free_box_operator_check.mm',
'matter/tools/cardboard_coupon_resume.hpp','matter/tools/cardboard_coupon_resume_check.cpp','matter/tools/cardboard_probe.mm',
'matter/tools/cardboard_shipping_snapshot.hpp','matter/tools/cardboard_shipping_snapshot_check.cpp',
'matter/tools/cardboard_solver_environment.py','matter/tools/cardboard_stage_resolution_study.md',
'matter/tools/cardboard_stage_resolution_study.py','matter/tools/fgmres_true_residual_native_check.mm',
'matter/tools/test_cardboard_cached_operator_audit.py','matter/tools/test_cardboard_solver_environment.py',
'matter/tools/test_cardboard_stage_resolution_study.py','matter/tools/test_true_residual_certification_contract.py'
]
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def enc(x): return (json.dumps(x,indent=2,allow_nan=False,sort_keys=False)+'\n').encode()
def git(*args): return subprocess.check_output(['git',*args],cwd=C)
def atomic_replace(path,data,mode=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.publication037-replace-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if mode is not None: os.chmod(tmp,mode)
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise
def atomic_put(path,data,mode=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.is_file() and path.read_bytes()==data, 'nonidentical existing target: '+str(path)
        if mode is not None: os.chmod(path,mode)
        return
    fd,tmp=tempfile.mkstemp(prefix='.publication037-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if mode is not None: os.chmod(tmp,mode)
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise

assert git('rev-parse','HEAD').decode().strip()==EXPECTED_HEAD
assert D.is_dir() and SNAP210.is_dir() and SNAP206.is_dir()
assert sha_file(D/'summary.json')==OLD_SUMMARY_SHA
assert sha_file(D/'publication-binding-reference036.json')==OLD_BINDING_SHA
assert len(CASES)==len(set(CASES)) and len(CASES)==47
assert all((Q/c).is_dir() for c in CASES), [c for c in CASES if not (Q/c).is_dir()]
for rel in ACTIVE_EXCLUDED: assert (Q/rel).exists(), 'missing active exclusion '+rel
assert any(c.startswith('analysis/source204-') for c in CASES) and any(c.startswith('analysis/source205-') for c in CASES)
# Exact source210 map and build binding.
fm=json.loads((SNAP210/'frozen-source.json').read_text())
assert sha_file(SNAP210/'frozen-source.json')==SOURCE210_SHA
assert len(fm['source_sha256'])==819
build=json.loads((Q/'build-source210-combined-001/source-build.json').read_text())
assert sha_file(Q/'build-source210-combined-001/source-build.json')==SOURCE210_BUILD_SHA
assert build['success'] is True and build['source_sha256']==fm['source_sha256']
receipt=json.loads((Q/'analysis/source210-combined-build-001/receipt.json').read_text())
assert receipt['source_frozen_sha256']==SOURCE210_SHA and receipt['source_build_sha256']==SOURCE210_BUILD_SHA
assert all(receipt['commands'][i]['exit_code']==0 for i in range(len(receipt['commands'])))
delta=[]
for rel,digest in fm['source_sha256'].items():
    p=C/rel
    assert p.is_file() and not p.is_symlink() and sha_file(p)==digest, 'source210 mismatch '+rel
    old=subprocess.run(['git','show',EXPECTED_HEAD+':'+rel],cwd=C,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    oldhash=sha_bytes(old.stdout) if old.returncode==0 else None
    if oldhash!=digest: delta.append(rel)
assert sorted(delta)==sorted(EXPECTED_CHANGED), delta
# Source206 remains a separate reference-only package.
fm206=json.loads((SNAP206/'frozen-source.json').read_text())
assert sha_file(SNAP206/'frozen-source.json')==SOURCE206_SHA and len(fm206['source_sha256'])==821
r206=json.loads((Q/'analysis/source206-stable-green-reference-buildfix-002/report.json').read_text())
assert r206['frozen_source_sha256']==SOURCE206_SHA and r206['source_count']==821 and r206['cpu_exit_code']==0 and not r206['native_gpu_executed']
r544=json.loads((Q/'analysis/native544-periodic-observer-parity-002/report.json').read_text())
assert r544['verdict']=='PASS' and r544['source_frozen_sha256']==SOURCE210_SHA and r544['complete_controls']==7
m544=json.loads((Q/'analysis/native544-periodic-observer-parity-002/manifest.json').read_text())
assert sha_file(Q/'analysis/native544-periodic-observer-parity-002/report.json')=='99e448cb3b60959f9bdac715707a4ad295ec042282398c299b54c57ea3cd8735'
assert sha_file(Q/'analysis/native544-periodic-observer-parity-002/manifest.json')=='0dffb7ac62d021f0f6c7b138b06f3ebcf3e733623e725508a2693ee72b83aae9'
# Parent checkout must contain exactly the source210 delta before archive payload insertion.
status=git('status','--porcelain=v1','--untracked-files=all').decode().splitlines()
tracked={x[3:] for x in status if not x.startswith('?? ')}
untracked={x[3:] for x in status if x.startswith('?? ')}
added_sources={"matter/tools/cardboard_cohesive_capacity_area.hpp","matter/tools/cardboard_cohesive_capacity_area_check.cpp"}
assert tracked==set(EXPECTED_CHANGED)-added_sources, tracked
assert untracked=={'matter/tools/cardboard_cohesive_capacity_area.hpp','matter/tools/cardboard_cohesive_capacity_area_check.cpp'}, untracked

selection={'schema':'numi.cardboard.publication037-case-selection.v1','parent':EXPECTED_HEAD,'cases':CASES,'excluded_active':ACTIVE_EXCLUDED,'source210_frozen_sha256':SOURCE210_SHA,'source206_reference_frozen_sha256':SOURCE206_SHA,'native544_parity_report_sha256':sha_file(Q/'analysis/native544-periodic-observer-parity-002/report.json'),'all_six_qualification_axes_open':True,'full_coupon_qualification':False}
selection_bytes=enc(selection)
atomic_put(O/'case-selection.json',selection_bytes)
atomic_put(P/'README.md',b'# Publication037 candidate\n\nThis archive candidate appends evidence to the exact publication036 history without closing any qualification axis. Source210 is the canonical 819-file source/build binding. Source206 and stable-Green 017/018/019 artifacts are reference-only CPU evidence. Native544 is a seven-control 0.01-degree observer off/on parity check. Active native545 is excluded.\n')
excluded_dirs={'.git','__pycache__','CMakeFiles'}
old_bytes=(D/'summary.json').read_bytes()
old_summary=json.loads(old_bytes)
assert len(old_summary['records'])==1879
assert sha_file(D/'publication-binding-reference036.json')==OLD_BINDING_SHA
prior_paths={}
for rec in old_summary['records']:
    for f in rec.get('files',[]):
        rp=f.get('repository_copy') or f.get('repository_compressed_copy')
        if rp: assert rp not in prior_paths; prior_paths[rp]=f
old_cases={r.get('case') for r in old_summary['records']}
assert not old_cases.intersection(CASES)
planned=[]; raw_total=0; stored_estimate=0
for case in CASES:
    root=Q/case; case_files=0
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in excluded_dirs and not d.startswith('.air'))
        for name in sorted(names):
            p=Path(directory)/name
            assert p.is_file() and not p.is_symlink(), str(p)
            b=p.read_bytes(); mode=stat.S_IMODE(p.stat().st_mode); h=sha_bytes(b)
            target=Path(case)/p.relative_to(root); raw_total+=len(b); case_files+=1
            if str(target) in prior_paths:
                f=prior_paths[str(target)]
                existing=(D/target).read_bytes()
                if f.get('repository_compressed_copy'): existing=gzip.decompress(existing)
                assert f['bytes']==len(b) and f['sha256']==h and existing==b
                planned.append({'case':case,'source':str(p),'target':str(target),'kind':'historical_overlap','bytes':len(b),'sha256':h,'mode':mode})
                continue
            if len(b)>262144 and p.suffix!='.gz':
                payload=gzip.compress(b,compresslevel=6,mtime=0); assert gzip.decompress(payload)==b
                stored=target.with_name(target.name+'.gz'); kind='gzip'
            else: payload=b; stored=target; kind='raw'
            stored_estimate+=len(payload)
            planned.append({'case':case,'source':str(p),'target':str(stored),'kind':kind,'bytes':len(b),'stored_bytes':len(payload),'sha256':h,'stored_sha256':sha_bytes(payload),'mode':mode})
    assert case_files>0, 'empty case '+case
assert raw_total<500_000_000 and stored_estimate<MAX_COMPRESSED_BYTES, (raw_total,stored_estimate)
assert not any('/545' in x['source'] for x in planned)
preflight={'schema':'numi.cardboard.publication037-payload-preflight.v1','state':'PREFLIGHT_PASS','parent':EXPECTED_HEAD,'old_summary_sha256':OLD_SUMMARY_SHA,'old_binding_sha256':OLD_BINDING_SHA,'source210_frozen_sha256':SOURCE210_SHA,'source210_build_sha256':SOURCE210_BUILD_SHA,'source210_source_count':819,'source206_frozen_sha256':SOURCE206_SHA,'source206_source_count':821,'case_count':len(CASES),'cases':CASES,'active_exclusions':ACTIVE_EXCLUDED,'raw_payload_bytes':raw_total,'estimated_stored_payload_bytes':stored_estimate,'max_compressed_payload_bytes':MAX_COMPRESSED_BYTES,'planned_files':planned,'directory_exclusions':sorted(excluded_dirs),'qualification_axes_open':6,'native545_outcome':'active at publication cutoff; excluded without outcome'}
atomic_put(O/'preflight-report.json',enc(preflight))
# Install deterministic payloads only after every selected input and source map has passed preflight.
records=[]; overlaps=[]; stored_total=0; payload_total=0; raw_count=0; gzip_count=0
for case in CASES:
    root=Q/case; files=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in excluded_dirs and not d.startswith('.air'))
        for name in sorted(names):
            p=Path(directory)/name; b=p.read_bytes(); h=sha_bytes(b); mode=stat.S_IMODE(p.stat().st_mode)
            target=Path(case)/p.relative_to(root)
            if str(target) in prior_paths:
                overlaps.append({'case':case,'name':str(p.relative_to(root)),'sha256':h}); continue
            f={'name':str(p.relative_to(root)),'bytes':len(b),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
            if len(b)>262144 and p.suffix!='.gz':
                payload=gzip.compress(b,compresslevel=6,mtime=0); assert gzip.decompress(payload)==b
                stored=target.with_name(target.name+'.gz'); atomic_put(D/stored,payload,0o644)
                f.update({'repository_compressed_copy':str(stored),'compressed_bytes':len(payload),'compressed_sha256':sha_bytes(payload),'restore':'gunzip; restore recorded mode; verify original bytes and SHA-256'})
                gzip_count+=1
            else:
                payload=b; stored=target; atomic_put(D/stored,payload,mode)
                f.update({'repository_copy':str(stored),'restore':'copy exact bytes and restore recorded mode'})
                raw_count+=1
            stored_total+=len(payload); payload_total+=len(b); files.append(f)
    if files: records.append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':files})
assert len(records)==len(CASES)
assert stored_total<=stored_estimate
for rec in records:
    for f in rec['files']:
        stored_path=D/(f.get('repository_copy') or f.get('repository_compressed_copy'))
        buf=stored_path.read_bytes()
        restored=gzip.decompress(buf) if f.get('repository_compressed_copy') else buf
        assert len(restored)==f['bytes'] and sha_bytes(restored)==f['sha256'] and restored==Path(f['external_path']).read_bytes()
        assert stat.S_IMODE(stored_path.stat().st_mode)==(0o644 if f.get('repository_compressed_copy') else int(f['mode_octal'],8))
atomic_put(D/'summary-reference036.json',old_bytes)
summary=json.loads(old_bytes)
summary.setdefault('historical_summary_snapshots',{})['reference036']={'path':'summary-reference036.json','records':1879,'sha256':OLD_SUMMARY_SHA}
summary['records'].extend(records)
assert len(summary['records'])==1879+len(records) and summary['records'][:1879]==old_summary['records']
qual=summary['qualification']
qual['solver_correctness']=qual['solver_correctness']+' Native544 source210 observer off/on run completes seven small 0.01-degree controls; the independent parity audit reports all32 production-root files byte-identical, including four checkpoint snapshots. This establishes bounded observer parity only.'
qual['numerical_convergence']=('Fresh source203 native539 completed19 accepted controls, then failed control19 after64 Newton iterations with residual6.3575385e-7 above the unchanged5e-7 nonlinear limit. CGLS produced nonzero directions at selected Newton iterations but did not close the remaining gap. Source201 native535 also failed at control17/Newton34. Native545, a separate source210 full coupon, was active at the publication cutoff and is excluded without an outcome. Native544 is only a seven-control 0.01-degree observer parity fixture. No complete60-control coupon or mesh/time refinement series is qualified.')
qual['component_calibration']=qual['component_calibration']+' The source206 stable-Green strain implementation is an explicit CPU reference overlay with 14 CPU checks; its equations are algebraically equivalent and its selected regression does not establish material calibration or production trajectory behavior.'
summary['qualification']=qual
summary['milestone']=(summary.get('milestone','')+' Source210 composes periodic accepted-checkpoint, strict inner forcing, and free-box/operator checks on an exact819-file frozen source map. Its source-bound build and CPU contracts pass; native543 is a synthetic helper with zero physical transactions. Native544 observer off/on completes seven controls with 32 production-root files byte-identical and four checkpoint-prefix comparisons; it is not a full coupon. Source206 stable-Green work and MP017/018/019 are retained as reference-only arithmetic evidence. Native539 failed after19 accepted controls; native545 was active at cutoff and is excluded. All six qualification axes remain open.')
summary_data=enc(summary)
assert json.loads(summary_data)['records'][:1879]==old_summary['records']
summary_sha=sha_bytes(summary_data)
# Construct the publication binding with exact build/source/observer/reference links and explicit qualification limits.
bsource=Q/'analysis/source210-publication-binding-001'
binding={
'schema':'numi.cardboard.reference037-publication-binding.v1','parent':EXPECTED_HEAD,'prior_binding_sha256':OLD_BINDING_SHA,
'historical_prefix':{'path':'summary-reference036.json','records':1879,'sha256':OLD_SUMMARY_SHA},
'new_case_records':len(records),'cases':CASES,'case_selection_sha256':sha_bytes(selection_bytes),
'copied_files':sum(len(x['files']) for x in records),'raw_files':raw_count,'gzip_files':gzip_count,'original_payload_bytes':payload_total,'stored_payload_bytes':stored_total,'summary_sha256':summary_sha,
'source210':{'frozen_source_sha256':SOURCE210_SHA,'selected_source_files':819,'source_build_sha256':SOURCE210_BUILD_SHA,'build_receipt_sha256':sha_file(bsource/'source210-build-receipt.json'),'composition_sha256':sha_file(bsource/'source210-composition.json'),'binding_sha256':sha_file(bsource/'binding.json'),'changed_sources':{rel:sha_file(C/rel) for rel in EXPECTED_CHANGED},'composed_checkout_matches_all_frozen_files':True,'build_cpu_checks_exit_zero':True,'gpu_build_qualification':False},
'source206_reference_only':{'frozen_source_sha256':SOURCE206_SHA,'selected_source_files':821,'reference_binding_sha256':sha_file(Q/'analysis/source206-reference-binding-001/binding.json'),'classification':'separate CPU reference overlay; not canonical source210 or material calibration'},
'bounded_results':{'native539_exit_code':2,'native539_accepted_controls':19,'native539_first_failure':'control19 after64 Newton iterations; residual6.3575385e-7 >5e-7','native544_report_sha256':sha_file(Q/'analysis/native544-periodic-observer-parity-002/report.json'),'native544_verdict':'PASS','native544_complete_controls':7,'native544_production_root_files_identical':32,'native544_checkpoint_controls_compared':[1,3,5,6],'native544_scope':'observer on/off parity only at0.01 degree; not full coupon','native545':'active at cutoff; excluded without outcome'},
'qualification':{'all_six_axes_open':True,'full_coupon':False,'full_protocol_qualification':False,'physical_assembly':False,'crush_impact_prediction':False},
'excluded_active':ACTIVE_EXCLUDED,'excluded_followups':OTHER_EXCLUDED,'verified_historical_overlaps':overlaps,
'restore_policy':'Compressed payloads are deterministic gzip files; restore by gunzip, restore recorded mode, and verify byte count plus SHA-256.'
}
binding_data=enc(binding); binding_sha=sha_bytes(binding_data)
atomic_put(D/'publication037-case-selection.json',selection_bytes,0o644)
atomic_put(D/'publication-binding-reference037.json',binding_data,0o644)
atomic_replace(D/'summary.json',summary_data)
# Close the archive receipt outside the repository. Nothing is staged or committed here.
new_status=git('status','--porcelain=v1','--untracked-files=all').decode().splitlines()
staged=git('diff','--cached','--name-only').decode().splitlines()
report={'schema':'numi.cardboard.publication037-archive-receipt.v1','state':'ARCHIVE_COMPLETE_SOURCE_CANDIDATE_NOT_STAGED','parent':EXPECTED_HEAD,'head_after':git('rev-parse','HEAD').decode().strip(),'staged_paths':staged,'summary_before_sha256':OLD_SUMMARY_SHA,'summary_after_sha256':summary_sha,'binding_sha256':binding_sha,'case_selection_sha256':sha_file(D/'publication037-case-selection.json'),'case_count':len(records),'file_count':sum(len(r['files']) for r in records),'raw_file_count':raw_count,'gzip_file_count':gzip_count,'original_bytes':payload_total,'stored_bytes':stored_total,'source_count':819,'source210_frozen_sha256':SOURCE210_SHA,'source206_reference_sha256':SOURCE206_SHA,'active_exclusions':ACTIVE_EXCLUDED,'qualification_axes_open':6,'historical_record_prefix_exact':json.loads(summary_data)['records'][:1879]==old_summary['records'],'all_new_payloads_restore_hash_exact':True,'all_selected_sources_match':True,'working_tree_status_count':len(new_status),'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
atomic_put(O/'archive-receipt.json',enc(report))
# Exact proposed repository path manifest, without staging or committing.
paths=sorted(set(git('diff','--name-only').decode().splitlines()) | set(git('ls-files','--others','--exclude-standard').decode().splitlines()))
nul=b'\0'.join(x.encode() for x in paths)+b'\0'
(O/'proposed-paths.nul').write_bytes(nul)
(O/'proposed-status-lines.json').write_bytes(enc({'status_lines':sorted(set(git('status','--porcelain=v1','--untracked-files=all').decode().splitlines())),'staged':False,'committed':False}))
manifest={'schema':'numi.cardboard.publication037-candidate-package.v1','files':{'archive_reference037.py':sha_file(P/'archive_reference037.py'),'README.md':sha_file(P/'README.md'),'case-selection.json':sha_file(O/'case-selection.json'),'preflight-report.json':sha_file(O/'preflight-report.json'),'archive-receipt.json':sha_file(O/'archive-receipt.json'),'proposed-paths.nul':sha_file(O/'proposed-paths.nul'),'proposed-status-lines.json':sha_file(O/'proposed-status-lines.json')}}
atomic_put(O/'manifest.json',enc(manifest))
print(json.dumps(report,indent=2,allow_nan=False))
