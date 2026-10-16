from pathlib import Path
import datetime, gzip, hashlib, json, os, stat, subprocess, tempfile

Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
C=Q/'checkout-publication032'
D=C/'matter/examples/cardboard/evidence/native-implementation-20261014'
P=Q/'analysis/publication037-extension-source-001'
O=Q/'analysis/publication037-extension-output-001'
HEAD='76b3fdbd822bd3179c1937c4ebea7a17ce131cb1'
OLD_SUMMARY='29945277f4d8ae8a9577602393a1e2eca7643f0496f151f23f39affd6043f154'
OLD_BINDING='7f0d93661aa8785073a7b6de1a2dbb721eae65a288b39376b0495656913d4'
OLD_SELECTION='19c03d3e47928e67d9c7e1837d2caf715c52b522a82cc3f873bec87a68191a33'
BRANCH_REPORT='645028cb32280a3c0e82b5fa53784d61c9de805b20b59c819bc37c75951fdb01'
BRANCH_MANIFEST='a4db5cd7ecac68f3e4860de01baad806509c822ff5c83a2dc3d42785504d60b7'
SOURCE210='477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931'
SOURCE206='846aa41eab31797886190cc50df512a6af2d49c981244e1448cc1b024c913340'
CASES=[
'analysis/native541-critical23-input-002',
'analysis/native541-critical23-adjudication-001',
'analysis/native541-critical23-branch-adjudication-001',
'native/critical23-material-replay-541',
'analysis/publication037-extension-source-001'
]
ACTIVE=[
'analysis/native545-readiness-001',
'analysis/fresh-source210-strict-inner-full-coupon-545-command-001',
'solver/fresh-source210-strict-inner-full-coupon-545'
]
EXCLUDED_DIRS={'.git','__pycache__','CMakeFiles'}
MAX_STORED=50_000_000

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def enc(x): return (json.dumps(x,indent=2,allow_nan=False,sort_keys=False)+'\n').encode()
def git(*args): return subprocess.check_output(['git',*args],cwd=C)
def atomic(path,data,mode=None,replace=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        assert path.is_file() and path.read_bytes()==data, 'different existing target: '+str(path)
        return
    fd,tmp=tempfile.mkstemp(prefix='.publication037-extension-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if mode is not None: os.chmod(tmp,mode)
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def files_for(root):
    for d,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(x for x in dirs if x not in EXCLUDED_DIRS and not x.startswith('.air'))
        for n in sorted(names):
            p=Path(d)/n
            assert p.is_file() and not p.is_symlink(), str(p)
            yield p

assert git('rev-parse','HEAD').decode().strip()==HEAD
summary_path=D/'summary.json'; binding_path=D/'publication-binding-reference037.json'
selection_path=D/'publication037-case-selection.json'
old_summary_bytes=summary_path.read_bytes()
old_binding_bytes=binding_path.read_bytes()
old_selection_bytes=selection_path.read_bytes()
assert sha_bytes(old_summary_bytes)==OLD_SUMMARY
assert sha_bytes(old_binding_bytes)==OLD_BINDING
assert sha_bytes(old_selection_bytes)==OLD_SELECTION
old_summary=json.loads(old_summary_bytes)
old_binding=json.loads(old_binding_bytes)
old_selection=json.loads(old_selection_bytes)
assert len(old_summary['records'])==1926
assert len(old_binding['cases'])==47 and len(old_selection['cases'])==47
assert old_summary['records'][:1879]==json.loads((D/'summary-reference036.json').read_bytes())['records']
assert sha_file(Q/'analysis/native541-critical23-branch-adjudication-001/report.json')==BRANCH_REPORT
assert sha_file(Q/'analysis/native541-critical23-branch-adjudication-001/manifest.json')==BRANCH_MANIFEST
branch=json.loads((Q/'analysis/native541-critical23-branch-adjudication-001/report.json').read_text())
assert branch['execution']['rows']==138
assert branch['execution']['projection_converged']==138
assert branch['execution']['stress_valid']==138 and branch['execution']['tangent_valid']==138
assert branch['execution']['physical_fem_transactions']==0
assert branch['finite_difference_scope']['native_signal_exceeds_endpoint_discrepancy_sum']==30
assert branch['finite_difference_scope']['native_signal_not_above_endpoint_discrepancy_sum']==16
assert branch['finite_difference_scope']['error_metrics']['native_centered_fd_relative_to_strict_tangent']['all_46_le_1e-3']==10
assert branch['finite_difference_scope']['error_metrics']['native_centered_fd_relative_to_strict_tangent']['all_46_le_1e-6']==0
assert branch['branch_indicators']['y_sign_mismatch_details']
assert 'not proof of switching a discontinuous constitutive branch' in branch['indicator_erratum']
assert all((Q/c).is_dir() for c in CASES)
assert not set(CASES).intersection(old_binding['cases'])
assert not any('/545' in x for x in CASES)

# Preserve exact pre-extension candidate metadata outside the checkout.
atomic(O/'summary037-pre541.json',old_summary_bytes,0o644)
atomic(O/'binding037-pre541.json',old_binding_bytes,0o644)
atomic(O/'selection037-pre541.json',old_selection_bytes,0o644)

prior={}
for rec in old_summary['records']:
    for f in rec.get('files',[]):
        rp=f.get('repository_copy') or f.get('repository_compressed_copy')
        if rp:
            assert rp not in prior
            prior[rp]=f
records=[]; planned=[]; original_bytes=0; stored_bytes=0; raw_n=0; gzip_n=0
for case in CASES:
    root=Q/case
    rows=[]
    for p in files_for(root):
        b=p.read_bytes(); h=sha_bytes(b); mode=stat.S_IMODE(p.stat().st_mode)
        rel=str(p.relative_to(root)); target=Path(case)/rel
        rp=str(target.with_name(target.name+'.gz')) if len(b)>262144 and p.suffix!='.gz' else str(target)
        assert rp not in prior, 'unexpected record collision '+rp
        payload=gzip.compress(b,compresslevel=6,mtime=0) if rp!=str(target) else b
        if rp!=str(target): assert gzip.decompress(payload)==b
        planned.append({'case':case,'external_path':str(p),'repository_path':rp,'bytes':len(b),'sha256':h,'compressed':rp!=str(target),'stored_bytes':len(payload),'stored_sha256':sha_bytes(payload),'mode_octal':oct(mode)})
        rows.append((p,rel,target,rp,b,h,mode,payload))
        original_bytes+=len(b); stored_bytes+=len(payload)
    assert rows, 'empty case '+case
    records.append((case,root,rows))
assert stored_bytes<=MAX_STORED, stored_bytes
preflight={'schema':'numi.cardboard.publication037-extension-preflight.v1','state':'PREFLIGHT_PASS','parent':HEAD,
'prior_summary_sha256':OLD_SUMMARY,'prior_binding_sha256':OLD_BINDING,'prior_case_selection_sha256':OLD_SELECTION,
'prior_case_count':47,'extension_case_count':len(CASES),'extension_cases':CASES,'active_exclusions':ACTIVE,
'critical23_report_sha256':BRANCH_REPORT,'critical23_manifest_sha256':BRANCH_MANIFEST,
'payload_count':len(planned),'original_payload_bytes':original_bytes,'stored_payload_bytes':stored_bytes,'max_stored_payload_bytes':MAX_STORED,
'qualification_axes_open':6,'physical_box_qualification':False,'native545_outcome':'active and excluded without result','planned_payloads':planned}
atomic(O/'preflight.json',enc(preflight),0o644)

# Install each payload only after all selected source files and critical23 assertions passed.
extension_records=[]; installed=0
for case,root,rows in records:
    fs=[]
    for p,rel,target,rp,b,h,mode,payload in rows:
        dest=D/rp
        atomic(dest,payload,0o644 if rp.endswith('.gz') else mode)
        restored=gzip.decompress(dest.read_bytes()) if rp.endswith('.gz') else dest.read_bytes()
        assert restored==b and sha_bytes(restored)==h and len(restored)==len(b)
        assert stat.S_IMODE(dest.stat().st_mode)==(0o644 if rp.endswith('.gz') else mode)
        x={'name':rel,'bytes':len(b),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
        if rp.endswith('.gz'):
            x.update({'repository_compressed_copy':rp,'compressed_bytes':len(payload),'compressed_sha256':sha_bytes(payload),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
            gzip_n+=1
        else:
            x.update({'repository_copy':rp,'restore':'copy exact bytes and restore recorded mode'})
            raw_n+=1
        fs.append(x); installed+=len(payload)
    extension_records.append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':fs})

summary=json.loads(old_summary_bytes)
assert summary['records']==old_summary['records']
summary['records'].extend(extension_records)
assert len(summary['records'])==1931
assert summary['records'][:1926]==old_summary['records']
q=summary['qualification']
extra=(' Native541 critical23 replay is bounded to 23 selected tet/basis pairs (138 valid native projection/stress/tangent rows; zero physical FEM transactions). All 138 selected tangent rows agree with the strict FP64 tangent within 1e-3 (maximum relative error 2.961412e-4). Twelve y-sign indicator differences occur inside the smooth positive-FB / smoothed-hardening equations; they are not discontinuous constitutive branch crossings. Native finite differences remain unqualified: of 46 two-power comparisons, 30 signals exceed their measured endpoint-error sum and 16 do not; 10 of the informative 30 are within 1e-3, and none are within 1e-6. Those counts describe this selected replay only; 1e-6 is not imposed as a native-FD acceptance criterion. The batch is not all nine directions, material calibration, or coupon qualification.')
q['component_calibration']=q['component_calibration']+extra
summary['qualification']=q
summary['milestone']=(summary.get('milestone','')+' Native541 critical23 adds a source-bound, zero-transaction material replay with selected tangent agreement and explicit smooth-law branch-indicator adjudication. Its native finite-difference comparisons remain unqualified, and no calibration or full-coupon claim follows.')
summary_data=enc(summary); summary_sha=sha_bytes(summary_data)

selection={'schema':'numi.cardboard.publication037-case-selection.v2','parent':HEAD,
'cases':old_selection['cases']+CASES,'excluded_active':ACTIVE,
'source210_frozen_sha256':SOURCE210,'source206_reference_frozen_sha256':SOURCE206,
'critical23_branch_report_sha256':BRANCH_REPORT,'critical23_branch_manifest_sha256':BRANCH_MANIFEST,
'all_six_qualification_axes_open':True,'full_coupon_qualification':False}
selection_bytes=enc(selection); selection_sha=sha_bytes(selection_bytes)
binding=dict(old_binding)
binding['schema']='numi.cardboard.reference037-publication-binding.v2'
binding['prior_candidate_binding_sha256']=OLD_BINDING
binding['new_case_records']=len(binding['cases'])+len(CASES)
binding['extension_case_records']=len(CASES)
binding['cases']=old_binding['cases']+CASES
binding['case_selection_sha256']=selection_sha
binding['copied_files']=old_binding['copied_files']+sum(len(x['files']) for x in extension_records)
binding['raw_files']=old_binding['raw_files']+raw_n
binding['gzip_files']=old_binding['gzip_files']+gzip_n
binding['original_payload_bytes']=old_binding['original_payload_bytes']+original_bytes
binding['stored_payload_bytes']=old_binding['stored_payload_bytes']+installed
binding['summary_sha256']=summary_sha
binding['critical23_material_replay']={
'report_sha256':sha_file(Q/'analysis/native541-critical23-adjudication-001/report.json'),
'branch_report_sha256':BRANCH_REPORT,'branch_manifest_sha256':BRANCH_MANIFEST,
'input_sha256':sha_file(Q/'analysis/native541-critical23-input-002/input.json'),
'native_receipt_sha256':sha_file(Q/'native/critical23-material-replay-541/receipt.json'),
'native_output_sha256':sha_file(Q/'native/critical23-material-replay-541/output/paper-points-exact_package_budget32.jsonl'),
'selected_rows':138,'tangent_relative_error_max':branch['measurements']['max_tangent_rel_l2'],
'fd_pairs':46,'native_signal_above_measured_endpoint_error_sum':30,'native_signal_not_above_measured_endpoint_error_sum':16,
'informative_pairs_at_or_below_1e-3':10,'informative_pairs_at_or_below_1e-6':0,
'y_sign_indicators_smooth_law_mismatches':len(branch['branch_indicators']['y_sign_mismatch_details']),
'classification':'bounded native material replay; tangent comparison in scope; native FD unqualified; no physical FEM transaction, calibration or coupon qualification'}
binding['bounded_results']['native541_critical23_branch_report_sha256']=BRANCH_REPORT
binding['bounded_results']['native541_critical23_branch_manifest_sha256']=BRANCH_MANIFEST
binding['bounded_results']['native541_critical23_tangent_max_relative_error']=branch['measurements']['max_tangent_rel_l2']
binding['bounded_results']['native541_critical23_native_fd_scope']='46 two-power comparisons: 30 signals exceed measured endpoint-error sum, 16 do not; 10/30 informative pairs <=1e-3, 0/30 <=1e-6; descriptive counts only, no native-FD acceptance threshold asserted'
binding['qualification']['all_six_axes_open']=True
binding['qualification']['full_coupon']=False
binding['qualification']['physical_assembly']=False
binding['qualification']['crush_impact_prediction']=False
binding['critical23_cases']=CASES
binding['restore_policy']=old_binding['restore_policy']
binding_data=enc(binding); binding_sha=sha_bytes(binding_data)

# Verify 1879-record historical prefix and previous 47 records unchanged before replacing summary/binding.
assert summary['records'][:1879]==json.loads((D/'summary-reference036.json').read_bytes())['records']
assert summary['records'][1879:1926]==old_summary['records'][1879:1926]
atomic(D/'publication037-case-selection.json',selection_bytes,0o644,replace=True)
atomic(D/'publication-binding-reference037.json',binding_data,0o644,replace=True)
atomic(D/'summary.json',summary_data,0o644,replace=True)

# Full intended path allowlist: all source210 delta entries, every archived copy, and metadata.
source_delta=old_binding['source210']['changed_sources'].keys()
payload_paths=set()
for rec in summary['records']:
    for f in rec.get('files',[]):
        rp=f.get('repository_copy') or f.get('repository_compressed_copy')
        if rp: payload_paths.add(rp)
meta={'matter/examples/cardboard/evidence/native-implementation-20261014/summary.json',
'matter/examples/cardboard/evidence/native-implementation-20261014/summary-reference036.json',
'matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference037.json',
'matter/examples/cardboard/evidence/native-implementation-20261014/publication037-case-selection.json'}
intended=set(source_delta)|payload_paths|meta
paths=sorted(intended)
nul=b'\0'.join(x.encode() for x in paths)+b'\0'
# Ensure ignored artifacts are present in the allowlist; record them rather than staging them.
ignored=set(git('ls-files','--others','--ignored','--exclude-standard').decode().splitlines())
tracked=set(git('diff','--name-only').decode().splitlines())
untracked=set(git('ls-files','--others','--exclude-standard').decode().splitlines())
actual=tracked|untracked|ignored
missing=sorted(intended-actual)
unexpected=sorted(actual-intended)
assert not missing, 'intended paths absent from worktree: '+repr(missing[:20])
# Keep active545 outputs out even if they are untracked in the checkout.
assert not any(any(x in p for x in ACTIVE) for p in intended)
atomic(O/'proposed-paths-complete.nul',nul,0o644,replace=True)
receipt={'schema':'numi.cardboard.publication037-extension-receipt.v1','state':'ARCHIVE_COMPLETE_SOURCE_CANDIDATE_NOT_STAGED',
'parent':HEAD,'head_after':git('rev-parse','HEAD').decode().strip(),'staged_paths':git('diff','--cached','--name-only').decode().splitlines(),
'prior_summary_sha256':OLD_SUMMARY,'summary_after_sha256':summary_sha,
'prior_binding_sha256':OLD_BINDING,'binding_after_sha256':binding_sha,
'prior_case_selection_sha256':OLD_SELECTION,'case_selection_after_sha256':selection_sha,
'prior_records':1926,'extension_records':len(extension_records),'total_records':len(summary['records']),
'extension_payload_count':sum(len(r['files']) for r in extension_records),'extension_raw_files':raw_n,'extension_gzip_files':gzip_n,
'extension_original_bytes':original_bytes,'extension_stored_bytes':installed,
'total_payload_records':sum(len(r.get('files',[])) for r in summary['records']),
'intended_path_count':len(paths),'tracked_diff_path_count':len(tracked),'untracked_nonignored_path_count':len(untracked),'ignored_untracked_path_count':len(ignored),
'missing_intended_paths':missing,'unexpected_worktree_paths':unexpected,
'active545_excluded':ACTIVE,'qualification_axes_open':6,'historical_prefix_exact':True,
'previous47_records_exact':True,'all_extension_payloads_hash_restore_checked':True,
'critical23_report_sha256':BRANCH_REPORT,'critical23_manifest_sha256':BRANCH_MANIFEST,
'critical23_measurements':{'rows':138,'max_tangent_rel_error':branch['measurements']['max_tangent_rel_l2'],'native_fd_signal_above_error':30,'native_fd_signal_not_above_error':16,'informative_native_fd_le_1e-3':10,'informative_native_fd_le_1e-6':0,'smooth_y_sign_indicator_mismatches':len(branch['branch_indicators']['y_sign_mismatch_details'])},
'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
atomic(O/'extension-receipt.json',enc(receipt),0o644,replace=True)
extension_manifest={'schema':'numi.cardboard.publication037-extension-source.v1','files':{
'archive_extension037.py':sha_file(P/'archive_extension037.py'),'README.md':sha_file(P/'README.md'),
'preflight.json':sha_file(O/'preflight.json'),'extension-receipt.json':sha_file(O/'extension-receipt.json'),
'proposed-paths-complete.nul':sha_file(O/'proposed-paths-complete.nul'),
'summary037-pre541.json':sha_file(O/'summary037-pre541.json'),'binding037-pre541.json':sha_file(O/'binding037-pre541.json'),
'selection037-pre541.json':sha_file(O/'selection037-pre541.json')}}
atomic(O/'extension-manifest.json',enc(extension_manifest),0o644,replace=True)
print(json.dumps(receipt,indent=2,allow_nan=False))
