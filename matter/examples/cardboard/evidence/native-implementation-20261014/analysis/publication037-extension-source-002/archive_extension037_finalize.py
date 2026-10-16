from pathlib import Path
import datetime,gzip,hashlib,json,os,stat,subprocess,tempfile

Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
C=Q/'checkout-publication032'
D=C/'matter/examples/cardboard/evidence/native-implementation-20261014'
P=Q/'analysis/publication037-extension-source-002'
O=Q/'analysis/publication037-extension-output-002'
HEAD='76b3fdbd822bd3179c1937c4ebea7a17ce131cb1'
OLD_SUMMARY='64f250b44bae1b9f0256f70b2b20ec501f6356c920d4944aab1075eb89c1476b'
OLD_BINDING='99fe50f3f6a11657e7946e8f45f5517a2679818ba34a64d680c82c46f2a90c88'
OLD_SELECTION='3c5588835a859a4f39eec27304d304e5b6fb3b068f60938f1de86b37bcc7f3a6'
BRANCH_REPORT='645028cb32280a3c0e82b5fa53784d61c9de805b20b59c819bc37c75951fdb01'
BRANCH_MANIFEST='a4db5cd7ecac68f3e4860de01baad806509c822ff5c83a2dc3d42785504d60b7'
SOURCE210='477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931'
SOURCE206='846aa41eab31797886190cc50df512a6af2d49c981244e1448cc1b024c913340'
EVIDENCE_PREFIX='matter/examples/cardboard/evidence/native-implementation-20261014/'
CASE='analysis/publication037-extension-source-002'
ACTIVE=[
'analysis/native545-readiness-001',
'analysis/fresh-source210-strict-inner-full-coupon-545-command-001',
'solver/fresh-source210-strict-inner-full-coupon-545'
]
EXCLUDED_DIRS={'.git','__pycache__','CMakeFiles'}
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def enc(x): return (json.dumps(x,indent=2,allow_nan=False,sort_keys=False)+'\n').encode()
def git(*args): return subprocess.check_output(['git',*args],cwd=C)
def atomic(path,data,mode=None,replace=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        assert path.is_file() and path.read_bytes()==data, 'different existing target: '+str(path)
        return
    fd,tmp=tempfile.mkstemp(prefix='.publication037-finalize-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if mode is not None: os.chmod(tmp,mode)
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise
def walk(root):
    for d,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(x for x in dirs if x not in EXCLUDED_DIRS and not x.startswith('.air'))
        for name in sorted(names):
            p=Path(d)/name
            assert p.is_file() and not p.is_symlink(), str(p)
            yield p
def collect_records(summary):
    out={}
    for rec in summary['records']:
        for f in rec.get('files',[]):
            rp=f.get('repository_copy') or f.get('repository_compressed_copy')
            if rp:
                assert rp not in out, 'duplicate existing archive path '+rp
                out[rp]=f
    return out

assert git('rev-parse','HEAD').decode().strip()==HEAD
sp=D/'summary.json'; bp=D/'publication-binding-reference037.json'; cp=D/'publication037-case-selection.json'
old_sb=sp.read_bytes(); old_bb=bp.read_bytes(); old_cb=cp.read_bytes()
assert sha_bytes(old_sb)==OLD_SUMMARY and sha_bytes(old_bb)==OLD_BINDING and sha_bytes(old_cb)==OLD_SELECTION
summary=json.loads(old_sb); binding=json.loads(old_bb); selection=json.loads(old_cb)
assert len(summary['records'])==1931 and len(binding['cases'])==52 and len(selection['cases'])==52
assert summary['records'][:1879]==json.loads((D/'summary-reference036.json').read_bytes())['records']
assert summary['records'][1879:1926]==json.loads((Q/'analysis/publication037-extension-output-001/summary037-pre541.json').read_bytes())['records'][1879:1926]
assert not any(CASE in x for x in binding['cases'])
assert (P/'archive_extension037_finalize.py').is_file()
assert (P/'README.md').is_file()
assert (P/'failed-install-attempt001.json').is_file()
branch=json.loads((Q/'analysis/native541-critical23-branch-adjudication-001/report.json').read_text())
assert sha_file(Q/'analysis/native541-critical23-branch-adjudication-001/report.json')==BRANCH_REPORT
assert sha_file(Q/'analysis/native541-critical23-branch-adjudication-001/manifest.json')==BRANCH_MANIFEST
assert branch['execution']['rows']==138 and branch['execution']['projection_converged']==138
assert branch['execution']['stress_valid']==138 and branch['execution']['tangent_valid']==138
assert branch['execution']['physical_fem_transactions']==0
assert branch['finite_difference_scope']['native_signal_exceeds_endpoint_discrepancy_sum']==30
assert branch['finite_difference_scope']['native_signal_not_above_endpoint_discrepancy_sum']==16
assert branch['finite_difference_scope']['error_metrics']['native_centered_fd_relative_to_strict_tangent']['all_46_le_1e-3']==10
assert branch['finite_difference_scope']['error_metrics']['native_centered_fd_relative_to_strict_tangent']['all_46_le_1e-6']==0
assert len(branch['branch_indicators']['y_sign_mismatch_details'])==12

# Keep source-002 small, deterministic, and separate from the prior failed attempt package.
prior=collect_records(summary)
fs=[]
for p in walk(Q/CASE):
    data=p.read_bytes(); mode=stat.S_IMODE(p.stat().st_mode); h=sha_bytes(data)
    rel=str(p.relative_to(Q/CASE)); target=Path(CASE)/rel
    if len(data)>262144 and p.suffix!='.gz':
        stored=target.with_name(target.name+'.gz'); payload=gzip.compress(data,compresslevel=6,mtime=0); assert gzip.decompress(payload)==data
    else: stored=target; payload=data
    rp=str(stored)
    assert rp not in prior, 'payload path already exists '+rp
    dest=D/rp
    atomic(dest,payload,0o644 if rp.endswith('.gz') else mode)
    restored=gzip.decompress(dest.read_bytes()) if rp.endswith('.gz') else dest.read_bytes()
    assert restored==data and sha_bytes(restored)==h and len(restored)==len(data)
    entry={'name':rel,'bytes':len(data),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
    if rp.endswith('.gz'):
        entry.update({'repository_compressed_copy':rp,'compressed_bytes':len(payload),'compressed_sha256':sha_bytes(payload),'restore':'gunzip; restore recorded mode; verify byte count and SHA-256'})
    else:
        entry.update({'repository_copy':rp,'restore':'copy exact bytes and restore recorded mode'})
    fs.append(entry)
assert fs
new_record={'case':CASE,'external_root':str(Q/CASE),'physical_box_qualification':False,'files':fs}
summary=json.loads(old_sb); summary['records'].append(new_record)
assert len(summary['records'])==1932 and summary['records'][:1931]==json.loads(old_sb)['records']

# Append the finalizer record and update binding/selection without touching historical records.
selection=json.loads(old_cb)
selection['schema']='numi.cardboard.publication037-case-selection.v3'
selection['cases'].append(CASE)
selection['critical23_branch_report_sha256']=BRANCH_REPORT
selection['critical23_branch_manifest_sha256']=BRANCH_MANIFEST
selection_data=enc(selection); selection_sha=sha_bytes(selection_data)
binding=json.loads(old_bb)
binding['schema']='numi.cardboard.reference037-publication-binding.v3'
binding['prior_candidate_binding_sha256']=OLD_BINDING
binding['cases'].append(CASE)
binding['new_case_records']=len(binding['cases'])
binding['extension_case_records']=len(binding['cases'])-47
binding['case_selection_sha256']=selection_sha
binding['copied_files']+=len(fs)
binding['raw_files']+=sum(1 for x in fs if x.get('repository_copy'))
binding['gzip_files']+=sum(1 for x in fs if x.get('repository_compressed_copy'))
binding['original_payload_bytes']+=sum(x['bytes'] for x in fs)
binding['stored_payload_bytes']+=sum(x.get('compressed_bytes',x['bytes']) for x in fs)
binding['critical23_cases']=[
'analysis/native541-critical23-input-002',
'analysis/native541-critical23-adjudication-001',
'analysis/native541-critical23-branch-adjudication-001',
'native/critical23-material-replay-541'
]
binding['critical23_extension_audit_case']=CASE
summary_data=enc(summary); summary_sha=sha_bytes(summary_data)
binding['summary_sha256']=summary_sha
binding['critical23_material_replay']['finalization_case']=CASE
binding_data=enc(binding); binding_sha=sha_bytes(binding_data)

# Verify exact inherited record bytes before replacing summary metadata.
assert summary['records'][:1879]==json.loads((D/'summary-reference036.json').read_bytes())['records']
assert summary['records'][1879:1931]==json.loads(old_sb)['records'][1879:1931]
atomic(cp,selection_data,0o644,replace=True)
atomic(bp,binding_data,0o644,replace=True)
atomic(sp,summary_data,0o644,replace=True)

# Candidate allowlist is the source delta plus only newly archived records and summary metadata.
payload_paths=set()
for rec in summary['records'][1879:]:
    for f in rec.get('files',[]):
        rp=f.get('repository_copy') or f.get('repository_compressed_copy')
        if rp: payload_paths.add(EVIDENCE_PREFIX+rp)
sources=set(binding['source210']['changed_sources'])
metadata={EVIDENCE_PREFIX+x for x in ('summary.json','summary-reference036.json','publication-binding-reference037.json','publication037-case-selection.json')}
intended=payload_paths|sources|metadata
paths=sorted(intended)
nul=b'\0'.join(x.encode() for x in paths)+b'\0'
def gitset(*args): return set(git(*args).decode().splitlines())
tracked=gitset('diff','--name-only')
untracked=gitset('ls-files','--others','--exclude-standard')
ignored=gitset('ls-files','--others','--ignored','--exclude-standard')
actual=tracked|untracked|ignored
missing=sorted(intended-actual); unexpected=sorted(actual-intended)
assert not missing, 'intended paths absent '+repr(missing[:20])
# Unexpected files are retained and reported; never absorb unrelated/active outputs.
atomic(O/'proposed-paths-complete.nul',nul,0o644,replace=True)
status={'schema':'numi.cardboard.publication037-final-worktree-status.v1','tracked_diff':sorted(tracked),'untracked_nonignored':sorted(untracked),'ignored_untracked':sorted(ignored),'intended_allowlist':paths,'missing_intended':missing,'unexpected_not_in_allowlist':unexpected,'staged':git('diff','--cached','--name-only').decode().splitlines(),'committed':False}
atomic(O/'worktree-status.json',enc(status),0o644,replace=True)
receipt={
'schema':'numi.cardboard.publication037-finalization-receipt.v1','state':'ARCHIVE_COMPLETE_SOURCE_CANDIDATE_NOT_STAGED',
'parent':HEAD,'head_after':git('rev-parse','HEAD').decode().strip(),'staged_paths':status['staged'],
'prior_summary_sha256':OLD_SUMMARY,'summary_after_sha256':summary_sha,
'prior_binding_sha256':OLD_BINDING,'binding_after_sha256':binding_sha,
'prior_case_selection_sha256':OLD_SELECTION,'case_selection_after_sha256':selection_sha,
'historical_records':1879,'records_before_finalizer':1931,'finalizer_records':1,'total_records':1932,
'total_new_publication_records':len(summary['records'])-1879,
'critical23_report_sha256':BRANCH_REPORT,'critical23_manifest_sha256':BRANCH_MANIFEST,
'critical23_rows':138,'tangent_max_relative_error':branch['measurements']['max_tangent_rel_l2'],
'critical23_y_sign_indicators_with_smooth_law':12,
'native_fd_pairs':46,'native_fd_signal_above_endpoint_error_sum':30,'native_fd_signal_not_above_endpoint_error_sum':16,
'informative_native_fd_at_or_below_1e-3':10,'informative_native_fd_at_or_below_1e-6':0,
'fd_interpretation':'descriptive subset counts; no native-FD 1e-6 acceptance criterion',
'intended_path_count':len(intended),'tracked_diff_path_count':len(tracked),'untracked_nonignored_path_count':len(untracked),'ignored_untracked_path_count':len(ignored),
'missing_intended_paths':missing,'unexpected_not_in_allowlist':unexpected,'active545_excluded':ACTIVE,
'all_six_qualification_axes_open':True,'physical_box_qualification':False,
'historical_prefix_exact':summary['records'][:1879]==json.loads((D/'summary-reference036.json').read_bytes())['records'],
'prior_publication037_records_unchanged':summary['records'][1879:1931]==json.loads(old_sb)['records'][1879:1931],
'finalizer_payloads_restore_exact':True,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
}
atomic(O/'finalization-receipt.json',enc(receipt),0o644,replace=True)
manifest={'schema':'numi.cardboard.publication037-finalization-package.v1','files':{
'archive_extension037_finalize.py':sha_file(P/'archive_extension037_finalize.py'),
'README.md':sha_file(P/'README.md'),
'failed-install-attempt001.json':sha_file(P/'failed-install-attempt001.json'),
'proposed-paths-complete.nul':sha_file(O/'proposed-paths-complete.nul'),
'worktree-status.json':sha_file(O/'worktree-status.json'),
'finalization-receipt.json':sha_file(O/'finalization-receipt.json'),
'summary037-before-finalizer.json':sha_file(O/'summary037-before-finalizer.json'),
'binding037-before-finalizer.json':sha_file(O/'binding037-before-finalizer.json'),
'selection037-before-finalizer.json':sha_file(O/'selection037-before-finalizer.json')}}
atomic(O/'finalization-manifest.json',enc(manifest),0o644,replace=True)
print(json.dumps(receipt,indent=2,allow_nan=False))
