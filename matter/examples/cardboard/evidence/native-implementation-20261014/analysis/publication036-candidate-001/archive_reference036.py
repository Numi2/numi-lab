
from pathlib import Path
import datetime, gzip, hashlib, json, os, stat, subprocess, tempfile

Q = Path('/Users/n/cardboard-implementation-20261014-host-macmini')
C = Q / 'checkout-publication032'
REL = Path('matter/examples/cardboard')
EVIDREL = REL / 'evidence/native-implementation-20261014'
D = C / EVIDREL
SNAP201 = Q / 'snapshots/source-regional-rap-tail-preserve-201'
SNAP202 = Q / 'snapshots/source-exact-frame-capture-202-buildfix-002'
SNAP203 = Q / 'snapshots/source-repeated-damped-fallback-203'
O = Q / 'analysis/publication036-payload-preflight-001'
EXPECTED_HEAD = '9660fd612f94d69b85906145a309594b93ca165f'
OLD_SUMMARY_SHA = 'af24934ca100aa7bf9e92e0ad6690856ff990c018ace454a044256e4c91fc352'
OLD_BINDING_SHA = '007d3efa8e750ac1e851235848fa89eb5bc0fbe268e990b605f12f245a282da6'
OLD_CANDIDATE_BINDING_SHA = '85dbcc8c4c36eb33b5d1898a2214f75152f8574625753963ed1f58882a02031e'
OLD_ARCHIVE_SCRIPT_SHA = '35915e63277568eac076e11a97e5052ed9d13d51dad287bfe8fc591353b44878'
SOURCE203_SHA = 'ec7af9ddb1fc7e424afdaaf270125ec4870bc34c1a807b732fbdf4aa2bdf64bd'
SELECTION_PATH = D / 'publication036-case-selection.json'
EXPECTED_CHANGED = [
    'matter/src/metal/fem.metalinc',
    'matter/src/runtime.mm',
    'matter/tools/cardboard_bilinear_capture_layers.py',
    'matter/tools/cardboard_cached_operator_audit.py',
    'matter/tools/test_cardboard_cached_operator_audit.py',
]
CASES = [
    'analysis/source202-203-independent-review-001',
    'analysis/source202-exact-frame-capture-001',
    'analysis/source202-exact-frame-capture-buildfix-002',
    'analysis/source203-repeated-damped-fallback-001',
    'analysis/source203-publication-binding-001',
    'analysis/publication036-candidate-001',
    'analysis/native531-armijo-cgls-fallback-helper-002',
    'analysis/native535-first-failure-adjudication-001',
    'analysis/native535-source201-full-coupon-command-001',
    'solver/fresh-source201-damped-cgls-complete-coupon-535',
    'analysis/native537-actual-armijo-fallback-command-001',
    'native/armijo-cgls-fallback-537',
    'analysis/native538-frame-and-prefix-verification-001',
    'analysis/native538-source203-frame-observer-pair-command-001',
    'solver/native538-source203-frame-off',
    'solver/native538-source203-frame-on',
]
ACTIVE_EXCLUDED = [
    'analysis/native539-source203-cauchy-all-fallback-command-001',
    'solver/fresh-source203-cauchy-all-fallback-coupon-539',
]
OTHER_EXCLUDED = [
    'analysis/source204-*',
    'analysis/source205-*',
    'other active native runs and mutable work products',
]
MAX_COMPRESSED_BYTES = 200_000_000

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha_file(p):
    return sha_bytes(Path(p).read_bytes())

def enc(x):
    return (json.dumps(x, indent=2, allow_nan=False, sort_keys=False) + '\n').encode()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=C)

def atomic_replace(path, data, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.publication036-replace-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def atomic_put(path, data, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.is_file(), path
        if path.read_bytes() != data:
            archive_tool = D/'analysis/publication036-candidate-001/archive_reference036.py'
            assert path == archive_tool and sha_file(path) == OLD_ARCHIVE_SCRIPT_SHA, f'nonidentical existing target: {path}'
            atomic_replace(path, data, mode)
            return
        if mode is not None:
            os.chmod(path, mode)
        return
    fd, tmp = tempfile.mkstemp(prefix='.publication036-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

assert C.is_dir() and D.is_dir() and SNAP203.is_dir()
O.mkdir(parents=True, exist_ok=True)
assert git('rev-parse', 'HEAD').decode().strip() == EXPECTED_HEAD
assert sha_file(D/'summary.json') == OLD_SUMMARY_SHA
assert sha_file(D/'publication-binding-reference035.json') == OLD_BINDING_SHA
if (D/'summary-reference035.json').exists():
    assert sha_file(D/'summary-reference035.json') == OLD_SUMMARY_SHA
else:
    pass
selection = json.loads(SELECTION_PATH.read_text())
assert selection['schema'] == 'numi.cardboard.publication036-case-selection.v1'
assert selection['cases'] == CASES
assert selection['compressed_payload_budget_bytes'] == MAX_COMPRESSED_BYTES
assert len(CASES) == len(set(CASES))
assert not set(ACTIVE_EXCLUDED).intersection(CASES)
assert not any(c.startswith('analysis/source204-') or c.startswith('analysis/source205-') for c in CASES)
for rel in ACTIVE_EXCLUDED:
    assert (Q/rel).exists(), f'active exclusion should name retained job: {rel}'

actual_status = set(git('status', '--porcelain=v1', '--untracked-files=all').decode().splitlines())
expected_tracked_status = {
    ' M matter/src/metal/fem.metalinc',
    ' M matter/src/runtime.mm',
    ' M matter/tools/cardboard_bilinear_capture_layers.py',
    ' M matter/tools/cardboard_cached_operator_audit.py',
    ' M matter/tools/test_cardboard_cached_operator_audit.py',
    ' M matter/examples/cardboard/implementation-20261014.md',
}
assert {x for x in actual_status if not x.startswith('?? ')} == expected_tracked_status
assert git('diff', '--cached', '--name-only').decode().strip() == ''
tracked_diff = sorted(x for x in git('diff', '--name-only', 'HEAD', '--', 'matter').decode().splitlines())
assert tracked_diff == sorted(EXPECTED_CHANGED + ['matter/examples/cardboard/implementation-20261014.md']), tracked_diff

frozen = json.loads((SNAP203/'frozen-source.json').read_text())
source_map = frozen['source_sha256']
assert sha_file(SNAP203/'frozen-source.json') == SOURCE203_SHA
assert len(source_map) == 817
for rel, digest in source_map.items():
    p = C/rel
    assert p.is_file() and not p.is_symlink(), p
    assert sha_file(p) == digest, f'source203 mismatch: {rel}'
for rel in EXPECTED_CHANGED:
    assert source_map[rel] == sha_file(C/rel), rel
    before = git('show', f'HEAD:{rel}')
    assert before and sha_bytes(before) != source_map[rel], f'expected source delta missing: {rel}'
assert sha_file(C/'matter/tools/cardboard_full_contact_armijo_check.mm') == '1db145ef8679a79dc2eafc74fd416f7a5ee67c7b343e46b7f8a150980b19a233'

old_bytes = (D/'summary.json').read_bytes()
old_summary = json.loads(old_bytes)
assert len(old_summary['records']) == 1863
prior_paths = {}
for record in old_summary['records']:
    for f in record.get('files', []):
        rp = f.get('repository_copy') or f.get('repository_compressed_copy')
        if rp:
            assert rp not in prior_paths, rp
            prior_paths[rp] = f
old_case_names = {r.get('case') for r in old_summary['records']}
assert not old_case_names.intersection(CASES), old_case_names.intersection(CASES)
source201map = json.loads((SNAP201/'frozen-source.json').read_text())['source_sha256']
source202map = json.loads((SNAP202/'frozen-source.json').read_text())['source_sha256']
assert len(source201map) == len(source202map) == len(source_map) == 817

planned = []
raw_total = 0
stored_estimate = 0
excluded_directories = {'.git', '__pycache__', 'CMakeFiles'}
for case in CASES:
    root = Q/case
    assert root.is_dir(), case
    case_files = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded_directories and not d.startswith('.air'))
        for name in sorted(names):
            p = Path(directory)/name
            assert p.is_file() and not p.is_symlink(), p
            b = p.read_bytes()
            mode = stat.S_IMODE(p.stat().st_mode)
            target = Path(case)/p.relative_to(root)
            h = sha_bytes(b)
            raw_total += len(b)
            case_files += 1
            if str(target) in prior_paths:
                f = prior_paths[str(target)]
                existing = (D/target).read_bytes()
                if f.get('repository_compressed_copy'):
                    existing = gzip.decompress(existing)
                assert f['bytes'] == len(b) and f['sha256'] == h and existing == b, target
                planned.append({'case':case,'source':str(p),'target':str(target),'kind':'historical_overlap','bytes':len(b),'sha256':h,'mode':mode})
                continue
            if len(b) > 262144 and p.suffix != '.gz':
                payload = gzip.compress(b, compresslevel=6, mtime=0)
                assert gzip.decompress(payload) == b
                stored_name = target.with_name(target.name + '.gz')
                kind = 'gzip'
            else:
                payload = b
                stored_name = target
                kind = 'raw'
            stored_estimate += len(payload)
            planned.append({'case':case,'source':str(p),'target':str(stored_name),'kind':kind,'bytes':len(b),'stored_bytes':len(payload),'sha256':h,'stored_sha256':sha_bytes(payload),'mode':mode})
    assert case_files > 0, f'empty case {case}'
assert raw_total < 400_000_000, raw_total
assert stored_estimate < MAX_COMPRESSED_BYTES, stored_estimate
assert not any('/native539-' in x['source'] or '/source204-' in x['source'] or '/source205-' in x['source'] for x in planned)
expected_untracked = {
    'matter/examples/cardboard/solver-milestone-source203.md',
    'matter/examples/cardboard/evidence/native-implementation-20261014/publication036-case-selection.json',
    str(EVIDREL/'summary-reference035.json'),
    str(EVIDREL/'publication-binding-reference036.json'),
}
for item in planned:
    if item['kind'] != 'historical_overlap':
        expected_untracked.add(str(EVIDREL/Path(item['target'])))
actual_untracked = set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
ignore_probe = subprocess.run(
    ["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=C,
    input=("\0".join(sorted(expected_untracked)) + "\0").encode(),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
ignored_expected = set(ignore_probe.stdout.decode().split("\0")) - {""}
assert ignore_probe.returncode in (0, 1), ignore_probe.stderr.decode()
assert actual_untracked | ignored_expected == expected_untracked, {
    "unexpected_untracked":sorted(actual_untracked-expected_untracked),
    "unexpected_ignored":sorted(ignored_expected-expected_untracked),
    "missing_untracked_or_ignored":sorted(expected_untracked-(actual_untracked|ignored_expected))
}
assert all((C / p).is_file() for p in expected_untracked), "expected archive outputs missing"


# All source/build and archive inputs are preflighted before the first payload write.
preflight = {
 'schema':'numi.cardboard.publication036-payload-preflight.v1',
 'state':'PREFLIGHT_PASS',
 'parent':EXPECTED_HEAD,
 'old_summary_sha256':OLD_SUMMARY_SHA,
 'old_binding_sha256':OLD_BINDING_SHA,
 'source203_frozen_sha256':SOURCE203_SHA,
 'source203_source_count':len(source_map),
 'case_count':len(CASES),
 'cases':CASES,
 'active_exclusions':ACTIVE_EXCLUDED,
 'other_exclusions':OTHER_EXCLUDED,
 'raw_payload_bytes':raw_total,
 'estimated_stored_payload_bytes':stored_estimate,
 'max_compressed_payload_bytes':MAX_COMPRESSED_BYTES,
 'planned_files':planned,
 'directory_exclusions':sorted(excluded_directories),
 'qualification_axes_open':6,
 'native539_outcome':'excluded while active; no outcome recorded',
}
atomic_put(O/'case-selection.json', SELECTION_PATH.read_bytes())
atomic_replace(O/'preflight-report.json', enc(preflight))

records = []
verified_overlaps = []
stored_total = 0
payload_total = 0
raw_count = 0
gzip_count = 0
for case in CASES:
    root = Q/case
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded_directories and not d.startswith('.air'))
        for name in sorted(names):
            p = Path(directory)/name
            b = p.read_bytes()
            mode = stat.S_IMODE(p.stat().st_mode)
            h = sha_bytes(b)
            target = Path(case)/p.relative_to(root)
            if str(target) in prior_paths:
                verified_overlaps.append({'case':case,'name':str(p.relative_to(root)),'sha256':h})
                continue
            f = {'name':str(p.relative_to(root)),'bytes':len(b),'sha256':h,'external_path':str(p),'mode_octal':oct(mode)}
            if len(b) > 262144 and p.suffix != '.gz':
                payload = gzip.compress(b, compresslevel=6, mtime=0)
                assert gzip.decompress(payload) == b
                stored_path = target.with_name(target.name+'.gz')
                atomic_put(D/stored_path, payload, 0o644)
                f.update({
                    'repository_compressed_copy':str(stored_path),
                    'compressed_bytes':len(payload),
                    'compressed_sha256':sha_bytes(payload),
                    'restore':'gunzip; restore recorded mode; verify original bytes and SHA-256'
                })
                gzip_count += 1
            else:
                payload = b
                atomic_put(D/target, payload, mode)
                f.update({'repository_copy':str(target),'restore':'copy exact bytes and restore recorded mode'})
                raw_count += 1
            stored_total += len(payload)
            payload_total += len(b)
            files.append(f)
    if files:
        records.append({'case':case,'external_root':str(root),'physical_box_qualification':False,'files':files})
assert stored_total <= stored_estimate, (stored_total, stored_estimate)
assert len(records) == len(CASES), (len(records), len(CASES))

# Restore-check every newly installed payload from the repository tree.
for record in records:
    for f in record['files']:
        target = D/(f.get('repository_copy') or f.get('repository_compressed_copy'))
        payload = target.read_bytes()
        if f.get('repository_compressed_copy'):
            restored = gzip.decompress(payload)
            assert sha_bytes(payload) == f['compressed_sha256']
        else:
            restored = payload
        assert len(restored) == f['bytes'] and sha_bytes(restored) == f['sha256']
        assert stat.S_IMODE(target.stat().st_mode) == (0o644 if f.get('repository_compressed_copy') else int(f['mode_octal'],8))
        assert restored == Path(f['external_path']).read_bytes()

# Preserve the complete prior summary bytes as the historical prefix reference.
atomic_put(D/'summary-reference035.json', old_bytes)
summary = json.loads(old_bytes)
summary.setdefault('historical_summary_snapshots', {})['reference035'] = {
    'path':'summary-reference035.json',
    'sha256':OLD_SUMMARY_SHA,
    'records':1863
}
summary['records'].extend(records)
assert len(summary['records']) == 1863 + len(records)
assert summary['records'][:1863] == old_summary['records']

# Update only stale numerical wording, retaining all six open qualification axes.
qual = summary.get('qualification', {})
qual['numerical_convergence'] = (
 'Fresh source201 native535 full-protocol execution fails after17 accepted controls at zero-based control17/Newton34: '
 'the primary residual-Armijo search exhausts24 trials, with base residual1.6395273405578337e-6 and last trial '
 '1.6396469391111168e-6 at alpha2^-23. Its same-Newton linear solve reports healthy status after64 restart cycles, '
 'but the final replacement residual1.19334089e-7 remains above RHS1.63952757e-6. The common accepted prefix is byte-identical '
 'to source198-bound native521. The native530 checkpoint only establishes an accepted-prefix save/resume through control16. '
 'Native539 source203 full-coupon execution was active at the publication cutoff and is excluded without an outcome. '
 'No complete60-control coupon or mesh/time refinement series is qualified; the prior23 Armijo trial norms and a causal '
 'explanation for the non-descent were not captured.'
)
summary['qualification'] = qual
summary['milestone'] = (summary.get('milestone','') +
 ' Source203 adds native FP32 frame capture and permits the existing damped CGLS fallback to be scheduled at all Newton iterations under the unchanged eligibility and nonlinear acceptance gates. Native537 passes synthetic production helper checks at Metal O0/O3 with zero FEM transactions; native538 establishes one-control observer off/on parity and frame-row reconstruction only. Native535 source201 full-protocol execution fails at control17/Newton34. The source203 full coupon 539 was active at the publication cutoff and is excluded without an outcome. All six qualification axes remain open.')
summary_data = enc(summary)
assert json.loads(summary_data)['records'][:1863] == old_summary['records']
summary_sha = sha_bytes(summary_data)

source201_build = json.loads((SNAP201/'frozen-source.json').read_text())
source202_build = json.loads((SNAP202/'frozen-source.json').read_text())
source203_overlay = Q/'analysis/source203-repeated-damped-fallback-001/source-overlay.json'
source203_package = Q/'analysis/source203-repeated-damped-fallback-001/package-manifest.json'
source_copy = Q/'analysis/publication036-candidate-001/source-copy-receipt.json'
native535_report = Q/'analysis/native535-first-failure-adjudication-001/report.json'
native537_receipt = Q/'native/armijo-cgls-fallback-537/run-receipt.json'
native538_report = Q/'analysis/native538-frame-and-prefix-verification-001/report.json'
strict_checker = C/'matter/tools/cardboard_full_contact_armijo_check.mm'
binding = {
 'schema':'numi.cardboard.reference036-publication-binding.v1',
 'parent':EXPECTED_HEAD,
 'prior_binding_sha256':OLD_BINDING_SHA,
 'historical_prefix':{'path':'summary-reference035.json','records':1863,'sha256':OLD_SUMMARY_SHA},
 'new_case_records':len(records),
 'cases':CASES,
 'case_selection_sha256':sha_file(SELECTION_PATH),
 'copied_files':sum(len(r['files']) for r in records),
 'raw_files':raw_count,
 'gzip_files':gzip_count,
 'original_payload_bytes':payload_total,
 'stored_payload_bytes':stored_total,
 'summary_sha256':summary_sha,
 'source203':{
   'frozen_source_sha256':SOURCE203_SHA,
   'selected_source_files':len(source_map),
   'source201_frozen_sha256':sha_file(SNAP201/'frozen-source.json'),
   'source202_frozen_sha256':sha_file(SNAP202/'frozen-source.json'),
   'source203_source_overlay_sha256':sha_file(source203_overlay),
   'source203_package_manifest_sha256':sha_file(source203_package),
   'source_copy_receipt_sha256':sha_file(source_copy),
   'changed_sources':{rel:sha_file(C/rel) for rel in EXPECTED_CHANGED},
   'composed_checkout_matches_all_frozen_files':True,
   'preserved_source199_strict_checker_sha256':sha_file(strict_checker),
   'source199_strict_checker_retained':True,
   'source203_full_coupon_539':'active at cutoff and excluded; no outcome recorded'
 },
 'bounded_results':{
   'source202_203_review_sha256':sha_file(Q/'analysis/source202-203-independent-review-001/report.json'),
   'source535_failure_report_sha256':sha_file(native535_report),
   'source535_exit_code':json.loads((Q/'solver/fresh-source201-damped-cgls-complete-coupon-535/receipt.json').read_text())['exit_code'],
   'source535_first_failure':'control17/Newton34, nonlinear Armijo exhaustion after24 trials; full coupon failed',
   'source537_receipt_sha256':sha_file(native537_receipt),
   'source537_physical_transactions':json.loads(native537_receipt.read_text())['full_fem_transactions'],
   'source538_report_sha256':sha_file(native538_report),
   'source538_verdict':json.loads(native538_report.read_text())['verdict'],
   'source538_limits':'one-control observer parity and frame reconstruction; not full trajectory or native derivative floor'
 },
 'qualification':{
   'all_six_axes_open':True,
   'full_coupon':False,
   'full_protocol_qualification':False,
   'physical_assembly':False,
   'crush_impact_prediction':False
 },
 'excluded_active':ACTIVE_EXCLUDED,
 'excluded_followups':OTHER_EXCLUDED,
 'verified_historical_overlaps':verified_overlaps,
 'restore_policy':'Compressed payloads are deterministic gzip files; restore each listed path by gunzip, restore the recorded mode, and verify byte count plus SHA-256.'
}
binding_data = enc(binding)
binding_sha = sha_bytes(binding_data)
binding_path = D/'publication-binding-reference036.json'
if binding_path.exists():
    assert sha_file(binding_path) == OLD_CANDIDATE_BINDING_SHA
    atomic_replace(binding_path, binding_data, 0o644)
else:
    atomic_put(binding_path, binding_data, 0o644)
atomic_replace(D/'summary.json', summary_data)

# Seal command-side report and exact archive inventories outside the repository.
report = {
 'schema':'numi.cardboard.publication036-archive-receipt.v1',
 'state':'ARCHIVE_COMPLETE_SOURCE_CANDIDATE_NOT_STAGED',
 'parent':EXPECTED_HEAD,
 'head_after':git('rev-parse','HEAD').decode().strip(),
 'staged_paths':git('diff','--cached','--name-only').decode().splitlines(),
 'summary_before_sha256':OLD_SUMMARY_SHA,
 'summary_after_sha256':summary_sha,
 'binding_sha256':binding_sha,
 'case_selection_sha256':sha_file(SELECTION_PATH),
 'case_count':len(records),
 'file_count':sum(len(r['files']) for r in records),
 'raw_file_count':raw_count,
 'gzip_file_count':gzip_count,
 'original_bytes':payload_total,
 'stored_bytes':stored_total,
 'source_count':len(source_map),
 'source203_frozen_sha256':SOURCE203_SHA,
 'active_exclusions':ACTIVE_EXCLUDED,
 'qualification_axes_open':6,
 'historical_record_prefix_exact':json.loads(summary_data)['records'][:1863] == old_summary['records'],
 'all_new_payloads_restore_hash_exact':True,
 'all_selected_sources_match':True,
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
}
atomic_put(O/'archive-receipt.json', enc(report))
atomic_put(O/'manifest.json', enc({
 'schema':'numi.cardboard.publication036-payload-package.v1',
 'files':{
  'archive_reference036.py':sha_file(Q/'analysis/publication036-candidate-001/archive_reference036.py'),
  'case-selection.json':sha_file(O/'case-selection.json'),
  'preflight-report.json':sha_file(O/'preflight-report.json'),
  'archive-receipt.json':sha_file(O/'archive-receipt.json')
 }
}))
print(json.dumps(report,indent=2,allow_nan=False))
