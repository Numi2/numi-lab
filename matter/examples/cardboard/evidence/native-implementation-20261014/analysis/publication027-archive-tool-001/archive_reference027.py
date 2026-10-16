from pathlib import Path
import gzip
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
EXCLUDE = {'inputs', 'build', 'base', 'candidate', '__pycache__', 'archives', 'restore-check'}
ALIASES = {
    'analysis/native466-source154-default64-parity-command-002',
    'analysis/native467-source154-allcontrol-outer128-command-002',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def allowed(case, rel):
    parts = rel.parts
    if case in ALIASES and len(parts) == 2 and parts[0] == 'inputs':
        return True
    return not any(part in EXCLUDE or part.startswith(('CMakeFiles', '.air')) for part in parts)


oldbytes = (D / 'summary.json').read_bytes()
old = json.loads(oldbytes)
binding = json.loads((D / 'publication-binding-reference026.json').read_text())
pathbytes = (B / 'publication-reference026-pathspec.nul').read_bytes()
assert binding['summary_sha256'] == hashlib.sha256(oldbytes).hexdigest()
assert len(old['records']) == 1424
oldpaths = {x.decode() for x in pathbytes.split(b'\0') if x}
assert len(oldpaths) == 4344
cases = json.loads((B / 'publication027-cases-prepared.json').read_text())['cases']
assert len(cases) == len(set(cases))
assert not {x['case'] for x in old['records']}.intersection(cases)
assert not (D / 'summary-reference026.json').exists()
assert not (D / 'publication-binding-reference027.json').exists()
assert json.loads((B / 'analysis/source154-default64-native466-parity-001/report.json').read_text())['parity_pass']
assert json.loads((B / 'analysis/source154-root-integration-001/report.json').read_text())['source_files_verified'] == 787
assert json.loads((B / 'analysis/native467-launch-guard-001/receipt.json').read_text())['space_abort'] is True
assert json.loads((B / 'solver/cgls-complete-outer128-allcontrols-467/receipt.json').read_text())['exit_code'] == -15
(D / 'summary-reference026.json').write_bytes(oldbytes)
newrows = []
newpaths = set()
rawcount = gzcount = 0
for case in cases:
    root = B / case
    assert root.is_dir(), root
    rows = []
    for src in sorted(root.rglob('*')):
        rel = src.relative_to(root)
        if not src.is_file() or not allowed(case, rel):
            continue
        assert not src.is_symlink(), src
        raw = src.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        target = D / case / rel
        assert not target.exists(), target
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {'name': str(rel), 'bytes': len(raw), 'sha256': digest, 'external_path': str(src)}
        if len(raw) <= 262144 or src.suffix == '.gz':
            target.write_bytes(raw)
            assert sha(target) == digest
            row['repository_copy'] = str(target.relative_to(D))
            rawcount += 1
        else:
            packed = gzip.compress(raw, compresslevel=6, mtime=0)
            target = target.with_name(target.name + '.gz')
            assert not target.exists(), target
            target.write_bytes(packed)
            assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == digest
            row.update(repository_compressed_copy=str(target.relative_to(D)),
                       compressed_bytes=len(packed), compressed_sha256=sha(target),
                       restore='gzip decompression recovers original bytes; verify original sha256')
            gzcount += 1
        rows.append(row)
        newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/' + str(target.relative_to(D)))
    assert rows, case
    record = {'case': case, 'external_root': str(root), 'physical_box_qualification': False, 'files': rows}
    for key in ['receipt', 'prediction']:
        p = root / (key + '.json')
        if p.is_file():
            record[key] = json.loads(p.read_text())
    newrows.append(record)

verdicts = {
    'solver_correctness': {
        'status': 'partial_component_evidence; complete_nonlinear_solve_unqualified',
        'evidence': 'Native466 preserves the full native463 attempted history across source154 controls. Isolated observer464 preserves native462 history with a retained pre-layout selection metadata defect. Source156 CPU-only repair is not integrated or natively qualified. Captured patch absolute entry gate fails despite small relative error; current projected history and native FP32 derivative floor remain unresolved.'},
    'numerical_convergence': {
        'status': 'unqualified',
        'evidence': 'Native467 requests the full 128-Newton all-control cycle but the disk guard interrupts control1 after control0 completed. SIGTERM exit-15, no timeout, unchanged inputs. No complete coupon mesh/time/slower-loading study; bounded CPU base-regional preconditioning worsens truncated residual.'},
    'component_calibration': {
        'status': 'partial_source_data_only',
        'evidence': 'Published-data package remains grade-bound. Exact kraft/NSSC progressive failure and local starch strength/GIc/GIIc remain unidentified.'},
    'analogue_validation': {
        'status': 'unqualified_for_requested_full_sequence',
        'evidence': 'Grade, fixture, conditioning and rate transfer limits remain; no new validation series passed.'},
    'physical_assembly_in_simulation': {
        'status': 'unqualified',
        'evidence': 'No completed 12-score folding, lap/end bonding and released settled 300x200x200mm shipping box. Explicit-flute contact/memory admission remains open.'},
    'crush_impact_prediction': {
        'status': 'unexecuted',
        'evidence': 'No qualified compression first peak, capacity-derived masses, gravity crush or 100mm drop; exact bare-box impact validation unavailable.'},
}
summary = json.loads(oldbytes)
summary.setdefault('historical_summary_snapshots', {})['reference026'] = {
    'path': 'summary-reference026.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()}
summary['records'].extend(newrows)
assert summary['records'][:1424] == old['records']
summary['verdicts'] = verdicts
summary['milestone'] = ('Source154 adds bounded opt-in all-control CGLS selectors and admits128 outer iterations. '
                        'Native466 full-attempt compatibility passes. Isolated native464 regional observer preserves '
                        'production history but exposes stale selection metadata; CPU-only source156 repair retained. '
                        'Captured patch absolute entry gate remains failed; CPU regional conditioning counterexample '
                        'retained. Native467 is capacity-interrupted, not a completed solve or a nonlinear failure. '
                        'No coupon convergence, calibrated full corrugated failure, released shipping assembly or crush/drop qualification.')
summary['qualification']['solver_correctness'] = verdicts['solver_correctness']['evidence']
summary['qualification']['numerical_convergence'] = verdicts['numerical_convergence']['evidence']
write(D / 'summary.json', summary)
for name in ['summary-reference026.json', 'publication-binding-reference027.json']:
    newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/' + name)
assert not oldpaths.intersection(newpaths)
newbinding = {
    'schema': 'numi.cardboard.reference027-publication-binding.v1',
    'prior_binding_sha256': sha(D / 'publication-binding-reference026.json'),
    'prior_pathspec_sha256': hashlib.sha256(pathbytes).hexdigest(),
    'prior_pathspec_entries': len(oldpaths),
    'historical_prefix': {'path': 'summary-reference026.json', 'records': 1424, 'sha256': hashlib.sha256(oldbytes).hexdigest()},
    'cases': cases, 'summary_sha256': sha(D / 'summary.json'), 'previous_prefix_records': 1424,
    'new_case_records': len(newrows), 'copied_files': sum(len(x['files']) for x in newrows),
    'raw_files': rawcount, 'gzip_files': gzcount, 'all_archived_bytes_verified': True,
    'physical_validation': False, 'six_axis_verdicts': verdicts,
    'excluded_directory_parts': sorted(EXCLUDE), 'excluded_directory_prefixes': ['CMakeFiles', '.air'],
    'input_directory_exceptions': sorted(ALIASES),
    'input_exception_reason': 'Versioned command packages contain small uniquely renamed byte-identical input aliases required by the wrapper basename contract; their repair manifests bind originals and aliases.',
    'external_recovery_archives': 'Closed-input compaction019/020/023 archives remain external sole recovery archives; keep them. Restore metadata is published.',
}
write(D / 'publication-binding-reference027.json', newbinding)
p = B / 'publication-reference027-pathspec.nul'
assert not p.exists()
p.write_bytes(pathbytes + b''.join(x.encode() + b'\0' for x in sorted(newpaths)))
print(json.dumps({'cases': len(newrows), 'files': newbinding['copied_files'], 'raw': rawcount, 'gzip': gzcount,
                  'summary_records': len(summary['records']), 'binding_sha256': sha(D / 'publication-binding-reference027.json'),
                  'pathspec_entries': len(oldpaths) + len(newpaths)}, indent=2))
