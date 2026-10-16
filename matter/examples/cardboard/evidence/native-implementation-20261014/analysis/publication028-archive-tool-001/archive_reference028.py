from pathlib import Path
import gzip
import hashlib
import json
import os

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
D = R / 'matter/examples/cardboard/evidence/native-implementation-20261014'
EXCLUDE = {'inputs', 'build', 'base', 'candidate', '__pycache__', 'archives', 'restore-check'}
ALIASES = {
    'analysis/native468-source154-dt-half-cycle-command-001',
    'analysis/native468-source154-dt-half-cycle-command-002',
    'analysis/native468-source157-dt-half-cycle-command-003',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def install(path, raw, replace_old=None):
    """Resume only identical writes; replacing the summary requires exact old bytes."""
    if path.exists():
        existing = path.read_bytes()
        if existing == raw:
            return
        assert replace_old is not None and existing == replace_old, path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.publication028.tmp')
    if temporary.exists():
        assert temporary.read_bytes() == raw, temporary
    else:
        with temporary.open('xb') as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    os.replace(temporary, path)


def jsonbytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def allowed(case, rel):
    parts = rel.parts
    if case in ALIASES and len(parts) == 2 and parts[0] == 'inputs':
        return True
    return not any(part in EXCLUDE or part.startswith(('CMakeFiles', '.air')) for part in parts)


prior_summary = D / 'summary-reference027.json'
oldbytes = (prior_summary if prior_summary.exists() else D / 'summary.json').read_bytes()
old = json.loads(oldbytes)
binding = json.loads((D / 'publication-binding-reference027.json').read_text())
pathbytes = (B / 'publication-reference027-pathspec.nul').read_bytes()
assert binding['summary_sha256'] == hashlib.sha256(oldbytes).hexdigest()
assert len(old['records']) == 1460
oldpaths = {x.decode() for x in pathbytes.split(b'\0') if x}
assert len(oldpaths) == 4849
cases = json.loads((B / 'publication028-cases-prepared.json').read_text())['cases']
assert len(cases) == len(set(cases))
assert not {x['case'] for x in old['records']}.intersection(cases)
assert json.loads((B / 'analysis/source157-root-integration-001/report.json').read_text())['selected_source_hashes_verified'] == 787
assert json.loads((B / 'analysis/retained-source-snapshot157-001/manifest.json').read_text())['snapshots'][0]['complete_disk_restoration_verified']
assert json.loads((B / 'analysis/native468-executable-preflight-refusal-001/report.json').read_text())['native_launched'] is False
newrows = []
newpaths = set()
rawcount = gzcount = 0
staging = B / 'publication028-payload-staging'
staging.mkdir(exist_ok=True)
payloads = []
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
        row = {'name': str(rel), 'bytes': len(raw), 'sha256': digest, 'external_path': str(src)}
        if len(raw) <= 262144 or src.suffix == '.gz':
            payload = raw
            row['repository_copy'] = str(target.relative_to(D))
            rawcount += 1
        else:
            packed = gzip.compress(raw, compresslevel=6, mtime=0)
            target = target.with_name(target.name + '.gz')
            assert hashlib.sha256(gzip.decompress(packed)).hexdigest() == digest
            payload = packed
            row.update(repository_compressed_copy=str(target.relative_to(D)),
                       compressed_bytes=len(packed), compressed_sha256=hashlib.sha256(packed).hexdigest(),
                       restore='gzip decompression recovers original bytes; verify original sha256')
            gzcount += 1
        staged = staging / target.relative_to(D)
        install(staged, payload)
        if target.exists():
            assert sha(target) == sha(staged), target
        payloads.append((src, digest, staged, target))
        rows.append(row)
        newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/' + str(target.relative_to(D)))
    assert rows, case
    record = {'case': case, 'external_root': str(root), 'physical_box_qualification': False, 'files': rows}
    for key in ['receipt', 'prediction']:
        p = root / (key + '.json')
        if p.is_file():
            record[key] = json.loads(p.read_text())
    newrows.append(record)

# No repository mutation occurs until the complete case inventory and every
# prepared payload have passed. A partial installation is resumed by exact
# hashes; it cannot silently overwrite a changed historical payload.
assert len({str(x[3]) for x in payloads}) == len(payloads)
for src, digest, staged, target in payloads:
    assert sha(src) == digest, ('source changed during preparation', src)
    assert staged.is_file()

verdicts = {
    'solver_correctness': {
        'status': 'partial_component_evidence; complete_nonlinear_solve_unqualified',
        'evidence': 'Source157 fixes commanded-grip observations only. Four CPU schedules and ten compiled-reader checks pass; targets and solver are unchanged. A source-bound build is admitted. Complete nonlinear solve, native regional-map/transpose qualification and projected-history/noise-floor questions remain open.'},
    'numerical_convergence': {
        'status': 'unqualified',
        'evidence': 'Native468 command004 is prepared and executable/build checked; command003 was refused before native launch. The active native468 case and launch guard are deliberately excluded. No complete coupon mesh/time/slower-loading study or downstream refinement qualification is published here.'},
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
summary.setdefault('historical_summary_snapshots', {})['reference027'] = {
    'path': 'summary-reference027.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()}
summary['records'].extend(newrows)
assert summary['records'][:1460] == old['records']
summary['verdicts'] = verdicts
summary['milestone'] = ('Source157 corrects grip-release telemetry with unchanged target generation and solver. '
                        'CPU schedules, compiled production-reader checks and source-bound build admission pass. '
                        'The source157 half-timestep command is checked after preserving a refused executable/build mismatch; '
                        'the active native run is excluded. CPU preconditioner counterexamples, restore-tested CMakeFiles '
                        'compaction metadata and the separate metallib historical-evidence gap are retained. '
                        'All complete physical qualification axes remain open.')
summary['qualification']['solver_correctness'] = verdicts['solver_correctness']['evidence']
summary['qualification']['numerical_convergence'] = verdicts['numerical_convergence']['evidence']
for name in ['summary-reference027.json', 'publication-binding-reference028.json']:
    newpaths.add('matter/examples/cardboard/evidence/native-implementation-20261014/' + name)
assert not oldpaths.intersection(newpaths)
newbinding = {
    'schema': 'numi.cardboard.reference028-publication-binding.v1',
    'prior_binding_sha256': sha(D / 'publication-binding-reference027.json'),
    'prior_pathspec_sha256': hashlib.sha256(pathbytes).hexdigest(),
    'prior_pathspec_entries': len(oldpaths),
    'historical_prefix': {'path': 'summary-reference027.json', 'records': 1460, 'sha256': hashlib.sha256(oldbytes).hexdigest()},
    'cases': cases, 'summary_sha256': sha(D / 'summary.json'), 'previous_prefix_records': 1460,
    'new_case_records': len(newrows), 'copied_files': sum(len(x['files']) for x in newrows),
    'raw_files': rawcount, 'gzip_files': gzcount, 'all_archived_bytes_verified': True,
    'physical_validation': False, 'six_axis_verdicts': verdicts,
    'excluded_directory_parts': sorted(EXCLUDE), 'excluded_directory_prefixes': ['CMakeFiles', '.air'],
    'input_directory_exceptions': sorted(ALIASES),
    'input_exception_reason': 'Versioned command packages contain small uniquely renamed byte-identical input aliases required by the wrapper basename contract; their repair manifests bind originals and aliases.',
    'external_recovery_archives': 'Closed-input compaction019/020/023/024, build-host020 and 57 CMakeFiles archives remain external sole recovery archives; preserve them. Restore metadata is published. Metallib historical operation receipts were not recovered: its current-state census is not independent proof of earlier per-path replacement.',
}
p = B / 'publication-reference028-pathspec.nul'
newpathbytes = pathbytes + b''.join(x.encode() + b'\0' for x in sorted(newpaths))
newbinding['summary_sha256'] = hashlib.sha256(jsonbytes(summary)).hexdigest()
newbinding['archive_installation'] = 'Complete payload preflight; hash-checked resumable file installation; historical and current summary are installed last.'
for src, digest, staged, target in payloads:
    install(target, staged.read_bytes())
    assert sha(src) == digest
    assert sha(target) == sha(staged)
install(prior_summary, oldbytes)
install(D / 'publication-binding-reference028.json', jsonbytes(newbinding))
install(p, newpathbytes)
install(D / 'summary.json', jsonbytes(summary), replace_old=oldbytes)
print(json.dumps({'cases': len(newrows), 'files': newbinding['copied_files'], 'raw': rawcount, 'gzip': gzcount,
                  'summary_records': len(summary['records']), 'binding_sha256': sha(D / 'publication-binding-reference028.json'),
                  'pathspec_entries': len(oldpaths) + len(newpaths)}, indent=2))
