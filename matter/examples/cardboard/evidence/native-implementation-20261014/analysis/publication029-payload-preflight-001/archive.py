from pathlib import Path
import gzip
import hashlib
import json
import os

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
D = R / PREFIX
O = B / 'analysis/publication029-payload-preflight-001'
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
encode = lambda obj: (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode()
def install(path, raw):
    if path.exists():
        assert path.read_bytes() == raw, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)

oldbytes = (D / 'summary.json').read_bytes()
assert hashlib.sha256(oldbytes).hexdigest() == 'd025f2ce80dda69efb980ed24a21001cc8e2c6ac41a68299768bb25c26182196'
old = json.loads(oldbytes)
old_binding = json.loads((D / 'publication-binding-reference028.json').read_text())
assert old_binding['summary_sha256'] == hashlib.sha256(oldbytes).hexdigest()
old_pathbytes = (B / 'publication-reference028-pathspec.nul').read_bytes()
old_paths = {p.decode() for p in old_pathbytes.split(b'\0') if p}
assert len(old['records']) == 1480 and len(old_paths) == 5566
prepared = json.loads((B / 'publication029-cases-prepared.json').read_text())
cases = prepared['cases'] + [
    'analysis/bilinear-capture-layer-audit-source166-003',
    'analysis/reference003-portable-integration-001',
    'analysis/reference003-portable-test-attempt-001',
    'analysis/source168-reference-integration-001',
    'analysis/source168-portable-reference-validation-001',
    'analysis/base-cgls-warm-start-refinement-overlay-159-001',
    'analysis/base-cgls-warm-start-refinement-overlay-159-002',
    'analysis/warm-start-native-kernel-check-001',
]
existing_cases = {row['case'] for row in old['records']}
already_published = sorted(set(cases) & existing_cases)
cases = [case for case in cases if case not in existing_cases]
assert len(cases) == len(set(cases))
assert all('470' not in case for case in cases)
excluded = set(prepared['excluded_directory_parts'])
stage = B / 'publication029-payload-staging'
stage.mkdir(exist_ok=False)
records = []
paths = set()
payloads = []
raw_count = gzip_count = 0
for case in cases:
    root = B / case
    assert root.is_dir(), root
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in excluded and not name.startswith(('CMakeFiles', '.air')))
        for name in sorted(names):
            source = Path(directory) / name
            assert source.is_file() and not source.is_symlink(), source
            relative = source.relative_to(root)
            original = source.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            row = {'name': str(relative), 'bytes': len(original), 'sha256': digest, 'external_path': str(source)}
            target_relative = Path(case) / relative
            if len(original) <= 262144 or source.suffix == '.gz':
                payload = original
                row['repository_copy'] = str(target_relative)
                raw_count += 1
            else:
                payload = gzip.compress(original, compresslevel=6, mtime=0)
                assert gzip.decompress(payload) == original
                target_relative = target_relative.with_name(target_relative.name + '.gz')
                row.update(repository_compressed_copy=str(target_relative), compressed_bytes=len(payload),
                           compressed_sha256=hashlib.sha256(payload).hexdigest(), restore='gzip decompression; verify original SHA-256')
                gzip_count += 1
            staged = stage / target_relative
            install(staged, payload)
            target = D / target_relative
            if target.exists():
                assert target.read_bytes() == payload, target
            payloads.append((source, digest, staged, target))
            paths.add(PREFIX + str(target_relative))
            files.append(row)
    assert files, case
    row = {'case': case, 'external_root': str(root), 'physical_box_qualification': False, 'files': files}
    for name in ['receipt', 'prediction']:
        path = root / (name + '.json')
        if path.is_file():
            row[name] = json.loads(path.read_text())
    records.append(row)
assert len({str(row[3]) for row in payloads}) == len(payloads)
assert not old_paths.intersection(paths)
for source, digest, staged, target in payloads:
    assert sha(source) == digest, source

verdicts = dict(old['verdicts'])
verdicts['solver_correctness'] = {
    'status': 'partial_component_evidence; complete_nonlinear_solve_unqualified',
    'evidence': 'Source167 adds accepted coupon checkpoints and actual regional-consumer capture. Source168 fixes the independent FP64 FB Jacobian and adds a portable90-digit witness. Four helper and two Python CTest registrations pass. Native468/469 share byte-identical failed production histories;469 rich capture is retained. Four strict-floor center projection misses remain after the reference repair. Experimental warm-start CGLS is isolated; its first actual kernel check fails due checker mode-state leakage. No complete native solver qualification.'}
verdicts['numerical_convergence'] = {
    'status': 'unqualified',
    'evidence': 'Native468/469 both reject control20 after20 accepted bent controls; finalresidual5.424712981e-6 vs5e-7tolerance. No complete loading/withdrawal/release coupon history, accepted timestep triplet or mesh convergence. Native470 accepted-checkpoint prefix is active and excluded.'}
verdicts['physical_assembly_in_simulation'] = {
    'status': 'unqualified',
    'evidence': 'Accepted-surface clearance/coverage and bounded host workspace checks implemented; full assembly driver syntax passes. Explicit-flute shipping-scale contact/admission and fullprocess peak remain open. No physically scored/folded/bonded/released300x200x200mm checkpoint.'}
summary = json.loads(oldbytes)
summary.setdefault('historical_summary_snapshots', {})['reference028'] = {'path': 'summary-reference028.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()}
summary['records'].extend(records)
assert summary['records'][:1480] == old['records']
summary['verdicts'] = verdicts
summary['milestone'] = ('Source167 implements strict accepted coupon checkpoints, source-bound diagnostic environment and actual regional consumer capture; source168 repairs independent FP64 FB differentiation with a portable high-precision witness. Complete failed468/469 histories and diagnostic parity are retained. CPU contracts/build checks pass; four strict-floor center rows, full nonlinear solve, convergence and shipping assembly remain open. Active470 excluded. Experimental warm-start kernel-check failure retained.')
summary['qualification']['solver_correctness'] = verdicts['solver_correctness']['evidence']
summary['qualification']['numerical_convergence'] = verdicts['numerical_convergence']['evidence']
new_summary = encode(summary)
paths.update(PREFIX + name for name in ['summary-reference028.json', 'publication-binding-reference029.json'])
new_pathbytes = old_pathbytes + b''.join(name.encode() + b'\0' for name in sorted(paths))
binding = {
    'schema': 'numi.cardboard.reference029-publication-binding.v1',
    'prior_binding_sha256': sha(D / 'publication-binding-reference028.json'),
    'prior_pathspec_sha256': hashlib.sha256(old_pathbytes).hexdigest(),
    'historical_prefix': {'records': 1480, 'path': 'summary-reference028.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()},
    'prior_pathspec_entries': len(old_paths), 'new_case_records': len(records), 'cases': cases,
    'already_published_cases_excluded': already_published,
    'copied_files': len(payloads), 'raw_files': raw_count, 'gzip_files': gzip_count,
    'summary_sha256': hashlib.sha256(new_summary).hexdigest(),
    'excluded_directory_parts': sorted(excluded), 'excluded_directory_prefixes': ['CMakeFiles', '.air'],
    'exclusion_scope': 'Build/candidate/base/scratch copies remain external; changed source patches, exact manifests, compiler/checker receipts and all native histories are retained. Sole recovery archives preserved. No claim of a Git copy for every external scratch file.',
    'six_axis_verdicts': verdicts, 'physical_validation': False,
    'active_native470_excluded': True,
    'source168_frozen_sha256': sha(B / 'snapshots/source-reference-jacobian-168/frozen-source.json'),
    'source167_native_build_sha256': sha(R / 'build-cardboard-provenance-167-final/source-build.json'),
    'source168_cpu_build_sha256': sha(R / 'build-cardboard-provenance-168/source-build.json'),
    'source159_history_errata': 'Final003 overlay/inventory verified; initially announced003 seal bytes were overwritten before preservation and could not be recovered. Base-frozen metadata typo is recorded in source167 validation; admission independently verifies actual157files. No synthesized historical bytes.'
}
for source, digest, staged, target in payloads:
    assert sha(source) == digest
    install(target, staged.read_bytes())
    assert sha(target) == sha(staged)
install(D / 'summary-reference028.json', oldbytes)
install(D / 'publication-binding-reference029.json', encode(binding))
install(B / 'publication-reference029-pathspec.nul', new_pathbytes)
assert (D / 'summary.json').read_bytes() == oldbytes
(D / 'summary.json').write_bytes(new_summary)
report = {'schema': 'numi.cardboard.publication029-payload-preflight.v1',
          'cases': len(records), 'payloads': len(payloads), 'raw': raw_count, 'gzip': gzip_count,
          'summary_records': len(summary['records']), 'old_records': 1480,
          'summary_sha256': sha(D / 'summary.json'), 'binding_sha256': sha(D / 'publication-binding-reference029.json'),
          'pathspec_sha256': sha(B / 'publication-reference029-pathspec.nul'),
          'pathspec_entries': len(old_paths) + len(paths), 'active470_excluded': True}
(O / 'report.json').write_bytes(encode(report))
install(O / 'archive.py', Path(__file__).read_bytes())
print(json.dumps(report, indent=2))
