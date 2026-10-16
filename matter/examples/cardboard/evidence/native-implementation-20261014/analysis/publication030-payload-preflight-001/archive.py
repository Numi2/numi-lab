from pathlib import Path
import gzip
import hashlib
import json
import os

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
PREFIX = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
D = R / PREFIX
O = B / 'analysis/publication030-payload-preflight-001'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
encode = lambda v: (json.dumps(v, indent=2, allow_nan=False) + '\n').encode()

def install(path, data):
    if path.exists():
        assert path.read_bytes() == data, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)

def main():
    assert not O.exists()
    oldbytes = (D / 'summary.json').read_bytes()
    assert hashlib.sha256(oldbytes).hexdigest() == 'aaf1660cb3c68c02de4b81cd4e480d54a20e3674d7176c65ab5aad74ac2187c7'
    old = json.loads(oldbytes)
    assert len(old['records']) == 1524
    pathbytes = (B / 'publication-reference029-pathspec.nul').read_bytes()
    assert hashlib.sha256(pathbytes).hexdigest() == '2a3f1391a727810fd77b90ccc0c7ff37d3e4cf327186d44d31a55fbe5ff7e06b'
    prior_paths = {s.decode() for s in pathbytes.split(b'\0') if s}
    assert len(prior_paths) == 8376
    cases = ['solver/accepted-checkpoint-prefix-470', 'solver/source169-default-off-parity-472']
    analyses = '''source169-warm-start-integration-001
source169-build-cpu-validation-001
warm-start-native-kernel-check-002
shipping-checkpoint-physical-arena-overlay-171-001
base-cgls-warm-finite-breakdown-overlay-172-001
base-cgls-warm-finite-breakdown-overlay-172-002
hierarchical-sparse-private-contact-scan-overlay-170-001
native-sparse-private-scan-check-170-001
source175-sparse-private-scan-integration-001
native469-material-replay-174-001
native469-mp-fallback-004
native469-mp-fallback-005
native469-mixed-reference-assembly-001
native469-mixed-reference-assembly-002
native469-mixed-reference-assembly-003
native469-mixed-reference-adjudication-001
native469-mixed-reference-assembly-adjudication-002
native470-accepted-checkpoint-prefix-command-001
native470-accepted-checkpoint-prefix-command-002
native470-accepted-checkpoint-prefix-command-002/executions
native470-accepted-prefix-adjudication-001
native470-checkpoint-write-failure-001
native471-checkpoint-resume-command-001
native472-source169-default-off-command-001
native472-source169-default-off-command-002
native472-source169-default-off-command-003
native472-source169-default-off-command-003/executions
native472-default-off-production-parity-001
native473-474-warm-checkpoint-smoke-001
source173-integration-preflight-refusal-001
source173-checkpoint-warm-integration-001
source173-build-cpu-validation-001
source173-warm-kernel-check-001
source173-repository-integration-001
source173-macmini-host-build-and-check-001
publication029-staged-source-check-001
publication029-payload-preflight-001
publication029-independent-byte-audit-001
publication029-commit-and-push-001
publication029-duplicate-staging-retirement-001
closed-published-output-compaction-030'''
    cases += ['analysis/' + name for name in analyses.splitlines()]
    existing = {r['case'] for r in old['records']}
    assert len(set(cases)) == len(cases) and not (set(cases) & existing)
    # Sealed narrow candidate source files are included, unlike duplicated base
    # snapshots/build trees. No running native case or recovery archive is read.
    excluded = {'__pycache__', 'archives', 'base', 'build', 'executions', 'restore-check'}
    O.mkdir()
    records, paths, bindings = [], set(), []
    raw_count = gzip_count = 0
    for case in cases:
        root = B / case
        assert root.is_dir(), root
        files = []
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(n for n in dirs if n not in excluded and not n.startswith(('CMakeFiles', '.air')))
            for name in sorted(names):
                source = Path(directory) / name
                assert source.is_file() and not source.is_symlink(), source
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                rel = source.relative_to(root)
                target_rel = Path(case) / rel
                row = {'name': str(rel), 'bytes': len(data), 'sha256': digest, 'external_path': str(source)}
                if len(data) <= 262144 or source.suffix == '.gz':
                    payload = data
                    row['repository_copy'] = str(target_rel)
                    raw_count += 1
                else:
                    payload = gzip.compress(data, compresslevel=6, mtime=0)
                    assert gzip.decompress(payload) == data
                    target_rel = target_rel.with_name(target_rel.name + '.gz')
                    row.update(repository_compressed_copy=str(target_rel), compressed_bytes=len(payload),
                        compressed_sha256=hashlib.sha256(payload).hexdigest(), restore='gzip decompression; verify original SHA-256')
                    gzip_count += 1
                target = D / target_rel
                assert PREFIX + str(target_rel) not in prior_paths
                install(target, payload)
                assert target.read_bytes() == payload
                paths.add(PREFIX + str(target_rel))
                bindings.append((source, digest))
                files.append(row)
        assert files, case
        record = {'case': case, 'external_root': str(root), 'physical_box_qualification': False, 'files': files}
        for name in ('receipt', 'prediction'):
            p = root / (name + '.json')
            if p.is_file():
                record[name] = json.loads(p.read_text())
        records.append(record)
    for p, digest in bindings:
        assert sha(p) == digest, p
    verdicts = dict(old['verdicts'])
    verdicts['solver_correctness'] = {'status': 'partial_component_evidence; complete_nonlinear_solve_unqualified',
        'evidence': 'Source173 adds default-off warm CGLS refinement and finite-breakdown decline, with264actualMetalhelperchecks and29focusedCPUcases. Native472 preserves the entire default-off failed468attempt. CorrectedMP005 restores per-grade parameter identity; selected center cached-column/element/nodal errors5.722942e-4/5.363624e-5/8.507027e-6meet relative gates. Four strictFP64center roots and branch/noise gates remain open. InvalidMP004 and derivedassemblyanalyses are retained. Actual fullwarm-enabled nonlinearcycle remains unqualified.'}
    verdicts['numerical_convergence'] = {'status': 'unqualified',
        'evidence': 'Native472 rejects control20 after20acceptedbentcontrols exactly as468. No complete loading/withdrawal/release history, timestep triplet or mesh series passes. Native470 fails checkpointserialization after20acceptedcontrols; source173 fixes physicalarena record counts. Newhost476/477results are excluded while active/unexecuted; checkpointresume alone cannot qualify an uninterrupted cycle.'}
    verdicts['physical_assembly_in_simulation'] = {'status': 'unqualified',
        'evidence': 'Sealed170parallel sparse scan and175integration overlays pass CPU/compilechecks. ShippingN528444/T1477728still exceeds runtime8192/32768scope; oneenvsparsepairarena36189188bytes fits36MiB but no caplift or16GiBprocesspeak qualification follows. No physicallyscored/folded/bonded/released shippingcheckpoint.'}
    summary = json.loads(oldbytes)
    summary.setdefault('historical_summary_snapshots', {})['reference029'] = {'path': 'summary-reference029.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()}
    summary['records'].extend(records)
    assert summary['records'][:1524] == old['records']
    summary['verdicts'] = verdicts
    summary['milestone'] = 'Source173 implements opt-in warm CGLS candidate refinement and exact checkpointarena validation. Isolated nativehelper264checks and complete default-off failedattempt parity pass. Per-grade MP reference repair removes a reference-tool error; strictFP64and branch/noise limits remain. Complete coupon/resolution/calibration/assembly/crush qualification remains open; newhost476/477notpublishedasresults.'
    for key in ('solver_correctness', 'numerical_convergence', 'physical_assembly_in_simulation'):
        summary['qualification'][key] = verdicts[key]['evidence']
    newbytes = encode(summary)
    paths.update(PREFIX + n for n in ('summary-reference029.json', 'publication-binding-reference030.json'))
    newpathbytes = pathbytes + b''.join(p.encode() + b'\0' for p in sorted(paths))
    binding = {'schema': 'numi.cardboard.reference030-publication-binding.v1',
        'prior_binding_sha256': sha(D / 'publication-binding-reference029.json'),
        'prior_pathspec_sha256': hashlib.sha256(pathbytes).hexdigest(),
        'historical_prefix': {'records': 1524, 'path': 'summary-reference029.json', 'sha256': hashlib.sha256(oldbytes).hexdigest()},
        'prior_pathspec_entries': len(prior_paths), 'new_case_records': len(records), 'cases': cases,
        'copied_files': len(bindings), 'raw_files': raw_count, 'gzip_files': gzip_count,
        'summary_sha256': hashlib.sha256(newbytes).hexdigest(), 'six_axis_verdicts': verdicts,
        'source173_frozen_sha256': sha(B / 'snapshots/source-checkpoint-warm-breakdown-173/frozen-source.json'),
        'source173_original_host_build_sha256': sha(R / 'build-cardboard-provenance-173/source-build.json'),
        'excluded_directory_parts': sorted(excluded),
        'exclusion_scope': 'Duplicated full base/build trees remain external. Narrow candidate code, patches, exact manifests and closed receipts are included. Active native476/newhost477, active compaction031 and future native170/174 host executions excluded. No claim of a Git copy for every external binary or scratch file.',
        'physical_validation': False, 'new_host476477_results_excluded': True}
    install(D / 'summary-reference029.json', oldbytes)
    install(D / 'publication-binding-reference030.json', encode(binding))
    install(B / 'publication-reference030-pathspec.nul', newpathbytes)
    assert (D / 'summary.json').read_bytes() == oldbytes
    (D / 'summary.json').write_bytes(newbytes)
    report = {'schema': 'numi.cardboard.publication030-payload-preflight.v1', 'cases': len(records),
        'payloads': len(bindings), 'raw': raw_count, 'gzip': gzip_count,
        'summary_records': len(summary['records']), 'old_records': 1524,
        'summary_sha256': sha(D / 'summary.json'), 'binding_sha256': sha(D / 'publication-binding-reference030.json'),
        'pathspec_sha256': sha(B / 'publication-reference030-pathspec.nul'),
        'pathspec_entries': len(prior_paths) + len(paths), 'active_results_excluded': True}
    (O / 'report.json').write_bytes(encode(report))
    (O / 'archive.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
