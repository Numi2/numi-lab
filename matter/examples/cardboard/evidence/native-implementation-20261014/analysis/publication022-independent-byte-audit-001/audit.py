#!/usr/bin/env python3
"""Independent, read-only byte audit for publication binding reference 022."""
from __future__ import annotations
import gzip, hashlib, json, os, pathlib, subprocess, sys, time

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
REPO = pathlib.Path('/Users/home/numi-cardboard-20261006')
PUB = REPO / 'matter/examples/cardboard/evidence/native-implementation-20261014'
PATHSPEC = BASE / 'publication-reference022-pathspec.nul'
BINDING = PUB / 'publication-binding-reference022.json'
SUMMARY = PUB / 'summary.json'
OLD = PUB / 'summary-reference021.json'
OUT = BASE / 'analysis/publication022-independent-byte-audit-001'

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest_stream(path: pathlib.Path, *, compressed: bool) -> tuple[int, str]:
    h = hashlib.sha256(); n = 0
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block); n += len(block)
    return n, h.hexdigest()

def allocated_tree(path: pathlib.Path) -> dict:
    count = logical = allocated = 0
    for dp, _, names in os.walk(path):
        for name in names:
            p = pathlib.Path(dp) / name
            try: st = p.lstat()
            except OSError: continue
            count += 1; logical += st.st_size
            allocated += getattr(st, 'st_blocks', 0) * 512
    return {'file_count': count, 'logical_bytes': logical, 'st_blocks_bytes': allocated}

def command(argv: list[str]) -> str:
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return p.stdout.strip()

binding = json.loads(BINDING.read_text())
summary = json.loads(SUMMARY.read_text())
old = json.loads(OLD.read_text())
path_entries = PATHSPEC.read_bytes().split(b'\0')
paths = [x.decode('utf-8') for x in path_entries if x]
assert len(paths) == 1187 and len(set(paths)) == len(paths)
assert binding['schema'] == 'numi.cardboard.reference022-publication-binding.v1'
assert binding['summary_sha256'] == sha(SUMMARY)
assert binding['copied_files'] == 1184 and binding['all_archived_bytes_verified'] is True
assert binding['physical_validation'] is False
assert sha(OLD) == summary['historical_summary_snapshots']['reference021']['sha256']
assert len(old['records']) == 1225 and len(summary['records']) == 1282
assert summary['records'][:len(old['records'])] == old['records']
new_records = summary['records'][len(old['records']):]
assert len(new_records) == len(binding['cases']) == 57
assert [r['case'] for r in new_records] == binding['cases']
rows = [f for r in new_records for f in r['files']]
assert len(rows) == 1184
assert len({f.get('repository_copy') or f.get('repository_compressed_copy') for f in rows}) == 1184
repo_prefix = 'matter/examples/cardboard/evidence/native-implementation-20261014/'
expected_paths = {repo_prefix + (f.get('repository_copy') or f.get('repository_compressed_copy')) for f in rows} | {
    repo_prefix + 'summary.json', repo_prefix + 'summary-reference021.json',
    repo_prefix + 'publication-binding-reference022.json'
}
assert set(paths) == expected_paths

counts = {'raw_exact': 0, 'gzip_exact': 0}
bytes = {'raw_source_bytes': 0, 'gzip_source_bytes': 0, 'raw_archive_bytes': 0, 'gzip_archive_bytes': 0}
failures = []
for row in rows:
    rel = row.get('repository_copy') or row.get('repository_compressed_copy')
    if not rel:
        failures.append({'external_path': row.get('external_path'), 'reason': 'missing_repository_copy_field'}); continue
    dest = PUB / rel
    src = pathlib.Path(row['external_path'])
    expected_n = int(row['bytes']); expected_h = row['sha256']
    if not dest.is_file() or dest.is_symlink() or not src.is_file():
        failures.append({'repository_copy': rel, 'reason': 'missing_or_nonregular'}); continue
    src_stat = src.stat(); dest_stat = dest.stat()
    if src_stat.st_size != expected_n or sha(src) != expected_h:
        failures.append({'repository_copy': rel, 'reason': 'external_source_binding_mismatch'}); continue
    if 'repository_copy' in row and dest_stat.st_size == expected_n and sha(dest) == expected_h:
        counts['raw_exact'] += 1
        bytes['raw_source_bytes'] += expected_n; bytes['raw_archive_bytes'] += dest_stat.st_size
        continue
    try:
        got_n, got_h = digest_stream(dest, compressed=True)
    except (OSError, EOFError, gzip.BadGzipFile):
        got_n, got_h = -1, ''
    compressed_ok = ('repository_compressed_copy' in row and
                     dest_stat.st_size == int(row['compressed_bytes']) and
                     sha(dest) == row['compressed_sha256'])
    if got_n == expected_n and got_h == expected_h and compressed_ok:
        counts['gzip_exact'] += 1
        bytes['gzip_source_bytes'] += expected_n; bytes['gzip_archive_bytes'] += dest_stat.st_size
    else:
        failures.append({'repository_copy': rel, 'reason': 'archived_copy_mismatch',
                         'expected_bytes': expected_n, 'observed_bytes': dest_stat.st_size,
                         'decompressed_bytes': got_n, 'expected_sha256': expected_h,
                         'decompressed_sha256': got_h})
assert not failures, json.dumps(failures[:10])
assert sum(counts.values()) == 1184

stage = command(['git', '-C', str(REPO), 'diff', '--cached', '--name-only'])
staged = [line for line in stage.splitlines() if line]
status = command(['git', '-C', str(REPO), 'status', '--porcelain=v1', '--untracked-files=normal'])
guards = [
 'matter/examples/cardboard/evidence/native-implementation-20261014/solver/regional-bending-captured-cycle-057/analysis-057/replay-build-037/CMakeFiles/4.0.1/CompilerIdOBJCXX/',
 'matter/tools/evidence/cohesive-scratch-alias/'
]
assert not staged
assert all(g not in paths for g in guards)
assert all(any(line.startswith('?? ' + g) for line in status.splitlines()) for g in guards)

source_builds = {}
for n in (133, 134, 135, 136):
    p = PUB / 'analysis/publication022-preflight-001' / f'source-build-{n}.json'
    obj = json.loads(p.read_text())
    checks = []
    for artifact, expected in obj['artifacts_sha256'].items():
        ap = pathlib.Path(artifact)
        checks.append({'path': artifact, 'exists': ap.is_file(), 'sha256_matches': ap.is_file() and sha(ap) == expected})
    source_builds[str(n)] = {'metadata_path': str(p), 'metadata_sha256': sha(p),
                             'artifact_count': len(checks), 'artifacts_verified': all(x['exists'] and x['sha256_matches'] for x in checks),
                             'artifacts': checks}
assert all(v['artifacts_verified'] for v in source_builds.values())

lsof_result = subprocess.run(['lsof', '-n'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
old_paths = [
 '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-133',
 '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-134',
 '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-135',
 '/Users/home/numi-cardboard-20261006/build-cardboard-provenance-136',
]
open_old_build_handles = [line for line in lsof_result.stdout.splitlines()
                          if any(p in line for p in old_paths)]

snapshots = [
 (133, 'source-base-state-cauchy-armijo-133', 'retained-source-snapshot133-001'),
 (134, 'source-cache-projection-reuse-checker-fixed-134', 'retained-source-snapshot134-001'),
 (135, 'source-cauchy-positive-mass-admission-135', 'retained-source-snapshot135-001'),
 (136, 'source-cache-projection-reuse-runtime-136', 'retained-source-snapshot136-001'),
]
snapshot_data = []
for n, name, bundle_name in snapshots:
    snap = BASE / 'snapshots' / name
    bundle = REPO / 'matter/examples/cardboard/evidence/native-implementation-20261014/analysis' / bundle_name
    bm = json.loads((bundle / 'manifest.json').read_text())
    bundle_status = command(['git','-C',str(REPO),'status','--porcelain=v1','--',str(bundle.relative_to(REPO))])
    snapshot_data.append({'number': n, 'snapshot_path': str(snap), 'snapshot_allocated': allocated_tree(snap),
        'bundle_path': str(bundle), 'bundle_allocated': allocated_tree(bundle),
        'bundle_manifest_sha256': sha(bundle/'manifest.json'),
        'bundle_claims_complete_disk_restoration': bm['snapshots'][0]['complete_disk_restoration_verified'],
        'bundle_claims_all_reconstructed_bytes_verified': bm['snapshots'][0]['all_reconstructed_bytes_verified'],
        'bundle_file_count': bm['snapshots'][0]['files'], 'git_base_revision': bm['git_base_revision'],
        'bundle_git_status': bundle_status})
base_commit_present = subprocess.run(['git','-C',str(REPO),'cat-file','-e','8c00ae13d78331c938311ca80d78857c4deefe75^{commit}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0

build_cache = {}
for n in (133,134,135,136):
    root = REPO / f'build-cardboard-provenance-{n}'
    by_type = {}
    for dp, _, filenames in os.walk(root):
        for filename in filenames:
            p = pathlib.Path(dp)/filename
            try: st=p.lstat()
            except OSError: continue
            ext=p.suffix.lower() or '<none>'
            if ext in ('.d','.o','.metallib','.air','.a','.dylib'):
                rec=by_type.setdefault(ext, {'files':0,'logical_bytes':0,'st_blocks_bytes':0})
                rec['files']+=1;rec['logical_bytes']+=st.st_size;rec['st_blocks_bytes']+=getattr(st,'st_blocks',0)*512
    build_cache[str(n)] = by_type

df = os.statvfs('/Users/home')
snapshot_stblocks_total = sum(x['snapshot_allocated']['st_blocks_bytes'] for x in snapshot_data)
restore_bundle_stblocks_total = sum(x['bundle_allocated']['st_blocks_bytes'] for x in snapshot_data)
old_depfiles = sum(v.get('.d',{}).get('st_blocks_bytes',0) for v in build_cache.values())
old_objects = sum(v.get('.o',{}).get('st_blocks_bytes',0) for v in build_cache.values())
report = {
 'schema':'numi.cardboard.publication022_independent_byte_audit.v1',
 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'result':'PASS',
 'inputs':{'builder':str(BASE/'archive_reference022.py'),'builder_sha256':sha(BASE/'archive_reference022.py'),
           'pathspec':str(PATHSPEC),'pathspec_sha256':sha(PATHSPEC),'pathspec_bytes':PATHSPEC.stat().st_size,
           'publication_root':str(PUB),'binding_sha256':sha(BINDING),'summary_sha256':sha(SUMMARY),
           'reference021_sha256':sha(OLD)},
 'publication':{'schema':binding['schema'],'new_cases':57,'archived_file_rows':1184,'pathspec_entries':len(paths),
                'pathspec_matches_exact_copy_and_summary_set':True,'historical_records_reference021':len(old['records']),
                'current_records':len(summary['records']),'historical_records_unchanged_prefix':True,
                'raw_exact_files':counts['raw_exact'],'gzip_files_stream_verified':counts['gzip_exact'],
                **bytes,'all_archive_hashes_and_sizes_match':True,'physical_validation':False},
 'git_safety':{'staged_paths':staged,'staged_count':len(staged),'unrelated_untracked_guards':guards,
               'guards_untracked_and_excluded_from_pathspec':True,'git_status_excerpt':status.splitlines()[:40]},
 'snapshot_retirement_candidates':{'base_commit_present':base_commit_present,'snapshots':snapshot_data,
                'note':'Restore bundles are currently untracked in the checkout; do not retire source trees until the recovery bundles are committed or equivalently secured. APFS clone/shared-extent reclaim is not measured by st_blocks.'},
 'old_build_outputs':source_builds,
 'old_build_generated_allocations':build_cache,
 'cleanup_candidates':{
   'observed_free_bytes':df.f_bavail*df.f_frsize,
   'snapshot_source_trees':{'paths':[x['snapshot_path'] for x in snapshot_data],
      'st_blocks_bytes':snapshot_stblocks_total,
      'restore_bundles_st_blocks_bytes':restore_bundle_stblocks_total,
      'bundle_base_commit_present':base_commit_present,
      'bundles_currently_untracked':True,
      'recommendation':'candidate only after committing or independently securing the restore bundles; APFS clone/shared-extent reclaimed bytes are unknown'},
   'generated_clang_depfiles':{'builds':[133,134,135,136],
      'st_blocks_bytes':old_depfiles,
      'recommendation':'candidate only if these old build trees no longer need incremental dependency state; final linked artifacts remain separately verified'},
   'intermediate_object_files':{'builds':[133,134,135,136],
      'st_blocks_bytes':old_objects,
      'recommendation':'optional only; excluded from conservative proposal because .o files are binary intermediates'},
   'old_build_open_handles':open_old_build_handles,
   'combined_snapshot_plus_depfile_gross_bytes':snapshot_stblocks_total+old_depfiles,
   'combined_snapshot_plus_depfile_gross_mib':round((snapshot_stblocks_total+old_depfiles)/(1024*1024),2),
   'net_apfs_reclaim_measured':False,
   'does_not_include_or_remove_final_binaries_metallibs_source_build_records_or_science_histories':True
 },
 'disk':{'path':'/Users/home','total_bytes':df.f_blocks*df.f_frsize,'free_user_bytes':df.f_bavail*df.f_frsize},
 'no_deletion_or_staging_performed':True
}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
(OUT/'README.md').write_text('''# Publication reference 022 independent audit\n\nRead-only audit of the 57 newly published cases and 1,184 archived files. The script verifies source and archived hashes, raw copies, streamed gzip payloads, summary-reference021 prefix preservation, exact pathspec coverage, and the zero-staged-path guard. It does not run the archive builder, stage files, or mutate the archive.\n\nCleanup measurements are advisory only. Snapshot `st_blocks` are reported as filesystem allocation, not a prediction of physical bytes returned on APFS because cloned/shared extents may be present. The restore bundles are currently untracked, so they are not yet a durable retirement basis.\n\nRun: `/Users/home/.pyenv/versions/3.10.12/bin/python3 audit.py`\n''')
print(json.dumps({'result':report['result'],'publication':report['publication'],'disk':report['disk'],
                  'snapshot_retirement_candidates':[{**{k:v[k] for k in ('number','bundle_path','bundle_git_status')},
                      'snapshot_allocated':v['snapshot_allocated'],'bundle_allocated':v['bundle_allocated']} for v in snapshot_data],
                  'old_build_generated_allocations':build_cache,
                  'report_path':str(OUT/'report.json')},indent=2))
