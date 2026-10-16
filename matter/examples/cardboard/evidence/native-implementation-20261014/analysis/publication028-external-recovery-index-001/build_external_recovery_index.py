from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, os, shutil, tempfile

B = Path('/Users/home/cardboard-implementation-20261014')
A = B / 'analysis'
S = B / 'solver'
BUILDROOT = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157')
OUT = A / 'publication028-external-recovery-index-001'


def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(path): return sha_bytes(path.read_bytes())
def dump(path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
def verified_file(path, expected_sha=None, expected_bytes=None):
    if not path.is_file(): raise RuntimeError(f'missing external file: {path}')
    n = path.stat().st_size
    h = sha(path)
    if expected_sha is not None and h != expected_sha: raise RuntimeError(f'sha mismatch: {path}')
    if expected_bytes is not None and n != expected_bytes: raise RuntimeError(f'size mismatch: {path}')
    return {'path': str(path), 'bytes': n, 'sha256': h}

# Build all records and validate every referenced archive/pointer before writes.
manifest_path = A / 'cardboard-provenance-cmakefiles-compaction-001/archive-manifest-v2.json'
manifest = json.loads(manifest_path.read_text())
manifest_ref = verified_file(manifest_path)
cmake = []
pointer_payloads = []
for target in manifest['targets']:
    archive = Path(target['archive'])
    pointer = Path(target['pointer'])
    inventory = Path(target['inventory'])
    archive_ref = verified_file(archive, target['archive_sha256'], target['archive_bytes'])
    pointer_ref = verified_file(pointer, target['pointer_sha256'])
    inventory_ref = verified_file(inventory, target['inventory_gzip_sha256'])
    decompressed = gzip.decompress(inventory.read_bytes())
    if sha_bytes(decompressed) != target['inventory_uncompressed_sha256']:
        raise RuntimeError(f'inventory uncompressed sha mismatch: {inventory}')
    copied = f"pointers/cmakefiles/{target['build']}/RESTORE_FROM_COMPACTION.txt"
    pointer_payloads.append((pointer, copied, pointer.read_bytes(), pointer_ref['sha256']))
    cmake.append({
      'build': target['build'], 'archive': archive_ref,
      'archive_sha256_binding_source': str(manifest_path),
      'archive_inventory': inventory_ref,
      'inventory_uncompressed_sha256': target['inventory_uncompressed_sha256'],
      'restore_pointer': {**pointer_ref, 'published_copy': copied},
      'restore_verified_before_source_compaction': target['restore_verified_before_removal'],
      'restore_limits': manifest['restoration_limits'],
    })

old_groups = []
for name in ['closed-solver-input-compaction-019','closed-solver-input-compaction-020',
             'closed-solver-input-compaction-023','closed-solver-input-compaction-024']:
    root = A / name
    if name.endswith('019'):
        seal_path = root / 'final-seal.json'; seal = json.loads(seal_path.read_text())
        source_records = seal['cases']
        tuples = [(x['case'], Path(x['archive_path']), x.get('archive_sha256'), x.get('archive_bytes'),
                   Path(x['pointer_path']), x.get('pointer_sha256'), 'sealed_final_seal') for x in source_records]
    elif name.endswith('020'):
        seal_path = root / 'final-seal.json'; seal = json.loads(seal_path.read_text())
        tuples=[]
        for archive_path, expected in seal['archives'].items():
            ap=Path(archive_path); case=ap.parent.name; pp=S/case/'inputs.compaction.json'
            tuples.append((case,ap,expected,ap.stat().st_size if ap.exists() else None,pp,None,'sealed_final_seal'))
    else:
        seal_path = root / 'selected.json'; seal=json.loads(seal_path.read_text()); tuples=[]
        for case in seal['cases']:
            ap=root/'archives'/case/'inputs.tar.gz'; pp=S/case/'inputs.compaction.json'
            tuples.append((case,ap,None,ap.stat().st_size if ap.exists() else None,pp,None,'observed_and_crosschecked_pointer'))
    seal_ref = verified_file(seal_path)
    entries=[]
    for case,ap,expected_sha,expected_bytes,pp,expected_pointer_sha,binding_kind in tuples:
        archive_ref=verified_file(ap, expected_sha, expected_bytes)
        pointer_ref=verified_file(pp, expected_pointer_sha)
        pointer=json.loads(pp.read_text())
        if Path(pointer.get('archive_path','')) != ap:
            raise RuntimeError(f'pointer/archive path mismatch: {pp}')
        if pointer.get('archive_sha256') != archive_ref['sha256']:
            raise RuntimeError(f'pointer/archive sha mismatch: {pp}')
        copied=f"pointers/solver/{name}/{case}/inputs.compaction.json"
        pointer_payloads.append((pp,copied,pp.read_bytes(),pointer_ref['sha256']))
        entries.append({
          'case':case, 'archive':archive_ref, 'archive_binding':binding_kind,
          'pointer':{**pointer_ref,'published_copy':copied},
          'restore_script':pointer.get('restore_script'),
          'restore_required_to_recreate_prediction_inputs':pointer.get('restore_required_to_recreate_prediction_inputs'),
        })
    old_groups.append({'package':name,'seal':seal_ref,'archive_count':len(entries),'archives':entries,
                       'source_retains_archive_sha': name.endswith(('019','020')) or 'archive_sha256' in tuples[0][4].read_text()})

host_root=A/'mortar-union-four-patch-buildhost-compaction-020'
host_manifest_path=host_root/'archive-manifest.json'
host_manifest=json.loads(host_manifest_path.read_text()); host_manifest_ref=verified_file(host_manifest_path)
host_archives=[]
for key,spec in host_manifest['targets'].items():
    archive=host_root/spec['archive']; target=spec['relative_target']
    source_pointer=B / f'analysis/mortar-union-four-patch-overlay-{key}/build-host/RESTORE_FROM_COMPACTION.txt'
    archive_ref=verified_file(archive,spec['archive_sha256'],spec['archive_bytes'])
    pointer_ref=verified_file(source_pointer)
    copied=f'pointers/build-host/overlay-{key}/RESTORE_FROM_COMPACTION.txt'
    pointer_payloads.append((source_pointer,copied,source_pointer.read_bytes(),pointer_ref['sha256']))
    host_archives.append({'overlay':key,'target':target,'archive':archive_ref,
                          'restore_pointer':{**pointer_ref,'published_copy':copied},
                          'restore_command':f'python3 {host_root}/restore_build_host.py {key}',
                          'source_manifest_sha256':host_manifest_ref['sha256']})

# These source157 build outputs are referenced by command003 but intentionally not copied.
sourcebuild_path=A/'source157-telemetry-native-build-001/build-metadata/source-build.json'
sourcebuild=json.loads(sourcebuild_path.read_text()); sourcebuild_ref=verified_file(sourcebuild_path)
observed_artifacts=[]
for ap,expected in sourcebuild['artifacts_sha256'].items():
    ref=verified_file(Path(ap),expected)
    ref['copied_to_publication'] = False
    observed_artifacts.append(ref)

# Stage exact pointer bytes and JSON first; the index is installed last.
stage=OUT/'.pointer-stage'
if stage.exists(): raise RuntimeError(f'staging path already exists: {stage}')
stage.mkdir()
try:
    pointer_records=[]
    for source,rel,data,expected in pointer_payloads:
        if sha_bytes(data)!=expected: raise RuntimeError(f'pointer changed while staging: {source}')
        dest=stage/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        if sha(dest)!=expected: raise RuntimeError(f'copied pointer sha mismatch: {dest}')
        pointer_records.append({'source_path':str(source),'published_copy':rel,'bytes':len(data),'sha256':expected})
    index={
      'schema':'numi.cardboard.publication028.external_recovery_index.v1',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'External recovery location/hash index; archive payloads are not copied into the publication.',
      'cmakefiles_compaction':{'manifest':manifest_ref,'archive_count':len(cmake),
          'archives':cmake,'pointer_copies':sum(x['source_path'].startswith('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-') for x in pointer_records)},
      'closed_solver_input_compactions':old_groups,
      'buildhost_compaction':{'manifest':host_manifest_ref,'archives':host_archives},
      'source157_referenced_build_outputs':{
          'source_build_record':sourcebuild_ref,'artifacts':observed_artifacts,
          'recovery_note':'Command003 was prepared but never launched. Build outputs remain outside publication; reconstruct them from the retained source157 snapshot/build inputs and verify against listed SHA-256 values before any replay.'},
      'pointer_files':pointer_records,
      'verification':{'all_archive_files_present_and_sha_size_checked':True,
          'all_restore_pointers_present_sha_checked_and_byte_copied':True,
          'cmake_inventory_gzip_and_uncompressed_sha_checked':True,
          'archive_payloads_copied':False,
          'native468_active_case_or_launch_guard_included':False},
      'limits':['External archives still depend on their recorded absolute filesystem locations unless separately transported.',
          'Birthtime and inode/hardlink topology caveats are those in each source package restore manifest.',
          'This index establishes current archive and pointer integrity, not publication or physical qualification.']
    }
    dump(stage/'external-recovery-index.json',index)
    (stage/'README.md').write_text('# Publication 028 external recovery index\n\nThis package records exact external archive locations, sizes and SHA-256 checks, and carries byte-exact small restore-pointer files under safe relative names. The archive payloads themselves remain at their original paths and are not duplicated here.\n\nThe source157 build executable and metallibs are referenced by the prepared command but are not copied; the command was not launched. Rebuild from the retained source snapshot and verify the artifact hashes before replay.\n')
    # Refuse overwrite, then atomically move staged contents into the empty package directory.
    for child in stage.iterdir():
        target=OUT/child.name
        if target.exists(): raise RuntimeError(f'output already exists: {target}')
    for child in list(stage.iterdir()): os.replace(child,OUT/child.name)
    stage.rmdir()
except Exception:
    # Keep the stage as failure evidence; never clean partial bytes automatically.
    raise
print(json.dumps({'archive_count':len(cmake)+sum(x['archive_count'] for x in old_groups)+len(host_archives),
                  'pointer_copy_count':len(pointer_payloads),
                  'index_sha256':sha(OUT/'external-recovery-index.json'),
                  'pointer_bytes':sum(x[2].__len__() for x in pointer_payloads)},indent=2))
