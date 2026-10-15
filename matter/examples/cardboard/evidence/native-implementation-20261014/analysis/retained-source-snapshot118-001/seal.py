from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile

BASE = Path('/Users/home/cardboard-implementation-20261014')
REPO = Path('/Users/home/numi-cardboard-20261006')
OUT = BASE / 'analysis/retained-source-snapshot118-001'
REVISION = '7ebc64a666c4bd5e1ff61b3684655979ab80dc60'
OUT.mkdir(exist_ok=False)
(OUT / 'blobs').mkdir()
digest = lambda raw: hashlib.sha256(raw).hexdigest()

git = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
cache = {}

def git_bytes(relative):
    if relative not in cache:
        assert '\n' not in relative and '\r' not in relative
        git.stdin.write((REVISION + ':' + relative + '\n').encode())
        git.stdin.flush()
        header = git.stdout.readline().decode().rstrip('\n')
        if header.endswith(' missing'):
            cache[relative] = None
        else:
            _, kind, size = header.split()
            assert kind == 'blob'
            raw = git.stdout.read(int(size))
            assert len(raw) == int(size) and git.stdout.read(1) == b'\n'
            cache[relative] = raw
    return cache[relative]

snapshots = []
for number in (118,):
    build_path = REPO / f'build-cardboard-provenance-{number:03d}/source-build.json'
    build = json.loads(build_path.read_text())
    root = Path(build['source_snapshot'])
    support = json.loads((root / 'configure-support-source.json').read_text())['files']
    files = dict(build['source_sha256'])
    # Snapshot102 deliberately omitted configure support and failed. Preserve
    # that actual thin snapshot; never reconstruct it as a successful one.
    if number != 102:
        for relative, expected in support.items():
            assert relative not in files
            assert digest((root / relative).read_bytes()) == expected
            files[relative] = expected
    for relative in ('frozen-source.json', 'configure-support-source.json'):
        if (root / relative).is_file():
            files[relative] = digest((root / relative).read_bytes())
    rows = []
    for relative, expected in sorted(files.items()):
        path = root / relative
        raw = path.read_bytes()
        assert digest(raw) == expected, (number, relative)
        row = dict(path=relative, bytes=len(raw), sha256=expected,
                   executable=bool(path.stat().st_mode & 0o111))
        original = git_bytes(relative)
        if original is not None and digest(original) == expected:
            row['storage'] = 'git_base'
        else:
            row['storage'] = 'gzip_blob'
            blob = OUT / 'blobs' / (expected + '.gz')
            packed = gzip.compress(raw, mtime=0)
            if blob.exists():
                assert gzip.decompress(blob.read_bytes()) == raw
            else:
                blob.write_bytes(packed)
            row['blob'] = str(blob.relative_to(OUT))
            row['compressed_sha256'] = digest(blob.read_bytes())
        restored = original if row['storage'] == 'git_base' else gzip.decompress((OUT / row['blob']).read_bytes())
        assert restored == raw
        rows.append(row)
    manifest_raw = (json.dumps(dict(number=number, source_snapshot=str(root),
                                   source_build_sha256=digest(build_path.read_bytes()),
                                   actual_build_success=build['success'],
                                   files=rows), indent=2) + '\n').encode()
    manifest_path = OUT / f'source-{number:03d}.json.gz'
    manifest_path.write_bytes(gzip.compress(manifest_raw, mtime=0))
    assert gzip.decompress(manifest_path.read_bytes()) == manifest_raw
    snapshots.append(dict(number=number, manifest=manifest_path.name,
                          compressed_sha256=digest(manifest_path.read_bytes()),
                          uncompressed_sha256=digest(manifest_raw),
                          files=len(rows), all_reconstructed_bytes_verified=True))
    if number in (118,):
        with tempfile.TemporaryDirectory(prefix=f'cardboard-source{number}-restore-', dir=BASE) as tmp:
            destination = Path(tmp)
            for row in rows:
                restored = git_bytes(row['path']) if row['storage'] == 'git_base' else gzip.decompress((OUT / row['blob']).read_bytes())
                target = destination / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(restored)
                if row['executable']:
                    target.chmod(0o755)
                assert digest(target.read_bytes()) == row['sha256']
        snapshots[-1]['complete_disk_restoration_verified'] = True
git.stdin.close()
assert git.wait() == 0
report = dict(schema='numi.cardboard.lossless_source_snapshots.v1',
              git_base_revision=REVISION, snapshots=snapshots,
              unique_gzip_source_blobs=len(list((OUT / 'blobs').glob('*.gz'))),
              physical_qualification=False,
              scope='Reconstructs exact source118 selected build sources plus verified root-CMake support from the fixed published Git base. No binary or scientific outcome is reclassified.')
(OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
shutil.copy2(Path(__file__), OUT / 'seal.py')
print(json.dumps(report))
