"""Restore one exact snapshot into a new directory; requires its fixed Git base."""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import json
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repository', required=True, type=Path)
parser.add_argument('--number', required=True, type=int, choices=(109,))
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
package = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
index = json.loads((package / 'manifest.json').read_text())
selected = next(row for row in index['snapshots'] if row['number'] == args.number)
packed = (package / selected['manifest']).read_bytes()
assert sha(packed) == selected['compressed_sha256']
raw = gzip.decompress(packed)
assert sha(raw) == selected['uncompressed_sha256']
manifest = json.loads(raw)
assert manifest['number'] == args.number
names = set()
for row in manifest['files']:
    path = PurePosixPath(row['path'])
    assert not path.is_absolute() and '..' not in path.parts and path.parts
    assert str(path) not in names
    names.add(str(path))
args.output.mkdir(exist_ok=False)
git = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=args.repository,
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for row in manifest['files']:
    if row['storage'] == 'git_base':
        request = index['git_base_revision'] + ':' + row['path']
        assert '\n' not in request and '\r' not in request
        git.stdin.write((request + '\n').encode())
        git.stdin.flush()
        header = git.stdout.readline().decode().rstrip('\n')
        assert not header.endswith(' missing'), request
        _, kind, size = header.split()
        assert kind == 'blob'
        restored = git.stdout.read(int(size))
        assert len(restored) == int(size) and git.stdout.read(1) == b'\n'
    else:
        assert row['storage'] == 'gzip_blob'
        blob = (package / row['blob']).resolve()
        assert blob.is_relative_to(package)
        compressed = blob.read_bytes()
        assert sha(compressed) == row['compressed_sha256']
        restored = gzip.decompress(compressed)
    assert len(restored) == row['bytes'] and sha(restored) == row['sha256']
    target = args.output / row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(restored)
    target.chmod(0o755 if row['executable'] else 0o644)
    assert sha(target.read_bytes()) == row['sha256']
git.stdin.close()
assert git.wait() == 0
print(json.dumps(dict(number=args.number, files=len(names), exact_bytes_verified=True,
                      output=str(args.output), actual_build_success=manifest['actual_build_success'])))
