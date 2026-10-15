from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import tempfile

base = Path('/Users/home/cardboard-implementation-20261014')
source = base / 'native/controlled-material-sweep-native-306/output'
out = base / 'analysis/retained-controlled-material-sweep-306-001'
out.mkdir(exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
files = {}
for path in sorted(source.glob('paper-derivative-sweep-*.jsonl')):
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    assert records[0]['rows'] == records[-1]['rows'] == len(records) - 2 == 960
    compressed = gzip.compress(raw, mtime=0)
    (out / (path.name + '.gz')).write_bytes(compressed)
    files[path.name] = dict(raw_bytes=len(raw), raw_sha256=sha(raw),
        compressed_name=path.name + '.gz', compressed_bytes=len(compressed), compressed_sha256=sha(compressed))
assert len(files) == 2
(out / 'restore.py').write_text('''from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
a.directory.mkdir(exist_ok=False,parents=True)
for name,v in m['files'].items():
 data=(b/v['compressed_name']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==v['compressed_sha256']
 raw=gzip.decompress(data)
 assert len(raw)==v['raw_bytes'] and hashlib.sha256(raw).hexdigest()==v['raw_sha256']
 rows=[json.loads(line) for line in raw.splitlines()]
 assert rows[0]['rows']==rows[-1]['rows']==len(rows)-2==960
 with (a.directory/name).open('xb') as f:f.write(raw)
print(json.dumps(dict(restored_files=len(m['files']),samples_per_policy=960)))
''')
(out / 'README.md').write_text(
    '# Lossless controlled material sweep306\n\n'
    'Restores both original960-sample native policy outputs, including all actual '
    'input and65 output words. Run `python3 restore.py /absolute/new/directory`. '
    'The directory must not exist. Hashes and full record counts are checked. '
    'The sweep completes; branch-aware independent accuracy and arithmetic '
    'noise floors remain separate. It contains zero physical solver steps.\n')
manifest = dict(schema='numi.cardboard.lossless-controlled-sweep-package.v1',
    source=str(source), files=files, physical_steps=0, physical_validation=False)
(out / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tempfile.TemporaryDirectory() as directory:
    result = subprocess.run(['/Users/home/.pyenv/versions/3.10.12/bin/python3',
        str(out / 'restore.py'), str(Path(directory) / 'restored')], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
(out / 'restore-test.json').write_text(json.dumps(dict(exit_code=result.returncode,
    result=json.loads(result.stdout), package_sha256=sha((out / 'package.json').read_bytes())), indent=2) + '\n')
(out / 'package_sweep.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(manifest, indent=2))
