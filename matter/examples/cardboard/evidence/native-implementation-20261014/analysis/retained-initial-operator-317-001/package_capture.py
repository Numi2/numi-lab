from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import tempfile

base = Path('/Users/home/cardboard-implementation-20261014')
case = base / 'solver/current-newton-augmentation-off-bounded-317'
out = base / 'analysis/retained-initial-operator-317-001'
out.mkdir(exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (case / 'operator-capture').read_bytes()
records = [json.loads(line) for line in raw.splitlines()]
assert records[0]['root'] == 0 and records[0]['newton_iteration'] == 0
compressed = gzip.compress(raw, mtime=0)
(out / 'operator-capture.jsonl.gz').write_bytes(compressed)
(out / 'restore.py').write_text('''from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
data=(b/m['compressed_name']).read_bytes()
assert hashlib.sha256(data).hexdigest()==m['compressed_sha256']
raw=gzip.decompress(data)
assert len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256']
records=[json.loads(line) for line in raw.splitlines()]
assert records[0]['root']==0 and records[0]['newton_iteration']==0
with a.destination.open('xb') as f:f.write(raw)
print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),records=len(records))))
''')
(out / 'README.md').write_text('# Initial bounded operator317\n\nLossless source096 root0/Newton0 feature-off capture from the intentionally capped one-Newton protocol. Exact candidate/history/constraints and cached algorithmic tangents are retained. The initial linear solve uses one iteration and does not reach a restart; the final strict nonlinear certificate rejects.316 enabled counterpart fails eligibility before GPU and is not a parity comparison. This is no complete cycle, accepted assembled checkpoint or augmentation qualification.\n\nRun `python3 restore.py /absolute/new/operator-capture`. Existing files reject.\n')
manifest = dict(schema='numi.cardboard.lossless-actual-operator-package.v1',
                source_run=str(case), receipt=json.loads((case / 'receipt.json').read_text()),
                compressed_name='operator-capture.jsonl.gz', compressed_bytes=len(compressed),
                compressed_sha256=sha(compressed), raw_bytes=len(raw), raw_sha256=sha(raw),
                records=len(records), root=0, newton_iteration=0, complete_cycle=False,
                accepted_checkpoint=False, physical_validation=False)
(out / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tempfile.TemporaryDirectory() as directory:
    result = subprocess.run(['/Users/home/.pyenv/versions/3.10.12/bin/python3',
        str(out / 'restore.py'), str(Path(directory) / 'restored')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    restored = json.loads(result.stdout)
(out / 'restore-test.json').write_text(json.dumps(dict(exit_code=result.returncode,
    restored=restored, package_sha256=sha((out / 'package.json').read_bytes())), indent=2) + '\n')
(out / 'package_capture.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(manifest, indent=2))
