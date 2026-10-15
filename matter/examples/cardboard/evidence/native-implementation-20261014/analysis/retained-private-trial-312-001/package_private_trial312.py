from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile

base=Path('/Users/home/cardboard-implementation-20261014')
case=base/'native/corrected-private-trial-layout-312'
reader=base/'analysis/private-trial-reader-001'
out=base/'analysis/retained-private-trial-312-001'
out.mkdir(exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('capture_reader',reader/'trial_capture_reader.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
capture=module.Capture(case/'private-trial.coupled.json')
assert len(capture.arrays)==129 and len(capture.payload)==19467248
for name in ('trial_capture_reader.py','test_trial_capture_reader.py'):
    shutil.copy2(reader/name,out/name)
shutil.copy2(case/'private-trial.coupled.json',out/'private-trial.coupled.json')
compressed=gzip.compress(capture.payload,mtime=0)
(out/'private-trial.coupled.bin.gz').write_bytes(compressed)
(out/'restore.py').write_text('''from pathlib import Path
import argparse,gzip,hashlib,json,shutil
from trial_capture_reader import Capture
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'package.json').read_text())
for name,v in m['artifacts'].items():
 data=(b/name).read_bytes()
 assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'],name
raw=gzip.decompress((b/'private-trial.coupled.bin.gz').read_bytes())
assert len(raw)==m['raw_payload_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_payload_sha256']
a.directory.mkdir(exist_ok=False,parents=True)
with (a.directory/'private-trial.coupled.bin').open('xb') as f:f.write(raw)
with (a.directory/'private-trial.coupled.json').open('xb') as f:f.write((b/'private-trial.coupled.json').read_bytes())
c=Capture(a.directory/'private-trial.coupled.json')
print(json.dumps(c.summary(),indent=2,allow_nan=False))
''')
(out/'README.md').write_text('# Lossless private Armijo trial312\n\n'
    'Restores the exact129-array,19,467,248-byte actual failing native trial with '
    'strict JSON, adjacent payload identity, per-array hashes and typed extents. '
    'Run `python3 restore.py /absolute/new/directory`; the directory must not exist. '
    'NumPy is required by the reader. Original absolute payload metadata is retained '
    'bytewise but never followed. This is a rejected-trial diagnostic, not an '
    'accepted checkpoint, complete nonlinear solve or material validation.\n')
inputs=[]
for path in (case/'prediction.json',case/'receipt.json',case/'inputs/external/source-build.json'):
    inputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path.read_bytes())))
package=dict(schema='numi.cardboard.lossless-private-trial-package.v1',
    raw_payload_bytes=len(capture.payload),raw_payload_sha256=sha(capture.payload),
    artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
        for p in sorted(out.iterdir()) if p.is_file()},source_bindings=inputs,
    capture=capture.summary(),accepted_checkpoint=False,physical_validation=False)
(out/'package.json').write_text(json.dumps(package,indent=2,allow_nan=False)+'\n')
with tempfile.TemporaryDirectory() as temporary:
    result=subprocess.run(['/Users/home/.pyenv/versions/3.10.12/bin/python3',
        str(out/'restore.py'),str(Path(temporary)/'restored')],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    restored=json.loads(result.stdout)
    assert restored['payload_sha256']==package['raw_payload_sha256']
(out/'restore-test.json').write_text(json.dumps(dict(exit_code=result.returncode,
    result=restored,package_sha256=sha((out/'package.json').read_bytes())),indent=2)+'\n')
(out/'package_private_trial312.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(compressed_bytes=len(compressed),payload_sha256=package['raw_payload_sha256'],
    restore_test_exit_code=result.returncode)))
