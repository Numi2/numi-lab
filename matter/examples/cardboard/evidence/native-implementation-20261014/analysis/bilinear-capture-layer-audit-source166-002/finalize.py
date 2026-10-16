from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np

B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/bilinear-capture-layer-audit-source166-002'
V=O/'validation'
V.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(O/'candidate/matter/tools')+':'+str(B/'snapshots/source-right-grip-telemetry-157/matter/tools'),
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
command=['/Users/home/.pyenv/versions/3.10.12/bin/python3','-m','unittest',
         'test_cardboard_bilinear_capture_layers','-v']
result=subprocess.run(command,env=environment,capture_output=True,text=True)
(V/'tests.stdout.log').write_text(result.stdout)
(V/'tests.stderr.log').write_text(result.stderr)
assert result.returncode==0,result.stderr
before=B/'analysis/native469-bilinear-layer-replay-001/arrays.npz'
after=B/'analysis/native469-bilinear-layer-replay-002/arrays.npz'
with np.load(before) as a,np.load(after) as b:
    assert a.files==b.files
    equality={name:bool(np.array_equal(a[name],b[name],equal_nan=True)) for name in a.files}
    assert all(equality.values())
with tempfile.TemporaryDirectory(prefix='cardboard-source166002-patch-',dir=B) as temp:
    root=Path(temp)
    (root/'matter/tools').mkdir(parents=True)
    command=['patch','-p1','--batch','--fuzz=0','-i',str(O/'overlay.patch')]
    patched=subprocess.run(command,cwd=root,capture_output=True,text=True)
    (V/'patch.stdout.log').write_text(patched.stdout)
    (V/'patch.stderr.log').write_text(patched.stderr)
    assert patched.returncode==0,patched.stdout+patched.stderr
    for path in (O/'candidate/matter/tools').glob('*.py'):
        assert sha(root/'matter/tools'/path.name)==sha(path)
report=dict(schema='numi.cardboard.source166002-reporting-fix-validation.v1',
    tests_returncode=result.returncode,
    strict_json_report_written=True,
    source166001_failed_command=str(B/'analysis/native469-fp64-replay-command-001'),
    source166002_completed_command=str(B/'analysis/native469-fp64-replay-command-002'),
    partial_before_sha256=sha(before),completed_after_sha256=sha(after),
    all_numerical_array_values_identical=True,array_equality=equality,
    actual469_report_sha256=sha(B/'analysis/native469-bilinear-layer-replay-002/report.json'),
    actual469_manifest_sha256=sha(B/'analysis/native469-bilinear-layer-replay-002/manifest.json'),
    patch_application_exact=True,
    native_execution_performed=False,qualification=False)
(V/'report.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(Path(__file__),O/'finalize.py')
(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-files.v1',
    files_sha256={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*'))
                  if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}),indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(O/'manifest.json'),patch_sha256=sha(O/'overlay.patch'),
                     source_overlay_sha256=sha(O/'source-overlay.json'))))
