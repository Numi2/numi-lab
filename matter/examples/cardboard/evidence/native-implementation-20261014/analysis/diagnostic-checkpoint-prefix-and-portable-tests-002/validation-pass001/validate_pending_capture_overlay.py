from pathlib import Path
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

B=Path('/Users/home/cardboard-implementation-20261014')
S=B/'snapshots/source-right-grip-telemetry-157'
BASE=B/'analysis/cardboard-effective-solver-environment-source161-002'
O=B/'analysis/diagnostic-checkpoint-prefix-and-portable-tests-002'
V=O/'validation-pass002'
V.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
patch='';rows=[]
for path in sorted((O/'candidate/matter/tools').glob('*.py')):
    relative=str(path.relative_to(O/'candidate'))
    before=BASE/'source'/relative
    old=before.read_text() if before.is_file() else ''
    patch+=''.join(difflib.unified_diff(old.splitlines(keepends=True),path.read_text().splitlines(keepends=True),
                   fromfile='a/'+relative if before.is_file() else '/dev/null',tofile='b/'+relative))
    rows.append(dict(path=relative,base_sha256=sha(before) if before.is_file() else None,overlay_sha256=sha(path)))
(O/'overlay.patch').write_text(patch)
record=dict(schema='numi.cardboard.pending-capture-overlay-validation.v1',
            native_execution_performed=False,commands=[])
with tempfile.TemporaryDirectory(prefix='cardboard-pending-capture-validation-',dir=B) as temp:
    root=Path(temp)
    for relative,expected in json.loads((S/'frozen-source.json').read_text())['source_sha256'].items():
        source=S/relative
        assert sha(source)==expected
        target=root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for index, patch_path in enumerate([
        B/'analysis/selector5-restart-consumer-capture-source160-002/source-overlay.patch',
        BASE/'source-overlay.patch',O/'overlay.patch']):
        command=['patch','-p1','--batch','--fuzz=0','-i',str(patch_path)]
        result=subprocess.run(command,cwd=root,capture_output=True,text=True)
        (V/f'patch-{index:02d}.stdout.log').write_text(result.stdout)
        (V/f'patch-{index:02d}.stderr.log').write_text(result.stderr)
        record['commands'].append(dict(command=command,returncode=result.returncode))
        assert result.returncode==0,(patch_path,result.stdout,result.stderr)
    for row in rows:
        assert sha(root/row['path'])==row['overlay_sha256']
    command=['/Users/home/.pyenv/versions/3.10.12/bin/python3','-m','unittest','-v',
        'test_cardboard_solver_environment','test_cardboard_checkpoint_pending_capture',
        'test_cardboard_resolution_study','test_cardboard_convergence',
        'test_cardboard_cached_operator_audit','test_bct_source_bound_harness']
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(root/'matter/tools'),CARDBOARD_TEST_SOURCE_ROOT=str(root),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run(command,env=environment,cwd=root,capture_output=True,text=True)
    (V/'tests.stdout.log').write_text(result.stdout)
    (V/'tests.stderr.log').write_text(result.stderr)
    record['commands'].append(dict(command=command,returncode=result.returncode))
    record['test_source_root']=str(root)
    record['candidate_patch_bytes_verified']=True
    record['affected_test_modules']=6
    record['success']=result.returncode==0
    (V/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    for row in rows:
        compile((root/row['path']).read_text(),row['path'],'exec')
(O/'README.md').write_text('''# Pending diagnostics at an accepted coupon checkpoint

This incremental overlay applies after source161-002. A requested operator-FD
or selector-5 consumer capture may have no files only when the native process
exits successfully after an explicitly requested accepted coupon checkpoint
strictly before that capture's numeric control. The result/prefix schemas,
accepted cursor, both-environment metadata flag and checkpoint/metadata hashes
must agree. Paths remain fresh, bounded and local to the current case.

The resulting status is `validated_with_pending_checkpoint_targets`, with an
explicit per-target reason and no positive qualification. Full normal runs,
failed native processes and already-visited capture targets still reject absent
files. Existing capture consumers continue to require actual validated files.
This does not qualify native restore or uninterrupted/resumed trajectory parity.

The environment and resolution tests now use their checkout's source, with an
explicit test-only source-root option for isolated overlay validation. They do
not require this machine's historical snapshot/overlay directories.

The first seven-test attempt's duplicate diagnostic-directory fixture error and
its exact executed candidate remain in validation-attempt001. The fixture was
repaired; production directory freshness rules were not relaxed. Validation uses
the exact source157 selected files plus the source160-002/source161-002 patches,
then this patch. No native build or GPU work is performed here.
''')
shutil.copy2(Path(__file__),O/'validate.py')
(O/'source-overlay.json').write_text(json.dumps(dict(
    schema='numi.cardboard.source-overlay.v1',overlay_id=O.name,
    base_package=str(BASE),base_manifest_sha256=sha(BASE/'source-overlay.json'),
    patch_sha256=sha(O/'overlay.patch'),files=rows,native_execution_performed=False,
    positive_qualification_allowed=False),indent=2)+'\n')
(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-files.v1',
    files_sha256={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*'))
                  if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}),indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(O/'manifest.json'),patch_sha256=sha(O/'overlay.patch'),
                     source_overlay_sha256=sha(O/'source-overlay.json'))))
