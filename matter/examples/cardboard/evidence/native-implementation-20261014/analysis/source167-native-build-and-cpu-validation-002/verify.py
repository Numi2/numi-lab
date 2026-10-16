from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-diagnostic-and-resume-167-final'
BUILD=R/'build-cardboard-provenance-167-final'
O=B/'analysis/source167-native-build-and-cpu-validation-002'
O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((BUILD/'source-build.json').read_text())
assert build['success']
frozen=json.loads((S/'frozen-source.json').read_text())
assert build['source_sha256']==frozen['source_sha256']
assert len(build['source_sha256'])==800
for relative,expected in build['source_sha256'].items():
    assert sha(S/relative)==expected
for filename,expected in build['artifacts_sha256'].items():
    assert sha(Path(filename))==expected
rows=[]
with tempfile.TemporaryDirectory(prefix='cardboard167-cpu-checks-',dir=B) as temp:
    for name in ['coupon-resume','inside-clearance','shipping-clearance-memory','regional-consumer-capture']:
        executable=BUILD/('numi-matter-cardboard-'+name+'-check')
        assert str(executable) in build['artifacts_sha256']
        check_command=[str(executable)]+([str(S/"matter/src/runtime.mm")] if name=="regional-consumer-capture" else [])
        result=subprocess.run(check_command,cwd=temp,capture_output=True,text=True)
        (O/(name+'.stdout.log')).write_text(result.stdout)
        (O/(name+'.stderr.log')).write_text(result.stderr)
        rows.append(dict(command=check_command,executable_sha256=sha(executable),returncode=result.returncode))
        assert result.returncode==0,(name,result.stdout,result.stderr)
metadata=B/'analysis/source157-telemetry-native-build-001/build-metadata/compile_commands.json'
entries=json.loads(metadata.read_text())
variant=[r for r in entries if r['file'].endswith('cardboard_probe.mm')
         and '-DNUMI_CARDBOARD_BOX_NATIVE_DRIVER=1' in (r.get('command','')+' '.join(r.get('arguments',[])))]
assert len(variant)==1
entry=variant[0]
arguments=entry.get('arguments') or shlex.split(entry['command'])
command=[];skip=False
old=str(B/'snapshots/source-right-grip-telemetry-157')
for value in arguments:
    if skip:
        skip=False;continue
    if value=='-o':
        skip=True;continue
    if value=='-c':
        continue
    command.append(value.replace(old,str(S)))
command.append('-fsyntax-only')
result=subprocess.run(command,cwd=R,capture_output=True,text=True)
(O/'box-native-syntax.stdout.log').write_text(result.stdout)
(O/'box-native-syntax.stderr.log').write_text(result.stderr)
rows.append(dict(command=command,returncode=result.returncode,compile_commands_sha256=sha(metadata)))
assert result.returncode==0,result.stdout+result.stderr
report=dict(schema='numi.cardboard.source167-native-build-cpu-validation.v1',
    source=str(S),source_build_sha256=sha(BUILD/'source-build.json'),
    selected_sources_verified=800,build_artifacts_verified=len(build['artifacts_sha256']),checks=rows,
    source159_metadata_erratum=dict(package='source159-inside-surface-role-map-overlay-003',
       reported_base_frozen_sha256='a365fdf346743d45cd4a7e1c4796e718555709485627784b31d4d0833ba68876',
       actual_source157_frozen_sha256=sha(B/'snapshots/source-right-grip-telemetry-157/frozen-source.json'),
       handling='Package preserved. Root independently verified every787selected and1034support base file before applying the pinned patch; this metadata error is not used as source admission.'),
    preparation_attempts=dict(
       first='Integration001 built successfully but captured unrequested patch .orig and Python cache artifacts; it is retained and excluded from native use.',
       second='Integration002 explicitly rejected generated backup artifacts. Raw failed tree, patch logs and failure.json retained.',
       third='Integration003 uses atomic git apply and disables import bytecode; exact expected source-membership equality passes. Its consumer CPU checker invocation lacked the required runtime source argument and aborted; no GPU use.',
       fourth='Integration004 preserves the same intended physical code and corrects the consumer CMake test argument. All prior preparation/build/test artifacts remain retained.'),
    native_gpu_execution_performed=False,
    limitations=['CPU contracts and box-native syntax do not qualify actual Runtime restore or history parity.',
                 'Shipping allocation, complete physical assembly and crush/drop remain unexecuted.'])
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(Path(__file__),O/'verify.py')
(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-files.v1',
    files_sha256={p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()}),indent=2)+'\n')
print(json.dumps(dict(report_sha256=sha(O/'report.json'),manifest_sha256=sha(O/'manifest.json'))))
