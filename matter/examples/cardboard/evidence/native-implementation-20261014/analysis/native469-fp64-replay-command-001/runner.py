from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
P = B / 'analysis/bilinear-capture-layer-audit-source166-001/candidate/matter/tools'
S = B / 'snapshots/source-right-grip-telemetry-157/matter/tools'
O = B / 'analysis/native469-fp64-replay-command-001'
O.mkdir(exist_ok=False)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(P)+':'+str(S),
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
command = ['/Users/home/.pyenv/versions/3.10.12/bin/python3',
    str(P/'cardboard_bilinear_capture_layers.py'), '--case',
    str(B/'solver/final-newton-fd-capture-469'), '--output',
    str(B/'analysis/native469-bilinear-layer-replay-001')]
record = dict(command=command, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              tool_sha256=hashlib.sha256(Path(command[1]).read_bytes()).hexdigest(),
              environment={key:env[key] for key in ('PYTHONDONTWRITEBYTECODE','PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
              gpu_execution=False)
(O/'command.json').write_text(json.dumps(record,indent=2)+'\n')
with (O/'stdout.log').open('x') as stdout, (O/'stderr.log').open('x') as stderr:
    result = subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
record.update(returncode=result.returncode, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
(O/'runner.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(record))
raise SystemExit(result.returncode)
