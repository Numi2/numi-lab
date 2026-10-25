from pathlib import Path
import subprocess, os, signal, json, time, hashlib
D=Path('/Users/n/Work/numi-human-connected-surface-069/direct-authoring-080')
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
WRAP='/tmp/numi-human-geometry-slot.py'
DRIVER=D/'run_refined_connected_embedded_081_lsmr_control.py'
OUT=D/'launch-081-lsmr-control'
OUT.mkdir(exist_ok=False)
argv=['/usr/bin/time','-l',PY,WRAP,PY,str(DRIVER)]
start=time.monotonic()
with (OUT/'stdout.log').open('wb') as out,(OUT/'stderr.log').open('wb') as err:
    proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
    status='completed'
    try: rc=proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        status='outer_timeout_15s'
        os.killpg(proc.pid,signal.SIGKILL)
        rc=proc.wait()
report={'status':status,'returncode':rc,'elapsed_s':time.monotonic()-start,'argv':argv,
        'driver_sha256':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
        'solver_sha256':hashlib.sha256((D/'canonical/matter/tools/human_skin_fat_connected_repair.py').read_bytes()).hexdigest()}
(OUT/'launch-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
