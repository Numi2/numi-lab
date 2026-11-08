from pathlib import Path
import json,subprocess,sys,time
A=Path('artifacts/crow-rl-contact-precision-20261108');B=Path('build/crow-contact-precision-build-01');name=sys.argv[1];cmd=['python/.venv/bin/python',str(A/name)]+sys.argv[2:];tag=Path(name).stem;start=time.monotonic()
with (B/(tag+'.stdout')).open('xb') as out,(B/(tag+'.stderr')).open('xb') as err:
    try:code=subprocess.run(cmd,stdout=out,stderr=err,timeout=120).returncode
    except subprocess.TimeoutExpired:code=None
(A/(tag+'-execution.json')).write_text(json.dumps(dict(argv=cmd,returncode=code,seconds=time.monotonic()-start),indent=2,sort_keys=True)+'\n')
print('returncode',code);print((B/(tag+'.stdout')).read_text());print((B/(tag+'.stderr')).read_text());assert code==0
