from pathlib import Path
import json,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-witness-20261108';B=R/'build/crow-line-search-witness-build-01'
start=time.monotonic()
with (B/'build.stdout').open('xb') as stdout,(B/'build.stderr').open('xb') as stderr:
 code=subprocess.run(['cmake','--build','build','--config','Release','-j6'],stdout=stdout,stderr=stderr,timeout=300).returncode
report=dict(returncode=code,seconds=time.monotonic()-start,command=['cmake','--build','build','--config','Release','-j6'])
(A/'build-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True);assert code==0
