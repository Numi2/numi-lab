from pathlib import Path
import json,datetime,hashlib,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'build-cardboard-provenance-118-full';O=B/'cpu-checks-118-full';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=(B/'cpu-checks-115/selected-tests.txt').read_text().splitlines();assert len(names)==74
(O/'selected-tests.txt').write_text('\n'.join(names)+'\n')
cmd=['ctest','--test-dir',str(D),'--output-on-failure','-j','4','-R','^('+'|'.join(names)+')$'];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (O/'stdout.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
r=dict(schema='numi.cardboard.cpu-check-receipt.v1',argv=cmd,started_at_utc=start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,selected_count=len(names),source_build=str(D/'source-build.json'),source_build_sha256=sha(D/'source-build.json'),physical_steps=0,physical_validation=False)
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');(O/'run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(r));raise SystemExit(code)
