from pathlib import Path
import json,hashlib,datetime,subprocess
out=Path(__file__).parent
snapshot=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-canonical-fp32-replay-and-rigid-checkpoint-integrated-061')
source=out/'audit.mm';binary=out/'audit';input=Path('/Users/home/cardboard-implementation-20261014/native/canonical-fp32-all-center-replay-192/center-replay.jsonl')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[Path(__file__),source,input]+[snapshot/'matter/tools'/f for f in ['cardboard_solver_reference.hpp','cardboard_paper_reference.hpp']]
bindings={str(p):sha(p) for p in files}
prediction=dict(schema='numi.cardboard.cpu-analytic-derivative-prediction.v1',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bindings=bindings,prediction='Differentiate the complete source13 implicit equations independently at exact captured FP32-promoted inputs. Compare native accurate tangent before interpreting branch-crossing centered differences. Sweep both one-sided and centered differences, recording multiplier/yield gap and local failures. Full-vector relative comparisons are descriptive; no measured Metal absolute noise floors, source15/damage, full solver or physical qualification.',physical_steps=0)
(out/'prediction.json').write_text(json.dumps(prediction,indent=2)+'\n')
compile=['xcrun','clang++','-O2','-std=c++20','-fobjc-arc','-framework','Foundation','-I',str(snapshot/'matter/tools'),str(source),'-o',str(binary)]
with (out/'build.log').open('w') as f: code=subprocess.call(compile,stdout=f,stderr=subprocess.STDOUT)
if code:raise SystemExit(code)
with (out/'rows.jsonl').open('w') as f,(out/'stderr.log').open('w') as e:code=subprocess.call([str(binary),str(input)],stdout=f,stderr=e)
assert {str(p):sha(p) for p in files}==bindings
(out/'receipt.json').write_text(json.dumps(dict(schema='numi.cardboard.cpu-audit-receipt.v1',compile=compile,exit_code=code,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary_sha256=sha(binary),rows_sha256=sha(out/'rows.jsonl'),physical_steps=0,physical_validation=False),indent=2)+'\n')
print((out/'stderr.log').read_text());raise SystemExit(code)
