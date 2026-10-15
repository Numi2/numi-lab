from pathlib import Path
import json,hashlib,subprocess,time,datetime
b=Path('/Users/home/cardboard-implementation-20261014');p=b/'analysis/private-ccd-metal-compare-006';o=b/'analysis/private-ccd-metal-native-350';o.mkdir(exist_ok=False)
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert h(p/'manifest.json')=='66aa47099fc8066be13109a74739b7d43983d706e06a7c68de1bba039ab2de16'
m=json.loads((p/'manifest.json').read_text())
for r in m['files']:assert h(p/r['path'])==r['sha256'],r['path']
argv=[str(p/'build/run_native'),'--metallib',str(p/'build/private_ee_ccd_diagnostic.metallib'),'--cases',str(p/'captured-input/captured-ee-cases.float4.bin'),'--cases-manifest',str(p/'captured-input/captured-ee-cases.json'),'--build-manifest',str(p/'build/build-manifest.json'),'--output',str(o/'native-result.json')]
pred=dict(schema='numi.cardboard.native-diagnostic-prediction.v1',argv=argv,package_manifest_sha256=h(p/'manifest.json'),question='Does unchanged production Metal agree with extracted budget16 helper for exact captured EE rays, and does cap17 certify them?',prediction='Require raw scalar budget16 parity before attributing iteration exhaustion. Cap17 may certify or remain failclosed; enclosing physical trial remains unqualified.',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(o/'prediction.json').write_text(json.dumps(pred,indent=2)+'\n');t=time.monotonic()
with(o/'stdout.log').open('x')as out,(o/'stderr.log').open('x')as err:code=subprocess.call(argv,stdout=out,stderr=err)
for r in m['files']:assert h(p/r['path'])==r['sha256'],r['path']
r=dict(schema='numi.cardboard.native-diagnostic-receipt.v1',exit_code=code,wall_seconds=time.monotonic()-t,inputs_unchanged=True,physical_transactions=0,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(o/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');(o/'runner.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(r));raise SystemExit(code)
