from pathlib import Path
import json,subprocess,shutil,time,hashlib
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
C=A/'parent-bindings-native-identity-1353-003';D=C/'negative-rest-registration-002';D.mkdir(exist_ok=False)
for name in ('common-cardiac-map-f32.bin','common-cardiac-domains-f32.bin','common-cardiac-volumes-f32.bin'):shutil.copyfile(C/name,D/name)
r=json.loads((C/'resting-anatomy-receipt.json').read_text());r['passive_tissue_parent_bindings']['surfaces'][0]['vertices'][0][0][2]=100
rp=D/'rest-geometry-shift.json';rp.write_text(json.dumps(r))
decl=json.loads((C/'baseline/run-declaration.json').read_text());idx=decl['argv'].index('numilab_human.resting_run');env=dict(v.split('=',1) for v in decl['argv'][2:idx-2])
argv=list(decl['owner_cli_preview']['native_argv']);argv[argv.index('--resting-anatomy-receipt')+1]=str(rp);argv[4]=str(D/'native')
argv[argv.index('--resting-movie')+1]=str(D/'native/native-viewer.mov')
log=D/'native.log';at=time.monotonic()
with log.open('w') as f:r=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
passed=r.returncode!=0 and 'changes registered rest geometry' in log.read_text()
report=dict(passed=passed,returncode=r.returncode,wall_seconds=time.monotonic()-at,receipt_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),prior_attempt='negative-admission/rest_geometry_shift rejected earlier common-field absolute sidecar path; not a test of rest-geometry gate')
(D/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert passed
