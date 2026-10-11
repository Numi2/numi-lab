from pathlib import Path
import json,subprocess,time,hashlib
W=Path('/Users/n/numi-lab-parent-bindings-1353');B=Path('/Users/n/numi-lab-parent-bindings-build-1353-004')
report=json.loads((B/'build-report.json').read_text());argv=list(report['commands'][0]['argv'])
argv.remove('-DNUMI_HUMAN_RESTING_SCENE=1')
i=argv.index('-o');del argv[i:i+2]
if '-c' in argv:argv.remove('-c')
argv.append('-fsyntax-only');at=time.monotonic()
with (B/'nonrest-syntax.log').open('w') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=B.with_name('numi-lab-parent-bindings-build-1353-003')/'bin/numi-human-native'
result=dict(nonrest_syntax_returncode=r.returncode,nonrest_syntax_argv=argv,wall_seconds=time.monotonic()-at,final_binary_sha256=sha(B/'bin/numi-human-native'),native_tested_binary_sha256=sha(prior),binary_exact_to_native_tested=sha(B/'bin/numi-human-native')==sha(prior),source_changes_since_native_test='Preprocessor guard for optional field in non-resting target; indentation only. Compiled resting binary is byte-identical.')
(B/'final-build-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='nonrest_syntax_argv'}));assert r.returncode==0 and result['binary_exact_to_native_tested']
