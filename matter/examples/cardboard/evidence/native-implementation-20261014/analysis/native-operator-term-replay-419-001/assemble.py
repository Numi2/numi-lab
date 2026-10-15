from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
P=B/'analysis/native-cached-operator-replay-360-003'
T=B/'analysis/native-operator-term-replay-360-001'
O=B/'analysis/native-operator-term-replay-419-001'
S=B/'snapshots/source-status5-vtccd-diagnostics-125'
C=B/'solver/two-stage-first-failure-rich-fd-419'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O.mkdir(exist_ok=False);(O/'build').mkdir()
text=(P/'prepare_replay.py').read_text()
text=text.replace('source-boundary-reset-order-101','source-status5-vtccd-diagnostics-125').replace('reset-augmentation-first-failure-capture-360','two-stage-first-failure-rich-fd-419').replace('source101','source125')
text=text.replace('"2:23"','"1:48"').replace('selected 2:23','selected 1:48').replace('(2, 23)','(1, 48)').replace('root 2/Newton 23','root 1/Newton 48').replace('"root": 2, "newton_iteration": 23','"root": 1, "newton_iteration": 48')
before='prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_ROOT") == "2" and\n            prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR") == "23"'
after='prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_ROOT") == "1" and\n            prediction.get("environment", {}).get("NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR") is None'
assert text.count(before)==1;text=text.replace(before,after)
(O/'prepare_replay.py').write_text(text)
def run(name,argv):
 with (O/f'{name}.stdout.log').open('x') as out,(O/f'{name}.stderr.log').open('x') as err:
  p=subprocess.run(argv,stdout=out,stderr=err,timeout=180)
 assert p.returncode==0,(name,p.returncode)
run('prepare',['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(O/'prepare_replay.py')])
text=(T/'operator_terms_replay.mm').read_text().replace('source101','source125')
before='\\\"control_step\\\":2,\\\"newton_iteration\\\":23'
assert text.count(before)==1;text=text.replace(before,'\\\"control_step\\\":1,\\\"newton_iteration\\\":48')
(O/'operator_terms_replay.mm').write_text(text)
prior=json.loads((P/'package-manifest.json').read_text())
cmd=[x.replace(str(P/'cached_operator_replay.mm'),str(O/'operator_terms_replay.mm')).replace(str(P/'build/cached_operator_replay'),str(O/'build/operator_terms_replay')).replace(str(B/'snapshots/source-boundary-reset-order-101'),str(S)).replace(str(B/'solver/reset-augmentation-first-failure-capture-360'),str(C)).replace('build-cardboard-provenance-101','build-cardboard-provenance-125') for x in prior['compile']['command']]
run('compile',cmd)
argv=[str(O/'build/operator_terms_replay'),str(C/'output/compiled.nmatterpack'),str(O/'prepared/replay-input.bin')]
run('preflight',argv+['/tmp/cardboard419-preflight-unused','--preflight-only'])
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
manifest=dict(schema='numi.cardboard.native-operator-term-package.v1',base_packages={str(p):sha(p/'package-manifest.json' if p==P else p/'manifest.json') for p in [P,T]},capture_sha256=sha(C/'operator-capture'),source_snapshot=str(S),source_build_sha256=sha(R/'build-cardboard-provenance-125/source-build.json'),compile_argv=cmd,native_argv=argv+['OUTPUT_DIRECTORY'],preflight_pass=True,native_dispatched=False,physical_transactions=0,files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()})
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('operator419 package prepared',sha(O/'manifest.json'))
