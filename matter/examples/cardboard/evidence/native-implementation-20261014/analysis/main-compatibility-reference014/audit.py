from pathlib import Path
import json,hashlib,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');D=R/'build-cardboard-main-compatibility014';O=B/'analysis/main-compatibility-reference014';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((B/'native-380-command.json').read_text());native=base[base.index('--')+1:];native=[v.replace(str(R/'build-cardboard-provenance-109'),str(D)).replace(str(B/'snapshots/source-explicit-material-budget-109'),str(R)) for v in native];native[native.index('--output')+1]=str(O/'compile32/output');native+=['--compile-only']
cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(R/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(R),'--runtime-dir',str(D),'--build-provenance',str(D/'source-build.json'),'--case',str(O/'compile32'),'--timeout','120','--question','Does the rebased current-main build compile the explicitly requested accurate material budget32?','--prediction','Fresh probe and policy builds on main include concurrent03cd38cd Human header changes. All Matter production files match qualifiedsource109. The compile-only probe must report effective32 for allmaterials and perform zero physical transactions or Metal runtime initialization. Earlier native evidence remains bound to its original source/build.','--',*native]
(O/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (O/'runner.log').open('x') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
r=json.loads((O/'compile32/receipt.json').read_text());assert rc==0 and r['exit_code']==0 and r['source_inputs_unchanged'] and not r['timed_out'] and not r['termination_errors']
m=json.loads((O/'compile32/output/manifest.json').read_text());assert m['solver']['compiled_local_material_newton_iterations']==[32]*3
assert json.loads((O/'compile32/output/result.json').read_text())['status']=='compile_only'
a=json.loads((B/'analysis/source109-110-main-adoption-001/receipt.json').read_text());assert all(sha(R/k)==v for k,v in {**a['source109_files'],**a['evaluator110_files']}.items())
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
(O/'report.json').write_text(json.dumps(dict(schema='numi.cardboard.main-compatibility-reference014.v1',git_head=head,upstream='03cd38cd',fresh_targets=['numi-matter-cardboard-probe','numi-matter-cardboard-bilinear15-projection-policy-check'],source_build_sha256=sha(D/'source-build.json'),policy_cpu_check_exit_code=0,compile_only_effective_budgets=[32,32,32],main_cardboard_files_match109_110=True,native_metal_execution=False,physical_steps=0,prior_native_evidence_rebound_to_main=False),indent=2)+'\n')
(O/'source-build.json').write_bytes((D/'source-build.json').read_bytes());(O/'audit.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and 'inputs' not in p.relative_to(O).parts}),indent=2)+'\n');print((O/'report.json').read_text())
