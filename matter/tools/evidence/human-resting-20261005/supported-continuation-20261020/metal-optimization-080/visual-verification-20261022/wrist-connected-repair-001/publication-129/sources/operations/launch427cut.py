from pathlib import Path
import json,hashlib,subprocess,time,os,importlib.util
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-427/run-001';py='/Users/n/numi-human-prep-venv-20261005/bin/python';cfg=P/'backtrack-config.json';j=json.loads(cfg.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((D/'D/postfit-backtrack-424/run-001/backtrack-config.json').read_text());allowed={'helper','helper_sha256','out','initial_scales'}
assert all(j.get(k)==v for k,v in prior.items()if k not in allowed)
v=D/'D/local-backtracking-config-419/graded_context_validation_419.py';sp=importlib.util.spec_from_file_location('validation419',v);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);binding=m.validate_graded_context_419(prior)
runner=D/'barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py';assert sha(runner)=='a91b410c979701a29ab32f376637db566abb745e066b629c000f6a37fea13212'
assert sha(j['helper'])==j['helper_sha256']
env=os.environ.copy();env.update(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=j['local_step_context_npz'],NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=j['local_step_context_sha256'],NUMI_GRADED_LOCAL_STEP_KAPPA_423=str(j['local_step_context_kappa']))
pins={str(f):sha(f)for f in [cfg,Path(j['helper']),Path(j['initial_scales']),v,runner,D/'D/graded-local-step-423/graded_local_step_423.py']}
a=[py,str(runner),str(cfg)];(P/'launch.json').write_text(json.dumps(dict(argv=a,pins=pins,context_binding=binding,new_scope='Construction scale-field LP using frozen026 SAT rows; no physical admission',native_admission=False,accepted_physical_steps=0),indent=2)+'\n')
start=time.monotonic()
with(P/'execution.log').open('x')as f:r=subprocess.run(a,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
assert all(sha(k)==v for k,v in pins.items())
o=dict(returncode=r.returncode,wall_seconds=time.monotonic()-start,log_sha256=sha(P/'execution.log'),runtime_source_pins_unchanged=True,native_admission=False,accepted_physical_steps=0)
(P/'execution.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))

