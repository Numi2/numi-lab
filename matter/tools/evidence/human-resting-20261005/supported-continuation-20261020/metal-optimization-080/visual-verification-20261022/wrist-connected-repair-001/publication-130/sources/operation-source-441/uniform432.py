from pathlib import Path
import json,hashlib,subprocess,os,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069");out=D/"D/uniform-control-432/run-001";out.mkdir(parents=True,exist_ok=False)
cfg=json.loads((D/"D/postfit-backtrack-428/run-001/backtrack-config.json").read_text());cfg.update(out=str(out/"backtrack-373"),uniform_failure_backtracking=True,initial_scale=.5,maximum_exact_audits=8,maximum_construction_passes=12)
p=out/"config.json";p.write_text(json.dumps(cfg,indent=2)+chr(10))
env=os.environ.copy();env.update(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=cfg["local_step_context_npz"],NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=cfg["local_step_context_sha256"],NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1")
argv=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(D/"barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py"),str(p)]
pins={x:hashlib.sha256(Path(x).read_bytes()).hexdigest()for x in [argv[1],str(p),cfg["trial"],cfg["parent_fit"]]}
(out/"launch.json").write_text(json.dumps(dict(argv=argv,pins=pins,native_admission=False),indent=2))
t=time.monotonic()
with(out/"execution.log").open("x")as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pins.items())
(out/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,selected_result_exists=(out/"backtrack-373/selected-result.json").exists(),native_admission=False),indent=2))
print(r.returncode)
