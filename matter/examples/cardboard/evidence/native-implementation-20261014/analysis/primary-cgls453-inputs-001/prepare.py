from pathlib import Path
import datetime,hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-cgls-primary-policy-143';old=B/'solver/bounded-cgls64-exact-state-450';new=B/'solver/cgls-primary-after48-control1-453'
O=B/'analysis/primary-cgls453-inputs-001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
post=json.loads((B/'analysis/source143-primary-policy-integration-001/post-build.json').read_text());assert post['exit_code']==0
assert json.loads((B/'analysis/source143-primary-policy-independent-review-001/report.json').read_text())['blocking_findings']==[]
assert json.loads((B/'analysis/source143-host-policy-parser-check-001/report.json').read_text())['pass']
assert not new.exists();O.mkdir(exist_ok=False)
c=json.loads((B/'native-450-command.json').read_text())
oldS=B/'snapshots/source-bounded-native-cgls-mask-repair-142';oldBuild=R/'build-cardboard-provenance-142';newBuild=R/'build-cardboard-provenance-143'
c=[x.replace(str(old),str(new)).replace(str(oldS),str(S)).replace(str(oldBuild),str(newBuild)) for x in c]
c[c.index('NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=1:48')]='NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=1:all'
c[c.index('--question')+1]='Does explicit CGLS-primary direction selection from Newton48 complete the unchanged control1 and full coupon through the original strict certificates?'
c[c.index('--prediction')+1]=('Source142450 establishes a native CGLS direction that lowers the true nonlinear residual73.68%, followed by small accepted Newton steps and terminal rollback. Source143 deliberately selects that residual-minimizing direction before ordinary Armijo from N48 onward on control1, while preserving the same operator, material, dt, tooling, physical duration, Newton64/restart16/FGMRES1024/material32 budgets and final certificates. Prediction: the first48 primary Newton records and the N48 CGLS binary action fields match450; later residual decrease is stronger than450 and control1 may satisfy its unchanged certificate. Primary outcome is complete loading/hold/unload/release/relaxation coupon under all original gates, never an accepted-prefix success. A failed or timed-out complete attempt remains a failure. No damage/contact/convergence/box/impact qualification follows from this numerical direction experiment.')
files=[(B/'analysis/source143-primary-policy-independent-review-001/report.json','source143-independent-policy-review.json'),(B/'analysis/base-cgls-primary-policy-overlay-143-001/manifest.json','source143-policy-overlay-manifest.json'),(B/'analysis/source143-host-policy-parser-check-001/report.json','source143-host-parser-report.json'),(B/'analysis/native450-cgls-adjudication-001/report.json','native450-root-adjudication.json'),(old/'receipt.json','native450-baseline-receipt.json')]
bindings=[];insert=['--env','NM_FEM_BASE_CGLS_PRIMARY_AFTER=48']
for original,name in files:
 target=O/name;shutil.copy2(original,target);assert sha(target)==sha(original)
 bindings.append({'original':str(original),'alias':str(target),'sha256':sha(target)});insert+=['--input-file',str(target)]
(O/'input-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');insert+=['--input-file',str(O/'input-bindings.json')]
c[c.index('--'):c.index('--')]=insert
inputs=[Path(c[i+1]) for i,x in enumerate(c) if x=='--input-file'];assert len({p.name for p in inputs})==len(inputs)
assert all(p.is_file() for p in inputs)
(B/'native-453-command.json').write_text(json.dumps(c,indent=2)+'\n');(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
guard=(B/'run_native450.py').read_text().replace('native-450-command.json','native-453-command.json').replace('native450-launch-guard-001','native453-launch-guard-001').replace('native450-runner.log','native453-runner.log')
(B/'run_native453.py').write_text(guard)
(O/'protocol.json').write_text(json.dumps({'schema':'numi.cardboard.native453_primary_policy_protocol.v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command_sha256':sha(B/'native-453-command.json'),'source_snapshot':str(S),'source_build_sha256':sha(newBuild/'source-build.json'),'stage':'PREPARED_NOT_EXECUTED','physical_validation':False,'physical_parameters_unchanged_from450':True,'new_solver_policy':'CGLS64 primary from N48 inclusive on control1, original budgets and gates'},indent=2)+'\n')
print('453 prepared',sha(B/'native-453-command.json'))
