from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-primary-cache-fd-action-253-002';s=q/'snapshots/source-primary-cache-fd-action-253-003';d=q/'analysis/source253-primary-cache-fd-action-003';d.mkdir(exist_ok=False);shutil.copytree(base,s)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text())
p=s/'matter/tools/test_true_residual_certification_contract.py';t=p.read_text();old='self.assertIn("useCache =\\n                        state.mixedSolverValue.executionBudgets.w == 1u", cycle)'
new='self.assertRegex(cycle, r"const std::uint32_t useCache =\\s*numi::matter::detail::femPrimaryOperatorCacheMode\\(\\s*state.mixedSolverValue.executionBudgets.w\\)")'
assert t.count(old)==1;t=t.replace(old,new);p.write_text(t)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'};tests=[]
for name in ['test_cardboard_primary_cache_action.py','test_cardboard_cached_operator_audit.py','test_true_residual_certification_contract.py','test_cardboard_solver_environment.py']:
 a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/name)];r=subprocess.run(a,cwd=s,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/(name+'.stdout')).write_bytes(r.stdout);(d/(name+'.stderr')).write_bytes(r.stderr);tests.append({'argv':a,'exit_code':r.returncode});assert r.returncode==0,(name,r.stderr.decode()[-1000:])
f.update(base_snapshot=str(base),base_frozen_sha256=sha(base/'frozen-source.json'));f['source_sha256']={n:sha(s/n) for n in sorted(f['source_sha256'])};(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
report={'schema':'numi.cardboard.source253-policy-contract-003.v1','source':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'test_only_change_vs002':'True-residual staticcontract follows sharedprimarycachehelper; compiled6case policytest checks mapping0/1/unrecognizedflags. Productioncodeunchanged.','tests':tests,'all_pass':True,'native_validation_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
