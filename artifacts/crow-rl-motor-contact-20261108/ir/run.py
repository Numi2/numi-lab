from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();I=R/'artifacts/crow-rl-motor-contact-20261108/ir';B=R/'build/crow-motor-contact-build-01';p=json.loads((I/'plan.json').read_text())
for n,h in p['source_and_runtime_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not (B/'coupled.tsv').exists();env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(Path(p['dynamic_library']).parent),DYLD_PRINT_LIBRARIES='1');env.pop('METALROBO_BOUNDED_FAILURE_CAPTURE',None);env.pop('METALROBO_CCD_FAILURE_CAPTURE',None);start=time.monotonic()
with (B/'coupled.stdout').open('xb') as out,(B/'coupled.stderr').open('xb') as err:
 try:rc=subprocess.run(p['command'],env=env,stdout=out,stderr=err,timeout=p['budget']['timeout_seconds']).returncode
 except subprocess.TimeoutExpired:rc=None
loads=[s for s in (B/'coupled.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];execution={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'plan_sha256':hashlib.sha256((I/'plan.json').read_bytes()).hexdigest(),'command':p['command'],'exit_code':rc,'seconds':time.monotonic()-start,'loader_lines':loads};(I/'execution.json').write_text(json.dumps(execution,indent=2,sort_keys=True)+'\n')
rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((B/'coupled.tsv').open(),delimiter='\t')];report={'accepted_native_conditions':len(rows),'expected_conditions':144,'exit_code':rc,'policy_or_profile_promoted':False}
if rc!=0:
 report.update(verdict='contradicted_native_acceptance_or_validity',failure=(B/'coupled.stderr').read_text().splitlines()[-3:]);(I/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True);raise SystemExit(1)
assert loads and all(s.endswith(p['dynamic_library']) for s in loads) and len(rows)==144
key=lambda r:(r['variant'],r['h'],r['contact'],r['lane']);arms=[{key(r):r for r in rows if r['arm']==a} for a in range(3)];assert len(arms[0])==len(arms[1])==len(arms[2])==48
error=lambda r:abs(r['applied_force']-r['final_law_force']);primaries=[k for k,r in arms[2].items() if r['contact'] and r['lane']==0];reductions=[1-error(arms[2][k])/error(arms[0][k]) for k in primaries];force_control=max(abs(arms[0][k]['applied_force']-arms[1][k]['applied_force']) for k in arms[0]);max_error=max(error(r) for r in arms[2].values());passed=max_error<=1e-7 and min(reductions)>=.99 and force_control<=1e-7
report.update(verdict='supported' if passed else 'contradicted',candidate_max_force_law_error_N=max_error,primary_error_reduction_range=[min(reductions),max(reductions)],fixed_force_quality_vs_split_force_error_N=force_control,maximum_momentum_residual_Ns=max(abs(r['momentum_residual']) for r in rows),maximum_q_hv_residual_m=max(abs(r['q_residual']) for r in rows),candidate_max_penetration_m=max(max(-r['q'],0) for r in arms[2].values() if r['contact']),limits=p['limits']);(I/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
