from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-motor-contact-20261108';B=R/'build/crow-motor-contact-build-01';plan=json.loads((A/'plan.json').read_text())
for name,h in plan['source_and_runtime_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
assert not (B/'native.tsv').exists();env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(plan['dynamic_library']).parent);env['DYLD_PRINT_LIBRARIES']='1';env.pop('METALROBO_BOUNDED_FAILURE_CAPTURE',None);env.pop('METALROBO_CCD_FAILURE_CAPTURE',None)
start=time.monotonic()
with (B/'native.stdout').open('xb') as out,(B/'native.stderr').open('xb') as err:
 try:rc=subprocess.run(plan['command'],env=env,stdout=out,stderr=err,timeout=plan['budget']['timeout_seconds']).returncode
 except subprocess.TimeoutExpired:rc=None
loads=[s for s in (B/'native.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
execution={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':plan['command'],'exit_code':rc,'seconds':time.monotonic()-start,'loader_lines':loads,'plan_sha256':hashlib.sha256((A/'plan.json').read_bytes()).hexdigest()};(A/'execution.json').write_text(json.dumps(execution,indent=2,sort_keys=True)+'\n');print(json.dumps(execution),flush=True)
assert rc==0 and loads and all(s.endswith(plan['dynamic_library']) for s in loads)
rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((B/'native.tsv').open(),delimiter='\t')];assert len(rows)==96
primary=[r for r in rows if r['bounded'] and r['contact'] and r['lane']==0];assert len(primary)==6
null=[r for r in rows if r['bounded'] and (not r['contact'] or r['lane']==2)]
relative=[abs(r['applied_force']-r['final_law_force'])/abs(r['final_law_force']) for r in primary]
control=max(abs(r['applied_force']-r['final_law_force']) for r in null)
assert control<=1e-7
report={'accepted_native_states':len(rows),'primary_conditions':[dict(r,relative_force_law_discrepancy=e) for r,e in zip(primary,relative)],'primary_relative_discrepancy_range':[min(relative),max(relative)],'free_or_continuously_saturated_force_error_N':control,'maximum_momentum_residual_Ns':max(abs(r['momentum_residual']) for r in rows),'maximum_q_hv_residual_m':max(abs(r['q_residual']) for r in rows),'prediction_supported_on_registered_conditions':min(relative)>=.05,'physics_errors':0,'policy_or_profile_promoted':False,'limits':'A deterministic 1-DoF zero-gravity frictionless contact diagnosis. It identifies a force-law inconsistency but does not by itself establish the cause of the crow tilt or qualify a revised physical model.'};(A/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
