from pathlib import Path
import csv,hashlib,json,os,subprocess,time,sys
R=Path.cwd();I=R/'artifacts/crow-rl-motor-contact-20261108/precision';B=R/'build/crow-motor-contact-build-01';p=json.loads((I/'plan.json').read_text())
for n,h in p['source_and_runtime_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
commands=[];env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(Path(p['dynamic_library']).parent),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=p['metallib']);env.pop('METALROBO_BOUNDED_FAILURE_CAPTURE',None);env.pop('METALROBO_CCD_FAILURE_CAPTURE',None)
for c in p['commands']:
 start=time.monotonic()
 with (B/(c['name']+'.stdout')).open('xb') as out,(B/(c['name']+'.stderr')).open('xb') as err:
  try:rc=subprocess.run(c['args'],env=env,stdout=out,stderr=err,timeout=p['budget']['timeouts_seconds_each']).returncode
  except subprocess.TimeoutExpired:rc=None
 loads=[s for s in (B/(c['name']+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];entry=dict(c,exit_code=rc,seconds=time.monotonic()-start,loader_lines=loads);commands.append(entry);(I/'execution.json').write_text(json.dumps(commands,indent=2,sort_keys=True)+'\n');print(json.dumps(entry),flush=True)
 if rc!=c['expected_exit']:
  report={'verdict':'contradicted_native_acceptance_or_validity','failure_command':c['name'],'exit_code':rc,'failure':(B/(c['name']+'.stderr')).read_text().splitlines()[-3:],'policy_or_profile_promoted':False};(I/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True);raise SystemExit(1)
 assert loads and all(s.endswith(p['dynamic_library']) for s in loads)
rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((B/'coupled-precision.tsv').open(),delimiter='\t')];assert len(rows)==144
key=lambda r:(r['variant'],r['h'],r['contact'],r['lane']);arms=[{key(r):r for r in rows if r['arm']==a} for a in range(3)];assert all(len(a)==48 for a in arms)
error=lambda r:abs(r['applied_force']-r['final_law_force']);primary=[k for k,r in arms[2].items() if r['contact'] and r['lane']==0];assert len(primary)==6;reductions=[1-error(arms[2][k])/error(arms[0][k]) for k in primary];control=max(abs(arms[0][k]['applied_force']-arms[1][k]['applied_force']) for k in arms[0]);maxerror=max(error(r) for r in arms[2].values())
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
entries=tree();checks={}
with Objects() as o:
 raw=o.read(entries[p['default_reference_manifest']][1]);assert hashlib.sha256(raw).hexdigest()==p['default_reference_manifest_sha256'];m=json.loads(raw)
 for f in m['files']:
  if f['path'].startswith(p['default_reference_prefix']) and f['path'].split('/')[-1]!='runtime.json':
   name=f['path'].split('/')[-1];actual=(B/'default-crow'/name).read_bytes();expected=b''.join(content(f,entries,o));checks[name]={'byte_exact':actual==expected,'sha256':hashlib.sha256(actual).hexdigest()};assert actual==expected,name
assert len(checks)==6
passed=maxerror<=1e-7 and min(reductions)>=.99 and control<=1e-7
report={'verdict':'supported' if passed else 'contradicted','accepted_comparison_native_conditions':len(rows),'tight_momentum_engineering_replay_pass':True,'default_contact_owner_pass_previously_verified_on_same_library':True,'default_crow640transition_comparisons':checks,'candidate_max_force_law_error_N':maxerror,'primary_error_reduction_range':[min(reductions),max(reductions)],'fixed_force_quality_vs_split_force_error_N':control,'maximum_momentum_residual_Ns':max(abs(r['momentum_residual']) for r in rows),'maximum_q_hv_residual_m':max(abs(r['q_residual']) for r in rows),'candidate_max_penetration_m':max(max(-r['q'],0) for r in arms[2].values() if r['contact']),'limits':p['limits'],'policy_or_profile_promoted':False};(I/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
