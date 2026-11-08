from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01';P=B/'reference';O=B/'qualification-observations'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'qualification-plan.json').read_text());prior=json.loads((A/'integration-execution.json').read_text())
assert len(prior)==9 and prior[-1]['name']=='api-0' and prior[-1]['returncode'] is None
for p,h in plan['files'].items():assert sha(R/p)==h
O.mkdir();env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(B/'runtime/metalrobo/MetalRobo.metallib')}
execution=[]
for name in ['api-0','api-1']:
 local=dict(env);local['METALROBO_QUALITY_FAILURE_CAPTURE']=str(O/(name+'-capture.bin'))
 cmd=[P/'crow-fixture',O/name,'0','1','32','2','api-default'];start=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as stdout,(O/(name+'.stderr')).open('xb') as stderr:
  try:code=subprocess.run([str(x) for x in cmd],env=local,stdout=stdout,stderr=stderr,timeout=plan['limits']['per_command_seconds']).returncode
  except subprocess.TimeoutExpired:code=None
 execution.append(dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start));write(A/'qualification-execution.json',execution);print(json.dumps(execution[-1]),flush=True)
 assert code in [0,1], 'Bounded qualification interrupted; preserve partial outputs'
 assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib')) for line in (O/(name+'.stderr')).read_text().splitlines())
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<plan['limits']['raw_bytes']
cases=[]
for name in ['api-0','api-1']:
 p=O/name;trace=list(csv.DictReader((p/'trace.tsv').open(),delimiter='\t'))
 summary=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else None
 failure=json.loads((p/'failure.json').read_text()) if (p/'failure.json').exists() else None
 terminal=json.loads((p/'terminal.json').read_text()) if (p/'terminal.json').exists() else None
 penetration=max([0.]+[-float(r[k]) for r in trace for k in ['upper_gap_m','lower_gap_m']]) if trace else None
 files={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name!='runtime.json'}
 cap=O/(name+'-capture.bin');files['capture']=sha(cap) if cap.exists() else None
 cases.append(dict(name=name,summary=summary,failure=failure,terminal=terminal,trace_rows=len(trace),max_sampled_penetration_m=penetration,physical_files=files))
obs=dict(api_default_complete_640_twice=all(r['summary'] and r['summary']['accepted_transitions']==640 and not r['failure'] and not r['terminal'] for r in cases),
 api_default_penetration_max_m=max([r['max_sampled_penetration_m'] for r in cases if r['max_sampled_penetration_m'] is not None],default=None),
 repeated_api_physical_outputs_exact=cases[0]['physical_files']==cases[1]['physical_files'])
passed={k:(v==plan['predictions'][k] if k!='api_default_penetration_max_m' else v is not None and v<=plan['predictions'][k]) for k,v in obs.items()}
write(A/'qualification-analysis.json',dict(cases=cases))
write(A/'qualification-result.json',dict(observations=obs,predictions_supported=passed,all_predictions_supported=all(passed.values()),native_commands=len(execution),goal_complete=False,policy_or_profile_promoted=False))
print((A/'qualification-result.json').read_text(),flush=True)
