from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01';P=B/'reference';O=B/'integration-observations'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'integration-plan.json').read_text())
for p,h in plan['files'].items():assert sha(R/p)==h
O.mkdir();lib=B/'runtime/metalrobo/MetalRobo.metallib'
env={**os.environ,'DYLD_LIBRARY_PATH':str(B/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(lib)}
jobs=[('unified-owner',[R/'build/bin/metalrobo_metal_unified_quality_probe'],True),
 ('ir-owner',[R/'build/bin/metalrobo_constraint_ir_probe'],True),
 ('contact-owner',[R/'build/bin/metalrobo_metal_world_contact_probe',lib],True),
 ('motor',[R/'build/bin/metalrobo_motor_contact_probe',lib,O/'motor.tsv','--native-drive'],True),
 ('zero',[R/'build/bin/metalrobo_motor_contact_probe',lib,O/'zero.tsv','--zero-gain'],True),
 ('calibration',[P/'crow-fixture','--calibrate'],True),
 ('default',[P/'fixture-default',O/'default','0','1','32'],True),
 ('strict',[P/'crow-fixture',O/'strict','0','1','32','2','strict'],False),
 ('api-0',[P/'crow-fixture',O/'api-0','0','1','32','2','api-default'],False),
 ('api-1',[P/'crow-fixture',O/'api-1','0','1','32','2','api-default'],False)]
assert len(jobs)==10;execution=[];startAll=time.monotonic()
for name,cmd,required in jobs:
 assert time.monotonic()-startAll<plan['limits']['total_seconds']
 local=dict(env)
 if name in ['strict','api-0','api-1']:local['METALROBO_QUALITY_FAILURE_CAPTURE']=str(O/(name+'-capture.bin'))
 start=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as stdout,(O/(name+'.stderr')).open('xb') as stderr:
  try:code=subprocess.run([str(x) for x in cmd],env=local,stdout=stdout,stderr=stderr,timeout=180).returncode
  except subprocess.TimeoutExpired:code=None
 execution.append(dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start));write(A/'integration-execution.json',execution);print(json.dumps(execution[-1]),flush=True)
 assert code in ([0] if required else [0,1])
 assert any(line.rstrip().endswith(str(B/'runtime/libmetalrobo.dylib')) for line in (O/(name+'.stderr')).read_text().splitlines())
 assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<plan['limits']['raw_bytes']
def physicalTSV(p):
 rows=list(csv.DictReader(p.open(),delimiter='\t'))
 return [{k:v for k,v in r.items() if k!='gpu_ms'} for r in rows]
motor=physicalTSV(O/'motor.tsv');zero=physicalTSV(O/'zero.tsv')
assert motor==physicalTSV(P/'motor-gold.tsv') and zero==physicalTSV(P/'zero-gold.tsv')
default=sum((O/'default'/p.name).read_bytes()==p.read_bytes() for p in (P/'default-gold').iterdir())
gold=(P/'strict-gold.bin').read_bytes();strict=(O/'strict-capture.bin').read_bytes()
diff=[i for i,(x,y) in enumerate(zip(gold,strict)) if x!=y];strictmatch=len(gold)==len(strict) and set(diff).issubset(set(range(88,92)))
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
obs=dict(owning_regressions_pass=True,motor_physical_records_exact=len(motor)+len(zero),default_physical_files_exact=default,
 strict_capture_diff_only_hessian_stride=strictmatch,api_default_complete_640_twice=all(r['summary'] and r['summary']['accepted_transitions']==640 and not r['failure'] and not r['terminal'] for r in cases),
 api_default_penetration_max_m=max([r['max_sampled_penetration_m'] for r in cases if r['max_sampled_penetration_m'] is not None],default=None),
 repeated_api_physical_outputs_exact=cases[0]['physical_files']==cases[1]['physical_files'])
passed={k:(v==plan['predictions'][k] if k!='api_default_penetration_max_m' else v is not None and v<=plan['predictions'][k]) for k,v in obs.items()}
write(A/'integration-analysis.json',dict(cases=cases,strict_capture_different_offsets=diff))
write(A/'integration-result.json',dict(observations=obs,predictions_supported=passed,all_predictions_supported=all(passed.values()),native_commands=len(execution),goal_complete=False,policy_or_profile_promoted=False))
print((A/'integration-result.json').read_text(),flush=True)
