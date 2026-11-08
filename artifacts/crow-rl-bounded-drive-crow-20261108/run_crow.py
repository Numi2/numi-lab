from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd(); B=R/'build/crow-bounded-drive-integration-build-01'; A=R/'artifacts/crow-rl-bounded-drive-crow-20261108'; T=R/'.numi/retained-runtime/bounded-drive-integration-v2-20261108'
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert sha(R/p)==h,p
assert not subprocess.check_output(['git','status','--porcelain'])
seal=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
records=[];failure=None
for trial in plan['execution_order']:
 name=trial['name']; command=[str(B/trial['executable']),str(B/name),str(trial['arm']),str(trial['chunk']),str(trial['substeps'])]
 env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'.ccdsweep'))
 start=time.monotonic();timedout=False
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run(command,stdout=out,stderr=err,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:code=None;timedout=True
 record=dict(trial,command=command,exit_code=code,timedout=timedout,seconds=time.monotonic()-start,sealed_head=seal)
 records.append(record);write(B/'crow-execution.json',records);print(json.dumps(record),flush=True)
 if code!=0:
  failure=dict(record)
  if (B/name/'failure.json').exists():failure['native_failure']=json.loads((B/name/'failure.json').read_text())
  else:failure['stderr_tail']=(B/(name+'.stderr')).read_text()[-4000:]
  break
report={'sealed_head':seal,'commands':records,'failure':failure,'unrun':[t['name'] for t in plan['execution_order'][len(records):]],'full_crow_qualified':False,'policy_or_profile_promoted':False,'runs':{},'convergence':{},'chunk_equivalence':{},'limitations':plan['limitations']}
def rows(name):
 result=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'trace.tsv').open(),delimiter='\t')]
 assert len(result)==640 and {(int(r['tick']),int(r['lane'])) for r in result}=={(t,l) for t in range(160) for l in range(4)}
 return result
def length(x):return sum(v*v for v in x)**.5
for rec in records:
 if rec['exit_code']!=0:continue
 name=rec['name'];d=B/name;s=json.loads((d/'summary.json').read_text());timing=json.loads((d/'runtime.json').read_text())
 assert s['accepted_transitions']==640 and s['physics_rejections']==0 and timing['device']=='Apple M4 Pro'
 loader=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 assert loader and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loader)
 metrics={'accepted_transitions':640,'runtime':timing,'loader_lines':loader}
 if rec['chunk']==1:
  q=rows(name);initial=json.loads((d/'initial-geometry.json').read_text());lanes=[]
  for lane in range(4):
   v=[r for r in q if r['lane']==lane];hold=[r for r in v if 64<=r['tick']<=95];release=[r for r in v if r['tick']>=128];anchor=next(r for r in v if r['tick']==31)
   m={'minimum_lift_m':min(r['z']-initial[lane]['position'][2] for r in hold),'maximum_slip_m':max(length(r['rel_'+a]-anchor['rel_'+a] for a in 'xyz') for r in hold),'opposed_samples':sum(r['upper_Jn']>1e-9 and r['lower_Jn']>1e-9 for r in hold),'maximum_penetration_before96_m':max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in v if r['tick']<96)),'release_drop_m':min(r['z'] for r in hold)-v[-1]['z'],'minimum_release_gap_m':min(min(r['upper_gap_m'],r['lower_gap_m']) for r in release)}
   m['hold_pass']=m['minimum_lift_m']>=.02 and m['maximum_slip_m']<=.005 and m['opposed_samples']>=16
   m['mechanical_gates_pass']=m['hold_pass'] and m['maximum_penetration_before96_m']<=.0008 and m['release_drop_m']>=.03 and m['minimum_release_gap_m']>=.001
   lanes.append(m)
  metrics.update(lanes=lanes,all_mechanical_gates_pass=all(m['mechanical_gates_pass'] for m in lanes),all_fail_hold=all(not m['hold_pass'] for m in lanes))
 report['runs'][name]=metrics
for a,b in [('crow-bounded32','crow-bounded64'),('crow-bounded64','crow-bounded128'),('crow-unused64','crow-unused128')]:
 if a not in report['runs'] or b not in report['runs']:continue
 q=rows(a);z=rows(b);assert all((c['tick'],c['lane'])==(f['tick'],f['lane']) for c,f in zip(q,z))
 errors=[length(c[k]-f[k] for k in 'xyz') for c,f in zip(q,z)]
 phases={f'{lo}:{hi}':max(e for e,r in zip(errors,q) if lo<=r['tick']<=hi) for lo,hi in ((0,31),(32,63),(64,95),(96,127),(128,159))}
 report['convergence'][a+'__'+b]={'maximum_position_error_m':max(errors),'phase_errors_m':phases,'pass_5mm':max(errors)<=.005}
if {'crow-bounded32','crow-chunk8'}.issubset(report['runs']):
 report['chunk_equivalence']={f:(B/'crow-bounded32'/f).read_bytes()==(B/'crow-chunk8'/f).read_bytes() for f in ('final-q.f32','final-v.f32','final-scene.bin')}
report['all_declared_commands_completed']=len(records)==len(plan['execution_order']) and failure is None
report['accepted_control_transitions']=sum(v['accepted_transitions'] for v in report['runs'].values())
positive=['crow-bounded32','crow-bounded64','crow-bounded128','crow-unused64','crow-unused128'];negative=['crow-open','crow-nocontact','crow-frictionless']
report['all_positive_mechanical_pass']=all(n in report['runs'] and report['runs'][n]['all_mechanical_gates_pass'] for n in positive)
report['all_negative_fail_hold']=all(n in report['runs'] and report['runs'][n]['all_fail_hold'] for n in negative)
report['all_5mm_convergence_pass']=len(report['convergence'])==3 and all(x['pass_5mm'] for x in report['convergence'].values())
report['final_state_chunk_exact']=len(report['chunk_equivalence'])==3 and all(report['chunk_equivalence'].values())
report['verdict']='inconclusive_native_failure' if failure else ('bounded_drive_crow_fixture_prediction_supported' if all(report[k] for k in ('all_declared_commands_completed','all_positive_mechanical_pass','all_negative_fail_hold','all_5mm_convergence_pass','final_state_chunk_exact')) else 'bounded_drive_crow_fixture_prediction_contradicted')
write(A/'results.json',report);write(B/'crow-analysis.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('commands','runs')}),flush=True)
