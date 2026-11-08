from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();B=R/'build/crow-bounded-drive-integration-build-01';A=R/'artifacts/crow-rl-bounded-drive-crow-20261108/precision';T=R/'.numi/retained-runtime/bounded-drive-precision-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def length(v):return sum(x*x for x in v)**.5
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert sha(R/p)==h,p
assert not subprocess.check_output(['git','status','--porcelain'])
seal=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
commands=[]
def execute(name,cmd):
 env['METALROBO_BOUNDED_FAILURE_CAPTURE']=str(B/(name+'.motorsnp'));env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'.ccdsweep'))
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(p) for p in cmd],stdout=out,stderr=err,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:code=None
 rec={'name':name,'command':[str(p) for p in cmd],'exit_code':code,'seconds':time.monotonic()-start,'sealed_head':seal};commands.append(rec);write(B/'precision-execution.json',commands);print(json.dumps(rec),flush=True)
 return code
report={'sealed_head':seal,'commands':commands,'policy_or_profile_promoted':False,'full_crow_qualified':False,'runs':{},'component_checks':{},'failure':None}
code=execute('precision-constitutive',[T/'probe',T/'metalrobo/MetalRobo.metallib',B/'precision-constitutive.tsv'])
if code==0:
 rows=list(csv.DictReader((B/'precision-constitutive.tsv').open(),delimiter='\t'));assert len(rows)==976
 old=list(csv.DictReader((B/'native-v2.tsv').open(),delimiter='\t'))
 report['legacy448rows_byte_values_exact']=[r for r in rows if r['bounded']=='0']==[r for r in old if r['bounded']=='0']
 for mode in (0,1):
  r=[{k:float(v) for k,v in q.items()} for q in rows if int(q['bounded'])==mode]
  m={'rows':len(r),'maximum_velocity_reference_error_rad_s':max(abs(q['final_v']-q['reference_v']) for q in r),'maximum_q_integration_error_rad':max(abs(q['final_q']-q['q']-q['h']*q['final_v']) for q in r),'maximum_torque_error_Nm':max(abs(q['motor_torque']-q['reference_torque']) for q in r),'mass_flags_match':all(q['aba_implicit']==1-mode and q['contact_mass_implicit']==1-mode for q in r)}
  m['pass']=m['maximum_velocity_reference_error_rad_s']<=2e-6 and m['maximum_q_integration_error_rad']<=2e-7 and m['maximum_torque_error_Nm']<=1e-7 and m['mass_flags_match'];report['component_checks'][str(mode)]=m
else:report['failure']=commands[-1]
for name,helper,sub in [('precision-crow32','fixture-bounded',32),('precision-unused64','fixture-unused',64),('precision-unused128','fixture-unused',128)]:
 if report['failure']:break
 if execute(name,[B/helper,B/name,'0','1',str(sub)])!=0:
  report['failure']=dict(commands[-1]);p=B/name/'failure.json'
  if p.exists():report['failure']['native_failure']=json.loads(p.read_text())
  break
 d=B/name;s=json.loads((d/'summary.json').read_text());assert s['accepted_transitions']==640 and s['physics_rejections']==0
 q=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((d/'trace.tsv').open(),delimiter='\t')];assert len(q)==640 and {(int(r['tick']),int(r['lane'])) for r in q}=={(t,l) for t in range(160) for l in range(4)}
 initial=json.loads((d/'initial-geometry.json').read_text());lanes=[]
 for lane in range(4):
  v=[r for r in q if r['lane']==lane];hold=[r for r in v if 64<=r['tick']<=95];release=[r for r in v if r['tick']>=128];anchor=next(r for r in v if r['tick']==31)
  m={'minimum_lift_m':min(r['z']-initial[lane]['position'][2] for r in hold),'maximum_slip_m':max(length(r['rel_'+a]-anchor['rel_'+a] for a in 'xyz') for r in hold),'opposed_samples':sum(r['upper_Jn']>1e-9 and r['lower_Jn']>1e-9 for r in hold),'maximum_penetration_before96_m':max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in v if r['tick']<96)),'release_drop_m':min(r['z'] for r in hold)-v[-1]['z'],'minimum_release_gap_m':min(min(r['upper_gap_m'],r['lower_gap_m']) for r in release)}
  m['mechanical_gates_pass']=m['minimum_lift_m']>=.02 and m['maximum_slip_m']<=.005 and m['opposed_samples']>=16 and m['maximum_penetration_before96_m']<=.0008 and m['release_drop_m']>=.03 and m['minimum_release_gap_m']>=.001;lanes.append(m)
 timing=json.loads((d/'runtime.json').read_text());assert timing['device']=='Apple M4 Pro'
 report['runs'][name]={'accepted_transitions':640,'lanes':lanes,'all_mechanical_pass':all(m['mechanical_gates_pass'] for m in lanes),'runtime':timing}
 if name=='precision-crow32':
  pen=max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in q if r['tick']<=20));report['exposed_prefix_penetration_m']=pen;report['precision_alone_does_not_remove_contact_penetration']=.0015<=pen<=.0025
if {'precision-unused64','precision-unused128'}.issubset(report['runs']):
 def trace(name):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'trace.tsv').open(),delimiter='\t')]
 coarse=trace('precision-unused64');fine=trace('precision-unused128');assert all((c['tick'],c['lane'])==(f['tick'],f['lane']) for c,f in zip(coarse,fine))
 error=[length(c[a]-f[a] for a in 'xyz') for c,f in zip(coarse,fine)]
 report['unused_pose_64vs128']={'maximum_position_error_m':max(error),'pass_5mm':max(error)<=.005,'phase_errors_m':{f'{lo}:{hi}':max(e for e,r in zip(error,coarse) if lo<=r['tick']<=hi) for lo,hi in ((0,31),(32,63),(64,95),(96,127),(128,159))}}
for rec in commands:
 loads=[s for s in (B/(rec['name']+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];assert loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads);rec['loader_lines']=loads
report['all_native_commands_completed']=len(commands)==4 and report['failure'] is None
report['component_gates_pass']=len(report['component_checks'])==2 and all(m['pass'] for m in report['component_checks'].values())
report['completed_native_steps']=514*(commands[0]['exit_code']==0)+sum(r['accepted_transitions'] for r in report['runs'].values())
report['verdict']='inconclusive_native_failure' if report['failure'] else ('native_precision_prediction_supported' if report['component_gates_pass'] and report['all_native_commands_completed'] and report.get('precision_alone_does_not_remove_contact_penetration') else 'native_precision_prediction_contradicted')
write(A/'results.json',report);write(B/'precision-analysis.json',report);print(json.dumps(report),flush=True)
