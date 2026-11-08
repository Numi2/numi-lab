from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();B=R/'build/crow-bounded-drive-integration-build-01';A=R/'artifacts/crow-rl-bounded-drive-crow-20261108/precision-v2';T=R/'.numi/retained-runtime/bounded-drive-precision-v2-20261108';BAD=R/'.numi/retained-runtime/bounded-drive-precision-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def length(v):return sum(x*x for x in v)**.5
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert sha(R/p)==h,p
assert not subprocess.check_output(['git','status','--porcelain'])
seal=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
report={'sealed_head':seal,'commands':[],'policy_or_profile_promoted':False,'full_crow_qualified':False}
def finish(verdict):
 report['verdict']=verdict;write(A/'results.json',report);write(B/'precision-v2-analysis.json',report);print(json.dumps(report),flush=True)
def execute(name,cmd):
 env['METALROBO_BOUNDED_FAILURE_CAPTURE']=str(B/(name+'.motorsnp'));env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'.ccdsweep'))
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(p) for p in cmd],stdout=out,stderr=err,env=env,timeout=300).returncode
  except subprocess.TimeoutExpired:code=None
 rec={'name':name,'command':[str(p) for p in cmd],'exit_code':code,'seconds':time.monotonic()-start}
 loads=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];assert loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads);rec['loader_lines']=loads
 report['commands'].append(rec);write(B/'precision-v2-execution.json',report['commands']);print(json.dumps(rec),flush=True);return code

def analyze_rows(path):
 rows=[{k:float(v) for k,v in q.items()} for q in csv.DictReader(path.open(),delimiter='\t')]
 metrics={'rows':len(rows),'maximum_velocity_error_rad_s':max(abs(q['final_v']-q['reference_v']) for q in rows),'maximum_q_integration_error_rad':max(abs(q['final_q']-q['q']-q['h']*q['final_v']) for q in rows),'maximum_torque_error_Nm':max(abs(q['motor_torque']-q['reference_torque']) for q in rows),'mass_flags_match':all(q['aba_implicit']==1-q['bounded'] and q['contact_mass_implicit']==1-q['bounded'] for q in rows)}
 metrics['pass']=metrics['maximum_velocity_error_rad_s']<=2e-6 and metrics['maximum_q_integration_error_rad']<=2e-7 and metrics['maximum_torque_error_Nm']<=1e-7 and metrics['mass_flags_match'];return metrics
code=execute('precision-v2-negative',[T/'probe',BAD/'metalrobo/MetalRobo.metallib',B/'precision-v2-negative.tsv'])
report['negative_instrument_control']=analyze_rows(B/'precision-v2-negative.tsv')
report['negative_instrument_control']['typed_reference_failure']='Native motor reference gate failed' in (B/'precision-v2-negative.stderr').read_text()
if code!=1 or report['negative_instrument_control']['pass'] or not report['negative_instrument_control']['typed_reference_failure']:
 finish('inconclusive_failure_detection_control');raise SystemExit(1)
code=execute('precision-v2-constitutive',[T/'probe',T/'metalrobo/MetalRobo.metallib',B/'precision-v2-constitutive.tsv'])
if code!=0:finish('numerical_revision_rejected_before_crow');raise SystemExit(1)
report['positive_component_control']=analyze_rows(B/'precision-v2-constitutive.tsv')
if report['positive_component_control']['rows']!=976 or not report['positive_component_control']['pass']:
 finish('numerical_revision_rejected_before_crow');raise SystemExit(1)
code=execute('precision-v2-crow32',[B/'fixture-bounded',B/'precision-v2-crow32','0','1','32'])
d=B/'precision-v2-crow32'
if code!=0:
 if (d/'failure.json').exists():report['native_failure']=json.loads((d/'failure.json').read_text())
 finish('inconclusive_crow_native_failure');raise SystemExit(1)
s=json.loads((d/'summary.json').read_text());assert s['accepted_transitions']==640 and s['physics_rejections']==0
q=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((d/'trace.tsv').open(),delimiter='\t')];assert len(q)==640 and {(int(r['tick']),int(r['lane'])) for r in q}=={(t,l) for t in range(160) for l in range(4)}
initial=json.loads((d/'initial-geometry.json').read_text());lanes=[]
for lane in range(4):
 v=[r for r in q if r['lane']==lane];hold=[r for r in v if 64<=r['tick']<=95];release=[r for r in v if r['tick']>=128];anchor=next(r for r in v if r['tick']==31)
 m={'minimum_lift_m':min(r['z']-initial[lane]['position'][2] for r in hold),'maximum_slip_m':max(length(r['rel_'+a]-anchor['rel_'+a] for a in 'xyz') for r in hold),'opposed_samples':sum(r['upper_Jn']>1e-9 and r['lower_Jn']>1e-9 for r in hold),'maximum_penetration_before96_m':max(0.,-min(min(r['upper_gap_m'],r['lower_gap_m']) for r in v if r['tick']<96)),'release_drop_m':min(r['z'] for r in hold)-v[-1]['z'],'minimum_release_gap_m':min(min(r['upper_gap_m'],r['lower_gap_m']) for r in release)}
 m['mechanical_gates_pass']=m['minimum_lift_m']>=.02 and m['maximum_slip_m']<=.005 and m['opposed_samples']>=16 and m['maximum_penetration_before96_m']<=.0008 and m['release_drop_m']>=.03 and m['minimum_release_gap_m']>=.001;lanes.append(m)
report['crow']={'accepted_transitions':640,'lanes':lanes,'all_mechanical_pass':all(m['mechanical_gates_pass'] for m in lanes),'runtime':json.loads((d/'runtime.json').read_text())}
finish('motor_precision_supported_crow_physics_unqualified' if not report['crow']['all_mechanical_pass'] else 'motor_precision_supported_single_crow_fixture_passed')
