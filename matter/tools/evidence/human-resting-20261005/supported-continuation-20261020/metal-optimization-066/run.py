#!/usr/bin/env python3
import hashlib, json, math, os, re, statistics, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

repo = Path('/Users/numi/numi-human-aortic-contact-opt-20261019')
here = repo / 'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-066'
base_path = repo / 'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-performance-audit-063/inputs/062/invocation.json'
base = json.loads(base_path.read_text())
expected_head = 'aadcd68e0747540978edd88a23af0cd114932942'
expected_binary = 'c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828'
metrics = ['root_xyz_m','center_of_mass_xyz_m','root_linear_velocity_xyz_m_s','root_linear_speed_m_s','max_generalized_acceleration','penetration_m','contact_count','min_J','max_J','max_displacement_m','max_anchor_reaction_N','passive_ligament_max_tension_N','passive_ligament_force_residual_N','passive_ligament_moment_residual_Nm','body_contact_normal_force_N','body_contact_force_residual_N','body_contact_moment_residual_Nm']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def parse_metrics(s):
 out={}
 for line in s.splitlines():
  if not any(x in line for x in ('human_standing_progress=accepted','resting_FEM_terminal_audit=','resting_fascia_fem_terminal_audit=')): continue
  for k,v in re.findall(r'(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)',line):
   if k not in metrics: continue
   try:
    z=[float(x) for x in v[1:-1].split(',')] if v.startswith('[') else float(v)
    if all(math.isfinite(x) for x in (z if isinstance(z,list) else [z])): out[k]=z
   except ValueError: pass
 return out
def run(label, block):
 d=here/'runs'/label; d.mkdir(parents=True,exist_ok=True)
 argv=list(base['argv']); argv[4]=str(d/'output')
 argv[argv.index('--muscle-step-count')+1]='1'
 env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_','MTL_'))}
 env.update(base.get('environment', base.get('simulation_environment', {}))); env.pop('NUMI_HUMAN_RESTING_COUPLED_TANGENT_BLOCK',None)
 if block: env['NUMI_HUMAN_RESTING_COUPLED_TANGENT_BLOCK']='1'
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=expected_head or sha(Path(argv[0]))!=expected_binary: raise RuntimeError('frozen source or binary identity changed')
 inv={'label':label,'treatment':'coupled_tangent_block' if block else 'control','source_head':expected_head,'binary_sha256':sha(Path(argv[0])),'input_invocation_sha256':sha(base_path),'argv':argv,'cwd':str(d),'simulation_environment':{k:v for k,v in sorted(env.items()) if k.startswith(('NUMI_','NM_','MTL_'))},'started_utc':datetime.now(timezone.utc).isoformat()}
 (d/'invocation.json').write_text(json.dumps(inv,indent=2,sort_keys=True)+'\n')
 old=json.loads((d/'summary.json').read_text()) if (d/'stdout.log').exists() and (d/'summary.json').exists() else None
 if old is not None:
  print('REUSE_RETAINED',label,flush=True); wall=float(old['process_wall_seconds']); stdout=(d/'stdout.log').read_text(); stderr=(d/'stderr.log').read_text(); returncode=int(old['returncode'])
 else:
  print('START',label,flush=True); start=time.perf_counter()
  p=subprocess.run(argv,cwd=d,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  wall=time.perf_counter()-start; stdout=p.stdout; stderr=p.stderr; returncode=p.returncode
  (d/'stdout.log').write_text(stdout); (d/'stderr.log').write_text(stderr)
 ends=[int(x) for x in re.findall(r'human_execution_stage=authoritative_segment_end[^\n]*stage_step=(\d+)',stdout)]
 bodyrows=re.findall(r'resting_integrated_body=completed([^\n]*)',stdout)
 body=dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',bodyrows[-1])) if bodyrows else {}
 profiles=[{'step':int(a),'command_ms':float(b),'gpu_ms':float(c),'host_copy_ms':float(e)} for a,b,c,e in re.findall(r'resting_native_profile step=(\d+) command_ms=([^\s]+) gpu_ms=([^\s]+) host_copy_ms=([^\s]+)',stdout)]
 s={'label':label,'returncode':returncode,'process_wall_seconds':wall,'authoritative_root_steps':ends,'body':body,'root_profiles':profiles,'terminal_metrics':parse_metrics(stdout),'fem_terminal_passed':'resting_FEM_terminal_audit=passed' in stdout,'fascia_terminal_passed':'resting_fascia_fem_terminal_audit=passed' in stdout,'root_assistance_false':'root_assistance=false' in stdout and not re.search(r'root_assistance_(?:force_n|torque_nm)=(?!0(?:\.0+)?(?:\s|$))',stdout),'stdout_sha256':sha(d/'stdout.log'),'stderr_sha256':sha(d/'stderr.log'),'finished_utc':datetime.now(timezone.utc).isoformat()}
 s['valid']=(returncode==0 and ends==[1] and abs(float(body.get('simulated_s','nan'))-0.000125)<1e-8 and len(profiles)==1 and s['fem_terminal_passed'] and s['fascia_terminal_passed'] and s['root_assistance_false'])
 (d/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
 print('END',label,'wall_s='+format(wall,'.3f'),'valid='+str(int(s['valid'])),'rtf='+str(body.get('real_time_factor')),'gpu_ms='+str([round(x['gpu_ms'],3) for x in profiles]),flush=True)
 if not s['valid']: raise RuntimeError('run failed physical acceptance; retained and stopped')
 return s
def compare(a,b):
 out={}
 for k in metrics:
  if k not in a or k not in b:return False,{'missing':k}
  x=a[k] if isinstance(a[k],list) else [a[k]]; y=b[k] if isinstance(b[k],list) else [b[k]]
  if len(x)!=len(y): return False,{'length_mismatch':k}
  out[k]=[]
  for i,j in zip(x,y):
   tol=max(1e-7,0.01*max(abs(i),abs(j))); out[k].append({'control':i,'treatment':j,'absolute_delta':abs(i-j),'allowed':tol})
   if abs(i-j)>tol:return False,out
 return True,out
results={}
for label,block in [('control-1',False),('block-1',True),('block-2',True),('control-2',False)]: results[label]=run(label,block)
pairs=[]
for c,b in [('control-1','block-1'),('control-2','block-2')]:
 ok,d=compare(results[c]['terminal_metrics'],results[b]['terminal_metrics']); cr=float(results[c]['body']['real_time_factor']); br=float(results[b]['body']['real_time_factor'])
 pairs.append({'control':c,'treatment':b,'control_rtf':cr,'treatment_rtf':br,'speedup_ratio':br/cr,'physical_metrics_within_tolerance':ok,'metric_comparison':d})
median=statistics.median(x['speedup_ratio'] for x in pairs); passed=all(x['speedup_ratio']>=1.15 and x['physical_metrics_within_tolerance'] for x in pairs)
analysis={'schema':'numi.matter.coupled-fem-tangent-block-screen-066.v1','pairs':pairs,'median_speedup_ratio':median,'promotion_screen_passed':passed,'interpretation':'one-root exploratory native mechanics-prefix comparison; not a sustained or physiological qualification'}
(here/'analysis.json').write_text(json.dumps(analysis,indent=2,sort_keys=True)+'\n')
print('ANALYSIS',json.dumps({'pair_speedups':[round(x['speedup_ratio'],4) for x in pairs],'median_speedup':round(median,4),'physical_checks':[x['physical_metrics_within_tolerance'] for x in pairs],'promotion_screen_passed':passed},sort_keys=True),flush=True)
