#!/usr/bin/env python3
import hashlib,json,math,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
repo=Path('/Users/numi/numi-human-aortic-contact-opt-20261019'); here=repo/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-073'
basepath=repo/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-performance-audit-063/inputs/062/invocation.json'
base=json.loads(basepath.read_text()); argv=list(base['argv']); binary=repo/'build-integrated-validation-20261019/profile-build-039/bin/numi-human-native'; argv[0]=str(binary); outdir=here/'run'/'output'; argv[4]=str(outdir)
env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_','MTL_'))}; env.update(base.get('environment',base.get('simulation_environment',{})))
run=here/'run'; run.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
inv={'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'binary_sha256':sha(binary),'argv':argv,'cwd':str(run),'input_invocation_sha256':sha(basepath),'working_source_sha256':{'runtime.mm':sha(repo/'matter/src/runtime.mm'),'fgmres.metalinc':sha(repo/'matter/src/metal/fgmres.metalinc')},'working_tree_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','--','matter/src/runtime.mm','matter/src/metal/fgmres.metalinc'],cwd=repo)).hexdigest(),'simulation_environment':{k:v for k,v in sorted(env.items()) if k.startswith(('NUMI_','NM_','MTL_'))},'started_utc':datetime.now(timezone.utc).isoformat(),'treatment':'empty_field_skip_plus_gpu_fgmres_indirect_and_dynamic_width_orthogonalization'}
(run/'invocation.json').write_text(json.dumps(inv,indent=2,sort_keys=True)+'\n')
start=time.perf_counter(); p=subprocess.run(argv,cwd=run,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE); wall=time.perf_counter()-start
(run/'stdout.log').write_text(p.stdout);(run/'stderr.log').write_text(p.stderr)
metrics=['root_xyz_m','center_of_mass_xyz_m','root_linear_velocity_xyz_m_s','root_linear_speed_m_s','max_generalized_acceleration','penetration_m','contact_count','min_J','max_J','max_displacement_m','max_anchor_reaction_N','passive_ligament_max_tension_N','passive_ligament_force_residual_N','passive_ligament_moment_residual_Nm','body_contact_normal_force_N','body_contact_force_residual_N','body_contact_moment_residual_Nm']
m={}
for line in p.stdout.splitlines():
 if not any(t in line for t in ('human_standing_progress=accepted','resting_FEM_terminal_audit=','resting_fascia_fem_terminal_audit=')):continue
 for k,v in re.findall(r'(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)',line):
  if k not in metrics:continue
  try:
   x=[float(z) for z in v[1:-1].split(',')] if v.startswith('[') else float(v)
   if all(math.isfinite(z) for z in (x if isinstance(x,list) else [x])):m[k]=x
  except ValueError:pass
ends=[int(x) for x in re.findall(r'human_execution_stage=authoritative_segment_end[^\n]*stage_step=(\d+)',p.stdout)]
bodyrows=re.findall(r'resting_integrated_body=completed([^\n]*)',p.stdout); body=dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',bodyrows[-1])) if bodyrows else {}
profiles=[{'step':int(a),'command_ms':float(b),'gpu_ms':float(c),'host_copy_ms':float(d)} for a,b,c,d in re.findall(r'resting_native_profile step=(\d+) command_ms=([^\s]+) gpu_ms=([^\s]+) host_copy_ms=([^\s]+)',p.stdout)]
counts=[{'step':int(a),'skipped_dispatches':int(b),'multiphysics_objects':int(c),'field_boundaries':int(d)} for a,b,c,d in re.findall(r'matter_mixed_field_smoother_dispatch_count=skipped step=(\d+) count=(\d+) multiphysics_objects=(\d+) field_boundaries=(\d+)',p.stderr)]
indirect=[{'step':int(a),'node':int(b),'element':int(c),'environment_count':int(d)} for a,b,c,d in re.findall(r'matter_fgmres_gpu_indirect_dispatch_count=encoded step=(\d+) node=(\d+) element=(\d+) environment_count=(\d+)',p.stderr)]
s={'returncode':p.returncode,'process_wall_seconds':wall,'authoritative_root_steps':ends,'body':body,'root_profiles':profiles,'skipped_dispatch_counts':counts,'fgmres_indirect_dispatch_counts':indirect,'fem_terminal_passed':'resting_FEM_terminal_audit=passed' in p.stdout,'fascia_terminal_passed':'resting_fascia_fem_terminal_audit=passed' in p.stdout,'root_assistance_false':'root_assistance=false' in p.stdout and not re.search(r'root_assistance_(?:force_n|torque_nm)=(?!0(?:\.0+)?(?:\s|$))',p.stdout),'terminal_metrics':m,'stdout_sha256':sha(run/'stdout.log'),'stderr_sha256':sha(run/'stderr.log'),'finished_utc':datetime.now(timezone.utc).isoformat()}
s['valid']=(p.returncode==0 and ends==[1,2,3,4] and abs(float(body.get('simulated_s','nan'))-0.0005)<1e-8 and len(profiles)==4 and len(counts)>=1 and all(x['multiphysics_objects']==0 and x['field_boundaries']==0 for x in counts) and len(indirect)>=1 and all(x['environment_count']==1 and (x['node']+x['element'])>0 for x in indirect) and s['fem_terminal_passed'] and s['fascia_terminal_passed'] and s['root_assistance_false'])
(run/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
baseline_root=repo/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-064/runs'
def metrics_from(path):
 text=(path/'stdout.log').read_text(); ans={}
 for line in text.splitlines():
  if not any(t in line for t in ('human_standing_progress=accepted','resting_FEM_terminal_audit=','resting_fascia_fem_terminal_audit=')):continue
  for k,v in re.findall(r'(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)',line):
   if k in metrics:
    try:ans[k]=[float(z) for z in v[1:-1].split(',')] if v.startswith('[') else float(v)
    except ValueError:pass
 return ans
comparisons=[]
for label in ('control-1','control-2'):
 try: old=metrics_from(baseline_root/label)
 except Exception:old={}
 checks={}; passed=True
 for k in metrics:
  if k not in old or k not in m:passed=False;checks[k]='missing';continue
  x=old[k] if isinstance(old[k],list) else [old[k]]; y=m[k] if isinstance(m[k],list) else [m[k]]
  if len(x)!=len(y):passed=False;checks[k]='length_mismatch';continue
  checks[k]=[]
  for a,b in zip(x,y):
   tol=max(1e-7,0.01*max(abs(a),abs(b))); ok=abs(a-b)<=tol; passed &= ok; checks[k].append({'control':a,'treatment':b,'absolute_delta':abs(a-b),'allowed':tol,'passed':ok})
 comparisons.append({'baseline':label,'baseline_valid':bool(json.loads((baseline_root/label/'amended-summary.json').read_text()).get('valid')),'metrics_within_tolerance':passed,'metrics':checks})
report={'schema':'numi.matter.fgmres-gpu-indirect-measurement-073.v1','treatment':s,'baseline_comparisons':comparisons,'skipped_dispatches_total':sum(x['skipped_dispatches'] for x in counts),'fgmres_indirect_dispatch_totals':{'node':sum(x['node'] for x in indirect),'element':sum(x['element'] for x in indirect)},'baseline_process_wall_seconds':[json.loads((baseline_root/l/'amended-summary.json').read_text())['process_wall_seconds'] for l in ('control-1','control-2')],'baseline_body_rtf':[float(json.loads((baseline_root/l/'amended-summary.json').read_text())['body']['real_time_factor']) for l in ('control-1','control-2')],'interpretation':'before/after same-host short mechanics-prefix replay; not sustained full Human qualification'}
(here/'analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('RESULT',json.dumps({'valid':s['valid'],'process_wall_seconds':wall,'body_rtf':body.get('real_time_factor'),'gpu_ms_per_root':[round(x['gpu_ms'],3) for x in profiles],'accepted_roots':ends,'skipped_dispatches':sum(x['skipped_dispatches'] for x in counts),'skipped_step_counts':counts,'fgmres_indirect_dispatch_totals':report['fgmres_indirect_dispatch_totals'],'baseline_wall_s':report['baseline_process_wall_seconds'],'baseline_rtf':report['baseline_body_rtf'],'physics_comparisons':[x['metrics_within_tolerance'] for x in comparisons]},sort_keys=True),flush=True)
if not s['valid']: raise SystemExit('native acceptance or empty-field premise failed')
