from pathlib import Path
import json,hashlib,csv,datetime,subprocess,os,collections
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');c=q/'solver/primary-cache-strict-control19-final127-diagnostic-602';base=q/'solver/strict-state-source240-newton128-full-coupon-585';old=q/'solver/strict-state-source240-control19-final-newton-diagnostic-596';s=q/'snapshots/source-legacy-program-primary-cache-observer-256-002';d=q/'analysis/native602-corrected-actual-cache-adjudication-002';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((c/'receipt.json').read_text());assert r['exit_code']==2 and r['source_inputs_unchanged'] and not r['timed_out']
rows=lambda p:list(csv.DictReader(p.open()));a=[x for x in rows(base/'output/observations.csv') if int(x['step'])>=16];b=rows(c/'output/observations.csv');assert a==b and len(b)==8
inputs=[Path(__file__),c/'receipt.json',base/'receipt.json',old/'receipt.json'];same=[]
for p in sorted((c/'output').iterdir()):
 if p.is_file() and (p.name.startswith(('accepted_step_','rejected_step_','material_state_')) or p.name in ['compiled.nmatterpack','mesh.json','manifest.json']):
  other=base/'output'/p.name;assert sha(p)==sha(other),p.name;same.append(p.name);inputs.extend([p,other])
bins=[]
for p in sorted((c/'output/rejected-conditioning-000020').glob('*.bin')):
 other=base/'output/rejected-conditioning-000020'/p.name;assert sha(p)==sha(other),p.name;bins.append(p.name);inputs.extend([p,other])
cap=c/'diagnostics/capture-operator-fd-capture/capture';ocap=old/'diagnostics/capture-operator-fd-capture/capture';counts=collections.Counter();removed=collections.Counter();action_changed=0;policy=None
with cap.open() as nf,ocap.open() as of:
 for ln,(nl,ol) in enumerate(zip(nf,of)):
  n=json.loads(nl);o=json.loads(ol);assert n['record']==o['record'];kind=n['record'];counts[kind]+=1
  if kind=='header':
   policy=n.pop('operator_action_policy');assert policy['cache_mode']==1 and policy['selection']=='matches_primary_FGMRES_cache_mode';removed['header.operator_action_policy']+=1
  elif kind=='node':
   key='production_A_delta_v_impulse_Ns'
   if n[key]!=o[key]:action_changed+=1
   del n[key];del o[key];removed['node.'+key]+=1
  elif kind=='tetrahedron':
   key='operator_force_node_world_N_per_s';del n['center'][key];del o['center'][key];removed['tetrahedron.center.'+key]+=1
  assert json.dumps(n,sort_keys=True,allow_nan=True)==json.dumps(o,sort_keys=True,allow_nan=True),(ln,kind)
 assert not nf.readline() and not of.readline()
assert counts['node']==1764 and counts['tetrahedron']==3840 and policy
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'}
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_cached_operator_audit.py'),'--capture',str(cap),'--prediction',str(c/'prediction.json'),'--receipt',str(c/'receipt.json'),'--stderr',str(c/'stderr.log'),'--source-snapshot',str(s),'--environment','0','--output',str(d/'operator.json')]
with (d/'operator.stdout').open('xb') as out,(d/'operator.stderr').open('xb') as err:
 proc=subprocess.run(argv,env=env,stdout=out,stderr=err)
assert proc.returncode==0,proc.returncode
audit=json.loads((d/'operator.json').read_text());act=audit['operator']['action_parity']
report={'schema':'numi.cardboard.native602-corrected-actual-cache-adjudication.v1','verdict':'ACTUAL_FAILING_STATE_OBSERVER_PARITY_AND_CACHED_ACTION_MEASURED','production_suffix_observation_rows_exact':len(b),'production_files_exact':same,'rejected_conditioning_bins_exact':bins,'basis_compared_to596':{'captured_record_counts':dict(counts),'excluded_action_fields':dict(removed),'all_other_captured_values_identical':True,'nodes_with_changed_action':action_changed,'scope':'Header action metadata and cached rather than uncached nodal/element actions are the only permitted capture differences. Candidate states, correction, masses, material frames, accepted/projected histories, perturbations, direct local tangents and internal forces are identical.'},'operator_action_policy':policy,'cached_action_parity':act,'measured_relative_below_one_percent':act['relative_l2_error']<=.01,'empirical_FP32_noise_floor_qualified':False,'full_solver_or_convergence_qualified':False,'native_target_still_failed':True,'capture_sha256':sha(cap),'cached_operator_audit_sha256':sha(d/'operator.json'),'CPU_audit_argv':argv,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');inputs.extend([cap,ocap,s/'frozen-source.json',c/'output/observations.csv',base/'output/observations.csv'])
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(q)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},'artifacts':{p.name:sha(p) for p in d.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'records_identical_except_action':dict(counts),'cached_action_parity':act,'full_solver_qualified':False}),flush=True)
