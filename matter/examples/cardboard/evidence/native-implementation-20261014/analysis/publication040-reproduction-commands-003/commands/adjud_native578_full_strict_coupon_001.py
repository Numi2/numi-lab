from pathlib import Path
import csv,json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native578-full-strict-coupon-adjudication-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=q/'solver/fresh-source210-strict-inner-full-coupon-545';new=q/'solver/strict-state-source232-full-coupon-578'
m0=json.loads((old/'output/manifest.json').read_text());m1=json.loads((new/'output/manifest.json').read_text())
diffs=[]
def diff(a,b,path=''):
 if type(a)!=type(b):diffs.append({'path':path,'before':a,'after':b})
 elif isinstance(a,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:diffs.append({'path':path+'/'+k,'before':a.get(k),'after':b.get(k)})
   else:diff(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b)
  for i,(x,y) in enumerate(zip(a,b)):diff(x,y,path+'/'+str(i))
 elif a!=b:diffs.append({'path':path,'before':a,'after':b})
diff(m0,m1)
allowed={'/compensated_source_paper_projection/compiled_world_fingerprint','/compensated_source_paper_projection/effective_compiled_flags/liner_flags','/compensated_source_paper_projection/effective_compiled_flags/medium_flags','/compensated_source_paper_projection/source_map_sha256','/compensated_source_paper_projection/strict_scaled_correction_threshold','/compensated_source_paper_projection/strict_state_certification','/compiled_world_fingerprint','/materials/regional_map_sha256'}
assert {x['path'] for x in diffs}==allowed
receipt=json.loads((new/'receipt.json').read_text());assert receipt['execution_started'] and receipt['source_inputs_unchanged'] and receipt['exit_code']==2 and not receipt['timed_out']
result=json.loads((new/'output/result.json').read_text());assert result['status']=='failed' and result['accepted_steps']==19
rows=list(csv.DictReader((new/'output/observations.csv').open()));a=[x for x in rows if x['environment']=='0'];b=[x for x in rows if x['environment']=='1']
assert len(a)==len(b)==20
assert [int(x['step']) for x in a]==list(range(20))
assert all(x['step_accepted']=='1' for x in a[:19]) and a[-1]['step_accepted']=='0'
assert all(x['step_accepted']=='1' for x in b)
assert float(m1['solver']['relative_residual_tolerance'])==5e-7
observed_phases=sorted({x['phase'] for x in a})
comparisons=[];identical=0
for p in sorted((new/'output').glob('*.obj')):
 o=old/'output'/p.name
 if o.exists():
  eq=sha(p)==sha(o);identical+=int(eq);comparisons.append({'name':p.name,'identical':eq,'old_sha256':sha(o),'new_sha256':sha(p)})
traces={};final_cycles={}
with (new/'stderr.log').open() as f:
 for line in f:
  if not line.startswith('fem_true_residual_cycle='):continue
  j=json.loads(line.split('=',1)[1])
  if j['root']!=19:continue
  n=j['newton_iteration'];e=j['environments'][0]
  if j['restart_cycle']==0:traces[n]=e['rhs_l2']
  if j['final_cycle']:final_cycles[n]={'true_l2':e['true_l2'],'rhs_l2':e['rhs_l2'],'diagnostics':e['diagnostics']}
assert len(traces)==64 and sorted(traces)==list(range(64))
last=final_cycles[63]
r=float(a[-1]['certificate_residual'])
assert r>5e-7 and r<1.019589e-6
manifest=q/'snapshots/source-strict-local-projection-upstream-232/frozen-source.json'
bound=[old/'receipt.json',old/'output/manifest.json',old/'output/observations.csv',new/'receipt.json',new/'output/manifest.json',new/'output/observations.csv',new/'output/result.json',new/'stdout.log',new/'stderr.log',manifest,q/'build-source232-executed-bound-003/source-build.json']
report={'schema':'numi.cardboard.native578-full-strict-coupon-adjudication.v1','verdict':'FAIL_COMPLETE_HISTORY_NOT_REACHED','native_receipt':receipt,'accepted_active_controls':19,'first_rejected_control':19,'requested_controls':60,'observed_phases':observed_phases,'strict_policy_change_only_in_authored_manifest':diffs,'materials_recipe_geometry_mesh_loading_dt_outer_and_inner_controls_unchanged':True,'outer_residual_required':5e-7,'outer_residual_observed':r,'baseline545_outer_residual':float(list(csv.DictReader((old/'output/observations.csv').open()))[-2]['certificate_residual']),'final_active_observation':a[-1],'final_held_reference_observation':b[-1],'root19_first_cycle_rhs_l2_by_newton':traces,'last_inner_cycle':last,'last_inner_absolute_residual_below_1e_minus8':last['true_l2']<1e-8,'common_geometry_files':comparisons,'common_geometry_identical':identical,'common_geometry_count':len(comparisons),'strict_policy_is_numerical_update_change_not_observer_parity_test':True,'newton63_residual_still_above_unchanged_gate':True,'accepted_prefix_not_convergence_evidence':True,'all_six_full_sequence_axes_open':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in bound},'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'accepted':19,'rejected':19,'R':r,'final_inner_R':last,'geometry_identical':identical,'geometry_compared':len(comparisons)}))
