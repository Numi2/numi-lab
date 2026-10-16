from pathlib import Path
import csv,json,hashlib,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native585-full-strict-coupon-adjudication-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
new=q/'solver/strict-state-source240-newton128-full-coupon-585';old=q/'solver/strict-state-source232-full-coupon-578'
receipt=json.loads((new/'receipt.json').read_text());result=json.loads((new/'output/result.json').read_text())
assert receipt['execution_started'] and receipt['source_inputs_unchanged'] and not receipt['timed_out'] and receipt['exit_code']==2
assert result['accepted_steps']==19 and result['status']=='failed'
rows=list(csv.DictReader((new/'output/observations.csv').open()));a=[x for x in rows if x['environment']=='0'];b=[x for x in rows if x['environment']=='1']
assert len(a)==len(b)==20 and all(x['step_accepted']=='1' for x in a[:19]) and a[-1]['step_accepted']=='0'
assert all(x['step_accepted']=='1' for x in b)
oldrows=list(csv.DictReader((old/'output/observations.csv').open()));oldprefix=[x for x in oldrows if int(x['step'])<19];prefix=[x for x in rows if int(x['step'])<19]
assert oldprefix==prefix
geometries=[]
for p in sorted((new/'output').glob('accepted*.obj')):
 o=old/'output'/p.name
 if o.exists():geometries.append({'path':p.name,'identical':sha(p)==sha(o),'sha256':sha(p),'baseline_sha256':sha(o)})
assert all(x['identical'] for x in geometries)
first={};last={};lastcycles=[]
for line in (new/'stderr.log').read_text().splitlines():
 if not line.startswith('fem_true_residual_cycle='):continue
 j=json.loads(line.split('=',1)[1])
 if j['root']!=19:continue
 i=j['newton_iteration'];e=j['environments'][0]
 if j['restart_cycle']==0:first[i]=e['rhs_l2']
 if j['final_cycle']:last[i]={'true_l2':e['true_l2'],'rhs_l2':e['rhs_l2'],'diagnostics':e['diagnostics'],'restart_cycle':j['restart_cycle']}
 if i==127:lastcycles.append({'restart':j['restart_cycle'],**e})
assert len(first)==128 and len(last)==128
m0=json.loads((old/'output/manifest.json').read_text());m1=json.loads((new/'output/manifest.json').read_text());diffs=[]
def diff(a,b,p=''):
 if type(a)!=type(b):diffs.append({'path':p,'before':a,'after':b})
 elif isinstance(a,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:diffs.append({'path':p+'/'+k,'before':a.get(k),'after':b.get(k)})
   else:diff(a[k],b[k],p+'/'+k)
 elif isinstance(a,list):
  if len(a)!=len(b):diffs.append({'path':p,'before':a,'after':b})
  else:
   for i,(x,y) in enumerate(zip(a,b)):diff(x,y,p+'/'+str(i))
 elif a!=b:diffs.append({'path':p,'before':a,'after':b})
diff(m0,m1)
bound=[new/'receipt.json',new/'output/result.json',new/'output/manifest.json',new/'output/observations.csv',new/'stderr.log',old/'output/manifest.json',old/'output/observations.csv',q/'snapshots/source-refined-regional-graph-240-003/frozen-source.json',q/'build-source240-refined-regional-graph-003/source-build.json']
r={'schema':'numi.cardboard.native585-full-strict-coupon-adjudication.v1','verdict':'FAIL_COMPLETE_HISTORY_NOT_REACHED','accepted_active_controls':19,'requested_controls':60,'first_rejected_control':19,'outer_residual_required':5e-7,'outer_residual_observed':float(a[-1]['certificate_residual']),'newton_budget':128,'restart_ceiling':16,'fgmres_budget':1024,'final_active_observation':a[-1],'common_accepted_observations_byte_value_identical':True,'common_accepted_observation_rows':len(prefix),'common_accepted_geometry':geometries,'manifest_differences':diffs,'root19_first_cycle_rhs_l2_by_newton':first,'root19_last_inner_cycles_by_newton':last,'final_newton127_restart_trace':lastcycles,'final_true_inner_over_rhs':last[127]['true_l2']/last[127]['rhs_l2'],'final_inner_stagnates_last_restarts':len({x['true_l2'] for x in lastcycles[-4:]})==1,'accepted_prefix_not_resolution_evidence':True,'all_six_full_sequence_axes_open':True,'limits':'R falls with more Newton work but complete60controlhistory still fails. Final inner true residual stagnates; neither tolerance relaxation nor blind work increase is justified by this run. Exact failed candidate replay requested separately.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in bound},'report_sha256':sha(d/'report.json')},indent=2)+'\n')
print(json.dumps({'report_sha256':sha(d/'report.json'),'R':r['outer_residual_observed'],'final_inner_relative':r['final_true_inner_over_rhs'],'prefix_exact':len(prefix),'geometry_exact':len(geometries),'manifest_diffs':[x['path'] for x in diffs]}))
