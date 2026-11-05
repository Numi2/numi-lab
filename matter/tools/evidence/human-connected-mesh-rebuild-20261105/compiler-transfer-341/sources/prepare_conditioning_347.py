"""Test existing current-operator conditioning on the admitted341 mesh; no physical changes."""
from pathlib import Path
import json,shutil,hashlib
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');old=W/'reference-screened-native-344';N=W/'reference-screened-conditioned-native-347'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not N.exists()
failure=json.loads((old/'install-motion.execution.json').read_text())
assert failure['exit_code']==1 and failure['resource_stop_reason'] is None
original=json.loads((old/'install-motion.plan.json').read_text())
def remap(v):
 if isinstance(v,str):return v.replace(str(old/'install-motion'),str(N/'install-motion'))
 if isinstance(v,list):return [remap(x) for x in v]
 if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
 return v
p=remap(original);env=p['stages'][0]['environment']
changes={'NM_FEM_THIN_CELL_SCHWARZ':'1','NM_FEM_THIN_CELL_REGIONAL_COMPOSITE':'1','NM_FGMRES_GRAPH_FULL_PSD_CONTACT_BLOCKS':'1','NUMI_HUMAN_RESTING_FGMRES_BUDGET':'64','NM_FEM_NEWTON_TRACE_ROOT':'26','NM_VASCULAR_TRACE_ROOT':'26','NUMI_MATTER_VASCULAR_ROW16_DIRECTION_TRACE':'1'}
before={k:env.get(k) for k in changes};env.update(changes)
p['retained_rejected_trial']=ref(old/'install-motion.execution.json')
p['retained_rejected_plan']=ref(old/'install-motion.plan.json')
p['numerical_intervention']={'before':before,'after':changes,'physical_unknowns_removed':0,'physical_equations_changed':False,'reason':'Root26 exhausted15/16 linear solves; fresh nonvascular residual0.000590985466 and descending-thoracic-aorta storage residual1.57513823e-5 failed. Reuse checked fine-cell Schwarz/regional/contact conditioning and allow current linear certification to finish. Higher budget itself is not a speedup claim.','prior_evidence':'Prior269/273 used fewer columns but failed endpoint fidelity; not promoted. This test changes the current remesh and retains all original force/strain/physical gates.','attribution':'Combined numerical intervention; no independent gain claimed for its individual settings.'}
p['purpose']='Convergence repair test on screened mesh341 with existing checked preconditioning and bounded64-column current-operator solve'
p['measurement_scope']='One32-root integrated initialization diagnostic; require full accepted horizon and endpoint parity before timing promotion'
p['predeclared_checks']['prediction']='Current-operator local conditioning plus a certified longer linear correction may avoid root26 rejection. Refuted if any physical step rejects or original endpoint force/strain gates fail.'
p['predeclared_checks']['stopping_rule']='One32-root run under unchanged900s/3GiB additional-swap/12GiB disk guards. On failure retain witnesses and change the method, not the iteration budget again.'
p['predeclared_checks']['attribution']='Whole combined intervention only. Existing16-Newton budget,15.625us timestep, constitutive laws, mass, contact and physical certificates unchanged.'
analysis=W/'analyze_native_347.py';assert not analysis.exists()
s=(W/'analyze_native_344.py').read_text().replace("reference-screened-native-344","reference-screened-conditioned-native-347").replace('native-screened-344','native-screened-347').replace('native344','native347')
analysis.write_text(s);compile(s,str(analysis),'exec')
p['analysis_owner']=ref(analysis)
for path,r in p['runtime_dependencies'].items():assert sha(path)==r['sha256'],path
assert sha(p['native_binary']['path'])==p['native_binary']['sha256']
for r in p['required_transitive_input_pins']:assert sha(r['path'])==r['sha256'],r['path']
N.mkdir();save(N/'install-motion.plan.json',p);shutil.copy2(W/'run_native_plan_compact.py',N/'run_native_plan.py')
save(N/'prelaunch-checks.json',{'plan':ref(N/'install-motion.plan.json'),'newton_budget':env['NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS'],'numerical_changes':changes,'physical_input_pins_unchanged':p['input_pins']==original['input_pins'],'native_steps':0,'physical_admission':False,'requires_16GiB_free_before_launch':True})
print('PREPARED_CONDITIONING347_NOT_EXECUTED',N,'freeGiB',shutil.disk_usage(W).free/2**30)
