"""Pin the current CCD-qualified runtime for the screened mesh's native comparison."""
from pathlib import Path
import json,hashlib,shutil,subprocess
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
D=W/'reference-screened-interior-341';N=W/'reference-screened-native-344'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
p=json.loads((N/'install-motion.plan.json').read_text())
control=json.loads((W/'matched-ccd-baseline-335/install-motion.plan.json').read_text())
prior=json.loads((W/'compensated-interior-native-334/install-motion.plan.json').read_text())
assert not (N/'install-motion.execution.json').exists()
env=p['stages'][0]['environment']
archive=env.pop('NUMI_MATTER_PIPELINE_ARCHIVE',None)
if archive:p['runtime_dependencies'].pop(archive,None)
p['native_binary']=ref(control['native_binary']['path'])
assert p['native_binary']['sha256']==control['native_binary']['sha256']
p['runtime_dependencies']={path:ref(path) for path in control['runtime_dependencies']}
assert all(p['runtime_dependencies'][path]['sha256']==value['sha256'] for path,value in control['runtime_dependencies'].items())
p['ccd_repair']=control['ccd_repair']
p['optional_driver_archive_disabled']={'enabled':False,'reason':'Same current-metallib and optional-cache setting as matched control335 and prior candidate334'}
p['native_comparison_reference']=ref(W/'matched-ccd-baseline-335/install-motion.plan.json')
p['screened_reference_field']=ref(D/'independent-reference-field.json')
p['analysis_owner']=ref(W/'analyze_native_344.py')
p['geometry_baseline']=ref(D/'report.json')
p['purpose']='Complete native-scene comparison of reference-deformation-screened interior remesh341 against unchanged mesh335'
p['trial_scope']='32 roots at unchanged15.625us; initialization transient, not sustained motion or resting qualification'
p['measurement_scope']='Whole integrated body wall, active solver work and admitted endpoint force/strain/physiology; instrumented short run'
p['predeclared_checks']['reference']=p['native_comparison_reference']
p['predeclared_checks']['prediction']='The three rejected local operations remove the two fixed-field peak failures. Actual native force/strain parity may still fail from changed trajectory or insufficient convergence. A speed improvement requires lower body wall time; no forecast gain.'
p['predeclared_checks']['effect_rule']='Require existing1% force/physiology and5% regional strain/energy gates; one timing pair is provisional. No native promotion from fixed-field checks.'
p['predeclared_checks']['stopping_rule']='One32-root trial; retain any rejection or mismatch and diagnose before another mesh trial.'
p['runtime_launcher']=ref(W/'run_native_plan_compact.py')
# Assert that the physics/solver controls have not drifted from the comparison.
changed={}
for k in set(env)|set(control['stages'][0]['environment']):
 a=env.get(k);b=control['stages'][0]['environment'].get(k)
 if a==b:continue
 if not (k.startswith('NUMI_H706_FRACTIONAL_') or k.startswith('NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_') or k in ['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256','NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP','NUMI_HUMAN_SKIN_REFERENCE_DUMP','NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT','NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV']):
  changed[k]=dict(candidate=a,control=b)
assert not changed,changed
# Preserve original binary-build lineage and explicitly record current source overlays.
p['current_native_source_overlays']={key:ref(value['path']) for key,value in control['full_execution_source_pins']['sources'].items()}
p['native_source_overlay_scope']='Source inventory at trial scheduling; compiler Python edits after binary build are separate and do not imply a rebuilt binary.'
issues=[]
for r in p['required_transitive_input_pins']:
 if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']:issues.append(r['path'])
for path in p['input_pins']:
 if not Path(path).is_file():issues.append(path)
assert not issues,issues
assert shutil.disk_usage(W).free>=16*2**30
save(N/'install-motion.plan.json',p)
shutil.copy2(W/'run_native_plan_compact.py',N/'run_native_plan.py')
save(N/'current-runtime-prelaunch-checks.json',dict(plan=ref(N/'install-motion.plan.json'),runtime_matches_control=True,unrelated_environment_differences=changed,transitive_input_failures=issues,device=subprocess.run(['system_profiler','SPDisplaysDataType'],capture_output=True,text=True).stdout,available_bytes=shutil.disk_usage(W).free,physical_admission=False,native_steps=0))
print('SCREENED_NATIVE344_READY_NOT_EXECUTED',N)
