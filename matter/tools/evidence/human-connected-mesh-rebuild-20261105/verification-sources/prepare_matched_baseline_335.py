from pathlib import Path
import json,hashlib,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
old=W/'baseline-repeat-native-324';out=W/'matched-ccd-baseline-335'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ref(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
p=json.loads((old/'install-motion.plan.json').read_text())
def remap(v):
 if isinstance(v,str):return v.replace(str(old/'install-motion'),str(out/'install-motion'))
 if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
 if isinstance(v,list):return [remap(x) for x in v]
 return v
p=remap(p)
cache=p['stages'][0]['environment'].pop('NUMI_MATTER_PIPELINE_ARCHIVE')
p['optional_driver_archive_disabled']={'previous':p['runtime_dependencies'].pop(cache),'reason':'Match new CCD repaired candidate, no optional persistent driver archive.'}
for path in list(p['runtime_dependencies']):
 if path.endswith('/NumiMatter.metallib'):p['runtime_dependencies'][path]=ref(path)
p['retained_prior_plan']=ref(old/'install-motion.plan.json')
p['baseline_restoration']=ref(W/'baseline-repeat-332-restoration/receipt.json')
p['baseline_additional_inventory_restoration']=ref(W/'baseline-repeat-332b-restoration/receipt.json')
p['ccd_repair']=json.loads((W/'compensated-interior-native-334/install-motion.plan.json').read_text())['ccd_repair']
p['purpose']='Matched current-metallib unchanged-geometry control for interior326 conditioning comparison'
p['predeclared_checks']['prediction']='Unchanged source geometry reproduces baseline324 forces/strain/gases; timing under identical repaired metallib and disabled optional cache allows mesh comparison.'
p['runtime_launcher']=ref(W/'run_native_plan_compact.py')
assert sha(p['native_binary']['path'])==p['native_binary']['sha256']
for raw,r in p['runtime_dependencies'].items():assert sha(raw)==r['sha256'],raw
for r in p['required_transitive_input_pins']:assert sha(r['path'])==r['sha256'],r['path']
assert all(Path(r).is_file() for r in p['input_pins'])
assert shutil.disk_usage(W).free>=16*1024**3,shutil.disk_usage(W).free/2**30
out.mkdir(exist_ok=False)
(out/'install-motion.plan.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
shutil.copy2(W/'run_native_plan_compact.py',out/'run_native_plan.py')
print('PREPARED_MATCHED_CONTROL',out,'freeGiB',shutil.disk_usage(W).free/2**30)
