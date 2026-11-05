from pathlib import Path
import hashlib,json,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
old=W/'compensated-interior-native-331';out=W/'compensated-interior-native-334'
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def ref(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
assert shutil.disk_usage(W).free>=16*1024**3,shutil.disk_usage(W).free/2**30
assert json.loads((old/'install-motion.execution.json').read_text())['resource_stop_reason']=='disk_below_12GiB'
p=json.loads((old/'install-motion.plan.json').read_text())
def remap(v):
 if isinstance(v,str):return v.replace(str(old/'install-motion'),str(out/'install-motion'))
 if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
 if isinstance(v,list):return [remap(x) for x in v]
 return v
p=remap(p)
p['retained_resource_failure']=ref(old/'install-motion.execution.json')
p['storage_recovery']=ref(W/'resource-recovery-333/human-cold-motion-333-00/human-cold-motion-333-00-verification.json')
p['duplicate_pack_storage_recovery']=ref(W/'resource-recovery-333/duplicate-packs-receipt.json')
issues=[]
assert sha(p['native_binary']['path'])==p['native_binary']['sha256']
for raw,r in p['runtime_dependencies'].items():
 if sha(raw)!=r['sha256']:issues.append(raw)
for r in p['required_transitive_input_pins']:
 if sha(r['path'])!=r['sha256']:issues.append(r['path'])
missing=[r for r in p['input_pins'] if not Path(r).is_file()]
assert not issues and not missing,(issues,missing)
out.mkdir(exist_ok=False)
(out/'install-motion.plan.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
shutil.copy2(W/'run_native_plan_compact.py',out/'run_native_plan.py')
(out/'prelaunch-checks.json').write_text(json.dumps({'plan':ref(out/'install-motion.plan.json'),'runtime_and_transitive_mismatches':issues,'missing_inputs':missing,'disk_available_bytes':shutil.disk_usage(W).free,'unchanged_physical_settings':True,'unchanged_12GiB_runtime_guard':True},indent=2)+'\n')
print('PREPARED',out,'freeGiB',shutil.disk_usage(W).free/2**30)
