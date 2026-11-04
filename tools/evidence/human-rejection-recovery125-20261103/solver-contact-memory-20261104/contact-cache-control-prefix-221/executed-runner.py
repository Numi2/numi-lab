from pathlib import Path
import json,subprocess,sys,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
sys.path.insert(0,str(D))
from trial_tools import prepare,pin,run
base=D/'contact-cache-bounded-pool-prefix-220'
p=prepare(base,D/'contact-cache-control-prefix-221',{'NM_MIXED_CERTIFICATE_CONTACT_CACHE':'0'})
plan=json.loads((p/'install-motion.plan.json').read_text())
plan['stages'][0]['environment'].pop('NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP',None)
for k in plan['runtime_dependencies']:
 if k in plan['input_pins']:plan['input_pins'][k]=pin(k)
manifest=json.loads((p/'build-manifest.json').read_text())
rels=set(json.loads((base/'build-manifest.json').read_text())['source']['sources'])
rels.add('src/metal/MetalArticulatedOperator.mm')
manifest['source']['sources']={rel:pin(R/rel) for rel in sorted(rels)}
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json')
plan['full_execution_source_pins']=manifest['source']
plan['shared_rhs_change']='Unchanged source215 packed preconditioner. Source218 cache disabled as matched control; source219 bounded horizon autorelease lifetime and opt-in accepted-boundary memory observation.'
plan['purpose']='32 complete coupled Human transactions, identical full-coordinate initial state, 15.625us, all physical/certification/rollback checks retained. Compare all11 CSV and3 MRV exactly to220. Initialization separate. Additional cache resident bytes zero. This is a release-transient prefix, not supported resting or motion qualification.'
plan['hypothesis']='Same final binary with cache disabled: paired attribution of final-contact work cache. Bounded pool and observations unchanged.'
plan['measurement_scope']='32 accepted steps; root0 whole-stage GPU timing, unselected timing off roots1..31; accepted memory observation and5s OS resource samples.'
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with tarfile.open(p/'execution-source-snapshots.tar.gz','w:gz') as tar:
 for rel in manifest['source']['sources']:tar.add(R/rel,arcname=rel)
for source,dest in [('contact-cache-final-build-218.log','build-final.log'),('contact-cache-final-full-218.log','solver-final.stdout.log'),('contact-cache-final-focused-218.log','contact-cache-focused.stdout.log'),('contact-cache-final-runtime-218.log','runtime-gravity-replay.stdout.log')]:
 shutil.copy2('/tmp/'+source,p/dest)
shutil.copy2(__file__,p/'executed-runner.py')
free=shutil.disk_usage(p).free
(p/'preparation-resource-justification.json').write_text(json.dumps({'free_bytes':free,'preparation_minimum_GiB':15.8,'hard_runtime_minimum_GiB':12,'maximum_swap_growth_GiB':3,'new_resident_cache_bytes':0,'maximum_wall_s':900,'no_physics_or_accuracy_guard_changed':True},indent=2)+'\n')
assert free>=15.8*1024**3, free/1024**3
assert not subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).stdout
print('RUN',p,'freeGiB',free/1024**3,flush=True)
sys.exit(run(p))

