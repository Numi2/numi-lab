from pathlib import Path
import json,subprocess,sys,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
sys.path.insert(0,str(D))
from trial_tools import prepare,pin,run
base=D/'contact-cache-control-prefix-221'
p=prepare(base,D/'routine-control-prefix-226',{'NUMI_MATTER_GPU_TIMING':'0','NUMI_MATTER_GPU_TIMING_STAGES':'0'})
plan=json.loads((p/'install-motion.plan.json').read_text())
env=plan['stages'][0]['environment']
removed={}
for k in ['NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP','NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_OBJECT','NM_FEM_BACKTRACKING_TRACE_NEWTON','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT','NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA']:
 if k in env:removed[k]=env.pop(k)
for k in plan['runtime_dependencies']:
 if k in plan['input_pins']:plan['input_pins'][k]=pin(k)
manifest=json.loads((p/'build-manifest.json').read_text())
rels=set(json.loads((base/'build-manifest.json').read_text())['source']['sources'])
manifest['source']['sources']={rel:pin(R/rel) for rel in sorted(rels)}
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json')
plan['full_execution_source_pins']=manifest['source']
plan['shared_rhs_change']='Production215 packed preconditioner,219 bounded horizon pool and225 inactive history binding extent fix. Rejected218 contact cache removed.'
plan['purpose']='32 full-coordinate complete coupled Human transactions at unchanged15.625us; optional deep root26 trace and encoder-splitting GPU timing disabled. All physical/certification/rollback checks retained. Compare11 CSV and3 MRV exactly against221.'
plan['hypothesis']='Attribute selected-root temporary memory excursion to optional deep diagnostics and establish a non-stage-profiled throughput control; no solver speedup predicted.'
plan['measurement_scope']='32 accepted steps. Accepted-boundary memory and compact physical observations remain enabled; OS sampled every5s. Initialization separate; no stage or kernel timing.'
plan['removed_optional_diagnostics']=removed
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with tarfile.open(p/'execution-source-snapshots.tar.gz','w:gz') as tar:
 for rel in manifest['source']['sources']:tar.add(R/rel,arcname=rel)
shutil.copy2('/tmp/routine-control-build-226.log',p/'build-final.log')
shutil.copy2(__file__,p/'executed-runner.py')
free=shutil.disk_usage(p).free
(p/'preparation-resource-justification.json').write_text(json.dumps({'free_bytes':free,'preparation_minimum_GiB':14.25,'hard_runtime_minimum_GiB':12,'maximum_swap_growth_GiB':3,'maximum_wall_s':900,'basis':'221 measured maximum disk excursion2.03GiB plus0.22GiB margin above12GiB runtime floor','no_physics_or_accuracy_guard_changed':True},indent=2)+'\n')
assert free>=14.25*1024**3,free/1024**3
assert not subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).stdout
print('RUN',p,'freeGiB',free/1024**3,flush=True)
sys.exit(run(p))
