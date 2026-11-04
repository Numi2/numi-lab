from pathlib import Path
import sys,json,subprocess,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
sys.path.insert(0,str(D));from trial_tools import prepare,pin,run
base=D/'contact-cache-control-prefix-221'
p=prepare(base,D/'accepted-z0-coupled-prefix-222',{'NM_FEM_REGIONAL_SELECTOR6_ACCEPTED_Z0_RECYCLE':'1'})
plan=json.loads((p/'install-motion.plan.json').read_text())
manifest=json.loads((p/'build-manifest.json').read_text())
rels=set(json.loads((base/'build-manifest.json').read_text())['source']['sources'])
rels.add('matter/include/numi/matter/accepted_z0_recycle_admission.hpp')
manifest['source']['sources']={rel:pin(R/rel) for rel in sorted(rels)}
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json');plan['full_execution_source_pins']=manifest['source']
plan['shared_rhs_change']='Unchanged current source; exercise existing accepted Z0 correction recycling through its original static-source, contact/constraint/dt and current-operator admission.'
plan['purpose']='32 complete coupled Human transactions at15.625us against221, only accepted Z0 recycler enabled. Keep all source/physical/certification/rollback gates and current physical tangent. Not a stale Jacobian, old accepted solution or reduced physical model.'
plan['hypothesis']='A validated accepted correction added to first preconditioned search direction may reduce active FGMRES work across successive full-coordinate steps. It also disables incompatible preconditioner ICBs and may be slower. Measure counts and total time; no gain forecast. No bypass if admission fails.'
plan['predeclared_checks']['equivalent_algorithm_limit']='Compare CSV and MRV state numerically; any non-bitwise difference requires full existing force/energy fidelity gate before promotion. This bounded performance experiment is not a qualified production setting.'
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with tarfile.open(p/'execution-source-snapshots.tar.gz','w:gz') as tar:
 for rel in rels:tar.add(R/rel,arcname=rel)
shutil.copy2(__file__,p/'executed-runner.py')
free=shutil.disk_usage(p).free
(p/'preparation-resource-justification.json').write_text(json.dumps({'free_bytes':free,'preparation_minimum_GiB':14.5,'hard_runtime_minimum_GiB':12,'maximum_swap_growth_GiB':3,'maximum_wall_s':900,'rationale':'Same baseline scene. Validated recycler adds bounded accepted/candidate vectors and validity metadata. Observed220 plateau10.033GB GPU;221 consumed about2.03GiB disk at peak. Preparation14.5GiB retains12GiB runtime floor plus0.47GiB margin above that observed peak. Existing12GiB runtime floor,3GiB swap-growth stop and900s budget are unchanged. Stop on source admission failure.'},indent=2)+'\n')
assert free>=14.5*2**30,free/2**30
assert not subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).stdout
print('RUN',p,free/2**30,flush=True);sys.exit(run(p))

