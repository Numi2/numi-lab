from pathlib import Path
import sys,json,subprocess,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
sys.path.insert(0,str(D));from trial_tools import prepare,pin,run
base=D/'contact-cache-control-prefix-221'
p=prepare(base,D/'accepted-z0-unscaled-prefix-223',{'NM_FEM_REGIONAL_SELECTOR6_ACCEPTED_Z0_RECYCLE':'1'})
plan=json.loads((p/'install-motion.plan.json').read_text())
e=plan['stages'][0]['environment']
for k in ['NM_FGMRES_LEFT_ROW_EQUILIBRATION','NM_FGMRES_LEFT_ROW_EQUILIBRATION_HUMAN_ONLY','NM_FGMRES_LEFT_ROW_EQUILIBRATION_COUPLED_FEM']:e.pop(k,None)
args=plan['stages'][0]['argv'];args[args.index('--muscle-step-count')+1]='4'
plan['stages'][0]['expected_accepted_steps']=4
plan['expected_accepted_steps']=4
e['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='4'
e['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='0,4'
manifest=json.loads((p/'build-manifest.json').read_text())
rels=set(json.loads((base/'build-manifest.json').read_text())['source']['sources'])
rels.add('matter/include/numi/matter/accepted_z0_recycle_admission.hpp')
manifest['source']['sources']={rel:pin(R/rel) for rel in sorted(rels)}
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json');plan['full_execution_source_pins']=manifest['source']
plan['purpose']='4 full-coordinate coupled Human transactions at15.625us. Test existing accepted correction recycler in its originally admitted unscaled solver layout. The222 scaled combination was rejected before stepping. No admission restriction changed or bypassed. Full current physical operator, residual, certification and rollback remain mandatory.'
plan['hypothesis']='Joint configuration test of accepted Z0 recycling and the existing unscaled FGMRES operator. Measure against first4 steps of221; do not attribute differences to recycling alone. Retain first rejection and skip broader qualification unless this bounded screen improves active work and numerical outputs.'
plan['shared_rhs_change']='No source change. Existing supported unscaled operator plus accepted Z0; no reduced physical coordinates.'
plan['predeclared_checks']={'steps':4,'timestep_s':0.000015625,'full_coupled_true_residual':True,'failures':'retain source and numerical rejection; no bypass','fidelity':'compare first4 baseline trace rows and initial geometry; non-bitwise outputs require full force/energy fidelity before promotion','performance':'bounded screen only; no0.1x or motion qualification'}
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with tarfile.open(p/'execution-source-snapshots.tar.gz','w:gz') as tar:
 for rel in rels:tar.add(R/rel,arcname=rel)
shutil.copy2(__file__,p/'executed-runner.py')
free=shutil.disk_usage(p).free
(p/'preparation-resource-justification.json').write_text(json.dumps({'free_bytes':free,'preparation_minimum_GiB':14.5,'hard_runtime_minimum_GiB':12,'maximum_swap_growth_GiB':3,'maximum_wall_s':900,'rationale':'Bounded4-step existing-implementation screen; prior full-control disk excursion2.03GiB. Hard12GiB disk floor and3GiB swap-growth watchdog unchanged.'},indent=2)+'\n')
assert free>=14.5*2**30,free/2**30
assert not subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).stdout
print('RUN',p,free/2**30,flush=True);sys.exit(run(p))

