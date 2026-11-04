from pathlib import Path
import json,subprocess,sys,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
name='packed-regional-native-158'
proposal=D/'connected-conditioning-134/fixed-boundary-volume-proposal.npz'
report=json.loads((proposal.parent/'report.json').read_text())
overrides={'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ':str(proposal),
 'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256':report['proposal_sha256'],'NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR':'1','NM_CONTACT_COMPLETED_INCIDENCE_REUSE':'1',
 'NM_FEM_TANGENT_CENSUS':'0','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT':'26','NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT':'3620233','NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA':'1','NM_FEM_DEFORMABLE_LIMIT_TRACE_FAILURES':'1','NM_FEM_CONTACT_TRACE_ITERATION':'11','NUMI_MATTER_GPU_TIMING':'1','NUMI_MATTER_GPU_TIMING_ROOT':'0','NUMI_MATTER_GPU_TIMING_STAGES':'1','NUMI_MATTER_GPU_TIMING_SAMPLE_EVERY':'1','NUMI_MATTER_GPU_TIMING_KERNELS':'',
 'NM_FEM_NEWTON_TRACE_ROOT':'26','NM_FEM_BACKTRACKING_TRACE_ROOT':'26',
 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS':'0,32','NM_FGMRES_PREFIX_PRECONDITIONER_ICB':'1',
 'NUMI_HUMAN_SKIN_FAT_COMPACT_MASS_CAPTURE':'1',
 'NUMI_MATTER_HUMAN_PARTITIONED_PRECONDITIONER':'1',
 'NM_FEM_HUMAN_PARTITIONED_SPARSE_BASIS':'1','NUMI_HUMAN_PARTITIONED_MASS_WHITEN':'1',
 'NUMI_MATTER_HUMAN_PARTITIONED_FEM_TANGENT':'1','NM_FEM_HUMAN_PARTITIONED_FUSED_TANGENT':'1'}
subprocess.run([sys.executable,'/tmp/run-performance-068.py',name,'1','prepare',json.dumps(overrides)],check=True)
p=D/name
plan=json.loads((p/'install-motion.plan.json').read_text())
plan['stages'][0]['argv'][plan['stages'][0]['argv'].index('--muscle-step-count')+1]='32'
plan['stages'][0]['expected_accepted_steps']=32
plan['stages'][0]['environment']['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='32'
plan['stages'][0]['environment'].pop('NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP',None)
plan['stages'][0]['environment'].pop('NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT',None)
plan['stages'][0]['environment'].pop('NM_FEM_CONTACT_TRACE_ITERATION',None)
plan['purpose']='Exact immutable packed regional tile scheduling for B transpose f, B transpose D B and streamed FEM tangent; compare against151 source150. Whole-stage timing only at root0; roots1..31 uninstrumented.'
plan['hypothesis']='Avoid padding every region to the largest support. Preserve binary reduction order, exact positive-zero padding, equations, source geometry134, masses, timestep, convergence and final certificates. Require32 accepted steps and common CSV/geometry parity; measure gains rather than predict them.'
plan['predeclared_checks']={'steps':32,'timestep_s':0.000015625,'mandatory_coupled_true_residual':True,
 'material_mass_geometry':'fixed vertices and exact layer boundary/volume; conservative source/nodal mass recomposed by existing importer','coordinate_approximation':'none, preconditioner only',
 'numerical_comparison':'existing engineering <=1% forces/physiology, <=5% deformation/energy; prefixes are not sustained motion qualification',
 'failures':'retain first source, capacity, conditioning, residual or physical failure; do not bypass admission',
 'performance':'exploratory; no speedup forecast; initialization separate; compare final-production-native-086'}
sys.path.insert(0,str(D))
from trial_tools import pin,run
manifest=json.loads((p/'build-manifest.json').read_text())
for rel in list(json.loads((D/'streamed-sparse-088/preimages.json').read_text()))+['apps/NumiHumanSkinFatFixedCavityReconditioning.hpp','apps/NumiHumanSkinFatFixedCavityReconditioningCheck.cpp','matter/tools/partitioned_reduction_projection_gpu_check.mm']:
 manifest['source']['sources'][rel]=pin(R/rel)
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json')
plan['full_execution_source_pins']=manifest['source']
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
archive=p/'execution-source-snapshots.tar.gz'
if archive.is_symlink():archive.unlink()
with tarfile.open(archive,'w:gz') as tar:
 for rel in manifest['source']['sources']:tar.add(R/rel,arcname=rel)
for src,dest in [('packed-regional-build-154.log','build-final.log'),('packed-regional-solver-154.log','solver-final.stdout.log'),
 ('direct-gpu-096.log','gate-final.stdout.log'),('compiler-direct-096.log','compiler-final.stdout.log')]:
 shutil.copy2('/tmp/'+src,p/dest)
shutil.copy2('/tmp/connected-mass-test-130.log',p/'cavity-final.stdout.log')
shutil.copy2('/tmp/packed-regional-direct-154.log',p/'positive-gap-direct.stdout.log')
shutil.copy2(__file__,p/'executed-runner.py')
print('RUN',name,flush=True)
import shutil
assert shutil.disk_usage(p).free>=17*1024**3, 'Need17GiB before resource-guarded retry'
sys.exit(run(p))

