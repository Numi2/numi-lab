from pathlib import Path
import json,subprocess,sys,tarfile,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
name='bounded-sparse-witness-native-183'
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
plan['stages'][0]['argv'][plan['stages'][0]['argv'].index('--muscle-step-count')+1]='1'
plan['stages'][0]['expected_accepted_steps']=1
plan['stages'][0]['environment']['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']='1'
plan['stages'][0]['environment'].pop('NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP',None)
plan['stages'][0]['environment'].pop('NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT',None)
plan['stages'][0]['environment'].pop('NM_FEM_CONTACT_TRACE_ITERATION',None)
env=plan['stages'][0]['environment']
# Hybrid already enforces fresh full-operator/projected certification at every
# Newton/restart. Remove the ordinary-coordinate legacy selector and the
# OPTIONAL inexact certificate. The remaining linear target is stricter:
# min(relative target, vascular absolute tolerance), not an accuracy relaxation.
env.pop('NM_FEM_TRUE_RESIDUAL_CERTIFY',None)
env.pop('NM_FGMRES_INEXACT_COUPLED_LINEAR_CERTIFICATE',None)
env.update({
 'NM_FEM_HUMAN_HYBRID_COORDINATES':'1',
 'NM_FEM_HUMAN_HYBRID_EMBED_CONTACT_BOUNDARY':'1',
 'NM_FEM_HYBRID_RESIDUAL_WITNESS':'1',
 'NM_FGMRES_RELATIVE_INNER_FORCING':'0',
 'NM_FGMRES_LEFT_ROW_EQUILIBRATION':'0',
 'NM_FGMRES_LEFT_ROW_EQUILIBRATION_COUPLED_FEM':'0',
 'NM_FGMRES_LEFT_ROW_EQUILIBRATION_HUMAN_ONLY':'0',
 'NM_FGMRES_SKIP_COMPLETED_ZERO_OPERATOR_ACTION':'1',
 'NM_FGMRES_PREFIX_PRECONDITIONER_ICB':'0',
 'NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS':'0,1',
 'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'0',
 'NM_FEM_NEWTON_TRACE_ROOT':'0','NM_FEM_BACKTRACKING_TRACE_ROOT':'0',
 'NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT':'0'})
argv=plan['stages'][0]['argv']
if '--vascular-dense45' in argv:argv.remove('--vascular-dense45')
plan['purpose']='Bounded witness observer182 (4MiB maximum, full component vectors explicitly unavailable on Human; mandatory GPU certificates retained): capture a fresh current compact-basis physical residual witness for bounded source-bound refinement180. The existing aggressive reduced model remains known to fail regional force/strain gates. This diagnostic changes no source geometry, material, mass, timestep, contact or physical acceptance and cannot qualify reduced-model performance.'
plan['hypothesis']='Obtain exact current compact layout metadata and residual localization, retain failed-model status, and recompile only measured fine-row promotions before comparison against full coordinates. No fabricated witness transplantation.'
plan['predeclared_checks']={
 'steps':1,'timestep_s':0.000015625,
 'certification':'runtime forces hybrid true residual and restart replacement and production failure propagation, independent of legacy NM_FEM_TRUE_RESIDUAL_CERTIFY; STRICT min(relative, vascular absolute) linear target',
 'unchanged':'source134 skin/fat geometry and conservative mass; material laws; contact and positive-cell gates; timestep; nonlinear tolerances; transaction rollback',
 'coordinate_approximation':'existing immutable compiler-generated regional Bq, B transpose f and B transpose M B; explicitly a reduced model',
 'engineering_gates':'<=1% integrated forces and physiological outputs; <=5% regional deformation/strain energy; unchanged penetration bounds against matched full-coordinate endpoint',
 'failure_policy':'keep first admission/residual/contact/quality witness; no bypass, no throughput promotion without engineering gates',
 'qualification':'one prefix step is not supported rest, motion qualification, sustained performance or clinical validation'}
sys.path.insert(0,str(D))
from trial_tools import pin,run
manifest=json.loads((p/'build-manifest.json').read_text())
for rel in list(json.loads((D/'streamed-sparse-088/preimages.json').read_text()))+['apps/NumiHumanSkinFatFixedCavityReconditioning.hpp','apps/NumiHumanSkinFatFixedCavityReconditioningCheck.cpp','matter/tools/partitioned_reduction_projection_gpu_check.mm']:
 manifest['source']['sources'][rel]=pin(R/rel)
manifest['source']['sources']['apps/NumiHumanSkinFatReductionLabelsCheck.cpp']=pin(R/'apps/NumiHumanSkinFatReductionLabelsCheck.cpp')
(p/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan['build_manifest']=pin(p/'build-manifest.json')
plan['full_execution_source_pins']=manifest['source']
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
archive=p/'execution-source-snapshots.tar.gz'
if archive.is_symlink():archive.unlink()
with tarfile.open(archive,'w:gz') as tar:
 for rel in manifest['source']['sources']:tar.add(R/rel,arcname=rel)
for src,dest in [('bounded-hybrid-witness-build-182.log','build-final.log'),('fresh-scaled-preflight-full-177.log','solver-final.stdout.log'),
 ('direct-gpu-096.log','gate-final.stdout.log'),('compiler-direct-096.log','compiler-final.stdout.log')]:
 shutil.copy2('/tmp/'+src,p/dest)
shutil.copy2('/tmp/connected-mass-test-130.log',p/'cavity-final.stdout.log')
shutil.copy2('/tmp/packed-regional-direct-154.log',p/'positive-gap-direct.stdout.log')
shutil.copy2('/tmp/sparse-residual-refinement-check-180-admitted.log',p/'sparse-refinement-check.stdout.log')
shutil.copy2(__file__,p/'executed-runner.py')
print('RUN',name,flush=True)
import shutil
assert shutil.disk_usage(p).free>=15*1024**3, 'Need15GiB before compact resource-guarded trial'
sys.exit(run(p))

