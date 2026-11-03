from pathlib import Path
import json,tarfile,subprocess,sys,hashlib,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
sys.path.insert(0,str(D))
from trial_tools import prepare,run,pin
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
base=D/'completed-linear-builders-063'
name=sys.argv[1]
enabled=sys.argv[2]=='1'
keys=''
d=prepare(base,D/name,{
'NM_CONTACT_FINAL_ZERO_GEOMETRY_REUSE':'1' if enabled else '0',
'NM_SCHEDULER_TILED_FEM_OBSERVE':'1' if enabled else '0',
'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'0',
'NUMI_MATTER_GPU_TIMING':'0',
'NUMI_MATTER_GPU_TIMING_STAGES':'0',
'NUMI_GPU_PROGRESS':'0',
'NUMI_HUMAN_CONTACT_GPU_TIMING':'0',
'NUMI_MATTER_GPU_TIMING_KERNELS':keys,
})
p=json.loads((d/'install-motion.plan.json').read_text())
p['stages'][0]['environment'].pop('NM_FGMRES_CONTACT_DIRECTIONAL_CACHE',None)
p['stages'][0]['environment'].pop('NM_FGMRES_SHARED_PATCH_VALUES',None)
if len(sys.argv)>4:p['stages'][0]['environment'].update(json.loads(sys.argv[4]))
p['purpose']='Quiet matched four-step initialization-prefix performance measurement; full endpoint dump only on explicit override; physical gates unchanged'
p['predeclared_checks']={'accepted_steps':4,'endpoint':'only available when explicitly captured; quiet timing uses compact active-work, physiology and existing accepted geometry','physiology':'bitwise CSV parity','precision_and_physics':'unchanged','stage_timings':'root0 same selected kernels, whole roots separately','active_iterations':'must match','physical_admission':False}
if p['stages'][0]['environment'].get('NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE')=='1':
 p['predeclared_checks']['endpoint']='Bitwise full endpoint parity with published completed-linear-builders-063'
if len(sys.argv)>4 and 'NM_FGMRES_INEXACT_NEWTON_MAX_ETA' in json.loads(sys.argv[4]):
 p['purpose']='Bounded active-linear-work experiment using existing EW inner forcing; no final physical tolerance, force, material, collision or timestep change'
 p['predeclared_checks']['endpoint']='Existing owner comparator: force <=1%, regional deformation/strain-energy <=5%, unchanged mass/topology/source inputs'
 p['predeclared_checks']['physiology']='Same coupled CSV quantities within 1%; no independent curve changes'
 p['predeclared_checks']['acceptance']='All original final residual, contact and accepted-state gates unchanged; stop on failure'
 p['predeclared_checks']['inner_algorithm_change']=json.loads(sys.argv[4])
m=json.loads((d/'build-manifest.json').read_text())
sources=json.loads((base/'build-manifest.json').read_text())['source']['sources']
sources={k:pin(R/k) for k in sources}
sources['matter/src/metal/identification_scheduler.metalinc']=pin(R/'matter/src/metal/identification_scheduler.metalinc')
for f in ['CMakeLists.txt','matter/CMakeLists.txt','matter/tools/vascular_cavity_check.mm','matter/tools/fgmres_true_residual_native_check.mm']:sources[f]=pin(R/f)
m['source']['sources']=sources
(d/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
from quiet_launcher_068 import configure
configure(d,p,pin)
p['build_manifest']=pin(d/'build-manifest.json')
p['full_execution_source_pins']=m['source']
(d/'install-motion.plan.json').write_text(json.dumps(p,indent=2)+'\n')
snapshot=d/'execution-source-snapshots.tar.gz'
import os
reuse=None
for manifest in sorted(D.glob('*/build-manifest.json')):
 if manifest.parent==d:continue
 prior=json.loads(manifest.read_text()).get('source',{}).get('sources',{})
 archive=manifest.parent/'execution-source-snapshots.tar.gz'
 if set(prior)!=set(sources) or not archive.is_file():continue
 if not all(prior[k]['sha256']==sources[k]['sha256'] for k in sources):continue
 with tarfile.open(archive,'r:gz') as tar:
  for key in sources:
   member=tar.extractfile(key)
   if member is None or hashlib.file_digest(member,'sha256').hexdigest()!=sources[key]['sha256']:break
  else:reuse=archive
 if reuse is not None:break
if reuse is not None:
 snapshot.symlink_to(os.path.relpath(reuse,d))
else:
 with tarfile.open(snapshot,'w:gz') as tar:
  for f in sources:tar.add(R/f,arcname=f)

for a in [str(D/'completed-incidence-067/build-final.log'),str(D/'completed-incidence-067/solver-final.stdout.log'),str(D/'completed-incidence-067/gate-final.stdout.log'),str(D/'completed-incidence-067/cavity-final.stdout.log')]:
 if Path(a).is_file():shutil.copy2(a,d/Path(a).name)
print('prepared',d,flush=True)
if len(sys.argv)>3 and sys.argv[3]=='run':sys.exit(run(d))
