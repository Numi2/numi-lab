from pathlib import Path
import json,tarfile,subprocess,sys,hashlib,shutil
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
sys.path.insert(0,str(D))
from trial_tools import prepare,run,pin
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
base=D/'split-contact-control-054'
name=sys.argv[1]
enabled=sys.argv[2]=='1'
keys='nm_scheduler_observe,nm_scheduler_observe_fem_partials,nm_scheduler_observe_tiled'
d=prepare(base,D/name,{
'NM_CONTACT_FINAL_ZERO_GEOMETRY_REUSE':'1' if enabled else '0',
'NM_SCHEDULER_TILED_FEM_OBSERVE':'1' if enabled else '0',
'NM_FGMRES_SHARED_PATCH_VALUES':'1' if enabled else '0',
'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'1',
'NUMI_HUMAN_CONTACT_GPU_TIMING':'0',
'NUMI_MATTER_GPU_TIMING_KERNELS':keys,
})
p=json.loads((d/'install-motion.plan.json').read_text())
p['stages'][0]['environment'].pop('NM_FGMRES_CONTACT_DIRECTIONAL_CACHE',None)
p['stages'][0]['environment'].pop('NM_FGMRES_SHARED_PATCH_VALUES',None)
if len(sys.argv)>4:p['stages'][0]['environment'].update(json.loads(sys.argv[4]))
p['purpose']='Matched four-step initialization-prefix measurement of tiled exact scheduler observation and final exact-zero broadphase reuse; no supported-rest or motion qualification'
p['predeclared_checks']={'accepted_steps':4,'endpoint':'bitwise against the matched control and retained four-step endpoint','physiology':'bitwise CSV parity','precision_and_physics':'unchanged','stage_timings':'root0 same selected kernels, whole roots separately','active_iterations':'must match','physical_admission':False}
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
m['source']['sources']=sources
(d/'build-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p['build_manifest']=pin(d/'build-manifest.json')
p['full_execution_source_pins']=m['source']
(d/'install-motion.plan.json').write_text(json.dumps(p,indent=2)+'\n')
with tarfile.open(d/'execution-source-snapshots.tar.gz','w:gz') as tar:
 for f in sources:tar.add(R/f,arcname=f)
for a in ['/tmp/finalization-059-build.log','/tmp/finalization-059-full-check.log']:
 if Path(a).is_file():shutil.copy2(a,d/Path(a).name)
print('prepared',d,flush=True)
if len(sys.argv)>3 and sys.argv[3]=='run':sys.exit(run(d))
