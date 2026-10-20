#!/usr/bin/env python3
import json,pathlib,os,hashlib,subprocess,time,platform,argparse
ap=argparse.ArgumentParser()
ap.add_argument('source',type=pathlib.Path)
ap.add_argument('destination',type=pathlib.Path)
ap.add_argument('--steps',type=int,default=4)
ap.add_argument('--rollback',action='store_true')
ap.add_argument('--root-z',type=float)
ap.add_argument('--geometry-audit',type=pathlib.Path)
args=ap.parse_args()
if args.steps<1 or (args.rollback and args.steps!=2):ap.error('positive steps required; rollback fixture requires exactly two steps')
root=next(parent for parent in pathlib.Path(__file__).resolve().parents if (parent/'CMakeLists.txt').is_file() and (parent/'matter').is_dir())
p=args.destination.resolve();p.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
j=json.loads(args.source.read_text());j['argv'][4]=str(p/'output')
# Record the selected solver policy rather than inheriting it invisibly.
for policy in ('NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER','NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR'):
 if policy in os.environ:
  if os.environ[policy] not in ('0','1'):ap.error(policy+' must be exactly 0 or 1')
  j['environment'][policy]=os.environ[policy]
if '--resting-registered-ligaments' not in j['argv']:j['argv'].append('--resting-registered-ligaments')
j['argv'][j['argv'].index('--muscle-step-count')+1]=str(args.steps)
if args.root_z is not None:j['argv'][j['argv'].index('--root-pose')+3]=str(args.root_z)
j['cwd']=str(p)
j['environment']['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(p/'common-field-failure.json')
if args.rollback:j['environment']['NUMI_HUMAN_SHARED_TISSUE_ROLLBACK_TEST']='1'
else:j['environment'].pop('NUMI_HUMAN_SHARED_TISSUE_ROLLBACK_TEST',None)
j['verification_note']='Registered passive ligament fibers with common rigid supine skin/fat transfer; numerical checks only; exact run scope is step count times timestep.'
j['binary_sha256']=sha(j['argv'][0]);j['source_invocation_sha256']=sha(args.source)
j['source_sha256']=sha(root/'apps/numilab_human_myosim_visual_probe.mm')
(p/'invocation.json').write_text(json.dumps(j,indent=2)+'\n')
files=['apps/NumiHumanAcceptedFEMSkinTransfer.hpp','apps/NumiHumanCoarseTissuePreparation.hpp','apps/NumiHumanWholeBodySkinReduction.hpp','apps/NumiHumanRegisteredLigaments.hpp','apps/NumiHumanSkinRigidPlacement.hpp','apps/NumiHumanSkinFEMIntegration.wholebody.hpp','apps/numilab_human_myosim_visual_probe.mm','matter/src/metal/fgmres.metalinc','matter/src/metal/numi_human.metalinc','matter/src/metal/vascular.metalinc','matter/src/runtime.mm','matter/src/metal/fem.metalinc','matter/src/metal/contact.metalinc','matter/src/numi_human.mm','matter/include/numi/matter/numi_human_shared.h','matter/include/numi/matter/shared.h','include/metalrobo/MetalArticulatedOperator.hpp','src/metal/MetalArticulatedOperator.mm','include/metalrobo/NumiHumanMuscleEquilibrium.hpp','src/core/NumiHumanMuscleEquilibrium.cpp','tests/numi_human_static_support_test.cpp','tests/numi_human_per_contact_plane_metal_test.mm','CMakeLists.txt','matter/CMakeLists.txt','matter/include/numi/matter/fem_compensated_deformation.h','matter/src/compiler_impl.cpp','matter/src/validation_impl.cpp','matter/src/fem_reference_geometry.hpp','matter/src/metal/base_cgls_runtime.metalinc','matter/src/metal/common.metalinc','matter/include/numi/matter/paper_compensated_bytecode.h','include/metalrobo/compensated_translation_gpu.h','apps/numanx_matter_attachment_runtime_probe.mm','matter/tools/cardboard_cached_operator_audit.py','matter/tools/cardboard_offline_conditioning_study.py','matter/tools/cardboard_paper_exact_capture.py','matter/tools/cardboard_solver_capture_analysis.py','matter/tools/cardboard_solver_pair_analyzer.py','matter/tools/test_cardboard_cached_operator_audit.py','matter/tools/test_cardboard_offline_conditioning_study.py','matter/tools/fem_kinematic_release_check.mm','matter/tools/fem_mutable_snapshot_check.mm','matter/tools/cardboard_source370_material_point_check.mm','matter/tools/fem_reference_check.mm','matter/tools/cardboard_contact_geometry_check.mm','matter/tools/cardboard_contact_geometry_check.metal','tools/verify_human_supported_tissue.sh','matter/tools/evidence/human-resting-20261005/supported-skin-ligaments-20261020/reproduction.py']
(p/'provenance.json').write_text(json.dumps({'source_base_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'executable_sha256':j['binary_sha256'],'metallib_sha256':sha(root/'build-integrated-validation-20261019/profile-build-039/matter/shaders/NumiMatter.metallib'),'suite_library_sha256':sha(root/'build-integrated-validation-20261019/profile-build-039/lib/libmetalrobo.dylib'),'source_sha256':{f:sha(root/f) for f in files},'actual_pose_geometry_audit_sha256':sha(args.geometry_audit or root/'matter/tools/evidence/human-resting-20261005/supported-skin-ligaments-20261020/geometry/native-009-actual-pose-26.json'),'host':platform.node()},indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('NUMI_HUMAN_') and k!='NM_VASCULAR_TRACE_ROOT'};env.update(j['environment'])
start=time.monotonic()
with (p/'stdout.log').open('w') as out,(p/'stderr.log').open('w') as err:
 r=subprocess.run(j['argv'],cwd=p,env=env,stdout=out,stderr=err)
result={'exit_code':r.returncode,'wall_s':time.monotonic()-start}
(p/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(r.returncode)
