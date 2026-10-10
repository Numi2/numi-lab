from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys
BASE=Path(__file__).parent
ROOT=BASE.parent.parent
HUMAN=Path('/Users/n/numi-human-force-audit-provenance-1254')
SOURCE=Path('/Users/n/numi-human-accepted-force-observer-1253')
BUILD=Path('/Users/n/numi-human-accepted-force-build-1253-002')
LAB=Path('/Users/n/numi-human-performance-source-014')
PARENT=ROOT/'source-seam-connectivity-1247/native-fhl-10s-preparation-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def identity(p):
 return {'path':str(p),'revision':subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).splitlines()}
def opt(a,k):return a[a.index(k)+1]
assert sha(PARENT/'run-declaration.json')=='d30e512fa186a70b852fadbc704e1ae7fd25d65f302d37e74bf61be920b5d587'
parent=json.loads((PARENT/'run-declaration.json').read_text())
execution=json.loads((PARENT/'execution.json').read_text())
assert execution['returncode']==0 and not execution['changed_inputs']
assert sha(BUILD/'bin/numi-human-native')=='b53e781562a9b6671bfb0f0142461a230d5f118fcffb499cbe36cf58ac03d0f6'
assert sha(HUMAN/'src/numilab_human/resting_run.py')=='187c8a37145ad74fd4c7880ca46177e76aaf09cd5b233816a718467679ac910b'
assert sha(SOURCE/'apps/numilab_human_myosim_visual_probe.mm')=='f089fdff77f0b47371f2d9553004ef1a81a3b248ab5ca3c3c4aa3f4508117e7a'
source_id=identity(SOURCE);human_id=identity(HUMAN)
assert source_id['revision']=='b18b116e25b3525b468848208b3f86ad5a05d104'
assert human_id['revision']=='553bb21a4c8159752b23b81e34bc3d761383a582'
scene=Path(parent['body_scene']);receipt=Path(parent['anatomy_receipt'])
assert json.loads(scene.read_text()).get('bed',{}).get('heightfield') is None
tool=LAB/'tools/numi';a=parent['argv'];ti=a.index(str(tool));cli=a[ti+1:]
env=dict(x.split('=',1) for x in a[2:ti])
sys.path.insert(0,str(HUMAN/'src'))
from numilab_human.resting_run import command,invocation_environment
guard={'schema':'numi.human.native-smoke-launch-guard.v1','launcher_path':str(BASE/'launch_arm.py'),'launcher_sha256':sha(BASE/'launch_arm.py'),'arms':{}}
for arm in ('control','window'):
 out=BASE/arm/'native-run';out.parent.mkdir(exist_ok=False)
 args=argparse.Namespace(body_scene=scene,anatomy_receipt=receipt,tendon=Path(opt(cli,'--tendon')),lab=LAB,build=BUILD,output=out,circulation=Path(opt(cli,'--circulation')),respiration=Path(opt(cli,'--respiration')),seconds=1.024,dt=.002,dimension=512,mechanics_only=False,inspection_tour=True,inspection_period_seconds=8.0,drive_intervention=None,postural_activation_cap=.01,release_initialization=True,contoured_bed=False,upper_passive_joints=False,hip_capsule_reference=False,hip_capsule_scale=None,rigid_hands=True,contact_iterations=64)
 native,assets=command(args);native=list(native)
 e=dict(env);e.update(NUMI_BUILD_DIR=str(BUILD),NUMI_HUMAN_ROOT=str(HUMAN),NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(out/'common-field-failure.json'),NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS='0,512',NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT='1',NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP='464',NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP='512',NUMI_HUMAN_ACCEPTED_FORCE_AUDIT='0' if arm=='control' else '1')
 c=list(cli);c[c.index('--output')+1]=str(out);c[c.index('--seconds')+1]='1.024'
 argv=['/usr/bin/env','-i',*[k+'='+v for k,v in e.items()],str(tool),*c]
 for p in [Path(__file__),BASE/'launch_arm.py',BASE.parent/'build-002/build-report.json',BASE.parent/'build-002/build.py',PARENT/'run-declaration.json',PARENT/'execution.json',PARENT/'native-run/run-metadata.json',tool,LAB/'numi/commands/human-resting',HUMAN/'src/numilab_human/resting_run.py',HUMAN/'src/numilab_human/model.py',HUMAN/'tests/test_resting_run.py',HUMAN/'pyproject.toml',SOURCE/'apps/numilab_human_myosim_visual_probe.mm',SOURCE/'apps/NumiHumanAcceptedQAuditWindow.hpp',SOURCE/'include/metalrobo/numi_human_accepted_force_observer.hpp',SOURCE/'tests/numi_human_accepted_force_observer_test.cpp',SOURCE/'CMakeLists.txt',Path('/usr/bin/python3')]:
  assets[str(p.resolve())]=sha(p)
 # The parent binds the full model lineage; retain it as identity evidence rather
 # than copying obsolete build pins into the actively selected runtime set.
 declaration={'schema':parent['schema'],'status':'prepared_unlaunched','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'body_scene':str(scene),'anatomy_receipt':str(receipt),'seconds':1.024,'dt':.002,'accepted_steps':512,'capture_steps':[0,512],'contact_iterations':64,'flat_support_unchanged':True,'hip_reference_enabled':False,'rigid_hands_enabled':True,'release_initialization':True,'postural_activation_cap':.01,'immutable_assets':dict(sorted(assets.items())),'launch_adapter':{'path':str(BASE/'launch_arm.py'),'sha256':sha(BASE/'launch_arm.py')},'owner_cli_preview':{'native_argv':native,'owner':str(HUMAN/'src/numilab_human/resting_run.py'),'recorded_invocation_environment':invocation_environment(e)},'source_identity':source_id,'human_identity':human_id,'parent_declaration':{'path':str(PARENT/'run-declaration.json'),'sha256':sha(PARENT/'run-declaration.json')},'q_audit':{'enabled':True,'first_accepted_step':464,'last_accepted_step':512,'expected_rows':49},'force_audit':{'enabled':arm=='window','expected_rows':49 if arm=='window' else 0,'source':'already collected GPU result, observer only'},'physical_scope':{'same_current_FHL_scene_and_all_physical_control_settings_between_arms':True,'only_force_audit_and_output_paths_differ_between_arms':True,'same_bounded_Q_submission_schedule_between_arms':True,'native_runtime_library_sha256':sha(BUILD/'lib/libmetalrobo.dylib')},'qualification_boundary':'1.024 s instrumentation parity only, not long-horizon physiology, anatomy acceptance, drift diagnosis, or performance qualification'}
 write(out.parent/'run-declaration.json',declaration)
 guard['arms'][arm]={'run_declaration':str(out.parent/'run-declaration.json'),'run_declaration_sha256':sha(out.parent/'run-declaration.json'),'native_output':str(out)}
write(BASE/'launch-guard.json',guard)
print(json.dumps({'status':'prepared_unlaunched','arms':guard['arms']},indent=2))
