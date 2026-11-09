from pathlib import Path
import hashlib,json,secrets,subprocess
R=Path.cwd();A=R/'artifacts/crow-unused-layout-confirmation-round3-20261109'
def sha(path):
 h=hashlib.sha256()
 with (R/path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
files=['src/metal/CrowFlightControllerV1.metal','src/metal/LocomotionTask.metal','src/c_api.cpp','src/core/TaskProgram.cpp','include/metalrobo/task_program_types.h','include/metalrobo/c_api.h','apps/task_rollout.swift','apps/task_train.swift','bindings/swift/MetalRoboTaskRollout.swift','assets/crow_navigation_course/crow-navigation.active-head.visual-observation.json','build/bin/metalrobo_task_rollout','build/lib/libmetalrobo.dylib','build/shaders/MetalRobo.metallib','build/crow-four-native-observer-20261109/libmetalrobo.dylib','build/crow-four-expert-native-integration-20261109/route.policypack','build/crow-four-expert-native-integration-20261109/landing.policypack','artifacts/crow-four-expert-native-integration-20261109/selection.json','artifacts/crow-four-expert-native-integration-20261109/contract.json','artifacts/crow-unused-layout-confirmation-round3-20261109/run.py','artifacts/crow-unused-layout-confirmation-round3-20261109/register.py']
seeds=[]
while len(seeds)<2:
 s=secrets.randbelow(2**32-1)+1
 if s in seeds:continue
 p=subprocess.run(['git','grep','-I','-F','-q','--',str(s),'--','.'],cwd=R)
 if p.returncode not in (0,1):raise RuntimeError('seed search failed')
 if p.returncode==1:seeds.append(s)
frozen={p:sha(p) for p in files}
assert frozen['src/metal/CrowFlightControllerV1.metal']=='aa2fe10901cb0494d80ae445902303a17ce0e73d030072bab09461d8f3593ebe'
report={'status':'preregistered before executing either layout','layout_authoring_commit':'086bdb7a','layouts':dict(zip(['reserved-final-g','reserved-final-h'],seeds)),'seed_check':'independently generated uniform 32-bit positive seeds; git grep across tracked text found neither prior to registration','prior_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'bound_sha256':frozen,'controls':'16 full starts per layout, teacher off, first episodes only, no scheduled or intermediate resets; 3000 control steps; native v15 body and frozen flight law','success':'all five waypoints in order and at least 25 terminal accepted steps of foot-only support with positive upward impulse and low velocity; no physical failure and no simulator failure','prediction':'at least 15 supported finishes out of 16 on each layout','failure_reporting':'every attempted lane, first fatal contact or last sampled state if stalled, and exact box distance sampled at each accepted 20 ms control-step endpoint; perch excluded from obstacle clearance','development_results':'141/144 supported finishes in nine development seed sets across seven layouts, each set at least 15/16; known failures include slalom contact, body-perch contact and a waypoint stall','caveat':'two unused nominal layouts and one unused seed per layout provide bounded generalization evidence, not universal reliability; policy receives simulator-issued goal coordinates alongside visual and proprioceptive observations'}
(A/'plan.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'layouts':report['layouts'],'prior_revision':report['prior_revision']}),flush=True)
