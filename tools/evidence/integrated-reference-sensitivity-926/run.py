from pathlib import Path
import os,sys,json,subprocess,time,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005');D=E/'integrated-reference-sensitivity-926';L=Path('/Users/n/numi-human-performance-source-014')
sys.path.insert(0,str(L/'matter/tools'))
from resting_intervention_study import NATIVE_310S_REQUIRED_ENVIRONMENT,NATIVE_310S_DISABLED_EXPERIMENTS
p=json.loads((D/'protocol.json').read_text());env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','DYLD_'))};env.update(NATIVE_310S_REQUIRED_ENVIRONMENT);env.update(NATIVE_310S_DISABLED_EXPERIMENTS)
env.update(NUMI_HUMAN_ROOT='/Users/n/numi-human-lung-triangulation-candidate-003',NUMI_LAB_ROOT=str(L),NUMI_BUILD_DIR='/Users/n/numi-human-source-capture-build-016')
records=[]
for arm in p['arms']:
 if (D/'stop-before-next-arm').exists():print('Stopped between arms at explicit root request',flush=True);break
 out=D/'arms'/arm['arm_id'];out.parent.mkdir(exist_ok=True)
 assert not out.exists()
 args=[str(L/'tools/numi'),'human-resting','--body-scene',str(E/'common-atlas-skin-composition-907/resting-scene/resting-supine-scene.manifest.json'),'--anatomy-receipt',str(E/'lung-choroid-composition-924/final-v2/resting-anatomy-receipt.json'),'--tendon',str(E/'tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon'),'--lab',str(L),'--build','/Users/n/numi-human-source-capture-build-016','--output',str(out),'--circulation',str(E/'reference-circulation-001/resting_reference_lv15.native.v3.json'),'--respiration',arm['parameters'],'--seconds','24','--dt','0.002','--dimension','512','--postural-activation-cap','0.01','--release-initialization','--rigid-hands','--contact-iterations','64','--inspection-tour','--inspection-period-seconds','2.5']
 (D/(arm['arm_id']+'.launch.json')).write_text(json.dumps({'argv':args,'environment':{k:v for k,v in env.items() if k.startswith('NUMI_')}},indent=2)+'\n')
 print(json.dumps({'event':'start','arm':arm['arm_id']}),flush=True);start=time.monotonic()
 with (D/(arm['arm_id']+'.driver.log')).open('wb') as f:r=subprocess.run(args,cwd=L,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({'arm_id':arm['arm_id'],'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'output':str(out)})
 (D/'run-index.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records[-1]),flush=True)
print('Completed requested arm loop; each individual exit status retained',flush=True)
