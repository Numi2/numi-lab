from pathlib import Path
import sys,os,subprocess
E=Path("/Users/n/numi-human-resting-evidence-20261005")
A=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005")
L=Path("/Users/n/numi-human-performance-source-014")
sys.path.insert(0,str(L/"matter/tools"))
from resting_intervention_study import NATIVE_310S_REQUIRED_ENVIRONMENT,NATIVE_310S_DISABLED_EXPERIMENTS
env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","DYLD_"))}
env.update(NATIVE_310S_REQUIRED_ENVIRONMENT)
env.update(NATIVE_310S_DISABLED_EXPERIMENTS)
env.update(NUMI_HUMAN_ROOT="/Users/n/numi-human-lung-triangulation-candidate-003",
           NUMI_LAB_ROOT=str(L),NUMI_BUILD_DIR="/Users/n/numi-human-source-capture-build-016")
out=E/"native-skin-source-capture-912"
env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0,63"
env["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"]="1"
env["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT"]="1"
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-field-failure.json")
args=[str(L/"tools/numi"),"human-resting",
 "--body-scene",str(E/"common-atlas-skin-composition-907/resting-scene/resting-supine-scene.manifest.json"),
 "--anatomy-receipt",str(E/"common-atlas-skin-composition-907/resting-anatomy/resting-anatomy-receipt.json"),
 "--tendon",str(E/"tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon"),
 "--lab",str(L),"--build","/Users/n/numi-human-source-capture-build-016",
 "--output",str(out),"--circulation",str(E/"reference-circulation-001/resting_reference_lv15.native.v3.json"),
 "--respiration",str(E/"integrated-anatomy-engineering-lung-edge-collapse-761/resting-reference-respiration.json"),
 "--seconds","0.128","--dt","0.002","--dimension","512",
 "--postural-activation-cap","0.01","--release-initialization","--rigid-hands",
 "--contact-iterations","64","--inspection-tour","--inspection-period-seconds","2.5"]
raise SystemExit(subprocess.call(args,cwd=L,env=env))
