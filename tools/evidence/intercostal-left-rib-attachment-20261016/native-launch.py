"""One-step native diagnostic of reproduced rib/intercostal and skin topology; no duration admission."""
import argparse,hashlib,json,os,sys
from pathlib import Path
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
sys.path.insert(0,"/Users/n/numi-human-contact-refit-publish-20261016/src")
from numilab_human import resting_run
P=D/"intercostal-extended-rib-loader-legacy-scene-native-confirmation-1031"
scene=D/"intercostal313-local-diagonal-existing-native-owner-bindings-973/resting-supine-scene.manifest.json"
receipt=D/"intercostal313-local-diagonal-existing-native-owner-bindings-973/resting-anatomy-receipt.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(scene)=="5fd2cf1fe5718f4be3a18b1b2cbe186349e2ea6b433b9b6e35bc8d29b6425523"
assert sha(receipt)=="d6a6154985d54f5eeb49a5aef54928d9fd03fc53aae3003ecfa1e91ad417174e"
assert sha(D/"rib50-existing-tendon-owner-equivalence-953/compiled/numi-human-tendon-attachments.nhtendon")=="4cfaf9e26e995798ee2f78c49218894ece9af9137fb167c74c684535ac2048c3"
parser=argparse.ArgumentParser();resting_run.add_arguments(parser)
args=parser.parse_args([
"--body-scene",str(scene),"--anatomy-receipt",str(receipt),
"--tendon",str(D/"rib50-existing-tendon-owner-equivalence-953/compiled/numi-human-tendon-attachments.nhtendon"),
"--lab","/Users/n/numi-lab-resting-qualification-publish-1395",
"--build","/Users/n/numi-human-retained-delivery-20261009/native-intercostal-all-left-rib-launch-1030",
"--output",str(P/"native-run"),
"--circulation","/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json",
"--respiration","/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json",
"--seconds",".002","--dt",".002","--dimension","512","--minimum-free-gib","12",
"--inspection-tour","--inspection-period-seconds","2","--postural-activation-cap",".01","--release-initialization",
"--mtp-passive-stiffness-nm-per-rad","2","--rigid-hands","--contact-iterations","64"])
args.capture_steps=[0]
env=json.loads((D/"aortic7-connected-network036-deltoid868b-parent7-rest-registration-native-emission-003/native-run/invocation.json").read_text())["environment"]
os.environ.update(env);os.environ["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(args.output/"common-field-failure.json")
P.mkdir(exist_ok=False)
(P/"preparation.json").write_text(json.dumps({"scope":__doc__,"runner_sha256":sha(__file__),"scene_sha256":sha(scene),"anatomy_receipt_sha256":sha(receipt),"anatomical_and_duration_admission":False},indent=2)+"\n")
raise SystemExit(resting_run.run(args))
