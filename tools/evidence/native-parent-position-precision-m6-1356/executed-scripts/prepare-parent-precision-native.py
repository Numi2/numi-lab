import argparse,hashlib,json,shutil,sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--build",type=Path,required=True)
ap.add_argument("--shader",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
W=Path("/Users/n/numi-lab-parent-precision-1356").resolve()
B=args.build.resolve();S=args.shader.resolve();O=args.output.resolve()
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/forearm054-native-diagnostic-009-refined085").resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
parent=json.loads((D/"baseline/run-declaration.json").read_text())
execution=json.loads((D/"baseline/execution.json").read_text())
assert execution["returncode"]==0 and not execution["changed_inputs"]
build=json.loads((B/"build-report.json").read_text())
assert build["inputs_unchanged"] and all(c["returncode"]==0 for c in build["commands"])
assert sha(B/"bin/numi-human-native")==build["binary_sha256"]
for p,v in build["input_sha256"].items():assert sha(p)==v,p
assert str(S) in build["input_sha256"]
prep=json.loads(Path(str(S)+".receipt.json").read_text())
argv0=list(parent["argv"]);mi=argv0.index("numilab_human.resting_run")
env=dict(x.split("=",1) for x in argv0[2:mi-2])
tail=argv0[mi+1:]
for flag,value in (("--output",str(O/"baseline/native-run")),("--build",str(B)),("--lab",str(W))):
 tail[tail.index(flag)+1]=value
sys.path.insert(0,env["PYTHONPATH"])
from numilab_human import resting_run as rr
parser=argparse.ArgumentParser();rr.add_arguments(parser)
native,assets=rr.command(parser.parse_args(tail))
assert Path(native[0]).resolve()==B/"bin/numi-human-native"
env.update(NUMI_BUILD_DIR=str(B),NUMI_LAB_ROOT=str(W),NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(O/"baseline/native-run/common-field-failure.json"))
env["DYLD_LIBRARY_PATH"]=":".join(str((B/p).resolve()) for p in ("lib","matter"))
pins=dict(parent["immutable_assets"])
def pin(p,expected=None):
 p=Path(p).resolve();v=sha(p)
 assert expected is None or v==expected,str(p)
 if str(p) in pins:assert pins[str(p)]==v,str(p)
 pins[str(p)]=v
for p,v in list(pins.items()):pin(p,v)
for p,v in assets.items():pin(p,v)
for p,v in build["input_sha256"].items():pin(p,v)
for p in (Path(__file__),W/"tools/prepare_human_resting_skin_source.py",S,Path(str(S)+".receipt.json"),B/"build-report.json",D/"baseline/execution.json",D/"baseline/run-declaration.json"):pin(p)
# The prep receipt carries each preprocessor dependency; bind all existing
# file entries independent of their report grouping.
def dependencies(v):
 if isinstance(v,dict):
  if isinstance(v.get("path"),str) and isinstance(v.get("sha256"),str) and Path(v["path"]).is_file():pin(v["path"],v["sha256"])
  for x in v.values():dependencies(x)
 elif isinstance(v,list):
  for x in v:dependencies(x)
dependencies(prep)
O.mkdir();(O/"baseline").mkdir()
shutil.copyfile(D/"launch_arm.py",O/"launch_arm.py")
launcher=O/"launch_arm.py"
declaration=dict(parent)
declaration.update(immutable_assets=pins,argv=["/usr/bin/env","-i",*[k+"="+v for k,v in sorted(env.items())],"/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run",*tail],
 launch_adapter={"path":str(launcher),"sha256":sha(launcher)},owner_cli_preview={"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},
 qualification="Experimental source-compiled skin presentation comparison on unchanged009 anatomy. Physical and physiological pipelines retain the original libraries. Requires actual capture checks; no anatomical admission.",
 presentation_source={"path":str(S),"sha256":sha(S),"scope":"existing nm_human_resting_skin pipeline only"})
dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(declaration,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launcher),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(O/"baseline/native-run")}}},indent=2)+"\n")
(O/"preparation.json").write_text(json.dumps({"status":"prepared_experimental_not_admitted","parent_scene":str(D),"child_anatomy_reused_unchanged":True,"source_shader":str(S),"source_shader_sha256":sha(S),"binary_sha256":sha(B/"bin/numi-human-native"),"declaration_sha256":sha(dp),"immutable_pin_count":len(pins),"physical_and_physiological_libraries_unchanged":True},indent=2)+"\n")
print(json.dumps({"output":str(O),"pins":len(pins),"declaration_sha256":sha(dp),"launch":"/Users/n/numi-human-prep-venv-20261005/bin/python3.13 "+str(launcher)+" baseline"}))
