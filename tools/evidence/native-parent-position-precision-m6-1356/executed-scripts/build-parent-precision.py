import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--source",required=True,type=Path)
ap.add_argument("--output",required=True,type=Path)
args=ap.parse_args()
W=Path("/Users/n/numi-lab-parent-precision-1356")
B=args.output.resolve()
source=args.source.resolve()
assert source.is_file()
B.mkdir(exist_ok=False)
(B/"bin").mkdir()
base=Path("/Users/n/numi-lab-complete-body-capture-build-1276-001")
for name in ("lib","matter","shaders"):
 (B/name).symlink_to(base/name,target_is_directory=True)
old=json.loads(Path("/Users/n/numi-lab-parent-bindings-build-1353-003/build-report.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[Path(__file__),source,W/"apps/numilab_human_myosim_visual_probe.mm",W/"apps/NumiHumanRestingVisual.hpp",W/"apps/NumiHumanRestingTissueParents.hpp",W/"include/metalrobo/numi_human_resting_visual_gpu.h",W/"matter/src/human_respiration.metal"]
report={"source":str(W),"source_head":subprocess.check_output(["git","-C",str(W),"rev-parse","HEAD"],text=True).strip(),"input_sha256":{str(p):sha(p) for p in inputs},"commands":[],"native_execution":False,"shader_scope":"Startup compilation of existing skin presentation kernel only; physical/physiology pipelines retain original metallib."}
for entry in old["commands"]:
 argv=[a.replace("/Users/n/numi-lab-parent-bindings-build-1353-003",str(B)).replace("/Users/n/numi-lab-parent-bindings-1353",str(W)) for a in entry["argv"]]
 if entry["phase"]=="compile":
  argv.insert(1,'-DNUMI_HUMAN_RESTING_SKIN_SOURCE="'+str(source)+'"')
 at=time.monotonic()
 with (B/(entry["phase"]+".log")).open("w") as f:
  r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 report["commands"].append({"phase":entry["phase"],"argv":argv,"returncode":r.returncode,"wall_seconds":time.monotonic()-at})
 (B/"build-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(entry["phase"],r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
report["inputs_unchanged"]=all(sha(p)==v for p,v in report["input_sha256"].items())
assert report["inputs_unchanged"]
report["binary_sha256"]=sha(B/"bin/numi-human-native")
(B/"build-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(report["binary_sha256"])
