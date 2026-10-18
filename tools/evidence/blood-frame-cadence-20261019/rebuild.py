#!/usr/bin/env python3
import argparse, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

p=argparse.ArgumentParser(description="Rebuild the pinned one-frame cadence probe")
p.add_argument("--output-dir",type=Path,required=True)
p.add_argument("--run",action="store_true",help="also run once through the exclusive wrapper")
p.add_argument("--header-source",type=Path,help="alternate exact header path for reproducing a pre-publication build")
a=p.parse_args()
repo=Path(__file__).resolve().parents[3]
evidence=Path(__file__).resolve().parent
manifest=json.loads((evidence/"build-receipt.json").read_text())
inputs=json.loads((evidence/"external-inputs.json").read_text())
header_path=a.header_source.resolve() if a.header_source else repo/inputs["header"]["path"]
for name,entry in inputs.items():
    path=header_path if name=="header" else Path(entry["path"])
    if not path.is_file() or sha(path)!=entry["sha256"]:
        raise SystemExit(f"pinned {name} input missing or hash mismatch: {path}")
a.output_dir=a.output_dir.resolve()
a.output_dir.mkdir(parents=True,exist_ok=False)
with tempfile.TemporaryDirectory(prefix="numi-cadence-stage-") as td:
    stage=Path(td)
    source_dir=stage/"cadence-probe"
    staged_header=stage/"source/matter/tools/human_resting_runtime.hpp"
    source_dir.mkdir(parents=True)
    staged_header.parent.mkdir(parents=True)
    staged_header.symlink_to(header_path)
    staged_source=source_dir/"cadence_probe.mm"
    shutil.copy2(evidence/"cadence_probe.mm",staged_source)
    old_source=manifest["source"]
    old_build=Path(manifest["compile_argv"][manifest["compile_argv"].index("-o")+1]).parent
    compile_argv=[x.replace(str(old_build),str(a.output_dir)) for x in manifest["compile_argv"]]
    compile_argv=[str(staged_source) if x==old_source else x for x in compile_argv]
    link_argv=[x.replace(str(old_build),str(a.output_dir)) for x in manifest["link_argv"]]
    for i,x in enumerate(compile_argv):
        if x.endswith("/cadence_probe_grid.mm"): compile_argv[i]=str(staged_source)
    for label,argv in (("compile",compile_argv),("link",link_argv)):
        log=a.output_dir/f"{label}.log"
        with log.open("wb") as f: result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode: raise SystemExit(f"{label} failed ({result.returncode}); see {log}")
    binary=a.output_dir/"cadence_probe"
    print(json.dumps({"binary":str(binary),"binary_sha256":sha(binary)},indent=2))
    if a.run:
        network=Path(inputs["network"]["path"])
        config=Path(inputs["configuration"]["path"])
        material=Path(inputs["material"]["path"])
        matter=Path(inputs["matter_source"]["path"])
        env=os.environ.copy()
        env["DYLD_LIBRARY_PATH"]=str(Path(inputs["runtime_library"]["path"]).parent)
        env["NUMI_MATTER_TEST_SOURCE"]=str(matter)
        env["NUMI_MATTER_TEST_SOURCE_SHA256"]=inputs["matter_source"]["sha256"]
        for key in ("NUMI_HUMAN_BLOOD_MPM_GRID_BOUNDS_M","NUMI_HUMAN_BLOOD_MPM_ENABLED","NUMI_HUMAN_BLOOD_MPM_MATERIAL","NUMI_HUMAN_BLEEDING_ENABLED"):
            env.pop(key,None)
        command=["/Users/numi/numi-human-prep-venv-20261005/bin/python3",inputs["wrapper"]["path"],"--","/usr/bin/time","-l",str(binary),str(network),str(config),str(material)]
        with (a.output_dir/"run.log").open("wb") as f: result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(f"run_exit_code={result.returncode} log={a.output_dir/'run.log'}")
        if result.returncode: raise SystemExit(result.returncode)
