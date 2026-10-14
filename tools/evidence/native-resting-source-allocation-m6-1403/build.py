from pathlib import Path
import json,subprocess,time,hashlib
r=Path("/Users/n/numi-lab-resting-qualification-publish-1395")
d=Path(__file__).resolve().parent
base=Path("/Users/n/numi-lab-anatomy-parents-build-1384-003")
old=json.loads((base/"build-report.json").read_text())
commands=[]
for c in old["commands"]:
    argv=[arg.replace(str(base),str(d)) for arg in c["argv"]]
    argv=[str(r/"apps/numilab_human_myosim_visual_probe.mm") if arg=="/Users/n/numi-lab-anatomy-parent-publication-1384/apps/numilab_human_myosim_visual_probe.mm" else arg for arg in argv]
    argv=[arg.replace("-I/Users/n/numi-lab-anatomy-parent-publication-1384/", "-I"+str(r)+"/") for arg in argv]
    t=time.monotonic()
    with (d/(c["phase"]+".log")).open("wb") as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    commands.append(dict(phase=c["phase"],argv=argv,returncode=result.returncode,wall_seconds=time.monotonic()-t))
    if result.returncode:
        (d/"failure.json").write_text(json.dumps(dict(commands=commands),indent=2)+"\n")
        raise SystemExit(result.returncode)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={str(p):sha(p) for p in r.joinpath("apps").glob("NumiHuman*.hpp")}
inputs[str(r/"apps/numilab_human_myosim_visual_probe.mm")]=sha(r/"apps/numilab_human_myosim_visual_probe.mm")
report=dict(commands=commands,binary_sha256=sha(d/"bin/numi-human-native"),input_sha256=inputs,
 source_revision=subprocess.check_output(["git","-C",str(r),"rev-parse","HEAD"],text=True).strip(),
 worktree_status=subprocess.check_output(["git","-C",str(r),"status","--short"],text=True),
 reused_gpu_runtime=str(base),native_gpu_confirmation=False)
(d/"build-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(dict(binary_sha256=report["binary_sha256"],commands=commands),indent=2))
