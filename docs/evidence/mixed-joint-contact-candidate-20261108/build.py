from pathlib import Path
import hashlib
import json
import subprocess
root=Path(__file__).resolve().parent
repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-joint-contact-candidate-artifacts-20261108")
work.mkdir(parents=True,exist_ok=True)
base=Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metal")
base_lib=Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metallib")
probe=Path("/Users/numi/Work/mixed-full-operator-artifacts-20261108/probe")
expected=(root.parent/"mixed-normal-block-20261108/inputs.sha256").read_text().splitlines()
assert hashlib.sha256(base.read_bytes()).hexdigest()==expected[0].split()[0]
assert hashlib.sha256(base_lib.read_bytes()).hexdigest()==expected[1].split()[0]
prior=root.parent/"mixed-full-operator-20261108"
link=json.loads((prior/"build.json").read_text())[2]["command"]
airs=[Path(x) for x in link if x.endswith(".air") and not x.endswith("/baseline.air")]
hashes="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in airs)
assert hashes==(prior/"air-inputs.sha256").read_text()
(root/"inputs.sha256").write_text(hashes+"".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (base,base_lib,probe,root/"joint_math.metal",root/"joint_update.metal")))
s=base.read_text()
helper=(root/"joint_math.metal").read_text()
marker="inline void mrWorldWave32SolvePacket("
assert s.count(marker)==1
s=s.replace(marker,helper+"\n"+marker,1)
start=s.index("        // Isolated research candidate: exact normal pair")
end=s.index("        threadgroup_barrier(mem_flags::mem_device);",start)
s=s[:start]+(root/"joint_update.metal").read_text()+s[end:]
shader=work/"joint.metal"; air=work/"joint.air"; library=work/"joint.metallib"
shader.write_text(s)
cmd1=["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)]
cmd2=[str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]
assert str(air) in cmd2 and str(library) in cmd2
records=[]
for command in (cmd1,cmd2):
 print("running",command[0],command[3],flush=True)
 p=subprocess.run(command,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append({"command":command,"returncode":p.returncode,"output":p.stdout})
 (root/"build.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode: raise SystemExit(p.returncode)
(root/"artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (shader,air,library)))
