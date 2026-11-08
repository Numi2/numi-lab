from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
repo=root.parents[2]
prior=root.parent/"mixed-joint-contact-candidate-20261108"
work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108");work.mkdir(parents=True,exist_ok=True)
base=Path("/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/joint.metal")
assert hashlib.sha256(base.read_bytes()).hexdigest()==(prior/"v2-artifacts.sha256").read_text().splitlines()[0].split()[0]
s=base.read_text()
marker="inline void mrWorldWave32SolvePacket("
assert s.count(marker)==1
s=s.replace(marker,(root/"joint_band.metal").read_text()+"\n"+marker,1)
start=s.index("        // Research-only joint block.")
end=s.index("        threadgroup_barrier(mem_flags::mem_device);",start)
s=s[:start]+(root/"joint_update_long.metal").read_text()+s[end:]
shader=work/"long.metal";air=work/"long.air";library=work/"long.metallib";shader.write_text(s)
link=json.loads((root.parent/"mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
airs=[Path(x) for x in link if x.endswith(".air") and not x.endswith("/baseline.air")]
hashes="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in airs)
assert hashes==(root.parent/"mixed-full-operator-20261108/air-inputs.sha256").read_text()
(root/"inputs.sha256").write_text(hashes+"".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (base,root/"joint_band.metal",root/"joint_update_long.metal")))
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",str(repo/"include"),"-c",str(shader),"-o",str(air)],
 [str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]]
records=[]
for cmd in cmds:
 print(cmd[0],cmd[3],flush=True)
 p=subprocess.run(cmd,cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append({"command":cmd,"returncode":p.returncode,"output":p.stdout})
 (root/"build.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise SystemExit(p.returncode)
(root/"artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (shader,air,library)))
