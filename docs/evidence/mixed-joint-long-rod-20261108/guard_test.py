from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent;work=Path("/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108");work.mkdir(parents=True,exist_ok=True)
shader=work/"guard.metal";air=work/"guard.air";lib=work/"guard.metallib";exe=work/"guard_test"
shader.write_text("#include <metal_stdlib>\nusing namespace metal;\n"+(root/"joint_band.metal").read_text()+"\nkernel void guard_check(device const uint* a [[buffer(0)]],device uint* b [[buffer(1)]],uint i [[thread_position_in_grid]]) {const uint k=6u*i;b[i]=mrResearchRodWitnessesFitSelectedBand(a[k],a[k+1],a[k+2],a[k+3],a[k+4],a[k+5])?1u:0u;}\n")
cmds=[["xcrun","-sdk","macosx","metal","-std=metal4.0","-c",str(shader),"-o",str(air)],
 ["xcrun","-sdk","macosx","metallib",str(air),"-o",str(lib)],
 ["xcrun","clang++","-std=c++20","-fobjc-arc",str(root/"guard_test.mm"),"-framework","Foundation","-framework","Metal","-o",str(exe)],
 [str(exe),str(lib)]]
records=[]
for cmd in cmds:
 p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 records.append({"command":cmd,"returncode":p.returncode,"output":p.stdout})
 (root/"guard-test.json").write_text(json.dumps(records,indent=2)+"\n")
 if p.returncode:raise SystemExit(p.returncode)
print(records[-1]["output"])
(root/"guard-artifacts.sha256").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (shader,air,lib,exe)))
