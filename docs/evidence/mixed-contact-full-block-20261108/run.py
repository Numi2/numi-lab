import ast
import hashlib
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
repo=root.parents[2]
work=Path("/Users/numi/Work/mixed-contact-full-block-artifacts-20261108")
work.mkdir(parents=True,exist_ok=True)
source=Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metal")
control=Path("/Users/numi/Work/mixed-normal-block-artifacts-20261108/candidate.metallib")
probe=Path("/Users/numi/Work/mixed-full-operator-artifacts-20261108/probe")
expected=(root.parent/"mixed-normal-block-20261108/inputs.sha256").read_text().splitlines()
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected[0].split()[0]
assert hashlib.sha256(control.read_bytes()).hexdigest()==expected[1].split()[0]
old=root.parent/"mixed-full-operator-20261108"
link=json.loads((old/"build.json").read_text())[2]["command"]
airs=[Path(x) for x in link if x.endswith(".air") and not x.endswith("/baseline.air")]
hashes="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in airs)
assert hashes==(old/"air-inputs.sha256").read_text()
(root/"inputs.sha256").write_text(hashes+"".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n" for p in (source,control,probe)))
module=ast.parse((old/"prepare.py").read_text())
instrument=next(ast.literal_eval(node.value) for node in module.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="instrument" for t in node.targets))
s=source.read_text()
start=s.index("    float laneNormalResidual = 0.0f;",s.index("const float groupError = simd_max(max(finalNormal, finalTangent));"))
needle="        const float limitU =\n            cone.effectiveFrictionU * contact.impulses.x;"
pos=s.index(needle,start)
def run(name,cmd,raw):
 print(name,flush=True)
 p=subprocess.run([str(x) for x in cmd],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 (root/(name+".stderr")).write_text(p.stderr)
 (root/(name+(".json" if raw else ".log"))).write_text(p.stdout)
 if p.returncode: raise RuntimeError(name+" exit "+str(p.returncode))
run("control",[probe,control,"2"],True)
for axis in range(3):
 name="axis%d"%axis
 shader=work/(name+".metal");air=work/(name+".air");library=work/(name+".metallib")
 shader.write_text(s[:pos]+instrument.replace("AXIS_VALUE",str(axis)+"u")+s[pos:])
 run(name+"-compile",["xcrun","-sdk","macosx","metal","-std=metal4.0","-O3","-I",repo/"include","-c",shader,"-o",air],False)
 command=[str(air) if x.endswith("/baseline.air") else str(library) if x.endswith("/baseline.metallib") else x for x in link]
 assert str(air) in command and str(library) in command
 run(name+"-link",command,False)
 with (root/"artifacts.sha256").open("a") as f:
  for p in (shader,air,library): f.write(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p)+"\n")
 run(name,[probe,library,"2"],True)
