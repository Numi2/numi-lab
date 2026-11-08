import hashlib,json,os,struct,subprocess
from pathlib import Path
root=Path("/Users/numi/numi-reconciliation-build-20261108");assets=Path("/Users/numi/numi-v5-admission-artifacts-20261107/v5-continuation-aggregate-status-diagnostic")
payload=bytearray((assets/"synthetic.nhtbind").read_bytes());assert payload[:8]==b"NHTBIND2" and struct.unpack_from("<I",payload,260)[0]==20
struct.pack_into("<I",payload,260,25);altered=Path("owner-mismatch.nhtbind").resolve();altered.write_bytes(payload)
names=["rigid.nhrigid","muscle.nhmyo","support.nhcnt","equality.nheq","retained-visual.mrvpack","retained-vision.json","synthetic.nmatterpack"]
cmd=[str(root/"numi_human_costal_v5_admission_probe")]+[str(assets/n) for n in names]+[str(altered),str(assets/"synthetic.nhcartilage"),str(root/"shaders/MetalRobo.metallib")]
env=os.environ.copy();env["MRNX_MATTER_METALLIB"]=str(root/"matter/shaders/NumiMatter.metallib");env["MRNX_RUNTIME_DIAGNOSTICS"]="1"
runs=[]
for i in range(2):
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,env=env);runs.append(dict(command=cmd,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
 out={"changed_offset":260,"original_owner":20,"altered_owner":25,"binding_sha256":hashlib.sha256(payload).hexdigest(),"runs":runs};Path("owner-mismatch.json").write_text(json.dumps(out,indent=2)+"\n");print("run",i,"exit",p.returncode,flush=True)
assert all(r["exit"]==1 and "v5 creation failed status=3" in r["stderr"] and "costal attachment admission failed" in r["stderr"] for r in runs)
assert runs[0]["stdout"]==runs[1]["stdout"] and runs[0]["stderr"]==runs[1]["stderr"]
print("binding/world owner mismatch rejected after candidate fingerprint recomputation",flush=True)
