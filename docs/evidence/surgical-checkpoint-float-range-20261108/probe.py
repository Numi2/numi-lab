from pathlib import Path
import hashlib,json,subprocess,sys,tempfile
build=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parent
probe=build/"bin/metalrobo_dual_psm_suture_handoff_probe"
source=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else build/"testing/suture/tissue-checkpoint/tissue-rest.tsv"
raw=source.read_bytes()
lines=raw.decode().splitlines(keepends=True)
matter=[line.split("\t")[1] for line in lines if line.startswith("matter_snapshot\t")]
assert len(matter)==1
payload=source.parent/matter[0]
payloadhash=hashlib.sha256(payload.read_bytes()).hexdigest()
def execute(path):
 p=subprocess.run([str(probe),"--tissue-checkpoint-restore-only","--resume-tissue-checkpoint","tissue-rest",str(path)],cwd=build,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 return {"returncode":p.returncode,"output":p.stdout}
result={"control":execute(source),"cases":{}}
assert result["control"]["returncode"]==0 and "physics_advanced=no" in result["control"]["output"]
with tempfile.TemporaryDirectory(prefix="suture-float-range-",dir="/Users/numi/Work") as tmp:
 d=Path(tmp);(d/matter[0]).symlink_to(payload)
 for key in ("q","v","needle_position","needle_orientation_xyzw","needle_linear_velocity","needle_angular_velocity","thread_position","thread_velocity","thread_twist"):
  altered=lines.copy()
  row=next(i for i,line in enumerate(altered) if line.startswith(key+"\t"))
  fields=altered[row].rstrip("\n").split("\t")
  fields[2 if key.startswith("thread_") else 1]="1e100"
  altered[row]="\t".join(fields)+"\n"
  path=d/(key+".tsv");path.write_text("".join(altered))
  result["cases"][key]=execute(path)
assert source.read_bytes()==raw
assert hashlib.sha256(payload.read_bytes()).hexdigest()==payloadhash
result["checkpoint_sha256"]=hashlib.sha256(raw).hexdigest()
result["matter_sha256"]=payloadhash
(root/(sys.argv[2]+".json")).write_text(json.dumps(result,indent=2)+"\n")
for key,data in result["cases"].items():
 print(key, "exit",data["returncode"],"no_step_success", "physics_advanced=no" in data["output"],
       "tail", data["output"].splitlines()[-1:] )
