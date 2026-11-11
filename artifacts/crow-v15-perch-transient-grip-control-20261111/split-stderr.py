from pathlib import Path
import hashlib,json
D=Path("artifacts/crow-v15-perch-transient-grip-control-20261111/teacher-20261126")
src=D/"run.stderr.gz";overall=hashlib.sha256();parts=[]
with src.open("rb") as f:
 i=0
 while True:
  data=f.read(32*1024*1024)
  if not data:break
  path=D/("run.stderr.gz.part%02d"%i)
  if path.exists():raise RuntimeError("part already exists: "+str(path))
  path.write_bytes(data);overall.update(data);parts.append({"name":path.name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()});i+=1
(D/"stderr-parts.json").write_text(json.dumps({"compressed_sha256":overall.hexdigest(),"bytes":sum(x["bytes"] for x in parts),"parts":parts},indent=2)+"\n")
print(len(parts),"parts",src.stat().st_size,"bytes")
