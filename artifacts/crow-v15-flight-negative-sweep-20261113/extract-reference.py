from pathlib import Path
import gzip,json,hashlib,sys
p=Path(sys.argv[1]);out=Path(sys.argv[2]);meta=json.loads((p/"run.stdout").read_text())["first_episode_navigation"];manifest=json.loads((p/"stderr-parts.json").read_text());src=p/"run.stderr.gz";h=hashlib.sha256()
with src.open("rb") as f:
 for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
assert h.hexdigest()==manifest["sha256"]
state={}
with gzip.open(src,"rt") as f:
 for line in f:
  if not line.startswith("CROW_SUPPORT_SAMPLE "):continue
  x=json.loads(line.split(" ",1)[1]);e=x["lane"];t=x["step"]
  if e in (2,3) and t<=meta["steps_by_environment"][e]:
   key=f"{e},{t}";assert key not in state;state[key]={n:x[n] for n in ("q","v","policy_action","contacts")}
for e in (2,3):assert all(f"{e},{t}" in state for t in range(1,meta["steps_by_environment"][e]+1))
obj={"source_commit":"4e3d6735a60acbb692aea97d84bed43d25d50e42","source_log_sha256":manifest["sha256"],"ends":meta["steps_by_environment"],"states":state}
with out.open("wb") as raw:
 with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as g:g.write((json.dumps(obj,separators=(",",":"))+"\n").encode())
print("extracted",len(state),"verified records")
