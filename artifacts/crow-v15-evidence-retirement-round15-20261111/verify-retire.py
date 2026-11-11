from pathlib import Path
import gzip,hashlib,json,subprocess,tempfile
R=Path.cwd();I=Path("/tmp/crow-restore-check-20261110");A=R/"artifacts/crow-v15-evidence-retirement-round15-20261111";branch="agent/american-crow-rl-20261107"
subprocess.run(["git","-C",str(I),"fetch","origin",branch],check=True)
commit=subprocess.check_output(["git","-C",str(I),"rev-parse","FETCH_HEAD"],text=True).strip()
targets=['artifacts/crow-v15-autonomous-walk-hold-journey-20261111/autonomous-20261126', 'artifacts/crow-v15-walk-hold-wing-off-20261111/autonomous-20261126', 'artifacts/crow-v15-walk-hold-slower-cycle-20261111/autonomous-20261126']
def hfile(path,expanded=False):
 h=hashlib.sha256();op=gzip.open if expanded else open
 with op(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
results=[]
with tempfile.TemporaryDirectory(prefix="crow-independent-restore-") as temp:
 for item in targets:
  raw=R/item/"run.stderr.gz"
  if not raw.exists():raise RuntimeError("source missing: "+str(raw))
  spec=commit+":"+item+"/stderr-parts.json";m=json.loads(subprocess.check_output(["git","-C",str(I),"show",spec]));restored=Path(temp)/(str(len(results))+".gz");h=hashlib.sha256()
  with restored.open("xb") as f:
   for part in m["parts"]:
    data=subprocess.check_output(["git","-C",str(I),"show",commit+":"+item+"/"+part["name"]]);assert len(data)==part["bytes"] and hashlib.sha256(data).hexdigest()==part["sha256"];f.write(data);h.update(data)
  assert restored.stat().st_size==m["bytes"]==raw.stat().st_size and h.hexdigest()==m["compressed_sha256"]==hfile(raw)
  expanded=hfile(restored,True);assert expanded==hfile(raw,True)
  proc=subprocess.run(["lsof","-t","--",str(raw)],capture_output=True,text=True);assert proc.returncode in (0,1) and not proc.stdout.strip()
  assert hfile(raw)==m["compressed_sha256"]
  raw.unlink();results.append({"path":str(raw),"bytes":m["bytes"],"compressed_sha256":m["compressed_sha256"],"expanded_sha256":expanded,"retired":True,"open_handles":False})
  (A/"verification.json").write_text(json.dumps({"independently_fetched_commit":commit,"results":results},indent=2)+"\n")
print(json.dumps({"independently_fetched_commit":commit,"results":results},indent=2))
