from pathlib import Path
import hashlib,json,gzip,subprocess
R=Path("/Users/n/Developer/numi-crow-rl-20261107");T=Path("/tmp/crow-grip-parts-restore-20261111");A=R/"artifacts/crow-v15-grip-archive-restoration-20261111"
m=json.loads((T/"stderr-parts.json").read_text());whole=hashlib.sha256();combined=T/"restored.stderr.gz"
with combined.open("xb") as o:
 for part in m["parts"]:
  x=(T/part["name"]).read_bytes()
  assert len(x)==part["bytes"] and hashlib.sha256(x).hexdigest()==part["sha256"]
  o.write(x);whole.update(x)
assert whole.hexdigest()==m["compressed_sha256"] and combined.stat().st_size==m["bytes"]
raw=R/"artifacts/crow-v15-perch-transient-grip-control-20261111/teacher-20261126/run.stderr.gz"
def streamhash(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
with raw.open("rb") as f:assert streamhash(f)==m["compressed_sha256"]
with gzip.open(combined,"rb") as f:expanded=streamhash(f)
with gzip.open(raw,"rb") as f:assert streamhash(f)==expanded
p=subprocess.run(["lsof","-t","--",str(raw)],capture_output=True,text=True)
assert p.returncode in (0,1) and not p.stdout.strip()
result={"independently_fetched_commit":(T/"restored-commit").read_text().strip(),"parts_checked":m["parts"],"compressed_sha256":m["compressed_sha256"],"expanded_sha256":expanded,"bytes":m["bytes"],"path":str(raw),"open_handles":False,"retired":False}
(A/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
print("Remote parts reassembled and gzip expanded:",m["bytes"],"bytes")
