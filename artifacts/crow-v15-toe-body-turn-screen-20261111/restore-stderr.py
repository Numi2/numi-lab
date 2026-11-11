from pathlib import Path
import hashlib,json
D=Path("artifacts/crow-v15-toe-body-turn-screen-20261111");m=json.loads((D/"stderr-parts.json").read_text());p=D/"flight.stderr.gz"
if p.exists():raise RuntimeError("refusing overwrite")
tmp=D/"flight.stderr.gz.restoring";h=hashlib.sha256()
with tmp.open("xb") as f:
 for part in m["parts"]:
  b=(D/part["name"]).read_bytes()
  if len(b)!=part["bytes"] or hashlib.sha256(b).hexdigest()!=part["sha256"]:raise ValueError("bad part")
  f.write(b);h.update(b)
if h.hexdigest()!=m["compressed_sha256"] or tmp.stat().st_size!=m["bytes"]:raise ValueError("bad archive")
tmp.rename(p)
