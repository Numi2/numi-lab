from pathlib import Path
import hashlib,json
D=Path("artifacts/crow-v15-student-reference-approach-eval-20261111/autonomous-20261126")
manifest=json.loads((D/"stderr-parts.json").read_text());target=D/"run.stderr.gz"
if target.exists():raise RuntimeError("refusing to overwrite "+str(target))
temporary=D/"run.stderr.gz.restoring";total=hashlib.sha256()
with temporary.open("xb") as out:
 for part in manifest["parts"]:
  data=(D/part["name"]).read_bytes()
  if len(data)!=part["bytes"] or hashlib.sha256(data).hexdigest()!=part["sha256"]:raise ValueError("bad part "+part["name"])
  out.write(data);total.update(data)
if total.hexdigest()!=manifest["compressed_sha256"] or temporary.stat().st_size!=manifest["bytes"]:raise ValueError("combined hash or size mismatch")
temporary.rename(target)
print("Restored",target)
