#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
inv=json.loads((root/"payload-files-057.json").read_text())
manifest=json.loads((root/"publication-manifest-057.json").read_text())
actual={p.relative_to(root).as_posix() for p in (root/"payload").rglob("*") if p.is_file()}
expected={e["path"] for e in inv["files"]}
assert actual==expected,{"unlisted":sorted(actual-expected),"missing":sorted(expected-actual)}
assert inv["count"]==len(inv["files"]) and inv["total_bytes"]==sum(e["bytes"] for e in inv["files"])
for e in inv["files"]:
 p=root/e["path"]
 assert p.is_file() and p.stat().st_size==e["bytes"] and sha(p)==e["sha256"],e["path"]
assert sha(root/"payload-files-057.json")==manifest["payload_files_sha256"]
assert sha(root/"source-input-pins-057.json")==manifest["source_input_pins_sha256"]
assert sha(root/"scope-and-limits-057.md")==manifest["scope_note_sha256"]
assert manifest["file_count"]==inv["count"] and manifest["payload_bytes"]==inv["total_bytes"]
assert manifest["base_origin_main"]=="9226d20147fbb1cfed7fe5a4c1ce96a0ee13413f"
inh=inv["inherited_files"]
assert inv["inherited_count"]==len(inh) and inv["inherited_bytes"]==sum(e["bytes"] for e in inh)
assert manifest["inherited_056"]["commit"]==manifest["base_origin_main"]
assert manifest["inherited_056"]["file_count"]==len(inh) and manifest["inherited_056"]["payload_bytes"]==inv["inherited_bytes"]
repo=next(p for p in root.parents if (p/".git").exists())
for e in inh:
 blob=subprocess.run(["git","-C",str(repo),"cat-file","blob",f'{e["publication_056_commit"]}:{e["git_path"]}'],check=True,stdout=subprocess.PIPE).stdout
 assert len(blob)==e["bytes"] and sha_bytes(blob)==e["sha256"],e["git_path"]
print(json.dumps({"status":"PASS_PUBLICATION_057_INTERNAL_AND_INHERITED_HASHES","payload_count":inv["count"],"payload_bytes":inv["total_bytes"],"inherited_count":len(inh),"inherited_bytes":inv["inherited_bytes"],"base":manifest["base_origin_main"]},indent=2))
