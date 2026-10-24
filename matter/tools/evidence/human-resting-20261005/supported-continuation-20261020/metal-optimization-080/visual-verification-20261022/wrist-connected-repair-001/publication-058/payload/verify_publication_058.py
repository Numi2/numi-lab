#!/usr/bin/env python3
"""Verify local publication-058 payload bytes and inherited Git blob pins."""
from pathlib import Path
import hashlib, json, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
REPO = next(p for p in ROOT.parents if (p / ".git").exists())
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def git(*args): return subprocess.check_output(["git","-C",str(REPO),*args])
inv=json.loads((ROOT/"payload-files-058.json").read_text())
manifest=json.loads((ROOT/"publication-manifest-058.json").read_text())
pins=json.loads((ROOT/"source-input-pins-058.json").read_text())
scope=ROOT/"scope-and-limits-058.md"
actual={p.relative_to(ROOT).as_posix() for p in (ROOT/"payload").rglob("*") if p.is_file()}
expected={e["path"] for e in inv["files"]}
assert actual==expected,{"unlisted":sorted(actual-expected),"missing":sorted(expected-actual)}
assert inv["count"]==len(inv["files"])
assert inv["total_bytes"]==sum(e["bytes"] for e in inv["files"])
for e in inv["files"]:
    p=ROOT/e["path"]
    assert p.is_file() and p.stat().st_size==e["bytes"] and sha(p)==e["sha256"],e["path"]
assert manifest["payload_files_sha256"]==sha(ROOT/"payload-files-058.json")
assert manifest["source_input_pins_sha256"]==sha(ROOT/"source-input-pins-058.json")
assert manifest["scope_note_sha256"]==sha(scope)
assert manifest["file_count"]==inv["count"]
assert manifest["payload_bytes"]==inv["total_bytes"]
assert manifest["base_origin_main"]==pins["base_origin_main"]
for e in inv["inherited_references"]:
    blob=git("cat-file","blob",f'{e["via_commit"]}:{e["git_path"]}')
    assert len(blob)==e["bytes"] and sha_bytes(blob)==e["sha256"],e["git_path"]
# External omitted inputs are evidence pins; verify any source still present on this host.
checked_external=0
for e in inv["omitted_artifacts"]:
    p=Path(e["source_path"])
    if p.is_file():
        assert p.stat().st_size==e["bytes"] and sha(p)==e["sha256"],str(p)
        checked_external+=1
status=manifest["status"]
if status.startswith("DRAFT_"):
    assert not any(x["path"].startswith("payload/029/") for x in inv["files"])
    result="PASS_DRAFT_PAYLOAD_AND_INHERITED_HASHES_OWNER_RESULTS_PENDING"
else:
    assert any(x["path"].startswith("payload/029/") for x in inv["files"])
    result="PASS_PUBLICATION_058_PAYLOAD_AND_INHERITED_HASHES"
print(json.dumps({"status":result,"local_files":inv["count"],"local_bytes":inv["total_bytes"],"inherited_references":len(inv["inherited_references"]),"omitted_artifacts":len(inv["omitted_artifacts"]),"external_sources_checked":checked_external,"base":manifest["base_origin_main"]},indent=2))
