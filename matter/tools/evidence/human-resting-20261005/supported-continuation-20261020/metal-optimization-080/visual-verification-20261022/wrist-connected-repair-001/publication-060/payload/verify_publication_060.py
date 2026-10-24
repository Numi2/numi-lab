#!/usr/bin/env python3
"""Verify publication-060 payload bytes, external observations, and inherited Git blob."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/Users/n/numi-human-aortic-contact-opt-20261019")

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])

inv = json.loads((ROOT / "payload-files-060.json").read_text())
pins = json.loads((ROOT / "source-input-pins-060.json").read_text())
manifest = json.loads((ROOT / "publication-manifest-060.json").read_text())
scope = ROOT / "scope-and-limits-060.md"
readme = ROOT / "README.md"
actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "payload").rglob("*") if p.is_file()}
expected = {e["path"] for e in inv["files"]}
assert actual == expected, {"unlisted": sorted(actual-expected), "missing": sorted(expected-actual)}
assert inv["count"] == len(inv["files"])
assert inv["total_bytes"] == sum(e["bytes"] for e in inv["files"])
for e in inv["files"]:
    p = ROOT / e["path"]
    assert p.is_file() and p.stat().st_size == e["bytes"] and sha_file(p) == e["sha256"], e["path"]
assert manifest["payload_files_sha256"] == sha_file(ROOT / "payload-files-060.json")
assert manifest["source_input_pins_sha256"] == sha_file(ROOT / "source-input-pins-060.json")
assert manifest["scope_note_sha256"] == sha_file(scope)
assert manifest["readme_sha256"] == sha_file(readme)
assert manifest["preparer_sha256"] == sha_file(ROOT / "prepare_publication_060.py")
assert manifest["file_count"] == inv["count"]
assert manifest["payload_bytes"] == inv["total_bytes"]
assert manifest["base_origin_main"] == pins["base_origin_main"]
for e in pins["inherited_references"]:
    blob = git("cat-file", "blob", f'{e["via_commit"]}:{e["git_path"]}')
    assert len(blob) == e["bytes"] and sha_bytes(blob) == e["sha256"], e["git_path"]
checked_external = 0
missing_external = 0
for e in pins["source_artifacts"]:
    p = Path(e["source_path"])
    if p.is_file() and e["observed_sha256"] is not None:
        assert p.stat().st_size == e["bytes"] and sha_file(p) == e["observed_sha256"], str(p)
        checked_external += 1
    else:
        missing_external += 1
for e in pins["omitted_artifacts"]:
    p = Path(e["source_path"])
    if p.is_file():
        assert p.stat().st_size == e["bytes"] and sha_file(p) == e["sha256"], str(p)
        checked_external += 1
result = {
    "status": "PASS_PUBLICATION_060_LOCAL_AND_INHERITED_HASHES",
    "local_files": inv["count"],
    "local_bytes": inv["total_bytes"],
    "external_sources_checked": checked_external,
    "external_sources_not_present_or_not_observed": missing_external,
    "inherited_references": len(pins["inherited_references"]),
    "base_origin_main": manifest["base_origin_main"],
    "qualification": "source-preparation and bounded candidate evidence only",
}
print(json.dumps(result, indent=2))

