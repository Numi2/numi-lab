#!/usr/bin/env python3
"""Verify the frozen Publication 062 payload and its scoped file inventory."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parent
LIST = ROOT / "payload-files-062.json"
MANIFEST = ROOT / "publication-manifest-062.json"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()
items=json.loads(LIST.read_text())
manifest=json.loads(MANIFEST.read_text())
assert manifest["payload_file_list_sha256"] == sha(LIST)
assert manifest["verifier_sha256"] == sha(Path(__file__))
testenv=ROOT/"test-environment-062.json"
assert manifest["test_environment_sha256"] == sha(testenv)
tenv=json.loads(testenv.read_text())
assert tenv["host"] == "numi.local"
assert tenv["python_version"].startswith("3.13.16 ")
assert tenv["pytest_version"] == "9.1.1"
assert [x["returncode"] for x in tenv["runs"]] == [0,0]
assert "16 passed" in tenv["runs"][0]["stdout"]
assert "5 passed" in tenv["runs"][1]["stdout"]
seen=set()
for item in items:
    rel=item["path"]
    assert rel not in seen, f"duplicate inventory path: {rel}"
    seen.add(rel)
    p=ROOT/rel
    assert p.is_file() and not p.is_symlink(), f"missing or redirected file: {rel}"
    assert p.stat().st_size == item["bytes"], f"size mismatch: {rel}"
    assert sha(p) == item["sha256"], f"SHA-256 mismatch: {rel}"
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in {"payload-files-062.json","publication-manifest-062.json","verify_publication_062.py"}}
assert actual == seen, f"unlisted={sorted(actual-seen)} missing={sorted(seen-actual)}"
fit=json.loads((ROOT/"payload/040/attempt-004/fit-report.json").read_text())
assert fit.get("status") == "candidate_rejected_basic_gate"
neg=json.loads((ROOT/"payload/040/review/root-negative-native-query-gate-040.json").read_text())
assert neg.get("status") == "PASS_REJECTED_CANDIDATE_BLOCKED_BEFORE_NATIVE_LAUNCH"
assert neg.get("output_directory_created") is False and neg.get("native_launch") is False
pins=json.loads((ROOT/"source-input-pins-062.json").read_text())
assert pins["native_query_gate"]["positive_parameterized_query_path"].startswith("not run")
assert pins["attempt_004"]["native_query_rejection_sha256"] == sha(ROOT/"payload/040/review/root-negative-native-query-gate-040.json")
print(json.dumps({"status":"PASS","payload_files":len(items),"payload_bytes":sum(x["bytes"] for x in items),"candidate_status":fit["status"],"negative_query_gate":neg["status"],"positive_query_path":"not run"},sort_keys=True))

