#!/usr/bin/env python3
import hashlib,json,pathlib
Q=pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
P=Q/"analysis/source252-freebody-scene-observer-build-validation-001"
S=Q/"snapshots/source-freebody-scene-observer-252-002"
R=json.loads((P/"report.json").read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
F=json.loads((S/"frozen-source.json").read_text())["source_sha256"]
assert sha(S/"frozen-source.json")==R["frozen_source_sha256"] and len(F)==R["frozen_source_file_count_verified"]==836
for rel,h in F.items(): assert sha(S/rel)==h,rel
assert sha(Q/"analysis/source252-freebody-scene-observer-002/source-overlay.json")==R["overlay_manifest_sha256"]
assert sha(Q/"analysis/source252-freebody-scene-observer-002/overlay.patch")==R["patch_sha256"]
B=pathlib.Path(R["build_record"]); assert sha(B)==R["build_record_sha256"]
D=json.loads(B.read_text()); assert D["success"] and len(D["source_sha256"])==R["build_source_file_count"]==831
assert set(D["source_sha256"])<=set(F)
for rel,h in D["source_sha256"].items(): assert F[rel]==h and sha(S/rel)==h,rel
for path,h in D["artifacts_sha256"].items(): assert sha(pathlib.Path(path))==h,path
assert (P/"contract.exit.txt").read_text().strip()=="0"
assert (P/"contract.stdout.txt").read_text().strip()=="free_body_control_scene_evidence_contract=pass physical_transactions=0"
assert not (P/"contract.stderr.txt").read_text()
print("source252_build_validation=pass frozen=836 build_inputs=831 artifacts=%d physical_transactions=0"%len(D["artifacts_sha256"]))
