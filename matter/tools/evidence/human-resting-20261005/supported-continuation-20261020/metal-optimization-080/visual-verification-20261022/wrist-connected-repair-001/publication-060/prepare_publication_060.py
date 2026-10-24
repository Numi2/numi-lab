#!/usr/bin/env python3
"""Rebuild the review manifests for publication-060 without touching Git's normal index."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "payload"
REPO = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
BASE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PARENT = "d1d9b0935704794d0087e975d7ee378e7c5e1811"

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

def read_json(path):
    return json.loads(Path(path).read_text())

def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

# Ensure the latest frozen 038 inventory is represented.
frozen038 = BASE / "bone-clearance-fit-038-candidate-query/frozen-inventory.json"
target038 = PAYLOAD / "038/run-002/frozen-inventory.json"
if not target038.is_file():
    raise SystemExit("038 frozen inventory missing from package")
if sha_file(target038) != "9c5de6bc0828a1882eeaaad53151dcf00eee11270ca5a1cdf9da26874c5bb117":
    raise SystemExit("038 inventory pin differs from the latest frozen inventory")
if sha_file(PAYLOAD / "037/attempt-003-prepared-not-run/build_fit_037.py") != "51d2de3b4b8a9c0a40429beaf00dd2ec2315a75394412b4c03c6c95cbc561e8d":
    raise SystemExit("held 037 attempt-003 builder hash mismatch")
if sha_file(PAYLOAD / "037/attempt-003-prepared-not-run/held-preflight-receipt.json") != "fb8b25de3ed0e472f6767aad0f3517927cdf71048884da66c83d80818523884b":
    raise SystemExit("held 037 attempt-003 receipt hash mismatch")
if sha_file(PAYLOAD / "038/row946-entry-proof/row946-route-entry-proof.json") != "a96b200b1b1b82ae9084be715d1177ddeed06e78f193945b9dc5b80769427f62":
    raise SystemExit("row946 route-entry diagnostic hash mismatch")

# Local payload inventory, excluding links to prevent path aliasing.
files = []
for p in sorted(PAYLOAD.rglob("*")):
    if p.is_symlink():
        raise SystemExit(f"symlink in payload: {p}")
    if p.is_file():
        files.append({"path": p.relative_to(ROOT).as_posix(),
                      "bytes": p.stat().st_size,
                      "sha256": sha_file(p)})
payload_index = {
    "schema": "numi.human.publication.payload-files.v1",
    "count": len(files),
    "total_bytes": sum(x["bytes"] for x in files),
    "files": files,
}
write_json(ROOT / "payload-files-060.json", payload_index)

# Preserve the exact as-run input hash tables from the frozen receipts.
source033 = read_json(PAYLOAD / "033/source-pins.json")
fit034 = read_json(PAYLOAD / "034/fit-report.json")
preflight034 = read_json(PAYLOAD / "034/preflight.json")
launch035 = read_json(PAYLOAD / "035/audit/source-preflight-003/launch-manifest.snapshot.json")
report036 = read_json(PAYLOAD / "036/report.json")
prep037 = read_json(PAYLOAD / "037/preparation/preparation-report.json")
fit037a = read_json(PAYLOAD / "037/attempt-001/fit-report.json")
fit037b = read_json(PAYLOAD / "037/attempt-002/fit-report.json")
review037b = read_json(PAYLOAD / "037/attempt-002/root-review-037-component-attempt-002.json")
layers037b = read_json(PAYLOAD / "037/attempt-002/root-review-037002-layers-volume.json")
inventory038 = read_json(target038)
launch038 = read_json(PAYLOAD / "038/run-002/base-launch-manifest.json")
launch038_child = read_json(PAYLOAD / "038/run-002/source-preflight/launch-manifest.snapshot.json")
held003 = read_json(PAYLOAD / "037/attempt-003-prepared-not-run/held-preflight-receipt.json")
entry946 = read_json(PAYLOAD / "038/row946-entry-proof/row946-route-entry-proof.json")

def observed(path, expected_sha, scope):
    p = Path(path)
    result = {"scope": scope, "source_path": str(p), "declared_sha256": expected_sha}
    if p.is_file():
        result["bytes"] = p.stat().st_size
        result["observed_sha256"] = sha_file(p)
        result["observed_matches_declared"] = result["observed_sha256"] == expected_sha
    else:
        result["bytes"] = None
        result["observed_sha256"] = None
        result["observed_matches_declared"] = None
    return result

# Input/source references are preserved as declarations from the frozen run
# manifests. Current source paths are observed where available; they are not
# asserted to be independently archived if absent.
external = []
for item in source033.get("inputs", []):
    external.append(observed(item["path"], item["sha256"], "033 source pin"))
for path, digest in launch035.get("input_hashes", {}).items():
    external.append(observed(path, digest, "035 source-preflight-003 launch input"))
for path, digest in launch038.get("input_hashes", {}).items():
    external.append(observed(path, digest, "038 run-002 base launch input"))
for path, digest in launch038_child.get("input_hashes", {}).items():
    external.append(observed(path, digest, "038 run-002 source-preflight input"))
for path, digest in entry946.get("inputs", {}).items():
    external.append(observed(path, digest, "038 row946 route-entry diagnostic input"))
# Deduplicate same path + same declaration, retaining all scopes.
dedup = {}
for rec in external:
    key = (rec["source_path"], rec["declared_sha256"])
    if key not in dedup:
        dedup[key] = rec
    else:
        dedup[key]["scope"] += "; " + rec["scope"]
external = sorted(dedup.values(), key=lambda x: (x["source_path"], x["declared_sha256"]))

# The physical topology used by 034/037 is already published in 058.
inherited = [{
    "kind": "published_input",
    "git_path": "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/publication-058/payload/030/prism-input/candidate.npz",
    "via_commit": PARENT,
    "bytes": 905624,
    "sha256": "640b7bc56013748b26d5e4e12790f85a472602ee6a4732f5773243df6e2147cc",
    "used_as": "D37 physical topology input for 034/037",
}]

omitted = [{
    "kind": "external_large_input_not_duplicated",
    "source_path": "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-candidate020-bone-witness-029/source-preflight-002/registered-skin.csv",
    "bytes": 24135771,
    "sha256": "eccbc683f8450f27c8c2559471aff8fa8c62644871011d46fe3c3dbfdf3f2524",
    "recovery_claim": False,
    "note": "The file was present and hash-checked while preparing this package; it is not duplicated in the package and this manifest does not claim an independent archive copy.",
}]
pins = {
    "schema": "numi.human.publication.source-input-pins.v1",
    "publication": "060",
    "base_origin_main": PARENT,
    "candidate": {
        "id": "candidate020-d37-current-unpaired163-owner-integration-038",
        "source_payload_sha256": "7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb",
        "source_payload_path": "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bilateral-hand-functional-rebind-020/run-001/candidate.nhskin",
        "registered_skin_poses_sha256": "0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd",
        "identity": "candidate020 source; no scene activation",
    },
    "source_artifacts": external,
    "report_declared_input_hashes": {
        "033_constraint_npz": source033.get("other_input_pins"),
        "034_fit_report": fit034.get("input_sha256"),
        "034_preflight": preflight034.get("input_sha256"),
        "035_capture_launch_manifest_sha256": sha_file(PAYLOAD / "035/audit/source-preflight-003/launch-manifest.snapshot.json"),
        "036_affine_report": report036.get("input_sha256"),
        "037_preparation": prep037.get("pins_sha256"),
        "037_attempt_001": fit037a.get("input_sha256"),
        "037_attempt_002": fit037b.get("input_sha256"),
        "037_attempt_003_held_builder_sha256": "51d2de3b4b8a9c0a40429beaf00dd2ec2315a75394412b4c03c6c95cbc561e8d",
        "037_attempt_003_held_receipt_sha256": sha_file(PAYLOAD / "037/attempt-003-prepared-not-run/held-preflight-receipt.json"),
    },
    "fit_review": {
        "037_attempt_001_root_review": {
            "status": "REJECTED_GEOMETRY",
            "candidate_sha256": "f6ed92a8f1ed06631cee260b31bd707f6cf0d00ac1aa7c0829491638cc9ad1af",
            "nonpositive_cells": 359,
            "cells_below_q_1e_minus_5": 365,
        },
        "037_attempt_002_root_review": review037b,
        "037_attempt_002_layer_volume_review": layers037b,
    },
    "candidate_038": {
        "frozen_inventory_sha256": sha_file(target038),
        "query_report_sha256": inventory038["source_only_query"]["candidate_query_report_sha256"],
        "candidate_override_csv_sha256": inventory038["source_only_query"]["candidate_override_csv_sha256"],
        "scope": inventory038["source_only_query"]["limitation"],
        "physical_qualification": inventory038["source_only_query"]["physical_qualification"],
    },
    "candidate_038_counts": {
        "baseline_inside_node_bank_pairs": 139,
        "candidate_inside_node_bank_pairs": inventory038["source_only_query"]["coverage"]["inside"],
        "resolved_by_count": 138,
        "candidate_invalid_pairs": inventory038["source_only_query"]["coverage"]["invalid"],
        "scope": "source-only per-node closed-solid query; no triangle, tetrahedron, or physical qualification",
    },
    "row946_route_entry_diagnostic": {
        "path": "payload/038/row946-entry-proof/row946-route-entry-proof.json",
        "sha256": sha_file(PAYLOAD / "038/row946-entry-proof/row946-route-entry-proof.json"),
        "status": entry946.get("status"),
        "identity": entry946.get("identity"),
        "route_hits": entry946.get("route_hits"),
        "coverage_limit": entry946.get("coverage_limit"),
        "interpretation_limit": "one-column double-precision segment diagnostic over reconstructed Float32 triangles; exact-predicate/full-route follow-up remains pending",
    },
    "attempt_003": {
        "status": "PREPARED_NOT_RUN",
        "builder_path": "payload/037/attempt-003-prepared-not-run/build_fit_037.py",
        "builder_sha256": sha_file(PAYLOAD / "037/attempt-003-prepared-not-run/build_fit_037.py"),
        "held_preflight_receipt_sha256": sha_file(PAYLOAD / "037/attempt-003-prepared-not-run/held-preflight-receipt.json"),
        "optimizer_ran": False,
        "candidate_emitted": False,
    },
    "omitted_artifacts": omitted,
    "inherited_references": inherited,
    "limitations": [
        "038 is node-to-closed-solid only; no skin triangle, tetrahedron, full clearance, or native physical acceptance.",
        "Row946 route entry is a one-column double-precision segment diagnostic over reconstructed Float32 member triangles; exact-predicate and complete-column follow-up remain pending.",
        "037 attempt 003 was prepared but not run; it produced no candidate.",
        "037 attempt 002 is a bounded partial fit; it is not a completed/admitted candidate.",
        "Registered-skin CSV is pinned but not duplicated and no independent archive-recovery claim is made.",
        "No physiological thickness, anatomy, clinical, biological, or scene activation claim.",
    ],
}
write_json(ROOT / "source-input-pins-060.json", pins)

# The verifier itself is part of the package and therefore already in files.
manifest = {
    "schema": "numi.human.publication.manifest.v1",
    "publication": "060",
    "status": "READY_FOR_ROOT_REVIEW_NOT_PUSHED",
    "base_origin_main": PARENT,
    "local_head_at_prepare": git("rev-parse", "HEAD"),
    "normal_index_tree_at_prepare": git("write-tree"),
    "file_count": payload_index["count"],
    "payload_bytes": payload_index["total_bytes"],
    "payload_files_sha256": sha_file(ROOT / "payload-files-060.json"),
    "source_input_pins_sha256": sha_file(ROOT / "source-input-pins-060.json"),
    "scope_note_sha256": sha_file(ROOT / "scope-and-limits-060.md"),
    "readme_sha256": sha_file(ROOT / "README.md"),
    "preparer_sha256": sha_file(ROOT / "prepare_publication_060.py"),
    "inherited_references_count": len(inherited),
    "omitted_artifact_count": len(omitted),
    "limitations": pins["limitations"],
    "pending": ["Exact-predicate and complete-column follow-up for row946 route-entry diagnostic is outside this package."],
}
write_json(ROOT / "publication-manifest-060.json", manifest)
print(json.dumps({
    "payload_files": payload_index["count"],
    "payload_bytes": payload_index["total_bytes"],
    "external_source_records": len(external),
    "external_present_hash_matches": sum(x["observed_matches_declared"] is True for x in external),
    "external_present_hash_mismatches": sum(x["observed_matches_declared"] is False for x in external),
    "external_paths_absent": sum(x["observed_matches_declared"] is None for x in external),
    "manifest_sha256": sha_file(ROOT / "publication-manifest-060.json"),
    "payload_file_list_sha256": sha_file(ROOT / "payload-files-060.json"),
    "source_input_pins_sha256": sha_file(ROOT / "source-input-pins-060.json"),
    "local_head": manifest["local_head_at_prepare"],
    "normal_index_tree": manifest["normal_index_tree_at_prepare"],
    "base_origin_main": PARENT,
}, indent=2))

