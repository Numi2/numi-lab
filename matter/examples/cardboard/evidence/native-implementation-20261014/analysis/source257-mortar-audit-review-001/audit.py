#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
SRC = Q / "snapshots/source-mortar-first-uncertified-audit-257-005"
BASE = Q / "snapshots/source-freebody-scene-observer-252-002"
PKG = Q / "analysis/source-mortar-first-uncertified-audit-257-005"
BUILD = Q / "build-source257-mortar-audit-006"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

frozen = json.loads((SRC / "frozen-source.json").read_text())
base_frozen = json.loads((BASE / "frozen-source.json").read_text())
source_map = frozen["source_sha256"]
base_map = base_frozen["source_sha256"]
overlay = json.loads((PKG / "source-overlay.json").read_text())
contract = json.loads((PKG / "cpu_contract.json").read_text())
build = json.loads((BUILD / "source-build.json").read_text())
actual_source_mismatches = [
    rel for rel, expected in source_map.items()
    if not (SRC / rel).is_file() or sha(SRC / rel) != expected
]
changed = sorted(
    rel for rel in set(source_map) | set(base_map)
    if source_map.get(rel) != base_map.get(rel)
)
declared = sorted(overlay["changed_files"])
runtime = (SRC / "matter/src/runtime.mm").read_text()
metal = (SRC / "matter/src/metal/contact.metalinc").read_text()
header = (SRC / "matter/include/numi/matter/shared.h").read_text()
post_capture = 'if (postRebuild) {' in runtime[runtime.find('actual_mortar_complement_first_uncertified_query')-500:
                                                     runtime.find('actual_mortar_complement_first_uncertified_query')+100]
artifact_mismatches = []
for path, expected in build["artifacts_sha256"].items():
    p = Path(path)
    if not p.is_file() or sha(p) != expected:
        artifact_mismatches.append(path)
build_sources_equal = build.get("source_sha256") == source_map

checks = {
    "source_snapshot_has_831_frozen_entries": len(source_map) == 831,
    "source_files_match_frozen_map": not actual_source_mismatches,
    "declared_delta_is_exactly_three_files": changed == declared and len(changed) == 3,
    "patch_hash_matches_overlay": sha(PKG / "overlay.patch") == overlay["patch_sha256"],
    "overlay_frozen_hash_matches_snapshot": sha(SRC / "frozen-source.json") == overlay["candidate_frozen_source_sha256"],
    "cpu_contract_passed": contract.get("passed") is True,
    "record_abi_is_384_bytes": overlay.get("record_size_bytes") == 384 and "sizeof(NMMortarComplementAuditGPU) == 384" in header,
    "no_new_environment_keys": overlay.get("new_environment_keys") == [],
    "selector_uses_only_reserved_3_2_extension": "f[7] == 3u && f[8] == 2u" in runtime and "f[7] < 3u && f[8] < 2u" in runtime,
    "capture_array_is_post_rebuild_only": post_capture,
    "selected_candidate_initializes_first_writer_record": "mortarAudit[0] = {};" in metal,
    "first_writer_guard_is_preserved": "audit[0].attempt.x != 0u" in metal,
    "production_certification_accumulation_remains": "allCertified = allCertified && certified;" in metal,
    "build_source_hashes_match_snapshot": build_sources_equal,
    "build_succeeded": build.get("success") is True,
    "built_artifacts_match_record": not artifact_mismatches,
}
report = {
    "schema": "numi.cardboard.source257-mortar-audit-build-review.v1",
    "candidate": "source-mortar-first-uncertified-audit-257-005",
    "base": "source-freebody-scene-observer-252-002",
    "source_frozen_sha256": sha(SRC / "frozen-source.json"),
    "source_file_count": len(source_map),
    "source_mismatches": actual_source_mismatches,
    "changed_files": changed,
    "declared_changed_files": declared,
    "overlay_sha256": sha(PKG / "source-overlay.json"),
    "patch_sha256": sha(PKG / "overlay.patch"),
    "cpu_contract_sha256": sha(PKG / "cpu_contract.json"),
    "build_record": str(BUILD / "source-build.json"),
    "build_record_sha256": sha(BUILD / "source-build.json"),
    "build_success": build.get("success"),
    "target": build.get("targets"),
    "artifact_sha256": build.get("artifacts_sha256"),
    "artifact_mismatches": artifact_mismatches,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "gpu_execution": False,
    "claim_limit": "source/build validation only; no native query result or physical/contact qualification",
}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "report.json"
out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"all_checks_pass": report["all_checks_pass"], "checks": checks, "changed_files": changed, "build_success": build.get("success")}, indent=2))
if not report["all_checks_pass"]:
    sys.exit(1)
