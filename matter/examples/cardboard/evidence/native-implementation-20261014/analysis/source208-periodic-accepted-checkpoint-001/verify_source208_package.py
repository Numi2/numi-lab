#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
PKG = Q / "analysis/source208-periodic-accepted-checkpoint-001"
BASE = Q / "snapshots/source-cohesive-quadrature-area-205"
CANDIDATE = Q / "snapshots/source-periodic-accepted-checkpoint-208"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = json.loads((PKG / "manifest.json").read_text())
for entry in manifest["package_files"]:
    path = PKG / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        raise SystemExit("package file mismatch: " + entry["path"])
overlay = json.loads((PKG / "source-overlay.json").read_text())
if digest(PKG / "source-overlay.json") != manifest["overlay_manifest_sha256"]:
    raise SystemExit("overlay manifest digest mismatch")
if digest(BASE / "frozen-source.json") != overlay["base_frozen_source_sha256"]:
    raise SystemExit("base frozen-source digest mismatch")
if digest(CANDIDATE / "frozen-source.json") != overlay["candidate_frozen_source_sha256"]:
    raise SystemExit("candidate frozen-source digest mismatch")
frozen = json.loads((CANDIDATE / "frozen-source.json").read_text())
build = json.loads((PKG / "source-build.json").read_text())
if not build["success"] or build["source_sha256"] != frozen["source_sha256"]:
    raise SystemExit("build and candidate source maps differ")
if len(frozen["source_sha256"]) != 819:
    raise SystemExit("unexpected selected source count")
for relative, expected in frozen["source_sha256"].items():
    direct = CANDIDATE / relative
    nested = CANDIDATE / "matter" / relative
    path = direct if direct.is_file() else nested
    if not path.is_file() or digest(path) != expected:
        raise SystemExit("candidate source mismatch: " + relative)
changed = overlay["changed_sources"]
if set(changed) != {
    "matter/tools/cardboard_coupon_resume.hpp",
    "matter/tools/cardboard_coupon_resume_check.cpp",
    "matter/tools/cardboard_probe.mm",
    "matter/tools/cardboard_shipping_snapshot.hpp",
    "matter/tools/cardboard_shipping_snapshot_check.cpp",
}:
    raise SystemExit("unexpected changed source set")
for relative, hashes in changed.items():
    if digest(BASE / relative) != hashes["base_sha256"] or digest(CANDIDATE / relative) != hashes["source208_sha256"]:
        raise SystemExit("changed source digest mismatch: " + relative)
if digest(PKG / "overlay.patch") != overlay["patch_sha256"]:
    raise SystemExit("patch digest mismatch")
print(json.dumps({"valid": True, "package_files": len(manifest["package_files"]),
                  "selected_source_files": len(frozen["source_sha256"]),
                  "changed_sources": len(changed)}))
