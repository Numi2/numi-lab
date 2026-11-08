#!/usr/bin/env python3
"""Verify partial capture metadata; this deliberately cannot restore state."""
from pathlib import Path
import hashlib
import sys
from verify_partial_capsule import verify as verify_partial

def fnv(data):
    h = 1469598103934665603
    for b in data: h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h

def verify(stage, checkpoint):
    stage = Path(stage)
    base_info, digests = verify_partial(stage / "partial-owner-capsule", checkpoint)
    path = stage / "owner-boundary-evidence"
    if path.is_symlink() or not path.is_dir(): raise ValueError("unsafe boundary directory")
    metadata, component = {}, None
    manifest = path / "manifest.tsv"
    for line in manifest.read_text().splitlines():
        row = line.split("\t")
        if row[0] == "component":
            if component is not None or len(row) != 6: raise ValueError("bad component entry")
            component = row[1:]
        else:
            if len(row) != 2 or row[0] in metadata: raise ValueError("bad metadata entry")
            metadata[row[0]] = row[1]
    for key, value in {"schema":"numi.synthetic-owner-boundary-evidence.v1", "complete":"0", "restorable":"0", "capture_exclusion":"single_thread_private_fixture_after_submission_wait", "environment_count":"1", "supports_reset":"0", "task_fingerprint":"0", "millard_fingerprint":"0"}.items():
        if metadata.get(key) != value: raise ValueError("invalid boundary " + key)
    if metadata.get("world_fingerprint") != base_info["compiled_world_fingerprint"]: raise ValueError("world mismatch")
    if not metadata.get("missing") or int(metadata.get("device_physics_fingerprint", "0")) <= 0: raise ValueError("missing owner identity")
    if int(metadata.get("matter_source_fingerprint", "0")) <= 0 or int(metadata.get("matter_device_fingerprint", "0")) <= 0: raise ValueError("missing Matter identity")
    selectors = metadata.get("selectors", "").split(",")
    if len(selectors) != 9 or any(not x.isdigit() for x in selectors): raise ValueError("invalid owner selectors")
    prior = (stage / "partial-owner-capsule" / "manifest.tsv").read_bytes()
    if len(prior) != int(metadata["prior_capsule_manifest_size"]) or fnv(prior) != int(metadata["prior_capsule_manifest_fnv64"]): raise ValueError("partial binding mismatch")
    if component is None: raise ValueError("missing convex cache")
    name, count, itemsize, size, expected = component
    if name != "convex-caches.bin" or int(itemsize) != 80 or int(count) <= 0 or int(count) * 80 != int(size): raise ValueError("invalid cache dimensions")
    cache = path / name
    if cache.is_symlink() or not cache.is_file(): raise ValueError("missing cache")
    data = cache.read_bytes()
    if len(data) != int(size) or fnv(data) != int(expected): raise ValueError("convex cache mismatch")
    digests[str(cache)] = hashlib.sha256(data).hexdigest()
    digests[str(manifest)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return metadata, digests

if __name__ == "__main__":
    try:
        metadata, digests = verify(sys.argv[1], sys.argv[2])
        for path, digest in sorted(digests.items()): print(digest, path)
        print("verified_partial_nonrestorable_owner_evidence=1 world_fingerprint=" + metadata["world_fingerprint"])
    except (ValueError, OSError, KeyError, IndexError) as exc:
        print("verification_failed:", exc, file=sys.stderr)
        sys.exit(1)
