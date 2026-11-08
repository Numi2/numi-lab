#!/usr/bin/env python3
"""Independently verify a diagnostic export. This does not authorize restore."""
from pathlib import Path
import hashlib
import sys

COMPONENTS = {"q.bin", "v.bin", "scene-bodies.bin", "rod-nodes.bin", "rod-edges.bin", "manifold-headers.bin", "manifold-points.bin", "manifold-counts.bin", "rod-tool-witnesses.bin"}
BINDINGS = {"checkpoint", "matter", "rigid_cache"}

def verify(capsule, input_checkpoint):
    capsule = Path(capsule)
    if capsule.is_symlink() or not capsule.is_dir():
        raise ValueError("capsule must be a real directory")
    rows = [line.split("\t") for line in (capsule / "manifest.tsv").read_text().splitlines()]
    singles, components, bindings = {}, {}, {}
    for row in rows:
        if row[0] == "component":
            if len(row) != 6 or row[1] in components: raise ValueError("invalid component entry")
            components[row[1]] = row[2:]
        elif row[0] == "binding":
            if len(row) != 5 or row[1] in bindings: raise ValueError("invalid binding entry")
            bindings[row[1]] = row[2:]
        else:
            if len(row) != 2 or row[0] in singles: raise ValueError("invalid metadata entry")
            singles[row[0]] = row[1]
    if set(components) != COMPONENTS or set(bindings) != BINDINGS: raise ValueError("missing or extra component or binding")
    if singles.get("schema") != "numi.surgical-partial-owner-capsule.v1" or singles.get("restorable") != "0" or singles.get("complete") != "0": raise ValueError("unsafe capsule claim")
    if singles.get("lineage") != "synthetic_same_resident_input_bound_by_hash": raise ValueError("incorrect lineage")
    if int(singles.get("compiled_world_fingerprint", "0")) <= 0 or int(singles.get("accepted_state_step", "0")) <= 0: raise ValueError("missing owner identity")
    if not singles.get("missing"): raise ValueError("missing-state declaration absent")
    if int(singles.get("input_state_step", "0")) <= 0: raise ValueError("input step absent")
    digests = {}
    def check(base, name, size, expected):
        if not name or Path(name).name != name: raise ValueError("unsafe path")
        path = base / name
        if path.is_symlink() or not path.is_file(): raise ValueError("missing or symlinked file")
        data = path.read_bytes()
        if len(data) != int(size): raise ValueError("length mismatch")
        h = 1469598103934665603
        for b in data: h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
        if h != int(expected): raise ValueError("hash mismatch")
        digests[str(path)] = hashlib.sha256(data).hexdigest()
    for name, (count, itemsize, size, expected) in components.items():
        if min(int(count), int(itemsize), int(size)) <= 0 or int(count) * int(itemsize) != int(size): raise ValueError("invalid component dimensions")
        check(capsule, name, size, expected)
    for _, (name, size, expected) in bindings.items(): check(capsule.parent, name, size, expected)
    input_checkpoint = Path(input_checkpoint)
    if input_checkpoint.name != singles.get("input_checkpoint_name"): raise ValueError("input identity mismatch")
    check(input_checkpoint.parent, input_checkpoint.name, singles["input_checkpoint_size"], singles["input_checkpoint_fnv64"])
    # Confirm this exact checkpoint points to the Matter file bound above.
    checkpoint = capsule.parent / bindings["checkpoint"][0]
    matter_rows = [x.split("\t") for x in checkpoint.read_text().splitlines() if x.startswith("matter_snapshot\t")]
    if len(matter_rows) != 1 or len(matter_rows[0]) < 2 or matter_rows[0][1] != bindings["matter"][0]: raise ValueError("Matter binding mismatch")
    digests[str(capsule / "manifest.tsv")] = hashlib.sha256((capsule / "manifest.tsv").read_bytes()).hexdigest()
    return singles, digests

if __name__ == "__main__":
    try:
        info, hashes = verify(sys.argv[1], sys.argv[2])
        for name, digest in sorted(hashes.items()): print(digest, name)
        print("verified_partial_nonrestorable=1 compiled_world_fingerprint=" + info["compiled_world_fingerprint"])
    except (ValueError, OSError, IndexError, KeyError) as exc:
        print("verification_failed:", exc, file=sys.stderr)
        sys.exit(1)
