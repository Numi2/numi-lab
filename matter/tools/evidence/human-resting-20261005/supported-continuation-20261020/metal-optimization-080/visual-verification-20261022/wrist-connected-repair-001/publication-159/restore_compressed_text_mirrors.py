#!/usr/bin/env python3
"""Restore P159 text archives into a fresh directory and verify exact bytes and metadata."""
import argparse, hashlib, json, os, subprocess, stat
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--destination", required=True, help="new output root; original files are never overwritten")
p.add_argument("--package", default=str(Path(__file__).resolve().parents[0]), help="P159 package root")
a=p.parse_args(); package=Path(a.package); dest=Path(a.destination)
receipt=package/"evidence/surface-clearance-546/matched-connected-fit-575/run-001/compression-preservation/receipt.json"
items=json.loads(receipt.read_text())["files"]
for item in items:
    rel=Path(item["original_path"]).name
    target=dest/rel
    if target.exists(): raise SystemExit(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    archive=package/item["archive_path"]
    with target.open("xb") as out:
        subprocess.run(["zstd","-d","-q","-c",str(archive)],stdout=out,check=True)
        out.flush(); os.fsync(out.fileno())
    os.chmod(target,item["original_mode"]); os.utime(target,ns=(item["original_mtime_ns"],item["original_mtime_ns"]))
    h=hashlib.sha256(target.read_bytes()).hexdigest()
    if h!=item["original_sha256"] or target.stat().st_size!=item["original_bytes"]: raise SystemExit(f"verification failed: {target}")
    print(target, h, target.stat().st_size)
