#!/usr/bin/env python3
"""Remove ten ignored archive duplicates using the already verified Git mapping."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
out = root / "artifacts/crow-rl-git-recovery-20261107"
receipt = out / "ignored-copy-cleanup.json"
if receipt.exists():
    raise RuntimeError("Cleanup already recorded")
spec = importlib.util.spec_from_file_location("crow_archive", root / "tools/crow_evidence_archive.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
manifest = tool.load_manifest(out / "manifest.json")
known = {f["sha256"]: f for f in manifest["files"]}
head = tool.git("rev-parse", "HEAD").decode().strip()
remote = tool.git("ls-remote", "origin", "refs/heads/" + tool.BRANCH).decode().split()[0]
if head != remote:
    raise RuntimeError("Archive has not been pushed")
before = subprocess.check_output(["df", "-k", "."]).decode()
handles = subprocess.run(["/usr/sbin/lsof", "-Fpn", "+D", "artifacts"], capture_output=True, text=True, timeout=60)
if handles.returncode not in (0, 1) or handles.stdout or handles.stderr:
    raise RuntimeError("Archive files have open handles")
files = []
entries = tool.tree()
with tool.Objects() as objects:
    for i in range(1, 11):
        p = Path(f"artifacts/crow-rl-life-checkpoint-20261107/build-logs/world-checkpoint-build-{i:02d}.log")
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
            raise RuntimeError("Unexpected cleanup symlink")
        sha = tool.file_hash(p)
        source = known[sha]
        h = hashlib.sha256()
        size = 0
        for data in tool.content(source, entries, objects):
            h.update(data)
            size += len(data)
        if h.hexdigest() != sha or size != p.stat().st_size:
            raise RuntimeError("Ignored file is not recovered exactly from Git")
        files.append({"path": str(p), "bytes": size, "sha256": sha,
                      "archived_source": source["path"], "backing": source["backing"]})
for f in files:
    p = Path(f["path"])
    if tool.file_hash(p) != f["sha256"]:
        raise RuntimeError("Ignored copy changed before cleanup")
    p.unlink()
for p in [Path("artifacts/crow-rl-life-checkpoint-20261107/build-logs"),
          Path("artifacts/crow-rl-life-checkpoint-20261107")]:
    p.rmdir()
shader = Path("artifacts/crow-rl-recurrent-actor-20261107/inputs/PolicyABI3.metallib")
shader_sha = tool.file_hash(shader)
tool.write_json(receipt, {"archive_commit": head, "remote_commit": remote, "removed": files,
                         "open_handles": 0, "df_before": before,
                         "df_after": subprocess.check_output(["df", "-k", "."]).decode(),
                         "active_shader": {"path": str(shader), "sha256": shader_sha,
                                           "also_archived_source": known[shader_sha]["path"],
                                           "disposition": "Retained as essential input and explicitly tracked despite the general metallib ignore rule."}})
print(json.dumps({"ignored_duplicate_files_removed": len(files), "bytes": sum(f["bytes"] for f in files)}))
