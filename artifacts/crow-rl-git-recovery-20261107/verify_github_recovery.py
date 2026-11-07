#!/usr/bin/env python3
"""Fresh GitHub recovery, exact readback, and bounded temporary-copy cleanup."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

root = Path.cwd()
archive = root / "artifacts/crow-rl-git-recovery-20261107"
receipt = archive / "verification-github.json"
if receipt.exists():
    raise RuntimeError("Recovery receipt already exists")
spec = importlib.util.spec_from_file_location("crow_archive", root / "tools/crow_evidence_archive.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
manifest_rel = Path("artifacts/crow-rl-git-recovery-20261107/manifest.json")
expected = tool.git("rev-parse", "ce5bdb7c").decode().strip()
before = subprocess.check_output(["df", "-k", "."]).decode()
temporary = Path(tempfile.mkdtemp(prefix="crow-github-recovery-", dir=root / ".numi"))
bare = temporary / "repo.git"
clone = subprocess.run(["git", "clone", "--bare", "--depth=1", "--filter=blob:none", "--single-branch",
                        "--branch", tool.BRANCH, "https://github.com/Numi2/numi-lab.git", str(bare)],
                       capture_output=True, text=True, check=True, timeout=180)
os.chdir(bare)
if tool.git("rev-parse", "HEAD").decode().strip() != expected:
    raise RuntimeError("Fresh GitHub clone has an unexpected commit")
if (bare / "objects/info/alternates").exists():
    raise RuntimeError("Recovery clone must not borrow local objects")
manifest_rel.parent.mkdir(parents=True)
manifest_rel.write_bytes(tool.git("show", "HEAD:" + str(manifest_rel)))
if tool.file_hash(manifest_rel) != tool.file_hash(root / manifest_rel):
    raise RuntimeError("GitHub manifest differs from the captured manifest")
prefixes = [".numi/retained-runtime/recurrent-rollout-admission-04",
            ".numi/runs/american-crow/20261107-learner-probe-01/initial.rolloutpack"]
manifest = tool.load_manifest(manifest_rel)
recovered = []
results = []
for i, prefix in enumerate(prefixes):
    destination = temporary / ("restore-" + str(i))
    destination.mkdir()
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        tool.verify(manifest_rel, destination, prefix)
    results.append(json.loads(report.getvalue()))
    for f in manifest["files"]:
        if f["path"] != prefix and not f["path"].startswith(prefix + "/"):
            continue
        p = destination / f["path"]
        if tool.file_hash(p) != f["sha256"] or p.stat().st_size != f["bytes"] or stat.S_IMODE(p.stat().st_mode) != f["mode"]:
            raise RuntimeError("Restored file or executable-mode readback mismatch")
        recovered.append({"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"], "mode": f["mode"]})
os.chdir(root)
handles = subprocess.run(["/usr/sbin/lsof", "-Fpn", "+D", str(temporary)], capture_output=True, text=True, timeout=60)
if handles.returncode not in (0, 1) or handles.stdout or handles.stderr:
    raise RuntimeError("Recovery temporary tree has open handles")
disposable = []
for directory, dirs, names in os.walk(temporary, followlinks=False):
    if any((Path(directory) / d).is_symlink() for d in dirs):
        raise RuntimeError("Unexpected recovery directory symlink")
    for name in names:
        p = Path(directory) / name
        if not stat.S_ISREG(p.lstat().st_mode):
            raise RuntimeError("Unexpected recovery file type")
        disposable.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                           "sha256": tool.file_hash(p)})
for f in disposable:
    (root / f["path"]).unlink()
for directory, _, _ in os.walk(temporary, topdown=False):
    Path(directory).rmdir()
tool.write_json(receipt, {"github_commit": expected, "manifest_sha256": tool.file_hash(root / manifest_rel),
                         "clone_stdout": clone.stdout, "clone_stderr": clone.stderr,
                         "recovered": recovered, "verification": results,
                         "temporary_files_removed": disposable, "temporary_open_handles": 0,
                         "df_before": before, "df_after": subprocess.check_output(["df", "-k", "."]).decode(),
                         "scope": "Fresh independent GitHub blob recovery and disk readback; no scientific rerun or new capability qualification."})
print(json.dumps({"github_commit": expected, "recovered_files": len(recovered),
                  "recovered_bytes": sum(f["bytes"] for f in recovered), "temporary_tree_removed": not temporary.exists()}))
