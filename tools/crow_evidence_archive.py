#!/usr/bin/env python3
"""Content-deduplicated Git recovery for explicitly selected crow evidence roots.

Capture never removes inputs. Verification reads committed Git objects, including
archives omitted from the checkout. Pruning requires an exact remote HEAD match,
verified recovery of every file, no open handles, and unchanged source hashes.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

CHUNK = 32 * 1024 * 1024
BRANCH = "agent/american-crow-rl-20261107"


def git(*args, input=None):
    return subprocess.check_output(["git", *args], input=input)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(CHUNK), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tree(revision="HEAD"):
    result = {}
    for entry in git("ls-tree", "-rz", revision).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        if kind == b"blob":
            result[path.decode()] = (mode.decode(), oid.decode())
    return result


class Objects:
    def __enter__(self):
        self.p = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)
        return self

    def read(self, oid):
        self.p.stdin.write(oid.encode() + b"\n")
        self.p.stdin.flush()
        header = self.p.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"Missing Git blob: {oid}")
        data = self.p.stdout.read(int(header[2]))
        if len(data) != int(header[2]) or self.p.stdout.read(1) != b"\n":
            raise ValueError("Truncated Git object stream")
        if hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != oid:
            raise ValueError("Git blob identity mismatch")
        return data

    def __exit__(self, *exc):
        self.p.stdin.close()
        self.p.stdout.close()
        if self.p.wait() != 0 and not exc[0]:
            raise ValueError("Git object reader failed")


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    return Path(value)


def selected_roots():
    return [Path(".numi/runs/american-crow"), Path(".numi/retained-runtime")] + sorted(
        p for p in Path("build").iterdir()
        if p.is_dir() and (p.name.startswith("crow-") or p.name.startswith("gru-memory-")))


def capture(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "blobs").mkdir()
    entries = tree()
    roots = selected_roots()
    if any(str(safe_path(p)).startswith(tuple(str(r) + "/" for r in roots)) for p in entries):
        raise ValueError("Selected roots contain tracked files")
    known = {}
    indexed = 0
    with Objects() as objects:
        for path, (_, oid) in entries.items():
            if not path.startswith("artifacts/crow-"):
                continue
            raw = objects.read(oid)
            known.setdefault((digest(raw), len(raw)), {"path": path, "encoding": "raw"})
            if raw[:2] == b"\x1f\x8b":
                decoded = gzip.decompress(raw)
                known.setdefault((digest(decoded), len(decoded)), {"path": path, "encoding": "gzip"})
            indexed += 1
            if indexed % 2000 == 0:
                print(f"Indexed {indexed} committed archive blobs", flush=True)
    files = []
    added = 0
    for root in roots:
        if root.is_symlink():
            raise ValueError(f"Symlink root: {root}")
        for directory, dirs, names in os.walk(root, followlinks=False):
            if any((Path(directory) / d).is_symlink() for d in dirs):
                raise ValueError("Directory symlink requires separate review")
            for name in sorted(names):
                p = Path(directory) / name
                s = p.lstat()
                if not stat.S_ISREG(s.st_mode):
                    raise ValueError(f"Non-regular evidence requires separate review: {p}")
                sha = file_hash(p)
                key = (sha, s.st_size)
                backing = known.get(key)
                if backing is None:
                    chunks = []
                    with p.open("rb") as f:
                        while data := f.read(CHUNK):
                            part_sha = digest(data)
                            dest = out / "blobs" / (part_sha + ".gz")
                            if not dest.exists():
                                packed = gzip.compress(data, compresslevel=6, mtime=0)
                                dest.write_bytes(packed)
                                added += len(packed)
                            chunks.append({"path": str(dest), "sha256": part_sha, "bytes": len(data)})
                    backing = {"encoding": "chunks-gzip", "chunks": chunks}
                    known[key] = backing
                after = p.lstat()
                if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode) != (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode):
                    raise ValueError(f"Evidence changed during capture: {p}")
                files.append({"path": str(p), "sha256": sha, "bytes": s.st_size,
                              "mode": stat.S_IMODE(s.st_mode), "mtime_ns": s.st_mtime_ns,
                              "backing": backing})
                if len(files) % 2000 == 0:
                    print(f"Mapped {len(files)} source files; added {added} compressed bytes", flush=True)
    result = {"schema": "crow-git-evidence-archive-v1", "base_commit": git("rev-parse", "HEAD").decode().strip(),
              "roots": [str(r) for r in roots], "files": files,
              "summary": {"files": len(files), "source_bytes": sum(f["bytes"] for f in files),
                          "new_compressed_bytes": added, "existing_archive_blobs_indexed": indexed,
                          "unique_file_contents": len({(f["sha256"], f["bytes"]) for f in files})}}
    write_json(out / "manifest.json", result)
    (out / ".gitattributes").write_text("*.gz binary\n")
    print(json.dumps(result["summary"]), flush=True)


def load_manifest(path):
    value = json.loads(path.read_text())
    if value["schema"] != "crow-git-evidence-archive-v1":
        raise ValueError("Unsupported manifest")
    roots = [safe_path(r) for r in value["roots"]]
    for r in roots:
        if str(r) not in (".numi/runs/american-crow", ".numi/retained-runtime") and not (
                r.parent == Path("build") and r.name.startswith(("crow-", "gru-memory-"))):
            raise ValueError("Manifest root is outside the owned crow evidence scope")
    seen = set()
    for f in value["files"]:
        p = safe_path(f["path"])
        if not any(p.is_relative_to(r) for r in roots) or f["path"] in seen:
            raise ValueError("Duplicate or out-of-scope evidence path")
        seen.add(f["path"])
        b = f["backing"]
        for item in b.get("chunks", [b]):
            if not str(safe_path(item["path"])).startswith("artifacts/crow-"):
                raise ValueError("Backing must be a crow Git artifact")
    return value


def content(f, entries, objects):
    b = f["backing"]
    if b["encoding"] in ("raw", "gzip"):
        raw = objects.read(entries[b["path"]][1])
        yield gzip.decompress(raw) if b["encoding"] == "gzip" else raw
    elif b["encoding"] == "chunks-gzip":
        for part in b["chunks"]:
            data = gzip.decompress(objects.read(entries[part["path"]][1]))
            if len(data) != part["bytes"] or digest(data) != part["sha256"]:
                raise ValueError("Chunk recovery mismatch")
            yield data
    else:
        raise ValueError("Unknown archive encoding")


def verify(manifest, restore=None, prefix=None):
    entries = tree()
    # The manifest itself must already be committed with exactly these bytes.
    manifest_oid = entries[str(manifest)][1]
    cache = set()
    selected = 0
    with Objects() as objects:
        if objects.read(manifest_oid) != manifest.read_bytes():
            raise ValueError("Uncommitted manifest changes")
        value = load_manifest(manifest)
        for f in value["files"]:
            if prefix and not (f["path"] == prefix or f["path"].startswith(prefix.rstrip("/") + "/")):
                continue
            selected += 1
            key = (f["sha256"], f["bytes"])
            if key in cache and restore is None:
                continue
            target = None
            if restore is not None:
                target = restore / safe_path(f["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            stream = target.open("xb") if target else None
            try:
                for data in content(f, entries, objects):
                    h.update(data)
                    size += len(data)
                    if stream:
                        stream.write(data)
            finally:
                if stream:
                    stream.close()
            if size != f["bytes"] or h.hexdigest() != f["sha256"]:
                raise ValueError(f"Recovered evidence mismatch: {f['path']}")
            if target:
                target.chmod(f["mode"])
            cache.add(key)
    if not selected:
        raise ValueError("No evidence selected")
    print(json.dumps({"verified_files": selected, "verified_unique_contents": len(cache),
                      "manifest_sha256": file_hash(manifest), "commit": git("rev-parse", "HEAD").decode().strip()}), flush=True)
    return value


def prune(manifest, receipt):
    current = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "origin", "refs/heads/" + BRANCH).decode().split()[0]
    if current != remote or git("branch", "--show-current").decode().strip() != BRANCH:
        raise ValueError("Archive HEAD is not confirmed on the crow remote branch")
    value = verify(manifest)
    before = subprocess.check_output(["df", "-k", "."]).decode()
    handles = []
    for root in [".numi/runs/american-crow", ".numi/retained-runtime", "build"]:
        p = subprocess.run(["/usr/sbin/lsof", "-Fpn", "+D", root], capture_output=True, text=True, timeout=60)
        if p.returncode not in (0, 1) or p.stdout or p.stderr:
            raise ValueError(f"Open handles or lsof failure at {root}: {p.stdout}{p.stderr}")
        handles.append({"root": root, "exit_code": p.returncode, "open_handles": 0})
    # Validate the entire deletion set before removing its first file.
    for f in value["files"]:
        path = safe_path(f["path"])
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"Symlink parent prevents cleanup: {path}")
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_size != f["bytes"] or s.st_mtime_ns != f["mtime_ns"] or stat.S_IMODE(s.st_mode) != f["mode"] or file_hash(path) != f["sha256"]:
            raise ValueError(f"Changed evidence prevents cleanup: {path}")
    for f in value["files"]:
        safe_path(f["path"]).unlink()
    for root in value["roots"]:
        for directory, _, _ in os.walk(root, topdown=False):
            try:
                Path(directory).rmdir()
            except OSError:
                pass  # Never remove new, uncovered files.
    write_json(receipt, {"archive_commit": current, "remote_commit": remote,
                        "manifest": str(manifest), "manifest_sha256": file_hash(manifest),
                        "removed_files": len(value["files"]), "removed_logical_bytes": value["summary"]["source_bytes"],
                        "open_handle_checks": handles, "df_before": before,
                        "df_after": subprocess.check_output(["df", "-k", "."]).decode()})
    print(receipt.read_text(), flush=True)


def omit_archives(receipt):
    current = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "origin", "refs/heads/" + BRANCH).decode().split()[0]
    if current != remote:
        raise ValueError("Push the archive before omitting checkout copies")
    keep = ("artifacts/crow-rl-feeding-behavior-20261107/policies/",
            "artifacts/crow-rl-recurrent-actor-20261107/inputs/PolicyABI3.metallib",
            "artifacts/crow-rl-retinal-contract-20261107/inputs/legacy-MetalRobo.metallib")
    before = subprocess.check_output(["df", "-k", "."]).decode()
    handles = subprocess.run(["/usr/sbin/lsof", "-Fpn", "+D", "artifacts"],
                             capture_output=True, text=True, timeout=60)
    if handles.returncode not in (0, 1) or handles.stdout or handles.stderr:
        raise ValueError("Open archive handles or incomplete lsof inspection")
    files = []
    for name, (mode, oid) in tree().items():
        if not name.startswith("artifacts/crow-") or name.startswith(keep):
            continue
        if name.startswith("artifacts/crow-rl-git-recovery-") and "/blobs/" not in name:
            continue  # Keep compact recovery controls and receipts visible.
        path = safe_path(name)
        if not path.exists():
            continue
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode) or mode not in ("100644", "100755"):
            raise ValueError(f"Unexpected archive file type: {name}")
        h = hashlib.sha1(b"blob " + str(s.st_size).encode() + b"\0")
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(CHUNK), b""):
                h.update(data)
        if h.hexdigest() != oid:
            raise ValueError(f"Uncommitted archive content: {name}")
        files.append({"path": name, "git_blob": oid, "bytes": s.st_size})
    git("update-index", "--skip-worktree", "-z", "--stdin",
        input=b"".join(f["path"].encode() + b"\0" for f in files))
    for f in files:
        safe_path(f["path"]).unlink()
    for directory, _, _ in os.walk("artifacts", topdown=False):
        try:
            Path(directory).rmdir()
        except OSError:
            pass
    write_json(receipt, {"commit": current, "remote_commit": remote,
                        "omitted_files": files, "omitted_logical_bytes": sum(f["bytes"] for f in files),
                        "preserved_input_prefixes": list(keep), "open_handles": 0,
                        "df_before": before, "df_after": subprocess.check_output(["df", "-k", "."]).decode()})
    print(json.dumps({"omitted_files": len(files), "omitted_logical_bytes": sum(f["bytes"] for f in files),
                      "receipt": str(receipt)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify", "restore", "prune", "omit"])
    parser.add_argument("path", type=Path, help="New artifact directory for capture, new receipt for omit, manifest otherwise")
    parser.add_argument("--output", type=Path, help="New restore directory or cleanup receipt")
    parser.add_argument("--prefix", help="Exact evidence path or subtree to restore")
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.path)
    elif args.action == "verify":
        verify(args.path)
    elif args.action == "restore":
        if not args.output:
            parser.error("restore requires --output")
        args.output.mkdir(parents=True, exist_ok=False)
        verify(args.path, args.output, args.prefix)
    elif args.action == "prune":
        if not args.output or args.output.exists():
            parser.error("prune requires a new --output receipt")
        prune(args.path, args.output)
    else:
        if args.path.exists():
            parser.error("omit requires a new receipt path")
        omit_archives(args.path)


if __name__ == "__main__":
    main()
