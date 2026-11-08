#!/usr/bin/env python3
"""Verify a sparse source archive against Git object identities and stamp an install.

This verifies content identity, not authorship or that installed binaries were
built from that source. Expected hashes must come from an independent channel.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def object_id(kind, data):
    return hashlib.sha1(kind.encode() + b" " + str(len(data)).encode() + b"\0" + data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def tree_entries(data):
    result = {}
    offset = 0
    while offset < len(data):
        space = data.index(b" ", offset)
        nul = data.index(b"\0", space)
        mode, name = data[offset:space], data[space + 1:nul]
        oid = data[nul + 1:nul + 21]
        if len(oid) != 20 or not name or name in result:
            raise ValueError("malformed tree")
        result[name] = (mode, oid.hex())
        offset = nul + 21
    return result


def commit_tree(commit):
    first = commit.split(b"\n", 1)[0]
    if not first.startswith(b"tree ") or len(first) != 45:
        raise ValueError("invalid commit tree")
    return first[5:].decode("ascii")


def make_proof(repo, root):
    manifest = json.loads((root / "BUNDLE-MANIFEST.json").read_bytes())
    revision = manifest["git_revision"]
    if git(repo, "rev-parse", "--show-object-format").strip() != b"sha1":
        raise ValueError("only SHA-1 Git object format is supported")
    if git(repo, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("repository revision differs from manifest")
    commit = git(repo, "cat-file", "commit", revision)
    if object_id("commit", commit) != revision:
        raise ValueError("commit hash mismatch")
    trees = {}
    def get_tree(oid):
        if oid not in trees:
            data = git(repo, "cat-file", "tree", oid)
            if object_id("tree", data) != oid:
                raise ValueError("tree hash mismatch")
            trees[oid] = base64.b64encode(data).decode()
        return tree_entries(base64.b64decode(trees[oid]))
    for path in manifest["files"]:
        if not path.startswith("source/"):
            continue
        parts = path[len("source/"):].encode().split(b"/")
        oid = commit_tree(commit)
        for part in parts[:-1]:
            mode, oid = get_tree(oid)[part]
            if mode not in (b"40000", b"040000"):
                raise ValueError("non-tree path component")
        get_tree(oid)[parts[-1]]
    proof = {"schema": "numi.git-inclusion-proof.v1", "object_format": "sha1",
             "commit": base64.b64encode(commit).decode(), "trees": trees}
    (root / "GIT-PROOF.json").write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
    print("proof trees:", len(trees), "revision:", revision)


def verify_archive(archive, expected_sha, expected_revision, source=None):
    raw = archive.read_bytes()
    if sha256(raw) != expected_sha:
        raise ValueError("archive SHA-256 does not match independently supplied digest")
    with tarfile.open(fileobj=__import__("io").BytesIO(raw), mode="r:gz") as tf:
        members = {}
        for member in tf:
            name = member.name.rstrip("/")
            if name in members or name.startswith("/") or ".." in name.split("/"):
                raise ValueError("duplicate or unsafe archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("unsupported non-file archive entry")
            members[name] = (tf.extractfile(member).read(), member.mode & 0o777)
    manifest_bytes = members["BUNDLE-MANIFEST.json"][0]
    proof_bytes = members["GIT-PROOF.json"][0]
    manifest = json.loads(manifest_bytes)
    proof = json.loads(proof_bytes)
    if manifest["git_revision"] != expected_revision or proof["schema"] != "numi.git-inclusion-proof.v1" or proof["object_format"] != "sha1":
        raise ValueError("revision or proof format mismatch")
    commit = base64.b64decode(proof["commit"], validate=True)
    if object_id("commit", commit) != expected_revision:
        raise ValueError("commit object does not match expected revision")
    expected_files = set(manifest["files"])
    actual_files = {name for name in members if name.startswith(("source/", "qualification/"))}
    if expected_files != actual_files or manifest["file_count"] != len(expected_files):
        raise ValueError("archive file set differs from manifest")
    if source is not None:
        actual_source = {"source/" + str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() or p.is_symlink()}
        if actual_source != {p for p in expected_files if p.startswith("source/")}:
            raise ValueError("build source file set differs from archive")
    trees = {}
    def tree(oid):
        if oid not in trees:
            data = base64.b64decode(proof["trees"][oid], validate=True)
            if object_id("tree", data) != oid:
                raise ValueError("tree proof hash mismatch")
            trees[oid] = tree_entries(data)
        return trees[oid]
    for name, item in manifest["files"].items():
        data, mode = members[name]
        if item["kind"] != "file" or item["size_bytes"] != len(data) or item["sha256"] != sha256(data) or item["mode"] != oct(mode):
            raise ValueError("manifest content or mode mismatch: " + name)
        if name.startswith("source/"):
            if source is not None:
                path = source / name[len("source/"):]
                if path.is_symlink() or not path.is_file() or path.read_bytes() != data or (path.stat().st_mode & 0o777) != mode:
                    raise ValueError("build source content or mode mismatch: " + name)
            parts = name[len("source/"):].encode().split(b"/")
            oid = commit_tree(commit)
            for part in parts[:-1]:
                entry_mode, oid = tree(oid)[part]
                if entry_mode not in (b"40000", b"040000"):
                    raise ValueError("non-tree path component: " + name)
            entry_mode, blob = tree(oid)[parts[-1]]
            expected_mode = b"100755" if mode & 0o111 else b"100644"
            if entry_mode != expected_mode or object_id("blob", data) != blob:
                raise ValueError("Git blob or mode mismatch: " + name)
    return {"source_revision": expected_revision, "archive_sha256": expected_sha,
            "manifest_sha256": sha256(manifest_bytes), "proof_sha256": sha256(proof_bytes),
            "source_files": len([n for n in expected_files if n.startswith("source/")])}


ARTIFACTS = ("bin/metalrobo_task_rollout", "bin/metalrobo_task_train", "lib/libmetalrobo.dylib",
             "lib/metalrobo/MetalRobo.metallib", "share/numi/python/metalrobo/policy_selection.py")


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".provenance-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def stamp(archive, expected_sha, expected_revision, source, prefix):
    verified = verify_archive(archive, expected_sha, expected_revision, source)
    artifacts = {name: sha256((prefix / name).read_bytes()) for name in ARTIFACTS}
    verified.update({"schema": "numi.installed-source-provenance.v1", "artifacts": artifacts,
                     "limitations": "source identity and installed hashes only; not authorship or a build attestation"})
    dest = prefix / "share/numi"
    # Archive and receipt precede REVISION; verification failures write nothing.
    archive_bytes = archive.read_bytes()
    if sha256(archive_bytes) != expected_sha:
        raise ValueError("archive changed during verification")
    atomic_write(dest / "SOURCE-BUNDLE.tar.gz", archive_bytes)
    atomic_write(dest / "SOURCE-PROVENANCE.json", (json.dumps(verified, sort_keys=True, indent=2) + "\n").encode())
    atomic_write(dest / "REVISION", ("verified-source:" + expected_revision + "\n").encode())
    print(json.dumps(verified, sort_keys=True))


def audit(prefix, expected_sha, expected_revision):
    dest = prefix / "share/numi"
    actual = verify_archive(dest / "SOURCE-BUNDLE.tar.gz", expected_sha, expected_revision)
    receipt = json.loads((dest / "SOURCE-PROVENANCE.json").read_bytes())
    if receipt.get("schema") != "numi.installed-source-provenance.v1" or any(receipt.get(k) != v for k, v in actual.items()):
        raise ValueError("installed receipt mismatch")
    if (dest / "REVISION").read_text().strip() != "verified-source:" + expected_revision:
        raise ValueError("installed revision mismatch")
    if receipt.get("artifacts") != {name: sha256((prefix / name).read_bytes()) for name in ARTIFACTS}:
        raise ValueError("installed artifact hash mismatch")
    print(json.dumps(receipt, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-proof")
    create.add_argument("--repo", required=True, type=Path)
    create.add_argument("--bundle", required=True, type=Path)
    for name in ("stamp", "audit"):
        q = sub.add_parser(name)
        q.add_argument("--prefix", required=True, type=Path)
        q.add_argument("--expected-archive-sha256", required=True)
        q.add_argument("--expected-revision", required=True)
        if name == "stamp":
            q.add_argument("--archive", required=True, type=Path)
            q.add_argument("--source", required=True, type=Path)
    a = p.parse_args()
    if a.command == "create-proof":
        make_proof(a.repo, a.bundle)
    elif a.command == "stamp":
        stamp(a.archive, a.expected_archive_sha256, a.expected_revision, a.source, a.prefix)
    else:
        audit(a.prefix, a.expected_archive_sha256, a.expected_revision)


if __name__ == "__main__":
    main()
