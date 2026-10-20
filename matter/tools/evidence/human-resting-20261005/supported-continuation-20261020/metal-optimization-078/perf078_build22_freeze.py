#!/usr/bin/env python3
"""Freeze a completed build-22 cohort and every dirty source file it used.

This script only copies/hashes already-built artifacts; it never runs a build
or a simulator. Supply the three artifact paths from the completed CMake build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

REPO = Path("/Users/numi/numi-human-aortic-contact-opt-20261019")
BUILD_ROOT = REPO / "build-integrated-validation-20261019"
FROZEN = BUILD_ROOT / "perf078-build22-frozen"
EVIDENCE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-078"
BUILD_OUTPUT = BUILD_ROOT / "profile-build-039"
FLAT_SOURCE_MANIFEST = EVIDENCE / "native-build22-source-manifest.json"
PROVENANCE_MANIFEST = EVIDENCE / "native-build22-source-provenance.json"
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".m", ".mm", ".metal", ".metalinc", ".inc", ".inl", ".ipp",
    ".swift", ".py", ".cmake",
}
EXCLUDED_PARTS = {".git", "build", "build-integrated-validation-20261019"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, env: dict[str, str] | None = None) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO, env=env)


def dirty_source_paths() -> list[str]:
    tracked = git("diff", "--name-only", "--diff-filter=ACDMRTUXB", "HEAD").decode().splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    paths = sorted(set(tracked + untracked))
    selected: list[str] = []
    for rel in paths:
        path = Path(rel)
        parts = set(path.parts)
        if parts & EXCLUDED_PARTS or "evidence" in parts or ".numi" in parts:
            continue
        if path.name == "CMakeLists.txt" or path.suffix in SOURCE_SUFFIXES:
            selected.append(rel)
    if not selected:
        raise RuntimeError("no dirty source files found to freeze")
    return selected


def status_by_path(paths: list[str]) -> dict[str, str]:
    raw = git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths).decode()
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if len(line) >= 4:
            result[line[3:]] = line[:2]
    return result


def source_patch(paths: list[str]) -> bytes:
    # A temporary Git index lets the patch include modified, staged, deleted,
    # and untracked files without touching the user's real index.
    fd, index_name = tempfile.mkstemp(prefix="perf078-build22-index-")
    os.close(fd)
    index = Path(index_name)
    index.unlink()
    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(index)
    try:
        subprocess.run(["git", "read-tree", "HEAD"], cwd=REPO, env=env, check=True)
        subprocess.run(["git", "add", "-A", "--", *paths], cwd=REPO, env=env, check=True)
        return subprocess.check_output(
            ["git", "diff", "--cached", "--binary", "--full-index", "HEAD", "--", *paths],
            cwd=REPO, env=env)
    finally:
        index.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=BUILD_OUTPUT / "bin/numi-human-native",
                        help="completed build-22 executable (defaults to profile-build-039)")
    parser.add_argument("--dylib", type=Path,
                        default=BUILD_OUTPUT / "lib/libmetalrobo.dylib",
                        help="completed build-22 dylib (defaults to profile-build-039)")
    parser.add_argument("--metallib", type=Path,
                        default=BUILD_OUTPUT / "matter/shaders/NumiMatter.metallib",
                        help="completed build-22 metallib (defaults to profile-build-039)")
    parser.add_argument("--frozen-dir", type=Path, default=FROZEN)
    args = parser.parse_args()

    frozen = args.frozen_dir.resolve()
    if frozen.exists():
        parser.error(f"frozen output already exists: {frozen}")
    artifacts = {
        "numi-human-native": args.binary.resolve(),
        "libmetalrobo.dylib": args.dylib.resolve(),
        "NumiMatter.metallib": args.metallib.resolve(),
    }
    for name, path in artifacts.items():
        if not path.is_file():
            parser.error(f"missing completed artifact {name}: {path}")

    paths = dirty_source_paths()
    statuses = status_by_path(paths)
    base_revision = git("rev-parse", "HEAD").decode().strip()
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.mkdir(parents=False)
    for name, source in artifacts.items():
        shutil.copy2(source, frozen / name)

    source_hashes: dict[str, str] = {}
    with tarfile.open(frozen / "source-changes.tar.gz", "w:gz") as archive:
        for rel in paths:
            source = REPO / rel
            if source.is_file():
                archive.add(source, arcname=rel, recursive=False)
                source_hashes[rel] = sha256(source)
            else:
                source_hashes[rel] = "DELETED"
    (frozen / "source.patch").write_bytes(source_patch(paths))
    (frozen / "source-manifest.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")

    artifact_manifest = {
        name: sha256(frozen / name) for name in artifacts
    }
    artifact_manifest.update({
        "source.patch": sha256(frozen / "source.patch"),
        "source-changes.tar.gz": sha256(frozen / "source-changes.tar.gz"),
        "source-manifest.json": sha256(frozen / "source-manifest.json"),
    })
    artifact_manifest_path = frozen / "manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")

    FLAT_SOURCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    FLAT_SOURCE_MANIFEST.write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema": 1,
        "build": "perf078-build22",
        "base_revision": base_revision,
        "frozen_artifact_directory": str(frozen),
        "artifact_manifest_sha256": sha256(artifact_manifest_path),
        "artifacts": artifact_manifest,
        "source_files": [
            {"path": rel, "status": statuses.get(rel, "??"),
             "sha256": source_hashes[rel]}
            for rel in paths
        ],
        "source_patch_sha256": artifact_manifest["source.patch"],
        "source_archive_sha256": artifact_manifest["source-changes.tar.gz"],
    }
    PROVENANCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_MANIFEST.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "frozen_dir": str(frozen),
        "base_revision": base_revision,
        "source_count": len(paths),
        "source_manifest": str(FLAT_SOURCE_MANIFEST),
        "provenance_manifest": str(PROVENANCE_MANIFEST),
        "artifact_hashes": artifact_manifest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
