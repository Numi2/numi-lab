#!/usr/bin/env python3
"""Materialize the authenticated ARDY HyperPolicy source overlay for CMake.

Large legacy executor seams are patched into the build tree rather than
rewritten in-place. The generated files are the canonical compiler inputs;
source-tree inputs remain immutable and every patch is cardinality checked.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tarfile

EXPECTED_SHA256 = "fd151020faab300c6f280d394e6c27b9985c068b3fe0007e5c14633ba8acce6a"
PART_COUNT = 7

OVERLAY_INPUTS = (
    "include/metalrobo/HyperPolicyProgram.hpp",
    "src/core/hyper_policy/HyperPolicyProgram.00.inc",
    "src/core/hyper_policy/HyperPolicyProgram.01.inc",
    "src/core/hyper_policy/HyperPolicyProgram.02.inc",
    "src/core/hyper_policy/HyperPolicyProgram.03.inc",
    "src/core/HyperPolicyProgram.cpp",
    "include/metalrobo/MetalWorld.hpp",
    "src/metal/MetalWorld.mm",
    "include/metalrobo/c_api.h",
    "src/c_api.cpp",
    "bindings/swift/MetalRoboTaskRollout.swift",
    "apps/task_rollout.swift",
    "python/pyproject.toml",
    "python/metalrobo/hyperpolicy/mlx_actor.py",
    "python/metalrobo/hyperpolicy/runtime.py",
    "python/metalrobo/mlx_ardy_hyperpolicy.py",
)


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"unsafe HyperPolicy payload member: {member.name}")
    archive.extractall(destination)


def copy_source(source: Path, output: Path, relative: str) -> None:
    src = source / relative
    dst = output / relative
    if not src.is_file():
        raise RuntimeError(f"required HyperPolicy integration input is missing: {relative}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_patcher(path: Path):
    spec = importlib.util.spec_from_file_location("numi_hyperpolicy_patcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load HyperPolicy integration patcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    source = Path(arguments.source_root).resolve()
    output = Path(arguments.output_root).resolve()
    fragments = source / ".github/hyperpolicy_integration_payload"

    encoded = b"".join(
        (fragments / f"{index:02d}.part").read_bytes()
        for index in range(PART_COUNT)
    )
    payload = base64.b64decode(encoded, validate=True)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"HyperPolicy payload authentication failed: {digest} != {EXPECTED_SHA256}"
        )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    archive_path = output / ".hyperpolicy-source.tar.gz"
    archive_path.write_bytes(payload)
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, output)
    archive_path.unlink()

    for relative in OVERLAY_INPUTS:
        copy_source(source, output, relative)

    patcher = load_patcher(output / "tools/apply_ardy_hyperpolicy_integration.py")
    patcher.ROOT = output
    patcher.patch_hyper_program()
    patcher.patch_metal_world_header()
    patcher.patch_metal_world_impl()
    patcher.patch_c_api_header()
    patcher.patch_c_api_impl()
    patcher.patch_swift_binding()
    patcher.patch_task_rollout_app()
    patcher.patch_python()

    # HyperPolicy kernels live in a dedicated sidecar metallib so the existing
    # large MetalRobo shader link remains untouched. The runtime derives that
    # sidecar from the main metallib directory and keeps one MTLDevice/library.
    metal_runtime = output / "src/metal/MetalHyperPolicy.mm"
    text = metal_runtime.read_text()
    text = text.replace(
        "#include <limits>\n",
        "#include <limits>\n#include <filesystem>\n",
        1,
    )
    old = """        std::string path = configuration.metallibPath;\n        if (path.empty()) {\n            path = defaultMetallibPath();\n        }\n"""
    new = """        std::string path = configuration.metallibPath;\n        if (path.empty()) {\n            path = defaultMetallibPath();\n        }\n        if (!path.empty()) {\n            path = (std::filesystem::path{path}.parent_path() /\n                    \"MetalRoboHyperPolicy.metallib\").string();\n        }\n"""
    if text.count(old) != 1:
        raise RuntimeError("MetalHyperPolicy sidecar metallib seam drifted")
    metal_runtime.write_text(text.replace(old, new, 1))

    (output / ".materialized").write_text(EXPECTED_SHA256 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
