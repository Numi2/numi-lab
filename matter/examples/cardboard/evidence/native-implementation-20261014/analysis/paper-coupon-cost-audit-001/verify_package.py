#!/usr/bin/env python3
"""Source-bound static verification for the projection-reuse prototype."""
from __future__ import annotations
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())
BASE = pathlib.Path(MANIFEST["base_snapshot"])
PATCH = HERE / "projection-reuse.patch"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

for rel, meta in MANIFEST["files"].items():
    data = (BASE / rel).read_bytes()
    assert sha(data) == meta["base_sha256"], f"base hash mismatch: {rel}"

with tempfile.TemporaryDirectory(prefix="projection-reuse-audit-") as tmp:
    root = pathlib.Path(tmp)
    for rel in MANIFEST["files"]:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / rel, dst)
    proc = subprocess.run(
        ["patch", "-p1", "--input", str(PATCH)], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert proc.returncode == 0, proc.stdout
    for rel, meta in MANIFEST["files"].items():
        patched = (root / rel).read_bytes()
        assert sha(patched) == meta["prototype_sha256"], f"prototype hash mismatch: {rel}"

    fem_base = (BASE / "matter/src/metal/fem.metalinc").read_text()
    fem_new = (root / "matter/src/metal/fem.metalinc").read_text()
    def body(text: str, start: str, end: str) -> str:
        a = text.index(start)
        b = text.index(end, a)
        return text[a:b]
    internal_start = "kernel void nm_fem_internal_forces("
    internal_end = "kernel void nm_fem_cohesive_face_forces("
    assert body(fem_base, internal_start, internal_end) == body(fem_new, internal_start, internal_end)
    cache = body(fem_new, "kernel void nm_fem_build_algorithmic_tangent_cache(",
                  "kernel void nm_fem_apply_operator_elements(")
    assert "column < 9u" in cache
    assert "material.stateCount == 15u" in cache
    assert "usesCompensatedImplicitCandidate(material)" in cache
    assert "usesScaledCorrectionProjection(material)" in cache
    assert "column == 0u" in cache and "column > 0u" in cache
    assert "deformationLow" in cache
    common = (root / "matter/src/metal/common.metalinc").read_text()
    tangent = body(common, "inline void evaluateAlgorithmicStressTangent(",
                   "inline bool evaluateMaterialValidity(")
    assert "projectMaterialState(" in tangent
    assert "reusedProjectedState != nullptr" in tangent
    runtime_base = (BASE / "matter/src/runtime.mm").read_bytes()
    assert sha(runtime_base) == MANIFEST["runtime_untouched_sha256"]

print("PASS: exact base hashes, patch application, patched hashes, 9-column scope, unchanged Armijo force kernel, and runtime untouched")
print("LIMIT: source-shape only; no Metal compile, native parity, or performance result")
