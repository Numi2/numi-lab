#!/usr/bin/env python3
"""Offline integrity checks for the external coupled-capture runtime overlay."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text())
    base = HERE / "runtime.mm.base"
    patched = HERE / "runtime.mm"
    patch = HERE / manifest["patch_file"]
    assert sha256(base) == manifest["base_runtime_sha256"]
    assert sha256(patched) == manifest["overlay_runtime_sha256"]
    assert sha256(patch) == manifest["patch_sha256"]

    source = patched.read_text()
    start = source.index("const auto encodeFEMCoupledNewtonCapture")
    end = source.index("const auto capturedArrays = arrays;", start)
    capture = source[start:end]
    names = re.findall(r'append(?:Buffer|Borrowed|Host)\("([^\"]+)"', capture)
    duplicates = [name for name, count in Counter(names).items() if count != 1]
    assert not duplicates, f"duplicate array names: {duplicates}"
    assert len(names) == manifest["unique_named_array_slots"]
    names_hash = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
    assert names_hash == manifest["array_names_sha256"]
    for required in (
        "full_rhs_before_fgmres",
        "fem_accepted_nodes",
        "fem_candidate_nodes",
        "material_history_accepted",
        "contact_samples",
        "deformable_contacts",
        "cohesive_faces_accepted",
        "rigid_states",
        "coupled_generalized_candidate",
        "coupled_candidate_bodies",
        "fgmres_basis",
        "regional_graph_factors",
    ):
        assert required in names, f"missing required array {required}"
    assert "articulated_generalized_unknowns_not_supported" in capture
    assert "coupledArticulatedNv != 0u" in capture
    selector_start = source.index("const bool femCoupledCaptureNewtonSelected")
    selector_end = source.index(
        "if (femTrueResidualEnabled || femCoupledCaptureNewtonSelected)",
        selector_start,
    )
    assert "coupledArticulatedNv == 0u" in source[selector_start:selector_end]
    for field in (
        "bodyWrenchCount > 64u",
        "externalWrenchCount > 64u",
        "rigidContactConstraintStride > 64u",
    ):
        assert field in capture
    for required in (
        "request.rigid.qStride != 0u",
        "request.rigid.currentBodyCount != 0u",
        "request.rigid.externalWrenchCount != 0u",
        "request.rigidContactConstraintStride != 0u",
    ):
        assert required in capture
    assert "after_fgmres_measure_and_human_direction_map_before_armijo_trial" in source
    assert "captureCap = 32u * 1024u * 1024u" in capture
    assert "captureArrayCap = 192u" in capture
    assert manifest["max_arrays"] == 192
    assert len(names) <= manifest["max_arrays"]
    assert "O_EXCL" in source
    assert "\\\"max_arrays\\\":192" in source
    assert "\\\"max_payload_bytes\\\":33554432" in source

    # Apply to a temporary exact-base tree and compare the reconstructed file.
    with tempfile.TemporaryDirectory(prefix="coupled-capture-overlay-") as tmp:
        root = Path(tmp)
        target = root / "matter/src/runtime.mm"
        target.parent.mkdir(parents=True)
        target.write_bytes(base.read_bytes())
        applied = subprocess.run(
            ["patch", "-p1"], cwd=root, input=patch.read_bytes(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if applied.returncode != 0:
            raise AssertionError(applied.stdout.decode(errors="replace"))
        assert hashlib.sha256(target.read_bytes()).hexdigest() == sha256(patched)

    print(json.dumps({
        "status": "PASS",
        "array_count": len(names),
        "base_sha256": sha256(base),
        "overlay_sha256": sha256(patched),
        "patch_sha256": sha256(patch),
        "patch_reapplies_to_exact_base": True,
        "native_execution": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
