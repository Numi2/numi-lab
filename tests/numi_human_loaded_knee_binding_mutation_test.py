#!/usr/bin/env python3
"""Fail-closed mutation coverage for the native loaded-knee bundle loader."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


MANIFEST_NAME = "HumanPack.loaded-anatomy-knee.v1.json"
BINDING_NAME = "HumanPack.loaded-anatomy-knee.binding.v1.json"


def canonical(value: Any, *, newline: bool) -> bytes:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return encoded + (b"\n" if newline else b"")


def identity_without(value: dict[str, Any], excluded: str) -> str:
    reduced = dict(value)
    del reduced[excluded]
    return hashlib.sha256(canonical(reduced, newline=False)).hexdigest()


def seal_manifest_and_binding(
    manifest: dict[str, Any], binding: dict[str, Any]
) -> tuple[bytes, bytes]:
    manifest["manifest_sha256"] = identity_without(manifest, "manifest_sha256")
    manifest_bytes = canonical(manifest, newline=True)
    binding["manifest"] = manifest
    binding["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    binding["binding_sha256"] = identity_without(binding, "binding_sha256")
    return manifest_bytes, canonical(binding, newline=True)


def seal_ownership(ownership: dict[str, Any]) -> tuple[str, bytes]:
    identity = identity_without(ownership, "manifest_sha256")
    ownership["manifest_sha256"] = identity
    return identity, canonical(ownership, newline=True)


def reseal_embedded_lab_export(manifest: dict[str, Any]) -> None:
    export = manifest["lab_authoring_export"]
    export["manifest_sha256"] = identity_without(export, "manifest_sha256")
    export_bytes = canonical(export, newline=True)
    manifest["inputs"]["lab_export"] = {
        "schema": export["schema"],
        "file_sha256": hashlib.sha256(export_bytes).hexdigest(),
        "identity_sha256": export["manifest_sha256"],
    }
    mass = manifest["mass_partition"]
    mass["provenance"]["lab_export_manifest_sha256"] = export[
        "manifest_sha256"
    ]
    mass["identity_sha256"] = identity_without(mass, "identity_sha256")


def run_loader(
    cli: Path, manifest: Path, binding: Path, ownership: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(cli), str(manifest), str(binding), str(ownership)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def require_result(
    name: str, result: subprocess.CompletedProcess[str], expected: int
) -> None:
    if result.returncode != expected:
        raise AssertionError(
            f"{name}: expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    detail = result.stderr.strip() if expected != 0 else result.stdout.strip()
    print(f"{name}=passed: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cli", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("ownership", type=Path)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    binding_bytes = args.binding.read_bytes()
    ownership_bytes = args.ownership.read_bytes()
    base_manifest = json.loads(manifest_bytes)
    base_binding = json.loads(binding_bytes)

    # Prove the test reserializer is exactly the bundle's canonical form before
    # using it to construct self-consistent adversarial bundles.
    if canonical(base_manifest, newline=True) != manifest_bytes:
        raise AssertionError("manifest fixture is not Python-canonical JSON")
    if canonical(base_binding, newline=True) != binding_bytes:
        raise AssertionError("binding fixture is not Python-canonical JSON")
    if identity_without(base_manifest, "manifest_sha256") != base_manifest[
        "manifest_sha256"
    ]:
        raise AssertionError("manifest fixture identity is invalid")
    if identity_without(base_binding, "binding_sha256") != base_binding[
        "binding_sha256"
    ]:
        raise AssertionError("binding fixture identity is invalid")

    require_result(
        "baseline",
        run_loader(args.cli, args.manifest, args.binding, args.ownership),
        0,
    )

    def manifest_case(
        name: str, mutation: Callable[[dict[str, Any]], None]
    ) -> None:
        manifest = copy.deepcopy(base_manifest)
        binding = copy.deepcopy(base_binding)
        mutation(manifest)
        mutated_manifest, mutated_binding = seal_manifest_and_binding(
            manifest, binding
        )
        with tempfile.TemporaryDirectory(prefix=f"loaded-knee-{name}-") as raw:
            directory = Path(raw)
            manifest_path = directory / MANIFEST_NAME
            binding_path = directory / BINDING_NAME
            manifest_path.write_bytes(mutated_manifest)
            binding_path.write_bytes(mutated_binding)
            require_result(
                name,
                run_loader(args.cli, manifest_path, binding_path, args.ownership),
                1,
            )

    def ownership_case(
        name: str,
        mutation: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> None:
        manifest = copy.deepcopy(base_manifest)
        binding = copy.deepcopy(base_binding)
        ownership = json.loads(ownership_bytes)
        mutation(ownership, manifest)
        ownership_identity, mutated_ownership = seal_ownership(ownership)
        ownership_file_identity = hashlib.sha256(mutated_ownership).hexdigest()
        manifest["inputs"]["ownership"]["identity_sha256"] = ownership_identity
        manifest["inputs"]["ownership"]["file_sha256"] = ownership_file_identity
        manifest["ownership_manifest_sha256"] = ownership_identity
        manifest["semantic_scope"][
            "ownership_manifest_sha256"
        ] = ownership_identity
        mutated_manifest, mutated_binding = seal_manifest_and_binding(
            manifest, binding
        )
        with tempfile.TemporaryDirectory(prefix=f"loaded-knee-{name}-") as raw:
            directory = Path(raw)
            manifest_path = directory / MANIFEST_NAME
            binding_path = directory / BINDING_NAME
            ownership_path = directory / "ownership.json"
            manifest_path.write_bytes(mutated_manifest)
            binding_path.write_bytes(mutated_binding)
            ownership_path.write_bytes(mutated_ownership)
            require_result(
                name,
                run_loader(args.cli, manifest_path, binding_path, ownership_path),
                1,
            )

    for field in (
        "unloaded_reference_qualified",
        "prestrain_reference_reset_executed",
        "subject_material_calibrated",
        "mesh_convergence_qualified",
        "specimen_load_validation_qualified",
        "clinical_validity_qualified",
        "production_physical_ownership",
        "production_active_force",
        "runtime_x_current_accepted",
        "integrated_human_qualification",
    ):
        manifest_case(
            f"qualification-overclaim-{field.replace('_', '-')}",
            lambda value, field=field: value["qualification"].__setitem__(
                field, True
            ),
        )
    manifest_case(
        "unknown-qualification-key",
        lambda value: value["qualification"].__setitem__("unexpected", False),
    )
    manifest_case(
        "unknown-topology-key",
        lambda value: value["topology"].__setitem__("unexpected", 0),
    )
    manifest_case(
        "embedded-nul",
        lambda value: value.__setitem__("subject_id", value["subject_id"] + "\0x"),
    )
    manifest_case(
        "uint32-overflow",
        lambda value: value["topology"].__setitem__(
            "node_count", value["topology"]["node_count"] + (1 << 32)
        ),
    )
    manifest_case(
        "source-status-forgery",
        lambda value: value.__setitem__(
            "source_ownership_status",
            "partial" if value["source_ownership_status"] == "blocked" else "blocked",
        ),
    )
    manifest_case(
        "human-boundary-forgery",
        lambda value: value.__setitem__(
            "boundary", "Production and clinical qualification established."
        ),
    )

    def lab_export_case(name: str, field: str, replacement: str) -> None:
        def mutate(value: dict[str, Any]) -> None:
            value["lab_authoring_export"][field] = replacement
            reseal_embedded_lab_export(value)

        manifest_case(name, mutate)

    lab_export_case(
        "lab-export-schema-forgery",
        "schema",
        "numi.lab.loaded-knee-authoring-export.v2",
    )
    lab_export_case("lab-export-status-forgery", "status", "production")
    lab_export_case(
        "lab-export-boundary-forgery",
        "boundary",
        "Production and clinical qualification established.",
    )
    manifest_case(
        "raw-node-mass-algorithm-forgery",
        lambda value: value["mass_partition"]["raw_f32_node_mass"].__setitem__(
            "algorithm", "unbound-arithmetic.1"
        ),
    )

    def forge_mapping_encoding(value: dict[str, Any]) -> None:
        for mapping in (
            value["mass_partition"]["provenance"]["mapping"],
            value["lab_authoring_export"]["mapping"],
        ):
            mapping["code_identity_encoding"] = "sha256(unbound-source)"

    manifest_case("mapping-code-encoding-forgery", forge_mapping_encoding)

    def erase_mapping_regions(value: dict[str, Any]) -> None:
        for mapping in (
            value["mass_partition"]["provenance"]["mapping"],
            value["lab_authoring_export"]["mapping"],
        ):
            mapping["diagnostics"]["regions"] = []

    manifest_case("mapping-region-erasure", erase_mapping_regions)

    def forge_ptl_substeps(value: dict[str, Any]) -> None:
        for mapping in (
            value["mass_partition"]["provenance"]["mapping"],
            value["lab_authoring_export"]["mapping"],
        ):
            mapping["diagnostics"]["regions"][4]["substep_count"] = 4

    manifest_case("mapping-ptl-substep-forgery", forge_ptl_substeps)

    def add_mapping_region_key(value: dict[str, Any]) -> None:
        for mapping in (
            value["mass_partition"]["provenance"]["mapping"],
            value["lab_authoring_export"]["mapping"],
        ):
            mapping["diagnostics"]["regions"][0]["unexpected"] = False

    manifest_case("unknown-mapping-region-key", add_mapping_region_key)

    def forge_contact_pair_source_order(value: dict[str, Any]) -> None:
        pairs = value["articular_contact_pairs"]
        pairs[2], pairs[5] = pairs[5], pairs[2]

    manifest_case(
        "contact-pair-source-order-forgery", forge_contact_pair_source_order
    )

    def deep_nesting(value: dict[str, Any]) -> None:
        nested: Any = "leaf"
        for _ in range(192):
            nested = [nested]
        value["qualification"]["unexpected"] = nested

    manifest_case("excessive-json-nesting", deep_nesting)

    def forge_status(
        ownership: dict[str, Any], manifest: dict[str, Any]
    ) -> None:
        replacement = "partial" if ownership["status"] == "blocked" else "blocked"
        ownership["status"] = replacement
        manifest["source_ownership_status"] = replacement

    ownership_case("ownership-status-forgery", forge_status)

    ownership_case(
        "ownership-qualification-overclaim",
        lambda ownership, _manifest: ownership["qualification"].__setitem__(
            "production_physical_ownership", True
        ),
    )

    ownership_case(
        "unknown-ownership-key",
        lambda ownership, _manifest: ownership["records"][0]["owners"][
            "state"
        ].__setitem__("unexpected", False),
    )

    def forge_selected_owner(
        ownership: dict[str, Any], manifest: dict[str, Any]
    ) -> None:
        semantic = manifest["regions"][0]["topology_semantic_id"]
        record = next(
            item for item in ownership["records"] if item["semantic_id"] == semantic
        )
        role = record["owners"]["physical_volume"]
        if role["status"] != "candidate":
            raise AssertionError("selected fixture owner is not candidate")
        role["status"] = "unresolved"
        role["owner_id"] = None
        counts = ownership["counts"]["owners_by_role_and_status"][
            "physical_volume"
        ]
        counts["candidate"] -= 1
        counts["unresolved"] += 1

    ownership_case("selected-owner-forgery", forge_selected_owner)

    def forge_coverage_leaf(
        ownership: dict[str, Any], manifest: dict[str, Any]
    ) -> None:
        semantic = manifest["regions"][0]["topology_semantic_id"]
        record = next(
            item for item in ownership["records"] if item["semantic_id"] == semantic
        )
        forged = "11" * 32
        record["coverage_leaf_sha256s"].append(forged)
        record["coverage_leaf_sha256s"].sort()
        ownership["source_coverage"]["leaf_count"] += 1
        manifest["semantic_scope"]["coverage_leaf_sha256s"].append(forged)
        manifest["semantic_scope"]["coverage_leaf_sha256s"].sort()

    ownership_case("coverage-leaf-forgery", forge_coverage_leaf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
