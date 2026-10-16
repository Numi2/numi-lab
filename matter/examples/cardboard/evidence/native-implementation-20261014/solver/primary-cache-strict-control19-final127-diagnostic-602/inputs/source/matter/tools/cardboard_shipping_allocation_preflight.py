#!/usr/bin/env python3
"""Compile and run the CPU-only shipping allocation preflight with input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def included_sources(compiler: str, root: Path, source: Path) -> list[Path]:
    command = [
        compiler, "-std=c++20", "-O2", "-MM", "-MT", "shipping_preflight",
        "-I", str(root / "include"),
        "-I", str(root / "matter/include"),
        "-I", str(root / "matter/tools"),
        str(source),
    ]
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    tokens = result.stdout.replace("\\\n", " ").split()[1:]
    paths = []
    for token in tokens:
        path = (root / token).resolve()
        if path.is_file() and path not in paths:
            paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(root).as_posix())


def verify_probe_contact_semantics(root: Path) -> dict[str, object]:
    probe_path = root / "matter/tools/cardboard_probe.mm"
    probe = re.sub(r"\s+", " ", probe_path.read_text(encoding="utf-8"))
    expected = (
        "object.deformableContact = glueMaterial != nullptr && !arguments.disableSelfContact;",
        "object.deformableSelfContact = object.deformableContact;",
        'if (option == "--without-self-contact") { arguments.disableSelfContact = true; continue; }',
    )
    missing = [snippet for snippet in expected if snippet not in probe]
    if missing:
        raise RuntimeError(
            "probe self-contact source semantics changed; refusing preflight admission: "
            + repr(missing)
        )
    return {
        "source_verified": True,
        "finite_glue_default_enables_deformable_contact": True,
        "finite_glue_default_enables_deformable_self_contact": True,
        "without_self_contact_is_explicit_opt_out": True,
        "verified_probe_expressions": list(expected),
    }


def source_binding_inputs(root: Path) -> list[Path]:
    # The C++ binary calls the same sparse admission header as runtime. These
    # host sources bind the runtime/probe behavior that the preflight reports.
    paths = [
        root / "matter/tools/cardboard_probe.mm",
        root / "matter/src/runtime.mm",
        root / "matter/src/compiler_impl.cpp",
        root / "matter/src/validation_impl.cpp",
        root / "matter/include/numi/matter/sparse_private_contact_admission.hpp",
        root / "matter/src/metal/contact.metalinc",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("required source binding files are missing: " + ", ".join(missing))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--output", type=Path,
                        help="write source-bound JSON here; default writes to stdout")
    parser.add_argument("--superseded-source54", type=Path,
                        help="bind and label the retained raw v2 source54 report as superseded")
    parser.add_argument("--superseded-source55", type=Path,
                        help="preserve the raw source55 estimate that used max(Metal, host) rather than an additive allocation inventory")
    parser.add_argument("--superseded-source56", type=Path,
                        help="bind the source56 exact-topology estimate whose host source/compile reserve was left at zero")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().with_suffix(".cpp")
    compiler = arguments.compiler
    include_files = included_sources(compiler, root, source)
    if source not in include_files:
        include_files.append(source)
    script = Path(__file__).resolve()
    if script not in include_files:
        include_files.append(script)
    for path in source_binding_inputs(root):
        if path not in include_files:
            include_files.append(path)
    include_files.sort(key=lambda item: item.relative_to(root).as_posix())

    compile_command = [
        compiler, "-std=c++20", "-O2",
        "-I", str(root / "include"),
        "-I", str(root / "matter/include"),
        "-I", str(root / "matter/tools"),
        str(source), "-o",
    ]
    with tempfile.TemporaryDirectory(prefix="numi-cardboard-preflight-") as directory:
        executable = Path(directory) / "cardboard_shipping_allocation_preflight"
        subprocess.run(compile_command + [str(executable)], cwd=root, check=True)
        raw = subprocess.run([str(executable)], cwd=root, check=True,
                             capture_output=True, text=True).stdout
        result = json.loads(raw)
        result["source_verified_runtime_semantics"] = verify_probe_contact_semantics(root)
        result["whole_process_preallocation_admission"] = {
            "complete": False,
            "reason": (
                "the source/compile host reserve is a coarse planning estimate, not "
                "actual vector capacities or allocator peaks; no native runtime or "
                "host/device peak was measured, so this cannot admit a full 16 GiB run"
            ),
            "metal_runtime_initialized": False,
            "actual_metal_peak_measured": False,
            "actual_host_peak_measured": False,
        }
        topology = result["topology"]
        prepared_nodes = int(topology["prepared_nodes"])
        tetrahedra = int(topology["tetrahedra"])
        source_mesh_bytes = 256 * prepared_nodes + 512 * tetrahedra
        compiled_world_bytes = 128 * prepared_nodes + 256 * tetrahedra
        host_reserve_assumptions = result.get("host_reserve_assumptions", {})
        expected_host_assumptions = {
            "source_mesh_bytes": source_mesh_bytes,
            "compiled_world_bytes": compiled_world_bytes,
            "generated_metal_bytes": 64 * 1024 * 1024,
            "upload_staging_bytes": compiled_world_bytes,
        }
        for field, expected in expected_host_assumptions.items():
            if host_reserve_assumptions.get(field) != expected:
                raise RuntimeError(
                    f"C++ preflight host reserve {field} does not match its "
                    "documented formula"
                )
        result["host_reserve_assumptions"] = {
            **host_reserve_assumptions,
            "basis": "same coarse estimateShippingPrebuildMemoryBudget formulas used by the probe prebuild path",
            "source_mesh_bytes": source_mesh_bytes,
            "source_mesh_formula": "256*nodes + 512*tetrahedra",
            "compiled_world_bytes": compiled_world_bytes,
            "compiled_world_formula": "128*nodes + 256*tetrahedra",
            "generated_metal_bytes": 64 * 1024 * 1024,
            "upload_staging_bytes": compiled_world_bytes,
            "upload_staging_formula": "equal to compiled-world reserve",
            "host_reserve_is_measurement": False,
            "capacity_accuracy": "planning assumption only; actual vector capacities, allocator overhead, generated source/compiler peaks, and host/device peak remain unmeasured",
        }
        result["allocation_accounting"] = {
            "modeled_combined_formula": (
                "estimated_metal_peak_bytes + host_mesh_and_compile_bytes + "
                "host_checkpoint_payload_bytes"
            ),
            "shared_reserves": (
                "upload staging and one snapshot readback are present in both "
                "side subtotals and counted once in the combined model"
            ),
            "observed_native_measurements": (
                "runtime resident, Metal currentAllocatedSize, and available "
                "host footprint remain observed counters compared by maximum "
                "because they may overlap on unified memory"
            ),
            "model_is_native_peak_measurement": False,
            "qualification_allowed": False,
        }
        if arguments.superseded_source54 is not None:
            old_path = arguments.superseded_source54.resolve()
            old = json.loads(old_path.read_text(encoding="utf-8"))
            result["superseded_preflight"] = {
                "path": str(old_path),
                "sha256": sha256(old_path),
                "schema": old.get("schema"),
                "status": "retained_raw_report_superseded_for_stale_admission_claims",
                "raw_report_modified": False,
                "corrected_claims": [
                    "sparse contact uses the runtime shared byte/index admission helper; 65536 is the dense pair cap, not a sparse P cap",
                    "finite glue enables deformable self-contact by default unless --without-self-contact is selected",
                    "scratch aliases reduce private trial bytes only and do not reduce persistent counts or scope caps",
                ],
            }
        compiler_version = subprocess.run(
            [compiler, "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        result["provenance"] = {
            "git_head_at_run": git_revision(root),
            "compiled_executable_sha256": sha256(executable),
            "compiler": compiler_version,
            "compile_command": compile_command + ["<temporary executable path>"],
            "hashed_non_system_translation_dependencies": {
                str(path.relative_to(root)): sha256(path) for path in include_files
            },
            "artifact_binding_scope": (
                "fresh CPU preflight executable, every non-system translation dependency, "
                "and hashed current probe/runtime/compiler/validation/contact source; "
                "does not build or execute Metal and is not a runtime allocation measurement"
            ),
        }
        if arguments.superseded_source54 is not None:
            result["provenance"]["superseded_source54_sha256"] = \
                result["superseded_preflight"]["sha256"]
        if arguments.superseded_source55 is not None:
            old_path = arguments.superseded_source55.resolve()
            old = json.loads(old_path.read_text(encoding="utf-8"))
            scenarios = old.get("memory_model", {}).get("modeled_scenarios", {})
            result["superseded_source55_estimate"] = {
                "path": str(old_path),
                "sha256": sha256(old_path),
                "schema": old.get("schema"),
                "status": "preserved_underestimated_allocation_model_not_qualification",
                "raw_report_modified": False,
                "previous_combination_rule": "maximum of modeled Metal and host subtotals",
                "previous_modeled_scenarios": {
                    name: {
                        "modeled_metal_envelope_bytes": value.get(
                            "modeled_metal_envelope_bytes"),
                        "modeled_peak_domain_bytes": value.get(
                            "modeled_peak_domain_bytes"),
                    }
                    for name, value in scenarios.items()
                },
                "correction": (
                    "distinct host mesh/compile and retained checkpoint allocations "
                    "must be added to the Metal allocation envelope; the old max "
                    "reported an underestimate for simultaneous modeled allocations"
                ),
                "source55_qualification_claim": False,
            }
            result["provenance"]["superseded_source55_sha256"] = \
                result["superseded_source55_estimate"]["sha256"]
        if arguments.superseded_source56 is not None:
            old_path = arguments.superseded_source56.resolve()
            old = json.loads(old_path.read_text(encoding="utf-8"))
            old_scenarios = old.get("memory_model", {}).get("modeled_scenarios", {})
            result["superseded_source56_host_reserve"] = {
                "path": str(old_path),
                "sha256": sha256(old_path),
                "schema": old.get("schema"),
                "status": "retained_raw_report_superseded_for_zero_host_reserve",
                "raw_report_modified": False,
                "prior_exact_topology_host_mesh_and_compile_bytes": {
                    name: value.get("host_mesh_and_compile_bytes")
                    for name, value in old_scenarios.items()
                    if name.startswith("sparse_K65536_no_free_rigid")
                },
                "correction": (
                    "apply the probe's documented coarse source-mesh, compiled-world, "
                    "generated-Metal, and upload-staging reserves to the exact topology; "
                    "the reserve remains an assumption rather than a measurement"
                ),
                "source56_qualification_claim": False,
            }
            result["provenance"]["superseded_source56_sha256"] = \
                result["superseded_source56_host_reserve"]["sha256"]

    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
