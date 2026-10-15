#!/usr/bin/env python3
"""Source-bound CPU audit of the frozen074 full-box allocation admission.

This consumes the native-free generated-topology preflight JSON and rechecks
its exact input hashes against the frozen snapshot. It does not initialize
Metal, estimate solver accuracy, or admit a physical run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

EXPECTED_TOPOLOGY = {
    "prepared_nodes": 528444,
    "tetrahedra": 1477728,
    "cohesive_faces_total_if_both_laws_enabled": 260835,
    "exterior_faces_after_starch_split": 1054876,
    "contact_eligible_exterior_nodes": 527964,
}
GIB = 1024 ** 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_match(text: str, pattern: str, description: str) -> re.Match[str]:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    require(match is not None, f"frozen source no longer has expected {description}")
    return match


def audit(snapshot: Path, preflight_path: Path, prebuild_path: Path) -> dict:
    pre = json.loads(preflight_path.read_text())
    prebuild = json.loads(prebuild_path.read_text())
    require(pre.get("schema") == "numi.cardboard.shipping-allocation-preflight.v6",
            "preflight schema is not the reviewed v6 source-bound result")
    topo = pre["topology"]
    for key, value in EXPECTED_TOPOLOGY.items():
        require(int(topo.get(key, -1)) == value,
                f"generated shipping topology mismatch for {key}")

    bound = pre["provenance"]["hashed_non_system_translation_dependencies"]
    require(bound, "preflight contains no source dependency hashes")
    hash_mismatches = []
    for relative, expected in bound.items():
        path = snapshot / relative
        if not path.is_file() or sha256(path) != expected:
            hash_mismatches.append(relative)
    require(not hash_mismatches,
            "preflight source binding does not match frozen snapshot: " +
            ", ".join(hash_mismatches[:8]))

    validation = (snapshot / "matter/src/validation_impl.cpp").read_text()
    runtime = (snapshot / "matter/src/runtime.mm").read_text()
    graph = (snapshot / "matter/src/regional_graph_schwarz.hpp").read_text()
    memory = (snapshot / "matter/tools/cardboard_shipping_memory.hpp").read_text()
    probe = (snapshot / "matter/tools/cardboard_probe.mm").read_text()
    sparse = (snapshot / "matter/include/numi/matter/sparse_private_contact_admission.hpp").read_text()

    contact_guard = validation[validation.find("maximumContactGalerkinNodes"):
                               validation.find("if (solver.executionBudgets.z == 5u)")]
    require("maximumContactGalerkinNodes = 8192u" in contact_guard and
            "maximumContactGalerkinTetrahedra = 32768u" in contact_guard and
            "maximumContactGalerkinCohesiveFaces = 4096u" in contact_guard,
            "regional-contact preconditioner bounds changed")
    schwarz_guard = validation[validation.find("if (solver.executionBudgets.z == 5u)"):]
    require("maximumGraphSchwarzNodes = 4096u" in schwarz_guard and
            "maximumGraphSchwarzTetrahedra = 32768u" in schwarz_guard and
            "world_.dispatch.rigidProxyCount != 0u" in schwarz_guard and
            "world_.dispatch.contactPairCount != 0u" in schwarz_guard,
            "graph-Schwarz contact-free admission guard changed")
    require("maximumDeviceBytes = 96ull * 1024ull * 1024ull" in graph and
            "maximumPatchCount = 128u" in graph and
            "maximumPatchNodes = 256u" in graph and
            "anchorStride = 128u" in graph,
            "graph-Schwarz memory/layout bounds changed")
    require("kDensePrivateContactPairLimit = 65536u" in sparse and
            "kSparsePrivateContactScratchBudgetBytes =\n    36ull * 1024ull * 1024ull" in sparse,
            "sparse rigid private-contact limits changed")
    require("kShippingFullContactArmijoNodeLimit = 8192u" in memory and
            "deformableArmijoBytes <= 128ull * 1024ull * 1024ull" in memory and
            "faces <= 4096u" in memory and
            "maximumCohesiveArmijoBytes =\n            8ull * 1024ull * 1024ull" in memory,
            "full-contact/cohesive private arena caps changed")
    require("state.dispatch.femNodeCount > 8192u" in runtime and
            "state.dispatch.tetrahedronCount > 32768u" in runtime and
            "state.dispatch.cohesiveFaceCount > 4096u" in runtime and
            "state.dispatch.deformableContactCapacity > 131072u" in runtime,
            "runtime full-contact scope limits changed")

    # The default shipping path sets full-contact descent as required and runs
    # its pre-mesh screen before reading materials/building the exact mesh.
    require("input.fullContactArmijoRequired = arguments.shippingBox;" in probe,
            "shipping no longer requires full-contact residual descent")
    prebuild_start = probe.find("const auto prebuildBudget = shippingPrebuildMemoryBudget(")
    prebuild_gate = probe.find("if (shippingPhysicsRequested &&", prebuild_start)
    material_read = probe.find("const Digest linerDigest =", prebuild_start)
    require(prebuild_start >= 0 and prebuild_gate > prebuild_start and
            material_read > prebuild_gate,
            "pre-mesh shipping screen no longer precedes material/mesh compilation")
    require("options.requireFullContactResidualDescent = arguments.shippingBox;" in probe,
            "pre-mesh shipping budget no longer reserves residual descent")
    prebuild_options = memory[memory.find("struct ShippingPrebuildArenaOptions"):
                              memory.find("inline ShippingMemoryBudget estimateShippingPrebuildMemoryBudget")]
    prebuild_budget = probe[probe.find("numi::cardboard::ShippingMemoryBudget shippingPrebuildMemoryBudget("):
                            probe.find("struct ShippingResidualDescentAdmission")]
    require("sparsePrivateContact" not in prebuild_options and
            "sparsePrivateContact" not in prebuild_budget and
            "NM_FEM_FULL_CONTACT_SPARSE_PRIVATE_CAPACITY" not in prebuild_budget,
            "pre-mesh screen unexpectedly changed sparse-option propagation; re-audit first blocker")
    require(int(prebuild["prebuild_pair_upper_bound"]) ==
            int(prebuild["nominal_nodes"]) * int(prebuild["rigid_proxy_count"]) and
            int(prebuild["prebuild_pair_upper_bound"]) > 65536 and
            prebuild["private_rigid_admitted"] is False and
            prebuild["private_descent_admitted"] is False and
            prebuild["private_rigid_failure"] ==
                "dense private pair domain or sample arena is outside runtime admission",
            "source74 C++ prebuild probe does not reproduce the first dense-pair rejection")

    scenarios = pre["memory_model"]["modeled_scenarios"]
    default = scenarios["default_capacity_compact_flag_off_stride15"]
    reduced = scenarios["capacity_131072_compact_flag_on_stride15_best_case"]
    sparse = scenarios["sparse_K65536_one_free_BOX_six_DOF_on_stride15"]
    limit = 16 * GIB
    require(int(default["modeled_peak_domain_bytes"]) > limit,
            "default contact capacity no longer exceeds modeled 16 GiB envelope")
    require(int(reduced["modeled_peak_domain_bytes"]) < limit,
            "bounded-capacity/compact best-case model no longer fits 16 GiB")
    require(int(reduced["required_private_full_contact_bytes"]) > 128 * 1024 * 1024,
            "bounded best-case full-contact private arena unexpectedly fits current cap")
    require(int(reduced["required_private_cohesive_bytes"]) > 8 * 1024 * 1024,
            "both-law cohesive arena unexpectedly fits current cap")
    require(sparse["private_rigid_contact_admitted"] is True and
            sparse["private_arenas_admitted"] is False,
            "sparse bounded rigid control changed; expected it to solve only rigid scratch")

    n = int(topo["prepared_nodes"])
    anchor_stride = 128
    potential_node_anchor_count = (n + anchor_stride - 1) // anchor_stride
    if (n - 1) % anchor_stride != 0:
        potential_node_anchor_count += 1
    scenario_summary = {}
    for name, item in (("default_capacity_compact_flag_off", default),
                       ("131072_compact_surface_best_case", reduced),
                       ("sparse_K65536_one_free_BOX", sparse)):
        scenario_summary[name] = {
            "modeled_peak_domain_bytes": int(item["modeled_peak_domain_bytes"]),
            "modeled_peak_domain_gib": int(item["modeled_peak_domain_bytes"]) / GIB,
            "modeled_with_tangent_cache_bytes": int(item["modeled_combined_allocation_plus_tangent_cache_bytes"]),
            "private_arenas_admitted": bool(item["private_arenas_admitted"]),
            "rigid_private_arena_admitted": bool(item["private_rigid_contact_admitted"]),
            "private_arena_failure": item["private_arena_failure"],
        }

    contact_pairs = int(pre["capacities_and_private_arrays"]["rigid_pair_rows"])
    default_slots = int(pre["capacities_and_private_arrays"]["deformable_capacity_default_8_per_tet"])
    return {
        "schema": "numi.cardboard.shipping-arena-source-audit.v1",
        "evidence_class": "frozen-source CPU topology and admission audit; not a native allocation or solver run",
        "snapshot_root": str(snapshot),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256(preflight_path),
        "preflight_compiled_executable_sha256": pre["provenance"]["compiled_executable_sha256"],
        "prebuild_probe_path": str(prebuild_path),
        "prebuild_probe_sha256": sha256(prebuild_path),
        "prebuild_probe_source_sha256": sha256(prebuild_path.parent / "prebuild_probe.cpp"),
        "prebuild_probe_executable_sha256": sha256(prebuild_path.parent / "prebuild_probe"),
        "audit_tool_path": str(Path(__file__).resolve()),
        "audit_tool_sha256": sha256(Path(__file__).resolve()),
        "test_tool_sha256": sha256(Path(__file__).resolve().parent / "test_audit_shipping_admission.py"),
        "source_dependency_count_verified": len(bound),
        "source_dependency_tree_binding_sha256": hashlib.sha256(
            "\n".join(f"{k}:{bound[k]}" for k in sorted(bound)).encode()).hexdigest(),
        "topology": {key: int(topo[key]) for key in EXPECTED_TOPOLOGY} | {
            "rigid_pair_rows": contact_pairs,
            "default_8_slots_per_tet": default_slots,
            "default_slots_over_runtime_131072_limit": default_slots / 131072,
            "rigid_pairs_over_dense_private_65536_limit": contact_pairs / 65536,
        },
        "source_anchors": {
            "prebuild_gate": "matter/tools/cardboard_probe.mm:11756-11787",
            "prebuild_rigid_pair_budget": "matter/tools/cardboard_shipping_memory.hpp:679-735",
            "prebuild_deformable_cohesive_caps": "matter/tools/cardboard_shipping_memory.hpp:764-812",
            "regional_contact_preconditioner": "matter/src/validation_impl.cpp:580-637",
            "graph_schwarz_preconditioner": "matter/src/validation_impl.cpp:638-680 and matter/src/regional_graph_schwarz.hpp:79-100,162-174",
            "runtime_full_contact_nodes": "matter/src/runtime.mm:9238-9247",
        },
        "first_executable_blocker": {
            "phase": "shipping pre-mesh full-contact residual-descent screen",
            "default_request": "shippingBox requires full-contact descent; sparse private contact is opt-in",
            "sparse_prebuild_propagation": "not implemented: ShippingPrebuildArenaOptions and shippingPrebuildMemoryBudget do not carry the sparse request/capacity or bounded BOX proof",
            "first_admission_failure": "dense private rigid pair domain exceeds 65,536-pair runtime limit",
            "pair_count": contact_pairs,
            "pair_limit": 65536,
            "result": "prebuild_screen_rejects_before_material_read_mesh_build_or_Metal_initialization",
            "also_fails_default_memory_model": int(default["modeled_peak_domain_bytes"]) > limit,
        },
        "prebuild_exact_cpu_probe": {
            "nominal_nodes": int(prebuild["nominal_nodes"]),
            "nominal_tetrahedra": int(prebuild["nominal_tetrahedra"]),
            "rigid_proxies": int(prebuild["rigid_proxy_count"]),
            "pair_upper_bound": int(prebuild["prebuild_pair_upper_bound"]),
            "default_contact_slots": int(prebuild["default_deformable_capacity"]),
            "private_rigid_bytes": int(prebuild["required_rigid_bytes"]),
            "private_full_contact_bytes": int(prebuild["required_full_contact_bytes"]),
            "private_cohesive_bytes_if_law_requested": int(prebuild["required_cohesive_bytes"]),
            "modeled_peak_bytes": int(prebuild["modeled_peak_bytes"]),
            "private_rigid_failure": prebuild["private_rigid_failure"],
            "qualification": False,
        },
        "next_blockers_after_bounded_sparse_rigid_scratch": [
            {
                "stage": "deformable full-contact Armijo scope",
                "first_scope_reason": "fem_node_cap_8192",
                "actual": int(topo["prepared_nodes"]),
                "limit": 8192,
            },
            {
                "stage": "deformable full-contact Armijo scope",
                "actual_tetrahedra": int(topo["tetrahedra"]),
                "limit": 32768,
            },
            {
                "stage": "cohesive trial scope when both laws are enabled",
                "actual_possible_faces": int(topo["cohesive_faces_total_if_both_laws_enabled"]),
                "limit": 4096,
                "required_private_bytes": int(reduced["required_private_cohesive_bytes"]),
                "private_cap_bytes": 8 * 1024 * 1024,
            },
            {
                "stage": "contact-compatible selector 4 validation",
                "node_limit": 8192,
                "tetrahedron_limit": 32768,
                "cohesive_face_limit": 4096,
                "generalized_rigid_rows_supported": False,
            },
            {
                "stage": "selector 5 graph-Schwarz validation/layout",
                "node_limit": 4096,
                "tetrahedron_limit": 32768,
                "contact_or_proxy_allowed": False,
                "potential_nodes_for_runtime_all_node_mask": n,
                "minimum_anchor_patches_at_stride_128": potential_node_anchor_count,
                "runtime_maximum_patch_count": 128,
                "layout_device_budget_bytes": 96 * 1024 * 1024,
            },
        ],
        "modeled_scenarios": scenario_summary,
        "persistent_arenas_from_070_model": {
            "contact_surface_compact_off_bytes": int(default["persistent_contact_surface_bytes"]),
            "contact_surface_compact_on_bytes": int(reduced["persistent_contact_surface_bytes"]),
            "rigid_contact_dense_bytes": int(reduced["persistent_rigid_contact_bytes"]),
            "deformable_slots_default_8T_bytes": int(default["persistent_deformable_slots_bytes"]),
            "deformable_slots_131072_bytes": int(reduced["persistent_deformable_slots_bytes"]),
            "cohesive_260835_face_bytes": int(reduced["persistent_cohesive_bytes"]),
            "note": "these are model outputs calibrated from a prior resident counter, not current runtime measurement",
        },
        "mathematically_equivalent_extension_requirements": [
            "rigid contacts: exact conservative swept broadphase; canonical pair identity retained for active friction history; overflow aborts the whole transaction before commit; stream/tile all candidate pairs rather than silently dropping pairs",
            "self-contact: compact immutable exterior surface features; conservative BVH query on every Newton trial; stable feature-pair identities and bounded active-history map; any candidate/history overflow rejects, never truncates",
            "cohesive: evaluate per-face trial traction/history from accepted history and candidate geometry in bounded tiles; write irreversible state only after the selected trial is accepted; preserve activation reference geometry and rollback",
            "preconditioner: retain exact production A, residual, determinant/CCD, and acceptance; introduce contact/cohesive-aware sparse subdomain or multilevel preconditioning with checked symbolic fill/arena size, replacing contact-free 4096-node graph scope only after exact-A parity and native convergence tests",
        ],
        "limitations": [
            "16 GiB figures are modeled envelopes, not peak allocator measurements",
            "source032 anchor is explicitly not the full shipping world",
            "no Metal runtime was initialized and no physics step was run",
            "possible cohesive face count is conditional on both explicit laws being enabled",
            "no proposal raises capacities while preserving all-pair completeness; sparse paths need a complete broadphase proof and fail-closed overflow transaction",
        ],
        "qualification": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--prebuild", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = audit(args.snapshot.resolve(), args.preflight.resolve(), args.prebuild.resolve())
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f"shipping-arena audit rejected: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "rejected_as_expected",
        "first_blocker": result["first_executable_blocker"]["first_admission_failure"],
        "nodes": result["topology"]["prepared_nodes"],
        "tetrahedra": result["topology"]["tetrahedra"],
        "verified_source_dependencies": result["source_dependency_count_verified"],
        "report": str(args.output),
        "qualification": False,
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
