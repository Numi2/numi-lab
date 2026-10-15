#!/usr/bin/env python3
"""Strict reader/validator for selected-generation selector-5 RAP+graph captures.

This validates the byte-level capture contract and graph layout. It does not
claim the native preconditioner converged or that the material/physical model
is calibrated. It can emit checked arrays as little-endian NumPy .npz only
when explicitly requested by the offline analysis caller.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

SCHEMA = "numi.matter.regional_rap_capture.v1"
CAPTURE_POINT = "after_regional_rap_and_graph_factor_barrier_before_fgmres_correction"
DTYPE_BYTES = {
    "<f4": 4,
    "<u4": 4,
    "<f4x4": 16,
    "<u4x4": 16,
    "struct:NMFEMNodeStateGPU": 80,
    "struct:NMMatterStatusGPU": 48,
}
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_product(values: list[int]) -> int:
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("array dimensions must be positive integers")
        result *= value
        if result > (1 << 63) - 1:
            raise ValueError("array element count overflows signed 64-bit bounds")
    return result


def _floats(raw: bytes):
    if len(raw) % 4:
        raise ValueError("float array byte count is not divisible by four")
    return struct.iter_unpack("<f", raw)


def _u32s(raw: bytes):
    if len(raw) % 4:
        raise ValueError("u32 array byte count is not divisible by four")
    return struct.iter_unpack("<I", raw)


def rebuild_graph_layout(tetrahedra, free_mask, anchor_stride: int = 128,
                         maximum_nodes: int = 256) -> dict[str, list]:
    """Independent global-ID reconstruction of the selector-5 patch map.

    The input must be the exact ordered tet connectivity used by the capture.
    Count equality alone is deliberately insufficient for topology admission.
    """
    node_count = len(free_mask)
    adjacency = [set() for _ in range(node_count)]
    free_nodes = [i for i, value in enumerate(free_mask) if value]
    if not free_nodes:
        raise ValueError("topology-derived graph has no free FEM nodes")
    for tet in tetrahedra:
        ids = [int(value) for value in tet]
        if len(ids) != 4 or len(set(ids)) != 4 or any(value < 0 or value >= node_count for value in ids):
            raise ValueError("bound topology contains invalid tetrahedron global node IDs")
        for node in ids:
            if not free_mask[node]:
                continue
            for other in ids:
                if node != other and free_mask[other]:
                    adjacency[node].add(other)
    anchors = free_nodes[::anchor_stride]
    if not anchors or anchors[-1] != free_nodes[-1]:
        anchors.append(free_nodes[-1])
    patches_nodes: list[list[int]] = []
    covered = set()
    for anchor in anchors:
        selected = [anchor]
        visited = {anchor}
        frontier = [anchor]
        head = 0
        while head < len(frontier) and len(selected) < maximum_nodes:
            current = frontier[head]
            head += 1
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                selected.append(neighbor)
                frontier.append(neighbor)
                if len(selected) == maximum_nodes:
                    break
        selected.sort()
        patches_nodes.append(selected)
        covered.update(selected)
    patches_nodes.extend([[node] for node in free_nodes if node not in covered])

    patches: list[list[int]] = []
    patch_nodes: list[int] = []
    rows: list[list[int]] = []
    matrix_offset = pivot_offset = 0
    coverage = [0] * node_count
    for patch_index, nodes in enumerate(patches_nodes):
        patches.append([len(patch_nodes), len(nodes), matrix_offset, pivot_offset])
        patch_nodes.extend(nodes)
        dofs = 3 * len(nodes)
        rows.extend([[patch_index, local_node, axis, node]
                     for local_node, node in enumerate(nodes) for axis in range(3)])
        for node in nodes:
            coverage[node] += 1
        matrix_offset += dofs * dofs
        pivot_offset += dofs
    offsets = [0]
    for count in coverage:
        offsets.append(offsets[-1] + count)
    values = [0] * len(patch_nodes)
    cursor = offsets[:-1].copy()
    for value_index, node in enumerate(patch_nodes):
        values[cursor[node]] = value_index
        cursor[node] += 1
    return {"patches": patches, "patch_nodes": patch_nodes, "rows": rows,
            "node_patch_offsets": offsets, "node_patch_values": values,
            "node_coverage": coverage}


def _verify_bound_topology(manifest: dict[str, Any], topology_npz: Path,
                           fixed_tags: list[float], arrays: dict[str, dict[str, Any]],
                           checksum: int) -> dict[str, Any]:
    """Match an NPZ mesh to the native producer's exact canonical digest bytes."""
    graph = manifest.get("graph_schwarz", {})
    source_topology_hash = graph.get("source_topology_sha256")
    source_free_mask_hash = graph.get("source_free_mask_sha256")
    encoding = graph.get("source_topology_digest_encoding")
    expected_encoding = "little-endian u32 node_count, u32 tetrahedron_count, then row-major u32 tet connectivity; free mask is raw u8, 1=free 0=fixed"
    if encoding != expected_encoding:
        raise ValueError("source topology digest encoding is absent or unsupported")
    for name, value in (("source_topology_sha256", source_topology_hash),
                        ("source_free_mask_sha256", source_free_mask_hash)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} is absent or not a lowercase SHA-256")
    # The native builder admits only authored binary tags. Treating arbitrary
    # values around a threshold as free/fixed would certify a mask the native
    # selector-5 layout builder itself rejects.
    if any(tag not in (0.0, 1.0) for tag in fixed_tags):
        raise ValueError("selected candidate fixed tags are not exact binary source values")
    free_mask = [tag == 0.0 for tag in fixed_tags]
    free_bytes = bytes(1 if value else 0 for value in free_mask)
    free_mask_digest = sha256_bytes(free_bytes)
    if free_mask_digest != source_free_mask_hash:
        raise ValueError("candidate fixed mask differs from producer source_free_mask_sha256")
    import numpy as np
    with np.load(topology_npz, allow_pickle=False) as topology:
        if "nodes" not in topology.files or "node_count" not in topology.files:
            raise ValueError("topology NPZ must contain nodes and node_count")
        tetrahedra = np.asarray(topology["nodes"])
        raw_node_count = np.asarray(topology["node_count"])
        if raw_node_count.shape != () or raw_node_count.dtype.kind not in ("u", "i"):
            raise ValueError("topology node_count must be a scalar integer")
        node_count = int(raw_node_count.item())
    if tetrahedra.shape != (manifest["tetrahedron_count"], 4) or node_count != manifest["node_count"]:
        raise ValueError("bound topology dimensions differ from selected native capture")
    if tetrahedra.dtype.kind not in ("u", "i") or np.any(tetrahedra < 0) or np.any(tetrahedra >= node_count):
        raise ValueError("bound topology node IDs are invalid")
    if node_count > 0xffffffff or tetrahedra.shape[0] > 0xffffffff:
        raise ValueError("bound topology dimensions exceed the producer u32 digest encoding")
    connectivity = np.asarray(tetrahedra, dtype="<u4", order="C")
    canonical_topology = (struct.pack("<II", node_count, int(tetrahedra.shape[0])) +
                          connectivity.tobytes(order="C"))
    topology_digest = sha256_bytes(canonical_topology)
    if topology_digest != source_topology_hash:
        raise ValueError("NPZ topology differs from producer source_topology_sha256")
    expected = rebuild_graph_layout(tetrahedra, free_mask)
    names = {
        "patches": "graph_patches", "patch_nodes": "graph_patch_nodes",
        "rows": "graph_rows", "node_patch_offsets": "graph_node_patch_offsets",
        "node_patch_values": "graph_node_patch_values", "node_coverage": "graph_node_coverage",
    }
    for expected_name, capture_name in names.items():
        descriptor = arrays[capture_name]["descriptor"]
        actual = list(struct.unpack("<" + "I" * (len(arrays[capture_name]["bytes"]) // 4),
                                    arrays[capture_name]["bytes"]))
        expected_flat = [value for row in expected[expected_name] for value in row] \
            if expected[expected_name] and isinstance(expected[expected_name][0], list) \
            else expected[expected_name]
        if actual != expected_flat:
            raise ValueError(f"captured {capture_name} differs from exact topology-derived global-ID map")
        expected_dtype = "<u4x4" if expected_name in ("patches", "rows") else "<u4"
        if descriptor["dtype"] != expected_dtype:
            raise ValueError(f"captured {capture_name} is not u32")
    if fnv1a_layout(expected) != checksum:
        raise ValueError("topology-derived FNV layout checksum differs from capture")
    return {"status": "topology_and_global_ids_verified", "source_topology_sha256": topology_digest,
            "topology_npz_file_sha256": sha256_bytes(topology_npz.read_bytes()),
            "source_free_mask_sha256": free_mask_digest,
            "graph_layout_checksum_fnv1a32": checksum}


def fnv1a_layout(layout: dict[str, list]) -> int:
    checksum = 2166136261
    for name in ("patches", "rows", "patch_nodes", "node_patch_offsets", "node_patch_values"):
        values = layout[name]
        words = [value for row in values for value in row] if values and isinstance(values[0], list) else values
        for word in words:
            checksum = ((checksum ^ int(word)) * 16777619) & 0xffffffff
    return checksum


def validate_capture(manifest_path: Path, payload_path: Path,
                     selected_environment: int | None = None,
                     topology_npz: Path | None = None,
                     require_topology: bool = False) -> dict[str, Any]:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported regional capture schema")
    if manifest.get("byte_order") != "little_endian":
        raise ValueError("capture is not little-endian")
    if manifest.get("capture_point") != CAPTURE_POINT:
        raise ValueError("capture is not the selector-5 post-factor/pre-correction point")
    if manifest.get("preconditioner_selector") != 5:
        raise ValueError("capture does not identify selector 5")
    graph = manifest.get("graph_schwarz")
    if not isinstance(graph, dict) or graph.get("enabled") is not True:
        raise ValueError("selector-5 graph metadata is missing or disabled")
    env_count = manifest.get("environment_count")
    node_count = manifest.get("node_count")
    tet_count = manifest.get("tetrahedron_count")
    mode_count = manifest.get("mode_count")
    root = manifest.get("root")
    iteration = manifest.get("iteration")
    generation = manifest.get("linearization_generation")
    micro_generation = manifest.get("micro_generation")
    for name, value in (("environment_count", env_count), ("node_count", node_count),
                        ("tetrahedron_count", tet_count), ("mode_count", mode_count),
                        ("root", root), ("iteration", iteration),
                        ("linearization_generation", generation),
                        ("micro_generation", micro_generation)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid {name}")
    if env_count < 1 or env_count > 4 or node_count == 0 or tet_count == 0 or not (1 <= mode_count <= 32):
        raise ValueError("capture dimensions exceed declared source limits")
    if generation != ((micro_generation << 32) | iteration):
        raise ValueError("linearization generation encoding does not match root iteration")
    capture_environment = graph.get("capture_environment")
    if selected_environment is None:
        selected_environment = capture_environment
    if selected_environment != capture_environment or not isinstance(selected_environment, int) or not (0 <= selected_environment < env_count):
        raise ValueError("selected environment does not match graph factor capture environment")

    payload = payload_path.read_bytes()
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("capture payload exceeds the declared 32 MiB readback cap")
    payload_meta = manifest.get("payload")
    if not isinstance(payload_meta, dict) or payload_meta.get("byte_count") != len(payload) or payload_meta.get("sha256") != sha256_bytes(payload):
        raise ValueError("payload length or SHA-256 mismatch")
    rows = manifest.get("arrays")
    if not isinstance(rows, list) or not rows:
        raise ValueError("capture array inventory is absent")
    arrays: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("array descriptor is not an object")
        name, dtype = row.get("name"), row.get("dtype")
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError("duplicate or invalid array name")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported array dtype {dtype!r}")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"invalid shape for {name}")
        count = checked_product(shape)
        element_bytes = row.get("element_bytes")
        if element_bytes != DTYPE_BYTES[dtype] or row.get("byte_count") != count * element_bytes:
            raise ValueError(f"array {name} shape and byte count disagree")
        offset = row.get("offset")
        byte_count = row.get("byte_count")
        if not isinstance(offset, int) or offset < 0 or not isinstance(byte_count, int) or byte_count < 0 or offset + byte_count > len(payload):
            raise ValueError(f"array {name} has invalid payload bounds")
        data = payload[offset:offset + byte_count]
        if row.get("sha256") != sha256_bytes(data):
            raise ValueError(f"array {name} SHA-256 mismatch")
        arrays[name] = {"descriptor": row, "bytes": data}
        spans.append((offset, offset + byte_count, name))
    spans.sort()
    cursor = 0
    for start, end, name in spans:
        if start != cursor:
            raise ValueError(f"capture arrays have a gap or overlap before {name}")
        cursor = end
    if cursor != len(payload):
        raise ValueError("array inventory does not cover the exact payload")

    required = {
        "P", "effective_P", "AP", "pivoted_RAP", "pivots", "factor_valid",
        "candidate_nodes", "material_history", "scheduler_exponents",
        "deformation_high", "deformation_low", "production_status",
        "graph_patches", "graph_patch_nodes", "graph_rows",
        "graph_node_patch_offsets", "graph_node_patch_values",
        "graph_node_coverage", "graph_patch_matrices_lu", "graph_pivots",
        "graph_factor_valid",
    }
    if manifest.get("tangent_cache_used") is True:
        required |= {"algorithmic_tangent_cache", "algorithmic_tangent_cache_valid"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"capture is missing required arrays: {sorted(missing)}")

    def desc(name: str) -> dict:
        return arrays[name]["descriptor"]
    def data(name: str) -> bytes:
        return arrays[name]["bytes"]
    def expect(name: str, dtype: str, shape: list[int]) -> None:
        d = desc(name)
        if d["dtype"] != dtype or d["shape"] != shape:
            raise ValueError(f"{name} shape/dtype differs from expected {shape}/{dtype}")
    expect("P", "<f4x4", [mode_count, node_count])
    expect("effective_P", "<f4x4", [env_count, mode_count, node_count])
    expect("AP", "<f4x4", [env_count, mode_count, node_count])
    expect("pivoted_RAP", "<f4", [env_count, mode_count, mode_count])
    expect("pivots", "<u4", [env_count, mode_count])
    expect("factor_valid", "<u4", [env_count])
    expect("candidate_nodes", "struct:NMFEMNodeStateGPU", [env_count, node_count])
    expect("material_history", "<f4", [env_count, tet_count, manifest.get("material_state_stride")])
    expect("scheduler_exponents", "<u4", [env_count, tet_count])
    expect("deformation_high", "<f4x4", [env_count, tet_count, 3])
    expect("deformation_low", "<f4x4", [env_count, tet_count, 3])
    expect("production_status", "struct:NMMatterStatusGPU", [env_count])

    for name, row in arrays.items():
        if row["descriptor"]["dtype"] in ("<f4", "<f4x4"):
            if any(not math.isfinite(value[0]) for value in _floats(row["bytes"])):
                raise ValueError(f"array {name} contains a nonfinite float")
    if manifest.get("tangent_cache_used") is True:
        expect("algorithmic_tangent_cache", "<f4", [env_count, tet_count, 81])
        expect("algorithmic_tangent_cache_valid", "<u4", [env_count, tet_count])

    # Recompute the exact environment digest specified in the manifest.
    digest_rows = manifest.get("candidate_linearization_sha256")
    identity = next((row for row in digest_rows if row.get("environment") == selected_environment), None) if isinstance(digest_rows, list) else None
    if not identity or identity.get("identity_status") or not identity.get("sha256"):
        raise ValueError("selected environment candidate digest is missing")
    digest = hashlib.sha256()
    digest_names = ["candidate_nodes", "material_history", "scheduler_exponents", "deformation_high", "deformation_low"]
    if manifest.get("tangent_cache_used") is True:
        digest_names += ["algorithmic_tangent_cache", "algorithmic_tangent_cache_valid"]
    for name in digest_names:
        d = desc(name)
        if d["shape"][0] != env_count or d["byte_count"] % env_count:
            raise ValueError(f"{name} cannot be sliced evenly by environment")
        width = d["byte_count"] // env_count
        begin = selected_environment * width
        digest.update(data(name)[begin:begin + width])
    computed_digest = digest.hexdigest()
    if computed_digest != identity["sha256"]:
        raise ValueError("selected candidate/linearization digest mismatch")
    candidate_env = data("candidate_nodes")[selected_environment * node_count * 80:(selected_environment + 1) * node_count * 80]
    if sha256_bytes(candidate_env) != identity.get("candidate_nodes_sha256"):
        raise ValueError("selected candidate-node SHA-256 mismatch")

    patch_count = graph.get("patch_count")
    patch_node_count = graph.get("patch_node_count")
    dof_count = graph.get("dof_count")
    matrix_count = graph.get("matrix_float_count")
    pivot_count = graph.get("pivot_count")
    layout_checksum = graph.get("layout_checksum_fnv1a32")
    counts = (patch_count, patch_node_count, dof_count, matrix_count, pivot_count)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
        raise ValueError("selector-5 graph counts must be positive integers")
    if not isinstance(layout_checksum, int) or not (0 <= layout_checksum <= 0xffffffff):
        raise ValueError("selector-5 graph layout checksum is invalid")
    expect("graph_patches", "<u4x4", [patch_count])
    expect("graph_patch_nodes", "<u4", [patch_node_count])
    expect("graph_rows", "<u4x4", [dof_count])
    expect("graph_node_patch_offsets", "<u4", [node_count + 1])
    expect("graph_node_patch_values", "<u4", [patch_node_count])
    expect("graph_node_coverage", "<u4", [node_count])
    expect("graph_patch_matrices_lu", "<f4", [matrix_count])
    expect("graph_pivots", "<u4", [pivot_count])
    expect("graph_factor_valid", "<u4", [patch_count])
    u32 = lambda name: [x[0] for x in _u32s(data(name))]
    patches = [tuple(u32("graph_patches")[4 * i:4 * i + 4]) for i in range(patch_count)]
    patch_nodes = u32("graph_patch_nodes")
    rows_flat = u32("graph_rows")
    rows = [tuple(rows_flat[4 * i:4 * i + 4]) for i in range(dof_count)]
    offsets = u32("graph_node_patch_offsets")
    node_patch_values = u32("graph_node_patch_values")
    coverage = u32("graph_node_coverage")
    graph_pivots = u32("graph_pivots")
    factor_valid = u32("graph_factor_valid")
    if offsets[0] != 0 or offsets[-1] != patch_node_count or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("graph node-to-patch CSR offsets are invalid")
    if len(node_patch_values) != patch_node_count or len(coverage) != node_count:
        raise ValueError("graph node-to-patch maps are malformed")
    if any(value >= patch_node_count for value in node_patch_values):
        raise ValueError("graph node-to-patch value is out of range")
    if any(coverage[node] != offsets[node + 1] - offsets[node] for node in range(node_count)):
        raise ValueError("graph coverage disagrees with node-to-patch CSR")
    if any(not x for x in factor_valid):
        raise ValueError("one or more selector-5 graph patch factors are invalid")

    # Fixed mask in NMFEMNodeStateGPU: restAndFixed.w is float lane 11.
    state_floats = list(_floats(candidate_env))
    if len(state_floats) != node_count * 20 or any(
            not math.isfinite(value[0]) for value in state_floats):
        raise ValueError("selected candidate node state contains nonfinite/malformed floats")
    fixed_tags = [state_floats[node * 20 + 11][0] for node in range(node_count)]
    for patch_index, (node_offset, patch_size, matrix_offset, pivot_offset) in enumerate(patches):
        if not (1 <= patch_size <= 256) or node_offset + patch_size > patch_node_count:
            raise ValueError(f"graph patch {patch_index} node range is invalid")
        n = 3 * patch_size
        if matrix_offset + n * n > matrix_count or pivot_offset + n > pivot_count:
            raise ValueError(f"graph patch {patch_index} factor range is invalid")
        nodes = patch_nodes[node_offset:node_offset + patch_size]
        if nodes != sorted(set(nodes)) or any(node >= node_count for node in nodes):
            raise ValueError(f"graph patch {patch_index} node IDs are not sorted unique valid IDs")
        if any(fixed_tags[node] > 0.5 for node in nodes):
            raise ValueError(f"graph patch {patch_index} includes a fixed node")
        if matrix_offset != (patches[patch_index - 1][2] + (3 * patches[patch_index - 1][1]) ** 2 if patch_index else 0):
            raise ValueError("graph patch matrix segments are not canonical contiguous ranges")
        if pivot_offset != (patches[patch_index - 1][3] + 3 * patches[patch_index - 1][1] if patch_index else 0):
            raise ValueError("graph patch pivot segments are not canonical contiguous ranges")
        local_pivots = graph_pivots[pivot_offset:pivot_offset + n]
        if sorted(local_pivots) != list(range(n)):
            raise ValueError(f"graph patch {patch_index} pivot rows are not a permutation")
        if not all(math.isfinite(x) for x in local_pivots):
            raise ValueError("unreachable: integer pivots are nonfinite")
    if patches[-1][2] + (3 * patches[-1][1]) ** 2 != matrix_count or patches[-1][3] + 3 * patches[-1][1] != pivot_count:
        raise ValueError("graph factors do not consume exact declared factor arrays")

    membership = [0] * node_count
    for patch_index, (offset, size, _, _) in enumerate(patches):
        for local, node in enumerate(patch_nodes[offset:offset + size]):
            membership[node] += 1
            for axis in range(3):
                row = rows[3 * sum(p[1] for p in patches[:patch_index]) + 3 * local + axis]
                if row != (patch_index, local, axis, node):
                    raise ValueError(f"graph row descriptor mismatch at patch {patch_index}")
    for node, fixed in enumerate(fixed_tags):
        if membership[node] != coverage[node]:
            raise ValueError("node coverage does not match patch membership")
        mapped_patch_nodes = [patch_nodes[node_patch_values[item]]
                              for item in range(offsets[node], offsets[node + 1])]
        if mapped_patch_nodes != [node] * coverage[node]:
            raise ValueError("node-to-patch CSR values do not point back to their node")
        if fixed > 0.5 and coverage[node] != 0:
            raise ValueError("fixed node is present in graph patches")
        if fixed <= 0.5 and coverage[node] == 0:
            raise ValueError("free node is not covered by graph patches")
    if sum(3 * patch[1] for patch in patches) != dof_count:
        raise ValueError("graph row count differs from patch node count")

    checksum = 2166136261
    for word in [*u32("graph_patches"), *rows_flat, *patch_nodes, *offsets, *node_patch_values]:
        checksum = ((checksum ^ word) * 16777619) & 0xffffffff
    if checksum != layout_checksum:
        raise ValueError("graph layout FNV-1a checksum mismatch")

    # Verify selected RAP and graph factors are finite and selected coarse factor valid.
    coarse_valid = u32("factor_valid")
    if coarse_valid[selected_environment] != 1:
        raise ValueError("selected RAP coarse factor is not valid")
    # Check local LU float segments themselves; not their source-A reconstruction here.
    local_lu = [x[0] for x in _floats(data("graph_patch_matrices_lu"))]
    if len(local_lu) != matrix_count or not all(math.isfinite(x) for x in local_lu):
        raise ValueError("graph local LU factors contain nonfinite values")

    graph_topology = graph.get("source_topology_sha256")
    graph_free_mask = graph.get("source_free_mask_sha256")
    digest_encoding = graph.get("source_topology_digest_encoding")
    expected_digest_encoding = "little-endian u32 node_count, u32 tetrahedron_count, then row-major u32 tet connectivity; free mask is raw u8, 1=free 0=fixed"
    if not isinstance(graph_topology, str) or len(graph_topology) != 64 or \
            not isinstance(graph_free_mask, str) or len(graph_free_mask) != 64 or \
            any(c not in "0123456789abcdef" for c in graph_topology + graph_free_mask):
        raise ValueError("selector-5 capture lacks the producer's source topology/free-mask hashes")
    if digest_encoding != expected_digest_encoding:
        raise ValueError("selector-5 capture source topology digest encoding is absent or unsupported")
    topology_result = None
    if topology_npz is not None:
        topology_result = _verify_bound_topology(manifest, topology_npz,
            fixed_tags, arrays, layout_checksum)
    elif require_topology:
        raise ValueError("exact topology input is required for topology-qualified validation")
    else:
        if any(tag not in (0.0, 1.0) for tag in fixed_tags):
            raise ValueError("selected candidate fixed tags are not exact binary source values")
        free_bytes = bytes(1 if tag == 0.0 else 0 for tag in fixed_tags)
        if sha256_bytes(free_bytes) != graph_free_mask:
            raise ValueError("candidate fixed mask differs from producer source_free_mask_sha256")
        topology_result = {"status": "source_hash_present_topology_input_missing",
                           "source_topology_sha256": graph_topology,
                           "source_free_mask_sha256": graph_free_mask}
    return {
        "schema": SCHEMA,
        "capture_point": CAPTURE_POINT,
        "source_physics_fingerprint": manifest.get("source_physics_fingerprint"),
        "device_program_fingerprint": manifest.get("device_program_fingerprint"),
        "root": root,
        "iteration": iteration,
        "environment": selected_environment,
        "linearization_generation": generation,
        "candidate_linearization_sha256": computed_digest,
        "payload_sha256": sha256_bytes(payload),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "patch_count": patch_count,
        "patch_node_count": patch_node_count,
        "dof_count": dof_count,
        "matrix_float_count": matrix_count,
        "pivot_count": pivot_count,
        "layout_checksum_fnv1a32": layout_checksum,
        "all_graph_factors_valid": True,
        "all_free_nodes_covered": True,
        "topology_validation": topology_result,
        "limitations": ["byte/layout integrity only", "not a native FGMRES convergence result", "not material calibration or physical qualification"],
    }


def load_validated_arrays(manifest_path: Path, payload_path: Path):
    """Return validated manifest plus NumPy views for numeric arrays.

    Struct arrays remain byte arrays because decoding their ABI is deliberately
    outside this reader's numeric action scope. Each value is copied into an
    owned ndarray; there are no views into a mutable mapping.
    """
    import numpy as np
    summary = validate_capture(manifest_path, payload_path)
    manifest = json.loads(manifest_path.read_text())
    payload = payload_path.read_bytes()
    decoded = {}
    for row in manifest["arrays"]:
        name, dtype = row["name"], row["dtype"]
        raw = payload[row["offset"]:row["offset"] + row["byte_count"]]
        shape = tuple(row["shape"])
        if dtype == "<f4":
            decoded[name] = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
        elif dtype == "<u4":
            decoded[name] = np.frombuffer(raw, dtype="<u4").copy().reshape(shape)
        elif dtype == "<f4x4":
            decoded[name] = np.frombuffer(raw, dtype="<f4").copy().reshape(shape + (4,))
        elif dtype == "<u4x4":
            decoded[name] = np.frombuffer(raw, dtype="<u4").copy().reshape(shape + (4,))
        else:
            decoded[name] = np.frombuffer(raw, dtype="u1").copy().reshape(shape + (row["element_bytes"],))
    return manifest, decoded, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--environment", type=int)
    parser.add_argument("--topology-npz", type=Path,
                        help="same-run source-bound NPZ with nodes and node_count; enables exact global-ID map validation")
    parser.add_argument("--require-topology", action="store_true",
                        help="fail unless exact topology is supplied and bound by the capture manifest")
    parser.add_argument("--output", type=Path, help="optional path for validated summary JSON")
    args = parser.parse_args()
    result = validate_capture(args.manifest, args.payload, args.environment,
                              args.topology_npz, args.require_topology)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"selector5 capture rejected: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
