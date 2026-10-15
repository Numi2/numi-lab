#!/usr/bin/env python3
"""Prepare real-mesh inputs for production selector-5 Metal kernels.

The source-182 capture is selector 3 (RAP only), so this tool deliberately
does not call it a selector-5 capture. It rebuilds the selector-5 graph map
from the actual captured FEM connectivity and the same-generation FEM fixed
mask, using the C++ production layout helper from source 065.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

BASE = Path("/Users/home/cardboard-implementation-20261014")
DEFAULT_AUDIT = BASE / "analysis/selected-newton-regional-rap-audit-182"
DEFAULT_RUN = BASE / "solver/selected-newton-regional-rap-and-operator-capture-182"
DEFAULT_OUT = BASE / "analysis/selector5-native-kernel-check-001/canonical182-input"
EXPECTED_AUDIT_SHA = "7fc18d560019f74f00e5cf0b82326494316e0e0a145db0a96db493c0d3d646db"
EXPECTED_TOPOLOGY_SHA = "aee02f06cdf0abbd5e1fb8287a37324f0ede6b18977b41e681f267446a45436c"
EXPECTED_VECTOR_SHA = "a2acfe0b30d14fed1238372ce7970b759360da018931ea8ecc87650912b0cbcc"
EXPECTED_OPERATOR_SHA = "014efa29694ca8f2afc4a6838f005eb3ffd17f232f62db32a2cc3212d51e72f7"
EXPECTED_RAP_PAYLOAD_SHA = "5ea011af4b6c0835cf88cf9f6005514b79a14a850ba16c2c4a66374e118cb4d4"
EXPECTED_SOURCE_PHYSICS = "9405e23ae7e2bcd0"
EXPECTED_DEVICE_PROGRAM = "692ad59caf1de985"
EXPECTED_CAPTURE_DIGEST_ENV0 = "1c756428d7ef1f5cfa6a5e035faef068d39de9b38db6195b56009bbe3c581b41"
EXPECTED_KERNEL_SNAPSHOT = BASE / "snapshots/source-angular-and-graph-schwarz-integrated-065"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_simple_payload(manifest_path: Path, payload_path: Path):
    manifest = json.loads(manifest_path.read_text())
    raw = payload_path.read_bytes()
    require(manifest.get("capture_point") == "after_regional_rap_factor_barrier_before_fgmres_correction",
            "source182 sidecar is not the RAP-only capture point")
    require(manifest.get("source_physics_fingerprint") == EXPECTED_SOURCE_PHYSICS,
            "source182 physics fingerprint mismatch")
    require(manifest.get("device_program_fingerprint") == EXPECTED_DEVICE_PROGRAM,
            "source182 device-program fingerprint mismatch")
    require(manifest.get("root") == 3 and manifest.get("iteration") == 23,
            "source182 selected root/Newton iteration mismatch")
    require(manifest.get("environment_count") == 2 and manifest.get("node_count") == 882 and
            manifest.get("tetrahedron_count") == 1920 and manifest.get("mode_count") == 26,
            "source182 capture dimensions mismatch")
    require(manifest.get("payload", {}).get("sha256") == sha256_bytes(raw),
            "source182 RAP payload SHA-256 mismatch")
    arrays = {}
    expected_offset = 0
    for item in manifest.get("arrays", []):
        name = item["name"]
        require(name not in arrays, f"duplicate RAP array {name}")
        offset = item["offset"]
        length = item["byte_count"]
        require(offset == expected_offset and length >= 0 and offset + length <= len(raw),
                f"RAP array {name} has a gap, overlap, or invalid bounds")
        part = raw[offset:offset + length]
        require(item.get("sha256") == sha256_bytes(part), f"RAP array {name} hash mismatch")
        arrays[name] = (item, part)
        expected_offset += length
    require(expected_offset == len(raw), "RAP array inventory does not cover payload")
    require(manifest.get("preconditioner_selector") is None and "graph_schwarz" not in manifest,
            "input is unexpectedly represented as a native selector-5 capture")
    for name in ("P", "effective_P", "AP", "pivoted_RAP", "pivots", "factor_valid",
                 "candidate_nodes", "material_history", "scheduler_exponents",
                 "deformation_high", "deformation_low", "algorithmic_tangent_cache",
                 "algorithmic_tangent_cache_valid"):
        require(name in arrays, f"RAP capture lacks required {name}")
    digest = hashlib.sha256()
    for name in ("candidate_nodes", "material_history", "scheduler_exponents",
                 "deformation_high", "deformation_low", "algorithmic_tangent_cache",
                 "algorithmic_tangent_cache_valid"):
        item, part = arrays[name]
        width = item["byte_count"] // manifest["environment_count"]
        digest.update(part[:width])
    env_identity = next(row for row in manifest["candidate_linearization_sha256"]
                        if row["environment"] == 0)
    require(digest.hexdigest() == EXPECTED_CAPTURE_DIGEST_ENV0 == env_identity["sha256"],
            "source182 selected environment candidate/linearization digest mismatch")
    return manifest, raw, arrays


def array_from(arrays, name, dtype, shape, descriptor_dtype=None):
    desc, raw = arrays[name]
    accepted_descriptor_dtype = descriptor_dtype or dtype
    descriptor_shape = list(shape[:-1]) if accepted_descriptor_dtype == "<f4x4" else list(shape)
    require(desc["dtype"] == accepted_descriptor_dtype and desc["shape"] == descriptor_shape,
            f"{name} descriptor differs from expected shape/dtype")
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def build_python_layout(tetrahedra, free_mask, anchor_stride=128, maximum_nodes=256):
    """Independent readable port used to verify the C++ production layout."""
    node_count = len(free_mask)
    adjacency = [set() for _ in range(node_count)]
    free_nodes = [i for i, bit in enumerate(free_mask) if bit]
    for tet in tetrahedra:
        ids = [int(x) for x in tet]
        require(len(set(ids)) == 4 and all(0 <= x < node_count for x in ids),
                "topology contains repeated or out-of-range tet node ids")
        for i, node in enumerate(ids):
            if not free_mask[node]:
                continue
            for j, other in enumerate(ids):
                if i != j and free_mask[other]:
                    adjacency[node].add(other)
    anchors = free_nodes[::anchor_stride]
    if not anchors or anchors[-1] != free_nodes[-1]:
        anchors.append(free_nodes[-1])
    node_patches = []
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
        node_patches.append(selected)
        covered.update(selected)
    node_patches.extend([[node] for node in free_nodes if node not in covered])
    patches, patch_nodes = [], []
    matrix_offset = pivot_offset = 0
    for nodes in node_patches:
        patches.append([len(patch_nodes), len(nodes), matrix_offset, pivot_offset])
        patch_nodes.extend(nodes)
        dofs = 3 * len(nodes)
        matrix_offset += dofs * dofs
        pivot_offset += dofs
    rows = [[patch, local_node, axis, node]
            for patch, nodes in enumerate(node_patches)
            for local_node, node in enumerate(nodes)
            for axis in range(3)]
    coverage = [0] * node_count
    for node in patch_nodes:
        coverage[node] += 1
    offsets = [0]
    for count in coverage:
        offsets.append(offsets[-1] + count)
    values = [0] * len(patch_nodes)
    cursor = offsets[:-1].copy()
    for value_index, node in enumerate(patch_nodes):
        values[cursor[node]] = value_index
        cursor[node] += 1
    return {
        "patches": patches, "patch_nodes": patch_nodes, "rows": rows,
        "node_patch_offsets": offsets, "node_patch_values": values,
        "node_coverage": coverage,
    }


def read_layout_export(path: Path):
    raw = path.read_bytes()
    require(raw[:8] == b"NMGRO001", "bad C++ graph layout export magic")
    header = struct.unpack_from("<9I", raw, 8)
    version, node_count, tet_count, env_count, patch_count, patch_node_count, dof_count, matrix_count, pivot_count = header
    require(version == 1, "unsupported C++ graph layout version")
    cursor = 8 + 9 * 4
    specs = [
        ("patches", patch_count, 16, "<u4", (patch_count, 4)),
        ("patch_nodes", patch_node_count, 4, "<u4", (patch_node_count,)),
        ("rows", dof_count, 16, "<u4", (dof_count, 4)),
        ("node_patch_offsets", node_count + 1, 4, "<u4", (node_count + 1,)),
        ("node_patch_values", patch_node_count, 4, "<u4", (patch_node_count,)),
        ("node_coverage", node_count, 4, "<u4", (node_count,)),
    ]
    arrays = {}
    for name, count, width, dtype, shape in specs:
        array_count = struct.unpack_from("<Q", raw, cursor)[0]
        cursor += 8
        require(array_count == count, f"C++ export {name} count mismatch")
        byte_count = count * width
        require(cursor + byte_count <= len(raw), f"C++ export truncated at {name}")
        arrays[name] = np.frombuffer(raw[cursor:cursor + byte_count], dtype=dtype).reshape(shape).copy()
        cursor += byte_count
    device_bytes = struct.unpack_from("<Q", raw, cursor)[0]
    cursor += 8
    require(cursor == len(raw), "unexpected C++ layout export trailing bytes")
    return {
        "header": {
            "node_count": node_count, "tet_count": tet_count,
            "environment_count": env_count, "patch_count": patch_count,
            "patch_node_count": patch_node_count, "dof_count": dof_count,
            "matrix_float_count": matrix_count, "pivot_count": pivot_count,
            "device_bytes": device_bytes,
        },
        "arrays": arrays,
    }


def fnv1a_layout(arrays):
    checksum = 2166136261
    for name in ("patches", "rows", "patch_nodes", "node_patch_offsets", "node_patch_values"):
        for word in np.asarray(arrays[name], dtype="<u4").reshape(-1):
            checksum = ((checksum ^ int(word)) * 16777619) & 0xffffffff
    return checksum


def pack_bundle(path: Path, arrays, dims):
    """Write the fixed little-endian input ABI consumed by the .mm checker."""
    node_count, tet_count, mode_count, patch_count, patch_node_count, dof_count, matrix_count, pivot_count = dims
    header = b"NMSGKC1\0" + struct.pack(
        "<11I", 1, node_count, tet_count, mode_count, patch_count,
        patch_node_count, dof_count, matrix_count, pivot_count, 2, 0)
    names = [
        "rap_lu", "rap_pivots", "rap_valid", "basis", "regional_basis",
        "regional_ap", "patches", "patch_nodes", "node_patch_offsets",
        "node_patch_values", "node_coverage", "patch_matrices",
    ]
    raw = bytearray(header)
    descriptors = []
    for name in names:
        value = np.ascontiguousarray(arrays[name])
        offset = len(raw)
        blob = value.tobytes(order="C")
        raw.extend(blob)
        descriptors.append({
            "name": name, "dtype": value.dtype.str, "shape": list(value.shape),
            "offset": offset, "byte_count": len(blob), "sha256": sha256_bytes(blob),
        })
    path.write_bytes(raw)
    return {
        "schema": "numi.matter.selector5-kernel-check-input.v1",
        "source_capture_selector": 3,
        "native_selector5_capture_present": False,
        "kernel_input_qualification": "real source182 mesh and selector3 selected-generation operator arrays; selector5 graph rebuilt from source065 helper",
        "bundle_file": path.name,
        "bundle_byte_count": len(raw),
        "bundle_sha256": sha256_bytes(raw),
        "dimensions": {
            "node_count": node_count, "tetrahedron_count": tet_count,
            "mode_count": mode_count, "patch_count": patch_count,
            "patch_node_count": patch_node_count, "dof_count": dof_count,
            "matrix_float_count": matrix_count, "pivot_count": pivot_count,
            "source_environment_count": 2, "selected_environment": 0,
        },
        "arrays": descriptors,
    }


def prepare(args):
    audit_path = args.audit_dir / "operator-audit.json"
    require(sha256_file(audit_path) == EXPECTED_AUDIT_SHA, "operator audit SHA mismatch")
    audit = json.loads(audit_path.read_text())
    require(audit["capture_selection"] == {"root": 3, "newton_iteration": 23, "environment": 0},
            "operator audit selected root/iteration/environment differs")
    require(audit["binding"]["source_snapshot"].endswith("source-selected-newton-rap-and-validated-replay-integrated-060"),
            "operator audit source snapshot differs from source 060")
    for filename, expected in audit["matrix_artifacts"]["files_sha256"].items():
        require(sha256_file(args.audit_dir / filename) == expected,
                f"audit artifact hash mismatch: {filename}")
    require(audit["matrix_artifacts"]["files_sha256"]["operator-audit.topology.npz"] == EXPECTED_TOPOLOGY_SHA,
            "topology hash is not the canonical source182 topology")
    require(audit["matrix_artifacts"]["files_sha256"]["operator-audit.vectors.npz"] == EXPECTED_VECTOR_SHA,
            "vector hash is not the canonical source182 vector capture")
    require(audit["matrix_artifacts"]["files_sha256"]["operator-audit.operator.npz"] == EXPECTED_OPERATOR_SHA,
            "operator hash is not the canonical source182 assembled operator")

    capture_path = args.run_dir / "operator-capture"
    require(sha256_file(capture_path) == audit["binding"]["capture_sha256"],
            "full native capture SHA differs from source-bound operator audit")
    manifest, payload, captured = validate_simple_payload(
        args.run_dir / "operator-capture.regional.json",
        args.run_dir / "operator-capture.regional.bin")
    require(sha256_file(args.run_dir / "operator-capture.regional.bin") == EXPECTED_RAP_PAYLOAD_SHA,
            "RAP binary is not the selected source182 sidecar")

    node_count, tet_count = manifest["node_count"], manifest["tetrahedron_count"]
    env_count, modes = manifest["environment_count"], manifest["mode_count"]
    candidate_desc, candidate_bytes = captured["candidate_nodes"]
    require(candidate_desc["dtype"] == "struct:NMFEMNodeStateGPU" and
            candidate_desc["shape"] == [env_count, node_count] and
            candidate_desc["element_bytes"] == 80,
            "candidate node state descriptor differs from ABI")
    candidate = np.frombuffer(candidate_bytes, dtype="<f4").reshape(env_count, node_count, 20)
    fixed_lane = candidate[0, :, 11]
    require(np.isin(fixed_lane, np.array([0.0, 1.0], dtype="<f4")).all(),
            "candidate fixed mask is not exact binary state")
    free_mask = (fixed_lane == 0.0).astype(np.uint8)
    topology = np.load(args.audit_dir / "operator-audit.topology.npz", allow_pickle=False)
    tetrahedra = np.asarray(topology["nodes"], dtype="<u4")
    require(int(topology["node_count"]) == node_count and tetrahedra.shape == (tet_count, 4),
            "source topology dimensions differ from capture")
    require(np.flatnonzero(free_mask).size == 846 and np.count_nonzero(free_mask == 0) == 36,
            "source182 environment-0 free/fixed count changed")
    vectors = np.load(args.audit_dir / "operator-audit.vectors.npz", allow_pickle=False)
    free_node_ids = np.asarray(vectors["free_node_ids"], dtype=np.uint32)
    require(np.array_equal(free_node_ids, np.flatnonzero(free_mask).astype(np.uint32)),
            "operator rows do not map to the captured candidate free-node IDs")
    matrix_npz = np.load(args.audit_dir / "operator-audit.operator.npz", allow_pickle=False)
    A = csr_matrix((matrix_npz["data"], matrix_npz["indices"], matrix_npz["indptr"]),
                   shape=tuple(int(x) for x in matrix_npz["shape"]))
    require(A.shape == (3 * len(free_node_ids), 3 * len(free_node_ids)) and
            np.isfinite(A.data).all(), "source182 A dimensions or finite values are invalid")

    graph_input = args.output_dir / "graph-layout-input.bin"
    graph_input.parent.mkdir(parents=True, exist_ok=True)
    header = b"NMGRA001" + struct.pack("<4I", 1, node_count, tet_count, env_count)
    graph_input.write_bytes(header + free_mask.tobytes() + tetrahedra.tobytes(order="C"))
    generator = args.layout_exporter
    require(generator.is_file(), "C++ graph layout exporter is missing; compile graph_layout_export.cpp first")
    graph_output = args.output_dir / "graph-layout-output.bin"
    proc = subprocess.run([str(generator), str(graph_input), str(graph_output)],
                          check=False, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"C++ production graph builder failed: {proc.stderr.strip()}")
    exported = read_layout_export(graph_output)
    py_layout = build_python_layout(tetrahedra, free_mask.astype(bool))
    for name, expected in py_layout.items():
        actual = exported["arrays"][name]
        require(np.array_equal(actual, np.asarray(expected, dtype="<u4")),
                f"C++ production layout differs from independent actual-ID BFS for {name}")
    checksum = fnv1a_layout(exported["arrays"])
    dims = exported["header"]
    require(dims["environment_count"] == env_count,
            "C++ graph layout environment count differs from captured source")

    patches = exported["arrays"]["patches"]
    patch_nodes = exported["arrays"]["patch_nodes"]
    matrix_floats = dims["matrix_float_count"]
    patch_matrices = np.zeros(matrix_floats, dtype="<f4")
    free_row = np.full(node_count, -1, dtype=np.int64)
    free_row[free_node_ids] = np.arange(len(free_node_ids), dtype=np.int64)
    for patch_index, (node_offset, node_count_patch, matrix_offset, pivot_offset) in enumerate(patches):
        del pivot_offset
        nodes = patch_nodes[node_offset:node_offset + node_count_patch].astype(np.int64)
        n = 3 * len(nodes)
        require(np.all(free_row[nodes] >= 0), f"patch {patch_index} contains a non-free global node")
        row_ids = np.repeat(free_row[nodes] * 3, 3) + np.tile(np.arange(3), len(nodes))
        col_ids = np.repeat(free_row[nodes] * 3, 3) + np.tile(np.arange(3), len(nodes))
        # Dense principal submatrix assembled from the exact audited source A.
        block = A[row_ids[:, None], col_ids[None, :]].toarray().astype("<f4")
        require(block.shape == (n, n) and np.isfinite(block).all(), f"patch {patch_index} A block invalid")
        start, stop = int(matrix_offset), int(matrix_offset + n * n)
        patch_matrices[start:stop] = block.reshape(-1)

    P = array_from(captured, "effective_P", "<f4", (env_count, modes, node_count, 4),
                   descriptor_dtype="<f4x4")[0].copy()
    AP = array_from(captured, "AP", "<f4", (env_count, modes, node_count, 4),
                    descriptor_dtype="<f4x4")[0].copy()
    rap_lu = array_from(captured, "pivoted_RAP", "<f4", (env_count, modes, modes))[0].copy()
    rap_pivots = array_from(captured, "pivots", "<u4", (env_count, modes))[0].copy()
    rap_valid = array_from(captured, "factor_valid", "<u4", (env_count,))[0:1].copy()
    rhs_free = np.asarray(vectors["native_rhs_free_impulse_Ns"], dtype=np.float64)
    require(rhs_free.shape == (A.shape[0],) and np.isfinite(rhs_free).all(),
            "captured free RHS dimensions are invalid")
    basis = np.zeros((node_count, 4), dtype="<f4")
    basis[free_node_ids, :3] = rhs_free.reshape(-1, 3).astype("<f4")

    packed_arrays = {
        "rap_lu": rap_lu.astype("<f4", copy=False),
        "rap_pivots": rap_pivots.astype("<u4", copy=False),
        "rap_valid": rap_valid.astype("<u4", copy=False),
        "basis": basis,
        "regional_basis": P.astype("<f4", copy=False),
        "regional_ap": AP.astype("<f4", copy=False),
        "patches": patches.astype("<u4", copy=False),
        "patch_nodes": patch_nodes.astype("<u4", copy=False),
        "node_patch_offsets": exported["arrays"]["node_patch_offsets"].astype("<u4", copy=False),
        "node_patch_values": exported["arrays"]["node_patch_values"].astype("<u4", copy=False),
        "node_coverage": exported["arrays"]["node_coverage"].astype("<u4", copy=False),
        "patch_matrices": patch_matrices,
    }
    bundle_path = args.output_dir / "selector5-kernel-input.bin"
    bundle_manifest = pack_bundle(bundle_path, packed_arrays, (
        node_count, tet_count, modes, dims["patch_count"], dims["patch_node_count"],
        dims["dof_count"], dims["matrix_float_count"], dims["pivot_count"]))
    layout = {
        "schema": "numi.matter.regional_graph_layout_source_bound.v1",
        "construction": "production buildRegionalGraphSchwarzLayout from source065 plus independent Python BFS equality",
        "topology_source": "operator-audit.topology.npz actual 1920 tet global node IDs",
        "free_mask_source": "candidate_nodes env0 restAndFixed.w at selected RAP generation; 0 free, 1 fixed",
        "input_capture_is_selector5": False,
        "source_capture_selector": 3,
        "topology_npz_sha256": sha256_file(args.audit_dir / "operator-audit.topology.npz"),
        "candidate_sidecar_manifest_sha256": sha256_file(args.run_dir / "operator-capture.regional.json"),
        "candidate_sidecar_payload_sha256": sha256_file(args.run_dir / "operator-capture.regional.bin"),
        "free_mask_sha256": sha256_bytes(free_mask.tobytes()),
        "graph_builder_source_sha256": sha256_file(EXPECTED_KERNEL_SNAPSHOT / "matter/src/regional_graph_schwarz.hpp"),
        "independent_python_layout_equal": True,
        "fnv1a32_runtime_layout_checksum": checksum,
        "layout_exporter_stdout": proc.stdout.strip(),
        "layout_exporter_sha256": sha256_file(generator),
        "output_header": exported["header"],
        "patches": exported["arrays"]["patches"].tolist(),
        "patch_node_global_ids": exported["arrays"]["patch_nodes"].tolist(),
        "rows": exported["arrays"]["rows"].tolist(),
        "node_patch_offsets": exported["arrays"]["node_patch_offsets"].tolist(),
        "node_patch_values": exported["arrays"]["node_patch_values"].tolist(),
        "node_coverage": exported["arrays"]["node_coverage"].tolist(),
    }
    (args.output_dir / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    bundle_manifest.update({
        "operator_audit_sha256": sha256_file(audit_path),
        "operator_npz_sha256": sha256_file(args.audit_dir / "operator-audit.operator.npz"),
        "vectors_npz_sha256": sha256_file(args.audit_dir / "operator-audit.vectors.npz"),
        "topology_npz_sha256": sha256_file(args.audit_dir / "operator-audit.topology.npz"),
        "regional_capture_manifest_sha256": sha256_file(args.run_dir / "operator-capture.regional.json"),
        "regional_capture_payload_sha256": sha256_file(args.run_dir / "operator-capture.regional.bin"),
        "regional_capture_environment": 0,
        "regional_capture_source_selector": 3,
        "candidate_linearization_sha256": EXPECTED_CAPTURE_DIGEST_ENV0,
        "source_physics_fingerprint": EXPECTED_SOURCE_PHYSICS,
        "device_program_fingerprint": EXPECTED_DEVICE_PROGRAM,
        "layout_file": "layout.json",
        "layout_file_sha256": sha256_file(args.output_dir / "layout.json"),
        "arrays_are_cpu_prepared": True,
        "limits": [
            "This is a source-182 selector-3 operator capture; no native selector-5 capture was present.",
            "Graph maps use actual captured mesh global IDs and env0 fixed mask; source065 production helper rebuilt them.",
            "Patch matrices are exact principal blocks of the audited CSR A converted to FP32; native FEM graph matrix assembly is not dispatched here.",
            "This checks production factor/solve/apply GPU kernels only when run by the parent; it is not full native preconditioner or solver qualification.",
        ],
    })
    (args.output_dir / "bundle.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "prepared",
        "output_dir": str(args.output_dir),
        "bundle_sha256": bundle_manifest["bundle_sha256"],
        "layout_sha256": bundle_manifest["layout_file_sha256"],
        "layout_checksum_fnv1a32": checksum,
        "dimensions": bundle_manifest["dimensions"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layout-exporter", type=Path,
                        default=Path(__file__).with_name("graph_layout_export"))
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
