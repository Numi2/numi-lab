#!/usr/bin/env python3
"""Verify frozen-source free-patch parity and disjoint fixed-node supplements."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u32_words(data: bytes, offset: int, count: int, width: int = 1):
    byte_count = count * width * 4
    if offset + byte_count > len(data):
        raise ValueError("truncated uint32 array")
    fmt = "<" + str(count * width) + "I"
    values = struct.unpack_from(fmt, data, offset)
    if width == 1:
        return list(values), offset + byte_count
    return [tuple(values[i : i + width])
            for i in range(0, len(values), width)], offset + byte_count


def read_layout(path: Path):
    data = path.read_bytes()
    if len(data) < 44 or data[:8] != b"NMGRO001":
        raise ValueError(f"{path}: bad layout magic/header")
    version, nodes, tets, environments, patch_count, patch_node_count, dofs, matrix_floats, pivots = struct.unpack_from(
        "<9I", data, 8
    )
    if version != 1:
        raise ValueError(f"{path}: unsupported layout version")
    offset = 44

    def vector(width: int):
        nonlocal offset
        if offset + 8 > len(data):
            raise ValueError(f"{path}: missing vector count")
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        if width == 1:
            values, offset = u32_words(data, offset, count)
            return values
        values, offset = u32_words(data, offset, count, width)
        return values

    patches = vector(4)
    patch_nodes = vector(1)
    rows = vector(4)
    node_patch_offsets = vector(1)
    node_patch_values = vector(1)
    node_coverage = vector(1)
    if offset + 8 != len(data):
        raise ValueError(f"{path}: unexpected trailing/truncated bytes")
    device_bytes = struct.unpack_from("<Q", data, offset)[0]
    if (len(patches) != patch_count or len(patch_nodes) != patch_node_count or
            len(rows) != dofs or len(node_patch_offsets) != nodes + 1 or
            len(node_coverage) != nodes):
        raise ValueError(f"{path}: header counts disagree with arrays")
    return {
        "path": str(path), "sha256": sha256(path), "bytes": len(data),
        "node_count": nodes, "tetrahedron_count": tets,
        "environment_count": environments, "patches": patches,
        "patch_nodes": patch_nodes, "rows": rows,
        "node_patch_offsets": node_patch_offsets,
        "node_patch_values": node_patch_values,
        "node_coverage": node_coverage,
        "matrix_float_count": matrix_floats, "pivot_count": pivots,
        "device_bytes": device_bytes,
    }


def read_source_mask(path: Path):
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"NMGRA001":
        raise ValueError(f"{path}: bad source-topology input")
    version, nodes, tets, environments = struct.unpack_from("<4I", data, 8)
    if version != 1:
        raise ValueError(f"{path}: unsupported input version")
    mask_start = 24
    mask_end = mask_start + nodes
    mask = list(data[mask_start:mask_end])
    expected_bytes = mask_end + 16 * tets
    if expected_bytes != len(data) or any(value not in (0, 1) for value in mask):
        raise ValueError(f"{path}: bad mask or topology length")
    return {
        "sha256": sha256(path), "node_count": nodes,
        "tetrahedron_count": tets, "environment_count": environments,
        "free_mask": mask,
    }


def audit(source_mask_path: Path, baseline_path: Path,
          source_only_path: Path, combined_path: Path):
    source = read_source_mask(source_mask_path)
    baseline = read_layout(baseline_path)
    source_only = read_layout(source_only_path)
    combined = read_layout(combined_path)
    if baseline_path.read_bytes() != source_only_path.read_bytes():
        raise ValueError("new source-only layout is not byte-identical to frozen source helper")
    if (baseline["node_count"], baseline["tetrahedron_count"], baseline["environment_count"]) != (
            source["node_count"], source["tetrahedron_count"], source["environment_count"]):
        raise ValueError("baseline layout is not bound to the supplied source topology")
    count = len(baseline["patches"])
    node_entries = len(baseline["patch_nodes"])
    dof_entries = len(baseline["rows"])
    if combined["patches"][:count] != baseline["patches"]:
        raise ValueError("combined layout changed baseline patch descriptors")
    if combined["patch_nodes"][:node_entries] != baseline["patch_nodes"]:
        raise ValueError("combined layout changed baseline patch-node prefix")
    if combined["rows"][:dof_entries] != baseline["rows"]:
        raise ValueError("combined layout changed baseline row prefix")
    free_nodes = [node for node, free in enumerate(source["free_mask"]) if free]
    fixed_nodes = [node for node, free in enumerate(source["free_mask"]) if not free]
    for node in free_nodes:
        if combined["node_coverage"][node] != baseline["node_coverage"][node]:
            raise ValueError(f"source-free node {node} coverage changed")
        b0, b1 = baseline["node_patch_offsets"][node:node + 2]
        c0, c1 = combined["node_patch_offsets"][node:node + 2]
        if baseline["node_patch_values"][b0:b1] != combined["node_patch_values"][c0:c1]:
            raise ValueError(f"source-free node {node} CSR membership changed")
    tail_nodes = combined["patch_nodes"][node_entries:]
    if any(source["free_mask"][node] != 0 for node in tail_nodes):
        raise ValueError("supplementary patch contains an initially free node")
    uncovered_fixed = [node for node in fixed_nodes if combined["node_coverage"][node] == 0]
    if uncovered_fixed:
        raise ValueError(f"initially fixed nodes lack release coverage: {uncovered_fixed[:8]}")
    return {
        "schema": "numi.cardboard.graph_schwarz_stable_prefix_audit.v1",
        "source_mask": {key: value for key, value in source.items() if key != "free_mask"},
        "free_node_count": len(free_nodes),
        "initially_fixed_node_count": len(fixed_nodes),
        "baseline_frozen_layout": {
            key: baseline[key] for key in (
                "sha256", "patches", "patch_nodes", "rows",
                "node_patch_offsets", "node_patch_values", "node_coverage",
                "matrix_float_count", "pivot_count", "device_bytes"
            )
        },
        "combined_dynamic_layout": {
            key: combined[key] for key in (
                "sha256", "patches", "patch_nodes", "rows",
                "node_patch_offsets", "node_patch_values", "node_coverage",
                "matrix_float_count", "pivot_count", "device_bytes"
            )
        },
        "checks": {
            "source_only_new_helper_byte_identical_to_frozen_helper": True,
            "source_free_patch_descriptors_nodes_rows_exact_prefix": True,
            "source_free_node_coverage_unchanged": True,
            "source_free_node_patch_csr_unchanged": True,
            "supplementary_nodes_initially_fixed_only": True,
            "all_initially_fixed_nodes_covered": True,
        },
        "qualification": "CPU graph-layout identity check only; no native solver or physical qualification.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-only", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_mask, args.baseline, args.source_only, args.combined)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output), "checks": result["checks"]}))


if __name__ == "__main__":
    main()
