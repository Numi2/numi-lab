#!/usr/bin/env python3
"""Strict reader for the opt-in CGLS regional sidecar (schema v1).

This validates bytes and provenance joins. It does not certify a solver result,
and it never treats raw augmented buffers as current without matching tags.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


MAX_BYTES = 96 * 1024 * 1024
MAX_ARRAYS = 32
SCHEMA = "numi.matter.base_cgls_regional_state_capture.v1"
STATE_SCHEMA = "numi.matter.base_cgls_state_capture.v1"
EXPECTED_CAPTURE_POINT = (
    "selected_newton_after_fgmres_regional_rap_before_cgls_direction_"
    "construction_or_installation"
)
DTYPE_BYTES = {
    "float32": 4,
    "uint32": 4,
    "float32x4": 16,
    "uint32x4": 16,
    "record:NMIncidenceRangeGPU": 16,
}


class CaptureError(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise CaptureError(f"non-standard JSON numeric token: {value}")


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise CaptureError(f"manifest must be a regular adjacent file: {path.name}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_bad_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"manifest root is not an object: {path.name}")
    return value


def _local_payload(manifest_path: Path, entry: Any) -> tuple[Path, bytes]:
    if not isinstance(entry, dict):
        raise CaptureError("payload descriptor is not an object")
    name = entry.get("file")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise CaptureError("payload must name an adjacent basename")
    path = manifest_path.parent / name
    try:
        if path.is_symlink():
            raise CaptureError(f"payload must be an adjacent regular file: {name}")
        declared_bytes = _integer(entry.get("byte_count"), "payload byte_count")
        if declared_bytes > MAX_BYTES or path.stat().st_size != declared_bytes:
            raise CaptureError(f"payload extent exceeds cap or differs from manifest: {name}")
        payload = path.read_bytes()
    except OSError as exc:
        raise CaptureError(f"cannot read adjacent payload {name}: {exc}") from exc
    if len(payload) != declared_bytes:
        raise CaptureError(f"payload extent mismatch: {name}")
    if _sha(payload) != entry.get("sha256"):
        raise CaptureError(f"payload SHA-256 mismatch: {name}")
    if len(payload) > MAX_BYTES:
        raise CaptureError("payload exceeds the 96 MiB schema cap")
    return path, payload


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CaptureError(f"invalid integer {label}")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hex_u64(value: Any, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise CaptureError(f"invalid canonical u64 hex value: {label}")
    return int(value, 16)


def _validate_arrays(manifest: dict[str, Any], payload: bytes,
                     *, strict_dtype: bool) -> dict[str, memoryview]:
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise CaptureError("array table is empty or invalid")
    if len(arrays) > MAX_ARRAYS:
        raise CaptureError("array table exceeds the 32-array schema bound")
    if _integer(manifest.get("array_count"), "array_count", 1) != len(arrays):
        raise CaptureError("array_count does not match array table")
    names: set[str] = set()
    intervals: list[tuple[int, int, str]] = []
    views: dict[str, memoryview] = {}
    for entry in arrays:
        if not isinstance(entry, dict):
            raise CaptureError("array entry is not an object")
        name = entry.get("name")
        dtype = entry.get("dtype")
        if not isinstance(name, str) or not name or name in names:
            raise CaptureError("array names must be unique nonempty strings")
        names.add(name)
        if not isinstance(dtype, str):
            raise CaptureError(f"array {name} has no dtype")
        element_bytes = _integer(entry.get("element_bytes"), f"{name}.element_bytes", 1)
        if strict_dtype and DTYPE_BYTES.get(dtype) != element_bytes:
            raise CaptureError(f"array {name} dtype/element_bytes mismatch")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape:
            raise CaptureError(f"array {name} has invalid shape")
        count = 1
        for axis in shape:
            count *= _integer(axis, f"{name}.shape axis", 1)
            if count > MAX_BYTES:
                raise CaptureError(f"array {name} shape exceeds cap")
        byte_count = _integer(entry.get("byte_count"), f"{name}.byte_count")
        if count * element_bytes != byte_count:
            raise CaptureError(f"array {name} shape/byte_count mismatch")
        offset = _integer(entry.get("offset"), f"{name}.offset")
        if offset % 16 != 0:
            raise CaptureError(f"array {name} is not 16-byte aligned")
        end = offset + byte_count
        if end > len(payload):
            raise CaptureError(f"array {name} exceeds payload")
        if _sha(payload[offset:end]) != entry.get("sha256"):
            raise CaptureError(f"array {name} SHA-256 mismatch")
        intervals.append((offset, end, name))
        views[name] = memoryview(payload)[offset:end]
    intervals.sort()
    previous_end = 0
    for start, end, name in intervals:
        expected_start = (previous_end + 15) & ~15
        if start != expected_start:
            raise CaptureError(f"overlapping array range: {name}")
        if any(payload[previous_end:start]):
            raise CaptureError(f"nonzero allocation padding before array: {name}")
        previous_end = end
    if previous_end != len(payload):
        raise CaptureError("payload has unclaimed trailing bytes")
    return views


def _u32_words(data: memoryview) -> tuple[int, ...]:
    if len(data) % 4:
        raise CaptureError("uint32 array byte extent is not divisible by four")
    return tuple(int.from_bytes(data[i:i + 4], "little")
                 for i in range(0, len(data), 4))


def _f32_values(data: memoryview) -> tuple[float, ...]:
    if len(data) % 4:
        raise CaptureError("float32 array byte extent is not divisible by four")
    if not data:
        return ()
    return struct.unpack("<" + "f" * (len(data) // 4), data)


def _required_ints(manifest: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        _integer(manifest.get(key), key)


def read_capture(manifest_path: str | Path) -> dict[str, Any]:
    """Read and validate an adjacent `.regional.json` manifest and `.state` join."""
    path = Path(manifest_path)
    manifest = _strict_json(path)
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise CaptureError("unsupported or incomplete regional capture")
    if manifest.get("byte_order") != "little_endian":
        raise CaptureError("regional payload is not declared little endian")
    if manifest.get("capture_point") != EXPECTED_CAPTURE_POINT:
        raise CaptureError("unexpected capture point")
    _required_ints(manifest, (
        "root", "control_step", "physics_substep", "microtick",
        "microtick_count", "micro_generation", "newton_iteration",
        "environment_count", "object_count", "node_count",
        "tetrahedron_count", "material_state_stride", "unknown_count",
        "preconditioner_selector", "base_mode_count", "mode_stride",
        "augmented_mode_count", "cgls_iteration_budget",
    ))
    if not (1 <= manifest["environment_count"] <= 2 and
            1 <= manifest["node_count"] <= 4096 and
            1 <= manifest["tetrahedron_count"] <= 32768 and
            1 <= manifest["material_state_stride"] <= 64 and
            manifest["unknown_count"] == 2 * manifest["environment_count"] * manifest["node_count"] and
            1 <= manifest["base_mode_count"] <= 32 and
            manifest["base_mode_count"] <= manifest["mode_stride"] <= 32):
        raise CaptureError("regional capture dimensions exceed admitted bounds")
    env = manifest["environment_count"]
    if manifest["root"] != manifest["control_step"]:
        raise CaptureError("root/control_step mismatch")
    micro_generation = manifest["micro_generation"]
    newton_iteration = manifest["newton_iteration"]
    if micro_generation > 0xFFFFFFFF or newton_iteration > 0xFFFFFFFF:
        raise CaptureError("linearization generation components exceed uint32")
    expected_generation = f"{((micro_generation << 32) | newton_iteration):016x}"
    if manifest.get("linearization_generation") != expected_generation:
        raise CaptureError("linearization generation does not bind capture coordinates")
    if manifest["microtick_count"] < 1 or manifest["microtick"] >= manifest["microtick_count"]:
        raise CaptureError("invalid microtick identity")
    if manifest["cgls_iteration_budget"] not in (32, 64):
        raise CaptureError("unsupported captured CGLS iteration budget")
    dt = manifest.get("dt_seconds")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise CaptureError("invalid dt_seconds")

    _, payload = _local_payload(path, manifest.get("payload"))
    arrays = _validate_arrays(manifest, payload, strict_dtype=True)
    if manifest.get("array_count") != len(arrays):
        raise CaptureError("array count mismatch")
    state_binding = manifest.get("candidate_state_binding")
    if not isinstance(state_binding, dict) or state_binding.get("schema") != STATE_SCHEMA:
        raise CaptureError("missing candidate-state schema binding")
    state_manifest_name = state_binding.get("manifest_file")
    state_payload_name = state_binding.get("payload_file")
    if (not isinstance(state_manifest_name, str) or
            Path(state_manifest_name).name != state_manifest_name or
            not isinstance(state_payload_name, str) or
            Path(state_payload_name).name != state_payload_name):
        raise CaptureError("state binding must use adjacent basenames")
    state_path = path.parent / state_manifest_name
    state_manifest = _strict_json(state_path)
    if state_manifest.get("schema") != STATE_SCHEMA or state_manifest.get("complete") is not True:
        raise CaptureError("bound base state manifest is unsupported or incomplete")
    state_payload_entry = state_manifest.get("payload")
    if not isinstance(state_payload_entry, dict) or state_payload_entry.get("file") != state_payload_name:
        raise CaptureError("state payload filename binding mismatch")
    state_payload_path, state_payload = _local_payload(state_path, state_payload_entry)
    if state_payload_path.name != state_payload_name or _sha(state_payload) != state_binding.get("payload_sha256"):
        raise CaptureError("state payload digest does not match regional binding")
    _validate_arrays(state_manifest, state_payload, strict_dtype=False)
    state_entries = {entry["name"]: entry for entry in state_manifest["arrays"]}
    required_state_arrays = {
        "accepted_nodes": ("record:NMFEMNodeStateGPU", (env, manifest["node_count"]), 80),
        "candidate_nodes": ("record:NMFEMNodeStateGPU", (env, manifest["node_count"]), 80),
        "accepted_tetrahedra": ("record:NMTetrahedronGPU", (env, manifest["tetrahedron_count"]), 96),
        "candidate_tetrahedra": ("record:NMTetrahedronGPU", (env, manifest["tetrahedron_count"]), 96),
        "accepted_material_history": ("float32", (env, manifest["tetrahedron_count"], manifest["material_state_stride"]), 4),
        "candidate_material_state_buffer": ("float32", (env, manifest["tetrahedron_count"], manifest["material_state_stride"]), 4),
        "original_newton_correction": ("float32x4", (manifest["unknown_count"],), 16),
        "base_residual_rhs": ("float32x4", (manifest["unknown_count"],), 16),
        "algorithmic_tangent_cache": ("float32", (env, manifest["tetrahedron_count"], 9, 9), 4),
        "algorithmic_tangent_cache_valid": ("uint32", (env, manifest["tetrahedron_count"]), 4),
        "fgmres_state": ("record:NMFGMRESStateGPU", (env,), 32),
        "armijo_metrics": ("float32x4", (env,), 16),
    }
    for name, (dtype, shape, element_bytes) in required_state_arrays.items():
        entry = state_entries.get(name)
        if (entry is None or entry.get("dtype") != dtype or
                tuple(entry.get("shape", ())) != shape or
                entry.get("element_bytes") != element_bytes):
            raise CaptureError(f"bound state capture is missing a required typed array: {name}")
    for key in ("control_step", "physics_substep", "microtick", "microtick_count",
                "micro_generation",
                "newton_iteration", "environment_count", "node_count",
                "tetrahedron_count", "material_state_stride", "unknown_count"):
        if state_manifest.get(key) != manifest.get(key):
            raise CaptureError(f"state/regional identity mismatch: {key}")
    for key in ("source_physics_fingerprint_u64_hex", "world_fingerprint_u64_hex",
                "device_program_fingerprint_u64_hex"):
        if _hex_u64(state_manifest.get(key), key) != _hex_u64(manifest.get(key), key):
            raise CaptureError(f"state/regional source identity mismatch: {key}")
    if state_manifest.get("device") != manifest.get("device"):
        raise CaptureError("state/regional device identity mismatch")
    device = manifest.get("device")
    if (not isinstance(device, dict) or not isinstance(device.get("name"), str) or
            not device["name"] or _integer(device.get("registry_id"),
                                             "device.registry_id") > 0xFFFFFFFFFFFFFFFF):
        raise CaptureError("invalid captured device identity")
    state_dt = state_manifest.get("dt_seconds")
    if (isinstance(state_dt, bool) or not isinstance(state_dt, (int, float)) or
            not math.isfinite(state_dt) or state_dt != dt):
        raise CaptureError("state/regional time-step identity mismatch")
    projection = state_manifest.get("material_history_projection")
    if not isinstance(projection, dict) or projection.get("current_newton_status") != "unavailable_not_persisted":
        raise CaptureError("state history projection semantics changed or are missing")

    modes = manifest["base_mode_count"]
    stride = manifest["mode_stride"]
    nodes = manifest["node_count"]
    required_shapes = {
        "P": (modes, nodes),
        "effective_P": (env, stride, nodes),
        "AP": (env, stride, nodes),
        "base_RAP_lu": (env, modes, modes),
        "base_RAP_pivots": (env, modes),
        "base_RAP_valid": (env,),
        "base_coarse_rhs": (env, modes),
        "base_coarse_coefficients": (env, modes),
    }
    for name, shape in required_shapes.items():
        entry = next((a for a in manifest["arrays"] if a["name"] == name), None)
        if entry is None or tuple(entry["shape"]) != shape:
            raise CaptureError(f"required regional array shape mismatch: {name}")

    base_pivots = _u32_words(arrays["base_RAP_pivots"])
    base_factor_valid = _u32_words(arrays["base_RAP_valid"])
    if any(word not in (0, 1) for word in base_factor_valid):
        raise CaptureError("base RAP factor-valid word is not boolean")
    base_pivots_in_range = all(word < modes for word in base_pivots)

    selector = manifest["preconditioner_selector"]
    if selector not in (2, 3, 4, 5):
        raise CaptureError("regional preconditioner selector is unsupported")
    replay = manifest.get("regional_apply_replay")
    if (not isinstance(replay, dict) or
            replay.get("row_mapping") != (
                "candidate_nodes are environment-major; restAndFixed.w is the raw fixed-node tag; node_ranges is included for selectors 2-4; selector 5 carries graph_rows") or
            replay.get("fgmres_health") !=
                "read fgmres_state.diagnostics from the joined state payload" or
            replay.get("native_arbitrary_vector_action") != (
                "not captured; coarse RHS and coefficients only describe the last recorded coarse input")):
        raise CaptureError("regional apply replay limitations are missing")
    entries = {a["name"]: a for a in manifest["arrays"]}
    if selector in (2, 3, 4):
        fine_shapes = {
            "inverse_diagonal": (env, nodes),
            "node_ranges": (env, nodes),
        }
        for name, shape in fine_shapes.items():
            entry = entries.get(name)
            expected_dtype = "float32x4" if name == "inverse_diagonal" else "record:NMIncidenceRangeGPU"
            if (entry is None or tuple(entry.get("shape", ())) != shape or
                    entry.get("dtype") != expected_dtype or entry.get("element_bytes") != 16):
                raise CaptureError(f"fine regional apply array is missing or mistyped: {name}")
        if replay.get("fine_diagonal") != "inverse_diagonal is captured for selectors 2-4":
            raise CaptureError("fine diagonal replay semantics are missing")
        ranges = _u32_words(arrays["node_ranges"])
        row_ranges_well_formed = len(ranges) == env * nodes * 4
        if row_ranges_well_formed:
            for offset in range(0, len(ranges), 4):
                object_index = ranges[offset + 2]
                reserved = ranges[offset + 3]
                if ((object_index != 0xFFFFFFFF and object_index >= manifest["object_count"]) or
                        reserved != 0):
                    row_ranges_well_formed = False
                    break
        inverse_diagonal_finite = all(math.isfinite(x)
                                      for x in _f32_values(arrays["inverse_diagonal"]))
    elif "inverse_diagonal" in entries or "node_ranges" in entries:
        raise CaptureError("selector-5 graph replay must use graph rows, not nodal fine arrays")
    if selector in (3, 4):
        block = entries.get("node_block_inverse")
        if (block is None or tuple(block.get("shape", ())) != (env, nodes, 3) or
                block.get("dtype") != "float32x4" or block.get("element_bytes") != 16):
            raise CaptureError("selector-3/4 block apply is missing its exact inverse rows")
        if replay.get("fine_block") != (
                "node_block_inverse contains three float4 rows per node, including row0.w validity/fallback marker, for selectors 3-4"):
            raise CaptureError("nodal block inverse semantics are missing")
        block_words = _f32_values(arrays["node_block_inverse"])
        block_markers = tuple(block_words[i] for i in range(3, len(block_words), 16))
        block_markers_boolean = all(value in (0.0, 1.0) for value in block_markers)
    elif selector in (2, 5) and "node_block_inverse" in entries:
        raise CaptureError("unexpected nodal block inverse for selected preconditioner")
    else:
        block_markers_boolean = True
    if selector == 5:
        row_ranges_well_formed = None
        inverse_diagonal_finite = None
    selected_policy = manifest.get("primary_policy_selected_newton") is True
    selected_array = entries.get("primary_policy_selected_environment")
    if selected_policy:
        if (selected_array is None or tuple(selected_array.get("shape", ())) != (env,) or
                selected_array.get("dtype") != "uint32" or selected_array.get("element_bytes") != 4):
            raise CaptureError("selected CGLS policy is missing its per-environment selection words")
    elif selected_array is not None:
        raise CaptureError("unselected CGLS policy unexpectedly carries selection words")

    augmented = manifest.get("same_newton_restart_augmentation_enabled") is True
    generation_state: list[dict[str, Any]] = []
    if augmented:
        aug_modes = manifest["augmented_mode_count"]
        if aug_modes != modes + 1 or stride != aug_modes:
            raise CaptureError("augmented mode dimensions are inconsistent")
        aug_shapes = {
            "augmented_RAP_lu": (env, aug_modes, aug_modes),
            "augmented_RAP_pivots": (env, aug_modes),
            "augmented_RAP_valid": (env,),
            "augmented_coarse_rhs": (env, aug_modes),
            "augmented_coarse_coefficients": (env, aug_modes),
            "restart_generation": (env,),
            "restart_mode_valid": (env,),
        }
        entries = {a["name"]: a for a in manifest["arrays"]}
        for name, shape in aug_shapes.items():
            if name not in entries or tuple(entries[name]["shape"]) != shape:
                raise CaptureError(f"augmented array shape mismatch: {name}")
        generations = _u32_words(arrays["restart_generation"])
        mode_valid = _u32_words(arrays["restart_mode_valid"])
        factor_valid = _u32_words(arrays["augmented_RAP_valid"])
        augmented_pivots = _u32_words(arrays["augmented_RAP_pivots"])
        if any(word not in (0, 1) for word in factor_valid):
            raise CaptureError("augmented RAP factor-valid word is not boolean")
        if any(word >= aug_modes for word in augmented_pivots):
            raise CaptureError("augmented RAP pivot row is out of range")
        for e in range(env):
            tag = generations[4 * e:4 * e + 4]
            current = tag[:3] == (manifest["control_step"], manifest["physics_substep"], manifest["newton_iteration"])
            generation_state.append({
                "environment": e,
                "generation_tag": tag,
                "generation_matches_selected_newton": current,
                "mode_valid": mode_valid[e],
                "factor_valid": factor_valid[e],
                "augmented_data_admissible_for_same_newton": bool(
                    current and mode_valid[e] == 1 and factor_valid[e] == 1
                ),
            })
    else:
        if manifest["augmented_mode_count"] != 0:
            raise CaptureError("augmentation-disabled metadata has nonzero augmented mode count")
        if any(a["name"].startswith("augmented_") or a["name"].startswith("restart_")
               for a in manifest["arrays"]):
            raise CaptureError("augmentation-disabled capture contains augmented arrays")

    graph = manifest.get("graph_schwarz")
    if not isinstance(graph, dict):
        raise CaptureError("missing graph_schwarz metadata")
    graph_enabled = graph.get("enabled") is True
    graph_names = {
        "graph_patches", "graph_patch_nodes", "graph_rows",
        "graph_node_patch_offsets", "graph_node_patch_values",
        "graph_node_coverage", "graph_original_matrices", "graph_patch_lu",
        "graph_pivots", "graph_factor_valid",
    }
    present_graph_names = graph_names.intersection(arrays)
    if graph_enabled != (manifest["preconditioner_selector"] == 5):
        raise CaptureError("graph flag and preconditioner selector disagree")
    if graph_enabled and present_graph_names != graph_names:
        raise CaptureError("selector-5 graph arrays are incomplete")
    if not graph_enabled and present_graph_names:
        raise CaptureError("graph arrays present while selector-5 is disabled")
    graph_counts = {
        "patch_count": "graph_patches",
        "patch_node_count": "graph_patch_nodes",
        "dof_count": "graph_rows",
        "matrix_float_count": "graph_original_matrices",
        "pivot_count": "graph_pivots",
    }
    for key in graph_counts:
        _integer(graph.get(key), f"graph_schwarz.{key}")
    if graph_enabled:
        graph_shapes = {
            "graph_patches": (graph["patch_count"],),
            "graph_patch_nodes": (graph["patch_node_count"],),
            "graph_rows": (graph["dof_count"],),
            "graph_node_patch_offsets": (manifest["node_count"] + 1,),
            "graph_node_patch_values": (graph["patch_node_count"],),
            "graph_node_coverage": (manifest["node_count"],),
            "graph_original_matrices": (env, graph["matrix_float_count"]),
            "graph_patch_lu": (env, graph["matrix_float_count"]),
            "graph_pivots": (env, graph["pivot_count"]),
            "graph_factor_valid": (env, graph["patch_count"]),
        }
        graph_entries = {a["name"]: a for a in manifest["arrays"]}
        for name, shape in graph_shapes.items():
            if tuple(graph_entries[name]["shape"]) != shape:
                raise CaptureError(f"selector-5 graph array shape mismatch: {name}")
        for key in ("source_topology_sha256", "source_free_mask_sha256",
                    "potential_node_mask_sha256"):
            value = graph.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise CaptureError(f"selector-5 graph identity missing: {key}")
        patch_words = _u32_words(arrays["graph_patches"])
        patch_nodes = _u32_words(arrays["graph_patch_nodes"])
        row_words = _u32_words(arrays["graph_rows"])
        node_offsets = _u32_words(arrays["graph_node_patch_offsets"])
        node_patch_values = _u32_words(arrays["graph_node_patch_values"])
        coverage = _u32_words(arrays["graph_node_coverage"])
        patches: list[tuple[int, int, int, int]] = []
        matrix_cursor = 0
        pivot_cursor = 0
        dof_cursor = 0
        node_cursor = 0
        for patch_index in range(graph["patch_count"]):
            descriptor = patch_words[4 * patch_index:4 * patch_index + 4]
            if len(descriptor) != 4:
                raise CaptureError("truncated graph patch descriptor")
            node_offset, node_count, matrix_offset, pivot_offset = descriptor
            if (node_count == 0 or node_offset != node_cursor or
                    matrix_offset != matrix_cursor or pivot_offset != pivot_cursor):
                raise CaptureError("graph patch packed offsets are not canonical")
            end = node_offset + node_count
            if end > len(patch_nodes):
                raise CaptureError("graph patch exceeds node-membership array")
            members = patch_nodes[node_offset:end]
            if (any(node >= nodes for node in members) or
                    any(a >= b for a, b in zip(members, members[1:]))):
                raise CaptureError("graph patch node IDs are invalid or not strictly sorted")
            dofs = 3 * node_count
            matrix_cursor += dofs * dofs
            pivot_cursor += dofs
            dof_cursor += dofs
            node_cursor = end
            patches.append((node_offset, node_count, matrix_offset, pivot_offset))
            expected_rows = ((patch_index, local_node, axis, node)
                             for local_node, node in enumerate(members)
                             for axis in range(3))
            actual_rows = (tuple(row_words[i:i + 4])
                           for i in range((dof_cursor - dofs) * 4,
                                          dof_cursor * 4, 4))
            if list(actual_rows) != list(expected_rows):
                raise CaptureError("graph scalar row descriptors do not match patch membership")
        if (node_cursor != len(patch_nodes) or matrix_cursor != graph["matrix_float_count"] or
                pivot_cursor != graph["pivot_count"] or dof_cursor != graph["dof_count"]):
            raise CaptureError("graph packed counts differ from patch descriptors")
        if len(node_offsets) != nodes + 1 or node_offsets[0] != 0 or node_offsets[-1] != len(patch_nodes):
            raise CaptureError("graph node-to-patch CSR endpoints are invalid")
        for node in range(nodes):
            begin, end = node_offsets[node], node_offsets[node + 1]
            if begin > end or end > len(node_patch_values):
                raise CaptureError("graph node-to-patch CSR offsets are not monotone")
            values = node_patch_values[begin:end]
            if any(index >= len(patch_nodes) for index in values):
                raise CaptureError("graph node-to-patch CSR value is out of range")
            if any(patch_nodes[index] != node for index in values):
                raise CaptureError("graph node-to-patch CSR value maps to another node")
            if any(a >= b for a, b in zip(values, values[1:])):
                raise CaptureError("graph node-to-patch CSR values are not stable-sorted")
            if coverage[node] != len(values):
                raise CaptureError("graph node coverage differs from CSR membership")
        if node_offsets[-1] != len(node_patch_values):
            raise CaptureError("graph node-to-patch CSR does not cover all membership values")
        pivot_words = _u32_words(arrays["graph_pivots"])
        factor_words = _u32_words(arrays["graph_factor_valid"])
        if any(pivot >= 3 * patch[1]
               for env_index in range(env)
               for patch in patches
               for pivot in pivot_words[
                   env_index * graph["pivot_count"] + patch[3]:
                   env_index * graph["pivot_count"] + patch[3] + 3 * patch[1]]):
            raise CaptureError("graph pivot row exceeds its patch block")
        if any(word not in (0, 1) for word in factor_words):
            raise CaptureError("graph factor validity word is not boolean")
    node_inverse = next((a for a in manifest["arrays"]
                         if a["name"] == "node_block_inverse"), None)
    if node_inverse is not None and tuple(node_inverse["shape"]) != (env, nodes, 3):
        raise CaptureError("node block inverse dimensions mismatch")

    primary = manifest.get("primary_policy_selected_newton") is True
    selection = arrays.get("primary_policy_selected_environment")
    if primary != (selection is not None):
        raise CaptureError("primary selection array provenance mismatch")
    if selection is not None:
        selection_words = _u32_words(selection)
        if len(selection_words) != env or any(word not in (0, 1) for word in selection_words):
            raise CaptureError("invalid primary CGLS environment selection words")

    return {
        "manifest_path": str(path),
        "regional_payload_sha256": _sha(payload),
        "regional_payload_bytes": len(payload),
        "regional_array_count": len(arrays),
        "state_payload_sha256": _sha(state_payload),
        "state_array_count": len(state_manifest.get("arrays", [])),
        "identity_joined": True,
        "projected_material_history_currentness": "unavailable_not_persisted",
        "augmented_generation": generation_state,
        "graph_schwarz_captured": graph_enabled,
        "base_factor_replay_status": {
            "valid_by_environment": base_factor_valid,
            "all_pivots_in_range": base_pivots_in_range,
        },
        "fine_preconditioner_replay_status": {
            "node_ranges_well_formed": row_ranges_well_formed,
            "inverse_diagonal_finite": inverse_diagonal_finite,
            "block_validity_markers_boolean": block_markers_boolean,
        },
        "primary_policy_selection_captured": primary,
        "native_or_physical_qualification": False,
    }

