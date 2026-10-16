"""Source-derived CPU reference for selector-5 graph Schwarz plus base RAP.

This models only the captured base mode view. It never claims the base view was
what FGMRES consumed at an arbitrary callback, and it rejects requests for an
augmented/selected-map claim because the capture schema lacks the consuming
restart-cycle identity. Arithmetic is binary64; it is an algebraic reference,
not a bitwise Metal replay or a native qualification.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Sequence

import regional_rap_reference as rap
import read_regional_capture as reader

Matrix = rap.Matrix
Vector = rap.Vector

class GraphReferenceError(ValueError):
    pass

@dataclass(frozen=True)
class GraphPatch:
    nodes: tuple[int, ...]
    original: Matrix
    factor: rap.Factor

@dataclass(frozen=True)
class GraphCapture:
    node_count: int
    free_mask: tuple[bool, ...]
    patches: tuple[GraphPatch, ...]
    patch_nodes: tuple[int, ...]
    node_patch_offsets: tuple[int, ...]
    node_patch_values: tuple[int, ...]
    node_coverage: tuple[int, ...]
    P: Matrix                 # 3N x baseModeCount, effective P
    AP: Matrix                # 3N x baseModeCount, captured A*effective_P
    base_factor: rap.Factor   # base RAP only
    mode_stride: int
    augmentation_enabled: bool
    completed_tail_latch: bool = False
    identity: tuple[Any, ...] = ()
    view_label: str = "base_view_reference_only"

    @property
    def dof_count(self) -> int:
        return 3 * self.node_count

    @property
    def mode_count(self) -> int:
        return len(self.P[0]) if self.P else 0


def _finite(xs: Sequence[float]) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise GraphReferenceError("dot shape mismatch")
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def _matvec(A: Matrix, x: Sequence[float]) -> Vector:
    if not A or any(len(row) != len(x) for row in A):
        raise GraphReferenceError("matvec shape mismatch")
    return [_dot(row, x) for row in A]


def _transpose(A: Matrix) -> Matrix:
    if not A:
        return []
    return [[float(A[i][j]) for i in range(len(A))]
            for j in range(len(A[0]))]


def _matrix_shape(A: Matrix, rows: int, cols: int, label: str) -> None:
    if len(A) != rows or any(len(row) != cols for row in A):
        raise GraphReferenceError(f"{label} shape mismatch")
    if not _finite([v for row in A for v in row]):
        raise GraphReferenceError(f"{label} contains nonfinite values")


def _local_scalar(node_local: int, axis: int) -> int:
    return 3 * node_local + axis


def _membership_index_lists(capture: GraphCapture) -> list[list[int]]:
    N = capture.node_count
    members = [[] for _ in range(N)]
    for membership_index, node in enumerate(capture.patch_nodes):
        if not 0 <= node < N:
            raise GraphReferenceError("patch node outside global node range")
        members[node].append(membership_index)
    return members


def validate_capture(c: GraphCapture) -> None:
    if not (1 <= c.node_count <= 4096):
        raise GraphReferenceError("node count outside captured bound")
    if len(c.free_mask) != c.node_count or any(type(x) is not bool for x in c.free_mask):
        raise GraphReferenceError("free mask must be exact bool per node")
    if len(c.node_coverage) != c.node_count:
        raise GraphReferenceError("node coverage extent mismatch")
    if not (1 <= c.mode_count <= 32 and c.mode_count <= c.mode_stride <= 32):
        raise GraphReferenceError("base mode count/stride outside source bound")
    _matrix_shape(c.P, c.dof_count, c.mode_count, "effective P")
    _matrix_shape(c.AP, c.dof_count, c.mode_count, "AP")
    if len(c.base_factor.pivots) != c.mode_count or len(c.base_factor.lu) != c.mode_count:
        raise GraphReferenceError("base RAP factor extent differs from mode count")
    if any(len(row) != c.mode_count for row in c.base_factor.lu):
        raise GraphReferenceError("base RAP LU is not square")
    if c.base_factor.valid and sorted(c.base_factor.pivots) != list(range(c.mode_count)):
        raise GraphReferenceError("valid base RAP pivot words are not a permutation")
    # Projection is a source invariant: constrained xyz rows in effective P are zero.
    for node, free in enumerate(c.free_mask):
        if not free:
            for axis in range(3):
                if any(c.P[3 * node + axis][m] != 0.0 for m in range(c.mode_count)):
                    raise GraphReferenceError("effective P is nonzero on fixed coordinates")
    if len(c.patches) == 0:
        raise GraphReferenceError("selector-5 graph requires at least one patch")
    packed_nodes: list[int] = []
    matrix_cursor = 0
    pivot_cursor = 0
    for patch in c.patches:
        if not patch.nodes or tuple(sorted(set(patch.nodes))) != patch.nodes:
            raise GraphReferenceError("patch members must be nonempty, unique, sorted")
        if any(node < 0 or node >= c.node_count for node in patch.nodes):
            raise GraphReferenceError("patch member outside node range")
        n = 3 * len(patch.nodes)
        _matrix_shape(patch.original, n, n, "graph original patch matrix")
        if len(patch.factor.lu) != n or len(patch.factor.pivots) != n or any(len(r) != n for r in patch.factor.lu):
            raise GraphReferenceError("graph patch factor extent mismatch")
        if not _finite([x for row in patch.factor.lu for x in row]):
            raise GraphReferenceError("graph patch factor contains nonfinite values")
        if patch.factor.valid and sorted(patch.factor.pivots) != list(range(n)):
            raise GraphReferenceError("valid graph patch pivots are not a permutation")
        packed_nodes.extend(patch.nodes)
        matrix_cursor += n * n
        pivot_cursor += n
    if tuple(packed_nodes) != c.patch_nodes:
        raise GraphReferenceError("packed patch-node array differs from descriptors")
    expected_lists = _membership_index_lists(c)
    if len(c.node_patch_offsets) != c.node_count + 1 or c.node_patch_offsets[0] != 0:
        raise GraphReferenceError("node-to-patch CSR offset shape or origin invalid")
    if c.node_patch_offsets[-1] != len(c.node_patch_values):
        raise GraphReferenceError("node-to-patch CSR endpoint mismatch")
    for node in range(c.node_count):
        b, e = c.node_patch_offsets[node], c.node_patch_offsets[node + 1]
        if not (0 <= b <= e <= len(c.node_patch_values)):
            raise GraphReferenceError("node-to-patch CSR is not monotone/in-range")
        values = c.node_patch_values[b:e]
        if tuple(values) != tuple(expected_lists[node]):
            raise GraphReferenceError("node-to-patch CSR differs from exact patch memberships")
        if len(values) != c.node_coverage[node]:
            raise GraphReferenceError("node coverage differs from CSR membership count")
        if c.free_mask[node] and c.node_coverage[node] == 0:
            raise GraphReferenceError("free graph node has no patch coverage")
    if len(c.node_coverage) != c.node_count:
        raise GraphReferenceError("node coverage extent mismatch")
    # The source's graph kernels key their zero-output branch on
    # fgmresStates[env].diagnostics.z > 0.5, not an external selection flag.


def require_base_view(c: GraphCapture, *, request_actual_consumed_view: bool = False) -> str:
    """Return explicit base-view label; fail closed for unsupported augmented claims."""
    if request_actual_consumed_view:
        if c.augmentation_enabled:
            raise GraphReferenceError("actual selected view unsupported: augmented consuming restart generation is absent")
        raise GraphReferenceError("capture does not prove this base view was consumed by FGMRES at the requested callback")
    return "captured_base_view_reference_only_not_actual_callback_selection"


def _patch_rhs(c: GraphCapture, patch: GraphPatch, x: Sequence[float]) -> Vector:
    rhs = []
    for node in patch.nodes:
        for axis in range(3):
            rhs.append(float(x[3 * node + axis]) if c.free_mask[node] else 0.0)
    return rhs


def _solve_patch(c: GraphCapture, patch: GraphPatch, rhs: Sequence[float]) -> Vector:
    if not patch.factor.valid:
        raise GraphReferenceError("invalid graph patch factor: source production marks solver failure")
    x0 = rap.solve_rap(patch.factor, rhs)
    if len(x0) != len(rhs) or not _finite(x0):
        raise GraphReferenceError("nonfinite first graph-patch solve")
    residual = [float(rhs[i]) - _dot(patch.original[i], x0)
                for i in range(len(rhs))]
    correction = rap.solve_rap(patch.factor, residual)
    if not _finite(correction):
        raise GraphReferenceError("nonfinite graph-patch refinement solve")
    return [x0[i] + correction[i] for i in range(len(rhs))]


def _solve_patch_transpose(c: GraphCapture, patch: GraphPatch, rhs: Sequence[float]) -> Vector:
    if not patch.factor.valid:
        raise GraphReferenceError("invalid graph patch factor: transpose is not admissible")
    # D=F+F(I-AF)=2F-FAF in exact arithmetic, so D^T=2F^T-F^T A^T F^T.
    z = rap.solve_rap_transpose(patch.factor, rhs)
    atz = _matvec(_transpose(patch.original), z)
    w = rap.solve_rap_transpose(patch.factor, atz)
    result = [2.0 * z[i] - w[i] for i in range(len(z))]
    if not _finite(result):
        raise GraphReferenceError("nonfinite graph-patch transpose refinement")
    return result


def apply_H(c: GraphCapture, x: Sequence[float]) -> Vector:
    """Apply H=G D E with one FP64 reference refinement per graph patch."""
    validate_capture(c)
    if len(x) != c.dof_count or not _finite(x):
        raise GraphReferenceError("H input shape/nonfinite mismatch")
    if c.completed_tail_latch:
        return [0.0] * c.dof_count
    sums = [0.0] * c.dof_count
    for patch in c.patches:
        local = _solve_patch(c, patch, _patch_rhs(c, patch, x))
        for local_node, global_node in enumerate(patch.nodes):
            if not c.free_mask[global_node]:
                continue
            for axis in range(3):
                sums[3 * global_node + axis] += local[3 * local_node + axis]
    result = [0.0] * c.dof_count
    for node in range(c.node_count):
        coverage = c.node_coverage[node]
        if c.free_mask[node] and coverage:
            for axis in range(3):
                result[3 * node + axis] = sums[3 * node + axis] / coverage
    if not _finite(result):
        raise GraphReferenceError("H produced nonfinite output")
    return result


def apply_H_transpose(c: GraphCapture, y: Sequence[float]) -> Vector:
    """Apply H^T=E^T D^T G^T by reversing gather/solve/scatter."""
    validate_capture(c)
    if len(y) != c.dof_count or not _finite(y):
        raise GraphReferenceError("H transpose input shape/nonfinite mismatch")
    if c.completed_tail_latch:
        return [0.0] * c.dof_count
    out = [0.0] * c.dof_count
    for patch in c.patches:
        # G^T: distribute each global value to every patch copy with 1/coverage.
        local_rhs = []
        for node in patch.nodes:
            for axis in range(3):
                local_rhs.append(float(y[3 * node + axis]) / c.node_coverage[node]
                                 if c.free_mask[node] and c.node_coverage[node] else 0.0)
        local = _solve_patch_transpose(c, patch, local_rhs)
        # E^T: sum patch-local free entries back into the global vector.
        for local_node, global_node in enumerate(patch.nodes):
            if c.free_mask[global_node]:
                for axis in range(3):
                    out[3 * global_node + axis] += local[3 * local_node + axis]
    if not _finite(out):
        raise GraphReferenceError("H transpose produced nonfinite output")
    return out


def apply_base_S(c: GraphCapture, rhs: Sequence[float]) -> Vector:
    """Apply the source-derived unaugmented base selector-5 map S."""
    validate_capture(c)
    if len(rhs) != c.dof_count or not _finite(rhs):
        raise GraphReferenceError("S input shape/nonfinite mismatch")
    if c.completed_tail_latch:
        return [0.0] * c.dof_count
    k = c.mode_count
    P = c.P
    AP = c.AP
    coarse_rhs = [_dot([P[i][m] for i in range(c.dof_count)], rhs) for m in range(k)]
    coeff = rap.solve_rap(c.base_factor, coarse_rhs)
    residual = [float(rhs[i]) - _dot([AP[i][m] for m in range(k)], coeff)
                for i in range(c.dof_count)]
    fine = apply_H(c, residual)
    coarse = [sum(P[i][m] * coeff[m] for m in range(k))
              for i in range(c.dof_count)]
    result = [fine[i] + coarse[i] for i in range(c.dof_count)]
    if not _finite(result):
        raise GraphReferenceError("S produced nonfinite output")
    return result


def apply_base_S_transpose(c: GraphCapture, y: Sequence[float]) -> Vector:
    """Apply the algebraic transpose S^T without assuming symmetric RAP/A_p."""
    validate_capture(c)
    if len(y) != c.dof_count or not _finite(y):
        raise GraphReferenceError("S transpose input shape/nonfinite mismatch")
    if c.completed_tail_latch:
        return [0.0] * c.dof_count
    hty = apply_H_transpose(c, y)
    if not c.base_factor.valid:
        return hty  # production invalid RAP yields zero coefficients, retaining graph correction
    k = c.mode_count
    pty = [_dot([c.P[i][m] for i in range(c.dof_count)], y) for m in range(k)]
    apty = [_dot([c.AP[i][m] for i in range(c.dof_count)], hty) for m in range(k)]
    coarse_rhs = [pty[m] - apty[m] for m in range(k)]
    coeff = rap.solve_rap_transpose(c.base_factor, coarse_rhs)
    coarse = [sum(c.P[i][m] * coeff[m] for m in range(k))
              for i in range(c.dof_count)]
    result = [hty[i] + coarse[i] for i in range(c.dof_count)]
    if not _finite(result):
        raise GraphReferenceError("S transpose produced nonfinite output")
    return result


def reconstruct_matrix_from_lu(factor: rap.Factor) -> Matrix:
    """Reconstruct unpivoted matrix from final-permutation LU factors."""
    if not factor.valid:
        raise GraphReferenceError("cannot reconstruct invalid factor")
    n = len(factor.pivots)
    if sorted(factor.pivots) != list(range(n)):
        raise GraphReferenceError("factor pivot words are not a permutation")
    L = [[(1.0 if i == j else (factor.lu[i][j] if i > j else 0.0))
          for j in range(n)] for i in range(n)]
    U = [[factor.lu[i][j] if j >= i else 0.0 for j in range(n)]
         for i in range(n)]
    permuted = rap.matmul(L, U)
    original = [[0.0] * n for _ in range(n)]
    for row_in_factor, original_row in enumerate(factor.pivots):
        original[original_row] = permuted[row_in_factor]
    return original


def factor_reconstruction_error(original: Matrix, factor: rap.Factor) -> dict[str, float]:
    reconstructed = reconstruct_matrix_from_lu(factor)
    if len(original) != len(reconstructed):
        raise GraphReferenceError("factor reconstruction extent mismatch")
    scale = max((abs(x) for row in original for x in row), default=0.0)
    max_abs = max((abs(original[i][j] - reconstructed[i][j])
                   for i in range(len(original)) for j in range(len(original))), default=0.0)
    return {"max_abs": max_abs, "scale": scale,
            "relative_to_max_abs": max_abs / scale if scale else max_abs}


def base_rap_matrix(c: GraphCapture) -> Matrix:
    """Recompute P^T AP from captured base effective P/AP using fsum."""
    validate_capture(c)
    return [[math.fsum(c.P[i][a] * c.AP[i][b] for i in range(c.dof_count))
             for b in range(c.mode_count)] for a in range(c.mode_count)]


def compare_base_rap_to_ptap(c: GraphCapture, *, atol: float = 1e-10) -> dict[str, Any]:
    """Compare captured base LU reconstruction with a fresh P^T(AP) product."""
    validate_capture(c)
    if not c.base_factor.valid:
        return {"factor_valid": False, "comparison_available": False}
    fresh = base_rap_matrix(c)
    factored = reconstruct_matrix_from_lu(c.base_factor)
    scale = max((abs(v) for row in fresh for v in row), default=0.0)
    maximum = max((abs(fresh[i][j] - factored[i][j])
                   for i in range(c.mode_count) for j in range(c.mode_count)), default=0.0)
    return {"factor_valid": True, "comparison_available": True,
            "fresh_PtAP_max_abs": scale, "max_abs_difference": maximum,
            "relative_to_fresh_max_abs": maximum / scale if scale else maximum,
            "within_atol": maximum <= atol, "atol": atol}


def compare_patch_matrices_to_global_A(c: GraphCapture, A: Matrix, *,
                                      identity_expected: tuple[Any, ...],
                                      atol: float = 1e-10) -> dict[str, Any]:
    """Compare captured principal free blocks to a caller-bound same-generation A.

    Fixed coordinates in graph local matrices are source-authored identity rows
    and columns. The caller must bind `identity_expected` to the same world,
    candidate and Newton key before this numerical comparison is admissible.
    """
    validate_capture(c)
    if not c.identity or c.identity != identity_expected:
        raise GraphReferenceError("global operator identity does not match graph capture")
    _matrix_shape(A, c.dof_count, c.dof_count, "global operator A")
    worst_free = worst_fixed = 0.0
    compared_free = compared_fixed = 0
    for patch in c.patches:
        n = len(patch.nodes) * 3
        for lr in range(n):
            node_r = patch.nodes[lr // 3]
            gr = 3 * node_r + lr % 3
            for lc in range(n):
                node_c = patch.nodes[lc // 3]
                gc = 3 * node_c + lc % 3
                expected = A[gr][gc] if c.free_mask[node_r] and c.free_mask[node_c] else (1.0 if lr == lc and not c.free_mask[node_r] else 0.0)
                err = abs(patch.original[lr][lc] - expected)
                if c.free_mask[node_r] and c.free_mask[node_c]:
                    worst_free = max(worst_free, err); compared_free += 1
                else:
                    worst_fixed = max(worst_fixed, err); compared_fixed += 1
    return {"same_generation_identity_bound": True,
            "free_free_max_abs": worst_free,
            "fixed_identity_and_cross_max_abs": worst_fixed,
            "free_free_entries": compared_free,
            "fixed_identity_cross_entries": compared_fixed,
            "within_atol": worst_free <= atol and worst_fixed <= atol,
            "atol": atol}


def fnv1a32_graph_layout(patches: Sequence[Sequence[int]], rows: Sequence[Sequence[int]],
                         patch_nodes: Sequence[int], offsets: Sequence[int],
                         values: Sequence[int]) -> int:
    value = 2166136261
    for word in [*(w for p in patches for w in p),
                 *(w for r in rows for w in r), *patch_nodes, *offsets, *values]:
        if not (0 <= int(word) <= 0xffffffff):
            raise GraphReferenceError("layout checksum word outside uint32")
        value = ((value ^ int(word)) * 16777619) & 0xffffffff
    return value


def _read_f32(view: memoryview) -> tuple[float, ...]:
    if len(view) % 4:
        raise GraphReferenceError("f32 payload array has non-word size")
    vals = struct.unpack("<" + "f" * (len(view) // 4), view)
    if not _finite(vals):
        raise GraphReferenceError("payload array contains nonfinite float32")
    return tuple(float(v) for v in vals)


def _read_u32(view: memoryview) -> tuple[int, ...]:
    if len(view) % 4:
        raise GraphReferenceError("u32 payload array has non-word size")
    return tuple(int.from_bytes(view[i:i + 4], "little") for i in range(0, len(view), 4))


def _hash_identity_bytes(topology_bytes: bytes, free_mask_bytes: bytes,
                         potential_mask_bytes: bytes, graph_meta: dict[str, Any]) -> None:
    expected = {
        "source_topology_sha256": hashlib.sha256(topology_bytes).hexdigest(),
        "source_free_mask_sha256": hashlib.sha256(free_mask_bytes).hexdigest(),
        "potential_node_mask_sha256": hashlib.sha256(potential_mask_bytes).hexdigest(),
    }
    for key, digest in expected.items():
        if graph_meta.get(key) != digest:
            raise GraphReferenceError(f"compiled source identity mismatch for {key}")


def load_base_view(regional_manifest_path: str | Path, *, environment: int,
                   topology_bytes: bytes, free_mask_bytes: bytes,
                   potential_mask_bytes: bytes,
                   request_actual_consumed_view: bool = False) -> tuple[GraphCapture, dict[str, Any]]:
    """Strictly load one base view from a source151 joined regional/state pair.

    Caller-supplied exact compiled topology/free/potential-mask bytes are
    mandatory. This prevents the sidecar's self-reported hashes from being
    mistaken for independently verified world identity.
    """
    manifest_path = Path(regional_manifest_path)
    try:
        summary = reader.read_capture(manifest_path)
        manifest = reader._strict_json(manifest_path)
        payload_path, payload = reader._local_payload(manifest_path, manifest["payload"])
        arrays = reader._validate_arrays(manifest, payload, strict_dtype=True)
        state_path = manifest_path.parent / manifest["candidate_state_binding"]["manifest_file"]
        state_manifest = reader._strict_json(state_path)
        state_payload_path, state_payload = reader._local_payload(state_path, state_manifest["payload"])
        state_arrays = reader._validate_arrays(state_manifest, state_payload, strict_dtype=False)
    except Exception as exc:
        raise GraphReferenceError(f"strict source151 capture reader rejected payload: {exc}") from exc
    if manifest.get("preconditioner_selector") != 5 or manifest.get("graph_schwarz", {}).get("enabled") is not True:
        raise GraphReferenceError("capture is not selector-5 graph Schwarz")
    env_count, nodes = manifest["environment_count"], manifest["node_count"]
    if not (0 <= environment < env_count):
        raise GraphReferenceError("environment index outside capture")
    graph_meta = manifest["graph_schwarz"]
    _hash_identity_bytes(topology_bytes, free_mask_bytes, potential_mask_bytes, graph_meta)
    if len(free_mask_bytes) != nodes or len(potential_mask_bytes) != nodes:
        raise GraphReferenceError("source mask byte length differs from node count")
    if any(x not in (0, 1) for x in free_mask_bytes + potential_mask_bytes):
        raise GraphReferenceError("source masks must be raw binary u8 bytes")
    if topology_bytes[:4] != nodes.to_bytes(4, "little"):
        raise GraphReferenceError("topology digest input does not begin with expected node count")
    if len(topology_bytes) < 8:
        raise GraphReferenceError("topology digest bytes are truncated")
    tet_count = int.from_bytes(topology_bytes[4:8], "little")
    if tet_count != manifest["tetrahedron_count"] or len(topology_bytes) != 8 + 16 * tet_count:
        raise GraphReferenceError("topology digest extent/count differs from manifest")
    for offset in range(8, len(topology_bytes), 4):
        if int.from_bytes(topology_bytes[offset:offset + 4], "little") >= nodes:
            raise GraphReferenceError("topology digest contains out-of-range node ID")

    # Candidate nodes are 80-byte records; restAndFixed.w is the third float4's w.
    node_entry = next((x for x in state_manifest["arrays"] if x["name"] == "candidate_nodes"), None)
    if node_entry is None or node_entry.get("element_bytes") != 80 or tuple(node_entry.get("shape", ())) != (env_count, nodes):
        raise GraphReferenceError("joined state candidate node array is malformed")
    node_view = state_arrays["candidate_nodes"]
    start = environment * nodes * 80
    free_mask = []
    candidate_tag_bytes = bytearray()
    for node in range(nodes):
        tag = struct.unpack_from("<f", node_view, start + node * 80 + 44)[0]
        if not math.isfinite(tag) or tag not in (0.0, 1.0, 3.0):
            raise GraphReferenceError("candidate fixed tag is not a supported exact source tag")
        is_free = tag == 0.0
        free_mask.append(is_free)
        candidate_tag_bytes.append(1 if is_free else 0)
    if bytes(candidate_tag_bytes) != free_mask_bytes:
        raise GraphReferenceError("candidate free/fixed tags differ from compiled source mask")

    patch_words = _read_u32(arrays["graph_patches"])
    patch_nodes = _read_u32(arrays["graph_patch_nodes"])
    row_words = _read_u32(arrays["graph_rows"])
    offsets = _read_u32(arrays["graph_node_patch_offsets"])
    values = _read_u32(arrays["graph_node_patch_values"])
    coverage = _read_u32(arrays["graph_node_coverage"])
    patch_desc = [tuple(patch_words[4*i:4*i+4]) for i in range(graph_meta["patch_count"])]
    rows = [tuple(row_words[4*i:4*i+4]) for i in range(graph_meta["dof_count"])]
    checksum = fnv1a32_graph_layout(patch_desc, rows, patch_nodes, offsets, values)
    if graph_meta.get("layout_checksum_fnv1a32") != checksum:
        raise GraphReferenceError("graph layout FNV-1a checksum mismatch")
    if bytes(potential_mask_bytes) != bytes([1] * nodes):
        # Current source151 graph builder authors all nodes as potential. Keep
        # other variants unsupported until source-derived semantics are added.
        raise GraphReferenceError("unsupported graph potential-mask policy")

    mode_count = manifest["base_mode_count"]
    stride = manifest["mode_stride"]
    p_words = _read_f32(arrays["effective_P"])
    ap_words = _read_f32(arrays["AP"])
    expected_xyz4 = env_count * stride * nodes * 4
    if len(p_words) != expected_xyz4 or len(ap_words) != expected_xyz4:
        raise GraphReferenceError("effective P/AP payload extents differ from metadata")
    P = [[0.0] * mode_count for _ in range(3 * nodes)]
    AP = [[0.0] * mode_count for _ in range(3 * nodes)]
    for m in range(mode_count):
        base = (environment * stride + m) * nodes * 4
        for node in range(nodes):
            for axis in range(3):
                P[3*node+axis][m] = p_words[base + 4*node + axis]
                AP[3*node+axis][m] = ap_words[base + 4*node + axis]
    base_lu_words = _read_f32(arrays["base_RAP_lu"])
    base_piv_words = _read_u32(arrays["base_RAP_pivots"])
    base_valid_words = _read_u32(arrays["base_RAP_valid"])
    rap_start = environment * mode_count * mode_count
    rap_lu = [list(base_lu_words[rap_start+i*mode_count:rap_start+(i+1)*mode_count]) for i in range(mode_count)]
    rap_piv = tuple(base_piv_words[environment*mode_count:(environment+1)*mode_count])
    if base_valid_words[environment] not in (0, 1):
        raise GraphReferenceError("base RAP validity is not boolean")
    base_factor = rap.Factor(rap_lu, rap_piv, bool(base_valid_words[environment]))

    original_words = _read_f32(arrays["graph_original_matrices"])
    lu_words = _read_f32(arrays["graph_patch_lu"])
    pivot_words = _read_u32(arrays["graph_pivots"])
    valid_words = _read_u32(arrays["graph_factor_valid"])
    patches = []
    matrix_cursor = pivot_cursor = 0
    for pidx, (node_offset, node_count, matrix_offset, pivot_offset) in enumerate(patch_desc):
        if node_offset != len([n for patch in patches for n in patch.nodes]) or matrix_offset != matrix_cursor or pivot_offset != pivot_cursor:
            raise GraphReferenceError("noncanonical packed graph descriptor offsets")
        n_dof = 3 * node_count
        if matrix_offset + n_dof*n_dof > graph_meta["matrix_float_count"] or pivot_offset+n_dof > graph_meta["pivot_count"]:
            raise GraphReferenceError("graph patch exceeds packed matrix/pivot arena")
        mbase = environment * graph_meta["matrix_float_count"] + matrix_offset
        obase = [list(original_words[mbase+i*n_dof:mbase+(i+1)*n_dof]) for i in range(n_dof)]
        lbase = [list(lu_words[mbase+i*n_dof:mbase+(i+1)*n_dof]) for i in range(n_dof)]
        pb = environment * graph_meta["pivot_count"] + pivot_offset
        pvals = tuple(pivot_words[pb:pb+n_dof])
        if valid_words[environment*graph_meta["patch_count"]+pidx] not in (0, 1):
            raise GraphReferenceError("graph patch factor-valid word is not boolean")
        factor = rap.Factor(lbase, pvals, bool(valid_words[environment*graph_meta["patch_count"]+pidx]))
        nodes_for_patch = tuple(patch_nodes[node_offset:node_offset+node_count])
        patches.append(GraphPatch(nodes_for_patch, obase, factor))
        matrix_cursor += n_dof*n_dof
        pivot_cursor += n_dof
    capture = GraphCapture(
        node_count=nodes, free_mask=tuple(free_mask), patches=tuple(patches),
        patch_nodes=tuple(patch_nodes), node_patch_offsets=tuple(offsets),
        node_patch_values=tuple(values), node_coverage=tuple(coverage),
        P=P, AP=AP, base_factor=base_factor, mode_stride=stride,
        augmentation_enabled=manifest.get("same_newton_restart_augmentation_enabled") is True,
        # NMFGMRESStateGPU.diagnostics = float4(current, initial, latch, cols).
        # Graph kernels zero their output when diagnostics.z > 0.5.
        completed_tail_latch=(struct.unpack_from("<f", state_arrays["fgmres_state"], environment*32 + 8)[0] > 0.5),
        identity=(manifest.get("source_physics_fingerprint"),
                  manifest.get("device_program_fingerprint"),
                  manifest.get("root"), manifest.get("physics_substep"),
                  manifest.get("microtick"), manifest.get("micro_generation"),
                  manifest.get("newton_iteration"), manifest.get("dt_seconds"),
                  manifest.get("candidate_linearization_sha256")),
    )
    validate_capture(capture)
    view = require_base_view(capture, request_actual_consumed_view=request_actual_consumed_view)
    return capture, {"strict_reader_summary": summary, "mode_view": view,
                    "augmentation_enabled": capture.augmentation_enabled,
                    "actual_selected_view_admissible": False,
                    "unavailable": "selected restart-cycle consumption and arbitrary-vector native output"}
