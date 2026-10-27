"""Component-wide witness damping around immutable 582 retention logic.

Only damping scope changes: one failed witness halves every mutable node in
its mesh-edge-connected component. This is a construction heuristic only.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069")
CORE582 = BASE / "D/surface-clearance-546/local-witness-retention-582/retain_local_witnesses582.py"
CORE582_SHA256 = "08bb82034180636d0ff798ad503735b6a50327bf52cc3c62bffb75d0265ff57a"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validated_mesh(faces, node_count: int) -> np.ndarray:
    F = np.asarray(faces)
    if F.ndim != 2 or F.shape[1] != 3 or F.dtype.kind not in "iu" or len(F) == 0:
        raise ValueError("faces must be a nonempty integer Mx3 array")
    F = F.astype(np.int64, copy=False)
    if node_count < 1 or F.min(initial=0) < 0 or F.max(initial=-1) >= node_count:
        raise ValueError("faces reference outside the node domain")
    if np.any((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 2] == F[:, 0])):
        raise ValueError("face has a repeated node")
    return F


def mutable_components588(faces, node_count: int, mutable) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Connected components of mutable nodes under current triangle-edge adjacency."""
    F = _validated_mesh(faces, int(node_count))
    mask = np.asarray(mutable)
    if mask.shape != (node_count,) or mask.dtype.kind != "b":
        raise ValueError("mutable must be a Boolean vector in the point domain")
    parent = np.arange(node_count, dtype=np.int64)
    rank = np.zeros(node_count, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b, c in F:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if mask[u] and mask[v]:
                union(u, v)
    groups: dict[int, list[int]] = {}
    for node in np.flatnonzero(mask):
        groups.setdefault(find(int(node)), []).append(int(node))
    ordered = sorted((np.asarray(nodes, dtype=np.int64) for nodes in groups.values()),
                     key=lambda rows: int(rows[0]))
    labels = np.full(node_count, -1, dtype=np.int64)
    for component, nodes in enumerate(ordered):
        labels[nodes] = component
    return labels, tuple(ordered)


def expand_witness_components588(witness_nodes, mutable, component_labels,
                                 components) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mutable)
    labels = np.asarray(component_labels)
    nodes = np.asarray(witness_nodes)
    if mask.ndim != 1 or mask.dtype.kind != "b" or labels.shape != mask.shape:
        raise ValueError("component map and mutable mask have inconsistent domains")
    if nodes.ndim != 1 or nodes.dtype.kind not in "iu" or not len(nodes):
        raise ValueError("witness nodes must be a nonempty integer vector")
    nodes = np.unique(nodes.astype(np.int64, copy=False))
    if nodes.min() < 0 or nodes.max() >= len(mask):
        raise ValueError("witness node outside current point domain")
    seeds = nodes[mask[nodes]]
    if not len(seeds):
        raise ValueError("witness has no mutable nodes")
    component_ids = np.unique(labels[seeds])
    if np.any(component_ids < 0) or np.any(component_ids >= len(components)):
        raise ValueError("witness mutable node is missing from component map")
    expanded = np.unique(np.concatenate([np.asarray(components[int(k)], dtype=np.int64)
                                         for k in component_ids]))
    if not np.all(mask[expanded]):
        raise AssertionError("component expansion included an immutable node")
    return expanded, component_ids.astype(np.int64, copy=False)


def damp_connected588(alpha, witness_nodes, mutable, faces,
                      *, component_labels=None, components=None,
                      factor: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Damp whole mutable connected components and preserve all other nodes."""
    a = np.asarray(alpha)
    mask = np.asarray(mutable)
    if a.ndim != 1 or a.dtype.kind != "f" or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("alpha must be a finite floating point vector in [0,1]")
    if mask.shape != a.shape or mask.dtype.kind != "b":
        raise ValueError("mutable mask must be Boolean and match alpha domain")
    if not np.isfinite(factor) or not 0 < factor < 1:
        raise ValueError("damping factor must be finite and strictly between zero and one")
    if component_labels is None or components is None:
        component_labels, components = mutable_components588(faces, len(a), mask)
    expanded, ids = expand_witness_components588(witness_nodes, mask, component_labels, components)
    for component in ids:
        members = components[int(component)]
        if not np.all(a[members] == a[members[0]]):
            raise ValueError("component alpha is not uniform before damping")
    out = a.copy()
    out[expanded] *= factor
    if not np.array_equal(out[~mask].view(np.uint8), a[~mask].view(np.uint8)):
        raise AssertionError("zero-displacement nodes changed alpha")
    return out, expanded, ids


def boundary_nodes588(faces, node_count: int) -> np.ndarray:
    F = _validated_mesh(faces, node_count)
    edge_uses: dict[tuple[int, int], int] = {}
    for a, b, c in F:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (u, v) if u < v else (v, u)
            edge_uses[edge] = edge_uses.get(edge, 0) + 1
    if any(count > 2 for count in edge_uses.values()):
        raise ValueError("nonmanifold edge in boundary scan")
    return np.asarray(sorted({node for edge, count in edge_uses.items() if count == 1 for node in edge}),
                      dtype=np.int64)


def validate_zero_boundary588(base_points, target_points, faces) -> np.ndarray:
    base, target = np.asarray(base_points), np.asarray(target_points)
    if base.ndim != 2 or base.shape[1] != 3 or target.shape != base.shape:
        raise ValueError("base and target must share a point-domain shape")
    boundary = boundary_nodes588(faces, len(base))
    if not np.array_equal(base[boundary].view(np.uint8), target[boundary].view(np.uint8)):
        raise ValueError("boundary point targets must be bitwise zero displacement")
    return boundary


def install_damper588(core_module, faces, mutable):
    """Patch only 582 damping callback in memory; never edit the frozen core."""
    node_count = len(np.asarray(mutable))
    mask = np.asarray(mutable)
    labels, components = mutable_components588(faces, node_count, mask)
    events = []

    def component_damp(alpha, witness_nodes, current_mutable):
        current = np.asarray(current_mutable)
        if current.shape != mask.shape or current.dtype.kind != "b" or not np.array_equal(current, mask):
            raise ValueError("582 mutable scope changed after 588 component map was pinned")
        out, expanded, component_ids = damp_connected588(
            alpha, witness_nodes, current, faces,
            component_labels=labels, components=components)
        events.append({
            "witness_nodes": np.unique(np.asarray(witness_nodes, dtype=np.int64)).tolist(),
            "damped_component_ids": component_ids.tolist(),
            "damped_node_count": int(len(expanded)),
            "component_alpha_before": [float(np.asarray(alpha)[components[int(k)][0]]) for k in component_ids],
            "component_alpha_after": [float(out[components[int(k)][0]]) for k in component_ids],
        })
        return out, expanded

    core_module.damp = component_damp
    return {
        "component_labels": labels,
        "components": components,
        "events": events,
        "damping_method": "halve each mutable node in every mesh-edge-connected component touched by a failure witness",
        "mutable_component_count": int(len(components)),
        "mutable_component_sizes": [int(len(nodes)) for nodes in components],
        "boundary_zero_required": True,
    }
