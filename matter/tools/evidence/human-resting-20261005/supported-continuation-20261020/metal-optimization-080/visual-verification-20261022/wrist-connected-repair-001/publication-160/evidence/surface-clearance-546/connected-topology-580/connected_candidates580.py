"""Unified fixed-coordinate proposal families for connected topology repair.

Proposal generation only: no coordinates move and no physical or exact gates are
replaced. The root runtime must still run quality, exact 185-bank and self checks.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
N = HERE.parent
OWNER_PINS = {
    N / "connected-star-569/star_candidates569.py":
        "81b33966ba0c2182c86d2a6d66862064105cf48e91c2efe0cfcc0ef6359d1b2c",
    N / "connected-patch-570/patch_candidates570.py":
        "5e309fd361ac99de50f1d3f9aeca8d5f5711a406969525c09a1a80481bf0861d",
    N / "regional-patch-571/regional_candidates571.py":
        "48d53b5516bc4e604a8c8bbfb7b100039e2646b3efcc2298010ed957fc9c3cfc",
    N / "harmonic-patch-577/harmonic_candidates577.py":
        "5352353ae6ef692d4ad340a82fd07494fa4dc0218fd997f6d9d5ebab93ca5803",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned proposal owner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _owners():
    for path, expected in OWNER_PINS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"proposal owner SHA mismatch for {path}: {actual}")
    return {
        "single": _load("star_candidates569_for580", N / "connected-star-569/star_candidates569.py"),
        "double": _load("patch_candidates570_for580", N / "connected-patch-570/patch_candidates570.py"),
        "regional": _load("regional_candidates571_for580", N / "regional-patch-571/regional_candidates571.py"),
        "harmonic": _load("harmonic_candidates577_for580", N / "harmonic-patch-577/harmonic_candidates577.py"),
    }


def _oriented_signature(faces):
    # Cyclic starts and row order are irrelevant; reversing a face is distinct.
    result = []
    for tri in np.asarray(faces):
        a, b, c = map(int, tri)
        result.append(min((a, b, c), (b, c, a), (c, a, b)))
    return tuple(sorted(result))


def _candidate_valid(candidate, old, vertices, boundary, points, max_boundary):
    owners = _owners()
    regional = owners["regional"]
    faces = np.asarray(candidate)
    if faces.shape != old.shape or faces.dtype.kind not in "iu":
        return False
    if any(len(set(map(int, tri))) != 3 for tri in faces):
        return False
    if len({_oriented_signature([tri]) for tri in faces}) != len(faces):
        return False
    if set(map(int, faces.ravel())) != set(map(int, vertices)):
        return False
    try:
        _, _, candidate_boundary, candidate_vertices = regional._boundary_and_vertices(
            np.asarray(points), faces, np.arange(len(faces), dtype=np.int64), max_boundary
        )
    except ValueError:
        return False
    return tuple(candidate_boundary) == tuple(boundary) and set(map(int, candidate_vertices)) == set(map(int, vertices))


def _zero_interior_candidates(points, reference, faces, old, boundary, vertices, max_boundary):
    owners = _owners()
    regional, single = owners["regional"], owners["single"]
    if len(boundary) > 9 or len(vertices) != len(boundary):
        return []
    try:
        current_chart = regional._chart(points, boundary, vertices)
        reference_chart = regional._chart(reference, boundary, vertices)
    except ValueError:
        return []
    chart_data = (current_chart, reference_chart)
    old_sig = _oriented_signature(old)
    out = {}
    for polygon, _xy, eps in chart_data:
        for tri_indices in single.polygon_triangulations(polygon, eps):
            proposal = np.asarray(
                [[boundary[int(i)] for i in tri] for tri in tri_indices],
                dtype=np.asarray(faces).dtype,
            )
            if _oriented_signature(proposal) == old_sig:
                continue
            if not regional._valid_candidate(proposal, old, vertices, boundary, chart_data):
                continue
            out[_oriented_signature(proposal)] = proposal
    return [out[key] for key in sorted(out)]


def patch_candidates(points, reference, faces, rows, centers=None, max_boundary=48):
    """Union 569/570/571/577 proposal families with deterministic deduplication."""
    owners = _owners()
    single = owners["single"]
    double = owners["double"]
    regional = owners["regional"]
    harmonic = owners["harmonic"]

    points = np.asarray(points)
    reference = np.asarray(reference)
    faces = np.asarray(faces)
    if reference.shape != points.shape or reference.dtype.kind != "f" or not np.isfinite(reference).all():
        raise ValueError("reference must be finite floating Nx3 matching points")
    if isinstance(max_boundary, bool) or not isinstance(max_boundary, (int, np.integer)) or max_boundary < 3:
        raise ValueError("max_boundary must be an integer >= 3")
    row_ids, old, boundary, vertices = regional._boundary_and_vertices(
        points, faces, rows, int(max_boundary)
    )
    if centers is None:
        requested_centers = None
    else:
        requested_centers = tuple(map(int, centers))
        if len(set(requested_centers)) != len(requested_centers):
            raise ValueError("centers must be unique")
        interior = set(map(int, vertices)) - set(map(int, boundary))
        if not set(requested_centers).issubset(interior):
            raise ValueError("centers must be interior patch vertices")

    interior = sorted(set(map(int, vertices)) - set(map(int, boundary)))
    row_set = set(map(int, row_ids))
    candidates = {}

    def add_many(items):
        for item in items:
            proposal = np.asarray(item, dtype=faces.dtype)
            if not _candidate_valid(proposal, old, vertices, boundary, points, int(max_boundary)):
                continue
            key = _oriented_signature(proposal)
            if key != _oriented_signature(old):
                candidates[key] = proposal

    # No-interior disk: enumerate boundary triangulations in current and 037 PCA charts.
    if not interior:
        add_many(_zero_interior_candidates(
            points, reference, faces, old, boundary, vertices, int(max_boundary)
        ))

    # One-interior star: specialized 569 fan retessellation when rows are the
    # complete star; generic disk families still run below.
    if len(interior) == 1 and len(boundary) <= 9:
        center = interior[0]
        full_star = set(map(int, np.flatnonzero(np.any(faces == center, axis=1))))
        if full_star == row_set and (requested_centers is None or requested_centers == (center,)):
            try:
                add_many(single.star_candidates(
                    points, reference, faces, row_ids, center, max_degree=9
                ))
            except ValueError:
                pass

    # Two-interior union: use the certified 570 complete-two-star proposal
    # generator only when its complete-union contract is satisfied.
    if len(interior) == 2 and len(boundary) <= 9:
        centers2 = tuple(interior)
        if requested_centers is None or set(requested_centers) == set(centers2):
            try:
                add_many(double.patch_candidates(
                    points, reference, faces, row_ids, centers2, max_boundary=9
                ))
            except ValueError:
                pass

    # General disk proposals: PCA Delaunay (571) and positive-weight harmonic
    # Delaunay (577) cover all interior counts without changing body coordinates.
    for owner in (regional, harmonic):
        try:
            add_many(owner.patch_candidates(
                points, reference, faces, row_ids, centers=requested_centers,
                max_boundary=int(max_boundary)
            ))
        except ValueError:
            pass
    return [candidates[key] for key in sorted(candidates)]


def external_edge_conflict(faces, rows, new_faces, edge_map):
    """Reuse the pinned 569 outside-edge conflict check."""
    return bool(_owners()["single"].external_edge_conflict(faces, rows, new_faces, edge_map))

