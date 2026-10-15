#!/usr/bin/env python3
"""Bounded CPU prototype for mixed-mortar covered/uncovered triangle unions.

This is a geometry algorithm prototype only. It is not wired to the native
contact solver and does not qualify native CCD, forces, or cohesive history.
Coordinates are barycentric triples (w0,w1,w2) on one parent triangle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Bary = tuple[float, float, float]
Triangle = tuple[Point, Point, Point]


class PartitionError(ValueError):
    """Input is unsupported, invalid, or exceeds the explicit output bound."""


@dataclass(frozen=True)
class Patch:
    stable_id: int
    side_a: tuple[Bary, Bary, Bary]
    side_b: tuple[Bary, Bary, Bary]
    active: bool = True


@dataclass(frozen=True)
class Cell:
    side: str
    covered: bool
    patch_id: int | None
    vertices: Triangle


@dataclass(frozen=True)
class Partition:
    side: str
    covered_area: float
    uncovered_area: float
    cells: tuple[Cell, ...]

    @property
    def area_error(self) -> float:
        return self.covered_area + self.uncovered_area - 0.5


def _point(bary: Sequence[float]) -> Point:
    if len(bary) != 3 or not all(math.isfinite(float(v)) for v in bary):
        raise PartitionError("barycentric vertex must have three finite coordinates")
    values = tuple(float(v) for v in bary)
    if min(values) < -1.0e-12 or max(values) > 1.0 + 1.0e-12:
        raise PartitionError("barycentric vertex lies outside the parent simplex")
    if abs(sum(values) - 1.0) > 1.0e-10:
        raise PartitionError("barycentric vertex weights must sum to one")
    return values[1], values[2]


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _signed_area(poly: Sequence[Point]) -> float:
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1]
        - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    ) if len(poly) >= 3 else 0.0


def _normalized_triangle(vertices: Sequence[Bary], tolerance: float) -> Triangle:
    if len(vertices) != 3:
        raise PartitionError("each mortar side patch must have exactly three vertices")
    tri = tuple(_point(v) for v in vertices)
    area = _signed_area(tri)
    if abs(area) <= tolerance:
        raise PartitionError("patch triangle is degenerate at the configured area tolerance")
    if area < 0.0:
        tri = (tri[0], tri[2], tri[1])
    return tri  # type: ignore[return-value]


def _line_intersection(p: Point, q: Point, a: Point, b: Point) -> Point:
    vp = _cross(a, b, p)
    vq = _cross(a, b, q)
    denominator = vp - vq
    if denominator == 0.0:
        return p
    t = vp / denominator
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _clip_halfplane(poly: Sequence[Point], a: Point, b: Point,
                    keep_inside: bool, tolerance: float) -> list[Point]:
    if not poly:
        return []
    result: list[Point] = []
    previous = poly[-1]
    prev_value = _cross(a, b, previous)
    prev_inside = prev_value >= -tolerance if keep_inside else prev_value <= tolerance
    for current in poly:
        value = _cross(a, b, current)
        inside = value >= -tolerance if keep_inside else value <= tolerance
        if inside != prev_inside:
            result.append(_line_intersection(previous, current, a, b))
        if inside:
            result.append(current)
        previous, prev_inside = current, inside
    # Remove adjacent duplicates created when a vertex lies on a clip edge.
    cleaned: list[Point] = []
    for p in result:
        if not cleaned or math.dist(cleaned[-1], p) > tolerance:
            cleaned.append(p)
    if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned


def _subtract_triangle(subject: Sequence[Point], clip: Triangle,
                       tolerance: float) -> list[list[Point]]:
    """Return disjoint convex pieces of subject outside a CCW triangle."""
    inside_piece = list(subject)
    outside: list[list[Point]] = []
    for edge in range(3):
        a, b = clip[edge], clip[(edge + 1) % 3]
        piece = _clip_halfplane(inside_piece, a, b, False, tolerance)
        if len(piece) >= 3 and abs(_signed_area(piece)) > tolerance:
            if _signed_area(piece) < 0.0:
                piece.reverse()
            outside.append(piece)
        inside_piece = _clip_halfplane(inside_piece, a, b, True, tolerance)
        if len(inside_piece) < 3:
            break
    return outside


def _intersection_area(a: Triangle, b: Triangle, tolerance: float) -> float:
    polygon: list[Point] = list(a)
    for edge in range(3):
        polygon = _clip_halfplane(
            polygon, b[edge], b[(edge + 1) % 3], True, tolerance)
        if len(polygon) < 3:
            return 0.0
    return abs(_signed_area(polygon))


def _fan(poly: Sequence[Point]) -> Iterable[Triangle]:
    for i in range(1, len(poly) - 1):
        tri = (poly[0], poly[i], poly[i + 1])
        area = _signed_area(tri)
        if area > 0.0:
            yield tri
        elif area < 0.0:
            yield (tri[0], tri[2], tri[1])


def partition_side(patches: Sequence[Patch], side: str, *,
                   max_triangles: int = 256,
                   tolerance: float = 1.0e-12) -> Partition:
    """Partition one parent simplex into active covered patches and complement.

    Active patch interiors must be nonoverlapping. Shared edges and vertices
    are allowed. Dormant patches are omitted from runtime coverage. `side` is
    either "a" or "b"; the two sides are partitioned independently because
    their authored barycentric maps need not be identical.
    """
    if side not in ("a", "b"):
        raise PartitionError("side must be 'a' or 'b'")
    if max_triangles < 1 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise PartitionError("output bound and geometric tolerance must be positive")
    active = sorted((p for p in patches if p.active), key=lambda p: p.stable_id)
    if len(active) > max_triangles:
        raise PartitionError("active patch count exceeds configured bounded workspace")
    if len({p.stable_id for p in active}) != len(active):
        raise PartitionError("active patch stable identifiers must be unique")
    clip_triangles: list[tuple[int, Triangle]] = []
    for patch in active:
        vertices = patch.side_a if side == "a" else patch.side_b
        clip_triangles.append((patch.stable_id, _normalized_triangle(vertices, tolerance)))

    # Reject positive-area overlaps only; edge/vertex contacts have zero area.
    for i, (id_a, tri_a) in enumerate(clip_triangles):
        for id_b, tri_b in clip_triangles[i + 1:]:
            overlap = _intersection_area(tri_a, tri_b, tolerance)
            if overlap > tolerance:
                raise PartitionError(
                    f"active patches {id_a} and {id_b} overlap by area {overlap:.17g}")

    uncovered_polygons: list[list[Point]] = [[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]
    for _, clip in clip_triangles:
        next_polygons: list[list[Point]] = []
        for polygon in uncovered_polygons:
            next_polygons.extend(_subtract_triangle(polygon, clip, tolerance))
            if len(next_polygons) > max_triangles:
                raise PartitionError("polygon partition exceeds configured bounded workspace")
        uncovered_polygons = next_polygons

    cells: list[Cell] = []
    for stable_id, tri in clip_triangles:
        cells.append(Cell(side, True, stable_id, tri))
    if len(cells) > max_triangles:
        raise PartitionError("covered patch cells exceed configured bounded workspace")
    for polygon in uncovered_polygons:
        cells.extend(Cell(side, False, None, tri) for tri in _fan(polygon))
        if len(cells) > max_triangles:
            raise PartitionError("triangulated output exceeds configured bounded workspace")

    covered_area = sum(abs(_signed_area(tri)) for _, tri in clip_triangles)
    uncovered_area = sum(abs(_signed_area(cell.vertices))
                         for cell in cells if not cell.covered)
    area_error = covered_area + uncovered_area - 0.5
    if abs(area_error) > max(tolerance * 100.0, 1.0e-10):
        raise PartitionError(f"partition does not conserve parent area: error={area_error:.17g}")
    if any(_signed_area(cell.vertices) <= 0.0 for cell in cells):
        raise PartitionError("partition output contains a nonpositive-orientation triangle")
    return Partition(side, covered_area, uncovered_area, tuple(cells))


def covered_patch_ids_at(patches: Sequence[Patch], side: str,
                         point: Bary, *, tolerance: float = 1.0e-10) -> tuple[int, ...]:
    """Return sorted active patch IDs whose closed triangles contain a point.

    Shared patch-edge queries may return multiple IDs. Contact suppression is
    based on whether this tuple is nonempty, while constitutive integration
    retains one independent history per patch.
    """
    if side not in ("a", "b"):
        raise PartitionError("side must be 'a' or 'b'")
    p = _point(point)
    found: list[int] = []
    for patch in sorted((p for p in patches if p.active), key=lambda p: p.stable_id):
        vertices = patch.side_a if side == "a" else patch.side_b
        tri = _normalized_triangle(vertices, max(1.0e-15, tolerance * tolerance))
        if all(_cross(tri[i], tri[(i + 1) % 3], p) >= -tolerance
               for i in range(3)):
            found.append(patch.stable_id)
    return tuple(found)

