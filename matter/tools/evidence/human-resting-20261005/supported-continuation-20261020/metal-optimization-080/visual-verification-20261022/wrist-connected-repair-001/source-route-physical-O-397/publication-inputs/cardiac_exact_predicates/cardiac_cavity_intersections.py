"""Exact embeddedness and disjointness audit of published cavity geometry.

The executed geometry is the quotient's Float64 metre coordinates. Conversion
with Fraction.from_float preserves every binary coordinate exactly; a common
integer denominator then permits exact orientation and intersection predicates.
No tolerance, capping, vertex movement, physical stepping or overlap repair is
performed. Shared topological faces are never exempted wholesale: intersections
must be confined to their common vertex or edge. Nested closed solids are tested
by exact ray parity after the complete surface intersection test.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import hashlib
import itertools
import math
import struct

from .cardiac_cavity_geometry import analyze_topology
from .model import ImportError as HumanImportError
from .physiology import canonical

SCHEMA = "HumanPack.cardiac-cavity-intersections.v1"
ALGORITHM = "exact_binary64_integer_triangle_predicates_and_rational_ray_parity_v1"


def require(ok, message):
    if not ok:
        raise HumanImportError("cardiac cavity intersections: " + message)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _orient2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _axes(normal):
    return tuple(k for k in range(3) if k != max(range(3), key=lambda j: abs(normal[j])))


def _signs(numerator, denominator, triangle, normal):
    x, y = _axes(normal)
    return [(b[x]-a[x])*(numerator[y]-denominator*a[y]) -
            (b[y]-a[y])*(numerator[x]-denominator*a[x])
            for a, b in zip(triangle, triangle[1:]+triangle[:1])]


def _inside(signs):
    return min(signs) >= 0 or max(signs) <= 0


_FLOAT32_LATTICE_DENOMINATOR = 1 << 149


def float32_point_lattice_key(point):
    """Return the exact integer-lattice key for a packed Float32 xyz point.

    This is for interfaces exported from Float32 mesh payloads. Converting each
    packed coordinate through ``float`` preserves its value exactly, and the
    common 2**149 denominator is divisible by every finite Float32 denominator.
    Reject binary64-only coordinates so callers cannot accidentally compare a
    different representation from the one the native loader consumes.
    """
    require(len(point) == 3, "Float32 point must have three coordinates")
    result = []
    for value in point:
        coordinate = float(value)
        require(math.isfinite(coordinate), "Float32 point must be finite")
        try:
            packed = struct.pack("<f", coordinate)
        except (OverflowError, struct.error):
            require(False, "point coordinate is outside Float32 range")
        require(struct.unpack("<f", packed)[0] == coordinate,
                "point coordinate is not an exact Float32 value")
        numerator, denominator = coordinate.as_integer_ratio()
        require(_FLOAT32_LATTICE_DENOMINATOR % denominator == 0,
                "Float32 coordinate is not on the common integer lattice")
        result.append(numerator * (_FLOAT32_LATTICE_DENOMINATOR // denominator))
    return tuple(result)


def float32_triangle_lattice_key(vertices, face):
    """Return a winding-independent exact key for a packed Float32 triangle."""
    require(len(face) == 3, "Float32 face must have three vertex indices")
    points = tuple(float32_point_lattice_key(vertices[int(index)]) for index in face)
    require(len(set(points)) == 3, "Float32 face has repeated coordinates")
    return tuple(sorted(points))


def float32_lattice_point_on_triangle(point_numerator, denominator, triangle):
    """Test a rational point against an exact triangle on the Float32 lattice.

    ``point_numerator / denominator`` and ``triangle`` use the integer lattice
    above. The scalar denominator is shared by all three coordinates, matching
    the exact predicate owner used by intersection and source audits.
    """
    require(len(point_numerator) == 3 and len(triangle) == 3,
            "point and triangle must be xyz triples")
    require(isinstance(denominator, int) and denominator > 0,
            "point denominator must be a positive integer scalar")
    normal = _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))
    require(any(normal), "Float32 triangle is exactly degenerate")
    if sum(normal[k] * (triangle[0][k] * denominator - point_numerator[k])
           for k in range(3)) != 0:
        return False
    return _inside(_signs(point_numerator, denominator, triangle, normal))


def _coplanar_points(first, second, normal):
    axes = _axes(normal)
    a, b = [tuple(v[k] for k in axes) for v in first], [tuple(v[k] for k in axes) for v in second]
    def inside2(p, tri):
        return _inside([_orient2(tri[i], tri[(i+1) % 3], p) for i in range(3)])
    points = [first[i] for i, p in enumerate(a) if inside2(p, b)]
    points.extend(second[i] for i, p in enumerate(b) if inside2(p, a))
    # Collinear contact endpoints are included above. Strict edge crossings
    # supply every remaining vertex of the convex intersection polygon.
    for i in range(3):
        x, y = a[i], a[(i+1) % 3]
        for j in range(3):
            c, d = b[j], b[(j+1) % 3]
            u, v = _orient2(c, d, x), _orient2(c, d, y)
            r, s = _orient2(x, y, c), _orient2(x, y, d)
            if u*v < 0 and r*s < 0:
                t = Fraction(u, u-v)
                points.append(tuple(first[i][k]+t*(first[(i+1) % 3][k]-first[i][k]) for k in range(3)))
    return points


def _segment_triangle(p, q, triangle, normal):
    dp, dq = _dot(normal, _sub(p, triangle[0])), _dot(normal, _sub(q, triangle[0]))
    if (dp > 0 and dq > 0) or (dp < 0 and dq < 0) or dp == dq:
        return []
    denominator = dp-dq
    numerator = tuple(dp*q[k]-dq*p[k] for k in range(3))
    if denominator < 0:
        denominator, numerator = -denominator, tuple(-x for x in numerator)
    if not _inside(_signs(numerator, denominator, triangle, normal)):
        return []
    return [tuple(Fraction(x, denominator) for x in numerator)]


def _integer_lattice_normalization(points):
    """Map exact rational points to a small integer lattice, retaining its inverse."""
    denominator = math.lcm(*(Fraction(value).denominator
                             for point in points for value in point))
    integer_points = tuple(tuple(Fraction(value).numerator *
                                 (denominator // Fraction(value).denominator)
                                 for value in point) for point in points)
    origin = integer_points[0]
    scale = 0
    for point in integer_points:
        for axis in range(3):
            scale = math.gcd(scale, abs(point[axis] - origin[axis]))
    if scale == 0:
        scale = 1
    normalized = tuple(tuple((point[axis] - origin[axis]) // scale
                             for axis in range(3)) for point in integer_points)
    return normalized, (denominator, origin, scale)


def _normalized_query_lattice(point, transform):
    denominator, origin, scale = transform
    local = tuple((Fraction(point[axis]) * denominator - origin[axis]) / scale
                  for axis in range(3))
    common = math.lcm(*(coordinate.denominator for coordinate in local))
    numerators = tuple(coordinate.numerator * (common // coordinate.denominator)
                       for coordinate in local)
    return numerators, common


def _plane_numerator_at_local_query(numerator, denominator, triangle, normal):
    return _dot(normal, tuple(triangle[0][axis] * denominator - numerator[axis]
                              for axis in range(3)))


def _triangle_intersection_points_integer(first, second):
    """Unscaled exact predicate on integer-lattice triangles."""
    na = _cross(_sub(first[1], first[0]), _sub(first[2], first[0]))
    nb = _cross(_sub(second[1], second[0]), _sub(second[2], second[0]))
    require(any(na) and any(nb), "degenerate triangle")
    sa = [_dot(na, _sub(p, first[0])) for p in second]
    sb = [_dot(nb, _sub(p, second[0])) for p in first]
    if min(sa) > 0 or max(sa) < 0 or min(sb) > 0 or max(sb) < 0:
        return []
    if not any(sa):
        return _coplanar_points(first, second, na)
    points = []
    for i in range(3):
        points.extend(_segment_triangle(first[i], first[(i+1) % 3], second, nb))
        points.extend(_segment_triangle(second[i], second[(i+1) % 3], first, na))
    return points


def triangle_intersection_points(first, second):
    """Exact intersection vertices using a reversible pair-local integer lattice."""
    require(len(first) == 3 and len(second) == 3,
            "triangle intersection requires two triangles")
    exact_points = tuple(_exact_rational_xyz(point, "triangle intersection vertex")
                         for point in tuple(first) + tuple(second))
    normalized, transform = _integer_lattice_normalization(exact_points)
    local_points = _triangle_intersection_points_integer(normalized[:3], normalized[3:])
    denominator, origin, scale = transform
    return [tuple(Fraction(origin[axis], denominator) +
                  Fraction(scale, denominator) * point[axis]
                  for axis in range(3)) for point in local_points]


def _records(vertices, faces):
    records = []
    for index, ids in enumerate(faces):
        tri = tuple(vertices[k] for k in ids)
        require(any(_cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))), "exactly degenerate triangle")
        records.append((tri, tuple(min(v[k] for v in tri) for k in range(3)),
                        tuple(max(v[k] for v in tri) for k in range(3)), index, tuple(ids)))
    return records


def _allowed_shared_point(point, common):
    if point in common:
        return True
    if len(common) == 2:
        a, b = tuple(common)
        return _cross(_sub(point, a), _sub(b, a)) == (0, 0, 0) and all(
            min(a[k], b[k]) <= point[k] <= max(a[k], b[k]) for k in range(3))
    return False


@dataclass(frozen=True)
class _PreparedSurfaceAABB:
    records: tuple
    root: tuple | None


def _prepare_surface_aabb(records):
    """Snapshot exact immutable records and build the existing closed-box tree."""
    records = tuple(records)

    def tree(positions):
        lo = tuple(min(records[i][1][k] for i in positions) for k in range(3))
        hi = tuple(max(records[i][2][k] for i in positions) for k in range(3))
        if len(positions) <= 8:
            return lo, hi, tuple(positions), None, None
        axis = max(range(3), key=lambda k: hi[k]-lo[k])
        positions.sort(key=lambda i: records[i][1][axis]+records[i][2][axis])
        middle = len(positions)//2
        return lo, hi, None, tree(positions[:middle]), tree(positions[middle:])

    return _PreparedSurfaceAABB(
        records, tree(list(range(len(records)))) if records else None)


def _query_surface_aabb(queries, index):
    if index.root is None:
        return
    for row in queries:
        lo, hi = row[1:3]
        stack = [index.root]
        while stack:
            lower, upper, positions, left, right = stack.pop()
            if any(hi[k] < lower[k] or upper[k] < lo[k] for k in range(3)):
                continue
            if positions is None:
                stack.extend((right, left))
                continue
            for position in positions:
                other = index.records[position]
                if all(hi[k] >= other[1][k] and other[2][k] >= lo[k] for k in range(3)):
                    yield row, other


def _aabb_candidate_pairs(first, second, *, same_surface):
    """Enumerate closed-box overlaps with exact bounds and source face IDs."""
    if not first or not second:
        return
    for row, other in _query_surface_aabb(first, _prepare_surface_aabb(second)):
        if not same_surface or other[3] > row[3]:
            yield row, other


def _audit_pair_prepared_first(first_index, second, *, same_surface=False):
    """Reuse a first-surface index across targets without changing predicates.

    Querying the smaller target avoids repeatedly walking every skin face.
    Restore the original first/second argument order before the exact narrow
    phase, including face IDs and shared-index classification. The index owns
    a tuple snapshot, so replacing entries in a caller's list cannot stale it.
    """
    candidates = (
        (first, other)
        for other, first in _query_surface_aabb(second, first_index)
        if not same_surface or other[3] > first[3]
    )
    return _audit_record_pairs(candidates, same_surface=same_surface)


def _audit_pair(first, second, *, same_surface):
    if first is None or second is None:
        raise HumanImportError("missing triangle records: geometry preparation did not complete")
    return _audit_record_pairs(
        _aabb_candidate_pairs(first, second, same_surface=same_surface),
        same_surface=same_surface)


def _audit_self_pairs_touching_faces(records, face_ids):
    """Audit all self-pairs incident on selected faces in one complete record set.

    Face IDs need not be contiguous or sorted. Keep the original vertex IDs
    for shared-edge classification; never compact a patch independently from
    its complement. Canonicalize candidate order instead of filtering one
    direction, which can silently omit a crossing when a selected face has
    the lower ID.
    """
    if records is None:
        raise HumanImportError("missing triangle records: geometry preparation did not complete")
    snapshot = tuple(records)
    by_id = {row[3]: row for row in snapshot}
    require(len(by_id) == len(snapshot), "duplicate face IDs in selected-face audit")
    selected_ids = set(face_ids)
    require(selected_ids <= set(by_id), "selected face ID missing from complete record set")
    selected = tuple(row for row in snapshot if row[3] in selected_ids)

    def candidates():
        seen = set()
        for first, other in _query_surface_aabb(selected, _prepare_surface_aabb(snapshot)):
            i, j = first[3], other[3]
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            yield (first, other) if i < j else (other, first)

    return _audit_record_pairs(candidates(), same_surface=True)


def _audit_record_pairs(candidate_pairs, *, same_surface):
    candidates, allowed, pairs = 0, 0, []
    for (tri, lo, hi, i, ids), (other, lower, upper, j, other_ids) in candidate_pairs:
        candidates += 1
        points = triangle_intersection_points(tri, other)
        if not points:
            continue
        # Use shared *indices*, not coincident but topologically unrelated
        # coordinates. Duplicate triangles are defects even when coincident.
        shared_ids = set(ids) & set(other_ids) if same_surface else set()
        common = {tri[ids.index(index)] for index in shared_ids}
        if same_surface and len(shared_ids) in (1, 2) and all(_allowed_shared_point(p, common) for p in points):
            allowed += 1
        else:
            pairs.append([i, j])
    return {"triangle_pairs": sorted(pairs), "count": len(pairs), "aabb_candidate_pairs": candidates,
            "allowed_shared_vertex_or_edge_pairs": allowed}


def point_location(point, records):
    """Exact parity for a closed embedded surface; unresolved rays fail closed."""
    normals = [_cross(_sub(row[0][1], row[0][0]), _sub(row[0][2], row[0][0])) for row in records]
    for row, normal in zip(records, normals):
        tri = row[0]
        if _dot(normal, _sub(point, tri[0])) == 0 and _inside(_signs(point, 1, tri, normal)):
            return {"location": "boundary", "ray": None, "crossings": None}
    for attempt in range(1, 65):
        direction = (1, attempt, attempt*attempt)
        crossings, ambiguous = 0, False
        for row, normal in zip(records, normals):
            tri = row[0]
            numerator, denominator = _dot(normal, _sub(tri[0], point)), _dot(normal, direction)
            if denominator == 0:
                if numerator == 0:
                    ambiguous = True
                    break
                continue
            if numerator*denominator <= 0:
                continue
            hit = tuple(point[k]*denominator+numerator*direction[k] for k in range(3))
            if denominator < 0:
                denominator, hit = -denominator, tuple(-v for v in hit)
            signs = _signs(hit, denominator, tri, normal)
            if _inside(signs):
                if 0 in signs:
                    ambiguous = True
                    break
                crossings += 1
        if not ambiguous:
            return {"location": "inside" if crossings % 2 else "outside", "ray": list(direction),
                    "crossings": crossings, "attempts": attempt}
    return {"location": "indeterminate", "ray": None, "crossings": None, "attempts": 64}





def _prepare_exact_aabb_index(lower_bounds, upper_bounds, *, leaf_size=12):
    """Build the deterministic exact-box tree used by repeated point queries."""
    require(len(lower_bounds) == len(upper_bounds) and bool(lower_bounds),
            "point-location AABB index requires non-empty bounds")
    require(type(leaf_size) is int and leaf_size > 0, "invalid point-location AABB leaf size")
    rows = tuple((tuple(lower), tuple(upper))
                 for lower, upper in zip(lower_bounds, upper_bounds))
    node_count = 0

    def build(indices):
        nonlocal node_count
        node_count += 1
        lower = tuple(min(rows[index][0][axis] for index in indices) for axis in range(3))
        upper = tuple(max(rows[index][1][axis] for index in indices) for axis in range(3))
        if len(indices) <= leaf_size:
            return (lower, upper, tuple(sorted(indices)), None, None)
        centers = {axis: [rows[index][0][axis] + rows[index][1][axis]
                          for index in indices] for axis in range(3)}
        spans = {axis: max(centers[axis]) - min(centers[axis]) for axis in range(3)}
        axis = max(range(3), key=lambda coordinate: (spans[coordinate], -coordinate))
        if spans[axis] == 0:
            return (lower, upper, tuple(sorted(indices)), None, None)
        indices.sort(key=lambda index: (rows[index][0][axis] + rows[index][1][axis], index))
        middle = len(indices) // 2
        if middle == 0 or middle == len(indices):
            return (lower, upper, tuple(sorted(indices)), None, None)
        left = build(indices[:middle])
        right = build(indices[middle:])
        return (lower, upper, None, left, right)

    return build(list(range(len(rows)))), node_count


def _point_aabb_candidate_indices(point, prepared, *, include_forward=True):
    """Return sorted exact boundary-box and positive-ray face candidates."""
    denominator = math.lcm(*(Fraction(value).denominator for value in point))
    numerator = tuple(Fraction(value).numerator *
                      (denominator // Fraction(value).denominator) for value in point)
    boundary, forward = [], []
    stack = [prepared.aabb_root]
    while stack:
        lower, upper, indices, left, right = stack.pop()
        if any(numerator[axis] < lower[axis] * denominator or
               numerator[axis] > upper[axis] * denominator for axis in range(3)):
            continue
        if indices is not None:
            for index in indices:
                row_lower, row_upper = prepared.lower_bounds[index], prepared.upper_bounds[index]
                if all(row_lower[axis] * denominator <= numerator[axis] <=
                       row_upper[axis] * denominator for axis in range(3)):
                    boundary.append(index)
        else:
            stack.append(right)
            stack.append(left)

    if not include_forward:
        boundary.sort()
        return tuple(boundary), ()

    stack = [prepared.aabb_root]
    while stack:
        _lower, upper, indices, left, right = stack.pop()
        if any(upper[axis] * denominator < numerator[axis] for axis in range(3)):
            continue
        if indices is not None:
            for index in indices:
                if all(prepared.upper_bounds[index][axis] * denominator >= numerator[axis]
                       for axis in range(3)):
                    forward.append(index)
        else:
            stack.append(right)
            stack.append(left)
    boundary.sort()
    forward.sort()
    return tuple(boundary), tuple(forward)


def _ray_intersects_exact_aabb(point, direction, lower, upper):
    """Return whether an exact positive ray intersects a closed exact box."""
    entry, exit = Fraction(0), None
    for axis, component in enumerate(direction):
        first = (Fraction(lower[axis]) - point[axis]) / component
        last = (Fraction(upper[axis]) - point[axis]) / component
        if first > last:
            first, last = last, first
        if last < 0:
            return False
        entry = max(entry, first)
        exit = last if exit is None else min(exit, last)
        if entry > exit:
            return False
    return True


def _ray_aabb_candidate_indices(point, prepared, direction):
    """Return sorted triangle boxes intersected by an exact positive ray."""
    require(isinstance(prepared, PreparedPointLocation),
            "ray AABB query requires a PreparedPointLocation snapshot")
    point = _exact_rational_xyz(point, "ray AABB query")
    direction = tuple(direction)
    require(len(direction) == 3 and all(type(value) is int and value > 0 for value in direction),
            "ray AABB query requires an exact positive integer direction")
    stack, candidates = [prepared.aabb_root], []
    while stack:
        lower, upper, indices, left, right = stack.pop()
        if not _ray_intersects_exact_aabb(point, direction, lower, upper):
            continue
        if indices is None:
            stack.append(right)
            stack.append(left)
        else:
            for index in indices:
                if _ray_intersects_exact_aabb(point, direction,
                                              prepared.lower_bounds[index],
                                              prepared.upper_bounds[index]):
                    candidates.append(index)
    candidates.sort()
    return tuple(candidates)


@dataclass(frozen=True)
class PreparedPointLocation:
    """Immutable exact surface snapshot for repeated point-location queries.

    The first deterministic ray is prepared eagerly. Additional ray directions
    are cached only if an earlier ray is ambiguous, avoiding a 64-by-face
    Python-integer table for every target.
    """
    records: tuple
    triangles: tuple
    normals: tuple
    plane_constants: tuple
    lower_bounds: tuple
    upper_bounds: tuple
    normalization_transforms: tuple
    primary_denominators: tuple
    primary_parallel_indices: tuple
    record_identity_sha256: str
    aabb_root: tuple
    aabb_node_count: int
    _fallback_ray_data: dict


def _exact_rational_coordinate(value, context):
    if isinstance(value, bool):
        require(False, f"{context} must use exact integer or Fraction coordinates")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Fraction):
        return value
    require(False, f"{context} must use exact integer or Fraction coordinates")


def _exact_rational_xyz(value, context):
    try:
        coordinates = tuple(value)
    except TypeError as error:
        raise HumanImportError("cardiac cavity intersections: " + f"{context} must be an xyz triple") from error
    require(len(coordinates) == 3, f"{context} must be an xyz triple")
    return tuple(_exact_rational_coordinate(coordinate, context) for coordinate in coordinates)



def _update_identity_integer(hasher, value):
    value = int(value)
    magnitude = abs(value)
    encoded = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
    hasher.update(b"\x01" if value < 0 else b"\x00")
    hasher.update(struct.pack("<Q", len(encoded)))
    hasher.update(encoded)


def _point_location_records_identity_sha256(records):
    hasher = hashlib.sha256(b"HumanExactPointLocationRecords.v1\x00")
    hasher.update(struct.pack("<Q", len(records)))
    for triangle, _lower, _upper, face_id, ids in records:
        hasher.update(b"F")
        _update_identity_integer(hasher, face_id)
        for vertex_id in ids:
            hasher.update(b"I")
            _update_identity_integer(hasher, vertex_id)
        for point in triangle:
            hasher.update(b"P")
            for coordinate in point:
                rational = Fraction(coordinate)
                hasher.update(b"Q")
                _update_identity_integer(hasher, rational.numerator)
                _update_identity_integer(hasher, rational.denominator)
    return hasher.hexdigest()



def prepare_point_location(records):
    """Snapshot exact records and prepare per-triangle normalized queries."""
    require(isinstance(records, (list, tuple)) and bool(records),
            "prepared point-location requires non-empty records")
    require(len(records) <= 200_000, "prepared point-location surface exceeds the bounded face count")
    snapshots, triangles, normals = [], [], []
    plane_constants, lower_bounds, upper_bounds = [], [], []
    normalization_transforms = []
    for position, raw_row in enumerate(records):
        require(isinstance(raw_row, (list, tuple)) and len(raw_row) == 5,
                f"prepared point-location record {position} must have five fields")
        raw_triangle, raw_lower, raw_upper, raw_face_id, raw_ids = raw_row
        try:
            raw_points = tuple(raw_triangle)
        except TypeError as error:
            raise HumanImportError("cardiac cavity intersections: " +
                                   f"prepared point-location record {position} has no triangle") from error
        require(len(raw_points) == 3,
                f"prepared point-location record {position} triangle must have three points")
        triangle = tuple(_exact_rational_xyz(point, f"prepared point-location record {position} vertex")
                         for point in raw_points)
        lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
        upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
        require(_exact_rational_xyz(raw_lower, f"prepared point-location record {position} lower bound") == lower
                and _exact_rational_xyz(raw_upper, f"prepared point-location record {position} upper bound") == upper,
                f"prepared point-location record {position} bounds do not match its triangle")
        require(not isinstance(raw_face_id, bool) and isinstance(raw_face_id, Integral)
                and int(raw_face_id) >= 0,
                f"prepared point-location record {position} face ID is invalid")
        try:
            ids = tuple(raw_ids)
        except TypeError as error:
            raise HumanImportError("cardiac cavity intersections: " +
                                   f"prepared point-location record {position} vertex IDs are invalid") from error
        require(len(ids) == 3 and all(not isinstance(vertex_id, bool) and isinstance(vertex_id, Integral)
                                      and int(vertex_id) >= 0 for vertex_id in ids)
                and len({int(vertex_id) for vertex_id in ids}) == 3,
                f"prepared point-location record {position} vertex IDs are invalid")
        normalized, transform = _integer_lattice_normalization(triangle)
        normal = _cross(_sub(normalized[1], normalized[0]), _sub(normalized[2], normalized[0]))
        require(any(normal), f"prepared point-location record {position} is degenerate")
        snapshots.append((triangle, lower, upper, int(raw_face_id),
                          tuple(int(vertex_id) for vertex_id in ids)))
        triangles.append(normalized)
        normals.append(normal)
        plane_constants.append(_dot(normal, normalized[0]))
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        normalization_transforms.append(transform)

    primary_denominators = tuple(_dot(normal, (1, 1, 1)) for normal in normals)
    primary_parallel_indices = tuple(index for index, divisor in enumerate(primary_denominators)
                                     if divisor == 0)
    snapshot_tuple = tuple(snapshots)
    aabb_root, aabb_node_count = _prepare_exact_aabb_index(lower_bounds, upper_bounds)
    return PreparedPointLocation(
        records=snapshot_tuple,
        triangles=tuple(triangles),
        normals=tuple(normals),
        plane_constants=tuple(plane_constants),
        lower_bounds=tuple(lower_bounds),
        upper_bounds=tuple(upper_bounds),
        normalization_transforms=tuple(normalization_transforms),
        primary_denominators=primary_denominators,
        primary_parallel_indices=primary_parallel_indices,
        record_identity_sha256=_point_location_records_identity_sha256(snapshot_tuple),
        aabb_root=aabb_root,
        aabb_node_count=aabb_node_count,
        _fallback_ray_data={},
    )


def point_location_prepared_indexed(point, prepared):
    """Classify exactly using AABB candidates and normalized triangle arithmetic.

    The AABB test is inclusive for boundary candidates. Forward-ray candidates
    are conservative because every deterministic ray component is positive.
    All final plane and triangle-side tests use the same exact normalized
    predicates as point_location_prepared.
    """
    require(isinstance(prepared, PreparedPointLocation),
            "indexed point-location requires a PreparedPointLocation snapshot")
    point = _exact_rational_xyz(point, "indexed point-location query")
    boundary_indices, _ = _point_aabb_candidate_indices(
        point, prepared, include_forward=False)
    local_cache = {}

    def local_query(index):
        if index not in local_cache:
            local_cache[index] = _normalized_query_lattice(
                point, prepared.normalization_transforms[index])
        return local_cache[index]

    for index in boundary_indices:
        numerator, denominator = local_query(index)
        triangle = prepared.triangles[index]
        normal = prepared.normals[index]
        plane = _plane_numerator_at_local_query(numerator, denominator, triangle, normal)
        if plane == 0 and _inside(_signs(numerator, denominator, triangle, normal)):
            return {"location": "boundary", "ray": None, "crossings": None}

    for attempt in range(1, 65):
        direction = (1, attempt, attempt * attempt)
        divisors, parallel_indices = _prepared_ray_data(prepared, attempt)
        # Preserve the established retry rule: any surface plane containing
        # this ray makes the attempt ambiguous, even when that triangle's AABB
        # lies behind or away from the positive ray segment.
        if any(_plane_numerator_at_local_query(
                *local_query(index), prepared.triangles[index], prepared.normals[index]) == 0
               for index in parallel_indices):
            continue
        crossings, ambiguous = 0, False
        ray_indices = _ray_aabb_candidate_indices(point, prepared, direction)
        for index in ray_indices:
            numerator, query_denominator = local_query(index)
            triangle = prepared.triangles[index]
            normal = prepared.normals[index]
            plane = _plane_numerator_at_local_query(
                numerator, query_denominator, triangle, normal)
            divisor = _dot(normal, direction)
            if divisor == 0:
                if plane == 0:
                    ambiguous = True
                    break
                continue
            if plane * divisor <= 0:
                continue
            hit = tuple(numerator[axis] * divisor + plane * direction[axis]
                        for axis in range(3))
            hit_denominator = query_denominator * divisor
            if hit_denominator < 0:
                hit_denominator = -hit_denominator
                hit = tuple(-coordinate for coordinate in hit)
            signs = _signs(hit, hit_denominator, triangle, normal)
            if _inside(signs):
                if 0 in signs:
                    ambiguous = True
                    break
                crossings += 1
        if not ambiguous:
            return {"location": "inside" if crossings % 2 else "outside",
                    "ray": list(direction), "crossings": crossings, "attempts": attempt}
    return {"location": "indeterminate", "ray": None, "crossings": None, "attempts": 64}


def _prepared_ray_data(prepared, attempt):
    if attempt == 1:
        return prepared.primary_denominators, prepared.primary_parallel_indices
    cached = prepared._fallback_ray_data.get(attempt)
    if cached is not None:
        return cached
    direction = (1, attempt, attempt * attempt)
    denominators = tuple(_dot(normal, direction) for normal in prepared.normals)
    parallel_indices = tuple(index for index, denominator in enumerate(denominators)
                             if denominator == 0)
    result = (denominators, parallel_indices)
    # Concurrent first use of the same fallback ray can harmlessly compute the
    # same immutable tuple twice; setdefault keeps one canonical cached value.
    return prepared._fallback_ray_data.setdefault(attempt, result)


def point_location_prepared(point, prepared):
    """Classify one exact point using per-triangle normalized coordinates."""
    require(isinstance(prepared, PreparedPointLocation),
            "prepared point-location requires a PreparedPointLocation snapshot")
    point = _exact_rational_xyz(point, "prepared point-location query")
    local_cache = {}

    def local_query(index):
        if index not in local_cache:
            local_cache[index] = _normalized_query_lattice(
                point, prepared.normalization_transforms[index])
        return local_cache[index]

    for index, (lower, upper) in enumerate(zip(prepared.lower_bounds, prepared.upper_bounds)):
        if any(point[axis] < lower[axis] or point[axis] > upper[axis] for axis in range(3)):
            continue
        numerator, denominator = local_query(index)
        triangle = prepared.triangles[index]
        normal = prepared.normals[index]
        plane = _plane_numerator_at_local_query(numerator, denominator, triangle, normal)
        if plane == 0 and _inside(_signs(numerator, denominator, triangle, normal)):
            return {"location": "boundary", "ray": None, "crossings": None}

    forward_candidates = tuple(
        index for index, upper in enumerate(prepared.upper_bounds)
        if all(upper[axis] >= point[axis] for axis in range(3))
    )
    for attempt in range(1, 65):
        direction = (1, attempt, attempt * attempt)
        divisors, parallel_indices = _prepared_ray_data(prepared, attempt)
        if any(_plane_numerator_at_local_query(
                *local_query(index), prepared.triangles[index], prepared.normals[index]) == 0
               for index in parallel_indices):
            continue
        crossings = 0
        for index in forward_candidates:
            divisor = divisors[index]
            if divisor == 0:
                continue
            numerator, query_denominator = local_query(index)
            triangle = prepared.triangles[index]
            normal = prepared.normals[index]
            plane = _plane_numerator_at_local_query(
                numerator, query_denominator, triangle, normal)
            if plane * divisor <= 0:
                continue
            hit = tuple(numerator[axis] * divisor + plane * direction[axis]
                        for axis in range(3))
            hit_denominator = query_denominator * divisor
            if hit_denominator < 0:
                hit_denominator = -hit_denominator
                hit = tuple(-coordinate for coordinate in hit)
            signs = _signs(hit, hit_denominator, triangle, normal)
            if _inside(signs):
                if 0 in signs:
                    break
                crossings += 1
        else:
            return {"location": "inside" if crossings % 2 else "outside", "ray": list(direction),
                    "crossings": crossings, "attempts": attempt}
    return {"location": "indeterminate", "ray": None, "crossings": None, "attempts": 64}


@dataclass(frozen=True)
class PreparedSignedWinding:
    """Exact signed-winding query snapshot for a closed oriented triangle 2-cycle.

    The surface may self-intersect or have several closed components. Its
    combinatorial edge orientation and vertex links must still form closed
    oriented 2-manifolds. No inside/outside or material interpretation is made.
    """
    point_locator: PreparedPointLocation
    face_component_count: int
    record_identity_sha256: str


def _validate_closed_oriented_record_topology(records):
    """Validate a closed oriented 2-manifold using exact record vertex IDs."""
    require(bool(records), "signed winding requires non-empty records")
    vertex_points = {}
    edge_uses = {}
    vertex_links = {}
    face_keys = set()
    face_ids = set()
    face_adjacency = [set() for _ in records]

    for position, (triangle, _lower, _upper, face_id, ids) in enumerate(records):
        require(face_id not in face_ids, "signed winding record face IDs must be unique")
        face_ids.add(face_id)
        face_key = tuple(sorted(ids))
        require(face_key not in face_keys, "signed winding topology has duplicate faces")
        face_keys.add(face_key)
        for vertex_id, point in zip(ids, triangle):
            previous = vertex_points.setdefault(vertex_id, point)
            require(previous == point,
                    "signed winding vertex ID has inconsistent exact coordinates")
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge_uses.setdefault(tuple(sorted((a, b))), []).append((a, b, position))
        for local, vertex_id in enumerate(ids):
            a, b = ids[(local + 1) % 3], ids[(local + 2) % 3]
            key = tuple(sorted((a, b)))
            link = vertex_links.setdefault(vertex_id, {})
            link[key] = link.get(key, 0) + 1

    for uses in edge_uses.values():
        require(len(uses) == 2,
                "signed winding requires every topological edge to have exactly two incident faces")
        first, second = uses
        require(first[0] == second[1] and first[1] == second[0],
                "signed winding requires consistently oriented shared edges")
        face_adjacency[first[2]].add(second[2])
        face_adjacency[second[2]].add(first[2])

    for vertex_id, link_edges in vertex_links.items():
        require(all(count == 1 for count in link_edges.values()),
                f"signed winding vertex {vertex_id} has a repeated link edge")
        link_adjacency = {}
        for a, b in link_edges:
            link_adjacency.setdefault(a, set()).add(b)
            link_adjacency.setdefault(b, set()).add(a)
        require(bool(link_adjacency) and all(len(neighbors) == 2
                                             for neighbors in link_adjacency.values()),
                f"signed winding vertex {vertex_id} link is not a cycle")
        reached = set()
        pending = [min(link_adjacency)]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(link_adjacency[current] - reached)
        require(len(reached) == len(link_adjacency),
                f"signed winding vertex {vertex_id} link has multiple cycles")

    unseen = set(range(len(records)))
    component_count = 0
    while unseen:
        component_count += 1
        pending = [min(unseen)]
        while pending:
            face = pending.pop()
            if face not in unseen:
                continue
            unseen.remove(face)
            pending.extend(face_adjacency[face] & unseen)
    return component_count


def prepare_signed_winding(records):
    """Prepare exact signed winding for a closed oriented shell, including self-intersections.

    Every point and record coordinate remains an exact integer or Fraction.
    Open, nonmanifold, inconsistently oriented, duplicate-face, and degenerate
    input is rejected before any query can report a winding.
    """
    point_locator = prepare_point_location(records)
    component_count = _validate_closed_oriented_record_topology(point_locator.records)
    return PreparedSignedWinding(
        point_locator=point_locator,
        face_component_count=component_count,
        record_identity_sha256=point_locator.record_identity_sha256,
    )


def signed_winding_number(point, prepared):
    """Return exact signed winding using locally normalized triangle predicates."""
    require(isinstance(prepared, PreparedSignedWinding),
            "signed winding requires a PreparedSignedWinding snapshot")
    locator = prepared.point_locator
    point = _exact_rational_xyz(point, "signed-winding query")
    local_cache = {}

    def local_query(index):
        if index not in local_cache:
            local_cache[index] = _normalized_query_lattice(
                point, locator.normalization_transforms[index])
        return local_cache[index]

    def plane_numerator(index):
        numerator, denominator = local_query(index)
        return _plane_numerator_at_local_query(
            numerator, denominator, locator.triangles[index], locator.normals[index])

    for index, (lower, upper) in enumerate(zip(locator.lower_bounds, locator.upper_bounds)):
        if any(point[axis] < lower[axis] or point[axis] > upper[axis] for axis in range(3)):
            continue
        numerator, denominator = local_query(index)
        if plane_numerator(index) == 0 and _inside(
                _signs(numerator, denominator, locator.triangles[index], locator.normals[index])):
            return {"status": "boundary", "winding_number": None, "ray": None,
                    "crossings": None, "attempts": 0}

    forward_candidates = tuple(
        index for index, upper in enumerate(locator.upper_bounds)
        if all(upper[axis] >= point[axis] for axis in range(3))
    )
    for attempt in range(1, 65):
        direction = (1, attempt, attempt * attempt)
        divisors, parallel_indices = _prepared_ray_data(locator, attempt)
        if any(plane_numerator(index) == 0 for index in parallel_indices):
            continue
        winding = 0
        crossings = 0
        for index in forward_candidates:
            divisor = divisors[index]
            if divisor == 0:
                continue
            plane = plane_numerator(index)
            if plane * divisor <= 0:
                continue
            numerator, query_denominator = local_query(index)
            hit = tuple(numerator[axis] * divisor + plane * direction[axis]
                        for axis in range(3))
            hit_denominator = query_denominator * divisor
            if hit_denominator < 0:
                hit_denominator = -hit_denominator
                hit = tuple(-coordinate for coordinate in hit)
            signs = _signs(hit, hit_denominator, locator.triangles[index],
                           locator.normals[index])
            if _inside(signs):
                if 0 in signs:
                    break
                winding += 1 if divisor > 0 else -1
                crossings += 1
        else:
            return {"status": "resolved", "winding_number": winding,
                    "ray": list(direction), "crossings": crossings,
                    "attempts": attempt}
    return {"status": "indeterminate", "winding_number": None, "ray": None,
            "crossings": None, "attempts": 64}


def audit_cavity_intersections(cavities: dict) -> dict:
    """Audit quotient surfaces without trusting cached source topology claims."""
    require(isinstance(cavities, dict) and isinstance(cavities.get("chambers"), list) and
            1 <= len(cavities["chambers"]) <= 64, "invalid cavity collection")
    meshes, denominator = {}, 1
    for chamber in cavities["chambers"]:
        require(isinstance(chamber, dict) and isinstance(chamber.get("source_id"), str) and
                chamber["source_id"] and chamber["source_id"] not in meshes, "missing or duplicate surface ID")
        q = chamber.get("exact_coordinate_quotient")
        require(isinstance(q, dict), "missing exact-coordinate quotient")
        v, f = q.get("vertices_m"), q.get("triangles")
        require(isinstance(v, list) and 3 <= len(v) <= 20000 and isinstance(f, list) and 1 <= len(f) <= 40000,
                "empty or oversized mesh")
        require(all(isinstance(p, (list, tuple)) and len(p) == 3 and all(type(x) in (int, float) and
                    abs(x) <= 1e6 and math.isfinite(x) for x in p) for p in v), "invalid Float64 metre coordinates")
        topology = analyze_topology(v, f)
        rational = [tuple(Fraction.from_float(float(x)) for x in p) for p in v]
        for p in rational:
            for x in p:
                denominator = max(denominator, x.denominator)  # All denominators are powers of two.
        meshes[chamber["source_id"]] = (v, f, rational, topology)
    integer, records, surfaces = {}, {}, {}
    for name in sorted(meshes):
        v, f, rational, topology = meshes[name]
        points = [tuple(x.numerator*(denominator//x.denominator) for x in p) for p in rational]
        integer[name], records[name] = points, _records(points, f)
        report = _audit_pair(records[name], records[name], same_surface=True)
        # Containment samples vertex zero, so every published vertex must be
        # part of the connected surface. An unused point is not a domain witness.
        connected_closed = (topology["closed_oriented_manifold_candidate"]
                            and not topology["unused_vertex_ids"]
                            and topology["face_component_count"] == 1)
        report.update(closed_connected_oriented_manifold=connected_closed,
                      embedded_closed_surface=connected_closed and report["count"] == 0,
                      geometry_sha256=hashlib.sha256(canonical({"vertices_m": v, "triangles": f})).hexdigest())
        surfaces[name] = report
    pairs = []
    for first, second in itertools.combinations(sorted(meshes), 2):
        report = _audit_pair(records[first], records[second], same_surface=False)
        if not surfaces[first]["embedded_closed_surface"] or not surfaces[second]["embedded_closed_surface"]:
            containment = {"status": "not_checked_invalid_surface"}
        elif report["count"]:
            containment = {"status": "not_checked_intersecting_surfaces"}
        else:
            a = point_location(integer[first][0], records[second])
            b = point_location(integer[second][0], records[first])
            containment = {"status": "checked", "first_in_second": a, "second_in_first": b,
                           "representative_vertex_index": 0}
        disjoint = containment.get("status") == "checked" and all(
            containment[k]["location"] == "outside" for k in ("first_in_second", "second_in_first"))
        report.update(first=first, second=second, containment=containment, disjoint_closed_domains=disjoint)
        pairs.append(report)
    return {"schema": SCHEMA, "algorithm": ALGORITHM,
            "coordinate_semantics": "exact_rational_value_of_published_binary64_metres",
            "source_coordinates_modified": False, "added_faces": 0, "overlap_repair": False,
            "per_surface": surfaces, "per_pair": pairs,
            "all_surfaces_embedded": all(r["embedded_closed_surface"] for r in surfaces.values()),
            "all_domains_disjoint": all(r["embedded_closed_surface"] for r in surfaces.values()) and
                                    all(r["disjoint_closed_domains"] for r in pairs),
            "qualification": "geometric_intersection_audit_only"}
