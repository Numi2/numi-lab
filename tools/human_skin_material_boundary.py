"""Construction-only certification of conforming skin/fat tetrahedral boundaries.

No contact exclusions or correspondence are inferred from geometric proximity.
Faces use actual shared volume-node IDs; all orientation is outward from the
owning tetrahedron. The resulting internal interface is never an exterior
self-contact surface. Temporary construction facets keep admission false.
"""
from collections import defaultdict
import numpy as np

_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _indices(value, width, name, limit):
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != width or a.dtype.kind not in "iu":
        raise ValueError(f"{name}: integer Nx{width} array required")
    if np.any(a >= limit) or np.any(a < 0):
        raise ValueError(f"{name}: out-of-range node identity")
    if any(len(set(map(int, row))) != width for row in a):
        raise ValueError(f"{name}: repeated node identity")
    return a.astype(np.int64, copy=False)


def _parity(face):
    return -1 if sum(face[i] > face[j] for i in range(3) for j in range(i+1, 3)) % 2 else 1


def certify_material_boundaries(points, tetrahedra, material_ids, *,
                                 outer_faces, interface_faces, deep_faces,
                                 construction_faces=(), minimum_quality=1e-5):
    """Certify complete explicit O/I/D tables against a conforming volume.

    Skin material is 1, fat is 2. O and D are exterior oriented facets of their
    owning material. I is oriented outward from skin, and must have one skin
    and one fat incident tet with opposite orientations. Construction facets
    are diagnostic exterior closures, never native physical skin.
    """
    x = np.asarray(points, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite Float32 Nx3 points required")
    if not np.isfinite(minimum_quality) or not 0 < minimum_quality < 1:
        raise ValueError("invalid minimum quality")
    t = _indices(tetrahedra, 4, "tetrahedra", len(x))
    m = np.asarray(material_ids)
    if m.shape != (len(t),) or m.dtype.kind not in "iu" or not np.isin(m, [1, 2]).all():
        raise ValueError("one skin/fat material identity required per tetrahedron")
    if not len(t) or set(map(int, m)) != {1, 2}:
        raise ValueError("nonempty skin and fat volumes required")
    v = x[t].astype(np.float64)
    e = v[:, 1:] - v[:, :1]
    det = np.einsum("ij,ij->i", e[:, 0], np.cross(e[:, 1], e[:, 2]))
    longest = np.zeros(len(t))
    for i in range(4):
        for j in range(i+1, 4):
            longest = np.maximum(longest, np.linalg.norm(v[:, i]-v[:, j], axis=1))
    quality = det / np.maximum(longest**3, np.finfo(float).tiny)
    bad = np.flatnonzero(~np.isfinite(quality) | (quality < minimum_quality))
    if len(bad):
        row = int(bad[0])
        raise ValueError(f"invalid Float32 tetrahedron quality at row {row}: {quality[row]:.17g}")
    incidences = defaultdict(list)
    seen_tets = set()
    for row, tet in enumerate(t):
        key = tuple(sorted(map(int, tet)))
        if key in seen_tets:
            raise ValueError(f"duplicate tetrahedron at row {row}")
        seen_tets.add(key)
        for local in _LOCAL_FACES:
            face = tuple(int(tet[i]) for i in local)
            key = tuple(sorted(face))
            incidences[key].append((row, int(m[row]), face))
            if len(incidences[key]) > 2:
                raise ValueError(f"nonmanifold tetrahedral facet {key}")
    expected = {k: {} for k in ("outer", "interface", "deep")}
    exterior = {}
    for key, adjacent in sorted(incidences.items()):
        if len(adjacent) == 1:
            row, material, face = adjacent[0]
            exterior[key] = (row, material, face)
            expected["outer" if material == 1 else "deep"][key] = face
        else:
            a, b = adjacent
            if _parity(a[2]) == _parity(b[2]):
                raise ValueError(f"inconsistent internal facet orientation {key}")
            if a[1] != b[1]:
                expected["interface"][key] = a[2] if a[1] == 1 else b[2]
    c = np.asarray(construction_faces)
    if c.size == 0:
        c = np.empty((0, 3), dtype=np.int64)
    c = _indices(c, 3, "construction", len(x))
    construction_keys = set()
    for f in c:
        face = tuple(map(int, f)); key = tuple(sorted(face))
        if key in construction_keys or key not in exterior:
            raise ValueError(f"construction facet is duplicate or not exterior {key}")
        construction_keys.add(key)
        actual = exterior[key]
        if _parity(face) != _parity(actual[2]):
            raise ValueError(f"construction facet orientation differs {key}")
        del expected["outer" if actual[1] == 1 else "deep"][key]
    tables = {}
    for role, supplied in (("outer", outer_faces), ("interface", interface_faces),
                           ("deep", deep_faces)):
        f = _indices(supplied, 3, role, len(x))
        observed = {}
        for tri in f:
            face = tuple(map(int, tri)); key = tuple(sorted(face))
            if key in observed:
                raise ValueError(f"{role}: duplicate facet {key}")
            observed[key] = face
            if key not in expected[role]:
                raise ValueError(f"{role}: facet has wrong material, incidence or construction role {key}")
            if _parity(face) != _parity(expected[role][key]):
                raise ValueError(f"{role}: facet orientation differs {key}")
        missing = sorted(set(expected[role]) - set(observed))
        if missing:
            raise ValueError(f"{role}: missing facet {missing[0]}")
        tables[role] = np.asarray([observed[k] for k in sorted(observed)], dtype=np.int64).reshape(-1, 3)
    return {
        "outer_faces": tables["outer"], "interface_faces": tables["interface"],
        "deep_faces": tables["deep"], "construction_faces": c.copy(),
        "tetrahedron_count": len(t), "minimum_float32_quality": float(quality.min()),
        "material_volume_m3": {str(k): float(det[m == k].sum()/6) for k in (1, 2)},
        "construction_facet_count": len(c),
        "boundary_incidence_pass": True,
        "native_admission": False,  # geometry, mass, ancestry and motion gates remain mandatory
        "construction_only_facets_remaining": bool(len(c)),
    }

