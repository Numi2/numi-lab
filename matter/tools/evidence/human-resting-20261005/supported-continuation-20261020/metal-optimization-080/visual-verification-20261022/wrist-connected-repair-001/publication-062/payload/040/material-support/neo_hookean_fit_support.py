"""Vectorized compressible Neo-Hookean element energy and stress derivatives.

All coordinates and translations are metres; material parameters and Piola
stress are pascals. Tetrahedron corner columns identify shared O/I/D
translation degrees of freedom. No optimizer or native solver is run here.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import numpy as np

MATERIALS_PA = {
    1: (13255.0336, 658333.333),   # skin
    2: (551.724138, 5333.33333),   # fat
}
MAX_ABS_PIOLA_PA = 5.0e6
_REFERENCE_CACHE: OrderedDict[str, "NeoHookeanReference"] = OrderedDict()
_REFERENCE_CACHE_LIMIT = 8

@dataclass(frozen=True)
class NeoHookeanReference:
    """Validated reference tetrahedra; arrays are immutable and SI-scaled."""
    sha256: str
    invDm_per_m: np.ndarray       # [tet,3,3], reference edge matrix inverse
    volume_m3: np.ndarray         # [tet], positive reference volume
    shape_gradients_per_m: np.ndarray  # [tet,4,3]
    determinant_Dm_m3: np.ndarray      # [tet], positive oriented determinant


def _array_hash(x: np.ndarray) -> str:
    y = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    h = hashlib.sha256()
    h.update(b"neo-hookean-reference-tetrahedra-f64-metres-v1\\0")
    h.update(np.asarray(y.shape, dtype=np.int64).tobytes())
    h.update(y.tobytes())
    return h.hexdigest()


def prepare_reference_tetrahedra(reference_vertices_m: np.ndarray) -> NeoHookeanReference:
    """Build/cache inv(Dm), reference volumes and P1 gradients for exact input.

    `reference_vertices_m` is [tet,4,3] with positively oriented corners.
    Degenerate, inverted, nonfinite, or incorrectly-shaped reference cells fail
    before they can enter an optimizer.
    """
    R = np.asarray(reference_vertices_m, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (4, 3) or len(R) == 0 or not np.all(np.isfinite(R)):
        raise ValueError("reference vertices must be finite [tet,4,3] metres")
    key = _array_hash(R)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        _REFERENCE_CACHE.move_to_end(key)
        return cached
    Dm = np.stack((R[:, 1] - R[:, 0], R[:, 2] - R[:, 0], R[:, 3] - R[:, 0]), axis=2)
    det = np.linalg.det(Dm)
    bad = np.flatnonzero(~np.isfinite(det) | (det <= 1.0e-30))
    if len(bad):
        raise ValueError(f"reference tetrahedra must have positive nondegenerate orientation; first rows {bad[:8].tolist()}")
    invDm = np.linalg.inv(Dm)
    if not np.all(np.isfinite(invDm)):
        raise ValueError("reference inverse edge matrix is nonfinite")
    # F = Ds inv(Dm); grad(N1..N3) are rows of inv(Dm), grad(N0) closes sum.
    grad = np.empty((len(R), 4, 3), dtype=np.float64)
    grad[:, 1:, :] = invDm
    grad[:, 0, :] = -np.sum(invDm, axis=1)
    arrays = [invDm, det / 6.0, grad, det]
    for a in arrays:
        a.setflags(write=False)
    out = NeoHookeanReference(key, *arrays)
    _REFERENCE_CACHE[key] = out
    _REFERENCE_CACHE.move_to_end(key)
    while len(_REFERENCE_CACHE) > _REFERENCE_CACHE_LIMIT:
        _REFERENCE_CACHE.popitem(last=False)
    return out


def _compact_column_groups(corner_columns: np.ndarray):
    C = np.asarray(corner_columns)
    if C.ndim != 2 or C.shape[1] != 4 or C.dtype.kind not in "iu" or np.any(C < 0):
        raise ValueError("corner_columns must be nonnegative integer [tet,4]")
    C = C.astype(np.int64, copy=False)
    sorted_c = np.sort(C, axis=1)
    is_new = np.ones_like(sorted_c, dtype=bool)
    is_new[:, 1:] = sorted_c[:, 1:] != sorted_c[:, :-1]
    compact_ids = np.full_like(sorted_c, -1)
    rows = np.arange(len(C))
    sorted_slots = np.cumsum(is_new, axis=1) - 1
    for j in range(4):
        active = is_new[:, j]
        compact_ids[rows[active], sorted_slots[active, j]] = sorted_c[active, j]
    slots = np.zeros_like(C)
    for k in range(4):
        # Find each original corner in its row's sorted unique column list.
        pos = np.argmax(sorted_c == C[:, k, None], axis=1)
        slots[:, k] = np.cumsum(is_new, axis=1)[rows, pos] - 1
    return compact_ids, slots


def evaluate_neo_hookean_batch(
    current_vertices_m: np.ndarray,
    reference: NeoHookeanReference,
    material_ids: np.ndarray,
    corner_columns: np.ndarray,
    *,
    max_abs_piola_pa: float = MAX_ABS_PIOLA_PA,
) -> dict[str, np.ndarray | str]:
    """Evaluate energies, Piola stresses, and analytic shared-column Jacobians.

    Inputs are current vertices [tet,4,3] in metres, a cache returned by
    `prepare_reference_tetrahedra`, material IDs (1 skin, 2 fat), and each
    corner's shared translation-column ID. Duplicate column IDs within a cell
    are summed before returning derivatives.

    The returned per-cell Jacobians are local and sparse-by-construction:
    `jacobian_column_ids[tet,slot]` names up to four unique columns, with -1
    padding; energy Jacobian is [tet,slot,xyz], and Piola Jacobian is
    [tet,9,slot,xyz] in row-major signed P-component order. Energy is per
    reference cell in joules; P is pascals; Jacobians are respectively J/m and
    Pa/m. `stress_constraint_residual_pa` is [P_i-limit, -P_i-limit] per each
    of the nine signed components; feasible rows are <= 0.

    Invalid/nonpositive current J cells have `valid=False` and finite zero
    energy/stress/Jacobians. Callers must reject/mask those rows explicitly;
    they are never silently treated as accepted optimizer constraints.
    """
    X = np.asarray(current_vertices_m, dtype=np.float64)
    if X.ndim != 3 or X.shape != (len(reference.volume_m3), 4, 3) or not np.all(np.isfinite(X)):
        raise ValueError("current vertices must be finite [cached_tet,4,3] metres")
    mid = np.asarray(material_ids)
    if mid.shape != (len(X),) or mid.dtype.kind not in "iu":
        raise ValueError("material_ids must be integer [tet] (1=skin, 2=fat)")
    mid = mid.astype(np.int64, copy=False)
    if np.any((mid != 1) & (mid != 2)):
        raise ValueError("unsupported material ID; expected 1 skin or 2 fat")
    if not np.isfinite(max_abs_piola_pa) or max_abs_piola_pa <= 0:
        raise ValueError("max_abs_piola_pa must be finite and positive")
    compact_ids, slots = _compact_column_groups(corner_columns)
    if compact_ids.shape[0] != len(X):
        raise ValueError("corner column count differs from tetrahedron count")

    Dm_s = np.stack((X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2)
    F = Dm_s @ reference.invDm_per_m
    J = np.linalg.det(F)
    valid = np.isfinite(J) & (J > 0.0) & np.all(np.isfinite(F), axis=(1, 2))
    n = len(X)
    W = np.zeros(n, dtype=np.float64)
    energy = np.zeros(n, dtype=np.float64)
    P = np.zeros((n, 3, 3), dtype=np.float64)
    energy_jac = np.zeros((n, 4, 3), dtype=np.float64)
    piola_jac = np.zeros((n, 9, 4, 3), dtype=np.float64)

    mu_by_id = {k: MATERIALS_PA[k][0] for k in MATERIALS_PA}
    bulk_by_id = {k: MATERIALS_PA[k][1] for k in MATERIALS_PA}
    mu = np.asarray([mu_by_id[int(k)] for k in mid], dtype=np.float64)
    bulk = np.asarray([bulk_by_id[int(k)] for k in mid], dtype=np.float64)
    idx = np.flatnonzero(valid)
    if len(idx):
        try:
            Finv = np.linalg.inv(F[idx])
            inv_ok = np.all(np.isfinite(Finv), axis=(1, 2))
        except np.linalg.LinAlgError:
            # Rare numerically singular positive-determinant cells are isolated
            # so one bad cell cannot poison the remainder of a local batch.
            Finv = np.zeros_like(F[idx]); inv_ok = np.zeros(len(idx), dtype=bool)
            for j, row in enumerate(idx):
                try:
                    candidate = np.linalg.inv(F[row])
                    if np.all(np.isfinite(candidate)):
                        Finv[j] = candidate; inv_ok[j] = True
                except np.linalg.LinAlgError:
                    pass
        valid[idx[~inv_ok]] = False
        idx = idx[inv_ok]
        Finv = Finv[inv_ok]
    if len(idx):
        Jv = J[idx]
        logJ = np.log(Jv)
        Iv = np.einsum("nij,nij->n", F[idx], F[idx])
        Wv = 0.5 * mu[idx] * (Iv - 3.0 - 2.0 * logJ) + 0.5 * bulk[idx] * logJ * logJ
        FinvT = Finv.transpose(0, 2, 1)
        Pv = mu[idx, None, None] * F[idx] + (bulk[idx] * logJ - mu[idx])[:, None, None] * FinvT
        W[idx] = Wv
        energy[idx] = Wv * reference.volume_m3[idx]
        P[idx] = Pv

        # Sum P1 shape gradients for corners sharing one translation column.
        grad = reference.shape_gradients_per_m[idx]
        ids_local = compact_ids[idx]
        slots_local = slots[idx]
        grouped_grad = np.zeros((len(idx), 4, 3), dtype=np.float64)
        rlocal = np.arange(len(idx))
        for corner in range(4):
            np.add.at(grouped_grad, (rlocal, slots_local[:, corner]), grad[:, corner, :])
        energy_jac[idx] = reference.volume_m3[idx, None, None] * np.einsum("nij,nkj->nki", Pv, grouped_grad)

        # dF/d(delta_{column,axis}) = e_axis outer summed_shape_gradient.
        dF = np.einsum("ai,nkj->nkaij", np.eye(3), grouped_grad)
        trace = np.einsum("nij,nkaji->nka", Finv, dF)
        dFt = dF.swapaxes(-1, -2)
        left = FinvT[:, None, None, :, :]
        dPv = (mu[idx, None, None, None, None] * dF
               + bulk[idx, None, None, None, None] * trace[..., None, None] * left
               - (bulk[idx] * logJ - mu[idx])[:, None, None, None, None] * (left @ dFt @ left))
        piola_jac[idx] = dPv.transpose(0, 3, 4, 1, 2).reshape(len(idx), 9, 4, 3)

    components = P.reshape(n, 9)
    signed = np.stack((components - max_abs_piola_pa, -components - max_abs_piola_pa), axis=2).reshape(n, 18)
    signed_jac = np.stack((piola_jac, -piola_jac), axis=2).reshape(n, 18, 4, 3)
    return {
        "valid": valid,
        "J": J,
        "energy_density_pa": W,
        "energy_joule": energy,
        "piola_pa": P,
        "piola_components_pa": components,
        "jacobian_column_ids": compact_ids,
        "energy_jacobian_j_per_m": energy_jac,
        "piola_component_jacobian_pa_per_m": piola_jac,
        "stress_constraint_residual_pa": signed,
        "stress_constraint_jacobian_pa_per_m": signed_jac,
        "stress_constraint_feasible": valid[:, None] & (signed <= 0.0),
        "reference_sha256": reference.sha256,
    }
