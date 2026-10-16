"""CPU mathematical reference for source149 regional block-RAP action and transpose.

This models the regional outer-FGMRES preconditioner as a possible right map S.
It does not claim source149 CGLS currently uses regional RAP: its in-recurrence S
is the separate node-local block3 scale in base_cgls.metalinc.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Sequence

Vector = list[float]
Matrix = list[list[float]]

class ReferenceError(ValueError):
    pass

@dataclass(frozen=True)
class Identity:
    source_physics: str
    device_program: str
    world: str
    candidate_linearization: str
    microtick: int
    microtick_count: int
    micro_generation: int
    dt_f32_word: int

@dataclass(frozen=True)
class Generation:
    control_step: int
    physics_substep: int
    microtick: int
    microtick_count: int
    micro_generation: int
    newton_iteration: int
    dt_f32_word: int
    restart_cycle: int

    @property
    def linearization_key(self):
        return (self.control_step, self.physics_substep, self.microtick,
                self.microtick_count, self.micro_generation,
                self.newton_iteration, self.dt_f32_word)

@dataclass(frozen=True)
class Factor:
    lu: Matrix
    pivots: tuple[int, ...]  # final row permutation, as consumed by source solve_rap
    valid: bool

@dataclass(frozen=True)
class Variant:
    label: str
    basis_storage: Matrix      # n x mode_stride; xyz rows flattened, w omitted
    ap_storage: Matrix         # n x mode_stride
    mode_count: int
    mode_stride: int
    factor: Factor
    linearization: tuple[int, int, int, int, int, int, int]  # exact same-Newton capture key
    restart_generation: Optional[Generation] = None
    augmentation_valid: Optional[bool] = None

@dataclass(frozen=True)
class Capture:
    identity: Identity
    base: Variant
    augmented: Optional[Variant]
    augmentation_enabled: bool
    fine_inverse_rows: tuple[tuple[tuple[float, float, float, float], ...], ...]
    # Per node: three float4 rows. row0.w > 0.5 selects block; otherwise
    # source falls back to the componentwise inverseDiagonal entry.
    inverse_diagonal: tuple[tuple[float, float, float], ...]
    row_mask: tuple[tuple[int, int, int], ...]
    selected: bool = True
    healthy: bool = True


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(x)) for x in values)

def _check_matrix(a: Matrix, rows: int, cols: int, what: str) -> None:
    if len(a) != rows or any(len(row) != cols for row in a):
        raise ReferenceError(f"{what}: shape mismatch")
    if not _finite([x for row in a for x in row]):
        raise ReferenceError(f"{what}: nonfinite input")

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ReferenceError("dot: shape mismatch")
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))

def transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    return [[float(a[i][j]) for i in range(len(a))] for j in range(len(a[0]))]

def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    if not a or any(len(row) != len(x) for row in a):
        raise ReferenceError("matvec: shape mismatch")
    return [dot(row, x) for row in a]

def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or len(a[0]) != len(b):
        raise ReferenceError("matmul: shape mismatch")
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]

def max_abs(a: Matrix) -> float:
    return max((abs(float(x)) for row in a for x in row), default=0.0)

def factor_rap(matrix: Matrix) -> Factor:
    """Partial-pivoted LU with source149's final-permutation convention.

    Arithmetic is Python binary64 for the independent reference. Pivot choice,
    threshold, row swaps, and final permutation match source149's algorithm.
    """
    n = len(matrix)
    _check_matrix(matrix, n, n, "RAP")
    lu = [[float(x) for x in row] for row in matrix]
    piv = list(range(n))
    scale = max_abs(lu)
    floor = max(scale * 1.0e-10, 1.0e-30)
    if not (scale > 0.0 and math.isfinite(scale)):
        return Factor(lu, tuple(piv), False)
    for k in range(n):
        selected = k
        selected_mag = 0.0
        for r in range(k, n):
            mag = abs(lu[r][k])
            if mag > selected_mag:  # strict > retains first-row tie rule
                selected_mag, selected = mag, r
        if not (selected_mag > floor and math.isfinite(selected_mag)):
            return Factor(lu, tuple(piv), False)
        if selected != k:
            lu[k], lu[selected] = lu[selected], lu[k]
            piv[k], piv[selected] = piv[selected], piv[k]
        diag = lu[k][k]
        if not (abs(diag) > floor and math.isfinite(diag)):
            return Factor(lu, tuple(piv), False)
        for i in range(k + 1, n):
            lower = lu[i][k] / diag
            if not math.isfinite(lower):
                return Factor(lu, tuple(piv), False)
            lu[i][k] = lower
            for j in range(k + 1, n):
                lu[i][j] -= lower * lu[k][j]
                if not math.isfinite(lu[i][j]):
                    return Factor(lu, tuple(piv), False)
    return Factor(lu, tuple(piv), True)

def solve_rap(factor: Factor, rhs: Sequence[float]) -> Vector:
    """Forward/back solve with source149 `rhs[pivots[row]]` semantics."""
    n = len(factor.pivots)
    if len(rhs) != n or len(factor.lu) != n or any(len(r) != n for r in factor.lu):
        raise ReferenceError("RAP solve: shape mismatch")
    if not factor.valid:
        return [0.0] * n
    if sorted(factor.pivots) != list(range(n)):
        raise ReferenceError("RAP solve: pivot array is not a permutation")
    if not _finite(rhs) or not _finite([x for r in factor.lu for x in r]):
        raise ReferenceError("RAP solve: nonfinite input")
    y = [float(rhs[factor.pivots[i]]) for i in range(n)]
    for i in range(n):
        y[i] -= math.fsum(factor.lu[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        value = y[i] - math.fsum(factor.lu[i][j] * x[j] for j in range(i + 1, n))
        d = factor.lu[i][i]
        if not (abs(d) > 1.0e-30 and math.isfinite(d) and math.isfinite(value)):
            return [0.0] * n
        x[i] = value / d
        if not math.isfinite(x[i]):
            return [0.0] * n
    return x

def solve_rap_transpose(factor: Factor, rhs: Sequence[float]) -> Vector:
    """Solve R^T x=b from P R=L U without assuming R is symmetric.

    Source stores final row permutation `piv` with `P b = b[piv]`. Therefore
    R^T=U^T L^T P: solve U^T z=b, L^T w=z, then scatter `x[piv[i]]=w[i]`.
    """
    n = len(factor.pivots)
    if len(rhs) != n or len(factor.lu) != n or any(len(r) != n for r in factor.lu):
        raise ReferenceError("RAP transpose solve: shape mismatch")
    if not factor.valid:
        return [0.0] * n
    if sorted(factor.pivots) != list(range(n)):
        raise ReferenceError("RAP transpose solve: pivot array is not a permutation")
    if not _finite(rhs) or not _finite([x for r in factor.lu for x in r]):
        raise ReferenceError("RAP transpose solve: nonfinite input")
    z = [0.0] * n
    # U^T is lower triangular.
    for i in range(n):
        value = float(rhs[i]) - math.fsum(factor.lu[j][i] * z[j] for j in range(i))
        d = factor.lu[i][i]
        if not (abs(d) > 1.0e-30 and math.isfinite(d) and math.isfinite(value)):
            return [0.0] * n
        z[i] = value / d
    # L^T is upper triangular with unit diagonal.
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        w[i] = z[i] - math.fsum(factor.lu[j][i] * w[j] for j in range(i + 1, n))
    x = [0.0] * n
    for i, original_row in enumerate(factor.pivots):
        x[original_row] = w[i]
    return x

def make_fine_inverse(capture: Capture, use_block: bool) -> Matrix:
    """Expand source's nodal float4 block/fallback rows into exact B storage."""
    nodes = len(capture.inverse_diagonal)
    n = 3 * nodes
    if len(capture.fine_inverse_rows) != nodes or len(capture.row_mask) != nodes:
        raise ReferenceError("fine inverse/mask node count mismatch")
    B = [[0.0] * n for _ in range(n)]
    for node in range(nodes):
        mask = capture.row_mask[node]
        if len(mask) != 3 or any(m not in (0, 1) for m in mask):
            raise ReferenceError("row mask must be exact binary xyz")
        rows = capture.fine_inverse_rows[node]
        if len(rows) != 3 or any(len(row) != 4 for row in rows):
            raise ReferenceError("inverse block must contain three float4 rows")
        diag = capture.inverse_diagonal[node]
        if len(diag) != 3 or not _finite(diag):
            raise ReferenceError("invalid inverse diagonal")
        if use_block and float(rows[0][3]) > 0.5:
            if not _finite([x for row in rows for x in row]) or rows[1][3] != 0.0 or rows[2][3] != 0.0:
                raise ReferenceError("invalid marked inverse block")
            local = [[float(rows[i][j]) for j in range(3)] for i in range(3)]
        else:
            local = [[float(diag[i]) if i == j else 0.0 for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                # Fixed/eliminated rows are zeroed by the production builders.
                if mask[i] == 0 or mask[j] == 0:
                    if abs(local[i][j]) > 0.0:
                        raise ReferenceError("nonzero fine inverse on constrained row")
                B[3 * node + i][3 * node + j] = local[i][j]
    return B

def _validate_variant(variant: Variant, n: int) -> int:
    k, stride = variant.mode_count, variant.mode_stride
    if not (1 <= k <= 32 and k <= stride <= 32):
        raise ReferenceError("modeCount/modeStride outside source bounds")
    _check_matrix(variant.basis_storage, n, stride, "P storage")
    _check_matrix(variant.ap_storage, n, stride, "AP storage")
    if len(variant.factor.pivots) != k or len(variant.factor.lu) != k:
        raise ReferenceError("RAP factor shape does not match selected modeCount")
    if any(len(row) != k for row in variant.factor.lu):
        raise ReferenceError("RAP factor is not square")
    if variant.factor.valid and sorted(variant.factor.pivots) != list(range(k)):
        raise ReferenceError("valid RAP factor has invalid pivot permutation")
    return k

def select_variant(capture: Capture, expected_identity: Identity, expected_generation: Generation) -> Variant:
    """Select exactly the source149 base or same-restart augmented RAP state."""
    if capture.identity != expected_identity:
        raise ReferenceError("source/world/program/candidate identity mismatch")
    if not capture.selected:
        raise ReferenceError("environment is not selected")
    base = capture.base
    base_key = expected_generation.linearization_key
    if base.linearization != base_key:
        raise ReferenceError("base RAP belongs to another control/substep/Newton generation")
    if capture.augmentation_enabled and expected_generation.restart_cycle > 0:
        aug = capture.augmented
        if aug is None:
            raise ReferenceError("restart-augmented RAP requested but absent")
        if aug.restart_generation != expected_generation:
            raise ReferenceError("restart RAP generation mismatch; stale cycle is not admissible")
        if aug.linearization != base_key:
            raise ReferenceError("augmented RAP candidate linearization mismatch")
        if aug.augmentation_valid is None:
            raise ReferenceError("augmented RAP validity was not captured")
        if aug.mode_count != base.mode_count + 1 or aug.mode_stride != base.mode_stride:
            raise ReferenceError("augmented mode layout does not extend the captured base mode layout")
        if not aug.augmentation_valid:
            last = aug.mode_count - 1
            if any(row[last] != 0.0 for row in aug.basis_storage) or \
               any(row[last] != 0.0 for row in aug.ap_storage):
                raise ReferenceError("invalid augmentation retained nonzero P/AP words")
        return aug
    return base

def apply_regional(capture: Capture, rhs: Sequence[float], *,
                   expected_identity: Identity, expected_generation: Generation,
                   use_block: bool = True) -> Vector:
    """Apply production regional RAP/block-RAP map to one selected env vector."""
    variant = select_variant(capture, expected_identity, expected_generation)
    n = 3 * len(capture.row_mask)
    if len(rhs) != n or not _finite(rhs):
        raise ReferenceError("preconditioner input shape/nonfinite mismatch")
    if not capture.healthy:
        return [0.0] * n  # production state's diagnostics.z > .5 path
    k = _validate_variant(variant, n)
    P = [row[:k] for row in variant.basis_storage]
    AP = [row[:k] for row in variant.ap_storage]
    B = make_fine_inverse(capture, use_block=use_block)
    coarse_rhs = [math.fsum(P[i][j] * float(rhs[i]) for i in range(n)) for j in range(k)]
    coefficients = solve_rap(variant.factor, coarse_rhs)
    coarse_residual = matvec(AP, coefficients)
    coarse_correction = matvec(P, coefficients)
    residual = [float(rhs[i]) - coarse_residual[i] for i in range(n)]
    fine_correction = matvec(B, residual)
    return [fine_correction[i] + coarse_correction[i] for i in range(n)]

def apply_regional_transpose(capture: Capture, rhs: Sequence[float], *,
                             expected_identity: Identity, expected_generation: Generation,
                             use_block: bool = True) -> Vector:
    """Apply exact algebraic transpose of the selected regional preconditioner."""
    variant = select_variant(capture, expected_identity, expected_generation)
    n = 3 * len(capture.row_mask)
    if len(rhs) != n or not _finite(rhs):
        raise ReferenceError("transpose input shape/nonfinite mismatch")
    if not capture.healthy:
        return [0.0] * n
    k = _validate_variant(variant, n)
    P = [row[:k] for row in variant.basis_storage]
    AP = [row[:k] for row in variant.ap_storage]
    B = make_fine_inverse(capture, use_block=use_block)
    BT = transpose(B)
    fine_transpose = matvec(BT, rhs)
    pty = [math.fsum(P[i][j] * float(rhs[i]) for i in range(n)) for j in range(k)]
    apt_fine = [math.fsum(AP[i][j] * fine_transpose[i] for i in range(n)) for j in range(k)]
    coarse_rhs = [pty[j] - apt_fine[j] for j in range(k)]
    coefficients = solve_rap_transpose(variant.factor, coarse_rhs)
    coarse_transpose = matvec(P, coefficients)
    return [fine_transpose[i] + coarse_transpose[i] for i in range(n)]

def dense_regional_operator(capture: Capture, *, expected_identity: Identity,
                           expected_generation: Generation,
                           use_block: bool = True) -> Matrix:
    """Construct dense S from the reference action; use only for bounded tests."""
    n = 3 * len(capture.row_mask)
    cols = []
    for j in range(n):
        e = [0.0] * n; e[j] = 1.0
        cols.append(apply_regional(capture, e, expected_identity=expected_identity,
                                   expected_generation=expected_generation, use_block=use_block))
    return transpose(cols)
