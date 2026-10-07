#include <metal_stdlib>

#include "metalrobo/numi_human_joint_equality_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_stand_cpu_finish_gpu.h"
#include "metalrobo/numi_human_constraint_projection.h"
#include "metalrobo/numi_human_friction.h"
#include "metalrobo/numi_human_bilateral.h"
#include "metalrobo/numi_human_passive_joint.h"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

using namespace metal;

namespace {

constant float kPivotFloor = 1.0e-10f;
constant float kResponseRegularization = 1.0e-7f;
// 16 KiB factor cache leaves room for the three 160-entry work vectors on
// devices with 32 KiB threadgroup memory. Larger authored blocks retain the
// same device factor path; the supported equality count is unchanged.
constant uint kCachedEqualityCapacity = 64u;
constant uint kParallelContactConditionFailure = 0x40000000u;
constant uint kParallelLimitConditionFailure = 0x80000000u;
// The single-Human CPU finish has already performed every ordered sweep.
// Specializing this path removes the unused GPU constraint solver from its
// completion pipeline while the ordinary split stand keeps that solver.
constant bool kCpuFinishSpecialized [[function_constant(0)]];
constant bool kCachedLimitEqualityResponse [[function_constant(1)]];
constant bool kUseCachedLimitEqualityResponse =
    is_function_constant_defined(kCachedLimitEqualityResponse)
        ? kCachedLimitEqualityResponse : false;
constant bool kSparseStandOperator [[function_constant(2)]];
constant bool kUseSparseStandOperator =
    is_function_constant_defined(kSparseStandOperator)
        ? kSparseStandOperator : false;
constant bool kStandFactorOnlySpecialized [[function_constant(3)]];
constant bool kUseStandFactorOnlySpecialized =
    is_function_constant_defined(kStandFactorOnlySpecialized)
        ? kStandFactorOnlySpecialized : false;
// Read-only, opt-in work attribution. Ordinary finish specializations remove
// every counter and its output argument; no physical state uses these values.
constant bool kFinishWorkCounters [[function_constant(4)]];
constant bool kUseFinishWorkCounters =
    is_function_constant_defined(kFinishWorkCounters)
        ? kFinishWorkCounters : false;
// Experimental factor-only path for the small SPD R^T A R block. Undefined
// and false preserve the existing pivoted bilateral factor exactly.
constant bool kReducedStandCholesky [[function_constant(5)]];
constant bool kUseReducedStandCholesky =
    is_function_constant_defined(kReducedStandCholesky)
        ? kReducedStandCholesky : false;
// Experimental contact-only single-SIMD family sweep. The host specializes
// every finish pipeline explicitly; undefined/false keep the cooperative
// 128-thread contact loop unchanged.
constant bool kFirstSimdContactSweep [[function_constant(6)]];
constant bool kUseFirstSimdContactSweep =
    is_function_constant_defined(kFirstSimdContactSweep)
        ? kFirstSimdContactSweep : false;
// Experimental replacement for the repeated SIMD ballot preview scans.
// Undefined/false preserve the existing preview-and-select path.
constant bool kOnePassOrderedLimits [[function_constant(7)]];
constant bool kUseOnePassOrderedLimits =
    is_function_constant_defined(kOnePassOrderedLimits)
        ? kOnePassOrderedLimits : false;
// Optional speculative-contact admission broadening. The source contact
// activation slop remains the warm-start threshold; this value only admits
// existing source witnesses to the response and finish paths.
constant float kSpeculativeContactAdmissionDistanceMeters
    [[function_constant(8)]];
constant float kUseSpeculativeContactAdmissionDistanceMeters =
    is_function_constant_defined(kSpeculativeContactAdmissionDistanceMeters)
        ? kSpeculativeContactAdmissionDistanceMeters : 0.0f;
// Default-off diagnostic scheduling experiment. It changes only where the
// existing source-limit/equality multiplier and work evidence is evaluated;
// the coupled candidate-velocity PGS operations remain in authored order.
constant bool kDeferredStandEqualityDiagnostics
    [[function_constant(9)]];
constant bool kUseDeferredStandEqualityDiagnostics =
    is_function_constant_defined(kDeferredStandEqualityDiagnostics)
        ? kDeferredStandEqualityDiagnostics : false;
// Optional hybrid LU workspace for medium-sized equality blocks. False
// keeps the existing threadgroup/device workspace selection.
constant bool kHybridEqualityFactorCache [[function_constant(10)]];
constant bool kUseHybridEqualityFactorCache =
    is_function_constant_defined(kHybridEqualityFactorCache)
        ? kHybridEqualityFactorCache : false;
// Opt-in read-only per-limit-column attribution. Undefined/false leaves the
// ordinary projected-response pipeline and argument layout unchanged.
constant bool kReducedResponseDiagnostics [[function_constant(11)]];
constant bool kUseReducedResponseDiagnostics =
    is_function_constant_defined(kReducedResponseDiagnostics)
        ? kReducedResponseDiagnostics : false;
inline float standContactAdmissionDistanceMeters(
    const MRNumiHumanStandContactGPU support
) {
    return kUseSpeculativeContactAdmissionDistanceMeters > 0.0f
        ? max(support.frictionSlopAndStabilization.y,
            kUseSpeculativeContactAdmissionDistanceMeters)
        : support.frictionSlopAndStabilization.y;
}

inline uint standLegacyResponseStride(
    const uint nv, const uint contactCount, const uint equalityCount
) {
    const uint responseColumns =
        (3u * contactCount + equalityCount + nv) * nv;
    return responseColumns + equalityCount * (equalityCount + 3u) +
        nv * equalityCount;
}

inline uint standReducedResponseWorkspaceElements(
    const uint nv, const uint equalityCount
) {
    if (equalityCount > nv) return 0u;
    const uint freeDofs = nv - equalityCount;
    // Source A, reduced factor, scale/pivot vectors, coordinate map,
    // coefficients, grouped member offsets/indices, and readiness word.
    return nv * nv + freeDofs * freeDofs + 3u * freeDofs +
        3u * nv + 2u;
}

inline uint standResponseStride(
    constant const MRNumiHumanStandDispatchGPU& dispatch, const uint nv
) {
    const uint legacy = standLegacyResponseStride(
        nv, dispatch.supportContactCount, dispatch.jointEqualityCount);
    return (dispatch.flags &
            MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES) != 0u
        ? legacy + standReducedResponseWorkspaceElements(
            nv, dispatch.jointEqualityCount)
        : legacy;
}

// The dense factor reads the source lower triangle. Sparse Cholesky reads the
// source upper triangle before writing its lower factor. Mirror the exact
// selected source triangle when constructing the reduced operator or reactions.
inline float standSelectedSourceA(
    device const float* sourceA,
    const uint nv,
    const uint row,
    const uint column,
    const bool upperTriangle
) {
    const uint storedRow = upperTriangle ? min(row, column) : max(row, column);
    const uint storedColumn = upperTriangle ? max(row, column) : min(row, column);
    return sourceA[storedRow * nv + storedColumn];
}

struct MRStandFinishWorkCounters {
    uint sweeps, contactDecisions, contactContractions;
    uint normalChanges, tangentChanges, zeroContactChanges;
    uint frictionInterior, frictionBoundary, frictionBoundaryIterations;
    uint limitPreviews, limitPreviewBlocks, limitNonzero, limitSelectedZero;
    uint contactResponseCoefficients, limitResponseCoefficients;
    uint equalityResponseCoefficients, onePassLimitRoutes;
};

// Forward substitution keeps each row's original increasing-column FMA
// sequence. Completed blocks update independent future rows in parallel;
// lane zero resolves dependencies within each block and performs the original
// ordered backward substitution. All lanes enter every barrier together.
template<typename FactorPointer>
inline bool mrNumiHumanBilateralSolveCooperative(
    FactorPointer factor,
    device const float* inverseScale,
    device const float* pivots,
    threadgroup float* rhs,
    const uint n,
    const uint lane,
    const uint threadCount,
    threadgroup uint* succeeded
) {
    if (lane == 0u) {
        *succeeded = n != 0u && n <= MR_NUMI_HUMAN_STAND_MAX_DOFS ? 1u : 0u;
        for (uint i = 0u; i < n && *succeeded != 0u; ++i) {
            if (!(inverseScale[i] > 0.0f) ||
                !isfinite(inverseScale[i])) {
                *succeeded = 0u;
                break;
            }
            rhs[i] *= inverseScale[i];
            if (!isfinite(rhs[i])) *succeeded = 0u;
        }
        for (uint k = 0u; k < n && *succeeded != 0u; ++k) {
            const float pivotValue = pivots[k];
            if (!isfinite(pivotValue) ||
                pivotValue < float(k) || pivotValue >= float(n)) {
                *succeeded = 0u;
                break;
            }
            const uint pivot = uint(pivotValue);
            if (pivotValue != float(pivot)) {
                *succeeded = 0u;
                break;
            }
            const float value = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*succeeded == 0u) return false;

    constexpr uint blockSize = 4u;
    for (uint begin = 0u; begin < n; begin += blockSize) {
        const uint end = min(begin + blockSize, n);
        if (lane == 0u) {
            for (uint i = begin; i < end; ++i) {
                float value = rhs[i];
                for (uint j = begin; j < i; ++j)
                    value = mrNHBilateralFma(
                        -factor[i * n + j], rhs[j], value);
                rhs[i] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = end + lane; i < n; i += threadCount) {
            float value = rhs[i];
            for (uint j = begin; j < end; ++j)
                value = mrNHBilateralFma(
                    -factor[i * n + j], rhs[j], value);
            rhs[i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        for (uint reverse = 0u; reverse < n; ++reverse) {
            const uint i = n - 1u - reverse;
            float value = rhs[i];
            for (uint j = i + 1u; j < n; ++j)
                value = mrNHBilateralFma(
                    -factor[i * n + j], rhs[j], value);
            const float pivot = factor[i * n + i];
            if (pivot == 0.0f || !isfinite(pivot)) {
                *succeeded = 0u;
                break;
            }
            rhs[i] = value / pivot;
            if (!isfinite(rhs[i])) {
                *succeeded = 0u;
                break;
            }
        }
        for (uint i = 0u; i < n && *succeeded != 0u; ++i) {
            rhs[i] *= inverseScale[i];
            if (!isfinite(rhs[i])) *succeeded = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *succeeded != 0u;
}

// Route hybrid indices through the existing device factor allocation only
// after the threadgroup-cached prefix. The false specialization directly
// accesses the original single workspace.
template<bool hybridWorkspace, typename FactorPointer>
inline float mrNumiHumanBilateralFactorRead(
    FactorPointer factorCache,
    device float* hybridBacking,
    const uint index
) {
    if (hybridWorkspace &&
        index >= MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_ELEMENTS)
        return hybridBacking[index];
    return factorCache[index];
}

template<bool hybridWorkspace, typename FactorPointer>
inline void mrNumiHumanBilateralFactorWrite(
    FactorPointer factorCache,
    device float* hybridBacking,
    const uint index,
    const float value
) {
    if (hybridWorkspace &&
        index >= MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_ELEMENTS)
        hybridBacking[index] = value;
    else
        factorCache[index] = value;
}

// The pivot search and each row's arithmetic stay in their original order.
// Independent scaling, swaps, and elimination rows run across one SIMD group.
// The optional hybrid specialization keeps a contiguous prefix in threadgroup
// memory and stores the remainder in the existing device factor allocation.
template<bool deviceWorkspace, bool hybridWorkspace = false,
         typename FactorPointer>
inline bool mrNumiHumanBilateralFactorCooperativeWorkspace(
    device float* matrix,
    device float* inverseScale,
    device float* pivots,
    const uint n,
    const uint lane,
    const uint threadCount,
    FactorPointer factorCache,
    device float* hybridBacking,
    threadgroup float* scaleCache,
    threadgroup float* pivotCache,
    threadgroup atomic_uint* failure,
    threadgroup uint* selectedPivot
) {
    constexpr mem_flags workspaceFence = deviceWorkspace || hybridWorkspace
        ? mem_flags::mem_device | mem_flags::mem_threadgroup
        : mem_flags::mem_threadgroup;
    if (lane == 0u)
        atomic_store_explicit(failure, MR_INVALID_INDEX,
                              memory_order_relaxed);
    if (!deviceWorkspace) {
        const uint cacheElements = hybridWorkspace
            ? min(n * n,
                uint(MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_ELEMENTS))
            : n * n;
        for (uint index = lane; index < cacheElements;
             index += threadCount)
            factorCache[index] = matrix[index];
    }
    threadgroup_barrier(workspaceFence);

    for (uint i = lane; i < n; i += threadCount) {
        const float diagonal = mrNumiHumanBilateralFactorRead<hybridWorkspace>(
            factorCache, hybridBacking, i * n + i);
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            atomic_fetch_min_explicit(failure, i, memory_order_relaxed);
            continue;
        }
        scaleCache[i] = 1.0f / mrNHBilateralSqrt(diagonal);
        if (!isfinite(scaleCache[i]))
            atomic_fetch_min_explicit(failure, i, memory_order_relaxed);
    }
    threadgroup_barrier(workspaceFence);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    for (uint index = lane; index < n * n; index += threadCount) {
        const uint i = index / n;
        const uint j = index - i * n;
        const float value =
            (mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                 factorCache, hybridBacking, index) * scaleCache[i]) *
            scaleCache[j];
        mrNumiHumanBilateralFactorWrite<hybridWorkspace>(
            factorCache, hybridBacking, index, value);
        if (!isfinite(value))
            atomic_fetch_min_explicit(failure, index,
                                      memory_order_relaxed);
    }
    threadgroup_barrier(workspaceFence);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    for (uint k = 0u; k < n; ++k) {
        if (lane == 0u) {
            uint pivot = k;
            float largest = mrNHBilateralAbs(
                mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                    factorCache, hybridBacking, k * n + k));
            for (uint i = k + 1u; i < n; ++i) {
                const float value = mrNHBilateralAbs(
                    mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, i * n + k));
                if (value > largest) { largest = value; pivot = i; }
            }
            if (!(largest > 0.0f) || !isfinite(largest))
                atomic_fetch_min_explicit(failure, k,
                                          memory_order_relaxed);
            else {
                pivotCache[k] = float(pivot);
                *selectedPivot = pivot;
            }
        }
        threadgroup_barrier(workspaceFence);
        if (atomic_load_explicit(failure, memory_order_relaxed) !=
            MR_INVALID_INDEX) return false;

        if (*selectedPivot != k) {
            for (uint j = lane; j < n; j += threadCount) {
                const uint rowEntry = k * n + j;
                const uint pivotEntry = *selectedPivot * n + j;
                const float value =
                    mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, rowEntry);
                const float pivotValue =
                    mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, pivotEntry);
                mrNumiHumanBilateralFactorWrite<hybridWorkspace>(
                    factorCache, hybridBacking, rowEntry, pivotValue);
                mrNumiHumanBilateralFactorWrite<hybridWorkspace>(
                    factorCache, hybridBacking, pivotEntry, value);
            }
        }
        threadgroup_barrier(workspaceFence);
        for (uint i = k + 1u + lane; i < n; i += threadCount) {
            const uint columnEntry = i * n + k;
            const float pivot =
                mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                    factorCache, hybridBacking, k * n + k);
            const float value =
                mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                    factorCache, hybridBacking, columnEntry) / pivot;
            mrNumiHumanBilateralFactorWrite<hybridWorkspace>(
                factorCache, hybridBacking, columnEntry, value);
            if (!isfinite(value)) {
                atomic_fetch_min_explicit(failure, i,
                                          memory_order_relaxed);
                continue;
            }
            for (uint j = k + 1u; j < n; ++j) {
                const uint entry = i * n + j;
                const float value = mrNHBilateralFma(
                    -mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, columnEntry),
                    mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, k * n + j),
                    mrNumiHumanBilateralFactorRead<hybridWorkspace>(
                        factorCache, hybridBacking, entry));
                mrNumiHumanBilateralFactorWrite<hybridWorkspace>(
                    factorCache, hybridBacking, entry, value);
                if (!isfinite(value))
                    atomic_fetch_min_explicit(failure, i,
                                              memory_order_relaxed);
            }
        }
        threadgroup_barrier(workspaceFence);
        if (atomic_load_explicit(failure, memory_order_relaxed) !=
            MR_INVALID_INDEX) return false;
    }

    if (!deviceWorkspace) {
        const uint copyElements = hybridWorkspace
            ? min(n * n,
                uint(MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_ELEMENTS))
            : n * n;
        for (uint index = lane; index < copyElements;
             index += threadCount)
            matrix[index] = factorCache[index];
    }
    for (uint index = lane; index < n; index += threadCount) {
        inverseScale[index] = scaleCache[index];
        pivots[index] = pivotCache[index];
    }
    return true;
}

// Keep existing callers on the original single-workspace specialization.
template<bool deviceWorkspace, typename FactorPointer>
inline bool mrNumiHumanBilateralFactorCooperative(
    device float* matrix,
    device float* inverseScale,
    device float* pivots,
    const uint n,
    const uint lane,
    const uint threadCount,
    FactorPointer factorCache,
    threadgroup float* scaleCache,
    threadgroup float* pivotCache,
    threadgroup atomic_uint* failure,
    threadgroup uint* selectedPivot
) {
    return mrNumiHumanBilateralFactorCooperativeWorkspace<
        deviceWorkspace, false>(matrix, inverseScale, pivots, n, lane,
            threadCount, factorCache, matrix, scaleCache, pivotCache,
            failure, selectedPivot);
}


// Attempt an equilibrated Cholesky factor for the symmetric positive-definite
// reduced operator. This is only a candidate: on any invalid or non-positive
// pivot it leaves matrix untouched, allowing the caller to run the original
// pivoted bilateral factor and preserve its acceptance behavior.
//
// The existing bilateral solve consumes unit-lower L followed by upper U,
// whereas Cholesky naturally produces Lc Lc^T. Encode the equivalent no-pivot
// LU as L[i,j] = Lc[i,j]/Lc[j,j], U[i,j] = Lc[i,i]*Lc[j,i], and U[i,i] =
// Lc[i,i]^2. Thus L U = Lc Lc^T, and the existing scaled cooperative solve
// needs no alternate RHS, tolerance, or state path.
inline bool mrNumiHumanReducedCholeskyFactorCooperative(
    device float* matrix,
    device float* inverseScale,
    device float* pivots,
    const uint n,
    const uint lane,
    const uint threadCount,
    threadgroup float* factorCache,
    threadgroup float* scaleCache,
    threadgroup atomic_uint* failure
) {
    constexpr uint cacheElements =
        kCachedEqualityCapacity * kCachedEqualityCapacity;
    // Keep both the Cholesky factor and its converted LU in the existing
    // 16-KiB threadgroup arena. Larger reduced blocks use the old factor.
    if (n == 0u || n > MR_NUMI_HUMAN_STAND_MAX_DOFS ||
        2u * n * n > cacheElements) return false;

    if (lane == 0u)
        atomic_store_explicit(failure, MR_INVALID_INDEX,
                              memory_order_relaxed);
    const uint matrixElements = n * n;
    for (uint index = lane; index < matrixElements; index += threadCount)
        factorCache[index] = matrix[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < matrixElements; index += threadCount)
        if (!isfinite(factorCache[index]))
            atomic_fetch_min_explicit(failure, index,
                                      memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    for (uint i = lane; i < n; i += threadCount) {
        const float diagonal = factorCache[i * n + i];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            atomic_fetch_min_explicit(failure, i, memory_order_relaxed);
            continue;
        }
        scaleCache[i] = 1.0f / mrNHBilateralSqrt(diagonal);
        if (!(scaleCache[i] > 0.0f) || !isfinite(scaleCache[i]))
            atomic_fetch_min_explicit(failure, i, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / n;
        const uint column = index - row * n;
        if (row < column) continue;
        const float value =
            (factorCache[index] * scaleCache[row]) * scaleCache[column];
        factorCache[index] = value;
        if (!isfinite(value))
            atomic_fetch_min_explicit(failure, index,
                                      memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    // Each step computes one positive Cholesky diagonal on lane zero and
    // distributes the independent lower-column updates over the group.
    for (uint column = 0u; column < n; ++column) {
        if (lane == 0u) {
            float diagonal = factorCache[column * n + column];
            for (uint inner = 0u; inner < column; ++inner) {
                const float value = factorCache[column * n + inner];
                diagonal -= value * value;
            }
            if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
                atomic_fetch_min_explicit(failure, column,
                                          memory_order_relaxed);
            } else {
                const float pivot = mrNHBilateralSqrt(diagonal);
                if (!(pivot > 0.0f) || !isfinite(pivot))
                    atomic_fetch_min_explicit(failure, column,
                                              memory_order_relaxed);
                else
                    factorCache[column * n + column] = pivot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(failure, memory_order_relaxed) !=
            MR_INVALID_INDEX) return false;

        const float pivot = factorCache[column * n + column];
        for (uint row = column + 1u + lane; row < n;
             row += threadCount) {
            float value = factorCache[row * n + column];
            for (uint inner = 0u; inner < column; ++inner)
                value -= factorCache[row * n + inner] *
                    factorCache[column * n + inner];
            value /= pivot;
            if (!isfinite(value)) {
                atomic_fetch_min_explicit(failure, row,
                                          memory_order_relaxed);
                continue;
            }
            factorCache[row * n + column] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(failure, memory_order_relaxed) !=
            MR_INVALID_INDEX) return false;
    }

    // Preserve the unmodified Cholesky values in the unused tail of the same
    // cache before writing the solve-compatible LU representation in place.
    const uint choleskyBase = matrixElements;
    for (uint index = lane; index < matrixElements; index += threadCount)
        factorCache[choleskyBase + index] = factorCache[index];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / n;
        const uint column = index - row * n;
        threadgroup const float* cholesky = factorCache + choleskyBase;
        float value = cholesky[index];
        if (row > column)
            value /= cholesky[column * n + column];
        else if (row < column)
            value = cholesky[row * n + row] *
                cholesky[column * n + row];
        else
            value *= value;
        factorCache[index] = value;
        if (!isfinite(value) || (row == column && !(value > 0.0f)))
            atomic_fetch_min_explicit(failure, index,
                                      memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(failure, memory_order_relaxed) !=
        MR_INVALID_INDEX) return false;

    for (uint index = lane; index < matrixElements; index += threadCount)
        matrix[index] = factorCache[index];
    for (uint i = lane; i < n; i += threadCount) {
        inverseScale[i] = scaleCache[i];
        pivots[i] = float(i);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return true;
}

inline bool finite4(const float4 value) { return all(isfinite(value)); }

inline bool validTendonTransfer(
    device const MRNumiHumanTendonBindingGPU& binding,
    device const MRNumiHumanTendonTransferResultGPU& transfer,
    constant const MRNumiHumanStandDispatchGPU& dispatch,
    const uint environment,
    const uint endpoint
) {
    bool validTransfer =
        transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
        transfer.environment == environment &&
        transfer.bindingIndex == endpoint &&
        finite4(transfer.terminalWorldForce) &&
        finite4(transfer.residualsAndForce) &&
        transfer.residualsAndForce.x >= 0.0f &&
        transfer.residualsAndForce.y >= 0.0f &&
        transfer.residualsAndForce.z >= 0.0f;
    for (uint node = 0u; node < 4u && validTransfer; ++node)
        validTransfer = finite4(transfer.nodalWorldForces[node]) &&
            transfer.nodalWorldForces[node].w == 0.0f;
    if (binding.mode == MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT)
        return validTransfer && transfer.envelopeIndex == MR_INVALID_INDEX;
    if (binding.mode == MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE)
        return validTransfer &&
            binding.envelopeIndex < dispatch.tendonEnvelopeCount &&
            transfer.envelopeIndex == binding.envelopeIndex;
    return false;
}

inline float4 quaternionConjugate(const float4 value) {
    return float4(-value.xyz, value.w);
}

inline float4 quaternionMultiply(const float4 left, const float4 right) {
    return float4(
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
        left.w * right.w - dot(left.xyz, right.xyz)
    );
}

inline float3 quaternionRotate(const float4 quaternion, const float3 value) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}

inline bool normalizedQuaternion(const float4 input, thread float4& output) {
    const float normSquared = dot(input, input);
    if (!finite4(input) || !(normSquared > 1.0e-12f) || !isfinite(normSquared)) {
        return false;
    }
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline float4 quaternionFromRotationVector(const float3 rotationVector) {
    const float angleSquared = dot(rotationVector, rotationVector);
    if (angleSquared < 1.0e-12f) {
        return normalize(float4(0.5f * rotationVector, 1.0f));
    }
    const float angle = sqrt(angleSquared);
    return normalize(float4(
        rotationVector * (sin(0.5f * angle) / angle), cos(0.5f * angle)
    ));
}

inline float3 worldInertiaMultiply(
    device const MRBodyPropertiesGPU& body,
    const float4 orientation,
    const float3 worldVector
) {
    const float3 local = quaternionRotate(
        quaternionConjugate(orientation), worldVector
    );
    const float3 localResult{
        dot(body.inertiaRow0.xyz, local),
        dot(body.inertiaRow1.xyz, local),
        dot(body.inertiaRow2.xyz, local),
    };
    return quaternionRotate(orientation, localResult);
}

template <typename WorkspacePointer, typename OutputPointer>
inline bool solveFactor(
    device const float* factor,
    WorkspacePointer workspace,
    OutputPointer output,
    const uint nv,
    device const uint* sparseGraph
) {
    if (kUseSparseStandOperator) {
        device const uint* upperOffsets = sparseGraph + sparseGraph[4];
        device const uint* upperColumns = sparseGraph + sparseGraph[5];
        device const uint* lowerOffsets = sparseGraph + sparseGraph[6];
        device const uint* lowerColumns = sparseGraph + sparseGraph[7];
        for (uint reverse = 0u; reverse < nv; ++reverse) {
            const uint row = nv - 1u - reverse;
            float value = output[row];
            for (uint i = upperOffsets[row]; i < upperOffsets[row + 1u]; ++i) {
                const uint column = upperColumns[i];
                value -= factor[row * nv + column] * workspace[column];
            }
            const float diagonal = factor[row * nv + row];
            if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
            workspace[row] = value / diagonal;
        }
        for (uint row = 0u; row < nv; ++row) {
            float value = workspace[row];
            for (uint i = lowerOffsets[row]; i < lowerOffsets[row + 1u]; ++i) {
                const uint column = lowerColumns[i];
                value -= factor[column * nv + row] * output[column];
            }
            output[row] = value / factor[row * nv + row];
            if (!isfinite(output[row])) return false;
        }
        return true;
    }
    for (uint row = 0u; row < nv; ++row) {
        float value = output[row];
        for (uint column = 0u; column < row; ++column) {
            value -= factor[row * nv + column] * workspace[column];
        }
        const float diagonal = factor[row * nv + row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) return false;
        workspace[row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < nv; ++reverse) {
        const uint row = nv - 1u - reverse;
        float value = workspace[row];
        for (uint column = row + 1u; column < nv; ++column) {
            value -= factor[column * nv + row] * output[column];
        }
        output[row] = value / factor[row * nv + row];
        if (!isfinite(output[row])) return false;
    }
    return true;
}

// The free RHS is one ordered triangular solve. Resolve a small prefix on
// lane zero, then apply that prefix to independent future rows in parallel.
// Every row retains the scalar solver's increasing-column subtraction order;
// backward substitution also retains its original descending-row owner.
inline bool solveFactorCooperativeForward(
    device const float* factor,
    threadgroup float* workspace,
    threadgroup float* output,
    const uint nv,
    const uint lane,
    const uint threadCount,
    threadgroup uint* succeeded,
    device const uint* sparseGraph
) {
    if (lane == 0u) *succeeded = 1u;
    for (uint row = lane; row < nv; row += threadCount)
        workspace[row] = output[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint blockSize = 4u;
    if (kUseSparseStandOperator) {
        device const uint* upperOffsets = sparseGraph + sparseGraph[4];
        device const uint* upperColumns = sparseGraph + sparseGraph[5];
        device const uint* lowerOffsets = sparseGraph + sparseGraph[6];
        device const uint* lowerColumns = sparseGraph + sparseGraph[7];
        device const uint* levelOffsets = sparseGraph + sparseGraph[8];
        device const uint* levelColumns = sparseGraph + sparseGraph[9];
        threadgroup atomic_uint* failed =
            reinterpret_cast<threadgroup atomic_uint*>(succeeded);
        for (uint level = 0u; level < sparseGraph[10]; ++level) {
            for (uint j = levelOffsets[level] + lane;
                 j < levelOffsets[level + 1u]; j += threadCount) {
                const uint row = levelColumns[j];
                float value = output[row];
                for (uint i = upperOffsets[row]; i < upperOffsets[row + 1u]; ++i) {
                    const uint column = upperColumns[i];
                    value -= factor[row * nv + column] * workspace[column];
                }
                const float diagonal = factor[row * nv + row];
                if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
                    atomic_store_explicit(failed, 0u, memory_order_relaxed);
                } else {
                    workspace[row] = value / diagonal;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (atomic_load_explicit(failed, memory_order_relaxed) == 0u) return false;
        }
        for (uint reverse = 0u; reverse < sparseGraph[10]; ++reverse) {
            const uint level = sparseGraph[10] - 1u - reverse;
            for (uint j = levelOffsets[level] + lane;
                 j < levelOffsets[level + 1u]; j += threadCount) {
                const uint row = levelColumns[j];
                float value = workspace[row];
                for (uint i = lowerOffsets[row]; i < lowerOffsets[row + 1u]; ++i) {
                    const uint column = lowerColumns[i];
                    value -= factor[column * nv + row] * output[column];
                }
                output[row] = value / factor[row * nv + row];
                if (!isfinite(output[row]))
                    atomic_store_explicit(failed, 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (atomic_load_explicit(failed, memory_order_relaxed) == 0u) return false;
        }
        return true;
    }
    for (uint begin = 0u; begin < nv; begin += blockSize) {
        const uint end = min(begin + blockSize, nv);
        if (lane == 0u) {
            for (uint row = begin; row < end; ++row) {
                float value = workspace[row];
                for (uint column = begin; column < row; ++column)
                    value -= factor[row * nv + column] * workspace[column];
                const float diagonal = factor[row * nv + row];
                if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
                    *succeeded = 0u;
                    break;
                }
                workspace[row] = value / diagonal;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (*succeeded == 0u) return false;
        for (uint row = end + lane; row < nv; row += threadCount) {
            float value = workspace[row];
            for (uint column = begin; column < end; ++column)
                value -= factor[row * nv + column] * workspace[column];
            workspace[row] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        for (uint reverse = 0u; reverse < nv; ++reverse) {
            const uint row = nv - 1u - reverse;
            float value = workspace[row];
            for (uint column = row + 1u; column < nv; ++column)
                value -= factor[column * nv + row] * output[column];
            output[row] = value / factor[row * nv + row];
            if (!isfinite(output[row])) {
                *succeeded = 0u;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *succeeded != 0u;
}

inline float pointJacobianAxis(
    device const float* pointJacobians,
    const uint base,
    const uint point,
    const uint nv,
    const uint dof,
    const float3 direction
) {
    const uint pointBase = base + point * 3u * nv;
    return direction.x * pointJacobians[pointBase + 0u * nv + dof] +
        direction.y * pointJacobians[pointBase + 1u * nv + dof] +
        direction.z * pointJacobians[pointBase + 2u * nv + dof];
}

inline bool evaluateJointEquality(
    device const MRNumiHumanJointEqualityGPU& equality,
    device const float* q,
    const uint qBase,
    const uint nq,
    const uint nv,
    thread float& target,
    thread float& derivative,
    thread float& error
) {
    const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
        equality.indices.w == MR_INVALID_INDEX;
    const bool coupled = equality.indices.z < nq && equality.indices.w < nv;
    if (equality.indices.x >= nq || equality.indices.y >= nv ||
        (!fixed && !coupled) ||
        (coupled && (equality.indices.x == equality.indices.z ||
                     equality.indices.y == equality.indices.w)) ||
        !finite4(equality.referencesAndCoefficients0) ||
        !finite4(equality.coefficients1) || !finite4(equality.solref) ||
        !finite4(equality.solimp0) || !finite4(equality.solimp1) ||
        equality.coefficients1.w != 0.0f || equality.solref.z != 0.0f ||
        equality.solref.w != 0.0f || equality.solimp1.y != 0.0f ||
        equality.solimp1.z != 0.0f || equality.solimp1.w != 0.0f) {
        return false;
    }
    const float delta = fixed
        ? 0.0f
        : q[qBase + equality.indices.z] -
            equality.referencesAndCoefficients0.y;
    const float a0 = equality.referencesAndCoefficients0.z;
    const float a1 = equality.referencesAndCoefficients0.w;
    const float a2 = equality.coefficients1.x;
    const float a3 = equality.coefficients1.y;
    const float a4 = equality.coefficients1.z;
    const float polynomial = a0 + delta * (
        a1 + delta * (a2 + delta * (a3 + delta * a4))
    );
    derivative = fixed
        ? 0.0f
        : a1 + delta * (
            2.0f * a2 + delta * (3.0f * a3 + 4.0f * delta * a4)
        );
    target = equality.referencesAndCoefficients0.x + polynomial;
    error = q[qBase + equality.indices.x] - target;
    return isfinite(target) && isfinite(derivative) && isfinite(error);
}

inline void fail(
    device MRNumiHumanStandStatusGPU& status,
    const uint code,
    const uint index
) {
    if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS) {
        status.code = code;
        status.failingIndex = index;
    }
}

inline void publishParallelResponseFailure(
    device MRNumiHumanStandStatusGPU& status, uint ordinal
) {
    device atomic_uint* failure =
        reinterpret_cast<device atomic_uint*>(&status.failingIndex);
    atomic_fetch_min_explicit(failure, ordinal, memory_order_relaxed);
}

} // namespace

// Large-state Human dynamics deliberately consumes the already-authoritative
// Metal kinematics/Jacobian and MyoSim J^T streams. Matrix assembly is spread
// across the threadgroup; lane zero performs deterministic Cholesky, source
// support projection, and state publication. The first release retains a
// low-velocity bias model (gravity, gyroscopic and authored body damping) and
// exposes that evidence boundary to the host rather than pretending to be an
// exact high-speed RNEA replacement.
kernel void mr_numi_human_stand_step(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device float* qState [[buffer(5)]],
    device float* vState [[buffer(6)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(7)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const float* generalizedForceWorkspace [[buffer(10)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device float4* bodyMotionScratch [[buffer(13)]],
    device float* factorScratch [[buffer(14)]],
    device float* vectorScratch [[buffer(15)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanTendonBindingGPU* tendonBindings [[buffer(18)]],
    device const MRNumiHumanTendonTransferResultGPU* tendonTransfers [[buffer(19)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(21)]],
    device const float4* bodyPositionLow [[buffer(22)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const float* passiveJointProgram [[buffer(24)]],
    device float* sourceDynamicsWitness [[buffer(25)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    device const MRWorldGPU& world = worlds[0];
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint bodyCount = articulation.bodyCount;
    const uint nv = articulation.nv;
    const uint nq = articulation.nq;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint bodyPoseBase = environment * dispatch.bodyPoseStride;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase = environment * dispatch.pointJacobianStride;
    const uint forceBase = environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    const uint inertiaWeightedBase = spatialBase + bodyCount * 6u * nv;
    const uint bodyMotionBase = environment * bodyCount * 2u;
    const uint factorBase = environment * nv * nv;
    const uint vectorStride = nv + 3u * nv +
        12u * dispatch.supportContactCount + dispatch.jointEqualityCount +
        (((dispatch.flags & MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES) != 0u)
            ? nv + nq + nv : 0u);
    const uint preloadBase = environment * vectorStride;
    const uint vectorBase = preloadBase + nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint responseColumns = (dispatch.supportContactCount * 3u + equalityCount + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    const uint responseBase = environment * responseStride;
    device float* equalityFactor = responseScratch + responseBase + responseColumns;
    device float* equalityScale = equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    // These vectors are repeatedly read and updated by the scalar constraint
    // solve. Keep them in the threadgroup instead of round-tripping every
    // scalar dependency through device memory. Retain the exported scratch
    // layout and publish its final values below.
    threadgroup float equalityRhsStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float equalityFactorStorage[
        kCachedEqualityCapacity * kCachedEqualityCapacity];
    threadgroup float candidateVStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspaceStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float massRowScale[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float* equalityRhs = equalityRhsStorage;
    // Equality multiplier corrections for each conditioned limit response.
    device float* limitEqualityCorrections = equalityPivots + 2u * equalityCount;
    device float* bias = vectorScratch + vectorBase;
    threadgroup float* candidateV = candidateVStorage;
    threadgroup float* workspace = workspaceStorage;
    device float* lambdas = bias + 3u * nv;
    device float* equalityLambdas =
        lambdas + 3u * dispatch.supportContactCount;
    device float* contactMatrices =
        equalityLambdas + dispatch.jointEqualityCount;
    device float* sourceLimitImpulseEvidence =
        contactMatrices + 9u * dispatch.supportContactCount;
    device float* preProjectionQEvidence = sourceLimitImpulseEvidence + nv;
    device float* preProjectionVEvidence = preProjectionQEvidence + nq;
    device float* factor = factorScratch + factorBase;
    const bool captureSourceDynamics =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u;
    const uint sourceDynamicsBase = environment * 3u * nv;
    const bool massPrerequisitesOnly =
        !kUseStandFactorOnlySpecialized &&
        (dispatch.flags & MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY) != 0u;
    const bool massReady =
        kUseStandFactorOnlySpecialized ||
        (dispatch.flags & MR_NUMI_HUMAN_STAND_MASS_READY) != 0u;
    const bool factorOnly =
        kUseStandFactorOnlySpecialized ||
        (dispatch.flags & MR_NUMI_HUMAN_STAND_FACTOR_ONLY) != 0u;
    const bool responsesReady =
        !kUseStandFactorOnlySpecialized &&
        (dispatch.flags & MR_NUMI_HUMAN_STAND_RESPONSES_READY) != 0u;

    if (kUseStandFactorOnlySpecialized) {
        constexpr uint expected = MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
            MR_NUMI_HUMAN_STAND_MASS_READY | MR_NUMI_HUMAN_STAND_FACTOR_ONLY;
        constexpr uint phases = expected | MR_NUMI_HUMAN_STAND_RESPONSES_READY |
            MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY |
            MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY;
        if ((dispatch.flags & phases) != expected) {
            if (lane == 0u)
                fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, MR_INVALID_INDEX);
            return;
        }
    }

    if (massReady) {
        if (lane == 0u &&
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) == 0u ||
             massPrerequisitesOnly ||
             (status.flags & MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY) == 0u))
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH,
                 MR_INVALID_INDEX);
        threadgroup_barrier(mem_flags::mem_device);
        if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    } else {
    if (lane == 0u) {
        if (dispatch.stepIndex == 0u) {
            status = {};
            status.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            status.environment = environment;
            status.failingIndex = MR_INVALID_INDEX;
            status.jointEqualityCounts.w = MR_INVALID_INDEX;
            status.constraintImpulseOwners = uint4(MR_INVALID_INDEX);
            status.velocityDiagnosticOwners = uint4(MR_INVALID_INDEX);
            status.contactAndAcceleration.x =
                (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u &&
                    dispatch.supportContactCount != 0u
                    ? INFINITY
                    : 0.0f;
            status.factorAndAssistance.x = INFINITY;
            for (uint index = 0u;
                 index < 3u * dispatch.supportContactCount;
                 ++index) {
                lambdas[index] = 0.0f;
            }
        }
        if (dispatch.abiVersion != MR_NUMI_HUMAN_STAND_ABI_VERSION ||
            dispatch.stepCount == 0u ||
            dispatch.stepCount > MR_NUMI_HUMAN_STAND_MAX_HORIZON_STEPS ||
            dispatch.stepIndex >= dispatch.stepCount ||
            dispatch.articulationIndex >= world.articulationCount ||
            dispatch.qStride < nq || dispatch.vStride < nv ||
            dispatch.bodyPoseStride < bodyCount ||
            dispatch.generalizedForceStride < nv ||
            dispatch.bodyJacobianPointOffset > dispatch.pointWorldStride ||
            bodyCount >
                (dispatch.pointWorldStride -
                 dispatch.bodyJacobianPointOffset) / 4u ||
            dispatch.pointJacobianStride /
                max(3u * nv, 1u) < dispatch.pointWorldStride ||
            dispatch.supportContactCount > MR_NUMI_HUMAN_STAND_MAX_CONTACTS ||
            dispatch.jointEqualityCount > nv ||
            dispatch.contactIterationCount == 0u ||
            dispatch.contactIterationCount > 64u ||
            !(dispatch.groundPointAndTimestep.w > 0.0f) ||
            !finite4(dispatch.groundPointAndTimestep) ||
            !finite4(dispatch.groundNormal) ||
            !finite4(dispatch.targetRootPosition) ||
            !finite4(dispatch.targetRootOrientation) ||
            !finite4(dispatch.assistanceGains) ||
            !mrNumiHumanTimedRootForceValid(
                dispatch.timedRootForce, dispatch.stepCount) ||
            dispatch.groundNormal.w != 0.0f ||
            dispatch.targetRootPosition.w != 0.0f ||
            dispatch.tendonTransferStride < dispatch.tendonEndpointCount ||
            ((dispatch.tendonEndpointCount == 0u) !=
             ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS) == 0u)) ||
            (dispatch.tendonEndpointCount != 0u &&
             (dispatch.tendonEndpointCount % 2u) != 0u) ||
            ((dispatch.jointEqualityCount == 0u) !=
             ((dispatch.flags &
               MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES) == 0u)) ||
            (dispatch.flags & ~(
                MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
                MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE |
                MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS |
                MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES |
                MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY |
                MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM |
                MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES |
                MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
                MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY |
                MR_NUMI_HUMAN_STAND_MASS_READY |
                MR_NUMI_HUMAN_STAND_FACTOR_ONLY |
                MR_NUMI_HUMAN_STAND_RESPONSES_READY |
                MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES |
                MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE |
                MR_NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS |
                MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM
            )) != 0u ||
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE) != 0u &&
             (dispatch.flags & MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES) == 0u) ||
            (massPrerequisitesOnly &&
             ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) == 0u ||
              massReady)) ||
            (massReady &&
             (dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) == 0u) ||
            ((factorOnly || responsesReady) &&
             (!massReady ||
              (dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) == 0u ||
              massPrerequisitesOnly || (factorOnly && responsesReady))) ||
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) != 0u &&
             (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u) ||
            ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u &&
             ((dispatch.flags & (MR_NUMI_HUMAN_STAND_ENABLE_CONTACT |
                                 MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES)) != 0u ||
              dispatch.supportContactCount != 0u ||
              dispatch.jointEqualityCount != 0u))) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, MR_INVALID_INDEX);
        } else if (articulation.rootType != MR_ROOT_FLOATING ||
                   bodyCount == 0u || bodyCount > MR_NUMI_HUMAN_STAND_MAX_BODIES ||
                   nv < 6u || nv > MR_NUMI_HUMAN_STAND_MAX_DOFS ||
                   nq < 7u || nq > MR_NUMI_HUMAN_STAND_MAX_Q ||
                   articulation.firstBody + bodyCount > world.bodyCount ||
                   articulation.vOffset + nv > world.nv ||
                   articulation.qOffset + nq > world.nq) {
            fail(status, MR_NUMI_HUMAN_STAND_INVALID_MODEL, MR_INVALID_INDEX);
        } else {
            const float normalLengthSquared = dot(
                dispatch.groundNormal.xyz, dispatch.groundNormal.xyz
            );
            if (!isfinite(normalLengthSquared) ||
                abs(normalLengthSquared - 1.0f) > 2.0e-4f) {
                fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, MR_INVALID_INDEX);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Validate the exact per-step terminal-load transaction before any Human
    // state is advanced. These loads are wrench-equivalent to MyoSim's
    // existing source-route J^T force; they are exposed to bone/deformable
    // consumers and deliberately are not added here as a second joint torque.
    // A registered pre-dynamics consumer may already have replaced a declared
    // J^T share in generalizedForceWorkspace with a solved anchor reaction.
    threadgroup uint tendonFailureByLane[256];
    threadgroup uint tendonPointCountByLane[256];
    threadgroup uint tendonEnvelopeCountByLane[256];
    threadgroup float4 tendonDiagnosticByLane[256];
    if (dispatch.tendonEndpointCount != 0u) {
        const uint transferBase = environment * dispatch.tendonTransferStride;
        uint firstFailure = MR_INVALID_INDEX;
        uint pointCount = 0u;
        uint envelopeCount = 0u;
        float4 diagnostic = float4(0.0f);
        for (uint endpoint = lane; endpoint < dispatch.tendonEndpointCount;
             endpoint += threadCount) {
            device const MRNumiHumanTendonBindingGPU& binding =
                tendonBindings[endpoint];
            device const MRNumiHumanTendonTransferResultGPU& transfer =
                tendonTransfers[transferBase + endpoint];
            if (!validTendonTransfer(binding, transfer, dispatch,
                                     environment, endpoint)) {
                firstFailure = min(firstFailure, endpoint);
                continue;
            }
            if (binding.mode ==
                MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE)
                ++envelopeCount;
            else
                ++pointCount;
            diagnostic = max(diagnostic, abs(transfer.residualsAndForce));
        }
        tendonFailureByLane[lane] = firstFailure;
        tendonPointCountByLane[lane] = pointCount;
        tendonEnvelopeCountByLane[lane] = envelopeCount;
        tendonDiagnosticByLane[lane] = diagnostic;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            firstFailure = MR_INVALID_INDEX;
            for (uint index = 0u; index < threadCount; ++index)
                firstFailure = min(firstFailure, tendonFailureByLane[index]);
            if (firstFailure == MR_INVALID_INDEX) {
                for (uint index = 0u; index < threadCount; ++index) {
                    status.tendonPointTransferCount +=
                        tendonPointCountByLane[index];
                    status.tendonEnvelopeTransferCount +=
                        tendonEnvelopeCountByLane[index];
                    status.tendonDiagnostics = max(status.tendonDiagnostics,
                        tendonDiagnosticByLane[index]);
                }
                status.tendonTransferCount += dispatch.tendonEndpointCount;
            } else {
                // Reproduce the original ordered prefix and typed first
                // failure; no later transfer may enter an accepted record.
                for (uint endpoint = 0u; endpoint <= firstFailure;
                     ++endpoint) {
                    device const auto& binding = tendonBindings[endpoint];
                    device const auto& transfer =
                        tendonTransfers[transferBase + endpoint];
                    if (!validTendonTransfer(binding, transfer, dispatch,
                                             environment, endpoint)) {
                        ++status.tendonFailureCount;
                        fail(status, MR_NUMI_HUMAN_STAND_TENDON_TRANSFER_FAILED,
                             endpoint);
                        break;
                    }
                    ++status.tendonTransferCount;
                    if (binding.mode ==
                        MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE)
                        ++status.tendonEnvelopeTransferCount;
                    else
                        ++status.tendonPointTransferCount;
                    status.tendonDiagnostics = max(status.tendonDiagnostics,
                        abs(transfer.residualsAndForce));
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Lane-zero validation avoids racing writes to the diagnostic status.
    if (lane == 0u) {
        for (uint index = 0u; index < nq; ++index) {
            if (!isfinite(qState[qBase + index])) {
                fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, index);
                break;
            }
        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS) {
            for (uint index = 0u; index < nv; ++index) {
                if (!isfinite(vState[vBase + index]) ||
                    !isfinite(generalizedForceWorkspace[forceBase + index])) {
                    fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_INPUT, index);
                    break;
                }
            }
        }
        // Contact impulses start cold. Reusing the old values without first
        // applying them to candidateV would make the projected deltas wrong.
        for (uint index = 0u;
             index < 3u * dispatch.supportContactCount;
             ++index) {
            lambdas[index] = 0.0f;
        }
        for (uint index = 0u;
             index < dispatch.jointEqualityCount;
             ++index) {
            equalityLambdas[index] = 0.0f;
            device const MRNumiHumanJointEqualityGPU& equality =
                jointEqualities[index];
            float target = 0.0f;
            float derivative = 0.0f;
            float error = 0.0f;
            bool valid = evaluateJointEquality(
                equality, qState, qBase, nq, nv,
                target, derivative, error
            );
            valid = valid &&
                dofs[articulation.vOffset + equality.indices.y].qIndex ==
                    articulation.qOffset + equality.indices.x;
            if (valid && equality.indices.w != MR_INVALID_INDEX) {
                valid = dofs[
                    articulation.vOffset + equality.indices.w
                ].qIndex == articulation.qOffset + equality.indices.z;
            }
            for (uint prior = 0u; prior < index && valid; ++prior) {
                const uint priorDependent = jointEqualities[prior].indices.y;
                valid = priorDependent != equality.indices.y &&
                    priorDependent != equality.indices.w;
            }
            for (uint later = index + 1u;
                 later < dispatch.jointEqualityCount && valid; ++later) {
                valid = jointEqualities[later].indices.y != equality.indices.w;
            }
            if (!valid) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, index);
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;

    // Reconstruct a world spatial Jacobian for every body from its COM and
    // three unit body-axis point probes.
    const uint spatialElements = bodyCount * nv;
    for (uint index = lane; index < spatialElements; index += threadCount) {
        const uint localBody = index / nv;
        const uint dof = index - localBody * nv;
        const uint probe = dispatch.bodyJacobianPointOffset + 4u * localBody;
        const uint probeBase = pointJacobianBase + probe * 3u * nv;
        float3 linear;
        float3 angular;
        const float4 orientation = bodyPoses[bodyPoseBase + localBody].orientation;
        if ((dispatch.flags &
             MR_NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS) != 0u) {
            const uint base = spatialBase + localBody * 6u * nv + dof;
            angular = float3(
                spatialJacobianScratch[base + 0u * nv],
                spatialJacobianScratch[base + 1u * nv],
                spatialJacobianScratch[base + 2u * nv]);
            linear = float3(
                spatialJacobianScratch[base + 3u * nv],
                spatialJacobianScratch[base + 4u * nv],
                spatialJacobianScratch[base + 5u * nv]);
        } else {
            linear = float3(
                pointJacobians[probeBase + 0u * nv + dof],
                pointJacobians[probeBase + 1u * nv + dof],
                pointJacobians[probeBase + 2u * nv + dof]);
            const float3 dx{
                pointJacobians[probeBase + 3u * nv + 0u * nv + dof] - linear.x,
                pointJacobians[probeBase + 3u * nv + 1u * nv + dof] - linear.y,
                pointJacobians[probeBase + 3u * nv + 2u * nv + dof] - linear.z,
            };
            const float3 dy{
                pointJacobians[probeBase + 6u * nv + 0u * nv + dof] - linear.x,
                pointJacobians[probeBase + 6u * nv + 1u * nv + dof] - linear.y,
                pointJacobians[probeBase + 6u * nv + 2u * nv + dof] - linear.z,
            };
            const float3 dz{
                pointJacobians[probeBase + 9u * nv + 0u * nv + dof] - linear.x,
                pointJacobians[probeBase + 9u * nv + 1u * nv + dof] - linear.y,
                pointJacobians[probeBase + 9u * nv + 2u * nv + dof] - linear.z,
            };
            const float3 axisX = quaternionRotate(orientation, float3(1.0f, 0.0f, 0.0f));
            const float3 axisY = quaternionRotate(orientation, float3(0.0f, 1.0f, 0.0f));
            const float3 axisZ = quaternionRotate(orientation, float3(0.0f, 0.0f, 1.0f));
            angular = 0.5f * (
                cross(axisX, dx) + cross(axisY, dy) + cross(axisZ, dz)
            );
        }
        const uint base = spatialBase + localBody * 6u * nv + dof;
        spatialJacobianScratch[base + 0u * nv] = angular.x;
        spatialJacobianScratch[base + 1u * nv] = angular.y;
        spatialJacobianScratch[base + 2u * nv] = angular.z;
        spatialJacobianScratch[base + 3u * nv] = linear.x;
        spatialJacobianScratch[base + 4u * nv] = linear.y;
        spatialJacobianScratch[base + 5u * nv] = linear.z;
        // I_world J_angular is shared by every mass-matrix row. Compute it
        // once per body/column, retaining the same world-inertia operation
        // and body-ordered dot-product reduction used by the original path.
        const float3 inertiaWeighted = worldInertiaMultiply(
            bodies[articulation.firstBody + localBody], orientation, angular);
        const uint weighted = inertiaWeightedBase + localBody * 3u * nv + dof;
        spatialJacobianScratch[weighted + 0u * nv] = inertiaWeighted.x;
        spatialJacobianScratch[weighted + 1u * nv] = inertiaWeighted.y;
        spatialJacobianScratch[weighted + 2u * nv] = inertiaWeighted.z;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint localBody = lane; localBody < bodyCount; localBody += threadCount) {
        float3 angular{0.0f};
        float3 linear{0.0f};
        const uint base = spatialBase + localBody * 6u * nv;
        for (uint dof = 0u; dof < nv; ++dof) {
            const float velocity = vState[vBase + dof];
            angular += velocity * float3(
                spatialJacobianScratch[base + 0u * nv + dof],
                spatialJacobianScratch[base + 1u * nv + dof],
                spatialJacobianScratch[base + 2u * nv + dof]
            );
            linear += velocity * float3(
                spatialJacobianScratch[base + 3u * nv + dof],
                spatialJacobianScratch[base + 4u * nv + dof],
                spatialJacobianScratch[base + 5u * nv + dof]
            );
        }
        bodyMotionScratch[bodyMotionBase + 2u * localBody + 0u] = float4(angular, 0.0f);
        bodyMotionScratch[bodyMotionBase + 2u * localBody + 1u] = float4(linear, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint row = lane; row < nv; row += threadCount) {
        float value = 0.0f;
        MRCompensatedScalar bodySum{0.0f, 0.0f};
        const bool compensated = (dispatch.flags &
            MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM) != 0u;
        for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
            const uint globalBody = articulation.firstBody + localBody;
            device const MRBodyPropertiesGPU& body = bodies[globalBody];
            const uint base = spatialBase + localBody * 6u * nv;
            const float3 jw{
                spatialJacobianScratch[base + 0u * nv + row],
                spatialJacobianScratch[base + 1u * nv + row],
                spatialJacobianScratch[base + 2u * nv + row],
            };
            const float3 jv{
                spatialJacobianScratch[base + 3u * nv + row],
                spatialJacobianScratch[base + 4u * nv + row],
                spatialJacobianScratch[base + 5u * nv + row],
            };
            const float3 angular = bodyMotionScratch[
                bodyMotionBase + 2u * localBody + 0u
            ].xyz;
            const float3 linear = bodyMotionScratch[
                bodyMotionBase + 2u * localBody + 1u
            ].xyz;
            const float4 orientation = bodyPoses[bodyPoseBase + localBody].orientation;
            const float3 angularMomentum = worldInertiaMultiply(body, orientation, angular);
            const float3 requiredTorque = cross(angular, angularMomentum) +
                body.dampingAndSpeedLimits.y * angular;
            const float3 requiredForce =
                -body.massAndInverseMass.x * world.gravityAndTimestep.xyz +
                body.dampingAndSpeedLimits.x * linear;
            const float contribution =
                dot(jw, requiredTorque) + dot(jv, requiredForce);
            if (compensated)
                bodySum = mrCompensatedAdd(bodySum, {contribution, 0.0f});
            else value += contribution;
        }
        if (compensated) value = bodySum.high + bodySum.low;
        device const MRDofPropertiesGPU& dof =
            dofs[articulation.vOffset + row];
        // MyoSim joint damping is passive generalized resistance. The Human
        // payload deliberately carries it without MR_DOF_FLAG_DRIVE so these
        // coordinates remain muscle-driven rather than becoming hidden PD
        // motors.
        if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
            value += dof.drive.y * vState[vBase + row];
        }
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM) != 0u) {
            for (uint column = 6u; column < nv; ++column) {
                const float stiffness = passiveJointProgram[row * nv + column];
                if (stiffness == 0.0f) continue;
                const uint sourceQ = dofs[articulation.vOffset + column].qIndex;
                const float displacement = qState[qBase + sourceQ - articulation.qOffset] -
                    passiveJointProgram[nv * nv + column];
                value += mrNumiHumanPassiveImplicitBias(stiffness, displacement,
                    vState[vBase + column], dispatch.groundPointAndTimestep.w);
            }
        }
        bias[row] = value;
        if (captureSourceDynamics) {
            // Preserve the raw device bias before the vector row becomes the
            // free-velocity scratch later in this same kernel.
            sourceDynamicsWitness[sourceDynamicsBase + nv + row] = value;
        }
    }
    }
    if (massPrerequisitesOnly) {
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0u && status.code == MR_NUMI_HUMAN_STAND_SUCCESS)
            status.flags = dispatch.flags;
        return;
    }

    if (!massReady) {
    const uint matrixElements = nv * nv;
    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / nv;
        const uint column = index - row * nv;
        float value = 0.0f;
        MRCompensatedScalar bodySum{0.0f, 0.0f};
        const bool compensated = (dispatch.flags &
            MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM) != 0u;
        for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
            const uint globalBody = articulation.firstBody + localBody;
            device const MRBodyPropertiesGPU& body = bodies[globalBody];
            const uint base = spatialBase + localBody * 6u * nv;
            const float3 leftAngular{
                spatialJacobianScratch[base + 0u * nv + row],
                spatialJacobianScratch[base + 1u * nv + row],
                spatialJacobianScratch[base + 2u * nv + row],
            };
            const uint weighted = inertiaWeightedBase + localBody * 3u * nv + column;
            const float3 rightInertiaWeighted{
                spatialJacobianScratch[weighted + 0u * nv],
                spatialJacobianScratch[weighted + 1u * nv],
                spatialJacobianScratch[weighted + 2u * nv],
            };
            const float3 leftLinear{
                spatialJacobianScratch[base + 3u * nv + row],
                spatialJacobianScratch[base + 4u * nv + row],
                spatialJacobianScratch[base + 5u * nv + row],
            };
            const float3 rightLinear{
                spatialJacobianScratch[base + 3u * nv + column],
                spatialJacobianScratch[base + 4u * nv + column],
                spatialJacobianScratch[base + 5u * nv + column],
            };
            const float contribution = dot(leftAngular, rightInertiaWeighted) +
                body.massAndInverseMass.x * dot(leftLinear, rightLinear);
            if (compensated)
                bodySum = mrCompensatedAdd(bodySum, {contribution, 0.0f});
            else value += contribution;
        }
        if (compensated) value = bodySum.high + bodySum.low;
        if (row == column) {
            device const MRDofPropertiesGPU& dof =
                dofs[articulation.vOffset + row];
            value += dof.drive.z;
            if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u) {
                // Backward-Euler passive damping: (M + hD)a = tau-b-Dv.
                value += dispatch.groundPointAndTimestep.w * dof.drive.y;
            }
        }
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM) != 0u) {
            value += mrNumiHumanPassiveEffectiveInertia(
                passiveJointProgram[index], dispatch.groundPointAndTimestep.w);
        }
        factor[index] = value;
        if (captureSourceDynamics && row == column) {
            // The upper triangle remains the exact source A0, but Cholesky
            // overwrites its diagonal. Preserve that missing diagonal here,
            // before the factorization barrier.
            sourceDynamicsWitness[sourceDynamicsBase + row] = value;
        }
    }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    // The spatial Jacobian/inertia columns are dead after mass assembly and
    // the barrier above. Reuse two nv rows as per-step equality linearization
    // caches; unlike the persistent preload prefix, this scratch is rebuilt
    // before every stand step. q is unchanged throughout the coupled sweeps.
    device float* equalityDerivativeCache =
        spatialJacobianScratch + spatialBase;
    device float* equalityTargetVelocityCache =
        equalityDerivativeCache + nv;
    // The full-body spatial Jacobian is dead after response construction.
    // Reuse its arena for DOF-major equality responses consumed on every
    // coupled sweep. The derivative and target prefixes remain intact.
    device float* equalityResponseByDof =
        equalityTargetVelocityCache + nv;
    const bool cacheEqualityResponseByDof =
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS >=
        2u + equalityCount;
    device float* projectedContactEquality =
        equalityResponseByDof + nv * equalityCount;
    const bool useProjectedContacts =
        equalityCount != 0u &&
        equalityCount <= MR_NUMI_HUMAN_STAND_MAX_DOFS &&
        (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u &&
        dispatch.supportContactCount != 0u &&
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv >=
            (2u + equalityCount) * nv +
            3u * dispatch.supportContactCount * equalityCount;

    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    threadgroup atomic_uint sparseGraphFailure;
    if (!responsesReady) {
    // Freeze each row's original pivot scale before any factor entry changes.
    // Column k has one dependent diagonal; every row below it is independent.
    // Each entry retains the original increasing-inner subtraction order.
    for (uint row = lane; row < nv; row += threadCount) {
        float scale = 0.0f;
        for (uint column = 0u; column < nv; ++column)
            scale = max(scale, abs(factor[row * nv + column]));
        massRowScale[row] = scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kUseSparseStandOperator) {
        if (lane == 0u) {
            atomic_store_explicit(&sparseGraphFailure, MR_INVALID_INDEX,
                                  memory_order_relaxed);
            if (sparseGraph[0] != 0x4e485347u || sparseGraph[1] != nv ||
                sparseGraph[2] != (nv + 31u) / 32u)
                fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED,
                     MR_INVALID_INDEX);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
        device const uint* mask = sparseGraph + sparseGraph[3];
        device const uint* upperOffsets = sparseGraph + sparseGraph[4];
        device const uint* upperColumns = sparseGraph + sparseGraph[5];
        for (uint row = lane; row < nv; row += threadCount) {
            for (uint column = 0u; column < nv; ++column) {
                if ((mask[row * sparseGraph[2] + column / 32u] &
                     (1u << (column % 32u))) == 0u &&
                    factor[row * nv + column] != 0.0f)
                    atomic_fetch_min_explicit(&sparseGraphFailure, row,
                                              memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u && atomic_load_explicit(&sparseGraphFailure,
                memory_order_relaxed) != MR_INVALID_INDEX)
            fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED,
                 atomic_load_explicit(&sparseGraphFailure, memory_order_relaxed));
        threadgroup_barrier(mem_flags::mem_device);
        if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
        device const uint* levelOffsets = sparseGraph + sparseGraph[8];
        device const uint* levelColumns = sparseGraph + sparseGraph[9];
        device const uint* entryOffsets = sparseGraph + sparseGraph[11];
        device const uint* entries = sparseGraph + sparseGraph[12];
        for (uint level = 0u; level < sparseGraph[10]; ++level) {
            for (uint j = levelOffsets[level] + lane;
                 j < levelOffsets[level + 1u]; j += threadCount) {
                const uint column = levelColumns[j];
                float value = factor[column * nv + column];
                for (uint i = upperOffsets[column]; i < upperOffsets[column + 1u]; ++i) {
                    const uint inner = upperColumns[i];
                    value -= factor[column * nv + inner] * factor[column * nv + inner];
                }
                if (!(value > max(kPivotFloor,
                        massRowScale[column] * 8.0f * 1.1920928955078125e-7f)) ||
                    !isfinite(value)) {
                    atomic_fetch_min_explicit(&sparseGraphFailure, column,
                                              memory_order_relaxed);
                } else {
                    factor[column * nv + column] = sqrt(value);
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            const uint failed = atomic_load_explicit(&sparseGraphFailure, memory_order_relaxed);
            if (failed != MR_INVALID_INDEX) {
                if (lane == 0u)
                    fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, failed);
                return;
            }
            for (uint j = entryOffsets[level] + lane;
                 j < entryOffsets[level + 1u]; j += threadCount) {
                const uint column = entries[j] >> 16u;
                const uint row = entries[j] & 0xffffu;
                float value = factor[column * nv + row];
                for (uint i = upperOffsets[column]; i < upperOffsets[column + 1u]; ++i) {
                    const uint inner = upperColumns[i];
                    if ((mask[row * sparseGraph[2] + inner / 32u] &
                         (1u << (inner % 32u))) != 0u)
                        value -= factor[row * nv + inner] * factor[column * nv + inner];
                }
                factor[row * nv + column] = value / factor[column * nv + column];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        if (lane == 0u) {
            for (uint column = 0u; column < nv; ++column) {
                minimumPivot = min(minimumPivot, factor[column * nv + column]);
                maximumPivot = max(maximumPivot, factor[column * nv + column]);
            }
        }
    } else {
    for (uint column = 0u; column < nv; ++column) {
        if (lane == 0u) {
            float value = factor[column * nv + column];
            for (uint inner = 0u; inner < column; ++inner)
                value -= factor[column * nv + inner] *
                    factor[column * nv + inner];
            if (!(value > max(
                    kPivotFloor,
                    massRowScale[column] * 8.0f *
                        1.1920928955078125e-7f
                )) || !isfinite(value)) {
                fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED,
                     column);
            } else {
                factor[column * nv + column] = sqrt(value);
                minimumPivot = min(minimumPivot,
                                   factor[column * nv + column]);
                maximumPivot = max(maximumPivot,
                                   factor[column * nv + column]);
            }
        }
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
        if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
        for (uint row = column + 1u + lane; row < nv;
             row += threadCount) {
            float value = factor[row * nv + column];
            for (uint inner = 0u; inner < column; ++inner)
                value -= factor[row * nv + inner] *
                    factor[column * nv + inner];
            factor[row * nv + column] =
                value / factor[column * nv + column];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    }
    }
    // Every inverse-mass response has an independent RHS. Preserve each
    // scalar substitution's operation order while assigning columns to GPU
    // lanes, rather than running all contact/equality/limit solves on lane 0.
    threadgroup atomic_uint responseFailure;
    if (lane == 0u) {
        atomic_store_explicit(&responseFailure,
            responsesReady ? status.failingIndex : MR_INVALID_INDEX,
            memory_order_relaxed);
        if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u) {
            for (uint contact = 0u; contact < dispatch.supportContactCount; ++contact) {
                device const auto& support = contacts[contact];
                if (support.bodyIndex < articulation.firstBody ||
                    support.bodyIndex >= articulation.firstBody + bodyCount ||
                    support.pointQueryIndex >= dispatch.pointWorldStride ||
                    support.reserved0 != 0u ||
                    !finite4(support.frictionSlopAndStabilization) ||
                    support.frictionSlopAndStabilization.x < 0.0f ||
                    support.frictionSlopAndStabilization.y < 0.0f ||
                    support.frictionSlopAndStabilization.z < 0.0f ||
                    support.frictionSlopAndStabilization.z > 1.0f ||
                    support.frictionSlopAndStabilization.w < 0.0f) {
                    fail(status, MR_NUMI_HUMAN_STAND_INVALID_DISPATCH, contact);
                    break;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    if (factorOnly) return;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint equalityColumnsEnd = contactColumns + equalityCount;
    const bool contactEnabled = (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u;
    if (!responsesReady && (contactEnabled || equalityCount != 0u)) {
        const float3 responseNormal = dispatch.groundNormal.xyz;
        const float3 responseReference = abs(responseNormal.x) < 0.8f
            ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
        const float3 responseTangent0 = normalize(
            responseReference - dot(responseReference, responseNormal) * responseNormal);
        const float3 responseDirections[3] = {
            responseNormal, responseTangent0, cross(responseNormal, responseTangent0)};
        float independentWorkspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        for (uint column = lane; column < equalityColumnsEnd + nv; column += threadCount) {
            device float* response = responseScratch + responseBase + column * nv;
            if (column < contactColumns) {
                if (!contactEnabled) continue;
                device const auto& support = contacts[column / 3u];
                const uint pointIndex = pointBase + support.pointQueryIndex;
                const float gap = dot(mrCompensatedPositionDifference(
                    pointWorld[pointIndex].position, pointPositionLow[pointIndex],
                    dispatch.groundPointAndTimestep, float4(0.0f)).xyz, responseNormal);
                if (gap > standContactAdmissionDistanceMeters(support)) continue;
                for (uint dof = 0u; dof < nv; ++dof)
                    response[dof] = pointJacobianAxis(pointJacobians, pointJacobianBase,
                        support.pointQueryIndex, nv, dof, responseDirections[column % 3u]);
            } else if (column < equalityColumnsEnd) {
                device const auto& equality = jointEqualities[column - contactColumns];
                float target = 0.0f, derivative = 0.0f, error = 0.0f;
                if (!evaluateJointEquality(equality, qState, qBase, nq, nv,
                        target, derivative, error)) {
                    atomic_fetch_min_explicit(&responseFailure, column, memory_order_relaxed);
                    continue;
                }
                for (uint dof = 0u; dof < nv; ++dof) response[dof] = 0.0f;
                response[equality.indices.y] = 1.0f;
                if (equality.indices.w != MR_INVALID_INDEX) response[equality.indices.w] = -derivative;
            } else {
                const uint dof = column - equalityColumnsEnd;
                if (!contactEnabled ||
                    (dofs[articulation.vOffset + dof].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u)
                    continue;
                for (uint index = 0u; index < nv; ++index) response[index] = 0.0f;
                response[dof] = 1.0f;
            }
            if (!solveFactor(factor, independentWorkspace, response, nv, sparseGraph))
                atomic_fetch_min_explicit(&responseFailure, column, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u) {
        const uint failedResponse = atomic_load_explicit(&responseFailure, memory_order_relaxed);
        if (failedResponse != MR_INVALID_INDEX) {
            if (failedResponse < contactColumns)
                fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED, failedResponse / 3u);
            else if (failedResponse < equalityColumnsEnd) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, failedResponse - contactColumns);
            } else
                fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED, failedResponse - equalityColumnsEnd);

        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    const float timestep = dispatch.groundPointAndTimestep.w;
    float maximumEqualityPositionError = 0.0f;
    if (lane == 0u && equalityCount != 0u) {
        for (uint equalityIndex = 0u;
             equalityIndex < dispatch.jointEqualityCount;
             ++equalityIndex) {
            device const MRNumiHumanJointEqualityGPU& equality =
                jointEqualities[equalityIndex];
            float target = 0.0f;
            float derivative = 0.0f;
            float error = 0.0f;
            if (!evaluateJointEquality(
                    equality, qState, qBase, nq, nv,
                    target, derivative, error
                )) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                     equalityIndex);
                break;
            }
            equalityDerivativeCache[equalityIndex] = derivative;
            equalityTargetVelocityCache[equalityIndex] = clamp(
                -0.2f * error / timestep, -4.0f, 4.0f
            );
            maximumEqualityPositionError = max(
                maximumEqualityPositionError, abs(error)
            );

        }
        // Solve all bilateral rows together, not as scalar Gauss-Seidel
        // updates which can undo each other. Keep the actual nonsymmetric
        // FP32 contractions instead of silently adding diagonal compliance.
        for (uint row=0u; row<equalityCount &&
                status.code == MR_NUMI_HUMAN_STAND_SUCCESS; ++row) {
            device const MRNumiHumanJointEqualityGPU& equality=jointEqualities[row];
            const float derivative = equalityDerivativeCache[row];
            for (uint column=0u; column<equalityCount; ++column) {
                device const float* response=responseScratch+responseBase+
                    (3u*dispatch.supportContactCount+column)*nv;
                float value=response[equality.indices.y];
                if (equality.indices.w!=MR_INVALID_INDEX)
                    value=fma(-derivative,response[equality.indices.w],value);
                equalityFactor[row*equalityCount+column]=value;
            }
        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
            !mrNumiHumanBilateralFactor(equalityFactor,equalityScale,equalityPivots,equalityCount)) {
            fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,MR_INVALID_INDEX);

        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
            equalityCount <= kCachedEqualityCapacity) {
            for (uint index = 0u; index < equalityCount * equalityCount; ++index)
                equalityFactorStorage[index] = equalityFactor[index];
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    // Eliminate the bilateral response from each active contact column once.
    // The later contact sweeps can then update the same equality manifold
    // without resolving its dense block twice per sweep.
    if (useProjectedContacts && contactEnabled) {
        float independentRhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        for (uint column = lane; column < contactColumns;
             column += threadCount) {
            device const auto& support = contacts[column / 3u];
            const uint pointIndex = pointBase + support.pointQueryIndex;
            const float gap = dot(mrCompensatedPositionDifference(
                pointWorld[pointIndex].position,
                pointPositionLow[pointIndex],
                dispatch.groundPointAndTimestep, float4(0.0f)).xyz,
                dispatch.groundNormal.xyz);
            if (gap > standContactAdmissionDistanceMeters(support)) continue;
            device float* response = responseScratch + responseBase +
                column * nv;
            device float* reaction =
                projectedContactEquality + column * equalityCount;
            for (uint row = 0u; row < equalityCount; ++row)
                reaction[row] = 0.0f;
            bool valid = true;
            for (uint refinement = 0u; refinement < 2u && valid;
                 ++refinement) {
                for (uint row = 0u; row < equalityCount; ++row) {
                    device const auto& equality = jointEqualities[row];
                    float residual = response[equality.indices.y];
                    if (equality.indices.w != MR_INVALID_INDEX)
                        residual = fma(-equalityDerivativeCache[row],
                            response[equality.indices.w], residual);
                    independentRhs[row] = residual;
                }
                const bool solved = equalityCount <= kCachedEqualityCapacity
                    ? mrNumiHumanBilateralSolve(equalityFactorStorage,
                        equalityScale, equalityPivots, independentRhs, equalityCount)
                    : mrNumiHumanBilateralSolve(equalityFactor,
                        equalityScale, equalityPivots, independentRhs, equalityCount);
                if (!solved) {
                    valid = false;
                    break;
                }
                for (uint row = 0u; row < equalityCount; ++row)
                    reaction[row] -= independentRhs[row];
                for (uint dof = 0u; dof < nv; ++dof) {
                    float correction = 0.0f;
                    for (uint row = 0u; row < equalityCount; ++row) {
                        device const float* equalityResponse =
                            responseScratch + responseBase +
                            (contactColumns + row) * nv;
                        correction = fma(independentRhs[row],
                            equalityResponse[dof], correction);
                    }
                    response[dof] -= correction;
                    if (!isfinite(response[dof])) {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid)
                atomic_fetch_min_explicit(&responseFailure, column,
                    memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint failed = atomic_load_explicit(&responseFailure,
                memory_order_relaxed);
            if (failed != MR_INVALID_INDEX)
                fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED,
                    failed / 3u);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    }
    // Condition source-limit responses independently, preserving the original
    // two refinement passes and each column's exact reduction order. The
    // subsequent coupled impulse sweeps retain their sequential ordering.
    if (contactEnabled) {
        float independentRaw[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        float independentRhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        for (uint dof = lane; dof < nv; dof += threadCount) {
            if ((dofs[articulation.vOffset + dof].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u)
                continue;
            device float* response = responseScratch + responseBase +
                (equalityColumnsEnd + dof) * nv;
            bool validConditioning = true;
            // Eliminate ALL bilateral rows from this limit's response. A pair
            // correction can satisfy one equality while violating another. These
            // columns use the already factored E M_eff^-1 E^T; no new global
            // solve or artificial compliance is introduced.
            device float* equalityCorrection = limitEqualityCorrections +
                dof * equalityCount;
            for (uint ei = 0u; ei < equalityCount; ++ei)
                equalityCorrection[ei] = 0.0f;
            if (equalityCount != 0u) {
                const float rawDiagonal = response[dof];
                // Preserve the raw column in private per-lane scratch
                // for unresolved/rank-dependent directions without dropping rows.
                for (uint index = 0u; index < nv; ++index)
                    independentRaw[index] = response[index];
                for (uint refinement = 0u; refinement < 2u && validConditioning; ++refinement) {
                    for (uint ei = 0u; ei < equalityCount; ++ei) {
                        device const MRNumiHumanJointEqualityGPU& eq = jointEqualities[ei];
                        const float derivative = equalityDerivativeCache[ei];
                        float residual = response[eq.indices.y];
                        if (eq.indices.w != MR_INVALID_INDEX)
                            residual = fma(-derivative, response[eq.indices.w], residual);
                        independentRhs[ei] = residual;
                    }
                    const bool equalitySolved = equalityCount <= kCachedEqualityCapacity
                        ? mrNumiHumanBilateralSolve(equalityFactorStorage, equalityScale,
                            equalityPivots, independentRhs, equalityCount)
                        : mrNumiHumanBilateralSolve(equalityFactor, equalityScale,
                            equalityPivots, independentRhs, equalityCount);
                    if (!equalitySolved) {
                        atomic_fetch_min_explicit(&responseFailure, 2u * dof, memory_order_relaxed);
                        validConditioning = false;
                        break;
                    }
                    for (uint ei = 0u; ei < equalityCount; ++ei)
                        equalityCorrection[ei] -= independentRhs[ei];
                    for (uint index = 0u; index < nv; ++index) {
                        float correction = 0.0f;
                        for (uint ei = 0u; ei < equalityCount; ++ei) {
                            device const float* er = responseScratch + responseBase +
                                (3u * dispatch.supportContactCount + ei) * nv;
                            correction = fma(independentRhs[ei], er[index], correction);
                        }
                        response[index] -= correction;
                        if (!isfinite(response[index])) {
                            atomic_fetch_min_explicit(&responseFailure, 2u * dof + 1u, memory_order_relaxed);
                            validConditioning = false;
                            break;
                        }
                    }
                }
                // Cancellation in a direction already fixed by E must not be
                // inverted as a new independent limit. Retain the original scalar
                // coupled update for unresolved directions; do not manufacture
                // response with a diagonal floor, omit the row, or loosen gates.
                if (validConditioning && !(response[dof] > 1.0e-6f * rawDiagonal)) {
                    for (uint index = 0u; index < nv; ++index)
                        response[index] = independentRaw[index];
                    for (uint ei = 0u; ei < equalityCount; ++ei)
                        equalityCorrection[ei] = 0.0f;
                }
            }

        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // Prepare once while all lanes are available. The scalar coupled solve
    // otherwise fetches one value from each nv-strided response column for
    // every DOF on every sweep.
    if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        cacheEqualityResponseByDof) {
        for (uint dof = lane; dof < nv; dof += threadCount) {
            for (uint row = 0u; row < equalityCount; ++row) {
                equalityResponseByDof[dof * equalityCount + row] =
                    responseScratch[responseBase +
                        (3u * dispatch.supportContactCount + row) * nv + dof];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // An optional split owner keeps prepared matrices and response columns on
    // the device for a following completion kernel in this command buffer.
    // Report response failures before ending this phase; no state is advanced.
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREPARE_ONLY) != 0u) {
        if (lane == 0u) {
            const uint failedCondition = atomic_load_explicit(
                &responseFailure, memory_order_relaxed);
            if (failedCondition != MR_INVALID_INDEX) {
                fail(status, (failedCondition & 1u) == 0u
                    ? MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED
                    : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT,
                    failedCondition / 2u);
            }
        }
        return;
    }
#include "NumiHumanStandSolve.metalinc"

}

// Each inverse-mass response is independent after the current mass factor is
// complete. Dispatch columns over the GPU instead of solving every RHS in the
// single preparation threadgroup. A failed column atomically records the
// lowest source ordinal; the following preparation phase maps it to the same
// typed stand failure before any accepted q/v state can advance.
kernel void mr_numi_human_stand_response_assemble(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device const float* factorScratch [[buffer(14)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint environment = position.y;
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS ||
        (dispatch.flags & (MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
                           MR_NUMI_HUMAN_STAND_MASS_READY |
                           MR_NUMI_HUMAN_STAND_FACTOR_ONLY)) !=
            (MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
             MR_NUMI_HUMAN_STAND_MASS_READY |
             MR_NUMI_HUMAN_STAND_FACTOR_ONLY)) return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint equalityColumnsEnd = contactColumns + equalityCount;
    if (position.x >= equalityCount) return;
    const uint column = contactColumns + position.x;
    const bool contactEnabled =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u;
    if (!contactEnabled && equalityCount == 0u) return;
    const uint responseColumns = (equalityColumnsEnd + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    device float* response = responseScratch +
        environment * responseStride + column * nv;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase =
        environment * dispatch.pointJacobianStride;
    const uint qBase = environment * dispatch.qStride;
    if (column < contactColumns) {
        if (!contactEnabled) return;
        device const auto& support = contacts[column / 3u];
        const uint pointIndex = pointBase + support.pointQueryIndex;
        const float3 normal = dispatch.groundNormal.xyz;
        const float gap = dot(mrCompensatedPositionDifference(
            pointWorld[pointIndex].position, pointPositionLow[pointIndex],
            dispatch.groundPointAndTimestep, float4(0.0f)).xyz, normal);
        if (gap > standContactAdmissionDistanceMeters(support)) return;
        const float3 reference = abs(normal.x) < 0.8f
            ? float3(1.0f, 0.0f, 0.0f)
            : float3(0.0f, 1.0f, 0.0f);
        const float3 tangent0 = normalize(
            reference - dot(reference, normal) * normal);
        const uint axis = column % 3u;
        const float3 direction = axis == 0u ? normal :
            axis == 1u ? tangent0 : cross(normal, tangent0);
        for (uint dof = 0u; dof < nv; ++dof)
            response[dof] = pointJacobianAxis(pointJacobians,
                pointJacobianBase, support.pointQueryIndex, nv, dof,
                direction);
    } else if (column < equalityColumnsEnd) {
        device const auto& equality =
            jointEqualities[column - contactColumns];
        float target = 0.0f, derivative = 0.0f, error = 0.0f;
        if (!evaluateJointEquality(equality, qState, qBase,
                articulation.nq, nv, target, derivative, error)) {
            device atomic_uint* failure =
                reinterpret_cast<device atomic_uint*>(&status.failingIndex);
            atomic_fetch_min_explicit(failure, column,
                                      memory_order_relaxed);
            return;
        }
        for (uint dof = 0u; dof < nv; ++dof) response[dof] = 0.0f;
        response[equality.indices.y] = 1.0f;
        if (equality.indices.w != MR_INVALID_INDEX)
            response[equality.indices.w] = -derivative;
    } else {
        const uint dof = column - equalityColumnsEnd;
        if (!contactEnabled ||
            (dofs[articulation.vOffset + dof].flags &
             MR_DOF_FLAG_POSITION_LIMIT) == 0u) return;
        for (uint index = 0u; index < nv; ++index) response[index] = 0.0f;
        response[dof] = 1.0f;
    }
    float workspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    if (!solveFactor(factorScratch + environment * nv * nv,
                     workspace, response, nv, sparseGraph)) {
        device atomic_uint* failure =
            reinterpret_cast<device atomic_uint*>(&status.failingIndex);
        atomic_fetch_min_explicit(failure, column, memory_order_relaxed);
    }
}

// One threadgroup owns each independent equality response. The ordered
// triangular dependencies stay intact while future forward rows share the
// current block across SIMD lanes.
kernel void mr_numi_human_stand_equality_response_cooperative(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device const float* factorScratch [[buffer(14)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 groupSize [[threads_per_threadgroup]]
) {
    const uint environment = group.y;
    const uint equalityIndex = group.x;
    const uint threadCount = groupSize.x;
    if (environment >= dispatch.environmentCount ||
        equalityIndex >= dispatch.jointEqualityCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS ||
        (dispatch.flags & (MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
                           MR_NUMI_HUMAN_STAND_MASS_READY |
                           MR_NUMI_HUMAN_STAND_FACTOR_ONLY)) !=
            (MR_NUMI_HUMAN_STAND_PREPARE_ONLY |
             MR_NUMI_HUMAN_STAND_MASS_READY |
             MR_NUMI_HUMAN_STAND_FACTOR_ONLY)) return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint column = contactColumns + equalityIndex;
    const uint responseColumns =
        (contactColumns + equalityCount + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    device float* response = responseScratch +
        environment * responseStride + column * nv;
    threadgroup float rhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup uint rhsValid = 0u;
    threadgroup uint solveSucceeded;
    for (uint dof = lane; dof < nv; dof += threadCount)
        rhs[dof] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        device const auto& equality = jointEqualities[equalityIndex];
        float target = 0.0f, derivative = 0.0f, error = 0.0f;
        rhsValid = evaluateJointEquality(equality, qState,
            environment * dispatch.qStride, articulation.nq, nv,
            target, derivative, error) ? 1u : 0u;
        if (rhsValid != 0u) {
            rhs[equality.indices.y] = 1.0f;
            if (equality.indices.w != MR_INVALID_INDEX)
                rhs[equality.indices.w] = -derivative;
        } else {
            device atomic_uint* failure =
                reinterpret_cast<device atomic_uint*>(&status.failingIndex);
            atomic_fetch_min_explicit(failure, column,
                                      memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rhsValid == 0u) return;
    const bool solved = solveFactorCooperativeForward(
        factorScratch + environment * nv * nv, workspace, rhs,
        nv, lane, threadCount, &solveSucceeded, sparseGraph);
    if (!solved) {
        if (lane == 0u) {
            device atomic_uint* failure =
                reinterpret_cast<device atomic_uint*>(&status.failingIndex);
            atomic_fetch_min_explicit(failure, column,
                                      memory_order_relaxed);
        }
        return;
    }
    for (uint dof = lane; dof < nv; dof += threadCount)
        response[dof] = rhs[dof];
}

// The equality block depends on all equality response columns. Factor it
// once, publish the source linearization, and transpose the equality columns
// into the layout consumed by every coupled sweep.
// The source-A snapshot and this factor are per-environment scratch written
// in phase 1, consumed by phases 4/5 of the same root, then overwritten by the
// next phase-1 assembly. They never serve as persistent physical state.
inline bool prepareReducedStandProjection(
    device const float* sourceA,
    device const float* derivativeCache,
    device const MRNumiHumanJointEqualityGPU* jointEqualities,
    const uint nv,
    const uint equalityCount,
    const bool upperTriangle,
    const uint lane,
    const uint threadCount,
    device float* reducedFactor,
    device float* reducedScale,
    device float* reducedPivots,
    device uint* coordinateForDof,
    device float* coefficientForDof,
    device uint* coordinateOffsets,
    device uint* coordinateDofs,
    device uint* ready,
    threadgroup float* factorCache,
    threadgroup float* scaleCache,
    threadgroup float* pivotCache,
    threadgroup atomic_uint* factorFailure,
    threadgroup uint* selectedPivot
) {
    const uint freeDofs =
        equalityCount <= nv ? nv - equalityCount : 0u;
    if (lane == 0u) {
        *ready = 0u;
        atomic_store_explicit(factorFailure, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (equalityCount > nv || freeDofs == 0u || freeDofs > 64u)
        return false;

    // Reuse the factor cache before factorization for immutable dependency
    // tags. Each DOF has one writer. The scans retain the authored equality
    // order, but independent DOFs no longer serialize on lane zero.
    threadgroup uint* dependencyRows =
        reinterpret_cast<threadgroup uint*>(factorCache);
    for (uint dof = lane; dof < nv; dof += threadCount) {
        uint dependentRow = MR_INVALID_INDEX;
        for (uint row = 0u; row < equalityCount; ++row) {
            if (jointEqualities[row].indices.y != dof) continue;
            if (dependentRow != MR_INVALID_INDEX)
                atomic_store_explicit(factorFailure, 1u, memory_order_relaxed);
            dependentRow = row;
        }
        dependencyRows[dof] = dependentRow;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = lane; row < equalityCount; row += threadCount) {
        device const auto& equality = jointEqualities[row];
        const uint master = equality.indices.w;
        const bool fixed = equality.indices.z == MR_INVALID_INDEX &&
            master == MR_INVALID_INDEX;
        const bool coupled = equality.indices.z != MR_INVALID_INDEX &&
            master < nv;
        if (equality.indices.y >= nv || (!fixed && !coupled) ||
            (coupled && (dependencyRows[master] != MR_INVALID_INDEX ||
                         !isfinite(derivativeCache[row]))))
            atomic_store_explicit(factorFailure, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(factorFailure, memory_order_relaxed) != 0u)
        return false;

    for (uint dof = lane; dof < nv; dof += threadCount) {
        const uint row = dependencyRows[dof];
        const uint master = row == MR_INVALID_INDEX
            ? dof : jointEqualities[row].indices.w;
        uint coordinate = MR_INVALID_INDEX;
        float coefficient = 0.0f;
        if (master != MR_INVALID_INDEX) {
            coordinate = 0u;
            for (uint predecessor = 0u; predecessor < master; ++predecessor)
                coordinate += dependencyRows[predecessor] == MR_INVALID_INDEX
                    ? 1u : 0u;
            coefficient = row == MR_INVALID_INDEX
                ? 1.0f : derivativeCache[row];
        }
        coordinateForDof[dof] = coordinate;
        coefficientForDof[dof] = coefficient;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // A coordinate offset is the count of earlier active source DOFs.
    // These integer counts exactly match the previous serial prefix sum.
    for (uint coordinate = lane; coordinate <= freeDofs;
         coordinate += threadCount) {
        uint offset = 0u;
        for (uint dof = 0u; dof < nv; ++dof)
            offset += coordinateForDof[dof] < coordinate ? 1u : 0u;
        coordinateOffsets[coordinate] = offset;
    }
    if (lane == 0u) *ready = 1u;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    // Each coordinate owns a disjoint packed range. Preserve ascending source
    // DOF order while distributing independent scans across the existing group.
    for (uint coordinateIndex = lane; coordinateIndex < freeDofs;
         coordinateIndex += threadCount) {
        uint output = coordinateOffsets[coordinateIndex];
        for (uint dof = 0u; dof < nv; ++dof)
            if (coordinateForDof[dof] == coordinateIndex)
                coordinateDofs[output++] = dof;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    const uint matrixElements = freeDofs * freeDofs;
    for (uint index = lane; index < matrixElements; index += threadCount) {
        const uint row = index / freeDofs;
        const uint column = index - row * freeDofs;
        if (row < column) continue;
        float value = 0.0f;
        for (uint rowIndex = coordinateOffsets[row];
             rowIndex < coordinateOffsets[row + 1u]; ++rowIndex) {
            const uint sourceRow = coordinateDofs[rowIndex];
            const float rowCoefficient = coefficientForDof[sourceRow];
            for (uint columnIndex = coordinateOffsets[column];
                 columnIndex < coordinateOffsets[column + 1u];
                 ++columnIndex) {
                const uint sourceColumn = coordinateDofs[columnIndex];
                const float sourceValue = standSelectedSourceA(
                    sourceA, nv, sourceRow, sourceColumn, upperTriangle);
                value = fma(rowCoefficient * coefficientForDof[sourceColumn],
                            sourceValue, value);
            }
        }
        reducedFactor[row * freeDofs + column] = value;
        reducedFactor[column * freeDofs + row] = value;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool factored = false;
    if (kUseReducedStandCholesky)
        factored = mrNumiHumanReducedCholeskyFactorCooperative(
            reducedFactor, reducedScale, reducedPivots, freeDofs, lane,
            threadCount, factorCache, scaleCache, factorFailure);
    if (!factored)
        factored = mrNumiHumanBilateralFactorCooperative<false>(
            reducedFactor, reducedScale, reducedPivots, freeDofs, lane,
            threadCount, factorCache, scaleCache, pivotCache, factorFailure,
            selectedPivot);
    if (lane == 0u) *ready = factored ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return *ready != 0u;
}

kernel void mr_numi_human_stand_equality_prepare(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const float* qState [[buffer(5)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    threadgroup float* factorCache [[threadgroup(0)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadCount [[threads_per_threadgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint nq = articulation.nq;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    if (status.failingIndex != MR_INVALID_INDEX) {
        if (lane == 0u) {
            ++status.jointEqualityCounts.z;
            fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                 status.failingIndex - contactColumns);
        }
        return;
    }
    if (equalityCount == 0u) return;
    const uint responseColumns =
        (contactColumns + equalityCount + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    const uint responseBase = environment * responseStride;
    device float* equalityFactor =
        responseScratch + responseBase + responseColumns;
    device float* equalityScale =
        equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    threadgroup float scaleCache[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float pivotCache[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup atomic_uint factorFailure;
    threadgroup uint selectedPivot;
    const uint bodyCount = articulation.bodyCount;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    device float* derivativeCache =
        spatialJacobianScratch + spatialBase;
    device float* targetVelocityCache = derivativeCache + nv;
    device float* equalityResponseByDof = targetVelocityCache + nv;
    const uint qBase = environment * dispatch.qStride;
    const float timestep = dispatch.groundPointAndTimestep.w;
    if (lane == 0u) {
        for (uint equalityIndex = 0u; equalityIndex < equalityCount;
             ++equalityIndex) {
            float target = 0.0f, derivative = 0.0f, error = 0.0f;
            if (!evaluateJointEquality(jointEqualities[equalityIndex],
                    qState, qBase, nq, nv, target, derivative, error)) {
                ++status.jointEqualityCounts.z;
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                     equalityIndex);
                break;
            }
            derivativeCache[equalityIndex] = derivative;
            targetVelocityCache[equalityIndex] = clamp(
                -0.2f * error / timestep, -4.0f, 4.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    for (uint row = lane; row < equalityCount; row += threadCount) {
        device const auto& equality = jointEqualities[row];
        const float derivative = derivativeCache[row];
        for (uint column = 0u; column < equalityCount; ++column) {
            device const float* response = responseScratch + responseBase +
                (contactColumns + column) * nv;
            float value = response[equality.indices.y];
            if (equality.indices.w != MR_INVALID_INDEX)
                value = fma(-derivative,
                            response[equality.indices.w], value);
            equalityFactor[row * equalityCount + column] = value;
        }
    }
    if (bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS >=
        2u + equalityCount) {
        for (uint dof = lane; dof < nv; dof += threadCount) {
            for (uint row = 0u; row < equalityCount; ++row) {
                equalityResponseByDof[dof * equalityCount + row] =
                    responseScratch[responseBase +
                        (contactColumns + row) * nv + dof];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    const bool useHybridEqualityFactorCache =
        kUseHybridEqualityFactorCache &&
        equalityCount > kCachedEqualityCapacity && equalityCount <= 96u;
    if (useHybridEqualityFactorCache) {
        if (!mrNumiHumanBilateralFactorCooperativeWorkspace<false, true>(
                equalityFactor, equalityScale, equalityPivots,
                equalityCount, lane, threadCount, factorCache,
                equalityFactor, scaleCache, pivotCache, &factorFailure,
                &selectedPivot) && lane == 0u) {
            fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                 MR_INVALID_INDEX);
        }
    } else if (equalityCount <= kCachedEqualityCapacity) {
        if (!mrNumiHumanBilateralFactorCooperative<false>(
                equalityFactor, equalityScale, equalityPivots,
                equalityCount, lane, threadCount, factorCache,
                scaleCache, pivotCache, &factorFailure, &selectedPivot) &&
            lane == 0u) {
            fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                 MR_INVALID_INDEX);
        }
    } else if (equalityCount <= MR_NUMI_HUMAN_STAND_MAX_DOFS) {
        if (!mrNumiHumanBilateralFactorCooperative<true>(
                equalityFactor, equalityScale, equalityPivots,
                equalityCount, lane, threadCount, equalityFactor,
                scaleCache, pivotCache, &factorFailure, &selectedPivot) &&
            lane == 0u) {
            fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                 MR_INVALID_INDEX);
        }
    } else if (lane == 0u &&
               !mrNumiHumanBilateralFactor(
                   equalityFactor, equalityScale, equalityPivots,
                   equalityCount)) {
        fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
             MR_INVALID_INDEX);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    if (lane == 0u && useHybridEqualityFactorCache) {
        device atomic_uint* statusFlags =
            reinterpret_cast<device atomic_uint*>(&status.flags);
        atomic_fetch_or_explicit(statusFlags,
            uint(MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_USED),
            memory_order_relaxed);
    }
    if ((dispatch.flags &
         MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES) == 0u)
        return;
    const uint legacyStride = standLegacyResponseStride(
        nv, dispatch.supportContactCount, equalityCount);
    device float* reducedArena = responseScratch + responseBase + legacyStride;
    const uint freeDofs = nv - equalityCount;
    device const float* sourceA = reducedArena;
    device float* reducedFactor = responseScratch + responseBase +
        legacyStride + nv * nv;
    device float* reducedScale = reducedFactor + freeDofs * freeDofs;
    device float* reducedPivots = reducedScale + freeDofs;
    device uint* coordinateForDof =
        reinterpret_cast<device uint*>(reducedPivots + freeDofs);
    device float* coefficientForDof =
        reinterpret_cast<device float*>(coordinateForDof + nv);
    device uint* coordinateOffsets =
        reinterpret_cast<device uint*>(coefficientForDof + nv);
    device uint* coordinateDofs = coordinateOffsets + freeDofs + 1u;
    device uint* reducedReady = coordinateDofs + nv;
    const bool upperTriangle =
        (dispatch.flags &
         MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE) != 0u;
    const bool reducedReadyForRoot = prepareReducedStandProjection(
        sourceA, derivativeCache, jointEqualities, nv, equalityCount,
        upperTriangle, lane, threadCount, reducedFactor, reducedScale,
        reducedPivots,
        coordinateForDof, coefficientForDof, coordinateOffsets,
        coordinateDofs, reducedReady, factorCache, scaleCache, pivotCache,
        &factorFailure, &selectedPivot);
    if (lane == 0u && reducedReadyForRoot) {
        device atomic_uint* statusFlags =
            reinterpret_cast<device atomic_uint*>(&status.flags);
        atomic_fetch_or_explicit(statusFlags,
            uint(MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_READY),
            memory_order_relaxed);
    }
}

// Contact and position-limit responses have no cross-column dependency after
// the equality block is factored. Each grid lane builds its own mass response
// and removes the equality component in the same ordered two-pass solve.
kernel void mr_numi_human_stand_projected_response_assemble(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device const float* factorScratch [[buffer(14)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint environment = position.y;
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint equalityColumnsEnd = contactColumns + equalityCount;
    const uint positionIndex = position.x;
    if (positionIndex >= contactColumns + nv) return;
    const uint column = positionIndex < contactColumns
        ? positionIndex : equalityColumnsEnd + positionIndex - contactColumns;
    const uint responseColumns = (equalityColumnsEnd + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    const uint responseBase = environment * responseStride;
    device float* response = responseScratch + responseBase + column * nv;
    device float* equalityFactor =
        responseScratch + responseBase + responseColumns;
    device float* equalityScale =
        equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    device float* limitEqualityCorrections =
        equalityPivots + 2u * equalityCount;
    const uint bodyCount = articulation.bodyCount;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    device const float* derivativeCache =
        spatialJacobianScratch + spatialBase;
    device float* projectedContactEquality =
        spatialJacobianScratch + spatialBase +
        (2u + equalityCount) * nv;
    const bool contactEnabled =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u;
    const bool useProjectedContacts =
        equalityCount != 0u && equalityCount <= MR_NUMI_HUMAN_STAND_MAX_DOFS &&
        contactEnabled && dispatch.supportContactCount != 0u &&
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv >=
            (2u + equalityCount) * nv +
            3u * dispatch.supportContactCount * equalityCount;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase =
        environment * dispatch.pointJacobianStride;
    if (positionIndex < contactColumns) {
        if (!contactEnabled) return;
        device const auto& support = contacts[positionIndex / 3u];
        const uint pointIndex = pointBase + support.pointQueryIndex;
        const float3 normal = dispatch.groundNormal.xyz;
        const float gap = dot(mrCompensatedPositionDifference(
            pointWorld[pointIndex].position, pointPositionLow[pointIndex],
            dispatch.groundPointAndTimestep, float4(0.0f)).xyz, normal);
        if (gap > standContactAdmissionDistanceMeters(support)) return;
        const float3 reference = abs(normal.x) < 0.8f
            ? float3(1.0f, 0.0f, 0.0f)
            : float3(0.0f, 1.0f, 0.0f);
        const float3 tangent0 = normalize(
            reference - dot(reference, normal) * normal);
        const uint axis = positionIndex % 3u;
        const float3 direction = axis == 0u ? normal :
            axis == 1u ? tangent0 : cross(normal, tangent0);
        for (uint dof = 0u; dof < nv; ++dof)
            response[dof] = pointJacobianAxis(pointJacobians,
                pointJacobianBase, support.pointQueryIndex, nv, dof,
                direction);
    } else {
        const uint dof = positionIndex - contactColumns;
        if (!contactEnabled ||
            (dofs[articulation.vOffset + dof].flags &
             MR_DOF_FLAG_POSITION_LIMIT) == 0u) return;
        for (uint index = 0u; index < nv; ++index) response[index] = 0.0f;
        response[dof] = 1.0f;
    }
    float workspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    if (!solveFactor(factorScratch + environment * nv * nv,
                     workspace, response, nv, sparseGraph)) {
        publishParallelResponseFailure(status, column);
        return;
    }
    if (positionIndex < contactColumns) {
        if (!useProjectedContacts) return;
        device float* reaction = projectedContactEquality +
            positionIndex * equalityCount;
        for (uint row = 0u; row < equalityCount; ++row)
            reaction[row] = 0.0f;
        float rhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
        bool valid = true;
        for (uint refinement = 0u; refinement < 2u && valid;
             ++refinement) {
            for (uint row = 0u; row < equalityCount; ++row) {
                device const auto& equality = jointEqualities[row];
                float residual = response[equality.indices.y];
                if (equality.indices.w != MR_INVALID_INDEX)
                    residual = fma(-derivativeCache[row],
                        response[equality.indices.w], residual);
                rhs[row] = residual;
            }
            if (!mrNumiHumanBilateralSolve(equalityFactor,
                    equalityScale, equalityPivots, rhs,
                    equalityCount)) {
                valid = false;
                break;
            }
            for (uint row = 0u; row < equalityCount; ++row)
                reaction[row] -= rhs[row];
            for (uint dof = 0u; dof < nv; ++dof) {
                float correction = 0.0f;
                for (uint row = 0u; row < equalityCount; ++row) {
                    device const float* equalityResponse =
                        responseScratch + responseBase +
                        (contactColumns + row) * nv;
                    correction = fma(rhs[row],
                        equalityResponse[dof], correction);
                }
                response[dof] -= correction;
                if (!isfinite(response[dof])) {
                    valid = false;
                    break;
                }
            }
        }
        if (!valid)
            publishParallelResponseFailure(status,
                kParallelContactConditionFailure + positionIndex);
        return;
    }
    const uint dof = positionIndex - contactColumns;
    device float* equalityCorrection =
        limitEqualityCorrections + dof * equalityCount;
    for (uint row = 0u; row < equalityCount; ++row)
        equalityCorrection[row] = 0.0f;
    if (equalityCount == 0u) return;
    const float rawDiagonal = response[dof];
    float rawResponse[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    for (uint index = 0u; index < nv; ++index)
        rawResponse[index] = response[index];
    float rhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    bool valid = true;
    for (uint refinement = 0u; refinement < 2u && valid;
         ++refinement) {
        for (uint row = 0u; row < equalityCount; ++row) {
            device const auto& equality = jointEqualities[row];
            float residual = response[equality.indices.y];
            if (equality.indices.w != MR_INVALID_INDEX)
                residual = fma(-derivativeCache[row],
                               response[equality.indices.w], residual);
            rhs[row] = residual;
        }
        if (!mrNumiHumanBilateralSolve(equalityFactor, equalityScale,
                equalityPivots, rhs, equalityCount)) {
            publishParallelResponseFailure(status,
                kParallelLimitConditionFailure + 2u * dof);
            valid = false;
            break;
        }
        for (uint row = 0u; row < equalityCount; ++row)
            equalityCorrection[row] -= rhs[row];
        for (uint index = 0u; index < nv; ++index) {
            float correction = 0.0f;
            for (uint row = 0u; row < equalityCount; ++row) {
                device const float* equalityResponse =
                    responseScratch + responseBase +
                    (contactColumns + row) * nv;
                correction = fma(rhs[row],
                    equalityResponse[index], correction);
            }
            response[index] -= correction;
            if (!isfinite(response[index])) {
                publishParallelResponseFailure(status,
                    kParallelLimitConditionFailure + 2u * dof + 1u);
                valid = false;
                break;
            }
        }
    }
    if (valid && !(response[dof] > 1.0e-6f * rawDiagonal)) {
        for (uint index = 0u; index < nv; ++index)
            response[index] = rawResponse[index];
        for (uint row = 0u; row < equalityCount; ++row)
            equalityCorrection[row] = 0.0f;
    }
}

// Contact and limit right hand sides are independent after the equality
// factor is published. One SIMD group performs each ordered mass solve;
// lane zero retains the authored two-pass equality conditioning for that row.
kernel void mr_numi_human_stand_projected_response_cooperative(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device const float* factorScratch [[buffer(14)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    device uint* reducedResponseDiagnostics [[buffer(29), function_constant(kUseReducedResponseDiagnostics)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 groupSize [[threads_per_threadgroup]]
) {
    const uint environment = group.y;
    const uint positionIndex = group.x;
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint contactColumns = 3u * dispatch.supportContactCount;
    const uint equalityColumnsEnd = contactColumns + equalityCount;
    if (positionIndex >= contactColumns + nv) return;
    const uint column = positionIndex < contactColumns
        ? positionIndex : equalityColumnsEnd + positionIndex - contactColumns;
    const uint responseColumns = (equalityColumnsEnd + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    const uint responseBase = environment * responseStride;
    device float* response = responseScratch + responseBase + column * nv;
    device float* equalityFactor =
        responseScratch + responseBase + responseColumns;
    device float* equalityScale =
        equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    device float* limitEqualityCorrections =
        equalityPivots + 2u * equalityCount;
    const uint spatialBase = environment * articulation.bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    device const float* derivativeCache =
        spatialJacobianScratch + spatialBase;
    device float* projectedContactEquality =
        spatialJacobianScratch + spatialBase +
        (2u + equalityCount) * nv;
    const bool contactEnabled =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u;
    const bool useProjectedContacts =
        equalityCount != 0u && equalityCount <= MR_NUMI_HUMAN_STAND_MAX_DOFS &&
        contactEnabled && dispatch.supportContactCount != 0u &&
        articulation.bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv >=
            (2u + equalityCount) * nv +
            3u * dispatch.supportContactCount * equalityCount;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase =
        environment * dispatch.pointJacobianStride;
    const uint threadCount = groupSize.x;
    threadgroup float rhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspace[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float equalityRhs[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float rawResponse[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float reducedResponse[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float reducedLift[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float3 direction;
    threadgroup uint responseActive = 0u;
    threadgroup uint conditionValid = 1u;
    threadgroup uint restoreRawResponse = 0u;
    threadgroup uint solveSucceeded = 0u;
    threadgroup uint reducedSolveSucceeded = 0u;
    threadgroup atomic_uint correctionFailed;
    threadgroup atomic_uint reducedFailure;
    if (lane == 0u) {
        if (positionIndex < contactColumns) {
            if (contactEnabled) {
                device const auto& support = contacts[positionIndex / 3u];
                const uint pointIndex = pointBase + support.pointQueryIndex;
                const float3 normal = dispatch.groundNormal.xyz;
                const float gap = dot(mrCompensatedPositionDifference(
                    pointWorld[pointIndex].position, pointPositionLow[pointIndex],
                    dispatch.groundPointAndTimestep, float4(0.0f)).xyz,
                    normal);
                if (gap <= standContactAdmissionDistanceMeters(support)) {
                    const float3 reference = abs(normal.x) < 0.8f
                        ? float3(1.0f, 0.0f, 0.0f)
                        : float3(0.0f, 1.0f, 0.0f);
                    const float3 tangent0 = normalize(
                        reference - dot(reference, normal) * normal);
                    direction = positionIndex % 3u == 0u ? normal :
                        positionIndex % 3u == 1u ? tangent0 :
                        cross(normal, tangent0);
                    responseActive = 1u;
                }
            }
        } else {
            const uint dof = positionIndex - contactColumns;
            responseActive = contactEnabled &&
                (dofs[articulation.vOffset + dof].flags &
                 MR_DOF_FLAG_POSITION_LIMIT) != 0u ? 1u : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (responseActive == 0u) return;
    const bool projectedRawReady =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_PROJECTED_RAW_READY) != 0u;
    const bool reducedResponseRequested =
        (dispatch.flags &
         MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES) != 0u;
    for (uint dof = lane; dof < nv; dof += threadCount) {
        rhs[dof] = projectedRawReady ? response[dof] :
            positionIndex < contactColumns
                ? pointJacobianAxis(pointJacobians, pointJacobianBase,
                    contacts[positionIndex / 3u].pointQueryIndex, nv, dof,
                    direction)
                : (dof == positionIndex - contactColumns ? 1.0f : 0.0f);
        if (reducedResponseRequested && !projectedRawReady &&
            positionIndex < contactColumns)
            rawResponse[dof] = rhs[dof];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!projectedRawReady && !solveFactorCooperativeForward(
            factorScratch + environment * nv * nv,
            workspace, rhs, nv, lane, threadCount, &solveSucceeded, sparseGraph)) {
        if (lane == 0u) publishParallelResponseFailure(status, column);
        return;
    }
    const bool contactColumn = positionIndex < contactColumns;
    const bool projectEquality = contactColumn
        ? useProjectedContacts : equalityCount != 0u;
    const uint limitDof = contactColumn
        ? 0u : positionIndex - contactColumns;
    device float* reaction = projectedContactEquality +
        (contactColumn ? positionIndex : 0u) * equalityCount;
    device float* equalityCorrection =
        limitEqualityCorrections + limitDof * equalityCount;
    for (uint row = lane; row < equalityCount; row += threadCount) {
        if (contactColumn && useProjectedContacts) reaction[row] = 0.0f;
        if (!contactColumn) equalityCorrection[row] = 0.0f;
    }
    if (!contactColumn && projectEquality)
        for (uint dof = lane; dof < nv; dof += threadCount)
            rawResponse[dof] = rhs[dof];
    const float rawDiagonal = !contactColumn && projectEquality
        ? rhs[limitDof] : 0.0f;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    bool useReducedResponse = false;
    bool diagnosticAttempted = false;
    bool diagnosticWeakFallback = false;
    bool diagnosticOtherFallback = false;
    float diagnosticReducedDiagonal = 0.0f;
    if (reducedResponseRequested && projectEquality && !projectedRawReady) {
        if (kUseReducedResponseDiagnostics && !contactColumn)
            diagnosticAttempted = true;
        const uint legacyStride = standLegacyResponseStride(
            nv, dispatch.supportContactCount, equalityCount);
        device float* reducedArena =
            responseScratch + responseBase + legacyStride;
        const uint freeDofs = nv - equalityCount;
        device const float* sourceA = reducedArena;
        device float* reducedFactor = reducedArena + nv * nv;
        device const float* reducedScale =
            reducedFactor + freeDofs * freeDofs;
        device const float* reducedPivots = reducedScale + freeDofs;
        device const uint* coordinateForDof =
            reinterpret_cast<device const uint*>(reducedPivots + freeDofs);
        device const float* coefficientForDof =
            reinterpret_cast<device const float*>(coordinateForDof + nv);
        device const uint* coordinateOffsets =
            reinterpret_cast<device const uint*>(coefficientForDof + nv);
        device const uint* coordinateDofs = coordinateOffsets + freeDofs + 1u;
        device const uint* reducedReady = coordinateDofs + nv;
        const bool useUpperTriangle = (dispatch.flags &
            MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE) != 0u;
        if (*reducedReady != 0u) {
            for (uint coordinate = lane; coordinate < freeDofs;
                 coordinate += threadCount) {
                float value = 0.0f;
                for (uint index = coordinateOffsets[coordinate];
                     index < coordinateOffsets[coordinate + 1u]; ++index) {
                    const uint dof = coordinateDofs[index];
                    const float force = contactColumn
                        ? rawResponse[dof]
                        : dof == limitDof ? 1.0f : 0.0f;
                    value = fma(coefficientForDof[dof], force, value);
                }
                reducedResponse[coordinate] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const bool reducedSolved = mrNumiHumanBilateralSolveCooperative(
                reducedFactor, reducedScale, reducedPivots,
                reducedResponse, freeDofs, lane, threadCount,
                &reducedSolveSucceeded);
            if (lane == 0u) reducedSolveSucceeded = reducedSolved ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (reducedSolveSucceeded != 0u) {
                if (lane == 0u)
                    atomic_store_explicit(&reducedFailure, 0u,
                                          memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint dof = lane; dof < nv; dof += threadCount) {
                    const uint coordinate = coordinateForDof[dof];
                    reducedLift[dof] =
                        coordinate == MR_INVALID_INDEX ? 0.0f :
                        coefficientForDof[dof] * reducedResponse[coordinate];
                    if (!isfinite(reducedLift[dof]))
                        atomic_store_explicit(&reducedFailure, 1u,
                                              memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint row = lane; row < equalityCount;
                     row += threadCount) {
                    device const auto& equality = jointEqualities[row];
                    const uint dependent = equality.indices.y;
                    float reactionValue = contactColumn
                        ? -rawResponse[dependent]
                        : (dependent == limitDof ? -1.0f : 0.0f);
                    for (uint dof = 0u; dof < nv; ++dof) {
                        reactionValue = fma(
                            standSelectedSourceA(sourceA, nv, dependent, dof,
                                useUpperTriangle),
                            reducedLift[dof], reactionValue);
                    }
                    equalityRhs[row] = reactionValue;
                    if (!isfinite(reactionValue))
                        atomic_store_explicit(&reducedFailure, 1u,
                                              memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (atomic_load_explicit(&reducedFailure,
                        memory_order_relaxed) == 0u) {
                    if (!contactColumn)
                        diagnosticReducedDiagonal = reducedLift[limitDof];
                    if (!contactColumn &&
                        !(reducedLift[limitDof] > 1.0e-6f * rawDiagonal)) {
                        if (kUseReducedResponseDiagnostics)
                            diagnosticWeakFallback = true;
                        for (uint dof = lane; dof < nv; dof += threadCount)
                            reducedLift[dof] = rawResponse[dof];
                        for (uint row = lane; row < equalityCount;
                             row += threadCount)
                            equalityRhs[row] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint dof = lane; dof < nv; dof += threadCount)
                        rhs[dof] = reducedLift[dof];
                    for (uint row = lane; row < equalityCount;
                         row += threadCount) {
                        if (contactColumn) reaction[row] = equalityRhs[row];
                        else equalityCorrection[row] = equalityRhs[row];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    useReducedResponse = true;
                    if (lane == 0u) {
                        device atomic_uint* statusFlags =
                            reinterpret_cast<device atomic_uint*>(
                                &status.flags);
                        atomic_fetch_or_explicit(statusFlags,
                            uint(MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_USED),
                            memory_order_relaxed);
                    }
                } else if (kUseReducedResponseDiagnostics && !contactColumn) {
                    diagnosticOtherFallback = true;
                }
            } else if (kUseReducedResponseDiagnostics && !contactColumn) {
                diagnosticOtherFallback = true;
            }
        } else if (kUseReducedResponseDiagnostics && !contactColumn) {
            diagnosticOtherFallback = true;
        }
    }

    if (projectEquality && !useReducedResponse) {
        // Restore complete legacy reactions when the candidate factor or
        // response is ineligible; the original Schur correction stays intact.
        for (uint row = lane; row < equalityCount; row += threadCount) {
            if (contactColumn && useProjectedContacts) reaction[row] = 0.0f;
            if (!contactColumn) equalityCorrection[row] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint refinement = 0u; refinement < 2u; ++refinement) {
            if (lane == 0u) {
                for (uint row = 0u; row < equalityCount; ++row) {
                    device const auto& equality = jointEqualities[row];
                    float residual = rhs[equality.indices.y];
                    if (equality.indices.w != MR_INVALID_INDEX)
                        residual = fma(-derivativeCache[row],
                            rhs[equality.indices.w], residual);
                    equalityRhs[row] = residual;
                }
                if (!mrNumiHumanBilateralSolve(equalityFactor,
                        equalityScale, equalityPivots, equalityRhs,
                        equalityCount)) {
                    conditionValid = 0u;
                    publishParallelResponseFailure(status, contactColumn
                        ? kParallelContactConditionFailure + positionIndex
                        : kParallelLimitConditionFailure + 2u * limitDof);
                } else {
                    for (uint row = 0u; row < equalityCount; ++row) {
                        if (contactColumn) reaction[row] -= equalityRhs[row];
                        else equalityCorrection[row] -= equalityRhs[row];
                    }
                }
                atomic_store_explicit(&correctionFailed, 0u,
                                      memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (conditionValid == 0u) return;
            // Each DOF preserves the original ascending equality-row FMA
            // sequence; independent DOFs now occupy separate GPU lanes.
            for (uint dof = lane; dof < nv; dof += threadCount) {
                float correction = 0.0f;
                for (uint row = 0u; row < equalityCount; ++row) {
                    device const float* equalityResponse =
                        responseScratch + responseBase +
                        (contactColumns + row) * nv;
                    correction = fma(equalityRhs[row],
                        equalityResponse[dof], correction);
                }
                rhs[dof] -= correction;
                if (!isfinite(rhs[dof]))
                    atomic_store_explicit(&correctionFailed, 1u,
                                          memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u && atomic_load_explicit(
                    &correctionFailed, memory_order_relaxed) != 0u) {
                conditionValid = 0u;
                publishParallelResponseFailure(status, contactColumn
                    ? kParallelContactConditionFailure + positionIndex
                    : kParallelLimitConditionFailure + 2u * limitDof + 1u);
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (conditionValid == 0u) return;
        }
        if (!contactColumn) {
            if (lane == 0u && !(rhs[limitDof] >
                    1.0e-6f * rawDiagonal)) {
                restoreRawResponse = 1u;
                for (uint row = 0u; row < equalityCount; ++row)
                    equalityCorrection[row] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            if (restoreRawResponse != 0u)
                for (uint dof = lane; dof < nv; dof += threadCount)
                    rhs[dof] = rawResponse[dof];
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (conditionValid == 0u) return;
    if (kUseReducedResponseDiagnostics && lane == 0u && !contactColumn &&
        projectEquality && reducedResponseRequested && !projectedRawReady) {
        const uint diagnosticBase =
            (environment * nv + limitDof) * 4u;
        reducedResponseDiagnostics[diagnosticBase + 0u] = as_type<uint>(rawDiagonal);
        reducedResponseDiagnostics[diagnosticBase + 1u] =
            as_type<uint>(diagnosticReducedDiagonal);
        reducedResponseDiagnostics[diagnosticBase + 2u] = 1u |
            (diagnosticAttempted ? 2u : 0u) |
            (useReducedResponse && !diagnosticWeakFallback ? 4u : 0u) |
            (diagnosticWeakFallback ? 8u : 0u) |
            (diagnosticOtherFallback ? 16u : 0u);
        reducedResponseDiagnostics[diagnosticBase + 3u] = 0u;
    }
    for (uint dof = lane; dof < nv; dof += threadCount)
        response[dof] = rhs[dof];
}

// Assemble independent mass entries across GPU threadgroups after the
// prerequisite pass has published spatial and inertia-weighted Jacobians.
// The per-entry body reduction and passive terms match stand_step exactly.
kernel void mr_numi_human_stand_mass_assemble(
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device const float* spatialJacobianScratch [[buffer(12)]],
    device float* factorScratch [[buffer(14)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const float* passiveJointProgram [[buffer(24)]],
    device float* sourceDynamicsWitness [[buffer(25)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint environment = position.y;
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS ||
        (status.flags & MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY) == 0u)
        return;
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint nv = articulation.nv;
    const uint index = position.x;
    if (index >= nv * nv) return;
    const uint row = index / nv;
    const uint column = index - row * nv;
    const uint bodyCount = articulation.bodyCount;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    const uint inertiaWeightedBase = spatialBase + bodyCount * 6u * nv;
    float value = 0.0f;
    MRCompensatedScalar bodySum{0.0f, 0.0f};
    const bool compensated = (dispatch.flags &
        MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM) != 0u;
    for (uint localBody = 0u; localBody < bodyCount; ++localBody) {
        const uint globalBody = articulation.firstBody + localBody;
        device const MRBodyPropertiesGPU& body = bodies[globalBody];
        const uint base = spatialBase + localBody * 6u * nv;
        const float3 leftAngular{
            spatialJacobianScratch[base + 0u * nv + row],
            spatialJacobianScratch[base + 1u * nv + row],
            spatialJacobianScratch[base + 2u * nv + row],
        };
        const uint weighted = inertiaWeightedBase +
            localBody * 3u * nv + column;
        const float3 rightInertiaWeighted{
            spatialJacobianScratch[weighted + 0u * nv],
            spatialJacobianScratch[weighted + 1u * nv],
            spatialJacobianScratch[weighted + 2u * nv],
        };
        const float3 leftLinear{
            spatialJacobianScratch[base + 3u * nv + row],
            spatialJacobianScratch[base + 4u * nv + row],
            spatialJacobianScratch[base + 5u * nv + row],
        };
        const float3 rightLinear{
            spatialJacobianScratch[base + 3u * nv + column],
            spatialJacobianScratch[base + 4u * nv + column],
            spatialJacobianScratch[base + 5u * nv + column],
        };
        const float contribution = dot(leftAngular, rightInertiaWeighted) +
            body.massAndInverseMass.x * dot(leftLinear, rightLinear);
        if (compensated)
            bodySum = mrCompensatedAdd(bodySum, {contribution, 0.0f});
        else value += contribution;
    }
    if (compensated) value = bodySum.high + bodySum.low;
    if (row == column) {
        device const MRDofPropertiesGPU& dof =
            dofs[articulation.vOffset + row];
        value += dof.drive.z;
        if ((dof.flags & MR_DOF_FLAG_DRIVE) == 0u)
            value += dispatch.groundPointAndTimestep.w * dof.drive.y;
    }
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM) != 0u)
        value += mrNumiHumanPassiveEffectiveInertia(
            passiveJointProgram[index], dispatch.groundPointAndTimestep.w);
    factorScratch[environment * nv * nv + index] = value;
    if ((dispatch.flags &
         MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES) != 0u) {
        // Snapshot the current source A before phase 2 factors factorScratch.
        // Phase 4 consumes this per-root derived scratch to form R^T A R.
        const uint legacyStride = standLegacyResponseStride(
            nv, dispatch.supportContactCount, dispatch.jointEqualityCount);
        const uint responseBase = environment *
            standResponseStride(dispatch, nv);
        responseScratch[responseBase + legacyStride + index] = value;
    }
    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u &&
        row == column)
        sourceDynamicsWitness[environment * 3u * nv + row] = value;
}

// The split completion entry consumes the prepare phase in the same
// authoritative command buffer. Contact and limit decisions retain their
// ordered owner; contact, equality, and limit response updates distribute
// disjoint DOFs. Contact velocity axes and limit reaction rows also use
// independent lanes while scalar impulse work keeps its FP32 order.
kernel void mr_numi_human_stand_finish(
    device const MRWorldGPU* worlds [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodies [[buffer(3)]],
    constant const MRNumiHumanStandDispatchGPU& dispatch [[buffer(4)]],
    device float* qState [[buffer(5)]],
    device float* vState [[buffer(6)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(7)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(8)]],
    device const float* pointJacobians [[buffer(9)]],
    device const float* generalizedForceWorkspace [[buffer(10)]],
    device const MRNumiHumanStandContactGPU* contacts [[buffer(11)]],
    device float* spatialJacobianScratch [[buffer(12)]],
    device float4* bodyMotionScratch [[buffer(13)]],
    device float* factorScratch [[buffer(14)]],
    device float* vectorScratch [[buffer(15)]],
    device float* responseScratch [[buffer(16)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(17)]],
    device const MRNumiHumanTendonBindingGPU* tendonBindings [[buffer(18)]],
    device const MRNumiHumanTendonTransferResultGPU* tendonTransfers [[buffer(19)]],
    device const MRNumiHumanJointEqualityGPU* jointEqualities [[buffer(20)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(21)]],
    device const float4* bodyPositionLow [[buffer(22)]],
    device const float4* pointPositionLow [[buffer(23)]],
    device const float* passiveJointProgram [[buffer(24)]],
    device float* sourceDynamicsWitness [[buffer(25)]],
    device const MRNumiHumanStandCpuFinishGPU* cpuFinishes [[buffer(26)]],
    device float* cachedLimitEqualityResponse [[buffer(27), function_constant(kUseCachedLimitEqualityResponse)]],
    device const uint* sparseGraph [[buffer(28), function_constant(kUseSparseStandOperator)]],
    device uint* finishWorkCounters [[buffer(29)]],
    uint environment [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint threadCount [[threads_per_threadgroup]],
    uint simdWidth [[threads_per_simdgroup]]
) {
    if (environment >= dispatch.environmentCount) return;
    device MRNumiHumanStandStatusGPU& status = statuses[environment];
    device const MRArticulationGPU& articulation =
        articulations[dispatch.articulationIndex];
    const uint bodyCount = articulation.bodyCount;
    const uint nv = articulation.nv;
    const uint nq = articulation.nq;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const uint pointBase = environment * dispatch.pointWorldStride;
    const uint pointJacobianBase = environment * dispatch.pointJacobianStride;
    const uint forceBase = environment * dispatch.generalizedForceStride +
        dispatch.generalizedForceOffset;
    const uint spatialBase = environment * bodyCount *
        MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv;
    const uint factorBase = environment * nv * nv;
    const uint vectorStride = nv + 3u * nv +
        12u * dispatch.supportContactCount + dispatch.jointEqualityCount +
        (((dispatch.flags & MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES) != 0u)
            ? nv + nq + nv : 0u);
    const uint preloadBase = environment * vectorStride;
    const uint vectorBase = preloadBase + nv;
    const uint equalityCount = dispatch.jointEqualityCount;
    const uint responseColumns = (dispatch.supportContactCount * 3u + equalityCount + nv) * nv;
    const uint responseStride = standResponseStride(dispatch, nv);
    const uint responseBase = environment * responseStride;
    device float* equalityFactor = responseScratch + responseBase + responseColumns;
    device float* equalityScale = equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    // These vectors are repeatedly read and updated by the scalar constraint
    // solve. Keep them in the threadgroup instead of round-tripping every
    // scalar dependency through device memory. Retain the exported scratch
    // layout and publish its final values below.
    threadgroup float equalityRhsStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float equalityFactorStorage[
        kCachedEqualityCapacity * kCachedEqualityCapacity];
    threadgroup float candidateVStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float workspaceStorage[MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup uint cooperativeFreeSolveSucceeded;
    threadgroup uint cooperativeEqualitySucceeded;
    // Lanes evaluate ordered unilateral decisions together; lane zero owns
    // impulse history, and disjoint lanes apply each accepted response.
    threadgroup uint cooperativeLimitCount;
    threadgroup float cooperativeLimitEqualityImpulses[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float cooperativeLimitEqualityVelocities[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup uint cooperativeContactActive[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    threadgroup float cooperativeContactGap[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    threadgroup float cooperativeContactVelocities[3];
    threadgroup float cooperativeContactApplied[3];
    threadgroup float cooperativeContactEqualityWork[
        3u * MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    // A one-sweep journal lets the optional diagnostic replay run after the
    // source-ordered physical limit updates. It is reused at each sweep
    // boundary; no accepted-state or cross-root history is retained.
    threadgroup float deferredLimitImpulseJournal[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup float deferredEqualityVelocityStart[
        MR_NUMI_HUMAN_STAND_MAX_DOFS];
    threadgroup MRNumiHumanPreparedFrictionMetric cooperativeFrictionMetrics[
        MR_NUMI_HUMAN_STAND_MAX_CONTACTS];
    threadgroup float* equalityRhs = equalityRhsStorage;
    // Equality multiplier corrections for each conditioned limit response.
    device float* limitEqualityCorrections = equalityPivots + 2u * equalityCount;
    device float* bias = vectorScratch + vectorBase;
    threadgroup float* candidateV = candidateVStorage;
    threadgroup float* workspace = workspaceStorage;
    device float* lambdas = bias + 3u * nv;
    device float* equalityLambdas =
        lambdas + 3u * dispatch.supportContactCount;
    device float* contactMatrices =
        equalityLambdas + dispatch.jointEqualityCount;
    device float* sourceLimitImpulseEvidence =
        contactMatrices + 9u * dispatch.supportContactCount;
    device float* preProjectionQEvidence = sourceLimitImpulseEvidence + nv;
    device float* preProjectionVEvidence = preProjectionQEvidence + nq;
    device float* factor = factorScratch + factorBase;
    const bool captureSourceDynamics =
        (dispatch.flags & MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY) != 0u;
    const uint sourceDynamicsBase = environment * 3u * nv;

    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
    device float* equalityDerivativeCache = spatialJacobianScratch + spatialBase;
    device float* equalityTargetVelocityCache = equalityDerivativeCache + nv;
    device const float* equalityResponseByDof =
        equalityTargetVelocityCache + nv;
    const bool cacheEqualityResponseByDof =
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS >=
        2u + equalityCount;
    device const float* projectedContactEquality =
        equalityResponseByDof + nv * equalityCount;
    const bool useProjectedContacts =
        equalityCount != 0u &&
        equalityCount <= MR_NUMI_HUMAN_STAND_MAX_DOFS &&
        (dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT) != 0u &&
        dispatch.supportContactCount != 0u &&
        bodyCount * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv >=
            (2u + equalityCount) * nv +
            3u * dispatch.supportContactCount * equalityCount;
    const float timestep = dispatch.groundPointAndTimestep.w;
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    float maximumEqualityPositionError = 0.0f;
    threadgroup atomic_uint responseFailure;
    if (lane == 0u) {
        atomic_store_explicit(&responseFailure, MR_INVALID_INDEX,
                              memory_order_relaxed);
        const uint responseFailureOrdinal = status.failingIndex;
        if (responseFailureOrdinal != MR_INVALID_INDEX) {
            const uint contactColumns = 3u * dispatch.supportContactCount;
            const uint equalityColumnsEnd = contactColumns + equalityCount;
            if (responseFailureOrdinal < kParallelContactConditionFailure) {
                if (responseFailureOrdinal < contactColumns)
                    fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED,
                         responseFailureOrdinal / 3u);
                else if (responseFailureOrdinal < equalityColumnsEnd) {
                    ++status.jointEqualityCounts.z;
                    fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,
                         responseFailureOrdinal - contactColumns);
                } else
                    fail(status, MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED,
                         responseFailureOrdinal - equalityColumnsEnd);
            } else if (responseFailureOrdinal <
                       kParallelLimitConditionFailure) {
                fail(status, MR_NUMI_HUMAN_STAND_CONTACT_FAILED,
                     (responseFailureOrdinal -
                      kParallelContactConditionFailure) / 3u);
            } else {
                const uint limitFailure = responseFailureOrdinal -
                    kParallelLimitConditionFailure;
                fail(status, (limitFailure & 1u) == 0u
                    ? MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED
                    : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT,
                    limitFailure / 2u);
            }
        }
        if (status.code == MR_NUMI_HUMAN_STAND_SUCCESS) {
        for (uint row = 0u; row < nv; ++row) {
            const float pivot = factor[row * nv + row];
            minimumPivot = min(minimumPivot, pivot);
            maximumPivot = max(maximumPivot, pivot);
        }
        for (uint row = 0u; row < equalityCount; ++row) {
            float target = 0.0f, derivative = 0.0f, error = 0.0f;
            if (!evaluateJointEquality(jointEqualities[row], qState, qBase,
                                       nq, nv, target, derivative, error)) {
                fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, row);
                break;
            }
            maximumEqualityPositionError = max(
                maximumEqualityPositionError, abs(error));
        }
        if (equalityCount <= kCachedEqualityCapacity) {
            for (uint index = 0u; index < equalityCount * equalityCount;
                 ++index) {
                equalityFactorStorage[index] = equalityFactor[index];
            }
        }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status.code != MR_NUMI_HUMAN_STAND_SUCCESS) return;
#define MR_NH_COOPERATIVE_FINISH 1
    MRStandFinishWorkCounters finishCounts{};
#include "NumiHumanStandSolve.metalinc"
#undef MR_NH_COOPERATIVE_FINISH
    if (kUseFinishWorkCounters && lane == 0u) {
        const uint base = environment * (17u + nv);
        finishWorkCounters[base + 0u] = finishCounts.sweeps;
        finishWorkCounters[base + 1u] = finishCounts.contactDecisions;
        finishWorkCounters[base + 2u] = finishCounts.contactContractions;
        finishWorkCounters[base + 3u] = finishCounts.normalChanges;
        finishWorkCounters[base + 4u] = finishCounts.tangentChanges;
        finishWorkCounters[base + 5u] = finishCounts.zeroContactChanges;
        finishWorkCounters[base + 6u] = finishCounts.frictionInterior;
        finishWorkCounters[base + 7u] = finishCounts.frictionBoundary;
        finishWorkCounters[base + 8u] = finishCounts.frictionBoundaryIterations;
        finishWorkCounters[base + 9u] = finishCounts.limitPreviews;
        finishWorkCounters[base + 10u] = finishCounts.limitPreviewBlocks;
        finishWorkCounters[base + 11u] = finishCounts.limitNonzero;
        finishWorkCounters[base + 12u] = finishCounts.limitSelectedZero;
        finishWorkCounters[base + 13u] = finishCounts.contactResponseCoefficients;
        finishWorkCounters[base + 14u] = finishCounts.limitResponseCoefficients;
        finishWorkCounters[base + 15u] = finishCounts.equalityResponseCoefficients;
        finishWorkCounters[base + 16u] = finishCounts.onePassLimitRoutes;
    }
}

// Ordinary stand/tendon accepted-step owner. Derived poses/routes/factors are
// recomputed on the next step; the contact vector arena includes persistent
// warm starts and therefore belongs to the restored state.
kernel void mr_numi_human_stand_reconcile(
    constant uint4& shape [[buffer(0)]], // environments, attempted step, muscles, vectors
    constant uint4& strides [[buffer(1)]], // q, v
    device float* q [[buffer(2)]],
    device float* v [[buffer(3)]],
    device MRMujocoMuscleStateGPU* muscles [[buffer(4)]],
    device MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device float* vectors [[buffer(6)]],
    device const float* acceptedQ [[buffer(7)]],
    device const float* acceptedV [[buffer(8)]],
    device const MRMujocoMuscleStateGPU* acceptedMuscles [[buffer(9)]],
    device const MRNumiHumanStandStatusGPU* acceptedStatuses [[buffer(10)]],
    device const float* acceptedVectors [[buffer(11)]],
    device MRCompensatedRootTranslationGPU* rootTranslations [[buffer(12)]],
    device const MRCompensatedRootTranslationGPU* acceptedRootTranslations [[buffer(13)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= shape.x) return;
    const MRNumiHumanStandStatusGPU attempt = statuses[environment];
    if (attempt.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
        attempt.environment == environment && attempt.completedSteps == shape.y + 1u) return;
    rootTranslations[environment] = acceptedRootTranslations[environment];
    for (uint i=0u; i<strides.x; ++i) q[environment*strides.x+i] = acceptedQ[environment*strides.x+i];
    for (uint i=0u; i<strides.y; ++i) v[environment*strides.y+i] = acceptedV[environment*strides.y+i];
    for (uint i=0u; i<shape.z; ++i) muscles[environment*shape.z+i] = acceptedMuscles[environment*shape.z+i];
    for (uint i=0u; i<shape.w; ++i) vectors[environment*shape.w+i] = acceptedVectors[environment*shape.w+i];
    MRNumiHumanStandStatusGPU restored = shape.y == 0u
        ? MRNumiHumanStandStatusGPU{} : acceptedStatuses[environment];
    restored.environment = environment;
    restored.code = attempt.code == MR_NUMI_HUMAN_STAND_SUCCESS
        ? MR_NUMI_HUMAN_STAND_INVALID_DISPATCH : attempt.code;
    restored.failingIndex = attempt.failingIndex;
    statuses[environment] = restored;
}
