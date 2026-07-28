#include "metalrobo/ConstraintIR.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint32_t kKnownBlockFlags =
    constraintIRBlockNewImpact |
    constraintIRBlockWarmStarted |
    constraintIRBlockDisabled;
constexpr std::uint32_t kKnownRowFlags =
    constraintIRRowPositionStabilized |
    constraintIRRowUnilateral |
    constraintIRRowContactNormal |
    constraintIRRowContactTangent |
    constraintIRRowContactTorsion;
constexpr double kDirectionTolerance = 2.0e-4;
constexpr double kOrthogonalityTolerance = 4.0e-4;
constexpr double kWarmStartTolerance = 2.0e-5;
constexpr double kMinimumProjectionStep = 1.0e-6;
constexpr double kMaximumProjectionStep = 1.0e6;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct FrictionPatchPoint {
    double normal = 0.0;
    double tangentU = 0.0;
    double tangentV = 0.0;
    double torsion = 0.0;
};

ConstraintIRDiagnostics failure(
    const ConstraintIRStatus status,
    std::string message,
    const std::uint32_t block = kConstraintIRInvalidIndex,
    const std::uint32_t row = kConstraintIRInvalidIndex
) {
    ConstraintIRDiagnostics result;
    result.status = status;
    result.blockIndex = block;
    result.rowIndex = row;
    result.message = std::move(message);
    return result;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4& value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

Vec3 xyz(const mr_float4& value) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
    };
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

Vec3 operator/(const Vec3 value, const double divisor) {
    return {
        value.x / divisor,
        value.y / divisor,
        value.z / divisor,
    };
}

double dot(const Vec3 left, const Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool representableAsFloat(const double value) {
    return finite(value) &&
        std::abs(value) <=
            static_cast<double>(std::numeric_limits<float>::max());
}

bool knownConstraintType(const std::uint32_t type) {
    return type <= MR_CONSTRAINT_DRY_FRICTION;
}

bool knownEndpointRole(const std::uint32_t role) {
    return role <= constraintIREndpointWorld;
}

bool knownJacobianKind(const std::uint32_t kind) {
    return kind <= constraintIRJacobianAngular;
}

bool rangeFits(
    const std::uint32_t offset,
    const std::uint32_t count,
    const std::size_t size
) {
    const std::uint64_t end =
        static_cast<std::uint64_t>(offset) + count;
    return end <= size;
}

std::pair<Vec3, Vec3> contactBasis(const Vec3 unitNormal) {
    const Vec3 absolute{
        std::abs(unitNormal.x),
        std::abs(unitNormal.y),
        std::abs(unitNormal.z),
    };
    Vec3 reference{};
    if (absolute.x <= absolute.y && absolute.x <= absolute.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absolute.y <= absolute.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    const Vec3 tangent = cross(reference, unitNormal);
    const double tangentNorm = norm(tangent);
    return {
        tangent / tangentNorm,
        cross(unitNormal, tangent / tangentNorm),
    };
}

bool coneCoefficientsValid(const ConstraintIRCone& cone) {
    return
        finite(cone.staticFrictionU) &&
        finite(cone.staticFrictionV) &&
        finite(cone.dynamicFrictionU) &&
        finite(cone.dynamicFrictionV) &&
        finite(cone.rollingLength) &&
        finite(cone.torsionalLength) &&
        finite(cone.restitution) &&
        finite(cone.restitutionThreshold) &&
        finite(cone.adhesionImpulse) &&
        finite(cone.maximumNormalImpulse) &&
        finite(cone.stictionTransitionVelocity) &&
        finite(cone.reserved) &&
        cone.staticFrictionU >= cone.dynamicFrictionU &&
        cone.staticFrictionV >= cone.dynamicFrictionV &&
        cone.dynamicFrictionU >= 0.0F &&
        cone.dynamicFrictionV >= 0.0F &&
        cone.rollingLength >= 0.0F &&
        cone.torsionalLength >= 0.0F &&
        cone.restitution >= 0.0F &&
        cone.restitution <= 1.0F &&
        cone.restitutionThreshold >= 0.0F &&
        cone.adhesionImpulse >= 0.0F &&
        cone.maximumNormalImpulse >= 0.0F &&
        cone.stictionTransitionVelocity >= 0.0F &&
        cone.reserved == 0.0F &&
        (
            (cone.staticFrictionU == 0.0F &&
             cone.staticFrictionV == 0.0F) ||
            (cone.staticFrictionU > 0.0F &&
             cone.staticFrictionV > 0.0F)
        ) &&
        (
            (cone.dynamicFrictionU == 0.0F &&
             cone.dynamicFrictionV == 0.0F) ||
            (cone.dynamicFrictionU > 0.0F &&
             cone.dynamicFrictionV > 0.0F)
        );
}

double coneScaledTangent(
    const double tangentU,
    const double tangentV,
    const double frictionU,
    const double frictionV
) {
    if (frictionU == 0.0 && frictionV == 0.0) {
        return (tangentU == 0.0 && tangentV == 0.0)
            ? 0.0
            : std::numeric_limits<double>::infinity();
    }
    return std::hypot(
        tangentU / frictionU,
        tangentV / frictionV
    );
}

double geometricMean(const double left, const double right) {
    return std::sqrt(std::max(left * right, 0.0));
}

void hashWord(std::uint64_t& hash, const std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        hash ^= (value >> shift) & 0xffu;
        hash *= kFnvPrime;
    }
}

void hashFloat(std::uint64_t& hash, const float value) noexcept {
    hashWord(hash, std::bit_cast<std::uint32_t>(value));
}

void hashFloat4(std::uint64_t& hash, const mr_float4& value) noexcept {
    hashFloat(hash, value.x);
    hashFloat(hash, value.y);
    hashFloat(hash, value.z);
    hashFloat(hash, value.w);
}

FrictionPatchPoint projectTruncatedFrictionPatch(
    const FrictionPatchPoint input,
    const double maximumNormal,
    const bool hasTorsion
) {
    const double tangentNorm =
        std::hypot(input.tangentU, input.tangentV);
    const double torsionMagnitude =
        hasTorsion ? std::abs(input.torsion) : 0.0;

    // For a fixed normal coordinate n, the closest tangent and torsion
    // coordinates are their projections onto ||t|| <= n and |r| <= n.
    // The remaining one-dimensional objective is convex and piecewise
    // quadratic:
    //
    //   (n-a)^2 + max(||t||-n, 0)^2 + max(|r|-n, 0)^2.
    //
    // Its minimizer is therefore one of the interval endpoints, a kink, or
    // the stationary point of one of its four active sets.
    const auto clampNormal = [maximumNormal](const double value) {
        return std::clamp(value, 0.0, maximumNormal);
    };
    const std::array<double, 8> candidates{
        0.0,
        maximumNormal,
        clampNormal(input.normal),
        clampNormal(tangentNorm),
        clampNormal(torsionMagnitude),
        clampNormal(0.5 * (input.normal + tangentNorm)),
        clampNormal(0.5 * (input.normal + torsionMagnitude)),
        clampNormal(
            (input.normal + tangentNorm + torsionMagnitude) / 3.0
        ),
    };
    const auto objective = [
        &input,
        tangentNorm,
        torsionMagnitude,
        hasTorsion
    ](const double normal) {
        const double normalError = normal - input.normal;
        const double tangentError =
            std::max(tangentNorm - normal, 0.0);
        const double torsionError = hasTorsion
            ? std::max(torsionMagnitude - normal, 0.0)
            : 0.0;
        return
            normalError * normalError +
            tangentError * tangentError +
            torsionError * torsionError;
    };

    double projectedNormal = candidates.front();
    double bestObjective = objective(projectedNormal);
    for (const double candidate : candidates) {
        const double candidateObjective = objective(candidate);
        if (candidateObjective < bestObjective ||
            (candidateObjective == bestObjective &&
             candidate < projectedNormal)) {
            projectedNormal = candidate;
            bestObjective = candidateObjective;
        }
    }

    const double tangentScale =
        tangentNorm > projectedNormal
        ? projectedNormal / std::max(tangentNorm, 1.0e-300)
        : 1.0;
    const double projectedTorsion = hasTorsion
        ? std::clamp(
            input.torsion,
            -projectedNormal,
            projectedNormal
        )
        : 0.0;
    return {
        projectedNormal,
        input.tangentU * tangentScale,
        input.tangentV * tangentScale,
        projectedTorsion,
    };
}

void projectEvaluatedWarmContact(
    const ConstraintIRBlock& block,
    const EvaluatedConstraintIRCone& cone,
    std::vector<float>& warmImpulses
) {
    const std::uint32_t offset = block.impulseOffset;
    const double maximumNormal =
        cone.maximumNormalImpulse == 0.0F
        ? static_cast<double>(kConstraintIRUnbounded)
        : cone.maximumNormalImpulse;
    const double muU = cone.effectiveFrictionU;
    const double muV = cone.effectiveFrictionV;
    const bool hasTorsion = block.dimension == 4u;

    if (muU == 0.0 && muV == 0.0) {
        warmImpulses[offset] = static_cast<float>(std::clamp(
            static_cast<double>(warmImpulses[offset]),
            0.0,
            maximumNormal
        ));
        warmImpulses[offset + 1u] = 0.0F;
        warmImpulses[offset + 2u] = 0.0F;
        if (hasTorsion) {
            warmImpulses[offset + 3u] = 0.0F;
        }
        return;
    }

    const double torsionScale = hasTorsion
        ? geometricMean(muU, muV) * cone.torsionalLength
        : 0.0;
    const bool hasScaledTorsion =
        hasTorsion && torsionScale > 0.0;
    const FrictionPatchPoint transformed{
        warmImpulses[offset],
        static_cast<double>(warmImpulses[offset + 1u]) / muU,
        static_cast<double>(warmImpulses[offset + 2u]) / muV,
        hasScaledTorsion
            ? static_cast<double>(warmImpulses[offset + 3u]) /
                torsionScale
            : 0.0,
    };
    const FrictionPatchPoint projected =
        projectTruncatedFrictionPatch(
            transformed,
            maximumNormal,
            hasScaledTorsion
        );
    warmImpulses[offset] =
        static_cast<float>(projected.normal);
    warmImpulses[offset + 1u] =
        static_cast<float>(projected.tangentU * muU);
    warmImpulses[offset + 2u] =
        static_cast<float>(projected.tangentV * muV);
    if (hasTorsion) {
        warmImpulses[offset + 3u] = hasScaledTorsion
            ? static_cast<float>(
                projected.torsion * torsionScale
            )
            : 0.0F;
    }
}

double boundNaturalResidual(
    const double impulse,
    const double gradient,
    const double lower,
    const double upper,
    const double projectionStep
) {
    const double projected = std::clamp(
        impulse - projectionStep * gradient,
        lower,
        upper
    );
    return std::abs(impulse - projected) / projectionStep;
}

bool allFinite(const std::span<const float> values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const float value) { return finite(value); }
    );
}

ConstraintIRStableKey v1Key(
    const MRContactConstraintGPU& contact
) {
    ConstraintIRStableKey key{};
    key.words[0] =
        static_cast<std::uint32_t>(contact.pairKey >> 32u);
    key.words[1] =
        static_cast<std::uint32_t>(contact.pairKey);
    key.words[2] =
        static_cast<std::uint32_t>(contact.featureKey >> 32u);
    key.words[3] =
        static_cast<std::uint32_t>(contact.featureKey);
    return key;
}

bool validV1AdapterConfig(const ConstraintIRV1AdapterConfig& config) {
    return finite(config.timeConstant) &&
        finite(config.dampingRatio) &&
        finite(config.dissipation) &&
        finite(config.stictionTransitionVelocity) &&
        config.timeConstant > 0.0F &&
        config.dampingRatio >= 0.0F &&
        config.dissipation >= 0.0F &&
        config.stictionTransitionVelocity >= 0.0F;
}

ConstraintIRDiagnostics validateV1Contact(
    const MRContactConstraintGPU& contact,
    const std::uint32_t index
) {
    const std::uint32_t unknownFlags = contact.flags & ~kKnownBlockFlags;
    if (unknownFlags != 0u ||
        contact.bodyA == contact.bodyB ||
        contact.bodyA == MR_INVALID_INDEX ||
        contact.bodyB == MR_INVALID_INDEX ||
        contact.islandIndex == MR_INVALID_INDEX) {
        return failure(
            ConstraintIRStatus::invalidBlock,
            "v1 contact indices or flags are invalid",
            index
        );
    }
    if (!finite(contact.pointAndSeparation) ||
        !finite(contact.normal) ||
        !finite(contact.friction) ||
        !finite(contact.response) ||
        !finite(contact.targetVelocityAndPreSolveNormal) ||
        !finite(contact.impulses)) {
        return failure(
            ConstraintIRStatus::nonfiniteData,
            "v1 contact contains non-finite data",
            index
        );
    }
    const double normalNorm = norm(xyz(contact.normal));
    if (std::abs(normalNorm - 1.0) > kDirectionTolerance) {
        return failure(
            ConstraintIRStatus::invalidRow,
            "v1 contact normal is not unit length",
            index
        );
    }
    if (contact.friction.x < contact.friction.y ||
        contact.friction.y < 0.0F ||
        contact.friction.z < 0.0F ||
        contact.friction.w < 0.0F ||
        contact.response.x < 0.0F ||
        contact.response.x > 1.0F ||
        contact.response.y < 0.0F ||
        contact.response.z < 0.0F ||
        contact.response.w < 0.0F) {
        return failure(
            ConstraintIRStatus::invalidCone,
            "v1 contact response or friction values are invalid",
            index
        );
    }
    if (contact.friction.z > 0.0F) {
        return failure(
            ConstraintIRStatus::unsupportedSemantics,
            "rolling friction requires executable rolling rows",
            index
        );
    }
    return {};
}

} // namespace

bool constraintIRKeyLess(
    const ConstraintIRStableKey& left,
    const ConstraintIRStableKey& right
) noexcept {
    return std::lexicographical_compare(
        std::begin(left.words),
        std::end(left.words),
        std::begin(right.words),
        std::end(right.words)
    );
}

bool constraintIRKeyEqual(
    const ConstraintIRStableKey& left,
    const ConstraintIRStableKey& right
) noexcept {
    return std::equal(
        std::begin(left.words),
        std::end(left.words),
        std::begin(right.words)
    );
}

ConstraintIRDiagnostics validateConstraintIR(const ConstraintIR& ir) {
    if (ir.abiVersion != kConstraintIRAbiVersion) {
        return failure(
            ConstraintIRStatus::invalidAbiVersion,
            "constraint IR ABI version is not supported"
        );
    }
    constexpr std::size_t maximum =
        std::numeric_limits<std::uint32_t>::max();
    if (ir.blocks.size() > maximum ||
        ir.endpoints.size() > maximum ||
        ir.rows.size() > maximum ||
        ir.cones.size() > maximum ||
        ir.warmImpulses.size() > maximum) {
        return failure(
            ConstraintIRStatus::invalidCount,
            "constraint IR exceeds 32-bit packed indexing"
        );
    }
    if (ir.blocks.empty()) {
        if (!ir.endpoints.empty() || !ir.rows.empty() ||
            !ir.cones.empty() || !ir.warmImpulses.empty()) {
            return failure(
                ConstraintIRStatus::invalidRange,
                "empty block stream owns non-empty payload streams"
            );
        }
        return {};
    }

    std::uint64_t expectedEndpoint = 0u;
    std::uint64_t expectedRow = 0u;
    std::uint64_t expectedImpulse = 0u;
    std::uint64_t expectedCone = 0u;
    ConstraintIRStableKey previousKey{};
    bool havePrevious = false;

    for (std::uint32_t blockIndex = 0u;
         blockIndex < ir.blocks.size();
         ++blockIndex) {
        const ConstraintIRBlock& block = ir.blocks[blockIndex];
        if (havePrevious &&
            !constraintIRKeyLess(previousKey, block.key)) {
            return failure(
                ConstraintIRStatus::nonCanonicalOrder,
                "constraint keys are not strictly increasing",
                blockIndex
            );
        }
        previousKey = block.key;
        havePrevious = true;

        if (!knownConstraintType(block.type) ||
            block.dimension == 0u ||
            block.dimension > 6u ||
            (block.flags & ~kKnownBlockFlags) != 0u ||
            block.islandIndex == kConstraintIRInvalidIndex ||
            block.reserved0 != 0u ||
            block.reserved1 != 0u) {
            return failure(
                ConstraintIRStatus::invalidBlock,
                "constraint block metadata is invalid",
                blockIndex
            );
        }
        if (block.endpointOffset != expectedEndpoint ||
            block.rowOffset != expectedRow ||
            block.impulseOffset != expectedImpulse ||
            !rangeFits(
                block.endpointOffset,
                block.endpointCount,
                ir.endpoints.size()
            ) ||
            !rangeFits(
                block.rowOffset,
                block.dimension,
                ir.rows.size()
            ) ||
            !rangeFits(
                block.impulseOffset,
                block.dimension,
                ir.warmImpulses.size()
            )) {
            return failure(
                ConstraintIRStatus::invalidRange,
                "constraint payload ranges are not canonically packed",
                blockIndex
            );
        }
        expectedEndpoint += block.endpointCount;
        expectedRow += block.dimension;
        expectedImpulse += block.dimension;

        const bool isContact = block.type == MR_CONSTRAINT_CONTACT;
        if (isContact) {
            if (block.dimension != 3u && block.dimension != 4u) {
                return failure(
                    ConstraintIRStatus::invalidBlock,
                    "contact block dimension must be three or four",
                    blockIndex
                );
            }
            if (block.endpointCount != 2u ||
                block.coneIndex != expectedCone ||
                block.coneIndex >= ir.cones.size()) {
                return failure(
                    ConstraintIRStatus::invalidRange,
                    "contact endpoint or cone range is invalid",
                    blockIndex
                );
            }
            ++expectedCone;
        } else if (block.coneIndex != kConstraintIRInvalidIndex) {
            return failure(
                ConstraintIRStatus::invalidRange,
                "non-contact block unexpectedly references a cone",
                blockIndex
            );
        }

        for (std::uint32_t local = 0u;
             local < block.endpointCount;
             ++local) {
            const ConstraintIREndpoint& endpoint =
                ir.endpoints[block.endpointOffset + local];
            if (!knownEndpointRole(endpoint.role) ||
                !knownJacobianKind(endpoint.jacobianKind) ||
                endpoint.reserved0 != 0u ||
                endpoint.reserved1 != 0u ||
                !finite(endpoint.anchor) ||
                !finite(endpoint.axis) ||
                (
                    endpoint.role != constraintIREndpointWorld &&
                    endpoint.objectIndex == kConstraintIRInvalidIndex
                )) {
                return failure(
                    ConstraintIRStatus::invalidEndpoint,
                    "constraint endpoint is invalid",
                    blockIndex
                );
            }
        }
        if (isContact) {
            const ConstraintIREndpoint& endpointA =
                ir.endpoints[block.endpointOffset];
            const ConstraintIREndpoint& endpointB =
                ir.endpoints[block.endpointOffset + 1u];
            if (endpointA.role != constraintIREndpointA ||
                endpointB.role != constraintIREndpointB ||
                endpointA.objectIndex == endpointB.objectIndex) {
                return failure(
                    ConstraintIRStatus::invalidEndpoint,
                    "contact endpoints are not canonical A/B objects",
                    blockIndex
                );
            }
        }

        for (std::uint32_t local = 0u;
             local < block.dimension;
             ++local) {
            const std::uint32_t rowIndex = block.rowOffset + local;
            const ConstraintIRRow& row = ir.rows[rowIndex];
            if (!finite(row.direction) ||
                !finite(row.positionError) ||
                !finite(row.targetVelocity) ||
                !finite(row.compliance) ||
                !finite(row.dissipation) ||
                !finite(row.timeConstant) ||
                !finite(row.dampingRatio) ||
                !finite(row.impulseLower) ||
                !finite(row.impulseUpper)) {
                return failure(
                    ConstraintIRStatus::nonfiniteData,
                    "constraint row contains non-finite data",
                    blockIndex,
                    rowIndex
                );
            }
            if (row.direction.w != 0.0F ||
                row.compliance < 0.0F ||
                row.dissipation < 0.0F ||
                row.timeConstant <= 0.0F ||
                row.dampingRatio < 0.0F ||
                row.impulseLower > row.impulseUpper ||
                (row.flags & ~kKnownRowFlags) != 0u ||
                row.reserved0 != 0u ||
                row.reserved1 != 0u ||
                row.reserved2 != 0u) {
                return failure(
                    ConstraintIRStatus::invalidRow,
                    "constraint row semantics are invalid",
                    blockIndex,
                    rowIndex
                );
            }
            const double directionNorm = norm(xyz(row.direction));
            if (isContact &&
                std::abs(directionNorm - 1.0) >
                    kDirectionTolerance) {
                return failure(
                    ConstraintIRStatus::invalidRow,
                    "contact row direction is not unit length",
                    blockIndex,
                    rowIndex
                );
            }
            if (!isContact && directionNorm > kDirectionTolerance &&
                std::abs(directionNorm - 1.0) >
                    kDirectionTolerance) {
                return failure(
                    ConstraintIRStatus::invalidRow,
                    "spatial row direction is neither zero nor unit",
                    blockIndex,
                    rowIndex
                );
            }
            const float warm =
                ir.warmImpulses[block.impulseOffset + local];
            if (!finite(warm)) {
                return failure(
                    ConstraintIRStatus::nonfiniteData,
                    "warm impulse is non-finite",
                    blockIndex,
                    rowIndex
                );
            }
            if (warm <
                    row.impulseLower - kWarmStartTolerance ||
                warm >
                    row.impulseUpper + kWarmStartTolerance) {
                return failure(
                    ConstraintIRStatus::infeasibleWarmStart,
                    "warm impulse violates scalar row bounds",
                    blockIndex,
                    rowIndex
                );
            }
        }

        if (!isContact) {
            continue;
        }

        const ConstraintIRRow& normalRow =
            ir.rows[block.rowOffset];
        const ConstraintIRRow& tangentURow =
            ir.rows[block.rowOffset + 1u];
        const ConstraintIRRow& tangentVRow =
            ir.rows[block.rowOffset + 2u];
        const std::uint32_t expectedNormalFlags =
            constraintIRRowPositionStabilized |
            constraintIRRowUnilateral |
            constraintIRRowContactNormal;
        if (normalRow.flags != expectedNormalFlags ||
            tangentURow.flags != constraintIRRowContactTangent ||
            tangentVRow.flags != constraintIRRowContactTangent) {
            return failure(
                ConstraintIRStatus::invalidRow,
                "contact row roles are not canonical",
                blockIndex,
                block.rowOffset
            );
        }
        if (std::abs(dot(
                    xyz(normalRow.direction),
                    xyz(tangentURow.direction)
                )) > kOrthogonalityTolerance ||
            std::abs(dot(
                    xyz(normalRow.direction),
                    xyz(tangentVRow.direction)
                )) > kOrthogonalityTolerance ||
            std::abs(dot(
                    xyz(tangentURow.direction),
                    xyz(tangentVRow.direction)
                )) > kOrthogonalityTolerance) {
            return failure(
                ConstraintIRStatus::invalidRow,
                "contact frame is not orthogonal",
                blockIndex,
                block.rowOffset
            );
        }
        if (block.dimension == 4u &&
            ir.rows[block.rowOffset + 3u].flags !=
                constraintIRRowContactTorsion) {
            return failure(
                ConstraintIRStatus::invalidRow,
                "four-dimensional contact lacks a torsion row",
                blockIndex,
                block.rowOffset + 3u
            );
        }

        const ConstraintIRCone& cone = ir.cones[block.coneIndex];
        if (!coneCoefficientsValid(cone)) {
            return failure(
                ConstraintIRStatus::invalidCone,
                "contact cone parameters are invalid",
                blockIndex
            );
        }
        const float expectedNormalUpper =
            cone.maximumNormalImpulse == 0.0F
            ? kConstraintIRUnbounded
            : cone.maximumNormalImpulse;
        if (normalRow.impulseLower != 0.0F ||
            normalRow.impulseUpper != expectedNormalUpper) {
            return failure(
                ConstraintIRStatus::invalidRow,
                "contact normal bounds do not canonically mirror cone cap",
                blockIndex,
                block.rowOffset
            );
        }
        const auto hasCanonicalFreeBounds =
            [](const ConstraintIRRow& row) {
                return
                    row.impulseLower == -kConstraintIRUnbounded &&
                    row.impulseUpper == kConstraintIRUnbounded;
            };
        if (!hasCanonicalFreeBounds(tangentURow) ||
            !hasCanonicalFreeBounds(tangentVRow)) {
            const std::uint32_t invalidRow =
                !hasCanonicalFreeBounds(tangentURow)
                ? block.rowOffset + 1u
                : block.rowOffset + 2u;
            return failure(
                ConstraintIRStatus::invalidRow,
                "contact tangent rows must use canonical free bounds",
                blockIndex,
                invalidRow
            );
        }
        if (block.dimension == 4u &&
            !hasCanonicalFreeBounds(
                ir.rows[block.rowOffset + 3u]
            )) {
            return failure(
                ConstraintIRStatus::invalidRow,
                "contact torsion row must use canonical free bounds",
                blockIndex,
                block.rowOffset + 3u
            );
        }
        if (cone.rollingLength > 0.0F) {
            return failure(
                ConstraintIRStatus::unsupportedSemantics,
                "rolling friction requires executable rolling rows",
                blockIndex
            );
        }
        if (cone.adhesionImpulse > 0.0F) {
            return failure(
                ConstraintIRStatus::unsupportedSemantics,
                "adhesion requires a shifted-cone contact implementation",
                blockIndex
            );
        }
        if (cone.torsionalLength > 0.0F &&
            block.dimension != 4u) {
            return failure(
                ConstraintIRStatus::invalidCone,
                "torsional material requires a torsion row",
                blockIndex
            );
        }

        const double normalImpulse =
            ir.warmImpulses[block.impulseOffset];
        const double tangentU =
            ir.warmImpulses[block.impulseOffset + 1u];
        const double tangentV =
            ir.warmImpulses[block.impulseOffset + 2u];
        const double maximumNormal =
            cone.maximumNormalImpulse == 0.0F
            ? kConstraintIRUnbounded
            : cone.maximumNormalImpulse;
        const double scaledTangent = coneScaledTangent(
            tangentU,
            tangentV,
            cone.staticFrictionU,
            cone.staticFrictionV
        );
        if (normalImpulse < -kWarmStartTolerance ||
            normalImpulse >
                maximumNormal + kWarmStartTolerance ||
            scaledTangent >
                normalImpulse + kWarmStartTolerance) {
            return failure(
                ConstraintIRStatus::infeasibleWarmStart,
                "warm contact impulse violates its elliptic cone",
                blockIndex
            );
        }
        if (block.dimension == 4u) {
            const double torsion =
                ir.warmImpulses[block.impulseOffset + 3u];
            const double torsionLimit =
                geometricMean(
                    cone.staticFrictionU,
                    cone.staticFrictionV
                ) *
                cone.torsionalLength * normalImpulse;
            if (std::abs(torsion) >
                torsionLimit + kWarmStartTolerance) {
                return failure(
                    ConstraintIRStatus::infeasibleWarmStart,
                    "warm torsion impulse violates its patch limit",
                    blockIndex,
                    block.rowOffset + 3u
                );
            }
        }
    }

    if (expectedEndpoint != ir.endpoints.size() ||
        expectedRow != ir.rows.size() ||
        expectedImpulse != ir.warmImpulses.size() ||
        expectedCone != ir.cones.size()) {
        return failure(
            ConstraintIRStatus::invalidRange,
            "constraint payload contains unreferenced trailing records"
        );
    }
    return {};
}

ConstraintIREvaluationResult evaluateConstraintIR(
    const ConstraintIR& ir,
    const ConstraintIREvaluationInput& input,
    const ConstraintIREvaluationConfig& config
) {
    ConstraintIREvaluationResult result;
    result.diagnostics = validateConstraintIR(ir);
    if (!result.diagnostics.succeeded()) {
        return result;
    }
    if (!finite(config.timestep) ||
        !finite(config.penetrationSlop) ||
        !finite(config.maximumDepenetrationVelocity) ||
        !finite(config.minimumTimeConstantRatio) ||
        !finite(config.stictionTransitionVelocity) ||
        config.timestep <= 0.0 ||
        config.penetrationSlop < 0.0 ||
        config.maximumDepenetrationVelocity < 0.0 ||
        config.minimumTimeConstantRatio < 0.0 ||
        config.stictionTransitionVelocity < 0.0) {
        result.diagnostics = failure(
            ConstraintIRStatus::invalidEvaluationConfig,
            "constraint evaluation configuration is invalid"
        );
        return result;
    }
    if (input.relativeVelocities.size() != ir.rows.size() ||
        (
            !input.preSolveVelocities.empty() &&
            input.preSolveVelocities.size() != ir.rows.size()
        ) ||
        !allFinite(input.relativeVelocities) ||
        (
            !input.preSolveVelocities.empty() &&
            !allFinite(input.preSolveVelocities)
        )) {
        result.diagnostics = failure(
            ConstraintIRStatus::invalidEvaluationInput,
            "constraint evaluation velocity streams are invalid"
        );
        return result;
    }

    EvaluatedConstraintIR working;
    working.blocks = ir.blocks;
    working.endpoints = ir.endpoints;
    working.rows.resize(ir.rows.size());
    working.cones.resize(ir.cones.size());
    working.warmImpulses = ir.warmImpulses;

    const double h = config.timestep;
    for (std::uint32_t blockIndex = 0u;
         blockIndex < ir.blocks.size();
         ++blockIndex) {
        const ConstraintIRBlock& block = ir.blocks[blockIndex];
        const bool isContact = block.type == MR_CONSTRAINT_CONTACT;
        double restitutionVelocity = 0.0;

        if (isContact) {
            const ConstraintIRCone& sourceCone =
                ir.cones[block.coneIndex];
            const ConstraintIRRow& tangentURow =
                ir.rows[block.rowOffset + 1u];
            const ConstraintIRRow& tangentVRow =
                ir.rows[block.rowOffset + 2u];
            const double slip = std::hypot(
                static_cast<double>(
                    input.relativeVelocities[
                        block.rowOffset + 1u
                    ]
                ) - tangentURow.targetVelocity,
                static_cast<double>(
                    input.relativeVelocities[
                        block.rowOffset + 2u
                    ]
                ) - tangentVRow.targetVelocity
            );
            const double transition = std::max(
                config.stictionTransitionVelocity,
                static_cast<double>(
                    sourceCone.stictionTransitionVelocity
                )
            );
            const bool staticRegion = slip <= transition;
            EvaluatedConstraintIRCone evaluated{};
            evaluated.effectiveFrictionU = staticRegion
                ? sourceCone.staticFrictionU
                : sourceCone.dynamicFrictionU;
            evaluated.effectiveFrictionV = staticRegion
                ? sourceCone.staticFrictionV
                : sourceCone.dynamicFrictionV;
            evaluated.staticFrictionU =
                sourceCone.staticFrictionU;
            evaluated.staticFrictionV =
                sourceCone.staticFrictionV;
            evaluated.dynamicFrictionU =
                sourceCone.dynamicFrictionU;
            evaluated.dynamicFrictionV =
                sourceCone.dynamicFrictionV;
            evaluated.rollingLength = sourceCone.rollingLength;
            evaluated.torsionalLength =
                sourceCone.torsionalLength;
            evaluated.restitutionThreshold =
                sourceCone.restitutionThreshold;
            evaluated.adhesionImpulse =
                sourceCone.adhesionImpulse;
            evaluated.maximumNormalImpulse =
                sourceCone.maximumNormalImpulse;
            working.cones[block.coneIndex] = evaluated;
            // Static friction validates the authored cache. Slip selection
            // may narrow the executable cone, so publish only its exact
            // projection into the evaluated warm-start stream.
            projectEvaluatedWarmContact(
                block,
                evaluated,
                working.warmImpulses
            );
        }

        for (std::uint32_t local = 0u;
             local < block.dimension;
             ++local) {
            const std::uint32_t rowIndex = block.rowOffset + local;
            const ConstraintIRRow& source = ir.rows[rowIndex];
            const double relative =
                input.relativeVelocities[rowIndex];
            const double preSolve =
                input.preSolveVelocities.empty()
                ? relative
                : input.preSolveVelocities[rowIndex];
            double stabilization = 0.0;

            if ((source.flags &
                 constraintIRRowPositionStabilized) != 0u) {
                const double minimumTau =
                    config.minimumTimeConstantRatio * h;
                const double tau = std::max(
                    static_cast<double>(source.timeConstant),
                    minimumTau
                );
                double positionError = source.positionError;
                if ((source.flags &
                     constraintIRRowContactNormal) != 0u) {
                    positionError = std::min(
                        positionError + config.penetrationSlop,
                        0.0
                    );
                } else if ((source.flags &
                            constraintIRRowUnilateral) != 0u) {
                    positionError =
                        std::min(positionError, 0.0);
                }
                const double ratio = h / tau;
                const double denominator =
                    1.0 +
                    2.0 * source.dampingRatio * ratio +
                    ratio * ratio;
                if ((source.flags &
                     constraintIRRowUnilateral) != 0u) {
                    stabilization = std::max(
                        -h * positionError /
                            (tau * tau * denominator),
                        0.0
                    );
                } else {
                    const double relativeError =
                        relative - source.targetVelocity;
                    stabilization =
                        (
                            relativeError -
                            h * positionError / (tau * tau)
                        ) /
                        denominator;
                }
                stabilization = std::clamp(
                    stabilization,
                    (source.flags &
                     constraintIRRowUnilateral) != 0u
                        ? 0.0
                        : -config.maximumDepenetrationVelocity,
                    config.maximumDepenetrationVelocity
                );
            }

            if (isContact && local == 0u &&
                (block.flags & constraintIRBlockNewImpact) != 0u) {
                const ConstraintIRCone& cone =
                    ir.cones[block.coneIndex];
                const double incoming =
                    preSolve - source.targetVelocity;
                if (incoming <
                    -static_cast<double>(
                        cone.restitutionThreshold
                    )) {
                    restitutionVelocity =
                        -static_cast<double>(cone.restitution) *
                        incoming;
                    stabilization = std::max(
                        stabilization,
                        restitutionVelocity
                    );
                }
            }

            const double regularization =
                static_cast<double>(source.compliance) / (h * h) +
                static_cast<double>(source.dissipation) / h;
            const double target =
                static_cast<double>(source.targetVelocity) +
                stabilization;
            if (!representableAsFloat(target) ||
                !representableAsFloat(regularization) ||
                !representableAsFloat(stabilization)) {
                result.diagnostics = failure(
                    ConstraintIRStatus::nonfiniteData,
                    "evaluated constraint row is not finite FP32",
                    blockIndex,
                    rowIndex
                );
                return result;
            }

            EvaluatedConstraintIRRow evaluated{};
            evaluated.direction = source.direction;
            evaluated.targetVelocity =
                static_cast<float>(target);
            evaluated.regularization =
                static_cast<float>(regularization);
            evaluated.impulseLower = source.impulseLower;
            evaluated.impulseUpper = source.impulseUpper;
            evaluated.sourcePositionError =
                source.positionError;
            evaluated.stabilizationVelocity =
                static_cast<float>(stabilization);
            evaluated.sourceTargetVelocity =
                source.targetVelocity;
            evaluated.relativeVelocity =
                static_cast<float>(relative);
            evaluated.preSolveVelocity =
                static_cast<float>(preSolve);
            working.rows[rowIndex] = evaluated;
        }

        if (isContact) {
            working.cones[block.coneIndex].
                restitutionVelocity =
                    static_cast<float>(restitutionVelocity);
        }
    }

    working.semanticFingerprint =
        fingerprintConstraintSemantics(working);
    result.evaluated = std::move(working);
    return result;
}

ConstraintIREvaluationView makeConstraintIREvaluationView(
    const EvaluatedConstraintIR& evaluated,
    const ConstraintIRConsumer consumer
) noexcept {
    return {
        consumer,
        evaluated.blocks,
        evaluated.endpoints,
        evaluated.rows,
        evaluated.cones,
        evaluated.warmImpulses,
        evaluated.semanticFingerprint,
    };
}

namespace {

std::uint64_t fingerprintConstraintStreams(
    const std::span<const ConstraintIRBlock> blocks,
    const std::span<const ConstraintIREndpoint> endpoints,
    const std::span<const EvaluatedConstraintIRRow> rows,
    const std::span<const EvaluatedConstraintIRCone> cones,
    const std::span<const float> warmImpulses
) noexcept {
    std::uint64_t hash = kFnvOffset;
    hashWord(hash, static_cast<std::uint32_t>(
        blocks.size()
    ));
    hashWord(hash, static_cast<std::uint32_t>(
        endpoints.size()
    ));
    hashWord(hash, static_cast<std::uint32_t>(
        rows.size()
    ));
    hashWord(hash, static_cast<std::uint32_t>(
        cones.size()
    ));

    for (const ConstraintIRBlock& block : blocks) {
        for (const std::uint32_t word : block.key.words) {
            hashWord(hash, word);
        }
        hashWord(hash, block.type);
        hashWord(hash, block.dimension);
        hashWord(hash, block.flags);
        hashWord(hash, block.islandIndex);
        hashWord(hash, block.endpointOffset);
        hashWord(hash, block.endpointCount);
        hashWord(hash, block.rowOffset);
        hashWord(hash, block.impulseOffset);
        hashWord(hash, block.coneIndex);
        hashWord(hash, block.eventSlot);
        hashWord(hash, block.reserved0);
        hashWord(hash, block.reserved1);
    }
    for (const ConstraintIREndpoint& endpoint : endpoints) {
        hashWord(hash, endpoint.objectIndex);
        hashWord(hash, endpoint.articulationIndex);
        hashWord(hash, endpoint.linkIndex);
        hashWord(hash, endpoint.role);
        hashWord(hash, endpoint.jacobianKind);
        hashWord(hash, endpoint.flags);
        hashWord(hash, endpoint.reserved0);
        hashWord(hash, endpoint.reserved1);
        hashFloat4(hash, endpoint.anchor);
        hashFloat4(hash, endpoint.axis);
    }
    for (const EvaluatedConstraintIRRow& row : rows) {
        hashFloat4(hash, row.direction);
        hashFloat(hash, row.targetVelocity);
        hashFloat(hash, row.regularization);
        hashFloat(hash, row.impulseLower);
        hashFloat(hash, row.impulseUpper);
        hashFloat(hash, row.sourcePositionError);
        hashFloat(hash, row.stabilizationVelocity);
        hashFloat(hash, row.sourceTargetVelocity);
        hashFloat(hash, row.relativeVelocity);
        hashFloat(hash, row.preSolveVelocity);
        hashFloat(hash, row.reserved0);
        hashFloat(hash, row.reserved1);
        hashFloat(hash, row.reserved2);
    }
    for (const EvaluatedConstraintIRCone& cone : cones) {
        hashFloat(hash, cone.effectiveFrictionU);
        hashFloat(hash, cone.effectiveFrictionV);
        hashFloat(hash, cone.staticFrictionU);
        hashFloat(hash, cone.staticFrictionV);
        hashFloat(hash, cone.dynamicFrictionU);
        hashFloat(hash, cone.dynamicFrictionV);
        hashFloat(hash, cone.rollingLength);
        hashFloat(hash, cone.torsionalLength);
        hashFloat(hash, cone.restitutionVelocity);
        hashFloat(hash, cone.restitutionThreshold);
        hashFloat(hash, cone.adhesionImpulse);
        hashFloat(hash, cone.maximumNormalImpulse);
    }
    for (const float impulse : warmImpulses) {
        hashFloat(hash, impulse);
    }
    return hash;
}

} // namespace

std::uint64_t fingerprintConstraintSemantics(
    const EvaluatedConstraintIR& evaluated
) noexcept {
    return fingerprintConstraintStreams(
        evaluated.blocks,
        evaluated.endpoints,
        evaluated.rows,
        evaluated.cones,
        evaluated.warmImpulses
    );
}

namespace {

bool evaluatedConeValid(const EvaluatedConstraintIRCone& cone) {
    const bool effectiveIsStatic =
        cone.effectiveFrictionU == cone.staticFrictionU &&
        cone.effectiveFrictionV == cone.staticFrictionV;
    const bool effectiveIsDynamic =
        cone.effectiveFrictionU == cone.dynamicFrictionU &&
        cone.effectiveFrictionV == cone.dynamicFrictionV;
    const auto paired = [](const float u, const float v) {
        return (u == 0.0F && v == 0.0F) ||
            (u > 0.0F && v > 0.0F);
    };
    return
        finite(cone.effectiveFrictionU) &&
        finite(cone.effectiveFrictionV) &&
        finite(cone.staticFrictionU) &&
        finite(cone.staticFrictionV) &&
        finite(cone.dynamicFrictionU) &&
        finite(cone.dynamicFrictionV) &&
        finite(cone.rollingLength) &&
        finite(cone.torsionalLength) &&
        finite(cone.restitutionVelocity) &&
        finite(cone.restitutionThreshold) &&
        finite(cone.adhesionImpulse) &&
        finite(cone.maximumNormalImpulse) &&
        cone.staticFrictionU >= cone.dynamicFrictionU &&
        cone.staticFrictionV >= cone.dynamicFrictionV &&
        cone.dynamicFrictionU >= 0.0F &&
        cone.dynamicFrictionV >= 0.0F &&
        cone.rollingLength == 0.0F &&
        cone.torsionalLength >= 0.0F &&
        cone.restitutionVelocity >= 0.0F &&
        cone.restitutionThreshold >= 0.0F &&
        cone.adhesionImpulse == 0.0F &&
        cone.maximumNormalImpulse >= 0.0F &&
        paired(cone.staticFrictionU, cone.staticFrictionV) &&
        paired(cone.dynamicFrictionU, cone.dynamicFrictionV) &&
        paired(
            cone.effectiveFrictionU,
            cone.effectiveFrictionV
        ) &&
        (effectiveIsStatic || effectiveIsDynamic);
}

ConstraintIRDiagnostics validateConstraintIREvaluationView(
    const ConstraintIREvaluationView& view
) {
    constexpr std::size_t maximum =
        std::numeric_limits<std::uint32_t>::max();
    if ((view.consumer != ConstraintIRConsumer::quality &&
         view.consumer != ConstraintIRConsumer::throughput) ||
        view.blocks.size() > maximum ||
        view.endpoints.size() > maximum ||
        view.rows.size() > maximum ||
        view.cones.size() > maximum ||
        view.warmImpulses.size() > maximum) {
        return failure(
            ConstraintIRStatus::invalidResidualInput,
            "evaluated constraint view metadata is invalid"
        );
    }
    if (view.blocks.empty() &&
        (!view.endpoints.empty() || !view.rows.empty() ||
         !view.cones.empty() || !view.warmImpulses.empty())) {
        return failure(
            ConstraintIRStatus::invalidResidualInput,
            "empty evaluated block stream owns payload data"
        );
    }

    std::uint64_t expectedEndpoint = 0u;
    std::uint64_t expectedRow = 0u;
    std::uint64_t expectedImpulse = 0u;
    std::uint64_t expectedCone = 0u;
    ConstraintIRStableKey previousKey{};
    bool havePrevious = false;

    for (std::uint32_t blockIndex = 0u;
         blockIndex < view.blocks.size();
         ++blockIndex) {
        const ConstraintIRBlock& block = view.blocks[blockIndex];
        if (havePrevious &&
            !constraintIRKeyLess(previousKey, block.key)) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated constraint keys are not canonical",
                blockIndex
            );
        }
        previousKey = block.key;
        havePrevious = true;

        if (!knownConstraintType(block.type) ||
            block.dimension == 0u ||
            block.dimension > 6u ||
            (block.flags & ~kKnownBlockFlags) != 0u ||
            block.islandIndex == kConstraintIRInvalidIndex ||
            block.reserved0 != 0u ||
            block.reserved1 != 0u ||
            block.endpointOffset != expectedEndpoint ||
            block.rowOffset != expectedRow ||
            block.impulseOffset != expectedImpulse ||
            !rangeFits(
                block.endpointOffset,
                block.endpointCount,
                view.endpoints.size()
            ) ||
            !rangeFits(
                block.rowOffset,
                block.dimension,
                view.rows.size()
            ) ||
            !rangeFits(
                block.impulseOffset,
                block.dimension,
                view.warmImpulses.size()
            )) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated constraint block ranges are invalid",
                blockIndex
            );
        }
        expectedEndpoint += block.endpointCount;
        expectedRow += block.dimension;
        expectedImpulse += block.dimension;

        const bool isContact = block.type == MR_CONSTRAINT_CONTACT;
        if (isContact) {
            if ((block.dimension != 3u &&
                 block.dimension != 4u) ||
                block.endpointCount != 2u ||
                block.coneIndex != expectedCone ||
                block.coneIndex >= view.cones.size()) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated contact structure is invalid",
                    blockIndex
                );
            }
            ++expectedCone;
        } else if (block.coneIndex != kConstraintIRInvalidIndex) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated scalar block references a cone",
                blockIndex
            );
        }

        for (std::uint32_t local = 0u;
             local < block.endpointCount;
             ++local) {
            const ConstraintIREndpoint& endpoint =
                view.endpoints[block.endpointOffset + local];
            if (!knownEndpointRole(endpoint.role) ||
                !knownJacobianKind(endpoint.jacobianKind) ||
                endpoint.reserved0 != 0u ||
                endpoint.reserved1 != 0u ||
                !finite(endpoint.anchor) ||
                !finite(endpoint.axis) ||
                (
                    endpoint.role != constraintIREndpointWorld &&
                    endpoint.objectIndex == kConstraintIRInvalidIndex
                )) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated endpoint data is invalid",
                    blockIndex
                );
            }
        }
        if (isContact) {
            const ConstraintIREndpoint& endpointA =
                view.endpoints[block.endpointOffset];
            const ConstraintIREndpoint& endpointB =
                view.endpoints[block.endpointOffset + 1u];
            if (endpointA.role != constraintIREndpointA ||
                endpointB.role != constraintIREndpointB ||
                endpointA.objectIndex == endpointB.objectIndex) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated contact endpoints are not canonical",
                    blockIndex
                );
            }
        }

        for (std::uint32_t local = 0u;
             local < block.dimension;
             ++local) {
            const std::uint32_t rowIndex =
                block.rowOffset + local;
            const EvaluatedConstraintIRRow& row =
                view.rows[rowIndex];
            const float warm =
                view.warmImpulses[block.impulseOffset + local];
            if (!finite(row.direction) ||
                !finite(row.targetVelocity) ||
                !finite(row.regularization) ||
                !finite(row.impulseLower) ||
                !finite(row.impulseUpper) ||
                !finite(row.sourcePositionError) ||
                !finite(row.stabilizationVelocity) ||
                !finite(row.sourceTargetVelocity) ||
                !finite(row.relativeVelocity) ||
                !finite(row.preSolveVelocity) ||
                !finite(row.reserved0) ||
                !finite(row.reserved1) ||
                !finite(row.reserved2) ||
                !finite(warm) ||
                row.direction.w != 0.0F ||
                row.regularization < 0.0F ||
                row.impulseLower > row.impulseUpper ||
                row.reserved0 != 0.0F ||
                row.reserved1 != 0.0F ||
                row.reserved2 != 0.0F ||
                warm <
                    row.impulseLower - kWarmStartTolerance ||
                warm >
                    row.impulseUpper + kWarmStartTolerance) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated row or warm impulse is invalid",
                    blockIndex,
                    rowIndex
                );
            }
            const double directionNorm = norm(xyz(row.direction));
            if (isContact &&
                std::abs(directionNorm - 1.0) >
                    kDirectionTolerance) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated contact direction is not unit",
                    blockIndex,
                    rowIndex
                );
            }
            if (!isContact && directionNorm > kDirectionTolerance &&
                std::abs(directionNorm - 1.0) >
                    kDirectionTolerance) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated scalar direction is invalid",
                    blockIndex,
                    rowIndex
                );
            }
        }

        if (!isContact) {
            continue;
        }
        const EvaluatedConstraintIRCone& cone =
            view.cones[block.coneIndex];
        if (!evaluatedConeValid(cone) ||
            (cone.torsionalLength > 0.0F &&
             block.dimension != 4u)) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated contact cone is invalid",
                blockIndex
            );
        }
        const EvaluatedConstraintIRRow& normalRow =
            view.rows[block.rowOffset];
        const float expectedNormalUpper =
            cone.maximumNormalImpulse == 0.0F
            ? kConstraintIRUnbounded
            : cone.maximumNormalImpulse;
        if (normalRow.impulseLower != 0.0F ||
            normalRow.impulseUpper != expectedNormalUpper) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated normal bounds disagree with cone cap",
                blockIndex,
                block.rowOffset
            );
        }
        for (std::uint32_t local = 1u;
             local < block.dimension;
             ++local) {
            const EvaluatedConstraintIRRow& row =
                view.rows[block.rowOffset + local];
            if (row.impulseLower != -kConstraintIRUnbounded ||
                row.impulseUpper != kConstraintIRUnbounded) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated coupled row has finite scalar bounds",
                    blockIndex,
                    block.rowOffset + local
                );
            }
        }

        const Vec3 normalDirection =
            xyz(view.rows[block.rowOffset].direction);
        const Vec3 tangentUDirection =
            xyz(view.rows[block.rowOffset + 1u].direction);
        const Vec3 tangentVDirection =
            xyz(view.rows[block.rowOffset + 2u].direction);
        if (std::abs(dot(normalDirection, tangentUDirection)) >
                kOrthogonalityTolerance ||
            std::abs(dot(normalDirection, tangentVDirection)) >
                kOrthogonalityTolerance ||
            std::abs(dot(tangentUDirection, tangentVDirection)) >
                kOrthogonalityTolerance) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated contact frame is not orthogonal",
                blockIndex
            );
        }

        const double normalImpulse =
            view.warmImpulses[block.impulseOffset];
        const double tangentU =
            view.warmImpulses[block.impulseOffset + 1u];
        const double tangentV =
            view.warmImpulses[block.impulseOffset + 2u];
        const double maximumNormal =
            cone.maximumNormalImpulse == 0.0F
            ? static_cast<double>(kConstraintIRUnbounded)
            : cone.maximumNormalImpulse;
        const double scaledTangent = coneScaledTangent(
            tangentU,
            tangentV,
            cone.effectiveFrictionU,
            cone.effectiveFrictionV
        );
        if (normalImpulse < -kWarmStartTolerance ||
            normalImpulse >
                maximumNormal + kWarmStartTolerance ||
            scaledTangent >
                normalImpulse + kWarmStartTolerance) {
            return failure(
                ConstraintIRStatus::invalidResidualInput,
                "evaluated warm impulse is outside effective cone",
                blockIndex
            );
        }
        if (block.dimension == 4u) {
            const double torsion =
                view.warmImpulses[block.impulseOffset + 3u];
            const double torsionLimit =
                geometricMean(
                    cone.effectiveFrictionU,
                    cone.effectiveFrictionV
                ) * cone.torsionalLength * normalImpulse;
            if (std::abs(torsion) >
                torsionLimit + kWarmStartTolerance) {
                return failure(
                    ConstraintIRStatus::invalidResidualInput,
                    "evaluated warm torsion is outside effective patch",
                    blockIndex,
                    block.rowOffset + 3u
                );
            }
        }
    }

    if (expectedEndpoint != view.endpoints.size() ||
        expectedRow != view.rows.size() ||
        expectedImpulse != view.warmImpulses.size() ||
        expectedCone != view.cones.size()) {
        return failure(
            ConstraintIRStatus::invalidResidualInput,
            "evaluated view contains trailing payload data"
        );
    }
    if (view.semanticFingerprint != fingerprintConstraintStreams(
            view.blocks,
            view.endpoints,
            view.rows,
            view.cones,
            view.warmImpulses
        )) {
        return failure(
            ConstraintIRStatus::invalidResidualInput,
            "evaluated view fingerprint is stale or forged"
        );
    }
    return {};
}

} // namespace

bool ConstraintIRResidualReport::withinTolerance(
    const ConstraintIRResidualConfig& config
) const noexcept {
    if (!succeeded() || !finite(config.residualTolerance) ||
        config.residualTolerance < 0.0) {
        return false;
    }
    const double maximum = std::max({
        maximumNaturalResidual,
        maximumPrimalViolation,
        maximumDualViolation,
        maximumComplementarityResidual,
        maximumScalarKktResidual,
    });
    return maximum <= config.residualTolerance;
}

ConstraintIRResidualReport evaluateConstraintIRResidual(
    const ConstraintIREvaluationView& semantics,
    const std::span<const float> relativeVelocities,
    const std::span<const float> impulses,
    const ConstraintIRResidualConfig& config
) {
    ConstraintIRResidualReport report;
    if (!finite(config.projectionStep) ||
        !finite(config.impulseTolerance) ||
        !finite(config.residualTolerance) ||
        config.projectionStep < kMinimumProjectionStep ||
        config.projectionStep > kMaximumProjectionStep ||
        config.impulseTolerance < 0.0 ||
        config.residualTolerance < 0.0 ||
        relativeVelocities.size() != semantics.rows.size() ||
        impulses.size() != semantics.rows.size() ||
        !allFinite(relativeVelocities) ||
        !allFinite(impulses)) {
        report.status = ConstraintIRStatus::invalidResidualInput;
        report.message =
            "constraint residual configuration or streams are invalid";
        return report;
    }
    const ConstraintIRDiagnostics viewDiagnostics =
        validateConstraintIREvaluationView(semantics);
    if (!viewDiagnostics.succeeded()) {
        report.status = ConstraintIRStatus::invalidResidualInput;
        report.message = viewDiagnostics.message;
        return report;
    }

    for (const ConstraintIRBlock& block : semantics.blocks) {
        if ((block.flags & constraintIRBlockDisabled) != 0u) {
            continue;
        }
        ++report.activeBlocks;
        report.scalarRows += block.dimension;
        std::array<double, 6> gradient{};
        for (std::uint32_t local = 0u;
             local < block.dimension;
             ++local) {
            const std::uint32_t rowIndex = block.rowOffset + local;
            gradient[local] =
                static_cast<double>(relativeVelocities[rowIndex]) -
                semantics.rows[rowIndex].targetVelocity +
                static_cast<double>(
                    semantics.rows[rowIndex].regularization
                ) *
                impulses[rowIndex];
        }

        if (block.type != MR_CONSTRAINT_CONTACT) {
            for (std::uint32_t local = 0u;
                 local < block.dimension;
                 ++local) {
                const std::uint32_t rowIndex =
                    block.rowOffset + local;
                const EvaluatedConstraintIRRow& row =
                    semantics.rows[rowIndex];
                const double impulse = impulses[rowIndex];
                report.maximumPrimalViolation = std::max({
                    report.maximumPrimalViolation,
                    static_cast<double>(row.impulseLower) -
                        impulse,
                    impulse -
                        static_cast<double>(row.impulseUpper),
                    0.0,
                });
                const double natural = boundNaturalResidual(
                    impulse,
                    gradient[local],
                    row.impulseLower,
                    row.impulseUpper,
                    config.projectionStep
                );
                report.maximumNaturalResidual = std::max(
                    report.maximumNaturalResidual,
                    natural
                );
                report.maximumScalarKktResidual = std::max(
                    report.maximumScalarKktResidual,
                    natural
                );
            }
            continue;
        }

        if (block.coneIndex >= semantics.cones.size() ||
            block.dimension < 3u) {
            report.status = ConstraintIRStatus::invalidResidualInput;
            report.message =
                "evaluated contact ranges are inconsistent";
            return report;
        }
        const EvaluatedConstraintIRCone& cone =
            semantics.cones[block.coneIndex];
        const double normal =
            impulses[block.impulseOffset];
        const double tangentU =
            impulses[block.impulseOffset + 1u];
        const double tangentV =
            impulses[block.impulseOffset + 2u];
        const double muU = cone.effectiveFrictionU;
        const double muV = cone.effectiveFrictionV;
        const bool hasTorsion = block.dimension == 4u;
        const double torsion = hasTorsion
            ? impulses[block.impulseOffset + 3u]
            : 0.0;
        const double torsionScale = hasTorsion
            ? geometricMean(muU, muV) * cone.torsionalLength
            : 0.0;
        const bool hasScaledTorsion =
            hasTorsion && torsionScale > 0.0;
        if (hasTorsion) {
            ++report.coupledTorsionContacts;
        }
        const double maximumNormal =
            cone.maximumNormalImpulse == 0.0F
            ? static_cast<double>(kConstraintIRUnbounded)
            : cone.maximumNormalImpulse;
        const bool capped =
            maximumNormal < kConstraintIRUnbounded &&
            normal >= maximumNormal - config.impulseTolerance;
        if (capped) {
            ++report.cappedContacts;
        }

        report.maximumPrimalViolation = std::max({
            report.maximumPrimalViolation,
            -normal,
            normal - maximumNormal,
            0.0,
        });

        if (muU == 0.0 && muV == 0.0) {
            report.maximumPrimalViolation = std::max({
                report.maximumPrimalViolation,
                std::abs(tangentU),
                std::abs(tangentV),
                hasTorsion ? std::abs(torsion) : 0.0,
            });
            const double projectedNormal = std::clamp(
                normal -
                    config.projectionStep * gradient[0],
                0.0,
                maximumNormal
            );
            const double natural = std::sqrt(
                (normal - projectedNormal) *
                    (normal - projectedNormal) +
                tangentU * tangentU +
                tangentV * tangentV +
                (hasTorsion ? torsion * torsion : 0.0)
            ) / config.projectionStep;
            report.maximumNaturalResidual = std::max(
                report.maximumNaturalResidual,
                natural
            );
            if (!hasTorsion && !capped) {
                report.maximumDualViolation = std::max(
                    report.maximumDualViolation,
                    std::max(-gradient[0], 0.0)
                );
                report.maximumComplementarityResidual = std::max(
                    report.maximumComplementarityResidual,
                    std::abs(normal * gradient[0])
                );
            }
        } else {
            const double scaledTangent =
                std::hypot(tangentU / muU, tangentV / muV);
            report.maximumPrimalViolation = std::max(
                report.maximumPrimalViolation,
                std::max(scaledTangent - normal, 0.0)
            );
            if (hasTorsion) {
                const double torsionViolation = hasScaledTorsion
                    ? std::abs(torsion / torsionScale) - normal
                    : std::abs(torsion);
                report.maximumPrimalViolation = std::max(
                    report.maximumPrimalViolation,
                    std::max(torsionViolation, 0.0)
                );
            }

            const FrictionPatchPoint transformed{
                normal,
                tangentU / muU,
                tangentV / muV,
                hasScaledTorsion ? torsion / torsionScale : 0.0,
            };
            const FrictionPatchPoint candidate{
                transformed.normal -
                    config.projectionStep * gradient[0],
                transformed.tangentU -
                    config.projectionStep * muU * gradient[1],
                transformed.tangentV -
                    config.projectionStep * muV * gradient[2],
                hasScaledTorsion
                    ? transformed.torsion -
                        config.projectionStep *
                            torsionScale * gradient[3]
                    : 0.0,
            };
            const FrictionPatchPoint projected =
                projectTruncatedFrictionPatch(
                    candidate,
                    maximumNormal,
                    hasScaledTorsion
                );
            double naturalSquared =
                (transformed.normal - projected.normal) *
                    (transformed.normal - projected.normal) +
                (transformed.tangentU - projected.tangentU) *
                    (transformed.tangentU - projected.tangentU) +
                (transformed.tangentV - projected.tangentV) *
                    (transformed.tangentV - projected.tangentV);
            if (hasScaledTorsion) {
                naturalSquared +=
                    (transformed.torsion - projected.torsion) *
                    (transformed.torsion - projected.torsion);
            } else if (hasTorsion) {
                // A zero torsional scale degenerates the fourth feasible
                // coordinate to exactly zero.
                naturalSquared += torsion * torsion;
            }
            const double natural =
                std::sqrt(naturalSquared) /
                config.projectionStep;
            report.maximumNaturalResidual = std::max(
                report.maximumNaturalResidual,
                natural
            );

            if (!hasTorsion && !capped) {
                const double dualTangent = std::hypot(
                    muU * gradient[1],
                    muV * gradient[2]
                );
                report.maximumDualViolation = std::max(
                    report.maximumDualViolation,
                    std::max(dualTangent - gradient[0], 0.0)
                );
                report.maximumComplementarityResidual = std::max(
                    report.maximumComplementarityResidual,
                    std::abs(
                        normal * gradient[0] +
                        tangentU * gradient[1] +
                        tangentV * gradient[2]
                    )
                );
            }
        }
    }

    const std::array<double, 5> diagnostics{
        report.maximumNaturalResidual,
        report.maximumPrimalViolation,
        report.maximumDualViolation,
        report.maximumComplementarityResidual,
        report.maximumScalarKktResidual,
    };
    if (!std::all_of(
            diagnostics.begin(),
            diagnostics.end(),
            [](const double value) { return finite(value); }
        )) {
        report.status = ConstraintIRStatus::nonfiniteData;
        report.message =
            "constraint residual produced non-finite diagnostics";
    }
    return report;
}

ConstraintIRV1AdapterResult adaptV1ContactsToConstraintIR(
    const std::span<const MRContactConstraintGPU> contacts,
    const ConstraintIRV1AdapterConfig& config
) {
    ConstraintIRV1AdapterResult result;
    if (!validV1AdapterConfig(config)) {
        result.diagnostics = failure(
            ConstraintIRStatus::invalidEvaluationConfig,
            "v1 adapter configuration is invalid"
        );
        return result;
    }
    if (contacts.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        result.diagnostics = failure(
            ConstraintIRStatus::invalidCount,
            "v1 contact stream exceeds 32-bit indexing"
        );
        return result;
    }

    std::vector<std::uint32_t> order(contacts.size());
    std::iota(order.begin(), order.end(), 0u);
    for (std::uint32_t index = 0u;
         index < contacts.size();
         ++index) {
        const ConstraintIRDiagnostics diagnostics =
            validateV1Contact(contacts[index], index);
        if (!diagnostics.succeeded()) {
            result.diagnostics = diagnostics;
            return result;
        }
    }
    std::sort(
        order.begin(),
        order.end(),
        [&](const std::uint32_t left,
            const std::uint32_t right) {
            return std::tie(
                contacts[left].pairKey,
                contacts[left].featureKey
            ) < std::tie(
                contacts[right].pairKey,
                contacts[right].featureKey
            );
        }
    );
    for (std::size_t index = 1u; index < order.size(); ++index) {
        const MRContactConstraintGPU& previous =
            contacts[order[index - 1u]];
        const MRContactConstraintGPU& current =
            contacts[order[index]];
        if (previous.pairKey == current.pairKey &&
            previous.featureKey == current.featureKey) {
            result.diagnostics = failure(
                ConstraintIRStatus::nonCanonicalOrder,
                "v1 contact stream contains a duplicate stable key",
                order[index]
            );
            return result;
        }
    }

    ConstraintIR working;
    std::vector<float> preSolveWorking;
    working.blocks.reserve(contacts.size());
    working.endpoints.reserve(2u * contacts.size());
    working.cones.reserve(contacts.size());

    for (const std::uint32_t sourceIndex : order) {
        const MRContactConstraintGPU& contact =
            contacts[sourceIndex];
        const Vec3 normal = xyz(contact.normal);
        const auto [tangentU, tangentV] = contactBasis(normal);
        const bool hasTorsion =
            contact.friction.w > 0.0F ||
            contact.impulses.w != 0.0F;
        const std::uint32_t dimension = hasTorsion ? 4u : 3u;

        ConstraintIRBlock block{};
        block.key = v1Key(contact);
        block.type = MR_CONSTRAINT_CONTACT;
        block.dimension = dimension;
        block.flags = contact.flags;
        block.islandIndex = contact.islandIndex;
        block.endpointOffset =
            static_cast<std::uint32_t>(working.endpoints.size());
        block.endpointCount = 2u;
        block.rowOffset =
            static_cast<std::uint32_t>(working.rows.size());
        block.impulseOffset = static_cast<std::uint32_t>(
            working.warmImpulses.size()
        );
        block.coneIndex =
            static_cast<std::uint32_t>(working.cones.size());
        working.blocks.push_back(block);

        ConstraintIREndpoint endpointA{};
        endpointA.objectIndex = contact.bodyA;
        endpointA.role = constraintIREndpointA;
        endpointA.jacobianKind =
            constraintIRJacobianWorldPoint;
        endpointA.anchor = {
            contact.pointAndSeparation.x,
            contact.pointAndSeparation.y,
            contact.pointAndSeparation.z,
            0.0F,
        };
        ConstraintIREndpoint endpointB = endpointA;
        endpointB.objectIndex = contact.bodyB;
        endpointB.role = constraintIREndpointB;
        working.endpoints.push_back(endpointA);
        working.endpoints.push_back(endpointB);

        const Vec3 surfaceVelocity =
            xyz(contact.targetVelocityAndPreSolveNormal);
        const std::array<Vec3, 4> directions{
            normal,
            tangentU,
            tangentV,
            normal,
        };
        const std::array<float, 4> targets{
            static_cast<float>(dot(surfaceVelocity, normal)),
            static_cast<float>(dot(surfaceVelocity, tangentU)),
            static_cast<float>(dot(surfaceVelocity, tangentV)),
            0.0F,
        };
        for (std::uint32_t local = 0u;
             local < dimension;
             ++local) {
            ConstraintIRRow row{};
            row.direction = f4(
                directions[local].x,
                directions[local].y,
                directions[local].z
            );
            row.positionError =
                local == 0u
                ? contact.pointAndSeparation.w
                : 0.0F;
            row.targetVelocity = targets[local];
            row.compliance =
                local == 0u ? contact.response.z : 0.0F;
            row.dissipation =
                local == 0u ? config.dissipation : 0.0F;
            row.timeConstant = config.timeConstant;
            row.dampingRatio = config.dampingRatio;
            if (local == 0u) {
                row.impulseLower = 0.0F;
                row.impulseUpper =
                    contact.response.w > 0.0F
                    ? contact.response.w
                    : kConstraintIRUnbounded;
                row.flags =
                    constraintIRRowPositionStabilized |
                    constraintIRRowUnilateral |
                    constraintIRRowContactNormal;
            } else if (local < 3u) {
                row.flags = constraintIRRowContactTangent;
            } else {
                row.flags = constraintIRRowContactTorsion;
            }
            working.rows.push_back(row);
            preSolveWorking.push_back(
                local == 0u
                ? contact.targetVelocityAndPreSolveNormal.w
                : targets[local]
            );
        }
        working.warmImpulses.push_back(contact.impulses.x);
        working.warmImpulses.push_back(contact.impulses.y);
        working.warmImpulses.push_back(contact.impulses.z);
        if (hasTorsion) {
            working.warmImpulses.push_back(contact.impulses.w);
        }

        ConstraintIRCone cone{};
        cone.staticFrictionU = contact.friction.x;
        cone.staticFrictionV = contact.friction.x;
        cone.dynamicFrictionU = contact.friction.y;
        cone.dynamicFrictionV = contact.friction.y;
        cone.rollingLength = contact.friction.z;
        cone.torsionalLength = contact.friction.w;
        cone.restitution = contact.response.x;
        cone.restitutionThreshold = contact.response.y;
        cone.maximumNormalImpulse = contact.response.w;
        cone.stictionTransitionVelocity =
            config.stictionTransitionVelocity;
        working.cones.push_back(cone);
    }

    result.diagnostics = validateConstraintIR(working);
    if (!result.diagnostics.succeeded()) {
        return result;
    }
    result.ir = std::move(working);
    result.preSolveVelocities = std::move(preSolveWorking);
    return result;
}

} // namespace metalrobo
