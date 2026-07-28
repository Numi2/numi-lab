#include "metalrobo/ArticulatedRigidWorld.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

bool finite(const double value) {
    return std::isfinite(value);
}

bool representableAsFloat(const double value) {
    return finite(value) &&
        std::abs(value) <=
            static_cast<double>(std::numeric_limits<float>::max());
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
}

Vec3 operator*(const double scale, const Vec3 value) {
    return {
        scale * value.x,
        scale * value.y,
        scale * value.z,
    };
}

double dot(const Vec3 left, const Vec3 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

MRStepStatusCode codeForDynamics(
    const ArticulatedDynamicsStatus status
) {
    switch (status) {
    case ArticulatedDynamicsStatus::success:
        return MR_STEP_SUCCESS;
    case ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite:
        return MR_STEP_FACTORIZATION_FAILED;
    case ArticulatedDynamicsStatus::nonlinearSolveFailed:
        return MR_STEP_DID_NOT_CONVERGE;
    case ArticulatedDynamicsStatus::unsupportedTopology:
        return MR_STEP_UNSUPPORTED;
    case ArticulatedDynamicsStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case ArticulatedDynamicsStatus::invalidModel:
    case ArticulatedDynamicsStatus::invalidDimensions:
    case ArticulatedDynamicsStatus::nonfiniteInput:
    case ArticulatedDynamicsStatus::invalidQuaternion:
    case ArticulatedDynamicsStatus::jointLimitViolation:
    case ArticulatedDynamicsStatus::bodySpeedLimitViolation:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

MRStepStatusCode codeForActuation(
    const ArticulatedActuationStatus status
) {
    switch (status) {
    case ArticulatedActuationStatus::success:
        return MR_STEP_SUCCESS;
    case ArticulatedActuationStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case ArticulatedActuationStatus::invalidConfiguration:
    case ArticulatedActuationStatus::invalidArticulation:
    case ArticulatedActuationStatus::invalidDimensions:
    case ArticulatedActuationStatus::nonfiniteInput:
    case ArticulatedActuationStatus::invalidDofMetadata:
    case ArticulatedActuationStatus::invalidCommandMode:
    case ArticulatedActuationStatus::invalidCommandSemantics:
    case ArticulatedActuationStatus::rootActuationForbidden:
    case ArticulatedActuationStatus::unactuatedDof:
    case ArticulatedActuationStatus::missingEffortLimit:
    case ArticulatedActuationStatus::missingModelDrive:
    case ArticulatedActuationStatus::positionCoordinateUnavailable:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

MRStepStatusCode codeForJointLimits(
    const ArticulatedJointLimitStatus status
) {
    switch (status) {
    case ArticulatedJointLimitStatus::success:
        return MR_STEP_SUCCESS;
    case ArticulatedJointLimitStatus::capacityExceeded:
        return MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
    case ArticulatedJointLimitStatus::factorizationFailure:
        return MR_STEP_FACTORIZATION_FAILED;
    case ArticulatedJointLimitStatus::didNotConverge:
        return MR_STEP_DID_NOT_CONVERGE;
    case ArticulatedJointLimitStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case ArticulatedJointLimitStatus::invalidConfiguration:
    case ArticulatedJointLimitStatus::invalidArticulation:
    case ArticulatedJointLimitStatus::invalidDimensions:
    case ArticulatedJointLimitStatus::nonfiniteInput:
    case ArticulatedJointLimitStatus::invalidDofMetadata:
    case ArticulatedJointLimitStatus::dynamicsFailure:
    case ArticulatedJointLimitStatus::invalidProblem:
    case ArticulatedJointLimitStatus::invalidWarmStart:
    case ArticulatedJointLimitStatus::invalidImpulse:
    case ArticulatedJointLimitStatus::solverFailure:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

MRStepStatusCode codeForCoupled(
    const CoupledArticulatedRigidContactStatus status,
    const MRStepStatusCode solverCode
) {
    switch (status) {
    case CoupledArticulatedRigidContactStatus::success:
        return MR_STEP_SUCCESS;
    case CoupledArticulatedRigidContactStatus::factorizationFailure:
        return MR_STEP_FACTORIZATION_FAILED;
    case CoupledArticulatedRigidContactStatus::solverFailure:
        return solverCode == MR_STEP_SUCCESS
            ? MR_STEP_DID_NOT_CONVERGE
            : solverCode;
    case CoupledArticulatedRigidContactStatus::dynamicsFailure:
        return MR_STEP_NONFINITE_RESULT;
    case CoupledArticulatedRigidContactStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case CoupledArticulatedRigidContactStatus::invalidModel:
    case CoupledArticulatedRigidContactStatus::invalidDimensions:
    case CoupledArticulatedRigidContactStatus::invalidRigidBody:
    case CoupledArticulatedRigidContactStatus::invalidContact:
    case CoupledArticulatedRigidContactStatus::invalidJointLimit:
    case CoupledArticulatedRigidContactStatus::nonfiniteInput:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

ArticulatedRigidWorldStepDiagnostics fail(
    ArticulatedRigidWorldStepDiagnostics diagnostics,
    const MRStepStatusCode code,
    const ArticulatedRigidWorldFailure failure
) {
    diagnostics.code = code;
    diagnostics.failure = failure;
    return diagnostics;
}

bool validGraspConfig(
    const EngineModel& model,
    const ArticulatedRigidGraspConfig& config
) {
    if (!config.enabled) {
        return true;
    }
    return
        config.jawBodyA < model.bodies.size() &&
        config.jawBodyB < model.bodies.size() &&
        config.jawBodyA != config.jawBodyB &&
        model.bodies[config.jawBodyA].articulationIndex !=
            MR_INVALID_INDEX &&
        model.bodies[config.jawBodyB].articulationIndex !=
            MR_INVALID_INDEX &&
        config.minimumNormalImpulse >= 0.0 &&
        finite(config.minimumNormalImpulse) &&
        config.minimumFriction >= 0.0 &&
        finite(config.minimumFriction) &&
        config.maximumTangentialSlipSpeed >= 0.0 &&
        finite(config.maximumTangentialSlipSpeed) &&
        config.maximumOpposingNormalDot >= -1.0 &&
        config.maximumOpposingNormalDot <= 1.0 &&
        finite(config.maximumOpposingNormalDot) &&
        config.requiredConsecutiveSteps > 0u;
}

bool validCache(
    const ArticulatedRigidWorldCache& cache,
    const std::size_t rigidBodyCount
) {
    for (std::size_t index = 0u;
         index < cache.contactImpulses.size();
         ++index) {
        const ArticulatedRigidContactCacheEntry& entry =
            cache.contactImpulses[index];
        if (!std::ranges::all_of(
                entry.warmStart.worldImpulseOnB,
                finite
            ) ||
            entry.lastSeenStep > cache.step) {
            return false;
        }
        for (std::size_t previous = 0u;
             previous < index;
             ++previous) {
            if (cache.contactImpulses[previous].warmStart.key ==
                entry.warmStart.key) {
                return false;
            }
        }
    }
    for (std::size_t index = 0u;
         index < cache.jointLimitImpulses.size();
         ++index) {
        const ArticulatedRigidJointLimitCacheEntry& entry =
            cache.jointLimitImpulses[index];
        if (!finite(entry.impulse) ||
            entry.impulse < 0.0 ||
            entry.lastSeenStep > cache.step) {
            return false;
        }
        for (std::size_t previous = 0u;
             previous < index;
             ++previous) {
            if (cache.jointLimitImpulses[previous].stableKey ==
                entry.stableKey) {
                return false;
            }
        }
    }
    for (std::size_t index = 0u;
         index < cache.graspEvidence.size();
         ++index) {
        const ArticulatedRigidGraspCacheEntry& entry =
            cache.graspEvidence[index];
        if (entry.rigidBody >= rigidBodyCount ||
            entry.identity == 0u ||
            entry.lastSeenStep > cache.step) {
            return false;
        }
        for (std::size_t previous = 0u;
             previous < index;
             ++previous) {
            if (cache.graspEvidence[previous].rigidBody ==
                entry.rigidBody) {
                return false;
            }
        }
    }
    return true;
}

void hashIdentityWord(
    std::uint64_t& hash,
    std::uint64_t value
) {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::size_t byte = 0u; byte < 8u; ++byte) {
        hash ^= value & 0xffu;
        hash *= prime;
        value >>= 8u;
    }
}

std::uint64_t graspIdentity(
    const EngineModel& model,
    const ArticulatedRigidGraspConfig& config,
    const std::span<const MRBodyStateGPU> rigidBodies,
    const std::span<const MRShapeGPU> rigidShapes,
    const std::size_t rigidBody
) {
    if (rigidBody >= rigidBodies.size()) {
        return 0u;
    }
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char character : model.name) {
        hash ^= character;
        hash *= 1099511628211ull;
    }
    hashIdentityWord(hash, model.articulations.size());
    hashIdentityWord(hash, model.bodies.size());
    hashIdentityWord(hash, model.shapes.size());
    hashIdentityWord(hash, config.jawBodyA);
    hashIdentityWord(hash, config.jawBodyB);
    hashIdentityWord(
        hash,
        std::bit_cast<std::uint64_t>(config.minimumNormalImpulse)
    );
    hashIdentityWord(
        hash,
        std::bit_cast<std::uint64_t>(config.minimumFriction)
    );
    hashIdentityWord(
        hash,
        std::bit_cast<std::uint64_t>(
            config.maximumTangentialSlipSpeed
        )
    );
    hashIdentityWord(
        hash,
        std::bit_cast<std::uint64_t>(
            config.maximumOpposingNormalDot
        )
    );
    hashIdentityWord(hash, config.requiredConsecutiveSteps);
    for (std::size_t shapeIndex = 0u;
         shapeIndex < model.shapes.size();
         ++shapeIndex) {
        const MRShapeGPU& shape = model.shapes[shapeIndex];
        if (shape.bodyIndex != config.jawBodyA &&
            shape.bodyIndex != config.jawBodyB) {
            continue;
        }
        hashIdentityWord(hash, shape.bodyIndex);
        hashIdentityWord(hash, shapeIndex);
        hashIdentityWord(hash, shape.slotGeneration);
    }

    hashIdentityWord(hash, rigidBody);
    hashIdentityWord(
        hash,
        rigidBodies[rigidBody].flagsAndIndices[2]
    );
    hashIdentityWord(
        hash,
        rigidBodies[rigidBody].flagsAndIndices[3]
    );
    for (std::size_t shapeIndex = 0u;
         shapeIndex < rigidShapes.size();
         ++shapeIndex) {
        const MRShapeGPU& shape = rigidShapes[shapeIndex];
        if (shape.bodyIndex != rigidBody) {
            continue;
        }
        hashIdentityWord(hash, shapeIndex);
        hashIdentityWord(hash, shape.slotGeneration);
    }
    return hash == 0u ? 1u : hash;
}

bool publishRigidVelocities(
    const std::span<const CoupledRigidBodyVelocity> velocities,
    const std::span<MRBodyStateGPU> bodies
) {
    if (velocities.size() != bodies.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < bodies.size(); ++index) {
        const CoupledRigidBodyVelocity& velocity = velocities[index];
        if (!std::ranges::all_of(
                velocity.linear,
                representableAsFloat
            ) ||
            !std::ranges::all_of(
                velocity.angular,
                representableAsFloat
            )) {
            return false;
        }
        const float inverseMass =
            bodies[index].linearVelocityAndInverseMass.w;
        bodies[index].linearVelocityAndInverseMass = {
            static_cast<float>(velocity.linear[0]),
            static_cast<float>(velocity.linear[1]),
            static_cast<float>(velocity.linear[2]),
            inverseMass,
        };
        bodies[index].angularVelocity = {
            static_cast<float>(velocity.angular[0]),
            static_cast<float>(velocity.angular[1]),
            static_cast<float>(velocity.angular[2]),
            0.0f,
        };
    }
    return true;
}

bool reduceBodyPairContacts(
    ArticulatedRigidIslandCollisionResult& collision,
    const std::uint32_t maximumPerPair
) {
    if (maximumPerPair == 0u ||
        collision.contacts.size() != collision.metadata.size()) {
        return false;
    }
    if (collision.contacts.size() <= maximumPerPair) {
        return true;
    }
    const auto canonicalPair = [](
        const ArticulatedRigidIslandContactMetadata& metadata
    ) {
        std::array<std::uint64_t, 2> endpointA{
            static_cast<std::uint32_t>(
                metadata.key.endpointA.kind
            ),
            metadata.key.endpointA.bodyIndex,
        };
        std::array<std::uint64_t, 2> endpointB{
            static_cast<std::uint32_t>(
                metadata.key.endpointB.kind
            ),
            metadata.key.endpointB.bodyIndex,
        };
        if (endpointB < endpointA) {
            std::swap(endpointA, endpointB);
        }
        return std::array<std::uint64_t, 4>{
            endpointA[0],
            endpointA[1],
            endpointB[0],
            endpointB[1],
        };
    };
    std::vector<std::size_t> ranked(collision.contacts.size());
    for (std::size_t index = 0u; index < ranked.size(); ++index) {
        ranked[index] = index;
    }
    std::ranges::sort(
        ranked,
        [&](const std::size_t left, const std::size_t right) {
            const ArticulatedRigidIslandContactMetadata& leftMetadata =
                collision.metadata[left];
            const ArticulatedRigidIslandContactMetadata& rightMetadata =
                collision.metadata[right];
            const auto leftPair = canonicalPair(leftMetadata);
            const auto rightPair = canonicalPair(rightMetadata);
            if (leftPair != rightPair) {
                return leftPair < rightPair;
            }
            if (leftMetadata.effectiveSeparation !=
                rightMetadata.effectiveSeparation) {
                return leftMetadata.effectiveSeparation <
                    rightMetadata.effectiveSeparation;
            }
            const ArticulatedRigidIslandContactKey& leftKey =
                leftMetadata.key;
            const ArticulatedRigidIslandContactKey& rightKey =
                rightMetadata.key;
            return std::array<std::uint64_t, 8>{
                leftKey.endpointA.shapeIndex,
                leftKey.endpointA.feature,
                leftKey.endpointA.slotGeneration,
                leftKey.endpointA.motionType,
                leftKey.endpointB.shapeIndex,
                leftKey.endpointB.feature,
                leftKey.endpointB.slotGeneration,
                leftKey.endpointB.motionType,
            } < std::array{
                static_cast<std::uint64_t>(
                    rightKey.endpointA.shapeIndex
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointA.feature
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointA.slotGeneration
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointA.motionType
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointB.shapeIndex
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointB.feature
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointB.slotGeneration
                ),
                static_cast<std::uint64_t>(
                    rightKey.endpointB.motionType
                ),
            };
        }
    );

    std::vector<std::size_t> retained;
    retained.reserve(collision.contacts.size());
    std::array<std::uint64_t, 4> previousPair{
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
    };
    std::uint32_t retainedForPair = 0u;
    for (const std::size_t candidate : ranked) {
        const ArticulatedRigidIslandContactMetadata& metadata =
            collision.metadata[candidate];
        const std::array<std::uint64_t, 4> pair =
            canonicalPair(metadata);
        if (pair != previousPair) {
            previousPair = pair;
            retainedForPair = 0u;
        }
        if (retainedForPair < maximumPerPair) {
            retained.push_back(candidate);
            ++retainedForPair;
        }
    }
    std::ranges::sort(retained);
    std::vector<CoupledArticulatedRigidIslandContact> contacts;
    std::vector<ArticulatedRigidIslandContactMetadata> metadata;
    contacts.reserve(retained.size());
    metadata.reserve(retained.size());
    for (const std::size_t index : retained) {
        contacts.push_back(collision.contacts[index]);
        metadata.push_back(collision.metadata[index]);
    }
    collision.contacts = std::move(contacts);
    collision.metadata = std::move(metadata);
    collision.diagnostics.contactCount =
        static_cast<std::uint32_t>(collision.contacts.size());
    collision.diagnostics.articulatedDynamicContactCount = 0u;
    collision.diagnostics.articulatedPrescribedContactCount = 0u;
    collision.diagnostics.dynamicDynamicContactCount = 0u;
    collision.diagnostics.dynamicPrescribedContactCount = 0u;
    for (const ArticulatedRigidIslandContactMetadata& item :
         collision.metadata) {
        switch (item.pairClass) {
        case ArticulatedRigidIslandPairClass::
            articulatedDynamicScene:
            ++collision.diagnostics.
                articulatedDynamicContactCount;
            break;
        case ArticulatedRigidIslandPairClass::
            articulatedPrescribedScene:
            ++collision.diagnostics.
                articulatedPrescribedContactCount;
            break;
        case ArticulatedRigidIslandPairClass::
            dynamicSceneDynamicScene:
            ++collision.diagnostics.dynamicDynamicContactCount;
            break;
        case ArticulatedRigidIslandPairClass::
            dynamicScenePrescribedScene:
            ++collision.diagnostics.
                dynamicPrescribedContactCount;
            break;
        }
    }
    return true;
}

double maximumVelocityDifference(
    const std::span<const double> freeArticulation,
    const std::span<const double> postArticulation,
    const std::span<const MRBodyStateGPU> freeRigid,
    const std::span<const CoupledRigidBodyVelocity> postRigid
) {
    if (freeArticulation.size() != postArticulation.size() ||
        freeRigid.size() != postRigid.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (std::size_t index = 0u;
         index < freeArticulation.size();
         ++index) {
        maximum = std::max(
            maximum,
            std::abs(
                postArticulation[index] -
                freeArticulation[index]
            )
        );
    }
    for (std::size_t index = 0u; index < freeRigid.size(); ++index) {
        const std::array<double, 6> source{
            freeRigid[index].linearVelocityAndInverseMass.x,
            freeRigid[index].linearVelocityAndInverseMass.y,
            freeRigid[index].linearVelocityAndInverseMass.z,
            freeRigid[index].angularVelocity.x,
            freeRigid[index].angularVelocity.y,
            freeRigid[index].angularVelocity.z,
        };
        const std::array<double, 6> destination{
            postRigid[index].linear[0],
            postRigid[index].linear[1],
            postRigid[index].linear[2],
            postRigid[index].angular[0],
            postRigid[index].angular[1],
            postRigid[index].angular[2],
        };
        for (std::size_t axis = 0u; axis < source.size(); ++axis) {
            maximum = std::max(
                maximum,
                std::abs(destination[axis] - source[axis])
            );
        }
    }
    return maximum;
}

std::vector<ArticulatedRigidIslandContactWarmStart>
activeContactWarmStarts(
    const ArticulatedRigidWorldCache& cache
) {
    std::vector<ArticulatedRigidIslandContactWarmStart> result;
    result.reserve(cache.contactImpulses.size());
    for (const ArticulatedRigidContactCacheEntry& entry :
         cache.contactImpulses) {
        if (entry.lastSeenStep == cache.step) {
            result.push_back(entry.warmStart);
        }
    }
    return result;
}

std::vector<double> activeJointLimitWarmStarts(
    const ArticulatedRigidWorldCache& cache,
    const std::span<const ArticulatedJointLimitRow> rows,
    std::uint32_t& matchCount
) {
    std::vector<double> result(rows.size(), 0.0);
    for (std::size_t rowIndex = 0u;
         rowIndex < rows.size();
         ++rowIndex) {
        const auto entry = std::ranges::find_if(
            cache.jointLimitImpulses,
            [&](const ArticulatedRigidJointLimitCacheEntry& item) {
                return
                    item.stableKey == rows[rowIndex].stableKey &&
                    item.lastSeenStep == cache.step;
            }
        );
        if (entry != cache.jointLimitImpulses.end()) {
            result[rowIndex] = entry->impulse;
            ++matchCount;
        }
    }
    return result;
}

bool updateContactCache(
    ArticulatedRigidWorldCache& cache,
    const ArticulatedRigidIslandCollisionResult& collision,
    const std::span<const double> impulses,
    const std::uint64_t step,
    const std::uint64_t maximumAge
) {
    if (collision.contacts.size() != collision.metadata.size() ||
        impulses.size() != 3u * collision.contacts.size()) {
        return false;
    }
    for (std::size_t index = 0u;
         index < collision.contacts.size();
         ++index) {
        const CoupledArticulatedRigidIslandContact& contact =
            collision.contacts[index];
        const std::size_t offset = 3u * index;
        const Vec3 worldImpulse =
            impulses[offset] * vector(contact.normal) +
            impulses[offset + 1u] * vector(contact.tangentU) +
            impulses[offset + 2u] * vector(contact.tangentV);
        if (!finite(worldImpulse.x) ||
            !finite(worldImpulse.y) ||
            !finite(worldImpulse.z)) {
            return false;
        }
        ArticulatedRigidIslandContactWarmStart warm{
            .key = collision.metadata[index].key,
            .worldImpulseOnB = {
                worldImpulse.x,
                worldImpulse.y,
                worldImpulse.z,
            },
        };
        const auto existing = std::ranges::find_if(
            cache.contactImpulses,
            [&](const ArticulatedRigidContactCacheEntry& entry) {
                return entry.warmStart.key == warm.key;
            }
        );
        if (existing == cache.contactImpulses.end()) {
            cache.contactImpulses.push_back({
                .warmStart = warm,
                .lastSeenStep = step,
            });
        } else {
            existing->warmStart = warm;
            existing->lastSeenStep = step;
        }
    }
    std::erase_if(
        cache.contactImpulses,
        [step, maximumAge](
            const ArticulatedRigidContactCacheEntry& entry
        ) {
            return entry.lastSeenStep > step ||
                step - entry.lastSeenStep > maximumAge;
        }
    );
    return true;
}

bool updateJointLimitCache(
    ArticulatedRigidWorldCache& cache,
    const std::span<const ArticulatedJointLimitRow> rows,
    const std::span<const double> impulses,
    const std::uint64_t step,
    const std::uint64_t maximumAge
) {
    if (rows.size() != impulses.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < rows.size(); ++index) {
        if (!finite(impulses[index]) || impulses[index] < 0.0) {
            return false;
        }
        const auto existing = std::ranges::find_if(
            cache.jointLimitImpulses,
            [&](const ArticulatedRigidJointLimitCacheEntry& entry) {
                return entry.stableKey == rows[index].stableKey;
            }
        );
        if (existing == cache.jointLimitImpulses.end()) {
            cache.jointLimitImpulses.push_back({
                .stableKey = rows[index].stableKey,
                .impulse = impulses[index],
                .lastSeenStep = step,
            });
        } else {
            existing->impulse = impulses[index];
            existing->lastSeenStep = step;
        }
    }
    std::erase_if(
        cache.jointLimitImpulses,
        [step, maximumAge](
            const ArticulatedRigidJointLimitCacheEntry& entry
        ) {
            return entry.lastSeenStep > step ||
                step - entry.lastSeenStep > maximumAge;
        }
    );
    return true;
}

bool updateGraspEvidence(
    ArticulatedRigidWorldCache& cache,
    const EngineModel& model,
    const ArticulatedRigidIslandCollisionResult& collision,
    const CoupledArticulatedRigidContactDiagnostics& solve,
    const ArticulatedRigidGraspConfig& config,
    const std::span<const MRBodyStateGPU> rigidBodies,
    const std::span<const MRShapeGPU> rigidShapes,
    const std::uint64_t previousStep,
    const std::uint64_t step,
    std::vector<ArticulatedRigidGraspEvidence>& evidence
) {
    evidence.clear();
    if (!config.enabled) {
        cache.graspEvidence.clear();
        return true;
    }
    if (collision.contacts.size() != collision.metadata.size() ||
        solve.contactImpulses.size() !=
            3u * collision.contacts.size() ||
        solve.postContactVelocity.size() !=
            3u * collision.contacts.size()) {
        return false;
    }

    const std::size_t rigidBodyCount = rigidBodies.size();
    evidence.resize(rigidBodyCount);
    std::vector<Vec3> normalA(rigidBodyCount);
    std::vector<Vec3> normalB(rigidBodyCount);
    std::vector<std::uint64_t> identities(rigidBodyCount, 0u);
    for (std::size_t body = 0u; body < rigidBodyCount; ++body) {
        evidence[body].rigidBody =
            static_cast<std::uint32_t>(body);
        if (rigidBodies[body].flagsAndIndices[0] !=
            MR_MOTION_DYNAMIC) {
            continue;
        }
        identities[body] = graspIdentity(
            model,
            config,
            rigidBodies,
            rigidShapes,
            body
        );
        if (identities[body] == 0u) {
            return false;
        }
    }

    for (std::size_t contactIndex = 0u;
         contactIndex < collision.contacts.size();
         ++contactIndex) {
        const CoupledArticulatedRigidIslandContact& contact =
            collision.contacts[contactIndex];
        const bool articulatedA =
            contact.endpointA.kind ==
                CoupledContactEndpointKind::articulated;
        const bool articulatedB =
            contact.endpointB.kind ==
                CoupledContactEndpointKind::articulated;
        if (articulatedA == articulatedB) {
            continue;
        }
        const CoupledContactEndpoint& articulated =
            articulatedA ? contact.endpointA : contact.endpointB;
        const CoupledContactEndpoint& scene =
            articulatedA ? contact.endpointB : contact.endpointA;
        if (scene.kind != CoupledContactEndpointKind::sceneBody ||
            scene.body >= rigidBodyCount) {
            return false;
        }
        if (rigidBodies[scene.body].flagsAndIndices[0] !=
            MR_MOTION_DYNAMIC) {
            continue;
        }
        const std::size_t offset = 3u * contactIndex;
        const double normalImpulse =
            solve.contactImpulses[offset];
        const double slip = std::hypot(
            solve.postContactVelocity[offset + 1u],
            solve.postContactVelocity[offset + 2u]
        );
        if (!finite(normalImpulse) || !finite(slip)) {
            return false;
        }
        ArticulatedRigidGraspEvidence& item =
            evidence[scene.body];
        const double normalSign = articulatedA ? 1.0 : -1.0;
        const Vec3 objectNormal =
            normalSign * vector(contact.normal);
        if (articulated.body == config.jawBodyA) {
            item.maximumTangentialSlipSpeed = std::max(
                item.maximumTangentialSlipSpeed,
                slip
            );
            if (normalImpulse > item.jawANormalImpulse) {
                item.jawAContact = true;
                item.jawANormalImpulse = normalImpulse;
                item.jawAFriction =
                    contact.friction;
                normalA[scene.body] = objectNormal;
            }
        }
        if (articulated.body == config.jawBodyB) {
            item.maximumTangentialSlipSpeed = std::max(
                item.maximumTangentialSlipSpeed,
                slip
            );
            if (normalImpulse > item.jawBNormalImpulse) {
                item.jawBContact = true;
                item.jawBNormalImpulse = normalImpulse;
                item.jawBFriction =
                    contact.friction;
                normalB[scene.body] = objectNormal;
            }
        }
    }

    for (std::size_t body = 0u; body < evidence.size(); ++body) {
        ArticulatedRigidGraspEvidence& item = evidence[body];
        if (rigidBodies[body].flagsAndIndices[0] !=
            MR_MOTION_DYNAMIC) {
            continue;
        }
        if (item.jawAContact && item.jawBContact) {
            const double denominator =
                norm(normalA[body]) * norm(normalB[body]);
            if (!(denominator > 0.0) || !finite(denominator)) {
                return false;
            }
            item.normalDot =
                dot(normalA[body], normalB[body]) / denominator;
        }
        item.qualifiedThisStep =
            item.jawAContact &&
            item.jawBContact &&
            item.jawANormalImpulse >=
                config.minimumNormalImpulse &&
            item.jawBNormalImpulse >=
                config.minimumNormalImpulse &&
            item.jawAFriction >= config.minimumFriction &&
            item.jawBFriction >= config.minimumFriction &&
            item.maximumTangentialSlipSpeed <=
                config.maximumTangentialSlipSpeed &&
            item.normalDot <= config.maximumOpposingNormalDot;

        auto cached = std::ranges::find_if(
            cache.graspEvidence,
            [body](const ArticulatedRigidGraspCacheEntry& entry) {
                return entry.rigidBody == body;
            }
        );
        std::uint32_t consecutive = 0u;
        if (item.qualifiedThisStep) {
            consecutive = 1u;
            if (cached != cache.graspEvidence.end() &&
                cached->identity == identities[body] &&
                cached->lastSeenStep == previousStep &&
                cached->consecutiveQualifiedSteps <
                    std::numeric_limits<std::uint32_t>::max()) {
                consecutive =
                    cached->consecutiveQualifiedSteps + 1u;
            }
        }
        item.consecutiveQualifiedSteps = consecutive;
        item.grasped =
            consecutive >= config.requiredConsecutiveSteps;
        const ArticulatedRigidGraspCacheEntry next{
            .rigidBody = static_cast<std::uint32_t>(body),
            .identity = identities[body],
            .consecutiveQualifiedSteps = consecutive,
            .grasped = item.grasped,
            .lastSeenStep = step,
        };
        if (cached == cache.graspEvidence.end()) {
            cache.graspEvidence.push_back(next);
        } else {
            *cached = next;
        }
    }
    std::erase_if(
        cache.graspEvidence,
        [step](const ArticulatedRigidGraspCacheEntry& entry) {
            return entry.lastSeenStep != step;
        }
    );
    return true;
}

} // namespace

ArticulatedRigidWorldStepDiagnostics stepArticulatedRigidWorldCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<double> q,
    const std::span<double> v,
    const std::span<const double> generalizedForce,
    const std::span<const ArticulatedBodyWrench> articulatedWrenches,
    const std::span<const MRBodyPropertiesGPU> rigidProperties,
    const std::span<MRBodyStateGPU> rigidBodies,
    const std::span<const MRShapeGPU> rigidShapes,
    const std::span<const MRMaterialGPU> rigidMaterials,
    const std::span<const BodyWrench> rigidWrenches,
    const ArticulatedRigidWorldConfig& config,
    ArticulatedRigidWorldCache& cache
) {
    ArticulatedRigidWorldStepDiagnostics diagnostics;
    std::string modelFailure;
    if (!model.valid(&modelFailure) ||
        articulationIndex >= model.articulations.size() ||
        model.articulations.size() != 1u ||
        config.dynamics.integrator !=
            ArticulatedIntegrator::symplecticEuler ||
        config.rigidFreeMotion.integrator !=
            FreeBodyIntegrator::symplecticEuler ||
        !(config.dynamics.timestep > 0.0) ||
        !finite(config.dynamics.timestep) ||
        cache.step == std::numeric_limits<std::uint64_t>::max() ||
        rigidProperties.empty() ||
        rigidProperties.size() != rigidBodies.size() ||
        std::ranges::none_of(
            rigidProperties,
            [](const MRBodyPropertiesGPU& body) {
                return body.motionType == MR_MOTION_DYNAMIC;
            }
        ) ||
        rigidShapes.empty() ||
        rigidMaterials.empty() ||
        config.maximumContactsPerBodyPair == 0u ||
        (!rigidWrenches.empty() &&
         rigidWrenches.size() != rigidBodies.size()) ||
        !validGraspConfig(model, config.grasp) ||
        !validCache(cache, rigidBodies.size())) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedRigidWorldFailure::invalidConfiguration
        );
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    if (q.size() != articulation.nq ||
        v.size() != articulation.nv ||
        generalizedForce.size() != articulation.nv ||
        (!articulatedWrenches.empty() &&
         articulatedWrenches.size() != model.bodies.size()) ||
        std::ranges::any_of(
            rigidProperties,
            [](const MRBodyPropertiesGPU& body) {
                return
                    body.articulationIndex != MR_INVALID_INDEX ||
                    body.motionType > MR_MOTION_DYNAMIC;
            }
        )) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_INPUT,
            ArticulatedRigidWorldFailure::invalidConfiguration
        );
    }

    std::vector<double> acceleration(v.size(), 0.0);
    diagnostics.articulatedFreeDynamics =
        computeArticulatedForwardDynamics(
            model,
            articulationIndex,
            q,
            v,
            generalizedForce,
            articulatedWrenches,
            acceleration,
            config.dynamics
        );
    if (!diagnostics.articulatedFreeDynamics.succeeded()) {
        return fail(
            diagnostics,
            codeForDynamics(
                diagnostics.articulatedFreeDynamics.status
            ),
            ArticulatedRigidWorldFailure::articulatedFreeDynamics
        );
    }
    std::vector<double> freeArticulation(v.size(), 0.0);
    for (std::size_t dof = 0u; dof < v.size(); ++dof) {
        freeArticulation[dof] =
            v[dof] + config.dynamics.timestep * acceleration[dof];
        if (!finite(freeArticulation[dof])) {
            return fail(
                diagnostics,
                MR_STEP_NONFINITE_RESULT,
                ArticulatedRigidWorldFailure::
                    articulatedFreeDynamics
            );
        }
    }

    std::vector<MRBodyStateGPU> workingRigid(
        rigidBodies.begin(),
        rigidBodies.end()
    );
    FreeBodyIntegratorConfig rigidFreeConfig =
        config.rigidFreeMotion;
    rigidFreeConfig.timestep = config.dynamics.timestep;
    rigidFreeConfig.gravity = {
        static_cast<float>(config.dynamics.gravity[0]),
        static_cast<float>(config.dynamics.gravity[1]),
        static_cast<float>(config.dynamics.gravity[2]),
        0.0f,
    };
    diagnostics.rigidFreeDynamics = predictFreeBodyVelocities(
        rigidProperties,
        workingRigid,
        rigidWrenches,
        rigidFreeConfig
    );
    if (!diagnostics.rigidFreeDynamics.succeeded()) {
        return fail(
            diagnostics,
            diagnostics.rigidFreeDynamics.code,
            ArticulatedRigidWorldFailure::rigidFreeDynamics
        );
    }

    ArticulatedRigidWorldCache workingCache = cache;
    const std::uint64_t nextStep = cache.step + 1u;
    const std::vector<ArticulatedRigidIslandContactWarmStart>
        contactWarmStarts = activeContactWarmStarts(cache);
    ArticulatedRigidCollisionConfig collisionConfig =
        config.collision;
    collisionConfig.dynamics = config.dynamics;
    collisionConfig.contact.contact.timestep =
        config.dynamics.timestep;
    ArticulatedRigidIslandCollisionResult collision =
        collideArticulatedRigidIslandContactsCpu(
            model,
            articulationIndex,
            q,
            freeArticulation,
            rigidShapes,
            rigidMaterials,
            workingRigid,
            workingCache.manifolds,
            collisionConfig,
            contactWarmStarts
        );
    diagnostics.collision = collision.diagnostics;
    diagnostics.contactCount =
        static_cast<std::uint32_t>(collision.contacts.size());
    diagnostics.articulatedDynamicContactCount =
        collision.diagnostics.articulatedDynamicContactCount;
    diagnostics.articulatedPrescribedContactCount =
        collision.diagnostics.articulatedPrescribedContactCount;
    diagnostics.dynamicDynamicContactCount =
        collision.diagnostics.dynamicDynamicContactCount;
    diagnostics.dynamicPrescribedContactCount =
        collision.diagnostics.dynamicPrescribedContactCount;
    diagnostics.maximumPenetration =
        collision.diagnostics.maximumPenetration;
    diagnostics.matchedContactWarmStarts =
        collision.diagnostics.matchedWarmStartCount;
    if (!collision.succeeded()) {
        return fail(
            diagnostics,
            collision.diagnostics.code,
            ArticulatedRigidWorldFailure::collision
        );
    }
    if (!reduceBodyPairContacts(
            collision,
            config.maximumContactsPerBodyPair
        )) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_RESULT,
            ArticulatedRigidWorldFailure::collision
        );
    }
    collision.diagnostics.matchedWarmStartCount =
        static_cast<std::uint32_t>(std::ranges::count_if(
            collision.metadata,
            [](const ArticulatedRigidIslandContactMetadata& item) {
                return item.warmStartMatched;
            }
        ));
    diagnostics.collision = collision.diagnostics;
    diagnostics.contactCount =
        static_cast<std::uint32_t>(collision.contacts.size());
    diagnostics.articulatedDynamicContactCount =
        collision.diagnostics.articulatedDynamicContactCount;
    diagnostics.articulatedPrescribedContactCount =
        collision.diagnostics.articulatedPrescribedContactCount;
    diagnostics.dynamicDynamicContactCount =
        collision.diagnostics.dynamicDynamicContactCount;
    diagnostics.dynamicPrescribedContactCount =
        collision.diagnostics.dynamicPrescribedContactCount;
    diagnostics.matchedContactWarmStarts =
        collision.diagnostics.matchedWarmStartCount;

    ArticulatedJointLimitConfig limitConfig =
        config.jointLimits;
    limitConfig.timestep = config.dynamics.timestep;
    std::vector<ArticulatedJointLimitRow> limitRows;
    diagnostics.jointLimitCompilation =
        compileArticulatedJointLimitRows(
            model,
            articulationIndex,
            q,
            freeArticulation,
            limitRows,
            limitConfig
        );
    diagnostics.jointLimitCount =
        static_cast<std::uint32_t>(limitRows.size());
    if (!diagnostics.jointLimitCompilation.succeeded()) {
        return fail(
            diagnostics,
            codeForJointLimits(
                diagnostics.jointLimitCompilation.status
            ),
            ArticulatedRigidWorldFailure::jointLimitCompilation
        );
    }
    std::vector<double> limitWarmStarts =
        activeJointLimitWarmStarts(
            cache,
            limitRows,
            diagnostics.matchedJointLimitWarmStarts
        );

    std::vector<double> postArticulation = freeArticulation;
    std::vector<CoupledRigidBodyVelocity> postRigid(
        workingRigid.size()
    );
    for (std::size_t body = 0u; body < workingRigid.size(); ++body) {
        postRigid[body].linear = {
            workingRigid[body].linearVelocityAndInverseMass.x,
            workingRigid[body].linearVelocityAndInverseMass.y,
            workingRigid[body].linearVelocityAndInverseMass.z,
        };
        postRigid[body].angular = {
            workingRigid[body].angularVelocity.x,
            workingRigid[body].angularVelocity.y,
            workingRigid[body].angularVelocity.z,
        };
    }

    if (!collision.contacts.empty() || !limitRows.empty()) {
        diagnostics.coupledSolve =
            solveCoupledArticulatedRigidIslandCpu(
                model,
                articulationIndex,
                q,
                freeArticulation,
                workingRigid,
                collision.contacts,
                postArticulation,
                postRigid,
                config.dynamics,
                config.quality,
                limitRows,
                limitWarmStarts
            );
        if (!diagnostics.coupledSolve.succeeded()) {
            return fail(
                diagnostics,
                codeForCoupled(
                    diagnostics.coupledSolve.status,
                    diagnostics.coupledSolve.quality.code
                ),
                ArticulatedRigidWorldFailure::coupledSolve
            );
        }
    } else {
        diagnostics.coupledSolve.status =
            CoupledArticulatedRigidContactStatus::success;
    }

    diagnostics.maximumVelocityCorrection =
        maximumVelocityDifference(
            freeArticulation,
            postArticulation,
            workingRigid,
            postRigid
        );
    if (!finite(diagnostics.maximumVelocityCorrection) ||
        !publishRigidVelocities(postRigid, workingRigid)) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_RESULT,
            ArticulatedRigidWorldFailure::velocityPublication
        );
    }
    for (std::size_t contact = 0u;
         contact < diagnostics.coupledSolve.contactImpulses.size();
         contact += 3u) {
        diagnostics.maximumNormalImpulse = std::max(
            diagnostics.maximumNormalImpulse,
            diagnostics.coupledSolve.contactImpulses[contact]
        );
    }
    for (const double impulse :
         diagnostics.coupledSolve.jointLimitImpulses) {
        diagnostics.maximumJointLimitImpulse = std::max(
            diagnostics.maximumJointLimitImpulse,
            impulse
        );
    }

    std::vector<double> integratedQ(q.begin(), q.end());
    diagnostics.articulatedIntegration =
        integrateArticulatedConfiguration(
            model,
            articulationIndex,
            integratedQ,
            postArticulation,
            config.dynamics
        );
    if (!diagnostics.articulatedIntegration.succeeded()) {
        return fail(
            diagnostics,
            codeForDynamics(
                diagnostics.articulatedIntegration.status
            ),
            ArticulatedRigidWorldFailure::articulatedIntegration
        );
    }
    diagnostics.rigidIntegration =
        integrateFreeBodyConfigurations(
            rigidProperties,
            workingRigid,
            config.dynamics.timestep
        );
    if (!diagnostics.rigidIntegration.succeeded()) {
        return fail(
            diagnostics,
            diagnostics.rigidIntegration.code,
            ArticulatedRigidWorldFailure::rigidIntegration
        );
    }

    if (!updateContactCache(
            workingCache,
            collision,
            diagnostics.coupledSolve.contactImpulses,
            nextStep,
            config.contactCacheMaximumAge
        ) ||
        !updateJointLimitCache(
            workingCache,
            limitRows,
            diagnostics.coupledSolve.jointLimitImpulses,
            nextStep,
            config.jointLimitCacheMaximumAge
        ) ||
        !updateGraspEvidence(
            workingCache,
            model,
            collision,
            diagnostics.coupledSolve,
            config.grasp,
            workingRigid,
            rigidShapes,
            cache.step,
            nextStep,
            diagnostics.graspEvidence
        )) {
        return fail(
            diagnostics,
            MR_STEP_NONFINITE_RESULT,
            ArticulatedRigidWorldFailure::graspEvidence
        );
    }
    diagnostics.graspedBodyCount =
        static_cast<std::uint32_t>(std::ranges::count_if(
            diagnostics.graspEvidence,
            [](const ArticulatedRigidGraspEvidence& evidence) {
                return evidence.grasped;
            }
        ));
    workingCache.step = nextStep;

    std::ranges::copy(integratedQ, q.begin());
    std::ranges::copy(postArticulation, v.begin());
    std::ranges::copy(workingRigid, rigidBodies.begin());
    cache = std::move(workingCache);
    diagnostics.code = MR_STEP_SUCCESS;
    diagnostics.failure = ArticulatedRigidWorldFailure::none;
    return diagnostics;
}

ArticulatedRigidWorldStepDiagnostics
stepControlledArticulatedRigidWorldCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<double> q,
    const std::span<double> v,
    const std::span<const ArticulatedDofCommand> commands,
    const std::span<const ArticulatedBodyWrench> articulatedWrenches,
    const std::span<const MRBodyPropertiesGPU> rigidProperties,
    const std::span<MRBodyStateGPU> rigidBodies,
    const std::span<const MRShapeGPU> rigidShapes,
    const std::span<const MRMaterialGPU> rigidMaterials,
    const std::span<const BodyWrench> rigidWrenches,
    const ArticulatedRigidWorldConfig& config,
    ArticulatedRigidWorldCache& cache
) {
    ArticulatedRigidWorldStepDiagnostics diagnostics;
    ArticulatedActuationResult actuation;
    diagnostics.actuation = evaluateArticulatedActuation(
        model,
        articulationIndex,
        q,
        v,
        commands,
        actuation,
        config.actuation
    );
    if (!diagnostics.actuation.succeeded()) {
        return fail(
            diagnostics,
            codeForActuation(diagnostics.actuation.status),
            ArticulatedRigidWorldFailure::actuation
        );
    }
    ArticulatedRigidWorldStepDiagnostics world =
        stepArticulatedRigidWorldCpu(
            model,
            articulationIndex,
            q,
            v,
            actuation.generalizedEffort,
            articulatedWrenches,
            rigidProperties,
            rigidBodies,
            rigidShapes,
            rigidMaterials,
            rigidWrenches,
            config,
            cache
        );
    world.actuation = diagnostics.actuation;
    return world;
}

} // namespace metalrobo
