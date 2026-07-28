#include "metalrobo/Collision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using PairKey = std::pair<std::uint32_t, std::uint32_t>;

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyStateGPU makeBody(
    const float x,
    const float y,
    const float z,
    const std::uint32_t motion,
    const mr_float4 orientation = f4(0.0f, 0.0f, 0.0f, 1.0f)
) {
    MRBodyStateGPU result{};
    result.position = f4(x, y, z, 1.0f);
    result.orientation = orientation;
    result.flagsAndIndices[0] = motion;
    return result;
}

MRShapeGPU makeShape(
    const std::uint32_t body,
    const std::uint32_t type,
    const mr_float4 dimensions,
    const std::uint32_t generation,
    const float contactOffset = 0.02f
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = type;
    result.collisionGroup = 1u;
    result.collisionMask = 1u;
    result.slotGeneration = generation;
    result.localPosition = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.dimensions = dimensions;
    result.contactRestAndBoundingRadius =
        f4(contactOffset, 0.0f, 0.0f, 0.0f);
    return result;
}

metalrobo::CollisionConfig ampleConfig() {
    metalrobo::CollisionConfig result;
    result.environment = 7u;
    result.capacities = {
        .pairCapacity = 64u,
        .rawContactCapacity = 128u,
        .manifoldCapacity = 64u,
    };
    return result;
}

struct PrimaryScene {
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
};

struct CapsulePairScene {
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
};

using SphereBoxScene = CapsulePairScene;

PrimaryScene makePrimaryScene() {
    constexpr float sineHalfTurn = 0.7071067811865475f;
    PrimaryScene result;
    result.bodies = {
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(-2.0f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(-1.25f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(
            1.0f,
            0.30f,
            0.0f,
            MR_MOTION_DYNAMIC,
            f4(0.0f, 0.0f, sineHalfTurn, sineHalfTurn)
        ),
        makeBody(3.0f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
        makeBody(5.0f, 0.45f, 0.0f, MR_MOTION_DYNAMIC),
    };
    result.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, f4(0.0f, 0.0f, 0.0f), 100u),
        makeShape(1u, MR_SHAPE_SPHERE, f4(0.5f, 0.0f, 0.0f), 101u),
        makeShape(2u, MR_SHAPE_SPHERE, f4(0.5f, 0.0f, 0.0f), 102u),
        makeShape(3u, MR_SHAPE_CAPSULE, f4(0.35f, 0.55f, 0.0f), 103u),
        makeShape(4u, MR_SHAPE_BOX, f4(0.5f, 0.5f, 0.5f), 104u),
        makeShape(
            5u,
            MR_SHAPE_CYLINDER,
            f4(0.30f, 0.50f, 0.0f),
            105u
        ),
    };
    return result;
}

CapsulePairScene makeCapsulePairScene() {
    constexpr float quarterTurn = 0.7071067811865475f;
    constexpr float boundaryOffset =
        0.3676884472372877f;
    CapsulePairScene result;

    auto appendPair = [&](
        const MRBodyStateGPU bodyA,
        MRShapeGPU shapeA,
        const MRBodyStateGPU bodyB,
        MRShapeGPU shapeB,
        const std::uint32_t group
    ) {
        const std::uint32_t bodyAIndex =
            static_cast<std::uint32_t>(result.bodies.size());
        const std::uint32_t bodyBIndex = bodyAIndex + 1u;
        shapeA.bodyIndex = bodyAIndex;
        shapeB.bodyIndex = bodyBIndex;
        shapeA.collisionGroup = group;
        shapeA.collisionMask = group;
        shapeB.collisionGroup = group;
        shapeB.collisionMask = group;
        shapeA.slotGeneration = 500u + bodyAIndex;
        shapeB.slotGeneration = 500u + bodyBIndex;
        result.bodies.push_back(bodyA);
        result.bodies.push_back(bodyB);
        result.shapes.push_back(shapeA);
        result.shapes.push_back(shapeB);
    };

    const auto sphere = [](const float contactOffset = 0.0f) {
        return makeShape(
            0u,
            MR_SHAPE_SPHERE,
            f4(0.40f, 0.0f, 0.0f),
            0u,
            contactOffset
        );
    };
    const auto capsule = [](const float contactOffset = 0.0f) {
        return makeShape(
            0u,
            MR_SHAPE_CAPSULE,
            f4(0.25f, 1.0f, 0.0f),
            0u,
            contactOffset
        );
    };

    // Sphere/capsule segment feature, in both canonical input orders.
    appendPair(
        makeBody(0.649f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 0u
    );
    appendPair(
        makeBody(5.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(5.649f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 1u
    );

    // Sphere/capsule positive endpoint feature, in both input orders.
    appendPair(
        makeBody(10.649f, 1.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(10.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 2u
    );
    appendPair(
        makeBody(15.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(15.649f, 1.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 3u
    );

    // Skew segments with an exact radius-sum separation.
    appendPair(
        makeBody(20.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(
            20.0f,
            0.0f,
            0.5f,
            MR_MOTION_DYNAMIC,
            f4(0.0f, 0.0f, quarterTurn, quarterTurn)
        ),
        capsule(),
        1u << 4u
    );

    // Parallel overlapping spans force an interior/endpoint closest pair.
    appendPair(
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(25.4f, 0.5f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 5u
    );

    // Coincident axes exercise the deterministic perpendicular fallback.
    appendPair(
        makeBody(30.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        makeBody(30.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(),
        1u << 6u
    );

    // Just inside the summed contact offset, then just outside it while the
    // expanded AABBs still overlap on every axis.
    appendPair(
        makeBody(35.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        makeBody(
            35.0f + boundaryOffset,
            0.0f,
            boundaryOffset,
            MR_MOTION_DYNAMIC
        ),
        capsule(0.01f),
        1u << 7u
    );
    appendPair(
        makeBody(40.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        makeBody(40.371f, 0.0f, 0.371f, MR_MOTION_DYNAMIC),
        capsule(0.01f),
        1u << 8u
    );
    return result;
}

SphereBoxScene makeSphereBoxScene() {
    constexpr float sineEighthTurn = 0.3826834323650898f;
    constexpr float cosineEighthTurn = 0.9238795325112867f;
    SphereBoxScene result;

    auto appendPair = [&](
        const MRBodyStateGPU bodyA,
        MRShapeGPU shapeA,
        const MRBodyStateGPU bodyB,
        MRShapeGPU shapeB,
        const std::uint32_t group
    ) {
        const std::uint32_t bodyAIndex =
            static_cast<std::uint32_t>(result.bodies.size());
        shapeA.bodyIndex = bodyAIndex;
        shapeB.bodyIndex = bodyAIndex + 1u;
        shapeA.collisionGroup = group;
        shapeA.collisionMask = group;
        shapeB.collisionGroup = group;
        shapeB.collisionMask = group;
        shapeA.slotGeneration = 700u + bodyAIndex;
        shapeB.slotGeneration = 701u + bodyAIndex;
        result.bodies.push_back(bodyA);
        result.bodies.push_back(bodyB);
        result.shapes.push_back(shapeA);
        result.shapes.push_back(shapeB);
    };
    const auto sphere = [] {
        return makeShape(
            0u,
            MR_SHAPE_SPHERE,
            f4(0.30f, 0.0f, 0.0f),
            0u,
            0.0f
        );
    };
    const auto box = [](
        const mr_float4 halfExtents =
            f4(0.50f, 0.50f, 0.50f)
    ) {
        return makeShape(
            0u,
            MR_SHAPE_BOX,
            halfExtents,
            0u,
            0.0f
        );
    };

    appendPair(
        makeBody(0.799f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 0u
    );
    appendPair(
        makeBody(5.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        makeBody(5.799f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        1u << 1u
    );
    appendPair(
        makeBody(10.70f, 0.70f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(10.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 2u
    );
    appendPair(
        makeBody(15.67f, 0.67f, 0.67f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(15.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(),
        1u << 3u
    );
    appendPair(
        makeBody(
            20.5649783103f,
            0.5649783103f,
            0.0f,
            MR_MOTION_DYNAMIC
        ),
        sphere(),
        makeBody(
            20.0f,
            0.0f,
            0.0f,
            MR_MOTION_DYNAMIC,
            f4(
                0.0f,
                0.0f,
                sineEighthTurn,
                cosineEighthTurn
            )
        ),
        box(),
        1u << 4u
    );
    appendPair(
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        sphere(),
        makeBody(25.0f, 0.0f, 0.0f, MR_MOTION_DYNAMIC),
        box(f4(0.50f, 0.60f, 0.70f)),
        1u << 5u
    );
    return result;
}

bool sameFloat4(const mr_float4 left, const mr_float4 right) {
    return
        left.x == right.x &&
        left.y == right.y &&
        left.z == right.z &&
        left.w == right.w;
}

bool sameRawContact(
    const MRRawContactGPU& left,
    const MRRawContactGPU& right
) {
    if (!sameFloat4(
            left.normalAndSeparation,
            right.normalAndSeparation
        ) ||
        !sameFloat4(left.pointAWorld, right.pointAWorld) ||
        !sameFloat4(left.pointBWorld, right.pointBWorld)) {
        return false;
    }
    for (std::size_t index = 0u; index < 4u; ++index) {
        if (left.featureAndFlags[index] !=
            right.featureAndFlags[index]) {
            return false;
        }
    }
    return true;
}

bool sameHeader(
    const MRManifoldHeaderGPU& left,
    const MRManifoldHeaderGPU& right
) {
    for (std::size_t index = 0; index < 4u; ++index) {
        if (left.pairAndCount[index] != right.pairAndCount[index] ||
            left.generationsAndFlags[index] !=
                right.generationsAndFlags[index]) {
            return false;
        }
    }
    return
        sameFloat4(left.normalAndAge, right.normalAndAge) &&
        sameFloat4(left.tangentAndMetric, right.tangentAndMetric);
}

bool samePoint(
    const MRManifoldPointGPU& left,
    const MRManifoldPointGPU& right
) {
    if (!sameFloat4(left.localAnchorA, right.localAnchorA) ||
        !sameFloat4(left.localAnchorB, right.localAnchorB) ||
        !sameFloat4(left.impulses, right.impulses)) {
        return false;
    }
    for (std::size_t index = 0; index < 4u; ++index) {
        if (left.featureAndLife[index] !=
            right.featureAndLife[index]) {
            return false;
        }
    }
    return true;
}

bool sameCache(
    const std::span<const metalrobo::PersistentManifold> left,
    const std::span<const metalrobo::PersistentManifold> right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t manifold = 0;
         manifold < left.size();
         ++manifold) {
        if (!sameHeader(
                left[manifold].header,
                right[manifold].header
            )) {
            return false;
        }
        for (std::size_t point = 0; point < 4u; ++point) {
            if (!samePoint(
                    left[manifold].points[point],
                    right[manifold].points[point]
                )) {
                return false;
            }
        }
    }
    return true;
}

bool emptyPayload(const metalrobo::CollisionFrame& frame) {
    return
        frame.worldAabbs.empty() &&
        frame.pairs.empty() &&
        frame.rawContacts.empty() &&
        frame.rawContactPairIndices.empty() &&
        frame.manifoldHeaders.empty() &&
        frame.manifoldPoints.empty();
}

PairKey keyOf(const MRCandidatePairGPU& pair) {
    return {pair.colliderA, pair.colliderB};
}

std::set<PairKey> pairSet(const metalrobo::CollisionFrame& frame) {
    std::set<PairKey> result;
    for (const MRCandidatePairGPU& pair : frame.pairs) {
        require(
            pair.colliderA < pair.colliderB,
            "candidate pair was not canonical"
        );
        require(
            result.emplace(keyOf(pair)).second,
            "duplicate candidate pair"
        );
    }
    return result;
}

bool containsPair(
    const metalrobo::CollisionFrame& frame,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    return pairSet(frame).contains({
        std::min(colliderA, colliderB),
        std::max(colliderA, colliderB),
    });
}

std::vector<MRRawContactGPU> contactsForPair(
    const metalrobo::CollisionFrame& frame,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    std::vector<MRRawContactGPU> result;
    for (std::size_t contactIndex = 0u;
         contactIndex < frame.rawContacts.size();
         ++contactIndex) {
        const std::uint32_t pairIndex =
            frame.rawContactPairIndices[contactIndex];
        require(
            pairIndex < frame.pairs.size(),
            "contact references an invalid candidate pair"
        );
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        if (pair.colliderA == colliderA &&
            pair.colliderB == colliderB) {
            result.push_back(frame.rawContacts[contactIndex]);
        }
    }
    return result;
}

std::uint32_t expectedFeature(
    const std::uint32_t shapeType,
    const std::uint32_t localFeature
) {
    return ((shapeType & 0x0fu) << 28u) |
        (localFeature & 0x0fffffffu);
}

void verifyWitnesses(const metalrobo::CollisionFrame& frame) {
    require(
        frame.rawContacts.size() ==
            frame.rawContactPairIndices.size(),
        "contact-to-pair map size mismatch"
    );
    for (std::size_t index = 0;
         index < frame.rawContacts.size();
         ++index) {
        const MRRawContactGPU& contact = frame.rawContacts[index];
        const double nx = contact.normalAndSeparation.x;
        const double ny = contact.normalAndSeparation.y;
        const double nz = contact.normalAndSeparation.z;
        const double normalLength =
            std::sqrt(nx * nx + ny * ny + nz * nz);
        const double dx =
            contact.pointBWorld.x - contact.pointAWorld.x;
        const double dy =
            contact.pointBWorld.y - contact.pointAWorld.y;
        const double dz =
            contact.pointBWorld.z - contact.pointAWorld.z;
        const double projected = dx * nx + dy * ny + dz * nz;
        require(
            std::isfinite(normalLength) &&
                std::abs(normalLength - 1.0) <= 2.0e-6,
            "raw contact normal is not finite and unit length"
        );
        require(
            std::abs(
                projected - contact.normalAndSeparation.w
            ) <= 3.0e-6,
            "raw contact witness violates separation convention"
        );
        require(
            frame.rawContactPairIndices[index] < frame.pairs.size(),
            "raw contact references an invalid pair"
        );
    }
}

void verifyPrimaryTopology(
    const metalrobo::CollisionFrame& frame
) {
    require(frame.succeeded(), "primary collision frame failed");
    require(frame.worldAabbs.size() == 6u, "missing world AABBs");
    require(frame.pairs.size() == 6u, "unexpected primary pair count");
    require(
        frame.rawContacts.size() == 13u,
        "unexpected primary raw-contact count"
    );
    require(
        frame.manifoldHeaders.size() == 6u &&
            frame.manifoldPoints.size() == 24u,
        "unexpected primary manifold layout"
    );
    require(
        frame.diagnostics.requiredPairs == 6u &&
            frame.diagnostics.requiredRawContacts == 13u &&
            frame.diagnostics.requiredManifolds == 6u,
        "primary preflight requirements are wrong"
    );

    const std::set<PairKey> expected{
        {0u, 1u},
        {0u, 2u},
        {0u, 3u},
        {0u, 4u},
        {0u, 5u},
        {1u, 2u},
    };
    require(pairSet(frame) == expected, "primary pair identities changed");

    std::array<bool, 6> pairClassSeen{};
    std::array<std::uint32_t, 6> contactsPerPair{};
    for (std::size_t pairIndex = 0;
         pairIndex < frame.pairs.size();
         ++pairIndex) {
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        require(
            pair.environment == 7u &&
                pair.flags >= metalrobo::collisionPairSphereSphere &&
                pair.flags <=
                    metalrobo::collisionPairCylinderPlane,
            "pair metadata is invalid"
        );
        pairClassSeen[pair.flags] = true;
    }
    for (const std::uint32_t pairIndex :
         frame.rawContactPairIndices) {
        ++contactsPerPair[pairIndex];
    }
    require(
        pairClassSeen[metalrobo::collisionPairSphereSphere] &&
            pairClassSeen[metalrobo::collisionPairSpherePlane] &&
            pairClassSeen[metalrobo::collisionPairCapsulePlane] &&
            pairClassSeen[metalrobo::collisionPairBoxPlane] &&
            pairClassSeen[metalrobo::collisionPairCylinderPlane],
        "not every analytic pair class was exercised"
    );

    for (std::size_t pairIndex = 0;
         pairIndex < frame.pairs.size();
         ++pairIndex) {
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        std::uint32_t expectedCount = 1u;
        if (pair.flags == metalrobo::collisionPairCapsulePlane) {
            expectedCount = 2u;
        } else if (
            pair.flags == metalrobo::collisionPairBoxPlane ||
            pair.flags == metalrobo::collisionPairCylinderPlane
        ) {
            expectedCount = 4u;
        }
        require(
            contactsPerPair[pairIndex] == expectedCount,
            "analytic contact cardinality changed"
        );
        require(
            frame.manifoldHeaders[pairIndex].pairAndCount[3] ==
                expectedCount,
            "manifold point cardinality changed"
        );
    }
    verifyWitnesses(frame);
}

void verifyStableTopology(
    const metalrobo::CollisionFrame& first,
    const metalrobo::CollisionFrame& second
) {
    require(
        first.pairs.size() == second.pairs.size() &&
            first.rawContacts.size() == second.rawContacts.size() &&
            first.manifoldHeaders.size() ==
                second.manifoldHeaders.size(),
        "stable replay changed collision topology"
    );
    for (std::size_t index = 0; index < first.pairs.size(); ++index) {
        require(
            first.pairs[index].environment ==
                    second.pairs[index].environment &&
                first.pairs[index].colliderA ==
                    second.pairs[index].colliderA &&
                first.pairs[index].colliderB ==
                    second.pairs[index].colliderB &&
                first.pairs[index].flags ==
                    second.pairs[index].flags,
            "candidate pair ID changed across replay"
        );
    }
    for (std::size_t index = 0;
         index < first.rawContacts.size();
         ++index) {
        require(
            first.rawContactPairIndices[index] ==
                    second.rawContactPairIndices[index] &&
                first.rawContacts[index].featureAndFlags[0] ==
                    second.rawContacts[index].featureAndFlags[0] &&
                first.rawContacts[index].featureAndFlags[1] ==
                    second.rawContacts[index].featureAndFlags[1],
            "raw-contact feature ID changed across replay"
        );
    }
    for (std::size_t manifold = 0;
         manifold < first.manifoldHeaders.size();
         ++manifold) {
        const MRManifoldHeaderGPU& firstHeader =
            first.manifoldHeaders[manifold];
        const MRManifoldHeaderGPU& secondHeader =
            second.manifoldHeaders[manifold];
        require(
            firstHeader.pairAndCount[0] ==
                    secondHeader.pairAndCount[0] &&
                firstHeader.pairAndCount[1] ==
                    secondHeader.pairAndCount[1] &&
                firstHeader.pairAndCount[2] ==
                    secondHeader.pairAndCount[2] &&
                firstHeader.pairAndCount[3] ==
                    secondHeader.pairAndCount[3] &&
                firstHeader.generationsAndFlags[0] ==
                    secondHeader.generationsAndFlags[0] &&
                firstHeader.generationsAndFlags[1] ==
                    secondHeader.generationsAndFlags[1] &&
                secondHeader.normalAndAge.w ==
                    firstHeader.normalAndAge.w + 1.0f,
            "persistent manifold identity or age changed"
        );
        for (std::size_t point = 0;
             point < firstHeader.pairAndCount[3];
             ++point) {
            const std::size_t offset = manifold * 4u + point;
            const MRManifoldPointGPU& firstPoint =
                first.manifoldPoints[offset];
            const MRManifoldPointGPU& secondPoint =
                second.manifoldPoints[offset];
            require(
                firstPoint.featureAndLife[0] ==
                        secondPoint.featureAndLife[0] &&
                    firstPoint.featureAndLife[1] ==
                        secondPoint.featureAndLife[1] &&
                    secondPoint.featureAndLife[2] ==
                        firstPoint.featureAndLife[2] + 1u,
                "persistent feature lifetime did not advance"
            );
        }
    }
}

void verifyFilters(
    const PrimaryScene& scene,
    const metalrobo::CollisionConfig& config
) {
    metalrobo::PersistentManifoldCache exclusionCache;
    const std::array exclusions{
        metalrobo::CollisionPairExclusion{
            .colliderA = 2u,
            .colliderB = 1u,
        },
    };
    const auto excluded = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        exclusionCache,
        exclusions
    );
    require(excluded.succeeded(), "excluded collision frame failed");
    require(
        !containsPair(excluded, 1u, 2u) &&
            excluded.pairs.size() == 5u,
        "reversed exclusion was not canonicalized"
    );

    std::vector<MRShapeGPU> maskedShapes = scene.shapes;
    maskedShapes[2].collisionMask = 0u;
    metalrobo::PersistentManifoldCache maskCache;
    const auto masked = metalrobo::collideCpuReference(
        maskedShapes,
        scene.bodies,
        config,
        maskCache
    );
    require(masked.succeeded(), "masked collision frame failed");
    for (const MRCandidatePairGPU& pair : masked.pairs) {
        require(
            pair.colliderA != 2u && pair.colliderB != 2u,
            "symmetric group/mask filtering failed"
        );
    }

    std::vector<MRBodyStateGPU> staticBodies = scene.bodies;
    staticBodies[1].flagsAndIndices[0] = MR_MOTION_STATIC;
    staticBodies[2].flagsAndIndices[0] = MR_MOTION_STATIC;
    metalrobo::PersistentManifoldCache staticCache;
    const auto staticFiltered = metalrobo::collideCpuReference(
        scene.shapes,
        staticBodies,
        config,
        staticCache
    );
    require(staticFiltered.succeeded(), "static filter frame failed");
    for (const MRCandidatePairGPU& pair : staticFiltered.pairs) {
        require(
            pair.colliderA != 1u && pair.colliderB != 1u &&
                pair.colliderA != 2u && pair.colliderB != 2u,
            "static-static pair escaped canonical filtering"
        );
    }
}

void verifyFourPointReduction() {
    const std::vector bodies{
        makeBody(0.0f, 0.0f, 0.0f, MR_MOTION_STATIC),
        makeBody(0.0f, -1.0f, 0.0f, MR_MOTION_DYNAMIC),
    };
    const std::vector shapes{
        makeShape(
            0u,
            MR_SHAPE_PLANE,
            f4(0.0f, 0.0f, 0.0f),
            200u
        ),
        makeShape(
            1u,
            MR_SHAPE_BOX,
            f4(0.5f, 0.5f, 0.5f),
            201u
        ),
    };
    metalrobo::CollisionConfig config = ampleConfig();
    config.environment = 9u;
    metalrobo::PersistentManifoldCache cache;
    const auto first = metalrobo::collideCpuReference(
        shapes,
        bodies,
        config,
        cache
    );
    require(
        first.succeeded() &&
            first.pairs.size() == 1u &&
            first.rawContacts.size() == 8u &&
            first.manifoldHeaders.size() == 1u &&
            first.manifoldHeaders[0].pairAndCount[3] == 4u,
        "eight-point box patch was not reduced to four"
    );
    const auto second = metalrobo::collideCpuReference(
        shapes,
        bodies,
        config,
        cache
    );
    require(
        second.succeeded() &&
            second.manifoldHeaders[0].pairAndCount[3] == 4u &&
            second.diagnostics.refreshedPoints == 4u,
        "reduced box patch did not persist"
    );
    verifyStableTopology(first, second);
    verifyWitnesses(first);
}

void verifyCapsulePairPrimitives() {
    const CapsulePairScene scene = makeCapsulePairScene();
    metalrobo::CollisionConfig config = ampleConfig();
    config.environment = 13u;
    config.capacities = {
        .pairCapacity = 16u,
        .rawContactCapacity = 16u,
        .manifoldCapacity = 16u,
    };
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame first =
        metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            config,
            cache
        );
    require(
        first.succeeded() &&
            first.pairs.size() == 9u &&
            first.rawContacts.size() == 8u &&
            first.manifoldHeaders.size() == 8u,
        "capsule pair adversarial topology is wrong"
    );
    verifyWitnesses(first);

    for (std::uint32_t pair = 0u; pair < 4u; ++pair) {
        const std::uint32_t colliderA = 2u * pair;
        const std::uint32_t colliderB = colliderA + 1u;
        require(
            containsPair(first, colliderA, colliderB),
            "sphere/capsule candidate pair is missing"
        );
        const auto contacts =
            contactsForPair(first, colliderA, colliderB);
        require(
            contacts.size() == 1u,
            "sphere/capsule did not emit exactly one contact"
        );
        const bool sphereFirst = pair % 2u == 0u;
        const std::uint32_t capsuleLocalFeature =
            pair < 2u ? 2u : 1u;
        require(
            contacts[0].featureAndFlags[0] ==
                expectedFeature(
                    sphereFirst
                        ? MR_SHAPE_SPHERE
                        : MR_SHAPE_CAPSULE,
                    sphereFirst ? 0u : capsuleLocalFeature
                ) &&
                contacts[0].featureAndFlags[1] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_CAPSULE
                            : MR_SHAPE_SPHERE,
                        sphereFirst ? capsuleLocalFeature : 0u
                    ),
            "sphere/capsule endpoint or segment feature is wrong"
        );
    }

    const auto sphereSegment =
        contactsForPair(first, 0u, 1u).front();
    const auto capsuleSegment =
        contactsForPair(first, 2u, 3u).front();
    const auto sphereEndpoint =
        contactsForPair(first, 4u, 5u).front();
    const auto capsuleEndpoint =
        contactsForPair(first, 6u, 7u).front();
    require(
        std::abs(
            sphereSegment.normalAndSeparation.x +
                capsuleSegment.normalAndSeparation.x
        ) <= 2.0e-6f &&
            std::abs(
                sphereSegment.normalAndSeparation.w -
                    capsuleSegment.normalAndSeparation.w
            ) <= 2.0e-6f &&
            std::abs(
                sphereEndpoint.normalAndSeparation.x +
                    capsuleEndpoint.normalAndSeparation.x
            ) <= 2.0e-6f &&
            std::abs(
                sphereEndpoint.normalAndSeparation.w -
                    capsuleEndpoint.normalAndSeparation.w
            ) <= 2.0e-6f,
        "sphere/capsule input-order reciprocity failed"
    );

    constexpr std::array<std::uint32_t, 5> featureA{
        2u, 2u, 0u, 0u, 0u,
    };
    constexpr std::array<std::uint32_t, 5> featureB{
        2u, 0u, 0u, 0u, 0u,
    };
    for (std::uint32_t localPair = 0u;
         localPair < featureA.size();
         ++localPair) {
        const std::uint32_t colliderA = 8u + 2u * localPair;
        const std::uint32_t colliderB = colliderA + 1u;
        require(
            containsPair(first, colliderA, colliderB),
            "capsule/capsule broadphase pair is missing"
        );
        const auto contacts =
            contactsForPair(first, colliderA, colliderB);
        const bool outsideOffset = localPair == 4u;
        require(
            contacts.size() == (outsideOffset ? 0u : 1u),
            "capsule/capsule contact-offset boundary is wrong"
        );
        if (!outsideOffset) {
            require(
                contacts[0].featureAndFlags[0] ==
                        expectedFeature(
                            MR_SHAPE_CAPSULE,
                            featureA[localPair]
                        ) &&
                    contacts[0].featureAndFlags[1] ==
                        expectedFeature(
                            MR_SHAPE_CAPSULE,
                            featureB[localPair]
                        ),
                "capsule/capsule closest-feature ID is wrong"
            );
        }
    }
    const auto skew = contactsForPair(first, 8u, 9u).front();
    const auto parallel =
        contactsForPair(first, 10u, 11u).front();
    const auto coincident =
        contactsForPair(first, 12u, 13u).front();
    const auto boundary =
        contactsForPair(first, 14u, 15u).front();
    require(
        std::abs(skew.normalAndSeparation.w) <= 2.0e-6f &&
            std::abs(
                parallel.normalAndSeparation.w + 0.10f
            ) <= 2.0e-6f &&
            std::abs(
                coincident.normalAndSeparation.w + 0.50f
            ) <= 2.0e-6f &&
            std::abs(
                boundary.normalAndSeparation.w - 0.01999f
            ) <= 2.0e-5f,
        "capsule/capsule skew, parallel, coincident, or boundary separation is wrong"
    );

    const metalrobo::CollisionFrame replay =
        metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            config,
            cache
        );
    require(
        replay.succeeded() &&
            replay.pairs.size() == first.pairs.size() &&
            replay.rawContacts.size() ==
                first.rawContacts.size(),
        "capsule pair deterministic replay topology changed"
    );
    for (std::size_t index = 0u;
         index < first.rawContacts.size();
         ++index) {
        require(
            first.rawContactPairIndices[index] ==
                    replay.rawContactPairIndices[index] &&
                sameRawContact(
                    first.rawContacts[index],
                    replay.rawContacts[index]
                ),
            "capsule pair deterministic replay changed witnesses"
        );
    }

    const std::vector<metalrobo::PersistentManifold> snapshot(
        cache.entries().begin(),
        cache.entries().end()
    );
    const auto expectCapacityFailure =
        [&](metalrobo::CollisionConfig failedConfig,
            const std::uint32_t expectedCode) {
            const auto failed = metalrobo::collideCpuReference(
                scene.shapes,
                scene.bodies,
                failedConfig,
                cache
            );
            const bool exactRequirements =
                failed.diagnostics.requiredPairs == 9u &&
                (
                    expectedCode ==
                        MR_STEP_PAIR_CAPACITY_OVERFLOW ||
                    (
                        failed.diagnostics
                                .requiredRawContacts == 8u &&
                        failed.diagnostics
                                .requiredManifolds == 8u
                    )
                );
            require(
                failed.diagnostics.code == expectedCode &&
                    exactRequirements &&
                    emptyPayload(failed) &&
                    sameCache(snapshot, cache.entries()),
                "capsule pair capacity failure was not exact and transactional"
            );
        };
    metalrobo::CollisionConfig pairOverflow = config;
    pairOverflow.capacities.pairCapacity = 8u;
    expectCapacityFailure(
        pairOverflow,
        MR_STEP_PAIR_CAPACITY_OVERFLOW
    );
    metalrobo::CollisionConfig contactOverflow = config;
    contactOverflow.capacities.rawContactCapacity = 7u;
    expectCapacityFailure(
        contactOverflow,
        MR_STEP_CONTACT_CAPACITY_OVERFLOW
    );
    metalrobo::CollisionConfig manifoldOverflow = config;
    manifoldOverflow.capacities.manifoldCapacity = 7u;
    expectCapacityFailure(
        manifoldOverflow,
        MR_STEP_MANIFOLD_CAPACITY_OVERFLOW
    );
}

void verifySphereBoxPrimitive() {
    const SphereBoxScene scene = makeSphereBoxScene();
    metalrobo::CollisionConfig config = ampleConfig();
    config.environment = 14u;
    config.capacities = {
        .pairCapacity = 8u,
        .rawContactCapacity = 8u,
        .manifoldCapacity = 8u,
    };
    metalrobo::PersistentManifoldCache cache;
    const auto first = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        cache
    );
    require(
        first.succeeded() &&
            first.pairs.size() == 6u &&
            first.rawContacts.size() == 6u &&
            first.manifoldHeaders.size() == 6u,
        "sphere/box adversarial topology is wrong"
    );
    verifyWitnesses(first);

    constexpr std::array<std::uint32_t, 6> boxFeatures{
        14u, 14u, 17u, 26u, 14u, 28u,
    };
    for (std::size_t localPair = 0u;
         localPair < boxFeatures.size();
         ++localPair) {
        const std::uint32_t colliderA =
            static_cast<std::uint32_t>(2u * localPair);
        const std::uint32_t colliderB = colliderA + 1u;
        const auto contacts =
            contactsForPair(first, colliderA, colliderB);
        require(
            containsPair(first, colliderA, colliderB) &&
                contacts.size() == 1u,
            "sphere/box pair did not emit exactly one contact"
        );
        const bool sphereFirst = localPair != 1u;
        require(
            contacts[0].featureAndFlags[0] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_SPHERE
                            : MR_SHAPE_BOX,
                        sphereFirst ? 0u : boxFeatures[localPair]
                    ) &&
                contacts[0].featureAndFlags[1] ==
                    expectedFeature(
                        sphereFirst
                            ? MR_SHAPE_BOX
                            : MR_SHAPE_SPHERE,
                        sphereFirst ? boxFeatures[localPair] : 0u
                    ),
            "sphere/box face, edge, corner, or interior feature is wrong"
        );
        require(
            first.pairs[localPair].flags ==
                metalrobo::collisionPairSphereBox,
            "sphere/box candidate pair class is wrong"
        );
    }

    const auto sphereFirst =
        contactsForPair(first, 0u, 1u).front();
    const auto boxFirst =
        contactsForPair(first, 2u, 3u).front();
    const auto rotated =
        contactsForPair(first, 8u, 9u).front();
    const auto inside =
        contactsForPair(first, 10u, 11u).front();
    require(
        std::abs(
            sphereFirst.normalAndSeparation.x +
                boxFirst.normalAndSeparation.x
        ) <= 2.0e-6f &&
            std::abs(
                sphereFirst.normalAndSeparation.w -
                    boxFirst.normalAndSeparation.w
            ) <= 2.0e-6f,
        "sphere/box swapped-order reciprocity failed"
    );
    require(
        std::abs(
            rotated.normalAndSeparation.x +
                0.7071067811865475f
        ) <= 3.0e-6f &&
            std::abs(
                rotated.normalAndSeparation.y +
                    0.7071067811865475f
            ) <= 3.0e-6f &&
            std::abs(
                rotated.normalAndSeparation.w + 0.001f
            ) <= 3.0e-6f,
        "rotated sphere/box face witness is wrong"
    );
    require(
        std::abs(inside.normalAndSeparation.x + 1.0f) <=
                2.0e-6f &&
            std::abs(
                inside.normalAndSeparation.w + 0.80f
            ) <= 2.0e-6f,
        "sphere-inside-box signed witness is wrong"
    );

    const auto replay = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        config,
        cache
    );
    require(
        replay.succeeded() &&
            replay.rawContacts.size() ==
                first.rawContacts.size(),
        "sphere/box replay topology changed"
    );
    for (std::size_t index = 0u;
         index < first.rawContacts.size();
         ++index) {
        require(
            first.rawContactPairIndices[index] ==
                    replay.rawContactPairIndices[index] &&
                sameRawContact(
                    first.rawContacts[index],
                    replay.rawContacts[index]
                ),
            "sphere/box deterministic replay changed witnesses"
        );
    }

    const std::vector<metalrobo::PersistentManifold> snapshot(
        cache.entries().begin(),
        cache.entries().end()
    );
    const auto expectCapacityFailure =
        [&](metalrobo::CollisionConfig failedConfig,
            const std::uint32_t expectedCode) {
            const auto failed = metalrobo::collideCpuReference(
                scene.shapes,
                scene.bodies,
                failedConfig,
                cache
            );
            const bool exactRequirements =
                failed.diagnostics.requiredPairs == 6u &&
                (
                    expectedCode ==
                        MR_STEP_PAIR_CAPACITY_OVERFLOW ||
                    (
                        failed.diagnostics
                                .requiredRawContacts == 6u &&
                        failed.diagnostics
                                .requiredManifolds == 6u
                    )
                );
            require(
                failed.diagnostics.code == expectedCode &&
                    exactRequirements &&
                    emptyPayload(failed) &&
                    sameCache(snapshot, cache.entries()),
                "sphere/box capacity failure was not exact and transactional"
            );
        };
    metalrobo::CollisionConfig pairOverflow = config;
    pairOverflow.capacities.pairCapacity = 5u;
    expectCapacityFailure(
        pairOverflow,
        MR_STEP_PAIR_CAPACITY_OVERFLOW
    );
    metalrobo::CollisionConfig contactOverflow = config;
    contactOverflow.capacities.rawContactCapacity = 5u;
    expectCapacityFailure(
        contactOverflow,
        MR_STEP_CONTACT_CAPACITY_OVERFLOW
    );
    metalrobo::CollisionConfig manifoldOverflow = config;
    manifoldOverflow.capacities.manifoldCapacity = 5u;
    expectCapacityFailure(
        manifoldOverflow,
        MR_STEP_MANIFOLD_CAPACITY_OVERFLOW
    );
}

bool aabbOverlap(const MRAabbGPU& left, const MRAabbGPU& right) {
    return
        left.lower.x <= right.upper.x &&
        left.upper.x >= right.lower.x &&
        left.lower.y <= right.upper.y &&
        left.upper.y >= right.lower.y &&
        left.lower.z <= right.upper.z &&
        left.upper.z >= right.lower.z;
}

float randomRange(
    std::uint32_t& state,
    const float minimum,
    const float maximum
) {
    state = state * 1664525u + 1013904223u;
    const double unit =
        static_cast<double>(state) /
        static_cast<double>(0xffffffffu);
    return static_cast<float>(
        static_cast<double>(minimum) +
        (static_cast<double>(maximum) - minimum) * unit
    );
}

std::size_t verifySweepAndPruneCorpus() {
    constexpr std::uint32_t count = 96u;
    std::vector<MRBodyStateGPU> bodies;
    std::vector<MRShapeGPU> shapes;
    bodies.reserve(count);
    shapes.reserve(count);
    std::uint32_t randomState = 0x6d2b79f5u;
    for (std::uint32_t index = 0u; index < count; ++index) {
        float x = randomRange(randomState, -5.0f, 5.0f);
        float y = randomRange(randomState, -5.0f, 5.0f);
        float z = randomRange(randomState, -5.0f, 5.0f);
        float radius = randomRange(randomState, 0.15f, 0.75f);
        float contactOffset = randomRange(
            randomState,
            0.0f,
            0.05f
        );
        if (index == 0u) {
            x = -8.0f;
            y = 0.0f;
            z = 0.0f;
            radius = 0.5f;
            contactOffset = 0.0f;
        } else if (index == 1u) {
            x = -7.0f;
            y = 0.0f;
            z = 0.0f;
            radius = 0.5f;
            contactOffset = 0.0f;
        }
        bodies.push_back(
            makeBody(x, y, z, MR_MOTION_DYNAMIC)
        );
        shapes.push_back(
            makeShape(
                index,
                MR_SHAPE_SPHERE,
                f4(radius, 0.0f, 0.0f),
                1000u + index,
                contactOffset
            )
        );
    }

    metalrobo::CollisionConfig config;
    config.environment = 11u;
    const std::uint32_t maximumPairs =
        count * (count - 1u) / 2u;
    config.capacities = {
        .pairCapacity = maximumPairs,
        .rawContactCapacity = maximumPairs,
        .manifoldCapacity = maximumPairs,
    };
    metalrobo::PersistentManifoldCache cache;
    const auto frame = metalrobo::collideCpuReference(
        shapes,
        bodies,
        config,
        cache
    );
    require(frame.succeeded(), "sweep-and-prune corpus failed");
    require(
        containsPair(frame, 0u, 1u),
        "boundary-touching AABBs were dropped"
    );

    std::set<PairKey> bruteForce;
    for (std::uint32_t left = 0u; left < count; ++left) {
        for (std::uint32_t right = left + 1u;
             right < count;
             ++right) {
            if (aabbOverlap(
                    frame.worldAabbs[left],
                    frame.worldAabbs[right]
                )) {
                bruteForce.emplace(left, right);
            }
        }
    }
    const std::set<PairKey> emitted = pairSet(frame);
    require(
        emitted == bruteForce,
        "sweep-and-prune differs from brute-force AABB corpus"
    );

    const auto replay = metalrobo::collideCpuReference(
        shapes,
        bodies,
        config,
        cache
    );
    require(replay.succeeded(), "corpus replay failed");
    require(
        pairSet(replay) == emitted,
        "broadphase pair IDs changed across replay"
    );
    return bruteForce.size();
}

void verifyTransactionalOverflow(
    const PrimaryScene& scene,
    const metalrobo::CollisionConfig& successfulConfig,
    metalrobo::PersistentManifoldCache& cache,
    const metalrobo::CollisionFrame& successfulFrame
) {
    const std::vector<metalrobo::PersistentManifold> snapshot(
        cache.entries().begin(),
        cache.entries().end()
    );
    auto expectUnchanged =
        [&](const metalrobo::CollisionFrame& failed,
            const std::uint32_t expectedCode) {
            require(
                !failed.succeeded() &&
                    failed.diagnostics.code == expectedCode,
                "capacity failure returned the wrong status"
            );
            require(
                emptyPayload(failed),
                "capacity failure leaked a partial payload"
            );
            require(
                sameCache(snapshot, cache.entries()),
                "capacity failure mutated the manifold cache"
            );
        };

    metalrobo::CollisionConfig pairOverflow = successfulConfig;
    pairOverflow.capacities.pairCapacity =
        static_cast<std::uint32_t>(
            successfulFrame.diagnostics.requiredPairs - 1u
        );
    const auto pairFailure = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        pairOverflow,
        cache
    );
    expectUnchanged(
        pairFailure,
        MR_STEP_PAIR_CAPACITY_OVERFLOW
    );
    require(
        pairFailure.diagnostics.requiredPairs ==
            successfulFrame.diagnostics.requiredPairs,
        "pair overflow omitted the exact required capacity"
    );

    metalrobo::CollisionConfig contactOverflow = successfulConfig;
    contactOverflow.capacities.rawContactCapacity =
        static_cast<std::uint32_t>(
            successfulFrame.diagnostics.requiredRawContacts - 1u
        );
    const auto contactFailure = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        contactOverflow,
        cache
    );
    expectUnchanged(
        contactFailure,
        MR_STEP_CONTACT_CAPACITY_OVERFLOW
    );
    require(
        contactFailure.diagnostics.requiredRawContacts ==
                successfulFrame.diagnostics.requiredRawContacts &&
            contactFailure.diagnostics.requiredManifolds ==
                successfulFrame.diagnostics.requiredManifolds,
        "contact overflow omitted exact preflight requirements"
    );

    metalrobo::CollisionConfig manifoldOverflow = successfulConfig;
    manifoldOverflow.capacities.manifoldCapacity =
        static_cast<std::uint32_t>(
            successfulFrame.diagnostics.requiredManifolds - 1u
        );
    const auto manifoldFailure = metalrobo::collideCpuReference(
        scene.shapes,
        scene.bodies,
        manifoldOverflow,
        cache
    );
    expectUnchanged(
        manifoldFailure,
        MR_STEP_MANIFOLD_CAPACITY_OVERFLOW
    );
    require(
        manifoldFailure.diagnostics.requiredRawContacts ==
                successfulFrame.diagnostics.requiredRawContacts &&
            manifoldFailure.diagnostics.requiredManifolds ==
                successfulFrame.diagnostics.requiredManifolds,
        "manifold overflow omitted exact preflight requirements"
    );
}

} // namespace

int main() {
    try {
        const PrimaryScene scene = makePrimaryScene();
        const metalrobo::CollisionConfig config = ampleConfig();
        metalrobo::PersistentManifoldCache cache;

        const auto first = metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            config,
            cache
        );
        verifyPrimaryTopology(first);
        require(
            first.diagnostics.newPoints == 13u &&
                first.diagnostics.refreshedPoints == 0u,
            "first collision frame did not build fresh manifolds"
        );

        const auto second = metalrobo::collideCpuReference(
            scene.shapes,
            scene.bodies,
            config,
            cache
        );
        verifyPrimaryTopology(second);
        verifyStableTopology(first, second);
        require(
            second.diagnostics.refreshedPoints == 13u &&
                second.diagnostics.newPoints == 0u,
            "persistent manifold refresh was not reused"
        );
        verifyFilters(scene, config);
        verifyFourPointReduction();
        verifyCapsulePairPrimitives();
        verifySphereBoxPrimitive();
        const std::size_t corpusPairs =
            verifySweepAndPruneCorpus();
        verifyTransactionalOverflow(scene, config, cache, second);

        std::cout
            << "collision=cpu_fp64"
            << " pairs=" << second.pairs.size()
            << " raw_contacts=" << second.rawContacts.size()
            << " manifolds=" << second.manifoldHeaders.size()
            << " sap_corpus_pairs=" << corpusPairs
            << " false_negatives=0"
            << " pair_classes=8"
            << " sphere_capsule_order=yes"
            << " capsule_capsule_adversarial=yes"
            << " sphere_box_adversarial=yes"
            << " stable_ids=yes"
            << " persistent_refresh=yes"
            << " manifold_reduction=8_to_4"
            << " canonical_filters=yes"
            << " overflow_transactional=yes"
            << " finite=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_collision_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
