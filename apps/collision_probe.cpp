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
    };
    result.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, f4(0.0f, 0.0f, 0.0f), 100u),
        makeShape(1u, MR_SHAPE_SPHERE, f4(0.5f, 0.0f, 0.0f), 101u),
        makeShape(2u, MR_SHAPE_SPHERE, f4(0.5f, 0.0f, 0.0f), 102u),
        makeShape(3u, MR_SHAPE_CAPSULE, f4(0.35f, 0.55f, 0.0f), 103u),
        makeShape(4u, MR_SHAPE_BOX, f4(0.5f, 0.5f, 0.5f), 104u),
    };
    return result;
}

bool sameFloat4(const mr_float4 left, const mr_float4 right) {
    return
        left.x == right.x &&
        left.y == right.y &&
        left.z == right.z &&
        left.w == right.w;
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
    require(frame.worldAabbs.size() == 5u, "missing world AABBs");
    require(frame.pairs.size() == 5u, "unexpected primary pair count");
    require(
        frame.rawContacts.size() == 9u,
        "unexpected primary raw-contact count"
    );
    require(
        frame.manifoldHeaders.size() == 5u &&
            frame.manifoldPoints.size() == 20u,
        "unexpected primary manifold layout"
    );
    require(
        frame.diagnostics.requiredPairs == 5u &&
            frame.diagnostics.requiredRawContacts == 9u &&
            frame.diagnostics.requiredManifolds == 5u,
        "primary preflight requirements are wrong"
    );

    const std::set<PairKey> expected{
        {0u, 1u},
        {0u, 2u},
        {0u, 3u},
        {0u, 4u},
        {1u, 2u},
    };
    require(pairSet(frame) == expected, "primary pair identities changed");

    std::array<bool, 5> pairClassSeen{};
    std::array<std::uint32_t, 5> contactsPerPair{};
    for (std::size_t pairIndex = 0;
         pairIndex < frame.pairs.size();
         ++pairIndex) {
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        require(
            pair.environment == 7u &&
                pair.flags >= metalrobo::collisionPairSphereSphere &&
                pair.flags <= metalrobo::collisionPairBoxPlane,
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
            pairClassSeen[metalrobo::collisionPairBoxPlane],
        "not every analytic pair class was exercised"
    );

    for (std::size_t pairIndex = 0;
         pairIndex < frame.pairs.size();
         ++pairIndex) {
        const MRCandidatePairGPU& pair = frame.pairs[pairIndex];
        std::uint32_t expectedCount = 1u;
        if (pair.flags == metalrobo::collisionPairCapsulePlane) {
            expectedCount = 2u;
        } else if (pair.flags == metalrobo::collisionPairBoxPlane) {
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
            excluded.pairs.size() == 4u,
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
            first.diagnostics.newPoints == 9u &&
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
            second.diagnostics.refreshedPoints == 9u &&
                second.diagnostics.newPoints == 0u,
            "persistent manifold refresh was not reused"
        );
        verifyFilters(scene, config);
        verifyFourPointReduction();
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
            << " pair_classes=4"
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
