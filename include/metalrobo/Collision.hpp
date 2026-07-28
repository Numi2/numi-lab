#pragma once

#include "metalrobo/engine_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

// Collision is evaluated for one logically isolated environment per call.
// Collider indices are stable positions in the shapes span.
struct CollisionPairExclusion {
    std::uint32_t colliderA = 0;
    std::uint32_t colliderB = 0;
};

enum CollisionPairClass : std::uint32_t {
    collisionPairSphereSphere = 1u,
    collisionPairSpherePlane = 2u,
    collisionPairCapsulePlane = 3u,
    collisionPairBoxPlane = 4u,
    collisionPairCylinderPlane = 5u,
};

struct CollisionCapacities {
    std::uint32_t pairCapacity = 0;
    std::uint32_t rawContactCapacity = 0;
    std::uint32_t manifoldCapacity = 0;
};

struct CollisionConfig {
    std::uint32_t environment = 0;
    CollisionCapacities capacities{};

    // Cached body-local anchors survive while their normal and tangential
    // drift remain inside these bounds.
    double manifoldBreakingSeparation = 0.02;
    double manifoldBreakingTangential = 0.02;
    double manifoldMergeDistance = 0.002;
    double manifoldNormalCosine = 0.95;
};

struct CollisionDiagnostics {
    MRStepStatusCode code = MR_STEP_SUCCESS;
    std::uint64_t requiredPairs = 0;
    std::uint64_t requiredRawContacts = 0;
    std::uint64_t requiredManifolds = 0;
    std::uint32_t broadphaseAxis = 0;
    std::uint32_t refreshedPoints = 0;
    std::uint32_t newPoints = 0;

    [[nodiscard]] bool succeeded() const noexcept {
        return code == MR_STEP_SUCCESS;
    }
};

struct PersistentManifold {
    MRManifoldHeaderGPU header{};
    std::array<MRManifoldPointGPU, 4> points{};
};

class PersistentManifoldCache {
public:
    void clear() noexcept;

    [[nodiscard]] std::span<const PersistentManifold>
    entries() const noexcept;

    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::vector<PersistentManifold> entries_;

    friend struct CollisionCacheAccess;
};

struct CollisionFrame {
    CollisionDiagnostics diagnostics{};
    std::vector<MRAabbGPU> worldAabbs;
    std::vector<MRCandidatePairGPU> pairs;
    std::vector<MRRawContactGPU> rawContacts;

    // One entry per raw contact, indexing pairs.
    std::vector<std::uint32_t> rawContactPairIndices;

    // Four point records are stored for every header. Only the first
    // pairAndCount[3] records are active.
    std::vector<MRManifoldHeaderGPU> manifoldHeaders;
    std::vector<MRManifoldPointGPU> manifoldPoints;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// FP64 CPU collision oracle. On every failure all CollisionFrame payload
// vectors are empty and cache is unchanged. Required capacities remain in
// diagnostics so callers can grow and replay transactionally.
[[nodiscard]] CollisionFrame collideCpuReference(
    std::span<const MRShapeGPU> shapes,
    std::span<const MRBodyStateGPU> bodies,
    const CollisionConfig& config,
    PersistentManifoldCache& cache,
    std::span<const CollisionPairExclusion> exclusions = {}
);

} // namespace metalrobo
