#pragma once

#include "metalrobo/EngineModel.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace metalrobo {

enum class GeometryCookStatus : std::uint32_t {
    success = 0u,
    invalidInput = 1u,
    degenerateGeometry = 2u,
    nonManifold = 3u,
    nonConvex = 4u,
    capacityOverflow = 5u,
};

struct GeometryCookConfig {
    double weldTolerance = 1.0e-7;
    double minimumTriangleArea = 1.0e-12;
    std::uint32_t meshLeafTriangleCapacity = 4u;
    bool removeDuplicateTriangles = true;
};

struct GeometryCookResult {
    GeometryCookStatus status = GeometryCookStatus::success;
    std::uint32_t geometryIndex = MR_INVALID_INDEX;
    std::uint32_t inputVertexCount = 0u;
    std::uint32_t outputVertexCount = 0u;
    std::uint32_t inputTriangleCount = 0u;
    std::uint32_t outputTriangleCount = 0u;
    std::uint32_t removedDegenerateTriangles = 0u;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == GeometryCookStatus::success;
    }
};

// Cooks an authored closed convex surface. The cooker deterministically welds
// vertices, repairs face winding around the vertex centroid, validates
// half-edge closure and convexity, and appends pointer-free geometry arenas to
// the model. Indices are packed xyz triplets.
[[nodiscard]] GeometryCookResult cookConvexGeometry(
    EngineModel& model,
    std::span<const mr_float4> vertices,
    std::span<const std::uint32_t> triangleIndices,
    const GeometryCookConfig& config = {}
);

// Cooks a static/kinematic triangle mesh with deterministic welding,
// degeneracy removal, adjacency/internal-edge ownership, and an inflated
// quantized BVH4 with threaded escape cursors.
[[nodiscard]] GeometryCookResult cookTriangleMeshGeometry(
    EngineModel& model,
    std::span<const mr_float4> vertices,
    std::span<const std::uint32_t> triangleIndices,
    std::span<const std::uint32_t> triangleMaterials = {},
    const GeometryCookConfig& config = {}
);

} // namespace metalrobo
