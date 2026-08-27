#include "metalrobo/NumiHumanTendon.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metalrobo {
namespace {

constexpr std::array<char, 8> kMagic{'N', 'H', 'T', 'E', 'N', 'D', '1', '\0'};
constexpr std::uint32_t kAbi = 1u;
constexpr std::size_t kHeaderBytes = 104u;
constexpr std::size_t kBindingBytes = 64u;
constexpr std::size_t kTriangleBytes = 64u;
constexpr double kPointTolerance = 1.0e-6;

NumiHumanTendonDiagnostics failure(
    const NumiHumanTendonStatus status,
    const std::uint32_t index = MR_INVALID_INDEX
) {
    return {status, index};
}

template <typename T>
bool read(std::span<const std::byte> bytes, std::size_t& offset, T& value) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) return false;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

bool finite(const std::array<double, 3>& value) {
    return std::all_of(value.begin(), value.end(), [](const double item) {
        return std::isfinite(item);
    });
}

double distance(const std::array<double, 3>& left, const std::array<double, 3>& right) {
    double squared = 0.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const double delta = left[axis] - right[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

std::array<double, 3> subtract(
    const std::array<double, 3>& left, const std::array<double, 3>& right
) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

std::array<double, 3> cross(
    const std::array<double, 3>& left, const std::array<double, 3>& right
) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

} // namespace

NumiHumanTendonDiagnostics decodeNumiHumanTendonPayload(
    const std::span<const std::byte> bytes,
    const std::span<const std::uint8_t> expectedSourceSha256,
    const std::span<const std::uint8_t> expectedMusclePayloadSha256,
    NumiHumanTendonPayload& payload
) {
    payload = {};
    if (bytes.size() < kHeaderBytes) return failure(NumiHumanTendonStatus::truncatedPayload);
    std::size_t offset = 0u;
    std::array<char, 8> magic{};
    std::uint32_t abi = 0u;
    std::uint32_t endpointCount = 0u;
    std::uint32_t triangleCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    if (!read(bytes, offset, magic) || !read(bytes, offset, abi) ||
        !read(bytes, offset, payload.bodyCount) || !read(bytes, offset, payload.muscleCount) ||
        !read(bytes, offset, payload.sourceSiteCount) || !read(bytes, offset, endpointCount) ||
        !read(bytes, offset, triangleCount) || !read(bytes, offset, reserved0) ||
        !read(bytes, offset, reserved1) || !read(bytes, offset, payload.sourceSha256) ||
        !read(bytes, offset, payload.musclePayloadSha256)) {
        return failure(NumiHumanTendonStatus::truncatedPayload);
    }
    if (magic != kMagic || abi != kAbi || payload.bodyCount == 0u ||
        payload.muscleCount == 0u || payload.sourceSiteCount == 0u ||
        endpointCount != 2u * payload.muscleCount || reserved0 != 0u || reserved1 != 0u) {
        return failure(NumiHumanTendonStatus::invalidPayload);
    }
    const std::size_t expectedBytes = kHeaderBytes +
        static_cast<std::size_t>(endpointCount) * kBindingBytes +
        static_cast<std::size_t>(triangleCount) * kTriangleBytes;
    if (bytes.size() != expectedBytes) {
        return failure(NumiHumanTendonStatus::invalidPayload);
    }
    if ((!expectedSourceSha256.empty() &&
         (expectedSourceSha256.size() != payload.sourceSha256.size() ||
          !std::equal(expectedSourceSha256.begin(), expectedSourceSha256.end(), payload.sourceSha256.begin()))) ||
        (!expectedMusclePayloadSha256.empty() &&
         (expectedMusclePayloadSha256.size() != payload.musclePayloadSha256.size() ||
          !std::equal(expectedMusclePayloadSha256.begin(), expectedMusclePayloadSha256.end(), payload.musclePayloadSha256.begin())))) {
        return failure(NumiHumanTendonStatus::sourceMismatch);
    }
    payload.bindings.reserve(endpointCount);
    for (std::uint32_t index = 0u; index < endpointCount; ++index) {
        std::array<std::uint32_t, 8> integers{};
        std::array<float, 8> values{};
        if (!read(bytes, offset, integers) || !read(bytes, offset, values)) {
            return failure(NumiHumanTendonStatus::truncatedPayload, index);
        }
        NumiHumanTendonBinding binding;
        binding.muscleIndex = integers[0];
        binding.endpointOrdinal = integers[1];
        binding.routeNodeIndex = integers[2];
        binding.sourceSiteIndex = integers[3];
        binding.bodyIndex = integers[4];
        if (integers[5] > static_cast<std::uint32_t>(NumiHumanTendonAttachmentMode::registeredBoneTriangle)) {
            return failure(NumiHumanTendonStatus::invalidBinding, index);
        }
        binding.mode = static_cast<NumiHumanTendonAttachmentMode>(integers[5]);
        binding.triangleIndex = integers[6];
        binding.boneStableId = integers[7];
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            binding.resolvedLocalPoint[axis] = values[axis];
            binding.barycentric[axis] = values[3u + axis];
        }
        binding.endpointMigration = values[6];
        if (values[7] != 0.0f || !finite(binding.resolvedLocalPoint) ||
            !finite(binding.barycentric) || !std::isfinite(binding.endpointMigration) ||
            binding.endpointMigration < 0.0) {
            return failure(NumiHumanTendonStatus::invalidBinding, index);
        }
        payload.bindings.push_back(binding);
    }
    payload.triangles.reserve(triangleCount);
    for (std::uint32_t index = 0u; index < triangleCount; ++index) {
        std::array<std::uint32_t, 4> integers{};
        std::array<float, 12> values{};
        if (!read(bytes, offset, integers) || !read(bytes, offset, values)) {
            return failure(NumiHumanTendonStatus::truncatedPayload, index);
        }
        if (integers[3] != 0u) return failure(NumiHumanTendonStatus::invalidBinding, index);
        NumiHumanTendonTriangle triangle;
        triangle.bodyIndex = integers[0];
        triangle.boneStableId = integers[1];
        triangle.sourceTriangleIndex = integers[2];
        for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                triangle.localVertices[vertex][axis] = values[4u * vertex + axis];
            }
            if (values[4u * vertex + 3u] != 0.0f || !finite(triangle.localVertices[vertex])) {
                return failure(NumiHumanTendonStatus::invalidBinding, index);
            }
        }
        payload.triangles.push_back(triangle);
    }
    return {};
}

NumiHumanTendonDiagnostics resolveNumiHumanTendonProgram(
    const NumiHumanTendonPayload& payload,
    const std::span<const MujocoMuscleSite> sourceSites,
    const std::span<const MujocoMuscleDefinition> sourceMuscles,
    NumiHumanTendonResolvedProgram& result
) {
    result = {};
    if (payload.muscleCount != sourceMuscles.size() ||
        payload.sourceSiteCount != sourceSites.size() ||
        payload.bindings.size() != 2u * sourceMuscles.size()) {
        return failure(NumiHumanTendonStatus::incompleteCoverage);
    }
    result.sites.assign(sourceSites.begin(), sourceSites.end());
    result.muscles.assign(sourceMuscles.begin(), sourceMuscles.end());
    std::vector<bool> observed(payload.bindings.size(), false);
    std::uint32_t expectedRouteNode = 0u;
    for (std::uint32_t muscleIndex = 0u; muscleIndex < sourceMuscles.size(); ++muscleIndex) {
        const MujocoMuscleDefinition& source = sourceMuscles[muscleIndex];
        if (source.route.size() < 2u || source.route.front().type != MujocoRouteNodeType::site ||
            source.route.back().type != MujocoRouteNodeType::site) {
            return failure(NumiHumanTendonStatus::invalidBinding, muscleIndex);
        }
        for (std::uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
            const std::size_t bindingIndex = 2u * muscleIndex + endpoint;
            const NumiHumanTendonBinding& binding = payload.bindings[bindingIndex];
            const std::uint32_t localRouteIndex = endpoint == 0u
                ? 0u : static_cast<std::uint32_t>(source.route.size() - 1u);
            const MujocoRouteNode& sourceNode = source.route[localRouteIndex];
            const std::uint32_t requiredAbsoluteIndex = expectedRouteNode + localRouteIndex;
            if (binding.muscleIndex != muscleIndex || binding.endpointOrdinal != endpoint ||
                binding.routeNodeIndex != requiredAbsoluteIndex ||
                binding.sourceSiteIndex != sourceNode.targetIndex || binding.sourceSiteIndex >= sourceSites.size() ||
                binding.bodyIndex != sourceSites[binding.sourceSiteIndex].bodyIndex || binding.bodyIndex >= payload.bodyCount) {
                return failure(NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(bindingIndex));
            }
            observed[bindingIndex] = true;
            if (binding.mode == NumiHumanTendonAttachmentMode::sourceSitePoint) {
                if (binding.triangleIndex != MR_INVALID_INDEX || binding.boneStableId != 0u ||
                    distance(binding.resolvedLocalPoint, sourceSites[binding.sourceSiteIndex].localPoint) > kPointTolerance ||
                    distance(binding.barycentric, std::array<double, 3>{}) > kPointTolerance ||
                    binding.endpointMigration > kPointTolerance) {
                    return failure(NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(bindingIndex));
                }
                ++result.pointBindingCount;
                continue;
            }
            if (binding.triangleIndex >= payload.triangles.size() || binding.boneStableId == 0u) {
                return failure(NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(bindingIndex));
            }
            const NumiHumanTendonTriangle& triangle = payload.triangles[binding.triangleIndex];
            const double weightSum = binding.barycentric[0] + binding.barycentric[1] + binding.barycentric[2];
            if (triangle.bodyIndex != binding.bodyIndex || triangle.boneStableId != binding.boneStableId ||
                std::any_of(binding.barycentric.begin(), binding.barycentric.end(), [](const double weight) {
                    return weight < 0.0 || weight > 1.0;
                }) || std::abs(weightSum - 1.0) > kPointTolerance) {
                return failure(NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(bindingIndex));
            }
            std::array<double, 3> trianglePoint{};
            for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    trianglePoint[axis] += binding.barycentric[vertex] * triangle.localVertices[vertex][axis];
                }
            }
            if (distance(trianglePoint, binding.resolvedLocalPoint) > kPointTolerance ||
                std::abs(distance(binding.resolvedLocalPoint, sourceSites[binding.sourceSiteIndex].localPoint) -
                         binding.endpointMigration) > kPointTolerance) {
                return failure(NumiHumanTendonStatus::invalidBinding, static_cast<std::uint32_t>(bindingIndex));
            }
            const std::uint32_t newSiteIndex = static_cast<std::uint32_t>(result.sites.size());
            result.sites.push_back({binding.bodyIndex, binding.resolvedLocalPoint});
            result.muscles[muscleIndex].route[localRouteIndex].targetIndex = newSiteIndex;
            ++result.triangleBindingCount;
            result.maximumEndpointMigration = std::max(result.maximumEndpointMigration, binding.endpointMigration);
        }
        expectedRouteNode += static_cast<std::uint32_t>(source.route.size());
    }
    if (!std::all_of(observed.begin(), observed.end(), [](const bool value) { return value; })) {
        return failure(NumiHumanTendonStatus::incompleteCoverage);
    }
    return {};
}

NumiHumanTendonDiagnostics evaluateNumiHumanTendonTraction(
    const NumiHumanTendonBinding& binding,
    const std::span<const std::array<double, 3>> worldTriangle,
    const std::array<double, 3>& terminalWorld,
    const std::array<double, 3>& adjacentRouteWorld,
    const std::array<double, 3>& bodyOriginWorld,
    const double actuatorForce,
    NumiHumanTendonTractionResult& result
) {
    result = {};
    if (!finite(terminalWorld) || !finite(adjacentRouteWorld) ||
        !finite(bodyOriginWorld) || !std::isfinite(actuatorForce)) {
        return failure(NumiHumanTendonStatus::nonfiniteResult);
    }
    const std::array<double, 3> directionVector = subtract(terminalWorld, adjacentRouteWorld);
    const double directionNorm = distance(terminalWorld, adjacentRouteWorld);
    if (!(directionNorm > 1.0e-12)) return failure(NumiHumanTendonStatus::invalidBinding);
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        result.terminalForce[axis] = actuatorForce * directionVector[axis] / directionNorm;
    }
    if (binding.mode == NumiHumanTendonAttachmentMode::sourceSitePoint) {
        if (!worldTriangle.empty()) return failure(NumiHumanTendonStatus::invalidBinding);
        result.nodalForces[0] = result.terminalForce;
        return {};
    }
    if (worldTriangle.size() != 3u) return failure(NumiHumanTendonStatus::invalidBinding);
    std::array<double, 3> representedPoint{};
    std::array<double, 3> resultant{};
    std::array<double, 3> representedMoment{};
    for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
        if (!finite(worldTriangle[vertex])) return failure(NumiHumanTendonStatus::nonfiniteResult);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            representedPoint[axis] += binding.barycentric[vertex] * worldTriangle[vertex][axis];
            result.nodalForces[vertex][axis] = binding.barycentric[vertex] * result.terminalForce[axis];
            resultant[axis] += result.nodalForces[vertex][axis];
        }
        const std::array<double, 3> nodalMoment = cross(
            subtract(worldTriangle[vertex], bodyOriginWorld), result.nodalForces[vertex]
        );
        for (std::size_t axis = 0u; axis < 3u; ++axis) representedMoment[axis] += nodalMoment[axis];
    }
    if (distance(representedPoint, terminalWorld) > 2.0e-6) {
        return failure(NumiHumanTendonStatus::invalidBinding);
    }
    result.forceResidual = distance(resultant, result.terminalForce);
    const std::array<double, 3> expectedMoment = cross(
        subtract(terminalWorld, bodyOriginWorld), result.terminalForce
    );
    result.momentResidual = distance(representedMoment, expectedMoment);
    if (!std::isfinite(result.forceResidual) || !std::isfinite(result.momentResidual)) {
        return failure(NumiHumanTendonStatus::nonfiniteResult);
    }
    return {};
}

const char* numiHumanTendonStatusName(const NumiHumanTendonStatus status) noexcept {
    switch (status) {
    case NumiHumanTendonStatus::success: return "success";
    case NumiHumanTendonStatus::truncatedPayload: return "truncated_payload";
    case NumiHumanTendonStatus::invalidPayload: return "invalid_payload";
    case NumiHumanTendonStatus::sourceMismatch: return "source_mismatch";
    case NumiHumanTendonStatus::incompleteCoverage: return "incomplete_coverage";
    case NumiHumanTendonStatus::invalidBinding: return "invalid_binding";
    case NumiHumanTendonStatus::nonfiniteResult: return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
