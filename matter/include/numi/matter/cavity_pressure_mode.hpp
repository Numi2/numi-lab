#pragma once

#include "shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace numi::matter::detail {

// Build one source-bound FEM direction proportional to dV_cavity/dx. The
// ordinary selector-6 coarse-mode path performs its existing two-pass
// orthogonalization and normalization afterward.
inline bool buildCavityPressureLoadMode(
    std::span<const NMFEMNodeStateGPU> nodes,
    std::span<const NMFEMTopologyNodeGPU> topologyNodes,
    std::span<const NMVascularCavityGPU> cavities,
    std::span<const NMVascularCavityFaceGPU> faces,
    std::uint32_t objectIndex,
    std::vector<nm_float4>& output,
    std::string& error,
    bool inverseLumpedMass = false) {
    output.clear();
    error.clear();
    if (nodes.empty() || topologyNodes.size() != nodes.size()) {
        error = "cavity pressure mode requires complete FEM node ownership";
        return false;
    }
    if (cavities.size() != 1u) {
        error = "cavity pressure mode requires exactly one registered cavity";
        return false;
    }
    const auto& cavity = cavities.front();
    if (cavity.identity.z != objectIndex || cavity.faces.y < 4u ||
        cavity.faces.x > faces.size() ||
        cavity.faces.y > faces.size() - cavity.faces.x) {
        error = "cavity pressure mode source identity or face extent is invalid";
        return false;
    }
    const std::size_t first = cavity.faces.x;
    const std::size_t end = first + cavity.faces.y;
    struct EdgeUse { std::uint32_t count = 0u; int orientation = 0; };
    std::unordered_map<std::uint64_t, EdgeUse> edges;
    std::unordered_set<std::uint32_t> stableFaces;
    std::vector<std::array<double, 3>> points(nodes.size());
    std::vector<std::uint8_t> touched(nodes.size(), 0u);
    std::array<double, 3> lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (std::size_t faceIndex = first; faceIndex < end; ++faceIndex) {
        const auto& face = faces[faceIndex];
        if (face.identity.x != 0u ||
            (face.identity.z != 1u && face.identity.z != 3u) ||
            face.identity.w != 0u || !stableFaces.insert(face.identity.y).second) {
            error = "cavity pressure mode has invalid or duplicate source faces";
            return false;
        }
        const std::array<std::uint32_t, 3> tri{
            face.nodes.x, face.nodes.y, face.nodes.z};
        if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) {
            error = "cavity pressure mode contains a degenerate face";
            return false;
        }
        for (std::uint32_t node : tri) {
            if (node >= nodes.size() || topologyNodes[node].identity.y != objectIndex) {
                error = "cavity pressure face escapes its compiled FEM object";
                return false;
            }
            if (touched[node] == 0u) {
                const auto& rest = nodes[node].restAndFixed;
                const float mass = nodes[node].positionAndMass.w;
                if (!std::isfinite(rest.w) || !std::isfinite(mass)) {
                    error = "cavity pressure mode has nonfinite node constraints or mass";
                    return false;
                }
                points[node] = {rest.x, rest.y, rest.z};
                for (unsigned axis = 0; axis < 3u; ++axis) {
                    if (!std::isfinite(points[node][axis])) {
                        error = "cavity pressure mode has nonfinite rest coordinates";
                        return false;
                    }
                    lower[axis] = std::min(lower[axis], points[node][axis]);
                    upper[axis] = std::max(upper[axis], points[node][axis]);
                }
                touched[node] = 1u;
            }
        }
        for (unsigned corner = 0; corner < 3u; ++corner) {
            const std::uint32_t a = tri[corner];
            const std::uint32_t b = tri[(corner + 1u) % 3u];
            const std::uint32_t lo = std::min(a, b), hi = std::max(a, b);
            auto& use = edges[(std::uint64_t(lo) << 32u) | hi];
            ++use.count;
            use.orientation += (a < b) ? 1 : -1;
        }
    }
    for (const auto& [edge, use] : edges) {
        (void)edge;
        if (use.count != 2u || use.orientation != 0) {
            error = "cavity pressure boundary is not a closed oriented two-manifold";
            return false;
        }
    }
    const std::array<double, 3> origin{
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        0.5 * (lower[2] + upper[2])};
    const double scale = std::max({upper[0] - lower[0], upper[1] - lower[1],
                                   upper[2] - lower[2]});
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        error = "cavity pressure boundary has zero or invalid extent";
        return false;
    }
    std::vector<std::array<double, 3>> gradient(nodes.size(), {0.0, 0.0, 0.0});
    double signedVolume = 0.0;
    const auto relative = [&](std::uint32_t node) {
        return std::array<double, 3>{points[node][0] - origin[0],
                                    points[node][1] - origin[1],
                                    points[node][2] - origin[2]};
    };
    const auto cross = [](const std::array<double, 3>& a,
                          const std::array<double, 3>& b) {
        return std::array<double, 3>{a[1] * b[2] - a[2] * b[1],
                                     a[2] * b[0] - a[0] * b[2],
                                     a[0] * b[1] - a[1] * b[0]};
    };
    for (std::size_t faceIndex = first; faceIndex < end; ++faceIndex) {
        const auto& tri = faces[faceIndex].nodes;
        const auto a = relative(tri.x), b = relative(tri.y), c = relative(tri.z);
        const auto bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
        signedVolume += (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]) / 6.0;
        const std::array<std::array<double, 3>, 3> local{bc, ca, ab};
        const std::array<std::uint32_t, 3> triNodes{tri.x, tri.y, tri.z};
        for (unsigned corner = 0; corner < 3u; ++corner) {
            const auto& node = nodes[triNodes[corner]];
            const bool authoredStaticConstraint =
                node.restAndFixed.w > 0.5f && node.restAndFixed.w < 1.5f;
            if (authoredStaticConstraint || !(node.positionAndMass.w > 0.0f))
                continue;
            for (unsigned axis = 0; axis < 3u; ++axis)
                gradient[triNodes[corner]][axis] += local[corner][axis] / 6.0;
        }
    }
    const double volumeFloor = std::max(scale * scale * scale * 1.0e-12, 1.0e-30);
    if (!std::isfinite(signedVolume) || !(signedVolume > volumeFloor)) {
        error = "cavity pressure mode has nonpositive or degenerate source volume";
        return false;
    }
    double gradientNormSquared = 0.0;
    std::vector<nm_float4> candidate(nodes.size(), nm_float4{0, 0, 0, 0});
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (touched[node] == 0u ||
            (nodes[node].restAndFixed.w > 0.5f &&
             nodes[node].restAndFixed.w < 1.5f) ||
            !(nodes[node].positionAndMass.w > 0.0f)) continue;
        const double mass = nodes[node].positionAndMass.w;
        for (unsigned axis = 0; axis < 3u; ++axis) {
            const double loadComponent = gradient[node][axis];
            const double component = inverseLumpedMass
                ? loadComponent / mass : loadComponent;
            if (!std::isfinite(component) ||
                std::abs(component) > std::numeric_limits<float>::max()) {
                error = inverseLumpedMass
                    ? "mass-weighted cavity pressure mode is nonfinite or out of range"
                    : "cavity pressure mode gradient is nonfinite or out of range";
                return false;
            }
            gradientNormSquared += component * component;
            if (axis == 0u) candidate[node].x = static_cast<float>(component);
            else if (axis == 1u) candidate[node].y = static_cast<float>(component);
            else candidate[node].z = static_cast<float>(component);
        }
    }
    const double areaScale = scale * scale;
    const double normFloor = inverseLumpedMass
        ? 1.0e-300
        : std::max(areaScale * areaScale * 1.0e-24, 1.0e-60);
    if (!std::isfinite(gradientNormSquared) ||
        !(gradientNormSquared > normFloor)) {
        error = "cavity pressure mode has no free-node gradient support";
        return false;
    }
    output.swap(candidate);
    return true;
}

} // namespace numi::matter::detail
