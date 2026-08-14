#include "metalrobo/SurgicalVisual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
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

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
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

Vec3 normalized(const Vec3 value) {
    const double length = norm(value);
    if (!(length > 1.0e-12) || !std::isfinite(length)) {
        throw std::invalid_argument("surgical visual direction is degenerate");
    }
    return value * (1.0 / length);
}

Vec3 fromArray(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
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

mr_float4 f4(const Vec3 value, const double w = 0.0) {
    return f4(value.x, value.y, value.z, w);
}

MRVisualMaterialGPUV2 material(
    const mr_float4 color,
    const float roughness,
    const float metallic,
    const float clearcoat,
    const float clearcoatRoughness,
    const std::uint32_t stableId
) {
    MRVisualMaterialGPUV2 result{};
    result.baseColorAndOpacity = color;
    result.emissionAndStrength = {0.0f, 0.0f, 0.0f, 0.0f};
    result.surface = {roughness, metallic, 1.0f, 1.0f};
    result.coatingAndAlphaCutoff = {
        clearcoat,
        clearcoatRoughness,
        1.0f,
        0.5f,
    };
    result.textureIndices0 = {
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    result.textureIndices1 = result.textureIndices0;
    result.flags = {
        MR_VISUAL_ALPHA_OPAQUE,
        MR_VISUAL_MATERIAL_DOUBLE_SIDED,
        0u,
        stableId,
    };
    result.reserved = result.textureIndices0;
    return result;
}

struct GeometryRange {
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    Vec3 lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec3 upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
};

void include(GeometryRange& range, const Vec3 point) {
    range.lower.x = std::min(range.lower.x, point.x);
    range.lower.y = std::min(range.lower.y, point.y);
    range.lower.z = std::min(range.lower.z, point.z);
    range.upper.x = std::max(range.upper.x, point.x);
    range.upper.y = std::max(range.upper.y, point.y);
    range.upper.z = std::max(range.upper.z, point.z);
}

MRVisualVertexGPUV2 vertex(
    const Vec3 position,
    const Vec3 normal,
    const Vec3 tangent,
    const double u,
    const double v,
    const mr_float4 color = {1.0f, 1.0f, 1.0f, 1.0f}
) {
    return {
        f4(position, 1.0),
        f4(normal, 1.0),
        f4(tangent),
        f4(u, v, 0.0, 0.0),
        color,
    };
}

GeometryRange appendTubeGeometry(
    VisualAssetPackV2& pack,
    const std::vector<Vec3>& points,
    const std::vector<double>& radii,
    const std::uint32_t radialSections,
    const bool capEnds
) {
    if (points.size() < 2u || points.size() != radii.size() ||
        radialSections < 3u) {
        throw std::invalid_argument("surgical visual tube is invalid");
    }
    const std::uint64_t vertexCount =
        static_cast<std::uint64_t>(points.size()) * radialSections +
        (capEnds ? 2u : 0u);
    if (vertexCount > std::numeric_limits<std::uint32_t>::max() ||
        pack.vertices.size() >
            std::numeric_limits<std::uint32_t>::max() - vertexCount) {
        throw std::overflow_error("surgical visual vertex capacity overflow");
    }

    GeometryRange range;
    range.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    const std::uint32_t firstVertex =
        static_cast<std::uint32_t>(pack.vertices.size());
    std::vector<Vec3> tangents(points.size());
    std::vector<Vec3> normals(points.size());
    std::vector<Vec3> binormals(points.size());
    std::vector<double> distances(points.size(), 0.0);
    for (std::size_t index = 0u; index < points.size(); ++index) {
        const Vec3 direction =
            index == 0u
            ? points[1u] - points[0u]
            : index + 1u == points.size()
                ? points[index] - points[index - 1u]
                : points[index + 1u] - points[index - 1u];
        tangents[index] = normalized(direction);
        if (index > 0u) {
            distances[index] = distances[index - 1u] +
                norm(points[index] - points[index - 1u]);
        }
    }
    const Vec3 initialReference =
        std::abs(tangents.front().z) < 0.85
        ? Vec3{0.0, 0.0, 1.0}
        : Vec3{0.0, 1.0, 0.0};
    normals.front() = normalized(cross(initialReference, tangents.front()));
    binormals.front() = normalized(
        cross(tangents.front(), normals.front())
    );
    for (std::size_t index = 1u; index < points.size(); ++index) {
        const Vec3 projected = normals[index - 1u] -
            tangents[index] * dot(normals[index - 1u], tangents[index]);
        if (norm(projected) > 1.0e-8) {
            normals[index] = normalized(projected);
        } else {
            const Vec3 reference =
                std::abs(tangents[index].z) < 0.85
                ? Vec3{0.0, 0.0, 1.0}
                : Vec3{0.0, 1.0, 0.0};
            normals[index] = normalized(cross(reference, tangents[index]));
        }
        binormals[index] = normalized(
            cross(tangents[index], normals[index])
        );
    }
    const double totalDistance = std::max(distances.back(), 1.0e-12);
    for (std::size_t ring = 0u; ring < points.size(); ++ring) {
        if (!(radii[ring] > 0.0) || !std::isfinite(radii[ring])) {
            throw std::invalid_argument("surgical visual radius is invalid");
        }
        for (std::uint32_t side = 0u;
             side < radialSections;
             ++side) {
            const double angle = 2.0 * std::numbers::pi *
                static_cast<double>(side) /
                static_cast<double>(radialSections);
            const Vec3 surfaceNormal =
                normals[ring] * std::cos(angle) +
                binormals[ring] * std::sin(angle);
            const Vec3 position = points[ring] +
                surfaceNormal * radii[ring];
            pack.vertices.push_back(vertex(
                position,
                surfaceNormal,
                tangents[ring],
                distances[ring] / totalDistance,
                static_cast<double>(side) /
                    static_cast<double>(radialSections)
            ));
            include(range, position);
        }
    }
    for (std::uint32_t ring = 0u;
         ring + 1u < points.size();
         ++ring) {
        for (std::uint32_t side = 0u;
             side < radialSections;
             ++side) {
            const std::uint32_t next = (side + 1u) % radialSections;
            const std::uint32_t a = firstVertex +
                ring * radialSections + side;
            const std::uint32_t b = firstVertex +
                ring * radialSections + next;
            const std::uint32_t c = firstVertex +
                (ring + 1u) * radialSections + side;
            const std::uint32_t d = firstVertex +
                (ring + 1u) * radialSections + next;
            pack.indices.insert(pack.indices.end(), {a, c, b, b, c, d});
        }
    }
    if (capEnds) {
        const std::uint32_t firstCenter =
            static_cast<std::uint32_t>(pack.vertices.size());
        pack.vertices.push_back(vertex(
            points.front(),
            tangents.front() * -1.0,
            normals.front(),
            0.0,
            0.5
        ));
        const std::uint32_t lastCenter =
            static_cast<std::uint32_t>(pack.vertices.size());
        pack.vertices.push_back(vertex(
            points.back(),
            tangents.back(),
            normals.back(),
            1.0,
            0.5
        ));
        const std::uint32_t lastRing = static_cast<std::uint32_t>(
            points.size() - 1u
        );
        for (std::uint32_t side = 0u;
             side < radialSections;
             ++side) {
            const std::uint32_t next = (side + 1u) % radialSections;
            pack.indices.insert(pack.indices.end(), {
                firstCenter,
                firstVertex + next,
                firstVertex + side,
                lastCenter,
                firstVertex + lastRing * radialSections + side,
                firstVertex + lastRing * radialSections + next,
            });
        }
    }
    range.indexCount = static_cast<std::uint32_t>(
        pack.indices.size() - range.firstIndex
    );
    return range;
}

GeometryRange appendBoxGeometry(
    VisualAssetPackV2& pack,
    const Vec3 center,
    const Vec3 halfExtent
) {
    if (!(halfExtent.x > 0.0) || !(halfExtent.y > 0.0) ||
        !(halfExtent.z > 0.0)) {
        throw std::invalid_argument("surgical visual box is invalid");
    }
    GeometryRange range;
    range.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    constexpr std::array<Vec3, 6u> normals{{
        { 1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        {0.0,  1.0, 0.0}, {0.0, -1.0, 0.0},
        {0.0, 0.0,  1.0}, {0.0, 0.0, -1.0},
    }};
    constexpr std::array<Vec3, 6u> tangents{{
        {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0},
        {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
    }};
    constexpr std::array<std::array<Vec3, 4u>, 6u> signs{{
        {{{1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {1, -1, 1}}},
        {{{-1, 1, -1}, {-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}}},
        {{{1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {1, 1, 1}}},
        {{{-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}}},
        {{{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}}},
        {{{1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}}},
    }};
    for (std::size_t face = 0u; face < signs.size(); ++face) {
        const std::uint32_t first =
            static_cast<std::uint32_t>(pack.vertices.size());
        for (std::size_t corner = 0u; corner < 4u; ++corner) {
            const Vec3 sign = signs[face][corner];
            const Vec3 position{
                center.x + sign.x * halfExtent.x,
                center.y + sign.y * halfExtent.y,
                center.z + sign.z * halfExtent.z,
            };
            pack.vertices.push_back(vertex(
                position,
                normals[face],
                tangents[face],
                corner == 1u || corner == 2u ? 1.0 : 0.0,
                corner >= 2u ? 1.0 : 0.0
            ));
            include(range, position);
        }
        pack.indices.insert(pack.indices.end(), {
            first, first + 1u, first + 2u,
            first, first + 2u, first + 3u,
        });
    }
    range.indexCount = static_cast<std::uint32_t>(
        pack.indices.size() - range.firstIndex
    );
    return range;
}

GeometryRange appendSurfaceGeometry(
    VisualAssetPackV2& pack,
    const std::vector<std::array<double, 3>>& sourcePositions,
    const std::vector<std::array<std::uint32_t, 3>>& triangles,
    const std::array<double, 3>& translation
) {
    if (sourcePositions.size() < 3u || triangles.empty()) {
        throw std::invalid_argument("surgical tissue surface is empty");
    }
    const Vec3 offset = fromArray(translation);
    std::vector<Vec3> positions;
    positions.reserve(sourcePositions.size());
    for (const auto& source : sourcePositions) {
        const Vec3 position = fromArray(source) + offset;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            throw std::invalid_argument(
                "surgical tissue surface contains a non-finite position"
            );
        }
        positions.push_back(position);
    }
    std::vector<Vec3> normals(positions.size());
    for (const auto& triangle : triangles) {
        if (triangle[0] >= positions.size() ||
            triangle[1] >= positions.size() ||
            triangle[2] >= positions.size()) {
            throw std::invalid_argument(
                "surgical tissue surface index is outside its vertex table"
            );
        }
        const Vec3 areaNormal = cross(
            positions[triangle[1]] - positions[triangle[0]],
            positions[triangle[2]] - positions[triangle[0]]
        );
        if (!(norm(areaNormal) > 1.0e-15)) {
            throw std::invalid_argument(
                "surgical tissue surface contains a degenerate triangle"
            );
        }
        for (const std::uint32_t vertexIndex : triangle) {
            normals[vertexIndex] = normals[vertexIndex] + areaNormal;
        }
    }
    GeometryRange range;
    range.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    const std::uint32_t firstVertex =
        static_cast<std::uint32_t>(pack.vertices.size());
    Vec3 lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec3 upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (const Vec3 position : positions) {
        lower.x = std::min(lower.x, position.x);
        lower.y = std::min(lower.y, position.y);
        lower.z = std::min(lower.z, position.z);
        upper.x = std::max(upper.x, position.x);
        upper.y = std::max(upper.y, position.y);
        upper.z = std::max(upper.z, position.z);
    }
    const double width = std::max(upper.x - lower.x, 1.0e-12);
    const double height = std::max(upper.y - lower.y, 1.0e-12);
    for (std::size_t index = 0u; index < positions.size(); ++index) {
        // Interior FEM nodes remain in the stable physics index space but are
        // not referenced by the extracted boundary. Give those unused slots a
        // finite placeholder normal; every rendered vertex has the accumulated
        // outward normal of at least one boundary face.
        const Vec3 normal = norm(normals[index]) > 1.0e-12
            ? normalized(normals[index])
            : Vec3{0.0, 0.0, 1.0};
        Vec3 tangent = Vec3{1.0, 0.0, 0.0} -
            normal * normal.x;
        if (norm(tangent) <= 1.0e-8) {
            tangent = Vec3{0.0, 1.0, 0.0} -
                normal * normal.y;
        }
        pack.vertices.push_back(vertex(
            positions[index],
            normal,
            normalized(tangent),
            (positions[index].x - lower.x) / width,
            (positions[index].y - lower.y) / height
        ));
        include(range, positions[index]);
    }
    for (const auto& triangle : triangles) {
        pack.indices.insert(pack.indices.end(), {
            firstVertex + triangle[0],
            firstVertex + triangle[1],
            firstVertex + triangle[2],
        });
    }
    range.indexCount = static_cast<std::uint32_t>(
        pack.indices.size() - range.firstIndex
    );
    return range;
}

GeometryRange appendPrimitiveGeometryCopy(
    VisualAssetPackV2& destination,
    const VisualAssetPackV2& source,
    const MRVisualPrimitiveGPUV2& primitive
) {
    const std::uint64_t end =
        static_cast<std::uint64_t>(primitive.geometry.x) +
        primitive.geometry.y;
    if (end > source.indices.size()) {
        throw std::logic_error(
            "source surgical instrument primitive is out of bounds"
        );
    }
    GeometryRange range;
    range.firstIndex = static_cast<std::uint32_t>(
        destination.indices.size()
    );
    for (std::uint32_t local = 0u; local < primitive.geometry.y; ++local) {
        const std::uint32_t sourceVertex =
            source.indices[primitive.geometry.x + local];
        if (sourceVertex >= source.vertices.size()) {
            throw std::logic_error(
                "source surgical instrument vertex is out of bounds"
            );
        }
        const MRVisualVertexGPUV2& value = source.vertices[sourceVertex];
        const std::uint32_t destinationVertex =
            static_cast<std::uint32_t>(destination.vertices.size());
        destination.vertices.push_back(value);
        destination.indices.push_back(destinationVertex);
        include(range, {
            value.position.x,
            value.position.y,
            value.position.z,
        });
    }
    range.indexCount = primitive.geometry.y;
    return range;
}

GeometryRange appendTaperedBoxGeometry(
    VisualAssetPackV2& pack,
    const Vec3 lowerCenter,
    const Vec3 upperCenter,
    const Vec3 lowerHalfExtent,
    const Vec3 upperHalfExtent
) {
    if (!(upperCenter.z > lowerCenter.z) ||
        !(lowerHalfExtent.x > 0.0) ||
        !(lowerHalfExtent.y > 0.0) ||
        !(upperHalfExtent.x > 0.0) ||
        !(upperHalfExtent.y > 0.0)) {
        throw std::invalid_argument(
            "surgical visual tapered box is invalid"
        );
    }
    const std::array<Vec3, 8u> corners{{
        {lowerCenter.x - lowerHalfExtent.x,
         lowerCenter.y - lowerHalfExtent.y, lowerCenter.z},
        {lowerCenter.x + lowerHalfExtent.x,
         lowerCenter.y - lowerHalfExtent.y, lowerCenter.z},
        {lowerCenter.x + lowerHalfExtent.x,
         lowerCenter.y + lowerHalfExtent.y, lowerCenter.z},
        {lowerCenter.x - lowerHalfExtent.x,
         lowerCenter.y + lowerHalfExtent.y, lowerCenter.z},
        {upperCenter.x - upperHalfExtent.x,
         upperCenter.y - upperHalfExtent.y, upperCenter.z},
        {upperCenter.x + upperHalfExtent.x,
         upperCenter.y - upperHalfExtent.y, upperCenter.z},
        {upperCenter.x + upperHalfExtent.x,
         upperCenter.y + upperHalfExtent.y, upperCenter.z},
        {upperCenter.x - upperHalfExtent.x,
         upperCenter.y + upperHalfExtent.y, upperCenter.z},
    }};
    struct Face {
        std::array<std::uint32_t, 4u> corners;
        Vec3 outward;
    };
    constexpr std::array<Face, 6u> faces{{
        {{{0u, 1u, 2u, 3u}}, {0.0, 0.0, -1.0}},
        {{{4u, 7u, 6u, 5u}}, {0.0, 0.0, 1.0}},
        {{{1u, 5u, 6u, 2u}}, {1.0, 0.0, 0.0}},
        {{{0u, 3u, 7u, 4u}}, {-1.0, 0.0, 0.0}},
        {{{3u, 2u, 6u, 7u}}, {0.0, 1.0, 0.0}},
        {{{0u, 4u, 5u, 1u}}, {0.0, -1.0, 0.0}},
    }};

    GeometryRange range;
    range.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    for (const Face& face : faces) {
        std::array<Vec3, 4u> positions{{
            corners[face.corners[0u]],
            corners[face.corners[1u]],
            corners[face.corners[2u]],
            corners[face.corners[3u]],
        }};
        Vec3 normal = normalized(cross(
            positions[1u] - positions[0u],
            positions[2u] - positions[0u]
        ));
        if (dot(normal, face.outward) < 0.0) {
            std::swap(positions[1u], positions[3u]);
            normal = normalized(cross(
                positions[1u] - positions[0u],
                positions[2u] - positions[0u]
            ));
        }
        const Vec3 tangent = normalized(
            positions[1u] - positions[0u]
        );
        const std::uint32_t first =
            static_cast<std::uint32_t>(pack.vertices.size());
        for (std::size_t corner = 0u; corner < positions.size(); ++corner) {
            pack.vertices.push_back(vertex(
                positions[corner],
                normal,
                tangent,
                corner == 1u || corner == 2u ? 1.0 : 0.0,
                corner >= 2u ? 1.0 : 0.0
            ));
            include(range, positions[corner]);
        }
        pack.indices.insert(pack.indices.end(), {
            first, first + 1u, first + 2u,
            first, first + 2u, first + 3u,
        });
    }
    range.indexCount = static_cast<std::uint32_t>(
        pack.indices.size() - range.firstIndex
    );
    return range;
}

void appendInstance(
    VisualAssetPackV2& pack,
    const GeometryRange& geometry,
    const std::uint32_t materialIndex,
    const std::uint32_t bindingKind,
    const std::uint32_t bodyIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    const std::uint32_t stableId,
    std::string node,
    std::string link
) {
    const std::uint32_t instanceIndex =
        static_cast<std::uint32_t>(pack.instances.size());
    const std::uint32_t primitiveIndex =
        static_cast<std::uint32_t>(pack.primitives.size());
    MRVisualInstanceGPUV2 instance{};
    instance.translationAndScale = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    instance.binding = {
        0u,
        bodyIndex,
        bindingKind,
        MR_VISUAL_INSTANCE_CASTS_SHADOW |
            MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
            MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
    };
    instance.identity = {semanticId, instanceId, bodyIndex, stableId};
    instance.geometry = {primitiveIndex, 1u, 0u, 0u};
    pack.instances.push_back(instance);

    MRVisualPrimitiveGPUV2 primitive{};
    primitive.geometry = {
        geometry.firstIndex,
        geometry.indexCount,
        materialIndex,
        instanceIndex,
    };
    primitive.identity = {semanticId, instanceId, bodyIndex, stableId};
    primitive.boundsMinimum = f4(geometry.lower, 1.0);
    primitive.boundsMaximum = f4(geometry.upper, 1.0);
    pack.primitives.push_back(primitive);
    pack.symbolicBindings.push_back({
        std::move(node),
        std::move(link),
        instanceIndex,
        bodyIndex,
        static_cast<MRVisualBindingKind>(bindingKind),
    });
}

void addTube(
    VisualAssetPackV2& pack,
    const std::vector<Vec3>& points,
    const std::vector<double>& radii,
    const std::uint32_t radialSections,
    const std::uint32_t materialIndex,
    const std::uint32_t bindingKind,
    const std::uint32_t bodyIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    const std::uint32_t stableId,
    const std::string& node,
    const std::string& link,
    const bool capEnds = true
) {
    appendInstance(
        pack,
        appendTubeGeometry(
            pack,
            points,
            radii,
            radialSections,
            capEnds
        ),
        materialIndex,
        bindingKind,
        bodyIndex,
        semanticId,
        instanceId,
        stableId,
        node,
        link
    );
}

void addBox(
    VisualAssetPackV2& pack,
    const Vec3 center,
    const Vec3 halfExtent,
    const std::uint32_t materialIndex,
    const std::uint32_t bindingKind,
    const std::uint32_t bodyIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    const std::uint32_t stableId,
    const std::string& node,
    const std::string& link
) {
    appendInstance(
        pack,
        appendBoxGeometry(pack, center, halfExtent),
        materialIndex,
        bindingKind,
        bodyIndex,
        semanticId,
        instanceId,
        stableId,
        node,
        link
    );
}

void addTaperedBox(
    VisualAssetPackV2& pack,
    const Vec3 lowerCenter,
    const Vec3 upperCenter,
    const Vec3 lowerHalfExtent,
    const Vec3 upperHalfExtent,
    const std::uint32_t materialIndex,
    const std::uint32_t bindingKind,
    const std::uint32_t bodyIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    const std::uint32_t stableId,
    const std::string& node,
    const std::string& link
) {
    appendInstance(
        pack,
        appendTaperedBoxGeometry(
            pack,
            lowerCenter,
            upperCenter,
            lowerHalfExtent,
            upperHalfExtent
        ),
        materialIndex,
        bindingKind,
        bodyIndex,
        semanticId,
        instanceId,
        stableId,
        node,
        link
    );
}

std::vector<Vec3> subdivideThread(
    const DiscreteElasticRodState& state,
    const std::uint32_t subdivisions
) {
    std::vector<Vec3> result;
    result.reserve(
        (state.positions.size() - 1u) * subdivisions + 1u
    );
    for (std::size_t edge = 0u;
         edge + 1u < state.positions.size();
         ++edge) {
        const Vec3 a = fromArray(state.positions[edge]);
        const Vec3 b = fromArray(state.positions[edge + 1u]);
        for (std::uint32_t part = 0u; part < subdivisions; ++part) {
            const double fraction =
                static_cast<double>(part) /
                static_cast<double>(subdivisions);
            result.push_back(a * (1.0 - fraction) + b * fraction);
        }
    }
    result.push_back(fromArray(state.positions.back()));
    return result;
}

double needleRadiusAt(
    const CurvedSutureNeedleSpec& spec,
    const double arcPosition
) {
    const double baseRadius = spec.crossSectionRadiusM.value;
    if (arcPosition < spec.swageLengthM.value) {
        const double fraction =
            1.0 - arcPosition / spec.swageLengthM.value;
        return baseRadius *
            (1.0 + (spec.swageRadiusRatio - 1.0) * fraction);
    }
    const double tipStart =
        spec.arcLengthM.value - spec.tipTaperLengthM.value;
    if (arcPosition > tipStart) {
        const double remaining =
            (spec.arcLengthM.value - arcPosition) /
            spec.tipTaperLengthM.value;
        return baseRadius *
            (spec.tipRadiusRatio +
             (1.0 - spec.tipRadiusRatio) * remaining);
    }
    return baseRadius;
}

void validateStyle(const DvrkSutureVisualStyle& style) {
    if (style.needleArcSections < 32u ||
        style.needleArcSections > 512u ||
        style.needleRadialSections < 8u ||
        style.needleRadialSections > 64u ||
        style.threadSubsectionsPerEdge == 0u ||
        style.threadSubsectionsPerEdge > 16u ||
        style.threadRadialSections < 6u ||
        style.threadRadialSections > 32u ||
        style.instrumentRadialSections < 8u ||
        style.instrumentRadialSections > 64u) {
        throw std::invalid_argument(
            "dVRK suture visual style is outside fixed capacity"
        );
    }
}

} // namespace

DvrkSutureVisualAsset makeDvrkSutureVisualAsset(
    const CurvedSutureNeedleAsset& needle,
    const DiscreteElasticRodModel& threadModel,
    const DiscreteElasticRodState& threadState,
    const DvrkSutureVisualBindings& bindings,
    const DvrkSutureVisualStyle& style,
    const DvrkSutureVisualScene& scene
) {
    std::string rodReason;
    if (needle.rigid.shapes.empty() ||
        !threadModel.valid(&rodReason) ||
        threadState.positions.size() != threadModel.restPositions.size() ||
        threadState.velocities.size() != threadModel.restPositions.size() ||
        threadState.twists.size() != threadModel.restTwists.size() ||
        threadState.twistRates.size() != threadModel.restTwists.size() ||
        bindings.needleBodyIndex == MR_INVALID_INDEX ||
        scene.tissuePositions.empty() != scene.tissueTriangles.empty()) {
        throw std::invalid_argument(
            "dVRK suture visual inputs are invalid: " + rodReason
        );
    }
    validateStyle(style);

    DvrkSutureVisualAsset result;
    VisualAssetPackV2& pack = result.pack;
    pack.id = "dvrk_lnd_gs21_suture_pickup_visual_v1";
    pack.sourceUri =
        "numi://orbit-surgical/6e47534+jhu-dvrk/53a401d/gs21";
    pack.sourceContentHash =
        "source:orbit-6e47534+jhu-53a401d+gs21-analytic";
    pack.license = "NOASSERTION";
    pack.preprocessingProvenance =
        "makeDvrkSutureVisualAsset/physics-bound-procedural-v2";
    pack.materials = {
        material({0.44f, 0.49f, 0.54f, 1.0f}, 0.27f, 0.92f,
                 0.16f, 0.16f, 1u),
        material({0.16f, 0.19f, 0.23f, 1.0f}, 0.31f, 0.88f,
                 0.10f, 0.20f, 2u),
        material({0.72f, 0.75f, 0.78f, 1.0f}, 0.075f, 1.0f,
                 0.36f, 0.055f, 3u),
        material({0.012f, 0.13f, 0.52f, 1.0f}, 0.29f, 0.04f,
                 0.48f, 0.12f, 4u),
        material({0.012f, 0.145f, 0.125f, 1.0f}, 0.82f, 0.0f,
                 0.015f, 0.38f, 5u),
        material({0.025f, 0.032f, 0.038f, 1.0f}, 0.56f, 0.18f,
                 0.04f, 0.30f, 6u),
        material({0.43f, 0.13f, 0.12f, 1.0f}, 0.86f, 0.0f,
                 0.03f, 0.32f, 7u),
        material({0.16f, 0.015f, 0.018f, 1.0f}, 0.69f, 0.0f,
                 0.02f, 0.28f, 8u),
    };

    constexpr std::uint32_t instrumentSemantic = 101u;
    constexpr std::uint32_t needleSemantic = 201u;
    constexpr std::uint32_t threadSemantic = 202u;
    constexpr std::uint32_t fieldSemantic = 301u;
    std::uint32_t stableId = 1u;

    const auto cylinder = [&](
        const std::uint32_t body,
        const Vec3 a,
        const Vec3 b,
        const double radiusA,
        const double radiusB,
        const std::uint32_t materialIndex,
        const std::string& node
    ) {
        addTube(
            pack,
            {a, b},
            {radiusA, radiusB},
            style.instrumentRadialSections,
            materialIndex,
            MR_VISUAL_BINDING_ARTICULATED_LINK,
            body,
            instrumentSemantic,
            1000u + body,
            stableId++,
            node,
            node
        );
    };
    cylinder(
        bindings.shaftBodyIndex,
        {0.0, 0.0, -0.410},
        {0.0, 0.0, 0.012},
        0.00435,
        0.00435,
        0u,
        "psm_tool_shaft"
    );
    cylinder(
        bindings.shaftBodyIndex,
        {0.0, 0.0, -0.030},
        {0.0, 0.0, 0.011},
        0.0072,
        0.0062,
        1u,
        "psm_distal_shaft_collar"
    );
    cylinder(
        bindings.shaftBodyIndex,
        {0.0, 0.0, -0.0275},
        {0.0, 0.0, -0.0225},
        0.00735,
        0.00735,
        5u,
        "psm_distal_shaft_grip_band"
    );
    cylinder(
        bindings.wristPitchBodyIndex,
        {0.0, 0.0, -0.004},
        {0.0, 0.0, 0.012},
        0.0068,
        0.0058,
        0u,
        "psm_wrist_pitch"
    );
    cylinder(
        bindings.wristYawBodyIndex,
        {0.0, 0.0, -0.004},
        {0.0, 0.0, 0.013},
        0.0058,
        0.0048,
        0u,
        "psm_wrist_yaw"
    );
    cylinder(
        bindings.wristYawBodyIndex,
        {-0.0065, 0.0, 0.0035},
        {0.0065, 0.0, 0.0035},
        0.00115,
        0.00115,
        2u,
        "psm_wrist_cross_pin"
    );
    cylinder(
        bindings.toolBodyIndex,
        {0.0, 0.0, -0.002},
        {0.0, 0.0, 0.021},
        0.0052,
        0.0030,
        1u,
        "large_needle_driver_clevis"
    );
    addBox(
        pack,
        {0.0040, 0.0, 0.0140},
        {0.00115, 0.0030, 0.0070},
        1u,
        MR_VISUAL_BINDING_ARTICULATED_LINK,
        bindings.toolBodyIndex,
        instrumentSemantic,
        1000u + bindings.toolBodyIndex,
        stableId++,
        "large_needle_driver_clevis_plate_a",
        "large_needle_driver_clevis_plate_a"
    );
    addBox(
        pack,
        {-0.0040, 0.0, 0.0140},
        {0.00115, 0.0030, 0.0070},
        1u,
        MR_VISUAL_BINDING_ARTICULATED_LINK,
        bindings.toolBodyIndex,
        instrumentSemantic,
        1000u + bindings.toolBodyIndex,
        stableId++,
        "large_needle_driver_clevis_plate_b",
        "large_needle_driver_clevis_plate_b"
    );
    cylinder(
        bindings.toolBodyIndex,
        {-0.0060, 0.0, 0.0140},
        {0.0060, 0.0, 0.0140},
        0.00125,
        0.00125,
        0u,
        "large_needle_driver_pivot_pin"
    );
    const auto jaw = [&](
        const std::uint32_t body,
        const double sign,
        const std::string& name
    ) {
        addTaperedBox(
            pack,
            {sign * 0.00135, 0.0, 0.0010},
            {sign * 0.00042, 0.0, 0.0250},
            {0.00155, 0.00215, 0.0},
            {0.00062, 0.00105, 0.0},
            1u,
            MR_VISUAL_BINDING_ARTICULATED_LINK,
            body,
            instrumentSemantic,
            1000u + body,
            stableId++,
            name,
            name
        );
        addBox(
            pack,
            {sign * 0.00016, 0.0, 0.0216},
            {0.00020, 0.00125, 0.0030},
            5u,
            MR_VISUAL_BINDING_ARTICULATED_LINK,
            body,
            instrumentSemantic,
            1000u + body,
            stableId++,
            name + "_tungsten_carbide_insert",
            name + "_tungsten_carbide_insert"
        );
    };
    jaw(
        bindings.jawABodyIndex,
        1.0,
        "large_needle_driver_jaw_a"
    );
    jaw(
        bindings.jawBBodyIndex,
        -1.0,
        "large_needle_driver_jaw_b"
    );
    for (std::uint32_t tooth = 0u; tooth < 5u; ++tooth) {
        const double z = 0.0193 + 0.00115 * tooth;
        cylinder(
            bindings.jawABodyIndex,
            {0.00004, -0.00110, z},
            {0.00004, 0.00110, z},
            0.00012,
            0.00012,
            0u,
            "large_needle_driver_jaw_a_tooth_" +
                std::to_string(tooth)
        );
        cylinder(
            bindings.jawBBodyIndex,
            {-0.00004, -0.00110, z},
            {-0.00004, 0.00110, z},
            0.00012,
            0.00012,
            0u,
            "large_needle_driver_jaw_b_tooth_" +
                std::to_string(tooth)
        );
    }

    if (scene.hasSecondaryInstrument) {
        DvrkSutureVisualBindings secondaryBindings =
            scene.secondaryInstrument;
        secondaryBindings.needleBodyIndex = bindings.needleBodyIndex;
        const DvrkSutureVisualAsset secondary =
            makeDvrkSutureVisualAsset(
                needle,
                threadModel,
                threadState,
                secondaryBindings,
                style,
                {}
            );
        for (std::size_t instanceIndex = 0u;
             instanceIndex < secondary.pack.instances.size();
             ++instanceIndex) {
            const MRVisualInstanceGPUV2& instance =
                secondary.pack.instances[instanceIndex];
            if (instance.identity.x != instrumentSemantic) {
                continue;
            }
            const auto symbolic = std::find_if(
                secondary.pack.symbolicBindings.begin(),
                secondary.pack.symbolicBindings.end(),
                [&](const VisualSymbolicBindingV2& binding) {
                    return binding.instanceIndex == instanceIndex;
                }
            );
            const std::string node = symbolic ==
                    secondary.pack.symbolicBindings.end()
                ? "receiver_instrument"
                : "receiver_" + symbolic->node;
            const std::string link = symbolic ==
                    secondary.pack.symbolicBindings.end() ||
                    symbolic->link.empty()
                ? std::string{}
                : "receiver_" + symbolic->link;
            for (std::uint32_t localPrimitive = 0u;
                 localPrimitive < instance.geometry.y;
                 ++localPrimitive) {
                const std::uint32_t primitiveIndex =
                    instance.geometry.x + localPrimitive;
                if (primitiveIndex >= secondary.pack.primitives.size()) {
                    throw std::logic_error(
                        "secondary surgical instrument primitive is missing"
                    );
                }
                const MRVisualPrimitiveGPUV2& primitive =
                    secondary.pack.primitives[primitiveIndex];
                appendInstance(
                    pack,
                    appendPrimitiveGeometryCopy(
                        pack,
                        secondary.pack,
                        primitive
                    ),
                    primitive.geometry.z,
                    instance.binding.z,
                    instance.binding.y,
                    instrumentSemantic,
                    10000u + instance.identity.y,
                    stableId++,
                    node,
                    link
                );
            }
        }
    }

    const double centerlineRadius =
        needle.spec.arcLengthM.value /
        needle.spec.arcAngleRad.value;
    const double startAngle = -0.5 * needle.spec.arcAngleRad.value;
    std::vector<Vec3> needlePoints;
    std::vector<double> needleRadii;
    needlePoints.reserve(style.needleArcSections + 1u);
    needleRadii.reserve(style.needleArcSections + 1u);
    for (std::uint32_t section = 0u;
         section <= style.needleArcSections;
         ++section) {
        const double fraction =
            static_cast<double>(section) /
            static_cast<double>(style.needleArcSections);
        const double angle = startAngle +
            needle.spec.arcAngleRad.value * fraction;
        needlePoints.push_back({
            centerlineRadius * std::cos(angle) -
                needle.rigid.geometryCenterOfMassM[0],
            centerlineRadius * std::sin(angle) -
                needle.rigid.geometryCenterOfMassM[1],
            -needle.rigid.geometryCenterOfMassM[2],
        });
        needleRadii.push_back(needleRadiusAt(
            needle.spec,
            needle.spec.arcLengthM.value * fraction
        ));
    }
    const std::size_t needleIndexBegin = pack.indices.size();
    addTube(
        pack,
        needlePoints,
        needleRadii,
        style.needleRadialSections,
        2u,
        MR_VISUAL_BINDING_RIGID_BODY,
        bindings.needleBodyIndex,
        needleSemantic,
        2001u,
        stableId++,
        "gs21_curved_needle",
        "gs21_curved_needle"
    );
    result.metrics.needleTriangleCount =
        static_cast<std::uint32_t>(
            (pack.indices.size() - needleIndexBegin) / 3u
        );

    const std::vector<Vec3> threadPoints = subdivideThread(
        threadState,
        style.threadSubsectionsPerEdge
    );
    const std::size_t threadIndexBegin = pack.indices.size();
    addTube(
        pack,
        threadPoints,
        std::vector<double>(threadPoints.size(), threadModel.radius),
        style.threadRadialSections,
        3u,
        MR_VISUAL_BINDING_WORLD,
        MR_INVALID_INDEX,
        threadSemantic,
        2002u,
        stableId++,
        "suture_thread",
        ""
    );
    result.metrics.threadTriangleCount =
        static_cast<std::uint32_t>(
            (pack.indices.size() - threadIndexBegin) / 3u
        );
    for (std::size_t index = 1u; index < threadPoints.size(); ++index) {
        result.metrics.threadCenterlineLengthM += norm(
            threadPoints[index] - threadPoints[index - 1u]
        );
    }

    Vec3 fieldCenter{};
    double minimumThreadZ = std::numeric_limits<double>::infinity();
    for (const Vec3 point : threadPoints) {
        fieldCenter = fieldCenter + point;
        minimumThreadZ = std::min(minimumThreadZ, point.z);
    }
    fieldCenter = fieldCenter * (1.0 / threadPoints.size());
    fieldCenter.z = minimumThreadZ - 0.006;
    addBox(
        pack,
        fieldCenter,
        {0.115, 0.085, 0.0025},
        4u,
        MR_VISUAL_BINDING_WORLD,
        MR_INVALID_INDEX,
        fieldSemantic,
        3001u,
        stableId++,
        "surgical_field",
        ""
    );
    if (!scene.tissuePositions.empty()) {
        const std::size_t tissueIndexBegin = pack.indices.size();
        appendInstance(
            pack,
            appendSurfaceGeometry(
                pack,
                scene.tissuePositions,
                scene.tissueTriangles,
                scene.tissueTranslationM
            ),
            6u,
            MR_VISUAL_BINDING_WORLD,
            MR_INVALID_INDEX,
            fieldSemantic,
            3002u,
            stableId++,
            "porcine_jejunum_enterotomy_coupon",
            ""
        );
        result.metrics.tissueTriangleCount =
            static_cast<std::uint32_t>(
                (pack.indices.size() - tissueIndexBegin) / 3u
            );
    } else {
        const Vec3 tissueCenter{
            fieldCenter.x,
            fieldCenter.y,
            minimumThreadZ - 0.002,
        };
        addBox(
            pack,
            tissueCenter,
            {0.045, 0.032, 0.0015},
            6u,
            MR_VISUAL_BINDING_WORLD,
            MR_INVALID_INDEX,
            fieldSemantic,
            3002u,
            stableId++,
            "suture_practice_tissue",
            ""
        );
        const double incisionZ = minimumThreadZ - 0.00018;
        addTube(
            pack,
            {
                {tissueCenter.x - 0.024, tissueCenter.y, incisionZ},
                {tissueCenter.x + 0.024, tissueCenter.y, incisionZ},
            },
            {0.00028, 0.00028},
            12u,
            7u,
            MR_VISUAL_BINDING_WORLD,
            MR_INVALID_INDEX,
            fieldSemantic,
            3003u,
            stableId++,
            "suture_practice_incision",
            ""
        );
        result.metrics.tissueTriangleCount = 12u;
    }

    pack.contentHash = computeVisualAssetPackContentHash(pack);
    std::string reason;
    if (!pack.valid(&reason)) {
        throw std::logic_error(
            "internal dVRK suture visual pack is invalid: " + reason
        );
    }
    result.metrics.vertexCount =
        static_cast<std::uint32_t>(pack.vertices.size());
    result.metrics.triangleCount =
        static_cast<std::uint32_t>(pack.indices.size() / 3u);
    result.metrics.instanceCount =
        static_cast<std::uint32_t>(pack.instances.size());
    return result;
}

} // namespace metalrobo
