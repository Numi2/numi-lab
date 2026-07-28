#include "metalrobo/GeometryCooker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator/(const Vec3 value, const double scale) {
    return value * (1.0 / scale);
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double lengthSquared(const Vec3 value) {
    return dot(value, value);
}

Vec3 minimum(const Vec3 a, const Vec3 b) {
    return {
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z),
    };
}

Vec3 maximum(const Vec3 a, const Vec3 b) {
    return {
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z),
    };
}

double component(const Vec3 value, const std::uint32_t axis) {
    return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
}

mr_float4 packed(const Vec3 value, const float w = 0.0f) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
}

struct Bounds {
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

    void include(const Vec3 point) {
        lower = minimum(lower, point);
        upper = maximum(upper, point);
    }

    void include(const Bounds& other) {
        lower = minimum(lower, other.lower);
        upper = maximum(upper, other.upper);
    }

    [[nodiscard]] Vec3 centroid() const {
        return (lower + upper) * 0.5;
    }

    [[nodiscard]] double surfaceArea() const {
        const Vec3 extent = maximum(
            upper - lower,
            {0.0, 0.0, 0.0}
        );
        return 2.0 * (
            extent.x * extent.y +
            extent.y * extent.z +
            extent.z * extent.x
        );
    }
};

struct Triangle {
    std::array<std::uint32_t, 3u> vertices{};
    std::uint32_t stableFeature = 0u;
    std::uint32_t material = 0u;
    Bounds bounds{};
};

GeometryCookResult failure(
    GeometryCookResult result,
    const GeometryCookStatus status,
    std::string message
) {
    result.status = status;
    result.message = std::move(message);
    return result;
}

bool validConfig(const GeometryCookConfig& config) {
    return
        std::isfinite(config.weldTolerance) &&
        std::isfinite(config.minimumTriangleArea) &&
        config.weldTolerance > 0.0 &&
        config.minimumTriangleArea > 0.0 &&
        config.meshLeafTriangleCapacity > 0u &&
        config.meshLeafTriangleCapacity <=
            MR_MESH_BVH_LEAF_COUNT_MASK;
}

bool finiteVertex(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::abs(value.x) <=
            MR_MAX_COLLISION_INPUT_COORDINATE &&
        std::abs(value.y) <=
            MR_MAX_COLLISION_INPUT_COORDINATE &&
        std::abs(value.z) <=
            MR_MAX_COLLISION_INPUT_COORDINATE;
}

using Cell = std::array<std::int64_t, 3u>;

Cell cellFor(const Vec3 point, const double inverseTolerance) {
    return {
        static_cast<std::int64_t>(
            std::floor(point.x * inverseTolerance)
        ),
        static_cast<std::int64_t>(
            std::floor(point.y * inverseTolerance)
        ),
        static_cast<std::int64_t>(
            std::floor(point.z * inverseTolerance)
        ),
    };
}

bool weldVertices(
    const std::span<const mr_float4> input,
    const double tolerance,
    std::vector<Vec3>& vertices,
    std::vector<std::uint32_t>& remap
) {
    const double inverseTolerance = 1.0 / tolerance;
    const double toleranceSquared = tolerance * tolerance;
    std::map<Cell, std::vector<std::uint32_t>> cells;
    remap.resize(input.size());
    for (std::size_t inputIndex = 0u;
         inputIndex < input.size();
         ++inputIndex) {
        if (!finiteVertex(input[inputIndex])) {
            return false;
        }
        const Vec3 point{
            input[inputIndex].x,
            input[inputIndex].y,
            input[inputIndex].z,
        };
        const Cell cell = cellFor(point, inverseTolerance);
        std::uint32_t selected = MR_INVALID_INDEX;
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
            for (std::int64_t dy = -1; dy <= 1; ++dy) {
                for (std::int64_t dx = -1; dx <= 1; ++dx) {
                    const Cell neighbor{
                        cell[0] + dx,
                        cell[1] + dy,
                        cell[2] + dz,
                    };
                    const auto found = cells.find(neighbor);
                    if (found == cells.end()) {
                        continue;
                    }
                    for (const std::uint32_t candidate :
                         found->second) {
                        if (lengthSquared(
                                vertices[candidate] - point
                            ) <= toleranceSquared) {
                            selected = std::min(
                                selected,
                                candidate
                            );
                        }
                    }
                }
            }
        }
        if (selected == MR_INVALID_INDEX) {
            if (vertices.size() >=
                std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            selected =
                static_cast<std::uint32_t>(vertices.size());
            vertices.push_back(point);
            cells[cell].push_back(selected);
        }
        remap[inputIndex] = selected;
    }
    return true;
}

bool cleanTriangles(
    const std::span<const std::uint32_t> inputIndices,
    const std::span<const std::uint32_t> inputMaterials,
    const std::span<const std::uint32_t> remap,
    const std::span<const Vec3> vertices,
    const GeometryCookConfig& config,
    std::vector<Triangle>& triangles,
    std::uint32_t& removed
) {
    if (inputIndices.size() % 3u != 0u ||
        (!inputMaterials.empty() &&
         inputMaterials.size() != inputIndices.size() / 3u)) {
        return false;
    }
    std::set<std::array<std::uint32_t, 3u>> duplicates;
    const double minimumCrossSquared =
        4.0 * config.minimumTriangleArea *
        config.minimumTriangleArea;
    for (std::size_t triangleIndex = 0u;
         triangleIndex < inputIndices.size() / 3u;
         ++triangleIndex) {
        std::array<std::uint32_t, 3u> mapped{};
        bool valid = true;
        for (std::uint32_t corner = 0u;
             corner < 3u;
             ++corner) {
            const std::uint32_t source =
                inputIndices[3u * triangleIndex + corner];
            if (source >= remap.size()) {
                valid = false;
                break;
            }
            mapped[corner] = remap[source];
        }
        if (!valid) {
            return false;
        }
        const Vec3 a = vertices[mapped[0]];
        const Vec3 b = vertices[mapped[1]];
        const Vec3 c = vertices[mapped[2]];
        const bool degenerate =
            mapped[0] == mapped[1] ||
            mapped[1] == mapped[2] ||
            mapped[2] == mapped[0] ||
            lengthSquared(cross(b - a, c - a)) <=
                minimumCrossSquared;
        std::array<std::uint32_t, 3u> canonical = mapped;
        std::sort(canonical.begin(), canonical.end());
        const bool duplicate =
            config.removeDuplicateTriangles &&
            !duplicates.insert(canonical).second;
        if (degenerate || duplicate) {
            ++removed;
            continue;
        }
        Triangle triangle{};
        triangle.vertices = mapped;
        triangle.stableFeature =
            static_cast<std::uint32_t>(triangleIndex);
        triangle.material = inputMaterials.empty()
            ? 0u
            : inputMaterials[triangleIndex];
        triangle.bounds.include(a);
        triangle.bounds.include(b);
        triangle.bounds.include(c);
        triangles.push_back(triangle);
    }
    return true;
}

Bounds vertexBounds(const std::span<const Vec3> vertices) {
    Bounds bounds;
    for (const Vec3 vertex : vertices) {
        bounds.include(vertex);
    }
    return bounds;
}

bool checkedArenaAppend(
    const std::size_t current,
    const std::size_t added
) {
    return current <=
        std::numeric_limits<std::uint32_t>::max() &&
        added <=
            std::numeric_limits<std::uint32_t>::max() -
                current;
}

void setComponent(
    mr_uint4& value,
    const std::uint32_t componentIndex,
    const std::uint32_t componentValue
) {
    if (componentIndex == 0u) {
        value.x = componentValue;
    } else if (componentIndex == 1u) {
        value.y = componentValue;
    } else if (componentIndex == 2u) {
        value.z = componentValue;
    } else {
        value.w = componentValue;
    }
}

std::uint32_t quantize(
    const double value,
    const double lower,
    const double upper,
    const bool roundUp
) {
    const double extent = upper - lower;
    if (!(extent > 0.0)) {
        return roundUp ? 65535u : 0u;
    }
    const double normalized =
        std::clamp((value - lower) / extent, 0.0, 1.0);
    const double scaled = normalized * 65535.0;
    const double rounded =
        roundUp ? std::ceil(scaled) : std::floor(scaled);
    return static_cast<std::uint32_t>(
        std::clamp(rounded, 0.0, 65535.0)
    );
}

struct BVHChild {
    Bounds bounds{};
    bool valid = false;
    bool leaf = false;
    std::uint32_t first = 0u;
    std::uint32_t count = 0u;
    std::uint32_t node = 0u;
};

struct BVHNode {
    std::array<BVHChild, MR_MESH_BVH_BRANCHING> children{};
};

Bounds rangeBounds(
    const std::span<const Triangle> triangles,
    const std::span<const std::uint32_t> order,
    const std::size_t begin,
    const std::size_t end
) {
    Bounds bounds;
    for (std::size_t index = begin; index < end; ++index) {
        bounds.include(triangles[order[index]].bounds);
    }
    return bounds;
}

double splitCost(
    const std::span<const Triangle> triangles,
    const std::span<const std::uint32_t> order,
    const std::size_t begin,
    const std::size_t end
) {
    const std::size_t count = end - begin;
    const std::size_t groupCount = std::min<std::size_t>(
        MR_MESH_BVH_BRANCHING,
        (count + 3u) / 4u
    );
    double cost = 0.0;
    for (std::size_t group = 0u;
         group < groupCount;
         ++group) {
        const std::size_t groupBegin =
            begin + count * group / groupCount;
        const std::size_t groupEnd =
            begin + count * (group + 1u) / groupCount;
        cost += rangeBounds(
            triangles,
            order,
            groupBegin,
            groupEnd
        ).surfaceArea() *
            static_cast<double>(groupEnd - groupBegin);
    }
    return cost;
}

std::uint32_t buildBVHNode(
    const std::span<const Triangle> triangles,
    std::vector<std::uint32_t>& order,
    const std::size_t begin,
    const std::size_t end,
    const std::uint32_t leafCapacity,
    std::vector<BVHNode>& nodes
) {
    const std::uint32_t nodeIndex =
        static_cast<std::uint32_t>(nodes.size());
    nodes.emplace_back();
    const std::size_t count = end - begin;
    if (count <= leafCapacity) {
        BVHChild child{};
        child.valid = true;
        child.leaf = true;
        child.first = static_cast<std::uint32_t>(begin);
        child.count = static_cast<std::uint32_t>(count);
        child.bounds = rangeBounds(
            triangles,
            order,
            begin,
            end
        );
        nodes[nodeIndex].children[0] = child;
        return nodeIndex;
    }

    std::uint32_t bestAxis = 0u;
    double bestCost = std::numeric_limits<double>::infinity();
    std::vector<std::uint32_t> bestOrder;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        std::vector<std::uint32_t> candidate(
            order.begin() + static_cast<std::ptrdiff_t>(begin),
            order.begin() + static_cast<std::ptrdiff_t>(end)
        );
        std::stable_sort(
            candidate.begin(),
            candidate.end(),
            [&](const std::uint32_t left,
                const std::uint32_t right) {
                const double leftValue = component(
                    triangles[left].bounds.centroid(),
                    axis
                );
                const double rightValue = component(
                    triangles[right].bounds.centroid(),
                    axis
                );
                return leftValue < rightValue ||
                    (leftValue == rightValue &&
                     triangles[left].stableFeature <
                         triangles[right].stableFeature);
            }
        );
        std::copy(
            candidate.begin(),
            candidate.end(),
            order.begin() + static_cast<std::ptrdiff_t>(begin)
        );
        const double cost = splitCost(
            triangles,
            order,
            begin,
            end
        );
        if (cost < bestCost) {
            bestCost = cost;
            bestAxis = axis;
            bestOrder = std::move(candidate);
        }
    }
    (void)bestAxis;
    std::copy(
        bestOrder.begin(),
        bestOrder.end(),
        order.begin() + static_cast<std::ptrdiff_t>(begin)
    );

    const std::size_t groupCount = std::min<std::size_t>(
        MR_MESH_BVH_BRANCHING,
        (count + leafCapacity - 1u) / leafCapacity
    );
    for (std::size_t group = 0u;
         group < groupCount;
         ++group) {
        const std::size_t groupBegin =
            begin + count * group / groupCount;
        const std::size_t groupEnd =
            begin + count * (group + 1u) / groupCount;
        BVHChild child{};
        child.valid = true;
        child.bounds = rangeBounds(
            triangles,
            order,
            groupBegin,
            groupEnd
        );
        if (groupEnd - groupBegin <= leafCapacity) {
            child.leaf = true;
            child.first =
                static_cast<std::uint32_t>(groupBegin);
            child.count =
                static_cast<std::uint32_t>(
                    groupEnd - groupBegin
                );
        } else {
            child.node = buildBVHNode(
                triangles,
                order,
                groupBegin,
                groupEnd,
                leafCapacity,
                nodes
            );
        }
        nodes[nodeIndex].children[group] = child;
    }
    return nodeIndex;
}

void encodeBVHNode(
    const std::uint32_t nodeIndex,
    const std::uint32_t nodeEscapeCursor,
    const Bounds& geometryBounds,
    const std::span<const BVHNode> source,
    std::span<MRMeshBVHNodeGPU> destination
) {
    MRMeshBVHNodeGPU encoded{};
    const BVHNode& node = source[nodeIndex];
    for (std::uint32_t slot = 0u;
         slot < MR_MESH_BVH_BRANCHING;
         ++slot) {
        const std::uint32_t escape =
            slot + 1u < MR_MESH_BVH_BRANCHING
            ? nodeIndex * MR_MESH_BVH_BRANCHING +
                slot + 1u
            : nodeEscapeCursor;
        const BVHChild& child = node.children[slot];
        setComponent(
            encoded.childIndices,
            slot,
            child.valid
                ? (child.leaf ? child.first : child.node)
                : MR_INVALID_INDEX
        );
        std::uint32_t meta =
            (std::min(
                 escape,
                 static_cast<std::uint32_t>(
                     MR_MESH_BVH_INVALID_ESCAPE
                 )
             ) << MR_MESH_BVH_ESCAPE_SHIFT);
        if (child.valid && child.leaf) {
            meta |= MR_MESH_BVH_LEAF_BIT |
                (child.count &
                 MR_MESH_BVH_LEAF_COUNT_MASK);
        }
        setComponent(encoded.childMeta, slot, meta);
        if (!child.valid) {
            continue;
        }
        const double inflation =
            8.0 * std::numeric_limits<float>::epsilon() *
            (std::max({
                 std::abs(child.bounds.lower.x),
                 std::abs(child.bounds.lower.y),
                 std::abs(child.bounds.lower.z),
                 std::abs(child.bounds.upper.x),
                 std::abs(child.bounds.upper.y),
                 std::abs(child.bounds.upper.z),
                 1.0
             }));
        encoded.quantizedLower[slot] = {
            quantize(
                child.bounds.lower.x - inflation,
                geometryBounds.lower.x,
                geometryBounds.upper.x,
                false
            ),
            quantize(
                child.bounds.lower.y - inflation,
                geometryBounds.lower.y,
                geometryBounds.upper.y,
                false
            ),
            quantize(
                child.bounds.lower.z - inflation,
                geometryBounds.lower.z,
                geometryBounds.upper.z,
                false
            ),
            0u,
        };
        encoded.quantizedUpper[slot] = {
            quantize(
                child.bounds.upper.x + inflation,
                geometryBounds.lower.x,
                geometryBounds.upper.x,
                true
            ),
            quantize(
                child.bounds.upper.y + inflation,
                geometryBounds.lower.y,
                geometryBounds.upper.y,
                true
            ),
            quantize(
                child.bounds.upper.z + inflation,
                geometryBounds.lower.z,
                geometryBounds.upper.z,
                true
            ),
            0u,
        };
        if (!child.leaf) {
            encodeBVHNode(
                child.node,
                escape,
                geometryBounds,
                source,
                destination
            );
        }
    }
    destination[nodeIndex] = encoded;
}

} // namespace

GeometryCookResult cookConvexGeometry(
    EngineModel& model,
    const std::span<const mr_float4> inputVertices,
    const std::span<const std::uint32_t> inputIndices,
    const GeometryCookConfig& config
) {
    GeometryCookResult result{};
    result.inputVertexCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            inputVertices.size(),
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.inputTriangleCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            inputIndices.size() / 3u,
            std::numeric_limits<std::uint32_t>::max()
        ));
    if (!validConfig(config) ||
        inputVertices.size() < 4u ||
        inputIndices.size() < 12u ||
        inputVertices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(
            std::move(result),
            GeometryCookStatus::invalidInput,
            "convex input or cooker configuration is invalid"
        );
    }

    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> remap;
    if (!weldVertices(
            inputVertices,
            config.weldTolerance,
            vertices,
            remap
        )) {
        return failure(
            std::move(result),
            GeometryCookStatus::invalidInput,
            "convex vertices are non-finite or exceed capacity"
        );
    }
    std::vector<Triangle> triangles;
    if (!cleanTriangles(
            inputIndices,
            {},
            remap,
            vertices,
            config,
            triangles,
            result.removedDegenerateTriangles
        ) ||
        vertices.size() < 4u ||
        triangles.size() < 4u) {
        return failure(
            std::move(result),
            GeometryCookStatus::degenerateGeometry,
            "convex surface is degenerate"
        );
    }

    Vec3 centroid{};
    for (const Vec3 vertex : vertices) {
        centroid = centroid + vertex;
    }
    centroid = centroid /
        static_cast<double>(vertices.size());
    std::vector<MRConvexFaceGPU> faces;
    std::vector<MRConvexHalfEdgeGPU> halfEdges;
    faces.reserve(triangles.size());
    halfEdges.resize(3u * triangles.size());
    for (std::size_t faceIndex = 0u;
         faceIndex < triangles.size();
         ++faceIndex) {
        Triangle& triangle = triangles[faceIndex];
        Vec3 a = vertices[triangle.vertices[0]];
        Vec3 b = vertices[triangle.vertices[1]];
        Vec3 c = vertices[triangle.vertices[2]];
        Vec3 normal = cross(b - a, c - a);
        if (dot(
                normal,
                (a + b + c) / 3.0 - centroid
            ) < 0.0) {
            std::swap(
                triangle.vertices[1],
                triangle.vertices[2]
            );
            b = vertices[triangle.vertices[1]];
            c = vertices[triangle.vertices[2]];
            normal = cross(b - a, c - a);
        }
        const double normalLength =
            std::sqrt(lengthSquared(normal));
        if (!(normalLength > 0.0)) {
            return failure(
                std::move(result),
                GeometryCookStatus::degenerateGeometry,
                "convex face has zero area"
            );
        }
        normal = normal / normalLength;
        const double offset = dot(normal, a);
        const double tolerance =
            8.0 * config.weldTolerance;
        for (const Vec3 vertex : vertices) {
            if (dot(normal, vertex) > offset + tolerance) {
                return failure(
                    std::move(result),
                    GeometryCookStatus::nonConvex,
                    "authored surface is not convex"
                );
            }
        }
        MRConvexFaceGPU face{};
        face.plane = packed(
            normal,
            static_cast<float>(offset)
        );
        face.firstHalfEdge =
            static_cast<std::uint32_t>(
                model.convexHalfEdges.size() +
                3u * faceIndex
            );
        face.halfEdgeCount = 3u;
        face.featureKey = triangle.stableFeature;
        faces.push_back(face);
    }

    using DirectedEdge =
        std::pair<std::uint32_t, std::uint32_t>;
    std::map<DirectedEdge, std::uint32_t> directed;
    for (std::size_t faceIndex = 0u;
         faceIndex < triangles.size();
         ++faceIndex) {
        for (std::uint32_t edge = 0u; edge < 3u; ++edge) {
            const std::uint32_t localIndex =
                static_cast<std::uint32_t>(
                    3u * faceIndex + edge
                );
            const std::uint32_t origin =
                triangles[faceIndex].vertices[edge];
            const std::uint32_t destination =
                triangles[faceIndex].vertices[
                    (edge + 1u) % 3u
                ];
            if (!directed.emplace(
                    DirectedEdge{origin, destination},
                    localIndex
                ).second) {
                return failure(
                    std::move(result),
                    GeometryCookStatus::nonManifold,
                    "convex contains a duplicated directed edge"
                );
            }
            MRConvexHalfEdgeGPU halfEdge{};
            halfEdge.originVertex =
                static_cast<std::uint32_t>(
                    model.geometryVertices.size()
                ) + origin;
            halfEdge.nextHalfEdge =
                static_cast<std::uint32_t>(
                    model.convexHalfEdges.size() +
                    3u * faceIndex +
                    (edge + 1u) % 3u
                );
            halfEdge.faceIndex =
                static_cast<std::uint32_t>(
                    model.convexFaces.size() +
                    faceIndex
                );
            halfEdges[localIndex] = halfEdge;
        }
    }
    for (const auto& [edge, localIndex] : directed) {
        const auto twin = directed.find({
            edge.second,
            edge.first,
        });
        if (twin == directed.end()) {
            return failure(
                std::move(result),
                GeometryCookStatus::nonManifold,
                "convex surface is not closed"
            );
        }
        halfEdges[localIndex].twinHalfEdge =
            static_cast<std::uint32_t>(
                model.convexHalfEdges.size()
            ) + twin->second;
    }

    if (!checkedArenaAppend(
            model.geometryVertices.size(),
            vertices.size()
        ) ||
        !checkedArenaAppend(
            model.geometryIndices.size(),
            3u * triangles.size()
        ) ||
        !checkedArenaAppend(
            model.convexFaces.size(),
            faces.size()
        ) ||
        !checkedArenaAppend(
            model.convexHalfEdges.size(),
            halfEdges.size()
        ) ||
        model.geometryHeaders.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(
            std::move(result),
            GeometryCookStatus::capacityOverflow,
            "convex arenas exceed the 32-bit cooked ABI"
        );
    }

    const Bounds bounds = vertexBounds(vertices);
    Bounds quantizationBounds = bounds;
    const double boundScale = std::max({
        std::abs(bounds.lower.x),
        std::abs(bounds.lower.y),
        std::abs(bounds.lower.z),
        std::abs(bounds.upper.x),
        std::abs(bounds.upper.y),
        std::abs(bounds.upper.z),
        1.0,
    });
    const double boundsInflation = std::max(
        config.weldTolerance,
        16.0 * std::numeric_limits<float>::epsilon() *
            boundScale
    );
    const Vec3 inflation{
        boundsInflation,
        boundsInflation,
        boundsInflation,
    };
    quantizationBounds.lower =
        quantizationBounds.lower - inflation;
    quantizationBounds.upper =
        quantizationBounds.upper + inflation;
    MRGeometryHeaderGPU header{};
    header.kind = MR_GEOMETRY_CONVEX;
    header.flags =
        MR_GEOMETRY_FLAG_CLOSED |
        MR_GEOMETRY_FLAG_CONVEX;
    header.vertexOffset =
        static_cast<std::uint32_t>(
            model.geometryVertices.size()
        );
    header.vertexCount =
        static_cast<std::uint32_t>(vertices.size());
    header.indexOffset =
        static_cast<std::uint32_t>(
            model.geometryIndices.size()
        );
    header.indexCount =
        static_cast<std::uint32_t>(
            3u * triangles.size()
        );
    header.faceOffset =
        static_cast<std::uint32_t>(
            model.convexFaces.size()
        );
    header.faceCount =
        static_cast<std::uint32_t>(faces.size());
    header.halfEdgeOffset =
        static_cast<std::uint32_t>(
            model.convexHalfEdges.size()
        );
    header.halfEdgeCount =
        static_cast<std::uint32_t>(halfEdges.size());
    header.localLower = packed(quantizationBounds.lower);
    header.localUpper = packed(quantizationBounds.upper);

    for (const Vec3 vertex : vertices) {
        model.geometryVertices.push_back(packed(vertex, 1.0f));
    }
    for (const Triangle& triangle : triangles) {
        for (const std::uint32_t vertex : triangle.vertices) {
            model.geometryIndices.push_back(
                header.vertexOffset + vertex
            );
        }
    }
    model.convexFaces.insert(
        model.convexFaces.end(),
        faces.begin(),
        faces.end()
    );
    model.convexHalfEdges.insert(
        model.convexHalfEdges.end(),
        halfEdges.begin(),
        halfEdges.end()
    );
    result.geometryIndex =
        static_cast<std::uint32_t>(
            model.geometryHeaders.size()
        );
    model.geometryHeaders.push_back(header);
    result.outputVertexCount = header.vertexCount;
    result.outputTriangleCount = header.faceCount;
    result.message = "ok";
    return result;
}

GeometryCookResult cookTriangleMeshGeometry(
    EngineModel& model,
    const std::span<const mr_float4> inputVertices,
    const std::span<const std::uint32_t> inputIndices,
    const std::span<const std::uint32_t> inputMaterials,
    const GeometryCookConfig& config
) {
    GeometryCookResult result{};
    result.inputVertexCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            inputVertices.size(),
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.inputTriangleCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            inputIndices.size() / 3u,
            std::numeric_limits<std::uint32_t>::max()
        ));
    if (!validConfig(config) ||
        inputVertices.size() < 3u ||
        inputIndices.size() < 3u ||
        inputVertices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(
            std::move(result),
            GeometryCookStatus::invalidInput,
            "mesh input or cooker configuration is invalid"
        );
    }
    if (std::any_of(
            inputMaterials.begin(),
            inputMaterials.end(),
            [&](const std::uint32_t material) {
                return material >= model.materials.size();
            }
        )) {
        return failure(
            std::move(result),
            GeometryCookStatus::invalidInput,
            "mesh triangle material index is out of range"
        );
    }
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> remap;
    if (!weldVertices(
            inputVertices,
            config.weldTolerance,
            vertices,
            remap
        )) {
        return failure(
            std::move(result),
            GeometryCookStatus::invalidInput,
            "mesh vertices are non-finite or exceed capacity"
        );
    }
    std::vector<Triangle> triangles;
    if (!cleanTriangles(
            inputIndices,
            inputMaterials,
            remap,
            vertices,
            config,
            triangles,
            result.removedDegenerateTriangles
        ) ||
        triangles.empty()) {
        return failure(
            std::move(result),
            GeometryCookStatus::degenerateGeometry,
            "mesh has no executable triangles"
        );
    }

    std::vector<std::uint32_t> order(triangles.size());
    std::iota(order.begin(), order.end(), 0u);
    std::vector<BVHNode> nodes;
    buildBVHNode(
        triangles,
        order,
        0u,
        order.size(),
        config.meshLeafTriangleCapacity,
        nodes
    );
    if (nodes.size() * MR_MESH_BVH_BRANCHING >
        MR_MESH_BVH_INVALID_ESCAPE) {
        return failure(
            std::move(result),
            GeometryCookStatus::capacityOverflow,
            "mesh BVH escape cursor exceeds its packed ABI"
        );
    }
    std::vector<Triangle> reordered;
    reordered.reserve(triangles.size());
    for (const std::uint32_t index : order) {
        reordered.push_back(triangles[index]);
    }
    triangles = std::move(reordered);

    struct EdgeReference {
        std::uint32_t triangle = 0u;
        std::uint32_t edge = 0u;
    };
    std::map<
        std::pair<std::uint32_t, std::uint32_t>,
        std::vector<EdgeReference>
    > edges;
    for (std::uint32_t triangle = 0u;
         triangle < triangles.size();
         ++triangle) {
        for (std::uint32_t edge = 0u; edge < 3u; ++edge) {
            const std::uint32_t a =
                triangles[triangle].vertices[edge];
            const std::uint32_t b =
                triangles[triangle].vertices[
                    (edge + 1u) % 3u
                ];
            edges[std::minmax(a, b)].push_back({
                triangle,
                edge,
            });
        }
    }
    for (const auto& [edge, references] : edges) {
        (void)edge;
        if (references.size() > 2u) {
            return failure(
                std::move(result),
                GeometryCookStatus::nonManifold,
                "mesh edge has more than two incident triangles"
            );
        }
    }

    if (!checkedArenaAppend(
            model.geometryVertices.size(),
            vertices.size()
        ) ||
        !checkedArenaAppend(
            model.meshTriangles.size(),
            triangles.size()
        ) ||
        !checkedArenaAppend(
            model.meshBvhNodes.size(),
            nodes.size()
        ) ||
        model.geometryHeaders.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
        return failure(
            std::move(result),
            GeometryCookStatus::capacityOverflow,
            "mesh arenas exceed the 32-bit cooked ABI"
        );
    }
    const Bounds bounds = vertexBounds(vertices);
    Bounds quantizationBounds = bounds;
    const double boundScale = std::max({
        std::abs(bounds.lower.x),
        std::abs(bounds.lower.y),
        std::abs(bounds.lower.z),
        std::abs(bounds.upper.x),
        std::abs(bounds.upper.y),
        std::abs(bounds.upper.z),
        1.0,
    });
    const double boundsInflation = std::max(
        config.weldTolerance,
        16.0 * std::numeric_limits<float>::epsilon() *
            boundScale
    );
    const Vec3 inflation{
        boundsInflation,
        boundsInflation,
        boundsInflation,
    };
    quantizationBounds.lower =
        quantizationBounds.lower - inflation;
    quantizationBounds.upper =
        quantizationBounds.upper + inflation;
    MRGeometryHeaderGPU header{};
    header.kind = MR_GEOMETRY_TRIANGLE_MESH;
    header.flags = MR_GEOMETRY_FLAG_QUANTIZED_BVH;
    header.vertexOffset =
        static_cast<std::uint32_t>(
            model.geometryVertices.size()
        );
    header.vertexCount =
        static_cast<std::uint32_t>(vertices.size());
    header.bvhOffset =
        static_cast<std::uint32_t>(
            model.meshBvhNodes.size()
        );
    header.bvhCount =
        static_cast<std::uint32_t>(nodes.size());
    header.triangleOffset =
        static_cast<std::uint32_t>(
            model.meshTriangles.size()
        );
    header.triangleCount =
        static_cast<std::uint32_t>(triangles.size());
    header.localLower = packed(quantizationBounds.lower);
    header.localUpper = packed(quantizationBounds.upper);

    std::vector<MRMeshTriangleGPU> cookedTriangles(
        triangles.size()
    );
    for (std::uint32_t triangle = 0u;
         triangle < triangles.size();
         ++triangle) {
        MRMeshTriangleGPU cooked{};
        cooked.verticesAndFeature = {
            header.vertexOffset +
                triangles[triangle].vertices[0],
            header.vertexOffset +
                triangles[triangle].vertices[1],
            header.vertexOffset +
                triangles[triangle].vertices[2],
            triangles[triangle].stableFeature,
        };
        cooked.adjacencyAndEdges = {
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            MR_INVALID_INDEX,
            0u,
        };
        cooked.materialAndFlags = {
            triangles[triangle].material,
            0u,
            0u,
            0u,
        };
        cookedTriangles[triangle] = cooked;
    }
    for (const auto& [edge, references] : edges) {
        (void)edge;
        if (references.size() == 1u) {
            cookedTriangles[references[0].triangle]
                .adjacencyAndEdges.w |=
                1u << references[0].edge;
            continue;
        }
        const EdgeReference first = references[0];
        const EdgeReference second = references[1];
        setComponent(
            cookedTriangles[first.triangle]
                .adjacencyAndEdges,
            first.edge,
            header.triangleOffset + second.triangle
        );
        setComponent(
            cookedTriangles[second.triangle]
                .adjacencyAndEdges,
            second.edge,
            header.triangleOffset + first.triangle
        );
        const bool firstOwns =
            triangles[first.triangle].stableFeature <=
            triangles[second.triangle].stableFeature;
        cookedTriangles[
            firstOwns ? first.triangle : second.triangle
        ].adjacencyAndEdges.w |=
            1u << (firstOwns ? first.edge : second.edge);
    }

    std::vector<MRMeshBVHNodeGPU> cookedNodes(nodes.size());
    encodeBVHNode(
        0u,
        MR_MESH_BVH_INVALID_ESCAPE,
        quantizationBounds,
        nodes,
        cookedNodes
    );
    for (const Vec3 vertex : vertices) {
        model.geometryVertices.push_back(packed(vertex, 1.0f));
    }
    model.meshTriangles.insert(
        model.meshTriangles.end(),
        cookedTriangles.begin(),
        cookedTriangles.end()
    );
    model.meshBvhNodes.insert(
        model.meshBvhNodes.end(),
        cookedNodes.begin(),
        cookedNodes.end()
    );
    result.geometryIndex =
        static_cast<std::uint32_t>(
            model.geometryHeaders.size()
        );
    model.geometryHeaders.push_back(header);
    result.outputVertexCount = header.vertexCount;
    result.outputTriangleCount = header.triangleCount;
    result.message = "ok";
    return result;
}

} // namespace metalrobo
