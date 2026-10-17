#pragma once

#include "metalrobo/VisualPresentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <stdexcept>
#include <vector>

// Presentation-only source-face window used by the opaque Skin inspection
// layer. It never alters the original sclera primitive or its Organ instance.
namespace numi_human_resting_ocular_window {

struct Point3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

struct Report {
    std::uint32_t sourceSurfaceStableID = 0u;
    enum class Kind : std::uint8_t { sclera_window, posterior_lens_backing };
    Kind kind = Kind::sclera_window;
    std::uint32_t corneaStableID = 0u;
    std::uint32_t irisStableID = 0u;
    std::uint32_t derivedStableID = 0u;
    std::uint32_t sourcePrimitiveIndex = 0u;
    std::uint32_t sourceInstanceIndex = 0u;
    std::array<double, 3> eyeAxis{};
    double irisFrontDepth = 0.0;
    double corneaFrontDepth = 0.0;
    std::uint32_t sourceFaceCount = 0u;
    std::uint32_t maskedFaceCount = 0u;
    std::vector<std::uint32_t> retainedSourceFaceOrdinals;
    std::vector<std::uint32_t> maskedSourceFaceOrdinals;
    std::vector<std::uint32_t> retainedIndices;
};

inline Point3 add(Point3 a, Point3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Point3 subtract(Point3 a, Point3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Point3 scale(Point3 a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline double dot(Point3 a, Point3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double length(Point3 a) {
    return std::sqrt(dot(a, a));
}

inline Point3 cross(Point3 a, Point3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline Point3 rotate(Point3 point, mr_float4 quaternion) {
    const Point3 axis{quaternion.x, quaternion.y, quaternion.z};
    const Point3 first = cross(axis, point);
    const Point3 second = cross(axis, first);
    return add(point, add(scale(first, 2.0 * quaternion.w), scale(second, 2.0)));
}

inline Point3 transform(
    const MRVisualInstanceGPUV2& instance,
    const MRVisualVertexGPUV2& vertex
) {
    Point3 point{vertex.position.x, vertex.position.y, vertex.position.z};
    point = scale(rotate(point, instance.orientation), instance.translationAndScale.w);
    return add(point, {instance.translationAndScale.x,
                       instance.translationAndScale.y,
                       instance.translationAndScale.z});
}

inline double cross2(Point2 o, Point2 a, Point2 b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

inline std::vector<Point2> convexHull(std::vector<Point2> points) {
    std::sort(points.begin(), points.end(), [](Point2 a, Point2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    points.erase(std::unique(points.begin(), points.end(), [](Point2 a, Point2 b) {
        return a.x == b.x && a.y == b.y;
    }), points.end());
    if (points.size() < 3u) return {};
    std::vector<Point2> hull;
    hull.reserve(points.size() * 2u);
    for (const Point2 point : points) {
        while (hull.size() >= 2u &&
               cross2(hull[hull.size() - 2u], hull.back(), point) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    const std::size_t lowerSize = hull.size();
    for (std::size_t i = points.size() - 1u; i > 0u; --i) {
        const Point2 point = points[i - 1u];
        while (hull.size() > lowerSize &&
               cross2(hull[hull.size() - 2u], hull.back(), point) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    if (!hull.empty()) hull.pop_back();
    return hull;
}

inline bool insideProjectedTriangle(Point2 point, Point2 a, Point2 b, Point2 c) {
    const double denominator = cross2(a, b, c);
    if (std::abs(denominator) < 1.0e-14) return false;
    const double wa = cross2(point, b, c) / denominator;
    const double wb = cross2(a, point, c) / denominator;
    const double wc = 1.0 - wa - wb;
    constexpr double barycentricTolerance = 1.0e-9;
    return wa >= -barycentricTolerance && wb >= -barycentricTolerance &&
        wc >= -barycentricTolerance;
}

inline bool insideConvexHull(const std::vector<Point2>& hull, Point2 point) {
    if (hull.size() < 3u) return false;
    constexpr double tolerance = 1.0e-10;
    for (std::size_t i = 0; i < hull.size(); ++i) {
        if (cross2(hull[i], hull[(i + 1u) % hull.size()], point) < -tolerance) {
            return false;
        }
    }
    return true;
}

inline const MRVisualPrimitiveGPUV2& primitiveFor(
    const metalrobo::VisualAssetPackV2& pack,
    std::uint32_t semantic,
    std::uint32_t stableID,
    std::uint32_t& primitiveIndex
) {
    const MRVisualPrimitiveGPUV2* found = nullptr;
    for (std::uint32_t i = 0; i < pack.primitives.size(); ++i) {
        const auto& primitive = pack.primitives[i];
        if (primitive.identity.x != semantic || primitive.identity.w != stableID) continue;
        if (found != nullptr) throw std::runtime_error("ocular window source identity is ambiguous");
        found = &primitive;
        primitiveIndex = i;
    }
    if (found == nullptr) throw std::runtime_error("ocular window source identity is absent");
    return *found;
}

inline std::vector<Point3> pointsFor(
    const metalrobo::VisualAssetPackV2& pack,
    const MRVisualPrimitiveGPUV2& primitive
) {
    if (primitive.geometry.w >= pack.instances.size()) {
        throw std::runtime_error("ocular source primitive references a missing instance");
    }
    const auto& instance = pack.instances[primitive.geometry.w];
    if (primitive.geometry.y == 0u || primitive.geometry.y % 3u != 0u ||
        primitive.geometry.x > pack.indices.size() ||
        primitive.geometry.y > pack.indices.size() - primitive.geometry.x) {
        throw std::runtime_error("ocular source primitive has an invalid triangle range");
    }
    std::vector<Point3> points;
    points.reserve(primitive.geometry.y);
    for (std::uint32_t i = primitive.geometry.x;
         i < primitive.geometry.x + primitive.geometry.y; ++i) {
        const auto vertexIndex = pack.indices[i];
        if (vertexIndex >= pack.vertices.size()) {
            throw std::runtime_error("ocular source primitive has an invalid vertex index");
        }
        points.push_back(transform(instance, pack.vertices[vertexIndex]));
    }
    return points;
}

enum class CanonicalExteriorSetState : std::uint8_t {
    absent,
    complete,
    partial,
    ambiguous,
};

struct CanonicalExteriorSetInspection {
    CanonicalExteriorSetState state = CanonicalExteriorSetState::absent;
    std::uint32_t presentSurfaceCount = 0u;
    std::uint32_t expectedSurfaceCount = 6u;
};

// The Skin window requires one canonical sclera, cornea, and iris for each eye.
// Older packs with none remain unchanged; partial or ambiguous typed sources fail closed.
inline CanonicalExteriorSetInspection inspectCanonicalExteriorSet(
    const metalrobo::VisualAssetPackV2& pack,
    std::uint32_t organSemantic
) {
    constexpr std::array<std::uint32_t, 6> requiredStableIDs{{386u, 383u, 384u, 395u, 392u, 393u}};
    CanonicalExteriorSetInspection result;
    bool ambiguous = false;
    for (const auto stableID : requiredStableIDs) {
        std::vector<std::uint32_t> primitiveIndices;
        std::vector<std::uint32_t> instanceIndices;
        for (std::uint32_t i = 0u; i < pack.primitives.size(); ++i) {
            const auto& primitive = pack.primitives[i];
            if (primitive.identity.x == organSemantic && primitive.identity.w == stableID) {
                primitiveIndices.push_back(i);
            }
        }
        for (std::uint32_t i = 0u; i < pack.instances.size(); ++i) {
            const auto& instance = pack.instances[i];
            if (instance.identity.x == organSemantic && instance.identity.w == stableID) {
                instanceIndices.push_back(i);
            }
        }
        if (primitiveIndices.empty() && instanceIndices.empty()) continue;
        ++result.presentSurfaceCount;
        if (primitiveIndices.size() != 1u || instanceIndices.size() != 1u) {
            ambiguous = true;
            continue;
        }
        const auto primitiveIndex = primitiveIndices.front();
        const auto instanceIndex = instanceIndices.front();
        const auto& primitive = pack.primitives[primitiveIndex];
        const auto& instance = pack.instances[instanceIndex];
        if (primitive.geometry.w != instanceIndex ||
            instance.geometry.x > pack.primitives.size() ||
            instance.geometry.y > pack.primitives.size() - instance.geometry.x ||
            primitiveIndex < instance.geometry.x ||
            primitiveIndex >= instance.geometry.x + instance.geometry.y ||
            primitive.geometry.y == 0u || primitive.geometry.y % 3u != 0u ||
            primitive.geometry.x > pack.indices.size() ||
            primitive.geometry.y > pack.indices.size() - primitive.geometry.x ||
            primitive.geometry.z >= pack.materials.size()) {
            ambiguous = true;
        }
    }
    if (ambiguous) {
        result.state = CanonicalExteriorSetState::ambiguous;
    } else if (result.presentSurfaceCount == 0u) {
        result.state = CanonicalExteriorSetState::absent;
    } else if (result.presentSurfaceCount == requiredStableIDs.size()) {
        result.state = CanonicalExteriorSetState::complete;
    } else {
        result.state = CanonicalExteriorSetState::partial;
    }
    return result;
}

inline bool facePartitionComplete(const Report& report) {
    if (report.sourceFaceCount == 0u ||
        report.retainedSourceFaceOrdinals.size() + report.maskedSourceFaceOrdinals.size() !=
            report.sourceFaceCount || report.retainedIndices.size() !=
            report.retainedSourceFaceOrdinals.size() * 3u ||
        report.maskedFaceCount != report.maskedSourceFaceOrdinals.size()) return false;
    std::vector<std::uint8_t> seen(report.sourceFaceCount, 0u);
    for (const auto face : report.retainedSourceFaceOrdinals) {
        if (face >= seen.size() || seen[face]++) return false;
    }
    for (const auto face : report.maskedSourceFaceOrdinals) {
        if (face >= seen.size() || seen[face]++) return false;
    }
    return std::all_of(seen.begin(), seen.end(), [](std::uint8_t count) { return count == 1u; });
}

inline Report buildScleraWindow(
    const metalrobo::VisualAssetPackV2& pack,
    std::uint32_t organSemantic,
    std::uint32_t scleraStableID,
    std::uint32_t corneaStableID,
    std::uint32_t irisStableID,
    std::uint32_t derivedStableID
) {
    Report report;
    report.sourceSurfaceStableID = scleraStableID;
    report.kind = Report::Kind::sclera_window;
    report.corneaStableID = corneaStableID;
    report.irisStableID = irisStableID;
    report.derivedStableID = derivedStableID;
    std::uint32_t scleraPrimitiveIndex = 0u;
    std::uint32_t corneaPrimitiveIndex = 0u;
    std::uint32_t irisPrimitiveIndex = 0u;
    const auto& sclera = primitiveFor(pack, organSemantic, scleraStableID, scleraPrimitiveIndex);
    const auto& cornea = primitiveFor(pack, organSemantic, corneaStableID, corneaPrimitiveIndex);
    const auto& iris = primitiveFor(pack, organSemantic, irisStableID, irisPrimitiveIndex);
    report.sourcePrimitiveIndex = scleraPrimitiveIndex;
    report.sourceInstanceIndex = sclera.geometry.w;
    const auto& scleraInstance = pack.instances.at(sclera.geometry.w);
    const auto& corneaInstance = pack.instances.at(cornea.geometry.w);
    const auto& irisInstance = pack.instances.at(iris.geometry.w);
    if (scleraInstance.binding.z != corneaInstance.binding.z ||
        scleraInstance.binding.z != irisInstance.binding.z ||
        scleraInstance.binding.y != corneaInstance.binding.y ||
        scleraInstance.binding.y != irisInstance.binding.y) {
        throw std::runtime_error("ocular source surfaces do not share a registered body frame");
    }
    const auto scleraPoints = pointsFor(pack, sclera);
    const auto corneaPoints = pointsFor(pack, cornea);
    const auto irisPoints = pointsFor(pack, iris);
    const auto mean = [](const std::vector<Point3>& points) {
        Point3 center{};
        for (const Point3 point : points) center = add(center, point);
        return scale(center, 1.0 / static_cast<double>(points.size()));
    };
    const Point3 scleraCenter = mean(scleraPoints);
    const Point3 corneaCenter = mean(corneaPoints);
    const Point3 irisCenter = mean(irisPoints);
    Point3 axis = subtract(corneaCenter, scleraCenter);
    const double axisLength = length(axis);
    if (!(std::isfinite(axisLength) && axisLength > 1.0e-8)) {
        throw std::runtime_error("ocular cornea-facing axis is degenerate");
    }
    axis = scale(axis, 1.0 / axisLength);
    report.eyeAxis = {axis.x, axis.y, axis.z};
    report.irisFrontDepth = dot(irisCenter, axis);
    report.corneaFrontDepth = dot(corneaCenter, axis);

    Point3 helper = std::abs(axis.x) < 0.8 ? Point3{1.0, 0.0, 0.0} : Point3{0.0, 1.0, 0.0};
    Point3 u = subtract(helper, scale(axis, dot(helper, axis)));
    const double uLength = length(u);
    if (!(std::isfinite(uLength) && uLength > 1.0e-8)) {
        throw std::runtime_error("ocular cornea window basis is degenerate");
    }
    u = scale(u, 1.0 / uLength);
    const Point3 v = cross(axis, u);
    const auto project = [&](Point3 point) {
        const Point3 relative = subtract(point, corneaCenter);
        return Point2{dot(relative, u), dot(relative, v)};
    };
    std::vector<Point2> corneaFootprint;
    corneaFootprint.reserve(corneaPoints.size());
    for (const Point3 point : corneaPoints) corneaFootprint.push_back(project(point));
    const auto footprintHull = convexHull(std::move(corneaFootprint));
    if (footprintHull.size() < 3u) {
        throw std::runtime_error("ocular cornea projected footprint is degenerate");
    }

    report.sourceFaceCount = sclera.geometry.y / 3u;
    report.retainedIndices.reserve(sclera.geometry.y);
    for (std::uint32_t face = 0u; face < report.sourceFaceCount; ++face) {
        const std::uint32_t offset = sclera.geometry.x + face * 3u;
        const std::uint32_t ia = pack.indices[offset];
        const std::uint32_t ib = pack.indices[offset + 1u];
        const std::uint32_t ic = pack.indices[offset + 2u];
        const Point3 a = transform(scleraInstance, pack.vertices.at(ia));
        const Point3 b = transform(scleraInstance, pack.vertices.at(ib));
        const Point3 c = transform(scleraInstance, pack.vertices.at(ic));
        const Point3 center = scale(add(add(a, b), c), 1.0 / 3.0);
        const Point3 normal = cross(subtract(b, a), subtract(c, a));
        const double normalLength = length(normal);
        const double depth = dot(center, axis);
        const bool capFace = insideConvexHull(footprintHull, project(center)) &&
            depth > report.irisFrontDepth && normalLength > 0.0 &&
            dot(scale(normal, 1.0 / normalLength), axis) > 0.15;
        if (capFace) {
            report.maskedSourceFaceOrdinals.push_back(face);
            continue;
        }
        report.retainedSourceFaceOrdinals.push_back(face);
        report.retainedIndices.insert(report.retainedIndices.end(), {ia, ib, ic});
    }
    report.maskedFaceCount = static_cast<std::uint32_t>(report.maskedSourceFaceOrdinals.size());
    if (report.maskedFaceCount == 0u || report.retainedIndices.empty()) {
        throw std::runtime_error("ocular sclera window produced no useful source-face subset");
    }
    if (!facePartitionComplete(report)) {
        throw std::runtime_error("ocular window source-face partition is incomplete");
    }
    return report;
}

// Select only real posterior lens source faces whose projected centroids fall
// inside the iris silhouette but in its central opening. The iris remains the
// visible anatomical surface; this small opaque clone is a Skin-only dark
// backing approximation, not a pupil or a replacement lens.
inline Report buildPosteriorLensBacking(
    const metalrobo::VisualAssetPackV2& pack,
    std::uint32_t organSemantic,
    std::uint32_t scleraStableID,
    std::uint32_t corneaStableID,
    std::uint32_t irisStableID,
    std::uint32_t lensStableID,
    std::uint32_t derivedStableID
) {
    Report report;
    report.sourceSurfaceStableID = lensStableID;
    report.kind = Report::Kind::posterior_lens_backing;
    report.corneaStableID = corneaStableID;
    report.irisStableID = irisStableID;
    report.derivedStableID = derivedStableID;
    std::uint32_t scleraPrimitiveIndex = 0u;
    std::uint32_t corneaPrimitiveIndex = 0u;
    std::uint32_t irisPrimitiveIndex = 0u;
    std::uint32_t lensPrimitiveIndex = 0u;
    const auto& sclera = primitiveFor(pack, organSemantic, scleraStableID, scleraPrimitiveIndex);
    const auto& cornea = primitiveFor(pack, organSemantic, corneaStableID, corneaPrimitiveIndex);
    const auto& iris = primitiveFor(pack, organSemantic, irisStableID, irisPrimitiveIndex);
    const auto& lens = primitiveFor(pack, organSemantic, lensStableID, lensPrimitiveIndex);
    report.sourcePrimitiveIndex = lensPrimitiveIndex;
    report.sourceInstanceIndex = lens.geometry.w;
    const auto& scleraInstance = pack.instances.at(sclera.geometry.w);
    const auto& corneaInstance = pack.instances.at(cornea.geometry.w);
    const auto& irisInstance = pack.instances.at(iris.geometry.w);
    const auto& lensInstance = pack.instances.at(lens.geometry.w);
    if (scleraInstance.binding.z != corneaInstance.binding.z ||
        scleraInstance.binding.z != irisInstance.binding.z ||
        scleraInstance.binding.z != lensInstance.binding.z ||
        scleraInstance.binding.y != corneaInstance.binding.y ||
        scleraInstance.binding.y != irisInstance.binding.y ||
        scleraInstance.binding.y != lensInstance.binding.y) {
        throw std::runtime_error("ocular source surfaces do not share a registered body frame");
    }
    const auto scleraPoints = pointsFor(pack, sclera);
    const auto corneaPoints = pointsFor(pack, cornea);
    const auto irisPoints = pointsFor(pack, iris);
    const auto lensPoints = pointsFor(pack, lens);
    const auto mean = [](const std::vector<Point3>& points) {
        Point3 center{};
        for (const Point3 point : points) center = add(center, point);
        return scale(center, 1.0 / static_cast<double>(points.size()));
    };
    const Point3 scleraCenter = mean(scleraPoints);
    const Point3 corneaCenter = mean(corneaPoints);
    const Point3 irisCenter = mean(irisPoints);
    Point3 axis = subtract(corneaCenter, scleraCenter);
    const double axisLength = length(axis);
    if (!(std::isfinite(axisLength) && axisLength > 1.0e-8)) {
        throw std::runtime_error("ocular cornea-facing axis is degenerate");
    }
    axis = scale(axis, 1.0 / axisLength);
    report.eyeAxis = {axis.x, axis.y, axis.z};
    report.irisFrontDepth = dot(irisCenter, axis);
    report.corneaFrontDepth = dot(mean(corneaPoints), axis);
    const Point3 helper = std::abs(axis.x) < 0.8 ? Point3{1.0, 0.0, 0.0} : Point3{0.0, 1.0, 0.0};
    Point3 u = subtract(helper, scale(axis, dot(helper, axis)));
    const double uLength = length(u);
    if (!(std::isfinite(uLength) && uLength > 1.0e-8)) {
        throw std::runtime_error("ocular backing projection basis is degenerate");
    }
    u = scale(u, 1.0 / uLength);
    const Point3 v = cross(axis, u);
    const auto project = [&](Point3 point) { return Point2{dot(point, u), dot(point, v)}; };
    std::vector<Point2> irisFootprint;
    irisFootprint.reserve(irisPoints.size());
    for (const Point3 point : irisPoints) irisFootprint.push_back(project(point));
    const auto irisHull = convexHull(std::move(irisFootprint));
    if (irisHull.size() < 3u) throw std::runtime_error("ocular iris projected footprint is degenerate");
    std::vector<std::array<Point2, 3>> projectedIrisTriangles;
    projectedIrisTriangles.reserve(iris.geometry.y / 3u);
    for (std::uint32_t face = 0u; face < iris.geometry.y / 3u; ++face) {
        projectedIrisTriangles.push_back({project(irisPoints[face * 3u]),
            project(irisPoints[face * 3u + 1u]), project(irisPoints[face * 3u + 2u])});
    }

    report.sourceFaceCount = lens.geometry.y / 3u;
    report.retainedIndices.reserve(lens.geometry.y / 10u);
    constexpr double behindIrisMargin = 1.0e-7;
    for (std::uint32_t face = 0u; face < report.sourceFaceCount; ++face) {
        const std::uint32_t offset = lens.geometry.x + face * 3u;
        const std::uint32_t ia = pack.indices.at(offset);
        const std::uint32_t ib = pack.indices.at(offset + 1u);
        const std::uint32_t ic = pack.indices.at(offset + 2u);
        const Point3 a = lensPoints[face * 3u];
        const Point3 b = lensPoints[face * 3u + 1u];
        const Point3 c = lensPoints[face * 3u + 2u];
        const Point3 center = scale(add(add(a, b), c), 1.0 / 3.0);
        const Point2 projectedCenter = project(center);
        bool coveredByIris = false;
        for (const auto& triangle : projectedIrisTriangles) {
            if (insideProjectedTriangle(projectedCenter, triangle[0], triangle[1], triangle[2])) {
                coveredByIris = true;
                break;
            }
        }
        const bool posterior = dot(center, axis) < report.irisFrontDepth - behindIrisMargin;
        if (insideConvexHull(irisHull, projectedCenter) && !coveredByIris && posterior) {
            report.retainedSourceFaceOrdinals.push_back(face);
            report.retainedIndices.insert(report.retainedIndices.end(), {ia, ib, ic});
        } else {
            report.maskedSourceFaceOrdinals.push_back(face);
        }
    }
    report.maskedFaceCount = static_cast<std::uint32_t>(report.maskedSourceFaceOrdinals.size());
    if (report.retainedIndices.empty() || !facePartitionComplete(report)) {
        throw std::runtime_error("posterior lens backing did not produce a complete nonempty source-face subset");
    }
    return report;
}

} // namespace numi_human_resting_ocular_window

namespace numi_human_resting_ocular_window {

inline void appendSkinWindowClone(
    metalrobo::VisualAssetPackV2& pack,
    const Report& report,
    std::uint32_t skinMaterialIndex
) {
    if (!facePartitionComplete(report)) {
        throw std::runtime_error("ocular window source-face partition is incomplete");
    }
    if (report.sourceInstanceIndex >= pack.instances.size() ||
        report.sourcePrimitiveIndex >= pack.primitives.size() ||
        skinMaterialIndex >= pack.materials.size() || report.retainedIndices.empty()) {
        throw std::runtime_error("ocular window clone inputs are out of range");
    }
    for (const auto& instance : pack.instances) {
        if (instance.identity.x == 51010u && instance.identity.w == report.derivedStableID) {
            throw std::runtime_error("ocular Skin window typed identity collides with a source instance");
        }
    }
    for (const auto& primitive : pack.primitives) {
        if (primitive.identity.x == 51010u && primitive.identity.w == report.derivedStableID) {
            throw std::runtime_error("ocular Skin window typed identity collides with a source primitive");
        }
    }
    if (report.retainedIndices.size() % 3u != 0u) {
        throw std::runtime_error("ocular Skin window index subset is not triangular");
    }
    const auto vertexCount = pack.vertices.size();
    const auto sourceInstance = pack.instances.at(report.sourceInstanceIndex);
    const auto sourcePrimitive = pack.primitives.at(report.sourcePrimitiveIndex);
    const std::uint32_t firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    const std::uint32_t instanceIndex = static_cast<std::uint32_t>(pack.instances.size());
    const std::uint32_t primitiveIndex = static_cast<std::uint32_t>(pack.primitives.size());
    for (const auto index : report.retainedIndices) {
        if (index >= pack.vertices.size()) {
            throw std::runtime_error("ocular Skin window references a missing source vertex");
        }
    }
    pack.indices.insert(pack.indices.end(), report.retainedIndices.begin(), report.retainedIndices.end());
    MRVisualPrimitiveGPUV2 primitive = sourcePrimitive;
    primitive.geometry = {firstIndex, static_cast<std::uint32_t>(report.retainedIndices.size()),
                          skinMaterialIndex, instanceIndex};
    primitive.identity.y = instanceIndex;
    primitive.identity.w = report.derivedStableID;
    MRVisualInstanceGPUV2 instance = sourceInstance;
    instance.identity.y = instanceIndex;
    instance.identity.w = report.derivedStableID;
    instance.geometry = {primitiveIndex, 1u, 0u, 0u};
    pack.primitives.push_back(primitive);
    pack.instances.push_back(instance);
    if (pack.vertices.size() != vertexCount) {
        throw std::runtime_error("ocular Skin window unexpectedly changed source vertex allocation");
    }
}

} // namespace numi_human_resting_ocular_window
