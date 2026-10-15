#pragma once

// Reference-geometry utilities for element-local crack-band regularization.
// This file does not define or enable a damage constitutive law.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace numi::cardboard {

struct FractureVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline FractureVec3 operator+(FractureVec3 a, FractureVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline FractureVec3 operator-(FractureVec3 a, FractureVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline FractureVec3 operator*(FractureVec3 a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}
inline FractureVec3 operator/(FractureVec3 a, double s) {
    return {a.x / s, a.y / s, a.z / s};
}
inline double fractureDot(FractureVec3 a, FractureVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline FractureVec3 fractureCross(FractureVec3 a, FractureVec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
inline double fractureNorm(FractureVec3 a) {
    return std::sqrt(fractureDot(a, a));
}
inline bool fractureFinite(FractureVec3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

// Row-major proper rotation mapping element-material vectors to reference space.
struct FractureRotation3 {
    std::array<double, 9> value{1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0};
};

enum class FractureNormalFrame {
    Reference,
    ElementMaterial,
};

struct TetCrackPlane {
    // The plane is n dot (X - Xcentroid) = offsetFromCentroidM. Both X and
    // offset are reference lengths in metres; the normal may be material-frame.
    FractureVec3 normal{1.0, 0.0, 0.0};
    FractureNormalFrame normalFrame = FractureNormalFrame::Reference;
    double offsetFromCentroidM = 0.0;
    FractureRotation3 materialToReference{};
    double relativePlaneTolerance = 1.0e-12;
};

struct TetCrackPlaneGeometry {
    double referenceVolumeM3 = 0.0;
    double referenceAreaM2 = 0.0;
    // Absent when the selected plane has no positive-area cut in this tet.
    std::optional<double> characteristicLengthM;
    FractureVec3 referenceUnitNormal{};
    std::vector<FractureVec3> cutPolygonReferenceM;

    bool hasPositiveAreaCut() const {
        return referenceAreaM2 > 0.0 && characteristicLengthM.has_value();
    }
};

namespace fracture_detail {

inline double determinant(const FractureRotation3& r) {
    const auto& m = r.value;
    return m[0] * (m[4] * m[8] - m[5] * m[7]) -
           m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline FractureVec3 rotate(const FractureRotation3& r, FractureVec3 v) {
    const auto& m = r.value;
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z};
}

inline void validateProperRotation(const FractureRotation3& r) {
    for (double x : r.value) {
        if (!std::isfinite(x))
            throw std::invalid_argument("material-to-reference rotation is nonfinite");
    }
    constexpr double tolerance = 1.0e-9;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double product = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                product += r.value[3 * k + i] * r.value[3 * k + j];
            const double expected = i == j ? 1.0 : 0.0;
            if (std::abs(product - expected) > tolerance)
                throw std::invalid_argument(
                    "material-to-reference transform must be orthonormal");
        }
    }
    if (std::abs(determinant(r) - 1.0) > tolerance)
        throw std::invalid_argument(
            "material-to-reference transform must be a proper rotation");
}

inline double maxEdgeLength(const std::array<FractureVec3, 4>& x) {
    double result = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = i + 1; j < 4; ++j)
            result = std::max(result, fractureNorm(x[j] - x[i]));
    return result;
}

inline bool nearPoint(FractureVec3 a, FractureVec3 b, double tolerance) {
    return fractureNorm(a - b) <= tolerance;
}

} // namespace fracture_detail

struct TetCrackPlaneClassification {
    double edgeScaleM = 0.0;
    double referenceVolumeM3 = 0.0;
    FractureVec3 referenceUnitNormal{};
    FractureVec3 planePointReferenceM{};
    double distanceToleranceM = 0.0;
    std::array<double, 4> signedDistanceM{};
    std::array<int, 4> side{};
};

// This is the single source of centroid, rotated normal, signed-distance, and
// tolerance arithmetic used by both polygon construction and the retained
// builder's full-face ambiguity gate.
inline TetCrackPlaneClassification classifyTetCrackPlane(
    const std::array<FractureVec3, 4>& referenceVerticesM,
    const TetCrackPlane& plane) {
    for (const FractureVec3 vertex : referenceVerticesM) {
        if (!fractureFinite(vertex))
            throw std::invalid_argument("tetrahedron reference vertex is nonfinite");
    }
    if (!(std::isfinite(plane.offsetFromCentroidM) &&
          std::isfinite(plane.relativePlaneTolerance) &&
          plane.relativePlaneTolerance > 0.0 &&
          plane.relativePlaneTolerance <= 1.0e-6))
        throw std::invalid_argument("crack-plane offset or tolerance is invalid");
    if (!fractureFinite(plane.normal))
        throw std::invalid_argument("crack-plane normal is nonfinite");

    TetCrackPlaneClassification classification;
    classification.edgeScaleM = fracture_detail::maxEdgeLength(referenceVerticesM);
    if (!(std::isfinite(classification.edgeScaleM) && classification.edgeScaleM > 0.0))
        throw std::invalid_argument("tetrahedron has zero or invalid edge scale");
    const FractureVec3 a = referenceVerticesM[0];
    const FractureVec3 b = referenceVerticesM[1];
    const FractureVec3 c = referenceVerticesM[2];
    const FractureVec3 d = referenceVerticesM[3];
    const double signedSixVolume = fractureDot(b - a, fractureCross(c - a, d - a));
    classification.referenceVolumeM3 = std::abs(signedSixVolume) / 6.0;
    const double degeneracyFloor = 128.0 * std::numeric_limits<double>::epsilon() *
                                   classification.edgeScaleM * classification.edgeScaleM *
                                   classification.edgeScaleM;
    if (!(std::isfinite(classification.referenceVolumeM3) &&
          classification.referenceVolumeM3 > degeneracyFloor))
        throw std::invalid_argument("tetrahedron reference volume is degenerate");

    FractureVec3 normal = plane.normal;
    if (plane.normalFrame == FractureNormalFrame::ElementMaterial) {
        fracture_detail::validateProperRotation(plane.materialToReference);
        normal = fracture_detail::rotate(plane.materialToReference, normal);
    } else if (plane.normalFrame != FractureNormalFrame::Reference) {
        throw std::invalid_argument("unknown crack-plane normal frame");
    }
    const double normalLength = fractureNorm(normal);
    if (!(std::isfinite(normalLength) && normalLength > 0.0))
        throw std::invalid_argument("crack-plane normal must be nonzero");
    normal = normal / normalLength;
    classification.referenceUnitNormal = normal;

    FractureVec3 centroid{};
    for (FractureVec3 vertex : referenceVerticesM) centroid = centroid + vertex;
    centroid = centroid / 4.0;
    classification.planePointReferenceM =
        centroid + normal * plane.offsetFromCentroidM;
    classification.distanceToleranceM = plane.relativePlaneTolerance *
                                        classification.edgeScaleM;
    for (std::size_t i = 0; i < 4; ++i) {
        classification.signedDistanceM[i] = fractureDot(
            normal, referenceVerticesM[i] - classification.planePointReferenceM);
        classification.side[i] =
            classification.signedDistanceM[i] > classification.distanceToleranceM ? 1 :
            (classification.signedDistanceM[i] < -classification.distanceToleranceM ? -1 : 0);
    }
    return classification;
}

// A face is numerically ambiguous when all three of its vertices lie in the
// exact classifier's on-plane band. This check uses the already-computed
// signed distances, so its threshold cannot drift from polygon construction.
inline bool tetCrackPlaneClearsEveryFullFace(
    const TetCrackPlaneClassification& classification) {
    constexpr std::array<std::array<std::size_t, 3>, 4> faces{{
        {{1, 2, 3}}, {{0, 3, 2}}, {{0, 1, 3}}, {{0, 2, 1}},
    }};
    for (const auto& face : faces) {
        double maximumAbsoluteDistance = 0.0;
        for (const std::size_t vertex : face) {
            const double distance = classification.signedDistanceM[vertex];
            if (!std::isfinite(distance)) return false;
            maximumAbsoluteDistance = std::max(maximumAbsoluteDistance, std::abs(distance));
        }
        if (!(maximumAbsoluteDistance > classification.distanceToleranceM)) return false;
    }
    return std::isfinite(classification.distanceToleranceM) &&
           classification.distanceToleranceM > 0.0;
}

// Computes the actual intersection polygon of a selected plane with one
// reference tetrahedron. Volume is abs(det)/6, so arbitrary input vertex
// winding is admitted without changing the result. A zero-area/no-cut result
// has no characteristic length and contributes zero fracture area.
enum class FullFaceAmbiguityPolicy {
    Allow,
    RejectWithinClassificationTolerance,
};

inline TetCrackPlaneGeometry computeTetCrackPlaneGeometry(
    const std::array<FractureVec3, 4>& referenceVerticesM,
    const TetCrackPlane& plane,
    const FullFaceAmbiguityPolicy facePolicy = FullFaceAmbiguityPolicy::Allow) {
    const TetCrackPlaneClassification classification =
        classifyTetCrackPlane(referenceVerticesM, plane);
    if (facePolicy == FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance &&
        !tetCrackPlaneClearsEveryFullFace(classification))
        throw std::invalid_argument(
            "crack plane is within classification tolerance of an entire tetrahedron face");
    if (facePolicy != FullFaceAmbiguityPolicy::Allow &&
        facePolicy != FullFaceAmbiguityPolicy::RejectWithinClassificationTolerance)
        throw std::invalid_argument("unknown full-face ambiguity policy");
    const double edgeScale = classification.edgeScaleM;
    const double volume = classification.referenceVolumeM3;
    const FractureVec3 normal = classification.referenceUnitNormal;
    const double distanceTolerance = classification.distanceToleranceM;
    const auto& distance = classification.signedDistanceM;
    const auto& side = classification.side;

    TetCrackPlaneGeometry result;
    result.referenceVolumeM3 = volume;
    result.referenceUnitNormal = normal;
    const double pointTolerance = 4.0 * distanceTolerance;
    auto addUnique = [&](FractureVec3 point) {
        for (FractureVec3 existing : result.cutPolygonReferenceM) {
            if (fracture_detail::nearPoint(existing, point, pointTolerance)) return;
        }
        result.cutPolygonReferenceM.push_back(point);
    };

    for (std::size_t i = 0; i < 4; ++i) {
        if (side[i] == 0) addUnique(referenceVerticesM[i]);
    }
    constexpr std::array<std::array<std::size_t, 2>, 6> edges{{
        {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}},
    }};
    for (const auto& edge : edges) {
        const std::size_t i = edge[0], j = edge[1];
        if (side[i] * side[j] < 0) {
            const double t = distance[i] / (distance[i] - distance[j]);
            addUnique(referenceVerticesM[i] +
                      (referenceVerticesM[j] - referenceVerticesM[i]) * t);
        }
    }

    if (result.cutPolygonReferenceM.size() < 3u) return result;

    FractureVec3 polygonCenter{};
    for (FractureVec3 point : result.cutPolygonReferenceM)
        polygonCenter = polygonCenter + point;
    polygonCenter = polygonCenter /
                    static_cast<double>(result.cutPolygonReferenceM.size());

    // Construct a deterministic in-plane basis and order the convex cut
    // polygon. The area uses the absolute signed projected shoelace sum.
    const FractureVec3 seed =
        std::abs(normal.x) <= std::abs(normal.y) &&
        std::abs(normal.x) <= std::abs(normal.z) ? FractureVec3{1.0, 0.0, 0.0} :
        (std::abs(normal.y) <= std::abs(normal.z) ? FractureVec3{0.0, 1.0, 0.0} :
                                                   FractureVec3{0.0, 0.0, 1.0});
    FractureVec3 u = fractureCross(normal, seed);
    u = u / fractureNorm(u);
    const FractureVec3 v = fractureCross(normal, u);
    std::sort(result.cutPolygonReferenceM.begin(), result.cutPolygonReferenceM.end(),
              [&](FractureVec3 lhs, FractureVec3 rhs) {
        const FractureVec3 l = lhs - polygonCenter;
        const FractureVec3 r = rhs - polygonCenter;
        return std::atan2(fractureDot(l, v), fractureDot(l, u)) <
               std::atan2(fractureDot(r, v), fractureDot(r, u));
    });

    double twiceArea = 0.0;
    for (std::size_t i = 0; i < result.cutPolygonReferenceM.size(); ++i) {
        const FractureVec3 p = result.cutPolygonReferenceM[i] - polygonCenter;
        const FractureVec3 q = result.cutPolygonReferenceM[
            (i + 1u) % result.cutPolygonReferenceM.size()] - polygonCenter;
        twiceArea += fractureDot(fractureCross(p, q), normal);
    }
    result.referenceAreaM2 = 0.5 * std::abs(twiceArea);
    if (!std::isfinite(result.referenceAreaM2))
        throw std::invalid_argument("tetrahedron crack-plane area is nonfinite");
    const double areaFloor = 128.0 * std::numeric_limits<double>::epsilon() *
                             edgeScale * edgeScale;
    if (result.referenceAreaM2 <= areaFloor) {
        result.referenceAreaM2 = 0.0;
        result.cutPolygonReferenceM.clear();
        return result;
    }
    result.characteristicLengthM = volume / result.referenceAreaM2;
    return result;
}

struct TetFractureEnergyBudget {
    double fractureEnergyReleaseRateJPerM2 = 0.0;
    double referenceCrackAreaM2 = 0.0;
    double characteristicLengthM = 0.0;
    double dissipatedEnergyJ = 0.0;
    double equivalentVolumetricDissipationJPerM3 = 0.0;
};

// For an element-local crack-band law whose dissipated work density integrates
// to Gc/l, the element budget is (Gc/l)*V = Gc*A. Geometry alone supplies A;
// the caller must supply a calibrated/source-supported Gc and damage law.
inline TetFractureEnergyBudget makeTetFractureEnergyBudget(
    const TetCrackPlaneGeometry& geometry,
    double fractureEnergyReleaseRateJPerM2) {
    if (!(std::isfinite(fractureEnergyReleaseRateJPerM2) &&
          fractureEnergyReleaseRateJPerM2 >= 0.0))
        throw std::invalid_argument("fracture energy release rate must be finite and nonnegative");
    if (!(std::isfinite(geometry.referenceVolumeM3) &&
          geometry.referenceVolumeM3 > 0.0 &&
          std::isfinite(geometry.referenceAreaM2) &&
          geometry.referenceAreaM2 >= 0.0))
        throw std::invalid_argument("tet fracture geometry has invalid reference volume/area");
    TetFractureEnergyBudget budget;
    budget.fractureEnergyReleaseRateJPerM2 = fractureEnergyReleaseRateJPerM2;
    budget.referenceCrackAreaM2 = geometry.referenceAreaM2;
    if (geometry.referenceAreaM2 == 0.0) {
        if (geometry.characteristicLengthM)
            throw std::invalid_argument("zero crack area cannot have a characteristic length");
        return budget;
    }
    if (!(geometry.characteristicLengthM &&
          std::isfinite(*geometry.characteristicLengthM) &&
          *geometry.characteristicLengthM > 0.0))
        throw std::invalid_argument("positive crack area requires valid tet length/volume");
    const double expectedLength = geometry.referenceVolumeM3 /
                                  geometry.referenceAreaM2;
    const double lengthTolerance = 256.0 * std::numeric_limits<double>::epsilon() *
        std::max(expectedLength, *geometry.characteristicLengthM);
    if (!std::isfinite(expectedLength) ||
        std::abs(expectedLength - *geometry.characteristicLengthM) > lengthTolerance)
        throw std::invalid_argument("tet characteristic length does not equal V/A");
    budget.characteristicLengthM = *geometry.characteristicLengthM;
    budget.dissipatedEnergyJ = fractureEnergyReleaseRateJPerM2 *
                               geometry.referenceAreaM2;
    budget.equivalentVolumetricDissipationJPerM3 =
        fractureEnergyReleaseRateJPerM2 / *geometry.characteristicLengthM;
    if (!(std::isfinite(budget.dissipatedEnergyJ) &&
          std::isfinite(budget.equivalentVolumetricDissipationJPerM3)))
        throw std::invalid_argument("fracture energy budget overflowed");
    return budget;
}

} // namespace numi::cardboard
