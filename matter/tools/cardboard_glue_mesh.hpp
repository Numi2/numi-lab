#pragma once

// Conforming 2D cross-section generator for a sinusoidal, finite-glue
// corrugated strip. This is an explicit fallback geometry, not a source-arc
// profile and not a calibrated adhesive model. Lengths are SI metres.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::cardboard {

enum class CellMaterial : std::uint32_t {
    liner = 0u,
    medium = 1u,
    glue = 2u,
};

struct GlueMeshConfig {
    double length = 0.0;
    double pitch = 0.0;
    double caliper = 0.0;
    double linerThickness = 0.0;
    double mediumThickness = 0.0;
    double glueMinimumGap = 0.0; // lower face; upper defaults to this value
    double bondWidth = 0.0; // lower horizontal footprint; upper defaults to this
    double upperGlueMinimumGap = 0.0; // zero selects glueMinimumGap
    double upperBondWidth = 0.0; // zero selects bondWidth
    std::uint32_t nxPerPitch = 0u;
    std::uint32_t thicknessSlices = 0u;
};

struct GlueMeshPoint {
    double x = 0.0;
    double z = 0.0;
};

struct GlueMeshTriangle {
    std::array<std::uint32_t, 3> points{};
    CellMaterial material = CellMaterial::liner;
    double frameAngle = 0.0;
};

struct GlueFootprint {
    bool crest = false;
    std::uint32_t period = 0u;
    double centerX = 0.0;
    double requestedHorizontalWidth = 0.0;
    double actualHorizontalWidth = 0.0;
    double centerlineParameterWidth = 0.0;
    double xLeft = 0.0;
    double xRight = 0.0;
    double parameterLeft = 0.0;
    double parameterRight = 0.0;
    double crossSectionArea = 0.0;
    double volumePerUnitBoardWidth = 0.0; // m^3 per m of board width
    std::uint64_t firstTriangle = 0u;
    std::uint64_t triangleCount = 0u;
};

struct ConformingGlueMesh {
    GlueMeshConfig config;
    std::vector<GlueMeshPoint> points;
    std::vector<GlueMeshTriangle> triangles;
    std::vector<GlueFootprint> glueFootprints;
    std::array<double, 3> crossSectionAreaByMaterial{};
    double mediumCenterlineRise = 0.0;
    double mediumAmplitude = 0.0;
    double centerlineTakeUpRatio = 0.0;
    double maximumCurvature = 0.0;
    std::string profileDescription =
        "sinusoidal fallback; source-arc profile not represented";
};

namespace cardboard_glue_detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kMergeScale = 1.0e12; // 1 pm coordinate quantization

inline bool nearEqual(double a, double b, double tolerance);

struct ProfileSample {
    GlueMeshPoint center;
    double tangent = 0.0;
};

struct PatchDefinition {
    bool crest = false;
    std::uint32_t period = 0u;
    double center = 0.0;
    double xLeft = 0.0;
    double xRight = 0.0;
    double uLeft = 0.0;
    double uRight = 0.0;
    double normalOffset = 0.0;
};

inline double resolvedUpperGlueGap(const GlueMeshConfig& config) {
    return config.upperGlueMinimumGap > 0.0
        ? config.upperGlueMinimumGap : config.glueMinimumGap;
}

inline double resolvedUpperBondWidth(const GlueMeshConfig& config) {
    return config.upperBondWidth > 0.0
        ? config.upperBondWidth : config.bondWidth;
}

inline void requirePositive(const double value, const char* name) {
    if (!std::isfinite(value) || !(value > 0.0))
        throw std::invalid_argument(std::string(name) +
                                    " must be finite and positive");
}

class SinusoidalProfile {
public:
    explicit SinusoidalProfile(const GlueMeshConfig& config)
        : config_(config),
          rise_(config.caliper - 2.0 * config.linerThickness -
                config.mediumThickness - config.glueMinimumGap -
                resolvedUpperGlueGap(config)),
          amplitude_(0.5 * rise_),
          midline_(0.5 * (config.linerThickness + config.glueMinimumGap +
                          0.5 * config.mediumThickness +
                          config.caliper - config.linerThickness -
                          resolvedUpperGlueGap(config) -
                          0.5 * config.mediumThickness)),
          waveNumber_(2.0 * kPi / config.pitch),
          maxCurvature_(amplitude_ * waveNumber_ * waveNumber_) {}

    [[nodiscard]] double rise() const { return rise_; }
    [[nodiscard]] double amplitude() const { return amplitude_; }
    [[nodiscard]] double maxCurvature() const { return maxCurvature_; }
    [[nodiscard]] double midline() const { return midline_; }
    [[nodiscard]] double centerlineTakeUpRatio() const {
        // Simpson integration with deterministic bounded sampling. This is a
        // useful geometry descriptor, not the paper grammage correction.
        constexpr std::uint32_t intervals = 2048u;
        const double h = config_.pitch / intervals;
        double sum = 1.0 + std::hypot(1.0, slope(config_.pitch));
        for (std::uint32_t i = 1u; i < intervals; ++i) {
            const double weight = i % 2u == 0u ? 2.0 : 4.0;
            sum += weight * std::hypot(1.0, slope(h * i));
        }
        const double arclength = h * sum / 3.0;
        return arclength / config_.pitch;
    }

    [[nodiscard]] double slope(const double x) const {
        return amplitude_ * waveNumber_ * std::sin(waveNumber_ * x);
    }

    [[nodiscard]] double curvature(const double x) const {
        const double first = slope(x);
        const double second = amplitude_ * waveNumber_ * waveNumber_ *
            std::cos(waveNumber_ * x);
        return second / std::pow(1.0 + first * first, 1.5);
    }

    [[nodiscard]] ProfileSample sample(const double x) const {
        const double z = midline_ - amplitude_ * std::cos(waveNumber_ * x);
        return {{x, z}, std::atan(slope(x))};
    }

    [[nodiscard]] GlueMeshPoint offset(const double x,
                                       const double signedDistance) const {
        const ProfileSample center = sample(x);
        return {
            x - signedDistance * std::sin(center.tangent),
            center.center.z + signedDistance * std::cos(center.tangent),
        };
    }

    [[nodiscard]] double offsetX(const double x,
                                 const double signedDistance) const {
        return offset(x, signedDistance).x;
    }

    [[nodiscard]] const GlueMeshConfig& config() const { return config_; }

private:
    GlueMeshConfig config_;
    double rise_ = 0.0;
    double amplitude_ = 0.0;
    double midline_ = 0.0;
    double waveNumber_ = 0.0;
    double maxCurvature_ = 0.0;
};

inline double cross(const GlueMeshPoint& a, const GlueMeshPoint& b,
                    const GlueMeshPoint& c) {
    return (b.x - a.x) * (c.z - a.z) -
        (b.z - a.z) * (c.x - a.x);
}

struct PointKey {
    std::int64_t x = 0;
    std::int64_t z = 0;
    friend bool operator<(const PointKey& a, const PointKey& b) {
        return a.x < b.x || (a.x == b.x && a.z < b.z);
    }
};

class PointRegistry {
public:
    std::uint32_t add(const GlueMeshPoint& point) {
        if (!std::isfinite(point.x) || !std::isfinite(point.z) ||
            std::abs(point.x) > 9.0e6 || std::abs(point.z) > 9.0e6)
            throw std::invalid_argument("mesh point is nonfinite or out of range");
        const PointKey key{
            static_cast<std::int64_t>(std::llround(point.x * kMergeScale)),
            static_cast<std::int64_t>(std::llround(point.z * kMergeScale)),
        };
        const auto found = indices_.find(key);
        if (found != indices_.end()) return found->second;
        if (points_.size() >= std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("cross-section node index exceeds uint32");
        const std::uint32_t index = static_cast<std::uint32_t>(points_.size());
        indices_.emplace(key, index);
        points_.push_back(point);
        return index;
    }

    [[nodiscard]] const std::vector<GlueMeshPoint>& points() const {
        return points_;
    }

private:
    std::map<PointKey, std::uint32_t> indices_;
    std::vector<GlueMeshPoint> points_;
};

inline std::vector<double> sortedUnique(std::vector<double> values,
                                        const double tolerance = 1.0e-13) {
    std::sort(values.begin(), values.end());
    std::vector<double> result;
    result.reserve(values.size());
    for (const double value : values) {
        if (result.empty() || std::abs(value - result.back()) > tolerance)
            result.push_back(value);
    }
    return result;
}

inline void addTriangle(
    const std::array<std::uint32_t, 3>& nodes,
    const CellMaterial material,
    const double frameAngle,
    const std::vector<GlueMeshPoint>& points,
    std::vector<GlueMeshTriangle>& triangles,
    std::array<double, 3>& areas) {
    std::array<std::uint32_t, 3> oriented = nodes;
    double signedArea2 = cross(points[oriented[0]], points[oriented[1]],
                               points[oriented[2]]);
    constexpr double minimumArea2 = 1.0e-22;
    if (!std::isfinite(signedArea2) || std::abs(signedArea2) <= minimumArea2)
        throw std::runtime_error("cross-section contains a zero-area triangle");
    if (signedArea2 < 0.0) {
        std::swap(oriented[1], oriented[2]);
        signedArea2 = -signedArea2;
    }
    const std::size_t materialIndex = static_cast<std::size_t>(material);
    areas.at(materialIndex) += 0.5 * signedArea2;
    triangles.push_back({oriented, material, frameAngle});
}

inline double addQuad(
    const std::array<std::uint32_t, 4>& nodes,
    const CellMaterial material,
    const double frameAngle,
    const std::vector<GlueMeshPoint>& points,
    std::vector<GlueMeshTriangle>& triangles,
    std::array<double, 3>& areas) {
    const double polygonArea2 =
        points[nodes[0]].x * points[nodes[1]].z -
        points[nodes[1]].x * points[nodes[0]].z +
        points[nodes[1]].x * points[nodes[2]].z -
        points[nodes[2]].x * points[nodes[1]].z +
        points[nodes[2]].x * points[nodes[3]].z -
        points[nodes[3]].x * points[nodes[2]].z +
        points[nodes[3]].x * points[nodes[0]].z -
        points[nodes[0]].x * points[nodes[3]].z;
    const double diagonal02a = cross(points[nodes[0]], points[nodes[1]],
                                     points[nodes[2]]);
    const double diagonal02b = cross(points[nodes[0]], points[nodes[2]],
                                     points[nodes[3]]);
    const double diagonal13a = cross(points[nodes[0]], points[nodes[1]],
                                     points[nodes[3]]);
    const double diagonal13b = cross(points[nodes[1]], points[nodes[2]],
                                     points[nodes[3]]);
    constexpr double tolerance = 1.0e-22;
    const auto validDiagonal = [polygonArea2](const double a,
                                             const double b) {
        return std::isfinite(polygonArea2) && std::abs(polygonArea2) > tolerance &&
            a * polygonArea2 > tolerance * std::abs(polygonArea2) &&
            b * polygonArea2 > tolerance * std::abs(polygonArea2);
    };
    double area = 0.0;
    if (validDiagonal(diagonal02a, diagonal02b)) {
        addTriangle({nodes[0], nodes[1], nodes[2]}, material, frameAngle,
                    points, triangles, areas);
        addTriangle({nodes[0], nodes[2], nodes[3]}, material, frameAngle,
                    points, triangles, areas);
        area = 0.5 * (std::abs(diagonal02a) + std::abs(diagonal02b));
    } else if (validDiagonal(diagonal13a, diagonal13b)) {
        addTriangle({nodes[0], nodes[1], nodes[3]}, material, frameAngle,
                    points, triangles, areas);
        addTriangle({nodes[1], nodes[2], nodes[3]}, material, frameAngle,
                    points, triangles, areas);
        area = 0.5 * (std::abs(diagonal13a) + std::abs(diagonal13b));
    } else {
        throw std::runtime_error(
            "cross-section quad is degenerate or cannot be triangulated without inversion");
    }
    return area;
}

inline double inverseOffsetX(const SinusoidalProfile& profile,
                             const double targetX,
                             const double center,
                             const double offset) {
    if (std::abs(targetX - center) <= 2.0e-15) return center;
    double lower = targetX < center
        ? center - 0.5 * profile.config().pitch
        : center;
    double upper = targetX < center
        ? center
        : center + 0.5 * profile.config().pitch;
    const double lowerValue = profile.offsetX(lower, offset);
    const double upperValue = profile.offsetX(upper, offset);
    if (targetX < lowerValue - 1.0e-13 ||
        targetX > upperValue + 1.0e-13)
        throw std::invalid_argument(
            "bond edge lies outside the invertible neighboring half-pitch");
    for (unsigned iteration = 0u; iteration < 64u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        const double value = profile.offsetX(middle, offset);
        if (value < targetX) lower = middle;
        else upper = middle;
    }
    return 0.5 * (lower + upper);
}

inline void validateConfig(const GlueMeshConfig& config) {
    requirePositive(config.length, "length");
    requirePositive(config.pitch, "pitch");
    requirePositive(config.caliper, "caliper");
    requirePositive(config.linerThickness, "liner thickness");
    requirePositive(config.mediumThickness, "medium thickness");
    requirePositive(config.glueMinimumGap, "glue minimum gap");
    requirePositive(config.bondWidth, "bond width");
    if (!std::isfinite(config.upperGlueMinimumGap) ||
        config.upperGlueMinimumGap < 0.0)
        throw std::invalid_argument(
            "upper glue minimum gap must be finite and nonnegative");
    if (!std::isfinite(config.upperBondWidth) || config.upperBondWidth < 0.0)
        throw std::invalid_argument(
            "upper bond width must be finite and nonnegative");
    const double upperGap = config.upperGlueMinimumGap > 0.0
        ? config.upperGlueMinimumGap : config.glueMinimumGap;
    const double upperWidth = config.upperBondWidth > 0.0
        ? config.upperBondWidth : config.bondWidth;
    if (config.nxPerPitch < 4u || config.nxPerPitch > 2048u)
        throw std::invalid_argument("nxPerPitch must lie in [4, 2048]");
    if (config.thicknessSlices == 0u || config.thicknessSlices > 128u)
        throw std::invalid_argument("thicknessSlices must lie in [1, 128]");
    const double pitchCount = config.length / config.pitch;
    const double rounded = std::round(pitchCount);
    if (rounded < 1.0 || rounded > 128.0 ||
        std::abs(pitchCount - rounded) > 1.0e-10)
        throw std::invalid_argument(
            "length must be an integer number of pitches in [1, 128]");
    if (!(config.bondWidth < 0.5 * config.pitch) ||
        !(upperWidth < 0.5 * config.pitch))
        throw std::invalid_argument(
            "lower and upper bond widths must each be less than half pitch so alternating footprints do not overlap");
    if (!(config.caliper > 2.0 * config.linerThickness +
          config.mediumThickness + config.glueMinimumGap + upperGap))
        throw std::invalid_argument(
            "caliper leaves no positive sinusoidal medium centerline rise");
    const double rise = config.caliper - 2.0 * config.linerThickness -
        config.mediumThickness - config.glueMinimumGap - upperGap;
    const double amplitude = 0.5 * rise;
    const double waveNumber = 2.0 * kPi / config.pitch;
    const double maximumCurvature = amplitude * waveNumber * waveNumber;
    if (!(0.5 * config.mediumThickness * maximumCurvature < 1.0))
        throw std::invalid_argument(
            "medium normal offset is singular or folds at the sinusoidal curvature");
    const std::uint64_t intervals = static_cast<std::uint64_t>(rounded) *
        config.nxPerPitch;
    if (intervals * config.thicknessSlices > 2000000ull)
        throw std::invalid_argument("requested cross-section exceeds cell bound");
}

inline std::vector<PatchDefinition> patchesFor(
    const GlueMeshConfig& config,
    const SinusoidalProfile& profile) {
    const auto pitchCount = static_cast<std::uint32_t>(
        std::llround(config.length / config.pitch));
    std::vector<PatchDefinition> patches;
    patches.reserve(2u * pitchCount + 1u);
    const auto addPatch = [&](const bool crest, const std::uint32_t period,
                              const double center, const double offset) {
        const double bondWidth = crest ? resolvedUpperBondWidth(config)
                                       : config.bondWidth;
        const double xLeft = std::max(0.0, center - 0.5 * bondWidth);
        const double xRight = std::min(config.length,
                                       center + 0.5 * bondWidth);
        if (!(xRight > xLeft)) return;
        const double uLeft = inverseOffsetX(profile, xLeft, center, offset);
        const double uRight = inverseOffsetX(profile, xRight, center, offset);
        if (!(uRight > uLeft))
            throw std::runtime_error("bond footprint inverse has nonpositive width");
        patches.push_back({crest, period, center, xLeft, xRight,
                           uLeft, uRight, offset});
    };
    for (std::uint32_t period = 0u; period < pitchCount; ++period) {
        const double valley = static_cast<double>(period) * config.pitch;
        addPatch(false, period, valley, -0.5 * config.mediumThickness);
        const double crest = valley + 0.5 * config.pitch;
        addPatch(true, period, crest, 0.5 * config.mediumThickness);
    }
    const double terminalValley = config.length;
    addPatch(false, pitchCount, terminalValley,
             -0.5 * config.mediumThickness);
    return patches;
}

} // namespace cardboard_glue_detail

inline ConformingGlueMesh buildConformingGlueCrossSection(
    const GlueMeshConfig& config) {
    using namespace cardboard_glue_detail;
    validateConfig(config);
    const SinusoidalProfile profile(config);
    ConformingGlueMesh mesh;
    mesh.config = config;
    mesh.mediumCenterlineRise = profile.rise();
    mesh.mediumAmplitude = profile.amplitude();
    mesh.centerlineTakeUpRatio = profile.centerlineTakeUpRatio();
    mesh.maximumCurvature = profile.maxCurvature();

    const auto patches = patchesFor(config, profile);
    const auto pitchCount = static_cast<std::uint32_t>(
        std::llround(config.length / config.pitch));
    std::vector<double> regularXs;
    const std::uint64_t intervalCount =
        static_cast<std::uint64_t>(pitchCount) * config.nxPerPitch;
    regularXs.reserve(static_cast<std::size_t>(intervalCount + 1u));
    for (std::uint64_t index = 0u; index <= intervalCount; ++index)
        regularXs.push_back(config.length * static_cast<double>(index) /
                            static_cast<double>(intervalCount));

    std::vector<double> centerlineXs = regularXs;
    for (const PatchDefinition& patch : patches) {
        centerlineXs.push_back(patch.uLeft);
        centerlineXs.push_back(patch.uRight);
    }
    centerlineXs = sortedUnique(std::move(centerlineXs));

    // The liner's flat bond face receives every curve-boundary abscissa. This
    // creates a one-to-one segmented interface edge on both materials.
    std::vector<double> linerXs = regularXs;
    // Regular samples inside a patch would subdivide the flat liner face
    // without subdividing the matching adhesive edge, creating T-junctions.
    // Keep the regular grid outside patches and insert every actual offset
    // surface abscissa below, so both sides of each bonded interface agree.
    linerXs.erase(std::remove_if(linerXs.begin(), linerXs.end(),
        [&](const double x) {
            for (const PatchDefinition& patch : patches) {
                const double left = profile.offsetX(patch.uLeft,
                                                     patch.normalOffset);
                const double right = profile.offsetX(patch.uRight,
                                                      patch.normalOffset);
                if (x > left + 1.0e-13 && x < right - 1.0e-13) return true;
            }
            return false;
        }), linerXs.end());
    for (const PatchDefinition& patch : patches) {
        for (const double u : centerlineXs) {
            if (u < patch.uLeft - 1.0e-13 || u > patch.uRight + 1.0e-13)
                continue;
            linerXs.push_back(profile.offsetX(u, patch.normalOffset));
        }
        linerXs.push_back(patch.xLeft);
        linerXs.push_back(patch.xRight);
    }
    linerXs = sortedUnique(std::move(linerXs));

    PointRegistry registry;
    std::vector<GlueMeshTriangle> triangles;
    std::array<double, 3> areas{};
    const auto pointAt = [&](const double x, const double z) {
        return registry.add({x, z});
    };

    // Two disconnected flat liners. They acquire connectivity to the core
    // only across the finite glue patch faces generated below.
    for (std::size_t xIndex = 0u; xIndex + 1u < linerXs.size(); ++xIndex) {
        const double x0 = linerXs[xIndex];
        const double x1 = linerXs[xIndex + 1u];
        for (std::uint32_t layer = 0u;
             layer < config.thicknessSlices; ++layer) {
            const double a0 = static_cast<double>(layer) /
                static_cast<double>(config.thicknessSlices);
            const double a1 = static_cast<double>(layer + 1u) /
                static_cast<double>(config.thicknessSlices);
            const double bottomZ0 = a0 * config.linerThickness;
            const double bottomZ1 = a1 * config.linerThickness;
            const double topBase = config.caliper - config.linerThickness;
            const double topZ0 = topBase + a0 * config.linerThickness;
            const double topZ1 = topBase + a1 * config.linerThickness;
            const auto bottomQuad = std::array<std::uint32_t, 4>{
                pointAt(x0, bottomZ0), pointAt(x1, bottomZ0),
                pointAt(x1, bottomZ1), pointAt(x0, bottomZ1)};
            const auto topQuad = std::array<std::uint32_t, 4>{
                pointAt(x0, topZ0), pointAt(x1, topZ0),
                pointAt(x1, topZ1), pointAt(x0, topZ1)};
            addQuad(bottomQuad, CellMaterial::liner, 0.0,
                    registry.points(), triangles, areas);
            addQuad(topQuad, CellMaterial::liner, 0.0,
                    registry.points(), triangles, areas);
        }
    }

    // Normal-thickness sinusoidal medium. Samples added at each inverse-offset
    // bond edge make its curved interface segments available to the glue mesh.
    std::vector<std::vector<std::uint32_t>> mediumNodes(
        centerlineXs.size(),
        std::vector<std::uint32_t>(config.thicknessSlices + 1u));
    for (std::size_t xIndex = 0u; xIndex < centerlineXs.size(); ++xIndex) {
        const double u = centerlineXs[xIndex];
        for (std::uint32_t layer = 0u;
             layer <= config.thicknessSlices; ++layer) {
            const double fraction = static_cast<double>(layer) /
                static_cast<double>(config.thicknessSlices);
            const double offset =
                (fraction - 0.5) * config.mediumThickness;
            const GlueMeshPoint point = profile.offset(u, offset);
            mediumNodes[xIndex][layer] = registry.add(point);
        }
    }
    for (std::size_t xIndex = 0u; xIndex + 1u < centerlineXs.size(); ++xIndex) {
        const double u0 = centerlineXs[xIndex];
        const double u1 = centerlineXs[xIndex + 1u];
        const double frame = std::atan(profile.slope(0.5 * (u0 + u1)));
        for (std::uint32_t layer = 0u;
             layer < config.thicknessSlices; ++layer) {
            const std::array<std::uint32_t, 4> quad{
                mediumNodes[xIndex][layer],
                mediumNodes[xIndex + 1u][layer],
                mediumNodes[xIndex + 1u][layer + 1u],
                mediumNodes[xIndex][layer + 1u],
            };
            addQuad(quad, CellMaterial::medium, frame,
                    registry.points(), triangles, areas);
        }
    }

    // Fill a positive-thickness adhesive wedge only at the finite-width crest
    // and valley footprints. Its curved boundary reuses medium node ids; its
    // flat boundary reuses liner surface node ids.
    for (const PatchDefinition& patch : patches) {
        const std::size_t surfaceLayer = patch.crest
            ? config.thicknessSlices
            : 0u;
        std::vector<std::size_t> indices;
        for (std::size_t index = 0u; index < centerlineXs.size(); ++index) {
            const double u = centerlineXs[index];
            if (u >= patch.uLeft - 1.0e-13 &&
                u <= patch.uRight + 1.0e-13)
                indices.push_back(index);
        }
        if (indices.size() < 2u)
            throw std::runtime_error("glue patch has fewer than two core-face nodes");

        const double linerFace = patch.crest
            ? config.caliper - config.linerThickness
            : config.linerThickness;
        std::vector<double> surfaceX;
        surfaceX.reserve(indices.size());
        for (const std::size_t index : indices) {
            const auto& corePoint = registry.points().at(
                mediumNodes[index][surfaceLayer]);
            surfaceX.push_back(corePoint.x);
        }
        if (!nearEqual(surfaceX.front(), patch.xLeft, 2.0e-12) ||
            !nearEqual(surfaceX.back(), patch.xRight, 2.0e-12))
            throw std::runtime_error("inverse-offset glue endpoints lost footprint closure");
        for (std::size_t index = 0u; index + 1u < surfaceX.size(); ++index)
            if (!(surfaceX[index + 1u] > surfaceX[index]))
                throw std::runtime_error("normal-offset glue boundary is not x-monotone");

        GlueFootprint footprint;
        footprint.crest = patch.crest;
        footprint.period = patch.period;
        footprint.centerX = patch.center;
        footprint.requestedHorizontalWidth = patch.crest
            ? resolvedUpperBondWidth(config) : config.bondWidth;
        footprint.actualHorizontalWidth = patch.xRight - patch.xLeft;
        footprint.centerlineParameterWidth = patch.uRight - patch.uLeft;
        footprint.xLeft = patch.xLeft;
        footprint.xRight = patch.xRight;
        footprint.parameterLeft = patch.uLeft;
        footprint.parameterRight = patch.uRight;
        footprint.firstTriangle = triangles.size();
        const std::size_t materialIndex =
            static_cast<std::size_t>(CellMaterial::glue);
        const double areaBefore = areas[materialIndex];

        for (std::size_t interval = 0u;
             interval + 1u < indices.size(); ++interval) {
            const std::size_t i0 = indices[interval];
            const std::size_t i1 = indices[interval + 1u];
            const double x0 = surfaceX[interval];
            const double x1 = surfaceX[interval + 1u];
            const double z0 = registry.points().at(
                mediumNodes[i0][surfaceLayer]).z;
            const double z1 = registry.points().at(
                mediumNodes[i1][surfaceLayer]).z;
            const double uMid = 0.5 * (centerlineXs[i0] + centerlineXs[i1]);
            const double frame = std::atan(profile.slope(uMid));
            std::vector<std::array<std::uint32_t, 2>> glueRows(
                config.thicknessSlices + 1u);
            for (std::uint32_t layer = 0u;
                 layer <= config.thicknessSlices; ++layer) {
                const double alpha = static_cast<double>(layer) /
                    static_cast<double>(config.thicknessSlices);
                const double leftZ = (1.0 - alpha) * z0 + alpha * linerFace;
                const double rightZ = (1.0 - alpha) * z1 + alpha * linerFace;
                if (patch.crest && (leftZ > linerFace + 1.0e-14 ||
                                    rightZ > linerFace + 1.0e-14))
                    throw std::runtime_error("crest glue overlaps the top liner");
                if (!patch.crest && (leftZ < linerFace - 1.0e-14 ||
                                     rightZ < linerFace - 1.0e-14))
                    throw std::runtime_error("valley glue overlaps the bottom liner");
                const auto nodeAt = [&](const std::size_t i,
                                        const std::size_t coreIndex) {
                    if (layer == 0u) return mediumNodes[coreIndex][surfaceLayer];
                    if (layer == config.thicknessSlices)
                        return registry.add({i == 0u ? x0 : x1, linerFace});
                    return registry.add({i == 0u ? x0 : x1,
                                         i == 0u ? leftZ : rightZ});
                };
                glueRows[layer][0] = nodeAt(0u, i0);
                glueRows[layer][1] = nodeAt(1u, i1);
            }
            for (std::uint32_t layer = 0u;
                 layer < config.thicknessSlices; ++layer) {
                const std::array<std::uint32_t, 4> quad{
                    glueRows[layer][0], glueRows[layer][1],
                    glueRows[layer + 1u][1], glueRows[layer + 1u][0],
                };
                footprint.crossSectionArea += addQuad(
                    quad, CellMaterial::glue, frame,
                    registry.points(), triangles, areas);
            }
        }
        footprint.crossSectionArea = areas[materialIndex] - areaBefore;
        footprint.volumePerUnitBoardWidth = footprint.crossSectionArea;
        footprint.triangleCount = triangles.size() - footprint.firstTriangle;
        if (!(footprint.crossSectionArea > 0.0) ||
            footprint.triangleCount == 0u)
            throw std::runtime_error("glue footprint has zero area");
        mesh.glueFootprints.push_back(footprint);
    }

    mesh.points = registry.points();
    mesh.triangles = std::move(triangles);
    mesh.crossSectionAreaByMaterial = areas;
    return mesh;
}

inline bool cardboard_glue_detail::nearEqual(const double a, const double b,
                                             const double tolerance) {
    return std::abs(a - b) <= tolerance;
}

} // namespace numi::cardboard
