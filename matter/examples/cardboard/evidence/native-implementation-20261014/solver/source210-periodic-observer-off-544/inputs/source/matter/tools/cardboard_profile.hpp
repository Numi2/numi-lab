#pragma once

// Source-parameterized circular-arc A-flute profile.
//
// Source: Nagasawa, Komiyama & Mitsomwang (2013), "Finite Element Analysis of
// Corrugated Board on Rotary Creasing Process", DOI 10.1299/jamdsm.7.103,
// Fig. 8 printed p.107 (r_F=1.7 mm, theta_F=64.8 degrees; lambda=9 mm and
// nominal t=5 mm, t_U=t_L=t_N=0.25 mm from section 2.1 and section 3).
// Interpretation of Fig. 8: one projected
// pitch is a symmetric valley-to-crest-to-valley centerline. Each half-pitch
// contains a radius-r circular transition from horizontal to a straight flank
// at theta, followed by a radius-r transition back to horizontal. The other
// half is its mirror. Thus the figure's r and theta are interpreted as centerline
// fillet radius and straight-flank tangent angle. The paper does not give arc
// endpoints or an explicit closure equation; this is an explicit geometric
// interpretation, not a claim that the paper specifies the construction below.
//
// All lengths use metres. This helper preserves pitch, radius, and angle
// exactly, then reports the resulting stack-caliper closure. It never rescales
// the profile to make a nominal board caliper fit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard {

struct Vec2 {
    double x = 0.0;
    double z = 0.0;
};

enum class ProfileSegment {
    valleyArc,
    straightFlank,
    crestArc,
};

struct ProfilePoint {
    Vec2 centerline;
    double tangentRadians = 0.0;
    ProfileSegment segment = ProfileSegment::valleyArc;
};

struct StackClosure {
    double nominalCoreRise = 0.0;
    double sourceProfileCoreRise = 0.0;
    double nominalBoardCaliper = 0.0;
    double impliedBoardCaliper = 0.0;
    double caliperMismatch = 0.0;

    [[nodiscard]] bool matches(const double absoluteTolerance) const {
        return std::abs(caliperMismatch) <= absoluteTolerance;
    }
};

struct FluteDimensions {
    double pitch = 0.0;
    double radius = 0.0;
    double inclinationDegrees = 0.0;
    double mediumThickness = 0.0;
    double boardCaliper = 0.0;
    double lowerLinerThickness = 0.0;
    double upperLinerThickness = 0.0;
};

class CircularArcFluteProfile {
public:
    explicit CircularArcFluteProfile(const FluteDimensions dimensions)
        : dimensions_(dimensions),
          theta_(dimensions.inclinationDegrees * kPi / 180.0),
          arcHorizontal_(dimensions.radius * std::sin(theta_)),
          flankHorizontal_(0.5 * dimensions.pitch - 2.0 * arcHorizontal_),
          flankLength_(flankHorizontal_ / std::cos(theta_)),
          arcRise_(dimensions.radius * (1.0 - std::cos(theta_))),
          coreRise_(2.0 * arcRise_ + flankLength_ * std::sin(theta_)),
          arcLength_(dimensions.radius * theta_),
          pitchLength_(4.0 * arcLength_ + 2.0 * flankLength_) {
        requireFinitePositive(dimensions_.pitch, "pitch");
        requireFinitePositive(dimensions_.radius, "radius");
        requireFinitePositive(dimensions_.mediumThickness, "medium thickness");
        requireFinitePositive(dimensions_.boardCaliper, "board caliper");
        requireFinitePositive(dimensions_.lowerLinerThickness,
                              "lower liner thickness");
        requireFinitePositive(dimensions_.upperLinerThickness,
                              "upper liner thickness");
        if (!std::isfinite(dimensions_.inclinationDegrees) ||
            !(dimensions_.inclinationDegrees > 0.0 &&
              dimensions_.inclinationDegrees < 90.0)) {
            throw std::invalid_argument(
                "flute inclination must be finite and between 0 and 90 degrees");
        }
        if (!(flankHorizontal_ > 0.0)) {
            throw std::invalid_argument(
                "reported pitch/radius/angle leave no positive straight flank");
        }
        if (!(dimensions_.mediumThickness < 2.0 * dimensions_.radius)) {
            throw std::invalid_argument(
                "normal-offset surfaces are singular: thickness must be less than twice the radius");
        }
        const double nominalCoreRise = dimensions_.boardCaliper -
            dimensions_.lowerLinerThickness -
            dimensions_.upperLinerThickness - dimensions_.mediumThickness;
        if (!(nominalCoreRise > 0.0)) {
            throw std::invalid_argument(
                "nominal stack leaves no positive medium centerline rise");
        }
        closure_ = {
            nominalCoreRise,
            coreRise_,
            dimensions_.boardCaliper,
            coreRise_ + dimensions_.lowerLinerThickness +
                dimensions_.upperLinerThickness + dimensions_.mediumThickness,
            coreRise_ + dimensions_.lowerLinerThickness +
                dimensions_.upperLinerThickness + dimensions_.mediumThickness -
                dimensions_.boardCaliper,
        };
    }

    [[nodiscard]] const FluteDimensions& dimensions() const { return dimensions_; }
    [[nodiscard]] const StackClosure& stackClosure() const { return closure_; }
    [[nodiscard]] double inclinationRadians() const { return theta_; }
    [[nodiscard]] double straightFlankLength() const { return flankLength_; }
    [[nodiscard]] double coreRise() const { return coreRise_; }
    [[nodiscard]] double centerlineArcLengthPerPitch() const {
        return pitchLength_;
    }
    [[nodiscard]] double takeUpRatio() const {
        return pitchLength_ / dimensions_.pitch;
    }

    // Evaluates one periodic valley/crest profile. x is returned unwrapped;
    // z and tangent repeat every projected pitch.
    [[nodiscard]] ProfilePoint sample(const double x) const {
        if (!std::isfinite(x))
            throw std::invalid_argument("profile sample x must be finite");
        double localX = std::fmod(x, dimensions_.pitch);
        if (localX < 0.0) localX += dimensions_.pitch;
        const double halfPitch = 0.5 * dimensions_.pitch;
        const bool descending = localX > halfPitch;
        const double risingX = descending
            ? dimensions_.pitch - localX
            : localX;
        ProfilePoint rising = sampleRising(risingX);
        if (descending) {
            rising.tangentRadians = -rising.tangentRadians;
        }
        rising.centerline.x = x;
        return rising;
    }

    // Signed normal distance from the centerline. Positive points toward the
    // local left-hand normal (-sin(theta), cos(theta)); pass +/- thickness/2
    // for the source medium's two normal-offset faces.
    [[nodiscard]] Vec2 normalOffset(const double x,
                                    const double signedDistance) const {
        if (!std::isfinite(signedDistance))
            throw std::invalid_argument("normal offset must be finite");
        const ProfilePoint point = sample(x);
        return {
            point.centerline.x - signedDistance *
                std::sin(point.tangentRadians),
            point.centerline.z + signedDistance *
                std::cos(point.tangentRadians),
        };
    }

    // With |offset| < r, both parallel-curve curvature factors are positive;
    // with theta < 90 degrees, the projected x derivative also stays positive.
    // These are exact local regularity conditions for the circular arcs and
    // straight flanks. The check executable additionally samples the two full
    // offset polylines and rejects any crossing between their boundaries.
    [[nodiscard]] bool normalOffsetsRegular() const {
        return 0.5 * dimensions_.mediumThickness < dimensions_.radius &&
            std::cos(theta_) > 0.0;
    }

    [[nodiscard]] bool offsetBoundariesIntersect(
        const std::size_t samplesPerPitch = 512u) const {
        if (samplesPerPitch < 8u) return true;
        std::vector<Vec2> lower;
        std::vector<Vec2> upper;
        lower.reserve(samplesPerPitch + 1u);
        upper.reserve(samplesPerPitch + 1u);
        const double halfThickness = 0.5 * dimensions_.mediumThickness;
        for (std::size_t index = 0; index <= samplesPerPitch; ++index) {
            const double x = dimensions_.pitch *
                static_cast<double>(index) /
                static_cast<double>(samplesPerPitch);
            lower.push_back(normalOffset(x, -halfThickness));
            upper.push_back(normalOffset(x, halfThickness));
        }
        for (std::size_t i = 0; i < samplesPerPitch; ++i) {
            for (std::size_t j = 0; j < samplesPerPitch; ++j) {
                if (segmentsIntersect(lower[i], lower[i + 1u],
                                      upper[j], upper[j + 1u]))
                    return true;
            }
        }
        return false;
    }

private:
    static constexpr double kPi = 3.141592653589793238462643383279502884;

    static void requireFinitePositive(const double value, const char* label) {
        if (!std::isfinite(value) || !(value > 0.0))
            throw std::invalid_argument(std::string(label) +
                                        " must be finite and positive");
    }

    [[nodiscard]] ProfilePoint sampleRising(const double x) const {
        const double arcEndX = arcHorizontal_;
        const double flankEndX = arcHorizontal_ + flankHorizontal_;
        const double clampedX = std::clamp(x, 0.0, 0.5 * dimensions_.pitch);
        if (clampedX <= arcEndX) {
            const double angle = std::asin(std::clamp(
                clampedX / dimensions_.radius, -1.0, 1.0));
            return {{clampedX,
                     dimensions_.radius * (1.0 - std::cos(angle))},
                    angle, ProfileSegment::valleyArc};
        }
        if (clampedX <= flankEndX) {
            const double alongFlank =
                (clampedX - arcEndX) / std::cos(theta_);
            return {{clampedX,
                     arcRise_ + alongFlank * std::sin(theta_)},
                    theta_, ProfileSegment::straightFlank};
        }
        const double alongCrestArc = clampedX - flankEndX;
        const double sinAngle = std::clamp(
            std::sin(theta_) - alongCrestArc / dimensions_.radius,
            0.0, std::sin(theta_));
        const double angle = std::asin(sinAngle);
        const double flankEndZ = arcRise_ +
            flankLength_ * std::sin(theta_);
        const double z = flankEndZ + dimensions_.radius *
            (std::cos(angle) - std::cos(theta_));
        return {{clampedX, z}, angle, ProfileSegment::crestArc};
    }

    static double orient(const Vec2& a, const Vec2& b, const Vec2& c) {
        return (b.x - a.x) * (c.z - a.z) -
            (b.z - a.z) * (c.x - a.x);
    }

    static bool onSegment(const Vec2& a, const Vec2& b, const Vec2& p) {
        constexpr double tolerance = 1.0e-15;
        return p.x >= std::min(a.x, b.x) - tolerance &&
            p.x <= std::max(a.x, b.x) + tolerance &&
            p.z >= std::min(a.z, b.z) - tolerance &&
            p.z <= std::max(a.z, b.z) + tolerance;
    }

    static bool segmentsIntersect(const Vec2& a, const Vec2& b,
                                  const Vec2& c, const Vec2& d) {
        constexpr double tolerance = 1.0e-18;
        const double o1 = orient(a, b, c);
        const double o2 = orient(a, b, d);
        const double o3 = orient(c, d, a);
        const double o4 = orient(c, d, b);
        if (((o1 > tolerance && o2 < -tolerance) ||
             (o1 < -tolerance && o2 > tolerance)) &&
            ((o3 > tolerance && o4 < -tolerance) ||
             (o3 < -tolerance && o4 > tolerance)))
            return true;
        if (std::abs(o1) <= tolerance && onSegment(a, b, c)) return true;
        if (std::abs(o2) <= tolerance && onSegment(a, b, d)) return true;
        if (std::abs(o3) <= tolerance && onSegment(c, d, a)) return true;
        if (std::abs(o4) <= tolerance && onSegment(c, d, b)) return true;
        return false;
    }

    FluteDimensions dimensions_;
    double theta_ = 0.0;
    double arcHorizontal_ = 0.0;
    double flankHorizontal_ = 0.0;
    double flankLength_ = 0.0;
    double arcRise_ = 0.0;
    double coreRise_ = 0.0;
    double arcLength_ = 0.0;
    double pitchLength_ = 0.0;
    StackClosure closure_;
};

} // namespace numi::cardboard
