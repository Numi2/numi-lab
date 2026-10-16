#include "cardboard_profile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using numi::cardboard::CircularArcFluteProfile;
using numi::cardboard::FluteDimensions;
using numi::cardboard::ProfileSegment;

unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

bool near(const double actual, const double expected,
          const double absoluteTolerance) {
    return std::abs(actual - expected) <= absoluteTolerance;
}

double distance(const numi::cardboard::Vec2& a,
                const numi::cardboard::Vec2& b) {
    return std::hypot(a.x - b.x, a.z - b.z);
}

double derivativeZ(const CircularArcFluteProfile& profile,
                   const double x, const double epsilon) {
    return (profile.sample(x + epsilon).centerline.z -
            profile.sample(x - epsilon).centerline.z) / (2.0 * epsilon);
}

void checkSourceFlute() {
    constexpr double mm = 1.0e-3;
    constexpr double radians = 3.141592653589793238462643383279502884 / 180.0;
    const FluteDimensions source{
        .pitch = 9.0 * mm,
        .radius = 1.7 * mm,
        .inclinationDegrees = 64.8,
        .mediumThickness = 0.25 * mm,
        .boardCaliper = 5.0 * mm,
        .lowerLinerThickness = 0.25 * mm,
        .upperLinerThickness = 0.25 * mm,
    };
    const CircularArcFluteProfile profile(source);
    const auto& closure = profile.stackClosure();
    constexpr double theta = 64.8 * radians;
    const double arcX = source.radius * std::sin(theta);
    const double flankHorizontal = 0.5 * source.pitch - 2.0 * arcX;
    const double flankLength = flankHorizontal / std::cos(theta);
    const double sourceRise = 2.0 * source.radius * (1.0 - std::cos(theta)) +
        flankLength * std::sin(theta);
    const double sourcePathLength = 4.0 * source.radius * theta +
        2.0 * flankLength;

    require(profile.normalOffsetsRegular(),
            "reported sheet thickness makes the source normal offsets singular");
    require(!profile.offsetBoundariesIntersect(512u),
            "sampled upper/lower normal-offset surfaces intersect");
    require(near(profile.straightFlankLength(), flankLength, 1.0e-12),
            "straight flank length does not close the projected half pitch");
    require(near(profile.coreRise(), sourceRise, 1.0e-12),
            "circular-arc centerline rise differs from the geometric construction");
    require(near(profile.centerlineArcLengthPerPitch(), sourcePathLength,
                 1.0e-12),
            "centerline arclength does not sum the four arcs and two flanks");
    require(near(profile.takeUpRatio(), sourcePathLength / source.pitch, 1.0e-12),
            "medium take-up ratio is inconsistent with source profile arclength");

    const double halfPitch = 0.5 * source.pitch;
    const double flankEndX = arcX + flankHorizontal;
    const auto valley = profile.sample(0.0);
    const auto arcEnd = profile.sample(arcX);
    const auto flankEnd = profile.sample(flankEndX);
    const auto crest = profile.sample(halfPitch);
    const auto nextValley = profile.sample(source.pitch);
    require(near(valley.centerline.z, 0.0, 1.0e-15) &&
                near(valley.tangentRadians, 0.0, 1.0e-15),
            "valley must start at zero height with a horizontal tangent");
    require(near(arcEnd.tangentRadians, theta, 1.0e-12) &&
                near(arcEnd.centerline.z,
                     source.radius * (1.0 - std::cos(theta)), 1.0e-12),
            "valley radius transition does not end at the reported flank angle");
    require(near(flankEnd.tangentRadians, theta, 1.0e-12),
            "straight flank must preserve the reported tangent angle");
    require(near(crest.centerline.z, sourceRise, 1.0e-12) &&
                near(crest.tangentRadians, 0.0, 1.0e-12),
            "crest arc must close at the half pitch with horizontal tangent");
    require(near(nextValley.centerline.z, 0.0, 1.0e-12) &&
                near(nextValley.tangentRadians, 0.0, 1.0e-12),
            "one projected pitch must close at the next valley");

    const std::array<double, 4> interiorSamples{
        0.5 * arcX,
        arcX + 0.4 * flankHorizontal,
        flankEndX + 0.5 * arcX,
        source.pitch - 0.5 * arcX,
    };
    constexpr double derivativeEpsilon = 1.0e-8;
    for (const double x : interiorSamples) {
        const auto point = profile.sample(x);
        const double numeric = derivativeZ(profile, x, derivativeEpsilon);
        require(near(numeric, std::tan(point.tangentRadians), 2.0e-6),
                "reported local tangent does not match the profile derivative");
        const double normalX = -std::sin(point.tangentRadians);
        const double normalZ = std::cos(point.tangentRadians);
        require(near(normalX * std::cos(point.tangentRadians) +
                     normalZ * std::sin(point.tangentRadians), 0.0, 1.0e-14),
                "local normal is not orthogonal to the tangent");
        const double halfThickness = 0.5 * source.mediumThickness;
        const auto upper = profile.normalOffset(x, halfThickness);
        const auto lower = profile.normalOffset(x, -halfThickness);
        require(near(distance(upper, lower), source.mediumThickness, 1.0e-14),
                "normal-offset surfaces do not preserve sheet thickness");
    }
    require(profile.sample(0.5 * arcX).segment == ProfileSegment::valleyArc &&
                profile.sample(arcX + 0.4 * flankHorizontal).segment ==
                    ProfileSegment::straightFlank &&
                profile.sample(flankEndX + 0.5 * arcX).segment ==
                    ProfileSegment::crestArc,
            "profile segment tagging changed away from its transition points");

    const double targetCoreRise = source.boardCaliper -
        source.lowerLinerThickness - source.upperLinerThickness -
        source.mediumThickness;
    require(near(closure.nominalCoreRise, targetCoreRise, 1.0e-15),
            "nominal liner/core stack closure is calculated incorrectly");
    require(!closure.matches(1.0e-6),
            "the source arc interpretation unexpectedly fits the nominal caliper");
    require(near(closure.impliedBoardCaliper, 5.727628948622616 * mm, 1.0e-12),
            "implied board caliper changed; review source-profile interpretation");
    require(near(closure.caliperMismatch, 0.727628948622616 * mm, 1.0e-12),
            "reported dimensions do not expose their caliper mismatch");

    std::cout << std::setprecision(12)
              << "profile_model=tangent circular arc + straight flank; "
                 "theta is flank angle; radius is centerline fillet radius\n"
              << "projected_pitch_m=" << source.pitch
              << " radius_m=" << source.radius
              << " inclination_deg=" << source.inclinationDegrees
              << " straight_flank_m=" << profile.straightFlankLength()
              << " centerline_rise_m=" << profile.coreRise()
              << " centerline_arclength_per_pitch_m="
              << profile.centerlineArcLengthPerPitch()
              << " take_up_ratio=" << profile.takeUpRatio() << '\n'
              << "nominal_caliper_m=" << closure.nominalBoardCaliper
              << " implied_caliper_m=" << closure.impliedBoardCaliper
              << " mismatch_m=" << closure.caliperMismatch
              << " stack_matches=" << closure.matches(1.0e-6) << '\n';
}

void checkRejectsImpossibleArcGeometry() {
    FluteDimensions noFlank{
        .pitch = 9.0e-3,
        .radius = 3.0e-3,
        .inclinationDegrees = 64.8,
        .mediumThickness = 0.25e-3,
        .boardCaliper = 5.0e-3,
        .lowerLinerThickness = 0.25e-3,
        .upperLinerThickness = 0.25e-3,
    };
    bool rejected = false;
    try {
        [[maybe_unused]] const CircularArcFluteProfile invalid(noFlank);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
            "profile accepted a radius/angle whose arcs consume more than half pitch");
}

} // namespace

int main() {
    try {
        checkSourceFlute();
        checkRejectsImpossibleArcGeometry();
        std::cout << "PASS cardboard_profile_check checks=" << checks
                  << " physical_validation=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL cardboard_profile_check checks=" << checks
                  << " error=" << error.what() << '\n';
        return 1;
    }
}
