#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/metal_world.hpp"
#include "numi/matter/surgical_tissue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef NUMI_JEJUNUM_MATERIAL
#define NUMI_JEJUNUM_MATERIAL ""
#endif

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

void appendStateHash(
    std::uint64_t& hash,
    const void* bytes,
    const std::size_t byteCount
) {
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < byteCount; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ull;
    }
}

template <typename T>
void appendStateHash(std::uint64_t& hash, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = values.size();
    appendStateHash(hash, &count, sizeof(count));
    if (!values.empty()) {
        appendStateHash(hash, values.data(), values.size() * sizeof(T));
    }
}

constexpr std::array<std::uint32_t, 4> kJawATeeth{15u, 18u, 20u, 22u};
constexpr std::array<std::uint32_t, 4> kJawBTeeth{17u, 19u, 21u, 23u};
constexpr std::uint32_t kNeedleFirstShape =
    2u * metalrobo::kSurgicalPSMShapeCount;
constexpr std::uint32_t kGiverNeedleShape = 14u;
constexpr std::uint32_t kReceiverNeedleShape = 19u;
constexpr double kControlTimestep = 0.002;
constexpr std::uint32_t kPhysicsSubsteps = 32u;
// The first puncture gate is intentionally a research qualification value,
// not a clinical insertion-force claim. At the live 62.5 us substep it is an
// 8 mN mean normal-force floor. Geometry crossing and accepted contact must
// both hold before topology may change; specimen-specific calibration still
// owns later promotion.
constexpr double kPunctureImpulseThresholdNs = 5.0e-7;
constexpr double kPunctureInitialClearanceM = 1.0e-5;
constexpr double kPunctureApproachSpeedMps = 2.0e-2;
// A bounded kinematic stress probe advances far enough to exercise a second
// sub-element channel segment without pretending to be a clinical motion
// profile. The surgical sequence retains the 20 mm/s entry speed above.
constexpr double kPunctureChannelProbeSpeedMps = 2.0e-1;
constexpr std::uint32_t kPunctureChannelProbeSteps = 16u;
// The through-wall qualification keeps the medically scaled needle at the
// same deliberate 20 mm/s terminal centerline speed as initial entry.  It is
// driven about the authored half-circle centre rather than translated through
// the wall, and advances in bounded chunks so every accepted Matter/DER state
// remains inspectable while resident GPU state is retained.
constexpr double kCurvedPassageSpeedMps = kPunctureApproachSpeedMps;
constexpr double kCurvedPassageExitClearanceM = 1.0e-4;
constexpr double kCurvedPassageMaximumExtensionM = 5.0e-4;
constexpr std::uint32_t kCurvedPassageChunkSteps = 32u;
constexpr std::uint32_t kCurvedPassageContactSegmentCount = 2u;
// Place the qualification bite 3 mm from the enterotomy centreline. This is
// an explicit training geometry pending procedure- and specimen-specific bite
// calibration, and is never reported as a clinical recommendation.
constexpr double kPunctureBiteOffsetM = 3.0e-3;
// With the 62.5 us microstep required by the first-edge flexural mode, a
// measured 36 ms pre-roll brings both steel and strand beneath their
// independent quiescence gates without adding artificial damping.
constexpr std::uint32_t kDefaultSettleSteps = 18u;
constexpr std::uint32_t kLongSettleSteps = 150u;
constexpr double kMaximumSettledNeedleLinearSpeed = 1.0e-3;
constexpr double kMaximumSettledNeedleAngularSpeed = 0.5;
constexpr double kMaximumSettledRodSpeed = 2.0e-2;
constexpr double kMaximumSettledRodStretch = 2.0e-6;
constexpr double kMinimumSettledRodSeparation = -5.0e-5;
constexpr double kPickupInsertion = 0.205;
constexpr double kReceiverHandoffInsertion = 0.230;
constexpr double kPickupRetraction = 0.004;
constexpr double kHandoffLift = 0.010;
// A surgeon-like deliberate lift: the cubic trajectory has a 15 mm/s peak
// jaw speed and zero endpoint velocity over 1.0 s. The former 0.5 s move
// peaked at 30 mm/s and exposed the rod/contact integration split repaired by
// the live constrained-rod configuration update.
constexpr std::uint32_t kHandoffLiftSteps = 500u;
// The 180 x 120 mm sterile pad is a pickup surface, not a safe dual-arm
// exchange volume. After lifting normal to the pad, the giver translates the
// needle 90 mm in +Y into a neutral handoff zone. CPU collision-oracle sweeps
// place the first collision-free opposed-port solution at 85 mm; retaining a
// further 5 mm margin avoids selecting the geometric boundary. The cubic
// trajectory is checked against every authored PSM joint velocity limit before
// it is dispatched to the live coupled island. Three seconds limits peak jaw
// speed to 45 mm/s; a subsequent 500 ms hold lets the free monofilament shed
// transport energy before terminal convergence is judged.
constexpr Vec3 kHandoffStagingOffset{0.0, 0.090, 0.0};
constexpr std::uint32_t kHandoffStagingSteps = 1500u;
constexpr std::uint32_t kHandoffStagingChunkSteps = 50u;
constexpr std::uint32_t kHandoffStagingSettleSteps = 250u;
constexpr double kMaximumCommandVelocityRatio = 0.80;
constexpr std::uint32_t kGraspStabilizationSteps = 25u;
// Pause with the receiver open after its Cartesian approach. This separates
// transport convergence from deliberate jaw closure and gives the coupled
// strand/contact island 100 ms at the acquired pose.
constexpr std::uint32_t kReceiverApproachSettleSteps = 50u;
// The giver may reseat the needle during receiver approach. Reacquire the
// resulting live handling frame with the receiver still open, then hold it
// clear before introducing the first receiver contact.
constexpr std::uint32_t kReceiverAlignmentMinimumSteps = 100u;
constexpr std::uint32_t kReceiverAlignmentSettleSteps = 50u;
// Receiver closure creates a new eight-patch steel contact and must be held
// before the temporal-cone residual is interpreted as dual positive control.
constexpr std::uint32_t kReceiverClosureSettleSteps = 100u;
// Exchange preload before the giver clears the needle: the receiver ramps
// from gentle 15 um overlap to the qualified 60 um seat while the giver ramps
// down by the same amount. A short hold qualifies that full receiver seat;
// only then does the giver continue to a visibly open clearance.
constexpr std::uint32_t kLoadExchangeSettleSteps = 50u;
constexpr std::uint32_t kGiverReleaseSettleSteps = 100u;
constexpr double kReceiverRetraction = 0.050;
// After the giver is visibly clear, retract the received needle 6.5 mm toward
// the receiver port. This exceeds the independent 6 mm carried-distance gate
// without unnecessarily tensioning the 250 mm monofilament against its table
// drape. Over 0.5 s the cubic trajectory peaks at 19.5 mm/s; the rejected
// 10 mm trajectories peaked at 75 and 30 mm/s and rolled the needle 182 and
// 145 um, respectively.
constexpr double kReceiverTransfer = 0.0065;
constexpr double kMinimumReceiverTransfer = 0.006;
constexpr std::uint32_t kReceiverTransferSteps = 250u;
// The cubic transfer reaches zero commanded velocity at its endpoint, but the
// coupled needle/strand island still carries contact and flexural momentum.
// Hold the acquired pose for 100 ms before terminal publication, matching the
// receiver-approach stabilization window rather than certifying a motion
// frame as a settled handoff.
constexpr std::uint32_t kReceiverTransferSettleSteps = 50u;
// The transverse insert rails close through the needle cross-section
// centreline. A former +120 um offset was compensating for the oversized
// legacy clevis and produced an off-axis rolling impulse at first contact.
constexpr double kPickupVerticalClearance = 0.00015;
constexpr double kInitialSupportPenetration = 2.0e-6;
// The hard swage must remain within the live rod constraint tolerance plus
// single-precision integration noise. A larger error means the thread is no
// longer mechanically carried by the needle, even if the rendered endpoints
// still appear close.
constexpr double kMaximumSwageAttachmentError = 2.5e-5;
// Keep the first resolved edge inside a conservative 1% outer-fibre bending-
// strain envelope: epsilon = radius * theta / edge length. PDO monofilament
// studies report break elongations of roughly 36-58%; this deliberately much
// smaller research ceiling bounds the linear Euler-Bernoulli model and is not
// a clinical damage threshold.
constexpr double kMaximumSwageTangentBendingStrain = 1.0e-2;
// Terminal grasp qualification is based on rigid-body point kinematics at
// the authored needle handling segment, not merely contact incidence. Once
// all four finite patches on each jaw have collision-seated the needle, the
// jaw-groove offset is stored in the needle body frame. Later phases may
// drift at most 100 um from that seated transform, irrespective of common
// rigid rotation. The remaining explicit research tolerances are 2 mm/s
// point slip and 0.6 rad/s relative angular motion at a commanded endpoint.
constexpr double kMaximumQualifiedGraspSeatDrift = 1.0e-4;
// During the long collision-free receiver approach, the controlling needle
// may settle deeper into the giver's finite V-groove. Permit one bounded,
// stable reseat below the 0.35 mm needle radius, then establish a fresh
// pre-closure reference. Subsequent grasp phases retain the 0.10 mm bound.
constexpr double kMaximumTransitionGraspReseating = 3.0e-4;
constexpr double kMaximumQualifiedRelativePointSpeed = 2.0e-3;
constexpr double kMaximumQualifiedRelativeAngularSpeed = 0.6;
// Temporal-cone contact publishes the final complementarity velocity residual
// in residuals.y and elliptic-cone feasibility in residuals.z. Terminal phase
// acceptance requires these live quantities directly; an un-dispatched
// unified-quality record is not a convergence certificate.
constexpr double kMaximumTerminalContactVelocityResidual = 2.0e-3;
constexpr double kMaximumTerminalConeViolation = 1.0e-5;
constexpr double kMaximumTerminalRodNodeSpeed = 5.0e-2;
constexpr double kMaximumTerminalRodEdgeLengthError = 2.0e-5;
// This protocol deliberately keeps the package coil separated and does not
// form a knot. The live DER projector prevents crossings; this independent
// terminal bound also rejects a strand that only barely avoids penetration.
constexpr double kMinimumThreadSelfCollisionClearance = 5.0e-5;
// The projector still owns a 50 um margin. Reconstructed capsule separation
// is measured from float positions after the solve, so permit 0.1 um of
// readback roundoff (the rejected midpoint differed by only 7.4 nm).
constexpr double kThreadClearanceReadbackTolerance = 1.0e-7;
// Each insert has two finite rows on either side of the needle centreline,
// split into two contact patches across the jaw width. The opposing jaw-centre
// distance must account for that V-groove offset before applying load. A
// 60 um per-patch radial engagement with the authored 50 um/N whole-system
// compliance resolves to roughly 1.2 N at one patch before coupled
// redistribution, or 4.8 N across one jaw's four patches. At the 9.7 mm jaw
// envelope that remains below the LND's authored 0.16 N*m joint-effort limit.
// The former 30 um setting reached 0.9998 friction utilization and eventually
// let the loaded needle roll out. This is a measured research calibration,
// not bulk steel deformation or a prescribed clinical clamp force.
constexpr double kGrooveRailRadialPreload = 6.0e-5;
// Once the giver has released, ramp the receiver from the 60 um exchange seat
// to a 75 um transport latch during retraction. This is 1.5 N per finite rail
// patch with the authored 50 um/N insert compliance, and its full-jaw moment
// remains independently checked against the LND effort limit below. It is a
// measured transport setting for the steel needle, not a tissue clamp load.
constexpr double kReceiverTransportRailRadialPreload = 7.5e-5;
// Establish receiver positive control gently while the giver still owns the
// needle. Full 60 um engagement is transferred in as the giver opens so two
// stiff V-grooves do not fight over the same curved steel body.
constexpr double kReceiverOverlapRailRadialPreload = 1.5e-5;
constexpr std::uint32_t kPreLiftRegripSteps = 50u;
constexpr Vec3 kGiverPortCalibration{0.001285, -0.000547, 0.0};
constexpr double kGiverYaw = 0.0;
constexpr double kGiverPitch = 0.78;
constexpr double kReceiverYaw = 1.20;
constexpr double kReceiverPitch = -0.78;
// Keep the opposed LND wrists staggered while preserving the receiver jaw
// midpoint and the intended one-third-to-one-half needle handling point. The
// exact staged-state sweep remains collision-free through approach and closure
// for neighbouring +/-0.05 rad postures around this nominal pair.
constexpr double kReceiverToolRollOffset = -0.20;
constexpr double kReceiverWristYaw = -0.30;
// A needle handoff requires opposing instrument shafts and mutually compatible
// insert frames. These two fixed-port angles come from an exact staged-state
// sweep: the receiver rail/live-needle tangent error is 0.0133 rad versus
// 0.0149 rad for the controlling giver, with zero approach/closure collisions
// and 8/8 receiver contacts. Neighbouring +/-0.05 rad samples remain clean.
constexpr double kReceiverBaseAzimuthOffset = 2.95;
constexpr double kReceiverNeedleAxisRoll = 2.55;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
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

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 vector(const nm_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

Vec3 vector(const std::array<float, 3>& value) {
    return {value[0], value[1], value[2]};
}

double segmentSegmentDistance(
    const Vec3 first0,
    const Vec3 first1,
    const Vec3 second0,
    const Vec3 second1
) {
    constexpr double kDegenerateSquared = 1.0e-24;
    const Vec3 firstDirection = first1 - first0;
    const Vec3 secondDirection = second1 - second0;
    const Vec3 offset = first0 - second0;
    const double firstLengthSquared = dot(firstDirection, firstDirection);
    const double secondLengthSquared = dot(secondDirection, secondDirection);
    const double secondProjection = dot(secondDirection, offset);
    double firstParameter = 0.0;
    double secondParameter = 0.0;
    if (firstLengthSquared <= kDegenerateSquared &&
        secondLengthSquared <= kDegenerateSquared) {
        return norm(offset);
    }
    if (firstLengthSquared <= kDegenerateSquared) {
        secondParameter = std::clamp(
            secondProjection / secondLengthSquared,
            0.0,
            1.0
        );
    } else {
        const double firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <= kDegenerateSquared) {
            firstParameter = std::clamp(
                -firstProjection / firstLengthSquared,
                0.0,
                1.0
            );
        } else {
            const double coupling = dot(
                firstDirection,
                secondDirection
            );
            const double denominator =
                firstLengthSquared * secondLengthSquared -
                coupling * coupling;
            if (denominator > kDegenerateSquared) {
                firstParameter = std::clamp(
                    (
                        coupling * secondProjection -
                        firstProjection * secondLengthSquared
                    ) / denominator,
                    0.0,
                    1.0
                );
            }
            secondParameter =
                (coupling * firstParameter + secondProjection) /
                secondLengthSquared;
            if (secondParameter < 0.0) {
                secondParameter = 0.0;
                firstParameter = std::clamp(
                    -firstProjection / firstLengthSquared,
                    0.0,
                    1.0
                );
            } else if (secondParameter > 1.0) {
                secondParameter = 1.0;
                firstParameter = std::clamp(
                    (coupling - firstProjection) /
                        firstLengthSquared,
                    0.0,
                    1.0
                );
            }
        }
    }
    const Vec3 closest =
        offset + firstDirection * firstParameter -
        secondDirection * secondParameter;
    return norm(closest);
}

double minimumNonNeighbourRodSurfaceSeparation(
    const metalrobo::MetalWorldResult& state,
    const double radius
) {
    require(
        state.finalRodNodes.size() >= 4u && radius > 0.0,
        "thread self-clearance state is incomplete"
    );
    double minimum = std::numeric_limits<double>::infinity();
    const std::size_t edgeCount = state.finalRodNodes.size() - 1u;
    for (std::size_t first = 0u; first < edgeCount; ++first) {
        for (std::size_t second = first + 2u;
             second < edgeCount;
             ++second) {
            minimum = std::min(
                minimum,
                segmentSegmentDistance(
                    vector(state.finalRodNodes[first].position),
                    vector(state.finalRodNodes[first + 1u].position),
                    vector(state.finalRodNodes[second].position),
                    vector(state.finalRodNodes[second + 1u].position)
                ) - 2.0 * radius
            );
        }
    }
    return minimum;
}

struct RodStateMetrics {
    double maximumNodeSpeed = 0.0;
    double maximumEdgeLengthError = 0.0;
    double minimumNonNeighbourSurfaceClearance =
        std::numeric_limits<double>::infinity();
};

RodStateMetrics rodStateMetrics(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    require(
        world.rods.size() == 1u &&
            state.finalRodNodes.size() ==
                world.rods[0].model.restPositions.size(),
        "terminal thread state does not match the authored DER topology"
    );
    RodStateMetrics result;
    for (std::size_t node = 0u;
         node < state.finalRodNodes.size();
         ++node) {
        result.maximumNodeSpeed = std::max(
            result.maximumNodeSpeed,
            norm(vector(state.finalRodNodes[node].velocity))
        );
        if (node != 0u) {
            result.maximumEdgeLengthError = std::max(
                result.maximumEdgeLengthError,
                std::abs(
                    norm(
                        vector(state.finalRodNodes[node].position) -
                        vector(state.finalRodNodes[node - 1u].position)
                    ) - world.rods[0].model.restLengths[node - 1u]
                )
            );
        }
    }
    result.minimumNonNeighbourSurfaceClearance =
        minimumNonNeighbourRodSurfaceSeparation(
            state,
            world.rods[0].model.radius
        );
    return result;
}

bool qualifiedTerminalRod(const RodStateMetrics& metrics) {
    return
        metrics.maximumNodeSpeed <= kMaximumTerminalRodNodeSpeed &&
        metrics.maximumEdgeLengthError <=
            kMaximumTerminalRodEdgeLengthError &&
        metrics.minimumNonNeighbourSurfaceClearance >=
            kMinimumThreadSelfCollisionClearance -
                kThreadClearanceReadbackTolerance;
}

bool qualifiedTransitionRod(const RodStateMetrics& metrics) {
    return
        metrics.maximumEdgeLengthError <=
            kMaximumTerminalRodEdgeLengthError &&
        metrics.minimumNonNeighbourSurfaceClearance >=
            kMinimumThreadSelfCollisionClearance -
                kThreadClearanceReadbackTolerance;
}

Quaternion multiply(const Quaternion a, const Quaternion b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

Quaternion conjugate(const Quaternion q) {
    return {-q.x, -q.y, -q.z, q.w};
}

Quaternion rotationX(const double angle) {
    return {
        std::sin(0.5 * angle),
        0.0,
        0.0,
        std::cos(0.5 * angle),
    };
}

Quaternion rotationY(const double angle) {
    return {
        0.0,
        std::sin(0.5 * angle),
        0.0,
        std::cos(0.5 * angle),
    };
}

Quaternion rotationZ(const double angle) {
    return {
        0.0,
        0.0,
        std::sin(0.5 * angle),
        std::cos(0.5 * angle),
    };
}

Quaternion axisAngle(const Vec3 axis, const double angle) {
    const double axisLength = norm(axis);
    require(
        axisLength > 1.0e-12 && std::isfinite(axisLength) &&
            std::isfinite(angle),
        "axis-angle rotation is degenerate"
    );
    const double scale = std::sin(0.5 * angle) / axisLength;
    return {
        axis.x * scale,
        axis.y * scale,
        axis.z * scale,
        std::cos(0.5 * angle),
    };
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 imaginary{q.x, q.y, q.z};
    const Vec3 doubled{
        2.0 * (imaginary.y * value.z - imaginary.z * value.y),
        2.0 * (imaginary.z * value.x - imaginary.x * value.z),
        2.0 * (imaginary.x * value.y - imaginary.y * value.x),
    };
    const Vec3 second{
        imaginary.y * doubled.z - imaginary.z * doubled.y,
        imaginary.z * doubled.x - imaginary.x * doubled.z,
        imaginary.x * doubled.y - imaginary.y * doubled.x,
    };
    return value + doubled * q.w + second;
}

struct NeedleTipCapsuleGeometry {
    Vec3 localTip{};
    Vec3 localBase{};
    Vec3 worldTip{};
    Vec3 worldBase{};
    Vec3 approachDirection{};
    double radiusM = 0.0;
    double lengthM = 0.0;
};

struct CurvedNeedleOrbit {
    Vec3 centerWorld{};
    Vec3 axisWorld{};
    Vec3 centerLocal{};
    Quaternion initialOrientation{};
    Vec3 initialBodyPosition{};
    double centerlineRadiusM = 0.0;
};

CurvedNeedleOrbit curvedNeedleOrbit(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const MRBodyStateGPU& body
) {
    const Quaternion orientation{
        body.orientation.x,
        body.orientation.y,
        body.orientation.z,
        body.orientation.w,
    };
    const Vec3 bodyPosition = vector(body.position);
    // SurgicalAssets cooks the analytic arc about local geometry origin and
    // then shifts every collider by the combined rigid COM.  The corresponding
    // body-local curvature centre is therefore the negative geometry COM.
    const Vec3 centerLocal{
        -needle.rigid.geometryCenterOfMassM[0],
        -needle.rigid.geometryCenterOfMassM[1],
        -needle.rigid.geometryCenterOfMassM[2],
    };
    const Vec3 axisWorld = rotate(orientation, {0.0, 0.0, 1.0});
    const double axisLength = norm(axisWorld);
    require(
        needle.metadata.centerlineRadiusM > 0.0 &&
            std::isfinite(needle.metadata.centerlineRadiusM) &&
            axisLength > 1.0e-12 && std::isfinite(axisLength),
        "curved needle orbit is invalid"
    );
    return {
        .centerWorld = bodyPosition + rotate(orientation, centerLocal),
        .axisWorld = axisWorld * (1.0 / axisLength),
        .centerLocal = centerLocal,
        .initialOrientation = orientation,
        .initialBodyPosition = bodyPosition,
        .centerlineRadiusM = needle.metadata.centerlineRadiusM,
    };
}

MRBodyStateGPU curvedNeedleTarget(
    const MRBodyStateGPU& authored,
    const CurvedNeedleOrbit& orbit,
    const double angleRad,
    const double angularSpeedRadPerS
) {
    require(
        std::isfinite(angleRad) &&
            std::isfinite(angularSpeedRadPerS) &&
            angularSpeedRadPerS > 0.0,
        "curved needle target has invalid motion"
    );
    const Quaternion orientation = multiply(
        axisAngle(orbit.axisWorld, angleRad),
        orbit.initialOrientation
    );
    const Vec3 position = orbit.centerWorld -
        rotate(orientation, orbit.centerLocal);
    const Vec3 angularVelocity =
        orbit.axisWorld * angularSpeedRadPerS;
    const Vec3 linearVelocity = cross(
        angularVelocity,
        position - orbit.centerWorld
    );
    MRBodyStateGPU target = authored;
    target.position.x = static_cast<float>(position.x);
    target.position.y = static_cast<float>(position.y);
    target.position.z = static_cast<float>(position.z);
    target.orientation = {
        static_cast<float>(orientation.x),
        static_cast<float>(orientation.y),
        static_cast<float>(orientation.z),
        static_cast<float>(orientation.w),
    };
    target.linearVelocityAndInverseMass = {
        static_cast<float>(linearVelocity.x),
        static_cast<float>(linearVelocity.y),
        static_cast<float>(linearVelocity.z),
        0.0f,
    };
    target.angularVelocity = {
        static_cast<float>(angularVelocity.x),
        static_cast<float>(angularVelocity.y),
        static_cast<float>(angularVelocity.z),
        0.0f,
    };
    target.inverseInertiaWorldRow0 = {};
    target.inverseInertiaWorldRow1 = {};
    target.inverseInertiaWorldRow2 = {};
    return target;
}

NeedleTipCapsuleGeometry needleTipCapsuleGeometry(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const MRBodyStateGPU& body
) {
    require(
        !needle.rigid.shapes.empty(),
        "curved needle has no tapered-tip collision segment"
    );
    const MRShapeGPU& segment = needle.rigid.shapes.back();
    require(
        segment.shapeType == MR_SHAPE_CAPSULE &&
            segment.dimensions.x > 0.0f &&
            segment.dimensions.y > 0.0f,
        "curved needle tip is not a finite capsule"
    );
    const Quaternion localRotation{
        segment.localRotation.x,
        segment.localRotation.y,
        segment.localRotation.z,
        segment.localRotation.w,
    };
    const Vec3 localAxis = rotate(localRotation, {0.0, 1.0, 0.0});
    const Vec3 localCenter = vector(segment.localPosition);
    const Vec3 localTip =
        localCenter + localAxis * segment.dimensions.y;
    const Vec3 localBase =
        localCenter - localAxis * segment.dimensions.y;
    const Quaternion bodyRotation{
        body.orientation.x,
        body.orientation.y,
        body.orientation.z,
        body.orientation.w,
    };
    const Vec3 bodyPosition = vector(body.position);
    const Vec3 worldTip = bodyPosition + rotate(bodyRotation, localTip);
    const Vec3 worldBase = bodyPosition + rotate(bodyRotation, localBase);
    const Vec3 tangent = worldTip - worldBase;
    const double lengthM = norm(tangent);
    require(
        lengthM > 1.0e-9 && std::isfinite(lengthM),
        "curved needle tip capsule is degenerate"
    );
    return {
        .localTip = localTip,
        .localBase = localBase,
        .worldTip = worldTip,
        .worldBase = worldBase,
        .approachDirection = tangent * (1.0 / lengthM),
        .radiusM = segment.dimensions.x,
        .lengthM = lengthM,
    };
}

double swageAttachmentError(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    require(
        !state.finalSceneBodies.empty() &&
            !state.finalRodNodes.empty() &&
            !world.rods.empty() &&
            !world.rods[0].rigidBindings.empty() &&
            world.rods[0].rigidBindings.size() ==
                world.rods[0].attachments.size(),
        "swage attachment state is incomplete"
    );
    const MRBodyStateGPU& needle = state.finalSceneBodies[0];
    double maximumError = 0.0;
    std::uint32_t hardAttachmentCount = 0u;
    for (std::size_t bindingIndex = 0u;
         bindingIndex < world.rods[0].rigidBindings.size();
         ++bindingIndex) {
        if (world.rods[0].attachments[bindingIndex].compliance != 0.0) {
            continue;
        }
        ++hardAttachmentCount;
        const auto& local =
            world.rods[0].rigidBindings[bindingIndex].localAnchor;
        const std::uint32_t node =
            world.rods[0].attachments[bindingIndex].nodeIndex;
        require(
            node < state.finalRodNodes.size(),
            "swage attachment node is out of range"
        );
        const Vec3 expected = vector(needle.position) + rotate(
            Quaternion{
                needle.orientation.x,
                needle.orientation.y,
                needle.orientation.z,
                needle.orientation.w,
            },
            {local[0], local[1], local[2]}
        );
        maximumError = std::max(
            maximumError,
            norm(vector(state.finalRodNodes[node].position) - expected)
        );
    }
    require(
        hardAttachmentCount == 1u,
        "swage must have exactly one hard root attachment"
    );
    return maximumError;
}

double maximumSwageTangentAngleError(
    const metalrobo::HeterogeneousWorld& world
) {
    require(
        !world.rods.empty() &&
            !world.rods[0].model.restLengths.empty() &&
            world.rods[0].model.radius > 0.0,
        "swage tangent material geometry is incomplete"
    );
    return
        kMaximumSwageTangentBendingStrain *
        world.rods[0].model.restLengths.front() /
        world.rods[0].model.radius;
}

double swageTangentLineError(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    require(
        !state.finalSceneBodies.empty() &&
            state.finalRodNodes.size() >= 2u &&
            !world.rods.empty() &&
            world.rods[0].tangentBindings.size() == 1u,
        "swage tangent-line state is incomplete"
    );
    const auto& binding = world.rods[0].tangentBindings[0];
    const MRBodyStateGPU& needle = state.finalSceneBodies[0];
    const Quaternion orientation{
        needle.orientation.x,
        needle.orientation.y,
        needle.orientation.z,
        needle.orientation.w,
    };
    const Vec3 expected = vector(needle.position) + rotate(
        orientation,
        vector(binding.localAnchor)
    );
    const std::uint32_t node = binding.edgeIndex + 1u;
    require(
        node < state.finalRodNodes.size(),
        "swage tangent-line node is out of range"
    );
    const Vec3 tangent = rotate(
        orientation,
        vector(binding.localTangent)
    );
    const Vec3 delta = vector(state.finalRodNodes[node].position) - expected;
    return norm(delta - tangent * dot(delta, tangent));
}

double swageTangentAngleError(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    require(
        !state.finalSceneBodies.empty() &&
            state.finalRodNodes.size() >= 2u &&
            !world.rods.empty() &&
            world.rods[0].tangentBindings.size() == 1u,
        "swage tangent state is incomplete"
    );
    const auto& binding = world.rods[0].tangentBindings[0];
    const std::uint32_t nodeA = binding.edgeIndex;
    const std::uint32_t nodeB = binding.edgeIndex + 1u;
    require(
        nodeA < state.finalRodNodes.size() &&
            nodeB < state.finalRodNodes.size() && nodeA != nodeB,
        "swage tangent nodes are invalid"
    );
    const Vec3 actual =
        vector(state.finalRodNodes[nodeB].position) -
        vector(state.finalRodNodes[nodeA].position);
    const MRBodyStateGPU& needle = state.finalSceneBodies[0];
    const Vec3 expected = rotate(
        Quaternion{
            needle.orientation.x,
            needle.orientation.y,
            needle.orientation.z,
            needle.orientation.w,
        },
        vector(binding.localTangent)
    );
    const double denominator = norm(actual) * norm(expected);
    require(
        denominator > 1.0e-12,
        "swage tangent contains a degenerate segment"
    );
    return std::acos(std::clamp(
        dot(actual, expected) / denominator,
        -1.0,
        1.0
    ));
}

double swageMaterialFrameError(
    const metalrobo::DiscreteRodRigidTwistAttachmentBinding& binding,
    const MRBodyStateGPU& body,
    const Vec3 nodeA,
    const Vec3 nodeB,
    const double twist
) {
    const Vec3 edge = nodeB - nodeA;
    const double edgeLength = norm(edge);
    require(
        edgeLength > 1.0e-12,
        "swage material frame contains a degenerate edge"
    );
    const Vec3 tangent = edge * (1.0 / edgeLength);
    const Quaternion orientation{
        body.orientation.x,
        body.orientation.y,
        body.orientation.z,
        body.orientation.w,
    };
    const Vec3 bodyTangent = rotate(
        orientation,
        vector(binding.localTangent)
    );
    const Vec3 localDirector = vector(binding.localMaterialDirector);
    const Vec3 worldDirector = rotate(
        orientation,
        localDirector
    );
    const auto transport = [](
        const Vec3 director,
        const Vec3 from,
        const Vec3 to
    ) {
        const double cosine = std::clamp(dot(from, to), -1.0, 1.0);
        require(
            cosine > -1.0 + 1.0e-10,
            "swage material-frame transport is antipodal"
        );
        const Vec3 axis = cross(from, to);
        const Vec3 first = cross(axis, director);
        const Vec3 second = cross(axis, first);
        Vec3 result =
            director + first + second * (1.0 / (1.0 + cosine));
        result = result - to * dot(result, to);
        const double resultLength = norm(result);
        require(
            resultLength > 1.0e-12,
            "swage material-frame transport is degenerate"
        );
        return result * (1.0 / resultLength);
    };
    const Vec3 reference = transport(
        vector(binding.referenceMaterialDirectorWorld),
        vector(binding.referenceTangentWorld),
        tangent
    );
    const Vec3 materialDirector = transport(
        worldDirector,
        bodyTangent,
        tangent
    );
    const double targetTwist = std::atan2(
        dot(tangent, cross(reference, materialDirector)),
        dot(reference, materialDirector)
    );
    return std::abs(std::atan2(
        std::sin(targetTwist - twist),
        std::cos(targetTwist - twist)
    ));
}

double swageMaterialFrameError(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    require(
        world.rods.size() == 1u &&
            world.rods[0].twistBindings.size() == 1u &&
            !state.finalSceneBodies.empty() &&
            state.finalRodNodes.size() >= 2u &&
            !state.finalRodEdges.empty(),
        "swage material-frame state is incomplete"
    );
    const auto& binding = world.rods[0].twistBindings[0];
    return swageMaterialFrameError(
        binding,
        state.finalSceneBodies.at(binding.bodyIndex),
        vector(state.finalRodNodes.at(binding.edgeIndex).position),
        vector(state.finalRodNodes.at(binding.edgeIndex + 1u).position),
        state.finalRodEdges.at(binding.edgeIndex).twistAndRate.x
    );
}

double initialSwageMaterialFrameError(
    const metalrobo::HeterogeneousWorld& world
) {
    require(
        world.rods.size() == 1u &&
            world.rods[0].twistBindings.size() == 1u,
        "initial swage material-frame state is incomplete"
    );
    const auto& rod = world.rods[0];
    const auto& binding = rod.twistBindings[0];
    return swageMaterialFrameError(
        binding,
        world.defaultSceneBodies.at(binding.bodyIndex),
        vector(rod.defaultState.positions.at(binding.edgeIndex)),
        vector(rod.defaultState.positions.at(binding.edgeIndex + 1u)),
        rod.defaultState.twists.at(binding.edgeIndex)
    );
}

double swageTangentBendingStrain(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    return
        world.rods[0].model.radius *
        swageTangentAngleError(world, state) /
        world.rods[0].model.restLengths.front();
}

double swageTangentBendingStressPa(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& state
) {
    const double radius = world.rods[0].model.radius;
    const double secondMoment =
        std::numbers::pi * std::pow(radius, 4.0) / 4.0;
    require(
        secondMoment > 0.0 &&
            !world.rods[0].model.bendStiffness.empty(),
        "swage tangent material stiffness is incomplete"
    );
    const double youngModulus =
        world.rods[0].model.bendStiffness.front() / secondMoment;
    return youngModulus * swageTangentBendingStrain(world, state);
}

metalrobo::SurgicalBasePose basePose(
    const Quaternion orientation,
    const Vec3 localJawMidpoint,
    const Vec3 desiredJawMidpoint
) {
    const Vec3 position =
        desiredJawMidpoint - rotate(orientation, localJawMidpoint);
    return {
        .position = {
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
        },
        .orientation = {
            static_cast<float>(orientation.x),
            static_cast<float>(orientation.y),
            static_cast<float>(orientation.z),
            static_cast<float>(orientation.w),
        },
    };
}

struct ShapeMotion {
    Vec3 position{};
    Vec3 linearVelocity{};
    Vec3 angularVelocity{};
};

ShapeMotion shapeMotion(
    const std::span<const metalrobo::ArticulatedBodyKinematics> bodies,
    const metalrobo::EngineModel& model,
    const std::uint32_t shapeIndex
) {
    const MRShapeGPU& shape = model.shapes.at(shapeIndex);
    const auto found = std::find_if(
        bodies.begin(),
        bodies.end(),
        [&](const metalrobo::ArticulatedBodyKinematics& body) {
            return body.bodyIndex == shape.bodyIndex;
        }
    );
    require(found != bodies.end(), "jaw shape body is missing");
    const Quaternion orientation{
        found->orientation[0],
        found->orientation[1],
        found->orientation[2],
        found->orientation[3],
    };
    const Vec3 offset = rotate(
        orientation,
        vector(shape.localPosition)
    );
    const Vec3 angularVelocity = vector(found->angularVelocity);
    return {
        .position = vector(found->centerOfMassPosition) + offset,
        .linearVelocity = vector(found->linearVelocity) +
            cross(angularVelocity, offset),
        .angularVelocity = angularVelocity,
    };
}

Vec3 shapeCenter(
    const std::span<const metalrobo::ArticulatedBodyKinematics> bodies,
    const metalrobo::EngineModel& model,
    const std::uint32_t shapeIndex
) {
    return shapeMotion(bodies, model, shapeIndex).position;
}

struct JawGeometry {
    Vec3 jawA{};
    Vec3 jawB{};
    Vec3 midpoint{};
    Vec3 jawAVelocity{};
    Vec3 jawBVelocity{};
    Vec3 midpointVelocity{};
    Vec3 meanAngularVelocity{};
    Vec3 railDirection{};
    Vec3 separationDirection{};
    double separation = 0.0;
};

JawGeometry jawGeometry(
    const metalrobo::EngineModel& model,
    const std::span<const double> q
) {
    const std::vector<double> zeroV(model.world.nv, 0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(
        model.articulations[0].bodyCount
    );
    const auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            zeroV,
            bodies,
            {}
        );
    require(diagnostics.succeeded(), "PSM jaw kinematics failed");
    JawGeometry result;
    constexpr double jawWeight = 1.0 / kJawATeeth.size();
    constexpr double angularWeight =
        1.0 / (kJawATeeth.size() + kJawBTeeth.size());
    Vec3 negativeRailCenter{};
    Vec3 positiveRailCenter{};
    constexpr double railCenterWeight = 0.25;
    for (std::size_t index = 0u; index < kJawATeeth.size(); ++index) {
        const std::uint32_t shape = kJawATeeth[index];
        const ShapeMotion motion = shapeMotion(bodies, model, shape);
        result.jawA = result.jawA + motion.position * jawWeight;
        result.jawAVelocity =
            result.jawAVelocity + motion.linearVelocity * jawWeight;
        result.meanAngularVelocity =
            result.meanAngularVelocity + motion.angularVelocity *
                angularWeight;
        (index < 2u ? negativeRailCenter : positiveRailCenter) =
            (index < 2u ? negativeRailCenter : positiveRailCenter) +
            motion.position * railCenterWeight;
    }
    for (std::size_t index = 0u; index < kJawBTeeth.size(); ++index) {
        const std::uint32_t shape = kJawBTeeth[index];
        const ShapeMotion motion = shapeMotion(bodies, model, shape);
        result.jawB = result.jawB + motion.position * jawWeight;
        result.jawBVelocity =
            result.jawBVelocity + motion.linearVelocity * jawWeight;
        result.meanAngularVelocity =
            result.meanAngularVelocity + motion.angularVelocity *
                angularWeight;
        (index < 2u ? negativeRailCenter : positiveRailCenter) =
            (index < 2u ? negativeRailCenter : positiveRailCenter) +
            motion.position * railCenterWeight;
    }
    result.midpoint = (result.jawA + result.jawB) * 0.5;
    result.midpointVelocity =
        (result.jawAVelocity + result.jawBVelocity) * 0.5;
    result.separation = norm(result.jawA - result.jawB);
    require(result.separation > 1.0e-9, "local jaw separation collapsed");
    result.separationDirection =
        (result.jawA - result.jawB) * (1.0 / result.separation);
    const Vec3 rail = positiveRailCenter - negativeRailCenter;
    require(norm(rail) > 1.0e-9, "local jaw rail direction collapsed");
    result.railDirection = rail * (1.0 / norm(rail));
    return result;
}

JawGeometry worldJawGeometry(
    const metalrobo::EngineModel& model,
    const std::uint32_t arm,
    const std::span<const float> q,
    const std::span<const float> v
) {
    require(
        q.size() == model.world.nq && v.size() == model.world.nv &&
            arm < model.articulations.size(),
        "world jaw state has invalid dimensions"
    );
    const std::vector<double> q64(q.begin(), q.end());
    const std::vector<double> v64(v.begin(), v.end());
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(
        model.articulations[arm].bodyCount
    );
    const auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            arm,
            std::span<const double>(q64).subspan(
                model.articulations[arm].qOffset,
                model.articulations[arm].nq
            ),
            std::span<const double>(v64).subspan(
                model.articulations[arm].vOffset,
                model.articulations[arm].nv
            ),
            bodies,
            {}
        );
    require(diagnostics.succeeded(), "world PSM jaw kinematics failed");
    const std::uint32_t firstShape =
        metalrobo::kSurgicalPSMShapeCount * arm;
    JawGeometry result;
    constexpr double jawWeight = 1.0 / kJawATeeth.size();
    constexpr double angularWeight =
        1.0 / (kJawATeeth.size() + kJawBTeeth.size());
    Vec3 negativeRailCenter{};
    Vec3 positiveRailCenter{};
    constexpr double railCenterWeight = 0.25;
    for (std::size_t index = 0u; index < kJawATeeth.size(); ++index) {
        const std::uint32_t shape = kJawATeeth[index];
        const ShapeMotion motion = shapeMotion(
            bodies,
            model,
            firstShape + shape
        );
        result.jawA = result.jawA + motion.position * jawWeight;
        result.jawAVelocity =
            result.jawAVelocity + motion.linearVelocity * jawWeight;
        result.meanAngularVelocity =
            result.meanAngularVelocity + motion.angularVelocity *
                angularWeight;
        (index < 2u ? negativeRailCenter : positiveRailCenter) =
            (index < 2u ? negativeRailCenter : positiveRailCenter) +
            motion.position * railCenterWeight;
    }
    for (std::size_t index = 0u; index < kJawBTeeth.size(); ++index) {
        const std::uint32_t shape = kJawBTeeth[index];
        const ShapeMotion motion = shapeMotion(
            bodies,
            model,
            firstShape + shape
        );
        result.jawB = result.jawB + motion.position * jawWeight;
        result.jawBVelocity =
            result.jawBVelocity + motion.linearVelocity * jawWeight;
        result.meanAngularVelocity =
            result.meanAngularVelocity + motion.angularVelocity *
                angularWeight;
        (index < 2u ? negativeRailCenter : positiveRailCenter) =
            (index < 2u ? negativeRailCenter : positiveRailCenter) +
            motion.position * railCenterWeight;
    }
    result.midpoint = (result.jawA + result.jawB) * 0.5;
    result.midpointVelocity =
        (result.jawAVelocity + result.jawBVelocity) * 0.5;
    result.separation = norm(result.jawA - result.jawB);
    require(result.separation > 1.0e-9, "world jaw separation collapsed");
    result.separationDirection =
        (result.jawA - result.jawB) * (1.0 / result.separation);
    const Vec3 rail = positiveRailCenter - negativeRailCenter;
    require(norm(rail) > 1.0e-9, "world jaw rail direction collapsed");
    result.railDirection = rail * (1.0 / norm(rail));
    return result;
}

double calibratedJawCoordinate(
    const metalrobo::EngineModel& model,
    const double needleRadius,
    const double separationAdjustment
) {
    const double toothRadius = model.shapes[kJawATeeth[0]].dimensions.x;
    const double desiredSeparation =
        2.0 * (needleRadius + toothRadius) + separationAdjustment;
    double best = 0.0;
    double bestError = std::numeric_limits<double>::infinity();
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = kPickupInsertion;
    q[3] = 0.0;
    q[4] = 0.0;
    q[5] = 0.0;
    for (std::uint32_t sample = 0u; sample <= 1600u; ++sample) {
        const double coordinate =
            0.12 * static_cast<double>(sample) / 1600.0;
        q[6] = -coordinate;
        q[7] = coordinate;
        const double error = std::abs(
            jawGeometry(model, q).separation - desiredSeparation
        );
        if (error < bestError) {
            bestError = error;
            best = coordinate;
        }
    }
    require(
        best > 0.0 && best < 0.12 && bestError < 2.0e-5,
        "PSM jaw could not be calibrated to the requested needle clearance"
    );
    return best;
}

double grooveDiametralClosure(
    const metalrobo::EngineModel& model,
    const double needleRadius,
    const double radialPreload = kGrooveRailRadialPreload
) {
    const double railRadius =
        model.shapes.at(kJawATeeth[0]).dimensions.x;
    const double railHalfSpacing = 0.5 * std::abs(
        static_cast<double>(
            model.shapes.at(kJawATeeth[1]).localPosition.z
        ) - model.shapes.at(kJawATeeth[0]).localPosition.z
    );
    const double contactRadius =
        needleRadius + railRadius - radialPreload;
    require(
        contactRadius > railHalfSpacing &&
            radialPreload > 0.0,
        "PSM insert groove cannot contain the requested needle gauge"
    );
    const double jawHalfSeparation = std::sqrt(
        contactRadius * contactRadius -
        railHalfSpacing * railHalfSpacing
    );
    return 2.0 * (
        needleRadius + railRadius - jawHalfSeparation
    );
}

std::vector<double> psmTarget(
    const metalrobo::EngineModel& model,
    const double insertion,
    const double jawCoordinate,
    const double yaw = 0.0,
    const double pitch = 0.0
) {
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    q[0] = yaw;
    q[1] = pitch;
    q[2] = insertion;
    // Counter base/shaft azimuth at tool roll so port placement does not
    // rotate the needle-driver jaws out of their authored handling plane.
    q[3] = -yaw;
    // Counter the remote-center shaft pitch at the distal wrist so the LND
    // jaws remain upright over the table while the ports stay separated.
    q[4] = -pitch;
    q[5] = 0.0;
    q[6] = -jawCoordinate;
    q[7] = jawCoordinate;
    return q;
}

std::vector<double> solvePsmJawTarget(
    const metalrobo::EngineModel& model,
    const metalrobo::SurgicalBasePose& base,
    std::vector<double> q,
    const Vec3 desiredWorld,
    const double jawCoordinate,
    const double maximumInsertionDeparture = 0.006,
    const double maximumResidual = 2.0e-5,
    const double maximumAngularDeparture = 0.03
) {
    require(
        q.size() == 8u && maximumInsertionDeparture >= 0.0 &&
            maximumResidual >= 0.0 && maximumAngularDeparture >= 0.0,
        "PSM IK seed has invalid dimensions"
    );
    q[6] = -jawCoordinate;
    q[7] = jawCoordinate;
    const Quaternion baseRotation{
        base.orientation[0],
        base.orientation[1],
        base.orientation[2],
        base.orientation[3],
    };
    const Vec3 basePosition{
        base.position[0],
        base.position[1],
        base.position[2],
    };
    const Vec3 desiredLocal = rotate(
        conjugate(baseRotation),
        desiredWorld - basePosition
    );
    constexpr std::array<double, 3> increments{
        1.0e-4,
        1.0e-4,
        1.0e-5,
    };
    constexpr std::array<double, 3> coordinateScales{
        0.20,
        0.20,
        1.0,
    };
    const std::array<double, 3> maximumSeedDeparture{
        maximumAngularDeparture,
        maximumAngularDeparture,
        maximumInsertionDeparture,
    };
    const std::array<double, 3> seed{q[0], q[1], q[2]};
    // The PSM remote-centre yaw/pitch joints move the shaft as well as the
    // jaw midpoint. Preserve the accepted distal-tool attitude by pairing
    // those coordinates with the compensating roll/wrist joints. Leaving
    // q3/q4 fixed turns a nominal Cartesian lift into an unintended needle-
    // driver rotation and can roll the needle out of an otherwise bilateral
    // groove grasp. The finite-difference Jacobian below must use the same
    // coupled coordinates as the accepted target trajectory.
    const double toolRollReference = q[0] + q[3];
    const double toolPitchReference = q[1] + q[4];
    const auto preserveToolAttitude = [&](std::vector<double>& value) {
        value[3] = toolRollReference - value[0];
        value[4] = toolPitchReference - value[1];
    };
    preserveToolAttitude(q);
    constexpr double kDampingSquared = 4.0e-4;
    for (std::uint32_t iteration = 0u; iteration < 32u; ++iteration) {
        const Vec3 current = jawGeometry(model, q).midpoint;
        const Vec3 error = desiredLocal - current;
        if (norm(error) <= 5.0e-6) {
            return q;
        }
        std::array<Vec3, 3> columns{};
        for (std::size_t coordinate = 0u;
             coordinate < columns.size();
             ++coordinate) {
            std::vector<double> perturbed = q;
            perturbed[coordinate] += increments[coordinate];
            preserveToolAttitude(perturbed);
            columns[coordinate] =
                (jawGeometry(model, perturbed).midpoint - current) *
                (1.0 /
                 (increments[coordinate] * coordinateScales[coordinate]));
        }
        const Vec3 normalColumn0{
            dot(columns[0], columns[0]) + kDampingSquared,
            dot(columns[1], columns[0]),
            dot(columns[2], columns[0]),
        };
        const Vec3 normalColumn1{
            dot(columns[0], columns[1]),
            dot(columns[1], columns[1]) + kDampingSquared,
            dot(columns[2], columns[1]),
        };
        const Vec3 normalColumn2{
            dot(columns[0], columns[2]),
            dot(columns[1], columns[2]),
            dot(columns[2], columns[2]) + kDampingSquared,
        };
        const Vec3 rightHandSide{
            dot(columns[0], error),
            dot(columns[1], error),
            dot(columns[2], error),
        };
        const double determinant = dot(
            normalColumn0,
            cross(normalColumn1, normalColumn2)
        );
        require(
            std::abs(determinant) > 1.0e-12 &&
                std::isfinite(determinant),
            "PSM jaw-target damped solve reached a singular posture"
        );
        const std::array<double, 3> scaledCorrection{
            dot(rightHandSide, cross(normalColumn1, normalColumn2)) /
                determinant,
            dot(normalColumn0, cross(rightHandSide, normalColumn2)) /
                determinant,
            dot(normalColumn0, cross(normalColumn1, rightHandSide)) /
                determinant,
        };
        for (std::uint32_t coordinate = 0u;
             coordinate < 3u;
             ++coordinate) {
            const double correction = std::clamp(
                scaledCorrection[coordinate] /
                    coordinateScales[coordinate],
                coordinate == 2u ? -0.002 : -0.01,
                coordinate == 2u ? 0.002 : 0.01
            );
            q[coordinate] = std::clamp(
                q[coordinate] + correction,
                seed[coordinate] - maximumSeedDeparture[coordinate],
                seed[coordinate] + maximumSeedDeparture[coordinate]
            );
        }
        q[0] = std::clamp(q[0], -1.58, 1.58);
        q[1] = std::clamp(q[1], -0.92, 0.92);
        q[2] = std::clamp(q[2], 0.0, 0.24);
        preserveToolAttitude(q);
    }
    const double residual = norm(
        jawGeometry(model, q).midpoint - desiredLocal
    );
    require(
        residual <= maximumResidual,
        "PSM jaw-target IK did not reach the needle segment: residual=" +
            std::to_string(residual)
    );
    return q;
}

std::vector<double> armLocalQ(
    const metalrobo::EngineModel& model,
    const std::uint32_t arm,
    const std::span<const float> globalQ
) {
    require(
        globalQ.size() == model.world.nq &&
            arm < model.articulations.size(),
        "accepted arm state has invalid dimensions"
    );
    const std::uint32_t offset =
        model.articulations[arm].qOffset + 7u;
    std::vector<double> local(8u);
    for (std::uint32_t coordinate = 0u; coordinate < 8u; ++coordinate) {
        local[coordinate] = globalQ[offset + coordinate];
    }
    return local;
}

metalrobo::SurgicalBasePose armBasePose(
    const metalrobo::EngineModel& model,
    const std::uint32_t arm,
    const std::span<const float> globalQ
) {
    require(
        globalQ.size() == model.world.nq &&
            arm < model.articulations.size(),
        "accepted arm base state has invalid dimensions"
    );
    const std::uint32_t offset = model.articulations[arm].qOffset;
    require(
        offset + metalrobo::kSurgicalFloatingRootQCount <=
            globalQ.size(),
        "accepted arm base pose is outside the world state"
    );
    return {
        .position = {
            globalQ[offset],
            globalQ[offset + 1u],
            globalQ[offset + 2u],
        },
        .orientation = {
            globalQ[offset + 3u],
            globalQ[offset + 4u],
            globalQ[offset + 5u],
            globalQ[offset + 6u],
        },
    };
}

double needleShapeAngle(
    const metalrobo::CurvedSutureNeedleSpec& spec,
    const std::uint32_t shape
) {
    const double fraction =
        (static_cast<double>(shape) + 0.5) /
        static_cast<double>(spec.arcSegments);
    return -0.5 * spec.arcAngleRad.value +
        fraction * spec.arcAngleRad.value;
}

Vec3 needleShapeWorldCenter(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::uint32_t shape,
    const Vec3 bodyPosition
) {
    return bodyPosition + vector(
        needle.rigid.shapes.at(shape).localPosition
    );
}

Vec3 needleShapeWorldTangent(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::uint32_t shape,
    const MRBodyStateGPU& body
) {
    const MRShapeGPU& segment = needle.rigid.shapes.at(shape);
    return rotate(
        multiply(
            Quaternion{
                body.orientation.x,
                body.orientation.y,
                body.orientation.z,
                body.orientation.w,
            },
            Quaternion{
                segment.localRotation.x,
                segment.localRotation.y,
                segment.localRotation.z,
                segment.localRotation.w,
            }
        ),
        Vec3{0.0, 1.0, 0.0}
    );
}

struct ReceiverAlignmentSolution {
    std::vector<double> localQ;
    Vec3 desiredNeedlePoint{};
    double centeringResidual =
        std::numeric_limits<double>::infinity();
    double railTangentAngle =
        std::numeric_limits<double>::infinity();
    double separationFrameAngle =
        std::numeric_limits<double>::infinity();
    double toolRollDeparture = 0.0;
    double wristYawDeparture = 0.0;
};

ReceiverAlignmentSolution solveReceiverAlignmentTarget(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::EngineModel& psm,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::span<const float> stateQ,
    const std::span<const float> stateV,
    const MRBodyStateGPU& needleBody,
    const double jawCoordinate
) {
    require(
        stateQ.size() == world.model.world.nq &&
            stateV.size() == world.model.world.nv,
        "receiver alignment state has invalid dimensions"
    );
    const auto base = armBasePose(world.model, 1u, stateQ);
    const Quaternion baseRotation{
        base.orientation[0],
        base.orientation[1],
        base.orientation[2],
        base.orientation[3],
    };
    const Vec3 basePosition{
        base.position[0],
        base.position[1],
        base.position[2],
    };
    const std::vector<double> current = armLocalQ(
        world.model,
        1u,
        stateQ
    );
    const double currentToolRoll = current[0] + current[3];
    const double currentWristYaw = current[5];
    const Quaternion needleOrientation{
        needleBody.orientation.x,
        needleBody.orientation.y,
        needleBody.orientation.z,
        needleBody.orientation.w,
    };
    const Vec3 desiredPoint = vector(needleBody.position) + rotate(
        needleOrientation,
        vector(needle.rigid.shapes.at(
            kReceiverNeedleShape
        ).localPosition)
    );
    const Vec3 desiredTangent = needleShapeWorldTangent(
        needle,
        kReceiverNeedleShape,
        needleBody
    );
    const JawGeometry giverJaw = worldJawGeometry(
        world.model,
        0u,
        stateQ,
        stateV
    );
    const MRShapeGPU& giverSegment =
        needle.rigid.shapes.at(kGiverNeedleShape);
    const MRShapeGPU& receiverSegment =
        needle.rigid.shapes.at(kReceiverNeedleShape);
    const Quaternion giverSegmentOrientation = multiply(
        needleOrientation,
        Quaternion{
            giverSegment.localRotation.x,
            giverSegment.localRotation.y,
            giverSegment.localRotation.z,
            giverSegment.localRotation.w,
        }
    );
    const Quaternion receiverSegmentOrientation = multiply(
        needleOrientation,
        Quaternion{
            receiverSegment.localRotation.x,
            receiverSegment.localRotation.y,
            receiverSegment.localRotation.z,
            receiverSegment.localRotation.w,
        }
    );
    const Vec3 giverSeparationInSegment = rotate(
        conjugate(giverSegmentOrientation),
        giverJaw.separationDirection
    );

    ReceiverAlignmentSolution best;
    best.desiredNeedlePoint = desiredPoint;
    double bestScore = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](const double rollDeparture,
                              const double wristDeparture) {
        std::vector<double> seed = current;
        seed[3] += rollDeparture;
        seed[5] += wristDeparture;
        std::vector<double> candidate;
        try {
            candidate = solvePsmJawTarget(
                psm,
                base,
                std::move(seed),
                desiredPoint,
                jawCoordinate,
                0.010,
                2.0e-5,
                0.08
            );
        } catch (const std::exception&) {
            return;
        }
        const JawGeometry localJaw = jawGeometry(psm, candidate);
        const Vec3 worldMidpoint = basePosition + rotate(
            baseRotation,
            localJaw.midpoint
        );
        const Vec3 worldRail = rotate(
            baseRotation,
            localJaw.railDirection
        );
        const Vec3 worldSeparation = rotate(
            baseRotation,
            localJaw.separationDirection
        );
        const double railAngle = std::acos(std::clamp(
            std::abs(dot(worldRail, desiredTangent)),
            0.0,
            1.0
        ));
        const Vec3 receiverSeparationInSegment = rotate(
            conjugate(receiverSegmentOrientation),
            worldSeparation
        );
        const double separationAngle = std::acos(std::clamp(
            std::abs(dot(
                giverSeparationInSegment,
                receiverSeparationInSegment
            )),
            0.0,
            1.0
        ));
        const double centeringResidual = norm(
            worldMidpoint - desiredPoint
        );
        const double score =
            railAngle * railAngle +
            0.25 * separationAngle * separationAngle +
            1.0e3 * centeringResidual * centeringResidual +
            1.0e-4 * (
                rollDeparture * rollDeparture +
                wristDeparture * wristDeparture
            );
        if (score < bestScore) {
            bestScore = score;
            best.localQ = std::move(candidate);
            best.centeringResidual = centeringResidual;
            best.railTangentAngle = railAngle;
            best.separationFrameAngle = separationAngle;
            best.toolRollDeparture =
                (best.localQ[0] + best.localQ[3]) - currentToolRoll;
            best.wristYawDeparture =
                best.localQ[5] - currentWristYaw;
        }
    };

    for (std::int32_t roll = -10; roll <= 10; ++roll) {
        for (std::int32_t wrist = -10; wrist <= 10; ++wrist) {
            evaluate(0.025 * roll, 0.025 * wrist);
        }
    }
    require(
        !best.localQ.empty(),
        "receiver alignment search found no reachable frame target"
    );
    const double coarseRoll = best.toolRollDeparture;
    const double coarseWrist = best.wristYawDeparture;
    for (std::int32_t roll = -6; roll <= 6; ++roll) {
        for (std::int32_t wrist = -6; wrist <= 6; ++wrist) {
            evaluate(
                coarseRoll + 0.005 * roll,
                coarseWrist + 0.005 * wrist
            );
        }
    }
    require(
        best.centeringResidual <= 2.0e-5 &&
            best.railTangentAngle <= 0.03 &&
            best.separationFrameAngle <= 0.12,
        "receiver alignment search could not recover the live needle frame"
    );
    return best;
}

struct ReceiverFrameError {
    double centering = std::numeric_limits<double>::infinity();
    double railTangentAngle = std::numeric_limits<double>::infinity();
    double separationFrameAngle =
        std::numeric_limits<double>::infinity();
};

ReceiverFrameError receiverFrameError(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::span<const float> stateQ,
    const std::span<const float> stateV,
    const MRBodyStateGPU& needleBody
) {
    const JawGeometry giverJaw = worldJawGeometry(
        world.model,
        0u,
        stateQ,
        stateV
    );
    const JawGeometry receiverJaw = worldJawGeometry(
        world.model,
        1u,
        stateQ,
        stateV
    );
    const Vec3 receiverPoint = vector(needleBody.position) + rotate(
        Quaternion{
            needleBody.orientation.x,
            needleBody.orientation.y,
            needleBody.orientation.z,
            needleBody.orientation.w,
        },
        vector(needle.rigid.shapes.at(
            kReceiverNeedleShape
        ).localPosition)
    );
    const Vec3 receiverTangent = needleShapeWorldTangent(
        needle,
        kReceiverNeedleShape,
        needleBody
    );
    const Quaternion needleOrientation{
        needleBody.orientation.x,
        needleBody.orientation.y,
        needleBody.orientation.z,
        needleBody.orientation.w,
    };
    const MRShapeGPU& giverSegment =
        needle.rigid.shapes.at(kGiverNeedleShape);
    const MRShapeGPU& receiverSegment =
        needle.rigid.shapes.at(kReceiverNeedleShape);
    const Quaternion giverSegmentOrientation = multiply(
        needleOrientation,
        Quaternion{
            giverSegment.localRotation.x,
            giverSegment.localRotation.y,
            giverSegment.localRotation.z,
            giverSegment.localRotation.w,
        }
    );
    const Quaternion receiverSegmentOrientation = multiply(
        needleOrientation,
        Quaternion{
            receiverSegment.localRotation.x,
            receiverSegment.localRotation.y,
            receiverSegment.localRotation.z,
            receiverSegment.localRotation.w,
        }
    );
    const Vec3 giverSeparationInSegment = rotate(
        conjugate(giverSegmentOrientation),
        giverJaw.separationDirection
    );
    const Vec3 receiverSeparationInSegment = rotate(
        conjugate(receiverSegmentOrientation),
        receiverJaw.separationDirection
    );
    return {
        .centering = norm(receiverJaw.midpoint - receiverPoint),
        .railTangentAngle = std::acos(std::clamp(
            std::abs(dot(receiverJaw.railDirection, receiverTangent)),
            0.0,
            1.0
        )),
        .separationFrameAngle = std::acos(std::clamp(
            std::abs(dot(
                giverSeparationInSegment,
                receiverSeparationInSegment
            )),
            0.0,
            1.0
        )),
    };
}

double pointSegmentDistance(
    const Vec3 point,
    const Vec3 segmentA,
    const Vec3 segmentB
) {
    const Vec3 edge = segmentB - segmentA;
    const double lengthSquared = dot(edge, edge);
    require(lengthSquared > 0.0, "needle capsule has zero length");
    const double weight = std::clamp(
        dot(point - segmentA, edge) / lengthSquared,
        0.0,
        1.0
    );
    return norm(point - (segmentA + edge * weight));
}

double minimumToothNeedleGap(
    const metalrobo::EngineModel& psm,
    const std::span<const double> localQ,
    const metalrobo::SurgicalBasePose& base,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const Vec3 needleBodyPosition
) {
    const std::vector<double> zeroV(psm.world.nv, 0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(
        psm.articulations[0].bodyCount
    );
    const auto diagnostics = metalrobo::computeArticulatedBodyKinematics(
        psm,
        0u,
        localQ,
        zeroV,
        bodies,
        {}
    );
    require(diagnostics.succeeded(), "PSM clearance kinematics failed");
    const Quaternion baseRotation{
        base.orientation[0],
        base.orientation[1],
        base.orientation[2],
        base.orientation[3],
    };
    const Vec3 basePosition = vector(base.position);
    double minimumGap = std::numeric_limits<double>::infinity();
    constexpr std::array<std::uint32_t, 6> teeth{
        15u, 17u, 18u, 19u, 20u, 21u,
    };
    for (const std::uint32_t toothIndex : teeth) {
        const MRShapeGPU& tooth = psm.shapes.at(toothIndex);
        const Vec3 toothCenter = basePosition + rotate(
            baseRotation,
            shapeCenter(bodies, psm, toothIndex)
        );
        for (const MRShapeGPU& segment : needle.rigid.shapes) {
            const Quaternion segmentRotation{
                segment.localRotation.x,
                segment.localRotation.y,
                segment.localRotation.z,
                segment.localRotation.w,
            };
            const Vec3 center = needleBodyPosition +
                vector(segment.localPosition);
            const Vec3 halfAxis = rotate(
                segmentRotation,
                Vec3{0.0, segment.dimensions.y, 0.0}
            );
            const double gap = pointSegmentDistance(
                toothCenter,
                center - halfAxis,
                center + halfAxis
            ) - tooth.dimensions.x - segment.dimensions.x;
            minimumGap = std::min(minimumGap, gap);
        }
    }
    return minimumGap;
}

double minimumPsmPadGap(
    const metalrobo::EngineModel& psm,
    const std::span<const double> localQ,
    const metalrobo::SurgicalBasePose& base,
    const double padTop
) {
    const std::vector<double> zeroV(psm.world.nv, 0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(
        psm.articulations[0].bodyCount
    );
    const auto diagnostics = metalrobo::computeArticulatedBodyKinematics(
        psm,
        0u,
        localQ,
        zeroV,
        bodies,
        {}
    );
    require(diagnostics.succeeded(), "PSM pad-clearance kinematics failed");
    const Quaternion baseRotation{
        base.orientation[0],
        base.orientation[1],
        base.orientation[2],
        base.orientation[3],
    };
    const Vec3 basePosition = vector(base.position);
    double minimumGap = std::numeric_limits<double>::infinity();
    // Only the distal yaw/clevis/jaw envelope enters the sterile field near
    // the pad. Proximal port-side bodies are outside this local table gate.
    for (std::uint32_t shapeIndex = 12u;
         shapeIndex < psm.shapes.size();
         ++shapeIndex) {
        const MRShapeGPU& shape = psm.shapes[shapeIndex];
        const auto found = std::find_if(
            bodies.begin(),
            bodies.end(),
            [&](const metalrobo::ArticulatedBodyKinematics& body) {
                return body.bodyIndex == shape.bodyIndex;
            }
        );
        require(found != bodies.end(), "PSM pad-clearance body is missing");
        const Quaternion bodyRotation{
            found->orientation[0],
            found->orientation[1],
            found->orientation[2],
            found->orientation[3],
        };
        const Quaternion shapeRotation{
            shape.localRotation.x,
            shape.localRotation.y,
            shape.localRotation.z,
            shape.localRotation.w,
        };
        const Quaternion worldRotation = multiply(
            baseRotation,
            multiply(bodyRotation, shapeRotation)
        );
        const Vec3 center = basePosition + rotate(
            baseRotation,
            shapeCenter(bodies, psm, shapeIndex)
        );
        double verticalExtent = 0.0;
        if (shape.shapeType == MR_SHAPE_SPHERE) {
            verticalExtent = shape.dimensions.x;
        } else if (shape.shapeType == MR_SHAPE_CAPSULE ||
                   shape.shapeType == MR_SHAPE_CYLINDER) {
            const Vec3 axis = rotate(worldRotation, {0.0, 1.0, 0.0});
            const double axialExtent =
                std::abs(axis.z) * shape.dimensions.y;
            const double radialExtent = shape.shapeType == MR_SHAPE_CAPSULE
                ? shape.dimensions.x
                : shape.dimensions.x * std::sqrt(std::max(
                    0.0,
                    1.0 - axis.z * axis.z
                ));
            verticalExtent = axialExtent + radialExtent;
        } else if (shape.shapeType == MR_SHAPE_BOX) {
            const Vec3 axisX = rotate(worldRotation, {1.0, 0.0, 0.0});
            const Vec3 axisY = rotate(worldRotation, {0.0, 1.0, 0.0});
            const Vec3 axisZ = rotate(worldRotation, {0.0, 0.0, 1.0});
            verticalExtent =
                std::abs(axisX.z) * shape.dimensions.x +
                std::abs(axisY.z) * shape.dimensions.y +
                std::abs(axisZ.z) * shape.dimensions.z;
        } else {
            continue;
        }
        minimumGap = std::min(
            minimumGap,
            center.z - verticalExtent - padTop
        );
    }
    return minimumGap;
}

Vec3 needleShapeWorldCenter(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::uint32_t shape,
    const MRBodyStateGPU& body
) {
    return vector(body.position) + rotate(
        Quaternion{
            body.orientation.x,
            body.orientation.y,
            body.orientation.z,
            body.orientation.w,
        },
        vector(needle.rigid.shapes.at(shape).localPosition)
    );
}

struct GraspKinematics {
    double seatDrift = 0.0;
    double relativePointSpeed = 0.0;
    double relativeAngularSpeed = 0.0;
    double relativeNeedleTangentSpin = 0.0;
    double jawPointSpeed = 0.0;
    double needlePointSpeed = 0.0;
};

struct GraspReference {
    Vec3 jawMidpointOffsetInNeedleFrame{};
};

GraspReference graspReference(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const metalrobo::MetalWorldResult& state,
    const std::uint32_t arm,
    const std::uint32_t needleShape
) {
    require(
        !state.finalSceneBodies.empty(),
        "grasp reference is missing the needle body"
    );
    const JawGeometry jaw = worldJawGeometry(
        world.model,
        arm,
        state.finalQ,
        state.finalV
    );
    const MRBodyStateGPU& body = state.finalSceneBodies[0];
    const Vec3 needlePoint = needleShapeWorldCenter(
        needle,
        needleShape,
        body
    );
    const Quaternion orientation{
        body.orientation.x,
        body.orientation.y,
        body.orientation.z,
        body.orientation.w,
    };
    return {
        .jawMidpointOffsetInNeedleFrame = rotate(
            conjugate(orientation),
            jaw.midpoint - needlePoint
        ),
    };
}

GraspKinematics graspKinematics(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const metalrobo::MetalWorldResult& state,
    const std::uint32_t arm,
    const std::uint32_t needleShape,
    const GraspReference& reference
) {
    require(
        !state.finalSceneBodies.empty(),
        "grasp kinematics are missing the needle body"
    );
    const JawGeometry jaw = worldJawGeometry(
        world.model,
        arm,
        state.finalQ,
        state.finalV
    );
    const MRBodyStateGPU& body = state.finalSceneBodies[0];
    const Vec3 needlePoint = needleShapeWorldCenter(
        needle,
        needleShape,
        body
    );
    const Vec3 angularVelocity = vector(body.angularVelocity);
    const Quaternion needleOrientation{
        body.orientation.x,
        body.orientation.y,
        body.orientation.z,
        body.orientation.w,
    };
    const Vec3 currentSeatOffset = rotate(
        conjugate(needleOrientation),
        jaw.midpoint - needlePoint
    );
    const MRShapeGPU& handlingSegment =
        needle.rigid.shapes.at(needleShape);
    const Vec3 segmentTangent = rotate(
        multiply(
            needleOrientation,
            Quaternion{
                handlingSegment.localRotation.x,
                handlingSegment.localRotation.y,
                handlingSegment.localRotation.z,
                handlingSegment.localRotation.w,
            }
        ),
        Vec3{0.0, 1.0, 0.0}
    );
    const Vec3 relativeAngularVelocity =
        angularVelocity - jaw.meanAngularVelocity;
    const Vec3 needlePointVelocity =
        vector(body.linearVelocityAndInverseMass) +
        cross(angularVelocity, needlePoint - vector(body.position));
    return {
        .seatDrift = norm(
            currentSeatOffset -
            reference.jawMidpointOffsetInNeedleFrame
        ),
        .relativePointSpeed =
            norm(needlePointVelocity - jaw.midpointVelocity),
        .relativeAngularSpeed = norm(relativeAngularVelocity),
        .relativeNeedleTangentSpin =
            std::abs(dot(relativeAngularVelocity, segmentTangent)),
        .jawPointSpeed = norm(jaw.midpointVelocity),
        .needlePointSpeed = norm(needlePointVelocity),
    };
}

bool qualifiedDrivenGrasp(const GraspKinematics& motion) {
    return
        motion.seatDrift <= kMaximumQualifiedGraspSeatDrift &&
        motion.relativePointSpeed <=
            kMaximumQualifiedRelativePointSpeed &&
        motion.relativeAngularSpeed <=
            kMaximumQualifiedRelativeAngularSpeed;
}

bool qualifiedTransitionGrasp(const GraspKinematics& motion) {
    return
        motion.seatDrift <= kMaximumTransitionGraspReseating &&
        motion.relativePointSpeed <=
            kMaximumQualifiedRelativePointSpeed &&
        motion.relativeAngularSpeed <=
            kMaximumQualifiedRelativeAngularSpeed;
}

const MRMetalWorldContactStatusGPU& requireTerminalResidual(
    const metalrobo::MetalWorldResult& state,
    const std::string_view phase,
    const bool enforceEndpointBounds = true
) {
    require(
        state.layout.dispatch.environmentCount == 1u &&
            state.layout.dispatch.controlStepCount != 0u &&
            state.contactStatuses.size() ==
                state.layout.contactStatusElements &&
            state.contactStatuses.size() ==
                state.layout.dispatch.controlStepCount,
        std::string(phase) +
            " did not publish its terminal contact residual certificate"
    );
    const MRMetalWorldContactStatusGPU& contact =
        state.contactStatuses.back();
    require(
        contact.environment == 0u &&
            contact.controlStep + 1u ==
                state.layout.dispatch.controlStepCount,
        std::string(phase) +
            " terminal contact residual certificate has the wrong index"
    );
    const auto finite4 = [](const mr_float4 value) {
        return
            std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z) && std::isfinite(value.w);
    };
    require(
        contact.code == MR_STEP_SUCCESS &&
            contact.solverIterations != 0u &&
            finite4(contact.residuals) &&
            (
                !enforceEndpointBounds ||
                (
                    contact.residuals.y <=
                        kMaximumTerminalContactVelocityResidual &&
                    contact.residuals.z <=
                        kMaximumTerminalConeViolation
                )
            ),
        std::string(phase) +
            " did not satisfy the live temporal-cone residual bounds: "
            "code=" + std::to_string(contact.code) +
            " iterations=" + std::to_string(contact.solverIterations) +
            " residuals=[" + std::to_string(contact.residuals.x) + "," +
            std::to_string(contact.residuals.y) + "," +
            std::to_string(contact.residuals.z) + "," +
            std::to_string(contact.residuals.w) + "]"
    );
    return contact;
}

void setArmTarget(
    std::vector<float>& globalQ,
    const metalrobo::EngineModel& model,
    const std::uint32_t arm,
    const std::span<const double> localQ
) {
    const std::uint32_t offset =
        model.articulations.at(arm).qOffset + 7u;
    require(
        localQ.size() == 8u &&
            offset + localQ.size() <= globalQ.size(),
        "dual PSM target has invalid dimensions"
    );
    for (std::size_t index = 0u; index < localQ.size(); ++index) {
        globalQ[offset + index] = static_cast<float>(localQ[index]);
    }
}

std::vector<float> computedTorquePositionTarget(
    const metalrobo::EngineModel& model,
    const std::span<const float> desiredQ,
    const std::span<const double> desiredV,
    const std::span<const double> desiredAcceleration
) {
    require(
        desiredQ.size() == model.world.nq &&
            desiredV.size() == model.world.nv &&
            desiredAcceleration.size() == model.world.nv,
        "computed-torque target has invalid dimensions"
    );
    std::vector<float> command(model.world.nv, 0.0f);
    for (std::uint32_t dof = 0u; dof < model.world.nv; ++dof) {
        const MRDofPropertiesGPU& properties = model.dofs[dof];
        if (properties.qIndex != MR_INVALID_INDEX &&
            properties.qIndex < desiredQ.size()) {
            command[dof] = desiredQ[properties.qIndex];
        }
    }
    metalrobo::ArticulatedDynamicsConfig dynamics;
    dynamics.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    dynamics.timestep = model.world.gravityAndTimestep.w;
    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < model.articulations.size();
         ++articulationIndex) {
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        const std::vector<double> localQ(
            desiredQ.begin() + articulation.qOffset,
            desiredQ.begin() + articulation.qOffset + articulation.nq
        );
        const std::span<const double> localV = desiredV.subspan(
            articulation.vOffset,
            articulation.nv
        );
        const std::span<const double> localAcceleration =
            desiredAcceleration.subspan(
                articulation.vOffset,
                articulation.nv
            );
        std::vector<double> feedforwardForce(articulation.nv, 0.0);
        const auto diagnostics =
            metalrobo::computeArticulatedInverseDynamics(
                model,
                articulationIndex,
                localQ,
                localV,
                localAcceleration,
                {},
                feedforwardForce,
                dynamics
            );
        require(
            diagnostics.succeeded(),
            "PSM computed-torque inverse dynamics failed"
        );
        for (std::uint32_t localDof = 0u;
             localDof < articulation.nv;
             ++localDof) {
            const std::uint32_t globalDof =
                articulation.vOffset + localDof;
            const MRDofPropertiesGPU& properties =
                model.dofs[globalDof];
            if ((properties.flags & MR_DOF_FLAG_DRIVE) == 0u ||
                properties.qIndex == MR_INVALID_INDEX ||
                !(properties.drive.x > 0.0f)) {
                continue;
            }
            double target =
                desiredQ[properties.qIndex] +
                (
                    feedforwardForce[localDof] +
                    (
                        properties.drive.y +
                        properties.drive.x *
                            (kControlTimestep / kPhysicsSubsteps)
                    ) * localV[localDof]
                ) / properties.drive.x;
            if ((properties.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u) {
                target = std::clamp(
                    target,
                    static_cast<double>(properties.limits.x),
                    static_cast<double>(properties.limits.y)
                );
            }
            command[globalDof] = static_cast<float>(target);
        }
    }
    return command;
}

std::vector<float> computedTorqueTrajectory(
    const metalrobo::EngineModel& model,
    const std::span<const float> initialQ,
    const std::span<const float> desiredQ,
    const std::uint32_t steps
) {
    require(
        initialQ.size() == model.world.nq &&
            desiredQ.size() ==
                static_cast<std::size_t>(steps) * model.world.nq &&
            steps != 0u,
        "computed-torque trajectory has invalid dimensions"
    );
    std::vector<float> efforts(
        static_cast<std::size_t>(steps) * model.world.nv,
        0.0f
    );
    std::vector<double> previousV(model.world.nv, 0.0);
    std::vector<float> previousQ(initialQ.begin(), initialQ.end());
    for (std::uint32_t step = 0u; step < steps; ++step) {
        const std::span<const float> currentQ = desiredQ.subspan(
            static_cast<std::size_t>(step) * model.world.nq,
            model.world.nq
        );
        std::vector<double> velocity(model.world.nv, 0.0);
        std::vector<double> acceleration(model.world.nv, 0.0);
        for (std::uint32_t dof = 0u; dof < model.world.nv; ++dof) {
            const MRDofPropertiesGPU& properties = model.dofs[dof];
            if (properties.qIndex == MR_INVALID_INDEX ||
                (properties.flags & MR_DOF_FLAG_ROOT) != 0u) {
                continue;
            }
            velocity[dof] =
                (currentQ[properties.qIndex] -
                 previousQ[properties.qIndex]) /
                kControlTimestep;
            acceleration[dof] =
                (velocity[dof] - previousV[dof]) /
                kControlTimestep;
        }
        const std::vector<float> command = computedTorquePositionTarget(
            model,
            currentQ,
            velocity,
            acceleration
        );
        std::copy(
            command.begin(),
            command.end(),
            efforts.begin() +
                static_cast<std::size_t>(step) * model.world.nv
        );
        previousQ.assign(currentQ.begin(), currentQ.end());
        previousV = std::move(velocity);
    }
    return efforts;
}

std::vector<float> interpolateTargets(
    const metalrobo::EngineModel& model,
    const std::vector<float>& begin,
    const std::vector<float>& end,
    const std::uint32_t steps
) {
    require(
        begin.size() == model.defaultQ.size() &&
            end.size() == model.defaultQ.size() && steps != 0u,
        "phase position target has invalid dimensions"
    );
    std::vector<float> desiredTrajectory(
        static_cast<std::size_t>(steps) * model.world.nq,
        0.0f
    );
    std::vector<float> desired = begin;
    for (std::uint32_t step = 0u; step < steps; ++step) {
        const double t = static_cast<double>(step + 1u) / steps;
        const double smooth = t * t * (3.0 - 2.0 * t);
        for (std::uint32_t dof = 0u; dof < model.world.nv; ++dof) {
            const MRDofPropertiesGPU& properties = model.dofs[dof];
            if (properties.qIndex == MR_INVALID_INDEX ||
                (properties.flags & MR_DOF_FLAG_ROOT) != 0u) {
                continue;
            }
            desired[properties.qIndex] = static_cast<float>(
                begin[properties.qIndex] +
                smooth * (
                    end[properties.qIndex] - begin[properties.qIndex]
                )
            );
        }
        std::copy(
            desired.begin(),
            desired.end(),
            desiredTrajectory.begin() +
                static_cast<std::size_t>(step) * model.world.nq
        );
    }
    return computedTorqueTrajectory(
        model,
        begin,
        desiredTrajectory,
        steps
    );
}

std::uint32_t velocityLimitedTargetSteps(
    const metalrobo::EngineModel& model,
    const std::span<const float> begin,
    const std::span<const float> end,
    const std::uint32_t minimumSteps
) {
    require(
        begin.size() == model.world.nq &&
            end.size() == model.world.nq && minimumSteps != 0u,
        "velocity-limited target has invalid dimensions"
    );
    std::uint32_t steps = minimumSteps;
    for (std::uint32_t dof = 0u; dof < model.world.nv; ++dof) {
        const MRDofPropertiesGPU& properties = model.dofs[dof];
        if (properties.qIndex == MR_INVALID_INDEX ||
            (properties.flags & MR_DOF_FLAG_ROOT) != 0u ||
            (properties.flags & MR_DOF_FLAG_VELOCITY_LIMIT) == 0u ||
            !(properties.limits.z > 0.0f)) {
            continue;
        }
        // A cubic smoothstep peaks at 1.5 times its mean coordinate
        // velocity. Retain the same 0.8 authored-limit margin used by the
        // Cartesian trajectory gate and leave four endpoint samples for
        // float/discrete cadence headroom.
        const double required =
            1.5 * std::abs(
                static_cast<double>(end[properties.qIndex]) -
                static_cast<double>(begin[properties.qIndex])
            ) /
            (
                kMaximumCommandVelocityRatio *
                static_cast<double>(properties.limits.z) *
                kControlTimestep
            );
        steps = std::max(
            steps,
            static_cast<std::uint32_t>(std::ceil(required)) + 4u
        );
    }
    return steps;
}

struct ArmTrajectory {
    std::vector<float> efforts;
    std::vector<float> desiredQ;
    std::vector<float> finalTarget;
    double maximumVelocityRatio = 0.0;
    double maximumVelocity = 0.0;
    double limitingVelocity = 0.0;
    std::uint32_t maximumVelocityDof = MR_INVALID_INDEX;
};

ArmTrajectory cartesianArmTrajectory(
    const metalrobo::EngineModel& worldModel,
    const metalrobo::EngineModel& psm,
    const std::uint32_t arm,
    const metalrobo::SurgicalBasePose& base,
    const std::vector<float>& begin,
    const Vec3 beginJawMidpoint,
    const Vec3 endJawMidpoint,
    const double beginJawCoordinate,
    const double endJawCoordinate,
    const std::uint32_t steps
) {
    require(
        begin.size() == worldModel.defaultQ.size() && steps != 0u,
        "Cartesian arm trajectory has invalid dimensions"
    );
    ArmTrajectory trajectory;
    trajectory.desiredQ.assign(
        static_cast<std::size_t>(steps) * worldModel.world.nq,
        0.0f
    );
    trajectory.finalTarget = begin;
    std::vector<double> local = armLocalQ(worldModel, arm, begin);
    for (std::uint32_t step = 0u; step < steps; ++step) {
        const double t = static_cast<double>(step + 1u) / steps;
        const double smooth = t * t * (3.0 - 2.0 * t);
        const Vec3 desired = beginJawMidpoint +
            (endJawMidpoint - beginJawMidpoint) * smooth;
        const double jawCoordinate =
            beginJawCoordinate +
            smooth * (endJawCoordinate - beginJawCoordinate);
        local = solvePsmJawTarget(
            psm,
            base,
            std::move(local),
            desired,
            jawCoordinate,
            0.003,
            2.0e-5,
            0.03
        );
        trajectory.finalTarget = begin;
        setArmTarget(
            trajectory.finalTarget,
            worldModel,
            arm,
            local
        );
        std::copy(
            trajectory.finalTarget.begin(),
            trajectory.finalTarget.end(),
            trajectory.desiredQ.begin() +
                static_cast<std::size_t>(step) * worldModel.world.nq
        );
    }
    std::vector<float> previousQ(begin.begin(), begin.end());
    for (std::uint32_t step = 0u; step < steps; ++step) {
        const std::span<const float> currentQ{
            trajectory.desiredQ.data() +
                static_cast<std::size_t>(step) * worldModel.world.nq,
            worldModel.world.nq,
        };
        for (std::uint32_t dof = 0u;
             dof < worldModel.world.nv;
             ++dof) {
            const MRDofPropertiesGPU& properties =
                worldModel.dofs[dof];
            if (properties.qIndex == MR_INVALID_INDEX ||
                (properties.flags & MR_DOF_FLAG_ROOT) != 0u ||
                (properties.flags & MR_DOF_FLAG_VELOCITY_LIMIT) == 0u ||
                !(properties.limits.z > 0.0f)) {
                continue;
            }
            const double velocity = std::abs(
                static_cast<double>(currentQ[properties.qIndex]) -
                previousQ[properties.qIndex]
            ) / kControlTimestep;
            const double ratio =
                velocity / static_cast<double>(properties.limits.z);
            if (ratio > trajectory.maximumVelocityRatio) {
                trajectory.maximumVelocityRatio = ratio;
                trajectory.maximumVelocity = velocity;
                trajectory.limitingVelocity = properties.limits.z;
                trajectory.maximumVelocityDof = dof;
            }
        }
        previousQ.assign(currentQ.begin(), currentQ.end());
    }
    trajectory.efforts = computedTorqueTrajectory(
        worldModel,
        begin,
        trajectory.desiredQ,
        steps
    );
    return trajectory;
}

struct ContactCounts {
    std::array<std::array<std::uint32_t, 2>, 2> jawContacts{};
    std::array<std::uint32_t, 2> graspZoneContacts{};
    std::uint32_t needlePadContacts = 0u;
    std::array<std::uint32_t, 2> armPadContacts{};
    std::array<std::uint32_t, 2> armNeedleContacts{};
    std::array<std::uint32_t, 2> firstArmPadCollider{
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    std::array<std::uint32_t, 2> firstArmNeedleCollider{
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
    };
    std::uint32_t crossArmContacts = 0u;
    std::uint32_t firstCrossArmBodyA = MR_INVALID_INDEX;
    std::uint32_t firstCrossArmBodyB = MR_INVALID_INDEX;
    std::uint32_t firstCrossArmColliderA = MR_INVALID_INDEX;
    std::uint32_t firstCrossArmColliderB = MR_INVALID_INDEX;
    double minimumCrossArmSeparation =
        std::numeric_limits<double>::infinity();
    double maximumCrossArmNormalImpulse = 0.0;
    std::uint32_t generalizedConstraints = 0u;
    std::uint32_t rodAttachmentConstraints = 0u;
    std::uint32_t rodTangentAttachmentConstraints = 0u;
    std::uint32_t rodTwistAttachmentConstraints = 0u;
    std::array<std::array<double, 3>, 2> rodAttachmentImpulses{};
    std::array<double, 2> rodTangentAttachmentImpulses{};
    std::uint32_t rodContacts = 0u;
    double maximumJawNormalImpulse = 0.0;
    double maximumJawFrictionUtilization = 0.0;
    double maximumGeneralizedImpulse = 0.0;
    double maximumRodAttachmentImpulse = 0.0;
    double maximumRodTwistAttachmentImpulse = 0.0;
    double maximumRodNormalImpulse = 0.0;
    double minimumRodSeparation = std::numeric_limits<double>::infinity();
};

std::string vectorSummary(Vec3 value);

ContactCounts contactCounts(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldResult& result,
    const metalrobo::CurvedSutureNeedleMetadata& needleMetadata,
    const std::uint32_t needleFirstShape
) {
    ContactCounts counts;
    if (result.contactStatuses.empty()) {
        return counts;
    }
    const std::size_t required = std::min<std::size_t>(
        result.contactStatuses.back().requiredConstraints,
        result.contactEvidence.blocks.size()
    );
    const std::array<std::array<std::uint32_t, 2>, 2> jawBodies{{
        {{
            world.model.articulations[0].firstBody + 7u,
            world.model.articulations[0].firstBody + 8u,
        }},
        {{
            world.model.articulations[1].firstBody + 7u,
            world.model.articulations[1].firstBody + 8u,
        }},
    }};
    const std::uint32_t needleBody = world.sceneBodyIndices.at(0u);
    const std::uint32_t padBody = world.sceneBodyIndices.at(1u);
    for (std::size_t constraint = 0u;
         constraint < required;
         ++constraint) {
        const auto& block = result.contactEvidence.blocks[constraint];
        if ((block.flags & MR_CONSTRAINT_IR_BLOCK_GENERALIZED) != 0u &&
            constraint < result.contactEvidence.contacts.size()) {
            ++counts.generalizedConstraints;
            const mr_float4 impulse =
                result.contactEvidence.contacts[constraint].impulses;
            counts.maximumGeneralizedImpulse = std::max({
                counts.maximumGeneralizedImpulse,
                std::abs(static_cast<double>(impulse.x)),
                std::abs(static_cast<double>(impulse.y)),
                std::abs(static_cast<double>(impulse.z)),
            });
            if ((block.flags &
                 MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT) != 0u) {
                ++counts.rodAttachmentConstraints;
                const std::uint32_t attachment = block.key.words[2];
                const std::uint32_t axis = block.key.words[3];
                if ((block.flags &
                     MR_CONSTRAINT_IR_BLOCK_ROD_TANGENT_ATTACHMENT) != 0u) {
                    ++counts.rodTangentAttachmentConstraints;
                    if (axis <
                        counts.rodTangentAttachmentImpulses.size()) {
                        counts.rodTangentAttachmentImpulses[axis] =
                            static_cast<double>(impulse.x);
                    }
                } else if (
                    attachment < counts.rodAttachmentImpulses.size() &&
                    axis < 3u
                ) {
                    counts.rodAttachmentImpulses[attachment][axis] =
                        static_cast<double>(impulse.x);
                }
                counts.maximumRodAttachmentImpulse = std::max({
                    counts.maximumRodAttachmentImpulse,
                    std::abs(static_cast<double>(impulse.x)),
                    std::abs(static_cast<double>(impulse.y)),
                    std::abs(static_cast<double>(impulse.z)),
                });
            }
            if ((block.flags &
                 MR_CONSTRAINT_IR_BLOCK_ROD_TWIST_ATTACHMENT) != 0u) {
                ++counts.rodTwistAttachmentConstraints;
                counts.maximumRodTwistAttachmentImpulse = std::max({
                    counts.maximumRodTwistAttachmentImpulse,
                    std::abs(static_cast<double>(impulse.x)),
                    std::abs(static_cast<double>(impulse.y)),
                    std::abs(static_cast<double>(impulse.z)),
                });
                counts.maximumRodAttachmentImpulse = std::max({
                    counts.maximumRodAttachmentImpulse,
                    std::abs(static_cast<double>(impulse.x)),
                    std::abs(static_cast<double>(impulse.y)),
                    std::abs(static_cast<double>(impulse.z)),
                });
            }
        }
        if ((block.flags & MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT) != 0u &&
            (block.flags & MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT) == 0u &&
            constraint < result.contactEvidence.contacts.size()) {
            const auto& contact =
                result.contactEvidence.contacts[constraint];
            if (contact.impulses.x > 1.0e-12f) {
                ++counts.rodContacts;
                counts.maximumRodNormalImpulse = std::max(
                    counts.maximumRodNormalImpulse,
                    static_cast<double>(contact.impulses.x)
                );
            }
            counts.minimumRodSeparation = std::min(
                counts.minimumRodSeparation,
                static_cast<double>(contact.pointAndSeparation.w)
            );
        }
        if ((block.flags &
             (MR_CONSTRAINT_IR_BLOCK_GENERALIZED |
              MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT |
              MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT)) != 0u ||
            block.endpointCount != 2u ||
            constraint >= result.contactEvidence.contacts.size() ||
            constraint >= result.contactEvidence.contactMetadata.size()) {
            continue;
        }
        const auto& contact =
            result.contactEvidence.contacts[constraint];
        if (!(contact.impulses.x > 1.0e-10f)) {
            continue;
        }
        const std::uint32_t bodyA = contact.bodyA;
        const std::uint32_t bodyB = contact.bodyB;
        const auto& metadata =
            result.contactEvidence.contactMetadata[constraint];
        if ((bodyA == needleBody && bodyB == padBody) ||
            (bodyA == padBody && bodyB == needleBody)) {
            ++counts.needlePadContacts;
        }
        for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
            for (std::uint32_t jaw = 0u; jaw < 2u; ++jaw) {
                const std::uint32_t jawBody = jawBodies[arm][jaw];
                if (!((bodyA == jawBody && bodyB == needleBody) ||
                      (bodyB == jawBody && bodyA == needleBody))) {
                    continue;
                }
                ++counts.jawContacts[arm][jaw];
                counts.maximumJawNormalImpulse = std::max(
                    counts.maximumJawNormalImpulse,
                    static_cast<double>(contact.impulses.x)
                );
                const double frictionLimit =
                    static_cast<double>(contact.friction.x) *
                    static_cast<double>(contact.impulses.x);
                if (frictionLimit > 0.0) {
                    counts.maximumJawFrictionUtilization = std::max(
                        counts.maximumJawFrictionUtilization,
                        std::hypot(
                            static_cast<double>(contact.impulses.y),
                            static_cast<double>(contact.impulses.z)
                        ) / frictionLimit
                    );
                }
                const std::uint32_t needleCollider =
                    bodyA == needleBody
                    ? metadata.colliderA
                    : metadata.colliderB;
                if (needleCollider >= needleFirstShape &&
                    needleCollider <
                        needleFirstShape +
                            needleMetadata.graspShapeEnd &&
                    needleCollider >=
                        needleFirstShape +
                            needleMetadata.graspShapeBegin) {
                    ++counts.graspZoneContacts[arm];
                }
            }
        }
        const bool aArm0 =
            bodyA >= world.model.articulations[0].firstBody &&
            bodyA < world.model.articulations[0].firstBody +
                world.model.articulations[0].bodyCount;
        const bool bArm0 =
            bodyB >= world.model.articulations[0].firstBody &&
            bodyB < world.model.articulations[0].firstBody +
                world.model.articulations[0].bodyCount;
        const bool aArm1 =
            bodyA >= world.model.articulations[1].firstBody &&
            bodyA < world.model.articulations[1].firstBody +
                world.model.articulations[1].bodyCount;
        const bool bArm1 =
            bodyB >= world.model.articulations[1].firstBody &&
            bodyB < world.model.articulations[1].firstBody +
                world.model.articulations[1].bodyCount;
        if ((aArm0 && bodyB == padBody) ||
            (bArm0 && bodyA == padBody)) {
            ++counts.armPadContacts[0];
            counts.firstArmPadCollider[0] = std::min(
                counts.firstArmPadCollider[0],
                aArm0 ? metadata.colliderA : metadata.colliderB
            );
        }
        if ((aArm1 && bodyB == padBody) ||
            (bArm1 && bodyA == padBody)) {
            ++counts.armPadContacts[1];
            counts.firstArmPadCollider[1] = std::min(
                counts.firstArmPadCollider[1],
                aArm1 ? metadata.colliderA : metadata.colliderB
            );
        }
        if ((aArm0 && bodyB == needleBody) ||
            (bArm0 && bodyA == needleBody)) {
            ++counts.armNeedleContacts[0];
            counts.firstArmNeedleCollider[0] = std::min(
                counts.firstArmNeedleCollider[0],
                aArm0 ? metadata.colliderA : metadata.colliderB
            );
        }
        if ((aArm1 && bodyB == needleBody) ||
            (bArm1 && bodyA == needleBody)) {
            ++counts.armNeedleContacts[1];
            counts.firstArmNeedleCollider[1] = std::min(
                counts.firstArmNeedleCollider[1],
                aArm1 ? metadata.colliderA : metadata.colliderB
            );
        }
        if ((aArm0 && bArm1) || (aArm1 && bArm0)) {
            ++counts.crossArmContacts;
            const double separation = static_cast<double>(
                contact.pointAndSeparation.w
            );
            if (separation < counts.minimumCrossArmSeparation) {
                counts.minimumCrossArmSeparation = separation;
                counts.firstCrossArmBodyA = bodyA;
                counts.firstCrossArmBodyB = bodyB;
                counts.firstCrossArmColliderA = metadata.colliderA;
                counts.firstCrossArmColliderB = metadata.colliderB;
            }
            counts.maximumCrossArmNormalImpulse = std::max(
                counts.maximumCrossArmNormalImpulse,
                static_cast<double>(contact.impulses.x)
            );
        }
    }
    return counts;
}

bool bilateral(const ContactCounts& counts, const std::uint32_t arm) {
    return counts.jawContacts[arm][0] != 0u &&
        counts.jawContacts[arm][1] != 0u &&
        counts.graspZoneContacts[arm] >= 2u;
}

bool cleanNeedleInteraction(
    const ContactCounts& counts,
    const bool giverControlsNeedle,
    const bool receiverControlsNeedle,
    const bool needleMustClearPad = true
) {
    const std::array<bool, 2> controls{
        giverControlsNeedle,
        receiverControlsNeedle,
    };
    for (std::uint32_t arm = 0u; arm < controls.size(); ++arm) {
        const std::uint32_t jawContactCount =
            counts.jawContacts[arm][0] + counts.jawContacts[arm][1];
        if (counts.armPadContacts[arm] != 0u ||
            counts.armNeedleContacts[arm] !=
                (controls[arm] ? jawContactCount : 0u)) {
            return false;
        }
    }
    return
        (!needleMustClearPad || counts.needlePadContacts == 0u) &&
        counts.crossArmContacts == 0u &&
        counts.maximumJawFrictionUtilization <= 1.0001 &&
        counts.minimumRodSeparation >= kMinimumSettledRodSeparation;
}

bool cleanGiverNeedleInteraction(
    const ContactCounts& counts,
    const bool needleMustClearPad = true
) {
    return cleanNeedleInteraction(
        counts,
        true,
        false,
        needleMustClearPad
    );
}

bool cleanReceiverApproachNeedleInteraction(
    const ContactCounts& counts
) {
    const std::uint32_t receiverJawContacts =
        counts.jawContacts[1][0] + counts.jawContacts[1][1];
    return
        !bilateral(counts, 1u) &&
        counts.graspZoneContacts[1] == receiverJawContacts &&
        cleanNeedleInteraction(counts, true, true);
}

std::string contactSummary(const ContactCounts& counts) {
    return
        " giver_jaws=" +
        std::to_string(counts.jawContacts[0][0]) + "/" +
        std::to_string(counts.jawContacts[0][1]) +
        " receiver_jaws=" +
        std::to_string(counts.jawContacts[1][0]) + "/" +
        std::to_string(counts.jawContacts[1][1]) +
        " grasp_zone=" +
        std::to_string(counts.graspZoneContacts[0]) + "/" +
        std::to_string(counts.graspZoneContacts[1]) +
        " pad=" + std::to_string(counts.needlePadContacts) +
        " arm_pad=" +
        std::to_string(counts.armPadContacts[0]) + "/" +
        std::to_string(counts.armPadContacts[1]) +
        " arm_needle=" +
        std::to_string(counts.armNeedleContacts[0]) + "/" +
        std::to_string(counts.armNeedleContacts[1]) +
        " first_arm_pad_collider=" +
        std::to_string(counts.firstArmPadCollider[0]) + "/" +
        std::to_string(counts.firstArmPadCollider[1]) +
        " first_arm_needle_collider=" +
        std::to_string(counts.firstArmNeedleCollider[0]) + "/" +
        std::to_string(counts.firstArmNeedleCollider[1]) +
        " cross_arm=" + std::to_string(counts.crossArmContacts) +
        " cross_arm_bodies=" +
        std::to_string(counts.firstCrossArmBodyA) + "/" +
        std::to_string(counts.firstCrossArmBodyB) +
        " cross_arm_colliders=" +
        std::to_string(counts.firstCrossArmColliderA) + "/" +
        std::to_string(counts.firstCrossArmColliderB) +
        " min_cross_arm_separation=" +
        std::to_string(
            std::isfinite(counts.minimumCrossArmSeparation)
                ? counts.minimumCrossArmSeparation
                : 0.0
        ) +
        " max_cross_arm_impulse=" +
        std::to_string(counts.maximumCrossArmNormalImpulse) +
        " rod_contacts=" + std::to_string(counts.rodContacts) +
        " max_rod_impulse=" +
        std::to_string(counts.maximumRodNormalImpulse) +
        " min_rod_separation=" +
        std::to_string(
            std::isfinite(counts.minimumRodSeparation)
                ? counts.minimumRodSeparation
                : 0.0
        ) +
        " generalized=" +
        std::to_string(counts.generalizedConstraints) +
        " rod_attachment=" +
        std::to_string(counts.rodAttachmentConstraints) +
        " rod_tangent_attachment=" +
        std::to_string(counts.rodTangentAttachmentConstraints) +
        " rod_twist_attachment=" +
        std::to_string(counts.rodTwistAttachmentConstraints) +
        " max_generalized_impulse=" +
        std::to_string(counts.maximumGeneralizedImpulse) +
        " max_rod_attachment_impulse=" +
        std::to_string(counts.maximumRodAttachmentImpulse) +
        " rod_attachment0_impulse=" +
        vectorSummary({
            counts.rodAttachmentImpulses[0][0],
            counts.rodAttachmentImpulses[0][1],
            counts.rodAttachmentImpulses[0][2],
        }) +
        " rod_attachment1_impulse=" +
        vectorSummary({
            counts.rodAttachmentImpulses[1][0],
            counts.rodAttachmentImpulses[1][1],
            counts.rodAttachmentImpulses[1][2],
        }) +
        " rod_tangent_attachment_impulse=[" +
        std::to_string(counts.rodTangentAttachmentImpulses[0]) + "," +
        std::to_string(counts.rodTangentAttachmentImpulses[1]) + "]" +
        " max_rod_twist_attachment_impulse=" +
        std::to_string(counts.maximumRodTwistAttachmentImpulse) +
        " max_normal_impulse=" +
        std::to_string(counts.maximumJawNormalImpulse) +
        " max_jaw_friction_utilization=" +
        std::to_string(counts.maximumJawFrictionUtilization);
}

std::string vectorSummary(const Vec3 value) {
    return
        "[" + std::to_string(value.x) + "," +
        std::to_string(value.y) + "," +
        std::to_string(value.z) + "]";
}

struct CrossArmCollisionScan {
    std::uint32_t samplesWithContact = 0u;
    std::uint32_t firstContactStep = MR_INVALID_INDEX;
    std::uint32_t lastContactStep = MR_INVALID_INDEX;
    std::uint32_t colliderA = MR_INVALID_INDEX;
    std::uint32_t colliderB = MR_INVALID_INDEX;
    std::uint32_t bodyA = MR_INVALID_INDEX;
    std::uint32_t bodyB = MR_INVALID_INDEX;
    double minimumSeparation = std::numeric_limits<double>::infinity();
    std::uint32_t samplesWithGiverPadContact = 0u;
    std::uint32_t firstGiverPadContactStep = MR_INVALID_INDEX;
    std::uint32_t lastGiverPadContactStep = MR_INVALID_INDEX;
    std::uint32_t giverPadCollider = MR_INVALID_INDEX;
    double minimumGiverPadSeparation =
        std::numeric_limits<double>::infinity();
    std::uint32_t samplesWithReceiverPadContact = 0u;
    std::uint32_t firstReceiverPadContactStep = MR_INVALID_INDEX;
    std::uint32_t lastReceiverPadContactStep = MR_INVALID_INDEX;
    std::uint32_t receiverPadCollider = MR_INVALID_INDEX;
    double minimumReceiverPadSeparation =
        std::numeric_limits<double>::infinity();
};

bool articulationOwnsBody(
    const metalrobo::EngineModel& model,
    const std::uint32_t articulation,
    const std::uint32_t body
) {
    const MRArticulationGPU& owner = model.articulations.at(articulation);
    return body >= owner.firstBody &&
        body < owner.firstBody + owner.bodyCount;
}

std::string semanticName(
    const std::span<const std::string> names,
    const std::uint32_t index
) {
    if (index < names.size() && !names[index].empty()) {
        return names[index];
    }
    return std::to_string(index);
}

CrossArmCollisionScan scanCrossArmTargetPath(
    const metalrobo::HeterogeneousWorld& world,
    const std::span<const float> begin,
    const std::span<const float> end,
    const std::uint32_t steps,
    const std::span<const float> explicitTrajectory = {}
) {
    require(
        begin.size() == world.model.world.nq &&
            end.size() == world.model.world.nq &&
            steps != 0u &&
            (
                explicitTrajectory.empty() ||
                explicitTrajectory.size() ==
                    static_cast<std::size_t>(steps) *
                        world.model.world.nq
            ),
        "cross-arm target scan has invalid generalized state"
    );
    metalrobo::CollisionConfig collisionConfig;
    collisionConfig.environment = 0x48414e44u;
    collisionConfig.capacities = {
        .pairCapacity = 8192u,
        .rawContactCapacity = 32768u,
        .manifoldCapacity = 8192u,
    };
    collisionConfig.manifoldBreakingSeparation = 5.0e-4;
    collisionConfig.manifoldBreakingTangential = 5.0e-4;
    collisionConfig.manifoldMergeDistance = 2.0e-5;

    CrossArmCollisionScan scan;
    std::vector<float> q(begin.begin(), begin.end());
    std::vector<float> v(world.model.world.nv, 0.0f);
    const std::uint32_t padBody = world.sceneBodyIndices.at(1u);
    for (std::uint32_t step = 0u; step <= steps; ++step) {
        if (!explicitTrajectory.empty() && step != 0u) {
            const float* sample = explicitTrajectory.data() +
                static_cast<std::size_t>(step - 1u) *
                    world.model.world.nq;
            q.assign(sample, sample + world.model.world.nq);
        } else if (explicitTrajectory.empty()) {
            const double t = static_cast<double>(step) / steps;
            const double smooth = t * t * (3.0 - 2.0 * t);
            for (std::uint32_t dof = 0u;
                 dof < world.model.world.nv;
                 ++dof) {
                const MRDofPropertiesGPU& properties =
                    world.model.dofs[dof];
                if (properties.qIndex == MR_INVALID_INDEX ||
                    (properties.flags & MR_DOF_FLAG_ROOT) != 0u) {
                    continue;
                }
                q[properties.qIndex] = static_cast<float>(
                    begin[properties.qIndex] +
                    smooth * (
                        end[properties.qIndex] - begin[properties.qIndex]
                    )
                );
            }
        }
        std::vector<MRBodyStateGPU> bodies;
        std::string reason;
        require(
            metalrobo::composeVisualBodyStates(
                world.model,
                1u,
                q,
                v,
                world.defaultSceneBodies,
                bodies,
                &reason
            ),
            "cross-arm target scan kinematics failed: " + reason
        );
        metalrobo::PersistentManifoldCache cache;
        const metalrobo::CollisionFrame collision =
            metalrobo::collideCpuReference(
                world.model.shapes,
                bodies,
                collisionConfig,
                cache,
                world.model.collisionExclusions
            );
        require(
            collision.succeeded(),
            "cross-arm target scan collision oracle failed"
        );
        std::uint32_t crossContacts = 0u;
        double stepMinimum = std::numeric_limits<double>::infinity();
        std::uint32_t stepColliderA = MR_INVALID_INDEX;
        std::uint32_t stepColliderB = MR_INVALID_INDEX;
        std::uint32_t stepBodyA = MR_INVALID_INDEX;
        std::uint32_t stepBodyB = MR_INVALID_INDEX;
        std::uint32_t giverPadContacts = 0u;
        double stepGiverPadMinimum =
            std::numeric_limits<double>::infinity();
        std::uint32_t stepGiverPadCollider = MR_INVALID_INDEX;
        std::uint32_t receiverPadContacts = 0u;
        double stepReceiverPadMinimum =
            std::numeric_limits<double>::infinity();
        std::uint32_t stepReceiverPadCollider = MR_INVALID_INDEX;
        for (std::size_t rawIndex = 0u;
             rawIndex < collision.rawContacts.size();
             ++rawIndex) {
            const std::uint32_t pairIndex =
                collision.rawContactPairIndices.at(rawIndex);
            const MRCandidatePairGPU& pair =
                collision.pairs.at(pairIndex);
            const std::uint32_t bodyA =
                world.model.shapes.at(pair.colliderA).bodyIndex;
            const std::uint32_t bodyB =
                world.model.shapes.at(pair.colliderB).bodyIndex;
            const bool cross =
                (articulationOwnsBody(world.model, 0u, bodyA) &&
                 articulationOwnsBody(world.model, 1u, bodyB)) ||
                (articulationOwnsBody(world.model, 1u, bodyA) &&
                 articulationOwnsBody(world.model, 0u, bodyB));
            const bool aGiver =
                articulationOwnsBody(world.model, 0u, bodyA);
            const bool bGiver =
                articulationOwnsBody(world.model, 0u, bodyB);
            const bool aReceiver =
                articulationOwnsBody(world.model, 1u, bodyA);
            const bool bReceiver =
                articulationOwnsBody(world.model, 1u, bodyB);
            const bool receiverPad =
                (aReceiver && bodyB == padBody) ||
                (bReceiver && bodyA == padBody);
            const bool giverPad =
                (aGiver && bodyB == padBody) ||
                (bGiver && bodyA == padBody);
            if (giverPad) {
                ++giverPadContacts;
                const double separation = static_cast<double>(
                    collision.rawContacts[rawIndex].normalAndSeparation.w
                );
                if (separation < stepGiverPadMinimum) {
                    stepGiverPadMinimum = separation;
                    stepGiverPadCollider = aGiver
                        ? pair.colliderA
                        : pair.colliderB;
                }
            }
            if (receiverPad) {
                ++receiverPadContacts;
                const double separation = static_cast<double>(
                    collision.rawContacts[rawIndex].normalAndSeparation.w
                );
                if (separation < stepReceiverPadMinimum) {
                    stepReceiverPadMinimum = separation;
                    stepReceiverPadCollider = aReceiver
                        ? pair.colliderA
                        : pair.colliderB;
                }
            }
            if (!cross) {
                continue;
            }
            ++crossContacts;
            const double separation = static_cast<double>(
                collision.rawContacts[rawIndex].normalAndSeparation.w
            );
            if (separation < stepMinimum) {
                stepMinimum = separation;
                stepColliderA = pair.colliderA;
                stepColliderB = pair.colliderB;
                stepBodyA = bodyA;
                stepBodyB = bodyB;
            }
        }
        if (giverPadContacts != 0u) {
            ++scan.samplesWithGiverPadContact;
            scan.firstGiverPadContactStep = std::min(
                scan.firstGiverPadContactStep,
                step
            );
            scan.lastGiverPadContactStep = step;
            if (stepGiverPadMinimum < scan.minimumGiverPadSeparation) {
                scan.minimumGiverPadSeparation = stepGiverPadMinimum;
                scan.giverPadCollider = stepGiverPadCollider;
            }
            std::cerr << "handoff_phase=giver_pad_collision_scan"
                << " step=" << step
                << " contacts=" << giverPadContacts
                << " minimum_separation_m=" << stepGiverPadMinimum
                << " collider=" << stepGiverPadCollider
                << " collider_name=\""
                << semanticName(
                    world.model.shapeNames,
                    stepGiverPadCollider
                ) << "\"\n";
        }
        if (receiverPadContacts != 0u) {
            ++scan.samplesWithReceiverPadContact;
            scan.firstReceiverPadContactStep = std::min(
                scan.firstReceiverPadContactStep,
                step
            );
            scan.lastReceiverPadContactStep = step;
            if (stepReceiverPadMinimum <
                scan.minimumReceiverPadSeparation) {
                scan.minimumReceiverPadSeparation = stepReceiverPadMinimum;
                scan.receiverPadCollider = stepReceiverPadCollider;
            }
            std::cerr << "handoff_phase=receiver_pad_collision_scan"
                << " step=" << step
                << " contacts=" << receiverPadContacts
                << " minimum_separation_m="
                << stepReceiverPadMinimum
                << " collider=" << stepReceiverPadCollider
                << " collider_name=\""
                << semanticName(
                    world.model.shapeNames,
                    stepReceiverPadCollider
                ) << "\"\n";
        }
        if (crossContacts == 0u) {
            continue;
        }
        ++scan.samplesWithContact;
        scan.firstContactStep = std::min(scan.firstContactStep, step);
        scan.lastContactStep = step;
        if (stepMinimum < scan.minimumSeparation) {
            scan.minimumSeparation = stepMinimum;
            scan.colliderA = stepColliderA;
            scan.colliderB = stepColliderB;
            scan.bodyA = stepBodyA;
            scan.bodyB = stepBodyB;
        }
        std::cerr << "handoff_phase=receiver_collision_scan"
            << " step=" << step
            << " cross_contacts=" << crossContacts
            << " minimum_separation_m=" << stepMinimum
            << " bodies=" << stepBodyA << '/' << stepBodyB
            << " body_names=\""
            << semanticName(world.model.bodyNames, stepBodyA) << "/"
            << semanticName(world.model.bodyNames, stepBodyB) << "\""
            << " colliders=" << stepColliderA << '/' << stepColliderB
            << " collider_names=\""
            << semanticName(world.model.shapeNames, stepColliderA) << "/"
            << semanticName(world.model.shapeNames, stepColliderB) << "\"\n";
    }
    return scan;
}

struct KinematicNeedleObservation {
    std::array<std::uint32_t, 2u> receiverJawContacts{};
    std::uint32_t receiverGraspZoneContacts = 0u;
    std::uint32_t crossArmContacts = 0u;
    double minimumCrossArmSeparation =
        std::numeric_limits<double>::infinity();
};

KinematicNeedleObservation observeKinematicNeedleContacts(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::CurvedSutureNeedleMetadata& needleMetadata,
    const std::span<const float> q
) {
    std::vector<float> v(world.model.world.nv, 0.0f);
    std::vector<MRBodyStateGPU> bodies;
    std::string reason;
    require(
        metalrobo::composeVisualBodyStates(
            world.model,
            1u,
            q,
            v,
            world.defaultSceneBodies,
            bodies,
            &reason
        ),
        "kinematic needle observation failed: " + reason
    );
    metalrobo::CollisionConfig config;
    config.environment = 0x4e454544u;
    config.capacities = {
        .pairCapacity = 8192u,
        .rawContactCapacity = 32768u,
        .manifoldCapacity = 8192u,
    };
    config.manifoldBreakingSeparation = 5.0e-4;
    config.manifoldBreakingTangential = 5.0e-4;
    config.manifoldMergeDistance = 2.0e-5;
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame collision =
        metalrobo::collideCpuReference(
            world.model.shapes,
            bodies,
            config,
            cache,
            world.model.collisionExclusions
        );
    require(
        collision.succeeded(),
        "kinematic needle collision observation failed"
    );
    const std::uint32_t needleBody = world.sceneBodyIndices.at(0u);
    const std::array<std::uint32_t, 2u> receiverJawBodies{
        world.model.articulations.at(1u).firstBody + 7u,
        world.model.articulations.at(1u).firstBody + 8u,
    };
    KinematicNeedleObservation observation;
    for (std::size_t rawIndex = 0u;
         rawIndex < collision.rawContacts.size();
         ++rawIndex) {
        const MRCandidatePairGPU& pair = collision.pairs.at(
            collision.rawContactPairIndices.at(rawIndex)
        );
        const std::uint32_t bodyA =
            world.model.shapes.at(pair.colliderA).bodyIndex;
        const std::uint32_t bodyB =
            world.model.shapes.at(pair.colliderB).bodyIndex;
        for (std::uint32_t jaw = 0u; jaw < 2u; ++jaw) {
            const std::uint32_t jawBody = receiverJawBodies[jaw];
            if (!((bodyA == jawBody && bodyB == needleBody) ||
                  (bodyB == jawBody && bodyA == needleBody))) {
                continue;
            }
            ++observation.receiverJawContacts[jaw];
            const std::uint32_t needleCollider =
                bodyA == needleBody ? pair.colliderA : pair.colliderB;
            if (needleCollider >=
                    kNeedleFirstShape + needleMetadata.graspShapeBegin &&
                needleCollider <
                    kNeedleFirstShape + needleMetadata.graspShapeEnd) {
                ++observation.receiverGraspZoneContacts;
            }
        }
        const bool cross =
            (articulationOwnsBody(world.model, 0u, bodyA) &&
             articulationOwnsBody(world.model, 1u, bodyB)) ||
            (articulationOwnsBody(world.model, 1u, bodyA) &&
             articulationOwnsBody(world.model, 0u, bodyB));
        if (cross) {
            ++observation.crossArmContacts;
            observation.minimumCrossArmSeparation = std::min(
                observation.minimumCrossArmSeparation,
                static_cast<double>(
                    collision.rawContacts[rawIndex]
                        .normalAndSeparation.w
                )
            );
        }
    }
    return observation;
}

std::string armTargetSummary(
    const metalrobo::EngineModel& model,
    const std::uint32_t arm,
    const std::span<const float> acceptedQ,
    const std::span<const float> targetQ
) {
    const std::uint32_t offset =
        model.articulations.at(arm).qOffset + 7u;
    const std::uint32_t rootOffset =
        model.articulations.at(arm).qOffset;
    std::string result = " root_error=[";
    for (std::uint32_t coordinate = 0u; coordinate < 3u; ++coordinate) {
        if (coordinate != 0u) {
            result += ',';
        }
        result += std::to_string(
            static_cast<double>(acceptedQ[rootOffset + coordinate]) -
            targetQ[rootOffset + coordinate]
        );
    }
    result += "] q_error=[";
    for (std::uint32_t coordinate = 0u; coordinate < 8u; ++coordinate) {
        if (coordinate != 0u) {
            result += ',';
        }
        result += std::to_string(
            static_cast<double>(acceptedQ[offset + coordinate]) -
            targetQ[offset + coordinate]
        );
    }
    return result + ']';
}

struct PhaseResult {
    metalrobo::MetalWorldResult result;
    metalrobo::MetalWorldDiagnostics diagnostics;
};

std::string matterCompileErrors(
    const std::vector<numi::matter::Diagnostic>& diagnostics
) {
    std::string result;
    for (const auto& diagnostic : diagnostics) {
        if (!result.empty()) {
            result += "; ";
        }
        result += std::to_string(diagnostic.line) + ":" +
            std::to_string(diagnostic.column) + " " + diagnostic.message;
    }
    return result;
}

numi::matter::CompiledWorld compileNeedleSutureTissueWorld(
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::CurvedSutureNeedleAsset& needleAsset,
    const double initialSurfaceOffsetM,
    const bool punctureTip,
    const std::uint32_t punctureContactSegmentCount,
    numi::matter::PorcineJejunumClosureCoupon& coupon
) {
    require(
        world.sceneBodyIndices.size() >= 1u &&
            world.defaultSceneBodies.size() >= 1u &&
            world.rods.size() == 1u &&
            world.rods[0].rigidBindings.size() == 1u &&
            world.rods[0].tangentBindings.size() == 1u,
        "tissue coupling requires the live needle-swage-thread topology"
    );
    auto parsed = numi::matter::parseMatterFile(NUMI_JEJUNUM_MATERIAL);
    require(
        parsed.succeeded(),
        "porcine jejunum material parse failed: " +
            matterCompileErrors(parsed.diagnostics)
    );

    // Retain the source-sized 30 x 24 x 0.77 mm porcine coupon and 16 mm
    // enterotomy. The contact-only swage regression uses the smallest
    // qualified 6x6x1 transaction mesh. A puncture uses the production
    // 18x16x2 wall so entry crosses a through-thickness volume rather than the
    // former single-layer contact surrogate.
    numi::matter::PorcineJejunumFungSpec spec;
    if (!punctureTip) {
        // Six cells per in-plane axis is the smallest topology qualified by
        // the owning surgical-tissue replay. A 4x4 mixed element block leaves
        // its normalized pressure mode dominated by rest-shape roundoff.
        spec.longitudinalCells = 6u;
        spec.circumferentialCells = 6u;
        spec.throughThicknessCells = 1u;
    }
    spec.fixLongitudinalEnds = true;
    std::string materialError;
    require(
        numi::matter::configurePorcineJejunumFungMaterial(
            parsed.material,
            spec,
            &materialError
        ),
        materialError
    );
    coupon = numi::matter::makePorcineJejunumClosureCoupon(0u, spec);
    coupon.object.mutationPolicy.punctureImpulseThreshold = punctureTip
        ? kPunctureImpulseThresholdNs
        : 0.0;

    const MRBodyStateGPU& needle = world.defaultSceneBodies[0];
    const Quaternion orientation{
        needle.orientation.x,
        needle.orientation.y,
        needle.orientation.z,
        needle.orientation.w,
    };
    const auto& swage = world.rods[0].rigidBindings[0].localAnchor;
    const auto& tangent = world.rods[0].tangentBindings[0].localAnchor;
    const NeedleTipCapsuleGeometry tip =
        needleTipCapsuleGeometry(needleAsset, needle);
    const Vec3 localEndpointA = punctureTip
        ? tip.localTip
        : Vec3{swage[0], swage[1], swage[2]};
    const Vec3 localEndpointB = punctureTip
        ? tip.localBase
        : Vec3{tangent[0], tangent[1], tangent[2]};
    const Vec3 endpointA = vector(needle.position) + rotate(
        orientation, localEndpointA
    );
    const Vec3 endpointB = vector(needle.position) + rotate(
        orientation, localEndpointB
    );
    const double needleRadiusM = punctureTip
        ? tip.radiusM
        : needleAsset.spec.crossSectionRadiusM.value;
    const Vec3 contactCenter = (endpointA + endpointB) * 0.5;
    require(
        std::isfinite(initialSurfaceOffsetM) &&
            initialSurfaceOffsetM > 0.0,
        "tissue coupling has an invalid signed surface offset"
    );
    std::uint32_t anchorNode = NM_INVALID_INDEX;
    double anchorPlanarRadiusSquared =
        std::numeric_limits<double>::infinity();
    const double topSurface = 0.5 * spec.thicknessM.value;
    const double targetLocalY = punctureTip
        ? -kPunctureBiteOffsetM
        : 0.0;
    for (const std::uint32_t node : coupon.object.femContactNodes) {
        require(
            node < coupon.object.femNodes.size(),
            "tissue contact surface contains an invalid node"
        );
        const Vec3 point = vector(coupon.object.femNodes[node]);
        if (std::abs(point.z - topSurface) > 1.0e-12) {
            continue;
        }
        const double planarRadiusSquared =
            point.x * point.x +
            (point.y - targetLocalY) * (point.y - targetLocalY);
        if (planarRadiusSquared < anchorPlanarRadiusSquared) {
            anchorPlanarRadiusSquared = planarRadiusSquared;
            anchorNode = node;
        }
    }
    require(
        anchorNode != NM_INVALID_INDEX,
        "tissue coupling has no top-surface contact anchor"
    );
    const Vec3 authoredAnchor = vector(coupon.object.femNodes[anchorNode]);
    const Vec3 capsuleEdge = endpointB - endpointA;
    const double capsuleLength = norm(capsuleEdge);
    require(capsuleLength > 0.0, "tissue coupling capsule is degenerate");
    const Vec3 capsuleAxis = capsuleEdge * (1.0 / capsuleLength);
    Vec3 targetContactPoint{};
    if (punctureTip) {
        // Align the coupon's outward top normal opposite the actual terminal
        // needle tangent. The sharp endpoint starts at a strictly positive
        // physical clearance; its closing velocity, accepted impulse and an
        // inward tract/tetrahedron intersection must all admit mutation on
        // Metal. No invalid initial penetration is used to force the event.
        const Vec3 basisZ = tip.approachDirection * -1.0;
        Vec3 basisX = rotate(orientation, {0.0, 0.0, 1.0});
        basisX = basisX - basisZ * dot(basisX, basisZ);
        const double basisXLength = norm(basisX);
        require(
            basisXLength > 1.0e-6,
            "needle binormal cannot orient the puncture coupon"
        );
        basisX = basisX * (1.0 / basisXLength);
        Vec3 basisY = cross(basisZ, basisX);
        const double basisYLength = norm(basisY);
        require(
            basisYLength > 1.0e-6,
            "puncture coupon frame is degenerate"
        );
        basisY = basisY * (1.0 / basisYLength);
        targetContactPoint = endpointA + tip.approachDirection *
            (needleRadiusM + initialSurfaceOffsetM);
        for (auto& position : coupon.object.femNodes) {
            const Vec3 relative = vector(position) - authoredAnchor;
            const Vec3 transformed = targetContactPoint +
                basisX * relative.x +
                basisY * relative.y +
                basisZ * relative.z;
            position = {transformed.x, transformed.y, transformed.z};
        }
        coupon.metadata.longitudinalAxis = {
            basisX.x, basisX.y, basisX.z,
        };
        coupon.metadata.circumferentialAxis = {
            basisY.x, basisY.y, basisY.z,
        };
        coupon.metadata.thicknessAxis = {
            basisZ.x, basisZ.y, basisZ.z,
        };
    } else {
        const Vec3 downward{0.0, 0.0, -1.0};
        Vec3 contactDirection =
            downward - capsuleAxis * dot(downward, capsuleAxis);
        const double contactDirectionLength = norm(contactDirection);
        require(
            contactDirectionLength > 1.0e-6,
            "tissue coupling capsule is parallel to the table normal"
        );
        contactDirection =
            contactDirection * (1.0 / contactDirectionLength);
        // A static preload begins just inside the 100 um IPC activation band.
        // The barrier separates the live needle and free tissue surface.
        targetContactPoint = contactCenter + contactDirection *
            (needleRadiusM + initialSurfaceOffsetM);
        const Vec3 translation = targetContactPoint - authoredAnchor;
        for (auto& position : coupon.object.femNodes) {
            position[0] += translation.x;
            position[1] += translation.y;
            position[2] += translation.z;
        }
    }
    double minimumAuthoredSeparation =
        std::numeric_limits<double>::infinity();
    for (const std::uint32_t node : coupon.object.femContactNodes) {
        require(
            node < coupon.object.femNodes.size(),
            "tissue contact surface contains an invalid node"
        );
        minimumAuthoredSeparation = std::min(
            minimumAuthoredSeparation,
            pointSegmentDistance(
                vector(coupon.object.femNodes[node]),
                endpointA,
                endpointB
            ) - needleRadiusM
        );
    }
    if (punctureTip) {
        require(
            minimumAuthoredSeparation > 0.0 &&
                std::abs(
                    minimumAuthoredSeparation - initialSurfaceOffsetM
                ) < 1.0e-7,
            "authored tapered tip does not have the requested positive "
            "entry clearance"
        );
    } else {
        require(
            minimumAuthoredSeparation > 0.0 &&
                std::abs(
                    minimumAuthoredSeparation - initialSurfaceOffsetM
                ) < 1.0e-7,
            "authored tissue boundary does not match the requested IPC gap: " +
                std::to_string(minimumAuthoredSeparation)
        );
    }
    std::cout << std::setprecision(9)
        << "tissue_contact_region="
        << (punctureTip ? "tapered_tip" : "swage")
        << " tissue_authored_minimum_separation_m="
        << minimumAuthoredSeparation
        << " tissue_surface_offset_m=" << initialSurfaceOffsetM
        << " tissue_anchor_node=" << anchorNode
        << " tissue_authored_capsule_first="
        << vectorSummary(endpointA)
        << " tissue_authored_capsule_second="
        << vectorSummary(endpointB) << '\n';

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = kControlTimestep / kPhysicsSubsteps;
    source.gravity = {0.0, 0.0, 0.0};
    source.contactSlop = 1.0e-4;
    source.maximumDepenetrationSpeed = 0.05;
    source.deterministic = true;
    // The contact-active coupon reaches the identical accepted state by the
    // fifth encoded Newton pass; later static tail passes do not alter it.
    source.mixedSolver.newtonIterations = 5u;
    source.mixedSolver.fgmresRestart = 10u;
    // This medically scaled coupon converges within seven Arnoldi columns on
    // the live Metal path. Keep a ten-column cycle (43% measured headroom)
    // without statically encoding empty restart cycles into every 62.5 us
    // surgical microstep.
    source.mixedSolver.fgmresIterations = 10u;
    source.mixedSolver.lineSearchSteps = 12u;
    source.mixedSolver.relativeResidual = 5.0e-4;
    source.mixedSolver.volumeTolerance = 5.0e-4;
    source.mixedSolver.pressureTolerance = 5.0e-4;
    source.mixedSolver.minimumContactSeparationRatio = 0.05;
    source.materials.push_back(std::move(parsed.material));

    // Matter borrows the real needle body. Contact-only regression uses its
    // swage-side binding capsule. Puncture always places the terminal tapered
    // capsule at proxy zero (sharp endpoint first under ABI v21), followed by
    // the requested adjacent authored arc segments. The through-wall probe
    // therefore resolves the widening taper behind the tip instead of passing
    // a zero-thickness point through tissue. Dynamic reactions enter the
    // global wrench arena before MetalWorld solves the hard swage and DER
    // strand; kinematic motion remains prescribed but still deforms tissue.
    const bool dynamicNeedle =
        needle.flagsAndIndices[0] == MR_MOTION_DYNAMIC;
    const auto appendNeedleProxy = [&source, &world, dynamicNeedle](
        const Vec3 first,
        const Vec3 second,
        const double radius,
        const bool sharpTip
    ) {
        numi::matter::RigidProxySource proxy;
        proxy.shape = NM_RIGID_CAPSULE;
        proxy.bodyIndex = world.sceneBodyIndices[0];
        proxy.sceneBodyIndex = dynamicNeedle ? 0u : NM_INVALID_INDEX;
        proxy.materialIndex = 0u;
        proxy.localCenter = {first.x, first.y, first.z};
        proxy.localExtent = {second.x, second.y, second.z};
        proxy.radiusOrOffset = radius;
        proxy.dynamic = dynamicNeedle;
        proxy.punctureTip = sharpTip;
        source.rigidProxies.push_back(proxy);
    };
    if (!punctureTip) {
        appendNeedleProxy(
            localEndpointA,
            localEndpointB,
            needleRadiusM,
            false
        );
    } else {
        require(
            punctureContactSegmentCount > 0u &&
                punctureContactSegmentCount <=
                    needleAsset.rigid.shapes.size(),
            "puncture contact segment count is outside the needle arc"
        );
        for (std::uint32_t localSegment = 0u;
             localSegment < punctureContactSegmentCount;
             ++localSegment) {
            const std::size_t shapeIndex =
                needleAsset.rigid.shapes.size() - 1u - localSegment;
            const MRShapeGPU& shape =
                needleAsset.rigid.shapes[shapeIndex];
            require(
                shape.shapeType == MR_SHAPE_CAPSULE &&
                    shape.dimensions.x > 0.0f &&
                    shape.dimensions.y > 0.0f,
                "puncture arc contains a non-capsule segment"
            );
            const Vec3 center = vector(shape.localPosition);
            const Vec3 axis = rotate(
                Quaternion{
                    shape.localRotation.x,
                    shape.localRotation.y,
                    shape.localRotation.z,
                    shape.localRotation.w,
                },
                {0.0, 1.0, 0.0}
            );
            // SurgicalAssets authors +Y along increasing arc length, so the
            // first endpoint is always the distal/tip-side endpoint.
            appendNeedleProxy(
                center + axis * shape.dimensions.y,
                center - axis * shape.dimensions.y,
                shape.dimensions.x,
                localSegment == 0u
            );
        }
    }
    source.objects.push_back(coupon.object);

    numi::matter::CompileOptions compileOptions;
    // The free needle is integrated by MetalWorld once per physics substep.
    // Keep Matter's tissue/contact transaction on that same cadence: encoding
    // several Matter microticks against one frozen source body would publish
    // the same free-body velocity cancellation more than once into the shared
    // wrench arena and is not a temporally coupled solve.
    compileOptions.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, compileOptions);
    require(
        compiled.succeeded(),
        "needle-suture tissue world compile failed: " +
            matterCompileErrors(compiled.diagnostics)
    );
    std::string layoutError;
    require(
        numi::matter::validateCompiledWorldLayout(
            compiled.world,
            &layoutError
        ) && compiled.world.dispatch.contactPairCount != 0u,
        "needle-suture tissue layout is invalid: " + layoutError
    );
    const std::uint32_t unifiedAnchor =
        compiled.world.dispatch.gridNodeCount + anchorNode;
    std::uint32_t anchorPairCount = 0u;
    for (const NMContactPairGPU& pair : compiled.world.contact.pairs) {
        if (pair.continuumNode == unifiedAnchor) {
            ++anchorPairCount;
        }
    }
    require(
        anchorNode < compiled.world.fem.nodes.size() &&
            anchorPairCount > 0u,
        "tissue contact anchor was not retained by the compiled pair topology"
    );
    std::cout << "tissue_contact_anchor_node=" << anchorNode
        << " fixed="
        << compiled.world.fem.nodes[anchorNode].restAndFixed.w
        << " compiled_pairs=" << anchorPairCount
        << " needle_contact_segments="
        << compiled.world.contact.rigidProxies.size()
        << " proxy_flags="
        << compiled.world.contact.rigidProxies.at(0u).flags << '\n';
    return std::move(compiled.world);
}

struct Arguments {
    std::string mode;
    std::filesystem::path stateOutputDirectory;
    std::filesystem::path resumeApproachHeldPath;
    std::filesystem::path resumeGiverClosedPath;
    std::filesystem::path resumeGiverLiftPath;
    std::filesystem::path resumeGiverHandoffStagePath;
    std::filesystem::path resumeGiverHandoffStagePrefixPath;
    std::filesystem::path resumeReceiverApproachMotionPath;
    std::filesystem::path resumeReceiverApproachPath;
    std::filesystem::path resumeReceiverAlignedPath;
    std::filesystem::path resumePositiveControlMotionPath;
    std::filesystem::path giverLiftReferencePath;
    std::filesystem::path giverGraspReferencePath;
    std::filesystem::path resumePositiveControlOverlapPath;
    std::filesystem::path resumeGiverReleaseMotionPath;
    std::filesystem::path resumeGiverReleasePath;
    std::filesystem::path resumeReceiverTransferMotionPath;
    std::filesystem::path receiverGraspReferencePath;
    std::filesystem::path receiverTransferStartReferencePath;
    std::uint32_t resumeStagingCompletedSteps = 0u;
    bool resumeStagingCompletedStepsProvided = false;
    std::uint32_t resumeLiftStepLimit = kHandoffLiftSteps;
    std::uint32_t liftDiagnosticChunkSteps = 0u;
    std::uint32_t settleStepLimit = 0u;
    std::uint32_t rodSolverIterations = 2048u;
    // The 16/8 sweep budget retained the jaw witnesses during the opening
    // lift but left a 9.4 mm/s final-sweep residual by step 100. A controlled
    // 32/16 replay reduced it to 6.4 mm/s without changing contact geometry,
    // friction, preload, substeps, or any acceptance tolerance.
    std::uint32_t velocityIterations = 32u;
    std::uint32_t finalVelocityIterations = 16u;
    bool resumeLiftStepLimitProvided = false;
    bool rodSolverIterationsProvided = false;
    bool velocityIterationsProvided = false;
    bool finalVelocityIterationsProvided = false;
    bool receiverCollisionScan = false;
    bool receiverAlignmentScan = false;
    double receiverToolRollOffset = kReceiverToolRollOffset;
    double receiverWristYaw = kReceiverWristYaw;
    double receiverBaseAzimuthOffset = kReceiverBaseAzimuthOffset;
    double receiverNeedleAxisRoll = kReceiverNeedleAxisRoll;
    double receiverScanHandoffHeightIncrement = 0.0;
    double receiverScanHandoffOffsetX = 0.0;
    double receiverScanHandoffOffsetY = 0.0;
    bool receiverToolRollOffsetProvided = false;
    bool receiverWristYawProvided = false;
    bool receiverBaseAzimuthOffsetProvided = false;
    bool receiverNeedleAxisRollProvided = false;
    bool receiverScanHandoffHeightIncrementProvided = false;
    bool receiverScanHandoffOffsetXProvided = false;
    bool receiverScanHandoffOffsetYProvided = false;
};

Arguments parseArguments(const int argc, const char* const argv[]) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--geometry-only" ||
            argument == "--settle-only" ||
            argument == "--tissue-coupling-only" ||
            argument == "--tissue-rest-only" ||
            argument == "--tissue-puncture-only" ||
            argument == "--tissue-puncture-advance-only" ||
            argument == "--tissue-curved-passage-only" ||
            argument == "--long-settle" ||
            argument == "--approach-only" ||
            argument == "--closure-only" ||
            argument == "--receiver-approach-only" ||
            argument == "--receiver-alignment-only" ||
            argument == "--receiver-closure-only" ||
            argument == "--load-exchange-only" ||
            argument == "--receiver-transfer-motion-only" ||
            argument == "--stage-only") {
            require(result.mode.empty(), "only one diagnostic mode is allowed");
            result.mode = argument;
        } else if (argument == "--state-output-dir") {
            require(
                result.stateOutputDirectory.empty() && index + 1 < argc,
                "--state-output-dir requires exactly one path"
            );
            result.stateOutputDirectory = argv[++index];
        } else if (argument == "--receiver-collision-scan") {
            require(
                !result.receiverCollisionScan,
                "--receiver-collision-scan may be provided only once"
            );
            result.receiverCollisionScan = true;
        } else if (argument == "--receiver-alignment-scan") {
            require(
                !result.receiverAlignmentScan,
                "--receiver-alignment-scan may be provided only once"
            );
            result.receiverAlignmentScan = true;
        } else if (argument == "--receiver-tool-roll-offset") {
            require(
                !result.receiverToolRollOffsetProvided &&
                    index + 1 < argc,
                "--receiver-tool-roll-offset requires one angle"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            result.receiverToolRollOffset = std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(result.receiverToolRollOffset) &&
                    std::abs(result.receiverToolRollOffset) <= 2.0,
                "receiver tool-roll offset must be finite and within 2 rad"
            );
            result.receiverToolRollOffsetProvided = true;
        } else if (argument == "--receiver-wrist-yaw") {
            require(
                !result.receiverWristYawProvided && index + 1 < argc,
                "--receiver-wrist-yaw requires one angle"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            result.receiverWristYaw = std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(result.receiverWristYaw) &&
                    std::abs(result.receiverWristYaw) <= 1.35,
                "receiver wrist yaw must be finite and within 1.35 rad"
            );
            result.receiverWristYawProvided = true;
        } else if (argument == "--receiver-base-azimuth-offset") {
            require(
                !result.receiverBaseAzimuthOffsetProvided &&
                    index + 1 < argc,
                "--receiver-base-azimuth-offset requires one angle"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            result.receiverBaseAzimuthOffset = std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(result.receiverBaseAzimuthOffset) &&
                    std::abs(result.receiverBaseAzimuthOffset) <=
                        std::numbers::pi,
                "receiver base azimuth must be finite and within pi rad"
            );
            result.receiverBaseAzimuthOffsetProvided = true;
        } else if (argument == "--receiver-needle-axis-roll") {
            require(
                !result.receiverNeedleAxisRollProvided && index + 1 < argc,
                "--receiver-needle-axis-roll requires one angle"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            result.receiverNeedleAxisRoll = std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(result.receiverNeedleAxisRoll) &&
                    std::abs(result.receiverNeedleAxisRoll) <=
                        std::numbers::pi,
                "receiver needle-axis roll must be finite and within pi"
            );
            result.receiverNeedleAxisRollProvided = true;
        } else if (argument == "--receiver-scan-handoff-height-increment") {
            require(
                !result.receiverScanHandoffHeightIncrementProvided &&
                    index + 1 < argc,
                "--receiver-scan-handoff-height-increment requires one "
                "distance"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            result.receiverScanHandoffHeightIncrement =
                std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(
                        result.receiverScanHandoffHeightIncrement
                    ) &&
                    result.receiverScanHandoffHeightIncrement >= 0.0 &&
                    result.receiverScanHandoffHeightIncrement <= 0.04,
                "receiver scan height increment must be in [0, 0.04] m"
            );
            result.receiverScanHandoffHeightIncrementProvided = true;
        } else if (argument == "--receiver-scan-handoff-offset-x" ||
                   argument == "--receiver-scan-handoff-offset-y") {
            const bool x =
                argument == "--receiver-scan-handoff-offset-x";
            bool& provided = x
                ? result.receiverScanHandoffOffsetXProvided
                : result.receiverScanHandoffOffsetYProvided;
            double& destination = x
                ? result.receiverScanHandoffOffsetX
                : result.receiverScanHandoffOffsetY;
            require(
                !provided && index + 1 < argc,
                std::string{argument} + " requires one distance"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            destination = std::stod(value, &consumed);
            require(
                consumed == value.size() &&
                    std::isfinite(destination) &&
                    std::abs(destination) <= 0.15,
                std::string{argument} + " must be within 0.15 m"
            );
            provided = true;
        } else if (argument == "--resume-giver-closed") {
            require(
                result.resumeGiverClosedPath.empty() &&
                    index + 1 < argc,
                "--resume-giver-closed requires exactly one path"
            );
            result.resumeGiverClosedPath = argv[++index];
        } else if (argument == "--resume-giver-lift") {
            require(
                result.resumeGiverLiftPath.empty() && index + 1 < argc,
                "--resume-giver-lift requires exactly one path"
            );
            result.resumeGiverLiftPath = argv[++index];
        } else if (argument == "--resume-giver-handoff-stage") {
            require(
                result.resumeGiverHandoffStagePath.empty() &&
                    index + 1 < argc,
                "--resume-giver-handoff-stage requires exactly one path"
            );
            result.resumeGiverHandoffStagePath = argv[++index];
        } else if (argument == "--resume-giver-handoff-stage-prefix") {
            require(
                result.resumeGiverHandoffStagePrefixPath.empty() &&
                    index + 1 < argc,
                "--resume-giver-handoff-stage-prefix requires one path"
            );
            result.resumeGiverHandoffStagePrefixPath = argv[++index];
        } else if (argument == "--resume-receiver-approach-motion") {
            require(
                result.resumeReceiverApproachMotionPath.empty() &&
                    index + 1 < argc,
                "--resume-receiver-approach-motion requires one path"
            );
            result.resumeReceiverApproachMotionPath = argv[++index];
        } else if (argument == "--resume-receiver-approach") {
            require(
                result.resumeReceiverApproachPath.empty() &&
                    index + 1 < argc,
                "--resume-receiver-approach requires one path"
            );
            result.resumeReceiverApproachPath = argv[++index];
        } else if (argument == "--resume-receiver-aligned") {
            require(
                result.resumeReceiverAlignedPath.empty() &&
                    index + 1 < argc,
                "--resume-receiver-aligned requires one path"
            );
            result.resumeReceiverAlignedPath = argv[++index];
        } else if (argument == "--resume-positive-control-motion") {
            require(
                result.resumePositiveControlMotionPath.empty() &&
                    index + 1 < argc,
                "--resume-positive-control-motion requires one path"
            );
            result.resumePositiveControlMotionPath = argv[++index];
        } else if (argument == "--giver-lift-reference") {
            require(
                result.giverLiftReferencePath.empty() &&
                    index + 1 < argc,
                "--giver-lift-reference requires one path"
            );
            result.giverLiftReferencePath = argv[++index];
        } else if (argument == "--giver-grasp-reference") {
            require(
                result.giverGraspReferencePath.empty() && index + 1 < argc,
                "--giver-grasp-reference requires one path"
            );
            result.giverGraspReferencePath = argv[++index];
        } else if (argument == "--receiver-grasp-reference") {
            require(
                result.receiverGraspReferencePath.empty() &&
                    index + 1 < argc,
                "--receiver-grasp-reference requires one path"
            );
            result.receiverGraspReferencePath = argv[++index];
        } else if (argument == "--receiver-transfer-start-reference") {
            require(
                result.receiverTransferStartReferencePath.empty() &&
                    index + 1 < argc,
                "--receiver-transfer-start-reference requires one path"
            );
            result.receiverTransferStartReferencePath = argv[++index];
        } else if (argument == "--resume-staging-completed-steps") {
            require(
                !result.resumeStagingCompletedStepsProvided &&
                    index + 1 < argc,
                "--resume-staging-completed-steps requires one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed > 0u &&
                    parsed < kHandoffStagingSteps,
                "completed staging steps must be inside the trajectory"
            );
            result.resumeStagingCompletedSteps =
                static_cast<std::uint32_t>(parsed);
            result.resumeStagingCompletedStepsProvided = true;
        } else if (argument == "--resume-positive-control-overlap") {
            require(
                result.resumePositiveControlOverlapPath.empty() &&
                    index + 1 < argc,
                "--resume-positive-control-overlap requires exactly one path"
            );
            result.resumePositiveControlOverlapPath = argv[++index];
        } else if (argument == "--resume-giver-release-motion") {
            require(
                result.resumeGiverReleaseMotionPath.empty() &&
                    index + 1 < argc,
                "--resume-giver-release-motion requires exactly one path"
            );
            result.resumeGiverReleaseMotionPath = argv[++index];
        } else if (argument == "--resume-giver-release") {
            require(
                result.resumeGiverReleasePath.empty() && index + 1 < argc,
                "--resume-giver-release requires exactly one path"
            );
            result.resumeGiverReleasePath = argv[++index];
        } else if (argument == "--resume-receiver-transfer-motion") {
            require(
                result.resumeReceiverTransferMotionPath.empty() &&
                    index + 1 < argc,
                "--resume-receiver-transfer-motion requires one path"
            );
            result.resumeReceiverTransferMotionPath = argv[++index];
        } else if (argument == "--resume-approach-held") {
            require(
                result.resumeApproachHeldPath.empty() &&
                    index + 1 < argc,
                "--resume-approach-held requires exactly one path"
            );
            result.resumeApproachHeldPath = argv[++index];
        } else if (argument == "--resume-lift-step-limit") {
            require(
                !result.resumeLiftStepLimitProvided &&
                    index + 1 < argc,
                "--resume-lift-step-limit requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed > 0u &&
                    parsed <= kHandoffLiftSteps,
                "resume lift step limit exceeds the deliberate lift"
            );
            result.resumeLiftStepLimit =
                static_cast<std::uint32_t>(parsed);
            result.resumeLiftStepLimitProvided = true;
        } else if (argument == "--lift-diagnostic-chunk-steps") {
            require(
                result.liftDiagnosticChunkSteps == 0u &&
                    index + 1 < argc,
                "--lift-diagnostic-chunk-steps requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed > 0u &&
                    parsed <= kHandoffLiftSteps,
                "lift diagnostic chunk must fit within the deliberate lift"
            );
            result.liftDiagnosticChunkSteps =
                static_cast<std::uint32_t>(parsed);
        } else if (argument == "--settle-step-limit") {
            require(
                result.settleStepLimit == 0u && index + 1 < argc,
                "--settle-step-limit requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed > 0u && parsed <= 200u,
                "settle step limit must be in [1, 200]"
            );
            result.settleStepLimit =
                static_cast<std::uint32_t>(parsed);
        } else if (argument == "--rod-solver-iterations") {
            require(
                !result.rodSolverIterationsProvided && index + 1 < argc,
                "--rod-solver-iterations requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed >= 512u &&
                    parsed <= 2048u,
                "rod solver iterations must be in [512, 2048]"
            );
            result.rodSolverIterations =
                static_cast<std::uint32_t>(parsed);
            result.rodSolverIterationsProvided = true;
        } else if (argument == "--velocity-iterations") {
            require(
                !result.velocityIterationsProvided && index + 1 < argc,
                "--velocity-iterations requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed > 0u && parsed <= 128u,
                "velocity iterations must be in [1, 128]"
            );
            result.velocityIterations =
                static_cast<std::uint32_t>(parsed);
            result.velocityIterationsProvided = true;
        } else if (argument == "--final-velocity-iterations") {
            require(
                !result.finalVelocityIterationsProvided && index + 1 < argc,
                "--final-velocity-iterations requires exactly one count"
            );
            const std::string value{argv[++index]};
            std::size_t consumed = 0u;
            const unsigned long parsed = std::stoul(value, &consumed);
            require(
                consumed == value.size() && parsed <= 128u,
                "final velocity iterations must be in [0, 128]"
            );
            result.finalVelocityIterations =
                static_cast<std::uint32_t>(parsed);
            result.finalVelocityIterationsProvided = true;
        } else {
            throw std::invalid_argument(
                "unknown dual-PSM handoff argument: " +
                std::string{argument}
            );
        }
    }
    const std::uint32_t resumeCount =
        (!result.resumeApproachHeldPath.empty() ? 1u : 0u) +
        (!result.resumeGiverClosedPath.empty() ? 1u : 0u) +
        (!result.resumeGiverLiftPath.empty() ? 1u : 0u) +
        (!result.resumeGiverHandoffStagePath.empty() ? 1u : 0u) +
        (!result.resumeGiverHandoffStagePrefixPath.empty() ? 1u : 0u) +
        (!result.resumeReceiverApproachMotionPath.empty() ? 1u : 0u) +
        (!result.resumeReceiverApproachPath.empty() ? 1u : 0u) +
        (!result.resumeReceiverAlignedPath.empty() ? 1u : 0u) +
        (!result.resumePositiveControlMotionPath.empty() ? 1u : 0u) +
        (!result.resumePositiveControlOverlapPath.empty() ? 1u : 0u) +
        (!result.resumeGiverReleaseMotionPath.empty() ? 1u : 0u) +
        (!result.resumeGiverReleasePath.empty() ? 1u : 0u) +
        (!result.resumeReceiverTransferMotionPath.empty() ? 1u : 0u);
    require(resumeCount <= 1u, "handoff resumes are mutually exclusive");
    require(
        !result.resumeLiftStepLimitProvided ||
            !result.resumeGiverClosedPath.empty(),
        "resume lift step limit requires --resume-giver-closed"
    );
    require(
        result.liftDiagnosticChunkSteps == 0u ||
            !result.resumeGiverClosedPath.empty(),
        "lift diagnostic chunking requires --resume-giver-closed"
    );
    require(
        result.settleStepLimit == 0u ||
            (resumeCount == 0u && result.mode != "--geometry-only") ||
            !result.resumePositiveControlMotionPath.empty() ||
            !result.resumePositiveControlOverlapPath.empty() ||
            !result.resumeGiverReleaseMotionPath.empty() ||
            !result.resumeReceiverTransferMotionPath.empty(),
        "settle step limit requires a fresh state, positive-control "
        "resume, or terminal hold diagnostic"
    );
    require(
        result.resumeApproachHeldPath.empty() ||
            result.mode == "--closure-only",
        "approach-held resume requires --closure-only"
    );
    require(
        result.resumeGiverLiftPath.empty() || result.mode.empty() ||
            result.mode == "--stage-only",
        "giver-lift resume only supports the stage-only diagnostic mode"
    );
    require(
        result.resumeGiverHandoffStagePath.empty() || result.mode.empty() ||
            result.mode == "--receiver-approach-only",
        "giver handoff-stage resume only supports receiver-approach-only"
    );
    require(
        result.resumeGiverHandoffStagePrefixPath.empty() ||
            result.mode.empty() || result.mode == "--stage-only",
        "giver stage-prefix resume only supports stage-only diagnostic mode"
    );
    require(
        result.resumeGiverHandoffStagePrefixPath.empty() ==
                !result.resumeStagingCompletedStepsProvided &&
            result.resumeGiverHandoffStagePrefixPath.empty() ==
                result.giverLiftReferencePath.empty(),
        "stage-prefix resume requires its completed-step count and the "
        "original giver-lift reference"
    );
    require(
        result.resumeReceiverApproachMotionPath.empty() ||
            result.mode.empty() ||
            result.mode == "--receiver-approach-only",
        "receiver approach-motion resume only supports "
        "receiver-approach-only"
    );
    require(
        result.giverGraspReferencePath.empty() ||
            !result.resumeReceiverApproachMotionPath.empty() ||
            !result.resumePositiveControlMotionPath.empty() ||
            !result.resumePositiveControlOverlapPath.empty(),
        "giver grasp reference requires receiver approach-motion or "
        "positive-control resume"
    );
    require(
        result.resumeReceiverApproachPath.empty() || result.mode.empty() ||
            result.mode == "--receiver-alignment-only" ||
            result.mode == "--receiver-closure-only",
        "receiver approach resume only supports receiver alignment or "
        "closure diagnostics"
    );
    require(
        result.resumeReceiverAlignedPath.empty() || result.mode.empty() ||
            result.mode == "--receiver-closure-only",
        "receiver-aligned resume only supports receiver closure"
    );
    require(
        result.resumePositiveControlMotionPath.empty() ||
            (result.mode == "--receiver-closure-only" &&
             !result.giverGraspReferencePath.empty()),
        "positive-control motion resume requires receiver-closure-only and "
        "the original giver reference"
    );
    require(
        !result.receiverCollisionScan ||
            !result.resumeGiverLiftPath.empty() ||
            !result.resumeGiverHandoffStagePath.empty(),
        "receiver collision scan requires a giver-lift or staged-handoff "
        "checkpoint"
    );
    require(
        !result.receiverAlignmentScan ||
            (!result.resumeReceiverApproachPath.empty() &&
             !result.receiverCollisionScan && result.mode.empty()),
        "receiver alignment scan requires only a receiver-approach "
        "checkpoint"
    );
    require(
        (!result.receiverToolRollOffsetProvided &&
         !result.receiverWristYawProvided &&
         !result.receiverBaseAzimuthOffsetProvided &&
         !result.receiverNeedleAxisRollProvided &&
         !result.receiverScanHandoffHeightIncrementProvided &&
         !result.receiverScanHandoffOffsetXProvided &&
         !result.receiverScanHandoffOffsetYProvided) ||
            result.receiverCollisionScan,
        "receiver posture overrides are diagnostic-scan-only"
    );
    require(
        result.resumePositiveControlOverlapPath.empty() ||
            result.mode.empty() || result.mode == "--load-exchange-only",
        "positive-control resume only supports load-exchange-only"
    );
    const bool terminalResume =
        !result.resumeGiverReleaseMotionPath.empty() ||
        !result.resumeGiverReleasePath.empty() ||
        !result.resumeReceiverTransferMotionPath.empty();
    require(
        terminalResume == !result.receiverGraspReferencePath.empty(),
        "terminal handoff resume requires exactly one load-exchange "
        "receiver grasp reference"
    );
    require(
        result.resumeGiverReleasePath.empty() || result.mode.empty() ||
            result.mode == "--receiver-transfer-motion-only",
        "giver-release resume only supports receiver-transfer-motion-only"
    );
    require(
        (result.resumeGiverReleaseMotionPath.empty() &&
         result.resumeReceiverTransferMotionPath.empty()) ||
            result.mode.empty(),
        "terminal hold resume cannot be combined with a diagnostic mode"
    );
    require(
        result.mode != "--receiver-transfer-motion-only" ||
            !result.resumeGiverReleasePath.empty(),
        "receiver-transfer-motion-only requires --resume-giver-release"
    );
    require(
        result.resumeReceiverTransferMotionPath.empty() ==
            result.receiverTransferStartReferencePath.empty(),
        "receiver-transfer-motion resume requires exactly one giver-release "
        "transfer-start reference"
    );
    return result;
}

std::vector<std::string_view> splitTabs(const std::string& line) {
    std::vector<std::string_view> fields;
    std::size_t begin = 0u;
    while (begin <= line.size()) {
        const std::size_t end = line.find('\t', begin);
        fields.emplace_back(
            line.data() + begin,
            (end == std::string::npos ? line.size() : end) - begin
        );
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1u;
    }
    return fields;
}

double parseNumber(
    const std::string_view text,
    const std::string_view label
) {
    std::size_t consumed = 0u;
    const std::string owned{text};
    const double value = std::stod(owned, &consumed);
    require(
        consumed == owned.size() && std::isfinite(value),
        "handoff resume contains an invalid " + std::string{label}
    );
    return value;
}

std::uint32_t parseIndex(
    const std::string_view text,
    const std::string_view label
) {
    std::size_t consumed = 0u;
    const std::string owned{text};
    const unsigned long value = std::stoul(owned, &consumed);
    require(
        consumed == owned.size() &&
            value <= std::numeric_limits<std::uint32_t>::max(),
        "handoff resume contains an invalid " + std::string{label}
    );
    return static_cast<std::uint32_t>(value);
}

void updateNeedleInverseInertia(
    const MRBodyPropertiesGPU& properties,
    MRBodyStateGPU& state
) {
    const Quaternion orientation{
        state.orientation.x,
        state.orientation.y,
        state.orientation.z,
        state.orientation.w,
    };
    const std::array<Vec3, 3u> axes{{
        rotate(orientation, {1.0, 0.0, 0.0}),
        rotate(orientation, {0.0, 1.0, 0.0}),
        rotate(orientation, {0.0, 0.0, 1.0}),
    }};
    const auto component = [](const Vec3 value, const std::size_t axis) {
        return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
    };
    const double body[3][3] = {
        {properties.inverseInertiaRow0.x,
         properties.inverseInertiaRow0.y,
         properties.inverseInertiaRow0.z},
        {properties.inverseInertiaRow1.x,
         properties.inverseInertiaRow1.y,
         properties.inverseInertiaRow1.z},
        {properties.inverseInertiaRow2.x,
         properties.inverseInertiaRow2.y,
         properties.inverseInertiaRow2.z},
    };
    double world[3][3]{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            const double rowAxis[3] = {
                component(axes[0], row),
                component(axes[1], row),
                component(axes[2], row),
            };
            const double columnAxis[3] = {
                component(axes[0], column),
                component(axes[1], column),
                component(axes[2], column),
            };
            for (std::size_t innerRow = 0u;
                 innerRow < 3u;
                 ++innerRow) {
                for (std::size_t innerColumn = 0u;
                     innerColumn < 3u;
                     ++innerColumn) {
                    world[row][column] +=
                        rowAxis[innerRow] * body[innerRow][innerColumn] *
                        columnAxis[innerColumn];
                }
            }
        }
    }
    state.inverseInertiaWorldRow0 = {
        static_cast<float>(world[0][0]),
        static_cast<float>(world[0][1]),
        static_cast<float>(world[0][2]),
        0.0f,
    };
    state.inverseInertiaWorldRow1 = {
        static_cast<float>(world[1][0]),
        static_cast<float>(world[1][1]),
        static_cast<float>(world[1][2]),
        0.0f,
    };
    state.inverseInertiaWorldRow2 = {
        static_cast<float>(world[2][0]),
        static_cast<float>(world[2][1]),
        static_cast<float>(world[2][2]),
        0.0f,
    };
}

std::uint64_t loadHandoffState(
    const std::filesystem::path& path,
    const std::string_view expectedPhase,
    metalrobo::HeterogeneousWorld& world
) {
    std::ifstream input(path);
    require(input.good(), "could not open handoff resume state");
    std::vector<float> q;
    std::vector<float> v;
    MRBodyStateGPU needle = world.defaultSceneBodies.at(0u);
    std::vector<MRRodNodeStateGPU> nodes(
        world.rods.at(0u).model.restPositions.size()
    );
    std::vector<MRRodEdgeStateGPU> edges(
        world.rods.at(0u).model.restTwists.size()
    );
    std::vector<bool> havePosition(nodes.size(), false);
    std::vector<bool> haveVelocity(nodes.size(), false);
    std::vector<bool> haveEdge(edges.size(), false);
    bool schema = false;
    bool phase = false;
    bool needlePosition = false;
    bool needleOrientation = false;
    bool needleLinearVelocity = false;
    bool needleAngularVelocity = false;
    bool haveStep = false;
    std::uint64_t stateStep = 0u;
    std::string line;
    while (std::getline(input, line)) {
        const std::vector<std::string_view> fields = splitTabs(line);
        if (fields.empty()) {
            continue;
        }
        if (fields[0] == "schema") {
            schema = fields.size() == 2u &&
                fields[1] == "numi.dual-psm-suture-handoff-state.v2";
        } else if (fields[0] == "phase") {
            phase = fields.size() == 2u && fields[1] == expectedPhase;
        } else if (fields[0] == "step") {
            require(fields.size() == 2u, "invalid resume step");
            stateStep = parseIndex(fields[1], "step");
            haveStep = true;
        } else if (fields[0] == "q" || fields[0] == "v") {
            std::vector<float>& destination =
                fields[0] == "q" ? q : v;
            destination.reserve(fields.size() - 1u);
            for (std::size_t index = 1u; index < fields.size(); ++index) {
                destination.push_back(static_cast<float>(parseNumber(
                    fields[index], fields[0]
                )));
            }
        } else if (fields[0] == "needle_position") {
            require(fields.size() == 4u, "invalid resume needle position");
            needle.position = {
                static_cast<float>(parseNumber(fields[1], "needle x")),
                static_cast<float>(parseNumber(fields[2], "needle y")),
                static_cast<float>(parseNumber(fields[3], "needle z")),
                1.0f,
            };
            needlePosition = true;
        } else if (fields[0] == "needle_orientation_xyzw") {
            require(fields.size() == 5u, "invalid resume needle orientation");
            needle.orientation = {
                static_cast<float>(parseNumber(fields[1], "needle qx")),
                static_cast<float>(parseNumber(fields[2], "needle qy")),
                static_cast<float>(parseNumber(fields[3], "needle qz")),
                static_cast<float>(parseNumber(fields[4], "needle qw")),
            };
            needleOrientation = true;
        } else if (fields[0] == "needle_linear_velocity") {
            require(fields.size() == 4u, "invalid resume needle velocity");
            needle.linearVelocityAndInverseMass.x =
                static_cast<float>(parseNumber(fields[1], "needle vx"));
            needle.linearVelocityAndInverseMass.y =
                static_cast<float>(parseNumber(fields[2], "needle vy"));
            needle.linearVelocityAndInverseMass.z =
                static_cast<float>(parseNumber(fields[3], "needle vz"));
            needleLinearVelocity = true;
        } else if (fields[0] == "needle_angular_velocity") {
            require(fields.size() == 4u, "invalid resume needle angular velocity");
            needle.angularVelocity = {
                static_cast<float>(parseNumber(fields[1], "needle wx")),
                static_cast<float>(parseNumber(fields[2], "needle wy")),
                static_cast<float>(parseNumber(fields[3], "needle wz")),
                0.0f,
            };
            needleAngularVelocity = true;
        } else if (fields[0] == "thread_position" ||
                   fields[0] == "thread_velocity") {
            require(fields.size() == 5u, "invalid resume thread node");
            const std::uint32_t index = parseIndex(fields[1], "thread node");
            require(index < nodes.size(), "resume thread node is out of range");
            mr_float4& destination = fields[0] == "thread_position"
                ? nodes[index].position
                : nodes[index].velocity;
            destination = {
                static_cast<float>(parseNumber(fields[2], "thread x")),
                static_cast<float>(parseNumber(fields[3], "thread y")),
                static_cast<float>(parseNumber(fields[4], "thread z")),
                fields[0] == "thread_position" ? 1.0f : 0.0f,
            };
            if (fields[0] == "thread_position") {
                havePosition[index] = true;
            } else {
                haveVelocity[index] = true;
            }
        } else if (fields[0] == "thread_twist") {
            require(fields.size() == 4u, "invalid resume thread twist");
            const std::uint32_t index = parseIndex(fields[1], "thread edge");
            require(index < edges.size(), "resume thread edge is out of range");
            edges[index].twistAndRate = {
                static_cast<float>(parseNumber(fields[2], "thread twist")),
                static_cast<float>(parseNumber(fields[3], "thread twist rate")),
                0.0f,
                0.0f,
            };
            haveEdge[index] = true;
        }
    }
    require(
        schema && phase && haveStep &&
            q.size() == world.model.world.nq &&
            v.size() == world.model.world.nv &&
            needlePosition && needleOrientation &&
            needleLinearVelocity && needleAngularVelocity &&
            std::ranges::all_of(havePosition, [](const bool value) {
                return value;
            }) &&
            std::ranges::all_of(haveVelocity, [](const bool value) {
                return value;
            }),
        "handoff resume state is incomplete or has the wrong phase"
    );
    if (!std::ranges::all_of(haveEdge, [](const bool value) {
            return value;
        })) {
        // v1 artifacts written before twist publication retain the authored
        // zero material frame. The accepted close state is nearly untwisted;
        // all newly written artifacts carry the exact edge state.
        std::ranges::fill(edges, MRRodEdgeStateGPU{});
    }
    const std::uint32_t needleBody = world.sceneBodyIndices.at(0u);
    updateNeedleInverseInertia(world.model.bodies.at(needleBody), needle);
    require(
        world.rods[0u].attachments.size() ==
            world.rods[0u].rigidBindings.size(),
        "handoff resume swage binding count changed"
    );
    for (std::size_t bindingIndex = 0u;
         bindingIndex < world.rods[0u].rigidBindings.size();
         ++bindingIndex) {
        const auto& localAnchor =
            world.rods[0u].rigidBindings[bindingIndex].localAnchor;
        const Vec3 anchorOffset = rotate(
            Quaternion{
                needle.orientation.x,
                needle.orientation.y,
                needle.orientation.z,
                needle.orientation.w,
            },
            {localAnchor[0], localAnchor[1], localAnchor[2]}
        );
        metalrobo::DiscreteRodAttachment& attachment =
            world.rods[0u].attachments[bindingIndex];
        attachment.targetPosition = {
            needle.position.x + anchorOffset.x,
            needle.position.y + anchorOffset.y,
            needle.position.z + anchorOffset.z,
        };
        const Vec3 anchorVelocity =
            vector(needle.linearVelocityAndInverseMass) +
            cross(vector(needle.angularVelocity), anchorOffset);
        attachment.targetVelocity = {
            anchorVelocity.x,
            anchorVelocity.y,
            anchorVelocity.z,
        };
    }
    world.model.defaultQ = std::move(q);
    world.model.defaultV = std::move(v);
    world.defaultSceneBodies[0u] = needle;
    auto& rodState = world.rods[0u].defaultState;
    for (std::size_t index = 0u; index < nodes.size(); ++index) {
        rodState.positions[index] = {
            nodes[index].position.x,
            nodes[index].position.y,
            nodes[index].position.z,
        };
        rodState.velocities[index] = {
            nodes[index].velocity.x,
            nodes[index].velocity.y,
            nodes[index].velocity.z,
        };
    }
    for (std::size_t index = 0u; index < edges.size(); ++index) {
        rodState.twists[index] = edges[index].twistAndRate.x;
        rodState.twistRates[index] = edges[index].twistAndRate.y;
    }
    return stateStep;
}

void writeHandoffStateArtifact(
    const std::filesystem::path& directory,
    const std::string_view phase,
    const std::uint64_t step,
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::BowelAnastomosisSutureSpec& suture,
    const metalrobo::MetalWorldResult& state
) {
    if (directory.empty()) {
        return;
    }
    require(
        state.finalQ.size() == world.model.world.nq &&
            state.finalV.size() == world.model.world.nv &&
        state.finalSceneBodies.size() >= 1u &&
            state.finalRodNodes.size() ==
                world.rods[0].model.restPositions.size() &&
            state.finalRodEdges.size() ==
                world.rods[0].model.restTwists.size(),
        "handoff visual state is incomplete"
    );
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    require(!error, "could not create handoff state output directory");
    const std::filesystem::path path =
        directory / (std::string{phase} + ".tsv");
    std::ofstream output(path, std::ios::trunc);
    require(output.good(), "could not open handoff state artifact");
    const MRBodyStateGPU& needle = state.finalSceneBodies[0];
    const numi::matter::PorcineJejunumFungSpec tissue;
    output << std::setprecision(17)
        << "schema\tnumi.dual-psm-suture-handoff-state.v2\n"
        << "phase\t" << phase << '\n'
        << "step\t" << step << '\n'
        << "control_timestep_s\t" << kControlTimestep << '\n'
        << "model\t" << world.model.name << '\n'
        << "q";
    for (const float value : state.finalQ) {
        output << '\t' << value;
    }
    output << "\nv";
    for (const float value : state.finalV) {
        output << '\t' << value;
    }
    output
        << "\nneedle_position\t" << needle.position.x << '\t'
        << needle.position.y << '\t' << needle.position.z << '\n'
        << "needle_orientation_xyzw\t" << needle.orientation.x << '\t'
        << needle.orientation.y << '\t' << needle.orientation.z << '\t'
        << needle.orientation.w << '\n'
        << "needle_linear_velocity\t"
        << needle.linearVelocityAndInverseMass.x << '\t'
        << needle.linearVelocityAndInverseMass.y << '\t'
        << needle.linearVelocityAndInverseMass.z << '\n'
        << "needle_angular_velocity\t" << needle.angularVelocity.x << '\t'
        << needle.angularVelocity.y << '\t' << needle.angularVelocity.z
        << '\n'
        << "thread_model\t" << world.rods[0].model.radius << '\t'
        << state.finalRodNodes.size() << '\t'
        << suture.threadLengthM.value << '\n'
        << "tissue_model\tporcine_jejunum_fung\t"
        << tissue.lengthM.value << '\t' << tissue.widthM.value << '\t'
        << tissue.thicknessM.value << '\t' << tissue.incisionGapM.value
        << '\n';
    for (std::size_t node = 0u; node < state.finalRodNodes.size(); ++node) {
        const MRRodNodeStateGPU& value = state.finalRodNodes[node];
        output << "thread_position\t" << node << '\t'
            << value.position.x << '\t' << value.position.y << '\t'
            << value.position.z << '\n'
            << "thread_velocity\t" << node << '\t'
            << value.velocity.x << '\t' << value.velocity.y << '\t'
            << value.velocity.z << '\n';
    }
    for (std::size_t edge = 0u; edge < state.finalRodEdges.size(); ++edge) {
        const MRRodEdgeStateGPU& value = state.finalRodEdges[edge];
        output << "thread_twist\t" << edge << '\t'
            << value.twistAndRate.x << '\t' << value.twistAndRate.y
            << '\n';
    }
    output.close();
    require(output.good(), "could not publish handoff state artifact");
}

PhaseResult initializePhase(
    metalrobo::MetalWorldContext& context,
    const metalrobo::CompiledWorld& compiled,
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldStepConfig& config,
    metalrobo::MetalWorldResidentState& resident,
    const std::vector<float>& efforts,
    const std::uint32_t steps
) {
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = steps,
        .initialQ = world.model.defaultQ,
        .initialV = world.model.defaultV,
        .efforts = efforts,
        .initialSceneBodies = world.defaultSceneBodies,
    };
    metalrobo::MetalWorldSubmission submission;
    const auto submitted = context.initializeResidentState(
        compiled,
        batch,
        config,
        resident,
        submission
    );
    require(
        submitted.succeeded() && submission.valid(),
        "initial handoff phase submission failed: " + submitted.message
    );
    PhaseResult phase;
    phase.diagnostics = submission.wait(phase.result);
    require(
        phase.diagnostics.succeeded() &&
            phase.diagnostics.failedStepCount == 0u,
        "initial handoff phase failed: " + phase.diagnostics.message
    );
    return phase;
}

PhaseResult initializePhaseUnchecked(
    metalrobo::MetalWorldContext& context,
    const metalrobo::CompiledWorld& compiled,
    const metalrobo::HeterogeneousWorld& world,
    const metalrobo::MetalWorldStepConfig& config,
    metalrobo::MetalWorldResidentState& resident,
    const std::vector<float>& efforts,
    const std::uint32_t steps
) {
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = steps,
        .initialQ = world.model.defaultQ,
        .initialV = world.model.defaultV,
        .efforts = efforts,
        .initialSceneBodies = world.defaultSceneBodies,
    };
    metalrobo::MetalWorldSubmission submission;
    const auto submitted = context.initializeResidentState(
        compiled,
        batch,
        config,
        resident,
        submission
    );
    require(
        submitted.succeeded() && submission.valid(),
        "initial diagnostic submission failed: " + submitted.message
    );
    PhaseResult phase;
    phase.diagnostics = submission.wait(phase.result);
    return phase;
}

PhaseResult continuePhase(
    metalrobo::MetalWorldContext& context,
    const metalrobo::CompiledWorld& compiled,
    const metalrobo::MetalWorldStepConfig& config,
    metalrobo::MetalWorldResidentState& resident,
    const std::vector<float>& efforts,
    const std::uint32_t steps,
    const std::string& name
) {
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = steps,
        .efforts = efforts,
    };
    metalrobo::MetalWorldSubmission submission;
    const auto submitted = context.submitResident(
        compiled,
        batch,
        config,
        resident,
        submission
    );
    require(
        submitted.succeeded() && submission.valid(),
        name + " submission failed: " + submitted.message
    );
    PhaseResult phase;
    phase.diagnostics = submission.wait(phase.result);
    require(
        phase.diagnostics.succeeded() &&
            phase.diagnostics.failedStepCount == 0u,
        name + " failed: " + phase.diagnostics.message
    );
    return phase;
}

PhaseResult continuePhaseWithKinematicTargets(
    metalrobo::MetalWorldContext& context,
    const metalrobo::CompiledWorld& compiled,
    const metalrobo::MetalWorldStepConfig& config,
    metalrobo::MetalWorldResidentState& resident,
    const std::vector<float>& efforts,
    const std::vector<MRBodyStateGPU>& kinematicTargets,
    const std::uint32_t steps,
    const std::string& name
) {
    require(
        kinematicTargets.size() ==
            static_cast<std::size_t>(steps) * compiled.sceneBodyCount(),
        name + " kinematic target stream has the wrong dimensions"
    );
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = steps,
        .efforts = efforts,
        .kinematicTargets = kinematicTargets,
    };
    metalrobo::MetalWorldSubmission submission;
    const auto submitted = context.submitResident(
        compiled,
        batch,
        config,
        resident,
        submission
    );
    require(
        submitted.succeeded() && submission.valid(),
        name + " submission failed: " + submitted.message
    );
    PhaseResult phase;
    phase.diagnostics = submission.wait(phase.result);
    require(
        phase.diagnostics.succeeded() &&
            phase.diagnostics.failedStepCount == 0u,
        name + " failed: " + phase.diagnostics.message
    );
    return phase;
}

PhaseResult continuePhaseUnchecked(
    metalrobo::MetalWorldContext& context,
    const metalrobo::CompiledWorld& compiled,
    const metalrobo::MetalWorldStepConfig& config,
    metalrobo::MetalWorldResidentState& resident,
    const std::vector<float>& efforts,
    const std::uint32_t steps,
    const std::string& name
) {
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = steps,
        .efforts = efforts,
    };
    metalrobo::MetalWorldSubmission submission;
    const auto submitted = context.submitResident(
        compiled,
        batch,
        config,
        resident,
        submission
    );
    require(
        submitted.succeeded() && submission.valid(),
        name + " submission failed: " + submitted.message
    );
    PhaseResult phase;
    phase.diagnostics = submission.wait(phase.result);
    return phase;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        const Arguments options = parseArguments(argc, argv);
        const bool geometryOnly = options.mode == "--geometry-only";
        const bool longSettle = options.mode == "--long-settle";
        const bool settleOnly = longSettle ||
            options.mode == "--settle-only";
        const bool tissueCouplingOnly =
            options.mode == "--tissue-coupling-only";
        const bool tissueRestOnly =
            options.mode == "--tissue-rest-only";
        const bool tissuePunctureOnly =
            options.mode == "--tissue-puncture-only" ||
            options.mode == "--tissue-puncture-advance-only" ||
            options.mode == "--tissue-curved-passage-only";
        const bool tissuePunctureAdvanceOnly =
            options.mode == "--tissue-puncture-advance-only";
        const bool tissueCurvedPassageOnly =
            options.mode == "--tissue-curved-passage-only";
        const bool tissueMatterOnly =
            tissueCouplingOnly || tissueRestOnly || tissuePunctureOnly;
        const bool approachOnly = options.mode == "--approach-only";
        const bool closureOnly = options.mode == "--closure-only";
        const bool receiverApproachOnly =
            options.mode == "--receiver-approach-only";
        const bool receiverAlignmentOnly =
            options.mode == "--receiver-alignment-only";
        const bool receiverClosureOnly =
            options.mode == "--receiver-closure-only";
        const bool loadExchangeOnly =
            options.mode == "--load-exchange-only";
        const bool stageOnly = options.mode == "--stage-only";
        const metalrobo::EngineModel psm =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        const auto sutureSpec =
            metalrobo::makeBowelAnastomosisSutureSpec();
        const auto needleForPlacement =
            metalrobo::makeCurvedSutureNeedleAsset(
                {},
                sutureSpec.needle
            );
        require(
            kGiverNeedleShape >=
                    needleForPlacement.metadata.graspShapeBegin &&
                kGiverNeedleShape <
                    needleForPlacement.metadata.graspShapeEnd &&
                kReceiverNeedleShape >=
                    needleForPlacement.metadata.graspShapeBegin &&
                kReceiverNeedleShape <
                    needleForPlacement.metadata.graspShapeEnd,
            "handoff grasps left the one-third-to-one-half handling zone"
        );
        const double graspDiametralClosure = grooveDiametralClosure(
            psm,
            sutureSpec.needle.crossSectionRadiusM.value
        );
        const double receiverOverlapDiametralClosure =
            grooveDiametralClosure(
                psm,
                sutureSpec.needle.crossSectionRadiusM.value,
                kReceiverOverlapRailRadialPreload
            );
        const double receiverTransportDiametralClosure =
            grooveDiametralClosure(
                psm,
                sutureSpec.needle.crossSectionRadiusM.value,
                kReceiverTransportRailRadialPreload
            );
        const auto& jawMetadata = metalrobo::surgicalPSMMetadata();
        const double estimatedPatchNormalForce =
            kGrooveRailRadialPreload /
            jawMetadata.insertSystemNormalComplianceMPerN;
        const double estimatedReceiverOverlapPatchNormalForce =
            kReceiverOverlapRailRadialPreload /
            jawMetadata.insertSystemNormalComplianceMPerN;
        const double estimatedReceiverTransportPatchNormalForce =
            kReceiverTransportRailRadialPreload /
            jawMetadata.insertSystemNormalComplianceMPerN;
        const double estimatedJawInsertMoment =
            static_cast<double>(kJawATeeth.size()) *
            estimatedPatchNormalForce *
            jawMetadata.largeNeedleDriverJawLength;
        const double estimatedReceiverTransportJawInsertMoment =
            static_cast<double>(kJawATeeth.size()) *
            estimatedReceiverTransportPatchNormalForce *
            jawMetadata.largeNeedleDriverJawLength;
        require(
            std::isfinite(estimatedPatchNormalForce) &&
                estimatedPatchNormalForce > 0.0 &&
                std::isfinite(estimatedReceiverOverlapPatchNormalForce) &&
                estimatedReceiverOverlapPatchNormalForce > 0.0 &&
                estimatedReceiverOverlapPatchNormalForce <
                    estimatedPatchNormalForce &&
                std::isfinite(
                    estimatedReceiverTransportPatchNormalForce
                ) &&
                estimatedReceiverTransportPatchNormalForce >
                    estimatedPatchNormalForce &&
                estimatedJawInsertMoment > 0.0 &&
                estimatedJawInsertMoment < psm.dofs[6].limits.w &&
                estimatedReceiverTransportJawInsertMoment >
                    estimatedJawInsertMoment &&
                estimatedReceiverTransportJawInsertMoment <
                    psm.dofs[6].limits.w,
            "research jaw preload exceeds the authored LND effort envelope"
        );
        const double closeJawCoordinate = calibratedJawCoordinate(
            psm,
            sutureSpec.needle.crossSectionRadiusM.value,
            -graspDiametralClosure
        );
        const double receiverOverlapJawCoordinate = calibratedJawCoordinate(
            psm,
            sutureSpec.needle.crossSectionRadiusM.value,
            -receiverOverlapDiametralClosure
        );
        const double receiverTransportJawCoordinate = calibratedJawCoordinate(
            psm,
            sutureSpec.needle.crossSectionRadiusM.value,
            -receiverTransportDiametralClosure
        );
        // A 0.30 mm diametral clearance admits the 0.70 mm needle. The
        // trajectory duration below is derived from the calibrated travel so
        // the LND's authored 0.2 rad/s speed limit remains authoritative even
        // when the physical jaw length changes.
        const double openJawCoordinate = calibratedJawCoordinate(
            psm,
            sutureSpec.needle.crossSectionRadiusM.value,
            3.0e-4
        );
        require(
            openJawCoordinate > receiverOverlapJawCoordinate &&
                receiverOverlapJawCoordinate > closeJawCoordinate &&
                closeJawCoordinate > receiverTransportJawCoordinate &&
                openJawCoordinate - closeJawCoordinate < 0.08,
            "gauge-calibrated jaw travel is outside the physical jaw range"
        );
        const std::uint32_t jawTravelSteps =
            static_cast<std::uint32_t>(std::ceil(
                3.0 * (openJawCoordinate - closeJawCoordinate) /
                (0.18 * kControlTimestep)
            )) + 4u;
        const std::uint32_t receiverOverlapTravelSteps =
            static_cast<std::uint32_t>(std::ceil(
                3.0 * (
                    openJawCoordinate - receiverOverlapJawCoordinate
                ) / (0.18 * kControlTimestep)
            )) + 4u;
        const auto giverPickupClosedNominal = psmTarget(
            psm,
            kPickupInsertion,
            closeJawCoordinate,
            kGiverYaw,
            kGiverPitch
        );
        const auto receiverHandoffOverlapNominal = psmTarget(
            psm,
            kReceiverHandoffInsertion,
            receiverOverlapJawCoordinate,
            kReceiverYaw,
            kReceiverPitch
        );
        const JawGeometry giverLocalJaw = jawGeometry(
            psm,
            giverPickupClosedNominal
        );
        const JawGeometry receiverLocalJaw = jawGeometry(
            psm,
            receiverHandoffOverlapNominal
        );
        const double insertRailHalfSpacing = 0.5 * std::abs(
            static_cast<double>(
                psm.shapes.at(kJawATeeth[1]).localPosition.z
            ) - psm.shapes.at(kJawATeeth[0]).localPosition.z
        );
        const double closedRailNeedleCenterDistance = std::hypot(
            0.5 * giverLocalJaw.separation,
            insertRailHalfSpacing
        );
        const double requiredRailNeedleCenterDistance =
            sutureSpec.needle.crossSectionRadiusM.value +
            psm.shapes.at(kJawATeeth[0]).dimensions.x -
            kGrooveRailRadialPreload;
        const double receiverOverlapRailNeedleCenterDistance = std::hypot(
            0.5 * receiverLocalJaw.separation,
            insertRailHalfSpacing
        );
        const double requiredReceiverOverlapRailNeedleCenterDistance =
            sutureSpec.needle.crossSectionRadiusM.value +
            psm.shapes.at(kJawATeeth[0]).dimensions.x -
            kReceiverOverlapRailRadialPreload;
        require(
            std::abs(
                closedRailNeedleCenterDistance -
                requiredRailNeedleCenterDistance
            ) <= 2.0e-5 &&
                std::abs(
                    receiverOverlapRailNeedleCenterDistance -
                    requiredReceiverOverlapRailNeedleCenterDistance
                ) <= 2.0e-5,
            "gauge-calibrated closure no longer seats the needle in the "
            "eight-patch V-groove"
        );

        metalrobo::DualPsmNeedleThreadNeutralZoneConfig config;
        config.surgical =
            metalrobo::makeBowelAnastomosisNeedleThreadWorldConfig(
                sutureSpec
            );
        const double threadShearModulus =
            sutureSpec.threadYoungModulusPa.value /
            (2.0 * (1.0 + sutureSpec.threadPoissonRatio.value));
        const double threadPolarMoment =
            0.5 * std::numbers::pi *
            std::pow(sutureSpec.threadRadiusM.value, 4.0);
        const double firstEdgeLength =
            sutureSpec.threadLengthM.value /
            static_cast<double>(sutureSpec.threadNodeCount - 1u);
        config.surgical.torsionalAttachmentComplianceRadPerNm =
            0.5 * firstEdgeLength /
            (threadShearModulus * threadPolarMoment);
        config.threadSolverIterations = options.rodSolverIterations;
        config.threadConstraintToleranceM = 2.0e-5;
        const double padTop = 0.5 * config.pad.thicknessM.value;
        const Vec3 needlePosition{
            0.0,
            0.0,
            padTop - needleForPlacement.rigid.localAabbLowerM[2] -
                kInitialSupportPenetration,
        };
        config.surgical.needlePose.position = {
            static_cast<float>(needlePosition.x),
            static_cast<float>(needlePosition.y),
            static_cast<float>(needlePosition.z),
        };

        const Vec3 giverPoint = needleShapeWorldCenter(
            needleForPlacement,
            kGiverNeedleShape,
            needlePosition
        );
        const Quaternion giverOrientation = multiply(
            rotationZ(needleShapeAngle(
                sutureSpec.needle,
                kGiverNeedleShape
            )),
            rotationX(std::numbers::pi)
        );
        const Quaternion receiverOrientation = multiply(
            multiply(
                rotationZ(needleShapeAngle(
                    sutureSpec.needle,
                    kReceiverNeedleShape
                ) + options.receiverBaseAzimuthOffset),
                rotationX(std::numbers::pi)
            ),
            rotationY(options.receiverNeedleAxisRoll)
        );
        config.surgical.robots.leftBase = basePose(
            giverOrientation,
            giverLocalJaw.midpoint,
            giverPoint
        );
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            config.surgical.robots.leftBase.position[axis] +=
                static_cast<float>((std::array{
                    kGiverPortCalibration.x,
                    kGiverPortCalibration.y,
                    kGiverPortCalibration.z,
                })[axis]);
        }
        const auto giverPickupClearanceClosed = solvePsmJawTarget(
            psm,
            config.surgical.robots.leftBase,
            giverPickupClosedNominal,
            giverPoint + Vec3{0.0, 0.0, kPickupVerticalClearance},
            closeJawCoordinate
        );
        const double nominalDistalPadGap = minimumPsmPadGap(
            psm,
            giverPickupClearanceClosed,
            config.surgical.robots.leftBase,
            padTop
        );
        const double nominalToothNeedleGap = minimumToothNeedleGap(
            psm,
            giverPickupClearanceClosed,
            config.surgical.robots.leftBase,
            needleForPlacement,
            needlePosition
        );
        require(
            nominalDistalPadGap > 5.0e-5 &&
                nominalToothNeedleGap < 0.0,
            "pickup pose does not simultaneously clear the pad and enter "
            "the authored needle contact envelope"
        );
        const Vec3 handoffStagingOffset = options.receiverCollisionScan
            ? Vec3{
                options.receiverScanHandoffOffsetX,
                options.receiverScanHandoffOffsetY,
                options.receiverScanHandoffHeightIncrement,
            }
            : kHandoffStagingOffset;
        const Vec3 receiverPoint = needleShapeWorldCenter(
            needleForPlacement,
            kReceiverNeedleShape,
            needlePosition
        ) + handoffStagingOffset + Vec3{0.0, 0.0, kHandoffLift};
        config.surgical.robots.rightBase = basePose(
            receiverOrientation,
            receiverLocalJaw.midpoint,
            receiverPoint
        );
        const Vec3 baseSeparation =
            vector(config.surgical.robots.leftBase.position) -
            vector(config.surgical.robots.rightBase.position);
        require(
            norm(baseSeparation) > 0.22,
            "dual PSM port geometry does not separate the arm bases"
        );
        config.surgical.robots.timestep =
            static_cast<float>(kControlTimestep);

        if (geometryOnly) {
            std::cout << std::scientific << std::setprecision(9);
            std::cout << "close_jaw_coordinate_rad="
                << closeJawCoordinate
                << " groove_diametral_closure_m="
                << graspDiametralClosure
                << " estimated_patch_normal_force_n="
                << estimatedPatchNormalForce
                << " estimated_jaw_insert_moment_nm="
                << estimatedJawInsertMoment
                << " rail_needle_center_distance_m="
                << closedRailNeedleCenterDistance
                << " open_jaw_coordinate_rad="
                << openJawCoordinate
                << " closure_steps=" << jawTravelSteps
                << " receiver_overlap_radial_preload_m="
                << kReceiverOverlapRailRadialPreload
                << " receiver_overlap_patch_normal_force_n="
                << estimatedReceiverOverlapPatchNormalForce
                << " receiver_overlap_jaw_coordinate_rad="
                << receiverOverlapJawCoordinate
                << " receiver_overlap_closure_steps="
                << receiverOverlapTravelSteps
                << " receiver_transport_radial_preload_m="
                << kReceiverTransportRailRadialPreload
                << " receiver_transport_patch_normal_force_n="
                << estimatedReceiverTransportPatchNormalForce
                << " receiver_transport_jaw_insert_moment_nm="
                << estimatedReceiverTransportJawInsertMoment
                << " receiver_transport_jaw_coordinate_rad="
                << receiverTransportJawCoordinate << '\n';
            std::cout << "pickup_vertical_offset_m="
                << kPickupVerticalClearance
                << " nominal_distal_pad_gap_m="
                << nominalDistalPadGap
                << " nominal_tooth_needle_gap_m="
                << nominalToothNeedleGap << '\n';
            const double centerlineRadius =
                sutureSpec.needle.arcLengthM.value /
                sutureSpec.needle.arcAngleRad.value;
            const double startAngle =
                -0.5 * sutureSpec.needle.arcAngleRad.value;
            std::cout << "swage_anchor_local_m="
                << vectorSummary({
                    centerlineRadius * std::cos(startAngle) -
                        needleForPlacement.rigid.geometryCenterOfMassM[0],
                    centerlineRadius * std::sin(startAngle) -
                        needleForPlacement.rigid.geometryCenterOfMassM[1],
                    -needleForPlacement.rigid.geometryCenterOfMassM[2],
                }) << '\n';
            for (std::uint32_t sample = 0u; sample <= 20u; ++sample) {
                const double retraction = 0.001 * sample;
                const auto q = psmTarget(
                    psm,
                    kPickupInsertion - retraction,
                    openJawCoordinate,
                    kGiverYaw,
                    kGiverPitch
                );
                std::cout << "retraction_m=" << retraction
                    << " minimum_tooth_needle_gap_m="
                    << minimumToothNeedleGap(
                           psm,
                           q,
                           config.surgical.robots.leftBase,
                           needleForPlacement,
                           needlePosition
                       ) << '\n';
            }
            return 0;
        }

        metalrobo::HeterogeneousWorld world;
        const auto composed =
            metalrobo::
                makeDualDvrkPsmNeedleThreadNeutralZoneHeterogeneousWorld(
                    world,
                    config
                );
        require(
            composed.succeeded(),
            "neutral-zone handoff world composition failed: " +
                composed.message
        );
        require(
            world.model.materials.size() == 7u,
            "neutral-zone handoff material topology changed"
        );
        const auto& psmMetadata = metalrobo::surgicalPSMMetadata();
        const double effectiveInsertStaticFriction = std::sqrt(
            static_cast<double>(world.model.materials[1].friction.x) *
            world.model.materials[4].friction.x
        );
        const double effectiveInsertDynamicFriction = std::sqrt(
            static_cast<double>(world.model.materials[1].friction.y) *
            world.model.materials[4].friction.y
        );
        require(
            world.model.articulations.size() == 2u &&
                world.sceneBodyIndices.size() == 2u &&
                world.rods.size() == 1u &&
                world.model.materials.size() == 7u &&
                world.rods[0].collision.ownedMaterial.has_value() &&
                world.rods[0].collision.materialIndex == 6u &&
                world.rods[0].collision.enableToolCollision &&
                world.rods[0].collision.enableCCD &&
                world.model.materials[6].friction.x == 0.18f &&
                world.model.materials[6].friction.y == 0.12f &&
                world.model.materials[1].response.z ==
                    psmMetadata.insertSystemNormalComplianceMPerN &&
                world.model.materials[3].response.z ==
                    psmMetadata.insertSystemNormalComplianceMPerN &&
                std::abs(
                    effectiveInsertStaticFriction -
                    psmMetadata.targetNeedleInsertStaticFriction
                ) < 1.0e-6 &&
                std::abs(
                    effectiveInsertDynamicFriction -
                    psmMetadata.targetNeedleInsertDynamicFriction
                ) < 1.0e-6 &&
                world.rods[0].attachments.size() == 1u &&
                world.rods[0].tangentBindings.size() == 1u &&
                world.rods[0].twistBindings.size() == 1u &&
                world.rods[0].attachments[0].compliance == 0.0 &&
                world.rods[0].tangentBindings[0].edgeIndex == 0u &&
                world.rods[0].tangentBindings[0]
                        .complianceRadPerNm ==
                    config.surgical
                        .tangentAttachmentComplianceRadPerNm &&
                world.rods[0].twistBindings[0]
                        .complianceRadPerNm ==
                    config.surgical
                        .torsionalAttachmentComplianceRadPerNm &&
                world.rods[0].model.restPositions.size() == 128u,
            "handoff world lost a robot, scene body, or suture node"
        );
        // Self contact remains live even though the authored 3 mm-pitch coil
        // starts separated. Terminal clearance alone cannot prove that a
        // moving monofilament stayed separated earlier in a phase, so the DER
        // projector enforces capsule separation at every physics substep.
        require(
                world.rods[0].stepConfig.solverIterations ==
                    options.rodSolverIterations &&
                world.rods[0].stepConfig.constraintTolerance ==
                    config.threadConstraintToleranceM &&
                world.rods[0].stepConfig.selfCollisionMargin ==
                    kMinimumThreadSelfCollisionClearance &&
                world.rods[0].stepConfig.linearDamping == 8.0 &&
                world.rods[0].stepConfig.twistDamping == 8.0,
            "neutral-zone world lost its certified thread step defaults"
        );
        require(
            world.rods[0].stepConfig.enableSelfCollision,
            "neutral-zone world disabled per-substep thread self contact"
        );
        // The needle, both instruments, and neutral-zone pad remain ordinary
        // collision partners for every free thread edge. The compiler removes
        // only the edge/body pairs adjacent to the two explicit swage
        // attachments, so distant needle/thread contact is still physical.
        world.rods[0].collision.collisionMask = ~0u;

        const auto giverStart = psmTarget(
            psm,
            kPickupInsertion - kPickupRetraction,
            openJawCoordinate,
            kGiverYaw,
            kGiverPitch
        );
        const auto receiverStart = psmTarget(
            psm,
            kReceiverHandoffInsertion - kReceiverRetraction,
            openJawCoordinate,
            kReceiverYaw,
            kReceiverPitch
        );
        const std::vector<float> authoredWorldQ = world.model.defaultQ;
        const std::vector<float> authoredWorldV = world.model.defaultV;
        std::uint64_t loadedStateStep = 0u;
        if (options.resumeGiverClosedPath.empty() &&
            options.resumeGiverLiftPath.empty() &&
            options.resumeGiverHandoffStagePath.empty() &&
            options.resumeGiverHandoffStagePrefixPath.empty() &&
            options.resumeReceiverApproachMotionPath.empty() &&
            options.resumeReceiverApproachPath.empty() &&
            options.resumeReceiverAlignedPath.empty() &&
            options.resumePositiveControlMotionPath.empty() &&
            options.resumePositiveControlOverlapPath.empty() &&
            options.resumeGiverReleaseMotionPath.empty() &&
            options.resumeGiverReleasePath.empty() &&
            options.resumeReceiverTransferMotionPath.empty() &&
            options.resumeApproachHeldPath.empty()) {
            setArmTarget(world.model.defaultQ, world.model, 0u, giverStart);
            setArmTarget(world.model.defaultQ, world.model, 1u, receiverStart);
        } else if (!options.resumeGiverClosedPath.empty()) {
            require(
                options.mode.empty(),
                "giver-closed resume cannot be combined with another mode"
            );
            loadedStateStep = loadHandoffState(
                options.resumeGiverClosedPath,
                "giver-closed",
                world
            );
        } else if (!options.resumeGiverLiftPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeGiverLiftPath,
                "giver-lift",
                world
            );
        } else if (!options.resumeGiverHandoffStagePath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeGiverHandoffStagePath,
                "giver-handoff-stage",
                world
            );
        } else if (!options.resumeGiverHandoffStagePrefixPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeGiverHandoffStagePrefixPath,
                "giver-handoff-stage-prefix",
                world
            );
        } else if (!options.resumeReceiverApproachMotionPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeReceiverApproachMotionPath,
                "receiver-approach-motion",
                world
            );
        } else if (!options.resumeReceiverApproachPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeReceiverApproachPath,
                "receiver-approach",
                world
            );
        } else if (!options.resumeReceiverAlignedPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeReceiverAlignedPath,
                "receiver-aligned",
                world
            );
        } else if (!options.resumePositiveControlMotionPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumePositiveControlMotionPath,
                "positive-control-motion",
                world
            );
        } else if (!options.resumePositiveControlOverlapPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumePositiveControlOverlapPath,
                "positive-control-overlap",
                world
            );
        } else if (!options.resumeGiverReleaseMotionPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeGiverReleaseMotionPath,
                "giver-release-motion",
                world
            );
        } else if (!options.resumeGiverReleasePath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeGiverReleasePath,
                "giver-release",
                world
            );
        } else if (!options.resumeReceiverTransferMotionPath.empty()) {
            loadedStateStep = loadHandoffState(
                options.resumeReceiverTransferMotionPath,
                "receiver-transfer-motion",
                world
            );
        } else {
            loadedStateStep = loadHandoffState(
                options.resumeApproachHeldPath,
                "giver-approach-held",
                world
            );
        }
        if (!options.resumeGiverLiftPath.empty() ||
            !options.resumeGiverHandoffStagePath.empty() ||
            !options.resumeGiverHandoffStagePrefixPath.empty()) {
            const MRArticulationGPU& receiver =
                world.model.articulations.at(1u);
            require(
                receiver.qOffset + 7u <= authoredWorldQ.size() &&
                    receiver.vOffset + 6u <= authoredWorldV.size(),
                "receiver base root is outside the authored world state"
            );
            // Pre-receiver checkpoints may predate a port-placement change,
            // and the receiver has not interacted with the scene yet. Fixed
            // RCM roots are authored world geometry rather than dynamic
            // checkpoint state, so restore the current receiver root while
            // retaining every dynamic giver/needle/thread coordinate.
            std::copy_n(
                authoredWorldQ.begin() + receiver.qOffset,
                7u,
                world.model.defaultQ.begin() + receiver.qOffset
            );
            std::copy_n(
                authoredWorldV.begin() + receiver.vOffset,
                6u,
                world.model.defaultV.begin() + receiver.vOffset
            );
        }
        world.fingerprint = metalrobo::heterogeneousWorldFingerprint(world);
        std::string reason;
        require(
            world.valid(&reason),
            "authored handoff reset is invalid: " + reason
        );
        const double authoredMaterialFrameError =
            initialSwageMaterialFrameError(world);
        require(
            std::isfinite(authoredMaterialFrameError),
            "authored swage material-frame error is non-finite"
        );
        std::cerr << "handoff_phase=authored_reset"
            << " swage_material_frame_error_rad="
            << authoredMaterialFrameError << '\n';

        if (options.receiverAlignmentScan) {
            const JawGeometry receiverJaw = worldJawGeometry(
                world.model,
                1u,
                world.model.defaultQ,
                world.model.defaultV
            );
            const Vec3 receiverNeedlePoint = needleShapeWorldCenter(
                needleForPlacement,
                kReceiverNeedleShape,
                world.defaultSceneBodies.at(0u)
            );
            const Vec3 centeringDelta =
                receiverNeedlePoint - receiverJaw.midpoint;
            const Vec3 insertNormal = cross(
                receiverJaw.railDirection,
                receiverJaw.separationDirection
            );
            const Vec3 needleTangent = needleShapeWorldTangent(
                needleForPlacement,
                kReceiverNeedleShape,
                world.defaultSceneBodies.at(0u)
            );
            const double railTangentAngle = std::acos(std::clamp(
                std::abs(dot(receiverJaw.railDirection, needleTangent)),
                0.0,
                1.0
            ));
            const ReceiverAlignmentSolution aligned =
                solveReceiverAlignmentTarget(
                    world,
                    psm,
                    needleForPlacement,
                    world.model.defaultQ,
                    world.model.defaultV,
                    world.defaultSceneBodies.at(0u),
                    openJawCoordinate
                );
            std::vector<float> alignedTarget = world.model.defaultQ;
            setArmTarget(
                alignedTarget,
                world.model,
                1u,
                aligned.localQ
            );
            const std::uint32_t alignedMotionSteps =
                velocityLimitedTargetSteps(
                    world.model,
                    world.model.defaultQ,
                    alignedTarget,
                    kReceiverAlignmentMinimumSteps
                );
            const CrossArmCollisionScan alignmentPath =
                scanCrossArmTargetPath(
                    world,
                    world.model.defaultQ,
                    alignedTarget,
                    100u
                );
            const KinematicNeedleObservation alignedObservation =
                observeKinematicNeedleContacts(
                    world,
                    needleForPlacement.metadata,
                    alignedTarget
                );
            require(
                alignmentPath.samplesWithContact == 0u &&
                    alignmentPath.samplesWithGiverPadContact == 0u &&
                    alignmentPath.samplesWithReceiverPadContact == 0u &&
                    alignedObservation.receiverJawContacts[0] == 0u &&
                    alignedObservation.receiverJawContacts[1] == 0u,
                "receiver live-frame alignment path is not collision-free"
            );
            std::cout << std::setprecision(9)
                << "receiver_alignment_scan=ok"
                << " centering_error_m=" << norm(centeringDelta)
                << " rail_offset_m="
                << dot(centeringDelta, receiverJaw.railDirection)
                << " separation_offset_m="
                << dot(centeringDelta, receiverJaw.separationDirection)
                << " insert_normal_offset_m="
                << dot(centeringDelta, insertNormal)
                << " rail_tangent_angle_rad=" << railTangentAngle
                << " aligned_centering_residual_m="
                << aligned.centeringResidual
                << " aligned_rail_tangent_angle_rad="
                << aligned.railTangentAngle
                << " aligned_separation_frame_angle_rad="
                << aligned.separationFrameAngle
                << " aligned_tool_roll_departure_rad="
                << aligned.toolRollDeparture
                << " aligned_wrist_yaw_departure_rad="
                << aligned.wristYawDeparture
                << " aligned_motion_steps=" << alignedMotionSteps
                << " aligned_path_cross_arm_samples="
                << alignmentPath.samplesWithContact
                << " jaw_midpoint="
                << vectorSummary(receiverJaw.midpoint)
                << " needle_point="
                << vectorSummary(receiverNeedlePoint) << '\n';
            return 0;
        }

        if (options.receiverCollisionScan) {
            const Vec3 scanHandoffOffset = handoffStagingOffset;
            double stagingMaximumVelocityRatio = 0.0;
            double stagingMaximumVelocity = 0.0;
            double stagingLimitingVelocity = 0.0;
            std::uint32_t stagingMaximumVelocityDof = MR_INVALID_INDEX;
            CrossArmCollisionScan stagingScan;
            const bool scanFromStagedCheckpoint =
                !options.resumeGiverHandoffStagePath.empty();
            if (norm(scanHandoffOffset) > 0.0 &&
                !scanFromStagedCheckpoint) {
                const JawGeometry giverJaw = worldJawGeometry(
                    world.model,
                    0u,
                    world.model.defaultQ,
                    world.model.defaultV
                );
                const ArmTrajectory staging = cartesianArmTrajectory(
                    world.model,
                    psm,
                    0u,
                    config.surgical.robots.leftBase,
                    world.model.defaultQ,
                    giverJaw.midpoint,
                    giverJaw.midpoint + scanHandoffOffset,
                    closeJawCoordinate,
                    closeJawCoordinate,
                    kHandoffStagingSteps
                );
                stagingMaximumVelocityRatio =
                    staging.maximumVelocityRatio;
                stagingMaximumVelocity = staging.maximumVelocity;
                stagingLimitingVelocity = staging.limitingVelocity;
                stagingMaximumVelocityDof = staging.maximumVelocityDof;
                stagingScan = scanCrossArmTargetPath(
                    world,
                    world.model.defaultQ,
                    staging.finalTarget,
                    kHandoffStagingSteps,
                    staging.desiredQ
                );
                world.model.defaultQ = staging.finalTarget;
                world.defaultSceneBodies[0u].position.x +=
                    static_cast<float>(scanHandoffOffset.x);
                world.defaultSceneBodies[0u].position.y +=
                    static_cast<float>(scanHandoffOffset.y);
                world.defaultSceneBodies[0u].position.z +=
                    static_cast<float>(scanHandoffOffset.z);
            }
            const Vec3 liftedReceiverPoint = needleShapeWorldCenter(
                needleForPlacement,
                kReceiverNeedleShape,
                world.defaultSceneBodies.at(0u)
            );
            auto receiverSeed = psmTarget(
                psm,
                kReceiverHandoffInsertion,
                openJawCoordinate,
                kReceiverYaw,
                kReceiverPitch
            );
            receiverSeed[3] += options.receiverToolRollOffset;
            receiverSeed[5] = options.receiverWristYaw;
            const auto receiverHandoffOpen = solvePsmJawTarget(
                psm,
                config.surgical.robots.rightBase,
                std::move(receiverSeed),
                liftedReceiverPoint,
                openJawCoordinate,
                0.015
            );
            std::vector<float> target = world.model.defaultQ;
            setArmTarget(
                target,
                world.model,
                1u,
                receiverHandoffOpen
            );
            const CrossArmCollisionScan scan = scanCrossArmTargetPath(
                world,
                world.model.defaultQ,
                target,
                150u
            );
            std::vector<float> closedTarget = target;
            auto receiverHandoffClosed = receiverHandoffOpen;
            receiverHandoffClosed[6] = -receiverOverlapJawCoordinate;
            receiverHandoffClosed[7] = receiverOverlapJawCoordinate;
            setArmTarget(
                closedTarget,
                world.model,
                1u,
                receiverHandoffClosed
            );
            const CrossArmCollisionScan closureScan =
                scanCrossArmTargetPath(
                    world,
                    target,
                    closedTarget,
                    50u
                );
            const KinematicNeedleObservation closedObservation =
                observeKinematicNeedleContacts(
                    world,
                    needleForPlacement.metadata,
                    closedTarget
                );
            const std::vector<float> zeroVelocity(
                world.model.world.nv,
                0.0f
            );
            const JawGeometry closedReceiverJaw = worldJawGeometry(
                world.model,
                1u,
                closedTarget,
                zeroVelocity
            );
            const JawGeometry openReceiverJaw = worldJawGeometry(
                world.model,
                1u,
                target,
                zeroVelocity
            );
            const double uncompensatedClosureMidpointSweep = norm(
                closedReceiverJaw.midpoint - openReceiverJaw.midpoint
            );
            const Vec3 receiverNeedleTangent = needleShapeWorldTangent(
                needleForPlacement,
                kReceiverNeedleShape,
                world.defaultSceneBodies.at(0u)
            );
            const double closedRailTangentAngle = std::acos(std::clamp(
                std::abs(dot(
                    closedReceiverJaw.railDirection,
                    receiverNeedleTangent
                )),
                0.0,
                1.0
            ));
            const JawGeometry stagedGiverJaw = worldJawGeometry(
                world.model,
                0u,
                world.model.defaultQ,
                world.model.defaultV
            );
            const Vec3 giverNeedleTangent = needleShapeWorldTangent(
                needleForPlacement,
                kGiverNeedleShape,
                world.defaultSceneBodies.at(0u)
            );
            const double giverRailTangentAngle = std::acos(std::clamp(
                std::abs(dot(
                    stagedGiverJaw.railDirection,
                    giverNeedleTangent
                )),
                0.0,
                1.0
            ));
            const MRBodyStateGPU& stagedNeedleBody =
                world.defaultSceneBodies.at(0u);
            const Quaternion stagedNeedleOrientation{
                stagedNeedleBody.orientation.x,
                stagedNeedleBody.orientation.y,
                stagedNeedleBody.orientation.z,
                stagedNeedleBody.orientation.w,
            };
            const MRShapeGPU& giverSegment =
                needleForPlacement.rigid.shapes.at(kGiverNeedleShape);
            const MRShapeGPU& receiverSegment =
                needleForPlacement.rigid.shapes.at(kReceiverNeedleShape);
            const Quaternion giverSegmentOrientation = multiply(
                stagedNeedleOrientation,
                Quaternion{
                    giverSegment.localRotation.x,
                    giverSegment.localRotation.y,
                    giverSegment.localRotation.z,
                    giverSegment.localRotation.w,
                }
            );
            const Quaternion receiverSegmentOrientation = multiply(
                stagedNeedleOrientation,
                Quaternion{
                    receiverSegment.localRotation.x,
                    receiverSegment.localRotation.y,
                    receiverSegment.localRotation.z,
                    receiverSegment.localRotation.w,
                }
            );
            const Vec3 giverSeparationInSegment = rotate(
                conjugate(giverSegmentOrientation),
                stagedGiverJaw.separationDirection
            );
            const Vec3 receiverSeparationInSegment = rotate(
                conjugate(receiverSegmentOrientation),
                closedReceiverJaw.separationDirection
            );
            const double separationFrameAngle = std::acos(std::clamp(
                std::abs(dot(
                    giverSeparationInSegment,
                    receiverSeparationInSegment
                )),
                0.0,
                1.0
            ));
            std::cout << std::setprecision(9)
                << "receiver_collision_scan=ok"
                << " tool_roll_offset_rad="
                << options.receiverToolRollOffset
                << " wrist_yaw_rad=" << options.receiverWristYaw
                << " base_azimuth_offset_rad="
                << options.receiverBaseAzimuthOffset
                << " needle_axis_roll_rad="
                << options.receiverNeedleAxisRoll
                << " handoff_height_increment_m="
                << options.receiverScanHandoffHeightIncrement
                << " handoff_offset_x_m="
                << options.receiverScanHandoffOffsetX
                << " handoff_offset_y_m="
                << options.receiverScanHandoffOffsetY
                << " staging_steps=" << kHandoffStagingSteps
                << " staging_max_velocity_ratio="
                << stagingMaximumVelocityRatio
                << " staging_max_velocity="
                << stagingMaximumVelocity
                << " staging_velocity_limit="
                << stagingLimitingVelocity
                << " staging_velocity_dof="
                << stagingMaximumVelocityDof
                << " staging_cross_arm_samples="
                << stagingScan.samplesWithContact
                << " staging_giver_pad_samples="
                << stagingScan.samplesWithGiverPadContact
                << " staging_receiver_pad_samples="
                << stagingScan.samplesWithReceiverPadContact
                << " receiver_root_x_m="
                << world.model.defaultQ[
                    world.model.articulations.at(1u).qOffset
                ]
                << " receiver_root_y_m="
                << world.model.defaultQ[
                    world.model.articulations.at(1u).qOffset + 1u
                ]
                << " receiver_root_z_m="
                << world.model.defaultQ[
                    world.model.articulations.at(1u).qOffset + 2u
                ]
                << " base_separation_m=" << norm(baseSeparation)
                << " lifted_needle_z_m="
                << world.defaultSceneBodies.at(0u).position.z
                << " samples_with_contact="
                << scan.samplesWithContact
                << " first_contact_step=" << scan.firstContactStep
                << " last_contact_step=" << scan.lastContactStep
                << " minimum_separation_m="
                << (
                    std::isfinite(scan.minimumSeparation)
                    ? scan.minimumSeparation
                    : 0.0
                )
                << " bodies=" << scan.bodyA << '/' << scan.bodyB
                << " colliders=" << scan.colliderA << '/'
                << scan.colliderB
                << " giver_pad_samples="
                << scan.samplesWithGiverPadContact
                << " giver_pad_first_step="
                << scan.firstGiverPadContactStep
                << " giver_pad_last_step="
                << scan.lastGiverPadContactStep
                << " giver_pad_minimum_separation_m="
                << (
                    std::isfinite(scan.minimumGiverPadSeparation)
                    ? scan.minimumGiverPadSeparation
                    : 0.0
                )
                << " giver_pad_collider=" << scan.giverPadCollider
                << " receiver_pad_samples="
                << scan.samplesWithReceiverPadContact
                << " receiver_pad_first_step="
                << scan.firstReceiverPadContactStep
                << " receiver_pad_last_step="
                << scan.lastReceiverPadContactStep
                << " receiver_pad_minimum_separation_m="
                << (
                    std::isfinite(scan.minimumReceiverPadSeparation)
                    ? scan.minimumReceiverPadSeparation
                    : 0.0
                )
                << " receiver_pad_collider="
                << scan.receiverPadCollider
                << " closure_samples_with_contact="
                << closureScan.samplesWithContact
                << " closure_receiver_pad_samples="
                << closureScan.samplesWithReceiverPadContact
                << " closure_receiver_pad_minimum_separation_m="
                << (
                    std::isfinite(
                        closureScan.minimumReceiverPadSeparation
                    )
                    ? closureScan.minimumReceiverPadSeparation
                    : 0.0
                )
                << " closure_receiver_pad_collider="
                << closureScan.receiverPadCollider
                << " closed_cross_arm_contacts="
                << closedObservation.crossArmContacts
                << " closed_minimum_cross_arm_separation_m="
                << (
                    std::isfinite(
                        closedObservation.minimumCrossArmSeparation
                    )
                    ? closedObservation.minimumCrossArmSeparation
                    : 0.0
                )
                << " closed_receiver_jaws="
                << closedObservation.receiverJawContacts[0] << '/'
                << closedObservation.receiverJawContacts[1]
                << " closed_receiver_grasp_zone="
                << closedObservation.receiverGraspZoneContacts
                << " closed_rail_tangent_angle_rad="
                << closedRailTangentAngle
                << " giver_rail_tangent_angle_rad="
                << giverRailTangentAngle
                << " uncompensated_closure_midpoint_sweep_m="
                << uncompensatedClosureMidpointSweep
                << " receiver_giver_separation_frame_angle_rad="
                << separationFrameAngle << '\n';
            return 0;
        }

        numi::matter::Runtime tissueRuntime;
        numi::matter::CompiledWorld tissueWorld;
        numi::matter::PorcineJejunumClosureCoupon tissueCoupon;
        std::optional<CurvedNeedleOrbit> tissueNeedleOrbit;
        double tissueNeedleAngularSpeedRadPerS = 0.0;
        if (tissueMatterOnly) {
            const NeedleTipCapsuleGeometry tip = needleTipCapsuleGeometry(
                needleForPlacement,
                world.defaultSceneBodies[0]
            );
            Vec3 initialNeedleVelocity = tissuePunctureOnly
                ? tip.approachDirection * (
                    tissuePunctureAdvanceOnly
                        ? kPunctureChannelProbeSpeedMps
                        : kPunctureApproachSpeedMps)
                : Vec3{};
            Vec3 initialNeedleAngularVelocity{};
            if (tissuePunctureAdvanceOnly || tissueCurvedPassageOnly) {
                require(
                    world.sceneBodyIndices[0] < world.model.bodies.size(),
                    "needle scene body has no compiled model owner"
                );
                MRBodyPropertiesGPU& needleProperties =
                    world.model.bodies[world.sceneBodyIndices[0]];
                needleProperties.motionType = MR_MOTION_KINEMATIC;
                needleProperties.massAndInverseMass.y = 0.0f;
                world.defaultSceneBodies[0].flagsAndIndices[0] =
                    MR_MOTION_KINEMATIC;
                world.defaultSceneBodies[0]
                    .linearVelocityAndInverseMass.w = 0.0f;
                world.defaultSceneBodies[0].inverseInertiaWorldRow0 = {};
                world.defaultSceneBodies[0].inverseInertiaWorldRow1 = {};
                world.defaultSceneBodies[0].inverseInertiaWorldRow2 = {};
            }
            if (tissueCurvedPassageOnly) {
                tissueNeedleOrbit = curvedNeedleOrbit(
                    needleForPlacement,
                    world.defaultSceneBodies[0]
                );
                tissueNeedleAngularSpeedRadPerS =
                    kCurvedPassageSpeedMps /
                    tissueNeedleOrbit->centerlineRadiusM;
                world.defaultSceneBodies[0] = curvedNeedleTarget(
                    world.defaultSceneBodies[0],
                    *tissueNeedleOrbit,
                    0.0,
                    tissueNeedleAngularSpeedRadPerS
                );
                initialNeedleVelocity = vector(
                    world.defaultSceneBodies[0]
                        .linearVelocityAndInverseMass
                );
                initialNeedleAngularVelocity = vector(
                    world.defaultSceneBodies[0].angularVelocity
                );
                const NeedleTipCapsuleGeometry drivenTip =
                    needleTipCapsuleGeometry(
                        needleForPlacement,
                        world.defaultSceneBodies[0]
                    );
                const Vec3 terminalVelocity = initialNeedleVelocity + cross(
                    initialNeedleAngularVelocity,
                    drivenTip.worldTip -
                        vector(world.defaultSceneBodies[0].position)
                );
                require(
                    std::abs(
                        norm(terminalVelocity) -
                        kCurvedPassageSpeedMps
                    ) <= 2.0e-5 &&
                        dot(
                            terminalVelocity,
                            drivenTip.approachDirection
                        ) > 0.999 * kCurvedPassageSpeedMps,
                    "curved needle orbit does not preserve terminal entry speed"
                );
            }
            world.defaultSceneBodies[0].linearVelocityAndInverseMass.x =
                static_cast<float>(initialNeedleVelocity.x);
            world.defaultSceneBodies[0].linearVelocityAndInverseMass.y =
                static_cast<float>(initialNeedleVelocity.y);
            world.defaultSceneBodies[0].linearVelocityAndInverseMass.z =
                static_cast<float>(initialNeedleVelocity.z);
            if (tissueCurvedPassageOnly) {
                const MRBodyStateGPU& needle = world.defaultSceneBodies[0];
                const Vec3 bodyPosition = vector(needle.position);
                const Quaternion orientation{
                    needle.orientation.x,
                    needle.orientation.y,
                    needle.orientation.z,
                    needle.orientation.w,
                };
                const Vec3 localAnchor = vector(
                    world.rods[0].rigidBindings[0].localAnchor
                );
                const Vec3 anchorOffset = rotate(
                    orientation,
                    localAnchor
                );
                const Vec3 anchorVelocity = initialNeedleVelocity + cross(
                    initialNeedleAngularVelocity,
                    anchorOffset
                );
                world.rods[0].attachments[0].targetVelocity = {
                    anchorVelocity.x,
                    anchorVelocity.y,
                    anchorVelocity.z,
                };
                // Match the hard root and its tangent edge to the prescribed
                // rigid twist. The rest of the loose 250 mm monofilament starts
                // at rest and is accelerated only through the physical swage.
                const std::size_t carriedNodes = std::min<std::size_t>(
                    2u,
                    world.rods[0].defaultState.velocities.size()
                );
                for (std::size_t node = 0u;
                     node < carriedNodes;
                     ++node) {
                    const Vec3 point = vector(
                        world.rods[0].defaultState.positions[node]
                    );
                    const Vec3 velocity = initialNeedleVelocity + cross(
                        initialNeedleAngularVelocity,
                        point - bodyPosition
                    );
                    world.rods[0].defaultState.velocities[node] = {
                        velocity.x,
                        velocity.y,
                        velocity.z,
                    };
                }
            } else {
                world.rods[0].attachments[0].targetVelocity = {
                    initialNeedleVelocity.x,
                    initialNeedleVelocity.y,
                    initialNeedleVelocity.z,
                };
                // Translate the reset strand with the needle so the puncture
                // transaction does not begin from an artificial swage impulse.
                for (std::size_t node = 0u;
                     node < world.rods[0].defaultState.velocities.size();
                     ++node) {
                    world.rods[0].defaultState.velocities[node] = {
                        initialNeedleVelocity.x,
                        initialNeedleVelocity.y,
                        initialNeedleVelocity.z,
                    };
                }
            }
            world.fingerprint =
                metalrobo::heterogeneousWorldFingerprint(world);
            tissueWorld = compileNeedleSutureTissueWorld(
                world,
                needleForPlacement,
                tissuePunctureOnly
                    ? kPunctureInitialClearanceM
                    : (tissueRestOnly ? 1.5e-4 : 5.0e-5),
                tissuePunctureOnly,
                tissueCurvedPassageOnly
                    ? kCurvedPassageContactSegmentCount : 1u,
                tissueCoupon
            );
            const auto initialized = tissueRuntime.initialize(
                tissueWorld,
                {
                    .metallib = NUMI_MATTER_METALLIB,
                    .environmentCount = 1u,
                    .captureEvents = true,
                    .captureDiagnostics = true,
                    .automaticIdentification = false,
                    .adaptiveTransfer = false,
                }
            );
            require(
                initialized.encoded && tissueRuntime.valid(),
                "needle-suture tissue runtime initialization failed: " +
                    initialized.message
            );
        }

        metalrobo::CompiledWorld compiled;
        const auto compile = metalrobo::compileMetalWorld(
            world,
            compiled
        );
        require(
            compile.succeeded() &&
                compiled.articulationCount() == 2u &&
                compiled.sceneBodyCount() == 2u &&
                compiled.rodCount() == 1u,
            "handoff world did not compile for persistent Metal: " +
                compile.message
        );

        metalrobo::MetalWorldStepConfig stepConfig;
        stepConfig.timestepSeconds =
            static_cast<float>(kControlTimestep);
        stepConfig.physicsSubsteps = kPhysicsSubsteps;
        stepConfig.solverMode =
            metalrobo::MetalWorldSolverMode::temporalCone;
        stepConfig.actuationMode =
            metalrobo::MetalWorldActuationMode::implicitPositionDrive;
        // Base locks, jaw gears, both articulated chains, the supported rod,
        // the swage, and rigid contact remain one coupled island. Four rod
        // endpoint/contact sweeps reduce the hard-swage/strand split exposed
        // by the full lift while the live terminal residual remains the
        // acceptance authority; no contact, attachment, or DER tolerance is
        // weakened.
        stepConfig.velocityIterations = options.velocityIterations;
        stepConfig.finalVelocityIterations =
            options.finalVelocityIterations;
        stepConfig.ccdMode = metalrobo::MetalWorldCCDMode::speculative;
        stepConfig.maxConservativeAdvancementIterations = 24u;
        stepConfig.deterministic = true;
        stepConfig.warmStart = true;
        stepConfig.captureContactEvidence = true;
        stepConfig.publishFinalState = true;
        stepConfig.publishStateTrajectory = false;
        stepConfig.rodContactOuterIterations = 4u;
        if (tissueMatterOnly) {
            // Qualify one atomic 62.5 us cross-domain transaction. The
            // production control step contains 32 of these substeps, but a
            // later substep must not erase evidence from the boundary being
            // qualified here; long-horizon puncture stability is gated by
            // the separate surgical-sequence replay.
            stepConfig.timestepSeconds = static_cast<float>(
                kControlTimestep / static_cast<double>(kPhysicsSubsteps)
            );
            stepConfig.physicsSubsteps = 1u;
            stepConfig.devicePhysicsProgram =
                numi::matter::makeMetalWorldDevicePhysicsProgram(
                    tissueRuntime
                );
                require(
                    stepConfig.devicePhysicsProgram.valid() &&
                    (tissuePunctureAdvanceOnly ||
                     tissueCurvedPassageOnly ||
                     (stepConfig.devicePhysicsProgram.flags &
                      metalrobo::
                          MetalWorldDevicePhysicsWritesBodyWrenches) != 0u),
                "tissue runtime did not publish the required MetalWorld "
                "coupling direction"
            );
        }

        std::vector<float> targetStart = world.model.defaultQ;
        std::vector<float> target = targetStart;
        metalrobo::MetalWorldContext context;
        metalrobo::MetalWorldResidentState resident;
        std::vector<float> efforts;
        std::optional<PhaseResult> qualifiedLift;
        std::optional<PhaseResult> qualifiedOverlap;
        bool receiverApproachMotionAlreadyCompleted = false;
        bool receiverApproachAlreadyCompleted = false;
        bool receiverAlignmentAlreadyCompleted = false;
        std::optional<GraspReference> giverGraspReference;
        std::optional<GraspReference> receiverGraspReference;
        std::uint64_t preReceiverSuccessfulSteps = 0u;
        double preReceiverGpuMilliseconds = 0.0;
        std::uint64_t preReleaseSuccessfulSteps = 0u;
        double preReleaseGpuMilliseconds = 0.0;

        if (tissueMatterOnly) {
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                1u
            );
            metalrobo::MetalWorldStepConfig referenceConfig = stepConfig;
            referenceConfig.devicePhysicsProgram = {};
            metalrobo::MetalWorldContext referenceContext;
            metalrobo::MetalWorldResidentState referenceResident;
            const PhaseResult reference = initializePhase(
                referenceContext,
                compiled,
                world,
                referenceConfig,
                referenceResident,
                efforts,
                1u
            );
            const PhaseResult coupled = initializePhaseUnchecked(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                1u
            );
            if (!coupled.diagnostics.succeeded() ||
                coupled.diagnostics.failedStepCount != 0u) {
                std::string failure =
                    "coupled tissue transaction failed: " +
                    coupled.diagnostics.message;
                const numi::matter::RuntimeStateSnapshot failedMatter =
                    tissueRuntime.snapshot();
                if (failedMatter.available &&
                    !failedMatter.statuses.empty()) {
                    const NMMatterStatusGPU& status =
                        failedMatter.statuses[0];
                    failure += " matter_status=" +
                        std::to_string(status.code) +
                        " object=" +
                        std::to_string(status.objectIndex) +
                        " failing_index=" +
                        std::to_string(status.failingIndex) +
                        " microsteps=" +
                        std::to_string(status.completedMicrosteps) +
                        " fgmres=" +
                        std::to_string(status.fgmresIterations) +
                        " contacts=" +
                        std::to_string(status.contactCount) +
                        " matter_events=" +
                        std::to_string(status.eventCount) +
                        " diagnostics=(" +
                        std::to_string(status.diagnostics.x) + "," +
                        std::to_string(status.diagnostics.y) + "," +
                        std::to_string(status.diagnostics.z) + "," +
                        std::to_string(status.diagnostics.w) + ")";
                }
                if (!coupled.result.statuses.empty()) {
                    const MRMetalWorldStatusGPU& status =
                        coupled.result.statuses[0];
                    failure += " world_status=" +
                        std::to_string(status.code) +
                        " failing_substep=" +
                        std::to_string(status.failingSubstep) +
                        " failing_index=" +
                        std::to_string(status.failingIndex) +
                        " diagnostics=(" +
                        std::to_string(status.diagnostics.x) + "," +
                        std::to_string(status.diagnostics.y) + "," +
                        std::to_string(status.diagnostics.z) + "," +
                        std::to_string(status.diagnostics.w) + ")";
                }
                if (!coupled.result.environmentStatuses.empty()) {
                    const metalrobo::MetalWorldStatus& status =
                        coupled.result.environmentStatuses[0];
                    failure += " first_pair=" +
                        std::to_string(status.firstFailingPair) +
                        " first_constraint=" +
                        std::to_string(status.firstFailingConstraint) +
                        " maximum_residuals=(" +
                        std::to_string(status.maximumResiduals[0]) + "," +
                        std::to_string(status.maximumResiduals[1]) + "," +
                        std::to_string(status.maximumResiduals[2]) + "," +
                        std::to_string(status.maximumResiduals[3]) + ")";
                }
                std::uint32_t validContactSamples = 0u;
                double minimumCandidateSeparation =
                    std::numeric_limits<double>::infinity();
                double maximumCandidateHessian = 0.0;
                for (const NMContactSampleGPU& sample :
                     failedMatter.contactSamples) {
                    if ((sample.identity.w & NM_CONTACT_VALID) == 0u) {
                        continue;
                    }
                    ++validContactSamples;
                    minimumCandidateSeparation = std::min(
                        minimumCandidateSeparation,
                        static_cast<double>(sample.pointAndSeparation.w)
                    );
                    maximumCandidateHessian = std::max({
                        maximumCandidateHessian,
                        std::abs(static_cast<double>(
                            sample.barrierHessianRow0.x)),
                        std::abs(static_cast<double>(
                            sample.barrierHessianRow1.y)),
                        std::abs(static_cast<double>(
                            sample.barrierHessianRow2.z)),
                    });
                }
                failure += " candidate_contacts=" +
                    std::to_string(validContactSamples) +
                    " minimum_candidate_separation=" +
                    std::to_string(minimumCandidateSeparation) +
                    " maximum_candidate_hessian=" +
                    std::to_string(maximumCandidateHessian);
                if (!failedMatter.rigidStates.empty()) {
                    const NMRigidStateGPU& rigid =
                        failedMatter.rigidStates[0];
                    failure += " projected_capsule_first=" +
                        vectorSummary(vector(rigid.centerAndRadius)) +
                        " projected_capsule_second=" +
                        vectorSummary(vector(rigid.extent)) +
                        " projected_capsule_radius=" +
                        std::to_string(rigid.centerAndRadius.w);
                    double snapshotMinimumSeparation =
                        std::numeric_limits<double>::infinity();
                    std::uint32_t snapshotMinimumNode = NM_INVALID_INDEX;
                    for (const std::uint32_t node :
                         tissueCoupon.object.femContactNodes) {
                        if (node >= failedMatter.femNodes.size()) {
                            continue;
                        }
                        const double separation = pointSegmentDistance(
                            vector(failedMatter.femNodes[node]
                                       .positionAndMass),
                            vector(rigid.centerAndRadius),
                            vector(rigid.extent)
                        ) - rigid.centerAndRadius.w;
                        if (separation < snapshotMinimumSeparation) {
                            snapshotMinimumSeparation = separation;
                            snapshotMinimumNode = node;
                        }
                    }
                    failure += " snapshot_minimum_separation=" +
                        std::to_string(snapshotMinimumSeparation) +
                        " snapshot_minimum_node=" +
                        std::to_string(snapshotMinimumNode);
                    if (snapshotMinimumNode <
                        failedMatter.femNodes.size()) {
                        failure += " snapshot_minimum_inverse_mass=" +
                            std::to_string(
                                failedMatter.femNodes[snapshotMinimumNode]
                                    .velocityAndInverseMass.w
                            );
                    }
                }
                throw std::runtime_error(failure);
            }
            const numi::matter::RuntimeStateSnapshot snapshot =
                tissueRuntime.snapshot();
            require(
                snapshot.available && !snapshot.reactions.empty() &&
                    !snapshot.femNodes.empty() &&
                    !snapshot.solverCertificates.empty(),
                "tissue coupling did not publish accepted Matter state: " +
                    snapshot.message
            );

            std::uint32_t activeContacts = 0u;
            double normalImpulse = 0.0;
            double minimumContactSeparation =
                std::numeric_limits<double>::infinity();
            double minimumContactNormalVelocity =
                std::numeric_limits<double>::infinity();
            double minimumAdmissionNormalVelocity =
                std::numeric_limits<double>::infinity();
            for (const NMContactSampleGPU& sample :
                 snapshot.contactSamples) {
                if ((sample.identity.w & NM_CONTACT_VALID) == 0u) {
                    continue;
                }
                ++activeContacts;
                normalImpulse += sample.impulseAndNormal.w;
                minimumContactSeparation = std::min(
                    minimumContactSeparation,
                    static_cast<double>(sample.pointAndSeparation.w)
                );
                minimumContactNormalVelocity = std::min(
                    minimumContactNormalVelocity,
                    static_cast<double>(sample.normalAndVelocity.w)
                );
                minimumAdmissionNormalVelocity = std::min(
                    minimumAdmissionNormalVelocity,
                    static_cast<double>(
                        sample.admissionVelocityAndNormal.w
                    )
                );
            }
            const NMRigidReactionGPU& reaction = snapshot.reactions[0];
            const Vec3 reactionImpulse{
                reaction.impulseAndCount.x,
                reaction.impulseAndCount.y,
                reaction.impulseAndCount.z,
            };
            const Vec3 referenceNeedleVelocity = vector(
                reference.result.finalSceneBodies.at(0u)
                    .linearVelocityAndInverseMass
            );
            const Vec3 coupledNeedleVelocity = vector(
                coupled.result.finalSceneBodies.at(0u)
                    .linearVelocityAndInverseMass
            );
            const double needleVelocityDelta = norm(
                coupledNeedleVelocity - referenceNeedleVelocity
            );
            const Vec3 referenceRootVelocity = vector(
                reference.result.finalRodNodes.at(0u).velocity
            );
            const Vec3 coupledRootVelocity = vector(
                coupled.result.finalRodNodes.at(0u).velocity
            );
            const double threadRootVelocityDelta = norm(
                coupledRootVelocity - referenceRootVelocity
            );
            double maximumTissueDisplacement = 0.0;
            double maximumTissuePositionUlp = 0.0;
            // Static equilibrium is bounded by one representable FP32 step
            // per authored coordinate. This rejects physical drift without
            // mistaking a round-trip through device storage for motion.
            const auto coordinateUlp = [](const float value) {
                return std::max(
                    std::abs(static_cast<double>(std::nextafter(
                        value,
                        std::numeric_limits<float>::infinity()
                    )) - static_cast<double>(value)),
                    std::abs(static_cast<double>(std::nextafter(
                        value,
                        -std::numeric_limits<float>::infinity()
                    )) - static_cast<double>(value))
                );
            };
            for (std::size_t node = 0u;
                 node < snapshot.femNodes.size(); ++node) {
                const Vec3 accepted = vector(
                    snapshot.femNodes[node].positionAndMass
                );
                const auto& initialState =
                    tissueWorld.fem.nodes.at(node).positionAndMass;
                const Vec3 initial = vector(initialState);
                maximumTissueDisplacement = std::max(
                    maximumTissueDisplacement,
                    norm(accepted - initial)
                );
                maximumTissuePositionUlp = std::max(
                    maximumTissuePositionUlp,
                    norm(Vec3{
                        coordinateUlp(initialState.x),
                        coordinateUlp(initialState.y),
                        coordinateUlp(initialState.z),
                    })
                );
            }
            double maximumNonlinearResidual = 0.0;
            double maximumRelativeCorrection = 0.0;
            double maximumVolumeResidual = 0.0;
            double minimumDeterminant =
                std::numeric_limits<double>::infinity();
            bool certificatesAccepted = true;
            std::uint32_t maximumFGMRESIterations = 0u;
            for (const NMMatterStatusGPU& status : snapshot.statuses) {
                maximumFGMRESIterations = std::max(
                    maximumFGMRESIterations,
                    status.fgmresIterations
                );
            }
            for (const NMSolverCertificateGPU& certificate :
                 snapshot.solverCertificates) {
                maximumNonlinearResidual = std::max(
                    maximumNonlinearResidual,
                    static_cast<double>(certificate.nonlinear.x)
                );
                maximumVolumeResidual = std::max(
                    maximumVolumeResidual,
                    static_cast<double>(certificate.nonlinear.z)
                );
                maximumRelativeCorrection = std::max(
                    maximumRelativeCorrection,
                    static_cast<double>(certificate.nonlinear.y)
                );
                minimumDeterminant = std::min(
                    minimumDeterminant,
                    static_cast<double>(certificate.validity.x)
                );
                certificatesAccepted = certificatesAccepted &&
                    certificate.validity.w > 0.5f;
            }
            std::uint32_t activeChannels = 0u;
            std::uint32_t activeTetrahedra = 0u;
            double removedMassKg = 0.0;
            const NMPunctureChannelGPU* acceptedChannel = nullptr;
            for (const NMPunctureChannelGPU& channel :
                 snapshot.punctureChannels) {
                if ((channel.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) {
                    continue;
                }
                ++activeChannels;
                acceptedChannel = &channel;
            }
            for (const NMTetrahedronGPU& tetrahedron :
                 snapshot.femTopologyTetrahedra) {
                activeTetrahedra +=
                    (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
            }
            for (const NMFEMTopologyStateGPU& topology :
                 snapshot.topologyStates) {
                removedMassKg += topology.accounting.y;
            }
            const NeedleTipCapsuleGeometry initialTip =
                needleTipCapsuleGeometry(
                    needleForPlacement,
                    world.defaultSceneBodies[0]
                );
            const double expectedEntryTractLengthM =
                2.0 * initialTip.radiusM;
            const double analyticTipTractMassKg =
                tissueCoupon.spec.densityKgPerM3.value *
                std::numbers::pi * initialTip.radiusM *
                initialTip.radiusM * expectedEntryTractLengthM;
            const double removedToTipTractMassRatio =
                analyticTipTractMassKg > 0.0
                ? removedMassKg / analyticTipTractMassKg
                : std::numeric_limits<double>::infinity();
            double channelRadiusM = 0.0;
            double channelLengthM = 0.0;
            double channelAxisAlignment = 0.0;
            std::uint32_t channelReleaseContacts = 0u;
            std::uint32_t channelReleaseChannels = 0u;
            std::uint32_t channelReleaseLinks = 0u;
            double channelReleaseTotalLengthM = 0.0;
            double channelReleaseSignedNeedleMotionM = 0.0;
            double channelReleaseMinimumDeterminant = 0.0;
            double channelReleaseMaximumResidual = 0.0;
            double channelReleaseSwageErrorM = 0.0;
            std::uint64_t channelReleaseStateHash = 0u;
            if (acceptedChannel != nullptr) {
                channelRadiusM = acceptedChannel->originAndRadius.w;
                channelLengthM =
                    2.0 * acceptedChannel->axisAndHalfLength.w;
                channelAxisAlignment = std::abs(dot(
                    vector(acceptedChannel->axisAndHalfLength),
                    initialTip.approachDirection
                ));
                if (tissueCurvedPassageOnly &&
                    tissueNeedleOrbit.has_value()) {
                    const Vec3 axis = vector(
                        acceptedChannel->axisAndHalfLength
                    );
                    const Vec3 origin = vector(
                        acceptedChannel->originAndRadius
                    );
                    const Vec3 proximal = origin - axis *
                        acceptedChannel->axisAndHalfLength.w;
                    const Vec3 distal = origin + axis *
                        acceptedChannel->axisAndHalfLength.w;
                    std::cout << std::setprecision(9)
                        << "curved_passage_entry_proximal_orbit_error_m="
                        << norm(
                            proximal - tissueNeedleOrbit->centerWorld
                        ) - tissueNeedleOrbit->centerlineRadiusM
                        << " curved_passage_entry_distal_orbit_error_m="
                        << norm(
                            distal - tissueNeedleOrbit->centerWorld
                        ) - tissueNeedleOrbit->centerlineRadiusM
                        << '\n';
                }
            }
            const double swageError = swageAttachmentError(
                world,
                coupled.result
            );
            std::uint64_t acceptedStateHash = 1469598103934665603ull;
            appendStateHash(acceptedStateHash, coupled.result.finalQ);
            appendStateHash(acceptedStateHash, coupled.result.finalV);
            appendStateHash(
                acceptedStateHash, coupled.result.finalSceneBodies);
            appendStateHash(
                acceptedStateHash, coupled.result.finalRodNodes);
            appendStateHash(
                acceptedStateHash, coupled.result.finalRodEdges);
            appendStateHash(acceptedStateHash, snapshot.femNodes);
            appendStateHash(acceptedStateHash, snapshot.femFields);
            appendStateHash(
                acceptedStateHash, snapshot.femTopologyTetrahedra);
            appendStateHash(
                acceptedStateHash, snapshot.punctureChannels);
            appendStateHash(
                acceptedStateHash, snapshot.topologyStates);
            appendStateHash(acceptedStateHash, snapshot.reactions);
            appendStateHash(acceptedStateHash, snapshot.contactSamples);
            appendStateHash(
                acceptedStateHash, snapshot.solverCertificates);
            if (tissueRestOnly) {
                std::cout << std::setprecision(17)
                    << "tissue_static_control reaction_count="
                    << reaction.impulseAndCount.w
                    << " active_contacts=" << activeContacts
                    << " reaction_norm=" << norm(reactionImpulse)
                    << " needle_velocity_delta=" << needleVelocityDelta
                    << " thread_root_velocity_delta="
                    << threadRootVelocityDelta
                    << " maximum_tissue_displacement="
                    << maximumTissueDisplacement
                    << " maximum_tissue_position_ulp="
                    << maximumTissuePositionUlp
                    << " certificates_accepted="
                    << certificatesAccepted
                    << " minimum_determinant=" << minimumDeterminant
                    << " swage_error=" << swageError
                    << " relative_correction="
                    << maximumRelativeCorrection << '\n';
                require(
                    reaction.impulseAndCount.w == 0.0f &&
                        activeContacts == 0u &&
                        norm(reactionImpulse) == 0.0 &&
                        needleVelocityDelta == 0.0 &&
                        threadRootVelocityDelta == 0.0 &&
                        maximumTissueDisplacement <=
                            maximumTissuePositionUlp &&
                        certificatesAccepted &&
                        std::isfinite(maximumRelativeCorrection) &&
                        std::isfinite(minimumDeterminant) &&
                        minimumDeterminant > 0.0 &&
                        swageError < kMaximumSwageAttachmentError,
                    "contact-free tissue rest state produced physical motion "
                    "or nonfinite telemetry: reaction_contacts=" +
                        std::to_string(reaction.impulseAndCount.w) +
                        " active_contacts=" +
                        std::to_string(activeContacts) +
                        " reaction_impulse=" +
                        vectorSummary(reactionImpulse) +
                        " needle_velocity_delta=" +
                        std::to_string(needleVelocityDelta) +
                        " thread_root_velocity_delta=" +
                        std::to_string(threadRootVelocityDelta) +
                        " maximum_tissue_displacement=" +
                        std::to_string(maximumTissueDisplacement) +
                        " maximum_tissue_position_ulp=" +
                        std::to_string(maximumTissuePositionUlp) +
                        " certificates_accepted=" +
                        std::to_string(certificatesAccepted) +
                        " minimum_determinant=" +
                        std::to_string(minimumDeterminant) +
                        " swage_error=" +
                        std::to_string(swageError) +
                        " relative_correction=" +
                        std::to_string(maximumRelativeCorrection)
                );
            } else if (tissuePunctureOnly) {
                require(
                    tissueWorld.contact.rigidProxies.size() ==
                        (tissueCurvedPassageOnly
                            ? kCurvedPassageContactSegmentCount : 1u) &&
                        (tissueWorld.contact.rigidProxies[0].flags &
                         NM_RIGID_PUNCTURE_TIP) != 0u &&
                        activeChannels == 1u &&
                        acceptedChannel != nullptr &&
                        activeTetrahedra ==
                            tissueCoupon.metadata.tetrahedronCount &&
                        removedMassKg == 0.0 &&
                        std::isfinite(removedToTipTractMassRatio) &&
                        removedToTipTractMassRatio == 0.0 &&
                        channelRadiusM > 0.0 &&
                        std::abs(
                            channelRadiusM - initialTip.radiusM
                        ) <= 2.0e-7 &&
                        std::abs(
                            channelLengthM - expectedEntryTractLengthM
                        ) <= 2.0e-7 &&
                        channelAxisAlignment >= 0.999 &&
                        activeContacts >= 1u &&
                        normalImpulse >= kPunctureImpulseThresholdNs &&
                        std::isfinite(minimumContactSeparation) &&
                        minimumContactSeparation > 0.0 &&
                        std::isfinite(minimumAdmissionNormalVelocity) &&
                        minimumAdmissionNormalVelocity < -1.0e-6 &&
                        certificatesAccepted &&
                        std::isfinite(minimumDeterminant) &&
                        minimumDeterminant > 0.0 &&
                        swageError < kMaximumSwageAttachmentError,
                    "tapered needle entry did not create one accepted, "
                    "mass-conserving geometry-matched tissue tract: "
                    "active_channels=" +
                        std::to_string(activeChannels) +
                        " active_tetrahedra=" +
                        std::to_string(activeTetrahedra) +
                        " authored_tetrahedra=" +
                        std::to_string(
                            tissueCoupon.metadata.tetrahedronCount
                        ) +
                        " removed_mass=" +
                        std::to_string(removedMassKg) +
                        " active_contacts=" +
                        std::to_string(activeContacts) +
                        " normal_impulse=" +
                        std::to_string(normalImpulse) +
                        " minimum_contact_separation=" +
                        std::to_string(minimumContactSeparation) +
                        " minimum_contact_normal_velocity=" +
                        std::to_string(minimumContactNormalVelocity) +
                        " minimum_admission_normal_velocity=" +
                        std::to_string(minimumAdmissionNormalVelocity) +
                        " tract_mass_ratio=" +
                        std::to_string(removedToTipTractMassRatio) +
                        " channel_radius=" +
                        std::to_string(channelRadiusM) +
                        " expected_radius=" +
                        std::to_string(initialTip.radiusM) +
                        " channel_length=" +
                        std::to_string(channelLengthM) +
                        " expected_length=" +
                        std::to_string(expectedEntryTractLengthM) +
                        " axis_alignment=" +
                        std::to_string(channelAxisAlignment) +
                        " minimum_determinant=" +
                        std::to_string(minimumDeterminant) +
                        " swage_error=" + std::to_string(swageError)
                );
                const NeedleTipCapsuleGeometry entryTip =
                    needleTipCapsuleGeometry(
                        needleForPlacement,
                        coupled.result.finalSceneBodies.at(0u)
                    );
                if (tissueCurvedPassageOnly) {
                    require(
                        tissueNeedleOrbit.has_value() &&
                            tissueNeedleAngularSpeedRadPerS > 0.0,
                        "curved passage lost its authored needle orbit"
                    );
                    Vec3 thicknessAxis = vector(
                        tissueCoupon.metadata.thicknessAxis
                    );
                    const double thicknessAxisLength = norm(thicknessAxis);
                    require(
                        thicknessAxisLength > 1.0e-12 &&
                            std::isfinite(thicknessAxisLength),
                        "curved passage tissue thickness axis is invalid"
                    );
                    thicknessAxis = thicknessAxis *
                        (1.0 / thicknessAxisLength);
                    double authoredTopProjection =
                        -std::numeric_limits<double>::infinity();
                    double authoredBottomProjection =
                        std::numeric_limits<double>::infinity();
                    for (const auto& authoredPosition :
                         tissueCoupon.object.femNodes) {
                        const double projection = dot(
                            vector(authoredPosition),
                            thicknessAxis
                        );
                        authoredTopProjection = std::max(
                            authoredTopProjection,
                            projection
                        );
                        authoredBottomProjection = std::min(
                            authoredBottomProjection,
                            projection
                        );
                    }
                    const double authoredWallThicknessM =
                        authoredTopProjection - authoredBottomProjection;
                    require(
                        std::isfinite(authoredWallThicknessM) &&
                            std::abs(
                                authoredWallThicknessM -
                                tissueCoupon.spec.thicknessM.value
                            ) <= 2.0e-7,
                        "curved passage wall thickness changed during cooking"
                    );
                    const double requiredTipAdvanceM =
                        authoredWallThicknessM +
                        2.0 * (
                            initialTip.radiusM +
                            kCurvedPassageExitClearanceM
                        );
                    const double requiredSine = requiredTipAdvanceM /
                        tissueNeedleOrbit->centerlineRadiusM;
                    require(
                        requiredSine > 0.0 && requiredSine < 0.5,
                        "curved passage exceeds the needle orbit envelope"
                    );
                    const double targetPassageAngleRad =
                        std::asin(requiredSine);
                    const double microstepSeconds =
                        static_cast<double>(stepConfig.timestepSeconds);
                    const double anglePerMicrostepRad =
                        tissueNeedleAngularSpeedRadPerS * microstepSeconds;
                    const std::uint32_t minimumPassageSteps =
                        static_cast<std::uint32_t>(std::ceil(
                            targetPassageAngleRad /
                            anglePerMicrostepRad
                        )) + 4u;
                    const std::uint32_t maximumExtensionSteps =
                        static_cast<std::uint32_t>(std::ceil(
                            kCurvedPassageMaximumExtensionM /
                            (kCurvedPassageSpeedMps * microstepSeconds)
                        ));
                    const std::uint32_t maximumPassageSteps =
                        minimumPassageSteps + maximumExtensionSteps;
                    require(
                        minimumPassageSteps > 1u &&
                            maximumPassageSteps > minimumPassageSteps &&
                            maximumPassageSteps < 4096u,
                        "curved passage step count is invalid"
                    );

                    PhaseResult passage = coupled;
                    std::uint32_t completedPassageSteps = 1u;
                    std::uint32_t totalPassageSteps = 0u;
                    double passageGpuMilliseconds =
                        coupled.diagnostics.gpuElapsedMilliseconds;
                    bool measuredExitReached = false;
                    while (completedPassageSteps < maximumPassageSteps) {
                        std::uint32_t chunkSteps = std::min(
                            kCurvedPassageChunkSteps,
                            maximumPassageSteps - completedPassageSteps
                        );
                        if (completedPassageSteps < minimumPassageSteps) {
                            chunkSteps = std::min(
                                chunkSteps,
                                minimumPassageSteps -
                                    completedPassageSteps
                            );
                        }
                        std::vector<MRBodyStateGPU> kinematicTargets;
                        kinematicTargets.reserve(
                            static_cast<std::size_t>(chunkSteps) *
                            compiled.sceneBodyCount()
                        );
                        for (std::uint32_t localStep = 0u;
                             localStep < chunkSteps;
                             ++localStep) {
                            const double angle = anglePerMicrostepRad *
                                static_cast<double>(
                                    completedPassageSteps + localStep
                                );
                            for (std::size_t sceneBody = 0u;
                                 sceneBody < compiled.sceneBodyCount();
                                 ++sceneBody) {
                                kinematicTargets.push_back(
                                    sceneBody == 0u
                                        ? curvedNeedleTarget(
                                            world.defaultSceneBodies[0],
                                            *tissueNeedleOrbit,
                                            angle,
                                            tissueNeedleAngularSpeedRadPerS
                                        )
                                        : world.defaultSceneBodies.at(
                                            sceneBody)
                                );
                            }
                        }
                        const std::vector<float> passageEfforts =
                            interpolateTargets(
                                world.model,
                                targetStart,
                                targetStart,
                                chunkSteps
                            );
                        passage = continuePhaseWithKinematicTargets(
                            context,
                            compiled,
                            stepConfig,
                            resident,
                            passageEfforts,
                            kinematicTargets,
                            chunkSteps,
                            "curvature-following tissue passage"
                        );
                        completedPassageSteps += chunkSteps;
                        passageGpuMilliseconds +=
                            passage.diagnostics.gpuElapsedMilliseconds;
                        const NeedleTipCapsuleGeometry progressTip =
                            needleTipCapsuleGeometry(
                                needleForPlacement,
                                passage.result.finalSceneBodies.at(0u)
                            );
                        std::cout << std::setprecision(9)
                            << "curved_passage_progress_steps="
                            << completedPassageSteps << '/'
                            << minimumPassageSteps
                            << " curved_passage_progress_m="
                            << dot(
                                progressTip.worldTip - initialTip.worldTip,
                                initialTip.approachDirection
                            )
                            << " chunk_gpu_ms="
                            << passage.diagnostics.gpuElapsedMilliseconds
                            << '\n';
                        if (completedPassageSteps < minimumPassageSteps) {
                            continue;
                        }
                        const numi::matter::RuntimeStateSnapshot
                            progressSnapshot = tissueRuntime.snapshot();
                        require(
                            progressSnapshot.available &&
                                tissueCoupon.metadata.nodeCount <=
                                    progressSnapshot.femNodes.size(),
                            "curved passage could not inspect distal clearance"
                        );
                        double progressBottomProjection =
                            std::numeric_limits<double>::infinity();
                        for (std::size_t nodeIndex = 0u;
                             nodeIndex < tissueCoupon.metadata.nodeCount;
                             ++nodeIndex) {
                            progressBottomProjection = std::min(
                                progressBottomProjection,
                                dot(
                                    vector(
                                        progressSnapshot.femNodes[nodeIndex]
                                            .positionAndMass
                                    ),
                                    thicknessAxis
                                )
                            );
                        }
                        const double progressDistalClearanceM =
                            progressBottomProjection -
                            dot(progressTip.worldTip, thicknessAxis) -
                            initialTip.radiusM;
                        std::cout << std::setprecision(9)
                            << "curved_passage_measured_distal_clearance_m="
                            << progressDistalClearanceM
                            << " curved_passage_extension_steps="
                            << completedPassageSteps -
                                minimumPassageSteps << '\n';
                        if (progressDistalClearanceM >=
                            kCurvedPassageExitClearanceM) {
                            measuredExitReached = true;
                            break;
                        }
                    }
                    totalPassageSteps = completedPassageSteps;
                    require(
                        measuredExitReached,
                        "curved passage exhausted its bounded extension before "
                        "the deformed distal surface cleared the needle tip"
                    );

                    const numi::matter::RuntimeStateSnapshot passageSnapshot =
                        tissueRuntime.snapshot();
                    require(
                        passageSnapshot.available &&
                            !passageSnapshot.femNodes.empty() &&
                            !passageSnapshot.punctureChannels.empty() &&
                            !passageSnapshot.solverCertificates.empty(),
                        "curved passage did not publish accepted Matter state"
                    );
                    std::vector<NMPunctureChannelGPU> passageChannels;
                    std::uint32_t passageContacts = 0u;
                    std::uint32_t passageTetrahedra = 0u;
                    double passageRemovedMassKg = 0.0;
                    double finalBottomProjection =
                        std::numeric_limits<double>::infinity();
                    double maximumPassageTissueDisplacementM = 0.0;
                    require(
                        tissueCoupon.metadata.nodeCount <=
                            passageSnapshot.femNodes.size() &&
                            tissueCoupon.metadata.nodeCount <=
                                tissueWorld.fem.nodes.size(),
                        "curved passage lost authored FEM nodes"
                    );
                    for (std::size_t nodeIndex = 0u;
                         nodeIndex < tissueCoupon.metadata.nodeCount;
                         ++nodeIndex) {
                        const Vec3 finalPosition = vector(
                            passageSnapshot.femNodes[nodeIndex]
                                .positionAndMass
                        );
                        finalBottomProjection = std::min(
                            finalBottomProjection,
                            dot(finalPosition, thicknessAxis)
                        );
                        maximumPassageTissueDisplacementM = std::max(
                            maximumPassageTissueDisplacementM,
                            norm(
                                finalPosition -
                                vector(
                                    tissueWorld.fem.nodes.at(nodeIndex)
                                        .positionAndMass
                                )
                            )
                        );
                    }
                    for (const NMContactSampleGPU& sample :
                         passageSnapshot.contactSamples) {
                        passageContacts +=
                            (sample.identity.w & NM_CONTACT_VALID) != 0u;
                    }
                    for (const NMPunctureChannelGPU& channel :
                         passageSnapshot.punctureChannels) {
                        if ((channel.identity.w & NM_TOPOLOGY_ACTIVE) != 0u) {
                            passageChannels.push_back(channel);
                        }
                    }
                    for (const NMTetrahedronGPU& tetrahedron :
                         passageSnapshot.femTopologyTetrahedra) {
                        passageTetrahedra +=
                            (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
                    }
                    for (const NMFEMTopologyStateGPU& topology :
                         passageSnapshot.topologyStates) {
                        passageRemovedMassKg += topology.accounting.y;
                    }
                    std::sort(
                        passageChannels.begin(),
                        passageChannels.end(),
                        [](const NMPunctureChannelGPU& left,
                           const NMPunctureChannelGPU& right) {
                            return left.identity.z < right.identity.z;
                        }
                    );
                    double channelMinimumProjection =
                        std::numeric_limits<double>::infinity();
                    double channelMaximumProjection =
                        -std::numeric_limits<double>::infinity();
                    double channelMaximumJoinGapM = 0.0;
                    double channelMinimumTangentAlignment = 1.0;
                    double channelMaximumOrbitErrorM = 0.0;
                    double channelMinimumSignedOrbitErrorM =
                        std::numeric_limits<double>::infinity();
                    double channelMaximumSignedOrbitErrorM =
                        -std::numeric_limits<double>::infinity();
                    double channelTotalLengthM = 0.0;
                    bool channelGeometryValid = true;
                    std::uint32_t channelLinks = 0u;
                    Vec3 previousDistal{};
                    Vec3 previousAxis{};
                    for (std::size_t channelIndex = 0u;
                         channelIndex < passageChannels.size();
                         ++channelIndex) {
                        const NMPunctureChannelGPU& channel =
                            passageChannels[channelIndex];
                        const Vec3 rawAxis = vector(
                            channel.axisAndHalfLength
                        );
                        const double axisLength = norm(rawAxis);
                        const double halfLength =
                            channel.axisAndHalfLength.w;
                        const double radius = channel.originAndRadius.w;
                        channelGeometryValid = channelGeometryValid &&
                            channel.identity.x == 0u &&
                            channel.identity.y == 0x80000000u &&
                            channel.identity.z != 0u &&
                            axisLength > 1.0e-12 &&
                            std::abs(axisLength - 1.0) <= 1.0e-5 &&
                            std::abs(radius - initialTip.radiusM) <= 2.0e-7 &&
                            std::abs(
                                2.0 * halfLength -
                                expectedEntryTractLengthM
                            ) <= 2.0e-7;
                        if (!(axisLength > 1.0e-12) ||
                            !(halfLength > 0.0)) {
                            continue;
                        }
                        const Vec3 axis = rawAxis * (1.0 / axisLength);
                        const Vec3 origin = vector(
                            channel.originAndRadius
                        );
                        const Vec3 proximal = origin - axis * halfLength;
                        const Vec3 distal = origin + axis * halfLength;
                        channelTotalLengthM += 2.0 * halfLength;
                        for (const Vec3 endpoint : {proximal, distal}) {
                            const double projection = dot(
                                endpoint,
                                thicknessAxis
                            );
                            channelMinimumProjection = std::min(
                                channelMinimumProjection,
                                projection
                            );
                            channelMaximumProjection = std::max(
                                channelMaximumProjection,
                                projection
                            );
                            const double signedOrbitErrorM =
                                norm(
                                    endpoint -
                                    tissueNeedleOrbit->centerWorld
                                ) -
                                tissueNeedleOrbit->centerlineRadiusM;
                            channelMaximumOrbitErrorM = std::max(
                                channelMaximumOrbitErrorM,
                                std::abs(signedOrbitErrorM)
                            );
                            channelMinimumSignedOrbitErrorM = std::min(
                                channelMinimumSignedOrbitErrorM,
                                signedOrbitErrorM
                            );
                            channelMaximumSignedOrbitErrorM = std::max(
                                channelMaximumSignedOrbitErrorM,
                                signedOrbitErrorM
                            );
                        }
                        if (channelIndex != 0u) {
                            const double joinGap = norm(
                                proximal - previousDistal
                            );
                            channelMaximumJoinGapM = std::max(
                                channelMaximumJoinGapM,
                                joinGap
                            );
                            channelMinimumTangentAlignment = std::min(
                                channelMinimumTangentAlignment,
                                dot(previousAxis, axis)
                            );
                            channelGeometryValid = channelGeometryValid &&
                                channel.identity.z >
                                    passageChannels[channelIndex - 1u]
                                        .identity.z &&
                                dot(
                                    cross(previousAxis, axis),
                                    tissueNeedleOrbit->axisWorld
                                ) >= -1.0e-5;
                            channelLinks +=
                                joinGap <= 0.25 * initialTip.radiusM;
                        }
                        previousDistal = distal;
                        previousAxis = axis;
                    }
                    bool passageCertificatesAccepted = true;
                    std::uint32_t passageMaximumFGMRESIterations = 0u;
                    for (const NMMatterStatusGPU& status :
                         passageSnapshot.statuses) {
                        passageMaximumFGMRESIterations = std::max(
                            passageMaximumFGMRESIterations,
                            status.fgmresIterations
                        );
                    }
                    double passageMinimumDeterminant =
                        std::numeric_limits<double>::infinity();
                    double passageMaximumResidual = 0.0;
                    for (const NMSolverCertificateGPU& certificate :
                         passageSnapshot.solverCertificates) {
                        passageCertificatesAccepted =
                            passageCertificatesAccepted &&
                            certificate.validity.w > 0.5f;
                        passageMinimumDeterminant = std::min(
                            passageMinimumDeterminant,
                            static_cast<double>(certificate.validity.x)
                        );
                        passageMaximumResidual = std::max({
                            passageMaximumResidual,
                            static_cast<double>(certificate.nonlinear.x),
                            static_cast<double>(certificate.nonlinear.z),
                            static_cast<double>(certificate.nonlinear.w),
                        });
                    }
                    const NeedleTipCapsuleGeometry exitTip =
                        needleTipCapsuleGeometry(
                            needleForPlacement,
                            passage.result.finalSceneBodies.at(0u)
                        );
                    const MRBodyStateGPU& exitNeedle =
                        passage.result.finalSceneBodies.at(0u);
                    const Vec3 exitTerminalVelocity =
                        vector(exitNeedle.linearVelocityAndInverseMass) +
                        cross(
                            vector(exitNeedle.angularVelocity),
                            exitTip.worldTip - vector(exitNeedle.position)
                        );
                    const double exitTipAdvanceM = dot(
                        exitTip.worldTip - initialTip.worldTip,
                        initialTip.approachDirection
                    );
                    const double distalSurfaceClearanceM =
                        finalBottomProjection -
                        dot(exitTip.worldTip, thicknessAxis) -
                        initialTip.radiusM;
                    const double exitOrbitErrorM = std::abs(
                        norm(
                            exitTip.worldTip -
                            tissueNeedleOrbit->centerWorld
                        ) - tissueNeedleOrbit->centerlineRadiusM
                    );
                    const double passageSwageErrorM = swageAttachmentError(
                        world,
                        passage.result
                    );
                    const RodStateMetrics passageRod = rodStateMetrics(
                        world,
                        passage.result
                    );
                    std::uint64_t passageStateHash =
                        1469598103934665603ull;
                    appendStateHash(passageStateHash, passage.result.finalQ);
                    appendStateHash(passageStateHash, passage.result.finalV);
                    appendStateHash(
                        passageStateHash,
                        passage.result.finalSceneBodies
                    );
                    appendStateHash(
                        passageStateHash,
                        passage.result.finalRodNodes
                    );
                    appendStateHash(
                        passageStateHash,
                        passage.result.finalRodEdges
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.femNodes
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.femFields
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.femTopologyTetrahedra
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.punctureChannels
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.topologyStates
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.reactions
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.contactSamples
                    );
                    appendStateHash(
                        passageStateHash,
                        passageSnapshot.solverCertificates
                    );
                    const std::uint32_t minimumWallChannels =
                        static_cast<std::uint32_t>(std::ceil(
                            authoredWallThicknessM /
                            expectedEntryTractLengthM
                        ));
                    require(
                        passageChannels.size() >= minimumWallChannels &&
                            channelLinks + 1u == passageChannels.size() &&
                            channelGeometryValid &&
                            channelMinimumTangentAlignment >= 0.99 &&
                            channelMaximumJoinGapM <=
                                0.25 * initialTip.radiusM &&
                            channelMaximumOrbitErrorM <=
                                initialTip.radiusM &&
                            channelMaximumSignedOrbitErrorM -
                                channelMinimumSignedOrbitErrorM <=
                                0.25 * initialTip.radiusM &&
                            channelMaximumProjection >=
                                authoredTopProjection -
                                0.25 * initialTip.radiusM &&
                            channelMinimumProjection <=
                                authoredBottomProjection +
                                0.25 * initialTip.radiusM &&
                            channelTotalLengthM >= authoredWallThicknessM &&
                            passageTetrahedra ==
                                tissueCoupon.metadata.tetrahedronCount &&
                            passageRemovedMassKg == 0.0 &&
                            passageCertificatesAccepted &&
                            std::isfinite(passageMinimumDeterminant) &&
                            passageMinimumDeterminant > 0.0 &&
                            std::isfinite(passageMaximumResidual) &&
                            exitTipAdvanceM >= requiredTipAdvanceM &&
                            distalSurfaceClearanceM >=
                                kCurvedPassageExitClearanceM &&
                            exitOrbitErrorM <= 2.0e-6 &&
                            std::abs(
                                norm(exitTerminalVelocity) -
                                kCurvedPassageSpeedMps
                            ) <= 2.0e-5 &&
                            dot(
                                exitTerminalVelocity,
                                exitTip.approachDirection
                            ) >= 0.999 * kCurvedPassageSpeedMps &&
                            passageSwageErrorM <
                                kMaximumSwageAttachmentError &&
                            qualifiedTransitionRod(passageRod),
                        "curvature-following needle did not form a connected "
                        "mass-conserving through-wall tract: channels=" +
                            std::to_string(passageChannels.size()) +
                            " links=" + std::to_string(channelLinks) +
                            " channel_min=" +
                            std::to_string(channelMinimumProjection) +
                            " authored_bottom=" +
                            std::to_string(authoredBottomProjection) +
                            " distal_clearance=" +
                            std::to_string(distalSurfaceClearanceM) +
                            " tip_advance=" +
                            std::to_string(exitTipAdvanceM) +
                            " required_advance=" +
                            std::to_string(requiredTipAdvanceM) +
                            " max_join_gap=" +
                            std::to_string(channelMaximumJoinGapM) +
                            " min_tangent_alignment=" +
                            std::to_string(
                                channelMinimumTangentAlignment
                            ) +
                            " max_orbit_error=" +
                            std::to_string(channelMaximumOrbitErrorM) +
                            " orbit_error_range=" +
                            std::to_string(
                                channelMaximumSignedOrbitErrorM -
                                channelMinimumSignedOrbitErrorM
                            ) +
                            " active_contacts=" +
                            std::to_string(passageContacts) +
                            " removed_mass=" +
                            std::to_string(passageRemovedMassKg) +
                            " minimum_determinant=" +
                            std::to_string(passageMinimumDeterminant) +
                            " maximum_residual=" +
                            std::to_string(passageMaximumResidual) +
                            " swage_error=" +
                            std::to_string(passageSwageErrorM) +
                            " rod_edge_error=" +
                            std::to_string(
                                passageRod.maximumEdgeLengthError
                            ) +
                            " rod_clearance=" +
                            std::to_string(
                                passageRod
                                    .minimumNonNeighbourSurfaceClearance
                            )
                    );
                    std::cout << std::setprecision(9)
                        << "tissue_curved_through_wall_passage=ok"
                        << " passage_steps=" << totalPassageSteps
                        << " needle_arc_angle_rad="
                        << anglePerMicrostepRad * totalPassageSteps
                        << " needle_tip_advance_m=" << exitTipAdvanceM
                        << " distal_surface_clearance_m="
                        << distalSurfaceClearanceM
                        << " active_puncture_channels="
                        << passageChannels.size()
                        << " puncture_channel_links=" << channelLinks
                        << " puncture_channel_total_length_m="
                        << channelTotalLengthM
                        << " puncture_channel_max_join_gap_m="
                        << channelMaximumJoinGapM
                        << " puncture_channel_min_tangent_alignment="
                        << channelMinimumTangentAlignment
                        << " puncture_channel_max_orbit_error_m="
                        << channelMaximumOrbitErrorM
                        << " puncture_channel_orbit_error_range_m="
                        << channelMaximumSignedOrbitErrorM -
                            channelMinimumSignedOrbitErrorM
                        << " active_contacts=" << passageContacts
                        << " active_tetrahedra=" << passageTetrahedra
                        << " removed_tissue_mass_kg="
                        << passageRemovedMassKg
                        << " maximum_tissue_displacement_m="
                        << maximumPassageTissueDisplacementM
                        << " matter_minimum_determinant="
                        << passageMinimumDeterminant
                        << " matter_maximum_residual="
                        << passageMaximumResidual
                        << " matter_maximum_fgmres_iterations="
                        << passageMaximumFGMRESIterations
                        << " terminal_tip_speed_mps="
                        << norm(exitTerminalVelocity)
                        << " hard_swage_root_error_m="
                        << passageSwageErrorM
                        << " thread_maximum_edge_error_m="
                        << passageRod.maximumEdgeLengthError
                        << " thread_minimum_clearance_m="
                        << passageRod.minimumNonNeighbourSurfaceClearance
                        << " passage_gpu_ms=" << passageGpuMilliseconds
                        << " passage_state_fnv64=0x" << std::hex
                        << passageStateHash << std::dec
                        << " failed_steps="
                        << passage.diagnostics.failedStepCount << '\n';
                    return 0;
                }
                const std::uint32_t channelReleaseSteps =
                    tissuePunctureAdvanceOnly
                        ? kPunctureChannelProbeSteps : 1u;
                const std::vector<float> channelReleaseEfforts =
                    tissuePunctureAdvanceOnly
                        ? interpolateTargets(
                            world.model,
                            targetStart,
                            targetStart,
                            channelReleaseSteps
                        )
                        : efforts;
                const PhaseResult channelRelease = continuePhase(
                    context,
                    compiled,
                    stepConfig,
                    resident,
                    channelReleaseEfforts,
                    channelReleaseSteps,
                    "embedded puncture-channel release"
                );
                const numi::matter::RuntimeStateSnapshot releaseSnapshot =
                    tissueRuntime.snapshot();
                require(
                    releaseSnapshot.available &&
                        !releaseSnapshot.femNodes.empty(),
                    "puncture-channel release did not publish Matter state"
                );
                std::uint32_t releaseTetrahedra = 0u;
                double releaseRemovedMassKg = 0.0;
                std::vector<NMPunctureChannelGPU> releaseActiveChannels;
                for (const NMContactSampleGPU& sample :
                     releaseSnapshot.contactSamples) {
                    channelReleaseContacts +=
                        (sample.identity.w & NM_CONTACT_VALID) != 0u;
                }
                for (const NMPunctureChannelGPU& channel :
                     releaseSnapshot.punctureChannels) {
                    if ((channel.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) {
                        continue;
                    }
                    ++channelReleaseChannels;
                    channelReleaseTotalLengthM +=
                        2.0 * channel.axisAndHalfLength.w;
                    releaseActiveChannels.push_back(channel);
                }
                std::set<std::uint32_t> releaseGenerations;
                bool releaseChannelGeometryValid = true;
                for (const NMPunctureChannelGPU& channel :
                     releaseActiveChannels) {
                    releaseGenerations.insert(channel.identity.z);
                    releaseChannelGeometryValid =
                        releaseChannelGeometryValid &&
                        channel.identity.x == 0u &&
                        channel.identity.y == 0x80000000u &&
                        channel.identity.z != 0u &&
                        std::abs(
                            norm(vector(channel.axisAndHalfLength)) - 1.0
                        ) <= 1.0e-5 &&
                        std::abs(
                            static_cast<double>(
                                channel.originAndRadius.w) -
                            initialTip.radiusM
                        ) <= 2.0e-7 &&
                        std::abs(
                            2.0 * static_cast<double>(
                                channel.axisAndHalfLength.w) -
                            expectedEntryTractLengthM
                        ) <= 2.0e-7;
                }
                for (std::size_t firstIndex = 0u;
                     firstIndex < releaseActiveChannels.size();
                     ++firstIndex) {
                    const NMPunctureChannelGPU& firstChannel =
                        releaseActiveChannels[firstIndex];
                    const Vec3 firstAxisValue =
                        vector(firstChannel.axisAndHalfLength);
                    if (!(norm(firstAxisValue) > 1.0e-12)) {
                        releaseChannelGeometryValid = false;
                        continue;
                    }
                    const Vec3 firstAxis = firstAxisValue *
                        (1.0 / norm(firstAxisValue));
                    const Vec3 distal =
                        vector(firstChannel.originAndRadius) +
                        firstAxis * firstChannel.axisAndHalfLength.w;
                    for (std::size_t secondIndex = 0u;
                         secondIndex < releaseActiveChannels.size();
                         ++secondIndex) {
                        if (firstIndex == secondIndex) continue;
                        const NMPunctureChannelGPU& secondChannel =
                            releaseActiveChannels[secondIndex];
                        const Vec3 secondAxisValue =
                            vector(secondChannel.axisAndHalfLength);
                        if (!(norm(secondAxisValue) > 1.0e-12)) {
                            releaseChannelGeometryValid = false;
                            continue;
                        }
                        const Vec3 secondAxis = secondAxisValue *
                            (1.0 / norm(secondAxisValue));
                        const Vec3 proximal =
                            vector(secondChannel.originAndRadius) -
                            secondAxis *
                                secondChannel.axisAndHalfLength.w;
                        if (norm(distal - proximal) <=
                            0.25 * initialTip.radiusM) {
                            ++channelReleaseLinks;
                        }
                    }
                }
                for (const NMTetrahedronGPU& tetrahedron :
                     releaseSnapshot.femTopologyTetrahedra) {
                    releaseTetrahedra +=
                        (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
                }
                for (const NMFEMTopologyStateGPU& topology :
                     releaseSnapshot.topologyStates) {
                    releaseRemovedMassKg += topology.accounting.y;
                }
                bool releaseCertificatesAccepted = true;
                channelReleaseMinimumDeterminant =
                    std::numeric_limits<double>::infinity();
                for (const NMSolverCertificateGPU& certificate :
                     releaseSnapshot.solverCertificates) {
                    releaseCertificatesAccepted =
                        releaseCertificatesAccepted &&
                        certificate.validity.w > 0.5f;
                    channelReleaseMinimumDeterminant = std::min(
                        channelReleaseMinimumDeterminant,
                        static_cast<double>(certificate.validity.x)
                    );
                    channelReleaseMaximumResidual = std::max({
                        channelReleaseMaximumResidual,
                        static_cast<double>(certificate.nonlinear.x),
                        static_cast<double>(certificate.nonlinear.z),
                        static_cast<double>(certificate.nonlinear.w),
                    });
                }
                const NeedleTipCapsuleGeometry releasedTip =
                    needleTipCapsuleGeometry(
                        needleForPlacement,
                        channelRelease.result.finalSceneBodies.at(0u)
                    );
                channelReleaseSignedNeedleMotionM = dot(
                    releasedTip.worldTip - entryTip.worldTip,
                    initialTip.approachDirection
                );
                channelReleaseSwageErrorM = swageAttachmentError(
                    world,
                    channelRelease.result
                );
                channelReleaseStateHash = 1469598103934665603ull;
                appendStateHash(
                    channelReleaseStateHash,
                    channelRelease.result.finalQ
                );
                appendStateHash(
                    channelReleaseStateHash,
                    channelRelease.result.finalV
                );
                appendStateHash(
                    channelReleaseStateHash,
                    channelRelease.result.finalSceneBodies
                );
                appendStateHash(
                    channelReleaseStateHash,
                    channelRelease.result.finalRodNodes
                );
                appendStateHash(
                    channelReleaseStateHash,
                    channelRelease.result.finalRodEdges
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.femNodes
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.femFields
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.femTopologyTetrahedra
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.punctureChannels
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.topologyStates
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.reactions
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.contactSamples
                );
                appendStateHash(
                    channelReleaseStateHash,
                    releaseSnapshot.solverCertificates
                );
                require(
                    channelReleaseContacts == 0u &&
                        channelReleaseChannels >=
                            (tissuePunctureAdvanceOnly ? 2u : 1u) &&
                        releaseGenerations.size() ==
                            channelReleaseChannels &&
                        releaseChannelGeometryValid &&
                        channelReleaseLinks + 1u ==
                            channelReleaseChannels &&
                        (!tissuePunctureAdvanceOnly ||
                         channelReleaseTotalLengthM >=
                            2.0 * expectedEntryTractLengthM) &&
                        releaseTetrahedra ==
                            tissueCoupon.metadata.tetrahedronCount &&
                        releaseRemovedMassKg == 0.0 &&
                        releaseCertificatesAccepted &&
                        std::isfinite(channelReleaseMinimumDeterminant) &&
                        channelReleaseMinimumDeterminant > 0.0 &&
                        std::isfinite(channelReleaseMaximumResidual) &&
                        std::isfinite(channelReleaseSignedNeedleMotionM) &&
                        (tissuePunctureAdvanceOnly
                            ? channelReleaseSignedNeedleMotionM > 0.0
                            : std::abs(channelReleaseSignedNeedleMotionM) <
                                1.0e-3) &&
                        channelReleaseSwageErrorM <
                            kMaximumSwageAttachmentError,
                    "embedded puncture channel did not release the entering "
                    "needle without tissue mass loss: contacts=" +
                        std::to_string(channelReleaseContacts) +
                        " channels=" +
                        std::to_string(channelReleaseChannels) +
                        " total_channel_length=" +
                        std::to_string(channelReleaseTotalLengthM) +
                        " links=" +
                        std::to_string(channelReleaseLinks) +
                        " unique_generations=" +
                        std::to_string(releaseGenerations.size()) +
                        " active_tetrahedra=" +
                        std::to_string(releaseTetrahedra) +
                        " removed_mass=" +
                        std::to_string(releaseRemovedMassKg) +
                        " minimum_determinant=" +
                        std::to_string(channelReleaseMinimumDeterminant) +
                        " maximum_residual=" +
                        std::to_string(channelReleaseMaximumResidual) +
                        " swage_error=" +
                        std::to_string(channelReleaseSwageErrorM) +
                        " signed_needle_motion=" +
                        std::to_string(channelReleaseSignedNeedleMotionM)
                );
            } else {
                require(
                    reaction.impulseAndCount.w >= 1.0f &&
                        norm(reactionImpulse) > 1.0e-9 &&
                        needleVelocityDelta > 1.0e-9 &&
                        threadRootVelocityDelta > 1.0e-9 &&
                        maximumTissueDisplacement > 0.0 &&
                        certificatesAccepted &&
                        std::isfinite(minimumDeterminant) &&
                        minimumDeterminant > 0.0 &&
                        swageError < kMaximumSwageAttachmentError,
                    "tissue reaction did not traverse the needle-swage-thread "
                    "chain in the accepted transaction: reaction_contacts=" +
                        std::to_string(reaction.impulseAndCount.w) +
                        " active_contacts=" +
                        std::to_string(activeContacts) +
                        " reaction_impulse=" +
                        vectorSummary(reactionImpulse) +
                        " needle_velocity_delta=" +
                        std::to_string(needleVelocityDelta) +
                        " thread_root_velocity_delta=" +
                        std::to_string(threadRootVelocityDelta) +
                        " maximum_tissue_displacement=" +
                        std::to_string(maximumTissueDisplacement) +
                        " certificates_accepted=" +
                        std::to_string(certificatesAccepted) +
                        " minimum_determinant=" +
                        std::to_string(minimumDeterminant) +
                        " swage_error=" + std::to_string(swageError)
                );
            }
            std::cout << std::setprecision(9)
                << (tissueRestOnly
                    ? "tissue_static_equilibrium=ok"
                    : (tissuePunctureOnly
                        ? (tissuePunctureAdvanceOnly
                            ? "tissue_puncture_channel_advance=ok"
                            : "tissue_tapered_tip_puncture=ok")
                        : "tissue_suture_coupling=ok"))
                << " reaction_contacts="
                << reaction.impulseAndCount.w
                << " final_active_contacts=" << activeContacts
                << " final_normal_impulse_ns=" << normalImpulse
                << " reaction_impulse_ns="
                << vectorSummary(reactionImpulse)
                << " needle_velocity_delta_mps="
                << needleVelocityDelta
                << " thread_root_velocity_delta_mps="
                << threadRootVelocityDelta
                << " maximum_tissue_displacement_m="
                << maximumTissueDisplacement
                << " hard_swage_root_error_m=" << swageError
                << " matter_maximum_nonlinear_residual="
                << maximumNonlinearResidual
                << " matter_maximum_relative_correction="
                << maximumRelativeCorrection
                << " matter_maximum_volume_residual="
                << maximumVolumeResidual
                << " matter_minimum_determinant="
                << minimumDeterminant
                << " matter_maximum_fgmres_iterations="
                << maximumFGMRESIterations
                << " active_puncture_channels=" << activeChannels
                << " active_tetrahedra=" << activeTetrahedra
                << " removed_tissue_mass_kg=" << removedMassKg
                << " analytic_tip_tract_mass_kg="
                << analyticTipTractMassKg
                << " removed_to_tip_tract_mass_ratio="
                << removedToTipTractMassRatio
                << " puncture_channel_radius_m=" << channelRadiusM
                << " puncture_channel_length_m=" << channelLengthM
                << " puncture_channel_axis_alignment="
                << channelAxisAlignment
                << " puncture_impulse_threshold_ns="
                << (tissuePunctureOnly
                    ? kPunctureImpulseThresholdNs
                    : 0.0)
                << " channel_release_contacts="
                << channelReleaseContacts
                << " channel_release_channels="
                << channelReleaseChannels
                << " channel_release_links="
                << channelReleaseLinks
                << " channel_release_total_length_m="
                << channelReleaseTotalLengthM
                << " channel_release_signed_needle_motion_m="
                << channelReleaseSignedNeedleMotionM
                << " channel_release_minimum_determinant="
                << channelReleaseMinimumDeterminant
                << " channel_release_maximum_residual="
                << channelReleaseMaximumResidual
                << " channel_release_swage_error_m="
                << channelReleaseSwageErrorM
                << " channel_release_state_fnv64=0x" << std::hex
                << channelReleaseStateHash << std::dec
                << " accepted_state_fnv64=0x" << std::hex
                << acceptedStateHash << std::dec
                << " reference_gpu_ms="
                << reference.diagnostics.gpuElapsedMilliseconds
                << " coupled_gpu_ms="
                << coupled.diagnostics.gpuElapsedMilliseconds
                << " failed_steps="
                << coupled.diagnostics.failedStepCount << '\n';
            return 0;
        }

        if (!options.resumeGiverReleaseMotionPath.empty()) {
            const std::uint32_t holdSteps = options.settleStepLimit != 0u
                ? options.settleStepLimit
                : kGiverReleaseSettleSteps;
            metalrobo::HeterogeneousWorld referenceWorld = world;
            (void)loadHandoffState(
                options.receiverGraspReferencePath,
                "load-exchange",
                referenceWorld
            );
            metalrobo::MetalWorldResult referenceState;
            referenceState.finalQ = referenceWorld.model.defaultQ;
            referenceState.finalV = referenceWorld.model.defaultV;
            referenceState.finalSceneBodies =
                referenceWorld.defaultSceneBodies;
            const GraspReference receiverReference = graspReference(
                world,
                needleForPlacement,
                referenceState,
                1u,
                kReceiverNeedleShape
            );
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                holdSteps
            );
            PhaseResult released = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                holdSteps
            );
            const ContactCounts releaseContacts = contactCounts(
                world,
                released.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const double releaseSwageError =
                swageAttachmentError(world, released.result);
            const double releaseSwageTangentError =
                swageTangentAngleError(world, released.result);
            const GraspKinematics releaseReceiverMotion = graspKinematics(
                world,
                needleForPlacement,
                released.result,
                1u,
                kReceiverNeedleShape,
                receiverReference
            );
            const RodStateMetrics releaseRod = rodStateMetrics(
                world,
                released.result
            );
            const MRMetalWorldContactStatusGPU& releaseResidual =
                requireTerminalResidual(
                    released.result,
                    "resumed giver release"
                );
            std::cerr << "handoff_phase=giver_release_resume"
                << " hold_steps=" << holdSteps
                << " hard_swage_root_error_m=" << releaseSwageError
                << " swage_tangent_angle_error_rad="
                << releaseSwageTangentError
                << " receiver_seat_drift_m="
                << releaseReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << releaseReceiverMotion.relativePointSpeed
                << " receiver_relative_angular_speed_radps="
                << releaseReceiverMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << releaseRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << releaseRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << releaseRod.maximumEdgeLengthError
                << " contact_residuals=["
                << releaseResidual.residuals.x << ','
                << releaseResidual.residuals.y << ','
                << releaseResidual.residuals.z << ','
                << releaseResidual.residuals.w << ']'
                << contactSummary(releaseContacts) << '\n';
            require(
                !bilateral(releaseContacts, 0u) &&
                    bilateral(releaseContacts, 1u) &&
                    cleanNeedleInteraction(
                        releaseContacts,
                        false,
                        true
                    ) &&
                    qualifiedDrivenGrasp(releaseReceiverMotion) &&
                    qualifiedTransitionRod(releaseRod) &&
                    releaseSwageError < kMaximumSwageAttachmentError &&
                    releaseSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "resumed receiver did not retain sole positive control"
            );
            const bool fullyQualified =
                holdSteps >= kGiverReleaseSettleSteps;
            if (fullyQualified) {
                writeHandoffStateArtifact(
                    options.stateOutputDirectory,
                    "giver-release",
                    loadedStateStep + holdSteps,
                    world,
                    sutureSpec,
                    released.result
                );
            }
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_giver_release_resume=ok"
                << " hold_steps=" << holdSteps
                << " qualification="
                << (fullyQualified ? "yes" : "diagnostic")
                << " receiver_seat_drift_m="
                << releaseReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << releaseReceiverMotion.relativePointSpeed
                << " thread_self_clearance_m="
                << releaseRod.minimumNonNeighbourSurfaceClearance
                << " contact_residuals=["
                << releaseResidual.residuals.x << ','
                << releaseResidual.residuals.y << ','
                << releaseResidual.residuals.z << ','
                << releaseResidual.residuals.w << ']'
                << " gpu_ms="
                << released.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }

        if (!options.resumeReceiverTransferMotionPath.empty()) {
            const std::uint32_t holdSteps = options.settleStepLimit != 0u
                ? options.settleStepLimit
                : kReceiverTransferSettleSteps;
            metalrobo::HeterogeneousWorld receiverReferenceWorld = world;
            (void)loadHandoffState(
                options.receiverGraspReferencePath,
                "load-exchange",
                receiverReferenceWorld
            );
            metalrobo::MetalWorldResult receiverReferenceState;
            receiverReferenceState.finalQ =
                receiverReferenceWorld.model.defaultQ;
            receiverReferenceState.finalV =
                receiverReferenceWorld.model.defaultV;
            receiverReferenceState.finalSceneBodies =
                receiverReferenceWorld.defaultSceneBodies;
            const GraspReference receiverReference = graspReference(
                world,
                needleForPlacement,
                receiverReferenceState,
                1u,
                kReceiverNeedleShape
            );

            metalrobo::HeterogeneousWorld transferStartWorld = world;
            (void)loadHandoffState(
                options.receiverTransferStartReferencePath,
                "giver-release",
                transferStartWorld
            );
            const Vec3 beforeTransfer = vector(
                transferStartWorld.defaultSceneBodies.at(0u).position
            );
            target = transferStartWorld.model.defaultQ;
            auto receiverTransferred = armLocalQ(
                transferStartWorld.model,
                1u,
                transferStartWorld.model.defaultQ
            );
            receiverTransferred[2] = std::max(
                0.0,
                receiverTransferred[2] - kReceiverTransfer
            );
            receiverTransferred[6] = -receiverTransportJawCoordinate;
            receiverTransferred[7] = receiverTransportJawCoordinate;
            setArmTarget(
                target,
                transferStartWorld.model,
                1u,
                receiverTransferred
            );
            efforts = interpolateTargets(
                world.model,
                targetStart,
                target,
                holdSteps
            );
            PhaseResult transferred = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                holdSteps
            );
            const ContactCounts transferContacts = contactCounts(
                world,
                transferred.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const Vec3 afterTransfer = vector(
                transferred.result.finalSceneBodies.at(0u).position
            );
            const double receiverFollow = norm(
                afterTransfer - beforeTransfer
            );
            const double finalSwageAttachmentError =
                swageAttachmentError(world, transferred.result);
            const double finalSwageTangentError =
                swageTangentAngleError(world, transferred.result);
            const GraspKinematics transferReceiverMotion = graspKinematics(
                world,
                needleForPlacement,
                transferred.result,
                1u,
                kReceiverNeedleShape,
                receiverReference
            );
            const RodStateMetrics transferRod = rodStateMetrics(
                world,
                transferred.result
            );
            const MRMetalWorldContactStatusGPU& transferResidual =
                requireTerminalResidual(
                    transferred.result,
                    "resumed receiver-transfer hold"
                );
            std::cerr
                << "handoff_phase=receiver_transfer_motion_resume"
                << " hold_steps=" << holdSteps
                << " receiver_follow_m=" << receiverFollow
                << " hard_swage_root_error_m="
                << finalSwageAttachmentError
                << " swage_tangent_angle_error_rad="
                << finalSwageTangentError
                << " receiver_seat_drift_m="
                << transferReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << transferReceiverMotion.relativePointSpeed
                << " receiver_relative_angular_speed_radps="
                << transferReceiverMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << transferRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << transferRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << transferRod.maximumEdgeLengthError
                << " contact_residuals=["
                << transferResidual.residuals.x << ','
                << transferResidual.residuals.y << ','
                << transferResidual.residuals.z << ','
                << transferResidual.residuals.w << ']'
                << contactSummary(transferContacts) << '\n';
            require(
                bilateral(transferContacts, 1u) &&
                    !bilateral(transferContacts, 0u) &&
                    cleanNeedleInteraction(
                        transferContacts,
                        false,
                        true
                    ) &&
                    qualifiedDrivenGrasp(transferReceiverMotion) &&
                    qualifiedTerminalRod(transferRod) &&
                    receiverFollow > kMinimumReceiverTransfer &&
                    finalSwageAttachmentError <
                        kMaximumSwageAttachmentError &&
                    finalSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "resumed receiver-transfer hold did not retain the needle "
                "and attached thread"
            );
            const bool fullyQualified =
                holdSteps >= kReceiverTransferSettleSteps;
            if (fullyQualified) {
                writeHandoffStateArtifact(
                    options.stateOutputDirectory,
                    "receiver-transfer",
                    loadedStateStep + holdSteps,
                    world,
                    sutureSpec,
                    transferred.result
                );
            }
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_receiver_transfer_hold_resume=ok"
                << " hold_steps=" << holdSteps
                << " transfer_command_m=" << kReceiverTransfer
                << " qualification="
                << (fullyQualified ? "yes" : "diagnostic")
                << " receiver_follow_m=" << receiverFollow
                << " receiver_seat_drift_m="
                << transferReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << transferReceiverMotion.relativePointSpeed
                << " thread_self_clearance_m="
                << transferRod.minimumNonNeighbourSurfaceClearance
                << " contact_residuals=["
                << transferResidual.residuals.x << ','
                << transferResidual.residuals.y << ','
                << transferResidual.residuals.z << ','
                << transferResidual.residuals.w << ']'
                << " gpu_ms="
                << transferred.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }

        if (!options.resumeGiverReleasePath.empty()) {
            metalrobo::HeterogeneousWorld referenceWorld = world;
            (void)loadHandoffState(
                options.receiverGraspReferencePath,
                "load-exchange",
                referenceWorld
            );
            metalrobo::MetalWorldResult referenceState;
            referenceState.finalQ = referenceWorld.model.defaultQ;
            referenceState.finalV = referenceWorld.model.defaultV;
            referenceState.finalSceneBodies =
                referenceWorld.defaultSceneBodies;
            const GraspReference receiverReference = graspReference(
                world,
                needleForPlacement,
                referenceState,
                1u,
                kReceiverNeedleShape
            );
            target = targetStart;
            auto receiverTransferred = armLocalQ(
                world.model,
                1u,
                targetStart
            );
            receiverTransferred[2] = std::max(
                0.0,
                receiverTransferred[2] - kReceiverTransfer
            );
            receiverTransferred[6] = -receiverTransportJawCoordinate;
            receiverTransferred[7] = receiverTransportJawCoordinate;
            setArmTarget(
                target,
                world.model,
                1u,
                receiverTransferred
            );
            const CrossArmCollisionScan transferPreflight =
                scanCrossArmTargetPath(
                    world,
                    targetStart,
                    target,
                    kReceiverTransferSteps
                );
            require(
                transferPreflight.samplesWithContact == 0u &&
                    transferPreflight.samplesWithGiverPadContact == 0u &&
                    transferPreflight.samplesWithReceiverPadContact == 0u,
                "receiver transfer intersects the giver or table"
            );
            efforts = interpolateTargets(
                world.model,
                targetStart,
                target,
                kReceiverTransferSteps
            );
            const Vec3 beforeTransfer = vector(
                world.defaultSceneBodies.at(0u).position
            );
            PhaseResult transferMotion = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kReceiverTransferSteps
            );
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                "receiver-transfer-motion",
                loadedStateStep + kReceiverTransferSteps,
                world,
                sutureSpec,
                transferMotion.result
            );
            if (options.mode == "--receiver-transfer-motion-only") {
                const ContactCounts motionContacts = contactCounts(
                    world,
                    transferMotion.result,
                    needleForPlacement.metadata,
                    kNeedleFirstShape
                );
                const Vec3 afterMotion = vector(
                    transferMotion.result.finalSceneBodies.at(0u).position
                );
                const double receiverFollow = norm(
                    afterMotion - beforeTransfer
                );
                const double motionSwageError =
                    swageAttachmentError(world, transferMotion.result);
                const double motionSwageTangentError =
                    swageTangentAngleError(world, transferMotion.result);
                const GraspKinematics motionReceiver = graspKinematics(
                    world,
                    needleForPlacement,
                    transferMotion.result,
                    1u,
                    kReceiverNeedleShape,
                    receiverReference
                );
                const RodStateMetrics motionRod = rodStateMetrics(
                    world,
                    transferMotion.result
                );
                std::cerr
                    << "handoff_phase=receiver_transfer_motion"
                    << " receiver_follow_m=" << receiverFollow
                    << " hard_swage_root_error_m=" << motionSwageError
                    << " swage_tangent_angle_error_rad="
                    << motionSwageTangentError
                    << " receiver_seat_drift_m="
                    << motionReceiver.seatDrift
                    << " receiver_relative_point_speed_mps="
                    << motionReceiver.relativePointSpeed
                    << " receiver_relative_angular_speed_radps="
                    << motionReceiver.relativeAngularSpeed
                    << " thread_self_clearance_m="
                    << motionRod.minimumNonNeighbourSurfaceClearance
                    << " thread_max_node_speed_mps="
                    << motionRod.maximumNodeSpeed
                    << " thread_max_edge_length_error_m="
                    << motionRod.maximumEdgeLengthError
                    << contactSummary(motionContacts) << '\n';
                require(
                    bilateral(motionContacts, 1u) &&
                        !bilateral(motionContacts, 0u) &&
                        cleanNeedleInteraction(
                            motionContacts,
                            false,
                            true
                        ) &&
                        qualifiedDrivenGrasp(motionReceiver) &&
                        qualifiedTransitionRod(motionRod) &&
                        receiverFollow > kMinimumReceiverTransfer &&
                        motionSwageError <
                            kMaximumSwageAttachmentError &&
                        motionSwageTangentError <
                            maximumSwageTangentAngleError(world),
                    "slow receiver-transfer motion did not retain the "
                    "needle and attached thread"
                );
                std::cout << std::setprecision(9)
                    << "dual_psm_suture_handoff_receiver_transfer_motion=ok"
                    << " transfer_steps=" << kReceiverTransferSteps
                    << " transfer_command_m=" << kReceiverTransfer
                    << " qualification=motion_checkpoint"
                    << " receiver_follow_m=" << receiverFollow
                    << " receiver_seat_drift_m="
                    << motionReceiver.seatDrift
                    << " receiver_relative_point_speed_mps="
                    << motionReceiver.relativePointSpeed
                    << " thread_self_clearance_m="
                    << motionRod.minimumNonNeighbourSurfaceClearance
                    << " gpu_ms="
                    << transferMotion.diagnostics.gpuElapsedMilliseconds
                    << '\n';
                return 0;
            }
            efforts = interpolateTargets(
                world.model,
                transferMotion.result.finalQ,
                target,
                kReceiverTransferSettleSteps
            );
            PhaseResult transferred = continuePhase(
                context,
                compiled,
                stepConfig,
                resident,
                efforts,
                kReceiverTransferSettleSteps,
                "resumed receiver-transfer hold"
            );
            transferred.diagnostics.successfulStepCount +=
                transferMotion.diagnostics.successfulStepCount;
            transferred.diagnostics.gpuElapsedMilliseconds +=
                transferMotion.diagnostics.gpuElapsedMilliseconds;
            const ContactCounts transferContacts = contactCounts(
                world,
                transferred.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const Vec3 afterTransfer = vector(
                transferred.result.finalSceneBodies.at(0u).position
            );
            const double receiverFollow = norm(
                afterTransfer - beforeTransfer
            );
            const double finalSwageAttachmentError =
                swageAttachmentError(world, transferred.result);
            const double finalSwageTangentError =
                swageTangentAngleError(world, transferred.result);
            const GraspKinematics transferReceiverMotion = graspKinematics(
                world,
                needleForPlacement,
                transferred.result,
                1u,
                kReceiverNeedleShape,
                receiverReference
            );
            const RodStateMetrics transferRod = rodStateMetrics(
                world,
                transferred.result
            );
            const MRMetalWorldContactStatusGPU& transferResidual =
                requireTerminalResidual(
                    transferred.result,
                    "resumed receiver transfer"
                );
            std::cerr
                << "handoff_phase=receiver_transfer_resume"
                << " receiver_follow_m=" << receiverFollow
                << " hard_swage_root_error_m="
                << finalSwageAttachmentError
                << " swage_tangent_angle_error_rad="
                << finalSwageTangentError
                << " receiver_seat_drift_m="
                << transferReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << transferReceiverMotion.relativePointSpeed
                << " receiver_relative_angular_speed_radps="
                << transferReceiverMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << transferRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << transferRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << transferRod.maximumEdgeLengthError
                << " contact_residuals=["
                << transferResidual.residuals.x << ','
                << transferResidual.residuals.y << ','
                << transferResidual.residuals.z << ','
                << transferResidual.residuals.w << ']'
                << contactSummary(transferContacts) << '\n';
            require(
                bilateral(transferContacts, 1u) &&
                    !bilateral(transferContacts, 0u) &&
                    cleanNeedleInteraction(
                        transferContacts,
                        false,
                        true
                    ) &&
                    qualifiedDrivenGrasp(transferReceiverMotion) &&
                    qualifiedTerminalRod(transferRod) &&
                    receiverFollow > kMinimumReceiverTransfer &&
                    finalSwageAttachmentError <
                        kMaximumSwageAttachmentError &&
                    finalSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "resumed receiver transfer did not carry the needle and "
                "attached thread"
            );
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                "receiver-transfer",
                loadedStateStep + kReceiverTransferSteps +
                    kReceiverTransferSettleSteps,
                world,
                sutureSpec,
                transferred.result
            );
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_receiver_transfer_resume=ok"
                << " transfer_steps=" << kReceiverTransferSteps
                << " transfer_command_m=" << kReceiverTransfer
                << " settling_steps=" << kReceiverTransferSettleSteps
                << " receiver_follow_m=" << receiverFollow
                << " receiver_seat_drift_m="
                << transferReceiverMotion.seatDrift
                << " receiver_relative_point_speed_mps="
                << transferReceiverMotion.relativePointSpeed
                << " thread_self_clearance_m="
                << transferRod.minimumNonNeighbourSurfaceClearance
                << " contact_residuals=["
                << transferResidual.residuals.x << ','
                << transferResidual.residuals.y << ','
                << transferResidual.residuals.z << ','
                << transferResidual.residuals.w << ']'
                << " gpu_ms="
                << transferred.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }

        if (!options.resumePositiveControlMotionPath.empty()) {
            const std::uint32_t positiveControlResumeSettleSteps =
                options.settleStepLimit != 0u
                ? options.settleStepLimit
                : kReceiverClosureSettleSteps;
            metalrobo::MetalWorldResult loadedPositiveControlState;
            loadedPositiveControlState.finalQ = world.model.defaultQ;
            loadedPositiveControlState.finalV = world.model.defaultV;
            loadedPositiveControlState.finalSceneBodies =
                world.defaultSceneBodies;
            receiverGraspReference = graspReference(
                world,
                needleForPlacement,
                loadedPositiveControlState,
                1u,
                kReceiverNeedleShape
            );
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                positiveControlResumeSettleSteps
            );
            PhaseResult overlap = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                positiveControlResumeSettleSteps
            );
            const ContactCounts overlapContacts = contactCounts(
                world,
                overlap.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            metalrobo::HeterogeneousWorld referenceWorld = world;
            (void)loadHandoffState(
                options.giverGraspReferencePath,
                "receiver-aligned",
                referenceWorld
            );
            metalrobo::MetalWorldResult referenceState;
            referenceState.finalQ = referenceWorld.model.defaultQ;
            referenceState.finalV = referenceWorld.model.defaultV;
            referenceState.finalSceneBodies =
                referenceWorld.defaultSceneBodies;
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                referenceState,
                0u,
                kGiverNeedleShape
            );
            const GraspKinematics overlapGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                overlap.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const GraspKinematics overlapReceiverMotion = graspKinematics(
                world,
                needleForPlacement,
                overlap.result,
                1u,
                kReceiverNeedleShape,
                *receiverGraspReference
            );
            const RodStateMetrics overlapRod = rodStateMetrics(
                world,
                overlap.result
            );
            const double overlapSwageError =
                swageAttachmentError(world, overlap.result);
            const double overlapSwageTangentError =
                swageTangentAngleError(world, overlap.result);
            require(
                !overlap.result.contactStatuses.empty(),
                "positive-control resumed hold published no contact status"
            );
            const MRMetalWorldContactStatusGPU& overlapResidual =
                overlap.result.contactStatuses.back();
            std::cerr << "handoff_phase=positive_control_motion_resume"
                << " hard_swage_root_error_m=" << overlapSwageError
                << " swage_tangent_angle_error_rad="
                << overlapSwageTangentError
                << " giver_seat_drift_m="
                << overlapGiverMotion.seatDrift
                << " receiver_seat_drift_m="
                << overlapReceiverMotion.seatDrift
                << " giver_relative_point_speed_mps="
                << overlapGiverMotion.relativePointSpeed
                << " receiver_relative_point_speed_mps="
                << overlapReceiverMotion.relativePointSpeed
                << " thread_self_clearance_m="
                << overlapRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << overlapRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << overlapRod.maximumEdgeLengthError
                << " contact_residuals=["
                << overlapResidual.residuals.x << ','
                << overlapResidual.residuals.y << ','
                << overlapResidual.residuals.z << ','
                << overlapResidual.residuals.w << ']'
                << contactSummary(overlapContacts) << '\n';
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                "positive-control-motion",
                loadedStateStep + positiveControlResumeSettleSteps,
                world,
                sutureSpec,
                overlap.result
            );
            (void)requireTerminalResidual(
                overlap.result,
                "positive-control motion resumed hold"
            );
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                "positive-control-overlap",
                loadedStateStep + positiveControlResumeSettleSteps,
                world,
                sutureSpec,
                overlap.result
            );
            require(
                bilateral(overlapContacts, 0u) &&
                    bilateral(overlapContacts, 1u) &&
                    cleanNeedleInteraction(
                        overlapContacts,
                        true,
                        true
                    ) &&
                    qualifiedTransitionGrasp(overlapGiverMotion) &&
                    qualifiedDrivenGrasp(overlapReceiverMotion) &&
                    qualifiedTransitionRod(overlapRod) &&
                    overlapSwageError < kMaximumSwageAttachmentError &&
                    overlapSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "resumed positive-control motion did not settle into "
                "bounded dual control"
            );
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_positive_control=ok"
                << " closure_steps=0"
                << " settling_steps="
                << positiveControlResumeSettleSteps
                << " giver_seat_drift_m="
                << overlapGiverMotion.seatDrift
                << " receiver_seat_drift_m="
                << overlapReceiverMotion.seatDrift
                << " thread_self_clearance_m="
                << overlapRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << overlapRod.maximumNodeSpeed
                << " contact_residuals=["
                << overlapResidual.residuals.x << ','
                << overlapResidual.residuals.y << ','
                << overlapResidual.residuals.z << ','
                << overlapResidual.residuals.w << ']'
                << contactSummary(overlapContacts)
                << " gpu_ms="
                << overlap.diagnostics.gpuElapsedMilliseconds << '\n';
            return 0;
        } else if (!options.resumeGiverLiftPath.empty() ||
            !options.resumeGiverHandoffStagePath.empty() ||
            !options.resumeGiverHandoffStagePrefixPath.empty()) {
            const bool resumedHandoffStage =
                !options.resumeGiverHandoffStagePath.empty();
            const bool resumedHandoffStagePrefix =
                !options.resumeGiverHandoffStagePrefixPath.empty();
            const std::string_view resumedPhase = resumedHandoffStage
                ? "giver-handoff-stage"
                : resumedHandoffStagePrefix
                    ? "giver-handoff-stage-prefix"
                    : "giver-lift";
            const std::uint32_t kResumeHoldSteps =
                (resumedHandoffStage || resumedHandoffStagePrefix)
                ? 25u
                : 1u;
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kResumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const double resumedNeedleLift =
                resumed.result.finalSceneBodies[0].position.z -
                config.surgical.needlePose.position[2];
            const double resumedSwageError =
                swageAttachmentError(world, resumed.result);
            const double resumedSwageTangentError =
                swageTangentAngleError(world, resumed.result);
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape
            );
            if (resumedHandoffStagePrefix) {
                metalrobo::HeterogeneousWorld referenceWorld = world;
                (void)loadHandoffState(
                    options.giverLiftReferencePath,
                    "giver-lift",
                    referenceWorld
                );
                metalrobo::MetalWorldResult referenceState;
                referenceState.finalQ = referenceWorld.model.defaultQ;
                referenceState.finalV = referenceWorld.model.defaultV;
                referenceState.finalSceneBodies =
                    referenceWorld.defaultSceneBodies;
                giverGraspReference = graspReference(
                    world,
                    needleForPlacement,
                    referenceState,
                    0u,
                    kGiverNeedleShape
                );
            }
            const GraspKinematics resumedGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const RodStateMetrics resumedRod = rodStateMetrics(
                world,
                resumed.result
            );
            const MRMetalWorldContactStatusGPU& resumedResidual =
                requireTerminalResidual(
                    resumed.result,
                    std::string{resumedPhase} + " checkpoint",
                    !(resumedHandoffStage || resumedHandoffStagePrefix)
                );
            require(
                bilateral(resumedContacts, 0u) &&
                    !bilateral(resumedContacts, 1u) &&
                    cleanNeedleInteraction(
                        resumedContacts,
                        true,
                        false
                    ) &&
                    qualifiedDrivenGrasp(resumedGiverMotion) &&
                    (
                        (resumedHandoffStage ||
                         resumedHandoffStagePrefix)
                        ? (
                            resumedRod.maximumEdgeLengthError <=
                                kMaximumTerminalRodEdgeLengthError &&
                            resumedRod
                                    .minimumNonNeighbourSurfaceClearance >=
                                kMinimumThreadSelfCollisionClearance -
                                    kThreadClearanceReadbackTolerance
                        )
                        : qualifiedTerminalRod(resumedRod)
                    ) &&
                    resumedNeedleLift > 0.0075 &&
                    (
                        (!resumedHandoffStage &&
                         !resumedHandoffStagePrefix) ||
                        resumed.result.finalSceneBodies[0].position.y -
                                config.surgical.needlePose.position[1] >
                            (
                                resumedHandoffStage
                                ? kHandoffStagingOffset.y - 0.005
                                : 0.0
                            )
                    ) &&
                    resumedSwageError < kMaximumSwageAttachmentError &&
                    resumedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                std::string{resumedPhase} +
                " checkpoint did not retain a physically coupled "
                "table-clear grasp: seat_drift_m=" +
                std::to_string(resumedGiverMotion.seatDrift) +
                " relative_point_speed_mps=" +
                std::to_string(resumedGiverMotion.relativePointSpeed) +
                " relative_angular_speed_radps=" +
                std::to_string(resumedGiverMotion.relativeAngularSpeed) +
                " thread_clearance_m=" + std::to_string(
                    resumedRod.minimumNonNeighbourSurfaceClearance
                ) + " thread_speed_mps=" +
                std::to_string(resumedRod.maximumNodeSpeed) +
                " thread_edge_error_m=" +
                std::to_string(resumedRod.maximumEdgeLengthError) +
                " swage_error_m=" +
                std::to_string(resumedSwageError) +
                " swage_tangent_error_rad=" +
                std::to_string(resumedSwageTangentError) +
                contactSummary(resumedContacts)
            );
            std::cerr << "handoff_phase=" << resumedPhase << "_resume"
                << " needle_lift_m=" << resumedNeedleLift
                << " hard_swage_root_error_m=" << resumedSwageError
                << " swage_tangent_angle_error_rad="
                << resumedSwageTangentError
                << " relative_point_speed_mps="
                << resumedGiverMotion.relativePointSpeed
                << " relative_angular_speed_radps="
                << resumedGiverMotion.relativeAngularSpeed
                << " relative_needle_tangent_spin_radps="
                << resumedGiverMotion.relativeNeedleTangentSpin
                << " thread_self_clearance_m="
                << resumedRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << resumedRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << resumedRod.maximumEdgeLengthError
                << " contact_residuals=["
                << resumedResidual.residuals.x << ','
                << resumedResidual.residuals.y << ','
                << resumedResidual.residuals.z << ','
                << resumedResidual.residuals.w << ']'
                << contactSummary(resumedContacts) << '\n';
            preReceiverSuccessfulSteps =
                loadedStateStep + kResumeHoldSteps;
            preReceiverGpuMilliseconds =
                resumed.diagnostics.gpuElapsedMilliseconds;
            qualifiedLift.emplace(std::move(resumed));
        } else if (!options.resumeReceiverApproachMotionPath.empty()) {
            constexpr std::uint32_t kResumeHoldSteps = 1u;
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kResumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape
            );
            if (!options.giverGraspReferencePath.empty()) {
                metalrobo::HeterogeneousWorld referenceWorld = world;
                (void)loadHandoffState(
                    options.giverGraspReferencePath,
                    "giver-handoff-stage",
                    referenceWorld
                );
                metalrobo::MetalWorldResult referenceState;
                referenceState.finalQ = referenceWorld.model.defaultQ;
                referenceState.finalV = referenceWorld.model.defaultV;
                referenceState.finalSceneBodies =
                    referenceWorld.defaultSceneBodies;
                giverGraspReference = graspReference(
                    world,
                    needleForPlacement,
                    referenceState,
                    0u,
                    kGiverNeedleShape
                );
            }
            const GraspKinematics resumedGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const RodStateMetrics resumedRod = rodStateMetrics(
                world,
                resumed.result
            );
            const double resumedSwageError =
                swageAttachmentError(world, resumed.result);
            const double resumedSwageTangentError =
                swageTangentAngleError(world, resumed.result);
            const MRMetalWorldContactStatusGPU& resumedResidual =
                requireTerminalResidual(
                    resumed.result,
                    "receiver approach-motion checkpoint",
                    false
                );
            require(
                bilateral(resumedContacts, 0u) &&
                    cleanReceiverApproachNeedleInteraction(
                        resumedContacts
                    ) &&
                    qualifiedTransitionRod(resumedRod) &&
                    resumedSwageError < kMaximumSwageAttachmentError &&
                    resumedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "receiver approach-motion checkpoint lost its safe open-jaw "
                "handoff state: " + contactSummary(resumedContacts)
            );
            std::cerr << "handoff_phase=receiver_approach_motion_resume"
                << " hard_swage_root_error_m=" << resumedSwageError
                << " swage_tangent_angle_error_rad="
                << resumedSwageTangentError
                << " giver_seat_drift_m="
                << resumedGiverMotion.seatDrift
                << " giver_relative_point_speed_mps="
                << resumedGiverMotion.relativePointSpeed
                << " giver_relative_angular_speed_radps="
                << resumedGiverMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << resumedRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << resumedRod.maximumNodeSpeed
                << " contact_residuals=["
                << resumedResidual.residuals.x << ','
                << resumedResidual.residuals.y << ','
                << resumedResidual.residuals.z << ','
                << resumedResidual.residuals.w << ']'
                << contactSummary(resumedContacts) << '\n';
            preReceiverSuccessfulSteps =
                loadedStateStep + kResumeHoldSteps;
            // This one-step resume phase is moved into the receiver result
            // below, so its GPU time is accounted there exactly once.
            preReceiverGpuMilliseconds = 0.0;
            receiverApproachMotionAlreadyCompleted = true;
            qualifiedLift.emplace(std::move(resumed));
        } else if (!options.resumeReceiverApproachPath.empty()) {
            constexpr std::uint32_t kResumeHoldSteps = 50u;
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kResumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape
            );
            const GraspKinematics resumedGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const RodStateMetrics resumedRod = rodStateMetrics(
                world,
                resumed.result
            );
            const double resumedSwageError =
                swageAttachmentError(world, resumed.result);
            const double resumedSwageTangentError =
                swageTangentAngleError(world, resumed.result);
            const MRMetalWorldContactStatusGPU& resumedResidual =
                requireTerminalResidual(
                    resumed.result,
                    "receiver approach checkpoint"
                );
            require(
                bilateral(resumedContacts, 0u) &&
                    cleanReceiverApproachNeedleInteraction(
                        resumedContacts
                    ) &&
                    qualifiedDrivenGrasp(resumedGiverMotion) &&
                    qualifiedTransitionRod(resumedRod) &&
                    resumedSwageError < kMaximumSwageAttachmentError &&
                    resumedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "receiver approach checkpoint lost its safe open-jaw "
                "handoff state: " + contactSummary(resumedContacts)
            );
            std::cerr << "handoff_phase=receiver_approach_resume"
                << " hard_swage_root_error_m=" << resumedSwageError
                << " swage_tangent_angle_error_rad="
                << resumedSwageTangentError
                << " giver_seat_drift_m="
                << resumedGiverMotion.seatDrift
                << " giver_relative_point_speed_mps="
                << resumedGiverMotion.relativePointSpeed
                << " giver_relative_angular_speed_radps="
                << resumedGiverMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << resumedRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << resumedRod.maximumNodeSpeed
                << " contact_residuals=["
                << resumedResidual.residuals.x << ','
                << resumedResidual.residuals.y << ','
                << resumedResidual.residuals.z << ','
                << resumedResidual.residuals.w << ']'
                << contactSummary(resumedContacts) << '\n';
            preReceiverSuccessfulSteps =
                loadedStateStep + kResumeHoldSteps;
            // The resumed qualified approach becomes receiverApproached and
            // carries its re-entry-hold GPU diagnostic forward.
            preReceiverGpuMilliseconds = 0.0;
            receiverApproachAlreadyCompleted = true;
            qualifiedLift.emplace(std::move(resumed));
        } else if (!options.resumeReceiverAlignedPath.empty()) {
            constexpr std::uint32_t kResumeHoldSteps = 50u;
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kResumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape
            );
            const GraspKinematics resumedGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const ReceiverFrameError resumedFrame = receiverFrameError(
                world,
                needleForPlacement,
                resumed.result.finalQ,
                resumed.result.finalV,
                resumed.result.finalSceneBodies[0]
            );
            const RodStateMetrics resumedRod = rodStateMetrics(
                world,
                resumed.result
            );
            const double resumedSwageError =
                swageAttachmentError(world, resumed.result);
            const double resumedSwageTangentError =
                swageTangentAngleError(world, resumed.result);
            const MRMetalWorldContactStatusGPU& resumedResidual =
                requireTerminalResidual(
                    resumed.result,
                    "receiver-aligned checkpoint"
                );
            require(
                bilateral(resumedContacts, 0u) &&
                    cleanReceiverApproachNeedleInteraction(
                        resumedContacts
                    ) &&
                    qualifiedDrivenGrasp(resumedGiverMotion) &&
                    qualifiedTransitionRod(resumedRod) &&
                    resumedFrame.centering <= 5.0e-5 &&
                    resumedFrame.railTangentAngle <= 0.03 &&
                    resumedFrame.separationFrameAngle <= 0.12 &&
                    resumedSwageError < kMaximumSwageAttachmentError &&
                    resumedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "receiver-aligned checkpoint lost its clear live needle "
                "frame: " + contactSummary(resumedContacts)
            );
            std::cerr << "handoff_phase=receiver_alignment_resume"
                << " centering_error_m=" << resumedFrame.centering
                << " rail_tangent_angle_rad="
                << resumedFrame.railTangentAngle
                << " separation_frame_angle_rad="
                << resumedFrame.separationFrameAngle
                << " giver_relative_point_speed_mps="
                << resumedGiverMotion.relativePointSpeed
                << " hard_swage_root_error_m=" << resumedSwageError
                << " swage_tangent_angle_error_rad="
                << resumedSwageTangentError
                << " thread_self_clearance_m="
                << resumedRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << resumedRod.maximumNodeSpeed
                << " contact_residuals=["
                << resumedResidual.residuals.x << ','
                << resumedResidual.residuals.y << ','
                << resumedResidual.residuals.z << ','
                << resumedResidual.residuals.w << ']'
                << contactSummary(resumedContacts) << '\n';
            preReceiverSuccessfulSteps =
                loadedStateStep + kResumeHoldSteps;
            preReceiverGpuMilliseconds = 0.0;
            receiverAlignmentAlreadyCompleted = true;
            qualifiedLift.emplace(std::move(resumed));
        } else if (!options.resumePositiveControlOverlapPath.empty()) {
            const std::uint32_t resumeHoldSteps =
                options.settleStepLimit != 0u
                ? options.settleStepLimit
                : 50u;
            metalrobo::MetalWorldResult loadedPositiveControlState;
            loadedPositiveControlState.finalQ = world.model.defaultQ;
            loadedPositiveControlState.finalV = world.model.defaultV;
            loadedPositiveControlState.finalSceneBodies =
                world.defaultSceneBodies;
            const GraspReference receiverCheckpointReference =
                graspReference(
                    world,
                    needleForPlacement,
                    loadedPositiveControlState,
                    1u,
                    kReceiverNeedleShape
                );
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                resumeHoldSteps
            );
            PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                resumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const double resumedSwageError =
                swageAttachmentError(world, resumed.result);
            const double resumedSwageTangentError =
                swageTangentAngleError(world, resumed.result);
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape
            );
            if (!options.giverGraspReferencePath.empty()) {
                metalrobo::HeterogeneousWorld referenceWorld = world;
                (void)loadHandoffState(
                    options.giverGraspReferencePath,
                    "receiver-aligned",
                    referenceWorld
                );
                metalrobo::MetalWorldResult referenceState;
                referenceState.finalQ = referenceWorld.model.defaultQ;
                referenceState.finalV = referenceWorld.model.defaultV;
                referenceState.finalSceneBodies =
                    referenceWorld.defaultSceneBodies;
                giverGraspReference = graspReference(
                    world,
                    needleForPlacement,
                    referenceState,
                    0u,
                    kGiverNeedleShape
                );
            }
            const GraspKinematics resumedGiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const GraspKinematics resumedReceiverMotion = graspKinematics(
                world,
                needleForPlacement,
                resumed.result,
                1u,
                kReceiverNeedleShape,
                receiverCheckpointReference
            );
            const MRMetalWorldContactStatusGPU& resumedResidual =
                requireTerminalResidual(
                    resumed.result,
                    "positive-control checkpoint"
                );
            require(
                bilateral(resumedContacts, 0u) &&
                    bilateral(resumedContacts, 1u) &&
                    cleanNeedleInteraction(
                        resumedContacts,
                        true,
                        true
                    ) &&
                    (
                        options.giverGraspReferencePath.empty()
                        ? qualifiedDrivenGrasp(resumedGiverMotion)
                        : qualifiedTransitionGrasp(resumedGiverMotion)
                    ) &&
                    qualifiedDrivenGrasp(resumedReceiverMotion) &&
                    resumedSwageError < kMaximumSwageAttachmentError &&
                    resumedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "positive-control checkpoint did not retain two independent "
                "needle grasps: " + contactSummary(resumedContacts)
            );
            std::cerr << "handoff_phase=positive_control_overlap_resume"
                << " hard_swage_root_error_m=" << resumedSwageError
                << " swage_tangent_angle_error_rad="
                << resumedSwageTangentError
                << " giver_seat_drift_m="
                << resumedGiverMotion.seatDrift
                << " receiver_seat_drift_m="
                << resumedReceiverMotion.seatDrift
                << " giver_relative_point_speed_mps="
                << resumedGiverMotion.relativePointSpeed
                << " receiver_relative_point_speed_mps="
                << resumedReceiverMotion.relativePointSpeed
                << " contact_residuals=["
                << resumedResidual.residuals.x << ','
                << resumedResidual.residuals.y << ','
                << resumedResidual.residuals.z << ','
                << resumedResidual.residuals.w << ']'
                << contactSummary(resumedContacts) << '\n';
            // The checkpoint re-entry has now demonstrated that its receiver
            // seat is stable. Use the settled endpoint as the independent
            // receiver reference for the subsequent load exchange, while the
            // giver remains tied to its original pre-contact reference.
            receiverGraspReference = graspReference(
                world,
                needleForPlacement,
                resumed.result,
                1u,
                kReceiverNeedleShape
            );
            preReleaseSuccessfulSteps =
                loadedStateStep + resumeHoldSteps;
            preReleaseGpuMilliseconds =
                resumed.diagnostics.gpuElapsedMilliseconds;
            qualifiedOverlap.emplace(std::move(resumed));
        } else if (!options.resumeGiverClosedPath.empty()) {
            constexpr std::uint32_t kResumeHoldSteps = 1u;
            auto resumeEfforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            const PhaseResult resumed = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                resumeEfforts,
                kResumeHoldSteps
            );
            const ContactCounts resumedContacts = contactCounts(
                world,
                resumed.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            require(
                bilateral(resumedContacts, 0u) &&
                    !bilateral(resumedContacts, 1u),
                "resumed state did not retain the giver grasp: " +
                    contactSummary(resumedContacts)
            );
            targetStart = resumed.result.finalQ;
            std::vector<float> regripTarget = targetStart;
            const std::uint32_t giverQOffset =
                world.model.articulations[0].qOffset + 7u;
            regripTarget[giverQOffset + 6u] =
                static_cast<float>(-closeJawCoordinate);
            regripTarget[giverQOffset + 7u] =
                static_cast<float>(closeJawCoordinate);
            resumeEfforts = interpolateTargets(
                world.model,
                targetStart,
                regripTarget,
                kPreLiftRegripSteps
            );
            const PhaseResult regripped = continuePhase(
                context,
                compiled,
                stepConfig,
                resident,
                resumeEfforts,
                kPreLiftRegripSteps,
                "resumed giver calibrated re-grip"
            );
            const ContactCounts regripContacts = contactCounts(
                world,
                regripped.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const double regripSwageError =
                swageAttachmentError(world, regripped.result);
            const double regripSwageTangentError =
                swageTangentAngleError(world, regripped.result);
            require(
                bilateral(regripContacts, 0u) &&
                    cleanGiverNeedleInteraction(
                        regripContacts,
                        false
                    ) &&
                    regripSwageError <
                        kMaximumSwageAttachmentError &&
                    regripSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "calibrated pre-lift re-grip lost positive control: " +
                    contactSummary(regripContacts)
            );
            giverGraspReference = graspReference(
                world,
                needleForPlacement,
                regripped.result,
                0u,
                kGiverNeedleShape
            );
            std::cerr << "handoff_phase=giver_regripped"
                << " hard_swage_root_error_m=" << regripSwageError
                << " swage_tangent_angle_error_rad="
                << regripSwageTangentError
                << contactSummary(regripContacts) << '\n';
            const JawGeometry resumedJaw = worldJawGeometry(
                world.model,
                0u,
                regripped.result.finalQ,
                regripped.result.finalV
            );
            targetStart = regripped.result.finalQ;
            const ArmTrajectory resumeLift = cartesianArmTrajectory(
                world.model,
                psm,
                0u,
                config.surgical.robots.leftBase,
                targetStart,
                resumedJaw.midpoint,
                resumedJaw.midpoint + Vec3{0.0, 0.0, kHandoffLift},
                closeJawCoordinate,
                closeJawCoordinate,
                kHandoffLiftSteps
            );
            resumeEfforts = resumeLift.efforts;
            const std::uint32_t resumeLiftSteps =
                options.resumeLiftStepLimit;
            resumeEfforts.resize(
                static_cast<std::size_t>(resumeLiftSteps) *
                world.model.world.nv
            );
            PhaseResult lifted;
            if (options.liftDiagnosticChunkSteps == 0u) {
                lifted = continuePhaseUnchecked(
                    context,
                    compiled,
                    stepConfig,
                    resident,
                    resumeEfforts,
                    resumeLiftSteps,
                    "resumed giver lift"
                );
            } else {
                std::uint32_t completedLiftSteps = 0u;
                double accumulatedGpuMilliseconds = 0.0;
                while (completedLiftSteps < resumeLiftSteps) {
                    const std::uint32_t chunkSteps = std::min(
                        options.liftDiagnosticChunkSteps,
                        resumeLiftSteps - completedLiftSteps
                    );
                    const std::size_t commandBegin =
                        static_cast<std::size_t>(completedLiftSteps) *
                        world.model.world.nv;
                    const std::size_t commandEnd = commandBegin +
                        static_cast<std::size_t>(chunkSteps) *
                        world.model.world.nv;
                    const std::vector<float> chunkEfforts(
                        resumeEfforts.begin() + commandBegin,
                        resumeEfforts.begin() + commandEnd
                    );
                    PhaseResult checkpoint = continuePhaseUnchecked(
                        context,
                        compiled,
                        stepConfig,
                        resident,
                        chunkEfforts,
                        chunkSteps,
                        "resumed giver lift diagnostic chunk"
                    );
                    accumulatedGpuMilliseconds +=
                        checkpoint.diagnostics.gpuElapsedMilliseconds;
                    completedLiftSteps += std::min(
                        chunkSteps,
                        checkpoint.diagnostics.successfulStepCount
                    );
                    if (!checkpoint.result.finalSceneBodies.empty() &&
                        !checkpoint.result.finalQ.empty()) {
                        const ContactCounts checkpointContacts = contactCounts(
                            world,
                            checkpoint.result,
                            needleForPlacement.metadata,
                            kNeedleFirstShape
                        );
                        const GraspKinematics checkpointMotion =
                            graspKinematics(
                                world,
                                needleForPlacement,
                                checkpoint.result,
                                0u,
                                kGiverNeedleShape,
                                *giverGraspReference
                            );
                        double minimumTwist =
                            std::numeric_limits<double>::infinity();
                        double maximumTwist =
                            -std::numeric_limits<double>::infinity();
                        double maximumTwistRate = 0.0;
                        for (const MRRodEdgeStateGPU& edge :
                             checkpoint.result.finalRodEdges) {
                            minimumTwist = std::min(
                                minimumTwist,
                                static_cast<double>(edge.twistAndRate.x)
                            );
                            maximumTwist = std::max(
                                maximumTwist,
                                static_cast<double>(edge.twistAndRate.x)
                            );
                            maximumTwistRate = std::max(
                                maximumTwistRate,
                                std::abs(static_cast<double>(
                                    edge.twistAndRate.y
                                ))
                            );
                        }
                        const mr_float4 residuals =
                            checkpoint.result.contactStatuses.empty()
                            ? mr_float4{}
                            : checkpoint.result.contactStatuses.back()
                                  .residuals;
                        std::cerr
                            << "handoff_phase=lift_diagnostic"
                            << " completed_steps=" << completedLiftSteps
                            << " hard_swage_root_error_m="
                            << swageAttachmentError(
                                   world,
                                   checkpoint.result
                               )
                            << " swage_tangent_line_error_m="
                            << swageTangentLineError(
                                   world,
                                   checkpoint.result
                               )
                            << " swage_tangent_angle_error_rad="
                            << swageTangentAngleError(
                                   world,
                                   checkpoint.result
                               )
                            << " jaw_needle_seat_drift_m="
                            << checkpointMotion.seatDrift
                            << " relative_point_speed_mps="
                            << checkpointMotion.relativePointSpeed
                            << " relative_angular_speed_radps="
                            << checkpointMotion.relativeAngularSpeed
                            << " relative_needle_tangent_spin_radps="
                            << checkpointMotion.relativeNeedleTangentSpin
                            << " needle_linear_speed_mps="
                            << norm(vector(
                                   checkpoint.result.finalSceneBodies[0]
                                       .linearVelocityAndInverseMass
                               ))
                            << " needle_angular_speed_radps="
                            << norm(vector(
                                   checkpoint.result.finalSceneBodies[0]
                                       .angularVelocity
                               ))
                            << " twist_span_rad="
                            << (
                                std::isfinite(minimumTwist) &&
                                    std::isfinite(maximumTwist)
                                ? maximumTwist - minimumTwist
                                : 0.0
                            )
                            << " maximum_twist_rate_radps="
                            << maximumTwistRate
                            << " contact_residuals=["
                            << residuals.x << ',' << residuals.y << ','
                            << residuals.z << ',' << residuals.w << ']'
                            << contactSummary(checkpointContacts) << '\n';
                    }
                    lifted = std::move(checkpoint);
                    if (!lifted.diagnostics.succeeded() ||
                        lifted.diagnostics.failedStepCount != 0u) {
                        break;
                    }
                }
                lifted.diagnostics.successfulStepCount = completedLiftSteps;
                lifted.diagnostics.gpuElapsedMilliseconds =
                    accumulatedGpuMilliseconds;
            }
            if (!lifted.diagnostics.succeeded() ||
                lifted.diagnostics.failedStepCount != 0u) {
                if (!lifted.result.rodStatuses.empty()) {
                    const MRRodGPUStatus& status =
                        lifted.result.rodStatuses.front();
                    std::cerr << "handoff_phase=lift_failure"
                        << " rod_code=" << status.code
                        << " rod_iterations=" << status.iterations
                        << " rod_failing_index=" << status.failingIndex
                        << " rod_diagnostics=["
                        << status.diagnostics.x << ','
                        << status.diagnostics.y << ','
                        << status.diagnostics.z << ','
                        << status.diagnostics.w << "]\n";
                }
                if (!lifted.result.finalSceneBodies.empty()) {
                    const ContactCounts failureContacts = contactCounts(
                        world,
                        lifted.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
                    );
                    std::cerr
                        << "handoff_phase=lift_failure_state"
                        << " hard_swage_root_error_m="
                        << swageAttachmentError(world, lifted.result)
                        << " needle_linear_speed_mps="
                        << norm(vector(
                            lifted.result.finalSceneBodies[0]
                                .linearVelocityAndInverseMass
                        ))
                        << " needle_angular_speed_radps="
                        << norm(vector(
                            lifted.result.finalSceneBodies[0]
                                .angularVelocity
                        ))
                        << " needle_angular_velocity_radps="
                        << vectorSummary(vector(
                            lifted.result.finalSceneBodies[0]
                                .angularVelocity
                        ))
                        << contactSummary(failureContacts) << '\n';
                }
                if (!lifted.result.finalQ.empty() &&
                    !lifted.result.finalRodNodes.empty()) {
                    writeHandoffStateArtifact(
                        options.stateOutputDirectory,
                        "giver-lift-failure-checkpoint",
                        loadedStateStep + kResumeHoldSteps +
                            kPreLiftRegripSteps +
                            lifted.diagnostics.successfulStepCount,
                        world,
                        sutureSpec,
                        lifted.result
                    );
                }
                throw std::runtime_error(
                    "resumed giver lift failed: " +
                    lifted.diagnostics.message
                );
            }
            const ContactCounts liftContacts = contactCounts(
                world,
                lifted.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const double needleLift =
                lifted.result.finalSceneBodies[0].position.z -
                world.defaultSceneBodies[0].position.z;
            const Vec3 needleLinearVelocity = vector(
                lifted.result.finalSceneBodies[0]
                    .linearVelocityAndInverseMass
            );
            const Vec3 needleAngularVelocity = vector(
                lifted.result.finalSceneBodies[0].angularVelocity
            );
            const GraspKinematics liftMotion = graspKinematics(
                world,
                needleForPlacement,
                lifted.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const RodStateMetrics liftRod = rodStateMetrics(
                world,
                lifted.result
            );
            MRMetalWorldContactStatusGPU liftStatus{};
            if (resumeLiftSteps == kHandoffLiftSteps) {
                liftStatus = requireTerminalResidual(
                    lifted.result,
                    "resumed giver lift"
                );
            } else if (!lifted.result.contactStatuses.empty()) {
                liftStatus = lifted.result.contactStatuses.back();
            }
            const double liftedSwageError =
                swageAttachmentError(world, lifted.result);
            const double liftedSwageTangentError =
                swageTangentAngleError(world, lifted.result);
            std::cerr << "handoff_phase=lift_candidate"
                << " lift_steps=" << resumeLiftSteps
                << " hard_swage_root_error_m="
                << liftedSwageError
                << " swage_tangent_angle_error_rad="
                << liftedSwageTangentError
                << " jaw_needle_seat_drift_m="
                << liftMotion.seatDrift
                << " relative_needle_linear_speed_mps="
                << liftMotion.relativePointSpeed
                << " relative_needle_angular_speed_radps="
                << liftMotion.relativeAngularSpeed
                << " relative_needle_tangent_spin_radps="
                << liftMotion.relativeNeedleTangentSpin
                << " thread_self_clearance_m="
                << liftRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << liftRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << liftRod.maximumEdgeLengthError
                << " contact_residuals=["
                << liftStatus.residuals.x << ','
                << liftStatus.residuals.y << ','
                << liftStatus.residuals.z << ','
                << liftStatus.residuals.w << ']'
                << contactSummary(liftContacts) << '\n';
            require(
                bilateral(liftContacts, 0u) &&
                    cleanGiverNeedleInteraction(
                        liftContacts,
                        resumeLiftSteps == kHandoffLiftSteps
                    ) &&
                    (
                        resumeLiftSteps != kHandoffLiftSteps ||
                        (
                            qualifiedDrivenGrasp(liftMotion) &&
                            qualifiedTerminalRod(liftRod)
                        )
                    ) &&
                    liftedSwageError < kMaximumSwageAttachmentError &&
                    liftedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "resumed giver lost the bilateral needle grasp: " +
                    contactSummary(liftContacts)
            );
            if (resumeLiftSteps == kHandoffLiftSteps) {
                require(
                    liftContacts.needlePadContacts == 0u &&
                        needleLift > 0.0075,
                    "resumed giver did not lift the needle clear of the table: " +
                        contactSummary(liftContacts)
                );
            }
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                resumeLiftSteps == kHandoffLiftSteps
                    ? "giver-lift"
                    : "giver-lift-prefix",
                loadedStateStep + kResumeHoldSteps +
                    kPreLiftRegripSteps + resumeLiftSteps,
                world,
                sutureSpec,
                lifted.result
            );
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_resume="
                << (resumeLiftSteps == kHandoffLiftSteps
                    ? "ok"
                    : "prefix_ok")
                << " lift_steps=" << resumeLiftSteps
                << " needle_lift_m=" << needleLift
                << " needle_linear_speed_mps="
                << norm(needleLinearVelocity)
                << " needle_angular_speed_radps="
                << norm(needleAngularVelocity)
                << " jaw_midpoint_speed_mps="
                << liftMotion.jawPointSpeed
                << " needle_grasp_point_speed_mps="
                << liftMotion.needlePointSpeed
                << " relative_needle_linear_speed_mps="
                << liftMotion.relativePointSpeed
                << " relative_needle_angular_speed_radps="
                << liftMotion.relativeAngularSpeed
                << " relative_needle_tangent_spin_radps="
                << liftMotion.relativeNeedleTangentSpin
                << " jaw_needle_seat_drift_m="
                << liftMotion.seatDrift
                << " thread_self_clearance_m="
                << liftRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << liftRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << liftRod.maximumEdgeLengthError
                << " contact_residuals=["
                << liftStatus.residuals.x << ','
                << liftStatus.residuals.y << ','
                << liftStatus.residuals.z << ','
                << liftStatus.residuals.w << ']'
                << " hard_swage_root_error_m="
                << liftedSwageError
                << " swage_tangent_angle_error_rad="
                << liftedSwageTangentError
                << contactSummary(liftContacts)
                << " gpu_ms="
                << resumed.diagnostics.gpuElapsedMilliseconds +
                    regripped.diagnostics.gpuElapsedMilliseconds +
                    lifted.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }

        if (!qualifiedLift.has_value() && !qualifiedOverlap.has_value()) {
        constexpr std::uint32_t kApproachSteps = 50u;
        constexpr std::uint32_t kApproachHoldSteps = 40u;
        const std::uint32_t kSettleSteps =
            options.resumeApproachHeldPath.empty()
            ? (options.settleStepLimit != 0u
                ? options.settleStepLimit
                : (longSettle
                    ? kLongSettleSteps
                    : kDefaultSettleSteps))
            : 0u;
        PhaseResult settled;
        PhaseResult approached;
        PhaseResult approachHeld;
        std::uint64_t preClosureSuccessfulSteps = 0u;
        double preClosureGpuMilliseconds = 0.0;

        if (!options.resumeApproachHeldPath.empty()) {
            constexpr std::uint32_t kResumeHoldSteps = 1u;
            efforts = interpolateTargets(
                world.model,
                targetStart,
                targetStart,
                kResumeHoldSteps
            );
            approachHeld = initializePhase(
                context,
                compiled,
                world,
                stepConfig,
                resident,
                efforts,
                kResumeHoldSteps
            );
            target = targetStart;
            preClosureSuccessfulSteps =
                loadedStateStep + kResumeHoldSteps;
            preClosureGpuMilliseconds =
                approachHeld.diagnostics.gpuElapsedMilliseconds;
        } else {
        // Give the initially supported monofilament a deterministic physical
        // pre-roll before an instrument enters the neutral zone. Completion
        // alone is insufficient: the velocity, stretch, contact, swage, and
        // tangent bounds below must all establish a quiescent pickup state.
        efforts = interpolateTargets(
            world.model,
            targetStart,
            targetStart,
            kSettleSteps
        );
        settled = initializePhase(
            context,
            compiled,
            world,
            stepConfig,
            resident,
            efforts,
            kSettleSteps
        );
        const ContactCounts settleContacts = contactCounts(
            world,
            settled.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        require(
            settleContacts.needlePadContacts != 0u &&
                !bilateral(settleContacts, 0u) &&
                !bilateral(settleContacts, 1u) &&
                settleContacts.armPadContacts[0] == 0u &&
                settleContacts.armPadContacts[1] == 0u &&
                settleContacts.armNeedleContacts[0] == 0u &&
                settleContacts.armNeedleContacts[1] == 0u &&
                settleContacts.crossArmContacts == 0u,
            "needle did not settle on the physical neutral-zone pad"
        );
        const Vec3 settledNeedleLinearVelocity = vector(
            settled.result.finalSceneBodies[0].
                linearVelocityAndInverseMass
        );
        const Vec3 settledNeedleAngularVelocity = vector(
            settled.result.finalSceneBodies[0].angularVelocity
        );
        std::cerr << std::scientific << std::setprecision(9)
            << "handoff_phase=settled needle_target_shift_m="
            << norm(
                needleShapeWorldCenter(
                    needleForPlacement,
                    kGiverNeedleShape,
                    settled.result.finalSceneBodies[0]
                ) - giverPoint
            ) << " needle_com_delta="
            << vectorSummary(
                vector(settled.result.finalSceneBodies[0].position) -
                needlePosition
            )
            << " needle_linear_velocity="
            << vectorSummary(settledNeedleLinearVelocity)
            << " needle_angular_velocity="
            << vectorSummary(settledNeedleAngularVelocity)
            << armTargetSummary(
                world.model,
                0u,
                settled.result.finalQ,
                world.model.defaultQ
            ) << contactSummary(settleContacts)
            << " rod_attachment0_raw=["
            << settleContacts.rodAttachmentImpulses[0][0] << ','
            << settleContacts.rodAttachmentImpulses[0][1] << ','
            << settleContacts.rodAttachmentImpulses[0][2] << ']'
            << " rod_attachment1_raw=["
            << settleContacts.rodAttachmentImpulses[1][0] << ','
            << settleContacts.rodAttachmentImpulses[1][1] << ','
            << settleContacts.rodAttachmentImpulses[1][2] << ']'
            << '\n';
        double maximumRodSpeed = 0.0;
        std::size_t maximumRodSpeedNode = 0u;
        Vec3 maximumRodSpeedVector{};
        double maximumRodStretch = 0.0;
        double minimumRodZ = std::numeric_limits<double>::infinity();
        double maximumRodZ = -std::numeric_limits<double>::infinity();
        for (std::size_t node = 0u;
             node < settled.result.finalRodNodes.size();
             ++node) {
            const Vec3 position = vector(
                settled.result.finalRodNodes[node].position
            );
            const Vec3 nodeVelocity =
                vector(settled.result.finalRodNodes[node].velocity);
            const double nodeSpeed = norm(nodeVelocity);
            if (nodeSpeed > maximumRodSpeed) {
                maximumRodSpeed = nodeSpeed;
                maximumRodSpeedNode = node;
                maximumRodSpeedVector = nodeVelocity;
            }
            minimumRodZ = std::min(minimumRodZ, position.z);
            maximumRodZ = std::max(maximumRodZ, position.z);
            if (node != 0u) {
                maximumRodStretch = std::max(
                    maximumRodStretch,
                    std::abs(
                        norm(
                            position - vector(
                                settled.result.finalRodNodes[node - 1u]
                                    .position
                            )
                        ) - world.rods[0].model.restLengths[node - 1u]
                    )
                );
            }
        }
        const double settledSelfClearance =
            minimumNonNeighbourRodSurfaceSeparation(
                settled.result,
                world.rods[0].model.radius
            );
        std::uint32_t speedNodeContactCount = 0u;
        double speedNodeContactImpulse = 0.0;
        std::uint32_t speedNodeRigidCollider = MR_INVALID_INDEX;
        std::uint32_t speedNodeRigidBody = MR_INVALID_INDEX;
        if (!settled.result.contactStatuses.empty()) {
            const std::size_t constraintCount = std::min<std::size_t>(
                settled.result.contactStatuses.back().requiredConstraints,
                settled.result.contactEvidence.blocks.size()
            );
            for (std::size_t constraint = 0u;
                 constraint < constraintCount;
                 ++constraint) {
                const auto& block =
                    settled.result.contactEvidence.blocks[constraint];
                if ((block.flags &
                     MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT) == 0u ||
                    block.key.words[0] < world.model.shapes.size() ||
                    constraint >=
                        settled.result.contactEvidence.contacts.size()) {
                    continue;
                }
                const std::uint32_t edge =
                    block.key.words[0] -
                    static_cast<std::uint32_t>(world.model.shapes.size());
                if (edge + 1u < maximumRodSpeedNode ||
                    edge > maximumRodSpeedNode) {
                    continue;
                }
                ++speedNodeContactCount;
                const double impulse = std::abs(static_cast<double>(
                    settled.result.contactEvidence.contacts[constraint]
                        .impulses.x
                ));
                if (impulse > speedNodeContactImpulse) {
                    speedNodeContactImpulse = impulse;
                    speedNodeRigidCollider = block.key.words[1];
                    if (speedNodeRigidCollider < world.model.shapes.size()) {
                        speedNodeRigidBody = world.model.shapes[
                            speedNodeRigidCollider
                        ].bodyIndex;
                    }
                }
            }
        }
        std::cerr << "rod_max_speed_mps=" << maximumRodSpeed
            << " rod_max_speed_node=" << maximumRodSpeedNode
            << " rod_max_velocity="
            << vectorSummary(maximumRodSpeedVector)
            << " speed_node_contacts=" << speedNodeContactCount
            << " speed_node_max_contact_impulse="
            << speedNodeContactImpulse
            << " speed_node_rigid_collider="
            << speedNodeRigidCollider
            << " speed_node_rigid_body=" << speedNodeRigidBody
            << " rod_max_stretch_m=" << maximumRodStretch
            << " thread_self_clearance_m=" << settledSelfClearance
            << " hard_swage_root_error_m="
            << swageAttachmentError(world, settled.result)
            << " swage_tangent_line_error_m="
            << swageTangentLineError(world, settled.result)
            << " swage_tangent_angle_error_rad="
            << swageTangentAngleError(world, settled.result)
            << " swage_material_frame_error_rad="
            << swageMaterialFrameError(world, settled.result)
            << " swage_tangent_bending_strain="
            << swageTangentBendingStrain(world, settled.result)
            << " swage_tangent_bending_stress_pa="
            << swageTangentBendingStressPa(world, settled.result)
            << " maximum_swage_tangent_angle_rad="
            << maximumSwageTangentAngleError(world)
            << " rod_z_range_m=[" << minimumRodZ << ',' << maximumRodZ
            << "] gpu_ms=" << settled.diagnostics.gpuElapsedMilliseconds;
        if (!settled.result.contactStatuses.empty()) {
            const auto& status = settled.result.contactStatuses.back();
            std::cerr << " contact_residuals=["
                << status.residuals.x << ',' << status.residuals.y << ','
                << status.residuals.z << ',' << status.residuals.w << ']';
        }
        if (!settled.result.rodStatuses.empty()) {
            const auto& status = settled.result.rodStatuses.front();
            std::cerr << " rod_iterations=" << status.iterations
                << " rod_diagnostics=[" << status.diagnostics.x << ','
                << status.diagnostics.y << ',' << status.diagnostics.z << ','
                << status.diagnostics.w << ']';
        }
        std::cerr << '\n';
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "settled",
            kSettleSteps,
            world,
            sutureSpec,
            settled.result
        );
        require(
            norm(settledNeedleLinearVelocity) <=
                    kMaximumSettledNeedleLinearSpeed &&
                norm(settledNeedleAngularVelocity) <=
                    kMaximumSettledNeedleAngularSpeed &&
                maximumRodSpeed <= kMaximumSettledRodSpeed &&
                maximumRodStretch <= kMaximumSettledRodStretch &&
                settledSelfClearance >=
                    kMinimumThreadSelfCollisionClearance &&
                settleContacts.minimumRodSeparation >=
                    kMinimumSettledRodSeparation &&
                swageAttachmentError(world, settled.result) <
                    kMaximumSwageAttachmentError &&
                swageTangentAngleError(world, settled.result) <
                    maximumSwageTangentAngleError(world),
            "supported needle-thread reset did not reach the bounded "
            "quiescent pickup state"
        );
        if (settleOnly) {
            return 0;
        }

        targetStart = settled.result.finalQ;
        target = targetStart;

        const double approachVerticalClearance = kPickupVerticalClearance;
        const Vec3 settledGiverPoint = needleShapeWorldCenter(
            needleForPlacement,
            kGiverNeedleShape,
            settled.result.finalSceneBodies[0]
        ) + Vec3{0.0, 0.0, approachVerticalClearance};
        const JawGeometry settledGiverJaw = worldJawGeometry(
            world.model,
            0u,
            settled.result.finalQ,
            settled.result.finalV
        );
        // Follow a Cartesian jaw-midpoint path rather than interpolating PSM
        // joint coordinates. The oblique remote-centre insertion axis otherwise
        // sweeps a tooth through the needle before reaching the handling zone.
        // Four millimetres over 100 ms bounds the approach at 40 mm/s.
        ArmTrajectory giverApproach = cartesianArmTrajectory(
            world.model,
            psm,
            0u,
            config.surgical.robots.leftBase,
            targetStart,
            settledGiverJaw.midpoint,
            settledGiverPoint,
            openJawCoordinate,
            openJawCoordinate,
            kApproachSteps
        );
        target = giverApproach.finalTarget;
        efforts = std::move(giverApproach.efforts);
        approached = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kApproachSteps,
            "giver approach"
        );
        // A needle driver is steadied over the handling zone before the jaws
        // are closed. Hold the Cartesian endpoint for 80 ms so the implicit
        // drive dissipates approach tracking error instead of converting it
        // into an off-axis needle/thread impulse during closure.
        efforts = interpolateTargets(
            world.model,
            target,
            target,
            kApproachHoldSteps
        );
        approachHeld = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kApproachHoldSteps,
            "giver approach hold"
        );
        preClosureSuccessfulSteps =
            kSettleSteps + kApproachSteps + kApproachHoldSteps;
        preClosureGpuMilliseconds =
            settled.diagnostics.gpuElapsedMilliseconds +
            approached.diagnostics.gpuElapsedMilliseconds +
            approachHeld.diagnostics.gpuElapsedMilliseconds;
        }
        const ContactCounts approachContacts = contactCounts(
            world,
            approachHeld.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const Vec3 approachNeedlePoint = needleShapeWorldCenter(
            needleForPlacement,
            kGiverNeedleShape,
            approachHeld.result.finalSceneBodies[0]
        );
        const JawGeometry approachJaw = worldJawGeometry(
            world.model,
            0u,
            approachHeld.result.finalQ,
            approachHeld.result.finalV
        );
        std::cerr << "handoff_phase=giver_approach jaw_midpoint_error_m="
            << norm(approachJaw.midpoint - approachNeedlePoint)
            << " hard_swage_root_error_m="
            << swageAttachmentError(world, approachHeld.result)
            << " jaw_midpoint_delta="
            << vectorSummary(approachJaw.midpoint - approachNeedlePoint)
            << " jaw_a_delta="
            << vectorSummary(approachJaw.jawA - approachNeedlePoint)
            << " jaw_b_delta="
            << vectorSummary(approachJaw.jawB - approachNeedlePoint)
            << armTargetSummary(
                world.model,
                0u,
                approachHeld.result.finalQ,
                target
            ) << contactSummary(approachContacts) << '\n';
        require(
            !bilateral(approachContacts, 0u) &&
                approachContacts.armPadContacts[0] == 0u &&
                approachContacts.armNeedleContacts[0] == 0u &&
                approachContacts.crossArmContacts == 0u,
            "open giver touched the needle, table, or second arm"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "giver-approach-held",
            preClosureSuccessfulSteps,
            world,
            sutureSpec,
            approachHeld.result
        );
        if (approachOnly) {
            return 0;
        }
        targetStart = approachHeld.result.finalQ;
        target = targetStart;

        const std::uint32_t kCloseSteps = jawTravelSteps;
        const ArmTrajectory giverClosure = cartesianArmTrajectory(
            world.model,
            psm,
            0u,
            config.surgical.robots.leftBase,
            targetStart,
            approachJaw.midpoint,
            approachNeedlePoint +
                Vec3{0.0, 0.0, kPickupVerticalClearance},
            openJawCoordinate,
            closeJawCoordinate,
            kCloseSteps
        );
        target = giverClosure.finalTarget;
        efforts = giverClosure.efforts;
        const PhaseResult giverClosed = continuePhaseUnchecked(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kCloseSteps,
            "giver closure"
        );
        if (!giverClosed.diagnostics.succeeded() ||
            giverClosed.diagnostics.failedStepCount != 0u) {
            if (!giverClosed.result.rodStatuses.empty()) {
                const MRRodGPUStatus& status =
                    giverClosed.result.rodStatuses.front();
                std::cerr << "handoff_phase=closure_failure"
                    << " rod_code=" << status.code
                    << " rod_iterations=" << status.iterations
                    << " rod_failing_index=" << status.failingIndex
                    << " rod_diagnostics=["
                    << status.diagnostics.x << ','
                    << status.diagnostics.y << ','
                    << status.diagnostics.z << ','
                    << status.diagnostics.w << "]\n";
            }
            if (!giverClosed.result.finalQ.empty() &&
                !giverClosed.result.finalRodNodes.empty()) {
                writeHandoffStateArtifact(
                    options.stateOutputDirectory,
                    "giver-closure-failure-checkpoint",
                    preClosureSuccessfulSteps +
                        giverClosed.diagnostics.successfulStepCount,
                    world,
                    sutureSpec,
                    giverClosed.result
                );
            }
            throw std::runtime_error(
                "giver closure failed: " +
                giverClosed.diagnostics.message
            );
        }
        // Do not qualify a grasp from the last frame of closure. Hold the
        // commanded preload for 50 ms so contact discovery, friction, needle
        // inertia, and the attached monofilament must settle into persistent
        // bilateral positive control.
        efforts = interpolateTargets(
            world.model,
            target,
            target,
            kGraspStabilizationSteps
        );
        const PhaseResult giverSecured = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kGraspStabilizationSteps,
            "giver grasp stabilization"
        );
        const ContactCounts giverClosedContacts = contactCounts(
            world,
            giverSecured.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const Vec3 closedNeedlePoint = needleShapeWorldCenter(
            needleForPlacement,
            kGiverNeedleShape,
            giverSecured.result.finalSceneBodies[0]
        );
        const JawGeometry closedJaw = worldJawGeometry(
            world.model,
            0u,
            giverSecured.result.finalQ,
            giverSecured.result.finalV
        );
        double maximumGiverTargetError = 0.0;
        const std::uint32_t giverQOffset =
            world.model.articulations[0].qOffset + 7u;
        for (std::uint32_t coordinate = 0u;
             coordinate < 8u;
             ++coordinate) {
            maximumGiverTargetError = std::max(
                maximumGiverTargetError,
                std::abs(
                    static_cast<double>(giverSecured.result.finalQ[
                        giverQOffset + coordinate
                    ]) - target[giverQOffset + coordinate]
                )
            );
        }
        std::cerr << "handoff_phase=giver_closed needle_target_shift_m="
            << norm(
                closedNeedlePoint - giverPoint
            ) << " hard_swage_root_error_m="
            << swageAttachmentError(world, giverSecured.result)
            << " jaw_midpoint_error_m="
            << norm(closedJaw.midpoint - closedNeedlePoint)
            << " jaw_midpoint_delta="
            << vectorSummary(closedJaw.midpoint - closedNeedlePoint)
            << " jaw_a_delta="
            << vectorSummary(closedJaw.jawA - closedNeedlePoint)
            << " jaw_b_delta="
            << vectorSummary(closedJaw.jawB - closedNeedlePoint)
            << " jaw_a_radius_m="
            << norm(closedJaw.jawA - closedNeedlePoint)
            << " jaw_b_radius_m="
            << norm(closedJaw.jawB - closedNeedlePoint)
            << " jaw_separation_m=" << closedJaw.separation
            << " maximum_joint_target_error="
            << maximumGiverTargetError
            << armTargetSummary(
                world.model,
                0u,
                giverSecured.result.finalQ,
                target
            )
            << contactSummary(giverClosedContacts) << '\n';
        const double giverClosedSwageError =
            swageAttachmentError(world, giverSecured.result);
        const double giverClosedSwageTangentError =
            swageTangentAngleError(world, giverSecured.result);
        std::cerr << "handoff_phase=giver_closed_swage"
            << " tangent_angle_error_rad="
            << giverClosedSwageTangentError << '\n';
        require(
            bilateral(giverClosedContacts, 0u) &&
                giverClosedContacts.armPadContacts[0] == 0u &&
                giverClosedContacts.crossArmContacts == 0u &&
                giverClosedSwageError < kMaximumSwageAttachmentError &&
                giverClosedSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "giver did not establish a collision-resolved driving grasp:" +
                contactSummary(giverClosedContacts)
        );
        giverGraspReference = graspReference(
            world,
            needleForPlacement,
            giverSecured.result,
            0u,
            kGiverNeedleShape
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "giver-closed",
            preClosureSuccessfulSteps +
                kCloseSteps + kGraspStabilizationSteps,
            world,
            sutureSpec,
            giverSecured.result
        );
        if (closureOnly) {
            return 0;
        }
        targetStart = giverSecured.result.finalQ;
        target = targetStart;

        // Lift normal to the table before translating. Retraction along the
        // oblique remote-centre shaft would scrape the needle laterally while
        // it is still supported by the pad, which is not a controlled pickup.
        const ArmTrajectory giverLift = cartesianArmTrajectory(
            world.model,
            psm,
            0u,
            config.surgical.robots.leftBase,
            targetStart,
            closedJaw.midpoint,
            closedJaw.midpoint + Vec3{0.0, 0.0, kHandoffLift},
            closeJawCoordinate,
            closeJawCoordinate,
            kHandoffLiftSteps
        );
        target = giverLift.finalTarget;
        efforts = giverLift.efforts;
        const PhaseResult lifted = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kHandoffLiftSteps,
            "giver lift"
        );
        const ContactCounts liftContacts = contactCounts(
            world,
            lifted.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const double needleLift =
            lifted.result.finalSceneBodies[0].position.z -
            world.defaultSceneBodies[0].position.z;
        std::cerr << "handoff_phase=giver_lift needle_lift_m="
            << needleLift << " hard_swage_root_error_m="
            << swageAttachmentError(world, lifted.result)
            << " swage_tangent_angle_error_rad="
            << swageTangentAngleError(world, lifted.result)
            << contactSummary(liftContacts) << '\n';
        const double liftedSwageError =
            swageAttachmentError(world, lifted.result);
        const double liftedSwageTangentError =
            swageTangentAngleError(world, lifted.result);
        const GraspKinematics liftedMotion = graspKinematics(
            world,
            needleForPlacement,
            lifted.result,
            0u,
            kGiverNeedleShape,
            *giverGraspReference
        );
        const RodStateMetrics liftedRod = rodStateMetrics(
            world,
            lifted.result
        );
        const MRMetalWorldContactStatusGPU& liftedResidual =
            requireTerminalResidual(
                lifted.result,
                "giver lift"
            );
        std::cerr << " jaw_needle_seat_drift_m="
            << liftedMotion.seatDrift
            << " relative_needle_point_speed_mps="
            << liftedMotion.relativePointSpeed
            << " relative_needle_angular_speed_radps="
            << liftedMotion.relativeAngularSpeed
            << " thread_self_clearance_m="
            << liftedRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << liftedRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << liftedRod.maximumEdgeLengthError
            << " contact_residuals=["
            << liftedResidual.residuals.x << ','
            << liftedResidual.residuals.y << ','
            << liftedResidual.residuals.z << ','
            << liftedResidual.residuals.w << "]\n";
        require(
            bilateral(liftContacts, 0u) &&
                cleanGiverNeedleInteraction(liftContacts) &&
                qualifiedDrivenGrasp(liftedMotion) &&
                qualifiedTerminalRod(liftedRod) &&
                liftContacts.needlePadContacts == 0u &&
                needleLift > 0.0075 &&
                liftedSwageError < kMaximumSwageAttachmentError &&
                liftedSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "giver did not lift the needle clear of the table"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "giver-lift",
            preClosureSuccessfulSteps +
                kCloseSteps + kGraspStabilizationSteps +
                kHandoffLiftSteps,
            world,
            sutureSpec,
            lifted.result
        );
        preReceiverSuccessfulSteps =
            preClosureSuccessfulSteps +
            kCloseSteps + kGraspStabilizationSteps + kHandoffLiftSteps;
        preReceiverGpuMilliseconds =
            preClosureGpuMilliseconds +
            giverClosed.diagnostics.gpuElapsedMilliseconds +
            giverSecured.diagnostics.gpuElapsedMilliseconds +
            lifted.diagnostics.gpuElapsedMilliseconds;
        qualifiedLift.emplace(std::move(lifted));
        }

        if (!qualifiedOverlap.has_value() &&
            options.resumeGiverHandoffStagePath.empty() &&
            !receiverApproachMotionAlreadyCompleted &&
            !receiverApproachAlreadyCompleted &&
            !receiverAlignmentAlreadyCompleted) {
            require(
                qualifiedLift.has_value() &&
                    giverGraspReference.has_value(),
                "neutral-zone staging is missing the qualified giver grasp"
            );
            const PhaseResult& lifted = qualifiedLift.value();
            targetStart = lifted.result.finalQ;
            const bool resumingStagePrefix =
                !options.resumeGiverHandoffStagePrefixPath.empty();
            const std::uint32_t priorStagingSteps = resumingStagePrefix
                ? options.resumeStagingCompletedSteps
                : 0u;
            const std::uint32_t remainingStagingSteps =
                kHandoffStagingSteps - priorStagingSteps;
            Vec3 remainingStagingOffset = kHandoffStagingOffset;
            if (resumingStagePrefix) {
                const Vec3 currentNeedleTranslation =
                    vector(lifted.result.finalSceneBodies[0].position) -
                    vector(config.surgical.needlePose.position);
                remainingStagingOffset =
                    Vec3{0.0, kHandoffStagingOffset.y, kHandoffLift} -
                    currentNeedleTranslation;
            }
            const JawGeometry stagingStartJaw = worldJawGeometry(
                world.model,
                0u,
                lifted.result.finalQ,
                lifted.result.finalV
            );
            const ArmTrajectory giverStaging = cartesianArmTrajectory(
                world.model,
                psm,
                0u,
                config.surgical.robots.leftBase,
                targetStart,
                stagingStartJaw.midpoint,
                stagingStartJaw.midpoint + remainingStagingOffset,
                closeJawCoordinate,
                closeJawCoordinate,
                remainingStagingSteps
            );
            require(
                giverStaging.maximumVelocityRatio <=
                    kMaximumCommandVelocityRatio,
                "neutral-zone trajectory exceeds the authored PSM joint "
                "velocity envelope"
            );
            const CrossArmCollisionScan stagingPreflight =
                scanCrossArmTargetPath(
                    world,
                    targetStart,
                    giverStaging.finalTarget,
                    remainingStagingSteps,
                    giverStaging.desiredQ
                );
            require(
                stagingPreflight.samplesWithContact == 0u &&
                    stagingPreflight.samplesWithGiverPadContact == 0u &&
                    stagingPreflight.samplesWithReceiverPadContact == 0u,
                "giver neutral-zone trajectory intersects the receiver or "
                "sterile pad"
            );
            std::cerr << "handoff_phase=giver_handoff_stage_preflight"
                << " offset_m=" << vectorSummary(remainingStagingOffset)
                << " prior_steps=" << priorStagingSteps
                << " remaining_steps=" << remainingStagingSteps
                << " max_velocity_ratio="
                << giverStaging.maximumVelocityRatio
                << " max_velocity=" << giverStaging.maximumVelocity
                << " velocity_limit=" << giverStaging.limitingVelocity
                << " velocity_dof="
                << giverStaging.maximumVelocityDof << '\n';

            PhaseResult staged;
            std::uint32_t completedStagingSteps = 0u;
            double stagingGpuMilliseconds = 0.0;
            while (completedStagingSteps < remainingStagingSteps) {
                const std::uint32_t chunkSteps = std::min(
                    kHandoffStagingChunkSteps,
                    remainingStagingSteps - completedStagingSteps
                );
                const std::size_t commandBegin =
                    static_cast<std::size_t>(completedStagingSteps) *
                    world.model.world.nv;
                const std::size_t commandEnd = commandBegin +
                    static_cast<std::size_t>(chunkSteps) *
                    world.model.world.nv;
                const std::vector<float> chunkEfforts(
                    giverStaging.efforts.begin() + commandBegin,
                    giverStaging.efforts.begin() + commandEnd
                );
                PhaseResult checkpoint = continuePhase(
                    context,
                    compiled,
                    stepConfig,
                    resident,
                    chunkEfforts,
                    chunkSteps,
                    "giver neutral-zone staging"
                );
                completedStagingSteps += chunkSteps;
                stagingGpuMilliseconds +=
                    checkpoint.diagnostics.gpuElapsedMilliseconds;
                const ContactCounts checkpointContacts = contactCounts(
                    world,
                    checkpoint.result,
                    needleForPlacement.metadata,
                    kNeedleFirstShape
                );
                const GraspKinematics checkpointMotion = graspKinematics(
                    world,
                    needleForPlacement,
                    checkpoint.result,
                    0u,
                    kGiverNeedleShape,
                    *giverGraspReference
                );
                const RodStateMetrics checkpointRod = rodStateMetrics(
                    world,
                    checkpoint.result
                );
                const double checkpointSwageError =
                    swageAttachmentError(world, checkpoint.result);
                const double checkpointSwageTangentError =
                    swageTangentAngleError(world, checkpoint.result);
                const mr_float4 checkpointResiduals =
                    checkpoint.result.contactStatuses.empty()
                    ? mr_float4{}
                    : checkpoint.result.contactStatuses.back().residuals;
                std::cerr << "handoff_phase=giver_handoff_stage"
                    << " completed_steps="
                    << priorStagingSteps + completedStagingSteps
                    << " jaw_needle_seat_drift_m="
                    << checkpointMotion.seatDrift
                    << " relative_point_speed_mps="
                    << checkpointMotion.relativePointSpeed
                    << " relative_angular_speed_radps="
                    << checkpointMotion.relativeAngularSpeed
                    << " thread_self_clearance_m="
                    << checkpointRod
                           .minimumNonNeighbourSurfaceClearance
                    << " thread_max_node_speed_mps="
                    << checkpointRod.maximumNodeSpeed
                    << " thread_max_edge_length_error_m="
                    << checkpointRod.maximumEdgeLengthError
                    << " hard_swage_root_error_m="
                    << checkpointSwageError
                    << " swage_tangent_angle_error_rad="
                    << checkpointSwageTangentError
                    << " contact_residuals=["
                    << checkpointResiduals.x << ','
                    << checkpointResiduals.y << ','
                    << checkpointResiduals.z << ','
                    << checkpointResiduals.w << ']'
                    << contactSummary(checkpointContacts) << '\n';
                writeHandoffStateArtifact(
                    options.stateOutputDirectory,
                    "giver-handoff-stage-prefix",
                    preReceiverSuccessfulSteps + completedStagingSteps,
                    world,
                    sutureSpec,
                    checkpoint.result
                );
                require(
                    bilateral(checkpointContacts, 0u) &&
                        !bilateral(checkpointContacts, 1u) &&
                        cleanNeedleInteraction(
                            checkpointContacts,
                            true,
                            false
                        ) &&
                        checkpointMotion.seatDrift <=
                            kMaximumQualifiedGraspSeatDrift &&
                        checkpointRod
                                .minimumNonNeighbourSurfaceClearance >=
                            kMinimumThreadSelfCollisionClearance -
                                kThreadClearanceReadbackTolerance &&
                        checkpointRod.maximumEdgeLengthError <=
                            kMaximumTerminalRodEdgeLengthError &&
                        checkpointSwageError <
                            kMaximumSwageAttachmentError &&
                        checkpointSwageTangentError <
                            maximumSwageTangentAngleError(world),
                    "giver lost physical positive control while moving into "
                    "the neutral handoff zone: " +
                        contactSummary(checkpointContacts)
                );
                staged = std::move(checkpoint);
            }
            const std::vector<float> stagingSettleEfforts =
                interpolateTargets(
                    world.model,
                    staged.result.finalQ,
                    giverStaging.finalTarget,
                    kHandoffStagingSettleSteps
                );
            PhaseResult stagedSettled = continuePhase(
                context,
                compiled,
                stepConfig,
                resident,
                stagingSettleEfforts,
                kHandoffStagingSettleSteps,
                "giver neutral-zone settling hold"
            );
            stagingGpuMilliseconds +=
                stagedSettled.diagnostics.gpuElapsedMilliseconds;
            staged = std::move(stagedSettled);
            staged.diagnostics.successfulStepCount =
                completedStagingSteps + kHandoffStagingSettleSteps;
            staged.diagnostics.gpuElapsedMilliseconds =
                stagingGpuMilliseconds;
            const ContactCounts stagedContacts = contactCounts(
                world,
                staged.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
            const GraspKinematics stagedMotion = graspKinematics(
                world,
                needleForPlacement,
                staged.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
            const RodStateMetrics stagedRod = rodStateMetrics(
                world,
                staged.result
            );
            const MRMetalWorldContactStatusGPU& stagedResidual =
                requireTerminalResidual(
                    staged.result,
                    "giver neutral-zone staging"
                );
            const Vec3 stagedNeedleTranslation =
                vector(staged.result.finalSceneBodies[0].position) -
                vector(config.surgical.needlePose.position);
            const double stagedSwageError =
                swageAttachmentError(world, staged.result);
            const double stagedSwageTangentError =
                swageTangentAngleError(world, staged.result);
            require(
                bilateral(stagedContacts, 0u) &&
                    !bilateral(stagedContacts, 1u) &&
                    cleanNeedleInteraction(
                        stagedContacts,
                        true,
                        false
                    ) &&
                    qualifiedDrivenGrasp(stagedMotion) &&
                    qualifiedTerminalRod(stagedRod) &&
                    stagedNeedleTranslation.y > 0.080 &&
                    stagedSwageError < kMaximumSwageAttachmentError &&
                    stagedSwageTangentError <
                        maximumSwageTangentAngleError(world),
                "giver did not establish a qualified neutral-zone handoff "
                "state: translation_m=" +
                vectorSummary(stagedNeedleTranslation) +
                " seat_drift_m=" +
                std::to_string(stagedMotion.seatDrift) +
                " relative_point_speed_mps=" +
                std::to_string(stagedMotion.relativePointSpeed) +
                " relative_angular_speed_radps=" +
                std::to_string(stagedMotion.relativeAngularSpeed) +
                " thread_clearance_m=" + std::to_string(
                    stagedRod.minimumNonNeighbourSurfaceClearance
                ) + " thread_speed_mps=" +
                std::to_string(stagedRod.maximumNodeSpeed) +
                " thread_edge_error_m=" +
                std::to_string(stagedRod.maximumEdgeLengthError) +
                " swage_error_m=" +
                std::to_string(stagedSwageError) +
                " swage_tangent_error_rad=" +
                std::to_string(stagedSwageTangentError) +
                contactSummary(stagedContacts)
            );
            std::cerr << "handoff_phase=giver_handoff_stage_qualified"
                << " needle_translation_m="
                << vectorSummary(stagedNeedleTranslation)
                << " jaw_needle_seat_drift_m="
                << stagedMotion.seatDrift
                << " relative_point_speed_mps="
                << stagedMotion.relativePointSpeed
                << " relative_angular_speed_radps="
                << stagedMotion.relativeAngularSpeed
                << " thread_self_clearance_m="
                << stagedRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << stagedRod.maximumNodeSpeed
                << " thread_max_edge_length_error_m="
                << stagedRod.maximumEdgeLengthError
                << " hard_swage_root_error_m=" << stagedSwageError
                << " swage_tangent_angle_error_rad="
                << stagedSwageTangentError
                << " contact_residuals=["
                << stagedResidual.residuals.x << ','
                << stagedResidual.residuals.y << ','
                << stagedResidual.residuals.z << ','
                << stagedResidual.residuals.w << ']'
                << contactSummary(stagedContacts) << '\n';
            writeHandoffStateArtifact(
                options.stateOutputDirectory,
                "giver-handoff-stage",
                preReceiverSuccessfulSteps + remainingStagingSteps +
                    kHandoffStagingSettleSteps,
                world,
                sutureSpec,
                staged.result
            );
            if (stageOnly) {
                std::cout << std::setprecision(9)
                    << "dual_psm_suture_handoff_stage=ok"
                    << " staging_steps=" << kHandoffStagingSteps
                    << " resumed_prior_staging_steps="
                    << priorStagingSteps
                    << " settling_steps="
                    << kHandoffStagingSettleSteps
                    << " needle_translation_m="
                    << vectorSummary(stagedNeedleTranslation)
                    << " max_command_velocity_ratio="
                    << giverStaging.maximumVelocityRatio
                    << " jaw_needle_seat_drift_m="
                    << stagedMotion.seatDrift
                    << " thread_self_clearance_m="
                    << stagedRod.minimumNonNeighbourSurfaceClearance
                    << " thread_max_node_speed_mps="
                    << stagedRod.maximumNodeSpeed
                    << " thread_max_edge_length_error_m="
                    << stagedRod.maximumEdgeLengthError
                    << " hard_swage_root_error_m="
                    << stagedSwageError
                    << " swage_tangent_angle_error_rad="
                    << stagedSwageTangentError
                    << " contact_residuals=["
                    << stagedResidual.residuals.x << ','
                    << stagedResidual.residuals.y << ','
                    << stagedResidual.residuals.z << ','
                    << stagedResidual.residuals.w << ']'
                    << contactSummary(stagedContacts)
                    << " gpu_ms="
                    << preReceiverGpuMilliseconds +
                        stagingGpuMilliseconds << '\n';
                return 0;
            }
            preReceiverSuccessfulSteps +=
                remainingStagingSteps + kHandoffStagingSettleSteps;
            preReceiverGpuMilliseconds += stagingGpuMilliseconds;
            qualifiedLift.emplace(std::move(staged));
        }

        double needleLift = qualifiedOverlap.has_value()
            ? qualifiedOverlap->result.finalSceneBodies[0].position.z -
                config.surgical.needlePose.position[2]
            : 0.0;
        ContactCounts overlapContacts;
        constexpr std::uint32_t kReceiverApproachSteps = 150u;
        const std::uint32_t kReceiverCloseSteps =
            receiverOverlapTravelSteps;
        if (!qualifiedOverlap.has_value()) {
        require(
            giverGraspReference.has_value(),
            "receiver phase is missing the qualified giver seat reference"
        );
        const std::uint32_t receiverApproachMotionStepsThisRun =
            (receiverApproachMotionAlreadyCompleted ||
             receiverApproachAlreadyCompleted ||
             receiverAlignmentAlreadyCompleted)
            ? 0u
            : kReceiverApproachSteps;
        const std::uint32_t receiverApproachSettleStepsThisRun =
            (receiverApproachAlreadyCompleted ||
             receiverAlignmentAlreadyCompleted)
            ? 0u
            : kReceiverApproachSettleSteps;
        PhaseResult receiverApproached;
        if (receiverApproachAlreadyCompleted ||
            receiverAlignmentAlreadyCompleted) {
            receiverApproached = std::move(qualifiedLift.value());
            needleLift =
                receiverApproached.result.finalSceneBodies[0].position.z -
                config.surgical.needlePose.position[2];
            targetStart = receiverApproached.result.finalQ;
            target = targetStart;
        } else {
        PhaseResult receiverApproachMotion;
        if (receiverApproachMotionAlreadyCompleted) {
            receiverApproachMotion = std::move(qualifiedLift.value());
            needleLift =
                receiverApproachMotion.result.finalSceneBodies[0].position.z -
                config.surgical.needlePose.position[2];
            targetStart = receiverApproachMotion.result.finalQ;
            target = targetStart;
        } else {
        const PhaseResult& lifted = qualifiedLift.value();
        needleLift =
            lifted.result.finalSceneBodies[0].position.z -
            config.surgical.needlePose.position[2];
        targetStart = lifted.result.finalQ;
        target = targetStart;

        const Vec3 liftedReceiverPoint = needleShapeWorldCenter(
            needleForPlacement,
            kReceiverNeedleShape,
            lifted.result.finalSceneBodies[0]
        );
        auto receiverHandoffSeed = psmTarget(
            psm,
            kReceiverHandoffInsertion,
            openJawCoordinate,
            kReceiverYaw,
            kReceiverPitch
        );
        receiverHandoffSeed[3] += kReceiverToolRollOffset;
        receiverHandoffSeed[5] = kReceiverWristYaw;
        const auto receiverHandoffOpenKinematic = solvePsmJawTarget(
            psm,
            config.surgical.robots.rightBase,
            std::move(receiverHandoffSeed),
            liftedReceiverPoint,
            openJawCoordinate,
            0.015
        );
        const auto receiverHandoffOpen = receiverHandoffOpenKinematic;
        setArmTarget(
            target,
            world.model,
            1u,
            receiverHandoffOpen
        );
        const CrossArmCollisionScan receiverApproachPreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                kReceiverApproachSteps
            );
        require(
            receiverApproachPreflight.samplesWithContact == 0u &&
                receiverApproachPreflight
                    .samplesWithGiverPadContact == 0u &&
                receiverApproachPreflight
                    .samplesWithReceiverPadContact == 0u,
            "receiver target path intersects the giver or table before "
            "the handoff"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            kReceiverApproachSteps
        );
        receiverApproachMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverApproachSteps,
            "receiver approach"
        );
        const MRMetalWorldContactStatusGPU& receiverMotionResidual =
            requireTerminalResidual(
                receiverApproachMotion.result,
                "receiver approach motion",
                false
            );
        const RodStateMetrics receiverMotionRod = rodStateMetrics(
            world,
            receiverApproachMotion.result
        );
        std::cerr << "handoff_phase=receiver_approach_motion"
            << " thread_max_node_speed_mps="
            << receiverMotionRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << receiverMotionRod.maximumEdgeLengthError
            << " contact_residuals=["
            << receiverMotionResidual.residuals.x << ','
            << receiverMotionResidual.residuals.y << ','
            << receiverMotionResidual.residuals.z << ','
            << receiverMotionResidual.residuals.w << "]\n";
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-approach-motion",
            preReceiverSuccessfulSteps + kReceiverApproachSteps,
            world,
            sutureSpec,
            receiverApproachMotion.result
        );
        }
        efforts = interpolateTargets(
            world.model,
            receiverApproachMotion.result.finalQ,
            target,
            kReceiverApproachSettleSteps
        );
        receiverApproached = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverApproachSettleSteps,
            "receiver open-jaw settling hold"
        );
        receiverApproached.diagnostics.successfulStepCount +=
            receiverApproachMotion.diagnostics.successfulStepCount;
        receiverApproached.diagnostics.gpuElapsedMilliseconds +=
            receiverApproachMotion.diagnostics.gpuElapsedMilliseconds;
        }
        const ContactCounts receiverApproachContacts = contactCounts(
            world,
            receiverApproached.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const double receiverApproachSwageError =
            swageAttachmentError(world, receiverApproached.result);
        const double receiverApproachSwageTangentError =
            swageTangentAngleError(world, receiverApproached.result);
        const GraspKinematics receiverApproachGiverMotion =
            graspKinematics(
                world,
                needleForPlacement,
                receiverApproached.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
        const RodStateMetrics receiverApproachRod = rodStateMetrics(
            world,
            receiverApproached.result
        );
        const MRMetalWorldContactStatusGPU& receiverApproachResidual =
            requireTerminalResidual(
                receiverApproached.result,
                "receiver approach"
            );
        std::cerr << "handoff_phase=receiver_approach"
            << " hard_swage_root_error_m=" << receiverApproachSwageError
            << " swage_tangent_angle_error_rad="
            << receiverApproachSwageTangentError
            << " giver_seat_drift_m="
            << receiverApproachGiverMotion.seatDrift
            << " giver_relative_point_speed_mps="
            << receiverApproachGiverMotion.relativePointSpeed
            << " giver_relative_angular_speed_radps="
            << receiverApproachGiverMotion.relativeAngularSpeed
            << " thread_self_clearance_m="
            << receiverApproachRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << receiverApproachRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << receiverApproachRod.maximumEdgeLengthError
            << " contact_residuals=["
            << receiverApproachResidual.residuals.x << ','
            << receiverApproachResidual.residuals.y << ','
            << receiverApproachResidual.residuals.z << ','
            << receiverApproachResidual.residuals.w << ']'
            << contactSummary(receiverApproachContacts) << '\n';
        require(
            bilateral(receiverApproachContacts, 0u) &&
                cleanReceiverApproachNeedleInteraction(
                    receiverApproachContacts
                ) &&
                qualifiedTransitionGrasp(receiverApproachGiverMotion) &&
                qualifiedTransitionRod(receiverApproachRod) &&
                receiverApproachSwageError <
                    kMaximumSwageAttachmentError &&
                receiverApproachSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "receiver approach lost giver control or formed a bilateral "
            "receiver grasp while open"
        );
        // The collision-free approach may let the needle settle within the
        // giver groove. Closure must measure any receiver-induced motion from
        // this accepted pre-contact seat, not from the earlier transport seat.
        giverGraspReference = graspReference(
            world,
            needleForPlacement,
            receiverApproached.result,
            0u,
            kGiverNeedleShape
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-approach",
            preReceiverSuccessfulSteps +
                receiverApproachMotionStepsThisRun +
                receiverApproachSettleStepsThisRun,
            world,
            sutureSpec,
            receiverApproached.result
        );
        if (receiverApproachOnly) {
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_receiver_approach=ok"
                << " approach_steps="
                << receiverApproachMotionStepsThisRun
                << " settling_steps="
                << receiverApproachSettleStepsThisRun
                << " giver_seat_drift_m="
                << receiverApproachGiverMotion.seatDrift
                << " thread_self_clearance_m="
                << receiverApproachRod
                    .minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << receiverApproachRod.maximumNodeSpeed
                << " contact_residuals=["
                << receiverApproachResidual.residuals.x << ','
                << receiverApproachResidual.residuals.y << ','
                << receiverApproachResidual.residuals.z << ','
                << receiverApproachResidual.residuals.w << ']'
                << contactSummary(receiverApproachContacts)
                << " gpu_ms="
                << preReceiverGpuMilliseconds +
                    receiverApproached.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }
        targetStart = receiverApproached.result.finalQ;
        target = targetStart;

        PhaseResult receiverAligned;
        std::uint32_t receiverAlignmentStepsThisRun = 0u;
        std::uint32_t receiverAlignmentSettleStepsThisRun = 0u;
        if (receiverAlignmentAlreadyCompleted) {
            receiverAligned = std::move(receiverApproached);
            receiverApproached.diagnostics.successfulStepCount = 0u;
            receiverApproached.diagnostics.gpuElapsedMilliseconds = 0.0;
        } else {
        const ReceiverAlignmentSolution receiverAlignmentTarget =
            solveReceiverAlignmentTarget(
                world,
                psm,
                needleForPlacement,
                receiverApproached.result.finalQ,
                receiverApproached.result.finalV,
                receiverApproached.result.finalSceneBodies[0],
                openJawCoordinate
            );
        setArmTarget(
            target,
            world.model,
            1u,
            receiverAlignmentTarget.localQ
        );
        receiverAlignmentStepsThisRun = velocityLimitedTargetSteps(
                world.model,
                targetStart,
                target,
                kReceiverAlignmentMinimumSteps
            );
        receiverAlignmentSettleStepsThisRun =
            kReceiverAlignmentSettleSteps;
        const CrossArmCollisionScan receiverAlignmentPreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                receiverAlignmentStepsThisRun
            );
        require(
            receiverAlignmentPreflight.samplesWithContact == 0u &&
                receiverAlignmentPreflight
                    .samplesWithGiverPadContact == 0u &&
                receiverAlignmentPreflight
                    .samplesWithReceiverPadContact == 0u,
            "receiver live-frame alignment intersects an instrument or "
            "the table"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            receiverAlignmentStepsThisRun
        );
        PhaseResult receiverAlignmentMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            receiverAlignmentStepsThisRun,
            "receiver live-frame alignment"
        );
        const MRMetalWorldContactStatusGPU& receiverAlignmentMotionResidual =
            requireTerminalResidual(
                receiverAlignmentMotion.result,
                "receiver live-frame alignment motion",
                false
            );
        const RodStateMetrics receiverAlignmentMotionRod = rodStateMetrics(
            world,
            receiverAlignmentMotion.result
        );
        std::cerr << "handoff_phase=receiver_alignment_motion"
            << " planned_centering_residual_m="
            << receiverAlignmentTarget.centeringResidual
            << " planned_rail_tangent_angle_rad="
            << receiverAlignmentTarget.railTangentAngle
            << " planned_separation_frame_angle_rad="
            << receiverAlignmentTarget.separationFrameAngle
            << " thread_max_node_speed_mps="
            << receiverAlignmentMotionRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << receiverAlignmentMotionRod.maximumEdgeLengthError
            << " contact_residuals=["
            << receiverAlignmentMotionResidual.residuals.x << ','
            << receiverAlignmentMotionResidual.residuals.y << ','
            << receiverAlignmentMotionResidual.residuals.z << ','
            << receiverAlignmentMotionResidual.residuals.w << "]\n";
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-alignment-motion",
            preReceiverSuccessfulSteps +
                receiverApproachMotionStepsThisRun +
                receiverApproachSettleStepsThisRun +
                receiverAlignmentStepsThisRun,
            world,
            sutureSpec,
            receiverAlignmentMotion.result
        );
        efforts = interpolateTargets(
            world.model,
            receiverAlignmentMotion.result.finalQ,
            target,
            receiverAlignmentSettleStepsThisRun
        );
        receiverAligned = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            receiverAlignmentSettleStepsThisRun,
            "receiver live-frame alignment hold"
        );
        receiverAligned.diagnostics.successfulStepCount +=
            receiverAlignmentMotion.diagnostics.successfulStepCount;
        receiverAligned.diagnostics.gpuElapsedMilliseconds +=
            receiverAlignmentMotion.diagnostics.gpuElapsedMilliseconds;
        const ContactCounts receiverAlignmentContacts = contactCounts(
            world,
            receiverAligned.result,
            needleForPlacement.metadata,
            kNeedleFirstShape
        );
        const ReceiverFrameError receiverAlignmentError =
            receiverFrameError(
                world,
                needleForPlacement,
                receiverAligned.result.finalQ,
                receiverAligned.result.finalV,
                receiverAligned.result.finalSceneBodies[0]
            );
        const GraspKinematics receiverAlignmentGiverMotion =
            graspKinematics(
                world,
                needleForPlacement,
                receiverAligned.result,
                0u,
                kGiverNeedleShape,
                *giverGraspReference
            );
        const RodStateMetrics receiverAlignmentRod = rodStateMetrics(
            world,
            receiverAligned.result
        );
        const double receiverAlignmentSwageError =
            swageAttachmentError(world, receiverAligned.result);
        const double receiverAlignmentSwageTangentError =
            swageTangentAngleError(world, receiverAligned.result);
        const MRMetalWorldContactStatusGPU& receiverAlignmentResidual =
            requireTerminalResidual(
                receiverAligned.result,
                "receiver live-frame alignment"
            );
        std::cerr << "handoff_phase=receiver_alignment"
            << " centering_error_m=" << receiverAlignmentError.centering
            << " rail_tangent_angle_rad="
            << receiverAlignmentError.railTangentAngle
            << " separation_frame_angle_rad="
            << receiverAlignmentError.separationFrameAngle
            << " giver_seat_drift_m="
            << receiverAlignmentGiverMotion.seatDrift
            << " giver_relative_point_speed_mps="
            << receiverAlignmentGiverMotion.relativePointSpeed
            << " hard_swage_root_error_m="
            << receiverAlignmentSwageError
            << " swage_tangent_angle_error_rad="
            << receiverAlignmentSwageTangentError
            << " thread_self_clearance_m="
            << receiverAlignmentRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << receiverAlignmentRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << receiverAlignmentRod.maximumEdgeLengthError
            << " contact_residuals=["
            << receiverAlignmentResidual.residuals.x << ','
            << receiverAlignmentResidual.residuals.y << ','
            << receiverAlignmentResidual.residuals.z << ','
            << receiverAlignmentResidual.residuals.w << ']'
            << contactSummary(receiverAlignmentContacts) << '\n';
        require(
            bilateral(receiverAlignmentContacts, 0u) &&
                cleanReceiverApproachNeedleInteraction(
                    receiverAlignmentContacts
                ) &&
                qualifiedDrivenGrasp(receiverAlignmentGiverMotion) &&
                qualifiedTransitionRod(receiverAlignmentRod) &&
                receiverAlignmentError.centering <= 5.0e-5 &&
                receiverAlignmentError.railTangentAngle <= 0.03 &&
                receiverAlignmentError.separationFrameAngle <= 0.12 &&
                receiverAlignmentSwageError <
                    kMaximumSwageAttachmentError &&
                receiverAlignmentSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "receiver live-frame alignment did not preserve a clear, "
            "controlled needle"
        );
        giverGraspReference = graspReference(
            world,
            needleForPlacement,
            receiverAligned.result,
            0u,
            kGiverNeedleShape
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-aligned",
            preReceiverSuccessfulSteps +
                receiverApproachMotionStepsThisRun +
                receiverApproachSettleStepsThisRun +
                receiverAlignmentStepsThisRun +
                receiverAlignmentSettleStepsThisRun,
            world,
            sutureSpec,
            receiverAligned.result
        );
        if (receiverAlignmentOnly) {
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_receiver_alignment=ok"
                << " motion_steps=" << receiverAlignmentStepsThisRun
                << " settling_steps="
                << receiverAlignmentSettleStepsThisRun
                << " centering_error_m="
                << receiverAlignmentError.centering
                << " rail_tangent_angle_rad="
                << receiverAlignmentError.railTangentAngle
                << " separation_frame_angle_rad="
                << receiverAlignmentError.separationFrameAngle
                << " giver_seat_drift_m="
                << receiverAlignmentGiverMotion.seatDrift
                << " thread_self_clearance_m="
                << receiverAlignmentRod
                    .minimumNonNeighbourSurfaceClearance
                << " contact_residuals=["
                << receiverAlignmentResidual.residuals.x << ','
                << receiverAlignmentResidual.residuals.y << ','
                << receiverAlignmentResidual.residuals.z << ','
                << receiverAlignmentResidual.residuals.w << ']'
                << contactSummary(receiverAlignmentContacts)
                << " gpu_ms="
                << preReceiverGpuMilliseconds +
                    receiverApproached.diagnostics.gpuElapsedMilliseconds +
                    receiverAligned.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }
        }
        targetStart = receiverAligned.result.finalQ;
        target = targetStart;

        auto receiverHandoffClosed = armLocalQ(
            world.model,
            1u,
            receiverAligned.result.finalQ
        );
        receiverHandoffClosed[6] = -receiverOverlapJawCoordinate;
        receiverHandoffClosed[7] = receiverOverlapJawCoordinate;

        setArmTarget(
            target,
            world.model,
            1u,
            receiverHandoffClosed
        );
        const CrossArmCollisionScan receiverClosurePreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                kReceiverCloseSteps
            );
        require(
            receiverClosurePreflight.samplesWithContact == 0u &&
                receiverClosurePreflight
                    .samplesWithGiverPadContact == 0u &&
                receiverClosurePreflight
                    .samplesWithReceiverPadContact == 0u,
            "receiver closure target intersects the giver or table"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            kReceiverCloseSteps
        );
        PhaseResult overlapMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverCloseSteps,
            "positive-control overlap"
        );
        const MRMetalWorldContactStatusGPU& overlapMotionResidual =
            requireTerminalResidual(
                overlapMotion.result,
                "positive-control motion",
                false
            );
        const RodStateMetrics overlapMotionRod = rodStateMetrics(
            world,
            overlapMotion.result
        );
        std::cerr << "handoff_phase=positive_control_motion"
            << " thread_max_node_speed_mps="
            << overlapMotionRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << overlapMotionRod.maximumEdgeLengthError
            << " contact_residuals=["
            << overlapMotionResidual.residuals.x << ','
            << overlapMotionResidual.residuals.y << ','
            << overlapMotionResidual.residuals.z << ','
            << overlapMotionResidual.residuals.w << "]\n";
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "positive-control-motion",
            preReceiverSuccessfulSteps +
                receiverApproachMotionStepsThisRun +
                receiverApproachSettleStepsThisRun +
                receiverAlignmentStepsThisRun +
                receiverAlignmentSettleStepsThisRun +
                kReceiverCloseSteps,
            world,
            sutureSpec,
            overlapMotion.result
        );
        receiverGraspReference = graspReference(
            world,
            needleForPlacement,
            overlapMotion.result,
            1u,
            kReceiverNeedleShape
        );
        efforts = interpolateTargets(
            world.model,
            overlapMotion.result.finalQ,
            target,
            kReceiverClosureSettleSteps
        );
        PhaseResult overlap = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverClosureSettleSteps,
            "positive-control settling hold"
        );
        overlap.diagnostics.successfulStepCount +=
            overlapMotion.diagnostics.successfulStepCount;
        overlap.diagnostics.gpuElapsedMilliseconds +=
            overlapMotion.diagnostics.gpuElapsedMilliseconds;
        overlapContacts = contactCounts(
            world,
            overlap.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const double overlapSwageError =
            swageAttachmentError(world, overlap.result);
        const double overlapSwageTangentError =
            swageTangentAngleError(world, overlap.result);
        const GraspKinematics overlapGiverMotion = graspKinematics(
            world,
            needleForPlacement,
            overlap.result,
            0u,
            kGiverNeedleShape,
            *giverGraspReference
        );
        const GraspKinematics overlapReceiverMotion = graspKinematics(
            world,
            needleForPlacement,
            overlap.result,
            1u,
            kReceiverNeedleShape,
            *receiverGraspReference
        );
        const RodStateMetrics overlapRod = rodStateMetrics(
            world,
            overlap.result
        );
        const MRMetalWorldContactStatusGPU& overlapResidual =
            requireTerminalResidual(
                overlap.result,
                "positive-control overlap"
            );
        std::cerr << "handoff_phase=positive_control_overlap"
            << " hard_swage_root_error_m=" << overlapSwageError
            << " swage_tangent_angle_error_rad="
            << overlapSwageTangentError
            << " giver_seat_drift_m="
            << overlapGiverMotion.seatDrift
            << " receiver_seat_drift_m="
            << overlapReceiverMotion.seatDrift
            << " giver_relative_point_speed_mps="
            << overlapGiverMotion.relativePointSpeed
            << " receiver_relative_point_speed_mps="
            << overlapReceiverMotion.relativePointSpeed
            << " thread_self_clearance_m="
            << overlapRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << overlapRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << overlapRod.maximumEdgeLengthError
            << " contact_residuals=["
            << overlapResidual.residuals.x << ','
            << overlapResidual.residuals.y << ','
            << overlapResidual.residuals.z << ','
            << overlapResidual.residuals.w << ']'
            << contactSummary(overlapContacts) << '\n';
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "positive-control-overlap",
            preReceiverSuccessfulSteps +
                receiverApproachMotionStepsThisRun +
                receiverApproachSettleStepsThisRun +
                receiverAlignmentStepsThisRun +
                receiverAlignmentSettleStepsThisRun +
                kReceiverCloseSteps +
                kReceiverClosureSettleSteps,
            world,
            sutureSpec,
            overlap.result
        );
        require(
            bilateral(overlapContacts, 0u) &&
                bilateral(overlapContacts, 1u) &&
                cleanNeedleInteraction(overlapContacts, true, true) &&
                // First receiver contact may seat the curved needle between
                // two finite V-grooves. This is the same single sub-radius
                // transition allowance used for the open-jaw approach; the
                // original giver reference is retained through load exchange
                // so subsequent motion cannot be hidden by rebasing.
                qualifiedTransitionGrasp(overlapGiverMotion) &&
                qualifiedDrivenGrasp(overlapReceiverMotion) &&
                qualifiedTransitionRod(overlapRod) &&
                overlapSwageError < kMaximumSwageAttachmentError &&
                overlapSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "handoff did not establish collision-free dual positive control"
        );
        if (receiverClosureOnly) {
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_positive_control=ok"
                << " closure_steps=" << kReceiverCloseSteps
                << " settling_steps=" << kReceiverClosureSettleSteps
                << " giver_seat_drift_m="
                << overlapGiverMotion.seatDrift
                << " receiver_seat_drift_m="
                << overlapReceiverMotion.seatDrift
                << " thread_self_clearance_m="
                << overlapRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << overlapRod.maximumNodeSpeed
                << " contact_residuals=["
                << overlapResidual.residuals.x << ','
                << overlapResidual.residuals.y << ','
                << overlapResidual.residuals.z << ','
                << overlapResidual.residuals.w << ']'
                << contactSummary(overlapContacts)
                << " gpu_ms="
                << preReceiverGpuMilliseconds +
                    receiverApproached.diagnostics.gpuElapsedMilliseconds +
                    receiverAligned.diagnostics.gpuElapsedMilliseconds +
                    overlap.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }
        preReleaseSuccessfulSteps =
            preReceiverSuccessfulSteps +
            receiverApproachMotionStepsThisRun +
            receiverApproachSettleStepsThisRun +
            receiverAlignmentStepsThisRun +
            receiverAlignmentSettleStepsThisRun + kReceiverCloseSteps +
            kReceiverClosureSettleSteps;
        preReleaseGpuMilliseconds =
            preReceiverGpuMilliseconds +
            receiverApproached.diagnostics.gpuElapsedMilliseconds +
            receiverAligned.diagnostics.gpuElapsedMilliseconds +
            overlap.diagnostics.gpuElapsedMilliseconds;
        qualifiedOverlap.emplace(std::move(overlap));
        }

        const PhaseResult& overlap = qualifiedOverlap.value();
        require(
            giverGraspReference.has_value() &&
                receiverGraspReference.has_value(),
            "release phase is missing a qualified dual-grasp reference"
        );
        if (!options.resumePositiveControlOverlapPath.empty()) {
            overlapContacts = contactCounts(
                world,
                overlap.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
            );
        }
        targetStart = overlap.result.finalQ;
        target = targetStart;

        const std::uint32_t kLoadExchangeSteps =
            static_cast<std::uint32_t>(std::ceil(
                3.0 * (
                    receiverOverlapJawCoordinate - closeJawCoordinate
                ) / (0.18 * kControlTimestep)
            )) + 4u;
        auto giverLoadExchanged = armLocalQ(
            world.model,
            0u,
            overlap.result.finalQ
        );
        giverLoadExchanged[6] = -receiverOverlapJawCoordinate;
        giverLoadExchanged[7] = receiverOverlapJawCoordinate;
        auto receiverLoadExchanged = armLocalQ(
            world.model,
            1u,
            overlap.result.finalQ
        );
        receiverLoadExchanged[6] = -closeJawCoordinate;
        receiverLoadExchanged[7] = closeJawCoordinate;
        setArmTarget(
            target,
            world.model,
            0u,
            giverLoadExchanged
        );
        setArmTarget(
            target,
            world.model,
            1u,
            receiverLoadExchanged
        );
        const CrossArmCollisionScan loadExchangePreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                kLoadExchangeSteps
            );
        require(
            loadExchangePreflight.samplesWithContact == 0u &&
                loadExchangePreflight.samplesWithGiverPadContact == 0u &&
                loadExchangePreflight.samplesWithReceiverPadContact == 0u,
            "coordinated handoff load exchange intersects an instrument or "
            "the table"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            kLoadExchangeSteps
        );
        PhaseResult loadExchangeMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kLoadExchangeSteps,
            "coordinated handoff load exchange"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "load-exchange-motion",
            preReleaseSuccessfulSteps + kLoadExchangeSteps,
            world,
            sutureSpec,
            loadExchangeMotion.result
        );
        efforts = interpolateTargets(
            world.model,
            loadExchangeMotion.result.finalQ,
            target,
            kLoadExchangeSettleSteps
        );
        PhaseResult loadExchanged = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kLoadExchangeSettleSteps,
            "coordinated handoff load-exchange hold"
        );
        loadExchanged.diagnostics.successfulStepCount +=
            loadExchangeMotion.diagnostics.successfulStepCount;
        loadExchanged.diagnostics.gpuElapsedMilliseconds +=
            loadExchangeMotion.diagnostics.gpuElapsedMilliseconds;
        const ContactCounts loadExchangeContacts = contactCounts(
            world,
            loadExchanged.result,
            needleForPlacement.metadata,
            kNeedleFirstShape
        );
        const double loadExchangeSwageError =
            swageAttachmentError(world, loadExchanged.result);
        const double loadExchangeSwageTangentError =
            swageTangentAngleError(world, loadExchanged.result);
        const GraspKinematics loadExchangeGiverMotion = graspKinematics(
            world,
            needleForPlacement,
            loadExchanged.result,
            0u,
            kGiverNeedleShape,
            *giverGraspReference
        );
        const GraspKinematics loadExchangeReceiverMotion = graspKinematics(
            world,
            needleForPlacement,
            loadExchanged.result,
            1u,
            kReceiverNeedleShape,
            *receiverGraspReference
        );
        const RodStateMetrics loadExchangeRod = rodStateMetrics(
            world,
            loadExchanged.result
        );
        const MRMetalWorldContactStatusGPU& loadExchangeResidual =
            requireTerminalResidual(
                loadExchanged.result,
                "coordinated handoff load exchange"
            );
        std::cerr << "handoff_phase=load_exchange"
            << " hard_swage_root_error_m=" << loadExchangeSwageError
            << " swage_tangent_angle_error_rad="
            << loadExchangeSwageTangentError
            << " giver_seat_drift_m="
            << loadExchangeGiverMotion.seatDrift
            << " receiver_seat_drift_m="
            << loadExchangeReceiverMotion.seatDrift
            << " giver_relative_point_speed_mps="
            << loadExchangeGiverMotion.relativePointSpeed
            << " receiver_relative_point_speed_mps="
            << loadExchangeReceiverMotion.relativePointSpeed
            << " thread_self_clearance_m="
            << loadExchangeRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << loadExchangeRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << loadExchangeRod.maximumEdgeLengthError
            << " contact_residuals=["
            << loadExchangeResidual.residuals.x << ','
            << loadExchangeResidual.residuals.y << ','
            << loadExchangeResidual.residuals.z << ','
            << loadExchangeResidual.residuals.w << ']'
            << contactSummary(loadExchangeContacts) << '\n';
        require(
            bilateral(loadExchangeContacts, 0u) &&
                bilateral(loadExchangeContacts, 1u) &&
                cleanNeedleInteraction(
                    loadExchangeContacts,
                    true,
                    true
                ) &&
                qualifiedTransitionGrasp(loadExchangeGiverMotion) &&
                qualifiedTransitionGrasp(loadExchangeReceiverMotion) &&
                qualifiedTransitionRod(loadExchangeRod) &&
                loadExchangeSwageError < kMaximumSwageAttachmentError &&
                loadExchangeSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "coordinated load exchange did not preserve dual positive "
            "control"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "load-exchange",
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
                kLoadExchangeSettleSteps,
            world,
            sutureSpec,
            loadExchanged.result
        );
        if (loadExchangeOnly) {
            std::cout << std::setprecision(9)
                << "dual_psm_suture_handoff_load_exchange=ok"
                << " motion_steps=" << kLoadExchangeSteps
                << " settling_steps=" << kLoadExchangeSettleSteps
                << " giver_seat_drift_m="
                << loadExchangeGiverMotion.seatDrift
                << " receiver_seat_drift_m="
                << loadExchangeReceiverMotion.seatDrift
                << " thread_self_clearance_m="
                << loadExchangeRod.minimumNonNeighbourSurfaceClearance
                << " thread_max_node_speed_mps="
                << loadExchangeRod.maximumNodeSpeed
                << " contact_residuals=["
                << loadExchangeResidual.residuals.x << ','
                << loadExchangeResidual.residuals.y << ','
                << loadExchangeResidual.residuals.z << ','
                << loadExchangeResidual.residuals.w << ']'
                << contactSummary(loadExchangeContacts)
                << " gpu_ms="
                << preReleaseGpuMilliseconds +
                    loadExchanged.diagnostics.gpuElapsedMilliseconds
                << '\n';
            return 0;
        }

        // The full receiver preload has now acquired its own physical seat.
        // Measure the subsequent giver clearance and transfer against that
        // established receiver/needle transform rather than counting this
        // bounded load-sharing reseat twice.
        receiverGraspReference = graspReference(
            world,
            needleForPlacement,
            loadExchanged.result,
            1u,
            kReceiverNeedleShape
        );
        targetStart = loadExchanged.result.finalQ;
        target = targetStart;
        const std::uint32_t kReleaseSteps =
            receiverOverlapTravelSteps;
        auto giverReleased = armLocalQ(
            world.model,
            0u,
            loadExchanged.result.finalQ
        );
        giverReleased[6] = -openJawCoordinate;
        giverReleased[7] = openJawCoordinate;
        setArmTarget(target, world.model, 0u, giverReleased);
        const CrossArmCollisionScan giverReleasePreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                kReleaseSteps
            );
        require(
            giverReleasePreflight.samplesWithContact == 0u &&
                giverReleasePreflight.samplesWithGiverPadContact == 0u &&
                giverReleasePreflight.samplesWithReceiverPadContact == 0u,
            "giver clearance intersects the receiver or table"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            kReleaseSteps
        );
        PhaseResult releaseMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReleaseSteps,
            "giver clearance"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "giver-release-motion",
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
                kLoadExchangeSettleSteps + kReleaseSteps,
            world,
            sutureSpec,
            releaseMotion.result
        );
        efforts = interpolateTargets(
            world.model,
            releaseMotion.result.finalQ,
            target,
            kGiverReleaseSettleSteps
        );
        PhaseResult released = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kGiverReleaseSettleSteps,
            "giver-clear receiver-control hold"
        );
        released.diagnostics.successfulStepCount +=
            releaseMotion.diagnostics.successfulStepCount;
        released.diagnostics.gpuElapsedMilliseconds +=
            releaseMotion.diagnostics.gpuElapsedMilliseconds;
        const ContactCounts releaseContacts = contactCounts(
            world,
            released.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const double releaseSwageError =
            swageAttachmentError(world, released.result);
        const double releaseSwageTangentError =
            swageTangentAngleError(world, released.result);
        const GraspKinematics releaseReceiverMotion = graspKinematics(
            world,
            needleForPlacement,
            released.result,
            1u,
            kReceiverNeedleShape,
            *receiverGraspReference
        );
        const RodStateMetrics releaseRod = rodStateMetrics(
            world,
            released.result
        );
        const MRMetalWorldContactStatusGPU& releaseResidual =
            requireTerminalResidual(
                released.result,
                "giver release"
            );
        std::cerr << "handoff_phase=giver_release"
            << " hard_swage_root_error_m=" << releaseSwageError
            << " swage_tangent_angle_error_rad="
            << releaseSwageTangentError
            << " receiver_seat_drift_m="
            << releaseReceiverMotion.seatDrift
            << " receiver_relative_point_speed_mps="
            << releaseReceiverMotion.relativePointSpeed
            << " receiver_relative_angular_speed_radps="
            << releaseReceiverMotion.relativeAngularSpeed
            << " thread_self_clearance_m="
            << releaseRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << releaseRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << releaseRod.maximumEdgeLengthError
            << " contact_residuals=["
            << releaseResidual.residuals.x << ','
            << releaseResidual.residuals.y << ','
            << releaseResidual.residuals.z << ','
            << releaseResidual.residuals.w << ']'
            << contactSummary(releaseContacts) << '\n';
        require(
            !bilateral(releaseContacts, 0u) &&
                bilateral(releaseContacts, 1u) &&
                cleanNeedleInteraction(
                    releaseContacts,
                    false,
                    true
                ) &&
                qualifiedDrivenGrasp(releaseReceiverMotion) &&
                qualifiedTransitionRod(releaseRod) &&
                releaseSwageError < kMaximumSwageAttachmentError &&
                releaseSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "receiver did not retain sole positive control after release"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "giver-release",
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
                kLoadExchangeSettleSteps + kReleaseSteps +
                kGiverReleaseSettleSteps,
            world,
            sutureSpec,
            released.result
        );
        targetStart = released.result.finalQ;
        target = targetStart;

        auto receiverTransferred = armLocalQ(
            world.model,
            1u,
            released.result.finalQ
        );
        receiverTransferred[2] = std::max(
            0.0,
            receiverTransferred[2] - kReceiverTransfer
        );
        receiverTransferred[6] = -receiverTransportJawCoordinate;
        receiverTransferred[7] = receiverTransportJawCoordinate;
        setArmTarget(
            target,
            world.model,
            1u,
            receiverTransferred
        );
        const CrossArmCollisionScan transferPreflight =
            scanCrossArmTargetPath(
                world,
                targetStart,
                target,
                kReceiverTransferSteps
            );
        require(
            transferPreflight.samplesWithContact == 0u &&
                transferPreflight.samplesWithGiverPadContact == 0u &&
                transferPreflight.samplesWithReceiverPadContact == 0u,
            "receiver transfer intersects the giver or table"
        );
        efforts = interpolateTargets(
            world.model,
            targetStart,
            target,
            kReceiverTransferSteps
        );
        const Vec3 beforeTransfer = vector(
            released.result.finalSceneBodies[0].position
        );
        PhaseResult transferMotion = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverTransferSteps,
            "receiver transfer"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-transfer-motion",
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
                kLoadExchangeSettleSteps + kReleaseSteps +
                kGiverReleaseSettleSteps + kReceiverTransferSteps,
            world,
            sutureSpec,
            transferMotion.result
        );
        efforts = interpolateTargets(
            world.model,
            transferMotion.result.finalQ,
            target,
            kReceiverTransferSettleSteps
        );
        PhaseResult transferred = continuePhase(
            context,
            compiled,
            stepConfig,
            resident,
            efforts,
            kReceiverTransferSettleSteps,
            "receiver-transfer hold"
        );
        transferred.diagnostics.successfulStepCount +=
            transferMotion.diagnostics.successfulStepCount;
        transferred.diagnostics.gpuElapsedMilliseconds +=
            transferMotion.diagnostics.gpuElapsedMilliseconds;
        const ContactCounts transferContacts = contactCounts(
            world,
            transferred.result,
                needleForPlacement.metadata,
                kNeedleFirstShape
        );
        const Vec3 afterTransfer = vector(
            transferred.result.finalSceneBodies[0].position
        );
        const double receiverFollow = norm(afterTransfer - beforeTransfer);
        const double finalSwageAttachmentError =
            swageAttachmentError(world, transferred.result);
        const double finalSwageTangentError =
            swageTangentAngleError(world, transferred.result);
        const GraspKinematics transferReceiverMotion = graspKinematics(
            world,
            needleForPlacement,
            transferred.result,
            1u,
            kReceiverNeedleShape,
            *receiverGraspReference
        );
        const RodStateMetrics transferRod = rodStateMetrics(
            world,
            transferred.result
        );
        const MRMetalWorldContactStatusGPU& transferResidual =
            requireTerminalResidual(
                transferred.result,
                "receiver transfer"
            );
        std::cerr << "handoff_phase=receiver_transfer receiver_follow_m="
            << receiverFollow << " hard_swage_root_error_m="
            << finalSwageAttachmentError << contactSummary(transferContacts)
            << " swage_tangent_angle_error_rad="
            << finalSwageTangentError
            << " receiver_seat_drift_m="
            << transferReceiverMotion.seatDrift
            << " receiver_relative_point_speed_mps="
            << transferReceiverMotion.relativePointSpeed
            << " receiver_relative_angular_speed_radps="
            << transferReceiverMotion.relativeAngularSpeed
            << " thread_self_clearance_m="
            << transferRod.minimumNonNeighbourSurfaceClearance
            << " thread_max_node_speed_mps="
            << transferRod.maximumNodeSpeed
            << " thread_max_edge_length_error_m="
            << transferRod.maximumEdgeLengthError
            << " contact_residuals=["
            << transferResidual.residuals.x << ','
            << transferResidual.residuals.y << ','
            << transferResidual.residuals.z << ','
            << transferResidual.residuals.w << ']'
            << '\n';
        require(
            bilateral(transferContacts, 1u) &&
                !bilateral(transferContacts, 0u) &&
                cleanNeedleInteraction(
                    transferContacts,
                    false,
                    true
                ) &&
                qualifiedDrivenGrasp(transferReceiverMotion) &&
                qualifiedTerminalRod(transferRod) &&
                receiverFollow > kMinimumReceiverTransfer &&
                finalSwageAttachmentError <
                    kMaximumSwageAttachmentError &&
                finalSwageTangentError <
                    maximumSwageTangentAngleError(world),
            "receiver transfer did not carry the needle and attached thread"
        );
        writeHandoffStateArtifact(
            options.stateOutputDirectory,
            "receiver-transfer",
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
                kLoadExchangeSettleSteps + kReleaseSteps +
                kGiverReleaseSettleSteps + kReceiverTransferSteps +
                kReceiverTransferSettleSteps,
            world,
            sutureSpec,
            transferred.result
        );

        const auto stats = context.stats();
        const std::uint64_t successfulSteps =
            preReleaseSuccessfulSteps + kLoadExchangeSteps +
            kLoadExchangeSettleSteps + kReleaseSteps +
            kGiverReleaseSettleSteps + kReceiverTransferSteps +
            kReceiverTransferSettleSteps;
        const double totalGpuMilliseconds =
            preReleaseGpuMilliseconds +
            loadExchanged.diagnostics.gpuElapsedMilliseconds +
            released.diagnostics.gpuElapsedMilliseconds +
            transferred.diagnostics.gpuElapsedMilliseconds;
        std::cout << std::setprecision(9)
            << "{\"schema\":\"numi.dual-psm-suture-handoff.v1\""
            << ",\"device\":\""
            << transferred.diagnostics.deviceName << "\""
            << ",\"needle_arc_mm\":"
            << 1000.0 * sutureSpec.needle.arcLengthM.value
            << ",\"needle_curvature\":\"1/2-circle\""
            << ",\"suture_size\":\"3-0\""
            << ",\"thread_length_m\":"
            << sutureSpec.threadLengthM.value
            << ",\"thread_diameter_mm\":"
            << 2000.0 * sutureSpec.threadRadiusM.value
            << ",\"thread_nodes\":"
            << world.rods[0].model.restPositions.size()
            << ",\"hard_swaged_thread_nodes\":"
            << std::count_if(
                world.rods[0].attachments.begin(),
                world.rods[0].attachments.end(),
                [](const auto& attachment) {
                    return attachment.compliance == 0.0;
                }
            )
            << ",\"thread_boundary_nodes\":"
            << world.rods[0].attachments.size() +
                   world.rods[0].tangentBindings.size()
            << ",\"swage_tangent_compliance_rad_per_nm\":"
            << world.rods[0].tangentBindings[0].complianceRadPerNm
            << ",\"swage_torsional_compliance_rad_per_nm\":"
            << world.rods[0].twistBindings[0].complianceRadPerNm
            << ",\"jaw_length_mm\":"
            << 1000.0 *
                metalrobo::surgicalPSMMetadata().
                    largeNeedleDriverJawLength
            << ",\"groove_radial_preload_m\":"
            << kGrooveRailRadialPreload
            << ",\"receiver_overlap_radial_preload_m\":"
            << kReceiverOverlapRailRadialPreload
            << ",\"receiver_overlap_patch_normal_force_n\":"
            << estimatedReceiverOverlapPatchNormalForce
            << ",\"receiver_transport_radial_preload_m\":"
            << kReceiverTransportRailRadialPreload
            << ",\"receiver_transport_patch_normal_force_n\":"
            << estimatedReceiverTransportPatchNormalForce
            << ",\"jaw_close_rad\":" << closeJawCoordinate
            << ",\"receiver_overlap_jaw_rad\":"
            << receiverOverlapJawCoordinate
            << ",\"receiver_transport_jaw_rad\":"
            << receiverTransportJawCoordinate
            << ",\"grasp_zone\":["
            << sutureSpec.needle.graspZoneStartFraction.value << ','
            << sutureSpec.needle.graspZoneEndFraction.value << ']'
            << ",\"needle_lift_m\":" << needleLift
            << ",\"receiver_follow_m\":" << receiverFollow
            << ",\"receiver_grasp_seat_drift_m\":"
            << transferReceiverMotion.seatDrift
            << ",\"load_exchange_receiver_reseating_m\":"
            << loadExchangeReceiverMotion.seatDrift
            << ",\"load_exchange_giver_reseating_m\":"
            << loadExchangeGiverMotion.seatDrift
            << ",\"load_exchange_steps\":" << kLoadExchangeSteps
            << ",\"giver_release_steps\":" << kReleaseSteps
            << ",\"receiver_transfer_steps\":"
            << kReceiverTransferSteps
            << ",\"receiver_transfer_command_m\":"
            << kReceiverTransfer
            << ",\"minimum_receiver_transfer_m\":"
            << kMinimumReceiverTransfer
            << ",\"receiver_transfer_settling_steps\":"
            << kReceiverTransferSettleSteps
            << ",\"receiver_relative_point_speed_mps\":"
            << transferReceiverMotion.relativePointSpeed
            << ",\"receiver_relative_angular_speed_radps\":"
            << transferReceiverMotion.relativeAngularSpeed
            << ",\"thread_minimum_non_neighbour_surface_separation_m\":"
            << transferRod.minimumNonNeighbourSurfaceClearance
            << ",\"thread_maximum_node_speed_mps\":"
            << transferRod.maximumNodeSpeed
            << ",\"thread_maximum_edge_length_error_m\":"
            << transferRod.maximumEdgeLengthError
            << ",\"terminal_self_collision_clearance_qualified\":true"
            << ",\"self_collision_projector_enabled\":true"
            << ",\"self_collision_margin_m\":"
            << world.rods[0].stepConfig.selfCollisionMargin
            << ",\"thread_tool_collision_enabled\":true"
            << ",\"thread_tool_ccd_enabled\":true"
            << ",\"hard_swage_root_error_m\":"
            << finalSwageAttachmentError
            << ",\"swage_tangent_angle_error_rad\":"
            << finalSwageTangentError
            << ",\"swage_tangent_bending_strain\":"
            << world.rods[0].model.radius * finalSwageTangentError /
                world.rods[0].model.restLengths.front()
            << ",\"swage_tangent_bending_stress_pa\":"
            << swageTangentBendingStressPa(world, transferred.result)
            << ",\"swage_tangent_line_error_m\":"
            << swageTangentLineError(
                world,
                transferred.result
            )
            << ",\"overlap_giver_contacts\":"
            << overlapContacts.jawContacts[0][0] +
                   overlapContacts.jawContacts[0][1]
            << ",\"overlap_receiver_contacts\":"
            << overlapContacts.jawContacts[1][0] +
                   overlapContacts.jawContacts[1][1]
            << ",\"load_exchange_giver_contacts\":"
            << loadExchangeContacts.jawContacts[0][0] +
                   loadExchangeContacts.jawContacts[0][1]
            << ",\"load_exchange_receiver_contacts\":"
            << loadExchangeContacts.jawContacts[1][0] +
                   loadExchangeContacts.jawContacts[1][1]
            << ",\"cross_arm_contacts\":"
            << overlapContacts.crossArmContacts
            << ",\"terminal_max_friction_utilization\":"
            << transferContacts.maximumJawFrictionUtilization
            << ",\"terminal_min_rod_separation_m\":"
            << (
                std::isfinite(transferContacts.minimumRodSeparation)
                    ? transferContacts.minimumRodSeparation
                    : 0.0
            )
            << ",\"terminal_impulse_delta\":"
            << transferResidual.residuals.x
            << ",\"terminal_contact_velocity_residual_mps\":"
            << transferResidual.residuals.y
            << ",\"terminal_cone_violation\":"
            << transferResidual.residuals.z
            << ",\"terminal_contact_residual_aux\":"
            << transferResidual.residuals.w
            << ",\"successful_steps\":" << successfulSteps
            << ",\"failed_steps\":0"
            << ",\"gpu_ms\":" << totalGpuMilliseconds
            << ",\"resident_bytes\":" << stats.retainedBufferBytes
            << ",\"submissions\":" << stats.submissionCount
            << ",\"positive_control_overlap\":true"
            << ",\"coordinated_load_exchange\":true"
            << ",\"sole_receiver_control\":true}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "dual_psm_suture_handoff status=failed error=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
