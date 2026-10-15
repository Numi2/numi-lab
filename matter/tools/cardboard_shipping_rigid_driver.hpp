#pragma once

// Probe-local bridge for a native nonarticulated free body coupled through
// Matter's dynamic-proxy rows. The board remains an implicit FEM object; this
// header does not duplicate rigid integration or contact mechanics.

#include "metalrobo/EngineModel.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/metal_world.hpp"
#include "cardboard_shipping_roof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard {

struct ShippingRigidBodySpec {
    double massKg = 0.0;
    std::array<double, 3> halfExtentM{};
    std::array<double, 3> initialCenterM{};
    // Shipping protocol axes are x=length, y=upright height, z=width.
    std::array<double, 3> gravityMPerS2{0.0, -9.80665, 0.0};
    double timestepSeconds = 0.0;
};

inline MRABABodyWrenchGPU makeShippingVerticalBodyForceWrench(
    const double forceYNewtons) {
    if (!std::isfinite(forceYNewtons))
        throw std::invalid_argument(
            "shipping body-force wrench must be finite");
    const float forceY = static_cast<float>(forceYNewtons);
    if (!std::isfinite(forceY))
        throw std::invalid_argument(
            "shipping body-force wrench is outside the native FP32 range");
    MRABABodyWrenchGPU result{};
    result.force = {0.0f, forceY, 0.0f, 0.0f};
    result.torque = {0.0f, 0.0f, 0.0f, 0.0f};
    return result;
}

struct ShippingBodyWrenchTerms {
    double gravityForceYNewtons = 0.0;
    double actuatorForceYNewtons = 0.0;
    double totalExternalForceYNewtons = 0.0;
    MRABABodyWrenchGPU totalWrench{};
};

inline ShippingBodyWrenchTerms makeShippingBodyWrenchTerms(
    const double massKg,
    const double gravityYMetresPerSecondSquared,
    const double actuatorForceYNewtons) {
    if (!(std::isfinite(massKg) && massKg > 0.0) ||
        !std::isfinite(gravityYMetresPerSecondSquared) ||
        !std::isfinite(actuatorForceYNewtons))
        throw std::invalid_argument(
            "shipping body force terms require finite gravity, actuator, and positive mass");
    ShippingBodyWrenchTerms result;
    result.gravityForceYNewtons = massKg * gravityYMetresPerSecondSquared;
    result.actuatorForceYNewtons = actuatorForceYNewtons;
    result.totalExternalForceYNewtons = result.gravityForceYNewtons +
        result.actuatorForceYNewtons;
    if (!std::isfinite(result.totalExternalForceYNewtons))
        throw std::overflow_error("shipping total external force is nonfinite");
    result.totalWrench = makeShippingVerticalBodyForceWrench(
        result.totalExternalForceYNewtons);
    return result;
}

struct ShippingRigidWorldBundle {
    metalrobo::EngineModel model;
    metalrobo::HeterogeneousWorld world;
    metalrobo::CompiledWorld compiled;
    std::vector<MRBodyStateGPU> initialSceneBodies;
    std::vector<float> initialQ;
    std::vector<float> initialV;
    std::uint32_t toolSceneBodyCount = 0u;
    std::uint32_t freeBodyIndex = 0u;
    std::uint32_t compressionPlatenBodyIndex = 0u;
    std::uint32_t supportFloorBodyIndex = 0u;
};

struct ShippingBodyContactImpulse {
    std::array<double, 3> impulseNs{};
    std::uint32_t contactCount = 0u;
};

// Matter's accepted RuntimeStateSnapshot exposes the reaction impulse
// accumulated for every compiled rigid proxy during the control step. This is
// the evidence path for FEM-versus-tool contact; MetalWorld contactEvidence
// covers rigid-rigid pairs and is not a substitute for these values.
struct ShippingNativeBodyReaction {
    std::array<double, 3> impulseNs{};
    double contactSamples = 0.0;
    std::uint32_t matchedProxyCount = 0u;
};

struct ShippingRigidContactGate {
    double totalSamples = 0.0;
    double supportFloorSamples = 0.0;
    double forbiddenToolSamples = 0.0;
};

inline ShippingRigidContactGate classifyShippingRigidContactSamples(
    const std::span<const NMRigidReactionGPU> reactions,
    const std::span<const NMRigidProxyGPU> proxies,
    const std::uint32_t environmentCount,
    const std::uint32_t supportFloorBodyIndex = 3u) {
    if (environmentCount == 0u ||
        proxies.size() > std::numeric_limits<std::size_t>::max() /
            environmentCount ||
        reactions.size() != proxies.size() * environmentCount)
        throw std::invalid_argument(
            "shipping rigid-contact gate has mismatched proxy/reaction arenas");
    ShippingRigidContactGate result;
    for (std::uint32_t environment = 0u;
         environment < environmentCount; ++environment) {
        const std::size_t base = static_cast<std::size_t>(environment) *
            proxies.size();
        for (std::size_t proxyIndex = 0u;
             proxyIndex < proxies.size(); ++proxyIndex) {
            const double samples =
                reactions[base + proxyIndex].impulseAndCount.w;
            if (!(std::isfinite(samples) && samples >= 0.0))
                throw std::invalid_argument(
                    "shipping rigid-contact gate observed an invalid contact count");
            result.totalSamples += samples;
            if (proxies[proxyIndex].bodyIndex == supportFloorBodyIndex)
                result.supportFloorSamples += samples;
            else
                result.forbiddenToolSamples += samples;
        }
    }
    if (!std::isfinite(result.totalSamples) ||
        !std::isfinite(result.supportFloorSamples) ||
        !std::isfinite(result.forbiddenToolSamples))
        throw std::invalid_argument(
            "shipping rigid-contact gate overflowed its sample counters");
    return result;
}

struct ShippingSupportFloorPlacement {
    float centerY = 0.0f;
    float topY = 0.0f;
    double positiveClearanceM = 0.0;
};

inline std::array<double, 4> shippingSupportFloorOrientationXYWZ() {
    constexpr double halfSqrt2 = 0.707106781186547524400844362104849039;
    // Rotate the proxy's local +z thin axis onto world +y. This makes its top
    // face oppose the shipping gravity vector (0,-g,0).
    return {-halfSqrt2, 0.0, 0.0, halfSqrt2};
}

inline ShippingSupportFloorPlacement makeShippingNonpenetratingFloorPlacement(
    const double acceptedBoardMinimumY,
    const double floorHalfThicknessM,
    const double contactSlopM) {
    if (!(std::isfinite(acceptedBoardMinimumY) &&
          std::isfinite(floorHalfThicknessM) && floorHalfThicknessM > 0.0 &&
          std::isfinite(contactSlopM) && contactSlopM > 0.0))
        throw std::invalid_argument(
            "shipping floor placement needs finite surface, thickness and contact slop");
    const float boardMinimum = static_cast<float>(acceptedBoardMinimumY);
    const float halfThickness = static_cast<float>(floorHalfThicknessM);
    if (!(std::isfinite(boardMinimum) && std::isfinite(halfThickness) &&
          halfThickness > 0.0f))
        throw std::invalid_argument(
            "shipping floor placement is outside the native FP32 range");
    float center = boardMinimum - halfThickness;
    if (!std::isfinite(center))
        throw std::invalid_argument("shipping floor center is nonfinite");
    float top = center + halfThickness;
    // Keep the transformed support plane strictly below the accepted FEM
    // surface despite FP32 cancellation in center + halfThickness.
    for (unsigned adjustment = 0u; top >= boardMinimum && adjustment < 8u;
         ++adjustment) {
        center = std::nextafter(center,
            -std::numeric_limits<float>::infinity());
        top = center + halfThickness;
    }
    const double clearance = static_cast<double>(boardMinimum) - top;
    if (!(std::isfinite(top) && top < boardMinimum && clearance > 0.0 &&
          clearance < contactSlopM))
        throw std::invalid_argument(
            "shipping support floor cannot be placed below the board within contact slop");
    return {center, top, clearance};
}

struct ShippingMatterStepAcceptance {
    bool accepted = false;
    std::uint32_t statusCode = NM_STATUS_INVALID_DISPATCH;
    NMSolverCertificateGPU certificate{};
    std::string reason = "Matter status and certificate are unavailable";
};

inline ShippingMatterStepAcceptance inspectShippingMatterAcceptance(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    const numi::matter::CompiledWorld& compiled,
    const std::uint32_t environment = 0u) {
    ShippingMatterStepAcceptance result;
    if (!snapshot.available) {
        result.reason = "Matter completion-boundary snapshot is unavailable: " +
            snapshot.message;
        return result;
    }
    if (environment >= snapshot.statuses.size() ||
        environment >= snapshot.solverCertificates.size()) {
        result.reason = "Matter status or solver-certificate readback is truncated";
        return result;
    }
    const auto& status = snapshot.statuses[environment];
    const auto& certificate = snapshot.solverCertificates[environment];
    result.statusCode = status.code;
    result.certificate = certificate;
    const auto finite = [](const float value) { return std::isfinite(value); };
    const auto& tolerance = compiled.mixedSolver.residualTolerances;
    if (status.code != NM_STATUS_SUCCESS) {
        result.reason = "Matter rejected the native step with status " +
            std::to_string(status.code);
    } else if (!(certificate.validity.w > 0.5f) ||
               !finite(certificate.nonlinear.x) ||
               !finite(certificate.nonlinear.y) ||
               !finite(certificate.nonlinear.z) ||
               !finite(certificate.nonlinear.w)) {
        result.reason = "Matter solver certificate is missing, rejected, or nonfinite";
    } else if (!(tolerance.x > 0.0f && tolerance.y > 0.0f &&
                 tolerance.z > 0.0f) ||
               certificate.nonlinear.x > tolerance.x ||
               certificate.nonlinear.z > tolerance.y ||
               certificate.nonlinear.w > tolerance.z) {
        result.reason = "Matter accepted flag disagrees with the authored residual tolerances";
    } else {
        result.accepted = true;
        result.reason = "Matter status and solver certificate accepted";
    }
    return result;
}

struct ShippingRigidProxyProjectionError {
    double centerAndRadius = 0.0;
    double extent = 0.0;
    double orientation = 0.0;
    double linearVelocityAndInverseMass = 0.0;
    double angularVelocity = 0.0;
    double bodyCenter = 0.0;
    double inverseInertiaRows = 0.0;

    [[nodiscard]] double maximum() const noexcept {
        return std::max({centerAndRadius, extent, orientation,
                         linearVelocityAndInverseMass, angularVelocity,
                         bodyCenter, inverseInertiaRows});
    }
};

inline ShippingRigidProxyProjectionError compareShippingImpactProxyProjection(
    const NMRigidProxyGPU& proxy,
    const NMRigidStateGPU& projected,
    const MRBodyStateGPU& body) {
    if (proxy.shapeKind != NM_RIGID_BOX || proxy.bodyIndex != 4u ||
        (proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
        (proxy.flags & NM_RIGID_ARTICULATED) != 0u ||
        proxy.localCenterAndRadius.x != 0.0f ||
        proxy.localCenterAndRadius.y != 0.0f ||
        proxy.localCenterAndRadius.z != 0.0f ||
        proxy.localOrientation.x != 0.0f ||
        proxy.localOrientation.y != 0.0f ||
        proxy.localOrientation.z != 0.0f ||
        proxy.localOrientation.w != 1.0f)
        throw std::invalid_argument(
            "shipping proxy projection audit requires the authored axis-aligned free box");
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC)
        throw std::invalid_argument(
            "shipping impact proxy is not bound to a dynamic scene body");

    const std::array<float, 4> expectedCenter{
        body.position.x, body.position.y, body.position.z,
        proxy.localCenterAndRadius.w};
    const std::array<float, 4> expectedExtent{
        proxy.localExtent.x, proxy.localExtent.y,
        proxy.localExtent.z, proxy.localExtent.w};
    const std::array<float, 4> expectedOrientation{
        body.orientation.x, body.orientation.y,
        body.orientation.z, body.orientation.w};
    const std::array<float, 4> expectedLinear{
        body.linearVelocityAndInverseMass.x,
        body.linearVelocityAndInverseMass.y,
        body.linearVelocityAndInverseMass.z,
        body.linearVelocityAndInverseMass.w};
    const std::array<float, 4> expectedAngular{
        body.angularVelocity.x, body.angularVelocity.y,
        body.angularVelocity.z, body.angularVelocity.w};
    const std::array<float, 4> expectedBodyCenter{
        body.position.x, body.position.y, body.position.z, body.position.w};
    const std::array<std::array<float, 4>, 3> expectedInertia{{
        {body.inverseInertiaWorldRow0.x, body.inverseInertiaWorldRow0.y,
         body.inverseInertiaWorldRow0.z, body.inverseInertiaWorldRow0.w},
        {body.inverseInertiaWorldRow1.x, body.inverseInertiaWorldRow1.y,
         body.inverseInertiaWorldRow1.z, body.inverseInertiaWorldRow1.w},
        {body.inverseInertiaWorldRow2.x, body.inverseInertiaWorldRow2.y,
         body.inverseInertiaWorldRow2.z, body.inverseInertiaWorldRow2.w}}};
    const auto components = [](const nm_float4& value) {
        return std::array<float, 4>{value.x, value.y, value.z, value.w};
    };
    const auto maxDifference = [](const std::array<float, 4>& a,
                                  const std::array<float, 4>& b) {
        double result = 0.0;
        for (std::size_t index = 0u; index < 4u; ++index) {
            if (!std::isfinite(a[index]) || !std::isfinite(b[index]))
                return std::numeric_limits<double>::infinity();
            result = std::max(result, std::abs(
                static_cast<double>(a[index]) - b[index]));
        }
        return result;
    };

    ShippingRigidProxyProjectionError error;
    error.centerAndRadius = maxDifference(
        components(projected.centerAndRadius), expectedCenter);
    error.extent = maxDifference(components(projected.extent), expectedExtent);
    error.orientation = maxDifference(
        components(projected.orientation), expectedOrientation);
    error.linearVelocityAndInverseMass = maxDifference(
        components(projected.linearVelocityAndInverseMass), expectedLinear);
    error.angularVelocity = maxDifference(
        components(projected.angularVelocity), expectedAngular);
    error.bodyCenter = maxDifference(
        components(projected.bodyCenter), expectedBodyCenter);
    error.inverseInertiaRows = std::max({
        maxDifference(components(projected.inverseInertiaRow0), expectedInertia[0]),
        maxDifference(components(projected.inverseInertiaRow1), expectedInertia[1]),
        maxDifference(components(projected.inverseInertiaRow2), expectedInertia[2])});
    return error;
}

inline ShippingNativeBodyReaction accumulateShippingNativeBodyReaction(
    const std::span<const NMRigidReactionGPU> reactions,
    const std::span<const NMRigidProxyGPU> proxies,
    const std::uint32_t environmentCount,
    const std::uint32_t environment,
    const std::uint32_t bodyIndex) {
    if (environmentCount == 0u || environment >= environmentCount ||
        proxies.size() > std::numeric_limits<std::size_t>::max() / environmentCount ||
        reactions.size() != proxies.size() * environmentCount)
        throw std::invalid_argument(
            "shipping native reaction/proxy arenas do not match their environments");
    ShippingNativeBodyReaction result;
    const std::size_t base = static_cast<std::size_t>(environment) * proxies.size();
    for (std::size_t proxyIndex = 0u; proxyIndex < proxies.size(); ++proxyIndex) {
        if (proxies[proxyIndex].bodyIndex != bodyIndex) continue;
        const auto& reaction = reactions[base + proxyIndex];
        const std::array<double, 4> impulseAndCount{
            reaction.impulseAndCount.x, reaction.impulseAndCount.y,
            reaction.impulseAndCount.z, reaction.impulseAndCount.w};
        for (const double value : impulseAndCount)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "shipping native body reaction contains a nonfinite value");
        if (reaction.impulseAndCount.w < 0.0f)
            throw std::invalid_argument(
                "shipping native reaction contact-sample count is negative");
        result.impulseNs[0] += reaction.impulseAndCount.x;
        result.impulseNs[1] += reaction.impulseAndCount.y;
        result.impulseNs[2] += reaction.impulseAndCount.z;
        result.contactSamples += reaction.impulseAndCount.w;
        ++result.matchedProxyCount;
    }
    return result;
}

inline double shippingReactionForceYFromImpulse(
    const ShippingNativeBodyReaction& reaction,
    const double timestepSeconds) {
    if (!(std::isfinite(timestepSeconds) && timestepSeconds > 0.0) ||
        !std::isfinite(reaction.impulseNs[1]))
        throw std::invalid_argument(
            "shipping native reaction force requires a finite impulse and positive timestep");
    // Runtime reactions are impulses (N s), accumulated across the accepted
    // control step. Dividing by that exact step duration gives mean force (N).
    return reaction.impulseNs[1] / timestepSeconds;
}

struct ShippingRigidDynamicsObservation {
    double massKg = 0.0;
    std::array<double, 3> linearMomentumKgMPerS{};
    std::array<double, 3> angularMomentumKgM2PerS{};
    double translationalKineticEnergyJ = 0.0;
    double rotationalKineticEnergyJ = 0.0;
    double gravitationalPotentialRelativeJ = 0.0;

    [[nodiscard]] double kineticEnergyJ() const noexcept {
        return translationalKineticEnergyJ + rotationalKineticEnergyJ;
    }
    [[nodiscard]] double mechanicalEnergyRelativeJ() const noexcept {
        return kineticEnergyJ() + gravitationalPotentialRelativeJ;
    }
};

inline ShippingRigidDynamicsObservation observeShippingRigidDynamics(
    const MRBodyStateGPU& body,
    const std::array<double, 3>& gravityMPerS2,
    const std::array<double, 3>& referencePositionM) {
    const double inverseMass = body.linearVelocityAndInverseMass.w;
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
        !(std::isfinite(inverseMass) && inverseMass > 0.0f))
        throw std::invalid_argument(
            "shipping dynamics observation requires a finite-mass dynamic body");
    const std::array<double, 3> position{
        body.position.x, body.position.y, body.position.z};
    const std::array<double, 3> velocity{
        body.linearVelocityAndInverseMass.x,
        body.linearVelocityAndInverseMass.y,
        body.linearVelocityAndInverseMass.z};
    const std::array<double, 3> angularVelocity{
        body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z};
    std::array<std::array<double, 3>, 3> inverseInertia{{
        {body.inverseInertiaWorldRow0.x, body.inverseInertiaWorldRow0.y,
         body.inverseInertiaWorldRow0.z},
        {body.inverseInertiaWorldRow1.x, body.inverseInertiaWorldRow1.y,
         body.inverseInertiaWorldRow1.z},
        {body.inverseInertiaWorldRow2.x, body.inverseInertiaWorldRow2.y,
         body.inverseInertiaWorldRow2.z}}};
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        if (!std::isfinite(position[axis]) || !std::isfinite(velocity[axis]) ||
            !std::isfinite(angularVelocity[axis]) ||
            !std::isfinite(gravityMPerS2[axis]) ||
            !std::isfinite(referencePositionM[axis]))
            throw std::invalid_argument(
                "shipping dynamics observation contains a nonfinite component");
        for (std::size_t column = 0u; column < 3u; ++column)
            if (!std::isfinite(inverseInertia[axis][column]))
                throw std::invalid_argument(
                    "shipping dynamics observation has nonfinite inverse inertia");
    }
    for (std::size_t row = 0u; row < 3u; ++row)
        for (std::size_t column = row + 1u; column < 3u; ++column)
            if (std::abs(inverseInertia[row][column] -
                         inverseInertia[column][row]) > 2.0e-5 *
                    std::max({1.0, std::abs(inverseInertia[row][column]),
                              std::abs(inverseInertia[column][row])}))
                throw std::invalid_argument(
                    "shipping world inverse inertia is not symmetric");

    const double a = inverseInertia[0][0];
    const double b = inverseInertia[0][1];
    const double c = inverseInertia[0][2];
    const double d = inverseInertia[1][0];
    const double e = inverseInertia[1][1];
    const double f = inverseInertia[1][2];
    const double g = inverseInertia[2][0];
    const double h = inverseInertia[2][1];
    const double i = inverseInertia[2][2];
    const double determinant = a * (e * i - f * h) -
        b * (d * i - f * g) + c * (d * h - e * g);
    if (!(a > 0.0 && a * e - b * d > 0.0 && determinant > 0.0 &&
          std::isfinite(determinant)))
        throw std::invalid_argument(
            "shipping inverse inertia must be positive definite");
    const std::array<std::array<double, 3>, 3> inertia{{
        {(e * i - f * h) / determinant,
         (c * h - b * i) / determinant,
         (b * f - c * e) / determinant},
        {(f * g - d * i) / determinant,
         (a * i - c * g) / determinant,
         (c * d - a * f) / determinant},
        {(d * h - e * g) / determinant,
         (b * g - a * h) / determinant,
         (a * e - b * d) / determinant}}};

    ShippingRigidDynamicsObservation result;
    result.massKg = 1.0 / inverseMass;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        result.linearMomentumKgMPerS[axis] = result.massKg * velocity[axis];
        result.translationalKineticEnergyJ +=
            0.5 * result.massKg * velocity[axis] * velocity[axis];
        result.gravitationalPotentialRelativeJ -= result.massKg *
            gravityMPerS2[axis] * (position[axis] - referencePositionM[axis]);
        for (std::size_t column = 0u; column < 3u; ++column)
            result.angularMomentumKgM2PerS[axis] +=
                inertia[axis][column] * angularVelocity[column];
    }
    for (std::size_t axis = 0u; axis < 3u; ++axis)
        result.rotationalKineticEnergyJ += 0.5 * angularVelocity[axis] *
            result.angularMomentumKgM2PerS[axis];
    if (!(std::isfinite(result.massKg) &&
          std::isfinite(result.translationalKineticEnergyJ) &&
          std::isfinite(result.rotationalKineticEnergyJ) &&
          std::isfinite(result.gravitationalPotentialRelativeJ)) ||
        result.rotationalKineticEnergyJ < -1.0e-10)
        throw std::invalid_argument(
            "shipping dynamics observation produced invalid energy or mass");
    return result;
}

inline ShippingBodyContactImpulse accumulateShippingBodyContactImpulse(
    const std::span<const MRContactConstraintGPU> contacts,
    const std::uint32_t activeContactCount,
    const std::uint32_t bodyIndex) {
    ShippingBodyContactImpulse result;
    const std::size_t count = std::min<std::size_t>(
        contacts.size(), activeContactCount);
    for (std::size_t index = 0u; index < count; ++index) {
        const auto& contact = contacts[index];
        const bool isA = contact.bodyA == bodyIndex;
        const bool isB = contact.bodyB == bodyIndex;
        if (!isA && !isB) continue;
        const double sign = isA ? -1.0 : 1.0;
        const std::array<double, 3> normal{
            contact.normal.x, contact.normal.y, contact.normal.z};
        const std::array<double, 3> tangent{
            contact.tangent.x, contact.tangent.y, contact.tangent.z};
        const std::array<double, 3> tangentV{
            normal[1] * tangent[2] - normal[2] * tangent[1],
            normal[2] * tangent[0] - normal[0] * tangent[2],
            normal[0] * tangent[1] - normal[1] * tangent[0]};
        const std::array<double, 3> impulse{
            contact.impulses.x, contact.impulses.y, contact.impulses.z};
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            result.impulseNs[axis] += sign * (
                normal[axis] * impulse[0] +
                tangent[axis] * impulse[1] +
                tangentV[axis] * impulse[2]);
        ++result.contactCount;
    }
    return result;
}

inline MRBodyStateGPU makeShippingKinematicSceneBody(
    const std::array<double, 3>& position,
    const std::array<double, 4>& orientation = {0.0, 0.0, 0.0, 1.0}) {
    MRBodyStateGPU body{};
    body.position = {static_cast<float>(position[0]),
                     static_cast<float>(position[1]),
                     static_cast<float>(position[2]), 1.0f};
    double orientationNorm2 = 0.0;
    for (const double component : orientation) {
        if (!std::isfinite(component))
            throw std::invalid_argument("shipping tool orientation is nonfinite");
        orientationNorm2 += component * component;
    }
    if (!(orientationNorm2 > 1.0e-12))
        throw std::invalid_argument("shipping tool orientation is degenerate");
    const double inverseNorm = 1.0 / std::sqrt(orientationNorm2);
    body.orientation = {static_cast<float>(orientation[0] * inverseNorm),
                        static_cast<float>(orientation[1] * inverseNorm),
                        static_cast<float>(orientation[2] * inverseNorm),
                        static_cast<float>(orientation[3] * inverseNorm)};
    body.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
    body.flagsAndIndices[1] = MR_INVALID_INDEX;
    // Scene-only bodies are not bound to a model-global rigid-body row.
    body.flagsAndIndices[2] = MR_INVALID_INDEX;
    body.linearVelocityAndInverseMass = {0.0f, 0.0f, 0.0f, 0.0f};
    return body;
}

inline numi::matter::RigidProxySource makeShippingImpactorProxy(
    const ShippingRigidBodySpec& spec,
    const std::uint32_t bodyIndex = 4u,
    const std::uint32_t materialIndex = 0u) {
    if (bodyIndex < 4u)
        throw std::invalid_argument(
            "shipping impactor body slot must follow the legacy tool-body range");
    for (const double extent : spec.halfExtentM)
        if (!(std::isfinite(extent) && extent > 0.0))
            throw std::invalid_argument(
                "shipping impactor proxy half extents must be positive");
    numi::matter::RigidProxySource proxy;
    proxy.shape = NM_RIGID_BOX;
    proxy.bodyIndex = bodyIndex;
    proxy.sceneBodyIndex = bodyIndex;
    proxy.materialIndex = materialIndex;
    proxy.localCenter = {0.0, 0.0, 0.0};
    proxy.localExtent = {spec.halfExtentM[0],
                         spec.halfExtentM[1],
                         spec.halfExtentM[2]};
    proxy.localOrientation = {0.0, 0.0, 0.0, 1.0};
    proxy.articulated = false;
    proxy.dynamic = true;
    return proxy;
}

inline numi::matter::RigidProxySource makeShippingCompressionPlatenProxy(
    const std::array<double, 3>& halfExtentM,
    const std::uint32_t materialIndex = 0u,
    const std::uint32_t bodyIndex = 2u) {
    for (const double extent : halfExtentM)
        if (!(std::isfinite(extent) && extent > 0.0))
            throw std::invalid_argument(
                "shipping compression platen half extents must be positive");
    numi::matter::RigidProxySource proxy;
    proxy.shape = NM_RIGID_BOX;
    proxy.bodyIndex = bodyIndex;
    proxy.materialIndex = materialIndex;
    proxy.localExtent = {halfExtentM[0], halfExtentM[1], halfExtentM[2]};
    proxy.localOrientation = {0.0, 0.0, 0.0, 1.0};
    proxy.prescribedEndPoseTranslation = true;
    return proxy;
}

inline numi::matter::RigidProxySource makeShippingFloorProxy(
    const std::array<double, 2>& halfPlanExtentM,
    const double halfThicknessM,
    const std::uint32_t materialIndex = 0u,
    const std::uint32_t bodyIndex = 3u) {
    for (const double extent : halfPlanExtentM)
        if (!(std::isfinite(extent) && extent > 0.0))
            throw std::invalid_argument("shipping floor half extents must be positive");
    if (!(std::isfinite(halfThicknessM) && halfThicknessM > 0.0))
        throw std::invalid_argument("shipping floor thickness must be positive");
    numi::matter::RigidProxySource proxy;
    proxy.shape = NM_RIGID_BOX;
    proxy.bodyIndex = bodyIndex;
    proxy.materialIndex = materialIndex;
    // This local-z thin box is rotated onto the shipping x-z support plane by
    // scene body 3's accepted endpoint orientation.
    proxy.localExtent = {halfPlanExtentM[0], halfPlanExtentM[1], halfThicknessM};
    proxy.localOrientation = {0.0, 0.0, 0.0, 1.0};
    proxy.prescribedEndPoseTranslation = true;
    return proxy;
}

inline ShippingRigidWorldBundle makeShippingRigidWorld(
    const std::span<const MRBodyStateGPU> toolSceneBodies,
    const ShippingRigidBodySpec& spec) {
    if (toolSceneBodies.size() < 4u ||
        toolSceneBodies.size() >= std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(
            "shipping rigid world needs at least four tool scene bodies and a bounded count");
    if (!(std::isfinite(spec.massKg) && spec.massKg > 0.0) ||
        !(std::isfinite(spec.timestepSeconds) && spec.timestepSeconds > 0.0))
        throw std::invalid_argument("shipping rigid body mass and timestep must be positive");
    for (const double extent : spec.halfExtentM)
        if (!(std::isfinite(extent) && extent > 0.0))
            throw std::invalid_argument("shipping rigid-body half extents must be positive");
    for (const double coordinate : spec.initialCenterM)
        if (!std::isfinite(coordinate))
            throw std::invalid_argument("shipping rigid-body initial pose is nonfinite");
    for (const double gravity : spec.gravityMPerS2)
        if (!std::isfinite(gravity))
            throw std::invalid_argument("shipping rigid-body gravity is nonfinite");

    ShippingRigidWorldBundle result;
    result.model = metalrobo::makeFreeSphereEngineModel();
    auto& model = result.model;
    model.name = "cardboard_shipping_tools_and_finite_mass_impactor";
    const std::uint32_t toolBodyCount =
        static_cast<std::uint32_t>(toolSceneBodies.size());
    const std::uint32_t freeBodyIndex = toolBodyCount;
    model.world.bodyCount = toolBodyCount + 1u;
    model.world.articulationCount = 0u;
    model.world.jointCount = 0u;
    model.world.nq = 0u;
    model.world.nv = 0u;
    model.world.shapeCount = 0u;
    model.world.materialCount = 0u;
    model.world.gravityAndTimestep = {
        static_cast<float>(spec.gravityMPerS2[0]),
        static_cast<float>(spec.gravityMPerS2[1]),
        static_cast<float>(spec.gravityMPerS2[2]),
        static_cast<float>(spec.timestepSeconds)};
    model.shapes.clear();
    model.materials.clear();
    model.shapeNames.clear();
    model.articulations.clear();
    model.joints.clear();
    model.dofs.clear();
    model.defaultQ.clear();
    model.defaultV.clear();
    model.bodyNames.clear();
    model.jointNames.clear();
    model.dofNames.clear();
    model.bodies.assign(static_cast<std::size_t>(toolBodyCount) + 1u,
                        MRBodyPropertiesGPU{});
    for (std::uint32_t index = 0u; index < toolBodyCount; ++index) {
        MRBodyPropertiesGPU tool{};
        tool.articulationIndex = MR_INVALID_INDEX;
        tool.parentBody = MR_INVALID_INDEX;
        tool.inboundJoint = MR_INVALID_INDEX;
        tool.motionType = MR_MOTION_KINEMATIC;
        tool.dampingAndSpeedLimits = {0.0f, 0.0f, 1.0e6f, 1.0e6f};
        model.bodies[index] = tool;
    }
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = MR_MOTION_DYNAMIC;
    const double mass = spec.massKg;
    const double ix = mass * (spec.halfExtentM[1] * spec.halfExtentM[1] +
                              spec.halfExtentM[2] * spec.halfExtentM[2]) / 3.0;
    const double iy = mass * (spec.halfExtentM[0] * spec.halfExtentM[0] +
                              spec.halfExtentM[2] * spec.halfExtentM[2]) / 3.0;
    const double iz = mass * (spec.halfExtentM[0] * spec.halfExtentM[0] +
                              spec.halfExtentM[1] * spec.halfExtentM[1]) / 3.0;
    if (!(ix > 0.0 && iy > 0.0 && iz > 0.0) ||
        !std::isfinite(ix) || !std::isfinite(iy) || !std::isfinite(iz))
        throw std::invalid_argument("shipping rigid-body inertia is invalid");
    const auto fp32Positive = [](const double value) {
        const float narrowed = static_cast<float>(value);
        return std::isfinite(narrowed) && narrowed > 0.0f;
    };
    if (!fp32Positive(mass) || !fp32Positive(1.0 / mass) ||
        !fp32Positive(ix) || !fp32Positive(iy) || !fp32Positive(iz) ||
        !fp32Positive(1.0 / ix) || !fp32Positive(1.0 / iy) ||
        !fp32Positive(1.0 / iz))
        throw std::invalid_argument(
            "shipping rigid-body mass or inertia is outside positive FP32 range");
    for (const double extent : spec.halfExtentM)
        if (!fp32Positive(extent))
            throw std::invalid_argument(
                "shipping rigid-body extent is outside positive FP32 range");
    for (const double coordinate : spec.initialCenterM)
        if (!std::isfinite(static_cast<float>(coordinate)))
            throw std::invalid_argument(
                "shipping rigid-body pose is outside the FP32 range");
    body.massAndInverseMass = {static_cast<float>(mass),
                               static_cast<float>(1.0 / mass), 0.0f, 0.0f};
    body.centerOfMass = {0.0f, 0.0f, 0.0f, 0.0f};
    body.inertiaRow0 = {static_cast<float>(ix), 0.0f, 0.0f, 0.0f};
    body.inertiaRow1 = {0.0f, static_cast<float>(iy), 0.0f, 0.0f};
    body.inertiaRow2 = {0.0f, 0.0f, static_cast<float>(iz), 0.0f};
    body.inverseInertiaRow0 = {static_cast<float>(1.0 / ix), 0.0f, 0.0f, 0.0f};
    body.inverseInertiaRow1 = {0.0f, static_cast<float>(1.0 / iy), 0.0f, 0.0f};
    body.inverseInertiaRow2 = {0.0f, 0.0f, static_cast<float>(1.0 / iz), 0.0f};
    // The native free-body candidate owns all six rigid DOFs. Engine speed
    // caps alter that candidate after the coupled solve, so disable them for
    // the dynamic shipping body. Kinematic tool caps are independent.
    body.dampingAndSpeedLimits = {0.0f, 0.0f, 0.0f, 0.0f};
    model.bodies[freeBodyIndex] = body;
    result.initialQ.clear();
    result.initialV.clear();
    result.initialSceneBodies.assign(toolSceneBodies.begin(), toolSceneBodies.end());
    for (MRBodyStateGPU& toolBody : result.initialSceneBodies)
        toolBody.flagsAndIndices[2] = MR_INVALID_INDEX;
    MRBodyStateGPU freeBody{};
    freeBody.position = {static_cast<float>(spec.initialCenterM[0]),
                         static_cast<float>(spec.initialCenterM[1]),
                         static_cast<float>(spec.initialCenterM[2]), 1.0f};
    freeBody.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    freeBody.linearVelocityAndInverseMass = {
        0.0f, 0.0f, 0.0f, static_cast<float>(1.0 / mass)};
    freeBody.inverseInertiaWorldRow0 = body.inverseInertiaRow0;
    freeBody.inverseInertiaWorldRow1 = body.inverseInertiaRow1;
    freeBody.inverseInertiaWorldRow2 = body.inverseInertiaRow2;
    freeBody.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    freeBody.flagsAndIndices[1] = MR_INVALID_INDEX;
    freeBody.flagsAndIndices[2] = MR_INVALID_INDEX;
    result.initialSceneBodies.push_back(freeBody);
    result.toolSceneBodyCount = toolBodyCount;
    result.freeBodyIndex = freeBodyIndex;
    result.compressionPlatenBodyIndex = toolBodyCount - 2u;
    result.supportFloorBodyIndex = toolBodyCount - 1u;

    const metalrobo::HeterogeneousWorldComponent component{
        .model = &model,
        .instanceId = "cardboard-shipping-rigid-world",
        .defaultSceneBodies = result.initialSceneBodies};
    metalrobo::EngineModelComposeConfig composeConfig;
    composeConfig.name = model.name;
    composeConfig.gravityAndTimestep = model.world.gravityAndTimestep;
    composeConfig.solverType = model.world.solverType;
    const auto composed = metalrobo::composeHeterogeneousWorld(
        std::span<const metalrobo::HeterogeneousWorldComponent>(&component, 1u),
        {}, result.world, composeConfig);
    if (!composed.succeeded())
        throw std::runtime_error("shipping rigid world composition failed: " +
                                 composed.message);
    const auto compiled = metalrobo::compileMetalWorld(result.world, result.compiled);
    if (!compiled.succeeded())
        throw std::runtime_error("shipping rigid world compile failed: " +
                                 compiled.message);
    if (result.compiled.nq() != 0u || result.compiled.nv() != 0u ||
        result.compiled.bodyCount() != toolBodyCount + 1u ||
        result.compiled.sceneBodyCount() != toolBodyCount + 1u ||
        result.compiled.articulationCount() != 0u)
        throw std::runtime_error("shipping rigid world did not compile to its tool scene bodies and one nonarticulated 6-DoF free body");
    return result;
}

inline std::string shippingRigidBranchGeometryMismatchReason(
    const std::span<const NMRigidProxyGPU> compiledMatterProxies,
    const ShippingRigidWorldBundle& branch,
    const ShippingRigidBodySpec& spec) {
    if (!branch.compiled.valid() || branch.compiled.nq() != 0u ||
        branch.compiled.nv() != 0u ||
        branch.freeBodyIndex >= branch.initialSceneBodies.size() ||
        branch.freeBodyIndex >= branch.model.bodies.size())
        return "shipping impact branch lacks its expected scene-only free body";
    const NMRigidProxyGPU* impactor = nullptr;
    for (const auto& proxy : compiledMatterProxies) {
        if (proxy.bodyIndex != branch.freeBodyIndex) continue;
        if (impactor != nullptr)
            return "shipping impact branch has more than one Matter proxy for its free body";
        impactor = &proxy;
    }
    if (impactor == nullptr)
        return "shipping impact branch has no compiled Matter proxy for its free body";
    if (const std::string mismatch = shippingImpactorProxyMismatchReason(
            *impactor, branch.freeBodyIndex, spec.halfExtentM);
        !mismatch.empty())
        return mismatch;

    const MRBodyStateGPU& scene = branch.initialSceneBodies[branch.freeBodyIndex];
    const MRBodyPropertiesGPU& properties = branch.model.bodies[branch.freeBodyIndex];
    const auto fp = [](const double value) {
        return static_cast<float>(value);
    };
    const double mass = spec.massKg;
    const double ix = mass * (spec.halfExtentM[1] * spec.halfExtentM[1] +
                              spec.halfExtentM[2] * spec.halfExtentM[2]) / 3.0;
    const double iy = mass * (spec.halfExtentM[0] * spec.halfExtentM[0] +
                              spec.halfExtentM[2] * spec.halfExtentM[2]) / 3.0;
    const double iz = mass * (spec.halfExtentM[0] * spec.halfExtentM[0] +
                              spec.halfExtentM[1] * spec.halfExtentM[1]) / 3.0;
    if (scene.flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
        scene.flagsAndIndices[2] != MR_INVALID_INDEX ||
        properties.motionType != MR_MOTION_DYNAMIC ||
        scene.position.x != fp(spec.initialCenterM[0]) ||
        scene.position.y != fp(spec.initialCenterM[1]) ||
        scene.position.z != fp(spec.initialCenterM[2]) ||
        scene.orientation.x != 0.0f || scene.orientation.y != 0.0f ||
        scene.orientation.z != 0.0f || scene.orientation.w != 1.0f ||
        scene.linearVelocityAndInverseMass.x != 0.0f ||
        scene.linearVelocityAndInverseMass.y != 0.0f ||
        scene.linearVelocityAndInverseMass.z != 0.0f ||
        scene.linearVelocityAndInverseMass.w != fp(1.0 / mass))
        return "shipping MetalWorld scene pose, orientation, or mass differs from the explicit impactor spec";
    if (properties.massAndInverseMass.x != fp(mass) ||
        properties.massAndInverseMass.y != fp(1.0 / mass) ||
        properties.inertiaRow0.x != fp(ix) ||
        properties.inertiaRow1.y != fp(iy) ||
        properties.inertiaRow2.z != fp(iz) ||
        scene.inverseInertiaWorldRow0.x != fp(1.0 / ix) ||
        scene.inverseInertiaWorldRow1.y != fp(1.0 / iy) ||
        scene.inverseInertiaWorldRow2.z != fp(1.0 / iz))
        return "shipping MetalWorld mass/inertia does not match the compiled BOX geometry and branch mass";
    return {};
}

struct ShippingMatterBridgeContext {
    numi::matter::Runtime* runtime = nullptr;
    void* femTargets = nullptr;
    std::uint32_t femTargetCount = 0u;
    // Borrowed environment-major MRABABodyWrenchGPU input. The owner API
    // consumes it transactionally in Matter preDynamics; the MTLBuffer remains
    // owned by NativeRun for the complete MetalWorld transaction.
    void* externalWrenches = nullptr;
    std::uint32_t externalWrenchCount = 0u;
    std::uint32_t externalWrenchStride = 0u;
    void* cohesiveActivationCommands = nullptr;
    std::uint32_t cohesiveActivationCommandCount = 0u;
    std::uint32_t controlStepBase = 0u;
};

inline bool encodeShippingMatterBridge(
    void* context,
    const metalrobo::MetalWorldDevicePhysicsPass& pass) {
    auto* bridge = static_cast<ShippingMatterBridgeContext*>(context);
    if (bridge == nullptr || bridge->runtime == nullptr ||
        pass.commandBuffer == nullptr || pass.physicsSubsteps != 1u ||
        pass.physicsSubstep != 0u || !(pass.timestepSeconds > 0.0f))
        return false;
    numi::matter::EncodeRequest request{};
    request.commandBuffer = pass.commandBuffer;
    request.rigid.q = pass.q;
    request.rigid.v = pass.v;
    request.rigid.currentBodies = pass.currentBodies;
    request.rigid.bodyWrenches = pass.bodyWrenches;
    request.rigid.externalWrenches = bridge->externalWrenches;
    request.rigid.externalWrenchCount = bridge->externalWrenchCount;
    request.rigid.externalWrenchStride = bridge->externalWrenchStride;
    request.rigid.sceneBodies = pass.sceneBodies;
    request.rigid.currentBodyCount = pass.bodyCount;
    request.rigid.currentBodyStride = pass.bodyStateStride;
    request.rigid.bodyWrenchCount = pass.bodyCount;
    request.rigid.bodyWrenchStride = pass.bodyWrenchStride;
    request.rigid.sceneBodyCount = pass.sceneBodyCount;
    request.rigid.sceneStride = pass.sceneBodyStride;
    request.rigid.qStride = pass.qStride;
    request.rigid.vStride = pass.nv;
    request.environmentStatuses = pass.environmentStatuses;
    request.rigidContactConstraints = pass.contactConstraints;
    request.rigidContactStatuses = pass.contactStatuses;
    request.rigidContactConstraintStride = pass.contactConstraintStride;
    request.articulationRootBody = pass.articulationRootBody;
    request.controlStep = bridge->controlStepBase + pass.controlStep;
    request.physicsSubstep = 0u;
    request.physicsSubsteps = 1u;
    request.rigidWorldPhysicsSubstep = pass.physicsSubstep;
    request.seed = pass.seed;
    request.timestepSeconds = bridge->runtime->timestepSeconds();
    if (pass.timestepSeconds > 0.0f && std::isfinite(pass.timestepSeconds))
        request.timestepSeconds = pass.timestepSeconds;
    request.phase = pass.phase ==
        metalrobo::MetalWorldDevicePhysicsPhase::postCommit
        ? numi::matter::EncodePhase::postCommit
        : numi::matter::EncodePhase::preDynamics;
    if (request.phase == numi::matter::EncodePhase::preDynamics) {
        request.femKinematicTargets = bridge->femTargets;
        request.femKinematicTargetCount = bridge->femTargetCount;
        request.cohesiveActivationCommands = bridge->cohesiveActivationCommands;
        request.cohesiveActivationCommandCount =
            bridge->cohesiveActivationCommandCount;
        request.cohesiveActivationCommandStride =
            bridge->cohesiveActivationCommandCount == 0u
                ? 0u : sizeof(NMCohesiveActivationCommandGPU);
    }
    request.runAdaptiveTransfer = false;
    return bridge->runtime->encode(request).encoded;
}

inline void abortShippingMatterBridge(void* context, void* commandBuffer) {
    auto* bridge = static_cast<ShippingMatterBridgeContext*>(context);
    if (bridge != nullptr && bridge->runtime != nullptr && commandBuffer != nullptr)
        bridge->runtime->cancel(commandBuffer);
}

inline metalrobo::MetalWorldDevicePhysicsProgram makeShippingMatterBridgeProgram(
    numi::matter::Runtime& runtime,
    ShippingMatterBridgeContext& context) {
    auto program = numi::matter::makeMetalWorldDevicePhysicsProgram(runtime);
    if (!program.valid())
        throw std::runtime_error("Matter Runtime cannot supply a MetalWorld device-physics program");
    context.runtime = &runtime;
    program.context = &context;
    program.encode = &encodeShippingMatterBridge;
    program.abort = &abortShippingMatterBridge;
    program.flags |=
        metalrobo::MetalWorldDevicePhysicsWritesBodyWrenches |
        metalrobo::MetalWorldDevicePhysicsOwnsFreeBodyCandidate;
    // Bind the adapter contract itself into MetalWorld's execution fingerprint.
    constexpr std::uint64_t bridgeRevision = 0x43415244424f4152ull;
    program.fingerprint ^= bridgeRevision;
    if (program.fingerprint == 0u) program.fingerprint = bridgeRevision;
    return program;
}

class ShippingRigidWorldRunner {
public:
    ShippingRigidWorldRunner(ShippingRigidWorldBundle bundle,
                             numi::matter::Runtime& runtime,
                             const float timestepSeconds,
                             const numi::matter::CompiledWorld& matterWorld)
        : bundle_(std::move(bundle)), runtime_(runtime),
          matterWorld_(matterWorld),
          context_(metalrobo::MetalWorldConfig{}) {
        if (!(std::isfinite(timestepSeconds) && timestepSeconds > 0.0f))
            throw std::invalid_argument("shipping rigid runner timestep must be positive");
        defaultTimestepSeconds_ = timestepSeconds;
        if (runtime_.sourcePhysicsFingerprint() != matterWorld_.physicsFingerprint ||
            runtime_.environmentCount() != matterWorld_.dispatch.environmentCount)
            throw std::invalid_argument(
                "shipping rigid runner received a Matter world that differs from its initialized runtime");
        stepConfig_.timestepSeconds = timestepSeconds;
        stepConfig_.physicsSubsteps = 1u;
        stepConfig_.solverMode = metalrobo::MetalWorldSolverMode::freeMotionABA;
        stepConfig_.ccdMode = metalrobo::MetalWorldCCDMode::disabled;
        stepConfig_.applyBodyDamping = false;
        stepConfig_.devicePhysicsProgram =
            makeShippingMatterBridgeProgram(runtime_, bridge_);
        stepConfig_.publishFinalState = true;
        stepConfig_.publishStateTrajectory = false;
        stepConfig_.warmStart = false;
        q_ = bundle_.initialQ;
        v_ = bundle_.initialV;
        sceneBodies_ = bundle_.initialSceneBodies;
    }

    struct StepResult {
        metalrobo::MetalWorldDiagnostics diagnostics;
        metalrobo::MetalWorldResult result;
        ShippingMatterStepAcceptance matterAcceptance;
        ShippingBodyContactImpulse compressionPlaten{};
        ShippingBodyContactImpulse impactor{};
        std::uint32_t activeContactCount = 0u;
    };

    StepResult step(const std::span<const MRBodyStateGPU> endToolSceneBodies,
                    void* femTargets,
                    const std::uint32_t femTargetCount,
                    void* activationCommands,
                    const std::uint32_t activationCommandCount,
                    void* externalWrenches,
                    const std::uint32_t externalWrenchCount,
                    const std::uint32_t externalWrenchStride,
                    const std::uint32_t controlStepBase,
                    const float timestepSeconds = 0.0f,
                    const bool captureContactEvidence = false) {
        if (endToolSceneBodies.size() != bundle_.toolSceneBodyCount ||
            sceneBodies_.size() !=
                static_cast<std::size_t>(bundle_.toolSceneBodyCount) + 1u ||
            bundle_.compiled.sceneBodyCount() != sceneBodies_.size() ||
            bundle_.freeBodyIndex != bundle_.toolSceneBodyCount)
            throw std::invalid_argument(
                "shipping rigid step requires all tool targets and its appended dynamic body state");
        const float stepTimestep = timestepSeconds > 0.0f
            ? timestepSeconds : defaultTimestepSeconds_;
        if (!(std::isfinite(stepTimestep) && stepTimestep > 0.0f))
            throw std::invalid_argument("shipping rigid step timestep must be positive");
        stepConfig_.timestepSeconds = stepTimestep;
        stepConfig_.captureContactEvidence = captureContactEvidence;
        bridge_.femTargets = femTargets;
        bridge_.femTargetCount = femTargetCount;
        bridge_.externalWrenches = externalWrenches;
        bridge_.externalWrenchCount = externalWrenchCount;
        bridge_.externalWrenchStride = externalWrenchStride;
        bridge_.cohesiveActivationCommands = activationCommands;
        bridge_.cohesiveActivationCommandCount = activationCommandCount;
        bridge_.controlStepBase = controlStepBase;
        metalrobo::MetalWorldBatch batch;
        batch.environmentCount = 1u;
        batch.controlStepCount = 1u;
        batch.initialQ = q_;
        batch.initialV = v_;
        batch.initialSceneBodies = sceneBodies_;
        std::vector<MRBodyStateGPU> targetSceneBodies(
            endToolSceneBodies.begin(), endToolSceneBodies.end());
        // The caller controls only the four tool endpoints. MetalWorld owns
        // the finite-mass body's current and accepted endpoint state.
        targetSceneBodies.push_back(sceneBodies_.at(bundle_.freeBodyIndex));
        batch.kinematicTargets = targetSceneBodies;
        StepResult step;
        step.diagnostics = context_.run(bundle_.compiled, batch, stepConfig_, step.result);
        if (!step.diagnostics.succeeded()) return step;
        step.matterAcceptance = inspectShippingMatterAcceptance(
            runtime_.snapshot(), matterWorld_);
        // MetalWorld may finish its transaction while an embedded Matter
        // environment reports a rejected candidate. Never publish that
        // candidate scene pose into the caller's accepted continuation.
        if (!step.matterAcceptance.accepted) return step;
        if (captureContactEvidence) {
            const std::uint32_t activeContacts =
                step.result.contactStatuses.empty() ? 0u :
                step.result.contactStatuses.front().activeContacts;
            step.activeContactCount = activeContacts;
            step.compressionPlaten = accumulateShippingBodyContactImpulse(
                step.result.contactEvidence.contacts, activeContacts,
                bundle_.compressionPlatenBodyIndex);
            step.impactor = accumulateShippingBodyContactImpulse(
                step.result.contactEvidence.contacts, activeContacts,
                bundle_.freeBodyIndex);
        }
        if (step.result.finalQ.size() != bundle_.compiled.nq() ||
            step.result.finalV.size() != bundle_.compiled.nv() ||
            step.result.finalSceneBodies.size() != bundle_.compiled.sceneBodyCount())
            throw std::runtime_error("MetalWorld accepted result omitted coupled state publication");
        q_ = step.result.finalQ;
        v_ = step.result.finalV;
        sceneBodies_ = step.result.finalSceneBodies;
        ++acceptedSteps_;
        return step;
    }

    void resetRigidState() {
        q_ = bundle_.initialQ;
        v_ = bundle_.initialV;
        sceneBodies_ = bundle_.initialSceneBodies;
    }

    void setFreeBodyState(const std::array<double, 3>& center,
                          const std::array<double, 4>& orientation =
                              {0.0, 0.0, 0.0, 1.0},
                          const std::array<double, 3>& linearVelocity = {},
                          const std::array<double, 3>& angularVelocity = {}) {
        if (sceneBodies_.size() !=
                static_cast<std::size_t>(bundle_.toolSceneBodyCount) + 1u ||
            bundle_.freeBodyIndex >= sceneBodies_.size())
            throw std::logic_error("shipping free-body scene state has invalid dimensions");
        double norm2 = 0.0;
        for (const double value : orientation) {
            if (!std::isfinite(value))
                throw std::invalid_argument("shipping free-body orientation is nonfinite");
            norm2 += value * value;
        }
        if (!(norm2 > 1.0e-12))
            throw std::invalid_argument("shipping free-body orientation is degenerate");
        for (const double value : center)
            if (!std::isfinite(value))
                throw std::invalid_argument("shipping free-body center is nonfinite");
        for (const double value : linearVelocity)
            if (!std::isfinite(value))
                throw std::invalid_argument("shipping free-body linear velocity is nonfinite");
        for (const double value : angularVelocity)
            if (!std::isfinite(value))
                throw std::invalid_argument("shipping free-body angular velocity is nonfinite");
        const double inverseNorm = 1.0 / std::sqrt(norm2);
        MRBodyStateGPU& freeBody = sceneBodies_[bundle_.freeBodyIndex];
        freeBody.position = {static_cast<float>(center[0]),
                             static_cast<float>(center[1]),
                             static_cast<float>(center[2]), 1.0f};
        freeBody.orientation = {
            static_cast<float>(orientation[0] * inverseNorm),
            static_cast<float>(orientation[1] * inverseNorm),
            static_cast<float>(orientation[2] * inverseNorm),
            static_cast<float>(orientation[3] * inverseNorm)};
        freeBody.linearVelocityAndInverseMass = {
            static_cast<float>(linearVelocity[0]),
            static_cast<float>(linearVelocity[1]),
            static_cast<float>(linearVelocity[2]),
            freeBody.linearVelocityAndInverseMass.w};
        freeBody.angularVelocity = {static_cast<float>(angularVelocity[0]),
                                    static_cast<float>(angularVelocity[1]),
                                    static_cast<float>(angularVelocity[2]),
                                    freeBody.angularVelocity.w};
        freeBody.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    }

    void restoreAcceptedSceneBodies(
        const std::span<const MRBodyStateGPU> acceptedSceneBodies) {
        if (acceptedSceneBodies.size() != bundle_.compiled.sceneBodyCount() ||
            acceptedSceneBodies.size() != sceneBodies_.size() ||
            bundle_.freeBodyIndex >= acceptedSceneBodies.size())
            throw std::invalid_argument(
                "shipping rigid checkpoint scene-body count does not match the compiled world");
        for (const MRBodyStateGPU& state : acceptedSceneBodies) {
            const std::array<float, 28> values{
                state.position.x, state.position.y, state.position.z,
                state.position.w, state.orientation.x, state.orientation.y,
                state.orientation.z, state.orientation.w,
                state.linearVelocityAndInverseMass.x,
                state.linearVelocityAndInverseMass.y,
                state.linearVelocityAndInverseMass.z,
                state.linearVelocityAndInverseMass.w,
                state.angularVelocity.x, state.angularVelocity.y,
                state.angularVelocity.z, state.angularVelocity.w,
                state.inverseInertiaWorldRow0.x,
                state.inverseInertiaWorldRow0.y,
                state.inverseInertiaWorldRow0.z,
                state.inverseInertiaWorldRow0.w,
                state.inverseInertiaWorldRow1.x,
                state.inverseInertiaWorldRow1.y,
                state.inverseInertiaWorldRow1.z,
                state.inverseInertiaWorldRow1.w,
                state.inverseInertiaWorldRow2.x,
                state.inverseInertiaWorldRow2.y,
                state.inverseInertiaWorldRow2.z,
                state.inverseInertiaWorldRow2.w};
            for (const float value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "shipping rigid checkpoint scene state contains a nonfinite value");
        }
        const MRBodyStateGPU& freeBody =
            acceptedSceneBodies[bundle_.freeBodyIndex];
        const MRBodyPropertiesGPU& properties =
            bundle_.model.bodies.at(bundle_.freeBodyIndex);
        const double mass = properties.massAndInverseMass.x;
        const double inverseMass = freeBody.linearVelocityAndInverseMass.w;
        if (freeBody.flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
            !(mass > 0.0) || !(inverseMass > 0.0) ||
            std::abs(mass * inverseMass - 1.0) > 2.0e-5)
            throw std::invalid_argument(
                "shipping rigid checkpoint mass differs from the compiled branch body");
        const std::array<mr_float4, 3> expected{
            properties.inverseInertiaRow0,
            properties.inverseInertiaRow1,
            properties.inverseInertiaRow2};
        const std::array<mr_float4, 3> actual{
            freeBody.inverseInertiaWorldRow0,
            freeBody.inverseInertiaWorldRow1,
            freeBody.inverseInertiaWorldRow2};
        for (std::size_t row = 0u; row < expected.size(); ++row) {
            const std::array<double, 4> error{
                static_cast<double>(expected[row].x) - actual[row].x,
                static_cast<double>(expected[row].y) - actual[row].y,
                static_cast<double>(expected[row].z) - actual[row].z,
                static_cast<double>(expected[row].w) - actual[row].w};
            for (const double component : error)
                if (std::abs(component) > 2.0e-5 *
                        std::max(1.0, std::abs(static_cast<double>(expected[row].x))))
                    throw std::invalid_argument(
                        "shipping rigid checkpoint inertia differs from the compiled branch body");
        }
        sceneBodies_.assign(acceptedSceneBodies.begin(),
                            acceptedSceneBodies.end());
    }

    [[nodiscard]] const std::vector<float>& q() const noexcept { return q_; }
    [[nodiscard]] const std::vector<float>& v() const noexcept { return v_; }
    [[nodiscard]] const std::vector<MRBodyStateGPU>& sceneBodies() const noexcept {
        return sceneBodies_;
    }
    [[nodiscard]] double freeBodyMassKg() const noexcept {
        return bundle_.freeBodyIndex < bundle_.model.bodies.size()
            ? bundle_.model.bodies[bundle_.freeBodyIndex].massAndInverseMass.x : 0.0;
    }
    [[nodiscard]] std::uint32_t freeBodyIndex() const noexcept {
        return bundle_.freeBodyIndex;
    }
    [[nodiscard]] std::uint32_t toolSceneBodyCount() const noexcept {
        return bundle_.toolSceneBodyCount;
    }
    [[nodiscard]] std::uint32_t compressionPlatenBodyIndex() const noexcept {
        return bundle_.compressionPlatenBodyIndex;
    }
    [[nodiscard]] std::uint32_t supportFloorBodyIndex() const noexcept {
        return bundle_.supportFloorBodyIndex;
    }
    [[nodiscard]] std::uint64_t worldFingerprint() const noexcept {
        return bundle_.compiled.fingerprint();
    }
    [[nodiscard]] std::array<double, 3> gravityMPerS2() const noexcept {
        const auto& gravity = bundle_.model.world.gravityAndTimestep;
        return {gravity.x, gravity.y, gravity.z};
    }
    [[nodiscard]] std::uint64_t acceptedSteps() const noexcept { return acceptedSteps_; }
    [[nodiscard]] const metalrobo::CompiledWorld& compiled() const noexcept {
        return bundle_.compiled;
    }

private:
    ShippingRigidWorldBundle bundle_;
    numi::matter::Runtime& runtime_;
    const numi::matter::CompiledWorld& matterWorld_;
    ShippingMatterBridgeContext bridge_{};
    metalrobo::MetalWorldContext context_;
    metalrobo::MetalWorldStepConfig stepConfig_{};
    float defaultTimestepSeconds_ = 0.0f;
    std::vector<float> q_;
    std::vector<float> v_;
    std::vector<MRBodyStateGPU> sceneBodies_;
    std::uint64_t acceptedSteps_ = 0u;
};

} // namespace numi::cardboard
