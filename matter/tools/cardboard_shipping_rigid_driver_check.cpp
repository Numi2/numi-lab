#include "cardboard_shipping_rigid_driver.hpp"
#include "cardboard_shipping_crush.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace numi::cardboard;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ShippingRigidBodySpec fixtureSpec() {
    ShippingRigidBodySpec spec;
    spec.massKg = 0.75;
    spec.halfExtentM = {0.05, 0.025, 0.05};
    spec.initialCenterM = {0.15, 100.0, 0.45};
    spec.gravityMPerS2 = {0.0, -9.80665, 0.0};
    spec.timestepSeconds = 25.0e-6;
    return spec;
}

void testCompositionAndFiniteMassProxy() {
    const std::array<MRBodyStateGPU, 4> tools{
        makeShippingKinematicSceneBody({0.0, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.0, 0.0, -1.0}),
        makeShippingKinematicSceneBody({0.0, 100.0, 0.0}),
        makeShippingKinematicSceneBody({0.0, -100.0, 0.0})};
    const auto spec = fixtureSpec();
    const auto bundle = makeShippingRigidWorld(tools, spec);
    require(bundle.compiled.valid() && bundle.compiled.nq() == 0u &&
            bundle.compiled.nv() == 0u && bundle.compiled.bodyCount() == 5u &&
            bundle.compiled.sceneBodyCount() == 5u &&
            bundle.compiled.articulationCount() == 0u,
            "shipping rigid world must use one nonarticulated free body");
    require(bundle.initialQ.empty() && bundle.initialV.empty() &&
            bundle.initialSceneBodies.size() == 5u &&
            bundle.initialSceneBodies[0].flagsAndIndices[0] == MR_MOTION_KINEMATIC &&
            bundle.initialSceneBodies[0].flagsAndIndices[2] == MR_INVALID_INDEX &&
            bundle.initialSceneBodies[1].flagsAndIndices[0] == MR_MOTION_KINEMATIC &&
            bundle.initialSceneBodies[1].flagsAndIndices[2] == MR_INVALID_INDEX &&
            bundle.initialSceneBodies[2].flagsAndIndices[0] == MR_MOTION_KINEMATIC &&
            bundle.initialSceneBodies[2].flagsAndIndices[2] == MR_INVALID_INDEX &&
            bundle.initialSceneBodies[3].flagsAndIndices[0] == MR_MOTION_KINEMATIC &&
            bundle.initialSceneBodies[3].flagsAndIndices[2] == MR_INVALID_INDEX &&
            bundle.initialSceneBodies[4].flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
            bundle.initialSceneBodies[4].flagsAndIndices[2] == MR_INVALID_INDEX &&
            bundle.model.bodies[4].dampingAndSpeedLimits.x == 0.0f &&
            bundle.model.bodies[4].dampingAndSpeedLimits.y == 0.0f &&
            bundle.model.bodies[4].dampingAndSpeedLimits.z == 0.0f &&
            bundle.model.bodies[4].dampingAndSpeedLimits.w == 0.0f &&
            std::abs(bundle.initialSceneBodies[4].position.x - 0.15f) < 1.0e-7f &&
            std::abs(bundle.initialSceneBodies[4].position.y - 100.0f) < 1.0e-5f &&
            std::abs(bundle.initialSceneBodies[4].position.z - 0.45f) < 1.0e-7f &&
            std::abs(bundle.initialSceneBodies[4].linearVelocityAndInverseMass.w -
                     1.0f / static_cast<float>(spec.massKg)) < 1.0e-6f,
            "shipping tools must remain kinematic and the fifth scene body must carry the finite mass");

    const auto proxy = makeShippingImpactorProxy(spec);
    require(proxy.shape == NM_RIGID_BOX && proxy.bodyIndex == 4u &&
            !proxy.articulated && proxy.dynamic &&
            proxy.sceneBodyIndex == 4u &&
            proxy.localExtent[0] == spec.halfExtentM[0] &&
            proxy.localExtent[1] == spec.halfExtentM[1] &&
            proxy.localExtent[2] == spec.halfExtentM[2],
            "Matter impactor proxy must bind as a nonarticulated finite-mass box");
    const auto platen = makeShippingCompressionPlatenProxy({0.15, 0.005, 0.10});
    const auto floor = makeShippingFloorProxy({0.15, 0.10}, 0.002);
    require(platen.bodyIndex == 2u && floor.bodyIndex == 3u &&
            platen.shape == NM_RIGID_BOX && floor.shape == NM_RIGID_BOX &&
            platen.prescribedEndPoseTranslation &&
            floor.prescribedEndPoseTranslation,
            "shipping compression platen and floor bindings changed");

    const std::array<MRBodyStateGPU, 8> compactFamilyTools{
        makeShippingKinematicSceneBody({0.0, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.0, 0.0, -2.0}),
        makeShippingKinematicSceneBody({0.2, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.2, 0.0, -2.0}),
        makeShippingKinematicSceneBody({0.4, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.4, 0.0, -2.0}),
        makeShippingKinematicSceneBody({0.0, 100.0, 0.0}),
        makeShippingKinematicSceneBody({0.0, -100.0, 0.0})};
    const auto compactBundle = makeShippingRigidWorld(compactFamilyTools, spec);
    require(compactBundle.compiled.valid() &&
                compactBundle.compiled.bodyCount() == 9u &&
                compactBundle.compiled.sceneBodyCount() == 9u &&
                compactBundle.toolSceneBodyCount == 8u &&
                compactBundle.freeBodyIndex == 8u &&
                compactBundle.compressionPlatenBodyIndex == 6u &&
                compactBundle.supportFloorBodyIndex == 7u &&
                compactBundle.initialSceneBodies.size() == 9u &&
                compactBundle.initialSceneBodies[8u].flagsAndIndices[0] ==
                    MR_MOTION_DYNAMIC &&
                compactBundle.initialSceneBodies[8u].flagsAndIndices[2] ==
                    MR_INVALID_INDEX,
            "three-family shipping tool pack must append its free body after eight tools");
    const auto relocatedImpactor = makeShippingImpactorProxy(spec, 8u);
    require(relocatedImpactor.bodyIndex == 8u &&
                relocatedImpactor.sceneBodyIndex == 8u &&
                makeShippingCompressionPlatenProxy(
                    {0.15, 0.005, 0.10}, 0u, 6u).bodyIndex == 6u &&
                makeShippingFloorProxy({0.15, 0.10}, 0.002, 0u, 7u).bodyIndex == 7u &&
                makeShippingCompressionPlatenProxy(
                    {0.15, 0.005, 0.10}, 0u, 6u).prescribedEndPoseTranslation &&
                makeShippingFloorProxy({0.15, 0.10}, 0.002, 0u, 7u)
                    .prescribedEndPoseTranslation,
            "shipping proxy helpers did not preserve the eight-tool body-index layout");

    NMRigidProxyGPU cookedProxy{};
    cookedProxy.shapeKind = NM_RIGID_BOX;
    cookedProxy.bodyIndex = 4u;
    cookedProxy.sceneBodyIndex = 4u;
    cookedProxy.flags = NM_RIGID_DYNAMIC;
    cookedProxy.localExtent = {0.05f, 0.025f, 0.05f, 0.0f};
    cookedProxy.localOrientation = {0.0f, 0.0f, 0.0f, 1.0f};
    const auto& sceneBody = bundle.initialSceneBodies[4u];
    NMRigidStateGPU projected{};
    projected.centerAndRadius = {sceneBody.position.x, sceneBody.position.y,
                                 sceneBody.position.z,
                                 cookedProxy.localCenterAndRadius.w};
    projected.extent = cookedProxy.localExtent;
    projected.orientation = {sceneBody.orientation.x, sceneBody.orientation.y,
                             sceneBody.orientation.z, sceneBody.orientation.w};
    projected.linearVelocityAndInverseMass = {
        sceneBody.linearVelocityAndInverseMass.x,
        sceneBody.linearVelocityAndInverseMass.y,
        sceneBody.linearVelocityAndInverseMass.z,
        sceneBody.linearVelocityAndInverseMass.w};
    projected.angularVelocity = {sceneBody.angularVelocity.x,
                                 sceneBody.angularVelocity.y,
                                 sceneBody.angularVelocity.z,
                                 sceneBody.angularVelocity.w};
    projected.bodyCenter = {sceneBody.position.x, sceneBody.position.y,
                            sceneBody.position.z, sceneBody.position.w};
    projected.inverseInertiaRow0 = {
        sceneBody.inverseInertiaWorldRow0.x,
        sceneBody.inverseInertiaWorldRow0.y,
        sceneBody.inverseInertiaWorldRow0.z,
        sceneBody.inverseInertiaWorldRow0.w};
    projected.inverseInertiaRow1 = {
        sceneBody.inverseInertiaWorldRow1.x,
        sceneBody.inverseInertiaWorldRow1.y,
        sceneBody.inverseInertiaWorldRow1.z,
        sceneBody.inverseInertiaWorldRow1.w};
    projected.inverseInertiaRow2 = {
        sceneBody.inverseInertiaWorldRow2.x,
        sceneBody.inverseInertiaWorldRow2.y,
        sceneBody.inverseInertiaWorldRow2.z,
        sceneBody.inverseInertiaWorldRow2.w};
    require(compareShippingImpactProxyProjection(
                cookedProxy, projected, sceneBody).maximum() == 0.0,
            "projected Matter state does not match the finite-mass scene body");
    require(shippingImpactorProxyMismatchReason(
                cookedProxy, 4u, spec.halfExtentM).empty() &&
                shippingRigidBranchGeometryMismatchReason(
                    std::span<const NMRigidProxyGPU>(&cookedProxy, 1u),
                    bundle, spec).empty(),
            "explicit Matter proxy and MetalWorld impact box do not share one geometry");
    auto wrongExtent = cookedProxy;
    wrongExtent.localExtent.x += 1.0e-4f;
    require(!shippingImpactorProxyMismatchReason(
                 wrongExtent, 4u, spec.halfExtentM).empty(),
            "compiled BOX extent mismatch passed branch geometry admission");
    auto wrongLocalFrame = cookedProxy;
    wrongLocalFrame.localOrientation.z = 0.1f;
    require(!shippingImpactorProxyMismatchReason(
                 wrongLocalFrame, 4u, spec.halfExtentM).empty(),
            "compiled BOX local-frame mismatch passed branch geometry admission");
    // The fixture center is at 100 m, where FP32 ULP is about 7.6 um.
    // Perturb by a larger, still-small offset so the assertion tests the
    // verifier rather than float rounding at the requested threshold.
    projected.centerAndRadius.y += 1.0e-2f;
    require(compareShippingImpactProxyProjection(
                cookedProxy, projected, sceneBody).centerAndRadius >= 9.0e-3,
            "rigid snapshot verifier missed an accepted proxy/scene pose mismatch");
    projected.centerAndRadius.y = sceneBody.position.y;
    projected.linearVelocityAndInverseMass.w += 0.1f;
    require(compareShippingImpactProxyProjection(
                cookedProxy, projected, sceneBody)
                    .linearVelocityAndInverseMass > 0.09,
            "rigid snapshot verifier missed an accepted inverse-mass mismatch");
    projected.linearVelocityAndInverseMass.w =
        sceneBody.linearVelocityAndInverseMass.w;
    projected.inverseInertiaRow1.y += 0.1f;
    require(compareShippingImpactProxyProjection(
                cookedProxy, projected, sceneBody).inverseInertiaRows > 0.09,
            "rigid snapshot verifier missed an accepted inverse-inertia mismatch");
}

void testWorldInertiaCheckpointUsesAcceptedOrientation() {
    const std::array<MRBodyStateGPU, 4> tools{
        makeShippingKinematicSceneBody({0.0, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.0, 0.0, -1.0}),
        makeShippingKinematicSceneBody({0.0, 100.0, 0.0}),
        makeShippingKinematicSceneBody({0.0, -100.0, 0.0})};
    const auto bundle = makeShippingRigidWorld(tools, fixtureSpec());
    const MRBodyPropertiesGPU& properties = bundle.model.bodies.at(4u);
    MRBodyStateGPU state = bundle.initialSceneBodies.at(4u);
    require(shippingWorldInverseInertiaMatchesPose(properties, state),
            "identity-pose checkpoint did not match the body-frame inverse inertia");

    // A quarter turn about world z swaps the anisotropic x/y body axes. Write
    // that independently expected world tensor to catch a local-vs-world
    // checkpoint comparison regression.
    const float quarterTurn = static_cast<float>(std::sqrt(0.5));
    state.orientation = {0.0f, 0.0f, quarterTurn, quarterTurn};
    state.inverseInertiaWorldRow0 = {
        properties.inverseInertiaRow1.y, 0.0f, 0.0f, 0.0f};
    state.inverseInertiaWorldRow1 = {
        0.0f, properties.inverseInertiaRow0.x, 0.0f, 0.0f};
    state.inverseInertiaWorldRow2 = {
        0.0f, 0.0f, properties.inverseInertiaRow2.z, 0.0f};
    require(shippingWorldInverseInertiaMatchesPose(properties, state),
            "accepted rotated checkpoint was compared against a body-frame tensor");

    state.inverseInertiaWorldRow0.y = 0.1f;
    require(!shippingWorldInverseInertiaMatchesPose(properties, state),
            "checkpoint verifier accepted a corrupted rotated inertia component");
    state.inverseInertiaWorldRow0.y = 0.0f;
    state.orientation = {0.0f, 0.0f, 0.0f, 0.0f};
    require(!shippingWorldInverseInertiaMatchesPose(properties, state),
            "checkpoint verifier normalized a degenerate orientation into a valid tensor");
    state.orientation = {0.0f, 0.0f, 0.0f,
                         std::numeric_limits<float>::quiet_NaN()};
    require(!shippingWorldInverseInertiaMatchesPose(properties, state),
            "checkpoint verifier admitted a nonfinite orientation");
}

void testMeasuredRoofFootprintBinding() {
    const auto nominal = shippingAuthoredNominalImpactorHalfExtents(
        0.300, 0.200, 0.025);
    require(std::abs(4.0 * nominal[0] * nominal[2] / (0.300 * 0.200) - 1.2) < 1.0e-14 &&
                std::abs(nominal[0] / nominal[2] - 1.5) < 1.0e-14 && nominal[1] == 0.025,
            "authored nominal impactor must have 20 percent larger area and preserve aspect ratio");
    ShippingOuterRoofMeasurement measurement;
    measurement.bounds.minimumX = -0.050;
    measurement.bounds.maximumX = 0.050;
    measurement.bounds.minimumZ = 0.100;
    measurement.bounds.maximumZ = 0.300;
    measurement.bounds.valid = true;
    const std::array<double, 3> measured{0.5 * std::sqrt(1.2) * 0.100, 0.025,
                                         0.5 * std::sqrt(1.2) * 0.200};
    require(shippingImpactorHalfExtentsMatchMeasuredRoof(measurement, measured),
            "roof-derived half-extents were not matched to their measured bounds");
    auto mismatched = measured;
    mismatched[0] = 0.050;
    require(!shippingImpactorHalfExtentsMatchMeasuredRoof(measurement, mismatched),
            "different compiled footprint was incorrectly accepted as roof coverage");
    measurement.bounds.valid = false;
    require(!shippingImpactorHalfExtentsMatchMeasuredRoof(measurement, measured),
            "invalid roof bounds must not admit an impactor footprint");
}

void testInvalidGeometryFailsClosed() {
    const std::array<MRBodyStateGPU, 4> tools{
        makeShippingKinematicSceneBody({0.0, 0.0, 1.0}),
        makeShippingKinematicSceneBody({0.0, 0.0, -1.0}),
        makeShippingKinematicSceneBody({0.0, 100.0, 0.0}),
        makeShippingKinematicSceneBody({0.0, -100.0, 0.0})};
    const auto base = fixtureSpec();
    const auto expectInvalid = [&](const auto& mutate) {
        auto invalid = base;
        mutate(invalid);
        bool rejected = false;
        try { (void)makeShippingRigidWorld(tools, invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid finite-mass shipping configuration was admitted");
    };
    expectInvalid([](ShippingRigidBodySpec& spec) { spec.massKg = 0.0; });
    expectInvalid([](ShippingRigidBodySpec& spec) {
        spec.massKg = std::numeric_limits<double>::max();
    });
    expectInvalid([](ShippingRigidBodySpec& spec) {
        spec.halfExtentM[0] = 1.0e-100;
    });
    expectInvalid([](ShippingRigidBodySpec& spec) { spec.halfExtentM[1] = -0.01; });
    expectInvalid([](ShippingRigidBodySpec& spec) { spec.timestepSeconds = 0.0; });
    expectInvalid([](ShippingRigidBodySpec& spec) {
        spec.initialCenterM[2] = std::numeric_limits<double>::infinity();
    });
    bool rejectedIndex = false;
    try { (void)makeShippingImpactorProxy(base, 3u); }
    catch (const std::invalid_argument&) { rejectedIndex = true; }
    require(rejectedIndex, "impactor proxy was allowed to alias a kinematic tool body");
}

void testContactImpulseConventionAndActiveCount() {
    MRContactConstraintGPU contact{};
    contact.bodyA = 2u;
    contact.bodyB = 7u;
    contact.normal = {0.0f, 1.0f, 0.0f, 0.0f};
    contact.tangent = {1.0f, 0.0f, 0.0f, 0.0f};
    contact.impulses = {10.0f, 2.0f, 3.0f, 0.0f};
    const std::array<MRContactConstraintGPU, 2> contacts{
        contact, MRContactConstraintGPU{}};
    const auto onA = accumulateShippingBodyContactImpulse(contacts, 1u, 2u);
    const auto onB = accumulateShippingBodyContactImpulse(contacts, 2u, 7u);
    require(onA.contactCount == 1u && onB.contactCount == 1u &&
            std::abs(onA.impulseNs[1] + 10.0) < 1.0e-12 &&
            std::abs(onA.impulseNs[0] + 2.0) < 1.0e-12 &&
            std::abs(onA.impulseNs[2] - 3.0) < 1.0e-12 &&
            std::abs(onB.impulseNs[1] - 10.0) < 1.0e-12 &&
            std::abs(onB.impulseNs[0] - 2.0) < 1.0e-12 &&
            std::abs(onB.impulseNs[2] + 3.0) < 1.0e-12,
            "contact impulse reconstruction changed the A/B action-reaction convention");
    const auto inactive = accumulateShippingBodyContactImpulse(contacts, 0u, 2u);
    require(inactive.contactCount == 0u && inactive.impulseNs[0] == 0.0 &&
                inactive.impulseNs[1] == 0.0 && inactive.impulseNs[2] == 0.0,
            "contact summary consumed an inactive capacity slot");
}

void testNativeReactionMappingForceUnitsAndBalance() {
    std::array<NMRigidProxyGPU, 3> proxies{};
    proxies[0].bodyIndex = 2u;
    proxies[1].bodyIndex = 2u;
    proxies[2].bodyIndex = 4u;
    std::array<NMRigidReactionGPU, 3> reactions{};
    reactions[0].impulseAndCount = {0.0f, -4.0f, 0.0f, 1.0f};
    reactions[1].impulseAndCount = {0.0f, -6.0f, 0.0f, 2.0f};
    reactions[2].impulseAndCount = {0.0f, 10.0f, 0.0f, 3.0f};
    const auto platen = accumulateShippingNativeBodyReaction(
        reactions, proxies, 1u, 0u, 2u);
    const auto impactor = accumulateShippingNativeBodyReaction(
        reactions, proxies, 1u, 0u, 4u);
    require(platen.matchedProxyCount == 2u && impactor.matchedProxyCount == 1u &&
            platen.contactSamples == 3.0 && impactor.contactSamples == 3.0 &&
            platen.impulseNs[1] == -10.0 && impactor.impulseNs[1] == 10.0 &&
            platen.impulseNs[1] + impactor.impulseNs[1] == 0.0,
            "native reaction mapping failed its synthetic action-reaction balance");
    require(std::abs(shippingReactionForceYFromImpulse(platen, 0.25) + 40.0) < 1.0e-12 &&
            std::abs(shippingReactionForceYFromImpulse(impactor, 0.25) - 40.0) < 1.0e-12,
            "native reaction impulse divided by timestep did not produce force in newtons");
    bool rejectedMalformedArena = false;
    try {
        (void)accumulateShippingNativeBodyReaction(
            std::span<const NMRigidReactionGPU>(reactions.data(), 2u),
            proxies, 1u, 0u, 2u);
    } catch (const std::invalid_argument&) { rejectedMalformedArena = true; }
    require(rejectedMalformedArena,
            "native reaction mapper accepted mismatched proxy/reaction arenas");
}

void testShippingGroundSupportAndTemporaryToolGate() {
    const auto floorOrientation = shippingSupportFloorOrientationXYWZ();
    const double x = floorOrientation[0];
    const double y = floorOrientation[1];
    const double z = floorOrientation[2];
    const double w = floorOrientation[3];
    const std::array<double, 3> rotatedThinAxis{
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y)};
    const ShippingRigidBodySpec orientationContract;
    require(rotatedThinAxis[1] > 0.999999 &&
                std::abs(rotatedThinAxis[0]) < 1.0e-12 &&
                std::abs(rotatedThinAxis[2]) < 1.0e-12 &&
                orientationContract.gravityMPerS2[1] < 0.0 &&
                orientationContract.gravityMPerS2[0] == 0.0 &&
                orientationContract.gravityMPerS2[2] == 0.0,
            "shipping floor support normal and gravity do not share the authored vertical axis");
    std::array<NMRigidProxyGPU, 5> proxies{};
    for (std::uint32_t index = 0u; index < proxies.size(); ++index)
        proxies[index].bodyIndex = index;
    std::array<NMRigidReactionGPU, 5> reactions{};
    reactions[0].impulseAndCount.w = 1.0f; // scoring punch
    reactions[1].impulseAndCount.w = 1.0f; // scoring rail/anvil
    reactions[3].impulseAndCount.w = 2.0f; // permitted support floor
    reactions[4].impulseAndCount.w = 1.0f; // free body/tool contact
    const auto classified = classifyShippingRigidContactSamples(
        reactions, proxies, 1u);
    require(classified.totalSamples == 5.0 &&
                classified.supportFloorSamples == 2.0 &&
                classified.forbiddenToolSamples == 3.0,
            "released-box contact gate did not isolate the declared ground support from temporary tools");
    for (auto& reaction : reactions)
        reaction.impulseAndCount.w = 0.0f;
    reactions[3].impulseAndCount.w = 4.0f;
    const auto groundOnly = classifyShippingRigidContactSamples(
        reactions, proxies, 1u);
    require(groundOnly.totalSamples == 4.0 &&
                groundOnly.supportFloorSamples == 4.0 &&
                groundOnly.forbiddenToolSamples == 0.0,
            "natural supporting-floor contact incorrectly fails the released shipping tool gate");

    const auto floor = makeShippingNonpenetratingFloorPlacement(
        0.200000003, 0.002, 2.5e-6);
    require(floor.topY < static_cast<float>(0.200000003) &&
                floor.positiveClearanceM > 0.0 &&
                floor.positiveClearanceM < 2.5e-6,
            "shipping floor must be placed just below the accepted bottom surface inside contact range");
    bool rejectedFloorOutsideContactRange = false;
    try {
        (void)makeShippingNonpenetratingFloorPlacement(
            0.200000003, 0.002, 1.0e-10);
    } catch (const std::invalid_argument&) {
        rejectedFloorOutsideContactRange = true;
    }
    require(rejectedFloorOutsideContactRange,
            "shipping floor helper admitted a clearance outside the native contact range");
}

void testMatterCertificateAcceptanceGate() {
    numi::matter::CompiledWorld compiled;
    compiled.mixedSolver.residualTolerances = {1.0e-4f, 1.0e-4f,
                                                1.0e-4f, 0.0f};
    numi::matter::RuntimeStateSnapshot snapshot;
    snapshot.available = true;
    snapshot.statuses.resize(1u);
    snapshot.solverCertificates.resize(1u);
    auto& status = snapshot.statuses[0u];
    status.code = NM_STATUS_SUCCESS;
    auto& certificate = snapshot.solverCertificates[0u];
    certificate.validity.w = 1.0f;
    certificate.nonlinear = {2.0e-5f, 2.0e-5f, 3.0e-5f, 4.0e-5f};
    require(inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "native Matter acceptance gate rejected a finite accepted certificate");

    status.code = NM_STATUS_NONLINEAR_SOLVER_FAILURE;
    require(!inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "raw certificate flag overrode a failed native status");
    status.code = NM_STATUS_SUCCESS;
    certificate.nonlinear.x = 2.0e-4f;
    require(!inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "native acceptance gate ignored the authored residual tolerance");
    certificate.nonlinear.x = 2.0e-5f;
    certificate.validity.w = 0.0f;
    require(!inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "native acceptance gate ignored a rejected certificate flag");
    certificate.validity.w = 1.0f;
    certificate.nonlinear.w = std::numeric_limits<float>::quiet_NaN();
    require(!inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "native acceptance gate admitted a nonfinite certificate");
    snapshot.solverCertificates.clear();
    require(!inspectShippingMatterAcceptance(snapshot, compiled).accepted,
            "native acceptance gate admitted truncated certificate readback");
}

void testRigidMomentumAndEnergyLedger() {
    constexpr double mass = 2.0;
    constexpr std::array<double, 3> halfExtent{0.5, 0.25, 0.1};
    const double ix = mass * (halfExtent[1] * halfExtent[1] +
                              halfExtent[2] * halfExtent[2]) / 3.0;
    const double iy = mass * (halfExtent[0] * halfExtent[0] +
                              halfExtent[2] * halfExtent[2]) / 3.0;
    const double iz = mass * (halfExtent[0] * halfExtent[0] +
                              halfExtent[1] * halfExtent[1]) / 3.0;
    MRBodyStateGPU body{};
    body.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    body.position = {0.0f, 1.9f, 0.0f, 1.0f};
    body.linearVelocityAndInverseMass = {3.0f, 4.0f, 0.0f,
                                         static_cast<float>(1.0 / mass)};
    body.angularVelocity = {2.0f, -1.0f, 0.5f, 0.0f};
    body.inverseInertiaWorldRow0 = {static_cast<float>(1.0 / ix), 0.0f, 0.0f, 0.0f};
    body.inverseInertiaWorldRow1 = {0.0f, static_cast<float>(1.0 / iy), 0.0f, 0.0f};
    body.inverseInertiaWorldRow2 = {0.0f, 0.0f, static_cast<float>(1.0 / iz), 0.0f};
    const auto observed = observeShippingRigidDynamics(
        body, {0.0, -9.80665, 0.0}, {0.0, 2.0, 0.0});
    const double expectedRotation = 0.5 * (ix * 4.0 + iy * 1.0 + iz * 0.25);
    require(std::abs(observed.massKg - mass) < 1.0e-6 &&
                std::abs(observed.linearMomentumKgMPerS[0] - 6.0) < 1.0e-6 &&
                std::abs(observed.linearMomentumKgMPerS[1] - 8.0) < 1.0e-6 &&
                std::abs(observed.translationalKineticEnergyJ - 25.0) < 1.0e-6 &&
                std::abs(observed.rotationalKineticEnergyJ - expectedRotation) < 1.0e-6 &&
                std::abs(observed.gravitationalPotentialRelativeJ +
                         mass * 9.80665 * 0.1) < 1.0e-5 &&
                std::abs(observed.mechanicalEnergyRelativeJ() -
                    (25.0 + expectedRotation - mass * 9.80665 * 0.1)) < 1.0e-5,
            "shipping rigid dynamics ledger did not preserve momentum and energy units");
    body.inverseInertiaWorldRow1.y = -1.0f;
    bool rejectedIndefinite = false;
    try {
        (void)observeShippingRigidDynamics(
            body, {0.0, -9.80665, 0.0}, {0.0, 2.0, 0.0});
    } catch (const std::invalid_argument&) {
        rejectedIndefinite = true;
    }
    require(rejectedIndefinite,
            "shipping dynamics ledger accepted an indefinite inverse inertia");
}

void testGravityTransferAndExternalWorkBalance() {
    ShippingGravityTransferProtocol protocol;
    const std::uint32_t count = shippingGravityTransferStepCount(protocol);
    require(count == 1000u &&
                std::abs(shippingGravityTransferSupportFraction(0u, count) -
                         (1.0 - 0.5 / count)) < 1.0e-15 &&
                std::abs(shippingGravityTransferSupportFraction(count - 1u, count) -
                         0.5 / count) < 1.0e-15,
            "gravity transfer must use midpoint samples and reserve zero force for release");
    constexpr double mass = 12.0;
    constexpr double gravityY = -9.80665;
    const double fullSupport = shippingGravityTransferSupportForceY(
        mass, gravityY, 1.0);
    const double halfSupport = shippingGravityTransferSupportForceY(
        mass, gravityY, 0.5);
    require(std::abs(fullSupport - mass * 9.80665) < 1.0e-12 &&
                std::abs(halfSupport - 0.5 * fullSupport) < 1.0e-12,
            "upward support force must equal the declared fraction of body weight");
    const auto supported = makeShippingBodyWrenchTerms(
        mass, gravityY, fullSupport);
    const auto released = makeShippingBodyWrenchTerms(
        mass, gravityY, 0.0);
    const auto halfSupported = makeShippingBodyWrenchTerms(
        mass, gravityY, halfSupport);
    const auto& wrench = supported.totalWrench;
    require(std::abs(supported.gravityForceYNewtons + fullSupport) < 1.0e-12 &&
                std::abs(supported.totalExternalForceYNewtons) < 1.0e-12 &&
                supported.totalWrench.force.y == 0.0f &&
                std::abs(released.totalExternalForceYNewtons + fullSupport) < 1.0e-12 &&
                std::abs(halfSupported.totalExternalForceYNewtons + 0.5 * fullSupport) < 1.0e-12 &&
                std::abs(released.totalWrench.force.y + fullSupport) <
                    2.0e-7 * fullSupport &&
                wrench.force.x == 0.0f &&
                std::abs(wrench.force.y) <
                    2.0e-7 * fullSupport &&
                wrench.force.z == 0.0f && wrench.torque.x == 0.0f &&
                wrench.torque.y == 0.0f && wrench.torque.z == 0.0f,
            "external RHS must combine body gravity and actuator exactly once, while preserving both terms");
    bool rejectedUnrepresentableWrench = false;
    try {
        (void)makeShippingVerticalBodyForceWrench(
            std::numeric_limits<double>::max());
    } catch (const std::invalid_argument&) {
        rejectedUnrepresentableWrench = true;
    }
    require(rejectedUnrepresentableWrench,
            "external wrench accepted a finite double force outside native FP32 range");
    const double work = shippingActuatorWorkJ(
        {0.0, fullSupport, 0.0}, {0.0, -0.002, 0.0});
    require(work < 0.0 && std::abs(work + fullSupport * 0.002) < 1.0e-12,
            "an upward actuator must report negative work during downward body motion");
    const auto balance = shippingLinearMomentumBalance(
        {0.0, 0.0, 0.0}, {0.0, 0.003, 0.0},
        {0.0, gravityY, 0.0}, mass,
        {0.0, fullSupport * 0.5, 0.0},
        {0.0, 0.003 - mass * gravityY * 1.0e-3 -
                  fullSupport * 0.5 * 1.0e-3, 0.0},
        1.0e-3);
    require(balance.normalizedResidual < 1.0e-12 &&
                std::abs(balance.residualNs[1]) < 1.0e-15,
            "native contact, gravity, actuator, and momentum impulse ledger failed balance");
    bool rejectedBadStep = false;
    try { (void)shippingGravityTransferSupportFraction(2u, 2u); }
    catch (const std::invalid_argument&) { rejectedBadStep = true; }
    require(rejectedBadStep,
            "gravity-transfer schedule admitted an index outside the ramp");
}

void testCompressionAndFpeakSchedules() {
    ShippingCompressionProtocol protocol;
    protocol.platenSpeedMPerS = 6.25e-3 / 60.0;
    protocol.incrementM = 0.25e-3;
    protocol.targetStrokeFractionOfInsideHeight = 0.70;
    const auto schedule = makeShippingCompressionSchedule(0.200, protocol);
    double stroke = 0.0;
    double elapsed = 0.0;
    for (const auto& increment : schedule) {
        stroke += increment.displacementM;
        elapsed += increment.timestepSeconds;
        require(increment.displacementM <= protocol.incrementM &&
                    increment.timestepSeconds > 0.0,
                "shipping compression schedule exceeded its increment or time bound");
    }
    require(schedule.size() == 560u && std::abs(stroke - 0.140) < 1.0e-12 &&
                std::abs(elapsed - 1344.0) < 1.0e-9,
            "shipping compression schedule no longer covers 70 percent of 200 mm at 6.25 mm/min");

    ShippingFirstCompressionPeakDetector detector;
    const std::array<double, 5> forces{5.0, 10.0, 12.0, 11.95, 11.70};
    std::optional<ShippingQualifiedCompressionPeak> peak;
    for (std::size_t index = 0u; index < forces.size(); ++index) {
        peak = detector.observeAccepted(
            0.25e-3 * static_cast<double>(index), forces[index], 1.0);
        if (index < 4u)
            require(!peak.has_value(),
                    "compression Fpeak qualified before two declining samples");
    }
    require(peak.has_value() && peak->peak.sampleIndex == 2u &&
            peak->peak.forceN == 12.0 && peak->qualifyingSampleIndex == 4u &&
            peak->qualifyingDropFraction >= 0.01,
            "first compression local peak was not selected with the declared qualification rule");
    ShippingFirstCompressionPeakDetector brokenContact;
    (void)brokenContact.observeAccepted(0.0, 10.0, 1.0);
    (void)brokenContact.observeAccepted(0.001, 12.0, 1.0);
    (void)brokenContact.observeAccepted(0.002, 11.0, 1.0);
    (void)brokenContact.observeAccepted(0.003, 0.0, 0.0);
    (void)brokenContact.observeAccepted(0.004, 10.0, 1.0);
    require(!brokenContact.observeAccepted(0.005, 9.0, 1.0).has_value(),
            "compression peak detector joined disconnected contact episodes");

    require(std::abs(shippingImpactMassFromPeakForce(100.0, 0.5) -
                     50.0 / 9.80665) < 1.0e-12,
            "Fpeak/g impact-mass scaling changed");
    bool rejectedZeroForce = false;
    try { (void)shippingImpactMassFromPeakForce(0.0, 1.0); }
    catch (const std::invalid_argument&) { rejectedZeroForce = true; }
    require(rejectedZeroForce,
            "impact mass was admitted without a measured positive Fpeak");

    AcceptedCompressionPeak acceptedPeak;
    acceptedPeak.completed = true;
    acceptedPeak.acceptedEvidence = true;
    acceptedPeak.runId = "synthetic-drop-geometry-fixture";
    acceptedPeak.resultSha256 = std::string(64u, 'a');
    acceptedPeak.sourceAssemblyCheckpointSha256 = std::string(64u, 'b');
    acceptedPeak.boardRecipeId = "literature2009";
    acceptedPeak.acceptedStepCount = 1u;
    acceptedPeak.peakReactionForceN = 100.0;
    ReleasedAssemblyCheckpoint acceptedCheckpoint;
    acceptedCheckpoint.checkpointId = "synthetic-released-fixture";
    acceptedCheckpoint.checkpointSha256 = std::string(64u, 'b');
    acceptedCheckpoint.accepted = true;
    acceptedCheckpoint.completeNativeSnapshot = true;
    acceptedCheckpoint.sourcePhysicsFingerprint = 0x1234u;
    acceptedCheckpoint.compiledWorldFingerprint = 0x5678u;
    acceptedCheckpoint.temporaryGripsReleased = true;
    acceptedCheckpoint.toolingParkedOrRemoved = true;
    acceptedCheckpoint.settled = true;
    acceptedCheckpoint.manufacturerLapClosedByGeometry = true;
    acceptedCheckpoint.bottomEndClosedByGeometry = true;
    acceptedCheckpoint.topEndClosedByGeometry = true;
    acceptedCheckpoint.allAssemblyActionsAccepted = true;
    acceptedCheckpoint.rejectedStepCount = 0u;
    acceptedCheckpoint.maximumNodalSpeedMPerS = 0.0;
    acceptedCheckpoint.normalizedForceImbalance = 0.0;
    acceptedCheckpoint.maximumEndCornerGapM = 0.0;
    acceptedCheckpoint.measuredInside = {};
    acceptedCheckpoint.boardRecipeId = "literature2009";
    acceptedCheckpoint.measuredInsideCenterX = 0.0;
    acceptedCheckpoint.measuredInsideCenterZ = 0.0;
    acceptedCheckpoint.measuredRoofContactY = 0.2;
    acceptedCheckpoint.measuredOuterRoof.measuredFromAcceptedFixtureFreeSettledCheckpoint = true;
    acceptedCheckpoint.measuredOuterRoof.checkpointSha256 =
        acceptedCheckpoint.checkpointSha256;
    acceptedCheckpoint.measuredOuterRoof.sourcePhysicsFingerprint =
        acceptedCheckpoint.sourcePhysicsFingerprint;
    acceptedCheckpoint.measuredOuterRoof.compiledWorldFingerprint =
        acceptedCheckpoint.compiledWorldFingerprint;
    acceptedCheckpoint.measuredOuterRoof.bounds = {
        -0.150, 0.150, -0.100, 0.100, 0.200, 12u, true};
    const auto cases = makeShippingDropCases(acceptedPeak, acceptedCheckpoint);
    require(cases.size() == 6u && std::abs(cases[0].impactorPlanLengthM * cases[0].impactorPlanWidthM /
                         (0.300 * 0.200) - 1.2) < 1.0e-14 &&
            std::abs(cases[0].impactorPlanLengthM / cases[0].impactorPlanWidthM - 1.5) < 1.0e-14,
            "shipping drop impactor footprint must have 20 percent larger area at the measured aspect ratio");
    constexpr std::array<double, 3> ratios{0.5, 1.0, 1.5};
    for (std::size_t index = 0u; index < cases.size(); ++index) {
        const auto& item = cases[index];
        const std::size_t ratioIndex = index / 2u;
        const bool quarterWidth = index % 2u != 0u;
        const double expectedZ = quarterWidth ? 0.25 * 0.200 : 0.0;
        require(item.forcePeakMassRatio == ratios[ratioIndex] &&
                    std::abs(item.massKg - ratios[ratioIndex] * 100.0 / 9.80665) <
                        1.0e-12 &&
                    std::abs(item.impactorCenterX) < 1.0e-12 &&
                    std::abs(item.impactorCenterZ - expectedZ) < 1.0e-12 &&
                    std::abs(item.impactorStartCenterY - 0.325) < 1.0e-12 &&
                    item.restoreCheckpointId == acceptedCheckpoint.checkpointId &&
                    item.restoreCheckpointSha256 == acceptedCheckpoint.checkpointSha256 &&
                    item.peakRunId == acceptedPeak.runId &&
                    item.peakResultSha256 == acceptedPeak.resultSha256,
                "shipping mass/location cases changed the bound released checkpoint, Fpeak identity, or 100 mm centered/quarter-width pose");
    }
}

} // namespace

int main() {
    try {
        testCompositionAndFiniteMassProxy();
        testWorldInertiaCheckpointUsesAcceptedOrientation();
        testMeasuredRoofFootprintBinding();
        testInvalidGeometryFailsClosed();
        testContactImpulseConventionAndActiveCount();
        testNativeReactionMappingForceUnitsAndBalance();
        testShippingGroundSupportAndTemporaryToolGate();
        testMatterCertificateAcceptanceGate();
        testRigidMomentumAndEnergyLedger();
        testGravityTransferAndExternalWorkBalance();
        testCompressionAndFpeakSchedules();
        std::cout << "shipping rigid bridge CPU checks passed: 10 groups\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
