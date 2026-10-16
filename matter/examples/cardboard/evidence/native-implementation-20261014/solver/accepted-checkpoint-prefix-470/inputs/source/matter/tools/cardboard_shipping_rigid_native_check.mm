#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cardboard_shipping_crush.hpp"
#include "cardboard_shipping_rigid_driver.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef NUMI_CARDBOARD_METAL_WORLD
#error "shipping rigid native check requires the integrated MetalWorld build"
#endif

namespace {

using namespace numi::matter;
using namespace numi::cardboard;
unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string jsonEscapeDiagnostic(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20u) {
                    constexpr char hex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += hex[(character >> 4u) & 0x0fu];
                    escaped += hex[character & 0x0fu];
                } else {
                    escaped += static_cast<char>(character);
                }
        }
    }
    return escaped;
}

template<class T>
bool exactVectorEqual(const std::vector<T>& a, const std::vector<T>& b) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (a.size() != b.size()) return false;
    return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

bool sameAcceptedPhysicalState(const RuntimeStateSnapshot& a,
                               const RuntimeStateSnapshot& b) {
    return a.available && b.available &&
        a.sourcePhysicsFingerprint == b.sourcePhysicsFingerprint &&
        a.deviceProgramFingerprint == b.deviceProgramFingerprint &&
        exactVectorEqual(a.femNodes, b.femNodes) &&
        exactVectorEqual(a.femFields, b.femFields) &&
        exactVectorEqual(a.femMaterialState, b.femMaterialState) &&
        exactVectorEqual(a.femTopologyNodes, b.femTopologyNodes) &&
        exactVectorEqual(a.femTopologyTetrahedra, b.femTopologyTetrahedra) &&
        exactVectorEqual(a.topologyStates, b.topologyStates) &&
        exactVectorEqual(a.cohesiveFaces, b.cohesiveFaces) &&
        exactVectorEqual(a.contactHistories, b.contactHistories) &&
        exactVectorEqual(a.deformableContactHistories,
                         b.deformableContactHistories) &&
        exactVectorEqual(a.rigidStates, b.rigidStates);
}

id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device,
                               const std::size_t bytes,
                               const std::string& label) {
    if (bytes == 0u) throw std::invalid_argument(label + " buffer is empty");
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if (buffer == nil || buffer.contents == nullptr)
        throw std::runtime_error("cannot allocate " + label + " buffer");
    std::memset(buffer.contents, 0, bytes);
    return buffer;
}

struct NativeFixture {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    CompiledWorld matterWorld;
    Runtime runtime;
    std::unique_ptr<ShippingRigidWorldRunner> runner;
    id<MTLBuffer> wrenchBuffer = nil;
    std::array<MRBodyStateGPU, 4> tools{};
    ShippingRigidBodySpec bodySpec;
    double dt = 5.0e-4;

    NativeFixture(const std::string& metallibPath,
                  const std::string& materialPath) {
        if (device == nil) throw std::runtime_error("Metal device unavailable");
        const std::string deviceName = [[device name] UTF8String];
        if (deviceName.find("Apple") == std::string::npos ||
            deviceName.find("Paravirtual") != std::string::npos)
            throw std::runtime_error("native Apple Metal device required");
        bodySpec.massKg = 0.75;
        bodySpec.halfExtentM = {0.05, 0.025, 0.05};
        bodySpec.initialCenterM = {0.2, 1.0, 0.2};
        bodySpec.gravityMPerS2 = {0.0, -9.80665, 0.0};
        bodySpec.timestepSeconds = dt;
        tools = {makeShippingKinematicSceneBody({10.0, 0.0, 10.0}),
                 makeShippingKinematicSceneBody({-10.0, 0.0, 10.0}),
                 makeShippingKinematicSceneBody({10.0, 0.0, -10.0}),
                 makeShippingKinematicSceneBody({-10.0, 0.0, -10.0})};

        const ParseResult parsed = parseMatterFile(materialPath);
        require(parsed.succeeded(), "shipping rigid fixture material parse failed: " +
            materialPath);
        WorldSource source;
        source.environmentCount = 1u;
        source.frameTimestep = dt;
        source.gravity = {0.0, 0.0, 0.0};
        source.mixedSolver.relativeResidual = 1.0e-7;
        source.mixedSolver.newtonIterations = 12u;
        source.mixedSolver.fgmresIterations = 64u;
        source.materials = {parsed.material};
        source.rigidProxies = {makeShippingImpactorProxy(bodySpec)};
        ObjectSource fem;
        fem.name = "small_native_fem_certificate_and_rollback_control";
        fem.representation = Representation::fem;
        fem.materialIndex = 0u;
        fem.mixedFEM = false;
        fem.twoWayCoupling = true;
        fem.adaptive = false;
        fem.identifiable = false;
        fem.deformableContact = false;
        fem.deformableSelfContact = false;
        fem.characteristicLength = 0.01;
        fem.femNodes = {{{0.0, 0.0, 0.0}}, {{0.01, 0.0, 0.0}},
                        {{0.0, 0.01, 0.0}}, {{0.0, 0.0, 0.01}}};
        fem.femFixedNodes = {0u, 1u, 2u};
        fem.tetrahedra = {{{0u, 1u, 2u, 3u}}};
        source.objects.push_back(std::move(fem));
        CompileOptions options;
        options.maximumRateExponent = 0u;
        options.emitSpecializedMetal = false;
        options.localMaterialNewtonIterations = 16u;
        const CompileResult compiled = compileWorld(source, options);
        require(compiled.succeeded(), "shipping rigid fixture compile failed");
        matterWorld = compiled.world;
        require(matterWorld.dispatch.environmentCount == 1u &&
                    matterWorld.dispatch.objectCount == 1u &&
                    matterWorld.contact.rigidProxies.size() == 1u &&
                    matterWorld.contact.rigidProxies[0].bodyIndex == 4u &&
                    matterWorld.dispatch.rigidGeneralizedCapacity == 6u,
                "fixture did not compile to one six-DOF nonarticulated free body");

        RuntimeConfiguration config;
        config.metallib = metallibPath;
        config.environmentCount = 1u;
        config.captureDiagnostics = true;
        config.captureEvents = false;
        config.adaptiveTransfer = false;
        const RuntimeDiagnostics initialized = runtime.initialize(matterWorld, config);
        require(initialized.encoded, "shipping rigid runtime initialize failed: " +
            initialized.message);
        require(runtime.sourcePhysicsFingerprint() == matterWorld.physicsFingerprint,
                "shipping rigid native runtime source fingerprint mismatch");

        wrenchBuffer = makeSharedBuffer(device,
            5u * sizeof(MRABABodyWrenchGPU), "external body wrench");
        auto bundle = makeShippingRigidWorld(tools, bodySpec);
        runner = std::make_unique<ShippingRigidWorldRunner>(
            std::move(bundle), runtime, static_cast<float>(dt), matterWorld);
    }

    MRABABodyWrenchGPU* wrench() const {
        return static_cast<MRABABodyWrenchGPU*>(wrenchBuffer.contents);
    }

    ShippingRigidWorldRunner::StepResult step(
        const double actuatorY, const std::uint32_t controlStep) {
        const auto terms = makeShippingBodyWrenchTerms(
            runner->freeBodyMassKg(), runner->gravityMPerS2()[1], actuatorY);
        wrench()[4u] = terms.totalWrench;
        return runner->step(tools, nullptr, 0u, nullptr, 0u,
            (__bridge void*)wrenchBuffer, 5u, 5u, controlStep,
            static_cast<float>(dt), false);
    }
};

ShippingRigidDynamicsObservation observe(const NativeFixture& fixture) {
    return observeShippingRigidDynamics(
        fixture.runner->sceneBodies().at(4u),
        fixture.runner->gravityMPerS2(), {0.0, 1.0, 0.0});
}

void requireFixedMassInertia(const NativeFixture& fixture,
                             const std::string& label) {
    const auto& body = fixture.runner->sceneBodies().at(4u);
    const double mass = fixture.bodySpec.massKg;
    const auto half = fixture.bodySpec.halfExtentM;
    const std::array<double, 3> expectedInertia{
        mass * (half[1] * half[1] + half[2] * half[2]) / 3.0,
        mass * (half[0] * half[0] + half[2] * half[2]) / 3.0,
        mass * (half[0] * half[0] + half[1] * half[1]) / 3.0};
    require(std::abs(static_cast<double>(body.linearVelocityAndInverseMass.w) -
                     1.0 / mass) <= 2.0e-6 / mass,
            label + " changed the finite body's inverse mass");
    const std::array<std::array<float, 4>, 3> inverseRows{{
        {body.inverseInertiaWorldRow0.x, body.inverseInertiaWorldRow0.y,
         body.inverseInertiaWorldRow0.z, body.inverseInertiaWorldRow0.w},
        {body.inverseInertiaWorldRow1.x, body.inverseInertiaWorldRow1.y,
         body.inverseInertiaWorldRow1.z, body.inverseInertiaWorldRow1.w},
        {body.inverseInertiaWorldRow2.x, body.inverseInertiaWorldRow2.y,
         body.inverseInertiaWorldRow2.z, body.inverseInertiaWorldRow2.w}}};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            const double expected = row == column
                ? 1.0 / expectedInertia[row] : 0.0;
            require(std::abs(static_cast<double>(inverseRows[row][column]) -
                             expected) <= 2.0e-5 * std::max(1.0, std::abs(expected)),
                    label + " changed the finite body's inverse inertia");
        }
        require(inverseRows[row][3] == 0.0f,
                label + " changed the finite body's inverse-inertia padding");
    }
}

void requireAccepted(NativeFixture& fixture,
                     const ShippingRigidWorldRunner::StepResult& step,
                     const std::string& label) {
    const RuntimeStateSnapshot snapshot = fixture.runtime.snapshot();
    if (!step.diagnostics.succeeded() || !step.matterAcceptance.accepted) {
        std::cerr << std::setprecision(9)
            << "shipping_native_acceptance_diagnostic {\"label\":\""
            << jsonEscapeDiagnostic(label) << "\",\"metalworld_succeeded\":"
            << (step.diagnostics.succeeded() ? "true" : "false")
            << ",\"metalworld_message\":\""
            << jsonEscapeDiagnostic(step.diagnostics.message)
            << "\",\"matter_acceptance\":{\"accepted\":"
            << (step.matterAcceptance.accepted ? "true" : "false")
            << ",\"status_code\":" << step.matterAcceptance.statusCode
            << ",\"reason\":\""
            << jsonEscapeDiagnostic(step.matterAcceptance.reason)
            << "\",\"certificate\":["
            << step.matterAcceptance.certificate.nonlinear.x << ','
            << step.matterAcceptance.certificate.nonlinear.y << ','
            << step.matterAcceptance.certificate.nonlinear.z << ','
            << step.matterAcceptance.certificate.nonlinear.w << ','
            << step.matterAcceptance.certificate.validity.x << ','
            << step.matterAcceptance.certificate.validity.y << ','
            << step.matterAcceptance.certificate.validity.z << ','
            << step.matterAcceptance.certificate.validity.w << "]},"
            << "\"snapshot\":{\"available\":"
            << (snapshot.available ? "true" : "false")
            << ",\"message\":\""
            << jsonEscapeDiagnostic(snapshot.message)
            << "\",\"control_step\":" << snapshot.controlStep
            << ",\"physics_substep\":" << snapshot.physicsSubstep
            << ",\"fem_nodes\":" << snapshot.femNodes.size()
            << ",\"fem_material_state\":" << snapshot.femMaterialState.size()
            << ",\"rigid_states\":" << snapshot.rigidStates.size()
            << ",\"rigid_generalized_candidate\":"
            << snapshot.rigidGeneralizedCandidate.size()
            << ",\"fem_node_state\":[";
        const auto emitFloat4 = [](const nm_float4& value) {
            std::cerr << '[' << value.x << ',' << value.y << ','
                << value.z << ',' << value.w << ']';
        };
        for (std::size_t i = 0u; i < snapshot.femNodes.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            const auto& node = snapshot.femNodes[i];
            std::cerr << "{\"position_mass\":";
            emitFloat4(node.positionAndMass);
            std::cerr << ",\"velocity_inverse_mass\":";
            emitFloat4(node.velocityAndInverseMass);
            std::cerr << ",\"rest_fixed\":";
            emitFloat4(node.restAndFixed);
            std::cerr << ",\"delta_velocity\":";
            emitFloat4(node.deltaVelocity);
            std::cerr << ",\"reference_displacement_mode\":";
            emitFloat4(node.referenceDisplacementAndMode);
            std::cerr << '}';
        }
        std::cerr << "],\"fem_material_state_values\":[";
        for (std::size_t i = 0u; i < snapshot.femMaterialState.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            std::cerr << snapshot.femMaterialState[i];
        }
        std::cerr << "],\"rigid_proxy_state\":[";
        for (std::size_t i = 0u; i < snapshot.rigidStates.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            const auto& rigid = snapshot.rigidStates[i];
            std::cerr << "{\"center_radius\":";
            emitFloat4(rigid.centerAndRadius);
            std::cerr << ",\"orientation\":";
            emitFloat4(rigid.orientation);
            std::cerr << ",\"linear_velocity_inverse_mass\":";
            emitFloat4(rigid.linearVelocityAndInverseMass);
            std::cerr << ",\"angular_velocity\":";
            emitFloat4(rigid.angularVelocity);
            std::cerr << '}';
        }
        std::cerr << "],\"rigid_generalized_candidate_values\":[";
        for (std::size_t i = 0u; i < snapshot.rigidGeneralizedCandidate.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            std::cerr << snapshot.rigidGeneralizedCandidate[i];
        }
        std::cerr << "],\"statuses\":[";
        for (std::size_t i = 0u; i < snapshot.statuses.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            const auto& status = snapshot.statuses[i];
            std::cerr << "{\"code\":" << status.code
                << ",\"environment\":" << status.environment
                << ",\"object\":" << status.objectIndex
                << ",\"failing_index\":" << status.failingIndex
                << ",\"completed_microsteps\":" << status.completedMicrosteps
                << ",\"fgmres_iterations\":" << status.fgmresIterations
                << ",\"contact_count\":" << status.contactCount
                << ",\"event_count\":" << status.eventCount
                << ",\"diagnostics\":[" << status.diagnostics.x << ','
                << status.diagnostics.y << ',' << status.diagnostics.z << ','
                << status.diagnostics.w << "]}";
        }
        std::cerr << "],\"solver_certificates\":[";
        for (std::size_t i = 0u; i < snapshot.solverCertificates.size(); ++i) {
            if (i != 0u) std::cerr << ',';
            const auto& certificate = snapshot.solverCertificates[i];
            std::cerr << "{\"nonlinear\":[" << certificate.nonlinear.x << ','
                << certificate.nonlinear.y << ',' << certificate.nonlinear.z << ','
                << certificate.nonlinear.w << "],\"contact\":["
                << certificate.contact.x << ',' << certificate.contact.y << ','
                << certificate.contact.z << ',' << certificate.contact.w
                << "],\"validity\":[" << certificate.validity.x << ','
                << certificate.validity.y << ',' << certificate.validity.z << ','
                << certificate.validity.w << "]}";
        }
        std::cerr << "]}}\n";
    }
    require(step.diagnostics.succeeded(), label + " MetalWorld step failed: " +
        step.diagnostics.message);
    require(step.matterAcceptance.accepted,
        label + " Matter step failed: " + step.matterAcceptance.reason);
    require(snapshot.available && snapshot.rigidStates.size() == 1u,
            label + " omitted accepted native proxy state");
    require(snapshot.femNodes.size() == 4u &&
                !snapshot.femMaterialState.empty() &&
                snapshot.statuses.size() == 1u &&
                snapshot.solverCertificates.size() == 1u,
            label + " omitted the authored FEM state or per-object solver certificate");
    requireFixedMassInertia(fixture, label);
    const auto error = compareShippingImpactProxyProjection(
        fixture.matterWorld.contact.rigidProxies[0], snapshot.rigidStates[0],
        fixture.runner->sceneBodies()[4u]);
    require(error.maximum() <= 3.0e-5,
            label + " Matter proxy projection differs from the accepted scene body");
}

void runSupportTransferReleaseAndRollback(NativeFixture& fixture) {
    constexpr double gravityY = -9.80665;
    constexpr std::uint32_t rampSteps = 8u;
    const double mass = fixture.runner->freeBodyMassKg();
    const double fullSupport = -mass * gravityY;

    auto step = fixture.step(fullSupport, 0u);
    requireAccepted(fixture, step, "full-support seed");
    const auto seedObservation = observe(fixture);
    require(std::abs(seedObservation.linearMomentumKgMPerS[1]) < 2.0e-5,
            "zero-total-force support seed acquired vertical momentum");

    double maximumMomentumResidual = 0.0;
    for (std::uint32_t index = 0u; index < rampSteps; ++index) {
        const auto before = observe(fixture);
        const double fraction = shippingGravityTransferSupportFraction(
            index, rampSteps);
        const double actuator = shippingGravityTransferSupportForceY(
            mass, gravityY, fraction);
        step = fixture.step(actuator, index + 1u);
        requireAccepted(fixture, step,
            "gravity-to-actuator transfer step " + std::to_string(index));
        const auto after = observe(fixture);
        // Independent Newton-impulse oracle, not the wrench-construction helper.
        const double expectedImpulse = (mass * gravityY + actuator) *
            fixture.dt;
        const double residual = (after.linearMomentumKgMPerS[1] -
                                 before.linearMomentumKgMPerS[1]) - expectedImpulse;
        maximumMomentumResidual = std::max(maximumMomentumResidual,
                                            std::abs(residual));
        require(std::abs(residual) <= 2.0e-4 *
                    std::max(1.0e-6, std::abs(expectedImpulse)) + 2.0e-7,
                "native rigid momentum did not balance total gravity-plus-actuator impulse");
    }

    const auto beforeRelease = observe(fixture);
    step = fixture.step(0.0, rampSteps + 1u);
    requireAccepted(fixture, step, "separate actuator release");
    const auto afterRelease = observe(fixture);
    const double releaseImpulse = mass * gravityY * fixture.dt;
    require(std::abs((afterRelease.linearMomentumKgMPerS[1] -
                      beforeRelease.linearMomentumKgMPerS[1]) - releaseImpulse) <=
                2.0e-4 * std::abs(releaseImpulse) + 2.0e-7,
            "actuator release did not expose the full downward gravity impulse");
    step = fixture.step(0.0, rampSteps + 2u);
    requireAccepted(fixture, step, "post-release observation");
    require(fixture.runner->sceneBodies()[4u].linearVelocityAndInverseMass.y < 0.0f,
            "released finite-mass body did not accelerate downward");

    const RuntimeStateSnapshot acceptedBeforeReject = fixture.runtime.snapshot();
    const auto sceneBeforeReject = fixture.runner->sceneBodies();
    requireFixedMassInertia(fixture, "pre-rejection accepted state");
    fixture.wrench()[4u].force.y = std::numeric_limits<float>::quiet_NaN();
    const auto rejected = fixture.runner->step(
        fixture.tools, nullptr, 0u, nullptr, 0u,
        (__bridge void*)fixture.wrenchBuffer, 5u, 5u, rampSteps + 3u,
        static_cast<float>(fixture.dt), false);
    require(!rejected.matterAcceptance.accepted,
            "deliberately invalid native body wrench unexpectedly committed");
    require(fixture.runner->acceptedSteps() == rampSteps + 3u,
            "rejected native step advanced the finite-mass acceptance cursor");
    require(std::equal(sceneBeforeReject.begin(), sceneBeforeReject.end(),
                fixture.runner->sceneBodies().begin(),
                [](const MRBodyStateGPU& a, const MRBodyStateGPU& b) {
                    return std::memcmp(&a, &b, sizeof(MRBodyStateGPU)) == 0;
                }),
            "rejected native step changed the accepted MetalWorld scene pose");
    const RuntimeStateSnapshot afterReject = fixture.runtime.snapshot();
    require(sameAcceptedPhysicalState(acceptedBeforeReject, afterReject),
            "rejected native step changed accepted material/contact/rigid history");
    requireFixedMassInertia(fixture, "post-rejection accepted state");
    std::cout << "native_transfer_steps=" << rampSteps
              << " maximum_momentum_residual_kg_m_s="
              << maximumMomentumResidual
              << " release_velocity_y_m_s="
              << fixture.runner->sceneBodies()[4u].linearVelocityAndInverseMass.y
              << " rollback_status=" << rejected.matterAcceptance.statusCode
              << " rollback_reason=" << rejected.matterAcceptance.reason << '\n';
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 3) {
                std::cerr << "usage: cardboard-shipping-rigid-native-check "
                             "<NumiMatter.metallib> <material.nmatter>\n";
                return 2;
            }
            NativeFixture fixture(argv[1], argv[2]);
            runSupportTransferReleaseAndRollback(fixture);
            std::cout << "shipping finite-mass native checks passed: "
                      << checks << " assertions\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
