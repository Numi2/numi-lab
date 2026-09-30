#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif
#ifndef NUMI_MATTER_CARDIAC_MATERIAL
#define NUMI_MATTER_CARDIAC_MATERIAL ""
#endif

namespace {

void require(bool okay, const std::string& message) {
    if (!okay) throw std::runtime_error(message);
}

struct SuppliedCell {
    std::array<std::array<double, 3>, 4> nodes{};
    std::array<double, 4> frame{};
    float tension = 0.0f;
};

numi::matter::CompiledWorld makeWorld(const SuppliedCell* supplied = nullptr) {
    auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_CARDIAC_MATERIAL);
    require(parsed.succeeded(), "source ventricular material did not parse");
    bool density = false;
    for (auto& parameter : parsed.material.parameters) {
        if (parameter.name != "density") continue;
        parameter.defaultValue = 1050.0; // Explicit synthetic inertial fixture.
        density = true;
    }
    require(density, "source material has no density sentinel");
    parsed.material.mixed.fibreDirection = {1.0, 0.0, 0.0};
    parsed.material.mixed.maximumActiveTension = 120000.0;

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = supplied == nullptr ? 1.0e-4 : 1.0e-6;
    source.gravity = {0.0, 0.0, 0.0};
    source.mixedSolver.newtonIterations = 8u;
    source.materials.push_back(std::move(parsed.material));
    numi::matter::ObjectSource wall;
    wall.name = "framed_source_law_active_tension_fixture";
    wall.materialIndex = 0u;
    wall.representation = numi::matter::Representation::fem;
    wall.mixedFEM = false;
    wall.deformableContact = false;
    wall.deformableSelfContact = false;
    wall.characteristicLength = 0.01;
    wall.femNodes = supplied == nullptr
        ? std::vector<std::array<double, 3>>{
            {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
            {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01}}
        : std::vector<std::array<double, 3>>(
            supplied->nodes.begin(), supplied->nodes.end());
    wall.femFixedNodes = {0u, 1u, 2u};
    wall.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    constexpr double halfRootTwo = 0.7071067811865475244;
    wall.femMaterialFrameRotations = {supplied == nullptr
        ? std::array<double, 4>{0.0, -halfRootTwo, 0.0, halfRootTwo}
        : supplied->frame};
    wall.femMaterialFrameSourceIdentity = {1u, 2u, 3u, 4u};
    wall.femMaterialIndices = {0u};
    wall.femMaterialSourceIdentity = {5u, 6u, 7u, 8u};
    source.objects.push_back(std::move(wall));

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    std::string failure = "framed unmixed source material did not compile";
    for (const auto& row : compiled.diagnostics)
        failure += "; " + row.message;
    require(compiled.succeeded(), failure);
    require(compiled.world.fem.tetrahedra.size() == 1u &&
            (compiled.world.objects[0].flags & NM_OBJECT_MIXED_FEM) == 0u,
            "probe did not retain exact unmixed FEM ownership");
    return std::move(compiled.world);
}

struct Outcome {
    NMFEMNodeStateGPU initialTip{};
    NMFEMNodeStateGPU tip{};
    NMMatterStatusGPU status{};
    std::string device;
};

Outcome run(const numi::matter::CompiledWorld& world,
            std::optional<float> tension) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "no Metal queue");
        MRMetalWorldStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_SUCCESS;
        id<MTLBuffer> worldStatuses = [device
            newBufferWithBytes:&initialStatus length:sizeof(initialStatus)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> activeTensions = tension.has_value()
            ? [device newBufferWithBytes:&*tension length:sizeof(float)
                options:MTLResourceStorageModeShared]
            : nil;
        require(worldStatuses != nil &&
                (!tension.has_value() || activeTensions != nil),
                "probe buffer allocation failed");

        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(world, {
            .metallib = NUMI_MATTER_METALLIB,
            .environmentCount = 1u,
            .captureEvents = false,
            .captureDiagnostics = true,
            .automaticIdentification = false,
            .adaptiveTransfer = false,
        });
        require(initialized.encoded && runtime.valid(), initialized.message);
        const auto initial = runtime.snapshot();
        require(initial.available && initial.femNodes.size() == 4u,
                initial.message);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "no Metal command buffer");
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)worldStatuses;
        request.femActiveTensions = (__bridge void*)activeTensions;
        request.femActiveTensionCount = tension.has_value() ? 1u : 0u;
        request.controlStep = 0u;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.phase = numi::matter::EncodePhase::preDynamics;
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "preDynamics: " + encoded.message);
        request.phase = numi::matter::EncodePhase::postCommit;
        encoded = runtime.encode(request);
        require(encoded.encoded, "postCommit: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native command buffer did not complete");
        const auto snapshot = runtime.snapshot();
        require(snapshot.available && snapshot.femNodes.size() == 4u &&
                snapshot.statuses.size() == 1u, snapshot.message);
        return {initial.femNodes[3], snapshot.femNodes[3], snapshot.statuses[0],
                std::string([[device name] UTF8String])};
    }
}

void rejectBadBuffer(const numi::matter::CompiledWorld& world,
                     std::size_t bytes, std::uint32_t count,
                     const std::string& expected) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> input = [device newBufferWithLength:bytes
            options:MTLResourceStorageModeShared];
        MRMetalWorldStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_SUCCESS;
        id<MTLBuffer> statuses = [device
            newBufferWithBytes:&initialStatus length:sizeof(initialStatus)
            options:MTLResourceStorageModeShared];
        require(device != nil && queue != nil && input != nil && statuses != nil,
                "invalid-buffer control could not allocate Metal resources");
        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(world, {
            .metallib = NUMI_MATTER_METALLIB, .environmentCount = 1u,
            .adaptiveTransfer = false,
        });
        require(initialized.encoded, initialized.message);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.femActiveTensions = (__bridge void*)input;
        request.femActiveTensionCount = count;
        request.phase = numi::matter::EncodePhase::preDynamics;
        request.timestepSeconds = runtime.timestepSeconds();
        const auto encoded = runtime.encode(request);
        require(!encoded.encoded && encoded.message.find(expected) != std::string::npos,
                "bad active-tension buffer was not rejected: " + encoded.message);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        SuppliedCell supplied;
        if (argc != 1) {
            require(argc == 19 && std::string(argv[1]) == "--cell",
                    "usage: probe [--cell 12_positions 4_quaternion tension]");
            for (int i = 0; i < 12; ++i) {
                supplied.nodes[i / 3][i % 3] = std::strtod(argv[2 + i], nullptr);
                require(std::isfinite(supplied.nodes[i / 3][i % 3]),
                        "nonfinite source cell coordinate");
            }
            double norm = 0.0;
            for (int i = 0; i < 4; ++i) {
                supplied.frame[i] = std::strtod(argv[14 + i], nullptr);
                require(std::isfinite(supplied.frame[i]),
                        "nonfinite source material quaternion");
                norm += supplied.frame[i] * supplied.frame[i];
            }
            require(std::abs(norm - 1.0) < 1.0e-10,
                    "nonunit source material quaternion");
            supplied.tension = static_cast<float>(std::strtod(argv[18], nullptr));
            require(std::isfinite(supplied.tension) &&
                    supplied.tension > 0.0f && supplied.tension <= 120000.0f,
                    "invalid source cell tension");
        }
        const bool sourceCell = argc == 19;
        const auto world = makeWorld(sourceCell ? &supplied : nullptr);
        rejectBadBuffer(world, sizeof(float), 0u,
                        "active-tension field must exactly cover");
        rejectBadBuffer(world, 2u, 1u,
                        "active-tension Metal buffer is undersized");
        const auto baseline = run(world, std::nullopt);
        const auto explicitZero = run(world, 0.0f);
        const float selectedTension = sourceCell ? supplied.tension : 100.0f;
        const auto active = run(world, selectedTension);
        const auto replay = run(world, selectedTension);
        const auto rejected = run(world, 120001.0f);
        const auto negative = run(world, -1.0f);
        const auto nonfinite = run(world,
            std::numeric_limits<float>::quiet_NaN());
        require(baseline.status.code == NM_STATUS_SUCCESS &&
                explicitZero.status.code == NM_STATUS_SUCCESS &&
                active.status.code == NM_STATUS_SUCCESS &&
                replay.status.code == NM_STATUS_SUCCESS,
                "native baseline/active/replay did not accept");
        require(rejected.status.code == NM_STATUS_NONFINITE_INPUT &&
                negative.status.code == NM_STATUS_NONFINITE_INPUT &&
                nonfinite.status.code == NM_STATUS_NONFINITE_INPUT,
                "invalid active tension did not fail closed");
        require(std::memcmp(&baseline.tip, &explicitZero.tip,
                            sizeof(baseline.tip)) == 0,
                "zero active tension changed legacy FEM state");
        require(std::memcmp(&rejected.initialTip, &rejected.tip,
                            sizeof(rejected.tip)) == 0 &&
                std::memcmp(&negative.initialTip, &negative.tip,
                            sizeof(negative.tip)) == 0 &&
                std::memcmp(&nonfinite.initialTip, &nonfinite.tip,
                            sizeof(nonfinite.tip)) == 0,
                "rejected tension changed accepted FEM state");
        require(std::memcmp(&active.tip, &replay.tip, sizeof(active.tip)) == 0,
                "native active step did not replay bitwise");
        const float dx = active.tip.positionAndMass.x -
            baseline.tip.positionAndMass.x;
        const float dy = active.tip.positionAndMass.y -
            baseline.tip.positionAndMass.y;
        const float dz = active.tip.positionAndMass.z -
            baseline.tip.positionAndMass.z;
        const float displacement = std::sqrt(dx * dx + dy * dy + dz * dz);
        require(std::isfinite(displacement) && displacement > 1.0e-10f &&
                (sourceCell || dz < -1.0e-8f),
                "native active fibre did not move the free tip");
        std::printf("{\"device\":\"%s\",\"abi\":%u,"
                    "\"accepted_active_steps\":1,\"baseline_code\":%u,"
                    "\"active_code\":%u,\"rejected_code\":%u,"
                    "\"tip_dz_m\":%.9g,\"tip_delta_norm_m\":%.9g,"
                    "\"source_cell\":%s,\"replay_bitwise\":true}\n",
                    active.device.c_str(), NM_MATTER_ABI_VERSION,
                    baseline.status.code, active.status.code,
                    rejected.status.code, static_cast<double>(dz),
                    static_cast<double>(displacement),
                    sourceCell ? "true" : "false");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fem active tension probe: %s\n", error.what());
        return 1;
    }
}
