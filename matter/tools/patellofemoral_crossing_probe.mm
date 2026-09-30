#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif
#ifndef NUMI_MATTER_FIXTURE_MATERIAL
#define NUMI_MATTER_FIXTURE_MATERIAL ""
#endif

namespace {

void require(bool okay, const std::string& message) {
    if (!okay) throw std::runtime_error(message);
}

std::array<std::array<double, 3>, 8> readCells(const char* path) {
    std::ifstream input(path);
    require(input.good(), "cannot open eight-node crossing fixture");
    std::array<std::array<double, 3>, 8> nodes{};
    for (auto& node : nodes)
        for (double& coordinate : node)
            require(bool(input >> coordinate) && std::isfinite(coordinate),
                    "fixture must contain eight finite SI-position triples");
    std::string extra;
    require(!(input >> extra), "fixture has trailing values");
    return nodes;
}

numi::matter::CompiledWorld makeWorld(
        const std::array<std::array<double, 3>, 8>& nodes,
        const std::array<double, 3>& offset) {
    auto material = numi::matter::parseMatterFile(NUMI_MATTER_FIXTURE_MATERIAL);
    require(material.succeeded(), "fixture material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-6;
    source.gravity = {0.0, 0.0, 0.0};
    source.contactSlop = 1.0e-5;
    source.mixedSolver.newtonIterations = 8u;
    source.materials.push_back(std::move(material.material));
    for (std::uint32_t objectIndex = 0u; objectIndex < 2u; ++objectIndex) {
        numi::matter::ObjectSource object;
        object.name = objectIndex == 0u ? "source_ptc_cell" : "source_fmc_cell";
        object.representation = numi::matter::Representation::fem;
        object.materialIndex = 0u;
        object.mixedFEM = false;
        object.deformableContact = true;
        object.deformableSelfContact = false;
        object.characteristicLength = 0.001;
        object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
        object.femCapacity.deformableContacts = 32u;
        for (std::uint32_t local = 0u; local < 4u; ++local) {
            auto position = nodes[objectIndex * 4u + local];
            if (objectIndex == 0u)
                for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                    position[axis] += offset[axis];
            object.femNodes.push_back(position);
        }
        source.objects.push_back(std::move(object));
    }
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto cooked = numi::matter::compileWorld(source, options);
    std::string diagnostic = "source crossing cells did not compile";
    for (const auto& row : cooked.diagnostics)
        diagnostic += "; " + row.message;
    require(cooked.succeeded(), diagnostic);
    require(cooked.world.fem.tetrahedra.size() == 2u &&
            cooked.world.fem.nodes.size() == 8u,
            "cooked fixture topology drift");
    return std::move(cooked.world);
}

void run(const numi::matter::CompiledWorld& world) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MRMetalWorldStatusGPU worldStatus{};
        worldStatus.code = MR_STEP_SUCCESS;
        id<MTLBuffer> statuses = [device
            newBufferWithBytes:&worldStatus length:sizeof(worldStatus)
            options:MTLResourceStorageModeShared];
        require(queue != nil && statuses != nil, "Metal setup failed");
        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(world, {
            .metallib = NUMI_MATTER_METALLIB,
            .environmentCount = 1u,
            .captureEvents = false,
            .captureDiagnostics = true,
            .automaticIdentification = false,
            .adaptiveTransfer = false,
        });
        require(initialized.encoded && runtime.valid(),
                "runtime initialization: " + initialized.message);
        const auto before = runtime.snapshot();
        require(before.available && before.femNodes.size() == 8u,
                "initial snapshot: " + before.message);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "cannot create Metal command buffer");
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
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
                "Metal command did not complete");
        const auto after = runtime.snapshot();
        require(after.available && after.statuses.size() == 1u &&
                after.femNodes.size() == 8u, "completion snapshot failed");
        bool rolledBack = true;
        for (std::size_t index = 0u; index < 8u; ++index)
            rolledBack &= std::memcmp(&before.femNodes[index],
                &after.femNodes[index], sizeof(NMFEMNodeStateGPU)) == 0;
        const auto& status = after.statuses[0];
        std::size_t activeHistories = 0u;
        for (const auto& history : after.deformableContactHistories)
            activeHistories += history.laggedTangentAndFriction.w > 0.5f;
        std::printf("{\"device\":\"%s\",\"abi\":%u,"
                    "\"status_code\":%u,\"completed_microsteps\":%u,"
                    "\"contact_count\":%u,\"failing_index\":%u,"
                    "\"surface_faces\":%zu,\"active_deformable_histories\":%zu,"
                    "\"rollback_bitwise\":%s,\"diagnostics\":[%.9g,%.9g,%.9g,%.9g],"
                    "\"accepted_positions_m\":[",
                    [[device name] UTF8String], NM_MATTER_ABI_VERSION,
                    status.code, status.completedMicrosteps,
                    status.contactCount, status.failingIndex,
                    world.fem.surfaceFaces.size(), activeHistories,
                    rolledBack ? "true" : "false",
                    double(status.diagnostics.x), double(status.diagnostics.y),
                    double(status.diagnostics.z), double(status.diagnostics.w));
        for (std::size_t index = 0u; index < 8u; ++index) {
            const auto& position = after.femNodes[index].positionAndMass;
            std::printf("%s[%.9g,%.9g,%.9g]", index == 0u ? "" : ",",
                        double(position.x), double(position.y),
                        double(position.z));
        }
        std::printf("]}\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 5,
                "usage: probe eight-node-fixture.txt [ptc-offset-x y z]");
        const auto nodes = readCells(argv[1]);
        std::array<double, 3> offset{};
        if (argc == 5)
            for (int axis = 0; axis < 3; ++axis) {
                offset[axis] = std::strtod(argv[2 + axis], nullptr);
                require(std::isfinite(offset[axis]), "nonfinite offset");
            }
        run(makeWorld(nodes, offset));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "patellofemoral crossing probe: %s\n", error.what());
        return 1;
    }
}
