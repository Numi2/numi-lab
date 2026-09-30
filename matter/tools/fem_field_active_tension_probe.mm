#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
};

numi::matter::CompiledWorld makeWorld(bool fieldStress,
                                     const SuppliedCell* supplied) {
    auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_CARDIAC_MATERIAL);
    require(parsed.succeeded(), "source Guccione law did not parse");
    bool density = false;
    for (auto& parameter : parsed.material.parameters) {
        if (parameter.name != "density") continue;
        parameter.defaultValue = 1050.0; // Synthetic inertia.
        density = true;
    }
    require(density, "source law lacks density parameter");
    parsed.material.mixed.fibreDirection = {1.0, 0.0, 0.0};
    parsed.material.mixed.heatCapacity = 1.0; // Synthetic field mass.
    parsed.material.mixed.jouleHeatFraction = 0.0;
    parsed.material.mixed.poreStorage = 1.0; // Keep the unused pore row nonsingular.
    parsed.material.mixed.maximumActiveTension = 1000.0;
    parsed.material.mixed.electricalConductivity = 0.1;
    parsed.material.mixed.activationDiffusivity = 0.0;
    parsed.material.mixed.activationOnRate = 10000.0;
    parsed.material.mixed.activationOffRate = 0.0;
    parsed.material.mixed.activationThreshold = 0.2;
    parsed.material.mixed.activationSlope = 12.0;

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0e-4;
    source.gravity = {0.0, 0.0, 0.0};
    source.mixedSolver.newtonIterations = 10u;
    source.materials.push_back(std::move(parsed.material));
    numi::matter::ObjectSource wall;
    wall.name = "synthetic_electric_to_guccione_stress";
    wall.materialIndex = 0u;
    wall.representation = numi::matter::Representation::fem;
    wall.mixedFEM = false;
    wall.femFieldDrivenActiveTension = fieldStress;
    wall.deformableContact = false;
    wall.deformableSelfContact = false;
    wall.femNodes = supplied == nullptr
        ? std::vector<std::array<double, 3>>{
            {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
            {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01}}
        : std::vector<std::array<double, 3>>(
            supplied->nodes.begin(), supplied->nodes.end());
    wall.femFixedNodes = {0u, 1u, 2u};
    wall.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    constexpr double h = 0.7071067811865475244;
    wall.femMaterialFrameRotations = {supplied == nullptr
        ? std::array<double, 4>{0.0, -h, 0.0, h}
        : supplied->frame};
    wall.femMaterialFrameSourceIdentity = {1u, 2u, 3u, 4u};
    wall.multiphysics.enabled = true;
    wall.multiphysics.initialTemperature = 293.15;
    wall.multiphysics.initialElectricPotential = 0.5;
    wall.multiphysics.initialActivation = 0.0;
    numi::matter::FieldBoundarySource ground;
    ground.node = 0u;
    ground.stableIdentifier = 1u;
    ground.flags = NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL;
    ground.value = {0.0, 0.0, 0.0, 0.0};
    wall.fieldBoundaries.push_back(ground);
    numi::matter::FieldBoundarySource stimulus;
    stimulus.node = 1u;
    stimulus.stableIdentifier = 2u;
    stimulus.flags = NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL;
    stimulus.value = {0.0, 0.0, 1.0, 0.0};
    wall.fieldBoundaries.push_back(stimulus);
    source.objects.push_back(std::move(wall));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    std::string error = "non-mixed field-driven world did not compile";
    for (const auto& row : compiled.diagnostics) error += "; " + row.message;
    require(compiled.succeeded(), error);
    const auto flags = compiled.world.objects[0].flags;
    require((flags & NM_OBJECT_MIXED_FEM) == 0u &&
            (flags & NM_OBJECT_MULTIPHYSICS) != 0u &&
            ((flags & NM_OBJECT_FEM_FIELD_ACTIVE_TENSION) != 0u) == fieldStress,
            "compiled field and passive-law ownership changed");
    return std::move(compiled.world);
}

struct Outcome {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMFEMFieldStateGPU> fields;
    NMMatterStatusGPU status{};
    float initialActivation = 0.0f;
    std::string device;
};

Outcome run(const numi::matter::CompiledWorld& world) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MRMetalWorldStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_SUCCESS;
        id<MTLBuffer> worldStatus = [device
            newBufferWithBytes:&initialStatus length:sizeof(initialStatus)
            options:MTLResourceStorageModeShared];
        require(device != nil && queue != nil && worldStatus != nil,
                "no native Metal resources");
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
        const auto before = runtime.snapshot();
        require(before.available && before.femNodes.size() == 4u &&
                before.femFields.size() == 4u, before.message);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "no borrowed command buffer");
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)worldStatus;
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
                "native Metal command did not complete");
        const auto after = runtime.snapshot();
        require(after.available && after.femNodes.size() == 4u &&
                after.femFields.size() == 4u && after.statuses.size() == 1u,
                after.message);
        return {after.femNodes, after.femFields, after.statuses[0],
                before.femFields[3].secondary.x,
                std::string([[device name] UTF8String])};
    }
}

void rejectDoubleOwner(const numi::matter::CompiledWorld& world) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MRMetalWorldStatusGPU initialStatus{};
        initialStatus.code = MR_STEP_SUCCESS;
        float tension = 10.0f;
        id<MTLBuffer> statuses = [device
            newBufferWithBytes:&initialStatus length:sizeof(initialStatus)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> active = [device
            newBufferWithBytes:&tension length:sizeof(tension)
            options:MTLResourceStorageModeShared];
        require(device != nil && statuses != nil && active != nil,
                "double-owner control allocation failed");
        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(world, {
            .metallib = NUMI_MATTER_METALLIB,
            .environmentCount = 1u,
            .adaptiveTransfer = false,
        });
        require(initialized.encoded, initialized.message);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)statuses;
        request.femActiveTensions = (__bridge void*)active;
        request.femActiveTensionCount = 1u;
        request.phase = numi::matter::EncodePhase::preDynamics;
        request.timestepSeconds = runtime.timestepSeconds();
        const auto encoded = runtime.encode(request);
        require(!encoded.encoded &&
                encoded.message.find("cannot share one force owner") != std::string::npos,
                "two active stress owners were admitted: " + encoded.message);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        SuppliedCell supplied;
        if (argc != 1) {
            require(argc == 18 && std::string(argv[1]) == "--cell",
                    "usage: probe [--cell 12_positions 4_quaternion]");
            for (int i = 0; i < 12; ++i) {
                supplied.nodes[i / 3][i % 3] = std::strtod(argv[2 + i], nullptr);
                require(std::isfinite(supplied.nodes[i / 3][i % 3]),
                        "nonfinite source cell coordinate");
            }
            double norm = 0.0;
            for (int i = 0; i < 4; ++i) {
                supplied.frame[i] = std::strtod(argv[14 + i], nullptr);
                require(std::isfinite(supplied.frame[i]),
                        "nonfinite source material frame");
                norm += supplied.frame[i] * supplied.frame[i];
            }
            require(std::abs(norm - 1.0) < 1.0e-10,
                    "source material frame is not unit length");
        }
        const SuppliedCell* cell = argc == 1 ? nullptr : &supplied;
        const auto passive = makeWorld(false, cell);
        const auto coupled = makeWorld(true, cell);
        rejectDoubleOwner(coupled);
        const auto baseline = run(passive);
        const auto active = run(coupled);
        const auto replay = run(coupled);
        require(baseline.status.code == NM_STATUS_SUCCESS &&
                active.status.code == NM_STATUS_SUCCESS &&
                replay.status.code == NM_STATUS_SUCCESS &&
                active.status.completedMicrosteps == 1u &&
                baseline.status.completedMicrosteps == 1u,
                "native electrical/mechanical transaction did not accept: passive=" +
                std::to_string(baseline.status.code) + "/" +
                std::to_string(baseline.status.completedMicrosteps) +
                " active=" + std::to_string(active.status.code) + "/" +
                std::to_string(active.status.completedMicrosteps) +
                " replay=" + std::to_string(replay.status.code) + "/" +
                std::to_string(replay.status.completedMicrosteps) +
                " passive_index=" + std::to_string(baseline.status.failingIndex) +
                " active_index=" + std::to_string(active.status.failingIndex) +
                " detail=" + std::to_string(active.status.diagnostics.x) + "," +
                std::to_string(active.status.diagnostics.y) + "," +
                std::to_string(active.status.diagnostics.z) + "," +
                std::to_string(active.status.diagnostics.w));
        require(active.initialActivation == 0.0f &&
                active.fields[3].secondary.x > 0.0f &&
                baseline.fields[3].secondary.x > 0.0f,
                "native activation field did not evolve");
        require(std::memcmp(active.nodes.data(), replay.nodes.data(),
                            active.nodes.size()*sizeof(NMFEMNodeStateGPU)) == 0 &&
                std::memcmp(active.fields.data(), replay.fields.data(),
                            active.fields.size()*sizeof(NMFEMFieldStateGPU)) == 0,
                "coupled electrical/mechanical replay changed");
        const auto& a = active.nodes[3].positionAndMass;
        const auto& b = baseline.nodes[3].positionAndMass;
        const double distance = std::sqrt(
            std::pow(double(a.x)-b.x, 2) +
            std::pow(double(a.y)-b.y, 2) +
            std::pow(double(a.z)-b.z, 2));
        require(distance > 1.0e-10,
                "field-driven activation did not change mechanics");
        std::printf("{\"status\":\"accepted_nonmixed_field_to_active_stress\","
                    "\"device\":\"%s\",\"accepted_steps\":1,"
                    "\"initial_tip_activation\":%.9g,"
                    "\"accepted_tip_activation\":%.9g,"
                    "\"tip_change_vs_field_only_m\":%.12g,"
                    "\"replay_bitwise\":true,"
                    "\"double_active_stress_owner_rejected\":true,"
                    "\"synthetic_electrical_parameters\":true,"
                    "\"source_cell_geometry\":%s,"
                    "\"source_model_reproduced\":false,"
                    "\"heartbeat_qualified\":false}\n",
                    active.device.c_str(), active.initialActivation,
                    active.fields[3].secondary.x, distance,
                    cell == nullptr ? "false" : "true");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "field active-tension probe: %s\n", error.what());
        return 1;
    }
}
