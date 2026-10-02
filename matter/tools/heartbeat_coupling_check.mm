#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"
#include "vascular_cavity_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {

using namespace numi::matter;
namespace fixture = vascular_cavity_fixture;

void require(bool okay, const std::string& message) {
    if (!okay) throw std::runtime_error(message);
}

VascularConnectionSource directionalValve(std::uint32_t id, std::uint32_t from,
                                          std::uint32_t to, double forward,
                                          double reverse) {
    VascularConnectionSource edge;
    edge.stableIdentifier = id;
    edge.fromCompartment = from;
    edge.toCompartment = to;
    edge.resistance = forward;
    edge.reverseResistance = reverse;
    edge.flowScale = 1.0e-6;
    edge.pressureScale = 2.0e4;
    edge.flowResidualTolerance = 1.0e-5;
    edge.flowLaw = VascularFlowLaw::directionalResistance;
    return edge;
}

void configureReservoir(VascularCompartmentSource& reservoir,
                        std::uint32_t id, const char* name,
                        double referencePressure, double amount) {
    reservoir.stableIdentifier = id;
    reservoir.anatomicalIdentifier = name;
    reservoir.referenceVolume = 4.0e-6;
    reservoir.initialVolume = 4.0e-6;
    reservoir.referencePressure = referencePressure;
    reservoir.compliance = 1.0e-7;
    reservoir.externalPressure = 0.0;
    reservoir.volumeScale = 1.0e-6;
    reservoir.volumeResidualTolerance = 1.0e-5;
    reservoir.initialSpeciesAmounts = {amount};
    reservoir.pressureLaw = VascularPressureLaw::linearCompliance;
}

WorldSource heartbeatSource() {
    WorldSource source = fixture::world(false, 4.0e-3);
    require(!source.materials.empty(), "heartbeat fixture has no material owner");
    source.materials[0].mixed.fibreDirection = {1.0, 0.0, 0.0};
    source.materials[0].mixed.maximumActiveTension = 20000.0;
    bool shear = false;
    bool bulk = false;
    for (auto& parameter : source.materials[0].parameters) {
        if (parameter.name == "mu") {
            parameter.defaultValue = 25000.0;
            shear = true;
        } else if (parameter.name == "lambda") {
            parameter.defaultValue = 250000.0;
            bulk = true;
        }
    }
    require(shear && bulk, "heartbeat fixture passive material scale changed");
    auto& vascular = source.vascular;
    auto& cavity = vascular.compartments[0];
    auto venous = vascular.compartments[1];
    VascularCompartmentSource arterial;
    configureReservoir(venous, 12u, "synthetic:venous_return", 100.0, 4.0e-6);
    configureReservoir(arterial, 13u, "synthetic:arterial_storage", 500.0, 0.0);
    vascular.compartments = {cavity, venous, arterial};
    vascular.connections = {
        directionalValve(21u, 12u, 11u, 1.0e7, 1.0e14),
        directionalValve(22u, 11u, 13u, 2.0e9, 1.0e14),
    };
    vascular.cavities[0].pressureScale = 2.0e4;

    // Give every cell a circumferential material fibre about the hollow lumen.
    // The frame and the cavity are deliberately synthetic fixture inputs.
    auto& object = source.objects[0];
    object.femMaterialFrameRotations.reserve(object.tetrahedra.size());
    for (const auto& tet : object.tetrahedra) {
        std::array<double, 3> center{};
        for (const auto node : tet.nodes) {
            for (std::size_t axis = 0; axis < 3u; ++axis)
                center[axis] += object.femNodes[node][axis] * 0.25;
        }
        const double radius = std::hypot(center[0], center[1]);
        require(radius > 1.0e-12, "synthetic cell fibre has no circumferential direction");
        const double fibreX = -center[1] / radius;
        const double fibreY = center[0] / radius;
        const double cosine = std::clamp(fibreX, -1.0, 1.0);
        if (cosine < -0.999999) {
            object.femMaterialFrameRotations.push_back({0.0, 0.0, 1.0, 0.0});
        } else {
            object.femMaterialFrameRotations.push_back({
                0.0, 0.0,
                fibreY / std::sqrt(2.0 * (1.0 + cosine)),
                std::sqrt((1.0 + cosine) * 0.5),
            });
        }
    }
    object.femMaterialFrameSourceIdentity = {
        0x73796e74682d6d79ull, 0x6f2d666962726573ull,
        0x6865617462656174ull, 1ull,
    };
    return source;
}

double activeTension(double timeSeconds) {
    constexpr double period = 0.8;
    constexpr double onset = 0.08;
    constexpr double duration = 0.24;
    double phase = std::fmod(timeSeconds, period);
    if (phase < 0.0) phase += period;
    if (phase <= onset || phase >= onset + duration) return 0.0;
    const double normalized = (phase - onset) / duration;
    const double pulse = std::sin(std::numbers::pi * normalized);
    return 15000.0 * pulse * pulse;
}

std::vector<fixture::Vec> nodePositions(const NMFEMNodeStateGPU* nodes,
                                       std::size_t nodeCount,
                                       std::size_t environment) {
    std::vector<fixture::Vec> positions(nodeCount);
    const auto* envNodes = nodes + environment * nodeCount;
    for (std::size_t i = 0; i < nodeCount; ++i) {
        const auto& p = envNodes[i].positionAndMass;
        positions[i] = {p.x, p.y, p.z};
    }
    return positions;
}

double physical(const nm_float4* states, std::size_t stateCount,
                std::size_t environment, std::uint32_t row,
                const std::vector<NMVascularUnknownGPU>& unknowns) {
    require(row < stateCount && row < unknowns.size(), "vascular output row is outside the compiled layout");
    return double(states[environment * stateCount + row].x) *
        unknowns[row].initialAndScaling.y;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: numi-matter-heartbeat-coupling-check OUTPUT.csv");
        constexpr std::uint32_t stepsPerBeat = 200u;
        constexpr std::uint32_t beatCount = 2u;
        constexpr std::uint32_t totalSteps = stepsPerBeat * beatCount;
        constexpr std::size_t environmentCount = 2u;

        auto source = heartbeatSource();
        const auto shell = fixture::hollowShell();
        const auto compiled = compileWorld(source, {.maximumRateExponent = 0u});
        std::string compileErrors;
        for (const auto& diagnostic : compiled.diagnostics) {
            if (diagnostic.severity == Diagnostic::Severity::error)
                compileErrors += diagnostic.message + "; ";
        }
        require(compiled.succeeded(), "heartbeat fixture compile: " + compileErrors);
        const auto& world = compiled.world;
        require(world.dispatch.environmentCount == environmentCount &&
                world.dispatch.femNodeCount == 64u &&
                world.fem.tetrahedra.size() == 156u &&
                world.vascular.compartments.size() == 3u &&
                world.vascular.connections.size() == 2u,
                "compiled heartbeat fixture dimensions changed");

        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil && [[device name] rangeOfString:@"Apple"].location != NSNotFound,
                    "physical Apple Metal is required");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue unavailable");

            std::array<MRMetalWorldStatusGPU, environmentCount> initialWorldStatus{};
            for (std::size_t env = 0; env < environmentCount; ++env) {
                initialWorldStatus[env].environment = static_cast<std::uint32_t>(env);
                initialWorldStatus[env].code = MR_STEP_SUCCESS;
            }
            id<MTLBuffer> worldStatusBuffer = [device
                newBufferWithBytes:initialWorldStatus.data()
                length:sizeof(initialWorldStatus)
                options:MTLResourceStorageModeShared];
            const std::size_t nodeCount = world.dispatch.femNodeCount;
            const std::size_t unknownCount = world.vascular.unknowns.size();
            const std::size_t nodeBytes = environmentCount * nodeCount *
                sizeof(NMFEMNodeStateGPU);
            const std::size_t vascularBytes = environmentCount * unknownCount *
                sizeof(nm_float4);
            const std::size_t tensionCount = environmentCount *
                world.fem.tetrahedra.size();
            id<MTLBuffer> tensionBuffer = [device
                newBufferWithLength:tensionCount * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> nodeReadback = [device
                newBufferWithLength:nodeBytes
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> vascularReadback = [device
                newBufferWithLength:vascularBytes
                options:MTLResourceStorageModeShared];
            require(worldStatusBuffer != nil && tensionBuffer != nil &&
                    nodeReadback != nil && vascularReadback != nil,
                    "heartbeat fixture Metal buffer allocation failed");

            Runtime runtime;
            const auto initialized = runtime.initialize(world, {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = static_cast<std::uint32_t>(environmentCount),
                .captureEvents = false,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            });
            require(initialized.encoded && runtime.valid(),
                    "Matter heartbeat runtime: " + initialized.message);
            id<MTLBuffer> matterStatusBuffer =
                (__bridge id<MTLBuffer>)runtime.statusBuffer();
            auto* matterStatus = static_cast<NMMatterStatusGPU*>(matterStatusBuffer.contents);
            auto* tension = static_cast<float*>(tensionBuffer.contents);
            require(matterStatusBuffer != nil && matterStatus != nullptr && tension != nullptr,
                    "heartbeat fixture shared runtime buffer is unavailable");

            const auto* sourceNodes = static_cast<const NMFEMNodeStateGPU*>(world.fem.nodes.data());
            std::vector<double> cycleOut(environmentCount * beatCount, 0.0);
            std::vector<double> cycleIn(environmentCount * beatCount, 0.0);
            double maximumBloodVolumeError = 0.0;
            double maximumCavityVolumeError = 0.0;
            double maximumActiveWallDisplacement = 0.0;
            double maximumControlWallDisplacement = 0.0;
            double maximumActiveControlWallDifference = 0.0;
            double maximumActiveTension = 0.0;
            double maximumActiveAorticFlow = 0.0;
            double maximumControlAorticFlow = 0.0;
            std::uint64_t totalNewtonLinearIterations = 0u;
            const double initialBloodVolume = 9.0e-6;
            const double timestep = runtime.timestepSeconds();
            require(std::abs(timestep - 4.0e-3) < 1.0e-8,
                    "heartbeat fixture timestep changed");

            const auto csvPath = std::filesystem::path(argv[1]);
            if (csvPath.has_parent_path())
                std::filesystem::create_directories(csvPath.parent_path());
            std::ofstream trace(csvPath);
            require(trace.good(), "cannot create heartbeat trace");
            trace << std::setprecision(12)
                  << "time_s,active_tension_pa,active_wall_volume_m3,control_wall_volume_m3,"
                  << "active_hydraulic_volume_m3,control_hydraulic_volume_m3,"
                  << "active_inlet_flow_m3_s,active_aortic_flow_m3_s,"
                  << "control_aortic_flow_m3_s,active_chamber_pressure_pa,"
                  << "active_wall_max_displacement_m,control_wall_max_displacement_m,"
                  << "total_blood_volume_error_fraction\n";

            const auto start = std::chrono::steady_clock::now();
            const auto femAccepted = (__bridge id<MTLBuffer>)runtime.femAcceptedNodeBuffer();
            const auto vascularAccepted = (__bridge id<MTLBuffer>)runtime.vascularAcceptedStateBuffer();
            require(femAccepted != nil && vascularAccepted != nil,
                    "Matter accepted-state buffer is unavailable");
            for (std::uint32_t step = 0; step < totalSteps; ++step) {
                const double time = (double(step) + 0.5) * timestep;
                const float active = static_cast<float>(activeTension(time));
                maximumActiveTension = std::max(maximumActiveTension, double(active));
                const std::size_t elements = world.fem.tetrahedra.size();
                std::fill(tension, tension + elements, active);
                std::fill(tension + elements, tension + 2u * elements, 0.0f);
                auto* statuses = static_cast<MRMetalWorldStatusGPU*>(worldStatusBuffer.contents);
                for (std::size_t env = 0; env < environmentCount; ++env) {
                    statuses[env] = {};
                    statuses[env].environment = static_cast<std::uint32_t>(env);
                    statuses[env].code = MR_STEP_SUCCESS;
                }

                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "heartbeat command buffer unavailable");
                EncodeRequest request{};
                request.commandBuffer = (__bridge void*)command;
                request.environmentStatuses = (__bridge void*)worldStatusBuffer;
                request.femActiveTensions = (__bridge void*)tensionBuffer;
                request.femActiveTensionCount = static_cast<std::uint32_t>(tensionCount);
                request.controlStep = step;
                request.physicsSubstep = 0u;
                request.physicsSubsteps = 1u;
                request.timestepSeconds = timestep;
                request.runAdaptiveTransfer = false;
                request.phase = EncodePhase::preDynamics;
                auto encoded = runtime.encode(request);
                require(encoded.encoded, "heartbeat preDynamics: " + encoded.message);
                request.phase = EncodePhase::postCommit;
                encoded = runtime.encode(request);
                require(encoded.encoded, "heartbeat postCommit: " + encoded.message);

                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                require(blit != nil, "heartbeat readback encoder unavailable");
                [blit copyFromBuffer:femAccepted sourceOffset:0u
                            toBuffer:nodeReadback destinationOffset:0u size:nodeBytes];
                [blit copyFromBuffer:vascularAccepted sourceOffset:0u
                            toBuffer:vascularReadback destinationOffset:0u size:vascularBytes];
                [blit endEncoding];
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "heartbeat Metal command failed at step " + std::to_string(step));
                for (std::size_t env = 0; env < environmentCount; ++env) {
                    require(matterStatus[env].code == NM_STATUS_SUCCESS,
                            "heartbeat step " + std::to_string(step) +
                            " environment " + std::to_string(env) +
                            " rejected with status " + std::to_string(matterStatus[env].code) +
                            " completed_microsteps " + std::to_string(matterStatus[env].completedMicrosteps) +
                            " fgmres_iterations " + std::to_string(matterStatus[env].fgmresIterations) +
                            " failing_index " + std::to_string(matterStatus[env].failingIndex) +
                            " residual " + std::to_string(matterStatus[env].diagnostics.z));
                    totalNewtonLinearIterations += matterStatus[env].fgmresIterations;
                }

                const auto* acceptedNodes = static_cast<const NMFEMNodeStateGPU*>(nodeReadback.contents);
                const auto* acceptedVascular = static_cast<const nm_float4*>(vascularReadback.contents);
                const auto activeNodes = nodePositions(acceptedNodes, nodeCount, 0u);
                const auto controlNodes = nodePositions(acceptedNodes, nodeCount, 1u);
                const double activeWallVolume = fixture::volume(activeNodes, shell.lumenFaces);
                const double controlWallVolume = fixture::volume(controlNodes, shell.lumenFaces);
                const std::uint32_t cavityRow = 0u;
                const std::uint32_t inletRow = world.vascular.layout.offsets.y;
                const std::uint32_t aorticRow = inletRow + 1u;
                const std::uint32_t pressureRow = world.vascular.layout.cavities.z;
                const double activeChamberVolume = physical(acceptedVascular, unknownCount, 0u,
                    cavityRow, world.vascular.unknowns);
                const double controlChamberVolume = physical(acceptedVascular, unknownCount, 1u,
                    cavityRow, world.vascular.unknowns);
                const double activeInlet = physical(acceptedVascular, unknownCount, 0u,
                    inletRow, world.vascular.unknowns);
                const double activeAortic = physical(acceptedVascular, unknownCount, 0u,
                    aorticRow, world.vascular.unknowns);
                const double controlAortic = physical(acceptedVascular, unknownCount, 1u,
                    aorticRow, world.vascular.unknowns);
                const double activePressure = physical(acceptedVascular, unknownCount, 0u,
                    pressureRow, world.vascular.unknowns);
                const double activeTotal = physical(acceptedVascular, unknownCount, 0u, 0u,
                    world.vascular.unknowns) +
                    physical(acceptedVascular, unknownCount, 0u, 1u, world.vascular.unknowns) +
                    physical(acceptedVascular, unknownCount, 0u, 2u, world.vascular.unknowns);
                const double controlTotal = physical(acceptedVascular, unknownCount, 1u, 0u,
                    world.vascular.unknowns) +
                    physical(acceptedVascular, unknownCount, 1u, 1u, world.vascular.unknowns) +
                    physical(acceptedVascular, unknownCount, 1u, 2u, world.vascular.unknowns);
                const double totalError = std::max(std::abs(activeTotal - initialBloodVolume),
                    std::abs(controlTotal - initialBloodVolume)) / initialBloodVolume;
                maximumBloodVolumeError = std::max(maximumBloodVolumeError, totalError);
                maximumCavityVolumeError = std::max(maximumCavityVolumeError,
                    std::max(std::abs(activeWallVolume - activeChamberVolume),
                             std::abs(controlWallVolume - controlChamberVolume)) / 1.0e-6);

                double activeDisplacement = 0.0;
                double controlDisplacement = 0.0;
                double activeControlDifference = 0.0;
                for (std::size_t node = 0; node < nodeCount; ++node) {
                    const auto& initial = sourceNodes[node].positionAndMass;
                    const double ax = activeNodes[node][0] - initial.x;
                    const double ay = activeNodes[node][1] - initial.y;
                    const double az = activeNodes[node][2] - initial.z;
                    const double cx = controlNodes[node][0] - initial.x;
                    const double cy = controlNodes[node][1] - initial.y;
                    const double cz = controlNodes[node][2] - initial.z;
                    activeDisplacement = std::max(activeDisplacement,
                        std::sqrt(ax * ax + ay * ay + az * az));
                    controlDisplacement = std::max(controlDisplacement,
                        std::sqrt(cx * cx + cy * cy + cz * cz));
                    const double dx = activeNodes[node][0] - controlNodes[node][0];
                    const double dy = activeNodes[node][1] - controlNodes[node][1];
                    const double dz = activeNodes[node][2] - controlNodes[node][2];
                    activeControlDifference = std::max(activeControlDifference,
                        std::sqrt(dx * dx + dy * dy + dz * dz));
                }
                maximumActiveWallDisplacement = std::max(maximumActiveWallDisplacement,
                    activeDisplacement);
                maximumControlWallDisplacement = std::max(maximumControlWallDisplacement,
                    controlDisplacement);
                maximumActiveControlWallDifference = std::max(
                    maximumActiveControlWallDifference, activeControlDifference);
                maximumActiveAorticFlow = std::max(maximumActiveAorticFlow, activeAortic);
                maximumControlAorticFlow = std::max(maximumControlAorticFlow, controlAortic);

                const std::uint32_t beat = step / stepsPerBeat;
                cycleIn[beat] += activeInlet * timestep;
                cycleOut[beat] += activeAortic * timestep;
                const double controlInlet = physical(acceptedVascular, unknownCount, 1u,
                    inletRow, world.vascular.unknowns);
                cycleIn[beatCount + beat] += controlInlet * timestep;
                cycleOut[beatCount + beat] += controlAortic * timestep;
                trace << (double(step + 1u) * timestep) << ',' << active << ','
                      << activeWallVolume << ',' << controlWallVolume << ','
                      << activeChamberVolume << ',' << controlChamberVolume << ','
                      << activeInlet << ',' << activeAortic << ','
                      << controlAortic << ',' << activePressure << ','
                      << activeDisplacement << ',' << controlDisplacement << ','
                      << totalError << '\n';
            }
            trace.close();
            const double wallSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();

            std::array<double, beatCount> strokeDifference{};
            for (std::size_t beat = 0; beat < beatCount; ++beat) {
                const double activeStroke = cycleOut[beat];
                const double controlStroke = cycleOut[beatCount + beat];
                strokeDifference[beat] = activeStroke - controlStroke;
            }
            require(maximumBloodVolumeError < 1.0e-4,
                    "closed-loop blood volume conservation exceeded 1e-4");
            require(maximumCavityVolumeError < 5.0e-3,
                    "hydraulic chamber volume diverged from the deforming wall");
            require(maximumActiveTension > 1.0e4 &&
                    maximumActiveWallDisplacement > 1.0e-8 &&
                    maximumActiveControlWallDifference > 1.0e-8,
                    "active myocardium produced no resolved tissue response");
            require(strokeDifference[0] > 1.0e-10 && strokeDifference[1] > 1.0e-10,
                    "two active beats did not produce excess aortic ejection over the zero-tension control");

            std::printf(
                "{\"device\":\"%s\",\"abi\":%u,\"native_accepted_steps\":%u,"
                "\"beats\":%u,\"timestep_s\":%.9g,\"duration_s\":%.9g,"
                "\"cells\":%zu,\"nodes\":%u,\"directional_valves\":2,"
                "\"active_tension_peak_pa\":%.9g,\"active_wall_max_displacement_m\":%.12g,"
                "\"control_wall_max_displacement_m\":%.12g,\"active_control_wall_difference_m\":%.12g,"
                "\"active_aortic_peak_flow_m3_s\":%.12g,\"control_aortic_peak_flow_m3_s\":%.12g,"
                "\"beat1_active_ejection_m3\":%.12g,\"beat1_control_ejection_m3\":%.12g,"
                "\"beat2_active_ejection_m3\":%.12g,\"beat2_control_ejection_m3\":%.12g,"
                "\"beat1_excess_ejection_m3\":%.12g,\"beat2_excess_ejection_m3\":%.12g,"
                "\"max_blood_volume_error_fraction\":%.12g,"
                "\"max_cavity_volume_error_fraction\":%.12g,"
                "\"wall_seconds\":%.9g,\"total_fgmres_iterations\":%llu,"
                "\"fixture\":\"synthetic_64_node_hollow_tissue_shell\","
                "\"physiological_calibration\":false,\"anatomical_qualification\":false,"
                "\"heartbeat_mechanics_demonstrated\":true}\n",
                [[device name] UTF8String], NM_MATTER_ABI_VERSION, totalSteps, beatCount,
                timestep, timestep * totalSteps, world.fem.tetrahedra.size(),
                world.dispatch.femNodeCount, maximumActiveTension,
                maximumActiveWallDisplacement, maximumControlWallDisplacement,
                maximumActiveControlWallDifference, maximumActiveAorticFlow,
                maximumControlAorticFlow, cycleOut[0], cycleOut[beatCount],
                cycleOut[1], cycleOut[beatCount + 1u],
                strokeDifference[0], strokeDifference[1], maximumBloodVolumeError,
                maximumCavityVolumeError, wallSeconds,
                static_cast<unsigned long long>(totalNewtonLinearIterations));
            return 0;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native heartbeat coupling: %s\n", error.what());
        return 1;
    }
}
