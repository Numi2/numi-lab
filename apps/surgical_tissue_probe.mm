#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/surgical_tissue.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_JEJUNUM_MATERIAL
#define NUMI_JEJUNUM_MATERIAL ""
#endif

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {

constexpr double kTissueTableContactSlopM = 1.0e-5;
constexpr double kInitialTissueTableGapM = 1.1e-5;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string compileErrors(
    const std::vector<numi::matter::Diagnostic>& diagnostics
) {
    std::string result;
    for (const auto& diagnostic : diagnostics) {
        if (!result.empty()) {
            result += "; ";
        }
        result +=
            std::to_string(diagnostic.line) + ":" +
            std::to_string(diagnostic.column) + " " +
            diagnostic.message;
    }
    return result;
}

numi::matter::CompiledWorld compileTissueWorld(
    numi::matter::PorcineJejunumFungSpec& runtimeSpec,
    numi::matter::PorcineJejunumClosureCoupon& coupon
) {
    auto parsed = numi::matter::parseMatterFile(
        NUMI_JEJUNUM_MATERIAL
    );
    require(
        parsed.succeeded(),
        "porcine jejunum material parse failed: " +
            compileErrors(parsed.diagnostics)
    );
    std::string materialError;
    require(
        numi::matter::configurePorcineJejunumFungMaterial(
            parsed.material,
            runtimeSpec,
            &materialError
        ),
        materialError
    );

    coupon = numi::matter::makePorcineJejunumClosureCoupon(
        0u,
        runtimeSpec
    );
    for (auto& position : coupon.object.femNodes) {
        position[2] +=
            0.5 * runtimeSpec.thicknessM.value +
            kInitialTissueTableGapM;
    }

    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 1.0 / 2000.0;
    source.gravity = {0.0, 0.0, -9.81};
    source.contactSlop = kTissueTableContactSlopM;
    source.deterministic = true;
    source.mixedSolver.newtonIterations = 12u;
    source.mixedSolver.fgmresRestart = 12u;
    source.mixedSolver.fgmresIterations = 48u;
    source.mixedSolver.lineSearchSteps = 12u;
    source.mixedSolver.relativeResidual = 5.0e-4;
    source.mixedSolver.volumeTolerance = 5.0e-4;
    source.mixedSolver.pressureTolerance = 5.0e-4;
    source.materials.push_back(std::move(parsed.material));
    numi::matter::RigidProxySource table;
    table.shape = NM_RIGID_PLANE;
    table.localCenter = {0.0, 0.0, 1.0};
    table.radiusOrOffset = 0.0;
    source.rigidProxies.push_back(table);
    source.objects.push_back(coupon.object);

    numi::matter::CompileOptions options;
    options.maximumRateExponent = 6u;
    auto compiled = numi::matter::compileWorld(source, options);
    require(
        compiled.succeeded(),
        "porcine jejunum world compile failed: " +
            compileErrors(compiled.diagnostics)
    );
    std::string layoutError;
    require(
        numi::matter::validateCompiledWorldLayout(
            compiled.world,
            &layoutError
        ),
        "porcine jejunum cooked layout is invalid: " + layoutError
    );
    return std::move(compiled.world);
}

struct TissueRun {
    numi::matter::RuntimeStateSnapshot snapshot;
    std::string deviceName;
    double gpuMilliseconds = 0.0;
    std::uint64_t threadDispatches = 0u;
    std::uint64_t simdgroupDispatches = 0u;
    std::uint64_t indirectDispatches = 0u;
};

TissueRun runTissue(
    const numi::matter::CompiledWorld& world,
    const std::uint32_t controlSteps
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Apple Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "failed to create tissue command queue");
        id<MTLBuffer> worldStatuses = [device
            newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        require(
            worldStatuses != nil,
            "failed to allocate tissue transaction status"
        );

        numi::matter::Runtime runtime;
        const auto initialized = runtime.initialize(
            world,
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
            initialized.encoded && runtime.valid(),
            "tissue runtime initialization failed: " +
                initialized.message
        );

        TissueRun result;
        result.deviceName = device.name.UTF8String;
        for (std::uint32_t step = 0u; step < controlSteps; ++step) {
            auto* status = static_cast<MRMetalWorldStatusGPU*>(
                worldStatuses.contents
            );
            *status = {};
            status->code = MR_STEP_SUCCESS;
            status->environment = 0u;

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(
                commandBuffer != nil,
                "failed to create tissue command buffer"
            );
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.environmentStatuses = (__bridge void*)worldStatuses;
            request.phase = numi::matter::EncodePhase::preDynamics;
            request.controlStep = step;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.runAdaptiveTransfer = false;
            auto encoded = runtime.encode(request);
            require(
                encoded.encoded,
                "tissue pre-dynamics encode failed: " + encoded.message
            );
            result.threadDispatches += encoded.threadDispatchCount;
            result.simdgroupDispatches +=
                encoded.simdgroupDispatchCount;
            result.indirectDispatches +=
                encoded.indirectDispatchCount;

            request.phase = numi::matter::EncodePhase::postCommit;
            encoded = runtime.encode(request);
            require(
                encoded.encoded,
                "tissue post-commit encode failed: " + encoded.message
            );
            result.threadDispatches += encoded.threadDispatchCount;
            result.simdgroupDispatches +=
                encoded.simdgroupDispatchCount;
            result.indirectDispatches +=
                encoded.indirectDispatchCount;

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(
                commandBuffer.status == MTLCommandBufferStatusCompleted,
                "tissue Metal command buffer failed: " +
                    std::string(
                        commandBuffer.error == nil
                            ? "unknown"
                            : commandBuffer.error.localizedDescription.UTF8String
                    )
            );
            result.gpuMilliseconds += 1000.0 * std::max(
                0.0,
                commandBuffer.GPUEndTime - commandBuffer.GPUStartTime
            );
        }
        result.snapshot = runtime.snapshot();
        require(
            result.snapshot.available,
            "tissue snapshot failed: " + result.snapshot.message
        );
        return result;
    }
}

double meanFreeHeight(
    const std::vector<NMFEMNodeStateGPU>& nodes,
    const std::vector<std::uint32_t>& fixedNodes,
    const std::size_t activeNodeCount
) {
    require(
        activeNodeCount <= nodes.size(),
        "active tissue node count exceeds runtime capacity"
    );
    std::vector<bool> fixed(activeNodeCount, false);
    for (const std::uint32_t node : fixedNodes) {
        require(node < fixed.size(), "fixed tissue node is out of range");
        fixed[node] = true;
    }
    double sum = 0.0;
    std::size_t count = 0u;
    for (std::size_t node = 0u; node < activeNodeCount; ++node) {
        if (!fixed[node]) {
            sum += nodes[node].positionAndMass.z;
            ++count;
        }
    }
    require(count != 0u, "tissue specimen has no free nodes");
    return sum / static_cast<double>(count);
}

double minimumLipGap(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    const numi::matter::PorcineJejunumClosureMetadata& metadata
) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto pair : metadata.incisionLipNodePairs) {
        require(
            pair[0] < snapshot.femNodes.size() &&
                pair[1] < snapshot.femNodes.size(),
            "incision lip pair is out of range"
        );
        const auto& lower = snapshot.femNodes[pair[0]].positionAndMass;
        const auto& upper = snapshot.femNodes[pair[1]].positionAndMass;
        const double dx = upper.x - lower.x;
        const double dy = upper.y - lower.y;
        const double dz = upper.z - lower.z;
        minimum = std::min(
            minimum,
            std::sqrt(dx * dx + dy * dy + dz * dz)
        );
    }
    return minimum;
}

template <typename Value>
bool bitIdentical(
    const std::vector<Value>& first,
    const std::vector<Value>& second
) {
    return first.size() == second.size() &&
        (first.empty() ||
         std::memcmp(
             first.data(),
             second.data(),
             first.size() * sizeof(Value)
         ) == 0);
}

struct TissueContactMetrics {
    std::uint32_t activeCount = 0u;
    double minimumSeparationM =
        std::numeric_limits<double>::infinity();
    double maximumBarrierImpulse = 0.0;
};

TissueContactMetrics tissueTableContacts(
    const numi::matter::RuntimeStateSnapshot& snapshot
) {
    TissueContactMetrics result;
    for (const NMContactSampleGPU& contact : snapshot.contactSamples) {
        if ((contact.identity.w & NM_CONTACT_VALID) == 0u) {
            continue;
        }
        ++result.activeCount;
        result.minimumSeparationM = std::min(
            result.minimumSeparationM,
            static_cast<double>(contact.pointAndSeparation.w)
        );
        result.maximumBarrierImpulse = std::max(
            result.maximumBarrierImpulse,
            static_cast<double>(contact.barrier.x)
        );
    }
    return result;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        require(
            argc == 1 ||
                (argc == 2 &&
                 std::string{argv[1]} == "--production-resolution"),
            "usage: metalrobo_surgical_tissue_probe "
            "[--production-resolution]"
        );
        const bool productionResolution = argc == 2;

        // Build the full reusable closure asset first. The default runtime
        // replay uses a bounded mesh with identical material and wall
        // thickness; --production-resolution executes the complete topology.
        const auto production =
            numi::matter::makePorcineJejunumClosureCoupon(0u);
        require(
            production.metadata.nodeCount >= 900u &&
                production.metadata.tetrahedronCount == 3456u &&
                !production.metadata.incisionLipNodePairs.empty() &&
                !production.object.femContactNodes.empty() &&
                production.object.femContactNodes.size() <
                    production.metadata.nodeCount &&
                production.object.mutationPolicy.enabled &&
                production.object.femCapacity.punctureChannels >= 32u,
            "high-resolution jejunal closure asset lost required topology"
        );
        std::vector<bool> productionContactNodes(
            production.metadata.nodeCount,
            false
        );
        for (const std::uint32_t node :
             production.object.femContactNodes) {
            require(
                node < productionContactNodes.size() &&
                    !productionContactNodes[node],
                "jejunal collision surface contains an invalid node"
            );
            productionContactNodes[node] = true;
        }
        for (const auto& pair :
             production.metadata.incisionLipNodePairs) {
            require(
                productionContactNodes[pair[0]] &&
                    productionContactNodes[pair[1]],
                "incision lip is missing from the tissue collision surface"
            );
        }

        const auto response =
            numi::matter::evaluatePorcineJejunumFungResponse(
                production.spec,
                1.05,
                1.05
            );
        require(
            response.energyDensityPa > 0.0 &&
                response.longitudinalSecondPiolaPa > 0.0 &&
                response.longitudinalSecondPiolaPa >
                    response.circumferentialSecondPiolaPa,
            "published biaxial Fung calibration oracle is inconsistent"
        );

        numi::matter::PorcineJejunumFungSpec runtimeSpec;
        if (!productionResolution) {
            runtimeSpec.longitudinalCells = 6u;
            runtimeSpec.circumferentialCells = 6u;
            runtimeSpec.throughThicknessCells = 1u;
        }
        runtimeSpec.fixLongitudinalEnds = true;
        numi::matter::PorcineJejunumClosureCoupon runtimeCoupon;
        const auto world = compileTissueWorld(
            runtimeSpec,
            runtimeCoupon
        );
        require(
            world.dispatch.femNodeCount >=
                    runtimeCoupon.metadata.nodeCount &&
                world.dispatch.tetrahedronCount >=
                    runtimeCoupon.metadata.tetrahedronCount &&
                world.dispatch.maximumRateExponent <= 6u &&
                world.constitutive[0].material.hint ==
                    numi::matter::ConstitutiveHint::generic,
            "live tissue world did not retain the generic Fung FEM contract"
                ": nodes=" + std::to_string(world.dispatch.femNodeCount) +
                "/" + std::to_string(runtimeCoupon.metadata.nodeCount) +
                " tets=" +
                std::to_string(world.dispatch.tetrahedronCount) + "/" +
                std::to_string(runtimeCoupon.metadata.tetrahedronCount) +
                " rate=" +
                std::to_string(world.dispatch.maximumRateExponent) +
                " hint=" + std::to_string(static_cast<std::uint32_t>(
                    world.constitutive[0].material.hint
                ))
        );

        numi::matter::RuntimeStateSnapshot initial;
        initial.available = true;
        initial.femNodes = world.fem.nodes;
        const double initialHeight = meanFreeHeight(
            initial.femNodes,
            runtimeCoupon.object.femFixedNodes,
            runtimeCoupon.metadata.nodeCount
        );
        const double initialGap = minimumLipGap(
            initial,
            runtimeCoupon.metadata
        );

        const std::uint32_t controlSteps =
            productionResolution ? 2u : 6u;
        const TissueRun first = runTissue(world, controlSteps);
        const TissueRun replay = runTissue(world, controlSteps);
        require(
            first.snapshot.femNodes.size() == initial.femNodes.size() &&
                first.snapshot.femNodes.size() ==
                    replay.snapshot.femNodes.size() &&
                !first.deviceName.empty() &&
                first.deviceName == replay.deviceName,
            "tissue runtime changed FEM state capacity"
        );
        require(
            bitIdentical(
                first.snapshot.femNodes,
                replay.snapshot.femNodes
            ) &&
                bitIdentical(
                    first.snapshot.solverCertificates,
                    replay.snapshot.solverCertificates
                ) &&
                bitIdentical(
                    first.snapshot.contactSamples,
                    replay.snapshot.contactSamples
                ) &&
                bitIdentical(
                    first.snapshot.contactHistories,
                    replay.snapshot.contactHistories
                ) &&
                bitIdentical(
                    first.snapshot.deformableContactHistories,
                    replay.snapshot.deformableContactHistories
                ),
            "tissue Metal replay is not bit-identical"
        );

        double maximumFixedDrift = 0.0;
        for (const std::uint32_t node :
             runtimeCoupon.object.femFixedNodes) {
            const auto& before = initial.femNodes[node].positionAndMass;
            const auto& after =
                first.snapshot.femNodes[node].positionAndMass;
            const double dx = after.x - before.x;
            const double dy = after.y - before.y;
            const double dz = after.z - before.z;
            maximumFixedDrift = std::max(
                maximumFixedDrift,
                std::sqrt(dx * dx + dy * dy + dz * dz)
            );
        }
        require(
            maximumFixedDrift <= 1.0e-9,
            "tissue fixed boundary moved by " +
                std::to_string(maximumFixedDrift) + " m"
        );
        const double finalHeight = meanFreeHeight(
            first.snapshot.femNodes,
            runtimeCoupon.object.femFixedNodes,
            runtimeCoupon.metadata.nodeCount
        );
        const double finalGap = minimumLipGap(
            first.snapshot,
            runtimeCoupon.metadata
        );
        const TissueContactMetrics tableContacts =
            tissueTableContacts(first.snapshot);
        require(
            finalHeight < initialHeight - 1.0e-7 &&
                finalHeight > initialHeight - 0.003 &&
                finalGap > 0.5 * initialGap &&
                std::isfinite(finalGap) &&
                tableContacts.activeCount != 0u &&
                tableContacts.minimumSeparationM > 0.0 &&
                std::isfinite(tableContacts.minimumSeparationM) &&
                tableContacts.maximumBarrierImpulse > 0.0 &&
                std::isfinite(tableContacts.maximumBarrierImpulse),
            "jejunal wall did not produce bounded sag and positive table "
            "contact with an open incision"
        );

        float maximumResidual = 0.0f;
        float maximumRelativeCorrection = 0.0f;
        float minimumDeterminant =
            std::numeric_limits<float>::infinity();
        for (const auto& certificate :
             first.snapshot.solverCertificates) {
            require(
                certificate.validity.w > 0.5f &&
                    std::isfinite(certificate.nonlinear.x) &&
                    std::isfinite(certificate.nonlinear.y) &&
                    std::isfinite(certificate.validity.x),
                "tissue solver rejected or published a nonfinite certificate"
            );
            maximumResidual = std::max(
                maximumResidual,
                certificate.nonlinear.x
            );
            maximumRelativeCorrection = std::max(
                maximumRelativeCorrection,
                certificate.nonlinear.y
            );
            minimumDeterminant = std::min(
                minimumDeterminant,
                certificate.validity.x
            );
        }

        std::cout << std::setprecision(9)
            << "{\"schema\":\"numi.surgical-tissue-physics.v1\""
            << ",\"device\":\"" << first.deviceName << "\""
            << ",\"material\":\"porcine_jejunum_fung\""
            << ",\"runtime_resolution\":\""
            << (productionResolution ? "production" : "bounded-probe")
            << "\""
            << ",\"material_source\":\""
            << numi::matter::jejunalValueSourceReference(
                   numi::matter::JejunalValueBasis::
                       belliniPorcineBiaxialStudy
               ) << "\""
            << ",\"wall_thickness_mm\":"
            << 1000.0 * production.spec.thicknessM.value
            << ",\"incision_gap_mm\":"
            << 1000.0 * production.spec.incisionGapM.value
            << ",\"production_nodes\":"
            << production.metadata.nodeCount
            << ",\"production_tetrahedra\":"
            << production.metadata.tetrahedronCount
            << ",\"runtime_nodes\":"
            << runtimeCoupon.metadata.nodeCount
            << ",\"runtime_node_capacity\":"
            << world.dispatch.femNodeCount
            << ",\"runtime_tetrahedra\":"
            << runtimeCoupon.metadata.tetrahedronCount
            << ",\"runtime_tetrahedron_capacity\":"
            << world.dispatch.tetrahedronCount
            << ",\"rate_exponent\":"
            << world.dispatch.maximumRateExponent
            << ",\"fgmres_restart\":"
            << world.mixedSolver.nonlinearIterations.y
            << ",\"fgmres_iteration_budget\":"
            << world.mixedSolver.nonlinearIterations.z
            << ",\"field_smoother_passes\":"
            << world.mixedSolver.executionBudgets.x
            << ",\"control_steps\":" << controlSteps
            << ",\"free_sag_m\":" << initialHeight - finalHeight
            << ",\"minimum_incision_gap_m\":" << finalGap
            << ",\"active_table_contacts\":"
            << tableContacts.activeCount
            << ",\"minimum_table_separation_m\":"
            << tableContacts.minimumSeparationM
            << ",\"maximum_table_barrier_impulse\":"
            << tableContacts.maximumBarrierImpulse
            << ",\"minimum_J\":" << minimumDeterminant
            << ",\"maximum_fixed_drift_m\":" << maximumFixedDrift
            << ",\"maximum_nonlinear_residual\":" << maximumResidual
            << ",\"maximum_relative_correction_telemetry\":"
            << maximumRelativeCorrection
            << ",\"minimum_contact_separation_ratio\":"
            << world.mixedSolver.contactAcceptance.x
            << ",\"fung_energy_5pct_biaxial_pa\":"
            << response.energyDensityPa
            << ",\"gpu_ms\":" << first.gpuMilliseconds
            << ",\"thread_dispatches\":" << first.threadDispatches
            << ",\"simdgroup_dispatches\":"
            << first.simdgroupDispatches
            << ",\"indirect_dispatches\":"
            << first.indirectDispatches
            << ",\"replay_bit_identical\":true"
            << ",\"failed_steps\":0}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_tissue status=failed error=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
