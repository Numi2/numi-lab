#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/surgical_tissue.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kLongitudinalCells = 24u;
constexpr std::uint32_t kTransverseCells = 8u;
constexpr std::uint32_t kThroughThicknessCells = 2u;
constexpr std::uint32_t kExpectedTetrahedra =
    6u * kLongitudinalCells * kTransverseCells * kThroughThicknessCells;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct LipPair {
    std::uint32_t lower = 0u;
    std::uint32_t upper = 0u;
    double restX = 0.0;
    double restZ = 0.0;
};

std::vector<LipPair> topLipPairs(
    const numi::matter::SyntheticSkinWoundCoupon& coupon,
    const std::span<const std::uint32_t> stateIndexBySourceNode
) {
    const double topZ = 0.5 * coupon.spec.thicknessM;
    std::vector<LipPair> result;
    for (const auto& pair : coupon.metadata.incisionLipNodePairs) {
        require(pair[0] < stateIndexBySourceNode.size() &&
                    pair[1] < stateIndexBySourceNode.size() &&
                    stateIndexBySourceNode[pair[0]] != NM_INVALID_INDEX &&
                    stateIndexBySourceNode[pair[1]] != NM_INVALID_INDEX,
                "incision lip node has no compiled-state mapping");
        const auto& lower = coupon.object.femNodes.at(pair[0]);
        const auto& upper = coupon.object.femNodes.at(pair[1]);
        if (std::abs(lower[2] - topZ) > 1.0e-12 ||
            std::abs(upper[2] - topZ) > 1.0e-12) {
            continue;
        }
        result.push_back({
            stateIndexBySourceNode[pair[0]],
            stateIndexBySourceNode[pair[1]],
            lower[0],
            lower[2],
        });
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.restX < b.restX;
    });
    require(result.size() >= 8u,
            "top incision surface has too few paired lip nodes");
    for (std::size_t index = 1u; index < result.size(); ++index) {
        require(result[index - 1u].restX < result[index].restX,
                "top incision lip samples have duplicate longitudinal nodes");
    }
    return result;
}

std::vector<std::size_t> chooseBitePatch(
    const std::vector<LipPair>& pairs,
    const double biteX
) {
    constexpr double kPatchHalfWidthM = 0.0016;
    std::vector<std::size_t> selected;
    for (std::size_t index = 0u; index < pairs.size(); ++index) {
        if (std::abs(pairs[index].restX - biteX) <= kPatchHalfWidthM) {
            selected.push_back(index);
        }
    }
    require(selected.size() >= 3u,
            "authored suture bite patch has too few surface nodes");
    return selected;
}

double pairGap(
    const LipPair& pair,
    const std::vector<NMFEMNodeStateGPU>& nodes
) {
    const auto& lower = nodes.at(pair.lower).positionAndMass;
    const auto& upper = nodes.at(pair.upper).positionAndMass;
    const double dx = static_cast<double>(upper.x - lower.x);
    const double dy = static_cast<double>(upper.y - lower.y);
    const double dz = static_cast<double>(upper.z - lower.z);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double maximumDisplacement(
    const std::vector<NMFEMNodeStateGPU>& nodes,
    const std::vector<NMFEMNodeStateGPU>& rest
) {
    require(nodes.size() == rest.size(),
            "skin node state changed its cooked capacity");
    double maximum = 0.0;
    for (std::size_t index = 0u; index < nodes.size(); ++index) {
        const double dx = static_cast<double>(
            nodes[index].positionAndMass.x - rest[index].positionAndMass.x);
        const double dy = static_cast<double>(
            nodes[index].positionAndMass.y - rest[index].positionAndMass.y);
        const double dz = static_cast<double>(
            nodes[index].positionAndMass.z - rest[index].positionAndMass.z);
        maximum = std::max(maximum, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maximum;
}

void run(
    const std::filesystem::path& outputDirectory,
    const std::filesystem::path& materialPath,
    const std::filesystem::path& metallibPath,
    const double biteForcePerSiteN,
    const bool validateOnly
) {
    using namespace numi::matter;
    @autoreleasepool {
        constexpr double kPlannedLoadedForceN = 0.010;
        constexpr double kContactSlopM = 1.0e-5;
        constexpr double kZeroForceDriftLimitM = 1.0e-6;
        require(std::isfinite(biteForcePerSiteN) &&
                    (biteForcePerSiteN == 0.0 ||
                     biteForcePerSiteN == kPlannedLoadedForceN),
                "bite force must be the preregistered 0 N control or 0.010 N load");
        auto material = parseMatterFile(materialPath.string());
        require(material.succeeded(),
                "could not parse the synthetic-skin material");

        SyntheticSkinWoundSpec spec;
        spec.longitudinalCells = kLongitudinalCells;
        spec.transverseCells = kTransverseCells;
        spec.throughThicknessCells = kThroughThicknessCells;
        auto coupon = makeSyntheticSkinWoundCoupon(0u, spec);
        WorldSource source;
        source.frameTimestep = 0.001;
        source.gravity = {0.0, 0.0, 0.0};
        source.contactSlop = kContactSlopM;
        source.deterministic = true;
        source.materials.push_back(std::move(material.material));
        source.objects.push_back(coupon.object);
        auto compiled = compileWorld(source, {});
        require(compiled.succeeded(),
                "could not compile the synthetic-skin traction world");
        std::string layoutError;
        require(validateCompiledWorldLayout(compiled.world, &layoutError),
                "invalid synthetic-skin traction world: " + layoutError);
        require(compiled.world.objects.size() == 1u,
                "compiled traction world changed the object count");
        const NMContinuumObjectGPU& descriptor =
            compiled.world.objects.front();
        require(descriptor.stateCount ==
                        coupon.object.femNodes.size() + 64u &&
                    compiled.world.fem.tetrahedra.size() ==
                        coupon.object.tetrahedra.size() + 128u &&
                    static_cast<std::size_t>(descriptor.stateOffset) +
                            descriptor.stateCount <=
                        compiled.world.fem.nodes.size() &&
                    compiled.world.fem.topologyNodes.size() ==
                        compiled.world.fem.nodes.size(),
                "compiled traction world has invalid FEM node capacity");
        std::vector<std::uint32_t> stateIndexBySourceNode(
            coupon.object.femNodes.size(), NM_INVALID_INDEX
        );
        const std::size_t descriptorEnd =
            static_cast<std::size_t>(descriptor.stateOffset) +
            descriptor.stateCount;
        for (std::size_t stateIndex = descriptor.stateOffset;
             stateIndex < descriptorEnd;
             ++stateIndex) {
            const NMFEMTopologyNodeGPU& topology =
                compiled.world.fem.topologyNodes[stateIndex];
            if (topology.identity.y != 0u ||
                (topology.identity.w & NM_TOPOLOGY_ACTIVE) == 0u) {
                continue;
            }
            const std::uint32_t sourceNode = topology.identity.x;
            require(sourceNode < stateIndexBySourceNode.size() &&
                        stateIndexBySourceNode[sourceNode] == NM_INVALID_INDEX,
                    "compiled traction world duplicated an authored FEM node");
            stateIndexBySourceNode[sourceNode] =
                static_cast<std::uint32_t>(stateIndex);
        }
        for (std::size_t sourceNode = 0u;
             sourceNode < stateIndexBySourceNode.size();
             ++sourceNode) {
            const std::uint32_t stateIndex =
                stateIndexBySourceNode[sourceNode];
            require(stateIndex != NM_INVALID_INDEX,
                    "compiled traction world omitted an authored FEM node");
            const auto& rest =
                compiled.world.fem.nodes[stateIndex].restAndFixed;
            const auto& sourceNodePosition = coupon.object.femNodes[sourceNode];
            require(std::abs(static_cast<double>(rest.x) -
                             sourceNodePosition[0]) <= 1.0e-7 &&
                        std::abs(static_cast<double>(rest.y) -
                                 sourceNodePosition[1]) <= 1.0e-7 &&
                        std::abs(static_cast<double>(rest.z) -
                                 sourceNodePosition[2]) <= 1.0e-7,
                    "compiled FEM node mapping changed an authored rest position");
        }
        const std::vector<LipPair> lipPairs = topLipPairs(
            coupon, stateIndexBySourceNode
        );
        const std::vector<std::size_t> proximalPatch = chooseBitePatch(
            lipPairs, -0.004
        );
        const std::vector<std::size_t> distalPatch = chooseBitePatch(
            lipPairs, 0.004
        );

        if (validateOnly) {
            std::cout << "synthetic_skin_wound_traction_preflight=ok"
                      << " source_nodes=" << coupon.object.femNodes.size()
                      << " state_node_capacity="
                      << compiled.world.fem.nodes.size()
                      << " source_tetrahedra="
                      << coupon.object.tetrahedra.size()
                      << " state_tetrahedron_capacity="
                      << compiled.world.fem.tetrahedra.size()
                      << " top_lip_pairs=" << lipPairs.size()
                      << " mapped_lip_nodes=" << 2u * lipPairs.size()
                      << " expected_tetrahedra=" << kExpectedTetrahedra
                      << " proximal_bite_nodes=" << proximalPatch.size()
                      << " distal_bite_nodes=" << distalPatch.size()
                      << " force_ramp=cosine60"
                      << " newton_iterations="
                      << compiled.world.mixedSolver.nonlinearIterations.x
                      << " gpu_dispatch=false\n";
            return;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device is available");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(queue != nil, "could not create the skin traction queue");
        id<MTLBuffer> worldStatuses = [device
            newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
            options:MTLResourceStorageModeShared];
        const std::size_t forceValueCount =
            compiled.world.dispatch.femNodeCount * 4u;
        id<MTLBuffer> forceBuffer = [device
            newBufferWithLength:forceValueCount * sizeof(float)
            options:MTLResourceStorageModeShared];
        require(worldStatuses != nil && forceBuffer != nil,
                "could not allocate skin traction inputs");
        auto* worldStatus = static_cast<MRMetalWorldStatusGPU*>(
            worldStatuses.contents
        );
        auto* externalForces = static_cast<float*>(forceBuffer.contents);
        require(worldStatus != nullptr && externalForces != nullptr,
                "skin traction inputs are not CPU visible");

        Runtime runtime;
        const auto initialized = runtime.initialize(
            compiled.world,
            {
                .metallib = metallibPath.string(),
                .environmentCount = 1u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            }
        );
        require(initialized.encoded && runtime.valid(),
                initialized.message);

        std::filesystem::create_directories(outputDirectory);
        std::ofstream samples(outputDirectory / "traction-samples.csv");
        std::ofstream lips(outputDirectory / "lip-geometry.csv");
        require(samples.good() && lips.good(),
                "could not create the skin traction output files");
        samples << "step,time_s,bite_force_n,mean_gap_m,minimum_mean_gap_m,"
                   "center_gap_m,minimum_signed_gap_m,maximum_displacement_m,"
                   "minimum_J,maximum_residual,"
                   "active_tetrahedra,mass_kg,gpu_ms\n";
        lips << "step,time_s,pair,rest_x_m,rest_z_m,"
                "lower_x_m,lower_y_m,lower_z_m,"
                "upper_x_m,upper_y_m,upper_z_m\n";
        samples << std::setprecision(12);
        lips << std::setprecision(12);

        const std::vector<NMFEMNodeStateGPU>& restNodes =
            compiled.world.fem.nodes;
        double initialMass = 0.0;
        for (const auto& node : restNodes) {
            initialMass += node.positionAndMass.w;
        }
        auto* matterStatuses = static_cast<NMMatterStatusGPU*>(
            ((__bridge id<MTLBuffer>)runtime.statusBuffer()).contents
        );
        require(matterStatuses != nullptr,
                "skin traction Matter status is not CPU visible");

        constexpr std::uint32_t kSteps = 60u;
        constexpr std::uint32_t kRampSteps = kSteps;
        constexpr float kPi = 3.14159265358979323846f;
        double totalGpuMilliseconds = 0.0;
        double finalMeanGap = 0.0;
        double initialMeanGap = 0.0;
        double minimumMeanGap = std::numeric_limits<double>::infinity();
        double maximumMeanGapDrift = 0.0;
        double maximumDisplacementM = 0.0;
        double minimumJ = std::numeric_limits<double>::infinity();
        double minimumSignedGapM = std::numeric_limits<double>::infinity();
        double maximumResidual = 0.0;
        std::uint32_t finalActiveTetrahedra = 0u;
        double finalMass = 0.0;

        const auto setPatchForces = [&](
            const std::vector<std::size_t>& patch,
            const double biteX,
            const float forceMagnitude
        ) {
            std::vector<double> weights(patch.size(), 0.0);
            double weightSum = 0.0;
            for (std::size_t local = 0u; local < patch.size(); ++local) {
                const double distance = lipPairs[patch[local]].restX - biteX;
                weights[local] = std::exp(
                    -0.5 * distance * distance / (0.001 * 0.001)
                );
                weightSum += weights[local];
            }
            require(weightSum > 0.0,
                    "suture traction patch has no positive weight");
            for (std::size_t local = 0u; local < patch.size(); ++local) {
                const LipPair& pair = lipPairs[patch[local]];
                const float force = static_cast<float>(
                    forceMagnitude * weights[local] / weightSum
                );
                externalForces[pair.lower * 4u + 1u] += force;
                externalForces[pair.upper * 4u + 1u] -= force;
            }
        };

        for (std::uint32_t step = 0u; step < kSteps; ++step) {
            std::fill_n(externalForces, forceValueCount, 0.0f);
            float forceScale = 0.0f;
            if (step < kRampSteps) {
                const float rampProgress = static_cast<float>(step) /
                    static_cast<float>(kRampSteps - 1u);
                forceScale = 0.5f * (1.0f - std::cos(kPi * rampProgress));
            }
            const float biteForce =
                static_cast<float>(biteForcePerSiteN) * forceScale;
            setPatchForces(proximalPatch, -0.004, biteForce);
            setPatchForces(distalPatch, 0.004, biteForce);

            *worldStatus = {};
            worldStatus->code = MR_STEP_SUCCESS;
            worldStatus->environment = 0u;
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            require(commandBuffer != nil,
                    "could not allocate a skin traction command buffer");
            EncodeRequest request{};
            request.commandBuffer = (__bridge void*)commandBuffer;
            request.environmentStatuses = (__bridge void*)worldStatuses;
            request.femExternalForces = (__bridge void*)forceBuffer;
            request.femExternalForceCount =
                compiled.world.dispatch.femNodeCount;
            request.phase = EncodePhase::preDynamics;
            request.controlStep = step;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            auto encoded = runtime.encode(request);
            require(encoded.encoded,
                    "skin traction pre-dynamics: " + encoded.message);
            request.phase = EncodePhase::postCommit;
            encoded = runtime.encode(request);
            require(encoded.encoded,
                    "skin traction post-commit: " + encoded.message);
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            require(commandBuffer.status == MTLCommandBufferStatusCompleted,
                    "skin traction Metal command failed");
            const CFTimeInterval gpuStart = commandBuffer.GPUStartTime;
            const CFTimeInterval gpuEnd = commandBuffer.GPUEndTime;
            const double stepGpuMilliseconds =
                std::isfinite(gpuStart) && std::isfinite(gpuEnd) &&
                    gpuEnd >= gpuStart
                ? 1000.0 * (gpuEnd - gpuStart)
                : 0.0;
            totalGpuMilliseconds += stepGpuMilliseconds;
            if (matterStatuses[0].code != NM_STATUS_SUCCESS) {
                const NMMatterStatusGPU status = matterStatuses[0];
                throw std::runtime_error(
                    "skin traction Matter status failed at step " +
                    std::to_string(step) + " with code " +
                    std::to_string(status.code) + " object=" +
                    std::to_string(status.objectIndex) + " failing_index=" +
                    std::to_string(status.failingIndex) +
                    " completed_microsteps=" +
                    std::to_string(status.completedMicrosteps) +
                    " fgmres_iterations=" +
                    std::to_string(status.fgmresIterations) +
                    " contact_count=" + std::to_string(status.contactCount) +
                    " diagnostics=" +
                    std::to_string(status.diagnostics.x) + "," +
                    std::to_string(status.diagnostics.y) + "," +
                    std::to_string(status.diagnostics.z) + "," +
                    std::to_string(status.diagnostics.w)
                );
            }

            const RuntimeStateSnapshot snapshot = runtime.snapshot();
            require(snapshot.available,
                    "skin traction snapshot: " + snapshot.message);
            require(snapshot.femNodes.size() == restNodes.size(),
                    "skin traction snapshot changed the FEM node count");
            require(!snapshot.solverCertificates.empty(),
                    "skin traction omitted its solver certificate");
            const NMSolverCertificateGPU& certificate =
                snapshot.solverCertificates.front();
            require(certificate.validity.w > 0.5f &&
                        std::isfinite(certificate.validity.x) &&
                        certificate.validity.x > 0.20f,
                    "skin traction violated the determinant certificate");
            minimumJ = std::min(
                minimumJ,
                static_cast<double>(certificate.validity.x)
            );
            maximumResidual = std::max(
                maximumResidual,
                static_cast<double>(certificate.nonlinear.x)
            );
            maximumDisplacementM = std::max(
                maximumDisplacementM,
                maximumDisplacement(snapshot.femNodes, restNodes)
            );
            finalMass = 0.0;
            for (const auto& node : snapshot.femNodes) {
                finalMass += node.positionAndMass.w;
            }
            finalActiveTetrahedra = 0u;
            for (const auto& tetrahedron : snapshot.femTopologyTetrahedra) {
                finalActiveTetrahedra +=
                    (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
            }

            double meanGap = 0.0;
            double centerGap = 0.0;
            double stepMinimumSignedGapM =
                std::numeric_limits<double>::infinity();
            std::size_t centerIndex = 0u;
            double centerDistance = std::numeric_limits<double>::infinity();
            for (std::size_t pairIndex = 0u;
                 pairIndex < lipPairs.size();
                 ++pairIndex) {
                const LipPair& pair = lipPairs[pairIndex];
                meanGap += pairGap(pair, snapshot.femNodes);
                if (std::abs(pair.restX) < centerDistance) {
                    centerDistance = std::abs(pair.restX);
                    centerIndex = pairIndex;
                }
                const auto& lower = snapshot.femNodes[pair.lower]
                                        .positionAndMass;
                const auto& upper = snapshot.femNodes[pair.upper]
                                        .positionAndMass;
                const double signedGapM =
                    static_cast<double>(upper.y - lower.y);
                require(signedGapM >= 0.0,
                        "synthetic wound lips crossed at step " +
                            std::to_string(step));
                stepMinimumSignedGapM = std::min(
                    stepMinimumSignedGapM,
                    signedGapM
                );
                lips << step << ',' << (0.001 * step) << ',' << pairIndex
                     << ',' << pair.restX << ',' << pair.restZ << ','
                     << lower.x << ',' << lower.y << ',' << lower.z << ','
                     << upper.x << ',' << upper.y << ',' << upper.z << '\n';
            }
            meanGap /= static_cast<double>(lipPairs.size());
            centerGap = pairGap(lipPairs[centerIndex], snapshot.femNodes);
            if (step == 0u) {
                initialMeanGap = meanGap;
            }
            minimumMeanGap = std::min(minimumMeanGap, meanGap);
            maximumMeanGapDrift = std::max(
                maximumMeanGapDrift,
                std::abs(meanGap - initialMeanGap)
            );
            minimumSignedGapM = std::min(
                minimumSignedGapM,
                stepMinimumSignedGapM
            );
            finalMeanGap = meanGap;
            samples << step << ',' << (0.001 * (step + 1u)) << ','
                    << biteForce << ',' << meanGap << ',' << minimumMeanGap
                    << ',' << centerGap << ',' << stepMinimumSignedGapM << ','
                    << maximumDisplacementM << ',' << minimumJ << ','
                    << maximumResidual << ',' << finalActiveTetrahedra << ','
                    << finalMass << ',' << stepGpuMilliseconds << '\n';
        }

        const double relativeMassError = initialMass > 0.0
            ? std::abs(finalMass - initialMass) / initialMass
            : 1.0;
        require(finalActiveTetrahedra == coupon.object.tetrahedra.size(),
                "skin traction removed or deactivated tetrahedra");
        require(relativeMassError <= 1.0e-5,
                "skin traction did not conserve FEM mass");
        if (biteForcePerSiteN == 0.0) {
            require(maximumMeanGapDrift <= kZeroForceDriftLimitM,
                    "zero-force control drift exceeded 1 um");
        }
        std::cout << std::setprecision(9)
                  << "synthetic_skin_wound_traction=ok"
                  << " arm=" << (biteForcePerSiteN == 0.0
                          ? "zero_force_control"
                          : "paired_bite_traction")
                  << " mesh=" << kLongitudinalCells << 'x'
                  << kTransverseCells << 'x' << kThroughThicknessCells
                  << " top_lip_pairs=" << lipPairs.size()
                  << " bite_force_per_site_n=" << biteForcePerSiteN
                  << " force_ramp=cosine60"
                  << " newton_iterations="
                  << compiled.world.mixedSolver.nonlinearIterations.x
                  << " initial_mean_gap_m=" << initialMeanGap
                  << " minimum_mean_gap_m=" << minimumMeanGap
                  << " final_mean_gap_m=" << finalMeanGap
                  << " maximum_mean_gap_drift_m=" << maximumMeanGapDrift
                  << " minimum_signed_gap_m=" << minimumSignedGapM
                  << " peak_gap_reduction_fraction="
                  << (initialMeanGap > 0.0
                          ? (initialMeanGap - minimumMeanGap) / initialMeanGap
                          : 0.0)
                  << " maximum_tissue_displacement_m="
                  << maximumDisplacementM
                  << " minimum_J=" << minimumJ
                  << " maximum_solver_residual=" << maximumResidual
                  << " active_tetrahedra=" << finalActiveTetrahedra
                  << " mass_relative_error=" << relativeMassError
                  << " total_gpu_ms=" << totalGpuMilliseconds
                  << " physical_calibration=none"
                  << " thread_present=false"
                  << " robot_execution=false\n";
    }
}

} // namespace

int main(const int argc, char* argv[]) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--help") {
            std::cout
                << "usage: metalrobo_synthetic_skin_wound_traction_probe "
                   "OUTPUT_DIRECTORY MATERIAL_PATH MATTER_METALLIB "
                   "[--force-per-site-n 0|0.010 | --validate-only]\n"
                << "Runs a synthetic wound-lip traction response probe; "
                   "0 N is the matched control; 10 mN is the loaded arm. "
                   "--validate-only checks compiled node mapping without "
                   "Metal dispatch. No thread or robot is modeled.\n";
            return 0;
        }
        require(argc == 4 || argc == 5 || argc == 6,
                "usage: metalrobo_synthetic_skin_wound_traction_probe "
                "OUTPUT_DIRECTORY MATERIAL_PATH MATTER_METALLIB "
                "[--force-per-site-n 0|0.010 | --validate-only]");
        double biteForcePerSiteN = 0.010;
        bool validateOnly = false;
        if (argc == 5) {
            require(std::string_view{argv[4]} == "--validate-only",
                    "expected --validate-only");
            validateOnly = true;
        }
        if (argc == 6) {
            require(std::string_view{argv[4]} == "--force-per-site-n",
                    "expected --force-per-site-n");
            const std::string forceText{argv[5]};
            std::size_t parsedCharacters = 0u;
            biteForcePerSiteN = std::stod(forceText, &parsedCharacters);
            require(parsedCharacters == forceText.size(),
                    "force value contains trailing characters");
        }
        run(
            std::filesystem::path{argv[1]},
            std::filesystem::path{argv[2]},
            std::filesystem::path{argv[3]},
            biteForcePerSiteN,
            validateOnly
        );
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "synthetic_skin_wound_traction=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
