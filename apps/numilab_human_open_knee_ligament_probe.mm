#include "metalrobo/NumiHumanKnee.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <vector>

namespace {

using Vec3 = std::array<double, 3u>;

struct LigamentSpec {
    std::string_view name;
    double c1MPa;
    double bulkMPa;
    double initialStretch;
    double maximumLinearStretch;
};

constexpr std::array<LigamentSpec, 4u> ligamentSpecs{{
    {"PCL", 3.25, 243.90, 1.000, 1.035},
    {"ACL", 1.95, 146.41, 1.016, 1.046},
    {"MCL", 1.44, 793.65, 1.034, 1.063},
    {"LCL", 1.44, 793.65, 1.027, 1.063},
}};

constexpr LigamentSpec patellarTendonSpec{
    "PTL", 2.75, 206.61, 1.000, 1.042,
};

constexpr LigamentSpec quadricepsTendonSpec{
    "QAT", 2.75, 206.61, 1.000, 1.042,
};

struct LigamentRuntime {
    const LigamentSpec* specification = nullptr;
    std::uint32_t payloadRegion = 0u;
    std::uint32_t firstFEMNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t femurAnchorCount = 0u;
    std::uint32_t tibiaAnchorCount = 0u;
    std::uint32_t patellaAnchorCount = 0u;
};

struct LigamentSnapshotHeader {
    std::array<char, 8u> magic{};
    std::uint32_t abi = 0u;
    std::uint32_t side = 0u;
    std::uint32_t regionCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t poseKind = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct LigamentSnapshotRegion {
    std::array<char, 16u> name{};
    std::uint32_t payloadFirstNode = 0u;
    std::uint32_t snapshotFirstNode = 0u;
    std::uint32_t nodeCount = 0u;
};

struct LigamentSnapshotNode {
    float position[3]{};
};

static_assert(sizeof(LigamentSnapshotHeader) == 40u);
static_assert(sizeof(LigamentSnapshotRegion) == 28u);
static_assert(sizeof(LigamentSnapshotNode) == 12u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> readBytes(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Open Knee ligament payload did not open");
    const auto end = stream.tellg();
    if (end <= 0) throw std::runtime_error("Open Knee ligament payload is empty");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("Open Knee ligament payload read failed");
    return bytes;
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
}

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0u] + b[0u], a[1u] + b[1u], a[2u] + b[2u]};
}

Vec3 scale(const Vec3& value, const double factor) {
    return {factor * value[0u], factor * value[1u], factor * value[2u]};
}

double norm(const Vec3& value) {
    return std::sqrt(value[0u] * value[0u] + value[1u] * value[1u] +
                     value[2u] * value[2u]);
}

double signedVolume(
    const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d
) {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ad = subtract(d, a);
    return (
        ab[0u] * (ac[1u] * ad[2u] - ac[2u] * ad[1u]) -
        ab[1u] * (ac[0u] * ad[2u] - ac[2u] * ad[0u]) +
        ab[2u] * (ac[0u] * ad[1u] - ac[1u] * ad[0u])
    ) / 6.0;
}

Vec3 position(const NMFEMNodeStateGPU& node) {
    return {node.positionAndMass.x, node.positionAndMass.y,
            node.positionAndMass.z};
}

const LigamentSpec& specification(const std::string_view name) {
    if (name == patellarTendonSpec.name) return patellarTendonSpec;
    if (name == quadricepsTendonSpec.name) return quadricepsTendonSpec;
    const auto found = std::find_if(
        ligamentSpecs.begin(), ligamentSpecs.end(),
        [name](const LigamentSpec& value) { return value.name == name; });
    if (found == ligamentSpecs.end())
        throw std::runtime_error("Open Knee ligament source material is missing");
    return *found;
}

void setParameter(
    numi::matter::MaterialProgram& material,
    const std::string_view name,
    const double value
) {
    const auto found = std::find_if(
        material.parameters.begin(), material.parameters.end(),
        [name](const numi::matter::Parameter& parameter) {
            return parameter.name == name;
        });
    if (found == material.parameters.end() || !std::isfinite(value) ||
        value < found->lower || value > found->upper)
        throw std::runtime_error("Open Knee ligament material parameter is invalid");
    found->defaultValue = value;
}

std::uint32_t femurBody(const metalrobo::NumiHumanKneeSide side) {
    return side == metalrobo::NumiHumanKneeSide::left
        ? metalrobo::NUMI_HUMAN_KNEE_FEMUR_BODY
        : metalrobo::NUMI_HUMAN_KNEE_RIGHT_FEMUR_BODY;
}

std::uint32_t tibiaBody(const metalrobo::NumiHumanKneeSide side) {
    return side == metalrobo::NumiHumanKneeSide::left
        ? metalrobo::NUMI_HUMAN_KNEE_TIBIA_BODY
        : metalrobo::NUMI_HUMAN_KNEE_RIGHT_TIBIA_BODY;
}

std::uint32_t patellaBody(const metalrobo::NumiHumanKneeSide side) {
    return side == metalrobo::NumiHumanKneeSide::left
        ? metalrobo::NUMI_HUMAN_KNEE_PATELLA_BODY
        : metalrobo::NUMI_HUMAN_KNEE_RIGHT_PATELLA_BODY;
}

std::uint64_t peakResidentBytes() {
    rusage usage{};
    return getrusage(RUSAGE_SELF, &usage) == 0
        ? static_cast<std::uint64_t>(usage.ru_maxrss)
        : 0u;
}

void writeAcceptedSnapshot(
    const char* path,
    const metalrobo::NumiHumanKneePayload& payload,
    const std::span<const LigamentRuntime> ligaments,
    const numi::matter::RuntimeStateSnapshot& accepted,
    const std::uint32_t totalTetrahedra
) {
    LigamentSnapshotHeader header;
    header.magic = {'N', 'H', 'K', 'F', 'E', 'M', '2', '\0'};
    header.abi = 2u;
    header.side = payload.side == metalrobo::NumiHumanKneeSide::left ? 0u : 1u;
    header.regionCount = static_cast<std::uint32_t>(ligaments.size());
    header.nodeCount = static_cast<std::uint32_t>(accepted.femNodes.size());
    header.tetrahedronCount = totalTetrahedra;
    // This state is the accepted sub-micron attachment/reaction preflight,
    // not an arbitrary articulated pose or loaded flexion state.
    header.poseKind = 1u;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.good(), "Open Knee ligament snapshot did not open");
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (const LigamentRuntime& ligament : ligaments) {
        const auto& region = payload.regions[ligament.payloadRegion];
        LigamentSnapshotRegion disk;
        std::copy_n(region.name.data(),
                    std::min(region.name.size(), disk.name.size() - 1u),
                    disk.name.data());
        disk.payloadFirstNode = region.firstNode;
        disk.snapshotFirstNode = ligament.firstFEMNode;
        disk.nodeCount = ligament.nodeCount;
        output.write(reinterpret_cast<const char*>(&disk), sizeof(disk));
    }
    for (const NMFEMNodeStateGPU& node : accepted.femNodes) {
        const LigamentSnapshotNode disk{{
            node.positionAndMass.x,
            node.positionAndMass.y,
            node.positionAndMass.z,
        }};
        output.write(reinterpret_cast<const char*>(&disk), sizeof(disk));
    }
    require(output.good(), "Open Knee ligament snapshot write failed");
}

} // namespace

int main(const int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2 || argc == 3,
                    "usage: probe OPEN_KNEE_NHKNEE1 [ACCEPTED_NHKFEM1|--active-qat-newtons=FORCE]");
            double activeQATTractionNewtons = 0.0;
            if (argc == 3) {
                constexpr std::string_view prefix =
                    "--active-qat-newtons=";
                const std::string_view argument(argv[2]);
                if (argument.starts_with(prefix)) {
                    activeQATTractionNewtons = std::stod(
                        std::string(argument.substr(prefix.size())));
                    require(std::isfinite(activeQATTractionNewtons) &&
                                activeQATTractionNewtons > 0.0,
                            "Open Knee active QAT force must be positive");
                }
            }
            const bool activeQuadricepsTendon =
                activeQATTractionNewtons > 0.0;
            const std::vector<std::byte> bytes = readBytes(argv[1]);
            metalrobo::NumiHumanKneePayload payload;
            const auto decoded = metalrobo::decodeNumiHumanKneePayload(bytes, payload);
            if (!decoded.succeeded()) {
                throw std::runtime_error(
                    std::string("Open Knee ligament payload rejected: ") +
                    metalrobo::numiHumanKneeStatusName(decoded.status));
            }

            auto parsed = numi::matter::parseMatterFile(
                NUMI_HUMAN_OPEN_KNEE_LIGAMENT_MATERIAL);
            require(parsed.succeeded(), "Open Knee ligament material did not parse");

            numi::matter::WorldSource source;
            source.environmentCount = 1u;
            source.frameTimestep = 1.0e-4;
            source.gravity = {0.0, 0.0, 0.0};
            source.mixedSolver.newtonIterations = 4u;
            source.mixedSolver.fgmresRestart = 16u;
            source.mixedSolver.fgmresIterations = 32u;
            source.mixedSolver.lineSearchSteps = 6u;

            std::vector<LigamentRuntime> ligaments;
            std::uint32_t totalNodes = 0u;
            std::uint32_t totalTetrahedra = 0u;
            for (std::uint32_t regionIndex = 0u;
                 regionIndex < payload.regions.size(); ++regionIndex) {
                const auto& region = payload.regions[regionIndex];
                const bool exactTissue =
                    region.kind == metalrobo::NumiHumanKneeRegionKind::ligament ||
                    (region.kind == metalrobo::NumiHumanKneeRegionKind::tendon &&
                     (region.name == patellarTendonSpec.name ||
                      (activeQuadricepsTendon &&
                       region.name == quadricepsTendonSpec.name)));
                if (!exactTissue)
                    continue;
                const auto& spec = specification(region.name);
                require(region.material.hasHomogeneousFiber &&
                            region.material.hasIsochoricInSituStretch,
                        "Open Knee source fibre material is absent");
                ligaments.push_back({
                    .specification = &spec,
                    .payloadRegion = regionIndex,
                    .firstFEMNode = totalNodes,
                    .nodeCount = region.nodeCount,
                    .tetrahedronCount = region.tetrahedronCount,
                });
                totalNodes += region.nodeCount;
                totalTetrahedra += region.tetrahedronCount;
            }
            const std::size_t expectedRegionCount =
                ligamentSpecs.size() + (activeQuadricepsTendon ? 2u : 1u);
            const std::uint32_t expectedNodeCount =
                activeQuadricepsTendon ? 62402u : 47439u;
            const std::uint32_t expectedTetrahedronCount =
                activeQuadricepsTendon ? 264442u : 195032u;
            require(ligaments.size() == expectedRegionCount &&
                        totalNodes == expectedNodeCount &&
                        totalTetrahedra == expectedTetrahedronCount,
                    "Open Knee exact ligament/patellar-tendon topology drifted");

            std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads(totalNodes);
            std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors(totalNodes);
            for (auto& load : nodeLoads)
                std::fill_n(load.endpointIndex, 4u, NM_INVALID_INDEX);
            for (auto& anchor : nodeAnchors) anchor.bodyIndex = NM_INVALID_INDEX;

            const std::uint32_t sourceFemur = femurBody(payload.side);
            const std::uint32_t sourceTibia = tibiaBody(payload.side);
            const std::uint32_t sourcePatella = patellaBody(payload.side);
            Vec3 activeQATAxis{};
            std::uint32_t activeQATLoadNodeCount = 0u;
            std::uint32_t activePTLPatellaLoadNodeCount = 0u;
            std::uint32_t activePTLTibiaLoadNodeCount = 0u;
            for (LigamentRuntime& ligament : ligaments) {
                const auto& region = payload.regions[ligament.payloadRegion];
                numi::matter::MaterialProgram material = parsed.material;
                material.name = "open_knee_" + region.name +
                    "_transverse_isotropic_smooth_preflight";
                material.fingerprint = 0u;
                setParameter(material, "density", 1000.0);
                setParameter(material, "c1", 1.0e6 * region.material.c1MPa);
                setParameter(material, "c3", 1.0e6 * region.material.c3MPa);
                setParameter(material, "c4", region.material.c4);
                setParameter(material, "bulk",
                             1.0e6 * region.material.bulkModulusMPa);
                // The source final value is admitted in NHKNEE1 ABI 2, but a
                // one-step jump to that residual stress failed the bounded
                // nonlinear/performance gate. Apply neutral stretch until a
                // staged in-situ equilibrium ramp owns initialization.
                setParameter(material, "initial_stretch", 1.0);
                setParameter(material, "fiber_x",
                             region.material.homogeneousFiberWorld[0u]);
                setParameter(material, "fiber_y",
                             region.material.homogeneousFiberWorld[1u]);
                setParameter(material, "fiber_z",
                             region.material.homogeneousFiberWorld[2u]);
                setParameter(material, "tension_smoothing", 1.0e-4);
                setParameter(material, "numerical_viscosity", 25.0);
                const std::uint32_t materialIndex =
                    static_cast<std::uint32_t>(source.materials.size());
                source.materials.push_back(std::move(material));

                numi::matter::ObjectSource object;
                object.name = "numi_human_exact_open_knee_" + region.name;
                object.materialIndex = materialIndex;
                object.representation = numi::matter::Representation::fem;
                object.mixedFEM = false;
                object.deformableSelfContact = false;
                object.characteristicLength = 0.001;
                object.femNodes.reserve(region.nodeCount);
                object.tetrahedra.reserve(region.tetrahedronCount);
                for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
                    const std::uint32_t sourceNode = region.firstNode + local;
                    const auto& node = payload.nodes[sourceNode];
                    object.femNodes.push_back({
                        node.restWorld[0u], node.restWorld[1u], node.restWorld[2u]});
                    if (!node.rigidlyAttached) continue;
                    const std::uint32_t femNode = ligament.firstFEMNode + local;
                    auto& anchor = nodeAnchors[femNode];
                    if (node.anchorBodyIndex == sourceFemur) {
                        anchor.bodyIndex = 0u;
                        ++ligament.femurAnchorCount;
                    } else if (node.anchorBodyIndex == sourceTibia) {
                        anchor.bodyIndex = 1u;
                        ++ligament.tibiaAnchorCount;
                    } else if (node.anchorBodyIndex == sourcePatella) {
                        anchor.bodyIndex = 2u;
                        ++ligament.patellaAnchorCount;
                    } else {
                        throw std::runtime_error(
                            "Open Knee tissue anchor is not femur, tibia, or patella owned");
                    }
                    anchor.flags = NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
                    anchor.localPoint = {
                        node.restWorld[0u], node.restWorld[1u], node.restWorld[2u], 0.0f};
                    object.femFixedNodes.push_back(local);
                }
                for (std::uint32_t local = 0u;
                     local < region.tetrahedronCount; ++local) {
                    const auto& sourceTet =
                        payload.tetrahedra[region.firstTetrahedron + local];
                    std::array<std::uint32_t, 4u> nodes{};
                    for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
                        require(sourceTet[corner] >= region.firstNode &&
                                    sourceTet[corner] <
                                        region.firstNode + region.nodeCount,
                                "Open Knee ligament tetrahedron crosses regions");
                        nodes[corner] = sourceTet[corner] - region.firstNode;
                    }
                    object.tetrahedra.push_back({nodes});
                }
                const bool twoSidedAttachment = region.name == patellarTendonSpec.name
                    ? ligament.patellaAnchorCount > 0u &&
                        ligament.tibiaAnchorCount > 0u &&
                        ligament.femurAnchorCount == 0u
                    : region.name == quadricepsTendonSpec.name
                    ? ligament.patellaAnchorCount > 0u &&
                        ligament.tibiaAnchorCount == 0u &&
                        ligament.femurAnchorCount == 0u
                    : ligament.femurAnchorCount > 0u &&
                        ligament.tibiaAnchorCount > 0u &&
                        ligament.patellaAnchorCount == 0u;
                require(twoSidedAttachment &&
                            object.femFixedNodes.size() ==
                                ligament.femurAnchorCount +
                                ligament.tibiaAnchorCount +
                                ligament.patellaAnchorCount,
                        "Open Knee tissue lacks its exact two-sided bone attachments");
                source.objects.push_back(std::move(object));
            }

            if (activeQuadricepsTendon) {
                const auto qat = std::find_if(
                    ligaments.begin(), ligaments.end(),
                    [&payload](const LigamentRuntime& candidate) {
                        return payload.regions[candidate.payloadRegion].name ==
                            quadricepsTendonSpec.name;
                    });
                require(qat != ligaments.end(),
                        "Open Knee active QAT region is unavailable");
                const auto& region = payload.regions[qat->payloadRegion];
                Vec3 centroid{};
                Vec3 patellaCentroid{};
                std::uint32_t patellaSamples = 0u;
                for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
                    const auto& node = payload.nodes[region.firstNode + local];
                    const Vec3 point{node.restWorld[0u], node.restWorld[1u],
                                     node.restWorld[2u]};
                    centroid = add(centroid, point);
                    if (node.rigidlyAttached &&
                        node.anchorBodyIndex == sourcePatella) {
                        patellaCentroid = add(patellaCentroid, point);
                        ++patellaSamples;
                    }
                }
                require(patellaSamples == qat->patellaAnchorCount &&
                            patellaSamples > 0u,
                        "Open Knee active QAT patellar insertion is incomplete");
                centroid = scale(centroid, 1.0 / region.nodeCount);
                patellaCentroid = scale(
                    patellaCentroid, 1.0 / patellaSamples);
                activeQATAxis = subtract(centroid, patellaCentroid);
                const double axisLength = norm(activeQATAxis);
                require(std::isfinite(axisLength) && axisLength > 1.0e-6,
                        "Open Knee active QAT axis is invalid");
                activeQATAxis = scale(activeQATAxis, 1.0 / axisLength);
                // This focused path qualifies the efficient QAT force-transfer
                // law: the source quadriceps resultant is distributed over the
                // exact patellar enthesis.  Loading the free proximal volume
                // instead would be a volumetric tendon experiment and would
                // incorrectly couple the independent LCL/PCL/ACL/MCL Newton
                // solves to an extensor-chain force that they do not own.
                std::vector<std::uint32_t> loadNodes;
                for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
                    const auto& node = payload.nodes[region.firstNode + local];
                    if (node.rigidlyAttached &&
                        node.anchorBodyIndex == sourcePatella) {
                        loadNodes.push_back(qat->firstFEMNode + local);
                    }
                }
                activeQATLoadNodeCount = static_cast<std::uint32_t>(
                    loadNodes.size());
                require(activeQATLoadNodeCount == qat->patellaAnchorCount,
                        "Open Knee active QAT enthesis coverage is incomplete");
                const float weight =
                    1.0f / static_cast<float>(activeQATLoadNodeCount);
                for (const std::uint32_t node : loadNodes) {
                    nodeLoads[node].endpointIndex[0u] = 0u;
                    nodeLoads[node].scale.x = weight;
                }
                const auto ptl = std::find_if(
                    ligaments.begin(), ligaments.end(),
                    [&payload](const LigamentRuntime& candidate) {
                        return payload.regions[candidate.payloadRegion].name ==
                            patellarTendonSpec.name;
                    });
                require(ptl != ligaments.end(),
                        "Open Knee active PTL region is unavailable");
                const auto& ptlRegion = payload.regions[ptl->payloadRegion];
                activePTLPatellaLoadNodeCount = ptl->patellaAnchorCount;
                activePTLTibiaLoadNodeCount = ptl->tibiaAnchorCount;
                require(activePTLPatellaLoadNodeCount > 0u &&
                            activePTLTibiaLoadNodeCount > 0u,
                        "Open Knee active PTL entheses are incomplete");
                const float patellaWeight =
                    1.0f / static_cast<float>(activePTLPatellaLoadNodeCount);
                const float tibiaWeight =
                    -1.0f / static_cast<float>(activePTLTibiaLoadNodeCount);
                std::uint32_t mappedPatellaNodes = 0u;
                std::uint32_t mappedTibiaNodes = 0u;
                for (std::uint32_t local = 0u;
                     local < ptlRegion.nodeCount; ++local) {
                    const auto& node =
                        payload.nodes[ptlRegion.firstNode + local];
                    if (!node.rigidlyAttached) continue;
                    const std::uint32_t femNode = ptl->firstFEMNode + local;
                    nodeLoads[femNode].endpointIndex[0u] = 1u;
                    if (node.anchorBodyIndex == sourcePatella) {
                        nodeLoads[femNode].scale.x = patellaWeight;
                        ++mappedPatellaNodes;
                    } else if (node.anchorBodyIndex == sourceTibia) {
                        nodeLoads[femNode].scale.x = tibiaWeight;
                        ++mappedTibiaNodes;
                    }
                }
                require(mappedPatellaNodes == activePTLPatellaLoadNodeCount &&
                            mappedTibiaNodes == activePTLTibiaLoadNodeCount,
                        "Open Knee active PTL force-couple coverage drifted");
            }

            numi::matter::CompileOptions compileOptions;
            compileOptions.maximumRateExponent = 0u;
            const auto compileStart = std::chrono::steady_clock::now();
            auto compiled = numi::matter::compileWorld(source, compileOptions);
            const auto compileEnd = std::chrono::steady_clock::now();
            if (!compiled.succeeded()) {
                std::string message = "Open Knee ligament FEM world did not compile";
                for (const auto& diagnostic : compiled.diagnostics)
                    message += "; " + diagnostic.message;
                throw std::runtime_error(message);
            }

            numi::matter::Runtime runtime;
            const auto initialized = runtime.initialize(compiled.world, {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            });
            require(initialized.encoded && runtime.valid(),
                    "Open Knee ligament FEM runtime did not initialize");
            const auto initial = runtime.snapshot();
            require(initial.available && initial.femNodes.size() == totalNodes,
                    "Open Knee ligament initial snapshot is unavailable");

            std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
            if (activeQuadricepsTendon) {
                NMNumiHumanTendonFEMEndpointReplacementGPU replacement{};
                replacement.loadEndpointIndex = 0u;
                replacement.anchorEndpointIndex = 1u;
                replacement.flags =
                    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE |
                    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW |
                    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE;
                replacement.forceOwnerFraction.x = 1.0f;
                replacements.push_back(replacement);
            }
            numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
            require(adapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = replacements,
                        .endpointCount = activeQuadricepsTendon ? 2u : 1u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction =
                            activeQuadricepsTendon ? 1.0f : 0.0f,
                    }, {.metallib = NUMI_MATTER_METALLIB}),
                    "Open Knee ligament two-way adapter did not initialize");
            const auto program = adapter.program();
            require(program.valid(), "Open Knee ligament program is invalid");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(device != nil && queue != nil,
                    "Open Knee ligament Metal queue is unavailable");
            std::array<MRNumiHumanTendonBindingGPU, 2u> bindings{};
            std::array<MRNumiHumanTendonTransferResultGPU, 2u> transfers{};
            const std::uint32_t endpointCount =
                activeQuadricepsTendon ? 2u : 1u;
            for (std::uint32_t endpoint = 0u; endpoint < endpointCount;
                 ++endpoint) {
                bindings[endpoint].muscleIndex = 0u;
                bindings[endpoint].endpointOrdinal = endpoint;
                bindings[endpoint].bodyIndex = endpoint == 0u ? 0u : 1u;
                bindings[endpoint].mode =
                    MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
                bindings[endpoint].envelopeIndex = MR_INVALID_INDEX;
                transfers[endpoint].status =
                    MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
                transfers[endpoint].environment = 0u;
                transfers[endpoint].bindingIndex = endpoint;
                transfers[endpoint].envelopeIndex = MR_INVALID_INDEX;
            }
            if (activeQuadricepsTendon) {
                const Vec3 loadBoneForce = scale(
                    activeQATAxis, -activeQATTractionNewtons);
                transfers[0u].terminalWorldForce = {
                    static_cast<float>(loadBoneForce[0u]),
                    static_cast<float>(loadBoneForce[1u]),
                    static_cast<float>(loadBoneForce[2u]), 0.0f};
                transfers[1u].terminalWorldForce = {
                    static_cast<float>(-loadBoneForce[0u]),
                    static_cast<float>(-loadBoneForce[1u]),
                    static_cast<float>(-loadBoneForce[2u]), 0.0f};
            }
            id<MTLBuffer> bindingBuffer = [device
                newBufferWithBytes:bindings.data()
                length:endpointCount * sizeof(bindings.front())
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> transferBuffer = [device
                newBufferWithBytes:transfers.data()
                length:endpointCount * sizeof(transfers.front())
                options:MTLResourceStorageModeShared];
            MRNumiHumanStandStatusGPU stand{};
            stand.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            stand.environment = 0u;
            stand.completedSteps = 1u;
            stand.failingIndex = MR_INVALID_INDEX;
            id<MTLBuffer> standBuffer = [device
                newBufferWithBytes:&stand length:sizeof(stand)
                options:MTLResourceStorageModeShared];

            constexpr float prescribedTibiaTranslationX = 1.0e-7f;
            constexpr float prescribedTibiaTranslationZ = 2.0e-7f;
            std::array<MRArticulatedBodyPoseGPU, 3u> poses{};
            poses[0u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            poses[1u].position = {prescribedTibiaTranslationX, 0.0f,
                                  prescribedTibiaTranslationZ, 0.0f};
            poses[1u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            poses[2u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            id<MTLBuffer> poseBuffer = [device
                newBufferWithBytes:poses.data() length:sizeof(poses)
                options:MTLResourceStorageModeShared];

            constexpr std::uint32_t dofCount = 9u;
            constexpr std::uint32_t pointJacobianStride =
                4u * 3u * 3u * dofCount;
            std::array<float, pointJacobianStride> pointJacobians{};
            for (std::uint32_t body = 0u; body < 3u; ++body) {
                for (std::uint32_t point = 0u; point < 4u; ++point) {
                    const std::uint32_t base =
                        (4u * body + point) * 3u * dofCount;
                    pointJacobians[base + 3u * body] = 1.0f;
                    pointJacobians[base + dofCount + 3u * body + 1u] = 1.0f;
                    pointJacobians[base + 2u * dofCount + 3u * body + 2u] = 1.0f;
                }
            }
            id<MTLBuffer> jacobianBuffer = [device
                newBufferWithBytes:pointJacobians.data()
                length:sizeof(pointJacobians)
                options:MTLResourceStorageModeShared];
            std::array<float, 2u * dofCount> generalizedForces{};
            id<MTLBuffer> generalizedForceBuffer = [device
                newBufferWithBytes:generalizedForces.data()
                length:sizeof(generalizedForces)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> runtimeReactionBuffer = (__bridge id<MTLBuffer>)
                runtime.femConstraintReactionBuffer();
            id<MTLBuffer> reactionReadback = [device
                newBufferWithLength:totalNodes * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            require(bindingBuffer != nil && transferBuffer != nil &&
                        standBuffer != nil && poseBuffer != nil &&
                        jacobianBuffer != nil && generalizedForceBuffer != nil &&
                        runtimeReactionBuffer != nil && reactionReadback != nil,
                    "Open Knee ligament borrowed buffers are unavailable");

            const auto execute = [&](const std::uint32_t step,
                                     const bool accepted) {
                stand.code = accepted ? MR_NUMI_HUMAN_STAND_SUCCESS
                                      : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
                stand.completedSteps = accepted ? step + 1u : step;
                std::memcpy(standBuffer.contents, &stand, sizeof(stand));
                std::memset(generalizedForceBuffer.contents, 0,
                            generalizedForceBuffer.length);
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil,
                        "Open Knee ligament command buffer is unavailable");
                metalrobo::MetalNumiHumanTendonLoadPass pass{};
                pass.commandBuffer = (__bridge void*)command;
                pass.bindings = (__bridge void*)bindingBuffer;
                pass.transfers = (__bridge void*)transferBuffer;
                pass.generalizedForces = (__bridge void*)generalizedForceBuffer;
                pass.bodyPoses = (__bridge void*)poseBuffer;
                pass.pointJacobians = (__bridge void*)jacobianBuffer;
                pass.standStatuses = (__bridge void*)standBuffer;
                pass.stepIndex = step;
                pass.environmentCount = 1u;
                pass.endpointCount = endpointCount;
                pass.dofCount = dofCount;
                pass.muscleCount = 1u;
                pass.generalizedForceStride = 2u * dofCount;
                pass.generalizedForceOffset = dofCount;
                pass.pointJacobianStride = pointJacobianStride;
                pass.bodyJacobianPointOffset = 0u;
                pass.bodyPoseStride = 3u;
                pass.articulationFirstBody = 0u;
                if (!program.encodePreDynamics(program.context, pass) ||
                    !program.encodePostValidation(program.context, pass)) {
                    throw std::runtime_error(
                        "Open Knee ligament transaction rejected: " +
                        adapter.diagnostics().message);
                }
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                require(blit != nil,
                        "Open Knee ligament reaction blit is unavailable");
                [blit copyFromBuffer:runtimeReactionBuffer sourceOffset:0u
                            toBuffer:reactionReadback destinationOffset:0u
                                size:totalNodes * sizeof(nm_float4)];
                [blit endEncoding];
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "Open Knee ligament command did not complete");
            };

            const auto stepStart = std::chrono::steady_clock::now();
            execute(0u, true);
            const auto stepEnd = std::chrono::steady_clock::now();
            std::array<float, 2u * dofCount> acceptedGeneralizedForces{};
            std::memcpy(acceptedGeneralizedForces.data(),
                        generalizedForceBuffer.contents,
                        generalizedForceBuffer.length);
            const auto accepted = runtime.snapshot();
            require(accepted.available && accepted.femNodes.size() == totalNodes &&
                        accepted.statuses.size() == 1u,
                    "Open Knee ligament Matter snapshot is unavailable");
            const NMMatterStatusGPU& acceptedStatus = accepted.statuses[0u];
            require(acceptedStatus.code == NM_STATUS_SUCCESS,
                    ("Open Knee ligament Matter step was rejected: status=" +
                     std::to_string(acceptedStatus.code) + " object=" +
                     std::to_string(acceptedStatus.objectIndex) + " index=" +
                     std::to_string(acceptedStatus.failingIndex) +
                     " microsteps=" +
                     std::to_string(acceptedStatus.completedMicrosteps) +
                     " fgmres_iterations=" +
                     std::to_string(acceptedStatus.fgmresIterations) +
                     " diagnostics=(" +
                     std::to_string(acceptedStatus.diagnostics.x) + "," +
                     std::to_string(acceptedStatus.diagnostics.y) + "," +
                     std::to_string(acceptedStatus.diagnostics.z) + "," +
                     std::to_string(acceptedStatus.diagnostics.w) + ")"));

            std::vector<nm_float4> acceptedReactions(totalNodes);
            std::memcpy(acceptedReactions.data(), reactionReadback.contents,
                        acceptedReactions.size() * sizeof(acceptedReactions.front()));
            double maximumDisplacement = 0.0;
            double minimumJ = std::numeric_limits<double>::infinity();
            double maximumJ = 0.0;
            std::array<double, 3u> bodyReactionL1{};
            std::array<double, 3u> bodyGeneralizedForceL1{};
            Vec3 activeQATEnthesisReaction{};
            Vec3 activePTLPatellaReaction{};
            Vec3 activePTLTibiaReaction{};
            for (std::uint32_t body = 0u; body < 3u; ++body) {
                for (std::uint32_t axis = 0u; axis < 3u; ++axis)
                    bodyGeneralizedForceL1[body] += std::abs(
                        acceptedGeneralizedForces[dofCount + 3u * body + axis]);
            }
            for (LigamentRuntime& ligament : ligaments) {
                const auto& region = payload.regions[ligament.payloadRegion];
                double femurReaction = 0.0;
                double tibiaReaction = 0.0;
                double patellaReaction = 0.0;
                for (std::uint32_t local = 0u; local < ligament.nodeCount; ++local) {
                    const std::uint32_t index = ligament.firstFEMNode + local;
                    maximumDisplacement = std::max(maximumDisplacement, norm(subtract(
                        position(accepted.femNodes[index]),
                        position(initial.femNodes[index]))));
                    const auto& anchor = nodeAnchors[index];
                    if ((anchor.flags &
                         NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u)
                        continue;
                    const Vec3 reaction{acceptedReactions[index].x,
                                        acceptedReactions[index].y,
                                        acceptedReactions[index].z};
                    const double magnitude = norm(reaction);
                    require(std::isfinite(magnitude),
                            "Open Knee ligament reaction is nonfinite");
                    bodyReactionL1[anchor.bodyIndex] += magnitude;
                    if (anchor.bodyIndex == 0u) femurReaction += magnitude;
                    else if (anchor.bodyIndex == 1u) tibiaReaction += magnitude;
                    else patellaReaction += magnitude;
                    if (region.name == quadricepsTendonSpec.name &&
                        anchor.bodyIndex == 2u) {
                        activeQATEnthesisReaction = add(
                            activeQATEnthesisReaction, reaction);
                    }
                    if (region.name == patellarTendonSpec.name &&
                        anchor.bodyIndex == 2u) {
                        activePTLPatellaReaction = add(
                            activePTLPatellaReaction, reaction);
                    } else if (region.name == patellarTendonSpec.name &&
                               anchor.bodyIndex == 1u) {
                        activePTLTibiaReaction = add(
                            activePTLTibiaReaction, reaction);
                    }
                }
                const bool completeReaction = region.name == patellarTendonSpec.name
                    ? patellaReaction > 0.0 && tibiaReaction > 0.0
                    : region.name == quadricepsTendonSpec.name
                    ? patellaReaction > 0.0 && femurReaction == 0.0 &&
                        tibiaReaction == 0.0
                    : femurReaction > 0.0 && tibiaReaction > 0.0;
                require(completeReaction,
                        "Open Knee tissue two-sided reaction is incomplete");
                for (std::uint32_t local = 0u;
                     local < region.tetrahedronCount; ++local) {
                    const auto& sourceTet =
                        payload.tetrahedra[region.firstTetrahedron + local];
                    std::array<std::uint32_t, 4u> n{};
                    for (std::uint32_t corner = 0u; corner < 4u; ++corner)
                        n[corner] = ligament.firstFEMNode +
                            sourceTet[corner] - region.firstNode;
                    const double restVolume = signedVolume(
                        position(initial.femNodes[n[0u]]),
                        position(initial.femNodes[n[1u]]),
                        position(initial.femNodes[n[2u]]),
                        position(initial.femNodes[n[3u]]));
                    const double currentVolume = signedVolume(
                        position(accepted.femNodes[n[0u]]),
                        position(accepted.femNodes[n[1u]]),
                        position(accepted.femNodes[n[2u]]),
                        position(accepted.femNodes[n[3u]]));
                    require(std::abs(restVolume) > 1.0e-18,
                            "Open Knee ligament tetrahedron is degenerate");
                    const double j = currentVolume / restVolume;
                    require(std::isfinite(j),
                            "Open Knee ligament determinant is nonfinite");
                    minimumJ = std::min(minimumJ, j);
                    maximumJ = std::max(maximumJ, j);
                }
                std::cout << " tissue=" << region.name
                          << " kind=" << (region.name == patellarTendonSpec.name
                              ? "patellar_tendon"
                              : region.name == quadricepsTendonSpec.name
                              ? "quadriceps_tendon" : "ligament")
                          << " nodes=" << ligament.nodeCount
                          << " tetrahedra=" << ligament.tetrahedronCount
                          << " femur_anchors=" << ligament.femurAnchorCount
                          << " tibia_anchors=" << ligament.tibiaAnchorCount
                          << " patella_anchors=" << ligament.patellaAnchorCount
                          << " source_c1_mpa=" << ligament.specification->c1MPa
                          << " source_k_mpa=" << ligament.specification->bulkMPa
                          << " source_initial_stretch="
                          << ligament.specification->initialStretch
                          << " applied_initial_stretch=1"
                          << " source_lambda_max="
                          << region.material.lambdaMaximum
                          << " source_c3_mpa=" << region.material.c3MPa
                          << " source_c4=" << region.material.c4
                          << " source_c5_mpa=" << region.material.c5MPa
                          << " fiber_world="
                          << region.material.homogeneousFiberWorld[0u] << ","
                          << region.material.homogeneousFiberWorld[1u] << ","
                          << region.material.homogeneousFiberWorld[2u]
                          << " femur_reaction_l1_n=" << femurReaction
                          << " tibia_reaction_l1_n=" << tibiaReaction
                          << " patella_reaction_l1_n=" << patellaReaction
                          << "\n";
            }
            const double activeQATEnthesisReactionResultant =
                norm(activeQATEnthesisReaction);
            const double activePTLPatellaReactionResultant =
                norm(activePTLPatellaReaction);
            const double activePTLTibiaReactionResultant =
                norm(activePTLTibiaReaction);
            const double activeQATEnthesisRelativeError =
                activeQuadricepsTendon
                ? std::abs(activeQATEnthesisReactionResultant -
                      activeQATTractionNewtons) /
                      std::max(1.0, activeQATTractionNewtons)
                : 0.0;
            const double activePTLPatellaRelativeError =
                activeQuadricepsTendon
                ? std::abs(activePTLPatellaReactionResultant -
                      activeQATTractionNewtons) /
                      std::max(1.0, activeQATTractionNewtons)
                : 0.0;
            const double activePTLTibiaRelativeError =
                activeQuadricepsTendon
                ? std::abs(activePTLTibiaReactionResultant -
                      activeQATTractionNewtons) /
                      std::max(1.0, activeQATTractionNewtons)
                : 0.0;
            // No active endpoint is applied to the femur. Its fixed-node L1
            // reaction is a conservative measured allowance for the passive
            // tissue contribution superposed on each active enthesis reaction.
            const double passiveReactionAllowanceNewtons =
                activeQuadricepsTendon
                ? std::max(1.0e-4 * activeQATTractionNewtons,
                           bodyReactionL1[0u])
                : 0.0;
            require(maximumDisplacement > 0.0 && maximumDisplacement < 0.001 &&
                        minimumJ >= 0.50 && maximumJ <= 1.75 &&
                        bodyReactionL1[0u] > 0.0 && bodyReactionL1[1u] > 0.0 &&
                        bodyReactionL1[2u] > 0.0 &&
                        bodyGeneralizedForceL1[0u] > 0.0 &&
                        bodyGeneralizedForceL1[1u] > 0.0 &&
                        bodyGeneralizedForceL1[2u] > 0.0 &&
                        (!activeQuadricepsTendon ||
                         (std::isfinite(activeQATEnthesisRelativeError) &&
                          std::abs(activeQATEnthesisReactionResultant -
                                   activeQATTractionNewtons) <=
                              passiveReactionAllowanceNewtons &&
                          std::isfinite(activePTLPatellaRelativeError) &&
                          std::abs(activePTLPatellaReactionResultant -
                                   activeQATTractionNewtons) <=
                              passiveReactionAllowanceNewtons &&
                          std::isfinite(activePTLTibiaRelativeError) &&
                          std::abs(activePTLTibiaReactionResultant -
                                   activeQATTractionNewtons) <=
                              passiveReactionAllowanceNewtons)),
                    "Open Knee ligament accepted mechanics are invalid");

            execute(1u, false);
            const auto rolledBack = runtime.snapshot();
            require(rolledBack.available &&
                        std::memcmp(rolledBack.femNodes.data(),
                                    accepted.femNodes.data(),
                                    accepted.femNodes.size() *
                                        sizeof(NMFEMNodeStateGPU)) == 0,
                    "Open Knee ligament rejected step did not roll back");
            require(runtime.restore(initial).encoded,
                    "Open Knee ligament initial restore failed");
            execute(0u, true);
            const auto replay = runtime.snapshot();
            require(replay.available &&
                        std::memcmp(replay.femNodes.data(), accepted.femNodes.data(),
                                    accepted.femNodes.size() *
                                        sizeof(NMFEMNodeStateGPU)) == 0 &&
                        std::memcmp(reactionReadback.contents,
                                    acceptedReactions.data(),
                                    acceptedReactions.size() *
                                        sizeof(acceptedReactions.front())) == 0 &&
                        std::memcmp(generalizedForceBuffer.contents,
                                    acceptedGeneralizedForces.data(),
                                    generalizedForceBuffer.length) == 0,
                    "Open Knee ligament accepted replay is not bitwise");
            const auto diagnostics = adapter.diagnostics();
            require(diagnostics.initialized && diagnostics.encodedPassCount == 3u &&
                        diagnostics.abortCount == 0u &&
                        diagnostics.fingerprint != 0u &&
                        (!activeQuadricepsTendon ||
                         (std::abs(
                              diagnostics.assembledExternalForceL1Newtons -
                              3.0 * activeQATTractionNewtons) <=
                              1.0e-4 * std::max(
                                  1.0, 3.0 * activeQATTractionNewtons) &&
                          std::abs(
                              diagnostics.assembledExternalForceResultantNewtons -
                              activeQATTractionNewtons) <=
                              1.0e-4 * std::max(
                                  1.0, activeQATTractionNewtons))),
                    "Open Knee ligament adapter diagnostics are incomplete");
            if (argc == 3 && !activeQuadricepsTendon) {
                writeAcceptedSnapshot(
                    argv[2], payload, ligaments, accepted, totalTetrahedra);
            }

            const double compileMilliseconds =
                std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();
            const double stepMilliseconds =
                std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
            std::cout
                << "numi_human_open_knee_tissues=passed"
                << " side=" << (payload.side == metalrobo::NumiHumanKneeSide::left
                    ? "left" : "right_mirrored")
                << " device=\"" << initialized.device << "\""
                << " exact_regions=" << ligaments.size()
                << " fem_nodes=" << totalNodes
                << " tetrahedra=" << totalTetrahedra
                << " prescribed_tibia_translation_x_m="
                << prescribedTibiaTranslationX
                << " prescribed_tibia_translation_z_m="
                << prescribedTibiaTranslationZ
                << " maximum_displacement_m=" << maximumDisplacement
                << " min_J=" << minimumJ
                << " max_J=" << maximumJ
                << " femur_reaction_l1_n=" << bodyReactionL1[0u]
                << " tibia_reaction_l1_n=" << bodyReactionL1[1u]
                << " patella_reaction_l1_n=" << bodyReactionL1[2u]
                << " femur_generalized_force_l1_n="
                << bodyGeneralizedForceL1[0u]
                << " tibia_generalized_force_l1_n="
                << bodyGeneralizedForceL1[1u]
                << " patella_generalized_force_l1_n="
                << bodyGeneralizedForceL1[2u]
                << " compile_ms=" << compileMilliseconds
                << " accepted_step_wall_ms=" << stepMilliseconds
                << " peak_rss_bytes=" << peakResidentBytes()
                << " replay=bitwise rollback=verified"
                << " active_qat="
                << (activeQuadricepsTendon ? "true" : "false")
                << " active_qat_load_nodes=" << activeQATLoadNodeCount
                << " active_qat_traction_n=" << activeQATTractionNewtons
                << " active_qat_enthesis_reaction_resultant_n="
                << activeQATEnthesisReactionResultant
                << " active_qat_enthesis_relative_error="
                << activeQATEnthesisRelativeError
                << " passive_reaction_allowance_n="
                << passiveReactionAllowanceNewtons
                << " active_ptl_patella_load_nodes="
                << activePTLPatellaLoadNodeCount
                << " active_ptl_tibia_load_nodes="
                << activePTLTibiaLoadNodeCount
                << " active_ptl_patella_reaction_resultant_n="
                << activePTLPatellaReactionResultant
                << " active_ptl_tibia_reaction_resultant_n="
                << activePTLTibiaReactionResultant
                << " active_ptl_patella_relative_error="
                << activePTLPatellaRelativeError
                << " active_ptl_tibia_relative_error="
                << activePTLTibiaRelativeError
                << " assembled_external_force_l1_n="
                << diagnostics.assembledExternalForceL1Newtons
                << " assembled_external_force_resultant_n="
                << diagnostics.assembledExternalForceResultantNewtons
                << " production_owner_fraction="
                << (activeQuadricepsTendon ? 1 : 0)
                << " accepted_snapshot="
                << (argc == 3 && !activeQuadricepsTendon ? argv[2] : "none")
                << " material_boundary=source_homogeneous_fibre_axis_with_smooth_exponential_apple_gpu_approximation_source_in_situ_stretch_admitted_but_held_neutral_pending_staged_equilibrium_ramp_not_exact_FEBio_exp_linear_Ei_law"
                << " mechanics_boundary="
                << (activeQuadricepsTendon
                    ? "source_tendon_resultant_distributed_over_exact_QAT_patellar_enthesis_and_zero_resultant_PTL_patella_to_tibia_enthesis_force_couple_plus_passive_tissue_FEM_not_active_volumetric_tendon_or_clinical_validation"
                    : "exact_topology_three_body_attachment_and_reaction_transaction_not_loaded_joint_or_clinical_validation")
                << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "numi_human_open_knee_tissues=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
