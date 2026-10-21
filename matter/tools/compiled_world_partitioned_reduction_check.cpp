#include "numi/matter/compiled_world_partitioned_reduction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {
using Position = std::array<double, 3>;
using Tet = std::array<std::uint32_t, 4>;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "compiled world partitioned reduction check failed: "
              << message << '\n';
    std::exit(1);
}

void require(const bool condition, const std::string& message) {
    if (!condition) fail(message);
}

bool near(const double a, const double b, const double tolerance = 3.0e-5) {
    return std::abs(a - b) <= tolerance *
        std::max({1.0, std::abs(a), std::abs(b)});
}

constexpr std::uint32_t nx = 5u;
constexpr std::uint32_t ny = 4u;
constexpr std::uint32_t nz = 4u;
constexpr std::uint32_t sourceNodeCount = nx * ny * nz;
constexpr std::uint32_t targetGlobalOffset = 4u;

std::uint32_t nodeIndex(const std::uint32_t x, const std::uint32_t y,
                        const std::uint32_t z) {
    return x + nx * (y + ny * z);
}

void appendGridTetrahedra(numi::matter::CompiledWorld& world) {
    for (std::uint32_t z = 0u; z + 1u < nz; ++z) {
        for (std::uint32_t y = 0u; y + 1u < ny; ++y) {
            for (std::uint32_t x = 0u; x + 1u < nx; ++x) {
                const std::uint32_t v000 = targetGlobalOffset +
                    nodeIndex(x, y, z);
                const std::uint32_t v100 = targetGlobalOffset +
                    nodeIndex(x + 1u, y, z);
                const std::uint32_t v010 = targetGlobalOffset +
                    nodeIndex(x, y + 1u, z);
                const std::uint32_t v110 = targetGlobalOffset +
                    nodeIndex(x + 1u, y + 1u, z);
                const std::uint32_t v001 = targetGlobalOffset +
                    nodeIndex(x, y, z + 1u);
                const std::uint32_t v101 = targetGlobalOffset +
                    nodeIndex(x + 1u, y, z + 1u);
                const std::uint32_t v011 = targetGlobalOffset +
                    nodeIndex(x, y + 1u, z + 1u);
                const std::uint32_t v111 = targetGlobalOffset +
                    nodeIndex(x + 1u, y + 1u, z + 1u);
                const std::array<Tet, 6> cellTets{{
                    {v000, v100, v110, v111},
                    {v000, v110, v010, v111},
                    {v000, v010, v011, v111},
                    {v000, v011, v001, v111},
                    {v000, v001, v101, v111},
                    {v000, v101, v100, v111},
                }};
                const std::uint32_t label = x < 2u ? 0u : 1u;
                for (const Tet& nodes : cellTets) {
                    NMTetrahedronGPU tet{};
                    tet.nodes = {nodes[0], nodes[1], nodes[2], nodes[3]};
                    tet.inverseRestRow0.w = 1.0f / 6.0f;
                    tet.identity = {
                        0u, 1u, 1u,
                        NM_OBJECT_ACTIVE |
                            (label << NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT),
                    };
                    world.fem.tetrahedra.push_back(tet);
                }
            }
        }
    }
}

numi::matter::CompiledWorld makeCompiledWorld(
    const std::uint32_t attachmentNode,
    const std::uint32_t contactNode,
    const std::uint32_t thoraxNode,
    const std::uint32_t vesselBindingNode,
    const std::uint32_t fixedNode
) {
    using namespace numi::matter;
    CompiledWorld world;
    world.dispatch.gridNodeCount = 8u;
    world.mixedSolver.executionBudgets.z =
        static_cast<std::uint32_t>(FEMPreconditionerMode::regionalGalerkin);

    world.objects.resize(2u);
    world.objects[0].representation = NM_REPRESENTATION_FEM;
    world.objects[0].stateOffset = 0u;
    world.objects[0].stateCount = targetGlobalOffset;
    world.objects[0].elementOffset = 0u;
    world.objects[0].elementCount = 1u;
    world.objects[0].topologyGeneration = 1u;

    world.objects[1].representation = NM_REPRESENTATION_FEM;
    world.objects[1].flags = NM_OBJECT_FEM_REGIONAL_MATERIAL;
    world.objects[1].stateOffset = targetGlobalOffset;
    world.objects[1].stateCount = sourceNodeCount;
    world.objects[1].elementOffset = 1u;
    world.objects[1].topologyGeneration = 1u;

    for (std::uint32_t source = 0u; source < targetGlobalOffset; ++source) {
        NMFEMNodeStateGPU node{};
        node.positionAndMass = {
            static_cast<float>(source), 0.0f, 0.0f, 2.0f};
        node.restAndFixed = {
            static_cast<float>(source), 0.0f, 0.0f, 0.0f};
        world.fem.nodes.push_back(node);
        NMFEMTopologyNodeGPU topology{};
        topology.identity = {source, 0u, 1u, NM_TOPOLOGY_ACTIVE};
        world.fem.topologyNodes.push_back(topology);
    }
    NMTetrahedronGPU filler{};
    filler.nodes = {0u, 1u, 2u, 3u};
    filler.inverseRestRow0.w = 1.0f / 6.0f;
    filler.identity = {0u, 0u, 1u, NM_OBJECT_ACTIVE};
    world.fem.tetrahedra.push_back(filler);

    for (std::uint32_t source = 0u; source < sourceNodeCount; ++source) {
        const std::uint32_t x = source % nx;
        const std::uint32_t y = (source / nx) % ny;
        const std::uint32_t z = source / (nx * ny);
        NMFEMNodeStateGPU node{};
        node.positionAndMass = {
            0.5f * x, 0.75f * y, 1.25f * z,
            1.0f + 0.001f * source};
        node.restAndFixed = {
            0.5f * x, 0.75f * y, 1.25f * z,
            source == fixedNode ? 1.0f :
                source == attachmentNode ? 2.0f : 0.0f};
        world.fem.nodes.push_back(node);
        NMFEMTopologyNodeGPU topology{};
        topology.identity = {
            source, 1u, 1u, NM_TOPOLOGY_ACTIVE};
        world.fem.topologyNodes.push_back(topology);
    }
    world.objects[1].elementCount = 1u;
    appendGridTetrahedra(world);
    world.objects[1].elementCount = static_cast<std::uint32_t>(
        world.fem.tetrahedra.size() - world.objects[1].elementOffset);

    NMFEMHumanAttachmentGPU attachment{};
    attachment.identity = {
        targetGlobalOffset + attachmentNode, 0u, 1u, 1u};
    world.fem.humanAttachments.push_back(attachment);

    world.contact.pairs.push_back({
        world.dispatch.gridNodeCount + targetGlobalOffset + contactNode,
        0u, 1u, 0u});

    NMVascularTissueBindingGPU binding{};
    binding.identity = {0u, targetGlobalOffset + vesselBindingNode, 0u, 0u};
    binding.physical.x = 1.0f;
    world.vascular.tissueBindings.push_back(binding);

    NMVascularCavityGPU cavity{};
    cavity.identity = {1u, 0u, 1u, 0u};
    world.vascular.cavities.push_back(cavity);
    NMVascularCavityFaceGPU face{};
    face.nodes = {
        targetGlobalOffset + vesselBindingNode,
        targetGlobalOffset + nodeIndex(1u, 1u, 1u),
        targetGlobalOffset + nodeIndex(1u, 0u, 2u),
        0u};
    face.identity = {0u, 1u, 1u, 0u};
    world.vascular.cavityFaces.push_back(face);
    (void)thoraxNode;
    return world;
}

void runValidCase() {
    using namespace numi::matter;
    const std::uint32_t attachment = nodeIndex(0u, 3u, 3u);
    const std::uint32_t contact = nodeIndex(2u, 1u, 1u);
    const std::uint32_t thorax = nodeIndex(4u, 0u, 0u);
    const std::uint32_t vessel = nodeIndex(1u, 0u, 1u);
    const std::uint32_t fixed = nodeIndex(4u, 3u, 3u);
    CompiledWorld world = makeCompiledWorld(
        attachment, contact, thorax, vessel, fixed);
    const auto sourceNodes = world.fem.nodes;
    const auto sourceTets = world.fem.tetrahedra;
    const std::array<std::uint32_t, 1> thoraxAnchors{
        targetGlobalOffset + thorax};
    const std::array<std::uint32_t, 1> activeContactAnchors{
        targetGlobalOffset + contact};
    CompiledWorldPartitionedReductionOptions options;
    options.thoraxAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(thoraxAnchors);
    options.activeContactAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(activeContactAnchors);
    const auto output = buildCompiledWorldPartitionedReductionBasis(
        world, 1u, options);
    require(output.valid(), output.error.empty() ? output.basis.error :
        output.error);
    const PartitionedReductionBasis& basis = output.basis;
    require(output.objectIndex == 1u && basis.nodeCount == sourceNodeCount,
            "selected compiled FEM object or active source node count changed");
    require(basis.modeCount == 42u && basis.regions.size() == 2u,
            "expected two source regions, affine modes, and six retained anchors");
    require(basis.minimumCoordinateReductionFactor == 4u &&
                std::uint64_t(basis.modeCount) * 4u <=
                    basis.freeCoordinateCount,
            "affine basis did not meet the requested 4:1 free-coordinate reduction");
    require(basis.values.size() ==
                static_cast<std::size_t>(basis.modeCount) * basis.nodeCount,
            "basis is not mode-major per source node");
    require(output.globalFEMNodeIndices.front() == targetGlobalOffset &&
                output.globalFEMNodeIndices.back() ==
                    targetGlobalOffset + sourceNodeCount - 1u &&
                output.sourceNodeIndices.front() == 0u &&
                output.sourceNodeIndices.back() == sourceNodeCount - 1u,
            "compiled global and authored source ancestry maps were not preserved");
    require(output.globalFEMTetrahedronIndices.size() ==
                world.objects[1].elementCount &&
                output.sourceTetrahedronIndices.front() == 0u &&
                output.tetrahedronRegionLabels.size() ==
                    world.objects[1].elementCount,
            "compiled active tetrahedron provenance and labels were not retained");
    require(world.fem.nodes.size() == sourceNodes.size() &&
                world.fem.tetrahedra.size() == sourceTets.size(),
            "compiled source geometry was changed");
    for (std::size_t node = 0u; node < sourceNodes.size(); ++node) {
        require(world.fem.nodes[node].positionAndMass.x ==
                    sourceNodes[node].positionAndMass.x &&
                    world.fem.nodes[node].positionAndMass.y ==
                    sourceNodes[node].positionAndMass.y &&
                    world.fem.nodes[node].positionAndMass.z ==
                    sourceNodes[node].positionAndMass.z &&
                    world.fem.nodes[node].positionAndMass.w ==
                    sourceNodes[node].positionAndMass.w &&
                    world.fem.nodes[node].restAndFixed.w ==
                    sourceNodes[node].restAndFixed.w,
                "compiled nodal source values were modified");
    }
    for (std::size_t tet = 0u; tet < sourceTets.size(); ++tet) {
        const auto& after = world.fem.tetrahedra[tet];
        const auto& before = sourceTets[tet];
        require(after.nodes.x == before.nodes.x &&
                    after.nodes.y == before.nodes.y &&
                    after.nodes.z == before.nodes.z &&
                    after.nodes.w == before.nodes.w &&
                    after.identity.x == before.identity.x &&
                    after.identity.y == before.identity.y &&
                    after.identity.z == before.identity.z &&
                    after.identity.w == before.identity.w,
                "compiled tetrahedron source values were modified");
    }

    std::map<std::uint32_t, std::uint32_t> reasonByGlobal;
    for (std::size_t anchor = 0u; anchor < basis.anchorNodes.size(); ++anchor) {
        const std::uint32_t local = basis.anchorNodes[anchor];
        reasonByGlobal[output.globalFEMNodeIndices[local]] =
            output.anchorReasonMasks[anchor];
    }
    const auto hasReason = [&](const std::uint32_t local,
                               const std::uint32_t reason) {
        const auto found = reasonByGlobal.find(targetGlobalOffset + local);
        return found != reasonByGlobal.end() &&
            (found->second & reason) != 0u;
    };
    require(hasReason(attachment, compiledReductionAnchorAttachment),
            "compiled Human attachment was not retained as a fine anchor");
    require(hasReason(contact, compiledReductionAnchorContact),
            "compiled FEM contact candidate was not retained as a fine anchor");
    require(hasReason(thorax, compiledReductionAnchorThorax),
            "caller-selected thorax node was not retained as a fine anchor");
    require(hasReason(vessel, compiledReductionAnchorVascularBinding) &&
                hasReason(vessel, compiledReductionAnchorVascularCavity),
            "vascular binding/cavity nodes were not retained with their provenance");

    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
        double sum = 0.0;
        for (std::uint32_t partition = 0u;
             partition < basis.partitionCount; ++partition) {
            const double weight = basis.partitionWeights[
                static_cast<std::size_t>(node) * basis.partitionCount + partition];
            require(weight >= 0.0 && std::isfinite(weight),
                    "partition weight is negative or non-finite");
            sum += weight;
        }
        require(near(sum, node == fixed ? 0.0 : 1.0, 1.0e-12),
                "compiled-node partition of unity failed");
    }

    std::vector<Position> positions(basis.nodeCount);
    for (std::uint32_t local = 0u; local < basis.nodeCount; ++local) {
        const auto& node = world.fem.nodes[output.globalFEMNodeIndices[local]];
        positions[local] = {
            node.restAndFixed.x, node.restAndFixed.y, node.restAndFixed.z};
    }
    const Position translation{0.12, -0.30, 0.47};
    const std::array<double, 9> gradient{
        0.11, -0.07, 0.23,
        -0.19, 0.31, 0.08,
        0.04, 0.17, -0.13};
    std::vector<double> q(basis.modeCount, 0.0);
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        const PartitionedReductionMode& descriptor = basis.modes[mode];
        if (descriptor.kind == PartitionedReductionModeKind::regionalTranslation) {
            const auto region = std::find_if(
                basis.regions.begin(), basis.regions.end(),
                [&](const auto& item) {
                    return item.sourceLabel == descriptor.regionLabel;
                });
            require(region != basis.regions.end(),
                    "affine translation lost its source region");
            const std::uint32_t outputAxis = descriptor.component;
            q[mode] = translation[outputAxis] +
                gradient[outputAxis * 3u] * region->center[0] +
                gradient[outputAxis * 3u + 1u] * region->center[1] +
                gradient[outputAxis * 3u + 2u] * region->center[2];
        } else if (descriptor.kind ==
                   PartitionedReductionModeKind::regionalAffineGradient) {
            const std::uint32_t outputAxis = descriptor.component / 3u;
            const std::uint32_t inputAxis = descriptor.component % 3u;
            q[mode] = gradient[outputAxis * 3u + inputAxis] *
                descriptor.scale;
        } else {
            const Position value = compiled_reduction_detail::affineValue(
                translation, gradient, positions[descriptor.anchorNode]);
            q[mode] = value[descriptor.component];
        }
    }
    std::vector<Position> displacement(basis.nodeCount);
    require(combinePartitionedPreconditionerModes(basis, q, displacement),
            "affine basis rejected its correctly sized mode coefficients");
    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
        const Position expected = node == fixed ? Position{} :
            compiled_reduction_detail::affineValue(
                translation, gradient, positions[node]);
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            require(near(displacement[node][axis], expected[axis]),
                    "anatomically partitioned B does not exactly reproduce an affine displacement");

    }

    for (std::uint32_t row = 0u; row < basis.modeCount; ++row) {
        for (std::uint32_t column = 0u; column < basis.modeCount; ++column) {
            double expected = 0.0;
            for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
                const auto& a = basis.values[
                    static_cast<std::size_t>(row) * basis.nodeCount + node];
                const auto& b = basis.values[
                    static_cast<std::size_t>(column) * basis.nodeCount + node];
                const double mass = world.fem.nodes[
                    output.globalFEMNodeIndices[node]].positionAndMass.w;
                expected += mass * (double(a.x) * b.x +
                    double(a.y) * b.y + double(a.z) * b.z);
            }
            require(near(basis.projectedMass[
                             static_cast<std::size_t>(row) * basis.modeCount +
                             column],
                         expected, 2.0e-12),
                    "projected mass is not the exact cooked-mass B^T M B");
        }
    }

    std::vector<Position> force(basis.nodeCount);
    for (std::size_t node = 0u; node < force.size(); ++node)
        force[node] = {0.13 * (node + 1u), -0.07 * (node + 2u),
                       0.11 * (node % 9u)};
    std::vector<double> reducedForce(basis.modeCount, 0.0);
    require(projectPartitionedNodalForce(basis, force, reducedForce),
            "B^T f rejected dimensionally correct compiled nodal force");
    double reducedWork = 0.0;
    double nodalWork = 0.0;
    for (std::uint32_t mode = 0u; mode < basis.modeCount; ++mode) {
        double expected = 0.0;
        for (std::uint32_t node = 0u; node < basis.nodeCount; ++node) {
            const auto& b = basis.values[
                static_cast<std::size_t>(mode) * basis.nodeCount + node];
            expected += b.x * force[node][0] + b.y * force[node][1] +
                b.z * force[node][2];
        }
        require(near(reducedForce[mode], expected, 2.0e-12),
                "compiled force projection differs from direct B^T f");
        reducedWork += q[mode] * reducedForce[mode];
    }
    for (std::uint32_t node = 0u; node < basis.nodeCount; ++node)
        for (std::uint32_t axis = 0u; axis < 3u; ++axis)
            nodalWork += displacement[node][axis] * force[node][axis];
    require(near(reducedWork, nodalWork, 2.0e-12),
            "compiled basis projection violates virtual-work equality");
}

void runSelectorZeroExplicitSourceLabelsCase() {
    using namespace numi::matter;
    const std::uint32_t attachment = nodeIndex(0u, 3u, 3u);
    const std::uint32_t contact = nodeIndex(2u, 1u, 1u);
    const std::uint32_t thorax = nodeIndex(4u, 0u, 0u);
    const std::uint32_t vessel = nodeIndex(1u, 0u, 1u);
    const std::uint32_t fixed = nodeIndex(4u, 3u, 3u);
    CompiledWorld world = makeCompiledWorld(
        attachment, contact, thorax, vessel, fixed);
    world.mixedSolver.executionBudgets.z =
        static_cast<std::uint32_t>(FEMPreconditionerMode::scalarDiagonal);
    world.physicsFingerprint = 0x9a36u;
    std::vector<std::uint32_t> sourceLabels;
    for (std::size_t tet = world.objects[1].elementOffset;
         tet < world.objects[1].elementOffset + world.objects[1].elementCount;
         ++tet) {
        NMTetrahedronGPU& cooked = world.fem.tetrahedra[tet];
        const std::uint32_t sourceX =
            (cooked.nodes.x - targetGlobalOffset) % nx;
        sourceLabels.push_back(24u + (sourceX < 2u ? 0u : 1u));
        cooked.identity.w &= ~NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK;
    }
    const std::array<std::uint32_t, 1> activeContactAnchors{
        targetGlobalOffset + contact};
    const std::array<std::uint32_t, 1> thoraxAnchors{
        targetGlobalOffset + thorax};
    CompiledWorldPartitionedReductionOptions options;
    options.activeContactAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(activeContactAnchors);
    options.thoraxAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(thoraxAnchors);
    options.explicitAuthoredTetrahedronRegionLabels =
        std::span<const std::uint32_t>(sourceLabels);
    options.explicitRegionLabelObjectIndex = 1u;
    options.explicitRegionLabelWorldPhysicsFingerprint =
        world.physicsFingerprint;
    const auto output = buildCompiledWorldPartitionedReductionBasis(
        world, 1u, options);
    require(output.valid(), output.error.empty() ? output.basis.error :
        output.error);
    require(output.regionLabelSource ==
                CompiledReductionRegionLabelSource::callerBoundSourcePayload &&
                output.tetrahedronRegionLabels == sourceLabels,
            "selector-0 source labels lost their explicit provenance");
}

void runRejectCase() {
    using namespace numi::matter;
    const std::uint32_t attachment = nodeIndex(0u, 3u, 3u);
    const std::uint32_t contact = nodeIndex(2u, 1u, 1u);
    const std::uint32_t thorax = nodeIndex(4u, 0u, 0u);
    const std::uint32_t vessel = nodeIndex(1u, 0u, 1u);
    const std::uint32_t fixed = nodeIndex(4u, 3u, 3u);
    CompiledWorld world = makeCompiledWorld(
        attachment, contact, thorax, vessel, fixed);
    world.mixedSolver.executionBudgets.z =
        static_cast<std::uint32_t>(FEMPreconditionerMode::scalarDiagonal);
    const auto output = buildCompiledWorldPartitionedReductionBasis(world, 1u);
    require(!output.valid() &&
                output.error.find("exact source labels bound") != std::string::npos,
            "selector zero admitted without an exact object- and world-bound source label map");
}

void runRankDeficientRegionPromotionCase() {
    using namespace numi::matter;
    const std::uint32_t attachment = nodeIndex(0u, 3u, 3u);
    const std::uint32_t contact = nodeIndex(2u, 1u, 1u);
    const std::uint32_t thorax = nodeIndex(4u, 0u, 0u);
    const std::uint32_t vessel = nodeIndex(1u, 0u, 1u);
    const std::uint32_t fixed = nodeIndex(4u, 3u, 3u);
    CompiledWorld world = makeCompiledWorld(
        attachment, contact, thorax, vessel, fixed);
    std::set<std::uint32_t> protectedAnchors{
        targetGlobalOffset + attachment,
        targetGlobalOffset + contact,
        targetGlobalOffset + thorax,
        targetGlobalOffset + vessel,
        targetGlobalOffset + nodeIndex(1u, 1u, 1u),
        targetGlobalOffset + nodeIndex(1u, 0u, 2u),
    };
    std::set<std::uint32_t> regionOneNodes;
    for (std::uint32_t tetIndex = world.objects[1].elementOffset;
         tetIndex < world.objects[1].elementOffset +
             world.objects[1].elementCount; ++tetIndex) {
        const NMTetrahedronGPU& tet = world.fem.tetrahedra[tetIndex];
        const std::uint32_t label = (tet.identity.w &
            NM_TETRAHEDRON_PRECONDITIONER_REGION_MASK) >>
            NM_TETRAHEDRON_PRECONDITIONER_REGION_SHIFT;
        if (label != 1u) continue;
        for (const std::uint32_t global :
             {tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w})
            regionOneNodes.insert(global);
    }
    std::vector<std::uint32_t> retainedRegionOneNodes;
    for (const std::uint32_t global : regionOneNodes) {
        if (protectedAnchors.contains(global) ||
            world.fem.nodes[global].restAndFixed.w == 1.0f)
            continue;
        retainedRegionOneNodes.push_back(global);
        if (retainedRegionOneNodes.size() == 3u) break;
    }
    require(retainedRegionOneNodes.size() == 3u,
            "rank-deficiency fixture could not retain three free regional nodes");
    std::set<std::uint32_t> retained(retainedRegionOneNodes.begin(),
                                     retainedRegionOneNodes.end());
    std::vector<std::uint32_t> selectedFineRows;
    for (const std::uint32_t global : regionOneNodes) {
        if (retained.contains(global) || protectedAnchors.contains(global) ||
            world.fem.nodes[global].restAndFixed.w == 1.0f)
            continue;
        selectedFineRows.push_back(global);
    }
    require(selectedFineRows.size() > 12u,
            "rank-deficiency fixture does not expose a meaningful fine promotion");

    const std::array<std::uint32_t, 1> contactAnchors{
        targetGlobalOffset + contact};
    const std::array<std::uint32_t, 1> thoraxAnchors{
        targetGlobalOffset + thorax};
    CompiledWorldPartitionedReductionOptions options;
    options.minimumCoordinateReductionFactor = 2u;
    options.activeContactAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(contactAnchors);
    options.thoraxAnchorGlobalFEMNodes =
        std::span<const std::uint32_t>(thoraxAnchors);
    options.fineCoordinateOnlyGlobalFEMNodes =
        std::span<const std::uint32_t>(selectedFineRows);

    const auto rejected = buildCompiledWorldPartitionedReductionBasis(
        world, 1u, options);
    require(!rejected.valid() &&
                rejected.error.find("rank deficient") != std::string::npos,
            "rank-deficient fine candidate did not fail closed with promotion disabled");

    options.promoteRankDeficientRegionsToFine = true;
    const auto promoted = buildCompiledWorldPartitionedReductionBasis(
        world, 1u, options);
    require(promoted.valid(), promoted.error.empty() ? promoted.basis.error : promoted.error);
    require(promoted.rankPromotedFineRegionLabels ==
                std::vector<std::uint32_t>{1u},
            "rank fallback did not retire exactly the deficient source region");
    require(promoted.basis.regions.size() == 1u &&
                promoted.basis.regions.front().sourceLabel == 0u,
            "rank fallback retained the unsupported regional affine block");
    require(promoted.rankPromotedFineGlobalFEMNodeIndices.size() > 12u,
            "rank fallback did not expose the promoted fine source rows");
    require(promoted.tetrahedronRegionLabels.size() ==
                world.objects[1].elementCount &&
                std::find(promoted.tetrahedronRegionLabels.begin(),
                          promoted.tetrahedronRegionLabels.end(), 1u) !=
                    promoted.tetrahedronRegionLabels.end(),
            "rank fallback changed the exact source tetrahedron labels or coverage");
    require(promoted.globalFEMNodeIndices.front() == targetGlobalOffset &&
                promoted.globalFEMNodeIndices.back() ==
                    targetGlobalOffset + sourceNodeCount - 1u &&
                promoted.basis.nodeCount == sourceNodeCount,
            "rank fallback changed source node identity or row coverage");
    for (const std::uint32_t global :
         promoted.rankPromotedFineGlobalFEMNodeIndices) {
        require(regionOneNodes.contains(global) &&
                    !protectedAnchors.contains(global),
                "rank fallback promoted a row outside its region or stole a protected anchor");
        const std::uint32_t local = global - targetGlobalOffset;
        for (std::uint32_t mode = 0u; mode < promoted.basis.modeCount; ++mode) {
            const auto& descriptor = promoted.basis.modes[mode];
            if (descriptor.kind == PartitionedReductionModeKind::anchorTranslation)
                continue;
            const auto& value = promoted.basis.values[
                static_cast<std::size_t>(mode) * promoted.basis.nodeCount + local];
            require(value.x == 0.0f && value.y == 0.0f && value.z == 0.0f,
                    "promoted fine row still contributes to a reduced affine mode");
        }
    }
    require(std::uint64_t(promoted.basis.modeCount) * 2u <=
                promoted.basis.freeCoordinateCount,
            "rank fallback silently exceeded the requested fine/reduction budget");

    options.minimumCoordinateReductionFactor = 4u;
    const auto overBudget = buildCompiledWorldPartitionedReductionBasis(
        world, 1u, options);
    require(!overBudget.valid(),
            "rank fallback silently weakened the requested coordinate-reduction target");
}
} // namespace

int main() {
    runValidCase();
    runSelectorZeroExplicitSourceLabelsCase();
    runRejectCase();
    runRankDeficientRegionPromotionCase();
    std::cout << "compiled world partitioned reduction: compiled ancestry, fine anchors, "
                 "affine reproduction, B^T M B, B^T f, virtual work, and fail-closed "
                 "source labels passed\n";
    return 0;
}
