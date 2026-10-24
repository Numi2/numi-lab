#pragma once

#include "numi/matter/matter.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::matter::human {

inline constexpr std::uint32_t kMaterialColumnFlagShift = 10u;
inline constexpr std::uint32_t kMaterialColumnIdMask = 0x003fffffu;
inline constexpr std::uint32_t kTopologyLowFlagMask = 0x000001ffu;

struct MaterialColumnSelfContactAudit {
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    std::uint32_t nodeCount = 0u;
    std::uint32_t columnCount = 0u;
    std::uint64_t priorPhysicsFingerprint = 0u;
    std::uint64_t physicsFingerprint = 0u;
    std::uint64_t compiledFingerprint = 0u;
};

[[nodiscard]] inline std::uint32_t materialColumnTag(
    const std::uint32_t flags
) noexcept {
    return flags >> kMaterialColumnFlagShift;
}

[[nodiscard]] inline bool sameMaterialColumnAdjacency(
    const std::uint32_t leftObject,
    const std::uint32_t leftFlags,
    const std::uint32_t rightObject,
    const std::uint32_t rightFlags
) noexcept {
    const auto left = materialColumnTag(leftFlags);
    return leftObject == rightObject && left != 0u &&
        left == materialColumnTag(rightFlags);
}

// Zero is reserved for unpaired interior nodes. Only the three explicitly
// matched boundary roles may share a material-column contact exclusion.
inline void validateExplicitMaterialColumns(std::span<const std::uint32_t> columns,
        std::size_t nodeCount,std::uint32_t columnCount) {
    if(nodeCount>std::numeric_limits<std::uint32_t>::max() ||
       columns.size()!=nodeCount || columnCount==0u ||
       columnCount>kMaterialColumnIdMask)
        throw std::invalid_argument("explicit material-column range or count is invalid");
    std::vector<unsigned> uses(columnCount,0u);
    for(auto column:columns) {
        if(column>columnCount)
            throw std::invalid_argument("explicit material-column ID exceeds registered surface rows");
        if(column)++uses[column-1u];
    }
    for(auto count:uses)if(count!=3u)
        throw std::invalid_argument("explicit material column does not have exactly three boundary roles");
}

inline void validateMaterialColumnSource(
    const ObjectSource& source,
    const std::uint32_t nodesPerLayer,
    const std::span<const std::uint32_t> explicitColumns = {}
) {
    const auto fail = [](const char* why) -> void {
        throw std::invalid_argument(why);
    };
    if (nodesPerLayer == 0u || nodesPerLayer > kMaterialColumnIdMask)
        fail("three-layer shell column count is zero or exceeds 22-bit tag capacity");
    const std::uint64_t expectedNodes =
        explicitColumns.empty()?3ull*static_cast<std::uint64_t>(nodesPerLayer)
                               :static_cast<std::uint64_t>(explicitColumns.size());
    if (expectedNodes > std::numeric_limits<std::uint32_t>::max() ||
        source.femNodes.size() != expectedNodes ||
        source.tetrahedra.empty())
        fail("material-column binding requires exact authored volume nodes and nonempty tetrahedra");
    if(!explicitColumns.empty())
        validateExplicitMaterialColumns(explicitColumns,source.femNodes.size(),nodesPerLayer);
    if (source.representation != Representation::fem ||
        source.automaticRepresentation || source.adaptive || source.mixedFEM ||
        source.deformableContact == false || source.deformableSelfContact ||
        source.mutationPolicy.enabled || !source.mutationCommands.empty() ||
        source.femCapacity.mutationCommands != 0u ||
        source.femCapacity.cohesiveFaces != 0u ||
        source.femCapacity.punctureChannels != 0u ||
        !source.femCohesiveInterfaces.empty() ||
        !source.femCohesiveMortarPatches.empty() ||
        !source.femHumanAttachments.empty())
        fail("material-column binding requires an explicit immutable, contact-enabled, cohesive-free FEM shell with self-contact initially disabled");
}

// Preflight the complete local range before writing any flag, so a bad node
// identity cannot leave a partially tagged topology array.
inline void tagMaterialColumnNodes(
    const std::span<NMFEMTopologyNodeGPU> nodes,
    const std::uint32_t objectIndex,
    const std::uint32_t nodesPerLayer,
    const std::span<const std::uint32_t> explicitColumns = {}
) {
    if (nodesPerLayer == 0u || nodesPerLayer > kMaterialColumnIdMask ||
        (explicitColumns.empty()&&std::uint64_t(nodesPerLayer)*3u!=nodes.size()))
        throw std::invalid_argument("material-column node range does not match three-layer bounds");
    if(!explicitColumns.empty())
        validateExplicitMaterialColumns(explicitColumns,nodes.size(),nodesPerLayer);
    for (std::uint32_t local = 0u; local < nodes.size(); ++local) {
        const auto& identity = nodes[local].identity;
        if (identity.x != local || identity.y != objectIndex || identity.z != 1u ||
            identity.w != NM_TOPOLOGY_ACTIVE)
            throw std::invalid_argument("material-column topology identity is not the original active immutable source node");
    }
    for (std::uint32_t local = 0u; local < nodes.size(); ++local) {
        const std::uint32_t encodedColumn = explicitColumns.empty()
            ? local%nodesPerLayer+1u : explicitColumns[local];
        nodes[local].identity.w =
            (nodes[local].identity.w & kTopologyLowFlagMask) |
            (encodedColumn << kMaterialColumnFlagShift);
    }
}

[[nodiscard]] inline std::uint64_t materialColumnPhysicsFingerprint(
    const std::uint64_t sourceFingerprint,
    const std::uint32_t objectIndex,
    const std::uint32_t nodesPerLayer,
    const std::span<const std::uint32_t> explicitColumns = {}
) noexcept {
    // Extend the canonical source identity with this postcompile contact
    // specialization. The package fingerprint separately hashes every tag.
    constexpr char marker[] = "numi-human-three-layer-material-column-self-contact-v1";
    std::uint64_t hash = 1469598103934665603ull;
    const auto add = [&hash](const std::uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ull;
    };
    for (unsigned shift = 0u; shift < 64u; shift += 8u)
        add(static_cast<std::uint8_t>(sourceFingerprint >> shift));
    for (const char byte : marker) add(static_cast<std::uint8_t>(byte));
    for (unsigned shift = 0u; shift < 32u; shift += 8u) {
        add(static_cast<std::uint8_t>(objectIndex >> shift));
        add(static_cast<std::uint8_t>(nodesPerLayer >> shift));
    }
    if(!explicitColumns.empty()) {
        constexpr char explicitMarker[]="explicit-volume-column-rows-v1";
        for(const char byte:explicitMarker)add(static_cast<std::uint8_t>(byte));
        for(const auto column:explicitColumns)
            for(unsigned shift=0u;shift<32u;shift+=8u)
                add(static_cast<std::uint8_t>(column>>shift));
    }
    return hash == 0u ? 1u : hash;
}

[[nodiscard]] inline MaterialColumnSelfContactAudit
bindHumanThreeLayerMaterialColumnSelfContact(
    CompiledWorld& world,
    const ObjectSource& source,
    const std::uint32_t objectIndex,
    const std::uint32_t nodesPerLayer,
    const std::span<const std::uint32_t> explicitColumns = {}
) {
    validateMaterialColumnSource(source, nodesPerLayer, explicitColumns);
    const auto fail = [](const char* why) -> void {
        throw std::invalid_argument(why);
    };
    if (world.fingerprint == 0u ||
        world.fingerprint != compiledWorldFingerprint(world))
        fail("material-column binding requires the untouched compiled-world fingerprint");
    std::string layoutError;
    if (!validateCompiledWorldLayout(world, &layoutError))
        throw std::invalid_argument(
            "material-column binding requires a valid compiled world: " + layoutError);
    if (objectIndex >= world.objects.size() ||
        objectIndex >= world.dispatch.objectCount)
        fail("material-column target object is outside compiled object bounds");
    const auto& descriptor = world.objects[objectIndex];
    const std::uint64_t expectedNodeCount =
        explicitColumns.empty()?3ull*static_cast<std::uint64_t>(nodesPerLayer)
                               :static_cast<std::uint64_t>(explicitColumns.size());
    if ((world.dispatch.flags & NM_MATTER_STATIC_FEM_SURFACE_MAP) == 0u ||
        (world.dispatch.flags & (NM_MATTER_ADAPTIVE | NM_MATTER_MUTATION)) != 0u ||
        world.dispatch.mutationCommandCount != 0u ||
        world.dispatch.femNodeCount != world.fem.nodes.size() ||
        world.fem.topologyNodes.size() != world.fem.nodes.size() ||
        world.dispatch.tetrahedronCount != world.fem.tetrahedra.size() ||
        world.dispatch.surfaceFaceCount != world.fem.surfaceFaces.size() ||
        descriptor.representation != NM_REPRESENTATION_FEM ||
        (descriptor.flags & NM_OBJECT_ACTIVE) == 0u ||
        (descriptor.flags & (NM_OBJECT_ADAPTIVE | NM_OBJECT_MUTABLE_TOPOLOGY |
                             NM_OBJECT_MIXED_FEM |
                             NM_OBJECT_DISABLE_DEFORMABLE_CONTACT)) != 0u ||
        (descriptor.flags & NM_OBJECT_DISABLE_SELF_CONTACT) == 0u ||
        descriptor.topologyGeneration != 1u ||
        descriptor.stateCount != expectedNodeCount ||
        descriptor.stateCount != source.femNodes.size() ||
        descriptor.elementCount != source.tetrahedra.size() ||
        descriptor.stateOffset > world.fem.topologyNodes.size() ||
        descriptor.stateCount > world.fem.topologyNodes.size() - descriptor.stateOffset ||
        descriptor.elementOffset > world.fem.tetrahedra.size() ||
        descriptor.elementCount > world.fem.tetrahedra.size() - descriptor.elementOffset)
        fail("compiled material-column target is not the expected active immutable three-layer FEM shell");
    if (!world.fem.mutationCommands.empty() ||
        !world.fem.punctureChannels.empty())
        fail("material-column binding rejects worlds with mutation or puncture authority");
    for (const auto& capacity : world.fem.capacities)
        if (capacity.work.x != 0u || capacity.topology.w != 0u)
            fail("material-column binding rejects reserved mutation or puncture capacity");
    for (std::uint32_t localTet = 0u; localTet < descriptor.elementCount; ++localTet)
        if (world.fem.tetrahedra[descriptor.elementOffset + localTet].identity.y != objectIndex)
            fail("compiled material-column shell tetrahedron ownership differs from source object");
    for (const auto& face : world.fem.cohesiveFaces)
        if (face.adjacency.y == objectIndex)
            fail("material-column shell contains a compiled cohesive face");
    for (const auto& attachment : world.fem.humanAttachments)
        if (attachment.identity.z == objectIndex)
            fail("material-column shell contains a Human attachment");

    auto targetNodes = std::span<NMFEMTopologyNodeGPU>(world.fem.topologyNodes)
        .subspan(descriptor.stateOffset, descriptor.stateCount);
    // The tagger preflights every source ID/object/generation/active flag before
    // changing any node. Save the only two fields this binding is allowed to edit.
    std::vector<std::uint32_t> previousFlags;
    previousFlags.reserve(targetNodes.size());
    for (const auto& node : targetNodes) previousFlags.push_back(node.identity.w);
    const std::uint32_t previousObjectFlags = descriptor.flags;
    const std::uint64_t previousPhysicsFingerprint = world.physicsFingerprint;
    tagMaterialColumnNodes(targetNodes, objectIndex, nodesPerLayer, explicitColumns);
    world.objects[objectIndex].flags &= ~NM_OBJECT_DISABLE_SELF_CONTACT;
    world.physicsFingerprint = materialColumnPhysicsFingerprint(
        previousPhysicsFingerprint, objectIndex, nodesPerLayer, explicitColumns);
    world.fingerprint = compiledWorldFingerprint(world);
    if (!validateCompiledWorldLayout(world, &layoutError)) {
        for (std::size_t local = 0u; local < targetNodes.size(); ++local)
            targetNodes[local].identity.w = previousFlags[local];
        world.objects[objectIndex].flags = previousObjectFlags;
        world.physicsFingerprint = previousPhysicsFingerprint;
        world.fingerprint = compiledWorldFingerprint(world);
        throw std::runtime_error(
            "material-column specialization failed canonical layout validation: " + layoutError);
    }
    return {objectIndex, descriptor.stateCount, nodesPerLayer,
            previousPhysicsFingerprint, world.physicsFingerprint, world.fingerprint};
}

} // namespace numi::matter::human
