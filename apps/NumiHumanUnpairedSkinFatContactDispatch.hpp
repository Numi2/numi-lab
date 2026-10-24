#pragma once

#include "NumiHumanMaterialColumnSelfContact.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace numi::human {

struct IndependentSkinFatCanonicalContactAudit {
    std::uint32_t objectIndex = NM_INVALID_INDEX;
    std::size_t volumeNodeCount = 0u;
    std::size_t tetrahedronCount = 0u;
    std::size_t expectedExteriorFaceCount = 0u;
    std::size_t compiledExteriorFaceCount = 0u;
    std::size_t expectedContactFaceCount = 0u;
    std::size_t compiledContactFaceCount = 0u;
};

template<class Labels>
[[nodiscard]] inline bool usesIndependentSkinFatCanonicalContact(const Labels* labels) noexcept {
    return labels != nullptr && labels->explicitUnpairedVolumeTopology;
}

template<class Labels>
[[nodiscard]] inline bool validateIndependentSkinFatCanonicalContact(
    const numi::matter::CompiledWorld& world,
    const numi::matter::ObjectSource& source,
    const Labels& labels,
    const std::uint32_t objectIndex,
    IndependentSkinFatCanonicalContactAudit& audit,
    std::string& error) {
    const auto fail = [&](const std::string& message) { error = message; return false; };
    audit = {};
    if (!labels.explicitUnpairedVolumeTopology)
        return fail("independent self-contact requires explicit unpaired owner metadata");
    if (objectIndex >= world.objects.size() || objectIndex >= world.dispatch.objectCount)
        return fail("independent self-contact object index is outside the compiled world");
    if (source.representation != numi::matter::Representation::fem ||
        !source.deformableContact || !source.deformableSelfContact ||
        source.tetrahedra.empty() || source.femNodes.empty())
        return fail("independent self-contact source is not a contact-enabled FEM owner");
    const auto& object = world.objects[objectIndex];
    if (object.representation != NM_REPRESENTATION_FEM ||
        (object.flags & NM_OBJECT_ACTIVE) == 0u ||
        (object.flags & (NM_OBJECT_DISABLE_DEFORMABLE_CONTACT | NM_OBJECT_DISABLE_SELF_CONTACT |
                         NM_OBJECT_ADAPTIVE | NM_OBJECT_MUTABLE_TOPOLOGY | NM_OBJECT_MIXED_FEM)) != 0u ||
        object.topologyGeneration != 1u || object.stateCount != source.femNodes.size() ||
        object.elementCount != source.tetrahedra.size())
        return fail("compiled independent self-contact object flags or ranges differ from the authored source");
    if (world.dispatch.femNodeCount != world.fem.nodes.size() ||
        world.dispatch.tetrahedronCount != world.fem.tetrahedra.size() ||
        world.dispatch.surfaceFaceCount != world.fem.surfaceFaces.size() ||
        world.fem.topologyNodes.size() != world.fem.nodes.size() ||
        object.stateOffset > world.fem.topologyNodes.size() ||
        object.stateCount > world.fem.topologyNodes.size() - object.stateOffset ||
        object.elementOffset > world.fem.tetrahedra.size() ||
        object.elementCount > world.fem.tetrahedra.size() - object.elementOffset)
        return fail("compiled independent self-contact surface map or FEM ranges are inconsistent");

    std::vector<std::uint8_t> roleNodeUse(source.femNodes.size(), 0u);
    for (std::size_t role = 0; role < 3u; ++role) {
        const auto& faces = labels.registeredLayerFaces[role];
        const auto& stable = labels.registeredLayerStableFaceIdentities[role];
        const auto& rows = labels.registeredLayerVolumeNodeRows[role];
        if (labels.registeredLayerVertexCounts[role] == 0u ||
            labels.registeredLayerFaceCounts[role] != faces.size() || faces.empty() ||
            stable.size() != faces.size() || rows.size() != labels.registeredLayerVertexCounts[role])
            return fail("independent self-contact role metadata has incomplete surface counts or identities");
        for (const auto stableId : stable)
            if (stableId == NM_INVALID_INDEX)
                return fail("independent self-contact role has an invalid stable source-face identity");
        for (const auto& face : faces)
            for (const auto vertex : face)
                if (vertex >= labels.registeredLayerVertexCounts[role])
                    return fail("independent self-contact role face references an out-of-range role vertex");
        for (const auto row : rows) {
            if (row >= source.femNodes.size() || roleNodeUse[row] != 0u)
                return fail("independent self-contact role-to-volume rows are out of range or aliased");
            roleNodeUse[row] = 1u;
        }
    }
    for (std::uint32_t local = 0u; local < object.stateCount; ++local) {
        const auto& identity = world.fem.topologyNodes[object.stateOffset + local].identity;
        if (identity.x != local || identity.y != objectIndex || identity.z != 1u ||
            numi::matter::human::materialColumnTag(identity.w) != 0u)
            return fail("independent self-contact topology has stale identities or legacy material-column tags");
    }

    using FaceKey = std::array<std::uint32_t, 3>;
    struct SourceFacet { std::uint32_t uses = 0u; std::uint32_t sourceFace = NM_INVALID_INDEX; };
    std::map<FaceKey, SourceFacet> sourceFacets;
    if (source.tetrahedra.size() > static_cast<std::size_t>(UINT32_MAX / 4u))
        return fail("independent self-contact tetrahedron face identities overflow");
    for (std::size_t tetRow = 0u; tetRow < source.tetrahedra.size(); ++tetRow) {
        const auto& tet = source.tetrahedra[tetRow].nodes;
        std::array<std::uint32_t, 4> uniqueNodes{tet[0], tet[1], tet[2], tet[3]};
        std::sort(uniqueNodes.begin(), uniqueNodes.end());
        if (uniqueNodes[0] == uniqueNodes[1] || uniqueNodes[1] == uniqueNodes[2] ||
            uniqueNodes[2] == uniqueNodes[3] || uniqueNodes[3] >= source.femNodes.size())
            return fail("independent self-contact source tetrahedron has invalid node IDs");
        for (std::uint32_t omitted = 0u; omitted < 4u; ++omitted) {
            FaceKey key{}; std::size_t at = 0u;
            for (std::uint32_t corner = 0u; corner < 4u; ++corner)
                if (corner != omitted) key[at++] = tet[corner];
            std::sort(key.begin(), key.end());
            auto& facet = sourceFacets[key];
            ++facet.uses;
            if (facet.uses == 1u)
                facet.sourceFace = static_cast<std::uint32_t>(4u * tetRow + omitted);
            if (facet.uses > 2u)
                return fail("independent self-contact source has a non-manifold tetrahedron facet");
        }
    }
    auto disabled = source.femDisabledContactSurfaceFaces;
    std::sort(disabled.begin(), disabled.end());
    if (std::adjacent_find(disabled.begin(), disabled.end()) != disabled.end() ||
        (!disabled.empty() && disabled.back() >= 4u * source.tetrahedra.size()))
        return fail("independent self-contact source has invalid disabled-face identities");
    std::size_t expectedExterior = 0u, expectedContact = 0u;
    std::set<FaceKey> expectedKeys;
    for (const auto& [key, facet] : sourceFacets) {
        if (facet.uses != 1u) continue;
        ++expectedExterior;
        expectedKeys.insert(key);
        if (!std::binary_search(disabled.begin(), disabled.end(), facet.sourceFace)) ++expectedContact;
    }

    std::size_t compiledExterior = 0u, compiledContact = 0u;
    std::set<FaceKey> compiledKeys;
    for (const auto& face : world.fem.surfaceFaces) {
        if (face.adjacency.z != objectIndex || face.adjacency.y != NM_INVALID_INDEX) continue;
        ++compiledExterior;
        if (face.adjacency.x < object.elementOffset ||
            face.adjacency.x - object.elementOffset >= object.elementCount || face.sides.x >= 4u)
            return fail("compiled independent self-contact exterior face has invalid source tetrahedron identity");
        const auto& tet = world.fem.tetrahedra[face.adjacency.x];
        if (tet.identity.y != objectIndex)
            return fail("compiled independent self-contact exterior face belongs to another object");
        const std::array<std::uint32_t, 4> nodes{tet.nodes.x, tet.nodes.y, tet.nodes.z, tet.nodes.w};
        FaceKey key{}; std::size_t at = 0u;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
            if (corner == face.sides.x) continue;
            const auto globalNode = nodes[corner];
            if (globalNode < object.stateOffset || globalNode - object.stateOffset >= object.stateCount)
                return fail("compiled independent self-contact face node is outside its object range");
            key[at++] = globalNode - object.stateOffset;
        }
        std::sort(key.begin(), key.end());
        if (!expectedKeys.contains(key) || !compiledKeys.insert(key).second)
            return fail("compiled independent self-contact exterior face differs from source tetrahedron boundary");
        const bool expectedDisabled = std::binary_search(
            disabled.begin(), disabled.end(), 4u * (face.adjacency.x - object.elementOffset) + face.sides.x);
        const bool compiledDisabled = (face.sides.w & NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT) != 0u;
        if (expectedDisabled != compiledDisabled)
            return fail("compiled independent self-contact face contact flag differs from authored face policy");
        if (!compiledDisabled) ++compiledContact;
    }
    if (compiledExterior != expectedExterior || compiledKeys != expectedKeys ||
        compiledContact != expectedContact || compiledContact == 0u)
        return fail("compiled independent self-contact exterior/contact face counts differ from source incidence");

    audit.objectIndex = objectIndex;
    audit.volumeNodeCount = object.stateCount;
    audit.tetrahedronCount = object.elementCount;
    audit.expectedExteriorFaceCount = expectedExterior;
    audit.compiledExteriorFaceCount = compiledExterior;
    audit.expectedContactFaceCount = expectedContact;
    audit.compiledContactFaceCount = compiledContact;
    error.clear();
    return true;
}

} // namespace numi::human
