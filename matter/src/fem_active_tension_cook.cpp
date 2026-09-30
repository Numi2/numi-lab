#include "numi/matter/matter.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace numi::matter {

bool cookFEMActiveTensions(const WorldSource& source,
                           const CompiledWorld& cooked,
                           std::span<const float> sourceTensions,
                           std::vector<float>& cookedTensions,
                           std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (source.environmentCount != cooked.dispatch.environmentCount ||
        source.objects.size() != cooked.objects.size() ||
        source.objects.size() != cooked.dispatch.objectCount ||
        cooked.fem.tetrahedra.size() != cooked.dispatch.tetrahedronCount)
        return fail("source/cooked FEM world dimensions differ");

    std::uint64_t authoredPerEnvironment = 0u;
    for (const ObjectSource& object : source.objects) {
        if (object.representation != Representation::fem) continue;
        if (object.tetrahedra.size() >
            std::numeric_limits<std::uint32_t>::max() - authoredPerEnvironment)
            return fail("authored FEM tetrahedron count exceeds the cooked ABI");
        authoredPerEnvironment += object.tetrahedra.size();
    }
    const std::uint64_t sourceCount = authoredPerEnvironment *
        source.environmentCount;
    const std::uint64_t cookedCount =
        std::uint64_t(cooked.dispatch.tetrahedronCount) *
        source.environmentCount;
    if (sourceCount > std::numeric_limits<std::size_t>::max() ||
        cookedCount > std::numeric_limits<std::size_t>::max() ||
        sourceTensions.size() != sourceCount)
        return fail("active tension must cover every authored FEM tetrahedron in every environment");

    std::vector<float> mapped(static_cast<std::size_t>(cookedCount), 0.0f);
    std::vector<bool> seen(cooked.fem.tetrahedra.size(), false);
    std::size_t authoredOffset = 0u;
    for (std::size_t objectIndex = 0u; objectIndex < source.objects.size();
         ++objectIndex) {
        const ObjectSource& object = source.objects[objectIndex];
        if (object.representation != Representation::fem) continue;
        const NMContinuumObjectGPU& descriptor = cooked.objects[objectIndex];
        if (descriptor.representation != NM_REPRESENTATION_FEM ||
            ((descriptor.flags & NM_OBJECT_MIXED_FEM) != 0u) != object.mixedFEM ||
            descriptor.elementCount < object.tetrahedra.size() ||
            std::uint64_t(descriptor.elementOffset) + descriptor.elementCount >
                cooked.fem.tetrahedra.size() ||
            (!object.femMaterialIndices.empty() &&
             object.femMaterialIndices.size() != object.tetrahedra.size()))
            return fail("source/cooked FEM object layout differs");

        for (std::size_t local = 0u; local < object.tetrahedra.size(); ++local) {
            const std::size_t cookedIndex = descriptor.elementOffset + local;
            const NMTetrahedronGPU& tetrahedron =
                cooked.fem.tetrahedra[cookedIndex];
            const std::uint32_t materialIndex =
                object.femMaterialIndices.empty()
                    ? object.materialIndex : object.femMaterialIndices[local];
            if (seen[cookedIndex] || materialIndex >= cooked.mixedMaterials.size() ||
                tetrahedron.identity.x != materialIndex ||
                tetrahedron.identity.y != objectIndex ||
                (tetrahedron.identity.w & NM_OBJECT_ACTIVE) == 0u)
                return fail("source/cooked FEM tetrahedron identity differs");
            const TetrahedronSource& sourceTet = object.tetrahedra[local];
            const std::uint32_t cookedNodes[4] = {
                tetrahedron.nodes.x, tetrahedron.nodes.y,
                tetrahedron.nodes.z, tetrahedron.nodes.w};
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                if (sourceTet.nodes[corner] >= object.femNodes.size() ||
                    std::uint64_t(descriptor.stateOffset) +
                        sourceTet.nodes[corner] != cookedNodes[corner])
                    return fail("source/cooked FEM tetrahedron node order differs");
            }
            seen[cookedIndex] = true;
            const float maximum = cooked.mixedMaterials[materialIndex].fibre.w;
            if (!std::isfinite(maximum) || maximum < 0.0f)
                return fail("cooked FEM active-tension material bound is invalid");
            for (std::size_t environment = 0u;
                 environment < source.environmentCount; ++environment) {
                const float tension = sourceTensions[
                    environment * static_cast<std::size_t>(authoredPerEnvironment) +
                    authoredOffset + local];
                if (!std::isfinite(tension) || tension < 0.0f ||
                    tension > maximum ||
                    (tension > 0.0f && object.mixedFEM))
                    return fail("authored FEM active tension is invalid for its cooked material");
                mapped[environment * cooked.fem.tetrahedra.size() +
                       cookedIndex] = tension;
            }
        }
        for (std::size_t local = object.tetrahedra.size();
             local < descriptor.elementCount; ++local) {
            const std::size_t cookedIndex = descriptor.elementOffset + local;
            if (seen[cookedIndex] ||
                (cooked.fem.tetrahedra[cookedIndex].identity.w &
                 NM_OBJECT_ACTIVE) != 0u)
                return fail("dormant FEM capacity slot is active or shared");
            seen[cookedIndex] = true;
        }
        authoredOffset += object.tetrahedra.size();
    }
    for (std::size_t index = 0u; index < seen.size(); ++index) {
        if (!seen[index] ||
            ((cooked.fem.tetrahedra[index].identity.w & NM_OBJECT_ACTIVE) != 0u &&
             cooked.fem.tetrahedra[index].identity.y >= source.objects.size()))
            return fail("cooked FEM tetrahedron has no source owner");
    }
    if (error != nullptr) error->clear();
    cookedTensions.swap(mapped);
    return true;
}

} // namespace numi::matter
