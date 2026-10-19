#pragma once

#include "NumiHumanRestingAorticWall.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace numi::human::vessel_wall_coupling {
using namespace numi::matter;
using aortic_wall::Built;
using aortic_wall::Config;
using aortic_wall::SectionProof;
using aortic_wall::Vec;

inline constexpr std::uint32_t abdominalAortaCompartmentStableId = 7u;
inline constexpr std::uint32_t abdominalAortaWallStableId = 9007u;
inline constexpr std::string_view abdominalAortaAnatomicalId =
    "source_aggregate:CVSim21:abdominal_aorta";
inline constexpr std::string_view abdominalAortaContourSHA256 =
    "1a191fe2f6c4a1cb03ee290bc850b80011cc267577992ea86abb3a4cb39f4108";

inline SectionProof pinnedAbdominalAortaSectionProof() {
    SectionProof proof;
    proof.stableId = 9u;
    proof.sourceBody = 7u;
    proof.sourcePayloadSHA256 = std::string(abdominalAortaContourSHA256);
    proof.planePoint = {-0.03372473683003, 0.05810004997253, -0.00946755672337};
    proof.planeNormal = {0.0, 1.0, 0.0};
    proof.externalAreaM2 = 2.4192601236904974e-4;
#include "actual_nha_contour.inc"
    if (proof.orderedExternalContour.size() != 87u) {
        throw std::runtime_error("pinned abdominal-aorta contour must contain 87 authored points");
    }
    return proof;
}

inline const VascularCompartmentSource& requireRegisteredAbdominalAorta(
    const WorldSource& source) {
    const auto& vascular = source.vascular;
    if (vascular.compartments.size() != 21u || vascular.connections.size() != 24u) {
        throw std::runtime_error(
            "aortic wall binding requires the registered 21-compartment/24-connection CVSim21 graph");
    }
    const auto registeredCount = std::count_if(vascular.compartments.begin(),
        vascular.compartments.end(), [](const VascularCompartmentSource& compartment) {
            return compartment.stableIdentifier == abdominalAortaCompartmentStableId;
        });
    const auto found = std::find_if(vascular.compartments.begin(), vascular.compartments.end(),
        [](const VascularCompartmentSource& compartment) {
            return compartment.stableIdentifier == abdominalAortaCompartmentStableId;
        });
    if (registeredCount != 1 || found == vascular.compartments.end() ||
        found->anatomicalIdentifier != abdominalAortaAnatomicalId ||
        found->storageKind != VascularStorageKind::absoluteVolume ||
        found->pressureLaw != VascularPressureLaw::linearCompliance ||
        !(found->compliance > 0.0) || !(found->initialVolume > 0.0) ||
        !std::isfinite(found->initialVolume) ||
        found->initialSpeciesAmounts.size() != vascular.species.size()) {
        throw std::runtime_error(
            "stable CVSim21 abdominal-aorta registration is absent or has an unsupported volume law");
    }
    const auto anyIdentity = [](const auto& words) {
        return std::any_of(words.begin(), words.end(), [](std::uint64_t word) {
            return word != 0u;
        });
    };
    if (!anyIdentity(vascular.contentIdentity) || !anyIdentity(vascular.sourceIdentity)) {
        throw std::runtime_error(
            "aortic wall binding requires the authored CVSim21 graph provenance");
    }
    return *found;
}

inline Built appendRegisteredAbdominalAortaWall(
    WorldSource& source,
    const MRBodyStateGPU& attachmentPose,
    const Vec& attachmentCOM,
    std::uint32_t attachmentBody,
    const MRBodyStateGPU& donorPose,
    std::uint32_t donorBody,
    const Config& config = {}) {
    const auto& registered = requireRegisteredAbdominalAorta(source);
    const double initialVolume = registered.initialVolume;
    const std::vector<double> initialSpecies = registered.initialSpeciesAmounts;
    const std::size_t materialCount = source.materials.size();
    const std::size_t objectCount = source.objects.size();
    const std::size_t cavityCount = source.vascular.cavities.size();
    if (std::any_of(source.vascular.cavities.begin(), source.vascular.cavities.end(),
        [](const VascularCavitySource& cavity) {
            return cavity.stableIdentifier == abdominalAortaWallStableId;
        })) {
        throw std::runtime_error("pinned aortic wall cavity identity is already in use");
    }

    // The existing CV row’s initialVolume sets the cavity reference geometry.
    // appendToWorld changes that same row’s constitutive law; no second volume
    // or species state is introduced. The before/after checks enforce this.
    const SectionProof section = pinnedAbdominalAortaSectionProof();
    Built wall = aortic_wall::appendToWorld(source, section, attachmentPose,
        attachmentCOM, attachmentBody, donorPose, donorBody, config);

    const auto after = std::find_if(source.vascular.compartments.begin(),
        source.vascular.compartments.end(), [](const VascularCompartmentSource& compartment) {
            return compartment.stableIdentifier == abdominalAortaCompartmentStableId;
        });
    if (after == source.vascular.compartments.end() ||
        after->anatomicalIdentifier != abdominalAortaAnatomicalId ||
        after->pressureLaw != VascularPressureLaw::deformingCavity ||
        after->initialVolume != initialVolume ||
        after->initialSpeciesAmounts != initialSpecies ||
        source.materials.size() != materialCount + 1u ||
        source.objects.size() != objectCount + 1u ||
        source.vascular.cavities.size() != cavityCount + 1u ||
        wall.cavity.stableIdentifier != abdominalAortaWallStableId ||
        wall.cavity.compartment != abdominalAortaCompartmentStableId ||
        wall.objectIndex != objectCount || wall.cavity.objectIndex != objectCount) {
        throw std::runtime_error(
            "aortic wall binding changed CV volume/species or failed one-to-one object registration");
    }
    return wall;
}

struct OuterLateralContactSurfaceMapAudit {
    std::uint32_t queryNodeCount=0u;
    std::uint32_t originalSurfaceFaceCount=0u;
    std::uint32_t canonicalSurfaceFaceCount=0u;
    std::uint32_t retainedWallFaceCount=0u;
    std::uint32_t excludedWallFaceCount=0u;
};

// Verify the source-authored contact eligibility in the final compiled world.
// This is read-only: preparation and live-runtime recooks must carry exactly
// the same wall face selection, including when an MPM receiver is appended.
inline OuterLateralContactSurfaceMapAudit auditOuterLateralContactSurfaceFaces(
    const CompiledWorld& world,
    const Built& wall,
    std::span<const std::uint32_t> queryNodes) {
    const auto fail=[](const char* why)->void { throw std::runtime_error(why); };
    if((world.dispatch.flags&NM_MATTER_STATIC_FEM_SURFACE_MAP)==0u ||
       world.dispatch.surfaceFaceCount!=world.fem.surfaceFaces.size())
        fail("outer aortic contact requires the compiled immutable FEM exterior-face map");
    if((world.dispatch.flags&(NM_MATTER_ADAPTIVE|NM_MATTER_MUTATION))!=0u ||
       world.dispatch.mutationCommandCount!=0u)
        fail("outer aortic contact requires immutable, nonadaptive FEM topology");
    if(wall.objectIndex>=world.objects.size())
        fail("outer aortic contact wall object index is outside the compiled world");
    const auto& descriptor=world.objects[wall.objectIndex];
    if(descriptor.representation!=NM_REPRESENTATION_FEM ||
       (descriptor.flags&NM_OBJECT_ADAPTIVE)!=0u ||
       (descriptor.flags&NM_OBJECT_MUTABLE_TOPOLOGY)!=0u ||
       (descriptor.flags&NM_OBJECT_DISABLE_DEFORMABLE_CONTACT)!=0u ||
       (descriptor.flags&NM_OBJECT_DISABLE_SELF_CONTACT)==0u ||
       descriptor.stateCount!=wall.object.femNodes.size() ||
       descriptor.elementCount!=wall.object.tetrahedra.size() ||
       descriptor.stateOffset>world.fem.nodes.size() ||
       descriptor.stateCount>world.fem.nodes.size()-descriptor.stateOffset ||
       descriptor.elementOffset>world.fem.tetrahedra.size() ||
       descriptor.elementCount>world.fem.tetrahedra.size()-descriptor.elementOffset)
        fail("compiled aortic wall does not match the exterior-contact owner contract");
    std::uint64_t nodeCapacity=0u,tetCapacity=0u,mutationCapacity=0u;
    for(const auto& capacity:world.fem.capacities){
        nodeCapacity+=capacity.topology.x;tetCapacity+=capacity.topology.y;
        mutationCapacity+=capacity.work.x;
    }
    if(nodeCapacity!=world.dispatch.femNodeCount ||
       tetCapacity!=world.dispatch.tetrahedronCount || mutationCapacity!=0u ||
       world.dispatch.tetrahedronCount!=world.fem.tetrahedra.size())
        fail("outer aortic contact requires exact immutable FEM node/tet capacities");

    const std::uint64_t expectedNodes=(std::uint64_t(wall.axialSegments)+1u)*wall.circumferentialSegments;
    const std::uint64_t expectedWallFaces=2ull*wall.axialSegments*wall.circumferentialSegments;
    if(expectedNodes==0u || expectedNodes>std::numeric_limits<std::uint32_t>::max() ||
       expectedWallFaces>std::numeric_limits<std::uint32_t>::max() ||
       queryNodes.size()!=expectedNodes)
        fail("outer aortic contact node/face mask count differs from authored wall rings");
    std::vector<std::uint8_t> queryMask(descriptor.stateCount,0u);
    for(const auto local:queryNodes){
        if(local>=descriptor.stateCount || queryMask[local]!=0u)
            fail("outer aortic contact query-node mask is duplicate or out of range");
        queryMask[local]=1u;
    }

    OuterLateralContactSurfaceMapAudit audit;
    audit.queryNodeCount=static_cast<std::uint32_t>(queryNodes.size());
    audit.originalSurfaceFaceCount=static_cast<std::uint32_t>(world.fem.surfaceFaces.size());
    const std::uint64_t sourceFaceDomain=std::uint64_t(world.dispatch.tetrahedronCount)*4u;
    if(sourceFaceDomain>std::numeric_limits<std::size_t>::max())
        fail("outer aortic contact source-face domain overflows host indexing");
    std::vector<std::uint8_t> sourceFaceSeen(static_cast<std::size_t>(sourceFaceDomain),0u);
    std::vector<std::vector<std::uint8_t>> stableMapIdSeen(world.objects.size());
    for(std::size_t object=0u;object<world.objects.size();++object){
        const std::uint64_t domain=4ull*world.objects[object].elementCount;
        if(domain>std::numeric_limits<std::size_t>::max())
            fail("outer aortic contact per-object face-map domain overflows host indexing");
        stableMapIdSeen[object].assign(static_cast<std::size_t>(domain),0u);
    }
    std::vector<std::uint8_t> retainedNodeSeen(descriptor.stateCount,0u);
    std::uint32_t wallExteriorFaceCount=0u;
    for(const auto& face:world.fem.surfaceFaces){
        const auto tetIndex=face.adjacency.x;
        const auto opposite=face.sides.x;
        if(face.adjacency.y!=NM_INVALID_INDEX || tetIndex>=world.fem.tetrahedra.size() || opposite>=4u)
            fail("compiled static FEM map contains a non-exterior or invalid target face");
        const auto& tet=world.fem.tetrahedra[tetIndex];
        const std::array<std::uint32_t,4u> corners{tet.nodes.x,tet.nodes.y,tet.nodes.z,tet.nodes.w};
        if(face.adjacency.z>=world.objects.size() || tet.identity.y!=face.adjacency.z)
            fail("compiled surface target face object lineage is inconsistent");
        const std::uint64_t stableFace=4ull*tetIndex+opposite;
        if(stableFace>=sourceFaceSeen.size() ||
           sourceFaceSeen[static_cast<std::size_t>(stableFace)]!=0u ||
           face.sides.z>=stableMapIdSeen[face.adjacency.z].size() ||
           stableMapIdSeen[face.adjacency.z][face.sides.z]!=0u)
            fail("compiled static surface source-face or per-object stable-map identity is invalid or duplicated");
        sourceFaceSeen[static_cast<std::size_t>(stableFace)]=1u;
        stableMapIdSeen[face.adjacency.z][face.sides.z]=1u;
        if((face.sides.w&~NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT)!=0u)
            fail("compiled static surface map contains an unknown contact-face flag");
        if(face.adjacency.z!=wall.objectIndex){
            if(face.sides.w!=0u)
                fail("outer aortic contact found a preflagged nonwall surface face");
            continue;
        }
        ++wallExteriorFaceCount;
        if(tetIndex<descriptor.elementOffset ||
           tetIndex>=std::uint64_t(descriptor.elementOffset)+descriptor.elementCount)
            fail("aortic target face tet lies outside its registered source object");
        bool outerLateral=true;
        std::array<std::uint32_t,3u> triangle{};std::size_t out=0u;
        for(unsigned corner=0u;corner<4u;++corner)if(corner!=opposite){
            if(corners[corner]<descriptor.stateOffset ||
               corners[corner]>=std::uint64_t(descriptor.stateOffset)+descriptor.stateCount)
                fail("aortic target face node escapes its registered wall range");
            const auto local=corners[corner]-descriptor.stateOffset;
            triangle[out++]=local;
            outerLateral=outerLateral&&(queryMask[local]!=0u);
        }
        const auto expectedFlag=outerLateral?0u:NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT;
        if(face.sides.w!=expectedFlag)
            fail("compiled aortic contact-face flags differ from the authored outer surface");
        const auto localFace=static_cast<std::uint32_t>(
            4ull*(tetIndex-descriptor.elementOffset)+opposite);
        if(std::binary_search(wall.object.femDisabledContactSurfaceFaces.begin(),
                              wall.object.femDisabledContactSurfaceFaces.end(),localFace)
                ==outerLateral)
            fail("aortic source contact-face identity differs from compiled topology");
        if(outerLateral){
            for(const auto local:triangle)retainedNodeSeen[local]=1u;
            ++audit.retainedWallFaceCount;
        } else {
            ++audit.excludedWallFaceCount;
        }
    }
    if(audit.retainedWallFaceCount!=expectedWallFaces ||
       audit.retainedWallFaceCount+audit.excludedWallFaceCount!=wallExteriorFaceCount ||
       audit.excludedWallFaceCount==0u ||
       std::any_of(queryNodes.begin(),queryNodes.end(),[&](const auto local){return retainedNodeSeen[local]==0u;}))
        fail("outer aortic contact did not flag every nonouter wall boundary face while retaining the annular lateral surface");
    if(world.fem.surfaceFaces.size()!=audit.originalSurfaceFaceCount ||
       world.fem.surfaceFaces.empty() ||
       world.fem.surfaceFaces.size()>std::numeric_limits<std::uint32_t>::max())
        fail("outer aortic contact changed or overflowed the complete canonical exterior map");
    audit.canonicalSurfaceFaceCount=static_cast<std::uint32_t>(world.fem.surfaceFaces.size());
    if(world.fingerprint!=compiledWorldFingerprint(world))
        fail("compiled aortic contact-face flags do not match the canonical world fingerprint");
    std::string layoutError;
    if(!validateCompiledWorldLayout(world,&layoutError))
        throw std::runtime_error("flagged complete aortic static contact map failed canonical layout validation: "+layoutError);
    return audit;
}

inline void requireCompiledAbdominalAortaBinding(
    const CompiledWorld& world,
    const Built& wall) {
    const auto& vascular = world.vascular;
    const auto compiledRegisteredCount = std::count_if(vascular.compartments.begin(),
        vascular.compartments.end(), [](const NMVascularCompartmentGPU& value) {
            return value.identity.x == abdominalAortaCompartmentStableId;
        });
    const auto compartment = std::find_if(vascular.compartments.begin(), vascular.compartments.end(),
        [](const NMVascularCompartmentGPU& value) {
            return value.identity.x == abdominalAortaCompartmentStableId;
        });
    if (compiledRegisteredCount != 1 || compartment == vascular.compartments.end()) {
        throw std::runtime_error("compiled CVSim21 graph lost the registered abdominal-aorta row");
    }
    const std::size_t compartmentIndex =
        static_cast<std::size_t>(compartment - vascular.compartments.begin());
    if (compartmentIndex >= vascular.compartmentCavity.size()) {
        throw std::runtime_error("compiled abdominal-aorta cavity map is truncated");
    }
    const std::uint32_t cavityIndex = vascular.compartmentCavity[compartmentIndex];
    if (cavityIndex >= vascular.cavities.size()) {
        throw std::runtime_error("compiled abdominal-aorta row has no registered wall cavity");
    }
    const auto& cavity = vascular.cavities[cavityIndex];
    if (cavity.identity.x != abdominalAortaWallStableId ||
        cavity.identity.y != compartmentIndex ||
        cavity.identity.z != wall.objectIndex ||
        cavity.faces.y != wall.cavity.faces.size() ||
        cavity.identity.w >= vascular.unknowns.size() ||
        compartment->identity.w != static_cast<std::uint32_t>(VascularPressureLaw::deformingCavity) ||
        wall.objectIndex >= world.objects.size() ||
        world.objects[wall.objectIndex].stateCount != wall.object.femNodes.size() ||
        cavity.faces.x > vascular.cavityFaces.size() ||
        cavity.faces.y > vascular.cavityFaces.size() - cavity.faces.x) {
        throw std::runtime_error(
            "compiled abdominal-aorta pressure row, FEM wall, or cavity faces are not bound one-to-one");
    }
    if (!std::equal(std::begin(cavity.sourceIdentity), std::end(cavity.sourceIdentity),
                    wall.cavity.sourceIdentity.begin()) ||
        !std::equal(std::begin(cavity.mechanicalIdentity), std::end(cavity.mechanicalIdentity),
                    wall.cavity.mechanicalIdentity.begin())) {
        throw std::runtime_error("compiled aortic wall provenance differs from the registered source");
    }
    std::size_t matching = 0u;
    for (const auto& candidate : vascular.cavities) {
        if (candidate.identity.y == compartmentIndex) ++matching;
    }
    if (matching != 1u) {
        throw std::runtime_error("compiled abdominal-aorta compartment has duplicate wall owners");
    }
}

} // namespace numi::human::vessel_wall_coupling
