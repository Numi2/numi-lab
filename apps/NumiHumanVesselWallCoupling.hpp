#pragma once

#include "NumiHumanRestingAorticWall.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
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
