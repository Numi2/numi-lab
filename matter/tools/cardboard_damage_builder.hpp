#pragma once

// Per-element authoring for a tetrahedral, selected-plane crack-band
// adaptation of Robertsson et al. (2023). The source brick law and this
// tetrahedral candidate intentionally have different program identities.

#include "cardboard_fracture_regularization.hpp"
#include "robertsson2023_material.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace numi::cardboard::damage {

using numi::cardboard::FractureVec3;
using numi::cardboard::TetCrackPlane;
using numi::cardboard::TetCrackPlaneGeometry;
using numi::cardboard::computeTetCrackPlaneGeometry;
using numi::matter::MaterialProgram;
using EnergyConvention = numi_cardboard::robertsson2023::CompressionEnergyConvention;
using StrengthHardeningParameters =
    numi_cardboard::robertsson2023::StrengthHardeningParameters;

inline constexpr const char* kSourceLawDOI = "10.1002/pts.2774";
inline constexpr const char* kSourceLawGrade =
    "single-ply softwood paperboard; not the literature2009 corrugated recipe";
inline constexpr std::array<const char*, 4> kDamageSubsurfaceNames{
    "nu9", "nu10", "nu11", "nu12"};

struct SourceBinding {
    std::string sourceId;
    std::string datasetId;
    std::string datasetSha256;
    std::string mapping;
};

struct ExplicitSetBinding {
    std::string setId;
    std::string sha256;
    std::string evidenceClass;
};

inline bool isSha256(const std::string& value) {
    if (value.size() != 64u) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline SourceBinding publishedStrengthHardeningBinding() {
    return {"robertsson_2023", "robertsson_2023_strength_hardening",
            "f3f35358ca3bef1719b1ec62d52939951a6cdf5830d320bbefaae4087344727f",
            "Table C2 / Eq. 27, own-grade source parameter set"};
}

inline void validateStrengthHardeningBinding(
    const StrengthHardeningParameters& parameters,
    const SourceBinding& binding) {
    if (binding.sourceId.empty() || binding.datasetId.empty() ||
        binding.mapping.empty() || !isSha256(binding.datasetSha256))
        throw std::invalid_argument("strength/hardening source binding is incomplete");
    if (binding.datasetId == "robertsson_2023_strength_hardening") {
        const auto expected =
            numi_cardboard::robertsson2023::publishedStrengthHardening();
        if (binding.sourceId != "robertsson_2023" ||
            binding.datasetSha256 !=
                "f3f35358ca3bef1719b1ec62d52939951a6cdf5830d320bbefaae4087344727f")
            throw std::invalid_argument(
                "Robertsson strength/hardening binding does not match the registered Table C2 CSV");
        for (unsigned i=0;i<12;++i) {
            if (parameters.initialStrengthMPa[i] != expected.initialStrengthMPa[i] ||
                parameters.hardeningAmplitudeMPa[i] != expected.hardeningAmplitudeMPa[i] ||
                parameters.hardeningRate[i] != expected.hardeningRate[i])
                throw std::invalid_argument(
                    "Robertsson Table C2 binding does not match its twelve compiled modes");
        }
    }
}

struct ElementDamageRequest {
    std::uint32_t elementIndex = 0;
    std::array<FractureVec3, 4> referenceVerticesM{};
    // One selected reference crack plane per source damage subsurface nu9..12.
    std::array<TetCrackPlane, 4> selectedPlanes{};
    // Explicit, still-unidentified values for the current starch/liner grade.
    // J/m^2, one value for each selected plane. There are no defaults.
    std::array<double, 4> fractureEnergyJPerM2{};
    // All twelve strength/hardening modes are required. Haj-Ali's four
    // uniaxial MD/CD line fits do not identify this multiaxial array; a caller
    // must bind a published source set or an explicit feasible-set member.
    std::optional<StrengthHardeningParameters> strengthHardening;
    std::optional<SourceBinding> strengthHardeningBinding;
    std::optional<ExplicitSetBinding> fractureEnergyBinding;
    std::optional<EnergyConvention> energyConvention;
    std::optional<double> damageCap;
};

struct ElementDamageProgram {
    std::uint32_t elementIndex = 0;
    double referenceVolumeM3 = 0.0;
    std::array<TetCrackPlaneGeometry, 4> selectedPlaneGeometry{};
    std::array<double, 4> fractureEnergyJPerM2{};
    std::array<double, 4> modeAreaMm2{};
    std::array<double, 4> modeLengthM{};
    SourceBinding strengthHardeningBinding;
    ExplicitSetBinding fractureEnergyBinding;
    EnergyConvention energyConvention = EnergyConvention::paper_absolute;
    MaterialProgram program;
    std::string mappingIdentity =
        "Robertsson2023_tetra_selected_plane_crack_band_candidate";
};

// Shipping meshes use one material index per distinct native FP32 program,
// while each tetrahedron keeps its exact geometry and source bindings here.
// Equality is exact after the same FP32 conversion used by the compiler; no
// parameter bins or geometric quantization are used.
struct ElementDamageMapping {
    std::uint32_t elementIndex = 0;
    std::uint32_t materialIndex = 0;
    double referenceVolumeM3 = 0.0;
    std::array<TetCrackPlaneGeometry, 4> selectedPlaneGeometry{};
    std::array<double, 4> fractureEnergyJPerM2{};
    std::array<double, 4> modeAreaMm2{};
    std::array<double, 4> modeLengthM{};
    SourceBinding strengthHardeningBinding;
    ExplicitSetBinding fractureEnergyBinding;
    EnergyConvention energyConvention = EnergyConvention::paper_absolute;
    std::string mappingIdentity =
        "Robertsson2023_tetra_selected_plane_crack_band_candidate";
};

struct MeshDamageProgramPool {
    std::vector<MaterialProgram> materials;
    // Same order as the input tetrahedra; stores exact material-map indices.
    std::vector<std::uint32_t> materialIndices;
    // Geometry and provenance remain per tetrahedron even when code is shared.
    std::vector<ElementDamageMapping> elementMappings;
};

inline std::uint64_t nativeProgramSignature(const MaterialProgram& material) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](const void* data, const std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    const auto mixFloat = [&](const double value) {
        const auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(value));
        mix(&bits, sizeof(bits));
    };
    const auto mixString = [&](const std::string& value) {
        const auto size = static_cast<std::uint64_t>(value.size());
        mix(&size, sizeof(size));
        mix(value.data(), value.size());
    };
    const auto mixDimension = [&](const numi::matter::Dimension& value) {
        mix(&value, sizeof(value));
    };

    // Name suffixes bind provenance, not executable arithmetic; per-element
    // provenance is retained in ElementDamageMapping. The expression DAG,
    // state layout and all values the compiler lowers into FP32 are included.
    mix(&material.cacheExpressions, sizeof(material.cacheExpressions));
    mix(&material.paperHillScalarInitializer,
        sizeof(material.paperHillScalarInitializer));
    mix(&material.paperCompensatedArithmetic,
        sizeof(material.paperCompensatedArithmetic));
    mix(&material.paperHillAccurateProjection,
        sizeof(material.paperHillAccurateProjection));
    mix(&material.genericCompensatedImplicitProjection,
        sizeof(material.genericCompensatedImplicitProjection));
    const auto parameterCount = static_cast<std::uint64_t>(material.parameters.size());
    mix(&parameterCount, sizeof(parameterCount));
    for (const auto& parameter : material.parameters) {
        mixString(parameter.name);
        mixDimension(parameter.dimension);
        mixFloat(parameter.defaultValue);
        mixFloat(parameter.lower);
        mixFloat(parameter.upper);
        mix(&parameter.logarithmic, sizeof(parameter.logarithmic));
        mix(&parameter.identifiable, sizeof(parameter.identifiable));
    }
    const auto stateCount = static_cast<std::uint64_t>(material.internalState.size());
    mix(&stateCount, sizeof(stateCount));
    for (const auto& state : material.internalState) {
        mixString(state.name);
        mixDimension(state.dimension);
        mixFloat(state.initialValue);
        mix(&state.transfer, sizeof(state.transfer));
    }
    const auto expressionCount = static_cast<std::uint64_t>(material.expressions.nodes.size());
    mix(&expressionCount, sizeof(expressionCount));
    for (const auto& expression : material.expressions.nodes) {
        mix(&expression.kind, sizeof(expression.kind));
        mixDimension(expression.dimension);
        mixFloat(expression.constant);
        mix(&expression.index, sizeof(expression.index));
        mix(&expression.integer, sizeof(expression.integer));
        mix(expression.arguments.data(), sizeof(expression.arguments));
    }
    const auto mixRoots = [&](const std::vector<std::uint32_t>& roots) {
        const auto size = static_cast<std::uint64_t>(roots.size());
        mix(&size, sizeof(size));
        mix(roots.data(), roots.size() * sizeof(std::uint32_t));
    };
    mix(&material.energyRoot, sizeof(material.energyRoot));
    mix(&material.dissipationRoot, sizeof(material.dissipationRoot));
    mix(&material.validityRoot, sizeof(material.validityRoot));
    mixRoots(material.stateUpdateRoots);
    mixRoots(material.stateImplicitRoots);
    const auto representationCount =
        static_cast<std::uint64_t>(material.supportedRepresentations.size());
    mix(&representationCount, sizeof(representationCount));
    mix(material.supportedRepresentations.data(),
        material.supportedRepresentations.size() * sizeof(numi::matter::Representation));
    mix(&material.hint, sizeof(material.hint));
    const auto& mixed = material.mixed;
    for (double value : {mixed.bulkModulus, mixed.thermalExpansion,
                         mixed.biotCoefficient, mixed.referenceTemperature,
                         mixed.heatCapacity, mixed.thermalConductivity,
                         mixed.heatSource, mixed.jouleHeatFraction,
                         mixed.poreStorage, mixed.poreMobility, mixed.poreSource,
                         mixed.electricalConductivity, mixed.activationDiffusivity,
                         mixed.activationOnRate, mixed.activationOffRate,
                         mixed.maximumActiveTension, mixed.activationThreshold,
                         mixed.activationSlope, mixed.cohesiveStrength,
                         mixed.fractureEnergy})
        mixFloat(value);
    for (double value : mixed.fibreDirection) mixFloat(value);
    mixFloat(material.staticFriction);
    mixFloat(material.dynamicFriction);
    mixFloat(material.restitution);
    mixFloat(material.adhesion);
    mixFloat(material.minimumDeterminant);
    mixFloat(material.maximumDeterminant);
    mixFloat(material.maximumStress);
    mixFloat(material.maximumEnergyDensity);
    return hash;
}

inline bool sameNativeProgram(const MaterialProgram& a, const MaterialProgram& b) {
    if (a.cacheExpressions != b.cacheExpressions ||
        a.paperHillScalarInitializer != b.paperHillScalarInitializer ||
        a.paperCompensatedArithmetic != b.paperCompensatedArithmetic ||
        a.paperHillAccurateProjection != b.paperHillAccurateProjection ||
        a.genericCompensatedImplicitProjection !=
            b.genericCompensatedImplicitProjection ||
        a.parameters.size() != b.parameters.size() ||
        a.internalState.size() != b.internalState.size() ||
        a.expressions.nodes.size() != b.expressions.nodes.size() ||
        a.energyRoot != b.energyRoot || a.dissipationRoot != b.dissipationRoot ||
        a.validityRoot != b.validityRoot ||
        a.stateUpdateRoots != b.stateUpdateRoots ||
        a.stateImplicitRoots != b.stateImplicitRoots ||
        a.supportedRepresentations != b.supportedRepresentations || a.hint != b.hint)
        return false;
    const auto sameFloat = [](const double x, const double y) {
        return std::bit_cast<std::uint32_t>(static_cast<float>(x)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y));
    };
    for (const auto [x, y] : std::array<std::pair<double, double>, 8>{
             {{a.staticFriction, b.staticFriction},
              {a.dynamicFriction, b.dynamicFriction},
              {a.restitution, b.restitution}, {a.adhesion, b.adhesion},
              {a.minimumDeterminant, b.minimumDeterminant},
              {a.maximumDeterminant, b.maximumDeterminant},
              {a.maximumStress, b.maximumStress},
              {a.maximumEnergyDensity, b.maximumEnergyDensity}}})
        if (!sameFloat(x, y)) return false;
    for (std::size_t i = 0; i < a.parameters.size(); ++i) {
        const auto& x = a.parameters[i];
        const auto& y = b.parameters[i];
        if (x.name != y.name || x.dimension != y.dimension ||
            std::bit_cast<std::uint32_t>(static_cast<float>(x.defaultValue)) !=
                std::bit_cast<std::uint32_t>(static_cast<float>(y.defaultValue)) ||
            std::bit_cast<std::uint32_t>(static_cast<float>(x.lower)) !=
                std::bit_cast<std::uint32_t>(static_cast<float>(y.lower)) ||
            std::bit_cast<std::uint32_t>(static_cast<float>(x.upper)) !=
                std::bit_cast<std::uint32_t>(static_cast<float>(y.upper)) ||
            x.logarithmic != y.logarithmic || x.identifiable != y.identifiable)
            return false;
    }
    for (std::size_t i = 0; i < a.internalState.size(); ++i) {
        const auto& x = a.internalState[i];
        const auto& y = b.internalState[i];
        if (x.name != y.name || x.dimension != y.dimension ||
            std::bit_cast<std::uint32_t>(static_cast<float>(x.initialValue)) !=
                std::bit_cast<std::uint32_t>(static_cast<float>(y.initialValue)) ||
            x.transfer != y.transfer)
            return false;
    }
    for (std::size_t i = 0; i < a.expressions.nodes.size(); ++i) {
        const auto& x = a.expressions.nodes[i];
        const auto& y = b.expressions.nodes[i];
        if (x.kind != y.kind || x.dimension != y.dimension ||
            std::bit_cast<std::uint32_t>(static_cast<float>(x.constant)) !=
                std::bit_cast<std::uint32_t>(static_cast<float>(y.constant)) ||
            x.index != y.index || x.integer != y.integer ||
            x.arguments != y.arguments)
            return false;
    }
    // This builder emits only ordinary generic FEM programs. Compare all
    // mixed constitutive fields too so future changes cannot alias programs.
    const auto& x = a.mixed;
    const auto& y = b.mixed;
    const bool sameMixed =
        std::bit_cast<std::uint32_t>(static_cast<float>(x.bulkModulus)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.bulkModulus)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.thermalExpansion)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.thermalExpansion)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.biotCoefficient)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.biotCoefficient)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.referenceTemperature)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.referenceTemperature)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.heatCapacity)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.heatCapacity)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.thermalConductivity)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.thermalConductivity)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.heatSource)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.heatSource)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.jouleHeatFraction)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.jouleHeatFraction)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.poreStorage)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.poreStorage)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.poreMobility)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.poreMobility)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.poreSource)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.poreSource)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.electricalConductivity)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.electricalConductivity)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.activationDiffusivity)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.activationDiffusivity)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.activationOnRate)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.activationOnRate)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.activationOffRate)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.activationOffRate)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.maximumActiveTension)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.maximumActiveTension)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.activationThreshold)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.activationThreshold)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.activationSlope)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.activationSlope)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.cohesiveStrength)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.cohesiveStrength)) &&
        std::bit_cast<std::uint32_t>(static_cast<float>(x.fractureEnergy)) ==
            std::bit_cast<std::uint32_t>(static_cast<float>(y.fractureEnergy));
    for (std::size_t i = 0; i < x.fibreDirection.size(); ++i)
        if (std::bit_cast<std::uint32_t>(static_cast<float>(x.fibreDirection[i])) !=
            std::bit_cast<std::uint32_t>(static_cast<float>(y.fibreDirection[i])))
            return false;
    return sameMixed && !a.learned && !b.learned;
}

inline ElementDamageProgram buildElementDamageProgram(
    const ElementDamageRequest& request) {
    if (!request.energyConvention || !request.damageCap ||
        !request.strengthHardening || !request.strengthHardeningBinding ||
        !request.fractureEnergyBinding)
        throw std::invalid_argument(
            "energy convention, damage cap, strength/hardening, and both provenance bindings must be explicit");
    validateStrengthHardeningBinding(*request.strengthHardening,
                                     *request.strengthHardeningBinding);
    if (request.fractureEnergyBinding->setId.empty() ||
        !isSha256(request.fractureEnergyBinding->sha256) ||
        request.fractureEnergyBinding->evidenceClass.empty())
        throw std::invalid_argument("fracture-energy set binding is incomplete");
    if (*request.damageCap < 0.0 || *request.damageCap > 0.96 ||
        !std::isfinite(*request.damageCap))
        throw std::invalid_argument("damage cap must be explicit and in [0, 0.96]");
    for (double gc : request.fractureEnergyJPerM2) {
        if (!(std::isfinite(gc) && gc > 0.0))
            throw std::invalid_argument(
                "each selected mode requires an explicit positive fracture energy");
    }

    ElementDamageProgram result;
    result.elementIndex = request.elementIndex;
    result.fractureEnergyJPerM2 = request.fractureEnergyJPerM2;
    result.energyConvention = *request.energyConvention;
    result.strengthHardeningBinding = *request.strengthHardeningBinding;
    result.fractureEnergyBinding = *request.fractureEnergyBinding;
    bool firstMode = true;
    for (std::size_t mode = 0; mode < request.selectedPlanes.size(); ++mode) {
        auto geometry = computeTetCrackPlaneGeometry(
            request.referenceVerticesM, request.selectedPlanes[mode]);
        if (!geometry.hasPositiveAreaCut())
            throw std::invalid_argument(std::string("selected plane ") +
                kDamageSubsurfaceNames[mode] + " does not cut element " +
                std::to_string(request.elementIndex) + " with positive area");
        if (firstMode) {
            result.referenceVolumeM3 = geometry.referenceVolumeM3;
            firstMode = false;
        } else {
            const double scale = std::max(result.referenceVolumeM3,
                                          geometry.referenceVolumeM3);
            if (std::abs(result.referenceVolumeM3 - geometry.referenceVolumeM3) >
                64.0 * std::numeric_limits<double>::epsilon() * scale)
                throw std::invalid_argument(
                    "damage planes were not evaluated on the same element geometry");
        }
        result.modeAreaMm2[mode] = geometry.referenceAreaM2 * 1.0e6;
        result.modeLengthM[mode] = *geometry.characteristicLengthM;
        result.selectedPlaneGeometry[mode] = std::move(geometry);
    }

    result.program = numi_cardboard::robertsson2023::makeSelectedPlaneCandidate(
        *request.energyConvention, result.modeAreaMm2,
        result.fractureEnergyJPerM2, result.modeLengthM,
        *request.strengthHardening, *request.damageCap);
    result.program.name += "_hard_" +
        result.strengthHardeningBinding.datasetSha256.substr(0u, 8u) + "_gc_" +
        result.fractureEnergyBinding.sha256.substr(0u, 8u);
    return result;
}

inline MeshDamageProgramPool buildMeshDamageProgramPool(
    const std::vector<ElementDamageRequest>& requests) {
    if (requests.empty())
        throw std::invalid_argument("cannot build a damage program pool for an empty mesh");
    MeshDamageProgramPool pool;
    pool.materialIndices.reserve(requests.size());
    pool.elementMappings.reserve(requests.size());
    std::unordered_set<std::uint32_t> elementIds;
    elementIds.reserve(requests.size());
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> signatures;
    signatures.reserve(requests.size());

    for (const auto& request : requests) {
        if (!elementIds.insert(request.elementIndex).second)
            throw std::invalid_argument(
                "mesh damage requests contain a duplicate element index");
        auto built = buildElementDamageProgram(request);
        const auto signature = nativeProgramSignature(built.program);
        std::optional<std::uint32_t> materialIndex;
        const auto bucket = signatures.find(signature);
        if (bucket != signatures.end()) {
            for (const std::uint32_t candidateIndex : bucket->second) {
                if (sameNativeProgram(pool.materials.at(candidateIndex), built.program)) {
                    materialIndex = candidateIndex;
                    break;
                }
            }
        }
        if (!materialIndex) {
            materialIndex = static_cast<std::uint32_t>(pool.materials.size());
            pool.materials.push_back(std::move(built.program));
            signatures[signature].push_back(*materialIndex);
        }

        ElementDamageMapping mapping;
        mapping.elementIndex = built.elementIndex;
        mapping.materialIndex = *materialIndex;
        mapping.referenceVolumeM3 = built.referenceVolumeM3;
        mapping.selectedPlaneGeometry = std::move(built.selectedPlaneGeometry);
        mapping.fractureEnergyJPerM2 = built.fractureEnergyJPerM2;
        mapping.modeAreaMm2 = built.modeAreaMm2;
        mapping.modeLengthM = built.modeLengthM;
        mapping.strengthHardeningBinding = std::move(built.strengthHardeningBinding);
        mapping.fractureEnergyBinding = std::move(built.fractureEnergyBinding);
        mapping.energyConvention = built.energyConvention;
        mapping.mappingIdentity = built.mappingIdentity;
        pool.materialIndices.push_back(*materialIndex);
        pool.elementMappings.push_back(std::move(mapping));
    }
    return pool;
}

inline std::vector<ElementDamageProgram> buildMeshDamagePrograms(
    const std::vector<ElementDamageRequest>& requests) {
    if (requests.empty())
        throw std::invalid_argument("cannot build damage programs for an empty mesh");
    std::vector<ElementDamageProgram> result;
    result.reserve(requests.size());
    for (const auto& request : requests)
        result.push_back(buildElementDamageProgram(request));
    std::unordered_set<std::uint32_t> elementIds;
    elementIds.reserve(result.size());
    for (const auto& element : result)
        if (!elementIds.insert(element.elementIndex).second)
            throw std::invalid_argument(
                "mesh damage requests contain a duplicate element index");
    return result;
}

} // namespace numi::cardboard::damage
