#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>

#include "cardboard_retained_matrix_damage.hpp"
#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {
using namespace numi::matter;
using namespace numi::cardboard::retained_damage;

struct Vec3 { double x = 0.0, y = 0.0, z = 0.0; };

Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

constexpr double kEdgeM = 0.01;
constexpr double kLayerThicknessM = 0.001;
constexpr double kTimestepSeconds = 1.0e-3;
constexpr std::uint32_t kFirstFaceId = 701u;
constexpr std::uint32_t kSecondFaceId = 907u;
constexpr const char* kParameterSchema = "numi.cardboard.synthetic_damage_cohesive_fixture_parameters.v1";
unsigned checks = 0u;
std::uint64_t nativeTransactions = 0u;

void require(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string out;
    for (const auto& value : values) out += value.message + "; ";
    return out;
}

std::string sha256(const std::string& bytes) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    CC_SHA256_Update(&context, bytes.data(), static_cast<CC_LONG>(bytes.size()));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    CC_SHA256_Final(digest, &context);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const unsigned char byte : digest)
        result << std::setw(2) << static_cast<unsigned>(byte);
    return result.str();
}

std::string readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open source-bound input: " + path.string());
    std::ostringstream contents;
    contents << input.rdbuf();
    require(input.good() || input.eof(), "cannot read source-bound input: " + path.string());
    return contents.str();
}

std::array<std::uint64_t, 4> identityFromDigest(const std::string& hexDigest) {
    require(hexDigest.size() == 64u, "SHA-256 digest has the wrong length");
    std::array<std::uint64_t, 4> identity{};
    for (std::size_t word = 0u; word < identity.size(); ++word) {
        for (std::size_t byte = 0u; byte < 8u; ++byte) {
            const std::size_t offset = 2u * (word * 8u + byte);
            const unsigned value = static_cast<unsigned>(std::stoul(
                hexDigest.substr(offset, 2u), nullptr, 16));
            identity[word] |= std::uint64_t(value) << (8u * byte);
        }
    }
    return identity;
}

std::string canonicalParameterRecord() {
    // This exact byte string is also checked in the companion JSON input. It
    // binds every synthetic value used below without presenting it as measured.
    return R"JSON({
  "schema": "numi.cardboard.synthetic_damage_cohesive_fixture_parameters.v1",
  "evidence_class": "synthetic_test_only_not_calibration",
  "damage_set_id": "transaction_fixture_damage_feasible_member_v1",
  "hardening_binding": {
    "source_id": "fixture_only_hardening",
    "dataset_id": "fixture_only_hardening",
    "mapping": "synthetic transaction exercise; not a measured kraft/NSSC fit"
  },
  "fracture_binding": {"set_id": "fixture_only_fracture", "evidence_class": "synthetic_test_only_not_calibration"},
  "rate_binding": {"set_id": "fixture_only_rate", "evidence_class": "synthetic_test_only_not_calibration"},
  "densification_binding": {"set_id": "fixture_only_densification", "evidence_class": "synthetic_test_only_not_calibration"},
  "matrix_damage": {
    "plastic_hardening_Pa_11_22_33_23_13_12": [130000000, 24000000, 0, 0, 0, 0],
    "fracture_modes_0_to_7_strength_Pa_Gc_J_per_m2_rate_s": [[100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001], [100000, 10, 0.001]],
    "densification_modulus_Pa_11_22_33": [40000000, 30000000, 20000000],
    "densification_onset_strain_11_22_33": [0.08, 0.1, 0.12],
    "densification_rate_s_11_22_33": [0.04, 0.06, 0.08],
    "damage_cap": 1,
    "smooth_strain_width": 0.0000001
  },
  "starch_interface": {
    "normal_penalty_Pa_per_m": 1000000000,
    "shear_penalty_Pa_per_m": 800000000,
    "compression_penalty_Pa_per_m": 2000000000,
    "normal_strength_Pa": 1000000,
    "shear_strength_Pa": 800000,
    "fracture_energy_mode_I_J_per_m2": 1000,
    "fracture_energy_mode_II_J_per_m2": 2000,
    "bk_exponent": 1.5
  },
  "geometry": {
    "edge_m": 0.01,
    "each_layer_thickness_m": 0.001,
    "timestep_s": 0.001,
    "gravity_m_per_s2": [0, 0, 0],
    "liner_and_medium_each_tets": 3,
    "finite_glue_tets": 3,
    "mortar_activation_ids": [701, 907],
    "activation_gap_tolerance_m": 0.0000001,
    "matrix_damage_state_stride": 30
  },
  "qualification": "synthetic transaction exercise only; no grade calibration or box qualification"
}
)JSON";
}

struct MaterialAsset {
    MaterialProgram material;
    std::string sha256;
};

MaterialAsset readMaterial(const std::filesystem::path& path) {
    const ParseResult parsed = parseMatterFile(path.string());
    require(parsed.succeeded(), "material parse failed: " + path.string() + ": " +
        diagnostics(parsed.diagnostics));
    return {parsed.material, sha256(readBytes(path))};
}

double densityOf(const MaterialProgram& material) {
    const auto found = std::find_if(material.parameters.begin(), material.parameters.end(),
        [](const Parameter& p) { return p.name == "density"; });
    require(found != material.parameters.end() && found->defaultValue > 0.0,
        "fixture source material has no positive density");
    return found->defaultValue;
}

ExplicitDamageSet syntheticDamageSet(const std::string& parameterSha256) {
    ExplicitDamageSet set;
    set.setId = "transaction_fixture_damage_feasible_member_v1";
    set.sha256 = parameterSha256;
    set.evidenceClass = "synthetic_test_only_not_calibration";
    set.plasticHardeningPa = {1.3e8, 2.4e7, 0.0, 0.0, 0.0, 0.0};
    set.hardeningBinding = {
        "fixture_only_hardening", "fixture_only_hardening",
        parameterSha256,
        "synthetic transaction exercise; not a measured kraft/NSSC fit"};
    for (auto& mode : set.fractureModes)
        mode = {1.0e5, 10.0, 1.0e-3};
    set.fractureBinding = {
        "fixture_only_fracture", parameterSha256,
        "synthetic_test_only_not_calibration"};
    set.rateBinding = {
        "fixture_only_rate", parameterSha256,
        "synthetic_test_only_not_calibration"};
    set.densificationModulusPa = {4.0e7, 3.0e7, 2.0e7};
    set.densificationOnsetStrain = {0.08, 0.10, 0.12};
    set.densificationRateTimeConstantSeconds = {0.04, 0.06, 0.08};
    set.densificationBinding = {
        "fixture_only_densification", parameterSha256,
        "synthetic_test_only_not_calibration"};
    set.damageCap = 1.0;
    set.smoothStrainWidth = 1.0e-7;
    return set;
}

std::array<std::uint32_t, 4> positiveTet(
    std::array<std::uint32_t, 4> tet, const std::vector<Vec3>& nodes) {
    const Vec3 a = nodes.at(tet[0]);
    if (dot(nodes.at(tet[1])-a,
            cross(nodes.at(tet[2])-a, nodes.at(tet[3])-a)) < 0.0)
        std::swap(tet[1], tet[2]);
    const double volume6 = dot(nodes.at(tet[1])-a,
        cross(nodes.at(tet[2])-a, nodes.at(tet[3])-a));
    require(std::isfinite(volume6) && volume6 > 0.0,
        "manufactured constituent tetrahedron is not positively oriented");
    return tet;
}

struct Fixture {
    WorldSource source;
    std::vector<Vec3> nodes;
    std::array<std::array<std::uint32_t, 4>, 9> tets{};
    std::array<std::uint32_t, 9> tetMaterials{};
    std::array<std::uint32_t, 8> damageStateIndices{};
    std::array<std::string, 3> materialSha256{};
    std::string parameterSha256;
};

ElementRequest makeElementRequest(const MaterialAsset& matrixAsset,
                                 const std::string& region,
                                 const std::uint32_t sourceMaterialIndex,
                                 const std::uint32_t tetIndex,
                                 const std::array<std::uint32_t, 4>& tet,
                                 const std::vector<Vec3>& nodes,
                                 const ExplicitDamageSet& set) {
    const MaterialProgram& matrix = matrixAsset.material;
    ElementRequest request;
    request.elementIndex = tetIndex;
    request.sourceMaterialIndex = sourceMaterialIndex;
    request.regionId = region;
    request.sourceMatrix = matrix;
    request.sourceDensityKgPerM3 = densityOf(matrix);
    request.sourceBinding = {
        matrix.name, matrixAsset.sha256,
        "source material asset SHA-256; damage feasible set is synthetic"};
    for (std::size_t i = 0u; i < tet.size(); ++i) {
        const Vec3 p = nodes.at(tet[i]);
        request.referenceVerticesM[i] = {p.x, p.y, p.z};
    }
    request.selectedMaterialPlanes = {{
        {{1.0, 0.0, 0.0}, numi::cardboard::FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{0.0, 1.0, 0.0}, numi::cardboard::FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{0.0, 0.0, 1.0}, numi::cardboard::FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{1.0, 1.0, 1.0}, numi::cardboard::FractureNormalFrame::Reference, 0.0, {}, 1.0e-12}}};
    request.explicitSet = set;
    return request;
}

Fixture makeFixture(const std::filesystem::path& materials,
                    const std::filesystem::path& parameterPath) {
    const MaterialAsset linerAsset = readMaterial(materials / "hajali2009_liner_hill_ideal.nmatter");
    const MaterialAsset mediumAsset = readMaterial(materials / "hajali2009_medium_hill_ideal.nmatter");
    const MaterialAsset glueAsset = readMaterial(materials / "starch2007_finite_glue_elastic.nmatter");
    const std::string parameterBytes = readBytes(parameterPath);
    require(parameterBytes == canonicalParameterRecord(),
        "synthetic fixture parameter JSON differs from the exact compiled parameter record");
    const std::string parameterSha256 = sha256(parameterBytes);
    Fixture fixture;
    fixture.materialSha256 = {linerAsset.sha256, mediumAsset.sha256, glueAsset.sha256};
    fixture.parameterSha256 = parameterSha256;
    const MaterialProgram& liner = linerAsset.material;
    const MaterialProgram& medium = mediumAsset.material;
    const MaterialProgram& glue = glueAsset.material;

    // Three 10 mm triangular prisms, each decomposed into three tets. The
    // duplicated coordinates at z=-2 mm and z=-1 mm form two independent
    // dormant adhesive interfaces while the middle prism remains finite glue.
    const std::array<double, 6> layerZ{{
        -3.0*kLayerThicknessM, -2.0*kLayerThicknessM,
        -2.0*kLayerThicknessM, -1.0*kLayerThicknessM,
        -1.0*kLayerThicknessM, 0.0}};
    const std::array<std::array<Vec3, 6>, 3> layerNodes{{
        {{{0.0,0.0,layerZ[0]}, {kEdgeM,0.0,layerZ[0]}, {0.0,kEdgeM,layerZ[0]},
          {0.0,0.0,layerZ[1]}, {kEdgeM,0.0,layerZ[1]}, {0.0,kEdgeM,layerZ[1]}}},
        {{{0.0,0.0,layerZ[2]}, {kEdgeM,0.0,layerZ[2]}, {0.0,kEdgeM,layerZ[2]},
          {0.0,0.0,layerZ[3]}, {kEdgeM,0.0,layerZ[3]}, {0.0,kEdgeM,layerZ[3]}}},
        {{{0.0,0.0,layerZ[4]}, {kEdgeM,0.0,layerZ[4]}, {0.0,kEdgeM,layerZ[4]},
          {0.0,0.0,layerZ[5]}, {kEdgeM,0.0,layerZ[5]}, {0.0,kEdgeM,layerZ[5]}}}
    }};
    for (const auto& layer : layerNodes)
        fixture.nodes.insert(fixture.nodes.end(), layer.begin(), layer.end());

    const std::array<std::array<std::uint32_t, 4>, 3> localTets{{
        {{0u,1u,2u,5u}}, {{0u,1u,4u,5u}}, {{0u,3u,4u,5u}}}};
    for (std::uint32_t layer = 0u; layer < 3u; ++layer) {
        for (std::uint32_t local = 0u; local < 3u; ++local) {
            auto tet = localTets[local];
            for (auto& node : tet) node += 6u * layer;
            fixture.tets[3u*layer + local] = positiveTet(tet, fixture.nodes);
        }
        double layerVolume = 0.0;
        for (std::uint32_t local = 0u; local < 3u; ++local) {
            const auto& tet = fixture.tets[3u*layer + local];
            const Vec3 a = fixture.nodes[tet[0]];
            const double volume6 = dot(fixture.nodes[tet[1]]-a,
                cross(fixture.nodes[tet[2]]-a, fixture.nodes[tet[3]]-a));
            layerVolume += volume6 / 6.0;
        }
        const double expectedVolume = 0.5 * kEdgeM * kEdgeM * kLayerThicknessM;
        require(std::abs(layerVolume - expectedVolume) <= expectedVolume * 1.0e-12,
            "three-tet prism decomposition does not preserve constituent volume");
    }

    std::vector<ElementRequest> requests;
    requests.reserve(6u);
    const ExplicitDamageSet damage = syntheticDamageSet(parameterSha256);
    for (std::uint32_t tet = 0u; tet < 3u; ++tet)
        requests.push_back(makeElementRequest(linerAsset, "liner", 0u, tet,
            fixture.tets[tet], fixture.nodes, damage));
    for (std::uint32_t tet = 6u; tet < 9u; ++tet)
        requests.push_back(makeElementRequest(mediumAsset, "medium", 1u, tet,
            fixture.tets[tet], fixture.nodes, damage));
    const MeshProgramPool overlays = buildMeshProgramPool(requests);
    require(overlays.overlayPrograms.size() >= 2u &&
            overlays.overlayMaterialIndices.size() == 6u &&
            overlays.elementMappings.size() == 6u,
        "retained matrix damage builder did not return six per-tet mappings");
    for (std::size_t mappingIndex = 0u;
         mappingIndex < overlays.elementMappings.size(); ++mappingIndex) {
        const ElementMapping& mapping = overlays.elementMappings[mappingIndex];
        const std::string& expectedSourceSha = mapping.regionId == "liner"
            ? linerAsset.sha256 : mediumAsset.sha256;
        require(mapping.sourceMaterialSha256 == expectedSourceSha &&
                mapping.explicitSetSha256 == parameterSha256 &&
                mapping.evidenceClass == "synthetic_test_only_not_calibration",
            "overlay mapping lost exact source asset or synthetic parameter hash binding");
    }

    WorldSource& source = fixture.source;
    source.environmentCount = 1u;
    source.frameTimestep = kTimestepSeconds;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials = {liner, medium, glue};
    for (const MaterialProgram& overlay : overlays.overlayPrograms)
        source.materials.push_back(overlay);
    source.cohesiveMaterials = {{
        .name = "synthetic_starch_interface_fixture_only",
        .normalPenaltyPaPerM = 1.0e9,
        .shearPenaltyPaPerM = 8.0e8,
        .compressionPenaltyPaPerM = 2.0e9,
        .normalStrengthPa = 1.0e6,
        .shearStrengthPa = 8.0e5,
        .fractureEnergyModeIJPerM2 = 1000.0,
        .fractureEnergyModeIIJPerM2 = 2000.0,
        .bkExponent = 1.5}};
    source.mixedSolver.newtonIterations = 24u;
    source.mixedSolver.fgmresIterations = 128u;
    source.mixedSolver.fgmresRestart = 16u;
    source.mixedSolver.relativeResidual = 1.0e-5;

    ObjectSource object;
    object.name = "synthetic_retained_liner_glue_medium_transaction";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.mixedFEM = false;
    object.characteristicLength = kLayerThicknessM;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.femReferenceSourceIdentity = identityFromDigest(sha256(
        "triangular_prism_fixture_v1;edge_m=0.01;layer_m=0.001;layers=liner,glue,medium;split=coincident"));
    object.femMaterialSourceIdentity = identityFromDigest(sha256(
        linerAsset.sha256 + ";" + mediumAsset.sha256 + ";" + glueAsset.sha256 + ";" + parameterSha256));
    for (const Vec3 p : fixture.nodes) {
        object.femNodes.push_back({p.x,p.y,p.z});
        object.femReferenceNodes.push_back({p.x,p.y,p.z});
    }
    object.femReferenceDisplacementGradient = true;
    object.femFixedNodes = {0u, 1u, 2u};
    for (const auto& tet : fixture.tets) object.tetrahedra.push_back({tet});

    const std::uint32_t overlayStart = 3u;
    for (std::uint32_t tet = 0u; tet < 3u; ++tet)
        fixture.tetMaterials[tet] = overlayStart + overlays.overlayMaterialIndices[tet];
    for (std::uint32_t tet = 3u; tet < 6u; ++tet)
        fixture.tetMaterials[tet] = 2u;
    for (std::uint32_t tet = 6u; tet < 9u; ++tet)
        fixture.tetMaterials[tet] = overlayStart + overlays.overlayMaterialIndices[3u + tet - 6u];
    object.femMaterialIndices.assign(fixture.tetMaterials.begin(),
                                    fixture.tetMaterials.end());

    const auto makePatch = [&](const std::array<std::uint32_t,3> sideA,
                               const std::array<std::uint32_t,3> sideB,
                               const std::uint32_t firstTet,
                               const std::uint32_t secondTet,
                               const std::uint32_t stableId) {
        CohesiveMortarPatchSource patch;
        patch.sideANodes = sideA;
        patch.sideBNodes = sideB;
        patch.sideAVertexBarycentric = {{
            {{1.0,0.0,0.0}}, {{0.0,1.0,0.0}}, {{0.0,0.0,1.0}}}};
        patch.sideBVertexBarycentric = patch.sideAVertexBarycentric;
        patch.firstTetrahedron = firstTet;
        patch.secondTetrahedron = secondTet;
        patch.materialIndex = 0u;
        patch.stableIdentifier = stableId;
        patch.cohesiveMaterialIndex = 0u;
        patch.activationGapToleranceM = 1.0e-7;
        object.femCohesiveMortarPatches.push_back(patch);
    };
    makePatch({{3u,4u,5u}}, {{6u,7u,8u}}, 2u, 3u, kFirstFaceId);
    makePatch({{9u,10u,11u}}, {{12u,13u,14u}}, 5u, 6u, kSecondFaceId);
    source.objects.push_back(std::move(object));

    for (std::uint32_t state = 0u; state < 8u; ++state) {
        const std::string name = "overlay_fracture_excess_" + std::to_string(state);
        const auto found = std::find_if(source.materials[fixture.tetMaterials[0]].internalState.begin(),
            source.materials[fixture.tetMaterials[0]].internalState.end(),
            [&](const InternalState& value) { return value.name == name; });
        require(found != source.materials[fixture.tetMaterials[0]].internalState.end(),
            "generated liner overlay lost a damage history state");
        fixture.damageStateIndices[state] = static_cast<std::uint32_t>(
            found - source.materials[fixture.tetMaterials[0]].internalState.begin());
    }
    return fixture;
}

CompiledWorld compileWorldForFixture(const WorldSource& source) {
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const CompileResult compiled = numi::matter::compileWorld(source, options);
    require(compiled.succeeded(), "coupled damage/cohesive fixture failed compile: " +
        diagnostics(compiled.diagnostics));
    return compiled.world;
}

void cpuAdmission(const Fixture& fixture) {
    const CompiledWorld compiled = compileWorldForFixture(fixture.source);
    require(compiled.dispatch.tetrahedronCount == 9u &&
            compiled.dispatch.femNodeCount == 18u,
        "fixture did not retain three finite material volumes and duplicated interface nodes");
    require(compiled.dispatch.cohesiveFaceCount == 2u &&
            compiled.fem.cohesiveFaces.size() == 2u,
        "two independent starch/glue mortar interfaces did not compile");
    require(compiled.dispatch.materialStateStride == 30u,
        "both retained matrix regions must compile with the full 30-state damage overlay");
    for (const auto& face : compiled.fem.cohesiveFaces)
        require((face.adjacency.w & NM_TOPOLOGY_COHESIVE_DORMANT) != 0u &&
                (face.adjacency.w & NM_TOPOLOGY_COHESIVE_MORTAR) != 0u &&
                face.geometry.w > 0.0f &&
                face.lawIdentity.w == NM_COHESIVE_ACTIVATION_EXPLICIT_COMMAND,
            "both synthetic glue interfaces must start dormant with positive mapped area");
    require(densityOf(fixture.source.materials[2]) == 1500.0,
        "finite starch-glue density assumption changed unexpectedly");
    for (const std::uint32_t materialIndex : {3u, 4u}) {
        require(fixture.source.materials.at(materialIndex).internalState.size() == 30u,
            "each source-bound liner/medium damage overlay must carry 30 state variables");
    }
}

std::string rawHex(const void* bytes, std::size_t length) {
    const auto* raw = static_cast<const unsigned char*>(bytes);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t index = 0u; index < length; ++index)
        out << std::setw(2) << static_cast<unsigned>(raw[index]);
    return out.str();
}

template<class T>
std::string vectorHex(const std::vector<T>& values) {
    return values.empty() ? std::string{} : rawHex(values.data(), values.size()*sizeof(T));
}

void emitSnapshotTrace(const std::string& label,
                       const RuntimeStateSnapshot& snapshot,
                       const bool full) {
    std::cerr << "cardboard_native_trace event=transaction label=\"" << label << "\""
              << " available=" << snapshot.available
              << " control_step=" << snapshot.controlStep
              << " physics_substep=" << snapshot.physicsSubstep
              << " material_state_stride=" << snapshot.materialStateStride
              << " node_count=" << snapshot.femNodes.size()
              << " tetrahedron_count=" << snapshot.femTopologyTetrahedra.size()
              << " cohesive_face_count=" << snapshot.cohesiveFaces.size()
              << " status_count=" << snapshot.statuses.size()
              << " certificate_count=" << snapshot.solverCertificates.size();
    for (std::size_t i = 0u; i < snapshot.statuses.size(); ++i) {
        const auto& status = snapshot.statuses[i];
        std::cerr << " status" << i << "={code:" << status.code
                  << ",environment:" << status.environment
                  << ",object:" << status.objectIndex
                  << ",failing_index:" << status.failingIndex
                  << ",microsteps:" << status.completedMicrosteps
                  << ",fgmres:" << status.fgmresIterations
                  << ",contacts:" << status.contactCount
                  << ",events:" << status.eventCount
                  << ",diag:[" << status.diagnostics.x << ','
                  << status.diagnostics.y << ',' << status.diagnostics.z << ','
                  << status.diagnostics.w << "]}";
    }
    for (std::size_t i = 0u; i < snapshot.solverCertificates.size(); ++i) {
        const auto& certificate = snapshot.solverCertificates[i];
        std::cerr << " certificate" << i << "={nonlinear:["
                  << certificate.nonlinear.x << ',' << certificate.nonlinear.y << ','
                  << certificate.nonlinear.z << ',' << certificate.nonlinear.w
                  << "],validity:[" << certificate.validity.x << ','
                  << certificate.validity.y << ',' << certificate.validity.z << ','
                  << certificate.validity.w << "]}";
    }
    std::cerr << '\n';
    if (!full) return;
    // Hex payloads preserve the exact readback bytes needed to diagnose a
    // failed native transaction before any acceptance assertion is raised.
    std::cerr << "cardboard_native_trace event=failed_snapshot label=\"" << label << "\""
              << " particles_le_hex=" << vectorHex(snapshot.particles)
              << " fem_nodes_le_hex=" << vectorHex(snapshot.femNodes)
              << " fem_fields_le_hex=" << vectorHex(snapshot.femFields)
              << " vascular_state_le_hex=" << vectorHex(snapshot.vascularState)
              << " vascular_clock_le_hex=" << vectorHex(snapshot.vascularClock)
              << " topology_nodes_le_hex=" << vectorHex(snapshot.femTopologyNodes)
              << " topology_tets_le_hex=" << vectorHex(snapshot.femTopologyTetrahedra)
              << " puncture_channels_le_hex=" << vectorHex(snapshot.punctureChannels)
              << " topology_state_le_hex=" << vectorHex(snapshot.topologyStates)
              << " statuses_le_hex=" << vectorHex(snapshot.statuses)
              << " certificates_le_hex=" << vectorHex(snapshot.solverCertificates)
              << " mpm_active_nodes_le_hex=" << vectorHex(snapshot.mpmActiveNodeIndices)
              << " mpm_node_to_active_le_hex=" << vectorHex(snapshot.mpmNodeToActive)
              << " mpm_active_counts_le_hex=" << vectorHex(snapshot.mpmActiveNodeCounts)
              << " rigid_generalized_f32_le_hex=" << vectorHex(snapshot.rigidGeneralizedCandidate)
              << " learned_weights_f32_le_hex=" << vectorHex(snapshot.learnedWeights)
              << " adaptive_le_hex=" << vectorHex(snapshot.adaptive)
              << " reactions_le_hex=" << vectorHex(snapshot.reactions)
              << " contact_samples_le_hex=" << vectorHex(snapshot.contactSamples)
              << " deformable_contact_failures_le_hex="
              << vectorHex(snapshot.deformableContactFailures)
              << " human_support_histories_le_hex=" << vectorHex(snapshot.humanSupportHistories)
              << " human_support_consequences_le_hex="
              << vectorHex(snapshot.humanSupportConsequences)
              << " material_state_f32_le_hex=" << vectorHex(snapshot.femMaterialState)
              << " particle_material_state_f32_le_hex=" << vectorHex(snapshot.particleMaterialState)
              << " environment_parameters_f32_le_hex=" << vectorHex(snapshot.environmentParameters)
              << " identification_le_hex=" << vectorHex(snapshot.identification)
              << " schedulers_le_hex=" << vectorHex(snapshot.schedulers)
              << " cohesive_faces_le_hex=" << vectorHex(snapshot.cohesiveFaces)
              << " deformable_contact_history_le_hex="
              << vectorHex(snapshot.deformableContactHistories)
              << " contact_history_le_hex=" << vectorHex(snapshot.contactHistories)
              << " rigid_states_le_hex=" << vectorHex(snapshot.rigidStates)
              << '\n';
}

void emitBindings(const Fixture& fixture) {
    std::cerr << "cardboard_native_trace event=source_bindings schema=\""
              << kParameterSchema << "\" liner_asset_sha256=" << fixture.materialSha256[0]
              << " medium_asset_sha256=" << fixture.materialSha256[1]
              << " glue_asset_sha256=" << fixture.materialSha256[2]
              << " synthetic_parameter_json_sha256=" << fixture.parameterSha256
              << " evidence_class=synthetic_test_only_not_calibration\n";
}

template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* data = values.empty() ? static_cast<const void*>(&zero)
                                      : static_cast<const void*>(values.data());
    return [device newBufferWithBytes:data
        length:std::max<std::size_t>(1u, values.size()) * sizeof(T)
        options:MTLResourceStorageModeShared];
}

struct Run {
    Runtime runtime;
    CompiledWorld world;
    const Fixture& fixture;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> statuses = nil;
    std::uint32_t nextControlStep = 0u;

    Run(CompiledWorld compiled, const Fixture& fixtureValue)
        : world(std::move(compiled)), fixture(fixtureValue) {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "native Metal device unavailable");
        const std::string deviceName([[device name] UTF8String]);
        require(deviceName.find("Apple") != std::string::npos &&
                deviceName.find("Paravirtual") == std::string::npos,
            "a native Apple Metal device is required");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue unavailable");
        RuntimeConfiguration configuration;
        configuration.metallib = NUMI_MATTER_METALLIB;
        configuration.environmentCount = world.dispatch.environmentCount;
        configuration.captureEvents = false;
        configuration.captureDiagnostics = true;
        configuration.adaptiveTransfer = false;
        const RuntimeDiagnostics initialized = runtime.initialize(world, configuration);
        require(initialized.encoded, "runtime initialization failed: " + initialized.message);
        statuses = makeBuffer(device, std::vector<MRMetalWorldStatusGPU>(1u));
    }

    RuntimeStateSnapshot state(const std::string& label = "snapshot") {
        RuntimeStateSnapshot result = runtime.snapshot();
        if (!result.available) {
            std::cerr << "cardboard_native_trace event=snapshot_error label=\""
                      << label << "\" message=\"" << result.message << "\"\n";
            throw std::runtime_error("runtime snapshot failed: " + result.message);
        }
        return result;
    }

    void restore(const RuntimeStateSnapshot& snapshot) {
        const RuntimeDiagnostics result = runtime.restore(snapshot);
        require(result.encoded, "full coupled checkpoint restore failed: " + result.message);
        nextControlStep = snapshot.controlStep + 1u;
    }

    std::vector<nm_float4> targets(double topZDisplacement,
                                   bool singular = false) const {
        std::vector<nm_float4> result(fixture.nodes.size());
        for (std::size_t i = 0u; i < result.size(); ++i) {
            const Vec3 p = fixture.nodes[i];
            result[i] = {static_cast<float>(p.x), static_cast<float>(p.y),
                         static_cast<float>(p.z), 0.0f};
        }
        // The only driven nodes are the exterior top triangular grip patch.
        for (const std::uint32_t node : {15u,16u,17u}) {
            Vec3 p = fixture.nodes[node];
            p.z += topZDisplacement;
            if (singular) p = fixture.nodes[0];
            result[node] = {static_cast<float>(p.x), static_cast<float>(p.y),
                             static_cast<float>(p.z), 3.0f};
        }
        return result;
    }

    RuntimeStateSnapshot step(
        const std::string& label,
        const std::vector<nm_float4>& nodeTargets,
        const std::vector<std::uint32_t>& activationIds = {}) {
        require(nodeTargets.size() == world.dispatch.femNodeCount,
            "target array does not cover the compiled nodes");
        auto* status = static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        status[0] = {};
        status[0].environment = 0u;
        id<MTLBuffer> targetBuffer = makeBuffer(device, nodeTargets);
        id<MTLBuffer> activationBuffer = nil;
        std::vector<NMCohesiveActivationCommandGPU> commands;
        for (const std::uint32_t idValue : activationIds) {
            NMCohesiveActivationCommandGPU command{};
            command.identity = {0u, 0u, idValue, 0u};
            commands.push_back(command);
        }
        if (!commands.empty()) activationBuffer = makeBuffer(device, commands);

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        require(commandBuffer != nil, "Metal command buffer unavailable");
        EncodeRequest request;
        request.commandBuffer = (__bridge void*)commandBuffer;
        request.environmentStatuses = (__bridge void*)statuses;
        request.femKinematicTargets = (__bridge void*)targetBuffer;
        request.femKinematicTargetCount = static_cast<std::uint32_t>(nodeTargets.size());
        request.controlStep = nextControlStep;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        if (!commands.empty()) {
            request.cohesiveActivationCommands = (__bridge void*)activationBuffer;
            request.cohesiveActivationCommandCount =
                static_cast<std::uint32_t>(commands.size());
            request.cohesiveActivationCommandStride = sizeof(NMCohesiveActivationCommandGPU);
        }
        request.phase = EncodePhase::preDynamics;
        RuntimeDiagnostics encoded = runtime.encode(request);
        if (!encoded.encoded) {
            std::cerr << "cardboard_native_trace event=encode_error label=\""
                      << label << "\" phase=preDynamics control_step="
                      << nextControlStep << " message=\"" << encoded.message << "\"\n";
            throw std::runtime_error("preDynamics encode failed: " + encoded.message);
        }
        request.phase = EncodePhase::postCommit;
        request.cohesiveActivationCommands = nullptr;
        request.cohesiveActivationCommandCount = 0u;
        request.cohesiveActivationCommandStride = 0u;
        encoded = runtime.encode(request);
        if (!encoded.encoded) {
            std::cerr << "cardboard_native_trace event=encode_error label=\""
                      << label << "\" phase=postCommit control_step="
                      << nextControlStep << " message=\"" << encoded.message << "\"\n";
            throw std::runtime_error("postCommit encode failed: " + encoded.message);
        }
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "cardboard_native_trace event=command_buffer_error label=\""
                      << label << "\" status=" << static_cast<unsigned>(commandBuffer.status)
                      << " error=\"" << (commandBuffer.error == nil ? "none"
                          : [[commandBuffer.error localizedDescription] UTF8String]) << "\"\n";
            try {
                const RuntimeStateSnapshot failed = state(label + ".command_buffer_failure");
                emitSnapshotTrace(label + ".command_buffer_failure", failed, true);
            } catch (const std::exception& snapshotError) {
                std::cerr << "cardboard_native_trace event=failure_snapshot_unavailable label=\""
                          << label << "\" message=\"" << snapshotError.what() << "\"\n";
            }
            throw std::runtime_error("native transaction command buffer failed");
        }
        ++nativeTransactions;
        RuntimeStateSnapshot result = state(label);
        const bool accepted = result.statuses.size() == 1u &&
            result.statuses[0].code == NM_STATUS_SUCCESS &&
            result.solverCertificates.size() == 1u &&
            result.solverCertificates[0].validity.w > 0.5f;
        emitSnapshotTrace(label, result, !accepted);
        if (result.statuses.size() == 1u &&
            result.statuses[0].code == NM_STATUS_SUCCESS)
            nextControlStep = result.controlStep + 1u;
        return result;
    }
};

bool hasFlag(const NMCohesiveFaceGPU& face, std::uint32_t flag) {
    return (face.adjacency.w & flag) != 0u;
}

void requireAccepted(const RuntimeStateSnapshot& snapshot, const std::string& label) {
    const bool accepted = snapshot.statuses.size() == 1u &&
            snapshot.solverCertificates.size() == 1u &&
            snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
            snapshot.solverCertificates[0].validity.w > 0.5f;
    if (!accepted) emitSnapshotTrace(label, snapshot, true);
    require(accepted, label + " did not produce an accepted production FEM certificate");
}

void requireContinuationEqual(const RuntimeStateSnapshot& a,
                              const RuntimeStateSnapshot& b,
                              const std::string& label,
                              const bool compareCompletionDiagnostics = true) {
    const auto same = [](const auto& lhs, const auto& rhs) {
        return lhs.size() == rhs.size() &&
            (lhs.empty() || std::memcmp(lhs.data(), rhs.data(),
                lhs.size()*sizeof(lhs[0])) == 0);
    };
    const bool equal = a.controlStep == b.controlStep && a.physicsSubstep == b.physicsSubstep &&
            a.materialStateStride == b.materialStateStride &&
            a.allocationGeneration == b.allocationGeneration &&
            a.learnedWeightRevision == b.learnedWeightRevision &&
            same(a.particles,b.particles) && same(a.femFields,b.femFields) &&
            same(a.vascularState,b.vascularState) && same(a.vascularClock,b.vascularClock) &&
            same(a.femTopologyNodes,b.femTopologyNodes) &&
            same(a.femNodes,b.femNodes) && same(a.femTopologyTetrahedra,b.femTopologyTetrahedra) &&
            same(a.punctureChannels,b.punctureChannels) && same(a.topologyStates,b.topologyStates) &&
            same(a.mpmActiveNodeIndices,b.mpmActiveNodeIndices) &&
            same(a.mpmNodeToActive,b.mpmNodeToActive) &&
            same(a.mpmActiveNodeCounts,b.mpmActiveNodeCounts) &&
            same(a.rigidGeneralizedCandidate,b.rigidGeneralizedCandidate) &&
            same(a.learnedWeights,b.learnedWeights) && same(a.adaptive,b.adaptive) &&
            same(a.reactions,b.reactions) && same(a.rigidStates,b.rigidStates) &&
            same(a.humanSupportHistories,b.humanSupportHistories) &&
            same(a.humanSupportConsequences,b.humanSupportConsequences) &&
            same(a.femMaterialState,b.femMaterialState) &&
            same(a.particleMaterialState,b.particleMaterialState) &&
            same(a.environmentParameters,b.environmentParameters) &&
            (!compareCompletionDiagnostics ||
                (same(a.statuses,b.statuses) && same(a.solverCertificates,b.solverCertificates))) &&
            same(a.schedulers,b.schedulers) &&
            same(a.cohesiveFaces,b.cohesiveFaces) &&
            same(a.deformableContactHistories,b.deformableContactHistories) &&
            same(a.contactHistories,b.contactHistories);
    if (!equal) {
        emitSnapshotTrace(label + ".expected", a, true);
        emitSnapshotTrace(label + ".actual", b, true);
    }
    require(equal, label + " changed accepted FEM, 30-state material, scheduler, or interface history");
}

void requireSnapshot(bool value, const RuntimeStateSnapshot& snapshot,
                     const std::string& label, const std::string& message) {
    if (!value) emitSnapshotTrace(label, snapshot, true);
    require(value, message);
}

void nativeTransaction(const Fixture& fixture) {
    Run run(compileWorldForFixture(fixture.source), fixture);
    const RuntimeStateSnapshot initial = run.state();
    emitSnapshotTrace("initial", initial, false);
    require(initial.cohesiveFaces.size() == 2u && initial.materialStateStride == 30u,
        "native initial state does not contain both dormant interfaces and 30-state FEM");
    for (const auto& face : initial.cohesiveFaces)
        require(hasFlag(face, NM_TOPOLOGY_COHESIVE_DORMANT),
            "bond activated before an accepted closure/setup endpoint");

    // Accept the closed geometry with no command, then cure one interface at a
    // time in separate transactions. The reference displacement and all bulk
    // stress/history states remain resident across both activations.
    const RuntimeStateSnapshot closed = run.step("closed_setup", run.targets(0.0));
    requireAccepted(closed, "closed setup");
    const RuntimeStateSnapshot firstActive = run.step(
        "activate_face_701", run.targets(0.0), {kFirstFaceId});
    requireAccepted(firstActive, "first interface activation");
    requireSnapshot(!hasFlag(firstActive.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_DORMANT) &&
             hasFlag(firstActive.cohesiveFaces[1], NM_TOPOLOGY_COHESIVE_DORMANT), firstActive,
        "activate_face_701", "first activation did not remain independent from the second dormant interface");
    const RuntimeStateSnapshot activated = run.step(
        "activate_face_907", run.targets(0.0), {kSecondFaceId});
    requireAccepted(activated, "second interface activation");
    requireSnapshot(!hasFlag(activated.cohesiveFaces[0], NM_TOPOLOGY_COHESIVE_DORMANT) &&
            !hasFlag(activated.cohesiveFaces[1], NM_TOPOLOGY_COHESIVE_DORMANT), activated,
        "activate_face_907", "both synthetic glue interfaces did not activate through accepted commands");
    for (const auto& face : activated.cohesiveFaces)
        for (const nm_float4 gap : face.activationLocalGap) {
            const bool gapValid = std::sqrt(double(gap.x)*gap.x + double(gap.y)*gap.y +
                                            double(gap.z)*gap.z) < 1.0e-7;
            requireSnapshot(gapValid, activated, "activate_face_907",
                "activation reference was not latched at coincident accepted faces");
        }

    // Every overlay is compiled at 30 states; this controlled exterior top
    // grip drives a small load/unload cycle. The law values and rate are all
    // synthetic, so this only exercises transactional coupling.
    RuntimeStateSnapshot loaded = activated;
    std::uint32_t loadIndex = 0u;
    for (const double dz : {2.0e-5, 4.0e-5, 6.0e-5}) {
        ++loadIndex;
        loaded = run.step("synthetic_load_" + std::to_string(loadIndex), run.targets(dz));
        requireAccepted(loaded, "synthetic coupled loading");
    }
    const std::size_t stride = loaded.materialStateStride;
    require(stride == 30u && loaded.femMaterialState.size() == 9u*stride,
        "accepted snapshot lost per-tet 30-state material storage");
    bool damageHistoryAdvanced = false;
    for (const std::uint32_t tet : {0u,1u,2u,6u,7u,8u}) {
        for (const std::uint32_t stateIndex : fixture.damageStateIndices) {
            const std::size_t index = std::size_t(tet)*stride + stateIndex;
            if (index < loaded.femMaterialState.size() &&
                std::isfinite(loaded.femMaterialState[index]) &&
                loaded.femMaterialState[index] > 1.0e-9f)
                damageHistoryAdvanced = true;
        }
    }
    requireSnapshot(damageHistoryAdvanced, loaded, "synthetic_loading_complete",
        "accepted loading did not advance any synthetic matrix fracture history");
    const RuntimeStateSnapshot unloaded = run.step("synthetic_unload", run.targets(0.0));
    requireAccepted(unloaded, "synthetic coupled unloading");
    requireSnapshot(unloaded.femMaterialState.size() == loaded.femMaterialState.size(),
        unloaded, "synthetic_unload", "unloading dropped the persistent bulk state arena");
    for (const auto& face : unloaded.cohesiveFaces)
        requireSnapshot(!hasFlag(face, NM_TOPOLOGY_COHESIVE_DORMANT), unloaded,
            "synthetic_unload", "unloading reset an activated interface to dormant");

    // Rejection after both bonds and damage are active must restore the exact
    // accepted continuation, including scheduler and cohesive histories.
    const RuntimeStateSnapshot beforeReject = unloaded;
    const RuntimeStateSnapshot rejected = run.step("intentional_singular_endpoint_rejection",
        run.targets(0.0, true));
    requireSnapshot(rejected.statuses.size() == 1u &&
            rejected.statuses[0].code != NM_STATUS_SUCCESS, rejected,
        "intentional_singular_endpoint_rejection",
        "deliberately singular endpoint unexpectedly committed");
    requireSnapshot(rejected.solverCertificates.size() == 1u &&
            rejected.solverCertificates[0].validity.w <= 0.5f, rejected,
        "intentional_singular_endpoint_rejection",
        "rejected endpoint retained an accepted certificate");
    requireContinuationEqual(beforeReject, rejected, "rejected transaction rollback", false);

    Run resumed(compileWorldForFixture(fixture.source), fixture);
    const RuntimeDiagnostics restore = resumed.runtime.restore(beforeReject);
    if (!restore.encoded) {
        emitSnapshotTrace("checkpoint_restore_input", beforeReject, true);
        std::cerr << "cardboard_native_trace event=restore_error message=\""
                  << restore.message << "\"\n";
        throw std::runtime_error("full coupled checkpoint restore failed: " + restore.message);
    }
    resumed.nextControlStep = beforeReject.controlStep + 1u;
    requireContinuationEqual(beforeReject, resumed.state(),
        "paired checkpoint restore before continuation");
    const RuntimeStateSnapshot resumedNext = resumed.step("restored_continuation",
        resumed.targets(0.0));
    const RuntimeStateSnapshot directNext = run.step("uninterrupted_continuation",
        run.targets(0.0));
    requireAccepted(resumedNext, "restored continuation");
    requireAccepted(directNext, "uninterrupted continuation");
    requireContinuationEqual(resumedNext, directNext,
        "restored versus uninterrupted coupled continuation");
}

} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 3 || (argc == 4 &&
                std::string(argv[3]) == "--cpu-only"),
                "usage: cardboard-damage-cohesive-transaction-check MATERIALS_DIR PARAMETER_JSON [--cpu-only]");
            const Fixture fixture = makeFixture(argv[1], argv[2]);
            emitBindings(fixture);
            cpuAdmission(fixture);
            if (argc == 4) {
                std::cout << "PASS CPU coupled admission checks=" << checks
                    << " tets=9 finite_glue_tets=3 matrix_overlay_states=30"
                    << " independent_mortar_faces=2 synthetic_parameters=true"
                    << " parameter_sha256=" << fixture.parameterSha256
                    << " calibration=false native_transactions=0\n";
                return 0;
            }
            nativeTransaction(fixture);
            std::cout << "PASS native coupled synthetic transaction checks=" << checks
                << " native_transactions=" << nativeTransactions
                << " parameter_sha256=" << fixture.parameterSha256
                << " matrix_damage_history=true independent_interface_activation=true"
                << " full_checkpoint_rollback=true calibration=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "cardboard coupled damage/cohesive transaction check failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
