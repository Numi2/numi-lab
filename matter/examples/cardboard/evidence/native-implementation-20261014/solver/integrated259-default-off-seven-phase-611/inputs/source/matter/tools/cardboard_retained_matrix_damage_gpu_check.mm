#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "cardboard_retained_matrix_damage.hpp"
#include "cardboard_retained_damage_fp64_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB
#define NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB ""
#endif

namespace {
using namespace numi::matter;
using namespace numi::cardboard;
using namespace numi::cardboard::retained_damage;
namespace fp64 = numi::cardboard::retained_damage::fp64;

using Matrix = std::array<double, 9>;
constexpr Matrix identity{1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0};
constexpr std::size_t kStateOffset = 19u;
constexpr std::size_t kMetadataOffset = kStateOffset + NM_MAX_MATERIAL_STATE;
constexpr std::size_t kStateLowOffset = kMetadataOffset + 6u;
constexpr std::size_t kOutputCount = kStateLowOffset + NM_MAX_MATERIAL_STATE;


void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string sha256(const std::string& bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto value : digest) output << std::setw(2) << unsigned(value);
    return output.str();
}

std::string sha256File(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open source-bound input: " + path);
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    require(!input.bad(), "failed reading source-bound input: " + path);
    return sha256(bytes);
}

template <class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    const T zero{};
    const void* bytes = values.empty() ? &zero : values.data();
    const std::size_t count = std::max<std::size_t>(1u, values.size());
    return [device newBufferWithBytes:bytes
                               length:count * sizeof(T)
                              options:MTLResourceStorageModeShared];
}

std::string diagnostics(const std::vector<Diagnostic>& values) {
    std::string text;
    for (const auto& diagnostic : values) text += diagnostic.message + "; ";
    return text;
}

const Parameter& findParameter(const MaterialProgram& material,
                               const std::string& name) {
    const auto found = std::find_if(material.parameters.begin(),
        material.parameters.end(), [&](const Parameter& parameter) {
            return parameter.name == name;
        });
    if (found == material.parameters.end())
        throw std::runtime_error("material lacks parameter " + name);
    return *found;
}

std::uint32_t findState(const MaterialProgram& material,
                        const std::string& name) {
    for (std::uint32_t i = 0u; i < material.internalState.size(); ++i)
        if (material.internalState[i].name == name) return i;
    throw std::runtime_error("material lacks state " + name);
}

ExplicitDamageSet syntheticDamageSet(const std::string& hardeningCsvSha) {
    ExplicitDamageSet set;
    set.setId = "retained_matrix_native_instrument_synthetic_v1";
    set.evidenceClass = "synthetic_instrument_only_not_calibration";
    set.plasticHardeningPa = {1.3e8, 2.4e7, 0.0, 0.0, 0.0, 0.0};
    set.hardeningBinding = {
        "Haj-Ali 2009 fitted-tension-node dataset; overlay coefficients are synthetic",
        "haj_ali_2009_bilinear_tension_fit_nodes.csv",
        hardeningCsvSha,
        "instrument fixture values are not the published fit values"};

    for (auto& mode : set.fractureModes)
        mode = {1.0e8, 1.0e6, 0.04};
    set.fractureModes[0] = {2.0e6, 0.4, 0.004};
    set.fractureBinding = {
        "native_instrument_synthetic_fracture_parameters",
        sha256("normal0=2000000,0.4,0.004;other_modes=100000000,1000000,0.04"),
        "synthetic_instrument_only_not_calibration"};
    set.rateBinding = {
        "native_instrument_synthetic_rate_parameters",
        sha256("mode0_tau=0.004;other_modes_tau=0.04"),
        "synthetic_instrument_only_not_calibration"};
    set.densificationModulusPa = {4.0e7, 3.0e7, 2.0e7};
    set.densificationOnsetStrain = {0.08, 0.10, 0.12};
    set.densificationRateTimeConstantSeconds = {0.04, 0.06, 0.08};
    set.densificationBinding = {
        "native_instrument_synthetic_compaction_parameters",
        sha256("modulus=40000000,30000000,20000000;onset=0.08,0.10,0.12;"
               "tau=0.04,0.06,0.08"),
        "synthetic_instrument_only_not_calibration"};
    set.damageCap = 1.0;
    set.smoothStrainWidth = 1.0e-7;

    std::ostringstream identityText;
    identityText << std::setprecision(17) << set.setId << '|'
        << set.evidenceClass << '|';
    for (const auto value : set.plasticHardeningPa) identityText << value << ',';
    for (const auto& mode : set.fractureModes)
        identityText << mode.strengthPa << ',' << mode.fractureEnergyJPerM2
                     << ',' << mode.rateTimeConstantSeconds << ';';
    for (const auto value : set.densificationModulusPa) identityText << value << ',';
    for (const auto value : set.densificationOnsetStrain) identityText << value << ',';
    for (const auto value : set.densificationRateTimeConstantSeconds) identityText << value << ',';
    identityText << set.damageCap << '|' << set.smoothStrainWidth;
    set.sha256 = sha256(identityText.str());
    return set;
}

ElementRequest makeRequest(const MaterialProgram& source,
                           const std::string& sourcePath,
                           const std::string& hardeningCsvSha,
                           const std::string& region) {
    ElementRequest request;
    request.elementIndex = 0u;
    request.sourceMaterialIndex = 0u;
    request.regionId = region;
    request.sourceMatrix = source;
    request.sourceDensityKgPerM3 = findParameter(source, "density").defaultValue;
    request.sourceBinding = {
        source.name,
        sha256File(sourcePath),
        "exact_local_material_file_sha256"};
    request.referenceVerticesM = {{{0.0, 0.0, 0.0},
                                    {0.001, 0.0, 0.0},
                                    {0.0, 0.001, 0.0},
                                    {0.0, 0.0, 0.001}}};
    request.selectedMaterialPlanes = {{
        {{1.0, 0.0, 0.0}, FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{0.0, 1.0, 0.0}, FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{0.0, 0.0, 1.0}, FractureNormalFrame::Reference, 0.0, {}, 1.0e-12},
        {{1.0, 1.0, 1.0}, FractureNormalFrame::Reference, 0.0, {}, 1.0e-12}}};
    request.explicitSet = syntheticDamageSet(hardeningCsvSha);
    return request;
}

CompiledWorld compileInstrumentWorld(const MaterialProgram& material) {
    WorldSource source;
    source.materials = {material};
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.frameTimestep = 2.0e-3;
    ObjectSource object;
    object.name = "retained_matrix_constitutive_instrument";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{{0.0, 0.0, 0.0},
                        {0.001, 0.0, 0.0},
                        {0.0, 0.001, 0.0},
                        {0.0, 0.0, 0.001}}};
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    object.characteristicLength = 0.001;
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    const auto result = compileWorld(source, options);
    require(result.succeeded(), "overlay material compilation failed: " +
                               diagnostics(result.diagnostics));
    return result.world;
}

struct Response {
    bool valid = false;
    bool projectionValid = false;
    float residual = 0.0f;
    float stateNorm = 0.0f;
    std::uint32_t iterations = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t failure = 0u;
    Matrix stress{};
    Matrix tangent{};
    std::vector<float> state;
    std::vector<float> stateLow;
};

struct SweepObservation {
    double requestedEpsilon = 0.0;
    bool nativePairValid = false;
    bool oraclePairValid = false;
    std::string oracleFailure;
    Matrix plusF{};
    Matrix plusFHigh{};
    Matrix plusFLow{};
    Matrix minusF{};
    Matrix minusFHigh{};
    Matrix minusFLow{};
    Matrix realizedDirection{};
    double directionRealizationRelativeL2 = 0.0;
    double midpointOffsetInf = 0.0;
    Response plusNative;
    Response minusNative;
    fp64::Projection plusOracle;
    fp64::Projection minusOracle;
    Matrix plusOracleStress{};
    Matrix minusOracleStress{};
    Matrix nativeFiniteDifference{};
    Matrix oracleFiniteDifference{};
    Matrix nativeTangentAtRealizedDirection{};
    Matrix oracleTangentAtRealizedDirection{};
    std::string oracleCenterBranch;
    std::string oraclePlusBranch;
    std::string oracleMinusBranch;
    std::string nativeCenterBranch;
    std::string nativePlusBranch;
    std::string nativeMinusBranch;
    bool oracleBranchPreserved = false;
    bool nativeBranchMatchesOracle = false;
};

void writeJsonArray(std::ostream& output, const Matrix& values) {
    output << '[';
    for (std::size_t i = 0u; i < values.size(); ++i) {
        if (i != 0u) output << ',';
        const float exactNativeInput = static_cast<float>(values[i]);
        output << std::setprecision(std::numeric_limits<float>::max_digits10)
               << exactNativeInput;
    }
    output << ']';
}

float stateByName(const MaterialProgram& material,
                  const std::vector<float>& values,
                  const std::string& name) {
    for (std::size_t i = 0u; i < material.internalState.size(); ++i)
        if (material.internalState[i].name == name) return values.at(i);
    throw std::runtime_error("diagnostic cannot locate state " + name);
}

float parameterByName(const std::vector<Parameter>& parameters,
                      const std::string& name) {
    const auto found = std::find_if(parameters.begin(), parameters.end(),
        [&](const Parameter& value) { return value.name == name; });
    if (found == parameters.end())
        throw std::runtime_error("diagnostic cannot locate parameter " + name);
    return static_cast<float>(found->defaultValue);
}

void writeYieldBranchJson(std::ostream& output, const MaterialProgram& material,
                          const std::vector<Parameter>& parameters,
                          const std::vector<float>& accepted,
                          const std::vector<float>& projected) {
    const auto yield = [&](const std::vector<float>& state) {
        const float s11 = stateByName(material, state, "sbar11");
        const float s22 = stateByName(material, state, "sbar22");
        const float s33 = stateByName(material, state, "sbar33");
        const float s23 = stateByName(material, state, "sbar23");
        const float s13 = stateByName(material, state, "sbar13");
        const float s12 = stateByName(material, state, "sbar12");
        const float d23 = s22 - s33;
        const float d31 = s33 - s11;
        const float d12 = s11 - s22;
        const float q2 = parameterByName(parameters, "hill_F") * d23 * d23 +
            parameterByName(parameters, "hill_G") * d31 * d31 +
            parameterByName(parameters, "hill_H") * d12 * d12 +
            2.0f * parameterByName(parameters, "hill_L") * s23 * s23 +
            2.0f * parameterByName(parameters, "hill_M") * s13 * s13 +
            2.0f * parameterByName(parameters, "hill_N") * s12 * s12 +
            parameterByName(parameters, "q_floor") * parameterByName(parameters, "q_floor");
        float overlayHardening = 0.0f;
        for (unsigned i = 0u; i < 6u; ++i) {
            const auto suffix = std::to_string(i);
            overlayHardening += parameterByName(parameters,
                "overlay_plastic_hardening_" + suffix) * stateByName(material, state,
                "overlay_plastic_history_" + suffix);
        }
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            const auto suffix = std::to_string(axis);
            overlayHardening += parameterByName(parameters,
                "overlay_densification_modulus_pa_axis" + suffix) *
                stateByName(material, state, "overlay_densification_" + suffix);
        }
        const float yieldRatio = 1.0f + overlayHardening /
            parameterByName(parameters, "sigma_ref");
        return std::array<float, 4>{std::sqrt(std::max(0.0f, q2)), yieldRatio,
            yieldRatio - std::sqrt(std::max(0.0f, q2)),
            stateByName(material, state, "accumulated_multiplier")};
    };
    const auto oldValue = yield(accepted);
    const auto newValue = yield(projected);
    output << "{\"accepted_hill_q\":" << std::setprecision(9) << oldValue[0]
           << ",\"projected_hill_q\":" << newValue[0]
           << ",\"projected_yield_ratio\":" << newValue[1]
           << ",\"projected_yield_gap\":" << newValue[2]
           << ",\"accepted_lambda\":" << oldValue[3]
           << ",\"projected_lambda\":" << newValue[3]
           << ",\"delta_lambda\":" << (newValue[3] - oldValue[3]) << '}';
}

void writeResponseJson(std::ostream& output, const Response& response,
                       const MaterialProgram& material,
                       const std::vector<Parameter>& parameters,
                       const std::vector<float>& accepted) {
    output << "{\"valid\":" << (response.valid ? "true" : "false")
           << ",\"projection_valid\":"
           << (response.projectionValid ? "true" : "false")
           << ",\"projection_residual_inf\":" << std::setprecision(9)
           << response.residual << ",\"projection_state_norm_inf\":"
           << response.stateNorm << ",\"projection_iterations\":"
           << response.iterations << ",\"stage_flags\":" << response.flags
           << ",\"terminal_failure\":" << response.failure
           << ",\"stress_Pa\":";
    writeJsonArray(output, response.stress);
    output << ",\"tangent_action_Pa\":";
    writeJsonArray(output, response.tangent);
    output << ",\"projected_state\":{";
    for (std::size_t i = 0u; i < response.state.size(); ++i) {
        if (i != 0u) output << ',';
        output << '\"' << material.internalState.at(i).name << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << response.state[i];
    }
    output << '}';
    output << ",\"projected_state_low_f32\":{";
    for (std::size_t i = 0u; i < response.stateLow.size(); ++i) {
        if (i != 0u) output << ',';
        output << '\"' << material.internalState.at(i).name << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << response.stateLow[i];
    }
    output << '}';
    output << ",\"yield_branch\":";
    writeYieldBranchJson(output, material, parameters, accepted, response.state);
    output << ",\"damage_history\":{";
    for (unsigned i = 0u; i < 8u; ++i) {
        if (i != 0u) output << ',';
        const auto name = "overlay_fracture_excess_" + std::to_string(i);
        output << '\"' << name << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << stateByName(material, response.state, name);
    }
    output << "}}";
}

void writeDoubleArray(std::ostream& output, const Matrix& values) {
    output << '[';
    for (std::size_t i = 0u; i < values.size(); ++i) {
        if (i != 0u) output << ',';
        output << std::setprecision(17) << values[i];
    }
    output << ']';
}

fp64::State toFP64State(const std::vector<float>& values) {
    fp64::State result;
    result.reserve(values.size());
    for (const float value : values) result.push_back(static_cast<double>(value));
    return result;
}

Matrix exactF32Matrix(const Matrix& values) {
    Matrix result{};
    for (std::size_t i = 0u; i < values.size(); ++i)
        result[i] = static_cast<float>(values[i]);
    return result;
}

Matrix floatLowWords(const Matrix& exact, const Matrix& high) {
    Matrix low{};
    for (std::size_t i = 0u; i < low.size(); ++i) {
        const float highWord = static_cast<float>(high[i]);
        low[i] = static_cast<float>(exact[i] - static_cast<double>(highWord));
    }
    return low;
}

Matrix sumFloatWords(const Matrix& high, const Matrix& low) {
    return fp64::combineFloatWords(high, low);
}

template <class T>
void appendRaw(std::string& output, const std::vector<T>& values) {
    if (!values.empty())
        output.append(reinterpret_cast<const char*>(values.data()),
                      values.size() * sizeof(T));
}

std::string cookedWorldSha256(const CompiledWorld& world) {
    std::string bytes;
    appendRaw(bytes, world.materials);
    appendRaw(bytes, world.parameters);
    appendRaw(bytes, world.scalarPrograms);
    appendRaw(bytes, world.instructions);
    return sha256(bytes);
}

double l2RelativeError(const Matrix& actual, const Matrix& reference) {
    const double denominator = fp64::norm2(reference);
    if (!(denominator > 1.0e-12)) return -1.0;
    return fp64::differenceNorm(actual, reference) / denominator;
}

void writeOptionalRelativeError(std::ostream& output, double error) {
    if (error < 0.0 || !std::isfinite(error)) output << "null";
    else output << std::setprecision(17) << error;
}

void writeOracleResponse(std::ostream& output,
                         const fp64::Projection& projection,
                         const Matrix& stress, const Matrix* tangent,
                         const MaterialProgram& material) {
    output << "{\"projection_residual_inf\":" << std::setprecision(17)
           << projection.residualInf << ",\"projection_iterations\":"
           << projection.iterations << ",\"stress_Pa\":";
    writeDoubleArray(output, stress);
    output << ",\"tangent_action_Pa\":";
    if (tangent != nullptr) writeDoubleArray(output, *tangent);
    else output << "null";
    output << ",\"projected_state\":{";
    for (std::size_t i = 0u; i < projection.state.size(); ++i) {
        if (i != 0u) output << ',';
        output << '"' << material.internalState.at(i).name << "\":"
               << std::setprecision(17) << projection.state[i];
    }
    output << "}}";
}

double hillYieldGap(const MaterialProgram& material,
                    const std::vector<Parameter>& parameters,
                    const fp64::State& state) {
    const auto findStateValue = [&](const std::string& name) {
        for (std::size_t i = 0u; i < material.internalState.size(); ++i)
            if (material.internalState[i].name == name) return state.at(i);
        throw std::runtime_error("oracle missing state " + name);
    };
    const auto p = [&](const std::string& name) {
        return static_cast<double>(static_cast<float>(findParameter(material, name).defaultValue));
    };
    const double s11 = findStateValue("sbar11");
    const double s22 = findStateValue("sbar22");
    const double s33 = findStateValue("sbar33");
    const double s23 = findStateValue("sbar23");
    const double s13 = findStateValue("sbar13");
    const double s12 = findStateValue("sbar12");
    const double q2 = p("hill_F") * std::pow(s22 - s33, 2.0) +
        p("hill_G") * std::pow(s33 - s11, 2.0) +
        p("hill_H") * std::pow(s11 - s22, 2.0) +
        2.0 * p("hill_L") * s23 * s23 +
        2.0 * p("hill_M") * s13 * s13 +
        2.0 * p("hill_N") * s12 * s12 +
        p("q_floor") * p("q_floor");
    double overlayHardening = 0.0;
    for (unsigned i = 0u; i < 6u; ++i)
        overlayHardening += p("overlay_plastic_hardening_" + std::to_string(i)) *
            findStateValue("overlay_plastic_history_" + std::to_string(i));
    for (unsigned i = 0u; i < 3u; ++i)
        overlayHardening += p("overlay_densification_modulus_pa_axis" +
                              std::to_string(i)) *
            findStateValue("overlay_densification_" + std::to_string(i));
    const double ratio = 1.0 + overlayHardening / p("sigma_ref");
    (void)parameters;
    return ratio - std::sqrt(std::max(0.0, q2));
}

fp64::State combineStateWords(const Response& response) {
    require(response.state.size() == response.stateLow.size(),
            "native state high/low arrays have different lengths");
    fp64::State result(response.state.size());
    for (std::size_t i = 0u; i < result.size(); ++i)
        result[i] = static_cast<double>(response.state[i]) +
                    static_cast<double>(response.stateLow[i]);
    return result;
}

std::string hillBranchClass(const MaterialProgram& material,
                            const std::vector<Parameter>& parameters,
                            const fp64::State& accepted,
                            const fp64::State& projected) {
    const auto multiplier = findState(material, "accumulated_multiplier");
    const double deltaLambda = projected.at(multiplier) - accepted.at(multiplier);
    const double gap = hillYieldGap(material, parameters, projected);
    constexpr double activeMultiplierThreshold = 1.0e-7;
    constexpr double branchGapThreshold = 1.0e-4;
    if (deltaLambda > activeMultiplierThreshold &&
        std::abs(gap) <= branchGapThreshold)
        return "active_plastic";
    if (deltaLambda <= activeMultiplierThreshold && gap > branchGapThreshold)
        return "inactive_elastic";
    return "unresolved";
}

fp64::State nativeStateAsFP64(const Response& response) {
    return combineStateWords(response);
}

void writeTangentFailureDiagnostic(const MaterialProgram& material,
                                   const std::string& sourceMaterialSha256,
                                   const std::string& syntheticSetSha256,
                                   const std::string& hardeningDatasetSha256,
                                   const CompiledWorld& world,
                                   const ElementMapping& mapping,
                                   const Matrix& deformation,
                                   const Matrix& deformationLow,
                                   const Matrix& direction,
                                   const std::vector<float>& acceptedState,
                                   const Matrix& plusDeformation,
                                   const Matrix& plusDeformationLow,
                                   const Matrix& minusDeformation,
                                   const Matrix& minusDeformationLow,
                                   const bool compensatedDeformationPerturbations,
                                   const Response& center,
                                   const Response& upper,
                                   const Response& lower,
                                   const double epsilon,
                                   const std::vector<Parameter>& parameters) {
    std::cerr << "retained_damage_native_tangent_diagnostic={\"schema\":\"numi.cardboard.retained-damage-tangent-diagnostic.v1\","
              << "\"material\":\"" << material.name << "\","
              << "\"source_material_sha256\":\"" << sourceMaterialSha256 << "\","
              << "\"synthetic_set_sha256\":\"" << syntheticSetSha256 << "\","
              << "\"hardening_dataset_sha256\":\"" << hardeningDatasetSha256 << "\","
              << "\"evidence_class\":\"explicit_synthetic_instrument_only_not_calibration\","
              << "\"dt_seconds\":0.002,\"epsilon\":" << std::setprecision(17)
              << epsilon << ",\"deformation_F32\":";
    writeJsonArray(std::cerr, deformation);
    std::cerr << ",\"direction_F32\":";
    writeJsonArray(std::cerr, direction);
    std::cerr << ",\"plus_deformation_F32\":";
    writeJsonArray(std::cerr, plusDeformation);
    std::cerr << ",\"minus_deformation_F32\":";
    writeJsonArray(std::cerr, minusDeformation);
    if (compensatedDeformationPerturbations) {
        std::cerr << ",\"deformation_low_F32\":";
        writeJsonArray(std::cerr, deformationLow);
        std::cerr << ",\"plus_deformation_low_F32\":";
        writeJsonArray(std::cerr, plusDeformationLow);
        std::cerr << ",\"minus_deformation_low_F32\":";
        writeJsonArray(std::cerr, minusDeformationLow);
        std::cerr << ",\"F_word_encoding\":\"ordered_high_float32_plus_low_float32\"";
    }
    std::cerr << ",\"accepted_state\":{";
    for (std::size_t i = 0u; i < acceptedState.size(); ++i) {
        if (i != 0u) std::cerr << ',';
        std::cerr << '\"' << material.internalState.at(i).name << "\":"
                  << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << acceptedState[i];
    }
    std::cerr << "},\"parameters_F32\":{";
    for (std::size_t i = 0u; i < parameters.size(); ++i) {
        if (i != 0u) std::cerr << ',';
        std::cerr << '\"' << parameters[i].name << "\":"
                  << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << static_cast<float>(parameters[i].defaultValue);
    }
    const auto& gpuMaterial = world.materials.at(0u);
    std::cerr << "},\"gpu_parameters_F32\":{";
    require(world.parameters.size() == parameters.size(),
            "diagnostic source and cooked parameter arrays differ in length");
    for (std::size_t i = 0u; i < world.parameters.size(); ++i) {
        if (i != 0u) std::cerr << ',';
        std::cerr << '\"' << parameters[i].name << "\":"
                  << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << world.parameters[i].valueAndBounds.x;
    }
    std::cerr << "},\"gpu_material_descriptor\":{";
    std::cerr << "\"parameter_offset\":" << gpuMaterial.parameterOffset
              << ",\"parameter_count\":" << gpuMaterial.parameterCount
              << ",\"state_count\":" << gpuMaterial.stateCount
              << ",\"state_initial_offset\":" << gpuMaterial.stateInitialOffset
              << ",\"state_update_program_offset\":" << gpuMaterial.stateUpdateProgramOffset
              << ",\"implicit_residual_program_offset\":" << gpuMaterial.implicitResidualProgramOffset
              << ",\"implicit_jacobian_program_offset\":" << gpuMaterial.implicitJacobianProgramOffset
              << ",\"implicit_deformation_program_offset\":" << gpuMaterial.implicitDeformationProgramOffset
              << ",\"stress_state_derivative_program_offset\":" << gpuMaterial.stressStateDerivativeProgramOffset
              << ",\"stress_program_offset\":" << gpuMaterial.stressProgramOffset
              << ",\"tangent_program_offset\":" << gpuMaterial.tangentProgramOffset
              << ",\"flags\":" << gpuMaterial.flags
              << ",\"projection_kind\":" << gpuMaterial.projectionKind
              << ",\"local_newton_iterations\":" << gpuMaterial.localNewtonIterations
              << "},\"world_program_storage\":{\"scalar_program_count\":"
              << world.scalarPrograms.size() << ",\"instruction_count\":"
              << world.instructions.size() << "},\"selected_plane_geometry\":{";
    std::cerr << "\"reference_volume_m3\":" << std::setprecision(17)
              << mapping.referenceVolumeM3 << ",\"mode_length_m\":[";
    for (std::size_t i = 0u; i < mapping.modeLengthM.size(); ++i)
        std::cerr << (i == 0u ? "" : ",") << std::setprecision(17)
                  << mapping.modeLengthM[i];
    std::cerr << "],\"mode_onset_strain\":[";
    for (std::size_t i = 0u; i < mapping.modeOnsetStrain.size(); ++i)
        std::cerr << (i == 0u ? "" : ",") << std::setprecision(17)
                  << mapping.modeOnsetStrain[i];
    std::cerr << "],\"mode_softening_strain\":[";
    for (std::size_t i = 0u; i < mapping.modeSofteningStrain.size(); ++i)
        std::cerr << (i == 0u ? "" : ",") << std::setprecision(17)
                  << mapping.modeSofteningStrain[i];
    std::cerr << "]},\"center\":";
    writeResponseJson(std::cerr, center, material, parameters, acceptedState);
    std::cerr << ",\"plus\":";
    writeResponseJson(std::cerr, upper, material, parameters, acceptedState);
    std::cerr << ",\"minus\":";
    writeResponseJson(std::cerr, lower, material, parameters, acceptedState);
    std::cerr << "}\n";
}

class Instrument {
public:
    Instrument() {
        NSString* libraryPath = @NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB;
        metallibPath_ = [libraryPath fileSystemRepresentation];
        metallibSha256_ = sha256File(metallibPath_);
        device_ = MTLCreateSystemDefaultDevice();
        require(device_ != nil, "Metal device is unavailable");
        queue_ = [device_ newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device_ newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB]
            error:&error];
        require(library != nil, "retained-matrix Metal library unavailable");
        id<MTLFunction> function = [library
            newFunctionWithName:@"cardboard_retained_matrix_damage_check"];
        pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        require(pipeline_ != nil, "retained-matrix Metal kernel unavailable");
    }

    Response evaluate(const CompiledWorld& world, const Matrix& deformation,
                      const Matrix& direction,
                      const std::vector<float>& acceptedState) const {
        return evaluate(world, deformation, Matrix{}, direction, acceptedState, false);
    }

    Response evaluate(const CompiledWorld& world,
                      const Matrix& deformationHigh,
                      const Matrix& deformationLow,
                      const Matrix& direction,
                      const std::vector<float>& acceptedState,
                      const bool compensatedDeformation) const {
        require(world.materials.size() == 1u,
                "instrument requires exactly one compiled overlay material");
        require(world.materials[0].stateCount == acceptedState.size(),
                "accepted history does not match overlay state count");
        std::vector<float> inputs;
        inputs.reserve(28u);
        for (const double value : deformationHigh) inputs.push_back(float(value));
        for (const double value : direction) inputs.push_back(float(value));
        for (const double value : deformationLow) inputs.push_back(float(value));
        inputs.push_back(compensatedDeformation ? 1.0f : 0.0f);
        std::vector<float> parameters;
        parameters.reserve(world.parameters.size());
        for (const auto& parameter : world.parameters)
            parameters.push_back(parameter.valueAndBounds.x);
        std::vector<float> output(kOutputCount, 123456.0f);
        std::array<id<MTLBuffer>, 7> buffers{
            makeBuffer(device_, world.materials),
            makeBuffer(device_, world.scalarPrograms),
            makeBuffer(device_, world.instructions),
            makeBuffer(device_, parameters),
            makeBuffer(device_, acceptedState),
            makeBuffer(device_, inputs),
            makeBuffer(device_, output)};

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_];
        for (NSUInteger i = 0u; i < buffers.size(); ++i)
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native material command failed");

        const auto* values = static_cast<const float*>(buffers[6].contents);
        Response result;
        result.valid = values[0] == 1.0f;
        for (std::uint32_t i = 0u; i < 9u; ++i) {
            result.stress[i] = values[1u + i];
            result.tangent[i] = values[10u + i];
        }
        result.state.assign(values + kStateOffset,
                            values + kStateOffset + world.materials[0].stateCount);
        result.stateLow.assign(values + kStateLowOffset,
                               values + kStateLowOffset + world.materials[0].stateCount);
        result.projectionValid = values[kMetadataOffset] == 1.0f;
        result.residual = values[kMetadataOffset + 1u];
        result.stateNorm = values[kMetadataOffset + 2u];
        result.iterations = std::uint32_t(values[kMetadataOffset + 3u]);
        result.flags = std::uint32_t(values[kMetadataOffset + 4u]);
        result.failure = std::uint32_t(values[kMetadataOffset + 5u]);
        return result;
    }

    std::string deviceName() const {
        return [[device_ name] UTF8String];
    }

    const std::string& metallibSha256() const { return metallibSha256_; }
    const std::string& metallibPath() const { return metallibPath_; }

private:
    std::string metallibPath_;
    std::string metallibSha256_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
};

double relativeError(const Matrix& expected, const Matrix& actual) {
    double difference2 = 0.0;
    double reference2 = 0.0;
    for (std::size_t i = 0u; i < expected.size(); ++i) {
        const double delta = expected[i] - actual[i];
        difference2 += delta * delta;
        reference2 += expected[i] * expected[i];
    }
    return std::sqrt(difference2) / std::max(1.0, std::sqrt(reference2));
}

std::string jsonEscape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        if (character == '\n') escaped += "\\n";
        else if (character == '\r') escaped += "\\r";
        else escaped.push_back(character);
    }
    return escaped;
}

double flooredRelativeError(const Matrix& actual, const Matrix& reference) {
    return fp64::differenceNorm(actual, reference) /
        std::max(1.0, fp64::norm2(reference));
}

void writeCaptureFile(const std::filesystem::path& capturePath,
                      const Instrument& instrument,
                      const std::string& executableSha256,
                      const std::string& materialPath,
                      const std::string& hardeningCsvPath,
                      const MaterialProgram& material,
                      const std::string& sourceMaterialSha256,
                      const std::string& syntheticSetSha256,
                      const std::string& hardeningDatasetSha256,
                      const CompiledWorld& world,
                      const ElementMapping& mapping,
                      const Matrix& deformation,
                      const Matrix& direction,
                      const std::vector<float>& acceptedState,
                      const Response& center,
                      const fp64::Projection& oracleCenter,
                      const Matrix& oracleCenterStress,
                      const Matrix& oracleCenterTangent,
                      const std::vector<SweepObservation>& sweep,
                      const bool compensatedDeformationPerturbations,
                      const Matrix& deformationLow,
                      const std::vector<SweepObservation>& compensatedSweep,
                      const std::vector<Parameter>& sourceParameters,
                      double exploratoryError) {
    require(!std::filesystem::exists(capturePath),
            "refusing to overwrite retained native tangent capture: " + capturePath.string());
    std::ofstream output(capturePath, std::ios::binary | std::ios::out);
    require(output.good(), "cannot create retained native tangent capture");
    const auto& gpuMaterial = world.materials.at(0u);
    require(sourceParameters.size() == world.parameters.size(),
            "capture source/cooked parameter arrays differ in length");
    const std::string materialAbsolutePath =
        std::filesystem::absolute(materialPath).lexically_normal().string();
    const std::string csvAbsolutePath =
        std::filesystem::absolute(hardeningCsvPath).lexically_normal().string();

    output << "{\"schema\":\""
           << (compensatedDeformationPerturbations
                   ? "numi.cardboard.retained-damage-capture.v3"
                   : "numi.cardboard.retained-damage-capture.v2")
           << "\","
           << "\"evidence_class\":\"synthetic_instrument_only_not_calibration\","
           << "\"physical_fem_steps\":0,\"physical_calibration\":false,"
           << "\"generic_compensated_implicit_projection\":"
           << (material.genericCompensatedImplicitProjection ? "true" : "false") << ','
           << "\"candidate_state_precision\":\""
           << (material.genericCompensatedImplicitProjection
                   ? "temporary_fp32_high_plus_low; accepted_history_fp32_high"
                   : "fp32_high_only") << "\","
           ;
    if (compensatedDeformationPerturbations)
        output << "\"compensated_deformation_perturbations_enabled\":true,";
    output << "\"source_material\":{\"name\":\"" << jsonEscape(material.name)
           << "\",\"path\":\"" << jsonEscape(materialAbsolutePath)
           << "\",\"sha256\":\"" << sourceMaterialSha256 << "\"},"
           << "\"hardening_dataset\":{\"path\":\"" << jsonEscape(csvAbsolutePath)
           << "\",\"sha256\":\"" << hardeningDatasetSha256 << "\"},"
           << "\"synthetic_parameter_set_sha256\":\"" << syntheticSetSha256 << "\","
           << "\"native_artifacts\":{\"executable_sha256\":\""
           << executableSha256 << "\",\"metallib_path\":\""
           << jsonEscape(std::filesystem::absolute(instrument.metallibPath())
                  .lexically_normal().string())
           << "\",\"metallib_sha256\":\"" << instrument.metallibSha256()
           << "\",\"compiled_world_fingerprint_u64\":" << world.fingerprint
           << ",\"compiled_physics_fingerprint_u64\":" << world.physicsFingerprint
           << ",\"compiled_constitutive_fingerprint_u64\":"
           << world.constitutive.at(0u).fingerprint
           << ",\"cooked_world_arrays_sha256\":\"" << cookedWorldSha256(world)
           << "\",\"scalar_program_count\":" << world.scalarPrograms.size()
           << ",\"instruction_count\":" << world.instructions.size() << "},"
           << "\"compiled_material_bytecode\":{\"scalar_programs\":[";
    for (std::size_t i = 0u; i < world.scalarPrograms.size(); ++i) {
        if (i != 0u) output << ',';
        const auto& program = world.scalarPrograms[i];
        output << '[' << program.firstInstruction << ',' << program.instructionCount
               << ',' << program.maximumStack << ',' << program.flags << ']';
    }
    output << "],\"instructions\":[";
    for (std::size_t i = 0u; i < world.instructions.size(); ++i) {
        if (i != 0u) output << ',';
        const auto& instruction = world.instructions[i];
        output << '[' << instruction.opcode << ',' << instruction.index << ','
               << instruction.integer << ',' << std::setprecision(9)
               << instruction.immediate.x << ']';
    }
    output << "]},\"compile_configuration\":{\"maximum_rate_exponent\":0,"
           << "\"emit_specialized_metal\":false,\"local_material_newton_iterations\":16,"
           << "\"timestep_seconds_f32\":" << std::setprecision(9)
           << static_cast<float>(2.0e-3)
           << ",\"temperature_kelvin_f32\":" << static_cast<float>(293.15)
           << ",\"metal_language\":\"metal4.0\",\"metal_optimization\":\"-O3\","
           << "\"metal_fast_math\":false,\"candidate_state_precision\":\"FP32\","
           << "\"cpu_oracle_arithmetic\":\"FP64 over exact captured FP32 inputs\"},"
           << "\"cooked_material_descriptor\":{\"parameter_offset\":"
           << gpuMaterial.parameterOffset << ",\"parameter_count\":"
           << gpuMaterial.parameterCount << ",\"state_count\":"
           << gpuMaterial.stateCount << ",\"flags\":" << gpuMaterial.flags
           << ",\"projection_kind\":" << gpuMaterial.projectionKind
           << ",\"state_initial_offset\":" << gpuMaterial.stateInitialOffset
           << ",\"state_update_program_offset\":" << gpuMaterial.stateUpdateProgramOffset
           << ",\"implicit_residual_program_offset\":"
           << gpuMaterial.implicitResidualProgramOffset
           << ",\"implicit_jacobian_program_offset\":"
           << gpuMaterial.implicitJacobianProgramOffset
           << ",\"implicit_deformation_program_offset\":"
           << gpuMaterial.implicitDeformationProgramOffset
           << ",\"stress_state_derivative_program_offset\":"
           << gpuMaterial.stressStateDerivativeProgramOffset
           << ",\"stress_program_offset\":" << gpuMaterial.stressProgramOffset
           << ",\"tangent_program_offset\":" << gpuMaterial.tangentProgramOffset
           << ",\"local_newton_iterations\":" << gpuMaterial.localNewtonIterations << "},"
           << "\"source_parameters_f32\":{";
    for (std::size_t i = 0u; i < sourceParameters.size(); ++i) {
        if (i != 0u) output << ',';
        output << '"' << jsonEscape(sourceParameters[i].name) << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << static_cast<float>(sourceParameters[i].defaultValue);
    }
    output << "},\"cooked_parameters_f32\":{";
    for (std::size_t i = 0u; i < world.parameters.size(); ++i) {
        if (i != 0u) output << ',';
        output << '"' << jsonEscape(sourceParameters.at(i).name) << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << world.parameters[i].valueAndBounds.x;
    }
    const Matrix exactF = exactF32Matrix(deformation);
    const Matrix exactDirection = exactF32Matrix(direction);
    output << "},\"reference_volume_m3\":" << std::setprecision(17)
           << mapping.referenceVolumeM3 << ",\"selected_plane_mode_length_m\":[";
    for (std::size_t i = 0u; i < mapping.modeLengthM.size(); ++i)
        output << (i == 0u ? "" : ",") << std::setprecision(17) << mapping.modeLengthM[i];
    output << "],\"accepted_history_f32\":{";
    for (std::size_t i = 0u; i < acceptedState.size(); ++i) {
        if (i != 0u) output << ',';
        output << '"' << jsonEscape(material.internalState.at(i).name) << "\":"
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << acceptedState[i];
    }
    output << "},\"F_f32\":";
    writeJsonArray(output, exactF);
    if (compensatedDeformationPerturbations) {
        output << ",\"F_low_f32\":";
        writeJsonArray(output, deformationLow);
        output << ",\"F_sum_f64\":";
        writeDoubleArray(output, fp64::combineFloatWords(exactF, deformationLow));
        output << ",\"F_word_encoding\":\"ordered_high_float32_plus_low_float32\"";
    }
    output << ",\"direction_f32\":";
    writeJsonArray(output, exactDirection);
    output << ",\"center_native\":";
    writeResponseJson(output, center, material, sourceParameters, acceptedState);
    output << ",\"center_fp64_source_reprojection\":";
    writeOracleResponse(output, oracleCenter, oracleCenterStress,
                        &oracleCenterTangent, material);
    output << ",\"native_vs_fp64_center\":{\"stress_absolute_l2_pa\":"
           << std::setprecision(17) << fp64::differenceNorm(center.stress, oracleCenterStress)
           << ",\"stress_reference_l2_pa\":" << fp64::norm2(oracleCenterStress)
           << ",\"stress_relative_l2\":";
    writeOptionalRelativeError(output, l2RelativeError(center.stress, oracleCenterStress));
    output << ",\"state_absolute_l2\":";
    const auto oracleStateNormError = [&]() {
        double sum = 0.0;
        for (std::size_t i = 0u; i < acceptedState.size(); ++i) {
            const double delta = static_cast<double>(center.state[i]) - oracleCenter.state[i];
            sum += delta * delta;
        }
        return std::sqrt(sum);
    };
    output << oracleStateNormError();
    if (compensatedDeformationPerturbations) {
        const fp64::State candidateState = combineStateWords(center);
        double sum = 0.0;
        for (std::size_t i = 0u; i < acceptedState.size(); ++i) {
            const double delta = candidateState[i] - oracleCenter.state[i];
            sum += delta * delta;
        }
        output << ",\"candidate_high_plus_low_state_absolute_l2\":"
               << std::sqrt(sum);
    }
    output << ",\"tangent_absolute_l2_pa\":"
           << fp64::differenceNorm(center.tangent, oracleCenterTangent)
           << ",\"tangent_reference_l2_pa\":" << fp64::norm2(oracleCenterTangent)
           << ",\"tangent_relative_l2\":";
    writeOptionalRelativeError(output, l2RelativeError(center.tangent, oracleCenterTangent));
    output << "},"
           << "\"tangent_error_gates\":{\"existing_exploratory_scaled_error\":"
           << exploratoryError << ",\"exploratory_threshold\":0.03,"
           << "\"exploratory_pass\":" << (exploratoryError < 3.0e-2 ? "true" : "false")
           << ",\"software_qualification_threshold_1e-3\":0.001,"
           << "\"software_qualification_pass_at_reference_epsilon\":"
           << (exploratoryError <= 1.0e-3 ? "true" : "false") << "},\"perturbation_sweep\":[";
    for (std::size_t index = 0u; index < sweep.size(); ++index) {
        const auto& row = sweep[index];
        if (index != 0u) output << ',';
        output << "{\"requested_epsilon\":" << std::setprecision(17)
               << row.requestedEpsilon << ",\"plus_F_f32\":";
        writeJsonArray(output, row.plusF);
        output << ",\"minus_F_f32\":";
        writeJsonArray(output, row.minusF);
        output << ",\"realized_direction\":";
        writeDoubleArray(output, row.realizedDirection);
        output << ",\"direction_realization_relative_l2\":"
               << row.directionRealizationRelativeL2
               << ",\"midpoint_offset_inf_f\":" << row.midpointOffsetInf;
        output << ",\"native_pair_valid\":"
               << (row.nativePairValid ? "true" : "false")
               << ",\"native_plus\":";
        if (row.plusNative.valid)
            writeResponseJson(output, row.plusNative, material, sourceParameters, acceptedState);
        else output << "null";
        output << ",\"native_minus\":";
        if (row.minusNative.valid)
            writeResponseJson(output, row.minusNative, material, sourceParameters, acceptedState);
        else output << "null";
        output << ",\"oracle_pair_valid\":"
               << (row.oraclePairValid ? "true" : "false")
               << ",\"oracle_failure\":\"" << jsonEscape(row.oracleFailure) << "\""
               << ",\"fp64_plus\":";
        if (row.oraclePairValid)
            writeOracleResponse(output, row.plusOracle, row.plusOracleStress, nullptr, material);
        else output << "null";
        output << ",\"fp64_minus\":";
        if (row.oraclePairValid)
            writeOracleResponse(output, row.minusOracle, row.minusOracleStress, nullptr, material);
        else output << "null";
        output << ",\"native_fd_action_pa\":";
        if (row.nativePairValid) writeDoubleArray(output, row.nativeFiniteDifference);
        else output << "null";
        output << ",\"fp64_fd_action_pa\":";
        if (row.oraclePairValid) writeDoubleArray(output, row.oracleFiniteDifference);
        else output << "null";
        output << ",\"native_ad_action_at_realized_direction_pa\":";
        if (row.nativePairValid)
            writeDoubleArray(output, row.nativeTangentAtRealizedDirection);
        else output << "null";
        output << ",\"fp64_ad_action_at_realized_direction_pa\":";
        if (row.oraclePairValid)
            writeDoubleArray(output, row.oracleTangentAtRealizedDirection);
        else output << "null";
        output << ",\"native_ad_vs_native_fd_scaled_error\":"
               << (row.nativePairValid ? flooredRelativeError(
                    row.nativeTangentAtRealizedDirection,
                    row.nativeFiniteDifference) : -1.0)
               << ",\"native_ad_vs_fp64_fd_scaled_error\":"
               << (row.oraclePairValid ? flooredRelativeError(
                    row.nativeTangentAtRealizedDirection,
                    row.oracleFiniteDifference) : -1.0)
               << ",\"fp64_ad_vs_fp64_fd_scaled_error\":"
               << (row.oraclePairValid ? flooredRelativeError(
                    row.oracleTangentAtRealizedDirection,
                    row.oracleFiniteDifference) : -1.0)
               << ",\"native_ad_vs_fp64_ad_relative_l2\":";
        if (row.oraclePairValid)
            writeOptionalRelativeError(output, l2RelativeError(
                row.nativeTangentAtRealizedDirection,
                row.oracleTangentAtRealizedDirection));
        else output << "null";
        output << ",\"native_ad_vs_fp64_fd_relative_l2\":";
        if (row.oraclePairValid)
            writeOptionalRelativeError(output, l2RelativeError(
                row.nativeTangentAtRealizedDirection,
                row.oracleFiniteDifference));
        else output << "null";
        output << ",\"native_ad_reference_l2_pa\":"
               << (row.nativePairValid ? fp64::norm2(row.nativeTangentAtRealizedDirection) : 0.0)
               << ",\"fp64_fd_reference_l2_pa\":"
               << (row.oraclePairValid ? fp64::norm2(row.oracleFiniteDifference) : 0.0)
               << ",\"plus_minus_fp64_yield_gap\":";
        if (row.oraclePairValid)
            output << '[' << hillYieldGap(material, sourceParameters, row.plusOracle.state)
                   << ',' << hillYieldGap(material, sourceParameters, row.minusOracle.state) << ']';
        else output << "null";
        output << ",\"plus_minus_lambda_f32\":";
        if (row.nativePairValid) output << '['
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << stateByName(material, row.plusNative.state, "accumulated_multiplier") << ','
               << stateByName(material, row.minusNative.state, "accumulated_multiplier")
               << ']';
        else output << "null";
        output << ",\"plus_minus_damage_history_f32\":";
        if (row.nativePairValid) output << '['
               << stateByName(material, row.plusNative.state, "overlay_fracture_excess_0") << ','
               << stateByName(material, row.minusNative.state, "overlay_fracture_excess_0")
               << ']';
        else output << "null";
        output << '}';
    }
    output << ']';
    if (compensatedDeformationPerturbations) {
        output << ",\"compensated_deformation_perturbation_sweep\":[";
        for (std::size_t index = 0u; index < compensatedSweep.size(); ++index) {
            const auto& row = compensatedSweep[index];
            if (index != 0u) output << ',';
            output << "{\"requested_epsilon\":" << std::setprecision(17)
                   << row.requestedEpsilon << ",\"plus_F_high_f32\":";
            writeJsonArray(output, row.plusFHigh);
            output << ",\"plus_F_low_f32\":";
            writeJsonArray(output, row.plusFLow);
            output << ",\"plus_F_sum_f64\":";
            writeDoubleArray(output, row.plusF);
            output << ",\"minus_F_high_f32\":";
            writeJsonArray(output, row.minusFHigh);
            output << ",\"minus_F_low_f32\":";
            writeJsonArray(output, row.minusFLow);
            output << ",\"minus_F_sum_f64\":";
            writeDoubleArray(output, row.minusF);
            output << ",\"requested_direction_f32\":";
            writeJsonArray(output, exactDirection);
            output << ",\"realized_direction_f64\":";
            writeDoubleArray(output, row.realizedDirection);
            output << ",\"direction_realization_relative_l2\":"
                   << std::setprecision(17) << row.directionRealizationRelativeL2
                   << ",\"midpoint_offset_inf_f\":" << row.midpointOffsetInf
                   << ",\"native_pair_valid\":"
                   << (row.nativePairValid ? "true" : "false")
                   << ",\"native_plus\":";
            if (row.plusNative.valid)
                writeResponseJson(output, row.plusNative, material, sourceParameters,
                                  acceptedState);
            else output << "null";
            output << ",\"native_minus\":";
            if (row.minusNative.valid)
                writeResponseJson(output, row.minusNative, material, sourceParameters,
                                  acceptedState);
            else output << "null";
            output << ",\"oracle_pair_valid\":"
                   << (row.oraclePairValid ? "true" : "false")
                   << ",\"oracle_failure\":\"" << jsonEscape(row.oracleFailure)
                   << "\",\"fp64_plus\":";
            if (row.oraclePairValid)
                writeOracleResponse(output, row.plusOracle, row.plusOracleStress,
                                    nullptr, material);
            else output << "null";
            output << ",\"fp64_minus\":";
            if (row.oraclePairValid)
                writeOracleResponse(output, row.minusOracle, row.minusOracleStress,
                                    nullptr, material);
            else output << "null";
            output << ",\"native_fd_action_pa_per_s\":";
            if (row.nativePairValid) writeDoubleArray(output, row.nativeFiniteDifference);
            else output << "null";
            output << ",\"fp64_fd_action_pa_per_s\":";
            if (row.oraclePairValid) writeDoubleArray(output, row.oracleFiniteDifference);
            else output << "null";
            output << ",\"native_ad_action_at_requested_direction_pa_per_s\":";
            if (row.nativePairValid)
                writeDoubleArray(output, row.nativeTangentAtRealizedDirection);
            else output << "null";
            output << ",\"fp64_ad_action_at_requested_direction_pa_per_s\":";
            if (row.oraclePairValid)
                writeDoubleArray(output, row.oracleTangentAtRealizedDirection);
            else output << "null";
            output << ",\"native_ad_vs_fp64_fd_relative_l2\":";
            if (row.oraclePairValid)
                writeOptionalRelativeError(output, l2RelativeError(
                    row.nativeTangentAtRealizedDirection, row.oracleFiniteDifference));
            else output << "null";
            output << ",\"native_fd_vs_fp64_fd_relative_l2\":";
            if (row.oraclePairValid && row.nativePairValid)
                writeOptionalRelativeError(output, l2RelativeError(
                    row.nativeFiniteDifference, row.oracleFiniteDifference));
            else output << "null";
            output << ",\"fp64_ad_vs_fp64_fd_relative_l2\":";
            if (row.oraclePairValid)
                writeOptionalRelativeError(output, l2RelativeError(
                    row.oracleTangentAtRealizedDirection, row.oracleFiniteDifference));
            else output << "null";
            const double nativeAdOracleFdError = row.oraclePairValid
                ? l2RelativeError(row.nativeTangentAtRealizedDirection,
                                  row.oracleFiniteDifference) : -1.0;
            const double oracleAdFdError = row.oraclePairValid
                ? l2RelativeError(row.oracleTangentAtRealizedDirection,
                                  row.oracleFiniteDifference) : -1.0;
            const bool requestedDirectionGate =
                row.directionRealizationRelativeL2 <= 1.0e-3;
            const bool localDerivativeGate = row.nativePairValid && row.oraclePairValid &&
                nativeAdOracleFdError >= 0.0 && nativeAdOracleFdError <= 1.0e-3;
            const bool oracleStepGate = row.oraclePairValid && oracleAdFdError >= 0.0 &&
                oracleAdFdError <= 1.0e-3;
            output << ",\"oracle_center_hill_branch\":\""
                   << row.oracleCenterBranch << "\",\"oracle_plus_hill_branch\":\""
                   << row.oraclePlusBranch << "\",\"oracle_minus_hill_branch\":\""
                   << row.oracleMinusBranch << "\",\"oracle_branch_preserved\":"
                   << (row.oracleBranchPreserved ? "true" : "false")
                   << ",\"native_center_hill_branch\":\""
                   << row.nativeCenterBranch << "\",\"native_plus_hill_branch\":\""
                   << row.nativePlusBranch << "\",\"native_minus_hill_branch\":\""
                   << row.nativeMinusBranch << "\",\"native_branch_matches_fp64\":"
                   << (row.nativeBranchMatchesOracle ? "true" : "false")
                   << ",\"requested_direction_gate_1e_3\":"
                   << (requestedDirectionGate ? "true" : "false")
                   << ",\"fp64_fd_step_check_gate_1e_3\":"
                   << (oracleStepGate ? "true" : "false")
                   << ",\"native_ad_vs_fp64_fd_gate_1e_3\":"
                   << (localDerivativeGate ? "true" : "false")
                   << ",\"requested_direction_branch_and_tangent_gate_1e_3\":"
                   << (requestedDirectionGate && row.oracleBranchPreserved &&
                       row.nativeBranchMatchesOracle && oracleStepGate &&
                       localDerivativeGate ? "true" : "false");
            output << ",\"plus_minus_fp64_yield_gap\":";
            if (row.oraclePairValid)
                output << '[' << hillYieldGap(material, sourceParameters,
                                               row.plusOracle.state)
                       << ',' << hillYieldGap(material, sourceParameters,
                                               row.minusOracle.state) << ']';
            else output << "null";
            output << ",\"plus_minus_lambda_f32\":";
            if (row.nativePairValid)
                output << '[' << std::setprecision(
                    std::numeric_limits<float>::max_digits10)
                       << stateByName(material, row.plusNative.state,
                                      "accumulated_multiplier") << ','
                       << stateByName(material, row.minusNative.state,
                                      "accumulated_multiplier") << ']';
            else output << "null";
            output << '}';
        }
        output << ']';
    }
    output << "}\n";
    output.flush();
    require(output.good(), "failed writing retained native tangent capture");
}

double maxAbs(const Matrix& values) {
    double result = 0.0;
    for (const double value : values) result = std::max(result, std::abs(value));
    return result;
}

void checkMaterial(Instrument& instrument, const std::string& materialPath,
                   const std::string& hardeningCsvPath,
                   const std::string& hardeningCsvSha,
                   const std::filesystem::path& captureDirectory,
                   const std::string& executableSha256,
                   const bool compensatedImplicitProjection,
                   const bool compensatedDeformationPerturbations) {
    const auto parsed = parseMatterFile(materialPath);
    require(parsed.succeeded(), "could not parse source matrix: " +
                                diagnostics(parsed.diagnostics));
    const auto request = makeRequest(parsed.material, materialPath, hardeningCsvSha,
                                     parsed.material.name);
    const auto built = buildElementProgram(request);
    MaterialProgram instrumentMaterial = built.program;
    instrumentMaterial.genericCompensatedImplicitProjection =
        compensatedImplicitProjection;
    const auto world = compileInstrumentWorld(instrumentMaterial);
    require(world.materials[0].stateCount <= NM_MAX_MATERIAL_STATE,
            "overlay state count exceeds the production state arena");
    std::vector<float> accepted = world.stateInitials;
    require(accepted.size() == world.materials[0].stateCount,
            "compiled initial history has an unexpected shape");

    Matrix direction{};
    direction[0] = 1.0;
    direction[4] = 0.15;
    direction[1] = 0.12;
    Matrix deformation = identity;
    const std::uint32_t damageIndex = findState(
        instrumentMaterial, "overlay_fracture_excess_0");
    const std::uint32_t multiplierIndex = findState(
        instrumentMaterial, "accumulated_multiplier");

    Response previous;
    const std::array<double, 4> stretch{{1.0010, 1.0020, 1.0030, 1.0038}};
    double previousDamage = 0.0;
    std::vector<float> acceptedBeforeTangent;
    for (std::size_t step = 0u; step < stretch.size(); ++step) {
        deformation = identity;
        deformation[0] = stretch[step];
        if (step + 1u == stretch.size()) acceptedBeforeTangent = accepted;
        const Matrix stepFHigh = exactF32Matrix(deformation);
        const Matrix stepFLow = compensatedDeformationPerturbations
            ? floatLowWords(deformation, stepFHigh) : Matrix{};
        const auto response = instrument.evaluate(
            world, stepFHigh, stepFLow, exactF32Matrix(direction), accepted,
            compensatedDeformationPerturbations);
        require(response.valid && response.projectionValid,
                parsed.material.name + " projection failed at load step " +
                std::to_string(step) + " residual=" + std::to_string(response.residual) +
                " failure=" + std::to_string(response.failure));
        const double damage = response.state[damageIndex];
        require(std::isfinite(damage) && damage + 1.0e-7 >= previousDamage,
                "fracture excess history decreased during monotone loading");
        require(std::isfinite(response.state[multiplierIndex]) &&
                std::isfinite(maxAbs(response.stress)) &&
                std::isfinite(maxAbs(response.tangent)),
                "nonfinite native stress, tangent, or plastic multiplier");
        previousDamage = damage;
        previous = response;
        accepted = response.state;
    }
    require(previousDamage > 1.0e-7,
            parsed.material.name + " synthetic mode-0 damage did not activate");

    deformation = identity;
    deformation[0] = stretch.back();
    const Matrix centerFHigh = exactF32Matrix(deformation);
    const Matrix centerFLow = compensatedDeformationPerturbations
        ? floatLowWords(deformation, centerFHigh) : Matrix{};
    const Matrix exactDirection = exactF32Matrix(direction);
    const Matrix centerF = fp64::combineFloatWords(centerFHigh, centerFLow);
    const auto center = instrument.evaluate(world, centerFHigh, centerFLow,
        exactDirection, acceptedBeforeTangent, compensatedDeformationPerturbations);
    require(center.valid && center.projectionValid,
            "center tangent evaluation rejected its projected state");
    constexpr double epsilon = 1.0e-5;
    Matrix plus = deformation;
    Matrix minus = deformation;
    Matrix plusHigh{}, plusLow{}, minusHigh{}, minusLow{};
    Response upper, lower;
    if (compensatedDeformationPerturbations) {
        for (std::size_t i = 0u; i < deformation.size(); ++i) {
            const auto upperWords = fp64::perturbFloatWords(
                static_cast<float>(centerFHigh[i]), static_cast<float>(centerFLow[i]),
                static_cast<float>(exactDirection[i]), epsilon, 1.0);
            const auto lowerWords = fp64::perturbFloatWords(
                static_cast<float>(centerFHigh[i]), static_cast<float>(centerFLow[i]),
                static_cast<float>(exactDirection[i]), epsilon, -1.0);
            plusHigh[i] = upperWords.high;
            plusLow[i] = upperWords.low;
            minusHigh[i] = lowerWords.high;
            minusLow[i] = lowerWords.low;
        }
        plus = fp64::combineFloatWords(plusHigh, plusLow);
        minus = fp64::combineFloatWords(minusHigh, minusLow);
        upper = instrument.evaluate(world, plusHigh, plusLow, exactDirection,
                                    acceptedBeforeTangent, true);
        lower = instrument.evaluate(world, minusHigh, minusLow, exactDirection,
                                    acceptedBeforeTangent, true);
    } else {
        for (std::size_t i = 0u; i < deformation.size(); ++i) {
            plus[i] += epsilon * direction[i];
            minus[i] -= epsilon * direction[i];
        }
        upper = instrument.evaluate(world, plus, direction, acceptedBeforeTangent);
        lower = instrument.evaluate(world, minus, direction, acceptedBeforeTangent);
    }
    require(upper.valid && lower.valid && upper.projectionValid && lower.projectionValid,
            "native centered-tangent probes did not converge");
    Matrix finiteDifference{};
    for (std::size_t i = 0u; i < finiteDifference.size(); ++i)
        finiteDifference[i] = (upper.stress[i] - lower.stress[i]) / (2.0 * epsilon);
    const double tangentError = relativeError(finiteDifference, center.tangent);
    if (!std::isfinite(tangentError) || tangentError >= 3.0e-2) {
        writeTangentFailureDiagnostic(instrumentMaterial, request.sourceBinding.sha256,
            request.explicitSet.sha256,
            request.explicitSet.hardeningBinding.datasetSha256,
            world, built.mapping,
            deformation, centerFLow, direction,
            acceptedBeforeTangent, plus, plusLow, minus, minusLow,
            compensatedDeformationPerturbations, center, upper, lower, epsilon,
            built.program.parameters);
    }

    if (!captureDirectory.empty()) {
        const Matrix exactF = centerF;
        const Matrix legacyExactF = exactF32Matrix(deformation);
        const fp64::State acceptedFP64 = toFP64State(acceptedBeforeTangent);
        constexpr double nativeDt = static_cast<double>(static_cast<float>(2.0e-3));
        const auto& compiled = world.constitutive.at(0u);
        const auto legacyOracleCenter = fp64::project(
            compiled, legacyExactF, acceptedFP64, nativeDt);
        const auto oracleCenter = fp64::project(
            compiled, exactF, acceptedFP64, nativeDt);
        const Matrix oracleCenterStress = fp64::stress(
            compiled, exactF, acceptedFP64, oracleCenter.state, nativeDt);
        const Matrix oracleCenterTangent = fp64::tangentAction(
            compiled, exactF, acceptedFP64, oracleCenter.state,
            exactDirection, nativeDt);

        const std::array<double, 8> epsilons{{1.0e-3, 3.0e-4, 1.0e-4,
            3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7}};
        std::vector<SweepObservation> sweep;
        std::vector<SweepObservation> compensatedSweep;
        sweep.reserve(epsilons.size());
        if (compensatedDeformationPerturbations)
            compensatedSweep.reserve(epsilons.size());
        for (const double epsilon : epsilons) {
            SweepObservation row;
            row.requestedEpsilon = epsilon;
            for (std::size_t i = 0u; i < legacyExactF.size(); ++i) {
                row.plusF[i] = static_cast<float>(legacyExactF[i] + epsilon * exactDirection[i]);
                row.minusF[i] = static_cast<float>(legacyExactF[i] - epsilon * exactDirection[i]);
                row.realizedDirection[i] =
                    (row.plusF[i] - row.minusF[i]) / (2.0 * epsilon);
            }
            double directionDifference2 = 0.0;
            double requestedDirection2 = 0.0;
            for (std::size_t i = 0u; i < legacyExactF.size(); ++i) {
                const double difference = row.realizedDirection[i] - exactDirection[i];
                directionDifference2 += difference * difference;
                requestedDirection2 += exactDirection[i] * exactDirection[i];
                row.midpointOffsetInf = std::max(row.midpointOffsetInf,
                    std::abs(0.5 * (row.plusF[i] + row.minusF[i]) - legacyExactF[i]));
            }
            row.directionRealizationRelativeL2 = std::sqrt(directionDifference2) /
                std::max(std::numeric_limits<double>::min(), std::sqrt(requestedDirection2));
            row.plusNative = instrument.evaluate(world, row.plusF,
                                                   exactDirection, acceptedBeforeTangent);
            row.minusNative = instrument.evaluate(world, row.minusF,
                                                    exactDirection, acceptedBeforeTangent);
            row.nativePairValid = row.plusNative.valid && row.minusNative.valid &&
                row.plusNative.projectionValid && row.minusNative.projectionValid;
            if (row.nativePairValid) {
                const auto matched = instrument.evaluate(
                    world, legacyExactF, row.realizedDirection, acceptedBeforeTangent);
                if (!matched.valid || !matched.projectionValid) row.nativePairValid = false;
                else row.nativeTangentAtRealizedDirection = matched.tangent;
            }
            if (row.nativePairValid) {
                for (std::size_t i = 0u; i < row.nativeFiniteDifference.size(); ++i)
                    row.nativeFiniteDifference[i] =
                        (row.plusNative.stress[i] - row.minusNative.stress[i]) /
                        (2.0 * epsilon);
            }
            try {
                row.plusOracle = fp64::project(
                    compiled, row.plusF, acceptedFP64, nativeDt);
                row.minusOracle = fp64::project(
                    compiled, row.minusF, acceptedFP64, nativeDt);
                row.plusOracleStress = fp64::stress(
                    compiled, row.plusF, acceptedFP64, row.plusOracle.state, nativeDt);
                row.minusOracleStress = fp64::stress(
                    compiled, row.minusF, acceptedFP64, row.minusOracle.state, nativeDt);
                row.oracleTangentAtRealizedDirection = fp64::tangentAction(
                    compiled, legacyExactF, acceptedFP64, legacyOracleCenter.state,
                    row.realizedDirection, nativeDt);
                for (std::size_t i = 0u; i < row.oracleFiniteDifference.size(); ++i)
                    row.oracleFiniteDifference[i] =
                        (row.plusOracleStress[i] - row.minusOracleStress[i]) /
                        (2.0 * epsilon);
                row.oraclePairValid = true;
            } catch (const std::exception& error) {
                row.oracleFailure = error.what();
            }
            sweep.push_back(std::move(row));
        }
        if (compensatedDeformationPerturbations) {
            require(compensatedImplicitProjection,
                    "compensated deformation perturbations require the generic compensated projection mode");
            for (const double epsilon : epsilons) {
                SweepObservation row;
                row.requestedEpsilon = epsilon;
                for (std::size_t i = 0u; i < exactF.size(); ++i) {
                    const auto upperWords = fp64::perturbFloatWords(
                        static_cast<float>(centerFHigh[i]),
                        static_cast<float>(centerFLow[i]),
                        static_cast<float>(exactDirection[i]), epsilon, 1.0);
                    const auto lowerWords = fp64::perturbFloatWords(
                        static_cast<float>(centerFHigh[i]),
                        static_cast<float>(centerFLow[i]),
                        static_cast<float>(exactDirection[i]), epsilon, -1.0);
                    row.plusFHigh[i] = upperWords.high;
                    row.plusFLow[i] = upperWords.low;
                    row.minusFHigh[i] = lowerWords.high;
                    row.minusFLow[i] = lowerWords.low;
                }
                row.plusF = sumFloatWords(row.plusFHigh, row.plusFLow);
                row.minusF = sumFloatWords(row.minusFHigh, row.minusFLow);
                double directionDifference2 = 0.0;
                double requestedDirection2 = 0.0;
                for (std::size_t i = 0u; i < exactDirection.size(); ++i) {
                    row.realizedDirection[i] =
                        (row.plusF[i] - row.minusF[i]) / (2.0 * epsilon);
                    const double difference =
                        row.realizedDirection[i] - exactDirection[i];
                    directionDifference2 += difference * difference;
                    requestedDirection2 += exactDirection[i] * exactDirection[i];
                    row.midpointOffsetInf = std::max(row.midpointOffsetInf,
                        std::abs(0.5 * (row.plusF[i] + row.minusF[i]) - exactF[i]));
                }
                row.directionRealizationRelativeL2 = std::sqrt(directionDifference2) /
                    std::max(std::numeric_limits<double>::min(),
                             std::sqrt(requestedDirection2));
                row.plusNative = instrument.evaluate(world, row.plusFHigh,
                    row.plusFLow, exactDirection, acceptedBeforeTangent, true);
                row.minusNative = instrument.evaluate(world, row.minusFHigh,
                    row.minusFLow, exactDirection, acceptedBeforeTangent, true);
                row.nativePairValid = row.plusNative.valid && row.minusNative.valid &&
                    row.plusNative.projectionValid && row.minusNative.projectionValid;
                if (row.nativePairValid) {
                    for (std::size_t i = 0u; i < row.nativeFiniteDifference.size(); ++i)
                        row.nativeFiniteDifference[i] =
                            (row.plusNative.stress[i] - row.minusNative.stress[i]) /
                            (2.0 * epsilon);
                    // This is the production AD action at the requested
                    // captured direction and center high-plus-low F.
                    row.nativeTangentAtRealizedDirection = center.tangent;
                }
                try {
                    row.plusOracle = fp64::project(
                        compiled, row.plusF, acceptedFP64, nativeDt);
                    row.minusOracle = fp64::project(
                        compiled, row.minusF, acceptedFP64, nativeDt);
                    row.plusOracleStress = fp64::stress(
                        compiled, row.plusF, acceptedFP64,
                        row.plusOracle.state, nativeDt);
                    row.minusOracleStress = fp64::stress(
                        compiled, row.minusF, acceptedFP64,
                        row.minusOracle.state, nativeDt);
                    row.oracleTangentAtRealizedDirection = oracleCenterTangent;
                    for (std::size_t i = 0u; i < row.oracleFiniteDifference.size(); ++i)
                        row.oracleFiniteDifference[i] =
                            (row.plusOracleStress[i] - row.minusOracleStress[i]) /
                            (2.0 * epsilon);
                    row.oraclePairValid = true;
                } catch (const std::exception& error) {
                    row.oracleFailure = error.what();
                }
                row.oracleCenterBranch = hillBranchClass(
                    instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                    oracleCenter.state);
                row.nativeCenterBranch = hillBranchClass(
                    instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                    nativeStateAsFP64(center));
                if (row.oraclePairValid) {
                    row.oraclePlusBranch = hillBranchClass(
                        instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                        row.plusOracle.state);
                    row.oracleMinusBranch = hillBranchClass(
                        instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                        row.minusOracle.state);
                    row.oracleBranchPreserved =
                        row.oracleCenterBranch != "unresolved" &&
                        row.oraclePlusBranch == row.oracleCenterBranch &&
                        row.oracleMinusBranch == row.oracleCenterBranch;
                }
                if (row.nativePairValid) {
                    row.nativePlusBranch = hillBranchClass(
                        instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                        nativeStateAsFP64(row.plusNative));
                    row.nativeMinusBranch = hillBranchClass(
                        instrumentMaterial, instrumentMaterial.parameters, acceptedFP64,
                        nativeStateAsFP64(row.minusNative));
                    row.nativeBranchMatchesOracle = row.oraclePairValid &&
                        row.nativeCenterBranch == row.oracleCenterBranch &&
                        row.nativePlusBranch == row.oraclePlusBranch &&
                        row.nativeMinusBranch == row.oracleMinusBranch;
                }
                compensatedSweep.push_back(std::move(row));
            }
        }
        const std::filesystem::path capturePath = captureDirectory /
            (parsed.material.name + ".tangent-capture.json");
        writeCaptureFile(capturePath, instrument, executableSha256,
            materialPath, hardeningCsvPath, instrumentMaterial,
            request.sourceBinding.sha256, request.explicitSet.sha256,
            request.explicitSet.hardeningBinding.datasetSha256,
            world, built.mapping, deformation, direction, acceptedBeforeTangent,
            center, oracleCenter, oracleCenterStress, oracleCenterTangent, sweep,
            compensatedDeformationPerturbations, centerFLow, compensatedSweep,
            instrumentMaterial.parameters, tangentError);
        std::cout << "native_tangent_capture=" << capturePath.string()
                  << " fp64_center_stress_relative_l2="
                  << l2RelativeError(center.stress, oracleCenterStress)
                  << " fp64_center_tangent_relative_l2="
                  << l2RelativeError(center.tangent, oracleCenterTangent)
                  << " qualification_gate_1e-3="
                  << (tangentError <= 1.0e-3 ? "pass" : "not_passed") << '\n';
    }
    require(std::isfinite(tangentError) && tangentError < 3.0e-2,
            "native full-history tangent differs from centered production reprojection: " +
            std::to_string(tangentError));

    const auto unloaded = instrument.evaluate(world, identity, direction, accepted);
    require(unloaded.valid && unloaded.projectionValid,
            "native unload projection failed");
    require(unloaded.state[damageIndex] + 1.0e-7 >= previousDamage,
            "irreversible fracture history decreased on unload");

    std::cout << "material=" << parsed.material.name
              << " source_file_sha256=" << request.sourceBinding.sha256
              << " synthetic_set_sha256=" << request.explicitSet.sha256
              << " state_count=" << world.materials[0].stateCount
              << " damage_history=" << previousDamage
              << " hill_multiplier=" << previous.state[multiplierIndex]
              << " stress_inf_pa=" << maxAbs(previous.stress)
              << " tangent_fd_relative_error=" << tangentError
              << " residual_inf=" << previous.residual
              << " projection_iterations=" << previous.iterations << '\n';
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::vector<std::string> positional;
            std::filesystem::path captureDirectory;
            bool compensatedImplicitProjection = false;
            bool compensatedDeformationPerturbations = false;
            for (int index = 1; index < argc; ++index) {
                const std::string argument = argv[index];
                if (argument == "--capture-dir") {
                    require(index + 1 < argc, "--capture-dir requires a directory path");
                    captureDirectory = argv[++index];
                } else if (argument == "--compensated-implicit-projection") {
                    compensatedImplicitProjection = true;
                } else if (argument == "--compensated-deformation-perturbations") {
                    require(!compensatedDeformationPerturbations,
                            "duplicate --compensated-deformation-perturbations option");
                    compensatedDeformationPerturbations = true;
                } else {
                    positional.push_back(argument);
                }
            }
            require(positional.size() == 3u,
                "usage: check LINER.nmatter MEDIUM.nmatter HAJ_ALI_TENSION_FIT.csv "
                "[--capture-dir DIR] [--compensated-implicit-projection] "
                "[--compensated-deformation-perturbations]");
            require(!compensatedDeformationPerturbations ||
                    compensatedImplicitProjection,
                    "--compensated-deformation-perturbations requires "
                    "--compensated-implicit-projection");
            require(!compensatedDeformationPerturbations || !captureDirectory.empty(),
                    "--compensated-deformation-perturbations requires --capture-dir");
            if (!captureDirectory.empty())
                std::filesystem::create_directories(captureDirectory);
            const std::string hardeningCsvSha = sha256File(positional[2]);
            const std::string executableSha256 = sha256File(argv[0]);
            Instrument instrument;
            std::cout << "device=" << instrument.deviceName()
                      << " path=production_projection_stress_and_consistent_tangent"
                      << " damage_set=explicit_synthetic_instrument_only"
                      << " generic_compensated_implicit_projection="
                      << (compensatedImplicitProjection ? "enabled" : "disabled")
                      << " compensated_deformation_perturbations="
                      << (compensatedDeformationPerturbations ? "enabled" : "disabled")
                      << " local_projection_tolerance="
                      << (compensatedImplicitProjection ? "2e-8" : "2e-5")
                      << " candidate_history="
                      << (compensatedImplicitProjection
                              ? "temporary_fp32_high_low_accepted_fp32_high"
                              : "fp32_high_only")
                      << " physical_fem_steps=0 calibration=false"
                      << " capture=" << (captureDirectory.empty() ? "disabled" : "enabled")
                      << " executable_sha256=" << executableSha256
                      << " metallib_sha256=" << instrument.metallibSha256() << '\n';
            checkMaterial(instrument, positional[0], positional[2], hardeningCsvSha,
                          captureDirectory, executableSha256,
                          compensatedImplicitProjection,
                          compensatedDeformationPerturbations);
            checkMaterial(instrument, positional[1], positional[2], hardeningCsvSha,
                          captureDirectory, executableSha256,
                          compensatedImplicitProjection,
                          compensatedDeformationPerturbations);
            std::cout << "retained_matrix_damage_native_instrument_passed"
                      << " physical_validation=false\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "retained matrix Metal instrument failure: "
                      << error.what() << '\n';
            return 1;
        }
    }
}
