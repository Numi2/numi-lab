#include "cardboard_solver_reference.hpp"
#include "cardboard_paper_projection_scaling.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/paper_compensated_bytecode.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace paper = numi_cardboard_paper;
namespace oracle = numi_cardboard_solver;

namespace {
constexpr std::size_t kParameterCount = 20;
constexpr std::size_t kStateCount = 13;
constexpr std::array<const char*, kParameterCount> kParameterNames{
    "density", "c11", "c22", "c33", "c12", "c13", "c23", "g12", "g13",
    "g23", "sigma_ref", "hill_F", "hill_G", "hill_H", "hill_L", "hill_M",
    "hill_N", "q_floor", "fb_epsilon", "residual_scale"};

struct CompiledSourceMaterial {
    std::vector<NMScalarProgramGPU> residualPrograms;
    std::vector<NMExpressionInstructionGPU> instructions;
    std::array<double, kStateCount> rowScales{};
    std::string name;
};

CompiledSourceMaterial compileSourceMaterial(
    const std::string& path,
    const std::string& expectedName
) {
    using namespace numi::matter;
    ParseResult parsed = parseMatterFile(path);
    if (!parsed.succeeded()) throw std::runtime_error("cannot parse source material: " + path);
    if (parsed.material.name != expectedName)
        throw std::runtime_error("source material identity differs from captured descriptor");
    if (parsed.material.parameters.size() != kParameterCount)
        throw std::runtime_error("source material parameter count differs from 13-state schema");
    for (std::size_t i = 0; i < kParameterCount; ++i)
        if (parsed.material.parameters[i].name != kParameterNames[i])
            throw std::runtime_error("source material parameter order differs from captured descriptor schema");

    // Reapply the exact command-line material builder policies before lowering.
    parsed.material.paperHillScalarInitializer = true;
    parsed.material.paperCompensatedArithmetic = true;
    parsed.material.paperHillAccurateProjection = true;
    const auto rowScales = numi::cardboard::paperProjectionRowScales(parsed.material);
    parsed.material = numi::cardboard::scalePaperProjectionResiduals(std::move(parsed.material));
    const detail::ConstitutiveCompileResult compiled = detail::compileConstitutive(
        parsed.material, NM_EXPRESSION_STACK_CAPACITY);
    if (!compiled.succeeded()) throw std::runtime_error("cannot compile transformed source material: " + path);
    if (compiled.program.implicitResiduals.size() != kStateCount)
        throw std::runtime_error("source material does not compile to 13 implicit residuals");

    CompiledSourceMaterial result{};
    result.name = expectedName;
    result.rowScales = rowScales;
    for (const ScalarBytecode& bytecode : compiled.program.implicitResiduals) {
        NMScalarProgramGPU program{};
        program.firstInstruction = static_cast<nm_u32>(result.instructions.size());
        program.instructionCount = static_cast<nm_u32>(bytecode.instructions.size());
        program.maximumStack = bytecode.maximumStack;
        program.flags = 2u; // Same compensated scalar evaluator used by the native policy.
        result.instructions.insert(result.instructions.end(), bytecode.instructions.begin(),
                                   bytecode.instructions.end());
        result.residualPrograms.push_back(program);
    }
    return result;
}

template <class Values>
void writeArray(const Values& values, double scale = 1.0) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << static_cast<double>(values[i]) * scale;
    }
    std::cout << ']';
}

template <std::size_t N>
std::array<double, N> readF32Array(std::istream& input, const char* label) {
    std::array<double, N> values{};
    for (double& value : values) {
        if (!(input >> value) || !std::isfinite(value))
            throw std::runtime_error(std::string("invalid ") + label);
        value = static_cast<float>(value);
        if (!std::isfinite(value))
            throw std::runtime_error(std::string(label) + " outside finite FP32 range");
    }
    return values;
}

template <std::size_t N>
std::array<float, N> readF32Words(std::istream& input, const char* label) {
    std::array<float, N> values{};
    for (float& value : values) {
        double raw = 0.0;
        if (!(input >> raw) || !std::isfinite(raw))
            throw std::runtime_error(std::string("invalid ") + label);
        value = static_cast<float>(raw);
        if (!std::isfinite(value))
            throw std::runtime_error(std::string(label) + " outside finite FP32 range");
    }
    return values;
}

template <std::size_t N>
std::array<double, N> widen(const std::array<float, N>& values) {
    std::array<double, N> result{};
    for (std::size_t i = 0; i < N; ++i) result[i] = static_cast<double>(values[i]);
    return result;
}

template <std::size_t N>
struct F32WordPair {
    std::array<float, N> high{};
    std::array<float, N> low{};
    std::array<double, N> sum{};
};

template <std::size_t N>
F32WordPair<N> readF32WordPair(std::istream& input, const char* label) {
    F32WordPair<N> result{};
    result.high = readF32Words<N>(input, label);
    result.low = readF32Words<N>(input, label);
    for (std::size_t i = 0; i < N; ++i)
        result.sum[i] = static_cast<double>(result.high[i]) + static_cast<double>(result.low[i]);
    return result;
}

struct CapturedParameters {
    paper::Parameters paper{};
    double residualScale = 0.0;
};

CapturedParameters makeParameters(
    const std::string& kind,
    const std::array<double, kParameterCount>& values
) {
    if (kind != "liner" && kind != "medium")
        throw std::runtime_error("unsupported 13-state paper material kind");
    CapturedParameters result{};
    auto& p = result.paper;
    p.name = kind == "liner" ? "captured_liner_13state_hill" : "captured_medium_13state_hill";
    p.density = values[0];
    for (unsigned i = 0; i < 6; ++i) p.c[i] = values[1 + i] * 1.0e-6;
    for (unsigned i = 0; i < 3; ++i) p.shear[i] = values[7 + i] * 1.0e-6;
    p.sigmaReference = values[10] * 1.0e-6;
    for (unsigned i = 0; i < 6; ++i) p.hill[i] = values[11 + i];
    p.qFloor = values[17];
    p.fischerBurmeisterEpsilon = values[18];
    result.residualScale = values[19];
    if (!(p.density > 0.0 && p.c[0] > 0.0 && p.c[1] > 0.0 && p.c[2] > 0.0 &&
          p.shear[0] > 0.0 && p.shear[1] > 0.0 && p.shear[2] > 0.0 &&
          p.sigmaReference > 0.0 && p.qFloor > 0.0 &&
          p.fischerBurmeisterEpsilon > 0.0 && result.residualScale > 0.0))
        throw std::runtime_error("captured paper parameters contain a nonpositive scale");
    return result;
}

double maxAbs(const std::array<double, 13>& values) {
    double result = 0.0;
    for (double value : values) result = std::max(result, std::abs(value));
    return result;
}

std::array<double, 13> scaledResidual(
    const CapturedParameters& parameters,
    const std::array<double, 13>& sourceRowScales,
    const paper::Matrix& f,
    const paper::State& accepted,
    const paper::State& candidate
) {
    auto residual = paper::residual(parameters.paper, f, accepted, candidate);
    for (unsigned i = 0; i < residual.size(); ++i)
        residual[i] *= parameters.residualScale * sourceRowScales[i];
    return residual;
}

bool safeKey(const std::string& key) {
    return !key.empty() && key.size() <= 160 &&
        std::all_of(key.begin(), key.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-' || c == '.';
        });
}

void writeDerivativeSweep(const paper::Parameters& parameters,
                          const paper::Matrix& f, const paper::Matrix& direction,
                          const paper::State& accepted,
                          const paper::Matrix& tangent) {
    double directionScale = 0.0;
    for (double value : direction) directionScale = std::max(directionScale, std::abs(value));
    std::cout << ",\"FP64_derivative_sweep\":{\"history\":\"same_captured_accepted_state\","
                 "\"direction_scale_per_s\":" << directionScale
              << ",\"stress_units\":\"Pa/s\",\"smooth_gate\":1e-6,\"rows\":[";
    if (!(directionScale > 0.0)) {
        std::cout << "],\"verdict\":\"inconclusive_zero_direction\"}";
        return;
    }
    const auto trialQ = [&](const paper::Matrix& value) {
        return paper::hillEquivalent(parameters, paper::normalizedSecondPiola(
            parameters, value, paper::plasticStrain(accepted)));
    };
    const auto branch = [](double q) {
        return std::abs(q - 1.0) <= 1e-6 ? 0 : (q > 1.0 ? 1 : -1);
    };
    const int centerBranch = branch(trialQ(f));
    bool previousPass = false, adjacentPass = false;
    for (unsigned refinement = 0; refinement < 14u; ++refinement) {
        if (refinement) std::cout << ',';
        const double epsilon = std::ldexp(1e-5 / directionScale, -static_cast<int>(refinement));
        paper::Matrix plus = f, minus = f;
        for (unsigned i = 0; i < 9u; ++i) {
            plus[i] += epsilon * direction[i];
            minus[i] -= epsilon * direction[i];
        }
        std::cout << "{\"epsilon_seconds\":" << epsilon;
        try {
            const auto pp = oracle::project(parameters, plus, accepted);
            const auto pm = oracle::project(parameters, minus, accepted);
            const bool smooth = centerBranch != 0 && branch(trialQ(plus)) == centerBranch &&
                branch(trialQ(minus)) == centerBranch;
            double error = 0.0, scale = 0.0, realizedError = 0.0;
            for (unsigned i = 0; i < 9u; ++i) {
                const double fd = (pp.firstPiola[i] - pm.firstPiola[i]) / (2.0 * epsilon);
                error = std::max(error, std::abs(fd - tangent[i]) * 1e6);
                scale = std::max(scale, std::abs(tangent[i]) * 1e6);
                realizedError = std::max(realizedError, std::abs(
                    (plus[i] - minus[i]) / (2.0 * epsilon) - direction[i]));
            }
            const bool pass = smooth && scale > 0.0 && error / scale <= 1e-6;
            adjacentPass = adjacentPass || (pass && previousPass);
            previousPass = pass;
            std::cout << ",\"valid\":true,\"smooth_branch_eligible\":" << (smooth ? "true" : "false")
                      << ",\"absolute_error_Pa_per_s\":" << error
                      << ",\"reference_scale_Pa_per_s\":" << scale
                      << ",\"relative_error\":";
            if (scale > 0.0) std::cout << error / scale; else std::cout << "null";
            std::cout << ",\"realized_direction_absolute_error_per_s\":" << realizedError
                      << ",\"projection_residuals\": [" << pp.residual << ',' << pm.residual << "]}";
        } catch (const std::exception&) {
            previousPass = false;
            std::cout << ",\"valid\":false,\"reason\":\"independent_projection_failed\"}";
        }
    }
    std::cout << "],\"verdict\":\""
              << (adjacentPass ? "two_adjacent_smooth_refinements_pass" : "no_adjacent_resolved_pass")
              << "\",\"Metal_noise_floor_measured\":false}";
}

void replayRow(std::istringstream& input, const std::string& key, const std::string& kind,
               const CompiledSourceMaterial& source, bool derivativeSweep,
               bool directionHighLow) {
    const auto fWords = readF32WordPair<9>(input, "F high/low word pair");
    F32WordPair<9> directionPair{};
    if (directionHighLow) directionPair = readF32WordPair<9>(input, "direction high/low word pair");
    else {
        directionPair.high = readF32Words<9>(input, "local deformation direction");
        for (unsigned i = 0; i < 9; ++i) directionPair.sum[i] = static_cast<double>(directionPair.high[i]);
    }
    const auto acceptedWords = readF32Words<kStateCount>(input, "accepted material state");
    const auto candidateWords = readF32WordPair<kStateCount>(input, "candidate state high/low word pair");
    const auto parameterWords = readF32Words<kParameterCount>(input, "captured material parameters");
    std::string extra;
    if (input >> extra) throw std::runtime_error("extra replay row fields");

    const auto f = fWords.sum;
    const auto direction = directionPair.sum;
    const auto accepted = widen(acceptedWords);
    const auto candidate = candidateWords.sum;
    const auto rawParameters = widen(parameterWords);
    const CapturedParameters parameters = makeParameters(kind, rawParameters);
    const auto projection = oracle::project(parameters.paper, f, accepted);
    const auto tangentAtReprojected = oracle::directionalTangent(
        parameters.paper, f, direction, accepted, projection.state);
    const auto tangentAtCandidate = oracle::directionalTangent(
        parameters.paper, f, direction, accepted, candidate);
    const auto candidateStress = paper::firstPiola(
        parameters.paper, f, paper::plasticStrain(candidate));
    const auto candidateResidual = paper::residual(parameters.paper, f, accepted, candidate);
    const auto candidateScaledResidual = scaledResidual(
        parameters, source.rowScales, f, accepted, candidate);
    const auto projectedScaledResidual = scaledResidual(
        parameters, source.rowScales, f, accepted, projection.state);

    if (source.residualPrograms.size() != kStateCount)
        throw std::runtime_error("transformed source residual program count is invalid");
    NMPaperCandidateState candidateView{candidateWords.high.data(), candidateWords.low.data()};
    std::array<float, kStateCount> sourceVmResidual{};
    bool sourceVmValid = true;
    for (unsigned row = 0; row < kStateCount; ++row) {
        bool rowValid = true;
        sourceVmResidual[row] = nmPaperCompensatedBytecode<0u>(
            source.residualPrograms[row], source.instructions.data(), parameterWords.data(),
            acceptedWords.data(), candidateView, fWords.high.data(), nullptr, nullptr,
            0.0f, 0.0f, rowValid, nullptr, fWords.low.data());
        sourceVmValid = sourceVmValid && rowValid;
    }
    if (!sourceVmValid) throw std::runtime_error("transformed compensated source bytecode rejected captured row");
    double sourceVmResidualInf = 0.0;
    for (float value : sourceVmResidual)
        sourceVmResidualInf = std::max(sourceVmResidualInf, std::abs(static_cast<double>(value)));
    const auto acceptedTrialStress = paper::normalizedSecondPiola(
        parameters.paper, f, paper::plasticStrain(accepted));
    paper::Vector6 candidateNormalizedStress{};
    for (unsigned i = 0; i < 6; ++i) candidateNormalizedStress[i] = candidate[6 + i];

    double stateError = 0.0;
    for (unsigned i = 0; i < kStateCount; ++i)
        stateError = std::max(stateError, std::abs(projection.state[i] - candidate[i]));

    std::cout << "{\"schema\":\"numi.cardboard.solver-pair-replay.v1\",\"key\":\""
              << key << "\",\"material_kind\":\"" << kind
              << "\",\"valid\":true"
              << ",\"word_contract\":\"FP32_high_and_low_words_widened_separately_then_summed_in_FP64\""
              << ",\"F_local_FP64\":";
    writeArray(f);
    std::cout << ",\"direction_local_per_s\":";
    writeArray(direction);
    std::cout << ",\"accepted_state\":";
    writeArray(accepted);
    std::cout << ",\"candidate_state_FP64\":";
    writeArray(candidate);
    std::cout << ",\"candidate_state_reprojected_reference_error_max_abs\":" << stateError
              << ",\"oracle_reprojected_state\":";
    writeArray(projection.state);
    std::cout << ",\"oracle_reprojected_P_Pa\":";
    writeArray(projection.firstPiola, 1.0e6);
    std::cout << ",\"oracle_candidate_P_Pa\":";
    writeArray(candidateStress, 1.0e6);
    std::cout << ",\"oracle_tangent_at_reprojected_state_Pa_per_s\":";
    writeArray(tangentAtReprojected, 1.0e6);
    std::cout << ",\"oracle_tangent_at_captured_candidate_Pa_per_s\":";
    writeArray(tangentAtCandidate, 1.0e6);
    std::cout << ",\"oracle_residual_unscaled_inf\":" << projection.residual
              << ",\"oracle_residual_scaled_inf\":" << maxAbs(projectedScaledResidual)
              << ",\"candidate_residual_unscaled_inf\":" << maxAbs(candidateResidual)
              << ",\"candidate_residual_unscaled_rows\":";
    writeArray(candidateResidual);
    std::cout << ",\"candidate_residual_scaled_inf\":" << maxAbs(candidateScaledResidual)
              << ",\"candidate_residual_scaled_rows\":";
    writeArray(candidateScaledResidual);
    std::cout << ",\"source_vm_compensated_scaled_residual_inf\":" << sourceVmResidualInf
              << ",\"source_vm_compensated_scaled_residual_rows\":";
    writeArray(sourceVmResidual);
    std::cout << ",\"projection_row_scales_from_source_defaults\":";
    writeArray(source.rowScales);
    std::cout << ",\"source_residual_scale\":" << parameters.residualScale
              << ",\"projection_iterations\":" << projection.iterations
              << ",\"projected_q\":" << projection.equivalentStress
              << ",\"candidate_q\":" << paper::hillEquivalent(parameters.paper, candidateNormalizedStress)
              << ",\"elastic_trial_q\":" << paper::hillEquivalent(parameters.paper, acceptedTrialStress)
              << ",\"lambda_increment_candidate\":" << candidate[12] - accepted[12]
              << ",\"lambda_increment_reprojected\":" << projection.state[12] - accepted[12];
    if (derivativeSweep && key.ends_with("_s0"))
        writeDerivativeSweep(parameters.paper, f, direction, accepted, tangentAtReprojected);
    std::cout << "}\n";
}
} // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    const bool derivativeSweep = argc == 4 && std::string(argv[3]) == "--derivative-sweep";
    const bool directionHighLow = argc == 4 && std::string(argv[3]) == "--direction-high-low";
    if (argc != 3 && !derivativeSweep && !directionHighLow) {
        std::cerr << "usage: cardboard-solver-pair-replay LINER.nmatter MEDIUM.nmatter [--direction-high-low|--derivative-sweep]\n"
                  << "stdin rows: key liner|medium F_hi[9] F_lo[9] "
                     "[H_hi[9] H_lo[9] if --direction-high-low] accepted[13] "
                     "candidate_hi[13] candidate_lo[13] parameters[20]\n";
        return 1;
    }
    std::array<CompiledSourceMaterial, 2> sources{};
    try {
        sources[0] = compileSourceMaterial(argv[1], "hajali2009_liner_hill_ideal");
        sources[1] = compileSourceMaterial(argv[2], "hajali2009_medium_hill_ideal");
    } catch (const std::exception& error) {
        std::cerr << "FAIL transformed source material setup: " << error.what() << '\n';
        return 1;
    }
    std::string line;
    unsigned rowCount = 0;
    unsigned failedRows = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream input(line);
        std::string key, kind;
        if (!(input >> key >> kind) || !safeKey(key)) {
            std::cerr << "FAIL malformed replay row identity\n";
            return 1;
        }
        try {
            if (kind != "liner" && kind != "medium")
                throw std::runtime_error("unsupported captured paper kind");
            replayRow(input, key, kind, sources[kind == "liner" ? 0u : 1u], derivativeSweep, directionHighLow);
        } catch (const std::exception& error) {
            ++failedRows;
            std::cout << "{\"schema\":\"numi.cardboard.solver-pair-replay.v1\",\"key\":\""
                      << key << "\",\"material_kind\":\"" << kind
                      << "\",\"valid\":false,\"reason\":\"" << error.what() << "\"}\n";
        }
        ++rowCount;
    }
    if (rowCount == 0) {
        std::cerr << "FAIL no replay rows\n";
        return 1;
    }
    return failedRows == 0 ? 0 : 2;
}
