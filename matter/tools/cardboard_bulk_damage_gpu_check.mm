#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include "robertsson2023_material.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#ifndef NUMI_CARDBOARD_BULK_METALLIB
#define NUMI_CARDBOARD_BULK_METALLIB ""
#endif

using namespace numi::matter;
using Matrix = std::array<double, 9>;
constexpr Matrix identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
constexpr size_t kStressOffset = 8;
constexpr size_t kTangentOffset = 17;
constexpr size_t kCandidateStateOffset = 26;
constexpr size_t kAcceptedStateOffset =
    kCandidateStateOffset + NM_MAX_MATERIAL_STATE;
constexpr size_t kTerminalFailureOffset =
    kAcceptedStateOffset + NM_MAX_MATERIAL_STATE;
constexpr size_t kDiagnosticOutputCount = kTerminalFailureOffset + 1u;
constexpr size_t kGuardCount = 4;

enum class TerminalFailure : uint32_t {
    none = 0,
    projectionInputInvalid = 1,
    stateUpdateInvalid = 2,
    implicitEvaluationInvalid = 3,
    implicitEvaluationNonfinite = 4,
    localSolveFailed = 5,
    armijoNoAcceptedTrial = 6,
    projectionIterationExhausted = 7,
    specializedProjectionInvalid = 8,
    stressEvaluationInvalid = 9,
    stressNonfinite = 10,
    tangentEvaluationInvalid = 11,
    tangentNonfinite = 12,
    implicitTangentEvaluationInvalid = 13,
    implicitTangentEvaluationNonfinite = 14,
    implicitTangentSolveFailed = 15,
    implicitTangentDerivativeInvalid = 16,
    validityRejected = 17,
};

const char* terminalFailureName(const TerminalFailure failure) {
    switch (failure) {
        case TerminalFailure::none: return "none";
        case TerminalFailure::projectionInputInvalid: return "projection_input_invalid";
        case TerminalFailure::stateUpdateInvalid: return "state_update_invalid";
        case TerminalFailure::implicitEvaluationInvalid: return "implicit_evaluation_invalid";
        case TerminalFailure::implicitEvaluationNonfinite: return "implicit_evaluation_nonfinite";
        case TerminalFailure::localSolveFailed: return "local_solve_failed";
        case TerminalFailure::armijoNoAcceptedTrial: return "armijo_no_accepted_trial";
        case TerminalFailure::projectionIterationExhausted: return "projection_iteration_exhausted";
        case TerminalFailure::specializedProjectionInvalid: return "specialized_projection_invalid";
        case TerminalFailure::stressEvaluationInvalid: return "stress_evaluation_invalid";
        case TerminalFailure::stressNonfinite: return "stress_nonfinite";
        case TerminalFailure::tangentEvaluationInvalid: return "tangent_evaluation_invalid";
        case TerminalFailure::tangentNonfinite: return "tangent_nonfinite";
        case TerminalFailure::implicitTangentEvaluationInvalid: return "implicit_tangent_evaluation_invalid";
        case TerminalFailure::implicitTangentEvaluationNonfinite: return "implicit_tangent_evaluation_nonfinite";
        case TerminalFailure::implicitTangentSolveFailed: return "implicit_tangent_solve_failed";
        case TerminalFailure::implicitTangentDerivativeInvalid: return "implicit_tangent_derivative_invalid";
        case TerminalFailure::validityRejected: return "validity_rejected";
    }
    return "unknown";
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class T>
id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values) {
    T zero{};
    const void* data = values.empty() ? &zero : values.data();
    const size_t count = std::max(values.size(), size_t(1));
    return [device newBufferWithBytes:data
                               length:count * sizeof(T)
                              options:MTLResourceStorageModeShared];
}

struct Response {
    bool valid = false;
    bool validityEvaluated = false;
    float validityMargin = 0.0f;
    uint32_t stageFlags = 0;
    float projectionResidualNorm = -1.0f;
    float projectionStateNorm = 0.0f;
    uint32_t projectionIterations = 0;
    TerminalFailure terminalFailure = TerminalFailure::none;
    Matrix stress{};
    Matrix tangent{};
    std::vector<float> candidateState;
    std::vector<float> acceptedState;
};

struct Instrument {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::array<id<MTLBuffer>, 4> residentBuffers{};
    const CompiledWorld* residentWorld = nullptr;
    size_t residentStaticBytes = 0u;

    explicit Instrument(const CompiledWorld& world) : residentWorld(&world) {
        require(world.materials.size() == 1u,
                "diagnostic requires exactly one resident material");
        require(world.materials[0].stateCount <= NM_MAX_MATERIAL_STATE,
                "material state count exceeds diagnostic fixed-capacity storage");
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal unavailable");
        std::vector<float> residentParameters;
        residentParameters.reserve(world.parameters.size());
        for (const auto& parameter : world.parameters)
            residentParameters.push_back(parameter.valueAndBounds.x);
        residentBuffers = {
            buffer(device, std::vector<NMMaterialGPU>{world.materials[0]}),
            buffer(device, world.scalarPrograms),
            buffer(device, world.instructions),
            buffer(device, residentParameters),
        };
        residentStaticBytes = sizeof(NMMaterialGPU) +
            world.scalarPrograms.size() * sizeof(NMScalarProgramGPU) +
            world.instructions.size() * sizeof(NMExpressionInstructionGPU) +
            residentParameters.size() * sizeof(float);
        queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_BULK_METALLIB]
                        error:&error];
        require(library != nil, "bulk-damage diagnostic library unavailable");
        id<MTLFunction> function =
            [library newFunctionWithName:@"cardboard_bulk_damage_diagnostic"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipeline != nil, "bulk-damage diagnostic pipeline unavailable");
        std::cout << "device=" << [[device name] UTF8String]
                  << " instrument=production_algorithmic_constitutive_path"
                  << " static_material_world_buffers=resident"
                  << " static_material_world_bytes=" << residentStaticBytes
                  << " physical_FEM_steps=0\n"
                  << "stage_flag_legend=0x001:projection_entered,"
                     "0x002:state_updates_complete,"
                     "0x004:implicit_system_evaluated,"
                     "0x008:local_solve_attempted,"
                     "0x010:local_solve_succeeded,"
                     "0x020:newton_update_accepted,"
                     "0x040:projection_converged,"
                     "0x080:stress_evaluated,"
                     "0x100:tangent_evaluated,"
                     "0x200:implicit_tangent_entered,"
                     "0x400:implicit_tangent_solved\n"
                  << "projection_residual_inf=-1 means no local implicit "
                     "residual was evaluated\n"
                  << "terminal_failure_codes=0:none,1:projection_input_invalid,"
                     "2:state_update_invalid,3:implicit_evaluation_invalid,"
                     "4:implicit_evaluation_nonfinite,5:local_solve_failed,"
                     "6:armijo_no_accepted_trial,"
                     "7:projection_iteration_exhausted,"
                     "8:specialized_projection_invalid,"
                     "9:stress_evaluation_invalid,10:stress_nonfinite,"
                     "11:tangent_evaluation_invalid,12:tangent_nonfinite,"
                     "13:implicit_tangent_evaluation_invalid,"
                     "14:implicit_tangent_evaluation_nonfinite,"
                     "15:implicit_tangent_solve_failed,"
                     "16:implicit_tangent_derivative_invalid,"
                     "17:validity_rejected\n";
    }

    Response evaluate(const CompiledWorld& world, const Matrix& deformation,
                      const Matrix& deformationDirection,
                      const std::vector<float>& acceptedState) {
        require(&world == residentWorld,
                "evaluate called with a world other than the resident world");
        require(acceptedState.size() == world.materials[0].stateCount,
                "accepted-state count differs from the material state count");
        std::vector<float> inputs;
        inputs.reserve(18);
        for (double value : deformation)
            inputs.push_back(static_cast<float>(value));
        for (double value : deformationDirection)
            inputs.push_back(static_cast<float>(value));
        std::vector<float> output(kDiagnosticOutputCount + kGuardCount,
                                  123456.0f);
        std::array<id<MTLBuffer>, 7> buffers{
            residentBuffers[0], residentBuffers[1], residentBuffers[2],
            residentBuffers[3], buffer(device, acceptedState),
            buffer(device, inputs), buffer(device, output),
        };

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (size_t index = 0; index < buffers.size(); ++index)
            [encoder setBuffer:buffers[index] offset:0 atIndex:index];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "native bulk-damage material command failed");

        const auto* values =
            static_cast<const float*>(buffers[6].contents);
        for (size_t index = kDiagnosticOutputCount;
             index < output.size(); ++index) {
            require(values[index] == 123456.0f,
                    "bulk-damage diagnostic output overrun");
        }

        Response response;
        response.valid = values[0] == 1.0f;
        response.validityEvaluated = values[1] == 1.0f;
        response.validityMargin = values[2];
        response.stageFlags = static_cast<uint32_t>(values[3]);
        response.projectionResidualNorm = values[4];
        response.projectionStateNorm = values[5];
        response.projectionIterations = static_cast<uint32_t>(values[6]);
        const float stateCountValue = values[7];
        require(std::isfinite(stateCountValue) && stateCountValue >= 0.0f &&
                    stateCountValue <= float(NM_MAX_MATERIAL_STATE) &&
                    std::floor(stateCountValue) == stateCountValue,
                "native state count exceeds fixed-capacity diagnostic storage");
        const uint32_t stateCount = static_cast<uint32_t>(stateCountValue);
        require(stateCount == world.materials[0].stateCount,
                "native output state count mismatch");
        for (uint32_t index = 0; index < 9; ++index) {
            response.stress[index] = values[kStressOffset + index];
            response.tangent[index] = values[kTangentOffset + index];
        }
        const float failureCode = values[kTerminalFailureOffset];
        require(std::isfinite(failureCode) && failureCode >= 0.0f &&
                    failureCode <= 17.0f && std::floor(failureCode) == failureCode,
                "native terminal failure subtype is outside the diagnostic enum");
        response.terminalFailure = static_cast<TerminalFailure>(
            static_cast<uint32_t>(failureCode));
        require(response.valid ==
                    (response.terminalFailure == TerminalFailure::none),
                "constitutive acceptance and terminal failure subtype disagree");
        response.candidateState.assign(
            values + kCandidateStateOffset,
            values + kCandidateStateOffset + stateCount);
        response.acceptedState.assign(
            values + kAcceptedStateOffset,
            values + kAcceptedStateOffset + stateCount);
        require(response.acceptedState == acceptedState,
                "native evaluator changed its accepted-state input");
        return response;
    }
};

struct Options {
    double maximumShear = 0.3;
    uint32_t stepsPerLeg = 200;
    uint32_t checkpointEvery = 1;
    std::string csvPath;
    std::string seedStatePath;
    double startShear = 0.0;
    bool startShearWasSet = false;
    bool continueSeed = false;
};

double parsePositiveDouble(const std::string& text, const char* option) {
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(value) ||
        !(value > 0.0)) {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return value;
}

uint32_t parsePositiveCount(const std::string& text, const char* option) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value == 0ul ||
        value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return static_cast<uint32_t>(value);
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char* option) -> std::string {
            if (index + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + option);
            return argv[++index];
        };
        if (argument == "--seed-state") {
            options.seedStatePath = value("--seed-state");
            require(!options.seedStatePath.empty(),
                    "--seed-state path must not be empty");
        } else if (argument == "--start-shear") {
            const std::string text = value("--start-shear");
            char* end = nullptr;
            options.startShear = std::strtod(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0' ||
                !std::isfinite(options.startShear) || options.startShear < 0.0)
                throw std::runtime_error("invalid value for --start-shear");
            options.startShearWasSet = true;
        } else if (argument == "--continue-seed") {
            options.continueSeed = true;
        } else if (argument == "--max-shear") {
            options.maximumShear =
                parsePositiveDouble(value("--max-shear"), "--max-shear");
        } else if (argument == "--steps-per-leg") {
            options.stepsPerLeg =
                parsePositiveCount(value("--steps-per-leg"), "--steps-per-leg");
        } else if (argument == "--checkpoint-every") {
            options.checkpointEvery = parsePositiveCount(
                value("--checkpoint-every"), "--checkpoint-every");
        } else if (argument == "--csv") {
            options.csvPath = value("--csv");
            require(!options.csvPath.empty(), "--csv path must not be empty");
        } else if (argument == "--help") {
            std::cout << "usage: numi-matter-cardboard-bulk-damage-gpu-check "
                         "[--max-shear gamma] [--steps-per-leg N] "
                         "[--checkpoint-every N] [--csv path] "
                         "[--seed-state path --start-shear gamma] "
                         "[--continue-seed]\n"
                         "Runs a material-point path; this is not a physical "
                         "FEM step. Default path is 0 -> 0.3 -> 0. A seed "
                         "file contains exactly one material state vector as "
                         "whitespace/comma/semicolon-separated finite floats "
                         "in state-index order. Seed state is at --start-shear; "
                         "the path loads to --max-shear then unloads to "
                         "--start-shear. With --continue-seed, the imported "
                         "accepted history goes directly to the next shear "
                         "increment: the same-shear seed response is skipped. "
                         "This option requires both --seed-state and "
                         "--start-shear; the default still evaluates step 0.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown command-line option: " + argument);
        }
    }
    require(options.stepsPerLeg <= std::numeric_limits<uint32_t>::max() / 2u,
            "too many material increments");
    require(options.seedStatePath.empty() == !options.startShearWasSet,
            "--seed-state and --start-shear must be supplied together");
    require(options.seedStatePath.empty() ||
                options.startShear <= options.maximumShear,
            "--start-shear must not exceed --max-shear");
    require(!options.continueSeed || !options.seedStatePath.empty(),
            "--continue-seed requires --seed-state and --start-shear");
    return options;
}

std::vector<float> loadSeedState(const std::string& path, uint32_t stateCount) {
    std::ifstream input(path);
    require(bool(input), "cannot open material restart state");
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    for (char& character : contents) {
        if (character == ',' || character == ';') character = ' ';
    }
    std::istringstream values(contents);
    std::vector<float> state;
    std::string token;
    while (values >> token) {
        char* end = nullptr;
        const float value = std::strtof(token.c_str(), &end);
        require(end != token.c_str() && *end == '\0' && std::isfinite(value),
                "restart state contains a non-float or nonfinite value");
        state.push_back(value);
    }
    require(state.size() == stateCount,
            "restart state count differs from compiled material state count");
    return state;
}

CompiledWorld compileMaterial(MaterialProgram material) {
    WorldSource source;
    source.materials = {std::move(material)};
    source.environmentCount = 1;
    source.gravity = {0, 0, 0};
    source.frameTimestep = 1.0e-4;
    ObjectSource object;
    object.name = "full_source_material_instrument_fixture";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{{0, 0, 0}}, {{0.001, 0, 0}},
                       {{0, 0.001, 0}}, {{0, 0, 0.001}}};
    object.tetrahedra = {{{0, 1, 2, 3}}};
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16;
    const auto result = compileWorld(source, options);
    if (!result.succeeded()) {
        for (const auto& diagnostic : result.diagnostics)
            std::cerr << diagnostic.message << '\n';
        throw std::runtime_error("source material compile rejected");
    }
    return result.world;
}

void writeCsvHeader(std::ostream& output, uint32_t stateCount) {
    output << "step,command_shear,algorithmic_response_accepted,"
              "validity_evaluated,validity_margin,stage_flags,terminal_failure,"
              "projection_residual_inf,projection_state_inf,"
              "projection_iterations,P13_Pa,dP13_dgamma_Pa,damage";
    for (uint32_t state = 0; state < stateCount; ++state)
        output << ",accepted_state_" << state;
    for (uint32_t state = 0; state < stateCount; ++state)
        output << ",candidate_state_" << state;
    output << '\n';
}

void writeCsvRow(std::ostream& output, uint32_t step, double commandShear,
                 const Response& response,
                 const std::vector<float>& acceptedState) {
    const double damage = response.candidateState.size() > 26u
        ? response.candidateState[26] : 0.0;
    output << step << ',' << commandShear << ',' << (response.valid ? 1 : 0)
           << ',' << (response.validityEvaluated ? 1 : 0) << ','
           << response.validityMargin << ',' << response.stageFlags << ','
           << terminalFailureName(response.terminalFailure) << ','
           << response.projectionResidualNorm << ','
           << response.projectionStateNorm << ','
           << response.projectionIterations << ',' << response.stress[2] << ','
           << response.tangent[2] << ',' << damage;
    for (float value : acceptedState) output << ',' << value;
    for (float value : response.candidateState) output << ',' << value;
    output << '\n';
}

void writeStateVector(const char* label, const std::vector<float>& state) {
    std::cout << ' ' << label << "=[" << std::setprecision(9);
    for (size_t index = 0; index < state.size(); ++index) {
        if (index != 0u) std::cout << ';';
        std::cout << state[index];
    }
    std::cout << ']';
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const Options options = parseOptions(argc, argv);
            const CompiledWorld world =
                compileMaterial(numi_cardboard::robertsson2023::makeMaterial());
            const uint32_t stateCount = world.materials[0].stateCount;
            std::cout << "state_count=" << stateCount
                      << " instructions=" << world.instructions.size()
                      << " scalar_programs=" << world.scalarPrograms.size()
                      << " start_shear=" << options.startShear
                      << " max_shear=" << options.maximumShear
                      << " steps_per_leg=" << options.stepsPerLeg
                      << " seeded=" << !options.seedStatePath.empty()
                      << " continue_seed=" << options.continueSeed << '\n';

            Instrument instrument(world);
            Matrix direction{};
            // The rest/elastic oracle differentiates P33 with respect to F33.
            // The loading path below instead differentiates P13 with respect
            // to F13; keep those independent directional checks explicit.
            direction[8] = 1.0;
            const auto rest = instrument.evaluate(
                world, identity, direction, world.stateInitials);
            require(rest.valid, "full source rest rejected");
            double maximumRestStress = 0.0;
            for (double value : rest.stress)
                maximumRestStress = std::max(maximumRestStress, std::abs(value));
            std::cout << "rest_max_P_Pa=" << maximumRestStress
                      << " tangent_ZD_Pa=" << rest.tangent[8] << '\n';
            require(maximumRestStress < 500.0,
                    "source rest stress exceeds FP32 cancellation bound");
            require(std::abs(rest.tangent[8] - 56.8e6) < 0.005 * 56.8e6,
                    "source tensile ZD tangent mismatches Eq18");

            // Independent closed-form check of the source elastic ZD branch.
            for (double strain : {-0.0001, 0.0001}) {
                Matrix deformation = identity;
                deformation[8] += strain;
                const auto response = instrument.evaluate(
                    world, deformation, direction, world.stateInitials);
                require(response.valid, "elastic source point rejected");
                const double stretch = static_cast<float>(deformation[8]);
                const double expected = stretch >= 1.0
                    ? 28.4e6 * (1.0 - 1.0 / (stretch * stretch))
                    : 0.730e6 * stretch *
                        (1.0 - std::exp(-11.5 * (stretch * stretch - 1.0)));
                std::cout << "elastic_ZD_F33=" << stretch
                          << " P33=" << response.stress[8]
                          << " expected=" << expected << '\n';
                require(std::abs(response.stress[8] - expected) < 500.0,
                        "source closed-form ZD stress differs");
            }

            std::ofstream csv;
            if (!options.csvPath.empty()) {
                csv.open(options.csvPath);
                require(bool(csv), "cannot open material-state CSV");
                csv << std::setprecision(std::numeric_limits<double>::max_digits10);
                writeCsvHeader(csv, stateCount);
                csv.flush();
            }

            std::vector<float> acceptedState = options.seedStatePath.empty()
                ? world.stateInitials
                : loadSeedState(options.seedStatePath, stateCount);
            if (!options.seedStatePath.empty()) {
                std::cout << "restart_seed_at_shear=" << options.startShear;
                writeStateVector("accepted_state", acceptedState);
                if (options.continueSeed)
                    std::cout << " imported_accepted_history=used_directly_next_shear_increment"
                                 " same_shear_reprojection=skipped";
                std::cout << '\n' << std::flush;
            }
            direction = {};
            direction[2] = 1.0;
            uint32_t acceptedIncrements = 0;
            double maximumDamage = 0.0;
            const uint32_t finalStep = 2u * options.stepsPerLeg;
            const uint32_t firstStep = options.continueSeed ? 1u : 0u;
            for (uint32_t step = firstStep; step <= finalStep; ++step) {
                const double loadingFraction =
                    double(std::min(step, options.stepsPerLeg)) /
                    double(options.stepsPerLeg);
                const double unloadingFraction = step <= options.stepsPerLeg
                    ? 0.0
                    : double(step - options.stepsPerLeg) /
                        double(options.stepsPerLeg);
                const double commandShear = step <= options.stepsPerLeg
                    ? options.startShear +
                        (options.maximumShear - options.startShear) *
                            loadingFraction
                    : options.maximumShear -
                        (options.maximumShear - options.startShear) *
                            unloadingFraction;
                Matrix deformation = identity;
                deformation[2] = commandShear;
                const auto response = instrument.evaluate(
                    world, deformation, direction, acceptedState);

                if (csv) {
                    // Preserve the pre-response accepted history here. In continuation
                    // mode the first row carries the imported seed state before this
                    // next-increment candidate is accepted or rejected.
                    writeCsvRow(csv, step, commandShear, response, acceptedState);
                    if (step % options.checkpointEvery == 0u || !response.valid ||
                        step == finalStep) {
                        csv.flush();
                        require(bool(csv), "failed while flushing material-state CSV");
                    }
                }

                if (step % options.checkpointEvery == 0u || !response.valid ||
                    step == finalStep) {
                    const double damage = response.candidateState.size() > 26u
                        ? response.candidateState[26] : 0.0;
                    std::cout << "step=" << step
                              << " shear=" << commandShear
                              << " algorithmic_response_accepted=" << response.valid
                              << " stage_flags=0x" << std::hex
                              << response.stageFlags << std::dec
                              << " terminal_failure="
                              << terminalFailureName(response.terminalFailure)
                              << " local_residual_inf="
                              << response.projectionResidualNorm
                              << " state_inf=" << response.projectionStateNorm
                              << " local_iterations="
                              << response.projectionIterations
                              << " damage_candidate=" << damage
                              << " state_commit="
                              << (response.valid ? "material-point-only" : "none");
                    writeStateVector("accepted_state", acceptedState);
                    writeStateVector("candidate_state", response.candidateState);
                    std::cout << '\n' << std::flush;
                }

                if (!response.valid) {
                    std::cout << "material_path_rejected_step=" << step
                              << " accepted_material_increments="
                              << acceptedIncrements
                              << " accepted_state_preserved=1"
                              << " candidate_state_csv=" << bool(csv)
                              << '\n' << std::flush;
                    return 2;
                }
                require(response.candidateState.size() == acceptedState.size(),
                        "candidate state count mismatch");
                if (stateCount > 26u) {
                    const float candidateDamage = response.candidateState[26];
                    require(candidateDamage + 1.0e-6f >= acceptedState[26] &&
                                candidateDamage <= 0.96001f,
                            "irreversible damage bound violated");
                    maximumDamage = std::max(maximumDamage,
                                             double(candidateDamage));
                }
                acceptedState = response.candidateState;
                ++acceptedIncrements;
            }

            std::cout << "accepted_material_increments=" << acceptedIncrements
                      << " max_damage=" << maximumDamage
                      << " physical_FEM_steps=0\n";
            require(maximumDamage > 0.0,
                    "source shear path failed to activate damage");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "bulk damage diagnostic failure: " << error.what()
                      << '\n';
            return 1;
        }
    }
}
