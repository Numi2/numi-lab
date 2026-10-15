// CPU/compiler admission and oracle checks for opt-in compensated bytecode
// policies, including stateless stored energy and generic implicit state. This
// does not execute the Metal material or qualify a physical cardboard response.
#define main cardboard_retained_matrix_damage_legacy_cpu_main
#include "cardboard_retained_matrix_damage_check.cpp"
#undef main

#include "numi/matter/paper_compensated_bytecode.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::array<float, 9> f32(const Matrix9& values);

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

numi::matter::CompiledWorld compilePolicyWorld(
    const numi::matter::MaterialProgram& material) {
    using namespace numi::matter;
    WorldSource source;
    source.materials = {material};
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.frameTimestep = 2.0e-3;
    ObjectSource object;
    object.name = "generic_compensated_projection_cpu_fixture";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{0.0, 0.0, 0.0}, {0.001, 0.0, 0.0},
                       {0.0, 0.001, 0.0}, {0.0, 0.0, 0.001}};
    object.tetrahedra = {{0u, 1u, 2u, 3u}};
    object.characteristicLength = 0.001;
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    options.localMaterialNewtonIterations = 16u;
    auto compiled = compileWorld(source, options);
    if (!compiled.succeeded()) {
        std::string message = "generic policy world compile failed";
        for (const auto& diagnostic : compiled.diagnostics)
            message += ": " + diagnostic.message;
        throw std::runtime_error(message);
    }
    return compiled.world;
}

void checkCandidateLowWord() {
    using namespace numi::matter;
    std::array<NMExpressionInstructionGPU, 6> instructions{};
    instructions[0].opcode = NM_EXPR_NEXT_STATE;
    instructions[0].index = 0u;
    instructions[1].opcode = NM_EXPR_NEXT_STATE;
    instructions[1].index = 0u;
    instructions[2].opcode = NM_EXPR_MULTIPLY;
    instructions[3].opcode = NM_EXPR_CONSTANT;
    instructions[3].immediate.x = 1.0f;
    instructions[4].opcode = NM_EXPR_SUBTRACT;
    NMScalarProgramGPU program{};
    program.instructionCount = 5u;
    program.maximumStack = 2u;
    program.flags = 2u;

    const float accepted[1]{1.0f};
    const float high[1]{1.0f};
    const float low[1]{std::ldexp(1.0f, -25)};
    const float f[9]{};
    const float direction[9]{};
    const float rate[9]{};
    bool valid = true;
    const NMPaperCandidateState candidate{high, low};
    const float result = nmPaperCompensatedBytecode<0u>(
        program, instructions.data(), nullptr, accepted, candidate,
        f, direction, rate, 0.002f, 293.15f, valid);
    need(valid, "two-word VM rejected a finite low-word candidate");
    const double exact = (1.0 + std::ldexp(1.0, -25)) *
        (1.0 + std::ldexp(1.0, -25)) - 1.0;
    need(std::abs(static_cast<double>(result) - exact) <= 1.0e-14,
         "candidate low word was lost in a nonlinear scalar expression");

    instructions[1].opcode = NM_EXPR_CACHE_STORE;
    instructions[1].index = 0u;
    instructions[2].opcode = NM_EXPR_CONSTANT;
    instructions[2].immediate.x = 1.0f;
    instructions[3].opcode = NM_EXPR_SUBTRACT;
    instructions[4].opcode = NM_EXPR_CACHE_LOAD;
    instructions[4].index = 0u;
    instructions[5].opcode = NM_EXPR_MULTIPLY;
    program.flags = 3u;
    program.instructionCount = 6u;
    valid = true;
    const float cached = nmPaperCompensatedBytecode<1u>(
        program, instructions.data(), nullptr, accepted, candidate,
        f, direction, rate, 0.002f, 293.15f, valid);
    need(valid && cached == low[0],
         "compensated expression cache discarded the candidate low word");
}

void checkTranscendentals() {
    const auto expValue = nmCompensatedExp({0.125f, 0.0f});
    const double expActual = static_cast<double>(expValue.high) + expValue.low;
    need(std::abs(expActual - std::exp(0.125)) / std::exp(0.125) < 2.0e-7,
         "ordered exponential differs from the FP64 reference");
    const auto expm1Value = nmCompensatedExpm1MinusX({1.0e-4f, 0.0f});
    const double expm1Actual = static_cast<double>(expm1Value.high) + expm1Value.low;
    const double expm1Expected = std::expm1(1.0e-4) - 1.0e-4;
    need(std::abs(expm1Actual - expm1Expected) /
             std::max(std::abs(expm1Expected), 1.0e-30) < 2.0e-7,
         "ordered expm1-minus-x lost its small-argument value");
}

struct OrthotropicReference {
    Matrix9 stress{};
    Matrix9 tangent{};
};

double materialParameter(const numi::matter::MaterialProgram& material,
                         const std::string& name) {
    for (const auto& parameter : material.parameters)
        if (parameter.name == name)
            return static_cast<float>(parameter.defaultValue);
    throw std::runtime_error("missing Nagasawa parameter " + name);
}

OrthotropicReference evaluateNagasawaReference(
    const numi::matter::MaterialProgram& material,
    const Matrix9& f,
    const Matrix9& direction) {
    const double c[3][3] = {
        {materialParameter(material, "c_MM"), materialParameter(material, "c_MC"),
         materialParameter(material, "c_MT")},
        {materialParameter(material, "c_MC"), materialParameter(material, "c_CC"),
         materialParameter(material, "c_CT")},
        {materialParameter(material, "c_MT"), materialParameter(material, "c_CT"),
         materialParameter(material, "c_TT")},
    };
    const double shear[3][3] = {
        {0.0, materialParameter(material, "g_CM"), materialParameter(material, "g_MT")},
        {materialParameter(material, "g_CM"), 0.0, materialParameter(material, "g_CT")},
        {materialParameter(material, "g_MT"), materialParameter(material, "g_CT"), 0.0},
    };
    double rightCauchyGreen[3][3]{};
    double deltaCauchyGreen[3][3]{};
    double strain[3]{};
    double deltaStrain[3]{};
    for (unsigned i = 0u; i < 3u; ++i) {
        for (unsigned j = 0u; j < 3u; ++j) {
            for (unsigned row = 0u; row < 3u; ++row) {
                rightCauchyGreen[i][j] += f[row * 3u + i] * f[row * 3u + j];
                deltaCauchyGreen[i][j] +=
                    direction[row * 3u + i] * f[row * 3u + j] +
                    f[row * 3u + i] * direction[row * 3u + j];
            }
        }
        strain[i] = 0.5 * (rightCauchyGreen[i][i] - 1.0);
        deltaStrain[i] = 0.5 * deltaCauchyGreen[i][i];
    }
    double secondPiola[3][3]{};
    double deltaSecondPiola[3][3]{};
    for (unsigned i = 0u; i < 3u; ++i) {
        for (unsigned j = 0u; j < 3u; ++j) {
            if (i == j) {
                for (unsigned k = 0u; k < 3u; ++k) {
                    secondPiola[i][i] += c[i][k] * strain[k];
                    deltaSecondPiola[i][i] += c[i][k] * deltaStrain[k];
                }
            } else {
                secondPiola[i][j] = shear[i][j] * rightCauchyGreen[i][j];
                deltaSecondPiola[i][j] = shear[i][j] * deltaCauchyGreen[i][j];
            }
        }
    }
    OrthotropicReference result;
    for (unsigned row = 0u; row < 3u; ++row) {
        for (unsigned col = 0u; col < 3u; ++col) {
            for (unsigned k = 0u; k < 3u; ++k) {
                result.stress[row * 3u + col] += f[row * 3u + k] * secondPiola[k][col];
                result.tangent[row * 3u + col] +=
                    direction[row * 3u + k] * secondPiola[k][col] +
                    f[row * 3u + k] * deltaSecondPiola[k][col];
            }
        }
    }
    return result;
}

std::vector<float> localParameterValues(const numi::matter::CompiledWorld& world,
                                        const NMMaterialGPU& material) {
    std::vector<float> values;
    values.reserve(material.parameterCount);
    for (std::uint32_t i = 0u; i < material.parameterCount; ++i)
        values.push_back(world.parameters.at(material.parameterOffset + i).valueAndBounds.x);
    return values;
}

float evaluateOrdinaryFP32(
    const numi::matter::CompiledWorld& world,
    const NMScalarProgramGPU& program,
    const std::vector<float>& parameters,
    const std::array<float, 9>& f,
    const std::array<float, 9>& direction) {
    std::vector<float> stack;
    stack.reserve(program.maximumStack);
    const auto pop = [&]() {
        need(!stack.empty(), "ordinary FP32 VM stack underflow");
        const float value = stack.back();
        stack.pop_back();
        return value;
    };
    for (std::uint32_t i = 0u; i < program.instructionCount; ++i) {
        const auto& instruction = world.instructions.at(program.firstInstruction + i);
        float value = 0.0f;
        switch (instruction.opcode) {
        case NM_EXPR_CONSTANT: value = instruction.immediate.x; break;
        case NM_EXPR_PARAMETER: value = parameters.at(instruction.index); break;
        case NM_EXPR_F: value = f.at(instruction.index); break;
        case NM_EXPR_DF: value = direction.at(instruction.index); break;
        case NM_EXPR_RATE: value = 0.0f; break;
        case NM_EXPR_STATE:
        case NM_EXPR_NEXT_STATE:
            throw std::runtime_error("stateless material unexpectedly emitted a state opcode");
        case NM_EXPR_DT: value = 0.002f; break;
        case NM_EXPR_TEMPERATURE: value = 293.15f; break;
        case NM_EXPR_ADD: { const float b = pop(); value = pop() + b; break; }
        case NM_EXPR_SUBTRACT: { const float b = pop(); value = pop() - b; break; }
        case NM_EXPR_MULTIPLY: { const float b = pop(); value = pop() * b; break; }
        case NM_EXPR_DIVIDE: { const float b = pop(); value = pop() / b; break; }
        case NM_EXPR_NEGATE: value = -pop(); break;
        case NM_EXPR_SQRT: value = std::sqrt(pop()); break;
        case NM_EXPR_ABS: value = std::abs(pop()); break;
        case NM_EXPR_LOG: value = std::log(pop()); break;
        case NM_EXPR_EXP: value = std::exp(pop()); break;
        case NM_EXPR_POW_INTEGER: value = std::pow(pop(), instruction.integer); break;
        case NM_EXPR_MIN: { const float b = pop(); value = std::min(pop(), b); break; }
        case NM_EXPR_MAX: { const float b = pop(); value = std::max(pop(), b); break; }
        case NM_EXPR_CLAMP: {
            const float upper = pop(), lower = pop();
            value = std::clamp(pop(), lower, upper); break;
        }
        case NM_EXPR_SELECT_NONNEGATIVE: {
            const float negative = pop(), positive = pop(), condition = pop();
            value = condition >= 0.0f ? positive : negative; break;
        }
        case NM_EXPR_EXPM1_MINUS_X: { const float x = pop(); value = std::expm1(x) - x; break; }
        case NM_EXPR_CACHE_STORE:
        case NM_EXPR_CACHE_LOAD:
        case NM_EXPR_FIBER_EXP_LINEAR:
            throw std::runtime_error("unexpected cached or unsupported opcode in Nagasawa stateless fixture");
        default:
            throw std::runtime_error("unknown ordinary FP32 scalar opcode " +
                                     std::to_string(instruction.opcode));
        }
        need(std::isfinite(value), "ordinary FP32 VM produced nonfinite output");
        stack.push_back(value);
    }
    need(stack.size() == 1u, "ordinary FP32 VM did not reduce to one value");
    return stack.back();
}

double evaluateCompensatedFP32(
    const numi::matter::CompiledWorld& world,
    const NMScalarProgramGPU& program,
    const std::vector<float>& parameters,
    const Matrix9& f,
    const Matrix9& direction) {
    std::array<float, 9> high{}, low{}, directionHigh{}, rate{};
    for (unsigned i = 0u; i < 9u; ++i) {
        high[i] = static_cast<float>(f[i]);
        low[i] = static_cast<float>(f[i] - static_cast<double>(high[i]));
        directionHigh[i] = static_cast<float>(direction[i]);
        need(static_cast<double>(directionHigh[i]) == direction[i],
             "fixture tangent direction must be exactly representable in FP32");
    }
    const float zeroState[1]{0.0f};
    const NMPaperCandidateState candidate{zeroState, zeroState};
    bool valid = true;
    MRCompensatedScalar words{};
    (void)nmPaperCompensatedBytecode<0u>(program, world.instructions.data(),
        parameters.data(), zeroState, candidate, high.data(), directionHigh.data(),
        rate.data(), 0.002f, 293.15f, valid, nullptr, low.data(), &words);
    need(valid && std::isfinite(words.high) && std::isfinite(words.low),
         "two-word evaluator rejected stateless Nagasawa scalar program");
    return static_cast<double>(words.high) + static_cast<double>(words.low);
}

numi::matter::CompileResult compileWorldForMaterial(
    const numi::matter::MaterialProgram& material) {
    using namespace numi::matter;
    WorldSource source;
    source.materials = {material};
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.frameTimestep = 2.0e-3;
    ObjectSource object;
    object.name = "stateless_generic_compensated_cpu_fixture";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{0.0, 0.0, 0.0}, {0.001, 0.0, 0.0},
                       {0.0, 0.001, 0.0}, {0.0, 0.0, 0.001}};
    object.tetrahedra = {{0u, 1u, 2u, 3u}};
    object.characteristicLength = 0.001;
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    return compileWorld(source, options);
}

void checkStatelessNagasawa(const std::filesystem::path& materials) {
    using namespace numi::matter;
    MaterialProgram source = parse(materials / "nagasawa2013_medium_orthotropic.nmatter");
    need(source.name == "nagasawa2013_medium_orthotropic" &&
         source.hint == ConstitutiveHint::generic && source.internalState.empty() &&
         source.stateUpdateRoots.empty() && source.stateImplicitRoots.empty() &&
         source.dissipationRoot == NM_INVALID_INDEX && !source.learned.has_value(),
         "retained Nagasawa medium no longer matches the stateless-energy fixture contract");

    const auto ordinary = compileWorldForMaterial(source);
    need(ordinary.succeeded(), "unmodified Nagasawa material no longer compiles");
    need((ordinary.world.scalarPrograms.front().flags & 2u) == 0u,
         "default Nagasawa scalar program unexpectedly selected compensated evaluation");

    MaterialProgram compensatedSource = source;
    compensatedSource.paperCompensatedArithmetic = true;
    const auto compensated = compileWorldForMaterial(compensatedSource);
    if (!compensated.succeeded()) {
        std::string reason = "opt-in Nagasawa compensated world compile rejected";
        for (const auto& diagnostic : compensated.diagnostics) reason += ": " + diagnostic.message;
        throw std::runtime_error(reason);
    }
    need(validateCompiledWorldLayout(compensated.world),
         "compensated Nagasawa cooked layout failed validation");
    const auto& descriptor = compensated.world.materials.at(0u);
    need(descriptor.stateCount == 0u &&
         (descriptor.flags & NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC) != 0u &&
         (descriptor.flags & NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION) == 0u,
         "stateless generic policy cooked into an unrelated state policy");
    need(compensated.world.fingerprint != ordinary.world.fingerprint,
         "compiled world identity failed to bind opt-in arithmetic selection");
    need(compensated.world.instructions.size() == ordinary.world.instructions.size(),
         "compensated policy changed the authored scalar instruction count");
    for (std::size_t i = 0u; i < ordinary.world.instructions.size(); ++i) {
        const auto& a = ordinary.world.instructions[i];
        const auto& b = compensated.world.instructions[i];
        need(a.opcode == b.opcode && a.index == b.index && a.integer == b.integer &&
             a.reserved == b.reserved && a.immediate.x == b.immediate.x &&
             a.immediate.y == b.immediate.y && a.immediate.z == b.immediate.z &&
             a.immediate.w == b.immediate.w,
             "compensated policy modified authored bytecode/equations");
    }
    for (const auto& scalar : compensated.world.scalarPrograms)
        need((scalar.flags & 2u) != 0u,
             "compensated policy failed to mark a compiled scalar program");

    Matrix9 f{1.0 + std::ldexp(1.0, -25), std::ldexp(1.0, -25), 0.0,
              0.0, 1.0 - std::ldexp(1.0, -25), 0.0,
              0.0, 0.0, 1.0};
    Matrix9 direction{1.0, 0.25, -0.125, -0.25, 0.5, 0.125,
                      0.0625, -0.125, 0.25};
    const auto reference = evaluateNagasawaReference(source, f, direction);
    const auto& compiledMaterial = compensated.world.constitutive.at(0u);
    LocalState empty;
    const Matrix9 zero{};
    const auto compiledStress = stress(compiledMaterial, f, empty, empty, 0.002);
    const auto compiledTangent = consistentTangentAction(
        compiledMaterial, f, empty, empty, direction, 0.002);
    std::vector<double> refStress(reference.stress.begin(), reference.stress.end());
    std::vector<double> refTangent(reference.tangent.begin(), reference.tangent.end());
    std::vector<double> compiledStressVector(compiledStress.begin(), compiledStress.end());
    need(relativeError(compiledStressVector, refStress) < 2.0e-11,
         "compiled FP64 bytecode stress disagrees with independent StVK source equations");
    need(relativeError(compiledTangent, refTangent) < 2.0e-11,
         "compiled FP64 tangent disagrees with independent StVK directional derivative");

    const auto parameters = localParameterValues(compensated.world, descriptor);
    std::vector<double> ordinaryStress(9u), compensatedStress(9u),
                        compensatedTangent(9u);
    for (unsigned i = 0u; i < 9u; ++i) {
        ordinaryStress[i] = evaluateOrdinaryFP32(ordinary.world,
            ordinary.world.scalarPrograms.at(ordinary.world.materials[0].stressProgramOffset + i),
            localParameterValues(ordinary.world, ordinary.world.materials[0]), f32(f), f32(zero));
        compensatedStress[i] = evaluateCompensatedFP32(compensated.world,
            compensated.world.scalarPrograms.at(descriptor.stressProgramOffset + i),
            parameters, f, zero);
        compensatedTangent[i] = evaluateCompensatedFP32(compensated.world,
            compensated.world.scalarPrograms.at(descriptor.tangentProgramOffset + i),
            parameters, f, direction);
    }
    const double ordinaryError = relativeError(ordinaryStress, refStress);
    const double compensatedStressError = relativeError(compensatedStress, refStress);
    const double compensatedTangentError = relativeError(compensatedTangent, refTangent);
    need(ordinaryError > 0.25,
         "near-rest fixture did not expose FP32 Green-strain cancellation as expected");
    need(compensatedStressError < 2.0e-5,
         "ordered FP32 stress did not recover the independent near-rest response");
    need(compensatedTangentError < 2.0e-5,
         "ordered FP32 tangent did not match the independent source derivative");
    Matrix9 plus = f, minus = f;
    constexpr double epsilon = 1.0e-5;
    for (unsigned i = 0u; i < 9u; ++i) {
        plus[i] += epsilon * direction[i];
        minus[i] -= epsilon * direction[i];
    }
    const Matrix9 zeroDirection{};
    const auto plusReference = evaluateNagasawaReference(source, plus, zeroDirection);
    const auto minusReference = evaluateNagasawaReference(source, minus, zeroDirection);
    std::vector<double> referenceFD(9u);
    for (unsigned i = 0u; i < 9u; ++i)
        referenceFD[i] = (plusReference.stress[i] - minusReference.stress[i]) /
                         (2.0 * epsilon);
    need(relativeError(referenceFD, refTangent) < 2.0e-8,
         "independent StVK directional derivative failed its source-stress finite difference");

    auto badLayout = compensated.world;
    badLayout.materials[0].flags |= NM_MATERIAL_HAS_STATE;
    badLayout.fingerprint = compiledWorldFingerprint(badLayout);
    std::string layoutError;
    need(!validateCompiledWorldLayout(badLayout, &layoutError),
         "cooked stateless arithmetic layout admitted a stateful material descriptor");
    badLayout = compensated.world;
    auto& badStressInstruction = badLayout.instructions.at(
        badLayout.scalarPrograms.at(descriptor.stressProgramOffset).firstInstruction);
    badStressInstruction.opcode = NM_EXPR_RATE;
    badLayout.fingerprint = compiledWorldFingerprint(badLayout);
    layoutError.clear();
    need(!validateCompiledWorldLayout(badLayout, &layoutError),
         "cooked stateless arithmetic layout admitted a rate-dependent stress opcode");

    MaterialProgram stateful = source;
    stateful.internalState.push_back({"test_state", kDimensionless, 0.0});
    stateful.paperCompensatedArithmetic = true;
    need(!numi::matter::detail::compileConstitutive(stateful, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "stateless generic policy admitted an internal-state material");
    MaterialProgram rateDependent = source;
    Expr rateNode;
    rateNode.kind = ExprKind::deformationRate;
    rateNode.index = 0u;
    rateNode.dimension = kRate;
    rateDependent.energyRoot = rateDependent.expressions.append(rateNode);
    rateDependent.paperCompensatedArithmetic = true;
    need(!numi::matter::detail::compileConstitutive(rateDependent, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "stateless stored-energy policy admitted a rate-dependent energy root");
    MaterialProgram dissipative = source;
    dissipative.dissipationRoot = dissipative.energyRoot;
    dissipative.paperCompensatedArithmetic = true;
    need(!numi::matter::detail::compileConstitutive(dissipative, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "stateless generic policy admitted a dissipation root");

    MaterialProgram fiber = parse(materials / "open_knee_ligament_febio_exp_linear.nmatter");
    fiber.dissipationRoot = NM_INVALID_INDEX;
    fiber.paperCompensatedArithmetic = true;
    const auto fiberWorld = compileWorldForMaterial(fiber);
    need(!fiberWorld.succeeded(),
         "compensated generic compiler admitted unsupported fiber bytecode");
    bool fiberDiagnostic = false;
    for (const auto& diagnostic : fiberWorld.diagnostics)
        fiberDiagnostic |= diagnostic.message.find("fiber-specific exponential") != std::string::npos;
    need(fiberDiagnostic,
         "unsupported compensated fiber bytecode did not fail at compiler admission");

    std::cout << "stateless_generic_material=" << source.name
              << " ordinary_near_rest_stress_relative_error=" << ordinaryError
              << " compensated_stress_relative_error=" << compensatedStressError
              << " compensated_tangent_relative_error=" << compensatedTangentError
              << " policy_fingerprint=" << compensated.world.fingerprint
              << " unsupported_fiber_rejected=true native_metal_execution=not_run physical_validation=false\n";
}

std::array<float, 9> f32(const Matrix9& values) {
    std::array<float, 9> result{};
    for (unsigned i = 0u; i < 9u; ++i) result[i] = static_cast<float>(values[i]);
    return result;
}

void checkSourcePolicy(const std::filesystem::path& materials) {
    using namespace numi::matter;
    using namespace numi::cardboard;
    using namespace numi::cardboard::retained_damage;
    const auto source = parse(materials / "hajali2009_liner_hill_ideal.nmatter");
    auto input = request(source, "liner", 0u, {0.0, 0.0, 0.0}, 0.001);
    const auto built = buildElementProgram(input);
    need(built.program.internalState.size() == 30u,
         "fixture no longer exercises the 30-state retained-damage layout");

    const auto legacy = numi::matter::detail::compileConstitutive(
        built.program, NM_EXPRESSION_STACK_CAPACITY);
    need(legacy.succeeded(), "legacy 30-state material failed to compile");
    MaterialProgram candidate = built.program;
    candidate.genericCompensatedImplicitProjection = true;
    const auto compensated = numi::matter::detail::compileConstitutive(
        candidate, NM_EXPRESSION_STACK_CAPACITY);
    need(compensated.succeeded(), "generic compensated 30-state material was rejected");
    need(compensated.program.gpu.flags &
             NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION,
         "cooked material omitted the generic projection policy bit");
    need(compensated.program.fingerprint != legacy.program.fingerprint,
         "compiled material fingerprint did not bind the projection policy");
    need(compensated.program.material.genericCompensatedImplicitProjection,
         "compiled source-fidelity record lost the projection policy");
    need(numi::cardboard::damage::nativeProgramSignature(candidate) !=
             numi::cardboard::damage::nativeProgramSignature(built.program) &&
         !numi::cardboard::damage::sameNativeProgram(candidate, built.program),
         "element program pooling would alias distinct projection policies");
    need(compensated.program.implicitResiduals.size() == candidate.internalState.size() &&
         compensated.program.implicitJacobians.size() ==
             candidate.internalState.size() * candidate.internalState.size() &&
         compensated.program.implicitDeformationDirections.size() ==
             candidate.internalState.size(),
         "generic policy fixture lost a residual, Jacobian, or deformation derivative");
    for (const auto& scalar : compensated.program.stress)
        need(!scalar.instructions.empty(), "compiled stress program is empty");

    auto invalidRoots = candidate;
    invalidRoots.stateImplicitRoots.pop_back();
    need(!numi::matter::detail::compileConstitutive(
             invalidRoots, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "policy admitted a layout with an unbound implicit state row");
    auto invalidHint = candidate;
    invalidHint.hint = ConstitutiveHint::neoHookean;
    need(!numi::matter::detail::compileConstitutive(
             invalidHint, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "policy admitted a specialized constitutive hint");
    auto hillConflict = candidate;
    hillConflict.paperHillScalarInitializer = true;
    need(!numi::matter::detail::compileConstitutive(
             hillConflict, NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "generic projection combined with the Hill-only initializer");

    const auto compiledWorld = compilePolicyWorld(candidate);
    std::string error;
    need(validateCompiledWorldLayout(compiledWorld, &error),
         "compiled generic-policy world failed layout validation: " + error);
    need((compiledWorld.materials.at(0u).flags &
          NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION) != 0u,
         "world material descriptor omitted the opt-in policy");
    for (const auto& scalar : compiledWorld.scalarPrograms)
        need((scalar.flags & 2u) != 0u,
             "ordered two-word evaluator was not selected for every scalar program");

    auto corrupted = compiledWorld;
    corrupted.materials[0].stateCount = 0u;
    corrupted.fingerprint = compiledWorldFingerprint(corrupted);
    error.clear();
    need(!validateCompiledWorldLayout(corrupted, &error),
         "cooked validator admitted a generic policy with no projected state");
    corrupted = compiledWorld;
    corrupted.materials[0].flags &=
        ~NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION;
    corrupted.fingerprint = compiledWorldFingerprint(corrupted);
    error.clear();
    need(!validateCompiledWorldLayout(corrupted, &error),
         "cooked validator admitted a scalar-bytecode/policy fingerprint mismatch");

    const auto packagePath = std::filesystem::temp_directory_path() /
        "numi-cardboard-generic-compensated-policy-check.nmatterpkg";
    std::filesystem::remove(packagePath);
    CompileResult packageSource;
    packageSource.world = compiledWorld;
    need(writePackage(packageSource, packagePath, &error),
         "could not write policy package: " + error);
    CompiledWorld restored;
    need(readPackage(packagePath, restored, nullptr, &error),
         "could not restore policy package: " + error);
    std::filesystem::remove(packagePath);
    need(validateCompiledWorldLayout(restored, &error),
         "restored policy package failed validation: " + error);
    need(restored.fingerprint == compiledWorld.fingerprint &&
         (restored.materials[0].flags &
          NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION) != 0u,
         "package round trip lost the generic projection policy or fingerprint");

    Matrix9 f{1.0001, 1.0e-5, 0.0,
              0.0, 0.99995, 0.0,
              0.0, 0.0, 1.0};
    Matrix9 direction{1.0, 0.2, 0.0,
                      0.0, -0.15, 0.0,
                      0.0, 0.0, 0.0};
    LocalState accepted(candidate.internalState.size());
    for (std::size_t i = 0u; i < accepted.size(); ++i)
        accepted[i] = candidate.internalState[i].initialValue;
    const auto center = project(compensated.program, f, accepted, 0.002, 1.0e-12);
    const auto tangent = consistentTangentAction(compensated.program, f, accepted,
                                                  center.state, direction, 0.002);
    Matrix9 plus = f, minus = f;
    constexpr double epsilon = 1.0e-6;
    for (std::size_t i = 0u; i < 9u; ++i) {
        plus[i] += epsilon * direction[i];
        minus[i] -= epsilon * direction[i];
    }
    const auto plusState = project(compensated.program, plus, accepted, 0.002, 1.0e-12);
    const auto minusState = project(compensated.program, minus, accepted, 0.002, 1.0e-12);
    const auto plusStress = stress(compensated.program, plus, accepted,
                                   plusState.state, 0.002);
    const auto minusStress = stress(compensated.program, minus, accepted,
                                    minusState.state, 0.002);
    std::vector<double> finiteDifference(9u), tangentVector(9u);
    for (std::size_t i = 0u; i < 9u; ++i) {
        finiteDifference[i] = (plusStress[i] - minusStress[i]) / (2.0 * epsilon);
        tangentVector[i] = tangent[i];
    }
    const double derivativeError = relativeError(tangentVector, finiteDifference);
    need(std::isfinite(derivativeError) && derivativeError < 1.0e-5,
         "independent 30-state source tangent failed the smooth FP64 FD check: " +
             number(derivativeError));
    std::cout << "source_states=" << candidate.internalState.size()
              << " compiled_policy_fingerprint=" << compensated.program.fingerprint
              << " package_policy_roundtrip=pass"
              << " smooth_fp64_tangent_fd_relative_error=" << derivativeError
              << " native_metal_execution=not_run physical_validation=false\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto materials = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path(__FILE__).parent_path().parent_path() / "materials";
        checkCandidateLowWord();
        checkTranscendentals();
        checkSourcePolicy(materials);
        checkStatelessNagasawa(materials);
        std::cout << "generic_compensated_policy_cpu_check_passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generic compensated projection CPU check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
