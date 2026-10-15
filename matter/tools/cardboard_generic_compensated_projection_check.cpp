// CPU/compiler admission and oracle checks for the opt-in generic implicit
// two-word candidate policy. This does not execute the Metal material or
// qualify a physical cardboard response.
#define main cardboard_retained_matrix_damage_legacy_cpu_main
#include "cardboard_retained_matrix_damage_check.cpp"
#undef main

#include "numi/matter/paper_compensated_bytecode.h"

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
        std::cout << "generic_compensated_implicit_projection_cpu_check_passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generic compensated projection CPU check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
