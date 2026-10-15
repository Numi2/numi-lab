#define main cardboard_retained_matrix_damage_legacy_cpu_main
#include "cardboard_retained_matrix_damage_check.cpp"
#undef main

#include "numi/matter/detail.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

numi::matter::CompileResult compileOneMaterial(
    const numi::matter::MaterialProgram& material) {
    using namespace numi::matter;
    WorldSource source;
    source.materials = {material};
    source.gravity = {0.0, 0.0, 0.0};
    source.environmentCount = 1u;
    source.frameTimestep = 2.0e-3;
    ObjectSource object;
    object.name = "bilinear15_scaled_correction_cpu_fixture";
    object.representation = Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.femNodes = {{0.0,0.0,0.0},{0.001,0.0,0.0},
                       {0.0,0.001,0.0},{0.0,0.0,0.001}};
    object.tetrahedra = {{0u,1u,2u,3u}};
    object.characteristicLength = 0.001;
    source.objects = {object};
    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    return compileWorld(source, options);
}

void checkBilinear15ScaledCorrectionPolicy(const std::filesystem::path& materials) {
    using namespace numi::matter;
    MaterialProgram source = parse(materials / "hajali2009_medium_hill_bilinear_fit.nmatter");
    need(source.name == "hajali2009_medium_hill_bilinear_fit" &&
         source.internalState.size() == 15u && source.parameters.size() == 28u,
         "bilinear-15 source layout changed; do not silently retarget this policy");

    const auto legacy = numi::matter::detail::compileConstitutive(
        source, NM_EXPRESSION_STACK_CAPACITY);
    need(legacy.succeeded(), "unmodified bilinear-15 source no longer compiles");
    MaterialProgram compensatedDefault = source;
    compensatedDefault.genericCompensatedImplicitProjection = true;
    compensatedDefault.paperCompensatedArithmetic = true;
    const auto compensatedLegacy = numi::matter::detail::compileConstitutive(
        compensatedDefault, NM_EXPRESSION_STACK_CAPACITY);
    need(compensatedLegacy.succeeded(),
         "the production source15 combination of generic implicit and compensated paper arithmetic must compile");
    need(compensatedLegacy.program.gpu.localNewtonIterations == 8u &&
         (compensatedLegacy.program.gpu.flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) == 0u,
         "the default source15 compensated policy changed without the new opt-in flag");

    MaterialProgram candidate = compensatedDefault;
    candidate.genericBilinear15AccurateProjection = true;
    const auto compiled = numi::matter::detail::compileConstitutive(
        candidate, NM_EXPRESSION_STACK_CAPACITY);
    need(compiled.succeeded(), "scaled-correction bilinear-15 policy failed compiler admission");
    need((compiled.program.gpu.flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) != 0u &&
         (compiled.program.gpu.flags &
          NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION) != 0u,
         "compiled material omitted an opt-in projection policy flag");
    need(compiled.program.gpu.localNewtonIterations == 16u,
         "the scoped local iteration budget is not 16");
    need(compiled.program.fingerprint != legacy.program.fingerprint,
         "material fingerprint did not bind the scaled-correction policy");
    need(compiled.program.implicitResiduals.size() == 15u &&
         compiled.program.implicitJacobians.size() == 225u &&
         compiled.program.implicitDeformationDirections.size() == 15u &&
         compiled.program.stressStateDerivatives.size() == 135u,
         "policy changed or omitted complete 15-state derivative programs");

    auto noCompensatedCandidate = candidate;
    noCompensatedCandidate.genericCompensatedImplicitProjection = false;
    need(!numi::matter::detail::compileConstitutive(
             noCompensatedCandidate,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "scaled-correction policy was admitted without the generic compensated evaluator");
    auto hillConflict = candidate;
    hillConflict.paperHillAccurateProjection = true;
    need(!numi::matter::detail::compileConstitutive(
             hillConflict,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "generic bilinear-15 policy combined with an unrelated Hill-only projection policy");
    auto paperArithmeticOnly = source;
    paperArithmeticOnly.paperCompensatedArithmetic = true;
    need(!numi::matter::detail::compileConstitutive(
             paperArithmeticOnly,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "15-state source arithmetic flag was admitted without its generic compensated implicit evaluator");
    auto invalidFloor = candidate;
    for (auto& parameter : invalidFloor.parameters)
        if (parameter.name == "fb_epsilon") parameter.defaultValue = 0.0;
    need(!numi::matter::detail::compileConstitutive(
             invalidFloor,NM_EXPRESSION_STACK_CAPACITY).succeeded(),
         "scaled-correction policy admitted a nonpositive state scale floor");

    const auto world = compileOneMaterial(candidate);
    const std::string worldFailure = world.diagnostics.empty() ? "no compiler diagnostic" : world.diagnostics.front().message;
    need(world.succeeded(), "scoped projection world failed compilation: " + worldFailure);
    const auto legacyWorld = compileOneMaterial(compensatedDefault);
    need(legacyWorld.succeeded() &&
         legacyWorld.world.materials[0].localNewtonIterations == 8u &&
         (legacyWorld.world.materials[0].flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) == 0u,
         "the opt-in policy changed the production default world material budget or flags");
    std::string error;
    need(validateCompiledWorldLayout(world.world, &error),
         "scoped projection world failed package validation: " + error);
    auto wrongBudget = world.world;
    wrongBudget.materials[0].localNewtonIterations = 8u;
    wrongBudget.fingerprint = compiledWorldFingerprint(wrongBudget);
    error.clear();
    need(!validateCompiledWorldLayout(wrongBudget, &error),
         "validator admitted the scaled-correction flag with the old iteration budget");
    auto wrongStateCount = world.world;
    wrongStateCount.materials[0].stateCount = 14u;
    wrongStateCount.fingerprint = compiledWorldFingerprint(wrongStateCount);
    error.clear();
    need(!validateCompiledWorldLayout(wrongStateCount, &error),
         "validator admitted a malformed state count for the specialized convergence policy");

    const auto packagePath = std::filesystem::temp_directory_path() /
        "numi-cardboard-bilinear15-scaled-correction-policy.nmatterpkg";
    std::filesystem::remove(packagePath);
    CompileResult package;
    package.world = world.world;
    need(writePackage(package, packagePath, &error),
         "could not serialize the scoped policy package: " + error);
    CompiledWorld restored;
    need(readPackage(packagePath, restored, nullptr, &error),
         "could not restore the scoped policy package: " + error);
    std::filesystem::remove(packagePath);
    need(validateCompiledWorldLayout(restored, &error) &&
         restored.fingerprint == world.world.fingerprint &&
         restored.materials[0].localNewtonIterations == 16u &&
         (restored.materials[0].flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) != 0u,
         "package roundtrip lost the local projection policy or iteration budget");
    std::cout << "bilinear15_scaled_correction_policy_cpu_check_passed"
              << " state_count=15 parameter_count=28 local_iterations=16"
              << " package_roundtrip=pass native_metal_execution=not_run"
              << " physical_validation=false\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto materials = argc > 1
            ? std::filesystem::path(argv[1])
            : std::filesystem::path(__FILE__).parent_path().parent_path() / "materials";
        checkBilinear15ScaledCorrectionPolicy(materials);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bilinear-15 scaled-correction policy CPU check failed: "
                  << error.what() << '\n';
        return 1;
    }
}
