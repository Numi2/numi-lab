#define main cardboard_retained_matrix_damage_legacy_cpu_main
#include "cardboard_retained_matrix_damage_check.cpp"
#undef main

#include "numi/matter/detail.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void need(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

numi::matter::CompileResult compileOneMaterial(
    const numi::matter::MaterialProgram& material,
    const std::uint32_t localIterations = 8u,
    const std::optional<std::uint32_t> accurateOverride = std::nullopt) {
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
    options.localMaterialNewtonIterations = localIterations;
    options.accurateBilinear15LocalNewtonIterations = accurateOverride;
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
         "the standalone constitutive compiler default for this policy changed");
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
    need(world.world.materials[0].localNewtonIterations == 16u,
         "accurate bilinear-15 world default must remain 16 when no override is supplied");
    const auto explicit16 = compileOneMaterial(candidate, 32u, 16u);
    const auto explicit32 = compileOneMaterial(candidate, 32u, 32u);
    need(explicit16.succeeded() && explicit32.succeeded() &&
         explicit16.world.materials[0].localNewtonIterations == 16u &&
         explicit32.world.materials[0].localNewtonIterations == 32u,
         "explicit accurate bilinear-15 budgets 16 and 32 were not preserved by compilation");
    need(explicit16.world.fingerprint != explicit32.world.fingerprint,
         "compiled-world fingerprint did not bind the effective local projection budget");
    const auto explicit8 = compileOneMaterial(candidate, 32u, 8u);
    need(explicit8.succeeded() &&
         explicit8.world.materials[0].localNewtonIterations == 8u,
         "valid explicit accurate bilinear-15 budget 8 was not preserved");
    for (const std::uint32_t invalidBudget : {0u, 33u}) {
        const auto invalid = compileOneMaterial(candidate, 32u, invalidBudget);
        need(!invalid.succeeded(),
             "compiler admitted an accurate bilinear-15 budget outside 1..32");
    }
    const auto legacyWorld = compileOneMaterial(compensatedDefault);
    need(legacyWorld.succeeded() &&
         legacyWorld.world.materials[0].localNewtonIterations == 8u &&
         (legacyWorld.world.materials[0].flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) == 0u,
         "the opt-in policy changed the production default world material budget or flags");
    std::string error;
    need(validateCompiledWorldLayout(world.world, &error),
         "scoped projection world failed package validation: " + error);
    need(validateCompiledWorldLayout(explicit32.world, &error),
         "validator rejected a valid explicit accurate bilinear-15 budget: " + error);
    for (const std::uint32_t invalidBudget : {0u, 33u}) {
        auto wrongBudget = world.world;
        wrongBudget.materials[0].localNewtonIterations = invalidBudget;
        wrongBudget.fingerprint = compiledWorldFingerprint(wrongBudget);
        error.clear();
        need(!validateCompiledWorldLayout(wrongBudget, &error),
             "validator admitted an accurate bilinear-15 budget outside 1..32");
    }
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
    package.world = explicit32.world;
    need(writePackage(package, packagePath, &error),
         "could not serialize the explicit-32 scoped policy package: " + error);
    CompiledWorld restored;
    need(readPackage(packagePath, restored, nullptr, &error),
         "could not restore the explicit-32 scoped policy package: " + error);
    std::filesystem::remove(packagePath);
    need(validateCompiledWorldLayout(restored, &error) &&
         restored.fingerprint == explicit32.world.fingerprint &&
         restored.materials[0].localNewtonIterations == 32u &&
         (restored.materials[0].flags &
          NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION) != 0u,
         "package roundtrip lost the local projection policy or iteration budget");
    std::cout << "bilinear15_scaled_correction_policy_cpu_check_passed"
              << " state_count=15 parameter_count=28 default_iterations=16 explicit_iterations=8,16,32"
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
