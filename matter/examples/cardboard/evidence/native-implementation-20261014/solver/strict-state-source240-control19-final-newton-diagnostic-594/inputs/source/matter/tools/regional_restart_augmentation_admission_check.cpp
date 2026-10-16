#include "numi/matter/compiler.hpp"
#include "numi/matter/regional_restart_augmentation_admission.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace numi::matter;

namespace {
unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

std::string diagnosticText(const CompileResult& result) {
    std::string text;
    for (const auto& diagnostic : result.diagnostics)
        text += diagnostic.message + "; ";
    return text;
}

CompileResult compileFixture(const MaterialProgram& material,
                             const bool enableMultiphysics = false) {
    WorldSource source;
    source.gravity = {0.0, 0.0, 0.0};
    source.frameTimestep = 1.0e-4;
    source.materials = {material};

    ObjectSource object;
    object.name = "regional_restart_mechanical_admission_fixture";
    object.representation = Representation::fem;
    object.materialIndex = 0u;
    object.characteristicLength = 0.01;
    object.mixedFEM = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.multiphysics.enabled = enableMultiphysics;
    object.femNodes = {{0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
                       {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01}};
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    source.objects.push_back(std::move(object));

    CompileOptions options;
    options.maximumRateExponent = 0u;
    options.emitSpecializedMetal = false;
    return compileWorld(source, options);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected a source-bound FEM material asset path");
        const auto parsed = parseMatterFile(argv[1]);
        require(parsed.succeeded(), "material parse failed");

        const CompileResult mechanical = compileFixture(parsed.material);
        require(mechanical.succeeded(),
                "mechanical fixture compile failed: " + diagnosticText(mechanical));
        const CompiledWorld& compiled = mechanical.world;
        require(compiled.dispatch.femNodeCount == 4u,
                "compiler did not produce the expected four-node FEM fixture");
        require(compiled.fem.fields.size() == compiled.dispatch.femNodeCount &&
                !compiled.fem.fields.empty(),
                "compiler must create a nonempty field-state backing arena per FEM node");
        require(detail::regionalRestartFieldsAreMechanical(
                    compiled.objects, compiled.fem.fields,
                    compiled.dispatch.femNodeCount),
                "finite compiler-generated mechanical field rows should be admitted");
        require(compiled.fem.fields[0].primary.y != 0.0f,
                "fixture must retain the compiler's finite nonzero default temperature");

        const CompileResult multiphysics = compileFixture(parsed.material, true);
        require(multiphysics.succeeded(),
                "multiphysics control fixture compile failed: " +
                    diagnosticText(multiphysics));
        require(!detail::regionalRestartFieldsAreMechanical(
                    multiphysics.world.objects, multiphysics.world.fem.fields,
                    multiphysics.world.dispatch.femNodeCount),
                "compiled multiphysics object flag/state must be rejected");

        auto altered = compiled;
        altered.fem.fields[0].secondary.w = 1.0f;
        require(!detail::regionalRestartFieldsAreMechanical(
                    altered.objects, altered.fem.fields,
                    altered.dispatch.femNodeCount),
                "active per-row multiphysics flag must be rejected");
        altered = compiled;
        altered.fem.fields[0].primary.z =
            std::numeric_limits<float>::quiet_NaN();
        require(!detail::regionalRestartFieldsAreMechanical(
                    altered.objects, altered.fem.fields,
                    altered.dispatch.femNodeCount),
                "nonfinite field payload must be rejected");
        altered = compiled;
        altered.fem.fields.pop_back();
        require(!detail::regionalRestartFieldsAreMechanical(
                    altered.objects, altered.fem.fields,
                    altered.dispatch.femNodeCount),
                "truncated field backing arena must be rejected");
        altered = compiled;
        altered.fem.fields[1].secondary.y =
            std::numeric_limits<float>::infinity();
        require(!detail::regionalRestartFieldsAreMechanical(
                    altered.objects, altered.fem.fields,
                    altered.dispatch.femNodeCount),
                "infinite accumulated field history must be rejected");

        std::cout << "PASS regional restart field admission checks=" << checks
                  << " compiled_fem_nodes=" << compiled.dispatch.femNodeCount
                  << " backing_field_rows=" << compiled.fem.fields.size()
                  << " physics_steps=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL regional restart field admission after " << checks
                  << " checks: " << error.what() << '\n';
        return 1;
    }
}
