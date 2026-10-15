#include "numi/matter/compiler.hpp"
#include "numi/matter/detail.hpp"

#include <iostream>

int main(int argc, char** argv) {
    using namespace numi::matter;
    if (argc != 3) return 2;
    auto l = parseMatterFile(argv[1]);
    auto m = parseMatterFile(argv[2]);
    if (!l.succeeded() || !m.succeeded()) return 3;
    l.material.paperHillScalarInitializer = true;
    m.material.paperHillScalarInitializer = true;
    l.material.paperCompensatedArithmetic = true;
    m.material.paperCompensatedArithmetic = true;
    l.material.paperHillAccurateProjection = true;
    m.material.paperHillAccurateProjection = true;
    WorldSource source;
    source.materials = {l.material, m.material};
    CompileOptions options;
    options.localMaterialNewtonIterations = 16u;
    const auto compiled = compileWorld(source, options);
    if (!compiled.succeeded()) {
        for (const auto& diagnostic : compiled.diagnostics)
            std::cerr << diagnostic.message << '\n';
        return 4;
    }
    for (unsigned mat = 0; mat < 2; ++mat) {
        const auto& material = compiled.world.materials[mat];
        const unsigned first = material.implicitResidualProgramOffset;
        const unsigned index = compiled.world.scalarPrograms[first].firstInstruction;
        const auto& instruction = compiled.world.instructions[index];
        std::cout << "mat=" << mat << " offset=" << material.parameterOffset
                  << " flags=" << compiled.world.scalarPrograms[first].flags
                  << " first_parameter_index=" << instruction.index << '\n';
    }
}
