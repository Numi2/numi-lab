#include "cardboard_expression_cache_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace numi::matter;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string diagnostics(const CompileResult& result) {
    std::string text;
    for (const Diagnostic& diagnostic : result.diagnostics)
        text += diagnostic.message + "; ";
    return text;
}

void translate(ObjectSource& object, const double x) {
    for (auto& node : object.femNodes) node[0] += x;
}

} // namespace

int main() {
    try {
        WorldSource source =
            numi_cardboard_expression_cache::makeWorld(
                numi_cardboard_expression_cache::makeMaterial(false));
        source.objects.front().name = "eligible_ordinary_fem_a";

        ObjectSource second = source.objects.front();
        second.name = "eligible_ordinary_fem_b";
        translate(second, 0.03);
        source.objects.push_back(second);

        ObjectSource mixed = second;
        mixed.name = "mixed_fem_fallback_rows";
        mixed.mixedFEM = true;
        translate(mixed, 0.03);
        source.objects.push_back(std::move(mixed));

        source.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::coupledFEMTangentBlockJacobi;
        const CompileResult selected = compileWorld(source);
        require(selected.succeeded(),
            "coupled block selector rejected a multi-object mixed world: " +
                diagnostics(selected));
        require(selected.world.mixedSolver.executionBudgets.z == 6u,
            "compiled world lost the distinct selector 6");
        std::string layoutError;
        require(validateCompiledWorldLayout(selected.world, &layoutError),
            "compiled coupled block layout failed: " + layoutError);
        require(selected.world.objects.size() == 3u &&
                    selected.world.fem.nodes.size() == 12u &&
                    selected.world.fem.tetrahedra.size() == 3u,
            "selector changed the authored multi-object FEM layout");
        require(selected.world.mixedSolver.residualTolerances.x ==
                    static_cast<float>(source.mixedSolver.relativeResidual) &&
                selected.world.mixedSolver.residualTolerances.y ==
                    static_cast<float>(source.mixedSolver.volumeTolerance) &&
                selected.world.mixedSolver.residualTolerances.z ==
                    static_cast<float>(source.mixedSolver.pressureTolerance),
            "selector changed a physical residual/volume/pressure gate");

        WorldSource scalar = source;
        scalar.mixedSolver.femPreconditioner =
            FEMPreconditionerMode::scalarDiagonal;
        const CompileResult baseline = compileWorld(scalar);
        require(baseline.succeeded(),
            "scalar baseline did not compile: " + diagnostics(baseline));
        require(baseline.world.fem.nodes.size() ==
                    selected.world.fem.nodes.size() &&
                baseline.world.fem.tetrahedra.size() ==
                    selected.world.fem.tetrahedra.size(),
            "preconditioner selection changed the physical topology size");
        for (std::size_t i = 0u; i < baseline.world.fem.nodes.size(); ++i) {
            const auto& a = baseline.world.fem.nodes[i].positionAndMass;
            const auto& b = selected.world.fem.nodes[i].positionAndMass;
            require(a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w,
                "preconditioner selection changed initial FEM positions or mass");
        }
        for (std::size_t i = 0u; i < baseline.world.fem.tetrahedra.size(); ++i) {
            const auto& a = baseline.world.fem.tetrahedra[i];
            const auto& b = selected.world.fem.tetrahedra[i];
            require(a.nodes.x == b.nodes.x && a.nodes.y == b.nodes.y &&
                    a.nodes.z == b.nodes.z && a.nodes.w == b.nodes.w &&
                    a.identity.y == b.identity.y,
                "preconditioner selection changed FEM ancestry or ownership");
        }

        WorldSource noEligibleFEM = source;
        for (ObjectSource& object : noEligibleFEM.objects)
            object.mixedFEM = true;
        const CompileResult rejected = compileWorld(noEligibleFEM);
        require(!rejected.succeeded(),
            "coupled block selector admitted a world with no eligible ordinary FEM rows");

        std::cout << "coupled_fem_tangent_block_compiler_check=passed"
                  << " selector=6 objects=3 ordinary_fem=2 mixed_fem=1"
                  << " physical_gates_unchanged=true topology_unchanged=true"
                  << " rejected_without_eligible_fem=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "coupled_fem_tangent_block_compiler_check=failed error=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
