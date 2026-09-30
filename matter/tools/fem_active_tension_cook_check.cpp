#include "numi/matter/matter.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_CARDIAC_MATERIAL
#define NUMI_MATTER_CARDIAC_MATERIAL ""
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

numi::matter::ObjectSource object(const char* name,
                                   std::uint32_t material,
                                   double shift) {
    numi::matter::ObjectSource result;
    result.name = name;
    result.representation = numi::matter::Representation::fem;
    result.materialIndex = material;
    result.mixedFEM = false;
    result.deformableContact = false;
    result.deformableSelfContact = false;
    result.characteristicLength = 0.01;
    result.femNodes = {{shift, 0.0, 0.0}, {shift + 0.01, 0.0, 0.0},
                       {shift, 0.01, 0.0}, {shift, 0.0, 0.01}};
    result.femFixedNodes = {0u, 1u, 2u};
    result.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    return result;
}

} // namespace

int main() {
    try {
        auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_CARDIAC_MATERIAL);
        require(parsed.succeeded(), "cardiac source material did not parse");
        for (auto& parameter : parsed.material.parameters)
            if (parameter.name == "density") parameter.defaultValue = 1050.0;
        parsed.material.mixed.fibreDirection = {1.0, 0.0, 0.0};
        parsed.material.mixed.maximumActiveTension = 120000.0;
        numi::matter::WorldSource source;
        source.environmentCount = 2u;
        source.frameTimestep = 1.0e-4;
        source.gravity = {0.0, 0.0, 0.0};
        source.materials.push_back(parsed.material);
        auto second = parsed.material;
        second.name = "source_ventricular_tension_bound_50pa";
        second.mixed.maximumActiveTension = 50.0;
        source.materials.push_back(std::move(second));
        auto first = object("first", 0u, 0.0);
        first.femCapacity.tetrahedra = 3u;
        source.objects.push_back(std::move(first));
        source.objects.push_back(object("second", 1u, 0.03));
        numi::matter::CompileOptions options;
        options.maximumRateExponent = 0u;
        auto compiled = numi::matter::compileWorld(source, options);
        std::string compilation = "two source FEM objects did not compile";
        for (const auto& diagnostic : compiled.diagnostics)
            compilation += "; " + diagnostic.message;
        require(compiled.succeeded(), compilation);
        require(compiled.world.fem.tetrahedra.size() == 4u,
                "capacity fixture must have four cooked slots");

        std::vector<float> cooked;
        std::string error;
        const std::array<float, 4> tensions{100.0f, 40.0f, 110.0f, 0.0f};
        require(numi::matter::cookFEMActiveTensions(
                    source, compiled.world, tensions, cooked, &error), error);
        require(cooked == std::vector<float>({100.0f, 0.0f, 0.0f, 40.0f,
                                              110.0f, 0.0f, 0.0f, 0.0f}),
                "source fields did not map to environment-major cooked order");

        const auto rejected = [&](const numi::matter::WorldSource& sourceInput,
                                  const numi::matter::CompiledWorld& worldInput,
                                  const std::vector<float>& values,
                                  const char* reason) {
            auto unchanged = cooked;
            require(!numi::matter::cookFEMActiveTensions(
                        sourceInput, worldInput, values, cooked, &error) &&
                    error.find(reason) != std::string::npos &&
                    cooked == unchanged,
                    std::string("invalid mapping accepted or changed output: ") + reason);
        };
        rejected(source, compiled.world, {100.0f}, "cover every authored");
        rejected(source, compiled.world, {100.0f, 51.0f, 110.0f, 0.0f},
                 "invalid for its cooked material");
        rejected(source, compiled.world,
                 {100.0f, 40.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f},
                 "invalid for its cooked material");
        auto reordered = source;
        std::swap(reordered.objects[0].tetrahedra[0].nodes[1],
                  reordered.objects[0].tetrahedra[0].nodes[2]);
        rejected(reordered, compiled.world, {100.0f, 40.0f, 110.0f, 0.0f},
                 "node order differs");
        auto activatedCapacity = compiled.world;
        activatedCapacity.fem.tetrahedra[1].identity.w = NM_OBJECT_ACTIVE;
        rejected(source, activatedCapacity, {100.0f, 40.0f, 110.0f, 0.0f},
                 "capacity slot is active");
        auto mixed = source;
        mixed.objects[0].mixedFEM = true;
        rejected(mixed, compiled.world, {100.0f, 40.0f, 110.0f, 0.0f},
                 "object layout differs");
        std::printf("{\"status\":\"source_to_cooked_active_tension_pass\","
                    "\"environments\":2,\"authored_tetrahedra\":2,"
                    "\"cooked_slots\":4,\"controls\":6}\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FEM active-tension cook check: %s\n", error.what());
        return 1;
    }
}
