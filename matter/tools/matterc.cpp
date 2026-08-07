#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path material;
    std::filesystem::path output;
    std::filesystem::path generatedMetal;
    std::string mode = "both";
    std::uint32_t environments = 16u;
};

[[nodiscard]] Arguments parseArguments(const int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value after " + std::string(argument));
            }
            return argv[++index];
        };
        if (argument == "--material") {
            result.material = value();
        } else if (argument == "--output") {
            result.output = value();
        } else if (argument == "--generated-metal") {
            result.generatedMetal = value();
        } else if (argument == "--mode") {
            result.mode = value();
        } else if (argument == "--envs") {
            const std::string text(value());
            const unsigned long parsed = std::stoul(text);
            if (parsed == 0u || parsed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("--envs must be a positive 32-bit integer");
            }
            result.environments = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "numi-matterc --material file.nmatter --output world.nmatterpack "
                   "[--generated-metal file.metal] [--mode mpm|fem|both] [--envs N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    if (result.material.empty() || result.output.empty()) {
        throw std::runtime_error("--material and --output are required");
    }
    if (result.mode != "mpm" && result.mode != "fem" && result.mode != "both") {
        throw std::runtime_error("--mode must be mpm, fem, or both");
    }
    return result;
}

void printDiagnostics(const std::vector<numi::matter::Diagnostic>& diagnostics) {
    for (const auto& diagnostic : diagnostics) {
        const char* severity = diagnostic.severity ==
                numi::matter::Diagnostic::Severity::error
            ? "error"
            : diagnostic.severity == numi::matter::Diagnostic::Severity::warning
                ? "warning"
                : "note";
        std::cerr << severity << ':' << diagnostic.line << ':'
                  << diagnostic.column << ": " << diagnostic.message << '\n';
    }
}

[[nodiscard]] numi::matter::ObjectSource makeMPMCube(
    const std::uint32_t material,
    const double density
) {
    numi::matter::ObjectSource object;
    object.name = "silicone_mpm";
    object.materialIndex = material;
    object.representation = numi::matter::Representation::mpm;
    object.characteristicLength = 0.01;
    object.mpmGridMinimum = {-0.04, -0.04, -0.02};
    object.mpmGridMaximum = {0.04, 0.04, 0.12};
    object.promotionStrain = 0.04;
    object.demotionStrain = 0.002;
    constexpr double spacing = 0.01;
    constexpr double volume = spacing * spacing * spacing;
    for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                numi::matter::ParticleSource particle;
                particle.position = {
                    -0.01 + spacing * static_cast<double>(x),
                    -0.01 + spacing * static_cast<double>(y),
                    0.04 + spacing * static_cast<double>(z),
                };
                particle.mass = density * volume;
                particle.referenceVolume = volume;
                object.particles.push_back(particle);
            }
        }
    }
    return object;
}

[[nodiscard]] numi::matter::ObjectSource makeFEMTetrahedron(
    const std::uint32_t material
) {
    numi::matter::ObjectSource object;
    object.name = "silicone_fem";
    object.materialIndex = material;
    object.representation = numi::matter::Representation::fem;
    object.characteristicLength = 0.02;
    object.femNodes = {
        {0.04, 0.00, 0.04},
        {0.06, 0.00, 0.04},
        {0.04, 0.02, 0.04},
        {0.04, 0.00, 0.06},
    };
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    return object;
}

[[nodiscard]] double materialDensity(
    const numi::matter::MaterialProgram& material
) {
    const auto iterator = std::find_if(
        material.parameters.begin(),
        material.parameters.end(),
        [](const numi::matter::Parameter& parameter) {
            return parameter.name == "density";
        }
    );
    if (iterator == material.parameters.end()) {
        throw std::runtime_error("compiled material has no density parameter");
    }
    return iterator->defaultValue;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        const Arguments arguments = parseArguments(argc, argv);
        numi::matter::ParseResult parsed =
            numi::matter::parseMatterFile(arguments.material);
        printDiagnostics(parsed.diagnostics);
        if (!parsed.succeeded()) {
            return 2;
        }

        numi::matter::WorldSource source;
        source.environmentCount = arguments.environments;
        source.frameTimestep = 1.0 / 240.0;
        source.femPCGIterations = 24u;
        source.identificationCandidates = arguments.environments >= 4u
            ? std::min<std::uint32_t>(arguments.environments & ~1u, 8u)
            : 0u;
        source.materials.push_back(parsed.material);
        numi::matter::RigidProxySource ground;
        ground.shape = NM_RIGID_PLANE;
        ground.localCenter = {0.0, 0.0, 1.0};
        ground.radiusOrOffset = 0.0;
        source.rigidProxies.push_back(ground);
        if (arguments.mode == "mpm" || arguments.mode == "both") {
            source.objects.push_back(makeMPMCube(
                0u,
                materialDensity(parsed.material)
            ));
        }
        if (arguments.mode == "fem" || arguments.mode == "both") {
            source.objects.push_back(makeFEMTetrahedron(0u));
        }

        numi::matter::CompileResult compiled = numi::matter::compileWorld(source);
        printDiagnostics(compiled.diagnostics);
        if (!compiled.succeeded()) {
            return 3;
        }
        std::string packageError;
        if (!numi::matter::writePackage(
                compiled,
                arguments.output,
                &packageError
            )) {
            throw std::runtime_error(packageError);
        }
        if (!arguments.generatedMetal.empty()) {
            std::ofstream stream(
                arguments.generatedMetal,
                std::ios::binary | std::ios::trunc
            );
            if (!stream) {
                throw std::runtime_error(
                    "cannot create generated Metal output"
                );
            }
            stream << compiled.generatedMetal;
        }

        numi::matter::CompiledWorld roundTrip;
        std::string generatedRoundTrip;
        if (!numi::matter::readPackage(
                arguments.output,
                roundTrip,
                &generatedRoundTrip,
                &packageError
            )) {
            throw std::runtime_error("package readback failed: " + packageError);
        }
        if (roundTrip.fingerprint != compiled.world.fingerprint ||
            roundTrip.dispatch.objectCount != compiled.world.dispatch.objectCount ||
            roundTrip.dispatch.particleCount != compiled.world.dispatch.particleCount ||
            roundTrip.dispatch.tetrahedronCount !=
                compiled.world.dispatch.tetrahedronCount ||
            generatedRoundTrip != compiled.generatedMetal) {
            throw std::runtime_error(
                "package readback changed the compiled matter identity"
            );
        }

        std::cout
            << "{\n"
            << "  \"schema\": \"numi.matter.compile.v1\",\n"
            << "  \"fingerprint\": " << compiled.world.fingerprint << ",\n"
            << "  \"environments\": " << source.environmentCount << ",\n"
            << "  \"objects\": " << compiled.world.dispatch.objectCount << ",\n"
            << "  \"materials\": " << compiled.world.dispatch.materialCount << ",\n"
            << "  \"mpm_particles\": " << compiled.world.dispatch.particleCount << ",\n"
            << "  \"mpm_grid_nodes\": " << compiled.world.dispatch.gridNodeCount << ",\n"
            << "  \"fem_nodes\": " << compiled.world.dispatch.femNodeCount << ",\n"
            << "  \"tetrahedra\": " << compiled.world.dispatch.tetrahedronCount << ",\n"
            << "  \"contact_pairs\": " << compiled.world.dispatch.contactPairCount << ",\n"
            << "  \"rate_exponent\": " << compiled.world.dispatch.maximumRateExponent << "\n"
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
