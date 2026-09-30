#include "numi/matter/matter.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

template <typename T>
std::vector<T> readBinary(const std::filesystem::path& path,
                          std::size_t count) {
    require(std::filesystem::is_regular_file(path),
            "missing source buffer " + path.filename().string());
    require(std::filesystem::file_size(path) == count * sizeof(T),
            "wrong source buffer size " + path.filename().string());
    std::vector<T> result(count);
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open " + path.filename().string());
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    require(input.good(), "cannot read " + path.filename().string());
    return result;
}

std::array<std::uint64_t, 4> sourceIdentity(const std::string& sha) {
    require(sha.size() == 64u, "frame digest must contain 64 hex digits");
    std::array<std::uint64_t, 4> result{};
    for (std::size_t part = 0u; part < result.size(); ++part) {
        const std::string word = sha.substr(part * 16u, 16u);
        require(word.find_first_not_of("0123456789abcdef") == std::string::npos,
                "frame digest must be lowercase hexadecimal");
        result[part] = std::stoull(word, nullptr, 16);
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        static_assert(std::endian::native == std::endian::little);
        require(argc == 5 || argc == 7,
                "usage: ventricular-source-cook asset-dir frames.f64le "
                "tension.f32le frame-sha256 [output.nmpkg cooked-tension.f32le]");
        constexpr std::size_t nodeCount = 300965u;
        constexpr std::size_t cellCount = 1470083u;
        constexpr std::size_t ventricularCount = 1097534u;
        const std::filesystem::path asset = argv[1];
        const auto nodes = readBinary<double>(asset / "nodes.f64le", nodeCount * 3u);
        const auto tetrahedra = readBinary<std::uint32_t>(
            asset / "tetrahedra.u32le", cellCount * 4u);
        const auto labels = readBinary<std::uint32_t>(
            asset / "labels.u32le", cellCount);
        const auto frames = readBinary<double>(argv[2], cellCount * 4u);
        const auto tension = readBinary<float>(argv[3], cellCount);
        const auto frameIdentity = sourceIdentity(argv[4]);

        auto parsed = numi::matter::parseMatterFile(NUMI_MATTER_CARDIAC_MATERIAL);
        require(parsed.succeeded(), "source ventricular law did not parse");
        bool foundDensity = false;
        for (auto& parameter : parsed.material.parameters) {
            if (parameter.name != "density") continue;
            parameter.defaultValue = 1050.0; // Synthetic inertial fixture.
            foundDensity = true;
        }
        require(foundDensity, "source law has no density sentinel");
        parsed.material.mixed.fibreDirection = {1.0, 0.0, 0.0};
        parsed.material.mixed.maximumActiveTension = 120000.0;

        numi::matter::WorldSource source;
        source.environmentCount = 1u;
        source.frameTimestep = 1.0e-6;
        source.gravity = {0.0, 0.0, 0.0};
        source.materials.push_back(std::move(parsed.material));
        numi::matter::ObjectSource wall;
        wall.name = "source_ventricular_only_synthetic_inertia";
        wall.representation = numi::matter::Representation::fem;
        wall.materialIndex = 0u;
        wall.mixedFEM = false;
        wall.deformableContact = false;
        wall.deformableSelfContact = false;
        wall.characteristicLength = 0.01;
        wall.femMaterialFrameSourceIdentity = frameIdentity;
        wall.femNodes.reserve(218077u);
        wall.tetrahedra.reserve(ventricularCount);
        wall.femMaterialFrameRotations.reserve(ventricularCount);
        std::vector<float> authoredTensions;
        authoredTensions.reserve(ventricularCount);
        std::vector<std::uint32_t> localNode(nodeCount,
            std::numeric_limits<std::uint32_t>::max());
        for (std::size_t cell = 0u; cell < cellCount; ++cell) {
            if (labels[cell] != 1u && labels[cell] != 2u) {
                require(tension[cell] == 0.0f,
                        "nonventricular source tension is nonzero");
                continue;
            }
            numi::matter::TetrahedronSource tet{};
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                const std::uint32_t sourceNode = tetrahedra[cell * 4u + corner];
                require(sourceNode < nodeCount, "source tetrahedron node out of range");
                std::uint32_t& local = localNode[sourceNode];
                if (local == std::numeric_limits<std::uint32_t>::max()) {
                    local = static_cast<std::uint32_t>(wall.femNodes.size());
                    wall.femNodes.push_back({nodes[sourceNode * 3u],
                                             nodes[sourceNode * 3u + 1u],
                                             nodes[sourceNode * 3u + 2u]});
                }
                tet.nodes[corner] = local;
            }
            wall.tetrahedra.push_back(tet);
            wall.femMaterialFrameRotations.push_back({
                frames[cell * 4u], frames[cell * 4u + 1u],
                frames[cell * 4u + 2u], frames[cell * 4u + 3u]});
            require(std::isfinite(tension[cell]) && tension[cell] >= 0.0f &&
                    tension[cell] <= 120000.0f,
                    "invalid source ventricular tension");
            authoredTensions.push_back(tension[cell]);
        }
        require(wall.tetrahedra.size() == ventricularCount &&
                wall.femNodes.size() == 218077u,
                "source ventricular topology count differs");
        wall.femFixedNodes = {0u, 1u, 2u}; // Explicit synthetic support.
        source.objects.push_back(std::move(wall));

        numi::matter::CompileOptions options;
        options.maximumRateExponent = 0u;
        const auto compiled = numi::matter::compileWorld(source, options);
        std::string failure = "full source ventricular FEM cook failed";
        for (std::size_t index = 0u;
             index < compiled.diagnostics.size() && index < 5u; ++index)
            failure += "; " + compiled.diagnostics[index].message;
        require(compiled.succeeded(), failure);
        std::vector<float> mapped;
        std::string mappingError;
        require(numi::matter::cookFEMActiveTensions(
                    source, compiled.world, authoredTensions, mapped,
                    &mappingError), mappingError);
        require(mapped.size() == authoredTensions.size() &&
                std::memcmp(mapped.data(), authoredTensions.data(),
                            mapped.size() * sizeof(float)) == 0,
                "source/cooked full ventricular field differs");
        if (argc == 7) {
            std::string packageError;
            require(numi::matter::writePackage(compiled, argv[5], &packageError),
                    "cannot write full ventricular package: " + packageError);
            std::ofstream stream(argv[6], std::ios::binary | std::ios::trunc);
            require(stream.good(), "cannot create cooked ventricular tension");
            stream.write(reinterpret_cast<const char*>(mapped.data()),
                         static_cast<std::streamsize>(mapped.size() * sizeof(float)));
            require(stream.good(), "cannot write cooked ventricular tension");
        }
        double syntheticMass = 0.0;
        for (const auto& node : compiled.world.fem.nodes)
            syntheticMass += node.positionAndMass.w;
        std::printf("{\"status\":\"full_source_ventricular_cook_pass\","
                    "\"source_cells\":%zu,\"ventricular_tetrahedra\":%zu,"
                    "\"cooked_tetrahedra\":%zu,\"ventricular_nodes\":%zu,"
                    "\"synthetic_density_kg_m3\":1050,"
                    "\"synthetic_fixed_nodes\":3,"
                    "\"synthetic_assembled_mass_kg\":%.12g,"
                    "\"source_to_cooked_float32_bitwise\":true,"
                    "\"accepted_native_steps\":0,\"heartbeat_qualified\":false}\n",
                    cellCount, authoredTensions.size(),
                    compiled.world.fem.tetrahedra.size(),
                    compiled.world.fem.nodes.size(), syntheticMass);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ventricular source cook: %s\n", error.what());
        return 1;
    }
}
