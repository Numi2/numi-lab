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
        require(argc == 7 || argc == 10,
                "usage: ventricular-source-cook asset-dir frames.f64le "
                "tension.f32le frame-sha256 source-nodes.u32le "
                "dof-regions.u32le [output.nmpkg cooked-tension.f32le "
                "cooked-source-nodes.u32le]");
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
        const auto publishedSources = readBinary<std::uint32_t>(argv[5], 218080u);
        const auto publishedRegions = readBinary<std::uint32_t>(argv[6], 218080u);
        constexpr std::array<std::uint32_t, 3> pointOnly{17565u, 170947u, 235754u};
        for (std::uint32_t i = 0u; i < 218077u; ++i)
            require(publishedSources[i] < nodeCount && publishedRegions[i] == 0u &&
                    (i == 0u || publishedSources[i] > publishedSources[i-1u]),
                    "published unique ventricular source-node quotient changed");
        for (std::uint32_t i = 0u; i < pointOnly.size(); ++i)
            require(publishedSources[218077u+i] == pointOnly[i] &&
                    publishedRegions[218077u+i] == 2u,
                    "published point-only LV/RV split changed");

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
        wall.femNodes.reserve(218080u);
        wall.tetrahedra.reserve(ventricularCount);
        wall.femMaterialFrameRotations.reserve(ventricularCount);
        std::vector<float> authoredTensions;
        authoredTensions.reserve(ventricularCount);
        std::vector<std::uint32_t> localNode(nodeCount,
            std::numeric_limits<std::uint32_t>::max());
        std::vector<std::uint32_t> cookedSources;
        cookedSources.reserve(218080u);
        std::vector<std::uint8_t> lvMask(nodeCount), rvMask(nodeCount);
        // Preserve the incumbent first-encounter mechanical order for every
        // source node, then give the three point-only RV contacts their own
        // states. The complete-face interface remains shared.
        for (std::size_t cell = 0u; cell < cellCount; ++cell) {
            if (labels[cell] != 1u && labels[cell] != 2u) continue;
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                const std::uint32_t sourceNode = tetrahedra[cell * 4u + corner];
                require(sourceNode < nodeCount, "source tetrahedron node out of range");
                (labels[cell] == 1u ? lvMask[sourceNode] : rvMask[sourceNode]) = 1u;
                std::uint32_t& local = localNode[sourceNode];
                if (local == std::numeric_limits<std::uint32_t>::max()) {
                    local = static_cast<std::uint32_t>(wall.femNodes.size());
                    wall.femNodes.push_back({nodes[sourceNode * 3u],
                                             nodes[sourceNode * 3u + 1u],
                                             nodes[sourceNode * 3u + 2u]});
                    cookedSources.push_back(sourceNode);
                }
            }
        }
        require(wall.femNodes.size() == 218077u,
                "unique source ventricular node count changed");
        std::vector<std::uint8_t> publishedMask(nodeCount);
        for (std::uint32_t i = 0u; i < 218077u; ++i)
            publishedMask[publishedSources[i]] = 1u;
        std::size_t shared = 0u;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            require((localNode[node] != std::numeric_limits<std::uint32_t>::max())
                    == bool(publishedMask[node]),
                    "published ventricular source-node coverage changed");
            if (lvMask[node] && rvMask[node]) ++shared;
        }
        require(shared == 2631u, "source LV/RV shared-node count changed");
        std::vector<std::uint32_t> rvPointNode(nodeCount,
            std::numeric_limits<std::uint32_t>::max());
        for (const auto sourceNode : pointOnly) {
            require(lvMask[sourceNode] && rvMask[sourceNode],
                    "point-only source node does not touch both ventricles");
            rvPointNode[sourceNode] = static_cast<std::uint32_t>(wall.femNodes.size());
            wall.femNodes.push_back({nodes[sourceNode * 3u],
                                     nodes[sourceNode * 3u + 1u],
                                     nodes[sourceNode * 3u + 2u]});
            cookedSources.push_back(sourceNode);
        }
        for (std::size_t cell = 0u; cell < cellCount; ++cell) {
            if (labels[cell] != 1u && labels[cell] != 2u) {
                require(tension[cell] == 0.0f,
                        "nonventricular source tension is nonzero");
                continue;
            }
            numi::matter::TetrahedronSource tet{};
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                const auto sourceNode = tetrahedra[cell * 4u + corner];
                tet.nodes[corner] = labels[cell] == 2u &&
                    rvPointNode[sourceNode] != std::numeric_limits<std::uint32_t>::max()
                    ? rvPointNode[sourceNode] : localNode[sourceNode];
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
                wall.femNodes.size() == 218080u &&
                cookedSources.size() == 218080u,
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
        require(compiled.world.fem.nodes.size() == cookedSources.size(),
                "cooked point-only node count changed");
        for (std::size_t index = 0u; index < cookedSources.size(); ++index) {
            const auto sourceNode = cookedSources[index];
            const auto& position = compiled.world.fem.nodes[index].positionAndMass;
            require(position.x == static_cast<float>(nodes[3u*sourceNode]) &&
                    position.y == static_cast<float>(nodes[3u*sourceNode+1u]) &&
                    position.z == static_cast<float>(nodes[3u*sourceNode+2u]),
                    "cooked node order differs from source mapping");
            require(std::isfinite(position.w) && position.w > 0.0f,
                    "cooked split node has no positive mass");
        }
        std::size_t cookedCell = 0u;
        for (std::size_t cell = 0u; cell < cellCount; ++cell) {
            if (labels[cell] != 1u && labels[cell] != 2u) continue;
            const auto actual = compiled.world.fem.tetrahedra[cookedCell++].nodes;
            const std::array<std::uint32_t, 4> indices{
                actual.x, actual.y, actual.z, actual.w};
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                const auto sourceNode = tetrahedra[4u*cell+corner];
                const auto expected = labels[cell] == 2u &&
                    rvPointNode[sourceNode] != std::numeric_limits<std::uint32_t>::max()
                    ? rvPointNode[sourceNode] : localNode[sourceNode];
                require(indices[corner] == expected,
                        "cooked tetrahedron crosses point-only LV/RV contact");
            }
        }
        require(cookedCell == ventricularCount,
                "cooked source tetrahedron count changed");
        std::vector<float> mapped;
        std::string mappingError;
        require(numi::matter::cookFEMActiveTensions(
                    source, compiled.world, authoredTensions, mapped,
                    &mappingError), mappingError);
        require(mapped.size() == authoredTensions.size() &&
                std::memcmp(mapped.data(), authoredTensions.data(),
                            mapped.size() * sizeof(float)) == 0,
                "source/cooked full ventricular field differs");
        if (argc == 10) {
            std::string packageError;
            require(numi::matter::writePackage(compiled, argv[7], &packageError),
                    "cannot write full ventricular package: " + packageError);
            std::ofstream stream(argv[8], std::ios::binary | std::ios::trunc);
            require(stream.good(), "cannot create cooked ventricular tension");
            stream.write(reinterpret_cast<const char*>(mapped.data()),
                         static_cast<std::streamsize>(mapped.size() * sizeof(float)));
            require(stream.good(), "cannot write cooked ventricular tension");
            std::ofstream nodeMap(argv[9], std::ios::binary | std::ios::trunc);
            require(nodeMap.good(), "cannot create cooked source-node map");
            nodeMap.write(reinterpret_cast<const char*>(cookedSources.data()),
                          static_cast<std::streamsize>(cookedSources.size() *
                                                       sizeof(std::uint32_t)));
            require(nodeMap.good(), "cannot write cooked source-node map");
        }
        double syntheticMass = 0.0;
        for (const auto& node : compiled.world.fem.nodes)
            syntheticMass += node.positionAndMass.w;
        std::printf("{\"status\":\"full_source_ventricular_cook_pass\","
                    "\"source_cells\":%zu,\"ventricular_tetrahedra\":%zu,"
                    "\"cooked_tetrahedra\":%zu,\"ventricular_nodes\":%zu,"
                    "\"point_only_lv_rv_nodes_split\":3,"
                    "\"full_face_interface_nodes_shared\":2628,"
                    "\"compiled_tetra_node_mapping_checked\":true,"
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
