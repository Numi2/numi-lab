#include "numi/matter/matter.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <typename T>
std::vector<T> read(const std::filesystem::path& path, std::size_t count) {
    require(std::filesystem::is_regular_file(path) &&
            std::filesystem::file_size(path) == count*sizeof(T),
            "source or cooked-map file has wrong size");
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count*sizeof(T)));
    require(bool(stream), "source or cooked-map file read failed");
    return values;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4,
                "usage: ventricular-source-topology-check package.nmpkg "
                "asset-dir cooked-source-nodes.u32le");
        numi::matter::CompiledWorld world;
        std::string error;
        require(numi::matter::readPackage(argv[1], world, nullptr, &error),
                "cannot read ventricular package");
        constexpr std::size_t sourceNodeCount = 300965u;
        constexpr std::size_t sourceCellCount = 1470083u;
        constexpr std::size_t cookedNodeCount = 218080u;
        constexpr std::size_t ventricularCellCount = 1097534u;
        require(world.fem.nodes.size() == cookedNodeCount &&
                world.fem.tetrahedra.size() == ventricularCellCount,
                "cooked ventricular dimensions changed");
        const auto asset = std::filesystem::path(argv[2]);
        const auto nodes = read<double>(asset/"nodes.f64le", 3u*sourceNodeCount);
        const auto tetrahedra = read<std::uint32_t>(
            asset/"tetrahedra.u32le", 4u*sourceCellCount);
        const auto labels = read<std::uint32_t>(asset/"labels.u32le", sourceCellCount);
        const auto mapping = read<std::uint32_t>(argv[3], cookedNodeCount);
        constexpr std::array<std::uint32_t, 3> pointOnly{17565u, 170947u, 235754u};
        for (std::size_t i = 0; i < pointOnly.size(); ++i)
            require(mapping[218077u+i] == pointOnly[i],
                    "point-only RV cooked-source identity changed");
        std::vector<std::uint32_t> first(sourceNodeCount,
            std::numeric_limits<std::uint32_t>::max());
        std::uint32_t next = 0u;
        std::size_t lvCells = 0u, rvCells = 0u, cookedCell = 0u;
        for (std::size_t cell = 0u; cell < sourceCellCount; ++cell) {
            const auto label = labels[cell];
            if (label != 1u && label != 2u) continue;
            if (label == 1u) ++lvCells;
            else ++rvCells;
            const auto actual = world.fem.tetrahedra[cookedCell++].nodes;
            const std::array<std::uint32_t, 4> indices{
                actual.x, actual.y, actual.z, actual.w};
            for (std::size_t corner = 0u; corner < 4u; ++corner) {
                const auto source = tetrahedra[4u*cell+corner];
                require(source < sourceNodeCount, "source node out of range");
                if (first[source] == std::numeric_limits<std::uint32_t>::max()) {
                    require(next < 218077u && mapping[next] == source,
                            "first-encounter source order changed");
                    first[source] = next++;
                }
                std::uint32_t expected = first[source];
                if (label == 2u)
                    for (std::uint32_t i = 0u; i < pointOnly.size(); ++i)
                        if (source == pointOnly[i]) expected = 218077u+i;
                require(indices[corner] == expected,
                        "package cell crosses point-only LV/RV source contact");
            }
        }
        require(next == 218077u && cookedCell == ventricularCellCount &&
                lvCells == 722773u && rvCells == 374761u,
                "source ventricular partition changed");
        double totalMass = 0.0;
        for (std::size_t index = 0u; index < cookedNodeCount; ++index) {
            const auto source = mapping[index];
            require(source < sourceNodeCount, "cooked source identity out of range");
            const auto& node = world.fem.nodes[index];
            const auto& position = node.positionAndMass;
            const auto& rest = node.restAndFixed;
            const float x = static_cast<float>(nodes[3u*source]);
            const float y = static_cast<float>(nodes[3u*source+1u]);
            const float z = static_cast<float>(nodes[3u*source+2u]);
            require(position.x == x && position.y == y && position.z == z &&
                    rest.x == x && rest.y == y && rest.z == z &&
                    std::isfinite(position.w) && position.w > 0.0f,
                    "cooked node geometry or mass differs from source");
            totalMass += position.w;
        }
        std::printf("{\"status\":\"full_source_package_point_contact_split_verified\","
                    "\"ventricular_tetrahedra\":%zu,\"ventricular_nodes\":%zu,"
                    "\"lv_cells\":%zu,\"rv_cells\":%zu,"
                    "\"point_only_lv_rv_nodes_split\":3,"
                    "\"source_to_cooked_tetrahedra_checked\":%zu,"
                    "\"synthetic_assembled_mass_kg\":%.12g,"
                    "\"heartbeat_qualified\":false}\n",
                    cookedCell, world.fem.nodes.size(), lvCells, rvCells,
                    cookedCell, totalMass);
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "ventricular topology check: %s\n",
                     exception.what());
        return 1;
    }
}
