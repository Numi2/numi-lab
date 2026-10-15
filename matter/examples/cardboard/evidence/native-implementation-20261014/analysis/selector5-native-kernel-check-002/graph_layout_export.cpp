#include "../../snapshots/source-angular-and-graph-schwarz-integrated-065/matter/src/regional_graph_schwarz.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void readExact(std::istream& input, T& value, const char* label) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error(std::string("truncated input at ") + label);
}

template <typename T>
void writeVector(std::ostream& output, const char* name,
                 const std::vector<T>& values) {
    const std::uint64_t count = values.size();
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error(std::string("failed writing ") + name);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: graph_layout_export INPUT.bin OUTPUT.bin\n";
            return 64;
        }
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open input binary");
        char magic[8]{};
        input.read(magic, sizeof(magic));
        if (!input || std::string(magic, sizeof(magic)) != "NMGRA001")
            throw std::runtime_error("bad graph-layout input magic");
        std::uint32_t version = 0u;
        std::uint32_t nodeCount = 0u;
        std::uint32_t tetCount = 0u;
        std::uint32_t environmentCount = 0u;
        readExact(input, version, "version");
        readExact(input, nodeCount, "node count");
        readExact(input, tetCount, "tet count");
        readExact(input, environmentCount, "environment count");
        if (version != 1u) throw std::runtime_error("unsupported input version");
        std::vector<std::uint8_t> freeMask(nodeCount);
        input.read(reinterpret_cast<char*>(freeMask.data()),
                   static_cast<std::streamsize>(freeMask.size()));
        std::vector<std::array<std::uint32_t, 4>> tetrahedra(tetCount);
        input.read(reinterpret_cast<char*>(tetrahedra.data()),
                   static_cast<std::streamsize>(tetrahedra.size() * sizeof(tetrahedra[0])));
        if (!input) throw std::runtime_error("truncated input arrays");
        char trailing = 0;
        if (input.read(&trailing, 1)) throw std::runtime_error("unexpected trailing input bytes");

        const auto layout = numi::matter::detail::buildRegionalGraphSchwarzLayout(
            nodeCount, std::span<const std::array<std::uint32_t, 4>>(tetrahedra),
            std::span<const std::uint8_t>(freeMask), environmentCount);
        if (!layout.error.empty()) throw std::runtime_error(layout.error);

        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output binary");
        output.write("NMGRO001", 8);
        const std::uint32_t values[] = {
            1u, nodeCount, tetCount, environmentCount,
            static_cast<std::uint32_t>(layout.patches.size()),
            static_cast<std::uint32_t>(layout.patchNodes.size()),
            layout.totalPatchDofs, layout.matrixFloatCount,
            layout.pivotCount};
        output.write(reinterpret_cast<const char*>(values), sizeof(values));
        writeVector(output, "patches", layout.patches);
        writeVector(output, "patch_nodes", layout.patchNodes);
        writeVector(output, "rows", layout.rows);
        writeVector(output, "node_patch_offsets", layout.nodePatchOffsets);
        writeVector(output, "node_patch_values", layout.nodePatchValues);
        writeVector(output, "node_coverage", layout.nodeCoverage);
        const std::uint64_t deviceBytes = layout.deviceBytes;
        output.write(reinterpret_cast<const char*>(&deviceBytes), sizeof(deviceBytes));
        if (!output) throw std::runtime_error("failed writing layout header");
        output.close();
        if (!output) throw std::runtime_error("failed closing layout output");

        std::cout << "graph_layout_export=pass"
                  << " nodes=" << nodeCount
                  << " tetrahedra=" << tetCount
                  << " environments=" << environmentCount
                  << " patches=" << layout.patches.size()
                  << " patch_nodes=" << layout.patchNodes.size()
                  << " dofs=" << layout.totalPatchDofs
                  << " matrix_floats=" << layout.matrixFloatCount
                  << " pivots=" << layout.pivotCount
                  << " metadata_bytes=" << layout.metadataBytes
                  << " device_bytes=" << layout.deviceBytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph_layout_export=fail message=" << error.what() << '\n';
        return 1;
    }
}
