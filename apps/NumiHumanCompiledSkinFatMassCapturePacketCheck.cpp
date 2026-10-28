#include "NumiHumanCompiledSkinFatMassCapture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

struct CsvTable {
    std::unordered_map<std::string, std::size_t> column;
    std::vector<std::vector<std::string>> rows;
};

static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0u;
    for (;;) {
        const auto end = line.find(',', begin);
        fields.push_back(line.substr(begin, end == std::string::npos
                                              ? std::string::npos : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1u;
    }
    return fields;
}

static CsvTable readCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty CSV: " + path);
    auto header = splitCsvLine(line);
    if (!header.empty() && !header.back().empty() && header.back().back() == '\r')
        header.back().pop_back();
    CsvTable result;
    for (std::size_t i = 0; i < header.size(); ++i)
        result.column.emplace(header[i], i);
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto row = splitCsvLine(line);
        if (row.size() != header.size())
            throw std::runtime_error("CSV column count changed: " + path);
        result.rows.push_back(std::move(row));
    }
    return result;
}

static const std::string& field(const CsvTable& table,
                                const std::vector<std::string>& row,
                                const std::string& key) {
    const auto found = table.column.find(key);
    if (found == table.column.end()) throw std::runtime_error("missing CSV column: " + key);
    return row[found->second];
}

static double number(const CsvTable& table, const std::vector<std::string>& row,
                     const std::string& key) {
    return std::stod(field(table, row, key));
}

static std::uint32_t u32(const CsvTable& table, const std::vector<std::string>& row,
                         const std::string& key) {
    const auto value = std::stoull(field(table, row, key));
    if (value > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("CSV uint32 field is out of range: " + key);
    return static_cast<std::uint32_t>(value);
}

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: packet-check <decompressed-csv-directory>");
        const std::string root = argv[1];
        const auto nodes = readCsv(root + "/compiled-skin-fem-nodes.csv");
        const auto tets = readCsv(root + "/compiled-skin-fem-tets.csv");
        const auto raw = readCsv(root + "/premerge-layer-contributions.csv");
        const auto expectedCells = readCsv(root + "/baseline-parent-cell-mass.csv");
        const auto expectedSources = readCsv(root + "/baseline-parent-node-source-mass.csv");
        const auto expectedNodes = readCsv(root + "/baseline-compiled-parent-nodes.csv");
        require(nodes.rows.size() == 7233u && tets.rows.size() == 28632u &&
                    raw.rows.size() == 9644u,
                "unexpected full donor row counts");
        require(expectedCells.rows.size() == tets.rows.size() &&
                    expectedSources.rows.size() == raw.rows.size() &&
                    expectedNodes.rows.size() == nodes.rows.size(),
                "packet row counts differ from donor exports");

        constexpr std::uint32_t objectIndex = 15u;
        constexpr std::uint32_t stateOffset = 1666u;
        constexpr std::uint32_t elementOffset = 3315u;
        std::vector<const std::vector<std::string>*> nodeByLocal(nodes.rows.size(), nullptr);
        for (const auto& row : nodes.rows) {
            const auto local = u32(nodes, row, "object_local_node");
            require(local < nodeByLocal.size() && nodeByLocal[local] == nullptr,
                    "duplicate/out-of-range local node row");
            require(u32(nodes, row, "object_index") == objectIndex,
                    "unexpected parent object index");
            nodeByLocal[local] = &row;
        }
        require(std::all_of(nodeByLocal.begin(), nodeByLocal.end(),
                            [](const auto* row) { return row != nullptr; }),
                "missing local node row");

        using SourceKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
        std::vector<std::uint32_t> donorByNode(nodes.rows.size(), NM_INVALID_INDEX);
        std::vector<V3> localPointByNode(nodes.rows.size());
        std::vector<double> rawDonorMassByNode(nodes.rows.size(), 0.0);
        std::vector<bool> hasDonorPoint(nodes.rows.size(), false);
        std::map<SourceKey, double> expectedSourceMass;
        std::map<SourceKey, double> rawSourceMass;
        for (const auto& row : raw.rows) {
            const auto node = u32(raw, row, "combined_object_local_node_index");
            const auto donor = u32(raw, row, "node_frame_owner_body");
            const auto material = u32(raw, row, "source_material_index");
            require(rawSourceMass.emplace(SourceKey{node,material,donor},
                        number(raw,row,"mass_kg")).second,
                    "raw source node/material/donor identity is duplicated");
            require(node < nodes.rows.size(), "source contribution node is out of range");
            require(field(raw, row, "mass_semantics") == "owner_layer_lumped_nodal_mass",
                    "unexpected source mass semantics");
            if (donorByNode[node] == NM_INVALID_INDEX) donorByNode[node] = donor;
            require(donorByNode[node] == donor,
                    "actual source packet has multiple donor owners at one node");
            const V3 point{number(raw, row, "donor_local_x_m"),
                           number(raw, row, "donor_local_y_m"),
                           number(raw, row, "donor_local_z_m")};
            if (!hasDonorPoint[node]) {
                localPointByNode[node] = point;
                hasDonorPoint[node] = true;
            } else {
                V3 delta{};
                for (std::size_t axis = 0; axis < 3u; ++axis)
                    delta[axis] = localPointByNode[node][axis] - point[axis];
                require(norm(delta) <= 1.0e-12,
                        "actual source packet has multiple local points at one node");
            }
            rawDonorMassByNode[node] += number(raw, row, "mass_kg");
        }
        for (const auto& row : expectedSources.rows) {
            const auto key = SourceKey{u32(expectedSources, row, "parent_node_index"),
                                       u32(expectedSources, row, "material_index"),
                                       u32(expectedSources, row, "donor_body_index")};
            expectedSourceMass[key] += number(expectedSources, row,
                                              "compiled_corner_mass_share_kg");
        }
        require(expectedSourceMass.size() == expectedSources.rows.size(),
                "expected packet splits node/material/donor rows unexpectedly");
        require(rawSourceMass.size()==expectedSourceMass.size(),
                "raw and cooked source identity sets have different sizes");
        for(const auto& [key,mass]:rawSourceMass) {
            const auto cooked=expectedSourceMass.find(key);
            require(cooked!=expectedSourceMass.end(),
                    "cooked source packet changed a raw node/material/donor identity");
            require(std::abs(mass-cooked->second)<=
                        std::max(1.0e-10,5.0e-4*std::max(std::abs(mass),std::abs(cooked->second))),
                    "raw source contribution changed beyond the existing FP32 cook bound");
        }

        numi::matter::WorldSource source;
        source.materials.resize(17u);
        for (std::uint32_t material : {15u, 16u}) {
            source.materials[material].name = "skin-fat-compiled-material";
            source.materials[material].parameters.push_back(
                {.name="density", .defaultValue=1000.0, .lower=1.0, .upper=5000.0});
        }
        source.objects.resize(objectIndex + 1u);
        auto& object = source.objects[objectIndex];
        object.name = "compiled skin-fat parent export object 15";
        object.representation = numi::matter::Representation::fem;
        object.mixedFEM = false;
        object.materialIndex = 15u;
        object.femNodes.resize(nodes.rows.size());
        object.femReferenceNodes.resize(nodes.rows.size());
        object.femReferenceSourceIdentity[0] = 1u;
        object.femMaterialIndices.resize(tets.rows.size());
        object.femMaterialSourceIdentity[0] = 2u;
        object.tetrahedra.resize(tets.rows.size());

        numi::matter::CompiledWorld compiled;
        compiled.objects.resize(objectIndex + 1u);
        auto& descriptor = compiled.objects[objectIndex];
        descriptor.stateOffset = stateOffset;
        descriptor.stateCount = static_cast<std::uint32_t>(nodes.rows.size());
        descriptor.elementOffset = elementOffset;
        descriptor.elementCount = static_cast<std::uint32_t>(tets.rows.size());
        compiled.materials.resize(17u);
        compiled.fem.nodes.resize(stateOffset + nodes.rows.size());
        compiled.fem.tetrahedra.resize(elementOffset + tets.rows.size());
        std::vector<double> compiledDensity(17u, -1.0);

        for (std::size_t local = 0; local < nodes.rows.size(); ++local) {
            const auto& row = *nodeByLocal[local];
            const auto global = u32(nodes, row, "global_node");
            require(global == stateOffset + local, "compiled node offset is not dense as expected");
            const V3 current{number(nodes, row, "position_x"),
                             number(nodes, row, "position_y"),
                             number(nodes, row, "position_z")};
            const V3 rest{number(nodes, row, "rest_x"),
                          number(nodes, row, "rest_y"),
                          number(nodes, row, "rest_z")};
            object.femNodes[local] = current;
            object.femReferenceNodes[local] = rest;
            auto& cooked = compiled.fem.nodes[global];
            cooked.positionAndMass = {static_cast<float>(current[0]),
                                      static_cast<float>(current[1]),
                                      static_cast<float>(current[2]),
                                      static_cast<float>(number(nodes, row, "mass_kg"))};
            cooked.restAndFixed = {static_cast<float>(rest[0]),
                                   static_cast<float>(rest[1]),
                                   static_cast<float>(rest[2]),
                                   static_cast<float>(number(nodes, row, "fixed_value"))};
        }

        for (const auto& row : tets.rows) {
            const auto tetRow = u32(tets, row, "source_tet_row");
            const auto compiledRow = u32(tets, row, "compiled_tet_row");
            require(tetRow < tets.rows.size() && compiledRow == elementOffset + tetRow,
                    "compiled tet offset/order is not dense as expected");
            const auto material = u32(tets, row, "material_index");
            require(u32(tets, row, "object_index") == objectIndex &&
                        u32(tets, row, "compiled_identity_y") == objectIndex &&
                        u32(tets, row, "identity_active_bit") == 1u &&
                        u32(tets, row, "material_match") == 1u &&
                        u32(tets, row, "node_ids_match") == 1u,
                    "native tet identity/node/material checks did not pass");
            std::array<std::uint32_t, 4> local{};
            std::array<std::uint32_t, 4> global{};
            for (std::size_t corner = 0; corner < 4u; ++corner) {
                local[corner] = u32(tets, row, "source_node" + std::to_string(corner));
                global[corner] = u32(tets, row, "global_node" + std::to_string(corner));
                require(local[corner] < nodes.rows.size() &&
                            global[corner] == stateOffset + local[corner],
                        "native parent tet local/global node correspondence failed");
            }
            object.tetrahedra[tetRow].nodes = local;
            object.femMaterialIndices[tetRow] = material;
            const double density = number(tets, row, "compiled_bulk_x");
            if (compiledDensity[material] < 0.0) compiledDensity[material] = density;
            require(compiledDensity[material] == density,
                    "compiled bulk density differs across same-material cells");
            compiled.materials[material].bulk.x = static_cast<float>(density);
            auto& cooked = compiled.fem.tetrahedra[compiledRow];
            cooked.nodes = {global[0], global[1], global[2], global[3]};
            cooked.inverseRestRow0 = {
                static_cast<float>(number(tets, row, "inverse_rest_r0c0")),
                static_cast<float>(number(tets, row, "inverse_rest_r0c1")),
                static_cast<float>(number(tets, row, "inverse_rest_r0c2")),
                static_cast<float>(number(tets, row, "inverse_rest_w_m3"))};
            cooked.identity = {material, objectIndex, 0u, 0u};
        }

        std::vector<std::uint32_t> donors(nodes.rows.size(), NM_INVALID_INDEX);
        std::vector<V3> localPoints(nodes.rows.size());
        std::vector<metalrobo::NumiHumanTissueMassNode> prepared;
        prepared.reserve(nodes.rows.size());
        for (std::size_t local = 0; local < nodes.rows.size(); ++local) {
            require(donorByNode[local] != NM_INVALID_INDEX && hasDonorPoint[local] &&
                        rawDonorMassByNode[local] > 0.0,
                    "actual donor source export omits a node mass owner/point");
            donors[local] = donorByNode[local];
            localPoints[local] = localPointByNode[local];
            prepared.push_back({.nodeIndex=stateOffset + static_cast<std::uint32_t>(local),
                                .donorBody=donorByNode[local],
                                .massKg=rawDonorMassByNode[local],
                                .localPosition=localPointByNode[local]});
        }

        std::vector<double> parentMasses;
        std::vector<ExplicitUnpairedVolumeSourceMassContribution> sourceMass;
        CompiledSkinFatMassCaptureReport report;
        std::string error;
        require(captureCompiledSkinFatParentMassLedger(
                    source, compiled, objectIndex, donors, localPoints, prepared,
                    parentMasses, sourceMass, report, error),
                "actual donor capture failed: " + error);
        require(parentMasses.size() == tets.rows.size() &&
                    sourceMass.size() == expectedSourceMass.size(),
                "actual donor capture output row counts differ from packet");

        double maxCellError = 0.0;
        for (const auto& row : expectedCells.rows) {
            const auto tet = u32(expectedCells, row, "parent_tet_row");
            require(tet < parentMasses.size(), "expected parent tet row out of range");
            const double expected = number(expectedCells, row, "parent_cell_mass_kg");
            maxCellError = std::max(maxCellError, std::abs(parentMasses[tet] - expected));
            require(parentMasses[tet] == expected,
                    "captured parent cell mass differs from emitted full-donor packet");
            const auto& sourceTet = tets.rows[tet];
            require(u32(expectedCells, row, "material_index") ==
                        u32(tets, sourceTet, "material_index"),
                    "parent cell packet material differs from compiled source row");
            for (std::size_t corner = 0; corner < 4u; ++corner)
                require(u32(expectedCells, row, "parent_node" + std::to_string(corner)) ==
                            u32(tets, sourceTet, "source_node" + std::to_string(corner)),
                        "parent cell packet connectivity differs from compiled source row");
        }
        std::map<SourceKey, double> capturedSourceMass;
        for (const auto& row : sourceMass)
            capturedSourceMass[{row.parentNodeIndex, row.materialIndex,
                                row.donorBodyIndex}] += row.massKg;
        require(capturedSourceMass.size() == expectedSourceMass.size(),
                "captured source mass key set differs from emitted packet");
        double maxSourceError = 0.0;
        for (const auto& [key, expected] : expectedSourceMass) {
            const auto found = capturedSourceMass.find(key);
            require(found != capturedSourceMass.end(), "captured source row key missing from packet");
            maxSourceError = std::max(maxSourceError, std::abs(found->second - expected));
            require(found->second == expected,
                    "captured node/material/donor source mass differs from emitted packet");
        }

        double maxNodePacketError = 0.0;
        for (const auto& row : expectedNodes.rows) {
            const auto local = u32(expectedNodes, row, "parent_node_index");
            require(local < nodes.rows.size(), "expected node packet local index out of range");
            const auto& actual = *nodeByLocal[local];
            require(u32(expectedNodes, row, "compiled_global_node") ==
                        u32(nodes, actual, "global_node"),
                    "node packet global identity differs");
            const double expectedMass = number(expectedNodes, row, "compiled_lumped_mass_kg");
            const double actualMass = number(nodes, *nodeByLocal[local], "mass_kg");
            maxNodePacketError = std::max(maxNodePacketError, std::abs(expectedMass - actualMass));
            require(expectedMass == actualMass &&
                        u32(expectedNodes, row, "node_frame_owner_body") == donors[local],
                    "node packet compiled mass or owner differs");
            require(number(expectedNodes, row, "donor_local_x_m") == localPoints[local][0] &&
                        number(expectedNodes, row, "donor_local_y_m") == localPoints[local][1] &&
                        number(expectedNodes, row, "donor_local_z_m") == localPoints[local][2],
                    "node packet donor-local frame point differs");
            for (const auto& pair : {std::pair{"current_x_m", "position_x"},
                                     std::pair{"current_y_m", "position_y"},
                                     std::pair{"current_z_m", "position_z"},
                                     std::pair{"rest_x_m", "rest_x"},
                                     std::pair{"rest_y_m", "rest_y"},
                                     std::pair{"rest_z_m", "rest_z"}})
                require(number(expectedNodes, row, pair.first) ==
                            number(nodes, *nodeByLocal[local], pair.second),
                        "node packet current/rest coordinate differs");
        }

        std::cout << "compiled skin/fat full donor packet replay PASS"
                  << " nodes=" << report.nodeCount
                  << " tets=" << report.tetrahedronCount
                  << " source_rows=" << sourceMass.size()
                  << " parent_mass_kg=" << report.parentMassKg
                  << " exact_cell_rows=" << parentMasses.size()
                  << " exact_source_keys=" << capturedSourceMass.size()
                  << " max_cell_error_kg=" << maxCellError
                  << " max_source_error_kg=" << maxSourceError
                  << " max_node_packet_error_kg=" << maxNodePacketError
                  << " compiled_nodal_closure_kg="
                  << report.maximumCompiledNodalMassClosureErrorKg
                  << " prepared_donor_closure_kg="
                  << report.maximumPreparedDonorMassClosureErrorKg << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compiled skin/fat full donor packet replay FAIL: " << error.what() << '\n';
        return 1;
    }
}
