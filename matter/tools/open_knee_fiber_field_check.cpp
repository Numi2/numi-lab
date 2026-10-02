#include "numi/matter/open_knee_fiber_field.hpp"
#include "numi/matter/compiler.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using namespace numi_matter_open_knee;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double materialParameter(
    const numi::matter::MaterialProgram& material,
    const std::string& name
) {
    for (const auto& parameter : material.parameters)
        if (parameter.name == name) return parameter.defaultValue;
    throw std::runtime_error("source meniscus material lacks parameter " + name);
}

void appendU32LE(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned shift = 0u; shift < 32u; shift += 8u)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendF64LE(std::vector<std::uint8_t>& bytes, double value) {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    for (unsigned shift = 0u; shift < 64u; shift += 8u)
        bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
}

void appendRecord(std::vector<std::uint8_t>& bytes, std::uint32_t localId,
                  const std::array<double, 3>& vector) {
    appendU32LE(bytes, localId);
    for (const double component : vector) appendF64LE(bytes, component);
}

void appendSourceMeshBlock(std::vector<std::uint8_t>& bytes,
                           std::uint32_t materialId,
                           std::uint32_t firstNodeId,
                           std::uint32_t elementId) {
    bytes.insert(bytes.end(), {'N', 'O', 'K', 'M'});
    appendU32LE(bytes, 1u);
    appendU32LE(bytes, materialId);
    appendU32LE(bytes, 4u);
    appendU32LE(bytes, 1u);
    const std::array<std::array<double, 3>, 4> nodes{{
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
        {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01},
    }};
    for (std::size_t index = 0u; index < nodes.size(); ++index)
        appendRecord(bytes, firstNodeId + static_cast<std::uint32_t>(index), nodes[index]);
    appendU32LE(bytes, elementId);
    for (std::uint32_t index = 0u; index < 4u; ++index)
        appendU32LE(bytes, firstNodeId + index);
}

double norm(const std::array<double, 3>& vector) {
    return std::hypot(vector[0], vector[1], vector[2]);
}

std::vector<std::uint8_t> readBinaryBytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), std::string("cannot open source artifact ") + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

numi::matter::ObjectSource makeSourceFieldFixture(
    const std::string& name,
    const std::vector<ElementFiberRecord>& records,
    const std::vector<std::uint8_t>& bytes,
    std::size_t byteOffset,
    const std::array<std::uint64_t, 4>& identity,
    std::array<double, 3> origin
) {
    numi::matter::ObjectSource object;
    object.name = name;
    object.representation = numi::matter::Representation::fem;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.mixedFEM = false;
    object.characteristicLength = 0.01;
    object.femNodes.resize(records.size() * 4u);
    object.tetrahedra.resize(records.size());
    constexpr std::size_t side = 40u;
    constexpr double pitch = 0.02;
    constexpr std::array<std::array<double, 3>, 4> localNodes{{
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
        {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01},
    }};
    for (std::size_t index = 0; index < records.size(); ++index) {
        const std::size_t x = index % side;
        const std::size_t y = (index / side) % side;
        const std::size_t z = index / (side * side);
        const std::uint32_t first = static_cast<std::uint32_t>(index * 4u);
        for (std::size_t local = 0; local < localNodes.size(); ++local) {
            object.femNodes[std::size_t(first) + local] = {
                origin[0] + pitch * static_cast<double>(x) + localNodes[local][0],
                origin[1] + pitch * static_cast<double>(y) + localNodes[local][1],
                origin[2] + pitch * static_cast<double>(z) + localNodes[local][2],
            };
        }
        object.tetrahedra[index] = {{first, first + 1u, first + 2u, first + 3u}};
    }
    std::string error;
    require(attachElementFiberFrames(object, bytes, byteOffset,
                                     static_cast<std::uint32_t>(records.size()),
                                     identity, error), error);
    return object;
}

void checkFrame(const std::array<double, 4>& frame,
                const std::array<double, 3>& source) {
    double norm2 = 0.0;
    for (const double component : frame) norm2 += component * component;
    require(std::abs(norm2 - 1.0) < 1.0e-14,
            "source material frame is not a unit quaternion");
    const auto direction = rotateMaterialX(frame);
    const double length = norm(source);
    for (std::size_t axis = 0u; axis < 3u; ++axis)
        require(std::abs(direction[axis] - source[axis] / length) < 1.0e-13,
                "material +X does not map to the source fiber direction");
}

void checkCookedSourceMaterialFrames(
    const std::vector<std::uint8_t>& sidecar,
    const std::array<double, 3>& fiber0,
    const std::array<double, 3>& fiber1
) {
    const auto material = numi::matter::parseMatterFile(
        NUMI_OPEN_KNEE_SOURCE_MATERIAL
    );
    require(material.succeeded(), "Open Knee(s) source material did not parse");
    numi::matter::WorldSource source;
    source.gravity = {0.0, 0.0, 0.0};
    source.materials.push_back(material.material);
    numi::matter::ObjectSource object;
    object.name = "source_transiso_fiber_frame_fixture";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.deformableContact = false;
    object.mixedFEM = false;
    object.femNodes = {
        {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
        {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01},
        {0.0, 0.0, -0.01},
    };
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}, {{0u, 2u, 1u, 4u}}};
    constexpr std::array<std::uint64_t, 4> identity{1u, 2u, 3u, 4u};
    std::string error;
    require(attachElementFiberFrames(object, sidecar, 0u, 2u, identity, error), error);
    source.objects.push_back(std::move(object));

    const numi::matter::CompileResult compiled = numi::matter::compileWorld(
        source, {.maximumRateExponent = 0, .emitSpecializedMetal = false}
    );
    require(compiled.succeeded(), "source-frame Matter cook failed");
    require(compiled.world.fem.tetrahedra.size() == 2u,
            "source-frame Matter cook changed the tetrahedron count");
    checkFrame(source.objects[0].femMaterialFrameRotations[0], fiber0);
    checkFrame(source.objects[0].femMaterialFrameRotations[1], fiber1);
    for (std::size_t index = 0u; index < 2u; ++index) {
        const auto& expected = source.objects[0].femMaterialFrameRotations[index];
        const auto& actual = compiled.world.fem.tetrahedra[index].materialFrameRotation;
        require(std::abs(actual.x - static_cast<float>(expected[0])) < 1.0e-7f &&
                    std::abs(actual.y - static_cast<float>(expected[1])) < 1.0e-7f &&
                    std::abs(actual.z - static_cast<float>(expected[2])) < 1.0e-7f &&
                    std::abs(actual.w - static_cast<float>(expected[3])) < 1.0e-7f,
                "Matter compiler did not cook source fiber frame in element order");
    }
}

} // namespace

int checkSourceSidecar(const char* path) {
    try {
        std::ifstream input(path, std::ios::binary);
        require(input.good(), "cannot open Human Open Knee(s) fiber sidecar");
        const auto firstByte = std::istreambuf_iterator<char>(input);
        const auto lastByte = std::istreambuf_iterator<char>();
        const std::vector<std::uint8_t> bytes(firstByte, lastByte);
        constexpr std::array<std::uint32_t, 2> counts{51009u, 44953u};
        constexpr std::array<std::array<std::uint64_t, 4>, 2> identities{{
            {0x9c2596de039ddb60ull, 0x798d29695f4b3271ull,
             0x984262867d3de49aull, 0x6bbd46ce64f71fa4ull},
            {0x1be0688d22381786ull, 0x1a22b839f1845d40ull,
             0x6e99af95feb15ae3ull, 0xbd400951d2643b5cull},
        }};
        std::array<std::vector<ElementFiberRecord>, 2> decodedGroups;
        std::size_t offset = 0u;
        for (std::size_t group = 0u; group < counts.size(); ++group) {
            std::vector<ElementFiberRecord> records;
            std::string error;
            require(decodeElementFiberRecords(bytes, offset, counts[group],
                                              records, error), error);
            numi::matter::ObjectSource object;
            object.tetrahedra.resize(counts[group]);
            require(attachElementFiberFrames(object, bytes, offset, counts[group],
                                             identities[group], error), error);
            require(object.femMaterialFrameRotations.size() == counts[group],
                    "source fiber frame count changed during Matter ingestion");
            const std::array<std::size_t, 3> samples{
                0u, records.size() / 2u, records.size() - 1u
            };
            for (const std::size_t index : samples)
                checkFrame(object.femMaterialFrameRotations[index],
                           records[index].sourceVector);
            decodedGroups[group] = std::move(records);
            offset += std::size_t(counts[group]) * 28u;
        }
        require(offset == bytes.size(), "source fiber sidecar has unassigned trailing bytes");
        const auto material = numi::matter::parseMatterFile(
            NUMI_OPEN_KNEE_SOURCE_MENISCUS_MATERIAL
        );
        require(material.succeeded(), "Open Knee(s) source meniscus material did not parse");
        require(materialParameter(material.material, "c1") == 4.61e6,
                "Matter meniscus c1 drifted from source SI conversion");
        require(materialParameter(material.material, "c3") == 0.1197e6,
                "Matter meniscus c3 drifted from source SI conversion");
        require(materialParameter(material.material, "c4") == 150.0,
                "Matter meniscus c4 drifted from source: " +
                    std::to_string(materialParameter(material.material, "c4")));
        require(materialParameter(material.material, "c5") == 400.0e6,
                "Matter meniscus c5 drifted from source SI conversion");
        require(materialParameter(material.material, "lambda_max") == 1.019,
                "Matter meniscus lambda_max drifted from source: " +
                    std::to_string(materialParameter(material.material, "lambda_max")));
        require(materialParameter(material.material, "bulk") == 92.16e6,
                "Matter meniscus bulk drifted from source SI conversion");
        require(materialParameter(material.material, "initial_stretch") == 1.0,
                "Matter meniscus initial stretch drifted from source");
        require(materialParameter(material.material, "numerical_viscosity") == 0.0,
                "Matter meniscus viscosity drifted from source");
        numi::matter::WorldSource source;
        source.gravity = {0.0, 0.0, 0.0};
        source.materials.push_back(material.material);
        std::size_t groupOffset = 0u;
        for (std::size_t group = 0u; group < counts.size(); ++group) {
            auto object = makeSourceFieldFixture(
                group == 0u ? "MNS-M_source_field_compile_fixture" :
                              "MNS-L_source_field_compile_fixture",
                decodedGroups[group], bytes, groupOffset, identities[group],
                {0.0, 0.0, group == 0u ? 0.0 : 2.0}
            );
            object.materialIndex = 0u;
            source.objects.push_back(std::move(object));
            groupOffset += std::size_t(counts[group]) * 28u;
        }
        const numi::matter::CompileResult compiled = numi::matter::compileWorld(
            source, {.maximumRateExponent = 0, .emitSpecializedMetal = false}
        );
        require(compiled.succeeded(), "full source fiber-field Matter cook failed");
        require(compiled.world.fem.tetrahedra.size() == offset / 28u,
                "full source fiber-field Matter cook changed the element count");
        std::size_t compiledOffset = 0u;
        for (std::size_t group = 0u; group < source.objects.size(); ++group) {
            const auto& object = source.objects[group];
            for (std::size_t index = 0u; index < object.tetrahedra.size(); ++index) {
                const auto& expected = object.femMaterialFrameRotations[index];
                const auto& actual = compiled.world.fem.tetrahedra[compiledOffset + index]
                                         .materialFrameRotation;
                require(std::abs(actual.x - static_cast<float>(expected[0])) < 1.0e-7f &&
                            std::abs(actual.y - static_cast<float>(expected[1])) < 1.0e-7f &&
                            std::abs(actual.z - static_cast<float>(expected[2])) < 1.0e-7f &&
                            std::abs(actual.w - static_cast<float>(expected[3])) < 1.0e-7f,
                        "full source fiber-field Matter cook changed element-frame order");
            }
            compiledOffset += object.tetrahedra.size();
        }
        std::cout << "open_knee_source_fiber_sidecar=passed MNS-M=51009 MNS-L=44953 "
                     "source_material_ids=13,12 total=95962 cooked_frames=95962 "
                     "source_geometry=not_loaded "
                     "material_frame_basis=local_x_to_source_direction\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "open_knee_source_fiber_sidecar=failed "
                  << exception.what() << '\n';
        return 1;
    }
}

int checkSourceArtifacts(const char* fiberPath, const char* meshPath) {
    try {
        const std::vector<std::uint8_t> fiberBytes = readBinaryBytes(fiberPath);
        const std::vector<std::uint8_t> meshBytes = readBinaryBytes(meshPath);
        std::vector<SourceMeniscusMesh> meshes;
        std::string error;
        require(decodeSourceMeniscusMeshes(meshBytes, meshes, error), error);

        constexpr std::uint32_t medialCount = 51009u;
        constexpr std::uint32_t lateralCount = 44953u;
        constexpr std::array<std::array<std::uint64_t, 4>, 2> identities{{
            {0x9c2596de039ddb60ull, 0x798d29695f4b3271ull,
             0x984262867d3de49aull, 0x6bbd46ce64f71fa4ull},
            {0x1be0688d22381786ull, 0x1a22b839f1845d40ull,
             0x6e99af95feb15ae3ull, 0xbd400951d2643b5cull},
        }};
        const auto material = numi::matter::parseMatterFile(
            NUMI_OPEN_KNEE_SOURCE_MENISCUS_MATERIAL
        );
        require(material.succeeded(), "Open Knee(s) source meniscus material did not parse");
        require(materialParameter(material.material, "c1") == 4.61e6 &&
                    materialParameter(material.material, "c3") == 0.1197e6 &&
                    materialParameter(material.material, "c4") == 150.0 &&
                    materialParameter(material.material, "c5") == 400.0e6 &&
                    materialParameter(material.material, "lambda_max") == 1.019 &&
                    materialParameter(material.material, "bulk") == 92.16e6 &&
                    materialParameter(material.material, "initial_stretch") == 1.0,
                "Matter meniscus material defaults drifted from source IDs 12 and 13");

        numi::matter::WorldSource source;
        source.gravity = {0.0, 0.0, 0.0};
        source.materials.push_back(material.material);
        for (const SourceMeniscusMesh& mesh : meshes) {
            const bool medial = mesh.sourceMaterialId == 13u;
            const std::uint32_t expectedCount = medial ? medialCount : lateralCount;
            const std::size_t fiberOffset = medial ? 0u :
                std::size_t(medialCount) * 28u;
            const std::array<std::uint64_t, 4>& sourceIdentity = identities[medial ? 0u : 1u];
            require(mesh.tetrahedra.size() == expectedCount,
                    "source meniscus tetrahedron count drifted from MeshData");

            numi::matter::ObjectSource object;
            object.name = medial ? "MNS-M_source_mesh" : "MNS-L_source_mesh";
            object.materialIndex = 0u;
            object.representation = numi::matter::Representation::fem;
            object.deformableContact = false;
            object.deformableSelfContact = false;
            object.mixedFEM = false;
            object.characteristicLength = 0.001;
            object.femNodes.reserve(mesh.nodes.size());
            std::unordered_map<std::uint32_t, std::uint32_t> localNode;
            localNode.reserve(mesh.nodes.size());
            for (const SourceNodeRecord& node : mesh.nodes) {
                const std::uint32_t local = static_cast<std::uint32_t>(
                    object.femNodes.size()
                );
                localNode.emplace(node.sourceId, local);
                // Source anatomical coordinates are preserved in the Human
                // archive. Matter consumes SI lengths; this explicit mm->m
                // conversion is an assumption because the FEBio deck has no
                // unit declaration, so the physical scale remains unqualified.
                object.femNodes.push_back({
                    node.coordinates[0] * 0.001,
                    node.coordinates[1] * 0.001,
                    node.coordinates[2] * 0.001,
                });
            }
            object.tetrahedra.reserve(mesh.tetrahedra.size());
            for (const SourceTetrahedronRecord& sourceTet : mesh.tetrahedra) {
                std::array<std::uint32_t, 4> localTet{};
                for (std::size_t node = 0u; node < localTet.size(); ++node) {
                    const auto found = localNode.find(sourceTet.sourceNodeIds[node]);
                    require(found != localNode.end(),
                            "source meniscus tetrahedron references an unmapped node");
                    localTet[node] = found->second;
                }
                object.tetrahedra.push_back({localTet});
            }
            require(attachElementFiberFrames(object, fiberBytes, fiberOffset,
                                             expectedCount, sourceIdentity, error), error);
            source.objects.push_back(std::move(object));
        }

        const numi::matter::CompileResult compiled = numi::matter::compileWorld(
            source, {.maximumRateExponent = 0, .emitSpecializedMetal = false}
        );
        require(compiled.succeeded(), "source meniscus Matter cook failed");
        require(compiled.world.fem.tetrahedra.size() ==
                    std::size_t(medialCount) + lateralCount,
                "source meniscus Matter cook changed the source tetrahedron total");
        double minimumRestVolume = std::numeric_limits<double>::infinity();
        double maximumRestVolume = 0.0;
        for (const auto& tet : compiled.world.fem.tetrahedra) {
            const double volume = tet.inverseRestRow0.w;
            require(std::isfinite(volume) && volume > 0.0,
                    "source meniscus cook contains an invalid rest tetrahedron");
            minimumRestVolume = std::min(minimumRestVolume, volume);
            maximumRestVolume = std::max(maximumRestVolume, volume);
        }
        std::size_t tetrahedronOffset = 0u;
        for (const auto& object : source.objects) {
            for (std::size_t index = 0u; index < object.tetrahedra.size(); ++index) {
                const auto& expected = object.femMaterialFrameRotations[index];
                const auto& actual = compiled.world.fem.tetrahedra[
                    tetrahedronOffset + index
                ].materialFrameRotation;
                require(std::abs(actual.x - static_cast<float>(expected[0])) < 1.0e-7f &&
                            std::abs(actual.y - static_cast<float>(expected[1])) < 1.0e-7f &&
                            std::abs(actual.z - static_cast<float>(expected[2])) < 1.0e-7f &&
                            std::abs(actual.w - static_cast<float>(expected[3])) < 1.0e-7f,
                        "Matter cook changed source meniscus material-frame ordering");
            }
            tetrahedronOffset += object.tetrahedra.size();
        }
        std::cout << "open_knee_source_meniscus_mesh=passed source_materials=12,13 "
                     "source_nodes=" << source.objects[0].femNodes.size() +
                     source.objects[1].femNodes.size() << " source_tetrahedra=95962 "
                     "cooked_frames=95962 source_contact=not_assembled "
                     "source_prestrain=not_applied source_unit_scale=mm_to_m_assumed "
                     "rest_volume_range_m3=" << minimumRestVolume << ',' <<
                     maximumRestVolume << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "open_knee_source_meniscus_mesh=failed "
                  << exception.what() << '\n';
        return 1;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3]);
        if (argc == 3 && std::string(argv[1]) == "--source-sidecar")
            return checkSourceSidecar(argv[2]);
        require(argc == 1, "usage: open knee fiber field check [--source-sidecar PATH] "
                           "[--source-artifacts FIBER_PATH MENISCUS_MESH_PATH]");
        const std::array<double, 3> medial0{-0.27206761041769667,
                                             1.1419075817929256,
                                             0.08613731458137508};
        const std::array<double, 3> medial1{0.0, -2.0, 0.0};
        const std::array<double, 3> lateral0{0.0, 0.0, -4.0};
        std::vector<std::uint8_t> sidecar;
        appendRecord(sidecar, 1u, medial0);
        appendRecord(sidecar, 2u, medial1);
        const std::size_t lateralOffset = sidecar.size();
        appendRecord(sidecar, 1u, lateral0);

        std::vector<ElementFiberRecord> decoded;
        std::string error;
        require(decodeElementFiberRecords(sidecar, 0u, 2u, decoded, error), error);
        require(decoded.size() == 2u && decoded[0].localElementId == 1u &&
                    decoded[0].sourceVector == medial0 &&
                    decoded[1].localElementId == 2u &&
                    decoded[1].sourceVector == medial1,
                "source vector records were changed during decode");
        checkCookedSourceMaterialFrames(sidecar, medial0, medial1);

        std::vector<std::uint8_t> sourceMeshes;
        appendSourceMeshBlock(sourceMeshes, 12u, 101u, 5001u);
        appendSourceMeshBlock(sourceMeshes, 13u, 201u, 9001u);
        std::vector<SourceMeniscusMesh> decodedMeshes;
        std::string meshError;
        require(decodeSourceMeniscusMeshes(sourceMeshes, decodedMeshes, meshError),
                meshError);
        require(decodedMeshes.size() == 2u &&
                    decodedMeshes[0].sourceMaterialId == 12u &&
                    decodedMeshes[0].nodes[0].sourceId == 101u &&
                    decodedMeshes[0].tetrahedra[0].sourceId == 5001u &&
                    decodedMeshes[1].sourceMaterialId == 13u &&
                    decodedMeshes[1].tetrahedra[0].sourceNodeIds[3] == 204u,
                "source meniscus mesh records changed during decode");
        std::vector<SourceMeniscusMesh> preservedMeshes = decodedMeshes;
        require(!decodeSourceMeniscusMeshes(
                    std::span<const std::uint8_t>(sourceMeshes.data(),
                                                  sourceMeshes.size() - 1u),
                    preservedMeshes, meshError) &&
                    preservedMeshes.size() == 2u &&
                    preservedMeshes[0].sourceMaterialId == 12u,
                "truncated source mesh must reject without mutating output");

        constexpr std::array<std::uint64_t, 4> identity{
            0x6e756d692d6f7065ull, 0x6e2d6b6e65652d66ull,
            0x696265722d763100ull, 0x0000000000000001ull,
        };
        numi::matter::ObjectSource medial;
        medial.tetrahedra.resize(2u);
        require(attachElementFiberFrames(medial, sidecar, 0u, 2u, identity, error), error);
        require(medial.femMaterialFrameRotations.size() == 2u &&
                    medial.femMaterialFrameSourceIdentity == identity,
                "source fiber frames were not assigned to the Matter FEM owner");
        checkFrame(medial.femMaterialFrameRotations[0], medial0);
        checkFrame(medial.femMaterialFrameRotations[1], medial1);
        checkFrame(frameFromSourceFiber({-1.0, 1.0e-10, 0.0}),
                   {-1.0, 1.0e-10, 0.0});
        checkFrame(frameFromSourceFiber({-1.0, 0.0, 0.0}),
                   {-1.0, 0.0, 0.0});

        numi::matter::ObjectSource lateral;
        lateral.tetrahedra.resize(1u);
        require(attachElementFiberFrames(lateral, sidecar, lateralOffset, 1u,
                                         identity, error), error);
        checkFrame(lateral.femMaterialFrameRotations[0], lateral0);

        std::vector<ElementFiberRecord> preserved{{99u, {9.0, 8.0, 7.0}}};
        std::vector<std::uint8_t> badIds = sidecar;
        badIds[0] = 2u;
        require(!decodeElementFiberRecords(badIds, 0u, 2u, preserved, error) &&
                    preserved.size() == 1u && preserved[0].localElementId == 99u,
                "malformed source rows must reject without partial output");

        numi::matter::ObjectSource truncated;
        truncated.tetrahedra.resize(2u);
        require(!attachElementFiberFrames(truncated,
                    std::span<const std::uint8_t>(sidecar.data(), 2u * 28u - 1u),
                    0u, 2u, identity, error) &&
                    truncated.femMaterialFrameRotations.empty(),
                "truncated source rows must not partially mutate the FEM owner");
        require(!attachElementFiberFrames(medial, sidecar, 0u, 2u, identity, error),
                "source frame owner must reject duplicate field assignment");

        std::cout << "open_knee_element_fiber_frames=passed records=3 "
                     "matter_fem_material_frames=2 source_vectors_preserved=1\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "open_knee_element_fiber_frames=failed "
                  << exception.what() << '\n';
        return 1;
    }
}
