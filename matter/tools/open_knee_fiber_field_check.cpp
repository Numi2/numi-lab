#include "numi/matter/open_knee_fiber_field.hpp"
#include "numi/matter/compiler.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
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
    bytes.insert(bytes.end(), {'N', 'O', 'K', 'T'});
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

std::array<std::uint64_t, 4> identityFromHex(std::string_view hex) {
    require(hex.size() == 64u, "source material identity must be a SHA-256 hex string");
    std::array<std::uint64_t, 4> result{};
    const auto digit = [](char c) -> std::uint64_t {
        if (c >= '0' && c <= '9') return static_cast<std::uint64_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint64_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<std::uint64_t>(c - 'A' + 10);
        throw std::runtime_error("invalid source SHA-256 identity character");
    };
    for (std::size_t index = 0u; index < hex.size(); ++index) {
        result[index / 16u] = (result[index / 16u] << 4u) | digit(hex[index]);
    }
    return result;
}

void setMaterialParameter(numi::matter::MaterialProgram& material,
                          const std::string& name, double value) {
    for (auto& parameter : material.parameters) {
        if (parameter.name == name) {
            require(std::isfinite(value) && parameter.lower <= value &&
                        value <= parameter.upper,
                    "source parameter lies outside Matter material bounds: " + name);
            parameter.defaultValue = value;
            return;
        }
    }
    throw std::runtime_error("source material lacks parameter " + name);
}

struct SourceTissueProgram {
    std::uint32_t materialId;
    const char* name;
    std::uint32_t nodeCount;
    std::uint32_t tetrahedronCount;
    bool ligament;
    bool meniscus;
    double c1;
    double c3;
    double c4;
    double c5;
    double lambdaMax;
    double bulk;
    double initialStretch;
    std::array<double, 3> fiber;
    const char* materialSha256;
};

const std::array<SourceTissueProgram, 12>& sourceTissuePrograms() {
    static const std::array<SourceTissueProgram, 12> programs{{
        {5u, "QAT", 14963u, 69410u, true, false, 2.75, .065, 115.89, 777.56, 1.042, 206.61, 1.0,
         {.0021382483725627954, -.22397946374811423, .9745915183875777}, "63bcda40f166f9db49fae7da4d28925151ca9f1c4771215b8e70725595d1c9e6"},
        {6u, "TBC-L", 40669u, 200079u, false, false, 2.54, 0., 0., 0., 1., 100., 1., {1., 0., 0.}, "ae4857ef797c6231bc113f484f5127743b964d6807477f761f8d08bf51d585eb"},
        {7u, "PCL", 3714u, 14379u, true, false, 3.25, .1196, 87.178, 431.063, 1.035, 243.9, 1.0,
         {-.13894548213429192, -.7522134084051663, -.6441033622097867}, "cf525f223facb390abf080db94e25a1252b0a5d5300def840919a7974c6b3d6d"},
        {8u, "PTC", 26121u, 121105u, false, false, 2.54, 0., 0., 0., 1., 100., 1., {1., 0., 0.}, "fefb3ee32356cb292e5d7dd7159b69326ec790e7ab94a171bd0078e401a8db8a"},
        {9u, "ACL", 15792u, 72552u, true, false, 1.95, .0139, 116.22, 535.039, 1.046, 146.41, 1.0,
         {-.1012763293512546, -.5108021004916831, .8537120821719818}, "e05ad25cf804201bb35be6c0232b55a7395ae1d4b6aff74738dd25925daefcc6"},
        {10u, "MCL", 15693u, 62712u, true, false, 1.44, .57, 48., 467.1, 1.063, 793.65, 1.0,
         {.2862545445153689, -.03912610167242837, .9573544191741205}, "dd4ad2e00a431147a41d2be29c2b6693e686659e7b61341e32568bb85b86a5ea"},
        {11u, "PTL", 9280u, 35616u, true, false, 2.75, .065, 115.89, 777.56, 1.042, 206.61, 1.0,
         {.2536841532034668, .41872016059340167, .8719620275711987}, "fc1cc8c97d219f15c3cc905d3febe536fdd6dd66406546f7c5c4ed38bc2a6549"},
        {12u, "MNS-L", 10901u, 44953u, false, true, 4.61, .1197, 150., 400., 1.019, 92.16, 1.0, {1., 0., 0.}, "1a74a5c8b48f1664ae6ddd0bfd4df320184369c0080b7248e61529fd63b7a6b6"},
        {13u, "MNS-M", 11706u, 51009u, false, true, 4.61, .1197, 150., 400., 1.019, 92.16, 1.0, {1., 0., 0.}, "bb4b9b15ea098c64b9a62e78084b834657934064888227642eb76d6189f6741e"},
        {14u, "LCL", 2960u, 9773u, true, false, 1.44, .57, 48., 467.1, 1.063, 793.65, 1.0,
         {.22214778048527417, .16495808684018612, .9609574356918686}, "acd5a29aedb080d1be5e0b2155da6297a1bcab1ba3a98b457928e296926ee2a4"},
        {15u, "TBC-M", 18060u, 75627u, false, false, 2.54, 0., 0., 0., 1., 100., 1., {1., 0., 0.}, "f930927d3c51a5312d762746e5bd73d4906c3d1dfd9bdbffee39ec9e4c15e69e"},
        {16u, "FMC", 24870u, 87072u, false, false, 2.54, 0., 0., 0., 1., 100., 1., {1., 0., 0.}, "6794baf3cf16114336939f42c089ddae47393fc3c83386fe4b4d405f284abb00"},
    }};
    return programs;
}

int checkSourceArtifacts(const char* fiberPath, const char* meshPath) {
    try {
        const std::vector<std::uint8_t> fiberBytes = readBinaryBytes(fiberPath);
        const std::vector<std::uint8_t> meshBytes = readBinaryBytes(meshPath);
        std::vector<SourceVolumeMesh> meshes;
        std::string error;
        require(decodeSourceVolumeMeshes(meshBytes, meshes, error), error);
        const auto& specs = sourceTissuePrograms();
        require(meshes.size() == specs.size(),
                "source volume input must contain all 12 source tissue programs");
        const auto ligamentTemplate = numi::matter::parseMatterFile(
            NUMI_OPEN_KNEE_SOURCE_MATERIAL
        );
        const auto meniscusTemplate = numi::matter::parseMatterFile(
            NUMI_OPEN_KNEE_SOURCE_MENISCUS_MATERIAL
        );
        const auto cartilageTemplate = numi::matter::parseMatterFile(
            NUMI_OPEN_KNEE_SOURCE_CARTILAGE_MATERIAL
        );
        require(ligamentTemplate.succeeded(), "source ligament material did not parse");
        require(meniscusTemplate.succeeded(), "source meniscus material did not parse");
        require(cartilageTemplate.succeeded(), "source cartilage material did not parse");

        constexpr std::array<std::array<std::uint64_t, 4>, 2> meshDataIdentities{{
            {0x9c2596de039ddb60ull, 0x798d29695f4b3271ull,
             0x984262867d3de49aull, 0x6bbd46ce64f71fa4ull},
            {0x1be0688d22381786ull, 0x1a22b839f1845d40ull,
             0x6e99af95feb15ae3ull, 0xbd400951d2643b5cull},
        }};
        constexpr std::uint32_t medialFiberCount = 51009u;
        constexpr std::uint32_t lateralFiberCount = 44953u;
        constexpr std::size_t lateralFiberOffset = std::size_t(medialFiberCount) * 28u;

        numi::matter::WorldSource source;
        source.gravity = {0.0, 0.0, 0.0};
        std::uint64_t expectedTetrahedra = 0u;
        for (std::size_t group = 0u; group < meshes.size(); ++group) {
            const SourceVolumeMesh& mesh = meshes[group];
            const SourceTissueProgram& spec = specs[group];
            require(mesh.sourceMaterialId == spec.materialId,
                    "source volume material order or identity drifted");
            require(mesh.nodes.size() == spec.nodeCount &&
                        mesh.tetrahedra.size() == spec.tetrahedronCount,
                    std::string("source volume counts drifted for ") + spec.name);

            numi::matter::MaterialProgram material = spec.ligament
                ? ligamentTemplate.material
                : (spec.meniscus ? meniscusTemplate.material : cartilageTemplate.material);
            material.name = std::string("open_knee_source_") + spec.name;
            // FEBio leaves coordinate units undeclared. These density and
            // length conversions are explicit tonne-mm-second assumptions.
            setMaterialParameter(material, "density", 1000.0);
            setMaterialParameter(material, "c1", spec.c1 * 1.0e6);
            setMaterialParameter(material, "bulk", spec.bulk * 1.0e6);
            if (spec.ligament || spec.meniscus) {
                setMaterialParameter(material, "initial_stretch", spec.initialStretch);
                setMaterialParameter(material, "c3", spec.c3 * 1.0e6);
                setMaterialParameter(material, "c4", spec.c4);
                setMaterialParameter(material, "c5", spec.c5 * 1.0e6);
                setMaterialParameter(material, "lambda_max", spec.lambdaMax);
                setMaterialParameter(material, "fiber_scale", 1.0);
                setMaterialParameter(material, "fiber_x", spec.fiber[0]);
                setMaterialParameter(material, "fiber_y", spec.fiber[1]);
                setMaterialParameter(material, "fiber_z", spec.fiber[2]);
            }
            if (spec.ligament || spec.meniscus)
                require(materialParameter(material, "numerical_viscosity") == 0.0,
                        "source elastic material unexpectedly enables numerical viscosity");
            source.materials.push_back(std::move(material));

            numi::matter::ObjectSource object;
            object.name = std::string("OpenKnee_") + spec.name;
            object.materialIndex = static_cast<std::uint32_t>(source.materials.size() - 1u);
            object.representation = numi::matter::Representation::fem;
            object.deformableContact = false;
            object.deformableSelfContact = false;
            object.mixedFEM = false;
            object.characteristicLength = 0.001;
            object.femNodes.reserve(mesh.nodes.size());
            std::unordered_map<std::uint32_t, std::uint32_t> localNode;
            localNode.reserve(mesh.nodes.size());
            for (const SourceNodeRecord& node : mesh.nodes) {
                const std::uint32_t local = static_cast<std::uint32_t>(object.femNodes.size());
                require(localNode.emplace(node.sourceId, local).second,
                        "source volume contains duplicate node identity");
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
                            "source volume tetrahedron references an unmapped node");
                    localTet[node] = found->second;
                }
                object.tetrahedra.push_back({localTet});
            }

            if (spec.meniscus) {
                const bool medial = spec.materialId == 13u;
                require(mesh.tetrahedra.size() ==
                            (medial ? medialFiberCount : lateralFiberCount),
                        "source meniscus fibre count differs from source volume");
                const std::size_t fiberOffset = medial ? 0u : lateralFiberOffset;
                const auto& identity = meshDataIdentities[medial ? 0u : 1u];
                require(attachElementFiberFrames(
                            object, fiberBytes, fiberOffset,
                            static_cast<std::uint32_t>(mesh.tetrahedra.size()),
                            identity, error), error);
            } else if (spec.ligament) {
                require(attachUniformElementFiberFrames(
                            object, spec.fiber, identityFromHex(spec.materialSha256), error),
                        error);
            }
            expectedTetrahedra += mesh.tetrahedra.size();
            source.objects.push_back(std::move(object));
        }
        require(expectedTetrahedra == 844287u,
                "source total tetrahedron count drifted from the pinned source");

        const numi::matter::CompileResult compiled = numi::matter::compileWorld(
            source, {.maximumRateExponent = 0, .emitSpecializedMetal = false}
        );
        require(compiled.succeeded(), "full Open Knee source-volume Matter cook failed");
        require(compiled.world.fem.tetrahedra.size() == expectedTetrahedra,
                "Matter cook changed the source tissue tetrahedron total");
        double minimumRestVolume = std::numeric_limits<double>::infinity();
        double maximumRestVolume = 0.0;
        for (const auto& tet : compiled.world.fem.tetrahedra) {
            const double volume = tet.inverseRestRow0.w;
            require(std::isfinite(volume) && volume > 0.0,
                    "source tissue cook contains an invalid rest tetrahedron");
            minimumRestVolume = std::min(minimumRestVolume, volume);
            maximumRestVolume = std::max(maximumRestVolume, volume);
        }
        std::size_t tetrahedronOffset = 0u;
        std::size_t cookedFrames = 0u;
        for (const auto& object : source.objects) {
            if (!object.femMaterialFrameRotations.empty()) {
                require(object.femMaterialFrameRotations.size() == object.tetrahedra.size(),
                        "source material-frame count differs from its tissue mesh");
                for (std::size_t index = 0u; index < object.tetrahedra.size(); ++index) {
                    const auto& expected = object.femMaterialFrameRotations[index];
                    const auto& actual = compiled.world.fem.tetrahedra[
                        tetrahedronOffset + index].materialFrameRotation;
                    require(std::abs(actual.x - static_cast<float>(expected[0])) < 1.0e-7f &&
                                std::abs(actual.y - static_cast<float>(expected[1])) < 1.0e-7f &&
                                std::abs(actual.z - static_cast<float>(expected[2])) < 1.0e-7f &&
                                std::abs(actual.w - static_cast<float>(expected[3])) < 1.0e-7f,
                            "Matter cook changed source material-frame ordering");
                }
                cookedFrames += object.femMaterialFrameRotations.size();
            }
            tetrahedronOffset += object.tetrahedra.size();
        }
        std::cout << "open_knee_source_volume_mesh=passed source_material_groups=12 "
                     "source_nodes=" << std::accumulate(source.objects.begin(), source.objects.end(),
                         std::uint64_t{0}, [](std::uint64_t total, const auto& object) {
                             return total + object.femNodes.size();
                         }) << " source_tetrahedra=" << expectedTetrahedra
                  << " cooked_frames=" << cookedFrames
                  << " source_contact=not_assembled source_initialization=not_solved "
                     "source_unit_scale=mm_to_m_assumed rest_volume_range_m3="
                  << minimumRestVolume << ',' << maximumRestVolume << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "open_knee_source_volume_mesh=failed " << exception.what() << '\n';
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
                           "[--source-artifacts FIBER_PATH SOURCE_VOLUME_MESH_PATH]");
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
        std::vector<SourceVolumeMesh> decodedMeshes;
        std::string meshError;
        require(decodeSourceVolumeMeshes(sourceMeshes, decodedMeshes, meshError),
                meshError);
        require(decodedMeshes.size() == 2u &&
                    decodedMeshes[0].sourceMaterialId == 12u &&
                    decodedMeshes[0].nodes[0].sourceId == 101u &&
                    decodedMeshes[0].tetrahedra[0].sourceId == 5001u &&
                    decodedMeshes[1].sourceMaterialId == 13u &&
                    decodedMeshes[1].tetrahedra[0].sourceNodeIds[3] == 204u,
                "source volume mesh records changed during decode");
        std::vector<SourceVolumeMesh> preservedMeshes = decodedMeshes;
        require(!decodeSourceVolumeMeshes(
                    std::span<const std::uint8_t>(sourceMeshes.data(),
                                                  sourceMeshes.size() - 1u),
                    preservedMeshes, meshError) &&
                    preservedMeshes.size() == 2u &&
                    preservedMeshes[0].sourceMaterialId == 12u,
                "truncated source volume must reject without mutating output");

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
