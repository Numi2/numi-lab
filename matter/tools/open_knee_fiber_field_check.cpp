#include "numi/matter/open_knee_fiber_field.hpp"
#include "numi/matter/open_knee_source_graph.hpp"
#include "numi/matter/open_knee_source_contact.hpp"
#include "numi/matter/open_knee_source_discrete.hpp"
#include "numi/matter/open_knee_source_projection.hpp"
#include "numi/matter/open_knee_source_rigid_ties.hpp"
#include "numi/matter/compiler.hpp"
#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#endif

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
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

std::uint64_t sourceFaceKey(std::array<std::uint32_t, 3> nodes) {
    std::ranges::sort(nodes);
    require(nodes[2] < (1u << 21u),
            "source contact node exceeds reversible face-key range");
    return (std::uint64_t(nodes[0]) << 42u) |
           (std::uint64_t(nodes[1]) << 21u) | nodes[2];
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

int checkSourceArtifacts(const char* fiberPath, const char* meshPath,
                         const char* rigidTiesPath = nullptr,
                         const char* rigidGraphPath = nullptr,
                         const char* sourceContactPath = nullptr,
                         const char* sourceDiscretePath = nullptr,
                         double solveTime = -1.0,
                         std::uint32_t newtonBudget = 7u,
                         std::uint32_t fgmresBudget = 10u,
                         const char* checkpointSeedPath = nullptr,
                         const char* elementForceOutputPath = nullptr) {
    try {
        require(elementForceOutputPath == nullptr || checkpointSeedPath != nullptr,
                "element force readback needs a source checkpoint seed");
        require(rigidGraphPath == nullptr || rigidTiesPath != nullptr,
                "source rigid graph requires the complete rigid-tie program");
        require(sourceContactPath == nullptr || rigidGraphPath != nullptr,
                "source contact requires the full tissue and rigid graph");
        require(sourceDiscretePath == nullptr ||
                    (sourceContactPath != nullptr && rigidGraphPath != nullptr),
                "source discrete edges require the full source assembly");
        require(solveTime < 0.0 || (sourceDiscretePath != nullptr &&
                    std::isfinite(solveTime) && solveTime >= 0.0 &&
                    solveTime <= 2.0),
                "source solve requires the complete program and a valid continuation time");
        require(newtonBudget > 0u && newtonBudget <= 64u &&
                fgmresBudget > 0u && fgmresBudget <= 512u,
                "source numerical iteration budgets are invalid");
        std::vector<std::array<float, 3u>> checkpointPositions;
        struct CheckpointBodyPose {
            std::uint32_t materialId;
            std::array<float, 3u> center;
            std::array<float, 4u> orientation;
        };
        std::vector<CheckpointBodyPose> checkpointBodies;
        std::array<std::uint8_t, 32u> checkpointDeck{};
        std::array<std::uint8_t, 32u> checkpointGeometry{};
        std::array<std::uint8_t, 32u> checkpointArchive{};
        if (checkpointSeedPath != nullptr) {
            require(solveTime >= 0.0, "checkpoint seed requires source solve");
            const auto bytes = readBinaryBytes(checkpointSeedPath);
            constexpr std::size_t headerBytes = 128u;
            require(bytes.size() == headerBytes + 194729u * 12u + 9u * 32u &&
                    std::memcmp(bytes.data(), "NOKSEED2", 8u) == 0,
                    "source checkpoint seed has invalid size or magic");
            std::uint32_t version = 0u, count = 0u, bodies = 0u, stateIndex = 0u;
            double checkpointTime = 0.0;
            std::memcpy(&version, bytes.data() + 8u, 4u);
            std::memcpy(&count, bytes.data() + 12u, 4u);
            std::memcpy(&bodies, bytes.data() + 16u, 4u);
            std::memcpy(&stateIndex, bytes.data() + 20u, 4u);
            std::memcpy(&checkpointTime, bytes.data() + 24u, 8u);
            std::memcpy(checkpointDeck.data(), bytes.data() + 32u, 32u);
            std::memcpy(checkpointGeometry.data(), bytes.data() + 64u, 32u);
            std::memcpy(checkpointArchive.data(), bytes.data() + 96u, 32u);
            require(version == 2u && count == 194729u && bodies == 9u &&
                    stateIndex > 0u && std::isfinite(checkpointTime) &&
                    std::abs(checkpointTime - solveTime) < 1.0e-6,
                    "source checkpoint seed has invalid header");
            checkpointPositions.resize(count);
            std::memcpy(checkpointPositions.data(), bytes.data() + headerBytes,
                        count * 12u);
            std::size_t cursor = headerBytes + count * 12u;
            for (std::size_t body = 0u; body < 9u; ++body) {
                CheckpointBodyPose row{};
                std::memcpy(&row.materialId, bytes.data() + cursor, 4u);
                std::memcpy(row.center.data(), bytes.data() + cursor + 4u, 12u);
                std::memcpy(row.orientation.data(), bytes.data() + cursor + 16u, 16u);
                checkpointBodies.push_back(row);
                cursor += 32u;
            }
            require(std::ranges::all_of(checkpointPositions,
                        [](const auto& position) {
                            return std::ranges::all_of(position,
                                [](float value) { return std::isfinite(value); });
                        }), "source checkpoint has nonfinite tissue positions");
        }
        const std::vector<std::uint8_t> fiberBytes = readBinaryBytes(fiberPath);
        const std::vector<std::uint8_t> meshBytes = readBinaryBytes(meshPath);
        std::vector<SourceVolumeMesh> meshes;
        std::string error;
        require(decodeSourceVolumeMeshes(meshBytes, meshes, error), error);
        const auto digestHex = [](const std::array<std::uint8_t, 32>& digest) {
            constexpr char digits[] = "0123456789abcdef";
            std::string result;
            result.reserve(64u);
            for (const auto byte : digest) {
                result.push_back(digits[byte >> 4u]);
                result.push_back(digits[byte & 15u]);
            }
            return result;
        };
        SourceRigidTieProgram sourceTies;
        std::vector<std::uint8_t> tieBytes;
        if (rigidTiesPath != nullptr) {
            tieBytes = readBinaryBytes(rigidTiesPath);
            require(decodeSourceRigidTieProgram(tieBytes, sourceTies, error), error);
            require(digestHex(sourceTies.deckSHA256) ==
                        "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a" &&
                    digestHex(sourceTies.geometrySHA256) ==
                        "4155db1d0d7b87ffb2c668102d2495870e4461a539b18e6708f1f4817b5601bf",
                    "source rigid-tie program is bound to different source files");
            if (checkpointSeedPath != nullptr)
                require(checkpointDeck == sourceTies.deckSHA256 &&
                            checkpointGeometry == sourceTies.geometrySHA256 &&
                            digestHex(checkpointArchive) ==
                                "c370ae9f94e9faee2d7060bf2a6819e03be1312e82ca79bd2e9cebf8b34398de",
                        "source checkpoint does not match pinned deck, geometry, or XPLT");
        }
        SourceRigidGraphProgram rigidGraph;
        std::vector<std::uint8_t> graphBytes;
        if (rigidGraphPath != nullptr) {
            graphBytes = readBinaryBytes(rigidGraphPath);
            require(decodeSourceRigidGraphProgram(graphBytes, rigidGraph, error), error);
            require(rigidGraph.deckSHA256 == sourceTies.deckSHA256 &&
                    rigidGraph.geometrySHA256 == sourceTies.geometrySHA256,
                    "source rigid graph and tissue ties bind different archives");
            if (checkpointSeedPath != nullptr) {
                require(checkpointBodies.size() == rigidGraph.bodies.size(),
                        "source checkpoint rigid body count differs from graph");
                for (const auto& body : rigidGraph.bodies) {
                    const auto count = std::ranges::count_if(checkpointBodies,
                        [&](const auto& row) {
                            return row.materialId == body.materialId;
                        });
                    require(count == 1u,
                            "source checkpoint is missing or duplicates a rigid body");
                }
                for (const auto& body : checkpointBodies) {
                    const double q2 = std::inner_product(body.orientation.begin(),
                        body.orientation.end(), body.orientation.begin(), 0.0);
                    require(std::ranges::all_of(body.center,
                                [](float x) { return std::isfinite(x); }) &&
                            std::isfinite(q2) && std::abs(q2 - 1.0) < 1.0e-4,
                            "source checkpoint rigid pose is invalid");
                }
            }
        }
        SourceSlidingContactProgram sourceContact;
        if (sourceContactPath != nullptr) {
            const auto bytes = readBinaryBytes(sourceContactPath);
            require(decodeSourceSlidingContactProgram(bytes, sourceContact, error),
                    error);
            require(sourceContact.deckSHA256 == rigidGraph.deckSHA256 &&
                    sourceContact.geometrySHA256 == rigidGraph.geometrySHA256 &&
                    digestHex(sourceContact.volumeMeshSHA256) ==
                        "39b86f2f55853c74968f36a8d6c67eaed94639a3d42b558bb17787e2af8ffdba" &&
                    digestHex(sourceContact.geometryBinarySHA256) ==
                        "97c5e7b1c09eb47193bca0b7d2a515088b40938da45fe7aa8c8fc7bb7dd54368",
                    "source contact and rigid graph bind different archives");
        }
        SourceDiscreteProgram sourceDiscrete;
        std::vector<std::uint8_t> discreteBytes;
        if (sourceDiscretePath != nullptr) {
            discreteBytes = readBinaryBytes(sourceDiscretePath);
            require(decodeSourceDiscreteProgram(discreteBytes, sourceDiscrete, error),
                    error);
            require(sourceDiscrete.deckSHA256 == rigidGraph.deckSHA256 &&
                    sourceDiscrete.geometrySHA256 == rigidGraph.geometrySHA256 &&
                    sourceDiscrete.geometryBinarySHA256 ==
                        sourceContact.geometryBinarySHA256,
                    "source discrete program binds different source files");
            for (const auto& graphCurve : rigidGraph.curves) {
                const auto found = std::ranges::find_if(sourceDiscrete.curves,
                    [&](const auto& curve) { return curve.id == graphCurve.id; });
                require(found != sourceDiscrete.curves.end() &&
                            found->points == graphCurve.points,
                        "source discrete and rigid graph load curves differ");
            }
        }
        std::unordered_map<std::uint32_t, SourceRigidTieRecord> tieBySourceNode;
        tieBySourceNode.reserve(sourceTies.rows.size());
        for (const auto& tie : sourceTies.rows)
            require(tieBySourceNode.emplace(tie.sourceNodeId, tie).second,
                    "source rigid-tie node was bound twice");
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
        if (solveTime >= 0.0) {
            source.mixedSolver.newtonIterations = newtonBudget;
            source.mixedSolver.fgmresRestart = std::min(10u, fgmresBudget);
            source.mixedSolver.fgmresIterations = fgmresBudget;
            source.mixedSolver.relativeResidual = 1.0e-5;
        }
        struct SourceNodeLocation {
            std::uint32_t cookedNode = 0u, object = 0u;
            std::array<double, 3> sourcePosition{};
        };
        std::unordered_map<std::uint32_t, SourceNodeLocation> sourceNodeLocations;
        if (rigidGraphPath != nullptr) sourceNodeLocations.reserve(194729u);
        std::uint32_t globalNodeBase = 0u;
        for (std::size_t index = 0u; index < rigidGraph.bodies.size(); ++index) {
            const auto& body = rigidGraph.bodies[index];
            numi::matter::RigidProxySource proxy;
            proxy.shape = NM_RIGID_SPHERE;
            proxy.frameOnly = true;
            proxy.bodyIndex = static_cast<std::uint32_t>(index);
            proxy.dynamic = body.materialId != 2u && body.materialId != 3u;
            proxy.quasiStatic = proxy.dynamic;
            if (proxy.dynamic) proxy.sceneBodyIndex = proxy.bodyIndex;
            source.rigidProxies.push_back(proxy);
        }
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
            // FEBio leaves coordinate units undeclared. Length and stiffness
            // use the explicit tonne-mm-second to SI hypothesis. The source
            // step is static with zero rigid masses; positive FEM density is
            // numerical bookkeeping and is omitted from static residuals.
            setMaterialParameter(material, "density", 1000.0);
            setMaterialParameter(material, "c1", spec.c1 * 1.0e6);
            setMaterialParameter(material, "bulk", spec.bulk * 1.0e6);
            if (spec.ligament || spec.meniscus) {
                // The retained FEBio preload curves start every ligament at
                // unit stretch. ACL/MCL/LCL reach spec.initialStretch only
                // after the first continuation interval; baking the target
                // into the t=0 material silently changes the source problem.
                setMaterialParameter(material, "initial_stretch",
                    rigidGraphPath != nullptr && spec.ligament
                        ? 1.0 : spec.initialStretch);
                setMaterialParameter(material, "c3", spec.c3 * 1.0e6);
                setMaterialParameter(material, "c4", spec.c4);
                setMaterialParameter(material, "c5", spec.c5 * 1.0e6);
                setMaterialParameter(material, "lambda_max", spec.lambdaMax);
                setMaterialParameter(material, "fiber_scale", 1.0);
                // Each ligament and meniscus element has a material frame
                // whose local +X is the authored source fibre. The generic
                // energy receives F in that frame; supplying the source-world
                // fibre here would rotate its direction a second time.
                setMaterialParameter(material, "fiber_x", 1.0);
                setMaterialParameter(material, "fiber_y", 0.0);
                setMaterialParameter(material, "fiber_z", 0.0);
            }
            if (spec.ligament || spec.meniscus)
                require(materialParameter(material, "numerical_viscosity") == 0.0,
                        "source elastic material unexpectedly enables numerical viscosity");
            // The archived FEBio deck fixes these coefficients. The generic
            // template advertises them for later calibration, but this source
            // experiment has no identification distributions.
            if (rigidGraphPath != nullptr)
                for (auto& parameter : material.parameters)
                    parameter.identifiable = false;
            source.materials.push_back(std::move(material));

            numi::matter::ObjectSource object;
            object.name = std::string("OpenKnee_") + spec.name;
            object.materialIndex = static_cast<std::uint32_t>(source.materials.size() - 1u);
            object.representation = numi::matter::Representation::fem;
            object.deformableContact = false;
            object.deformableSelfContact = false;
            object.quasiStatic = rigidGraphPath != nullptr;
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
                if (rigidGraphPath != nullptr)
                    require(sourceNodeLocations.emplace(node.sourceId,
                        SourceNodeLocation{globalNodeBase + local,
                            static_cast<std::uint32_t>(group), node.coordinates}).second,
                        "source volume node ID appears in two tissue groups");
                if (const auto tie = tieBySourceNode.find(node.sourceId);
                    tie != tieBySourceNode.end()) {
                    require(tie->second.sourceMaterialId == spec.materialId,
                            "source rigid-tie tissue differs from volume ownership");
                    object.femFixedNodes.push_back(local);
                    tieBySourceNode.erase(tie);
                }
            }
            if (!checkpointPositions.empty()) {
                object.femReferenceNodes = object.femNodes;
                object.femReferenceSourceIdentity = {
                    0x4155db1d0d7b87ffull, 0xb2c668102d249587ull,
                    0x0e4461a539b18e67ull, 0x08f1f4817b5601bfull};
                for (std::size_t local = 0u; local < object.femNodes.size(); ++local) {
                    const auto& current = checkpointPositions[globalNodeBase + local];
                    object.femNodes[local] = {current[0u], current[1u], current[2u]};
                }
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
            if (spec.ligament || spec.meniscus)
                require(materialParameter(source.materials[group], "fiber_x") == 1.0 &&
                            materialParameter(source.materials[group], "fiber_y") == 0.0 &&
                            materialParameter(source.materials[group], "fiber_z") == 0.0 &&
                            object.femMaterialFrameRotations.size() ==
                                object.tetrahedra.size(),
                        "source fibre was not expressed in its cooked local frame");
            expectedTetrahedra += mesh.tetrahedra.size();
            globalNodeBase += static_cast<std::uint32_t>(mesh.nodes.size());
            source.objects.push_back(std::move(object));
        }
        require(expectedTetrahedra == 844287u,
                "source total tetrahedron count drifted from the pinned source");
        if (rigidGraphPath != nullptr)
            for (std::size_t group = 0u; group < specs.size(); ++group)
                if (specs[group].ligament)
                    require(materialParameter(source.materials[group],
                                "initial_stretch") == 1.0,
                            "source preload starts with final ligament prestrain");
        require(tieBySourceNode.empty(),
                "source rigid-tie node is absent from all source tissue volumes");
        SourceContactProjectionSummary initialContactProjection;
        if (sourceContactPath != nullptr) {
            std::unordered_map<std::uint32_t, SourcePoint> sourcePositions;
            sourcePositions.reserve(sourceNodeLocations.size() +
                                    sourceContact.rigidNodes.size());
            for (const auto& [id, node] : sourceNodeLocations)
                require(sourcePositions.emplace(id, node.sourcePosition).second,
                        "source tissue contact node position is ambiguous");
            for (const auto& node : sourceContact.rigidNodes)
                require(sourcePositions.emplace(node.sourceNodeId,
                                                node.sourcePosition).second,
                        "source rigid contact node overlaps a tissue node");
            require(bindSourceInitialContactProjections(sourceContact,
                        sourcePositions, initialContactProjection, error), error);
            require(!initialContactProjection.activeRows.empty() &&
                        initialContactProjection.activePoints > 0u,
                    "source sliding-elastic initial projection has no active quadrature");
        }
        std::size_t boundDiscreteFEMEdges = 0u;
        std::size_t boundDiscreteRigidEdges = 0u;
        if (sourceDiscretePath != nullptr) {
            const auto rigidBody = [&](std::uint32_t id) {
                return std::ranges::any_of(rigidGraph.bodies,
                    [id](const auto& body) { return body.materialId == id; });
            };
            for (std::size_t edgeIndex = 0u;
                 edgeIndex < sourceDiscrete.edges.size(); ++edgeIndex) {
                const auto& edge = sourceDiscrete.edges[edgeIndex];
                const bool rigid = edgeIndex < 4u;
                const auto bind = [&](std::uint32_t id, std::uint32_t owner,
                                      const std::array<double, 3>& point) {
                    if (rigid) {
                        require(rigidBody(owner),
                                "source patellar discrete edge has no rigid owner");
                    } else {
                        const auto found = sourceNodeLocations.find(id);
                        require(found != sourceNodeLocations.end() &&
                                specs[found->second.object].materialId == owner &&
                                found->second.sourcePosition == point,
                                "source MCL-meniscus discrete edge has no exact FEM node");
                    }
                };
                bind(edge.nodeA, edge.ownerA, edge.pointA);
                bind(edge.nodeB, edge.ownerB, edge.pointB);
                require(edge.ownerA != edge.ownerB,
                        "source discrete edge connects one material to itself");
                if (rigid) ++boundDiscreteRigidEdges;
                else ++boundDiscreteFEMEdges;
            }
            require(boundDiscreteRigidEdges == 4u &&
                        boundDiscreteFEMEdges == 402u,
                    "source discrete endpoint binding is incomplete");
        }
        if (sourceContactPath != nullptr) {
            std::unordered_map<std::uint32_t, std::uint32_t> rigidNodeOwner;
            rigidNodeOwner.reserve(sourceContact.rigidNodes.size());
            for (const auto& node : sourceContact.rigidNodes) {
                const bool knownBody = std::ranges::any_of(rigidGraph.bodies,
                    [&](const auto& body) {
                        return body.materialId == node.materialId;
                    });
                require(knownBody &&
                        rigidNodeOwner.emplace(node.sourceNodeId,
                                               node.materialId).second,
                        "source contact rigid vertex has no unique source body");
            }
            for (const auto& surface : sourceContact.surfaces) {
                const auto owner = std::ranges::find_if(specs,
                    [&](const auto& spec) {
                        return spec.materialId == surface.materialId;
                    });
                const bool rigidOwner = std::ranges::any_of(rigidGraph.bodies,
                    [&](const auto& body) {
                        return body.materialId == surface.materialId;
                    });
                require(owner != specs.end() || rigidOwner,
                        "source contact surface has no tissue or rigid owner");
                for (std::size_t faceIndex = surface.firstFace;
                     faceIndex < std::size_t(surface.firstFace) +
                         surface.faceCount; ++faceIndex) {
                    const auto& face = sourceContact.faces[faceIndex];
                    for (const auto nodeId : face.sourceNodes) {
                        if (rigidOwner) {
                            require(rigidNodeOwner.contains(nodeId) &&
                                    rigidNodeOwner.at(nodeId) ==
                                        surface.materialId,
                                    "source contact rigid face has wrong vertex owner");
                        } else {
                            const auto node = sourceNodeLocations.find(nodeId);
                            require(node != sourceNodeLocations.end() &&
                                    node->second.object ==
                                        std::size_t(owner - specs.begin()),
                                    "source contact tissue face has wrong cooked FEM owner");
                        }
                    }
                }
            }
        }

        const numi::matter::CompileResult compiled = numi::matter::compileWorld(
            source, {.maximumRateExponent = 0, .emitSpecializedMetal = false}
        );
        if (!compiled.succeeded() && checkpointSeedPath != nullptr)
            for (const auto& diagnostic : compiled.diagnostics)
                std::cerr << "checkpoint_seed_cook_diagnostic="
                          << diagnostic.message << '\n';
        require(compiled.succeeded(), "full Open Knee source-volume Matter cook failed");
        require(compiled.world.fem.tetrahedra.size() == expectedTetrahedra,
                "Matter cook changed the source tissue tetrahedron total");
        if (rigidGraphPath != nullptr) {
            require(compiled.world.contact.rigidProxies.size() == 9u &&
                    compiled.world.dispatch.rigidGeneralizedCapacity == 42u &&
                    compiled.world.contact.pairs.empty(),
                    "source rigid frames created contact or changed free-body ownership");
            require(std::ranges::all_of(compiled.world.objects,
                        [](const auto& object) {
                            return (object.flags & NM_OBJECT_FEM_QUASISTATIC) != 0u;
                        }) &&
                    std::ranges::all_of(compiled.world.contact.rigidProxies,
                        [](const auto& proxy) {
                            return (proxy.flags & NM_RIGID_DYNAMIC) == 0u ||
                                (proxy.flags & NM_RIGID_SOURCE_QUASISTATIC) != 0u;
                        }),
                    "source static analysis was cooked with inertial unknowns");
        }
        if (rigidTiesPath != nullptr) {
            const auto fixedCount = std::count_if(
                compiled.world.fem.nodes.begin(), compiled.world.fem.nodes.end(),
                [](const auto& node) { return node.restAndFixed.w == 1.0f; });
            require(static_cast<std::size_t>(fixedCount) == sourceTies.rows.size(),
                    "Matter cook changed source rigid-tie node ownership");
        }
        std::vector<NMSourceCylindricalJointGPU> runtimeJoints;
        std::vector<NMSourceRigidSpringGPU> runtimeSprings;
        std::vector<NMSourceFEMRigidTieGPU> runtimeTies;
        std::vector<NMSourceFEMSpringGPU> runtimeFEMSprings;
        std::vector<NMSourcePrestrainGPU> runtimePrestrain;
        std::vector<NMSourceContactNodeGPU> runtimeContactNodes;
        std::vector<NMSourceContactFaceGPU> runtimeContactFaces;
        std::vector<NMSourceContactSurfaceGPU> runtimeContactSurfaces;
        std::vector<NMSourceSlidingPairGPU> runtimeSlidingPairs;
        std::size_t sourceContactBackingFaces = 0u;
        double minimumSourceAutoPenalty = std::numeric_limits<double>::infinity();
        double maximumSourceAutoPenalty = 0.0;
        bool runtimeProgramInitialized = false;
        std::size_t runtimeResidentBytes = 0u;
        if (rigidGraphPath != nullptr) {
            const auto bodyIndex = [&](const std::uint32_t materialId) {
                const auto found = std::ranges::find_if(rigidGraph.bodies,
                    [materialId](const auto& body) {
                        return body.materialId == materialId;
                    });
                require(found != rigidGraph.bodies.end(),
                        "source tie or connector references no rigid body");
                return static_cast<std::uint32_t>(found - rigidGraph.bodies.begin());
            };
            const auto point = [](const std::array<double, 3>& sourcePoint,
                                  const double scale) -> nm_float4 {
                return {static_cast<float>(sourcePoint[0] * scale),
                        static_cast<float>(sourcePoint[1] * scale),
                        static_cast<float>(sourcePoint[2] * scale), 0.0f};
            };
            if (sourceContactPath != nullptr) {
                std::unordered_map<std::uint32_t, const SourceRigidContactNode*>
                    rigidNodeById;
                rigidNodeById.reserve(sourceContact.rigidNodes.size());
                for (const auto& node : sourceContact.rigidNodes)
                    require(rigidNodeById.emplace(node.sourceNodeId, &node).second,
                            "source rigid contact node ID repeats");
                std::unordered_map<std::uint32_t, std::uint32_t> contactNodeSlot;
                contactNodeSlot.reserve(sourceNodeLocations.size() +
                                        sourceContact.rigidNodes.size());
                runtimeContactFaces.reserve(sourceContact.faces.size());
                runtimeContactSurfaces.reserve(sourceContact.surfaces.size());
                for (std::uint32_t surfaceIndex = 0u;
                     surfaceIndex < sourceContact.surfaces.size(); ++surfaceIndex) {
                    const auto& surface = sourceContact.surfaces[surfaceIndex];
                    const bool rigid = std::ranges::any_of(rigidGraph.bodies,
                        [&](const auto& body) {
                            return body.materialId == surface.materialId;
                        });
                    std::uint32_t owner = 0u;
                    if (rigid) owner = bodyIndex(surface.materialId);
                    else {
                        const auto found = std::ranges::find_if(specs,
                            [&](const auto& spec) {
                                return spec.materialId == surface.materialId;
                            });
                        require(found != specs.end(),
                                "source contact has no cooked tissue owner");
                        owner = static_cast<std::uint32_t>(found - specs.begin());
                    }
                    runtimeContactSurfaces.push_back({
                        {surface.firstFace, surface.faceCount, owner,
                         rigid ? 1u : 0u}});
                    for (std::uint32_t faceIndex = surface.firstFace;
                         faceIndex < surface.firstFace + surface.faceCount;
                         ++faceIndex) {
                        const auto& face = sourceContact.faces[faceIndex];
                        NMSourceContactFaceGPU row{};
                        row.identity = {surfaceIndex, face.sourceFaceId,
                                        surface.materialId, NM_INVALID_INDEX};
                        for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
                            const std::uint32_t id = face.sourceNodes[vertex];
                            auto [slot, inserted] = contactNodeSlot.emplace(
                                id, static_cast<std::uint32_t>(runtimeContactNodes.size()));
                            if (inserted) {
                                NMSourceContactNodeGPU node{};
                                if (rigid) {
                                    const auto found = rigidNodeById.find(id);
                                    require(found != rigidNodeById.end() &&
                                            found->second->materialId == surface.materialId,
                                            "source rigid face node changes owner");
                                    const auto& center = rigidGraph.bodies[owner].centerOfMass;
                                    std::array<double, 3> local{};
                                    for (std::size_t axis = 0u; axis < 3u; ++axis)
                                        local[axis] = found->second->sourcePosition[axis] -
                                                      center[axis];
                                    node.identity = {owner, 1u, id, 0u};
                                    node.localPoint = point(local, 0.001);
                                } else {
                                    const auto found = sourceNodeLocations.find(id);
                                    require(found != sourceNodeLocations.end() &&
                                            found->second.object == owner,
                                            "source tissue face node changes owner");
                                    node.identity = {found->second.cookedNode,
                                                     0u, id, 0u};
                                }
                                runtimeContactNodes.push_back(node);
                            }
                            if (vertex == 0u) row.nodes.x = slot->second;
                            else if (vertex == 1u) row.nodes.y = slot->second;
                            else row.nodes.z = slot->second;
                        }
                        runtimeContactFaces.push_back(row);
                    }
                }
                // FEBio's automatic penalty reads the elastic tangent and
                // volume of the element behind each slave face. Bind that
                // exact owner now; a face with zero or multiple matching
                // source tetrahedra cannot silently receive a default law.
                std::unordered_multimap<std::uint64_t, std::uint32_t>
                    requestedFaces;
                requestedFaces.reserve(runtimeContactFaces.size());
                std::size_t tissueFaceCount = 0u;
                for (std::uint32_t face = 0u;
                     face < runtimeContactFaces.size(); ++face) {
                    const auto& row = runtimeContactFaces[face];
                    if (runtimeContactSurfaces[row.identity.x].identity.w != 0u)
                        continue;
                    requestedFaces.emplace(sourceFaceKey(
                        sourceContact.faces[face].sourceNodes), face);
                    ++tissueFaceCount;
                }
                std::size_t ambiguousFaces = 0u;
                std::uint32_t globalTet = 0u;
                constexpr std::array<std::array<std::uint32_t, 3>, 4> tetFaces{{
                    {0u, 1u, 2u}, {0u, 1u, 3u},
                    {0u, 2u, 3u}, {1u, 2u, 3u}}};
                for (std::size_t group = 0u; group < meshes.size(); ++group) {
                    const auto& mesh = meshes[group];
                    const auto& spec = specs[group];
                    for (std::size_t localTet = 0u;
                         localTet < mesh.tetrahedra.size(); ++localTet) {
                        const auto& tet = mesh.tetrahedra[localTet];
                        for (const auto& local : tetFaces) {
                            const auto key = sourceFaceKey({
                                tet.sourceNodeIds[local[0]],
                                tet.sourceNodeIds[local[1]],
                                tet.sourceNodeIds[local[2]]});
                            const auto matches = requestedFaces.equal_range(key);
                            for (auto found = matches.first;
                                 found != matches.second; ++found) {
                                auto& face = runtimeContactFaces[found->second];
                                if (face.identity.z != mesh.sourceMaterialId)
                                    continue;
                                if (face.identity.w != NM_INVALID_INDEX)
                                    ++ambiguousFaces;
                                else {
                                    face.identity.w = globalTet;
                                    const auto& faceNodes = sourceContact.faces[
                                        found->second].sourceNodes;
                                    const SourcePoint a = sourceNodeLocations.at(
                                        faceNodes[0]).sourcePosition;
                                    const SourcePoint b = sourceNodeLocations.at(
                                        faceNodes[1]).sourcePosition;
                                    const SourcePoint c = sourceNodeLocations.at(
                                        faceNodes[2]).sourcePosition;
                                    const SourcePoint areaVector = cross(
                                        subtract(b, a), subtract(c, a));
                                    const double area2 = length(areaVector);
                                    const double area = 0.5 * area2;
                                    require(area2 > 0.0 && std::isfinite(area2),
                                            "source automatic penalty has degenerate face");
                                    const SourcePoint normal = scale(areaVector,
                                        1.0 / area2);
                                    const SourcePoint t0 = sourceNodeLocations.at(
                                        tet.sourceNodeIds[0]).sourcePosition;
                                    const SourcePoint t1 = sourceNodeLocations.at(
                                        tet.sourceNodeIds[1]).sourcePosition;
                                    const SourcePoint t2 = sourceNodeLocations.at(
                                        tet.sourceNodeIds[2]).sourcePosition;
                                    const SourcePoint t3 = sourceNodeLocations.at(
                                        tet.sourceNodeIds[3]).sourcePosition;
                                    const double volume = std::abs(dot(
                                        subtract(t1, t0), cross(
                                            subtract(t2, t0),
                                            subtract(t3, t0)))) / 6.0;
                                    require(area > 0.0 && volume > 0.0 &&
                                                std::isfinite(area) &&
                                                std::isfinite(volume),
                                            "source automatic penalty has invalid face geometry");
                                    // FEBio 2.9 AutoPenalty uses the inverse
                                    // identity-state elastic tangent contracted
                                    // twice with the reference face normal.
                                    // c2=0 and initial in-situ stretch=1 at
                                    // activation. The uncoupled fibre tangent
                                    // uses its tensile-side derivative at 1.
                                    const double h = spec.ligament || spec.meniscus
                                        ? spec.c3 * spec.c4 : 0.0;
                                    double anisotropy = 0.0;
                                    if (h > 0.0) {
                                        const auto& frames = source.objects[group]
                                            .femMaterialFrameRotations;
                                        require(localTet < frames.size(),
                                                "source backing tet lacks its fibre frame");
                                        const auto fiber = rotateMaterialX(
                                            frames[localTet]);
                                        const double cosine = dot(normal, fiber);
                                        const double deviatoric =
                                            cosine * cosine - 1.0 / 3.0;
                                        anisotropy = h * std::pow(
                                            deviatoric / (4.0 * spec.c1), 2.0) /
                                            (1.0 + h / (6.0 * spec.c1));
                                    }
                                    const double compliance =
                                        1.0 / (6.0 * spec.c1) +
                                        1.0 / (9.0 * spec.bulk) - anisotropy;
                                    require(compliance > 0.0 &&
                                                std::isfinite(compliance),
                                            "source automatic penalty has singular compliance");
                                    const double effectiveModulus =
                                        1.0 / compliance; // MPa
                                    const double penalty =
                                        effectiveModulus * 1.0e9 * area / volume;
                                    face.autoPenalty = {
                                        static_cast<float>(penalty),
                                        static_cast<float>(area * 1.0e-6),
                                        static_cast<float>(volume * 1.0e-9),
                                        static_cast<float>(effectiveModulus * 1.0e6)};
                                    require(std::isfinite(face.autoPenalty.x) &&
                                                face.autoPenalty.x > 0.0f,
                                            "source automatic penalty exceeds runtime precision");
                                }
                            }
                        }
                        ++globalTet;
                    }
                }
                const auto missingFaces = std::count_if(
                    runtimeContactFaces.begin(), runtimeContactFaces.end(),
                    [&](const auto& face) {
                        return runtimeContactSurfaces[face.identity.x]
                                   .identity.w == 0u &&
                               face.identity.w == NM_INVALID_INDEX;
                    });
                require(globalTet == compiled.world.fem.tetrahedra.size() &&
                            ambiguousFaces == 0u && missingFaces == 0u,
                        "source contact backing elements are missing or ambiguous: " +
                        std::to_string(tissueFaceCount) + " tissue faces, " +
                        std::to_string(missingFaces) + " missing, " +
                        std::to_string(ambiguousFaces) + " ambiguous");
                sourceContactBackingFaces = tissueFaceCount;
                for (const auto& face : runtimeContactFaces) {
                    if (face.identity.w == NM_INVALID_INDEX) continue;
                    const double sourceUnits = face.autoPenalty.x * 1.0e-9;
                    minimumSourceAutoPenalty = std::min(
                        minimumSourceAutoPenalty, sourceUnits);
                    maximumSourceAutoPenalty = std::max(
                        maximumSourceAutoPenalty, sourceUnits);
                }
                runtimeSlidingPairs.reserve(sourceContact.pairs.size());
                for (const auto& pair : sourceContact.pairs) {
                    require(pair.parameters[0] == 0.0 &&
                            pair.parameters[4] == 1.0 &&
                            pair.parameters[5] == 1.0 &&
                            pair.parameters[6] == 0.0 &&
                            pair.parameters[11] == 0.0,
                            "source sliding contact enforcement changed");
                    NMSourceSlidingPairGPU row{};
                    row.identity = {pair.master, pair.slave, 1u, 0u};
                    row.normal = {
                        static_cast<float>(pair.parameters[3]),
                        static_cast<float>(pair.parameters[2] * 0.001),
                        static_cast<float>(pair.parameters[7]),
                        static_cast<float>(pair.parameters[8] *
                            sourceKneeBoundingBoxRadius * 0.001)};
                    runtimeSlidingPairs.push_back(row);
                }
                require(runtimeContactFaces.size() == 345070u &&
                            runtimeContactSurfaces.size() == 36u &&
                            runtimeSlidingPairs.size() == 18u,
                        "source contact runtime topology is incomplete");
            }
            runtimeTies.reserve(sourceTies.rows.size());
            for (const auto& tie : sourceTies.rows) {
                const auto location = sourceNodeLocations.find(tie.sourceNodeId);
                require(location != sourceNodeLocations.end() &&
                        compiled.world.fem.nodeRanges[location->second.cookedNode]
                            .objectIndex == location->second.object &&
                        compiled.world.fem.nodes[location->second.cookedNode]
                            .restAndFixed.w == 1.0f,
                        "source tie does not address its cooked FEM node");
                const std::uint32_t proxy = bodyIndex(tie.rigidBodyMaterialId);
                const auto& center = rigidGraph.bodies[proxy].centerOfMass;
                std::array<double, 3> local{};
                for (std::size_t axis = 0u; axis < 3u; ++axis)
                    local[axis] = location->second.sourcePosition[axis] - center[axis];
                NMSourceFEMRigidTieGPU row{};
                row.identity = {location->second.cookedNode, proxy,
                                location->second.object, tie.sourceNodeId};
                row.localPoint = point(local, 0.001);
                runtimeTies.push_back(row);
            }
            runtimeJoints.reserve(rigidGraph.joints.size());
            for (const auto& joint : rigidGraph.joints) {
                const std::uint32_t a = bodyIndex(joint.bodyA);
                const std::uint32_t b = bodyIndex(joint.bodyB);
                double translationCurve = 1.0, rotationCurve = 1.0;
                require(sampleSourceLoadCurve(rigidGraph, joint.translationCurve,
                            0.0, translationCurve) &&
                        sampleSourceLoadCurve(rigidGraph, joint.rotationCurve,
                            0.0, rotationCurve),
                        "source joint has no initial load-curve value");
                if (joint.rotationCurve != -1) {
                    const auto curve = std::ranges::find_if(rigidGraph.curves,
                        [&](const auto& value) {
                            return value.id == static_cast<std::uint32_t>(
                                joint.rotationCurve);
                        });
                    require(joint.rotationCurve == 9 &&
                            curve != rigidGraph.curves.end() &&
                            curve->points == std::vector<std::array<double, 2>>{{
                                {0.0, 0.0}, {1.0, 0.0}, {2.0, 1.0}}},
                            "source flexion curve is not the supported authored ramp");
                }
                NMSourceCylindricalJointGPU row{};
                row.indices = {a, b, joint.prescribedTranslation ? 1u : 0u,
                               joint.prescribedRotation ? 1u : 0u};
                row.referenceA = point(rigidGraph.bodies[a].centerOfMass, 0.001);
                row.referenceB = point(rigidGraph.bodies[b].centerOfMass, 0.001);
                row.origin = point(joint.origin, 0.001);
                row.axis = point(joint.axis, 1.0);
                // The pinned deck has no units declaration. This is the
                // explicit tonne-mm-second to SI hypothesis, not admission.
                row.parameters = {static_cast<float>(joint.forcePenalty * 1000.0),
                                  static_cast<float>(joint.momentPenalty * 0.001),
                                  static_cast<float>(joint.translation *
                                      translationCurve * 0.001),
                                  static_cast<float>(joint.rotation *
                                      (sourceDiscretePath != nullptr ?
                                          1.0 : rotationCurve))};
                row.axial.z = sourceDiscretePath != nullptr &&
                    joint.rotationCurve == 9 ? 1.0f : 0.0f;
                runtimeJoints.push_back(row);
            }
            if (sourceDiscretePath != nullptr) {
                constexpr std::array<std::pair<std::uint32_t, double>, 6>
                    sourcePrestrainTargets{{
                        {5u, 1.0}, {7u, 1.0}, {9u, 1.016},
                        {10u, 1.034}, {11u, 1.0}, {14u, 1.027}}};
                constexpr std::array<std::uint32_t, 6> curveIds{
                    1u, 2u, 3u, 4u, 5u, 6u};
                for (std::size_t i = 0u; i < sourcePrestrainTargets.size(); ++i) {
                    const auto [materialId, target] = sourcePrestrainTargets[i];
                    const auto spec = std::ranges::find_if(specs,
                        [&](const auto& value) {
                            return value.materialId == materialId;
                        });
                    require(spec != specs.end() && spec->ligament,
                            "source prestrain material has no continuum");
                    const auto material = static_cast<std::size_t>(spec - specs.begin());
                    const auto& curve = sourceDiscrete.curves[curveIds[i] - 1u];
                    require(curve.id == curveIds[i] &&
                            curve.points == std::vector<std::array<double, 2>>{{
                                {0.0, 1.0}, {1.0, target}, {2.0, target}}},
                            "source prestrain continuation curve changed");
                    const auto& parameters = source.materials[material].parameters;
                    const auto parameter = std::ranges::find_if(parameters,
                        [](const auto& value) {
                            return value.name == "initial_stretch";
                        });
                    require(parameter != parameters.end(),
                            "source continuum lacks prestrain parameter");
                    const auto global = compiled.world.materials[material].parameterOffset +
                        static_cast<std::uint32_t>(parameter - parameters.begin());
                    require(global < compiled.world.parameters.size() &&
                            compiled.world.parameters[global].valueAndBounds.x == 1.0f,
                            "source prestrain parameter cook changed");
                    NMSourcePrestrainGPU row{};
                    row.identity = {global, static_cast<std::uint32_t>(material),
                                    curveIds[i], 0u};
                    row.stretch = {1.0f, static_cast<float>(target), 0.0f, 0.0f};
                    runtimePrestrain.push_back(row);
                }
            }
            runtimeSprings.reserve(rigidGraph.springs.size());
            for (const auto& spring : rigidGraph.springs) {
                const std::uint32_t a = bodyIndex(spring.bodyA);
                const std::uint32_t b = bodyIndex(spring.bodyB);
                NMSourceRigidSpringGPU row{};
                row.indices = {a, b, 0u, 0u};
                row.referenceA = point(rigidGraph.bodies[a].centerOfMass, 0.001);
                row.referenceB = point(rigidGraph.bodies[b].centerOfMass, 0.001);
                row.insertionA = point(spring.insertionA, 0.001);
                row.insertionB = point(spring.insertionB, 0.001);
                row.parameters = {static_cast<float>(spring.stiffness * 1000.0),
                                  static_cast<float>(spring.freeLength * 0.001),
                                  0.0f, 0.0f};
                runtimeSprings.push_back(row);
            }
            if (sourceDiscretePath != nullptr) {
                for (std::size_t index = 0u; index < 4u; ++index) {
                    const auto& edge = sourceDiscrete.edges[index];
                    const auto& set = sourceDiscrete.sets[index / 2u];
                    const auto& curve = sourceDiscrete.curves[set.curveId - 1u];
                    require(curve.points.size() == 3u &&
                            curve.points[0][0] == -1.0 &&
                            curve.points[1][0] == 0.0 &&
                            curve.points[2][0] == 1.0 &&
                            curve.points[0][1] == 0.0 &&
                            curve.points[1][1] == 0.0 &&
                            curve.points[2][1] == (index < 2u ? 50.0 : 8.0),
                            "source patellar spring curve is not the pinned piecewise law");
                    const auto a = bodyIndex(edge.ownerA);
                    const auto b = bodyIndex(edge.ownerB);
                    NMSourceRigidSpringGPU row{};
                    row.indices = {a, b, 0u, 0u};
                    row.referenceA = point(rigidGraph.bodies[a].centerOfMass, 0.001);
                    row.referenceB = point(rigidGraph.bodies[b].centerOfMass, 0.001);
                    row.insertionA = point(edge.pointA, 0.001);
                    row.insertionB = point(edge.pointB, 0.001);
                    row.parameters = {static_cast<float>(
                        set.scale * (curve.points[2][1] - curve.points[1][1]) * 1000.0),
                        0.0f, 1.0f, 0.001f};
                    runtimeSprings.push_back(row);
                }
                runtimeFEMSprings.reserve(boundDiscreteFEMEdges);
                const auto& set = sourceDiscrete.sets[2];
                for (std::size_t index = 4u;
                     index < sourceDiscrete.edges.size(); ++index) {
                    const auto& edge = sourceDiscrete.edges[index];
                    const auto a = sourceNodeLocations.find(edge.nodeA);
                    const auto b = sourceNodeLocations.find(edge.nodeB);
                    require(a != sourceNodeLocations.end() &&
                            b != sourceNodeLocations.end(),
                            "source FEM spring lost its cooked node");
                    NMSourceFEMSpringGPU row{};
                    row.identity = {a->second.cookedNode, b->second.cookedNode,
                                    static_cast<std::uint32_t>(index + 1u), 0u};
                    row.referenceA = point(edge.pointA, 0.001);
                    row.referenceB = point(edge.pointB, 0.001);
                    // Source E is force per source length. Both endpoints
                    // remain in the native FEM displacement solve.
                    row.parameters = {static_cast<float>(set.stiffness * 1000.0),
                                      0.0f, 0.0f, 0.0f};
                    runtimeFEMSprings.push_back(row);
                }
                require(runtimeFEMSprings.size() == 402u,
                        "source FEM spring lowering is incomplete");
            }
            std::uint64_t programFingerprint = 14695981039346656037ull;
            for (const auto* bytes : {&tieBytes, &graphBytes, &discreteBytes})
                for (const auto byte : *bytes)
                    programFingerprint = (programFingerprint ^ byte) *
                        1099511628211ull;
            require(programFingerprint != 0u,
                    "source rigid program has invalid fingerprint");
#ifdef __APPLE__
            numi::matter::RuntimeConfiguration configuration;
            configuration.metallib = NUMI_MATTER_METALLIB;
            configuration.adaptiveTransfer = false;
            configuration.captureEvents = false;
            configuration.captureDiagnostics = solveTime >= 0.0;
            configuration.sourceCylindricalJoints = runtimeJoints;
            configuration.sourceRigidSprings = runtimeSprings;
            configuration.sourceFEMRigidTies = runtimeTies;
            configuration.sourceFEMSprings = runtimeFEMSprings;
            configuration.sourcePrestrain = runtimePrestrain;
            configuration.sourceContactNodes = runtimeContactNodes;
            configuration.sourceContactFaces = runtimeContactFaces;
            configuration.sourceContactSurfaces = runtimeContactSurfaces;
            configuration.sourceSlidingPairs = runtimeSlidingPairs;
            configuration.sourceRigidConnectorFingerprint = programFingerprint;
            numi::matter::Runtime runtime;
            const auto initialized = runtime.initialize(compiled.world, configuration);
            require(initialized.encoded && runtime.valid(),
                    "whole source tissue/rigid Matter program initialization: " +
                        initialized.message);
            runtimeProgramInitialized = true;
            runtimeResidentBytes = initialized.residentBytes;
            if (solveTime >= 0.0) {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                require(device != nil, "source step requires Metal device");
                id<MTLCommandQueue> queue = [device newCommandQueue];
                require(queue != nil, "source step requires Metal queue");
                std::vector<MRBodyStateGPU> bodies(rigidGraph.bodies.size());
                for (std::size_t index = 0u; index < bodies.size(); ++index) {
                    auto& body = bodies[index];
                    const auto& center = rigidGraph.bodies[index].centerOfMass;
                    body.position = {static_cast<float>(center[0] * 0.001),
                                     static_cast<float>(center[1] * 0.001),
                                     static_cast<float>(center[2] * 0.001), 0.0f};
                    body.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
                    for (const auto& checkpoint : checkpointBodies)
                        if (checkpoint.materialId == rigidGraph.bodies[index].materialId) {
                            body.position = {checkpoint.center[0u], checkpoint.center[1u],
                                             checkpoint.center[2u], 0.0f};
                            body.orientation = {checkpoint.orientation[0u],
                                checkpoint.orientation[1u], checkpoint.orientation[2u],
                                checkpoint.orientation[3u]};
                        }
                    const bool free = source.rigidProxies[index].dynamic;
                    body.linearVelocityAndInverseMass.w = free ? 1.0f : 0.0f;
                    body.flagsAndIndices[0] = free ? MR_MOTION_DYNAMIC : MR_MOTION_STATIC;
                    body.inverseInertiaWorldRow0 = {free ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f};
                    body.inverseInertiaWorldRow1 = {0.0f, free ? 1.0f : 0.0f, 0.0f, 0.0f};
                    body.inverseInertiaWorldRow2 = {0.0f, 0.0f, free ? 1.0f : 0.0f, 0.0f};
                }
                MRMetalWorldStatusGPU worldStatus{};
                worldStatus.code = MR_STEP_SUCCESS;
                id<MTLBuffer> current = [device newBufferWithBytes:bodies.data()
                    length:bodies.size() * sizeof(MRBodyStateGPU)
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> scene = [device newBufferWithBytes:bodies.data()
                    length:bodies.size() * sizeof(MRBodyStateGPU)
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> wrenches = [device newBufferWithLength:
                    bodies.size() * sizeof(MRABABodyWrenchGPU)
                    options:MTLResourceStorageModeShared];
                id<MTLBuffer> initialElementForces = elementForceOutputPath == nullptr
                    ? nil : [device newBufferWithLength:
                        std::size_t(compiled.world.dispatch.tetrahedronCount) *
                            sizeof(NMFEMElementVectorGPU)
                        options:MTLResourceStorageModeShared];
                id<MTLBuffer> statuses = [device newBufferWithBytes:&worldStatus
                    length:sizeof(worldStatus)
                    options:MTLResourceStorageModeShared];
                require(current != nil && scene != nil && wrenches != nil &&
                    statuses != nil &&
                    (elementForceOutputPath == nullptr || initialElementForces != nil),
                    "source step Metal arenas unavailable");
                id<MTLCommandBuffer> command = [queue commandBuffer];
                numi::matter::EncodeRequest request{};
                request.commandBuffer = (__bridge void*)command;
                request.environmentStatuses = (__bridge void*)statuses;
                request.rigid.currentBodies = (__bridge void*)current;
                request.rigid.currentBodyCount = static_cast<std::uint32_t>(bodies.size());
                request.rigid.currentBodyStride = request.rigid.currentBodyCount;
                request.rigid.sceneBodies = (__bridge void*)scene;
                request.rigid.sceneBodyCount = request.rigid.currentBodyCount;
                request.rigid.sceneStride = request.rigid.currentBodyCount;
                request.rigid.bodyWrenches = (__bridge void*)wrenches;
                request.rigid.bodyWrenchCount = request.rigid.currentBodyCount;
                request.rigid.bodyWrenchStride = request.rigid.currentBodyCount;
                if (initialElementForces != nil) {
                    request.femInitialElementForces =
                        (__bridge void*)initialElementForces;
                    request.femInitialElementForceCount =
                        compiled.world.dispatch.tetrahedronCount;
                }
                request.timestepSeconds = runtime.timestepSeconds();
                request.sourceContinuationTime = static_cast<float>(solveTime);
                request.phase = numi::matter::EncodePhase::preDynamics;
                auto encoded = runtime.encode(request);
                if (!encoded.encoded) {
                    std::cout << "source_step=not_encoded reason=" << encoded.message
                              << " source_equivalence=rejected\n";
                    return 2;
                }
                request.phase = numi::matter::EncodePhase::postCommit;
                request.femInitialElementForces = nullptr;
                request.femInitialElementForceCount = 0u;
                encoded = runtime.encode(request);
                require(encoded.encoded, "source step postCommit: " + encoded.message);
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "source step command failed");
                if (initialElementForces != nil) {
                    const auto* forces = static_cast<const NMFEMElementVectorGPU*>(
                        initialElementForces.contents);
                    if (elementForceOutputPath != nullptr) {
                        std::ofstream output(elementForceOutputPath,
                                             std::ios::binary | std::ios::trunc);
                        require(output.good(), "cannot open checkpoint element force output");
                        output.write(reinterpret_cast<const char*>(forces),
                            std::streamsize(initialElementForces.length));
                        require(output.good(), "cannot write checkpoint element force output");
                    }
                    std::vector<bool> mclFemoralTie(
                        compiled.world.dispatch.femNodeCount, false);
                    for (const auto& tie : runtimeTies)
                        if (tie.identity.z == 5u &&
                            rigidGraph.bodies[tie.identity.y].materialId == 4u)
                            mclFemoralTie[tie.identity.x] = true;
                    const auto& mcl = compiled.world.objects[5u];
                    std::array<double, 3u> total{};
                    for (std::uint32_t local = 0u; local < mcl.elementCount;
                         ++local) {
                        const std::uint32_t element = mcl.elementOffset + local;
                        const auto& tet = compiled.world.fem.tetrahedra[element];
                        const auto& force = forces[element];
                        for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
                            const std::uint32_t localNode = slot == 0u ?
                                tet.nodes.x : slot == 1u ? tet.nodes.y :
                                slot == 2u ? tet.nodes.z : tet.nodes.w;
                            const std::uint32_t node = localNode;
                            if (!mclFemoralTie[node]) continue;
                            const auto& value = slot == 0u ? force.node0 :
                                slot == 1u ? force.node1 :
                                slot == 2u ? force.node2 : force.node3;
                            total[0u] += value.x;
                            total[1u] += value.y;
                            total[2u] += value.z;
                        }
                    }
                    std::cout << "checkpoint_initial_mcl_femoral_continuum_force_n="
                              << total[0u] << ',' << total[1u] << ','
                              << total[2u] << '\n';
                }
                const auto state = runtime.snapshot();
                require(state.available && !state.statuses.empty(),
                        "source step state readback unavailable");
                double maximumDisplacement = 0.0;
                for (std::size_t node = 0u; node < state.femNodes.size(); ++node) {
                    const auto& actual = state.femNodes[node].positionAndMass;
                    const auto& rest = compiled.world.fem.nodes[node].restAndFixed;
                    const double dx = actual.x - rest.x, dy = actual.y - rest.y,
                                 dz = actual.z - rest.z;
                    maximumDisplacement = std::max(maximumDisplacement,
                        std::sqrt(dx*dx + dy*dy + dz*dz));
                }
                const bool accepted = state.statuses[0u].code == NM_STATUS_SUCCESS;
                double maximumResidual = 0.0;
                double minimumJacobian = std::numeric_limits<double>::infinity();
                std::size_t worstObject = 0u;
                for (std::size_t object = 0u;
                     object < state.solverCertificates.size(); ++object) {
                    const auto& certificate = state.solverCertificates[object];
                    if (certificate.nonlinear.x > maximumResidual) {
                        maximumResidual = certificate.nonlinear.x;
                        worstObject = object;
                    }
                    minimumJacobian = std::min(minimumJacobian,
                        static_cast<double>(certificate.validity.x));
                }
                std::cout << "source_step=" << (accepted ? "accepted" : "rejected")
                          << " source_time=" << solveTime
                          << " newton_budget=" << newtonBudget
                          << " fgmres_budget=" << fgmresBudget
                          << " matter_status=" << state.statuses[0u].code
                          << " failing_object=" << state.statuses[0u].objectIndex
                          << " failing_index=" << state.statuses[0u].failingIndex
                          << " completed_microsteps=" << state.statuses[0u].completedMicrosteps
                          << " fgmres_iterations=" << state.statuses[0u].fgmresIterations
                          << " diagnostic0=" << state.statuses[0u].diagnostics.x
                          << " diagnostic1=" << state.statuses[0u].diagnostics.y
                          << " residual=" << state.statuses[0u].diagnostics.z
                          << " correction=" << state.statuses[0u].diagnostics.w
                          << " worst_object=" << worstObject
                          << " maximum_object_residual=" << maximumResidual
                          << " minimum_jacobian=" << minimumJacobian
                          << " max_fem_displacement_m=" << maximumDisplacement
                          << " source_contact_history=" << state.sourceContactHistory.size()
                          << " source_equivalence=unqualified\n";
                if (checkpointSeedPath != nullptr)
                    for (std::size_t body = 0u; body < state.rigidStates.size(); ++body) {
                        const auto& rigid = state.rigidStates[body];
                        std::cout << "checkpoint_rigid_body index=" << body
                                  << " material=" << rigidGraph.bodies[body].materialId
                                  << " center_m=" << rigid.bodyCenter.x << ','
                                  << rigid.bodyCenter.y << ',' << rigid.bodyCenter.z
                                  << " quaternion=" << rigid.orientation.x << ','
                                  << rigid.orientation.y << ',' << rigid.orientation.z
                                  << ',' << rigid.orientation.w << '\n';
                    }
                if (!accepted)
                    for (std::size_t object = 0u;
                         object < state.solverCertificates.size(); ++object) {
                        const auto& certificate = state.solverCertificates[object];
                        std::cout << "source_certificate object=" << object
                                  << " residual=" << certificate.nonlinear.x
                                  << " correction=" << certificate.nonlinear.y
                                  << " volume=" << certificate.nonlinear.z
                                  << " pressure=" << certificate.nonlinear.w
                                  << " minimum_jacobian=" << certificate.validity.x
                                  << " contact_gap=" << certificate.contact.y
                                  << " transport=" << certificate.transport.x << ','
                                  << certificate.transport.y << ','
                                  << certificate.transport.z << ','
                                  << certificate.transport.w << '\n';
                    }
                if (!accepted && !state.diagnosticGeneralizedResidual.empty()) {
                    const auto& residual = state.diagnosticGeneralizedResidual;
                    for (std::size_t object = 0u;
                         object < compiled.world.objects.size(); ++object) {
                        const auto& layout = compiled.world.objects[object];
                        double squared = 0.0, maximum = 0.0;
                        std::uint32_t worstNode = 0u;
                        for (std::uint32_t local = 0u;
                             local < layout.stateCount; ++local) {
                            const std::uint32_t node = layout.stateOffset + local;
                            const auto& row = residual[node];
                            const double magnitude = std::sqrt(double(row.x)*row.x +
                                double(row.y)*row.y + double(row.z)*row.z);
                            squared += magnitude*magnitude;
                            if (magnitude > maximum) {
                                maximum = magnitude;
                                worstNode = node;
                            }
                        }
                        std::cout << "source_residual_block object=" << object
                                  << " name=" << source.objects[object].name
                                  << " l2=" << std::sqrt(squared)
                                  << " maximum=" << maximum
                                  << " worst_node=" << worstNode << '\n';
                    }
                    const std::size_t rigidBase =
                        2u * compiled.world.dispatch.femNodeCount +
                        compiled.world.dispatch.mpmActiveNodeCapacity;
                    double rigidSquared = 0.0, rigidMaximum = 0.0;
                    std::size_t worstDof = 0u;
                    for (std::size_t dof = 0u;
                         dof < compiled.world.dispatch.rigidGeneralizedCapacity; ++dof) {
                        const double magnitude = std::abs(residual[rigidBase + dof].x);
                        rigidSquared += magnitude*magnitude;
                        if (magnitude > rigidMaximum) {
                            rigidMaximum = magnitude;
                            worstDof = dof;
                        }
                    }
                    std::cout << "source_residual_block name=rigid"
                              << " l2=" << std::sqrt(rigidSquared)
                              << " maximum=" << rigidMaximum
                              << " worst_dof=" << worstDof << '\n';
                    if (checkpointSeedPath != nullptr)
                        for (std::size_t dof = 0u;
                             dof < compiled.world.dispatch.rigidGeneralizedCapacity;
                             ++dof)
                            std::cout << "checkpoint_rigid_residual dof=" << dof
                                      << " impulse=" << residual[rigidBase + dof].x
                                      << '\n';
                }
                if (!accepted &&
                    state.diagnosticSourceContactNodeForces.size() ==
                        runtimeContactNodes.size() &&
                    !state.diagnosticGeneralizedResidual.empty()) {
                    std::vector<double> contactImpulseSquared(source.objects.size());
                    std::vector<double> contactResidualDot(source.objects.size());
                    std::vector<std::array<double, 3u>> rigidContactForces(
                        rigidGraph.bodies.size());
                    std::vector<std::array<double, 3u>> tiedContactForces(
                        source.objects.size() * rigidGraph.bodies.size());
                    std::vector<std::uint32_t> tieBodyByFEM(
                        compiled.world.dispatch.femNodeCount, UINT32_MAX);
                    for (const auto& tie : runtimeTies)
                        tieBodyByFEM[tie.identity.x] = tie.identity.y;
                    for (std::size_t slot = 0u;
                         slot < runtimeContactNodes.size(); ++slot) {
                        const auto& node = runtimeContactNodes[slot];
                        if (node.identity.y == 1u) {
                            const auto& force =
                                state.diagnosticSourceContactNodeForces[slot];
                            auto& sum = rigidContactForces[node.identity.x];
                            sum[0u] += force.x;
                            sum[1u] += force.y;
                            sum[2u] += force.z;
                            continue;
                        }
                        if (node.identity.y != 0u) continue;
                        const std::uint32_t fem = node.identity.x;
                        const std::uint32_t object =
                            compiled.world.fem.nodeRanges[fem].objectIndex;
                        const auto& force =
                            state.diagnosticSourceContactNodeForces[slot];
                        if (tieBodyByFEM[fem] != UINT32_MAX) {
                            auto& tied = tiedContactForces[
                                object * rigidGraph.bodies.size() +
                                tieBodyByFEM[fem]];
                            tied[0u] += force.x;
                            tied[1u] += force.y;
                            tied[2u] += force.z;
                        }
                        const auto& residual =
                            state.diagnosticGeneralizedResidual[fem];
                        const double dt = runtime.timestepSeconds();
                        contactImpulseSquared[object] += dt*dt *
                            (double(force.x)*force.x + double(force.y)*force.y +
                             double(force.z)*force.z);
                        contactResidualDot[object] += dt *
                            (double(force.x)*residual.x +
                             double(force.y)*residual.y +
                             double(force.z)*residual.z);
                    }
                    for (std::size_t object = 0u;
                         object < source.objects.size(); ++object)
                        std::cout << "source_contact_block object=" << object
                                  << " name=" << source.objects[object].name
                                  << " contact_impulse_l2="
                                  << std::sqrt(contactImpulseSquared[object])
                                  << " contact_residual_dot="
                                  << contactResidualDot[object] << '\n';
                    if (checkpointSeedPath != nullptr)
                        for (std::size_t body = 0u;
                             body < rigidContactForces.size(); ++body) {
                            const auto& force = rigidContactForces[body];
                            std::cout << "checkpoint_rigid_contact_force body="
                                      << body << " material="
                                      << rigidGraph.bodies[body].materialId
                                      << " newton=" << force[0u] << ','
                                      << force[1u] << ',' << force[2u] << '\n';
                        }
                    if (checkpointSeedPath != nullptr)
                        for (std::size_t object = 0u;
                             object < source.objects.size(); ++object)
                            for (std::size_t body = 0u;
                                 body < rigidGraph.bodies.size(); ++body) {
                                const auto& force = tiedContactForces[
                                    object * rigidGraph.bodies.size() + body];
                                if (std::abs(force[0u]) + std::abs(force[1u]) +
                                    std::abs(force[2u]) <= 1.0e-8) continue;
                                std::cout << "checkpoint_tied_contact_force object="
                                          << object << " name="
                                          << source.objects[object].name
                                          << " body_material="
                                          << rigidGraph.bodies[body].materialId
                                          << " newton=" << force[0u] << ','
                                          << force[1u] << ',' << force[2u] << '\n';
                            }
                }
                if (!accepted && checkpointSeedPath != nullptr &&
                    state.diagnosticSourceTieImpulses.size() == runtimeTies.size()) {
                    std::vector<std::array<double, 3u>> byObject(source.objects.size());
                    std::vector<std::array<double, 3u>> byBody(rigidGraph.bodies.size());
                    std::vector<std::array<double, 3u>> byObjectBody(
                        source.objects.size() * rigidGraph.bodies.size());
                    for (std::size_t row = 0u; row < runtimeTies.size(); ++row) {
                        const auto& tie = runtimeTies[row];
                        const auto& impulse = state.diagnosticSourceTieImpulses[row];
                        for (auto* sum : {&byObject[tie.identity.z],
                                          &byBody[tie.identity.y],
                                          &byObjectBody[tie.identity.z *
                                              rigidGraph.bodies.size() + tie.identity.y]}) {
                            (*sum)[0u] += impulse.x;
                            (*sum)[1u] += impulse.y;
                            (*sum)[2u] += impulse.z;
                        }
                    }
                    for (std::size_t object = 0u; object < byObject.size(); ++object) {
                        const auto& sum = byObject[object];
                        std::cout << "checkpoint_tie_object object=" << object
                                  << " name=" << source.objects[object].name
                                  << " impulse=" << sum[0u] << ','
                                  << sum[1u] << ',' << sum[2u] << '\n';
                    }
                    for (std::size_t body = 0u; body < byBody.size(); ++body) {
                        const auto& sum = byBody[body];
                        std::cout << "checkpoint_tie_body body=" << body
                                  << " material=" << rigidGraph.bodies[body].materialId
                                  << " impulse=" << sum[0u] << ','
                                  << sum[1u] << ',' << sum[2u] << '\n';
                    }
                    for (std::size_t object = 0u; object < byObject.size(); ++object)
                        for (std::size_t body = 0u; body < byBody.size(); ++body) {
                            const auto& sum = byObjectBody[
                                object * byBody.size() + body];
                            if (std::abs(sum[0u]) + std::abs(sum[1u]) +
                                std::abs(sum[2u]) <= 1.0e-8) continue;
                            std::cout << "checkpoint_tie_path object=" << object
                                      << " name=" << source.objects[object].name
                                      << " body_material="
                                      << rigidGraph.bodies[body].materialId
                                      << " impulse=" << sum[0u] << ','
                                      << sum[1u] << ',' << sum[2u] << '\n';
                        }
                }
                return accepted ? 0 : 2;
            }
#else
            throw std::runtime_error("whole source Matter runtime requires Apple Metal");
#endif
        }
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
                  << " source_rigid_tie_nodes=" << sourceTies.rows.size()
                  << " source_rigid_bodies=" << rigidGraph.bodies.size()
                  << " source_cylindrical_joints=" << rigidGraph.joints.size()
                  << " source_rigid_springs=" << rigidGraph.springs.size()
                  << " source_discrete_rigid_edges=" << boundDiscreteRigidEdges
                  << " source_prestrain_curves=" << runtimePrestrain.size()
                  << " source_discrete_fem_edges=" << boundDiscreteFEMEdges
                  << " source_discrete_fem="
                  << (sourceDiscretePath != nullptr
                      ? "included_in_newton_program_not_stepped"
                      : "not_assembled")
                  << " source_discrete_rigid="
                  << (sourceDiscretePath != nullptr
                      ? "source_constant_extended_curve_in_newton_program_not_stepped"
                      : "not_assembled")
                  << " coupled_runtime_program=" <<
                     (runtimeProgramInitialized ? "initialized" : "not_requested")
                  << " runtime_resident_bytes=" << runtimeResidentBytes
                  << " source_contact_faces=" << sourceContact.faces.size()
                  << " source_contact_runtime_nodes=" << runtimeContactNodes.size()
                  << " source_contact_runtime_faces=" << runtimeContactFaces.size()
                  << " source_contact_runtime_pairs=" << runtimeSlidingPairs.size()
                  << " source_contact_runtime_bvh_nodes="
                  << (runtimeContactFaces.empty() ? 0u :
                      2u * runtimeContactFaces.size() -
                          runtimeContactSurfaces.size())
                  << " source_contact_runtime_projection_slots="
                  << 3u * runtimeContactFaces.size()
                  << " source_contact_backing_tissue_faces="
                  << sourceContactBackingFaces
                  << " source_contact_auto_penalty_range_n_per_mm3="
                  << minimumSourceAutoPenalty << ','
                  << maximumSourceAutoPenalty
                  << " source_contact_initial_gauss_points="
                  << initialContactProjection.quadraturePoints
                  << " source_contact_initial_active_gauss_points="
                  << initialContactProjection.activePoints
                  << " source_contact_initial_active_faces="
                  << std::accumulate(
                        initialContactProjection.activeFacesBySurface.begin(),
                        initialContactProjection.activeFacesBySurface.end(),
                        std::uint64_t{0})
                  << " source_contact_initial_active_faces_by_surface=";
        for (std::size_t surface = 0u;
             surface < initialContactProjection.activeFacesBySurface.size();
             ++surface) {
            if (surface != 0u) std::cout << ',';
            std::cout << initialContactProjection.activeFacesBySurface[surface];
        }
        std::cout
                  << " source_contact_initial_max_gap_mm="
                  << initialContactProjection.maximumGap
                  << " source_contact_search_radius_mm="
                  << initialContactProjection.searchRadius
                  << " source_contact_rigid_vertices="
                  << sourceContact.rigidNodes.size()
                  << " source_contact="
                  << (sourceContactPath != nullptr
                      ? "candidate_bvh_traction_tangent_history_staged_fail_closed"
                      : "not_assembled")
                  << " source_analysis="
                  << (rigidGraphPath != nullptr
                      ? "quasistatic_inertia_excluded" : "unqualified")
                  << " source_prestrain="
                  << (rigidGraphPath != nullptr
                      ? (sourceDiscretePath != nullptr
                          ? "curve_schedule_in_newton_program_not_stepped"
                          : "unit_at_initial_state_target_not_continued")
                      : "target_without_continuation")
                  << " source_flexion="
                  << (sourceDiscretePath != nullptr
                      ? "curve9_in_rigid_newton_program_not_stepped"
                      : "not_continued")
                  << " source_initialization=not_solved "
                     "source_equivalence=rejected "
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
        if (argc == 17 && std::string(argv[1]) == "--source-artifacts" &&
            std::string(argv[8]) == "--solve-source-time" &&
            std::string(argv[10]) == "--solver-budget" &&
            std::string(argv[13]) == "--checkpoint-seed" &&
            std::string(argv[15]) == "--checkpoint-element-forces")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], std::stod(argv[9]),
                static_cast<std::uint32_t>(std::stoul(argv[11])),
                static_cast<std::uint32_t>(std::stoul(argv[12])), argv[14],
                argv[16]);
        if (argc == 15 && std::string(argv[1]) == "--source-artifacts" &&
            std::string(argv[8]) == "--solve-source-time" &&
            std::string(argv[10]) == "--solver-budget" &&
            std::string(argv[13]) == "--checkpoint-seed")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], std::stod(argv[9]),
                static_cast<std::uint32_t>(std::stoul(argv[11])),
                static_cast<std::uint32_t>(std::stoul(argv[12])), argv[14]);
        if (argc == 13 && std::string(argv[1]) == "--source-artifacts" &&
            std::string(argv[8]) == "--solve-source-time" &&
            std::string(argv[10]) == "--solver-budget")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], std::stod(argv[9]),
                static_cast<std::uint32_t>(std::stoul(argv[11])),
                static_cast<std::uint32_t>(std::stoul(argv[12])));
        if (argc == 10 && std::string(argv[1]) == "--source-artifacts" &&
            std::string(argv[8]) == "--solve-source-time")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                argv[6], argv[7], std::stod(argv[9]));
        if (argc == 4 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3]);
        if (argc == 5 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3], argv[4]);
        if (argc == 7 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                                        argv[6]);
        if (argc == 8 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5],
                                        argv[6], argv[7]);
        if (argc == 6 && std::string(argv[1]) == "--source-artifacts")
            return checkSourceArtifacts(argv[2], argv[3], argv[4], argv[5]);
        if (argc == 3 && std::string(argv[1]) == "--source-sidecar")
            return checkSourceSidecar(argv[2]);
        require(argc == 1, "usage: open knee fiber field check [--source-sidecar PATH] "
                           "[--source-artifacts FIBER_PATH SOURCE_VOLUME_MESH_PATH [SOURCE_RIGID_TIES_PATH [SOURCE_RIGID_GRAPH_PATH [SOURCE_CONTACT_PATH [SOURCE_DISCRETE_PATH]]]] [--solve-source-time TIME [--solver-budget NEWTON FGMRES [--checkpoint-seed PATH [--checkpoint-element-forces PATH]]]]]");
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
