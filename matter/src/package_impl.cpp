#include "numi/matter/detail.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace numi::matter {
namespace {

inline constexpr std::array<char, 16> kMagic{
    'N', 'U', 'M', 'I', 'M', 'A', 'T', 'T',
    'E', 'R', 'P', 'K', 'G', '\0', '\0', '\0',
};
inline constexpr std::uint32_t kPackageVersion = 8u;
inline constexpr std::uint32_t kEndianMarker = 0x01020304u;

enum class Section : std::uint32_t {
    dispatch = 1u,
    mixedSolver,
    materials,
    mixedMaterials,
    parameters,
    stateInitials,
    instructions,
    scalarPrograms,
    objects,
    mpmParticles,
    mpmNodes,
    mpmGrids,
    mpmBlocks,
    mpmBlockLookup,
    mpmStencils,
    mpmNodeIncidence,
    mpmNodeRanges,
    femNodes,
    femTetrahedra,
    femSurfaceFaces,
    femNodeIncidence,
    femNodeRanges,
    femCapacities,
    femFields,
    femFieldBoundaries,
    femTopologyNodes,
    femCohesiveFaces,
    femMutationCommands,
    femPunctureChannels,
    rigidProxies,
    contactPairs,
    contactNodeIncidence,
    contactNodeRanges,
    rigidIncidence,
    rigidRanges,
    adaptive,
    schedulers,
    identification,
    learnedMaterials,
    learnedLayers,
    learnedWeights,
    generatedMetal,
};

inline constexpr std::uint32_t kSectionCount =
    static_cast<std::uint32_t>(Section::generatedMetal);

struct PackageHeader {
    std::array<char, 16> magic{};
    std::uint32_t version = 0u;
    std::uint32_t endian = 0u;
    std::uint32_t matterAbiVersion = 0u;
    std::uint32_t sectionCount = 0u;
    std::uint64_t physicsFingerprint = 0u;
    std::uint64_t fingerprint = 0u;
    std::uint64_t headerHash = 0u;
};

struct SectionHeader {
    std::uint32_t id = 0u;
    std::uint32_t elementSize = 0u;
    std::uint64_t elementCount = 0u;
    std::uint64_t byteCount = 0u;
    std::uint64_t contentHash = 0u;
};

static_assert(std::is_trivially_copyable_v<PackageHeader>);
static_assert(std::is_trivially_copyable_v<SectionHeader>);

template <typename T>
[[nodiscard]] bool writeRaw(std::ofstream& stream, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

[[nodiscard]] bool writeBytes(
    std::ofstream& stream,
    const void* data,
    const std::size_t bytes
) {
    if (bytes == 0u) {
        return true;
    }
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(stream);
}

template <typename T>
[[nodiscard]] bool writeSection(
    std::ofstream& stream,
    const Section id,
    const std::span<const T> values
) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
        return false;
    }
    SectionHeader header{};
    header.id = static_cast<std::uint32_t>(id);
    header.elementSize = sizeof(T);
    header.elementCount = values.size();
    header.byteCount = values.size_bytes();
    header.contentHash = detail::hashBytes(values.data(), values.size_bytes());
    return writeRaw(stream, header) &&
        writeBytes(stream, values.data(), values.size_bytes());
}

[[nodiscard]] bool writeStringSection(
    std::ofstream& stream,
    const Section id,
    const std::string_view value
) {
    SectionHeader header{};
    header.id = static_cast<std::uint32_t>(id);
    header.elementSize = 1u;
    header.elementCount = value.size();
    header.byteCount = value.size();
    header.contentHash = detail::hashString(value);
    return writeRaw(stream, header) && writeBytes(stream, value.data(), value.size());
}

template <typename T>
[[nodiscard]] bool readRaw(std::ifstream& stream, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

[[nodiscard]] bool readBytes(
    std::ifstream& stream,
    void* data,
    const std::size_t bytes
) {
    if (bytes == 0u) {
        return true;
    }
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(stream);
}

template <typename T>
[[nodiscard]] bool decodeVector(
    std::ifstream& stream,
    const SectionHeader& header,
    std::vector<T>& output,
    std::string* error
) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (header.elementSize != sizeof(T) ||
        header.elementCount > std::numeric_limits<std::size_t>::max() ||
        header.byteCount != header.elementCount * sizeof(T) ||
        header.byteCount > std::numeric_limits<std::size_t>::max()) {
        if (error != nullptr) {
            *error = "matter package section has an incompatible element layout";
        }
        return false;
    }
    std::vector<T> candidate(static_cast<std::size_t>(header.elementCount));
    if (!readBytes(stream, candidate.data(), candidate.size() * sizeof(T))) {
        if (error != nullptr) {
            *error = "matter package ended inside a typed section";
        }
        return false;
    }
    if (detail::hashBytes(candidate.data(), candidate.size() * sizeof(T)) !=
        header.contentHash) {
        if (error != nullptr) {
            *error = "matter package section content hash mismatch";
        }
        return false;
    }
    output = std::move(candidate);
    return true;
}

[[nodiscard]] bool skipSection(
    std::ifstream& stream,
    const SectionHeader& header,
    std::string* error
) {
    if (header.byteCount > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())) {
        if (error != nullptr) {
            *error = "matter package section is too large to seek";
        }
        return false;
    }
    stream.seekg(static_cast<std::streamoff>(header.byteCount), std::ios::cur);
    if (!stream) {
        if (error != nullptr) {
            *error = "matter package ended while skipping an unknown section";
        }
        return false;
    }
    return true;
}

[[nodiscard]] std::uint64_t headerHash(PackageHeader header) {
    header.headerHash = 0u;
    return detail::hashBytes(&header, sizeof(header));
}

} // namespace

bool writePackage(
    const CompileResult& compiled,
    const std::filesystem::path& path,
    std::string* error
) {
    if (!compiled.succeeded()) {
        if (error != nullptr) {
            *error = "cannot write a matter package from a failed compile";
        }
        return false;
    }
    const CompiledWorld& world = compiled.world;
    std::string layoutError;
    if (!validateCompiledWorldLayout(world, &layoutError)) {
        if (error != nullptr) {
            *error = "cannot package an invalid Matter world: " + layoutError;
        }
        return false;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        if (error != nullptr) {
            *error = "cannot create matter package: " + path.string();
        }
        return false;
    }

    PackageHeader header{};
    header.magic = kMagic;
    header.version = kPackageVersion;
    header.endian = kEndianMarker;
    header.matterAbiVersion = NM_MATTER_ABI_VERSION;
    header.sectionCount = kSectionCount;
    header.physicsFingerprint = world.physicsFingerprint;
    header.fingerprint = world.fingerprint;
    header.headerHash = headerHash(header);
    if (!writeRaw(stream, header)) {
        if (error != nullptr) {
            *error = "failed to write matter package header";
        }
        return false;
    }

    const bool written =
        writeSection(stream, Section::dispatch,
            std::span<const NMMatterDispatchGPU>(&world.dispatch, 1u)) &&
        writeSection(stream, Section::mixedSolver,
            std::span<const NMMixedSolverGPU>(&world.mixedSolver, 1u)) &&
        writeSection(stream, Section::materials,
            std::span<const NMMaterialGPU>(world.materials)) &&
        writeSection(stream, Section::mixedMaterials,
            std::span<const NMMixedMaterialGPU>(world.mixedMaterials)) &&
        writeSection(stream, Section::parameters,
            std::span<const NMParameterRangeGPU>(world.parameters)) &&
        writeSection(stream, Section::stateInitials,
            std::span<const float>(world.stateInitials)) &&
        writeSection(stream, Section::instructions,
            std::span<const NMExpressionInstructionGPU>(world.instructions)) &&
        writeSection(stream, Section::scalarPrograms,
            std::span<const NMScalarProgramGPU>(world.scalarPrograms)) &&
        writeSection(stream, Section::objects,
            std::span<const NMContinuumObjectGPU>(world.objects)) &&
        writeSection(stream, Section::mpmParticles,
            std::span<const NMParticleStateGPU>(world.mpm.particles)) &&
        writeSection(stream, Section::mpmNodes,
            std::span<const NMGridNodeStateGPU>(world.mpm.nodes)) &&
        writeSection(stream, Section::mpmGrids,
            std::span<const NMMPMGridGPU>(world.mpm.grids)) &&
        writeSection(stream, Section::mpmBlocks,
            std::span<const NMMPMBlockGPU>(world.mpm.blocks)) &&
        writeSection(stream, Section::mpmBlockLookup,
            std::span<const std::uint32_t>(world.mpm.blockLookup)) &&
        writeSection(stream, Section::mpmStencils,
            std::span<const NMMPMStencilGPU>(world.mpm.stencils)) &&
        writeSection(stream, Section::mpmNodeIncidence,
            std::span<const std::uint32_t>(world.mpm.nodeIncidence)) &&
        writeSection(stream, Section::mpmNodeRanges,
            std::span<const NMIncidenceRangeGPU>(world.mpm.nodeRanges)) &&
        writeSection(stream, Section::femNodes,
            std::span<const NMFEMNodeStateGPU>(world.fem.nodes)) &&
        writeSection(stream, Section::femTetrahedra,
            std::span<const NMTetrahedronGPU>(world.fem.tetrahedra)) &&
        writeSection(stream, Section::femSurfaceFaces,
            std::span<const NMFEMSurfaceFaceGPU>(world.fem.surfaceFaces)) &&
        writeSection(stream, Section::femNodeIncidence,
            std::span<const std::uint32_t>(world.fem.nodeIncidence)) &&
        writeSection(stream, Section::femNodeRanges,
            std::span<const NMIncidenceRangeGPU>(world.fem.nodeRanges)) &&
        writeSection(stream, Section::femCapacities,
            std::span<const NMFEMCapacityGPU>(world.fem.capacities)) &&
        writeSection(stream, Section::femFields,
            std::span<const NMFEMFieldStateGPU>(world.fem.fields)) &&
        writeSection(stream, Section::femFieldBoundaries,
            std::span<const NMFieldBoundaryGPU>(world.fem.fieldBoundaries)) &&
        writeSection(stream, Section::femTopologyNodes,
            std::span<const NMFEMTopologyNodeGPU>(world.fem.topologyNodes)) &&
        writeSection(stream, Section::femCohesiveFaces,
            std::span<const NMCohesiveFaceGPU>(world.fem.cohesiveFaces)) &&
        writeSection(stream, Section::femMutationCommands,
            std::span<const NMMutationCommandGPU>(world.fem.mutationCommands)) &&
        writeSection(stream, Section::femPunctureChannels,
            std::span<const NMPunctureChannelGPU>(world.fem.punctureChannels)) &&
        writeSection(stream, Section::rigidProxies,
            std::span<const NMRigidProxyGPU>(world.contact.rigidProxies)) &&
        writeSection(stream, Section::contactPairs,
            std::span<const NMContactPairGPU>(world.contact.pairs)) &&
        writeSection(stream, Section::contactNodeIncidence,
            std::span<const std::uint32_t>(world.contact.nodeIncidence)) &&
        writeSection(stream, Section::contactNodeRanges,
            std::span<const NMIncidenceRangeGPU>(world.contact.nodeRanges)) &&
        writeSection(stream, Section::rigidIncidence,
            std::span<const std::uint32_t>(world.contact.rigidIncidence)) &&
        writeSection(stream, Section::rigidRanges,
            std::span<const NMIncidenceRangeGPU>(world.contact.rigidRanges)) &&
        writeSection(stream, Section::adaptive,
            std::span<const NMAdaptiveStateGPU>(world.adaptive)) &&
        writeSection(stream, Section::schedulers,
            std::span<const NMSchedulerStateGPU>(world.schedulers)) &&
        writeSection(stream, Section::identification,
            std::span<const NMIdentificationDistributionGPU>(world.identification)) &&
        writeSection(stream, Section::learnedMaterials,
            std::span<const NMLearnedMaterialGPU>(world.learnedMaterials)) &&
        writeSection(stream, Section::learnedLayers,
            std::span<const NMLearnedLayerGPU>(world.learnedLayers)) &&
        writeSection(stream, Section::learnedWeights,
            std::span<const float>(world.learnedWeights)) &&
        writeStringSection(stream, Section::generatedMetal, compiled.generatedMetal);

    if (!written || !stream.flush()) {
        if (error != nullptr) {
            *error = "failed while writing matter package sections";
        }
        return false;
    }
    return true;
}

bool readPackage(
    const std::filesystem::path& path,
    CompiledWorld& world,
    std::string* generatedMetal,
    std::string* error
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        if (error != nullptr) {
            *error = "cannot open matter package: " + path.string();
        }
        return false;
    }
    PackageHeader header{};
    if (!readRaw(stream, header)) {
        if (error != nullptr) *error = "invalid matter package header";
        return false;
    }
    if (header.magic == kMagic &&
        (header.version != kPackageVersion ||
         header.matterAbiVersion != NM_MATTER_ABI_VERSION)) {
        if (error != nullptr)
            *error = "matter package ABI mismatch: recook required";
        return false;
    }
    if (header.magic != kMagic ||
        header.endian != kEndianMarker ||
        header.headerHash != headerHash(header) ||
        header.sectionCount != kSectionCount) {
        if (error != nullptr) {
            *error = "invalid or unsupported matter package header";
        }
        return false;
    }

    CompiledWorld candidate;
    std::string generatedCandidate;
    std::set<std::uint32_t> seen;
    for (std::uint32_t sectionIndex = 0u;
         sectionIndex < header.sectionCount;
         ++sectionIndex) {
        SectionHeader section{};
        if (!readRaw(stream, section) || !seen.insert(section.id).second) {
            if (error != nullptr) {
                *error = "matter package contains a missing or duplicate section header";
            }
            return false;
        }
        const Section id = static_cast<Section>(section.id);
        bool decoded = false;
        switch (id) {
        case Section::dispatch: {
            std::vector<NMMatterDispatchGPU> values;
            decoded = decodeVector(stream, section, values, error) && values.size() == 1u;
            if (decoded) candidate.dispatch = values.front();
            break;
        }
        case Section::mixedSolver: {
            std::vector<NMMixedSolverGPU> values;
            decoded = decodeVector(stream, section, values, error) && values.size() == 1u;
            if (decoded) candidate.mixedSolver = values.front();
            break;
        }
        case Section::materials:
            decoded = decodeVector(stream, section, candidate.materials, error); break;
        case Section::mixedMaterials:
            decoded = decodeVector(stream, section, candidate.mixedMaterials, error); break;
        case Section::parameters:
            decoded = decodeVector(stream, section, candidate.parameters, error); break;
        case Section::stateInitials:
            decoded = decodeVector(stream, section, candidate.stateInitials, error); break;
        case Section::instructions:
            decoded = decodeVector(stream, section, candidate.instructions, error); break;
        case Section::scalarPrograms:
            decoded = decodeVector(stream, section, candidate.scalarPrograms, error); break;
        case Section::objects:
            decoded = decodeVector(stream, section, candidate.objects, error); break;
        case Section::mpmParticles:
            decoded = decodeVector(stream, section, candidate.mpm.particles, error); break;
        case Section::mpmNodes:
            decoded = decodeVector(stream, section, candidate.mpm.nodes, error); break;
        case Section::mpmGrids:
            decoded = decodeVector(stream, section, candidate.mpm.grids, error); break;
        case Section::mpmBlocks:
            decoded = decodeVector(stream, section, candidate.mpm.blocks, error); break;
        case Section::mpmBlockLookup:
            decoded = decodeVector(stream, section, candidate.mpm.blockLookup, error); break;
        case Section::mpmStencils:
            decoded = decodeVector(stream, section, candidate.mpm.stencils, error); break;
        case Section::mpmNodeIncidence:
            decoded = decodeVector(stream, section, candidate.mpm.nodeIncidence, error); break;
        case Section::mpmNodeRanges:
            decoded = decodeVector(stream, section, candidate.mpm.nodeRanges, error); break;
        case Section::femNodes:
            decoded = decodeVector(stream, section, candidate.fem.nodes, error); break;
        case Section::femTetrahedra:
            decoded = decodeVector(stream, section, candidate.fem.tetrahedra, error); break;
        case Section::femSurfaceFaces:
            decoded = decodeVector(stream, section, candidate.fem.surfaceFaces, error); break;
        case Section::femNodeIncidence:
            decoded = decodeVector(stream, section, candidate.fem.nodeIncidence, error); break;
        case Section::femNodeRanges:
            decoded = decodeVector(stream, section, candidate.fem.nodeRanges, error); break;
        case Section::femCapacities:
            decoded = decodeVector(stream, section, candidate.fem.capacities, error); break;
        case Section::femFields:
            decoded = decodeVector(stream, section, candidate.fem.fields, error); break;
        case Section::femFieldBoundaries:
            decoded = decodeVector(stream, section, candidate.fem.fieldBoundaries, error); break;
        case Section::femTopologyNodes:
            decoded = decodeVector(stream, section, candidate.fem.topologyNodes, error); break;
        case Section::femCohesiveFaces:
            decoded = decodeVector(stream, section, candidate.fem.cohesiveFaces, error); break;
        case Section::femMutationCommands:
            decoded = decodeVector(stream, section, candidate.fem.mutationCommands, error); break;
        case Section::femPunctureChannels:
            decoded = decodeVector(stream, section, candidate.fem.punctureChannels, error); break;
        case Section::rigidProxies:
            decoded = decodeVector(stream, section, candidate.contact.rigidProxies, error); break;
        case Section::contactPairs:
            decoded = decodeVector(stream, section, candidate.contact.pairs, error); break;
        case Section::contactNodeIncidence:
            decoded = decodeVector(stream, section, candidate.contact.nodeIncidence, error); break;
        case Section::contactNodeRanges:
            decoded = decodeVector(stream, section, candidate.contact.nodeRanges, error); break;
        case Section::rigidIncidence:
            decoded = decodeVector(stream, section, candidate.contact.rigidIncidence, error); break;
        case Section::rigidRanges:
            decoded = decodeVector(stream, section, candidate.contact.rigidRanges, error); break;
        case Section::adaptive:
            decoded = decodeVector(stream, section, candidate.adaptive, error); break;
        case Section::schedulers:
            decoded = decodeVector(stream, section, candidate.schedulers, error); break;
        case Section::identification:
            decoded = decodeVector(stream, section, candidate.identification, error); break;
        case Section::learnedMaterials:
            decoded = decodeVector(stream, section, candidate.learnedMaterials, error); break;
        case Section::learnedLayers:
            decoded = decodeVector(stream, section, candidate.learnedLayers, error); break;
        case Section::learnedWeights:
            decoded = decodeVector(stream, section, candidate.learnedWeights, error); break;
        case Section::generatedMetal: {
            if (section.elementSize != 1u ||
                section.elementCount != section.byteCount ||
                section.byteCount > std::numeric_limits<std::size_t>::max()) {
                if (error != nullptr) {
                    *error = "generated Metal section has an invalid layout";
                }
                return false;
            }
            generatedCandidate.resize(static_cast<std::size_t>(section.byteCount));
            decoded = readBytes(stream, generatedCandidate.data(), generatedCandidate.size()) &&
                detail::hashString(generatedCandidate) == section.contentHash;
            if (!decoded && error != nullptr) {
                *error = "generated Metal section is truncated or corrupted";
            }
            break;
        }
        default:
            decoded = skipSection(stream, section, error);
            break;
        }
        if (!decoded) {
            if (error != nullptr && error->empty()) {
                *error = "failed to decode matter package section " +
                    std::to_string(section.id);
            }
            return false;
        }
    }

    for (std::uint32_t id = 1u; id <= kSectionCount; ++id) {
        if (!seen.contains(id)) {
            if (error != nullptr) {
                *error = "matter package is missing required section " +
                    std::to_string(id);
            }
            return false;
        }
    }

    candidate.physicsFingerprint = header.physicsFingerprint;
    candidate.fingerprint = header.fingerprint;
    std::string layoutError;
    if (!validateCompiledWorldLayout(candidate, &layoutError)) {
        if (error != nullptr) {
            *error = "matter package contains an invalid execution layout: " +
                layoutError;
        }
        return false;
    }
    world = std::move(candidate);
    if (generatedMetal != nullptr) {
        *generatedMetal = std::move(generatedCandidate);
    }
    return true;
}

bool writeLearnedMaterial(
    const LearnedMaterialSource& material,
    const std::filesystem::path& path,
    std::string* error
) {
    struct Header {
        std::array<char, 16> magic{};
        std::uint32_t version = 1u;
        std::uint32_t endian = kEndianMarker;
        std::uint32_t invariantCount = 0u;
        std::uint32_t layerCount = 0u;
        float softplusBeta = 0.0f;
        float determinantFloor = 0.0f;
        float growthCoefficient = 0.0f;
        std::uint32_t reserved = 0u;
        std::uint64_t floatCount = 0u;
        std::uint64_t contentHash = 0u;
        std::uint64_t fingerprint = 0u;
        std::uint64_t headerHash = 0u;
    };
    struct Layer {
        std::uint32_t inputWidth = 0u;
        std::uint32_t outputWidth = 0u;
        std::uint32_t inputCount = 0u;
        std::uint32_t recurrentCount = 0u;
        std::uint32_t biasCount = 0u;
        std::uint32_t reserved0 = 0u;
        std::uint32_t reserved1 = 0u;
        std::uint32_t reserved2 = 0u;
    };
    constexpr std::array<char, 16> magic{
        'N','U','M','I','P','O','L','Y','I','C','N','N','\0','\0','\0','\0'
    };
    std::vector<Layer> layers;
    std::vector<float> values;
    layers.reserve(material.layers.size());
    for (const LearnedLayerSource& source : material.layers) {
        if (source.inputWeights.size() > std::numeric_limits<std::uint32_t>::max() ||
            source.recurrentWeights.size() > std::numeric_limits<std::uint32_t>::max() ||
            source.biases.size() > std::numeric_limits<std::uint32_t>::max()) {
            if (error != nullptr) *error = "learned layer exceeds 32-bit capacity";
            return false;
        }
        layers.push_back({
            source.inputWidth,
            source.outputWidth,
            static_cast<std::uint32_t>(source.inputWeights.size()),
            static_cast<std::uint32_t>(source.recurrentWeights.size()),
            static_cast<std::uint32_t>(source.biases.size()),
            0u, 0u, 0u,
        });
        values.insert(values.end(), source.inputWeights.begin(), source.inputWeights.end());
        values.insert(values.end(), source.recurrentWeights.begin(), source.recurrentWeights.end());
        values.insert(values.end(), source.biases.begin(), source.biases.end());
    }
    Header header{};
    header.magic = magic;
    header.invariantCount = material.invariantCount;
    header.layerCount = static_cast<std::uint32_t>(layers.size());
    header.softplusBeta = material.softplusBeta;
    header.determinantFloor = material.determinantFloor;
    header.growthCoefficient = material.growthCoefficient;
    header.floatCount = values.size();
    std::uint64_t contentHash = detail::hashBytes(
        layers.data(), layers.size() * sizeof(Layer)
    );
    contentHash = detail::hashBytes(
        values.data(), values.size() * sizeof(float), contentHash
    );
    header.contentHash = contentHash;
    header.fingerprint = material.fingerprint == 0u ? contentHash : material.fingerprint;
    Header hashable = header;
    hashable.headerHash = 0u;
    header.headerHash = detail::hashBytes(&hashable, sizeof(hashable));
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !writeRaw(stream, header) ||
        !writeBytes(stream, layers.data(), layers.size() * sizeof(Layer)) ||
        !writeBytes(stream, values.data(), values.size() * sizeof(float))) {
        if (error != nullptr) *error = "failed to write learned material";
        return false;
    }
    return true;
}

bool readLearnedMaterial(
    const std::filesystem::path& path,
    LearnedMaterialSource& material,
    std::string* error
) {
    struct Header {
        std::array<char, 16> magic{};
        std::uint32_t version = 0u;
        std::uint32_t endian = 0u;
        std::uint32_t invariantCount = 0u;
        std::uint32_t layerCount = 0u;
        float softplusBeta = 0.0f;
        float determinantFloor = 0.0f;
        float growthCoefficient = 0.0f;
        std::uint32_t reserved = 0u;
        std::uint64_t floatCount = 0u;
        std::uint64_t contentHash = 0u;
        std::uint64_t fingerprint = 0u;
        std::uint64_t headerHash = 0u;
    };
    struct Layer {
        std::uint32_t inputWidth = 0u;
        std::uint32_t outputWidth = 0u;
        std::uint32_t inputCount = 0u;
        std::uint32_t recurrentCount = 0u;
        std::uint32_t biasCount = 0u;
        std::uint32_t reserved0 = 0u;
        std::uint32_t reserved1 = 0u;
        std::uint32_t reserved2 = 0u;
    };
    constexpr std::array<char, 16> magic{
        'N','U','M','I','P','O','L','Y','I','C','N','N','\0','\0','\0','\0'
    };
    std::ifstream stream(path, std::ios::binary);
    Header header{};
    if (!stream || !readRaw(stream, header)) {
        if (error != nullptr) *error = "cannot read learned material header";
        return false;
    }
    Header hashable = header;
    hashable.headerHash = 0u;
    if (header.magic != magic || header.version != 1u ||
        header.endian != kEndianMarker ||
        header.headerHash != detail::hashBytes(&hashable, sizeof(hashable)) ||
        header.layerCount == 0u || header.invariantCount < 4u ||
        header.invariantCount > 8u || header.floatCount >
            std::numeric_limits<std::size_t>::max() ||
        !std::isfinite(header.softplusBeta) || header.softplusBeta <= 0.0f ||
        !std::isfinite(header.determinantFloor) ||
        header.determinantFloor <= 0.0f ||
        !std::isfinite(header.growthCoefficient) ||
        header.growthCoefficient < 0.0f) {
        if (error != nullptr) *error = "invalid learned material header";
        return false;
    }
    std::vector<Layer> layers(header.layerCount);
    std::vector<float> values(static_cast<std::size_t>(header.floatCount));
    if (!readBytes(stream, layers.data(), layers.size() * sizeof(Layer)) ||
        !readBytes(stream, values.data(), values.size() * sizeof(float))) {
        if (error != nullptr) *error = "learned material is truncated";
        return false;
    }
    std::uint64_t contentHash = detail::hashBytes(
        layers.data(), layers.size() * sizeof(Layer)
    );
    contentHash = detail::hashBytes(
        values.data(), values.size() * sizeof(float), contentHash
    );
    if (contentHash != header.contentHash) {
        if (error != nullptr) *error = "learned material content hash mismatch";
        return false;
    }
    LearnedMaterialSource candidate;
    candidate.invariantCount = header.invariantCount;
    candidate.softplusBeta = header.softplusBeta;
    candidate.determinantFloor = header.determinantFloor;
    candidate.growthCoefficient = header.growthCoefficient;
    candidate.fingerprint = header.fingerprint;
    std::size_t cursor = 0u;
    std::uint32_t previousWidth = 0u;
    for (const Layer layer : layers) {
        const std::uint64_t expectedInput =
            static_cast<std::uint64_t>(layer.inputWidth) * layer.outputWidth;
        const std::uint64_t expectedRecurrent =
            static_cast<std::uint64_t>(previousWidth) * layer.outputWidth;
        const std::uint64_t count = static_cast<std::uint64_t>(layer.inputCount) +
            layer.recurrentCount + layer.biasCount;
        if (layer.inputWidth != header.invariantCount || layer.outputWidth == 0u ||
            layer.inputCount != expectedInput ||
            layer.recurrentCount != expectedRecurrent ||
            layer.biasCount != layer.outputWidth || cursor > values.size() ||
            count > values.size() - cursor) {
            if (error != nullptr) *error = "learned material layer layout is invalid";
            return false;
        }
        LearnedLayerSource decoded;
        decoded.inputWidth = layer.inputWidth;
        decoded.outputWidth = layer.outputWidth;
        decoded.inputWeights.assign(values.begin() + cursor,
            values.begin() + cursor + layer.inputCount);
        cursor += layer.inputCount;
        decoded.recurrentWeights.assign(values.begin() + cursor,
            values.begin() + cursor + layer.recurrentCount);
        cursor += layer.recurrentCount;
        decoded.biases.assign(values.begin() + cursor,
            values.begin() + cursor + layer.biasCount);
        cursor += layer.biasCount;
        if (!std::ranges::all_of(decoded.inputWeights,
                [](const float value) {
                    return std::isfinite(value) && value >= 0.0f;
                }) ||
            !std::ranges::all_of(decoded.recurrentWeights,
                [](const float value) {
                    return std::isfinite(value) && value >= 0.0f;
                }) ||
            !std::ranges::all_of(decoded.biases,
                [](const float value) { return std::isfinite(value); })) {
            if (error != nullptr) *error =
                "learned material violates convex weight constraints";
            return false;
        }
        candidate.layers.push_back(std::move(decoded));
        previousWidth = layer.outputWidth;
    }
    if (cursor != values.size() || previousWidth != 1u) {
        if (error != nullptr) *error = "learned material has trailing values or no scalar output";
        return false;
    }
    material = std::move(candidate);
    return true;
}

} // namespace numi::matter
