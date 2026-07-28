#include "metalrobo/WorldPack.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <exception>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kEndianMarker = 0x01020304u;
constexpr std::array<char, 8> kMagic{
    'M', 'R', 'W', 'P', 'A', 'C', 'K', '1',
};

struct WorldPackFileHeader {
    std::array<char, 8> magic{};
    std::uint32_t formatVersion = 0u;
    std::uint32_t endianMarker = 0u;
    std::uint32_t compilerAbiVersion = 0u;
    std::uint32_t engineAbiVersion = 0u;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t contentHash = 0u;
    std::uint64_t familyFingerprint = 0u;
    std::uint64_t templateFingerprint = 0u;
    std::uint64_t programFingerprint = 0u;
    std::array<std::uint64_t, 3> reserved{};
};

static_assert(
    std::is_trivially_copyable_v<WorldPackFileHeader>
);
static_assert(sizeof(WorldPackFileHeader) == 88u);

WorldPackResult fail(
    const WorldPackStatus status,
    std::string message
) {
    return {status, std::move(message)};
}

bool setReason(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

std::uint64_t hashBytes(const std::span<const std::byte> bytes) {
    std::uint64_t hash = kFnvOffset;
    for (const std::byte value : bytes) {
        hash ^= std::to_integer<unsigned char>(value);
        hash *= kFnvPrime;
    }
    return hash;
}

class PayloadWriter {
public:
    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* begin =
            reinterpret_cast<const std::byte*>(&value);
        bytes_.insert(bytes_.end(), begin, begin + sizeof(T));
    }

    void string(const std::string& value) {
        const std::uint64_t size = value.size();
        pod(size);
        const auto* begin =
            reinterpret_cast<const std::byte*>(value.data());
        bytes_.insert(bytes_.end(), begin, begin + value.size());
    }

    template <typename T>
    void podVector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint64_t count = values.size();
        pod(count);
        if (!values.empty()) {
            const auto* begin =
                reinterpret_cast<const std::byte*>(values.data());
            bytes_.insert(
                bytes_.end(),
                begin,
                begin + values.size() * sizeof(T)
            );
        }
    }

    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept {
        return bytes_;
    }

private:
    std::vector<std::byte> bytes_;
};

class PayloadReader {
public:
    explicit PayloadReader(const std::span<const std::byte> bytes)
        : bytes_(bytes) {}

    template <typename T>
    bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (remaining() < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t size = 0u;
        if (!pod(size) ||
            size > remaining() ||
            size > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(
                bytes_.data() + offset_
            ),
            static_cast<std::size_t>(size)
        );
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    template <typename T>
    bool podVector(std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > std::numeric_limits<std::size_t>::max() ||
            count > remaining() / sizeof(T)) {
            return false;
        }
        const std::size_t size = static_cast<std::size_t>(count);
        std::vector<T> staged(size);
        if (size != 0u) {
            std::memcpy(
                staged.data(),
                bytes_.data() + offset_,
                size * sizeof(T)
            );
            offset_ += size * sizeof(T);
        }
        values = std::move(staged);
        return true;
    }

    [[nodiscard]] bool consumed() const noexcept {
        return offset_ == bytes_.size();
    }

private:
    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - offset_;
    }

    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0u;
};

void writePose(PayloadWriter& writer, const WorldPose& pose) {
    writer.pod(pose.position);
    writer.pod(pose.orientation);
}

bool readPose(PayloadReader& reader, WorldPose& pose) {
    return reader.pod(pose.position) &&
        reader.pod(pose.orientation);
}

void writeAsset(PayloadWriter& writer, const WorldAsset& asset) {
    writer.string(asset.id);
    writer.string(asset.semanticClass);
    writer.pod(asset.role);
    writer.pod(asset.render);
    writer.pod(asset.collision);
    writer.pod(asset.dynamics);
    writePose(writer, asset.initialPose);
    writer.pod(asset.uniformScale);
    writer.pod(asset.massScale);
    writer.pod(asset.frictionScale);
    writer.pod(asset.restitutionScale);
    writer.pod(asset.dampingScale);
    writer.pod(asset.controllerGainScale);
    writer.pod(asset.controllerDampingScale);
    writer.pod(asset.controllerLatencySeconds);
    writer.pod(asset.payloadScale);
    writer.pod(asset.renderAlternative);
    writer.pod(asset.collisionAlternative);
    writer.pod(asset.articulationIndex);
    writer.string(asset.topologyCohort);
    writer.podVector(asset.bodyIndices);
    writer.podVector(asset.shapeIndices);
    writer.podVector(asset.materialIndices);
    const std::uint64_t anchorCount = asset.anchors.size();
    writer.pod(anchorCount);
    for (const SemanticAnchor& anchor : asset.anchors) {
        writer.string(anchor.id);
        writePose(writer, anchor.localPose);
        writer.pod(anchor.radius);
        writer.pod(anchor.flags);
    }
}

bool readAsset(PayloadReader& reader, WorldAsset& asset) {
    if (!reader.string(asset.id) ||
        !reader.string(asset.semanticClass) ||
        !reader.pod(asset.role) ||
        !reader.pod(asset.render) ||
        !reader.pod(asset.collision) ||
        !reader.pod(asset.dynamics) ||
        !readPose(reader, asset.initialPose) ||
        !reader.pod(asset.uniformScale) ||
        !reader.pod(asset.massScale) ||
        !reader.pod(asset.frictionScale) ||
        !reader.pod(asset.restitutionScale) ||
        !reader.pod(asset.dampingScale) ||
        !reader.pod(asset.controllerGainScale) ||
        !reader.pod(asset.controllerDampingScale) ||
        !reader.pod(asset.controllerLatencySeconds) ||
        !reader.pod(asset.payloadScale) ||
        !reader.pod(asset.renderAlternative) ||
        !reader.pod(asset.collisionAlternative) ||
        !reader.pod(asset.articulationIndex) ||
        !reader.string(asset.topologyCohort) ||
        !reader.podVector(asset.bodyIndices) ||
        !reader.podVector(asset.shapeIndices) ||
        !reader.podVector(asset.materialIndices)) {
        return false;
    }
    std::uint64_t anchorCount = 0u;
    if (!reader.pod(anchorCount) ||
        anchorCount > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    std::vector<SemanticAnchor> anchors(
        static_cast<std::size_t>(anchorCount)
    );
    for (SemanticAnchor& anchor : anchors) {
        if (!reader.string(anchor.id) ||
            !readPose(reader, anchor.localPose) ||
            !reader.pod(anchor.radius) ||
            !reader.pod(anchor.flags)) {
            return false;
        }
    }
    asset.anchors = std::move(anchors);
    return true;
}

void writeSensor(PayloadWriter& writer, const SensorSpec& sensor) {
    writer.string(sensor.id);
    writer.string(sensor.parentAssetId);
    writer.pod(sensor.kind);
    writePose(writer, sensor.localPose);
    writer.pod(sensor.width);
    writer.pod(sensor.height);
    writer.pod(sensor.intrinsics);
    writer.pod(sensor.distortion);
    writer.pod(sensor.focalScale);
    writer.pod(sensor.colorNoiseSigma);
    writer.pod(sensor.depthNoiseSigma);
    writer.pod(sensor.depthDropout);
    writer.pod(sensor.latencySeconds);
}

bool readSensor(PayloadReader& reader, SensorSpec& sensor) {
    return reader.string(sensor.id) &&
        reader.string(sensor.parentAssetId) &&
        reader.pod(sensor.kind) &&
        readPose(reader, sensor.localPose) &&
        reader.pod(sensor.width) &&
        reader.pod(sensor.height) &&
        reader.pod(sensor.intrinsics) &&
        reader.pod(sensor.distortion) &&
        reader.pod(sensor.focalScale) &&
        reader.pod(sensor.colorNoiseSigma) &&
        reader.pod(sensor.depthNoiseSigma) &&
        reader.pod(sensor.depthDropout) &&
        reader.pod(sensor.latencySeconds);
}

void writeAppearance(
    PayloadWriter& writer,
    const AppearanceSpec& appearance
) {
    writer.string(appearance.id);
    writer.pod(appearance.exposureStops);
    writer.pod(appearance.whiteBalanceScale);
    writer.pod(appearance.saturation);
    writer.pod(appearance.lightIntensity);
    writer.pod(appearance.hueRadians);
    writer.pod(appearance.contrast);
    writer.pod(appearance.roughnessScale);
    writer.pod(appearance.metallicScale);
    writer.pod(appearance.alternative);
    writer.pod(appearance.environmentMap);
}

bool readAppearance(
    PayloadReader& reader,
    AppearanceSpec& appearance
) {
    return reader.string(appearance.id) &&
        reader.pod(appearance.exposureStops) &&
        reader.pod(appearance.whiteBalanceScale) &&
        reader.pod(appearance.saturation) &&
        reader.pod(appearance.lightIntensity) &&
        reader.pod(appearance.hueRadians) &&
        reader.pod(appearance.contrast) &&
        reader.pod(appearance.roughnessScale) &&
        reader.pod(appearance.metallicScale) &&
        reader.pod(appearance.alternative) &&
        reader.pod(appearance.environmentMap);
}

void writeArtifact(
    PayloadWriter& writer,
    const EpisodeArtifact& artifact
) {
    writer.string(artifact.id);
    writer.pod(artifact.kind);
    writer.pod(artifact.producer);
    writer.string(artifact.assetId);
    writer.string(artifact.uri);
    writer.string(artifact.contentHash);
    writer.pod(artifact.startTimeSeconds);
    writer.pod(artifact.endTimeSeconds);
}

bool readArtifact(
    PayloadReader& reader,
    EpisodeArtifact& artifact
) {
    return reader.string(artifact.id) &&
        reader.pod(artifact.kind) &&
        reader.pod(artifact.producer) &&
        reader.string(artifact.assetId) &&
        reader.string(artifact.uri) &&
        reader.string(artifact.contentHash) &&
        reader.pod(artifact.startTimeSeconds) &&
        reader.pod(artifact.endTimeSeconds);
}

void writeTask(PayloadWriter& writer, const TaskSpec& task) {
    writer.string(task.id);
    writer.string(task.robotAssetId);
    writer.string(task.manipulatedAssetId);
    writer.string(task.targetAssetId);
    writer.string(task.targetAnchorId);
    writer.pod(task.controlPeriodSeconds);
    writer.pod(task.horizonSeconds);
}

bool readTask(PayloadReader& reader, TaskSpec& task) {
    return reader.string(task.id) &&
        reader.string(task.robotAssetId) &&
        reader.string(task.manipulatedAssetId) &&
        reader.string(task.targetAssetId) &&
        reader.string(task.targetAnchorId) &&
        reader.pod(task.controlPeriodSeconds) &&
        reader.pod(task.horizonSeconds);
}

void writeEngineModel(
    PayloadWriter& writer,
    const EngineModel& model
) {
    writer.pod(model.world);
    writer.podVector(model.articulations);
    writer.podVector(model.joints);
    writer.podVector(model.dofs);
    writer.podVector(model.bodies);
    writer.podVector(model.shapes);
    writer.podVector(model.materials);
    writer.podVector(model.geometryHeaders);
    writer.podVector(model.geometryVertices);
    writer.podVector(model.geometryIndices);
    writer.podVector(model.convexFaces);
    writer.podVector(model.convexHalfEdges);
    writer.podVector(model.meshBvhNodes);
    writer.podVector(model.meshTriangles);
    writer.podVector(model.collisionExclusions);
    writer.pod(model.constraintProgram.abiVersion);
    writer.podVector(model.constraintProgram.blocks);
    writer.podVector(model.constraintProgram.endpoints);
    writer.podVector(model.constraintProgram.rows);
    writer.podVector(model.constraintProgram.cones);
    writer.podVector(model.constraintProgram.warmImpulses);
    writer.podVector(model.defaultQ);
    writer.podVector(model.defaultV);
    writer.string(model.name);
}

bool readEngineModel(
    PayloadReader& reader,
    EngineModel& model
) {
    return reader.pod(model.world) &&
        reader.podVector(model.articulations) &&
        reader.podVector(model.joints) &&
        reader.podVector(model.dofs) &&
        reader.podVector(model.bodies) &&
        reader.podVector(model.shapes) &&
        reader.podVector(model.materials) &&
        reader.podVector(model.geometryHeaders) &&
        reader.podVector(model.geometryVertices) &&
        reader.podVector(model.geometryIndices) &&
        reader.podVector(model.convexFaces) &&
        reader.podVector(model.convexHalfEdges) &&
        reader.podVector(model.meshBvhNodes) &&
        reader.podVector(model.meshTriangles) &&
        reader.podVector(model.collisionExclusions) &&
        reader.pod(model.constraintProgram.abiVersion) &&
        reader.podVector(model.constraintProgram.blocks) &&
        reader.podVector(model.constraintProgram.endpoints) &&
        reader.podVector(model.constraintProgram.rows) &&
        reader.podVector(model.constraintProgram.cones) &&
        reader.podVector(model.constraintProgram.warmImpulses) &&
        reader.podVector(model.defaultQ) &&
        reader.podVector(model.defaultV) &&
        reader.string(model.name);
}

template <typename T, typename WriteElement>
void writeRichVector(
    PayloadWriter& writer,
    const std::vector<T>& values,
    WriteElement&& writeElement
) {
    const std::uint64_t count = values.size();
    writer.pod(count);
    for (const T& value : values) {
        writeElement(writer, value);
    }
}

template <typename T, typename ReadElement>
bool readRichVector(
    PayloadReader& reader,
    std::vector<T>& values,
    ReadElement&& readElement
) {
    std::uint64_t count = 0u;
    if (!reader.pod(count) ||
        count > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    std::vector<T> staged(static_cast<std::size_t>(count));
    for (T& value : staged) {
        if (!readElement(reader, value)) {
            return false;
        }
    }
    values = std::move(staged);
    return true;
}

void writeStringVector(
    PayloadWriter& writer,
    const std::vector<std::string>& values
) {
    writeRichVector(
        writer,
        values,
        [](PayloadWriter& output, const std::string& value) {
            output.string(value);
        }
    );
}

bool readStringVector(
    PayloadReader& reader,
    std::vector<std::string>& values
) {
    return readRichVector(
        reader,
        values,
        [](PayloadReader& input, std::string& value) {
            return input.string(value);
        }
    );
}

std::vector<std::byte> serializeFamily(const WorldFamily& family) {
    PayloadWriter writer;
    writer.pod(family.fingerprint);

    const WorldTemplate& world = family.worldTemplate;
    writer.string(world.id);
    writer.pod(world.fingerprint);
    writer.pod(world.capabilities);
    writeEngineModel(writer, world.engineModel);
    writeRichVector(writer, world.assets, writeAsset);
    writeRichVector(writer, world.sensors, writeSensor);
    writeRichVector(writer, world.appearances, writeAppearance);
    writer.podVector(world.assetBindings);
    writer.podVector(world.bindingIndices);
    writeRichVector(writer, world.artifacts, writeArtifact);
    writeTask(writer, world.task);
    writeStringVector(writer, world.topologyCohorts);

    writer.string(family.program.id);
    writer.pod(family.program.fingerprint);
    writer.pod(family.program.instanceFlags);
    writer.podVector(family.program.variations);
    writer.podVector(family.program.categoricalValues);
    return writer.bytes();
}

bool deserializeFamily(
    const std::span<const std::byte> payload,
    WorldFamily& family
) {
    PayloadReader reader{payload};
    WorldFamily staged;
    WorldTemplate& world = staged.worldTemplate;
    if (!reader.pod(staged.fingerprint) ||
        !reader.string(world.id) ||
        !reader.pod(world.fingerprint) ||
        !reader.pod(world.capabilities) ||
        !readEngineModel(reader, world.engineModel) ||
        !readRichVector(reader, world.assets, readAsset) ||
        !readRichVector(reader, world.sensors, readSensor) ||
        !readRichVector(
            reader,
            world.appearances,
            readAppearance
        ) ||
        !reader.podVector(world.assetBindings) ||
        !reader.podVector(world.bindingIndices) ||
        !readRichVector(reader, world.artifacts, readArtifact) ||
        !readTask(reader, world.task) ||
        !readStringVector(reader, world.topologyCohorts) ||
        !reader.string(staged.program.id) ||
        !reader.pod(staged.program.fingerprint) ||
        !reader.pod(staged.program.instanceFlags) ||
        !reader.podVector(staged.program.variations) ||
        !reader.podVector(staged.program.categoricalValues) ||
        !reader.consumed()) {
        return false;
    }
    family = std::move(staged);
    return true;
}

bool validCompiledProgram(
    const CompiledWorldProgram& program,
    std::string* reason
) {
    if (program.id.empty() || program.fingerprint == 0u) {
        return setReason(reason, "world pack program identity is empty");
    }
    for (const MRWorldVariationGPU& variation :
         program.variations) {
        if (variation.binding.x > MR_WORLD_VARIATION_CAMERA ||
            variation.binding.y >
                MR_WORLD_DISTRIBUTION_CATEGORICAL ||
            variation.binding.z > MR_WORLD_TARGET_CLUTTER_SET) {
            return setReason(
                reason,
                "world pack contains an unknown variation"
            );
        }
        if (variation.binding.y ==
                MR_WORLD_DISTRIBUTION_CATEGORICAL &&
            (variation.categorical.y == 0u ||
             variation.categorical.x >
                 program.categoricalValues.size() ||
             variation.categorical.y >
                 program.categoricalValues.size() -
                     variation.categorical.x)) {
            return setReason(
                reason,
                "world pack categorical range is invalid"
            );
        }
    }
    return true;
}

} // namespace

bool MRWorldPack::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (formatVersion != kWorldPackFormatVersion ||
        contentHash == 0u ||
        family.fingerprint == 0u) {
        return setReason(reason, "world pack identity is invalid");
    }
    std::string familyReason;
    if (!family.worldTemplate.valid(&familyReason)) {
        return setReason(
            reason,
            "world pack template: " + familyReason
        );
    }
    if (!validCompiledProgram(family.program, &familyReason)) {
        return setReason(reason, std::move(familyReason));
    }
    const std::vector<std::byte> payload = serializeFamily(family);
    if (hashBytes(payload) != contentHash) {
        return setReason(
            reason,
            "world pack content hash does not match its family"
        );
    }
    const WorldInstanceBatch base = family.sample(1u, 0u);
    if (!base.valid(&familyReason)) {
        return setReason(
            reason,
            "world pack sampled base: " + familyReason
        );
    }
    return true;
}

WorldPackResult compileWorldPack(
    const WorldFamily& family,
    MRWorldPack& output
) {
    std::string reason;
    if (!family.worldTemplate.valid(&reason) ||
        !validCompiledProgram(family.program, &reason) ||
        family.fingerprint == 0u) {
        return fail(
            WorldPackStatus::invalidFamily,
            reason.empty()
                ? "world family fingerprint is empty"
                : std::move(reason)
        );
    }
    try {
        MRWorldPack candidate;
        candidate.family = family;
        const std::vector<std::byte> payload =
            serializeFamily(candidate.family);
        candidate.contentHash = hashBytes(payload);
        if (!candidate.valid(&reason)) {
            return fail(
                WorldPackStatus::invalidPack,
                std::move(reason)
            );
        }
        output = std::move(candidate);
        return {};
    } catch (const std::bad_alloc&) {
        return fail(
            WorldPackStatus::capacityOverflow,
            "host allocation failed while compiling world pack"
        );
    } catch (const std::exception& exception) {
        return fail(
            WorldPackStatus::internalFailure,
            exception.what()
        );
    }
}

WorldPackResult writeWorldPack(
    const MRWorldPack& pack,
    const std::filesystem::path& path
) {
    std::string reason;
    if (path.empty() || !pack.valid(&reason)) {
        return fail(
            WorldPackStatus::invalidPack,
            path.empty() ? "world pack path is empty" : std::move(reason)
        );
    }
    try {
        const std::vector<std::byte> payload =
            serializeFamily(pack.family);
        WorldPackFileHeader header;
        header.magic = kMagic;
        header.formatVersion = pack.formatVersion;
        header.endianMarker = kEndianMarker;
        header.compilerAbiVersion = MR_WORLD_COMPILER_ABI_VERSION;
        header.engineAbiVersion = MR_ENGINE_ABI_VERSION;
        header.payloadBytes = payload.size();
        header.contentHash = pack.contentHash;
        header.familyFingerprint = pack.family.fingerprint;
        header.templateFingerprint =
            pack.family.worldTemplate.fingerprint;
        header.programFingerprint =
            pack.family.program.fingerprint;

        std::filesystem::path temporary = path;
        temporary += "." + std::to_string(pack.contentHash) + ".tmp";
        {
            std::ofstream stream(
                temporary,
                std::ios::binary | std::ios::trunc
            );
            if (!stream) {
                return fail(
                    WorldPackStatus::ioFailure,
                    "could not open temporary world pack for writing"
                );
            }
            stream.write(
                reinterpret_cast<const char*>(&header),
                sizeof(header)
            );
            if (!payload.empty()) {
                stream.write(
                    reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size())
                );
            }
            stream.flush();
            if (!stream) {
                stream.close();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                return fail(
                    WorldPackStatus::ioFailure,
                    "failed while writing world pack"
                );
            }
        }
        if (std::rename(
                temporary.c_str(),
                path.c_str()
            ) != 0) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                WorldPackStatus::ioFailure,
                "could not publish world pack"
            );
        }
        return {};
    } catch (const std::bad_alloc&) {
        return fail(
            WorldPackStatus::capacityOverflow,
            "host allocation failed while writing world pack"
        );
    } catch (const std::exception& exception) {
        return fail(WorldPackStatus::ioFailure, exception.what());
    }
}

WorldPackResult readWorldPack(
    const std::filesystem::path& path,
    MRWorldPack& output
) {
    if (path.empty()) {
        return fail(
            WorldPackStatus::ioFailure,
            "world pack path is empty"
        );
    }
    try {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            return fail(
                WorldPackStatus::ioFailure,
                "could not open world pack"
            );
        }
        const std::streamoff fileSize = stream.tellg();
        if (fileSize < static_cast<std::streamoff>(
                sizeof(WorldPackFileHeader)
            )) {
            return fail(
                WorldPackStatus::corruptPayload,
                "world pack is shorter than its header"
            );
        }
        stream.seekg(0);
        WorldPackFileHeader header;
        stream.read(
            reinterpret_cast<char*>(&header),
            sizeof(header)
        );
        if (!stream || header.magic != kMagic) {
            return fail(
                WorldPackStatus::corruptPayload,
                "world pack magic is invalid"
            );
        }
        if (header.formatVersion != kWorldPackFormatVersion ||
            header.compilerAbiVersion !=
                MR_WORLD_COMPILER_ABI_VERSION ||
            header.engineAbiVersion != MR_ENGINE_ABI_VERSION ||
            header.endianMarker != kEndianMarker) {
            return fail(
                WorldPackStatus::unsupportedVersion,
                "world pack format or ABI is unsupported"
            );
        }
        const std::uint64_t available =
            static_cast<std::uint64_t>(fileSize) - sizeof(header);
        if (header.payloadBytes != available ||
            header.payloadBytes >
                std::numeric_limits<std::size_t>::max() ||
            header.payloadBytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamsize>::max()
                )) {
            return fail(
                WorldPackStatus::corruptPayload,
                "world pack payload length is invalid"
            );
        }
        std::vector<std::byte> payload(
            static_cast<std::size_t>(header.payloadBytes)
        );
        if (!payload.empty()) {
            stream.read(
                reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())
            );
        }
        if (!stream || hashBytes(payload) != header.contentHash) {
            return fail(
                WorldPackStatus::corruptPayload,
                "world pack payload hash is invalid"
            );
        }

        MRWorldPack candidate;
        candidate.formatVersion = header.formatVersion;
        candidate.contentHash = header.contentHash;
        if (!deserializeFamily(payload, candidate.family) ||
            candidate.family.fingerprint !=
                header.familyFingerprint ||
            candidate.family.worldTemplate.fingerprint !=
                header.templateFingerprint ||
            candidate.family.program.fingerprint !=
                header.programFingerprint) {
            return fail(
                WorldPackStatus::corruptPayload,
                "world pack family payload is inconsistent"
            );
        }
        std::string reason;
        if (!candidate.valid(&reason)) {
            return fail(
                WorldPackStatus::invalidPack,
                std::move(reason)
            );
        }
        output = std::move(candidate);
        return {};
    } catch (const std::bad_alloc&) {
        return fail(
            WorldPackStatus::capacityOverflow,
            "host allocation failed while reading world pack"
        );
    } catch (const std::exception& exception) {
        return fail(WorldPackStatus::ioFailure, exception.what());
    }
}

const char* worldPackStatusName(const WorldPackStatus status) noexcept {
    switch (status) {
    case WorldPackStatus::success:
        return "success";
    case WorldPackStatus::invalidFamily:
        return "invalid_family";
    case WorldPackStatus::invalidPack:
        return "invalid_pack";
    case WorldPackStatus::ioFailure:
        return "io_failure";
    case WorldPackStatus::unsupportedVersion:
        return "unsupported_version";
    case WorldPackStatus::corruptPayload:
        return "corrupt_payload";
    case WorldPackStatus::capacityOverflow:
        return "capacity_overflow";
    case WorldPackStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
