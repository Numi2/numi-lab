#include "metalrobo/EpisodeTwinCompiler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <system_error>
#include <type_traits>
#include <unordered_set>

namespace metalrobo {
namespace {

constexpr std::array<std::uint32_t, 64> kSHA256RoundConstants{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

class SHA256 {
public:
    void append(const void* data, const std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        byteCount_ += size;
        std::size_t offset = 0u;
        while (offset < size) {
            const std::size_t copied =
                std::min(size - offset, block_.size() - blockSize_);
            std::memcpy(block_.data() + blockSize_, bytes + offset, copied);
            blockSize_ += copied;
            offset += copied;
            if (blockSize_ == block_.size()) {
                transform(block_.data());
                blockSize_ = 0u;
            }
        }
    }

    void appendString(const std::string_view value) {
        const std::uint64_t size = value.size();
        append(&size, sizeof(size));
        append(value.data(), value.size());
    }

    template <typename Value> void appendScalar(const Value value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        append(&value, sizeof(value));
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bitCount =
            static_cast<std::uint64_t>(byteCount_) * 8u;
        const std::uint8_t delimiter = 0x80u;
        append(&delimiter, 1u);
        const std::uint8_t zero = 0u;
        while (blockSize_ != 56u) {
            append(&zero, 1u);
        }
        std::array<std::uint8_t, 8> length{};
        for (std::size_t index = 0u; index < length.size(); ++index) {
            length[length.size() - 1u - index] =
                static_cast<std::uint8_t>(bitCount >> (index * 8u));
        }
        append(length.data(), length.size());

        std::array<std::uint8_t, 32> result{};
        for (std::size_t index = 0u; index < state_.size(); ++index) {
            result[index * 4u + 0u] =
                static_cast<std::uint8_t>(state_[index] >> 24u);
            result[index * 4u + 1u] =
                static_cast<std::uint8_t>(state_[index] >> 16u);
            result[index * 4u + 2u] =
                static_cast<std::uint8_t>(state_[index] >> 8u);
            result[index * 4u + 3u] = static_cast<std::uint8_t>(state_[index]);
        }
        return result;
    }

private:
    static std::uint32_t loadBigEndian(const std::uint8_t* bytes) {
        return (static_cast<std::uint32_t>(bytes[0]) << 24u) |
               (static_cast<std::uint32_t>(bytes[1]) << 16u) |
               (static_cast<std::uint32_t>(bytes[2]) << 8u) |
               static_cast<std::uint32_t>(bytes[3]);
    }

    void transform(const std::uint8_t* bytes) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0u; index < 16u; ++index) {
            words[index] = loadBigEndian(bytes + index * 4u);
        }
        for (std::size_t index = 16u; index < words.size(); ++index) {
            const std::uint32_t a = words[index - 15u];
            const std::uint32_t b = words[index - 2u];
            const std::uint32_t s0 =
                std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3u);
            const std::uint32_t s1 =
                std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10u);
            words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0u; index < words.size(); ++index) {
            const std::uint32_t sum1 =
                std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t temporary1 =
                h + sum1 + choice + kSHA256RoundConstants[index] + words[index];
            const std::uint32_t sum0 =
                std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t blockSize_ = 0u;
    std::size_t byteCount_ = 0u;
};

std::string hexDigest(const std::array<std::uint8_t, 32>& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(64u, '0');
    for (std::size_t index = 0u; index < digest.size(); ++index) {
        result[index * 2u] = digits[digest[index] >> 4u];
        result[index * 2u + 1u] = digits[digest[index] & 0x0fu];
    }
    return result;
}

bool appendFile(SHA256& hash, const std::filesystem::path& path,
                std::uint64_t& byteCount, std::string* reason) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        if (reason != nullptr) {
            *reason = "could not open artifact source: " + path.string();
        }
        return false;
    }
    std::array<char, 1024u * 1024u> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize read = stream.gcount();
        if (read > 0) {
            hash.append(buffer.data(), static_cast<std::size_t>(read));
            byteCount += static_cast<std::uint64_t>(read);
        }
    }
    if (!stream.eof()) {
        if (reason != nullptr) {
            *reason = "could not read artifact source: " + path.string();
        }
        return false;
    }
    return true;
}

struct HashedPath {
    std::string digest;
    std::uint64_t byteCount = 0u;
    bool directory = false;
};

bool hashPath(const std::filesystem::path& source, HashedPath& output,
              std::string* reason) {
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(source, error);
    if (error) {
        if (reason != nullptr) {
            *reason = "could not inspect artifact source: " + error.message();
        }
        return false;
    }
    if (std::filesystem::is_symlink(status)) {
        if (reason != nullptr) {
            *reason = "artifact sources may not be symbolic links";
        }
        return false;
    }

    SHA256 hash;
    std::uint64_t byteCount = 0u;
    if (std::filesystem::is_regular_file(status)) {
        if (!appendFile(hash, source, byteCount, reason)) {
            return false;
        }
        output = {hexDigest(hash.finish()), byteCount, false};
        return true;
    }
    if (!std::filesystem::is_directory(status)) {
        if (reason != nullptr) {
            *reason = "artifact source is neither a file nor a directory";
        }
        return false;
    }

    const std::string domain = "MetalRoboDirectoryV1";
    hash.appendString(domain);
    std::vector<std::filesystem::path> files;
    for (std::filesystem::recursive_directory_iterator iterator(
             source, std::filesystem::directory_options::skip_permission_denied,
             error);
         !error && iterator != std::filesystem::recursive_directory_iterator{};
         iterator.increment(error)) {
        const auto entryStatus = iterator->symlink_status(error);
        if (error) {
            break;
        }
        if (std::filesystem::is_symlink(entryStatus)) {
            if (reason != nullptr) {
                *reason = "artifact directory contains a symbolic link";
            }
            return false;
        }
        if (std::filesystem::is_regular_file(entryStatus)) {
            files.push_back(iterator->path());
        }
    }
    if (error) {
        if (reason != nullptr) {
            *reason =
                "could not enumerate artifact directory: " + error.message();
        }
        return false;
    }
    std::ranges::sort(files, [&source](const auto& left, const auto& right) {
        return std::filesystem::relative(left, source).generic_string() <
               std::filesystem::relative(right, source).generic_string();
    });
    for (const std::filesystem::path& file : files) {
        const std::string relative =
            std::filesystem::relative(file, source).generic_string();
        hash.appendString(relative);
        const std::uint64_t fileSize = std::filesystem::file_size(file, error);
        if (error) {
            if (reason != nullptr) {
                *reason = "could not size artifact file: " + error.message();
            }
            return false;
        }
        hash.appendScalar(fileSize);
        if (!appendFile(hash, file, byteCount, reason)) {
            return false;
        }
    }
    output = {hexDigest(hash.finish()), byteCount, true};
    return true;
}

std::string normalizedExpectedHash(std::string value) {
    constexpr std::string_view prefix = "sha256:";
    if (value.starts_with(prefix)) {
        value.erase(0u, prefix.size());
    }
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool validHashText(const std::string& value) {
    if (value.empty()) {
        return true;
    }
    const std::string normalized = normalizedExpectedHash(value);
    return normalized.size() == 64u &&
           std::ranges::all_of(normalized, [](const unsigned char c) {
               return std::isxdigit(c) != 0;
           });
}

std::filesystem::path objectPath(const std::filesystem::path& root,
                                 const std::string& digest,
                                 const bool directory) {
    return root / "objects" / "sha256" / digest.substr(0u, 2u) /
           (digest.substr(2u) + (directory ? ".tree" : ""));
}

std::filesystem::path uniqueSibling(const std::filesystem::path& target) {
    static std::atomic_uint64_t counter{0u};
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return target.parent_path() /
           (target.filename().string() + ".tmp-" + std::to_string(tick) + "-" +
            std::to_string(counter.fetch_add(1u)));
}

EpisodeArtifactKind streamArtifactKind(const CaptureStreamKind kind) {
    switch (kind) {
    case CaptureStreamKind::cameraCalibration:
    case CaptureStreamKind::cameraPoses:
        return EpisodeArtifactKind::cameraCalibration;
    case CaptureStreamKind::robotTelemetry:
    case CaptureStreamKind::robotCommands:
    case CaptureStreamKind::gripperState:
        return EpisodeArtifactKind::robotTrajectory;
    case CaptureStreamKind::forceTorque:
        return EpisodeArtifactKind::interactionTrace;
    case CaptureStreamKind::cad:
    case CaptureStreamKind::urdf:
        return EpisodeArtifactKind::geometry;
    case CaptureStreamKind::rgb:
    case CaptureStreamKind::depth:
    case CaptureStreamKind::rgbd:
    case CaptureStreamKind::video:
        return EpisodeArtifactKind::capture;
    }
}

CaptureProduct streamProduct(const CaptureStream& stream) {
    CaptureProduct result;
    result.id = stream.id;
    result.stage = EpisodeTwinStage::ingest;
    result.kind = streamArtifactKind(stream.kind);
    result.producer = EpisodeArtifactProducer::measured;
    result.assetId = stream.assetId;
    result.source = stream.source;
    result.expectedContentHash = stream.expectedContentHash;
    result.startTimeSeconds = stream.startTimeSeconds;
    result.endTimeSeconds = stream.endTimeSeconds;
    result.payload.targetId =
        stream.sensorId.empty() ? stream.assetId : stream.sensorId;
    result.payload.calibration = stream.calibration;
    switch (stream.kind) {
    case CaptureStreamKind::rgb:
    case CaptureStreamKind::depth:
    case CaptureStreamKind::rgbd:
    case CaptureStreamKind::video:
        result.payload.kind =
            EpisodeTwinProductKind::synchronizedSensorStream;
        break;
    case CaptureStreamKind::cameraCalibration:
    case CaptureStreamKind::cameraPoses:
        result.payload.kind = EpisodeTwinProductKind::sensorCalibration;
        break;
    case CaptureStreamKind::robotTelemetry:
    case CaptureStreamKind::gripperState:
    case CaptureStreamKind::forceTorque:
        result.payload.kind = EpisodeTwinProductKind::robotStateTrace;
        break;
    case CaptureStreamKind::robotCommands:
        result.payload.kind = EpisodeTwinProductKind::robotCommandTrace;
        break;
    case CaptureStreamKind::cad:
    case CaptureStreamKind::urdf:
        result.payload.kind = EpisodeTwinProductKind::artifactOnly;
        break;
    }
    return result;
}

void appendProductPayload(SHA256& hash,
                          const EpisodeTwinProductPayload& payload) {
    hash.appendScalar(payload.kind);
    hash.appendString(payload.targetId);
    hash.appendScalar(payload.calibration.hasResolution);
    hash.appendScalar(payload.calibration.hasIntrinsics);
    hash.appendScalar(payload.calibration.hasDistortion);
    hash.appendScalar(payload.calibration.hasPose);
    hash.appendScalar(payload.calibration.width);
    hash.appendScalar(payload.calibration.height);
    hash.appendScalar(payload.calibration.intrinsics);
    hash.appendScalar(payload.calibration.distortion);
    hash.appendScalar(payload.calibration.worldFromSensor.position);
    hash.appendScalar(payload.calibration.worldFromSensor.orientation);
    hash.appendScalar(payload.hasWorldPose);
    hash.appendScalar(payload.worldPose.position);
    hash.appendScalar(payload.worldPose.orientation);
    hash.appendScalar(payload.hasPhysicalPrior);
    hash.appendScalar(payload.massScale);
    hash.appendScalar(payload.frictionScale);
    hash.appendScalar(payload.restitutionScale);
    hash.appendScalar(payload.dampingScale);
    hash.appendScalar(payload.renderRepresentation);
    hash.appendScalar(payload.collisionRepresentation);
    hash.appendScalar(payload.hasCollisionBox);
    hash.appendScalar(payload.collisionBoxHalfExtents);
}

void appendArtifactIdentity(SHA256& hash, const EpisodeArtifact& artifact) {
    hash.appendString(artifact.id);
    hash.appendScalar(artifact.kind);
    hash.appendScalar(artifact.producer);
    hash.appendString(artifact.assetId);
    hash.appendString(artifact.contentHash);
    hash.appendScalar(artifact.startTimeSeconds);
    hash.appendScalar(artifact.endTimeSeconds);
}

std::string makeStageKey(
    const EpisodeTwinStage stage, const CaptureManifest& manifest,
    const EngineModel& engineModel,
    const std::span<const EpisodeArtifact> inputs,
    const std::span<const EpisodeArtifact> declared,
    const std::span<const CaptureProduct> declaredProducts,
    const std::vector<std::shared_ptr<EpisodeTwinStageProvider>>& providers) {
    SHA256 hash;
    hash.appendString("MetalRoboEpisodeTwinStageV1");
    hash.appendScalar(stage);
    hash.appendString(manifest.id);
    hash.appendString(manifest.coordinateConvention);
    hash.appendString(manifest.engineModelId);
    hash.appendString(manifest.worldProgramId);
    hash.appendScalar(manifest.adapter);
    hash.appendScalar(manifest.schemaVersion);
    hash.appendScalar(manifest.profile);
    for (const CaptureStream& stream : manifest.streams) {
        hash.appendString(stream.id);
        hash.appendScalar(stream.kind);
        hash.appendString(stream.assetId);
        hash.appendString(stream.sensorId);
        hash.appendString(stream.timestampDomain);
        hash.appendScalar(stream.startTimeSeconds);
        hash.appendScalar(stream.endTimeSeconds);
        hash.appendScalar(stream.nominalRateHz);
        hash.appendScalar(stream.calibration.hasResolution);
        hash.appendScalar(stream.calibration.hasIntrinsics);
        hash.appendScalar(stream.calibration.hasDistortion);
        hash.appendScalar(stream.calibration.hasPose);
        if (stream.calibration.present()) {
            hash.appendScalar(stream.calibration.width);
            hash.appendScalar(stream.calibration.height);
            hash.appendScalar(stream.calibration.intrinsics);
            hash.appendScalar(stream.calibration.distortion);
            hash.appendScalar(stream.calibration.worldFromSensor.position);
            hash.appendScalar(stream.calibration.worldFromSensor.orientation);
        }
    }
    hash.appendScalar(
        worldCompilerFingerprint(manifest.seedEpisode, engineModel));
    for (const EpisodeArtifact& artifact : inputs) {
        appendArtifactIdentity(hash, artifact);
    }
    for (const EpisodeArtifact& artifact : declared) {
        appendArtifactIdentity(hash, artifact);
    }
    for (const CaptureProduct& product : declaredProducts) {
        appendProductPayload(hash, product.payload);
    }
    for (const auto& provider : providers) {
        if (provider != nullptr && provider->supports(stage, manifest)) {
            hash.appendString(provider->id());
            hash.appendString(provider->version());
        }
    }
    return hexDigest(hash.finish());
}

template <typename Value>
bool writeScalar(std::ostream& stream, const Value value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return static_cast<bool>(stream);
}

bool writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    return writeScalar(stream, size) &&
           static_cast<bool>(stream.write(
               value.data(), static_cast<std::streamsize>(value.size())));
}

template <typename Value> bool readScalar(std::istream& stream, Value& output) {
    stream.read(reinterpret_cast<char*>(&output), sizeof(output));
    return static_cast<bool>(stream);
}

bool readString(std::istream& stream, std::string& output) {
    std::uint64_t size = 0u;
    if (!readScalar(stream, size) || size > 16u * 1024u * 1024u) {
        return false;
    }
    output.resize(static_cast<std::size_t>(size));
    stream.read(output.data(), static_cast<std::streamsize>(output.size()));
    return static_cast<bool>(stream);
}

std::filesystem::path receiptPath(const std::filesystem::path& root,
                                  const EpisodeTwinStage stage,
                                  const std::string& key) {
    return root / "receipts" / episodeTwinStageName(stage) /
           (key + ".mrreceipt");
}

bool saveReceipt(const std::filesystem::path& root,
                 const EpisodeTwinStageReceipt& receipt, std::string* reason) {
    const std::filesystem::path target =
        receiptPath(root, receipt.stage, receipt.stageKey);
    std::error_code error;
    std::filesystem::create_directories(target.parent_path(), error);
    if (error) {
        if (reason != nullptr) {
            *reason = "could not prepare receipt directory: " + error.message();
        }
        return false;
    }
    const std::filesystem::path temporary = uniqueSibling(target);
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    constexpr std::array<char, 8> magic{'M', 'R', 'E', 'P', 'I', 'S', 'O', 'D'};
    const std::uint32_t version = 3u;
    const std::uint32_t stage = static_cast<std::uint32_t>(receipt.stage);
    const std::uint64_t count = receipt.products.size();
    stream.write(magic.data(), magic.size());
    bool okay = receipt.artifacts.size() == receipt.products.size() &&
                writeScalar(stream, version) && writeScalar(stream, stage) &&
                writeString(stream, receipt.stageKey) &&
                writeScalar(stream, count);
    for (const EpisodeTwinProduct& product : receipt.products) {
        const EpisodeArtifact& artifact = product.artifact;
        const EpisodeTwinProductPayload& payload = product.payload;
        okay = okay && writeString(stream, artifact.id) &&
               writeScalar(stream, artifact.kind) &&
               writeScalar(stream, artifact.producer) &&
               writeString(stream, artifact.assetId) &&
               writeString(stream, artifact.uri) &&
               writeString(stream, artifact.contentHash) &&
               writeScalar(stream, artifact.startTimeSeconds) &&
               writeScalar(stream, artifact.endTimeSeconds) &&
               writeScalar(stream, payload.kind) &&
               writeString(stream, payload.targetId) &&
               writeScalar(stream, payload.calibration.hasResolution) &&
               writeScalar(stream, payload.calibration.hasIntrinsics) &&
               writeScalar(stream, payload.calibration.hasDistortion) &&
               writeScalar(stream, payload.calibration.hasPose) &&
               writeScalar(stream, payload.calibration.width) &&
               writeScalar(stream, payload.calibration.height) &&
               writeScalar(stream, payload.calibration.intrinsics) &&
               writeScalar(stream, payload.calibration.distortion) &&
               writeScalar(
                   stream,
                   payload.calibration.worldFromSensor.position) &&
               writeScalar(
                   stream,
                   payload.calibration.worldFromSensor.orientation) &&
               writeScalar(stream, payload.hasWorldPose) &&
               writeScalar(stream, payload.worldPose.position) &&
               writeScalar(stream, payload.worldPose.orientation) &&
               writeScalar(stream, payload.hasPhysicalPrior) &&
               writeScalar(stream, payload.massScale) &&
               writeScalar(stream, payload.frictionScale) &&
               writeScalar(stream, payload.restitutionScale) &&
               writeScalar(stream, payload.dampingScale) &&
               writeScalar(stream, payload.renderRepresentation) &&
               writeScalar(stream, payload.collisionRepresentation) &&
               writeScalar(stream, payload.hasCollisionBox) &&
               writeScalar(stream, payload.collisionBoxHalfExtents);
    }
    stream.close();
    if (!okay || !stream) {
        std::filesystem::remove(temporary, error);
        if (reason != nullptr) {
            *reason = "could not write stage receipt";
        }
        return false;
    }
    std::filesystem::rename(temporary, target, error);
    if (error) {
        if (std::filesystem::exists(target)) {
            std::filesystem::remove(temporary, error);
            return true;
        }
        const std::string renameError = error.message();
        std::error_code cleanupError;
        std::filesystem::remove(temporary, cleanupError);
        if (reason != nullptr) {
            *reason = "could not publish stage receipt: " + renameError;
        }
        return false;
    }
    return true;
}

bool cachedArtifactsExist(const std::filesystem::path& root,
                          const std::span<const EpisodeArtifact> artifacts) {
    constexpr std::string_view prefix = "sha256:";
    for (const EpisodeArtifact& artifact : artifacts) {
        if (!artifact.contentHash.starts_with(prefix)) {
            return false;
        }
        const std::string digest = artifact.contentHash.substr(prefix.size());
        if (digest.size() != 64u ||
            (!std::filesystem::exists(objectPath(root, digest, false)) &&
             !std::filesystem::exists(objectPath(root, digest, true)))) {
            return false;
        }
    }
    return true;
}

bool loadReceipt(const std::filesystem::path& root,
                 const EpisodeTwinStage stage, const std::string& key,
                 EpisodeTwinStageReceipt& output) {
    std::ifstream stream(receiptPath(root, stage, key), std::ios::binary);
    if (!stream) {
        return false;
    }
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    constexpr std::array<char, 8> expected{'M', 'R', 'E', 'P',
                                           'I', 'S', 'O', 'D'};
    std::uint32_t version = 0u;
    std::uint32_t storedStage = 0u;
    std::uint64_t count = 0u;
    std::string storedKey;
    if (magic != expected || !readScalar(stream, version) ||
        !readScalar(stream, storedStage) || !readString(stream, storedKey) ||
        !readScalar(stream, count) || version != 3u ||
        storedStage != static_cast<std::uint32_t>(stage) || storedKey != key ||
        count > 1'000'000u) {
        return false;
    }
    EpisodeTwinStageReceipt candidate;
    candidate.stage = stage;
    candidate.stageKey = key;
    candidate.cacheHit = true;
    candidate.products.resize(static_cast<std::size_t>(count));
    candidate.artifacts.reserve(static_cast<std::size_t>(count));
    for (EpisodeTwinProduct& product : candidate.products) {
        EpisodeArtifact& artifact = product.artifact;
        EpisodeTwinProductPayload& payload = product.payload;
        if (!readString(stream, artifact.id) ||
            !readScalar(stream, artifact.kind) ||
            !readScalar(stream, artifact.producer) ||
            !readString(stream, artifact.assetId) ||
            !readString(stream, artifact.uri) ||
            !readString(stream, artifact.contentHash) ||
            !readScalar(stream, artifact.startTimeSeconds) ||
            !readScalar(stream, artifact.endTimeSeconds) ||
            !readScalar(stream, payload.kind) ||
            !readString(stream, payload.targetId) ||
            !readScalar(stream, payload.calibration.hasResolution) ||
            !readScalar(stream, payload.calibration.hasIntrinsics) ||
            !readScalar(stream, payload.calibration.hasDistortion) ||
            !readScalar(stream, payload.calibration.hasPose) ||
            !readScalar(stream, payload.calibration.width) ||
            !readScalar(stream, payload.calibration.height) ||
            !readScalar(stream, payload.calibration.intrinsics) ||
            !readScalar(stream, payload.calibration.distortion) ||
            !readScalar(
                stream,
                payload.calibration.worldFromSensor.position) ||
            !readScalar(
                stream,
                payload.calibration.worldFromSensor.orientation) ||
            !readScalar(stream, payload.hasWorldPose) ||
            !readScalar(stream, payload.worldPose.position) ||
            !readScalar(stream, payload.worldPose.orientation) ||
            !readScalar(stream, payload.hasPhysicalPrior) ||
            !readScalar(stream, payload.massScale) ||
            !readScalar(stream, payload.frictionScale) ||
            !readScalar(stream, payload.restitutionScale) ||
            !readScalar(stream, payload.dampingScale) ||
            !readScalar(stream, payload.renderRepresentation) ||
            !readScalar(stream, payload.collisionRepresentation) ||
            !readScalar(stream, payload.hasCollisionBox) ||
            !readScalar(stream, payload.collisionBoxHalfExtents)) {
            return false;
        }
        candidate.artifacts.push_back(artifact);
    }
    if (!cachedArtifactsExist(root, candidate.artifacts)) {
        return false;
    }
    output = std::move(candidate);
    return true;
}

EpisodeTwinCompilerResult failure(const EpisodeTwinCompilerStatus status,
                                  std::string message) {
    return {status, std::move(message)};
}

} // namespace

bool CaptureManifest::valid(std::string* reason) const {
    const auto invalid = [reason](const std::string& message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    const auto finite4 = [](const mr_float4& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
               std::isfinite(value.z) && std::isfinite(value.w);
    };
    if (schemaVersion != 1u && schemaVersion != 2u) {
        return invalid("capture manifest schema version is unsupported");
    }
    if (profile > CaptureProfile::frankaFixedRGBD ||
        (schemaVersion == 1u && profile != CaptureProfile::authoredSeed) ||
        (schemaVersion == 2u &&
         profile == CaptureProfile::authoredSeed)) {
        return invalid("capture manifest profile is invalid for its schema");
    }
    if (adapter > CaptureAdapterKind::videoFallback) {
        return invalid("capture manifest adapter is invalid");
    }
    if (id.empty() || engineModelId.empty() || worldProgramId.empty() ||
        coordinateConvention.empty()) {
        return invalid("capture manifest identity, model, program, or "
                       "coordinates are empty");
    }
    if (streams.empty() && products.empty()) {
        return invalid("capture manifest has no streams or products");
    }
    if (seedEpisode.assets.empty()) {
        return invalid("capture manifest seed episode has no world assets");
    }
    std::unordered_set<std::string> ids;
    for (const CaptureStream& stream : streams) {
        if (stream.id.empty() || !ids.insert(stream.id).second ||
            stream.kind > CaptureStreamKind::urdf || stream.source.empty() ||
            !validHashText(stream.expectedContentHash) ||
            !std::isfinite(stream.startTimeSeconds) ||
            !std::isfinite(stream.endTimeSeconds) ||
            !std::isfinite(stream.nominalRateHz) ||
            stream.endTimeSeconds < stream.startTimeSeconds ||
            stream.nominalRateHz < 0.0) {
            return invalid("capture manifest contains an invalid stream");
        }
        if (stream.calibration.present()) {
            if (!finite4(stream.calibration.intrinsics) ||
                !finite4(stream.calibration.distortion) ||
                !finite4(stream.calibration.worldFromSensor.position) ||
                !finite4(stream.calibration.worldFromSensor.orientation)) {
                return invalid(
                    "capture manifest contains invalid camera calibration");
            }
        }
    }
    for (const CaptureProduct& product : products) {
        if (product.id.empty() || !ids.insert(product.id).second ||
            product.stage > EpisodeTwinStage::publish ||
            product.source.empty() ||
            !validHashText(product.expectedContentHash) ||
            !std::isfinite(product.startTimeSeconds) ||
            !std::isfinite(product.endTimeSeconds) ||
            product.endTimeSeconds < product.startTimeSeconds ||
            product.kind > EpisodeArtifactKind::replay ||
            product.producer > EpisodeArtifactProducer::authored ||
            product.payload.kind > EpisodeTwinProductKind::taskEvents ||
            product.payload.renderRepresentation >
                MR_WORLD_RENDER_PROCEDURAL ||
            product.payload.collisionRepresentation >
                MR_WORLD_COLLISION_DEFORMABLE_SURFACE ||
            !finite4(product.payload.calibration.intrinsics) ||
            !finite4(product.payload.calibration.distortion) ||
            !finite4(
                product.payload.calibration.worldFromSensor.position) ||
            !finite4(
                product.payload.calibration.worldFromSensor.orientation) ||
            !finite4(product.payload.worldPose.position) ||
            !finite4(product.payload.worldPose.orientation) ||
            !finite4(product.payload.collisionBoxHalfExtents) ||
            !std::isfinite(product.payload.massScale) ||
            !std::isfinite(product.payload.frictionScale) ||
            !std::isfinite(product.payload.restitutionScale) ||
            !std::isfinite(product.payload.dampingScale) ||
            (product.payload.hasPhysicalPrior &&
             (product.payload.massScale <= 0.0f ||
              product.payload.frictionScale <= 0.0f ||
              product.payload.restitutionScale < 0.0f ||
              product.payload.dampingScale <= 0.0f)) ||
            (product.payload.hasCollisionBox &&
             (!(product.payload.collisionBoxHalfExtents.x > 0.0f) ||
              !(product.payload.collisionBoxHalfExtents.y > 0.0f) ||
              !(product.payload.collisionBoxHalfExtents.z > 0.0f)))) {
            return invalid("capture manifest contains an invalid product");
        }
    }
    return true;
}

EpisodeArtifactStore::EpisodeArtifactStore(std::filesystem::path root)
    : root_(std::move(root)) {}

const std::filesystem::path& EpisodeArtifactStore::root() const noexcept {
    return root_;
}

bool EpisodeArtifactStore::prepare(std::string* reason) {
    if (root_.empty()) {
        if (reason != nullptr) {
            *reason = "artifact store path is empty";
        }
        return false;
    }
    std::error_code error;
    std::filesystem::create_directories(root_ / "objects" / "sha256", error);
    if (!error) {
        std::filesystem::create_directories(root_ / "receipts", error);
    }
    if (error && reason != nullptr) {
        *reason = "could not prepare artifact store: " + error.message();
    }
    return !error;
}

bool EpisodeArtifactStore::importProduct(const CaptureProduct& product,
                                         EpisodeArtifactImport& output,
                                         std::string* reason) {
    HashedPath hashed;
    if (!hashPath(product.source, hashed, reason)) {
        return false;
    }
    const std::string expected =
        normalizedExpectedHash(product.expectedContentHash);
    if (!expected.empty() && expected != hashed.digest) {
        if (reason != nullptr) {
            *reason = "artifact hash mismatch for " + product.id +
                      ": expected sha256:" + expected +
                      ", observed sha256:" + hashed.digest;
        }
        return false;
    }

    const std::filesystem::path target =
        objectPath(root_, hashed.digest, hashed.directory);
    std::error_code error;
    std::filesystem::create_directories(target.parent_path(), error);
    if (error) {
        if (reason != nullptr) {
            *reason = "could not prepare artifact object directory: " +
                      error.message();
        }
        return false;
    }
    if (!std::filesystem::exists(target)) {
        const std::filesystem::path temporary = uniqueSibling(target);
        if (hashed.directory) {
            std::filesystem::copy(product.source, temporary,
                                  std::filesystem::copy_options::recursive,
                                  error);
        } else {
            std::filesystem::copy_file(product.source, temporary,
                                       std::filesystem::copy_options::none,
                                       error);
        }
        if (error) {
            const std::string copyError = error.message();
            std::error_code cleanupError;
            std::filesystem::remove_all(temporary, cleanupError);
            if (reason != nullptr) {
                *reason = "could not import artifact object: " + copyError;
            }
            return false;
        }
        std::filesystem::rename(temporary, target, error);
        if (error) {
            if (std::filesystem::exists(target)) {
                std::error_code cleanupError;
                std::filesystem::remove_all(temporary, cleanupError);
            } else {
                const std::string renameError = error.message();
                std::error_code cleanupError;
                std::filesystem::remove_all(temporary, cleanupError);
                if (reason != nullptr) {
                    *reason =
                        "could not publish artifact object: " + renameError;
                }
                return false;
            }
        }
    }

    EpisodeArtifactImport candidate;
    candidate.storedPath = target;
    candidate.byteCount = hashed.byteCount;
    candidate.artifact.id = product.id;
    candidate.artifact.kind = product.kind;
    candidate.artifact.producer = product.producer;
    candidate.artifact.assetId = product.assetId;
    candidate.artifact.uri = "mrstore://sha256/" + hashed.digest +
                             (hashed.directory ? "?type=tree" : "");
    candidate.artifact.contentHash = "sha256:" + hashed.digest;
    candidate.artifact.startTimeSeconds = product.startTimeSeconds;
    candidate.artifact.endTimeSeconds = product.endTimeSeconds;
    candidate.product.artifact = candidate.artifact;
    candidate.product.payload = product.payload;
    output = std::move(candidate);
    return true;
}

EpisodeTwinCompilerResult EpisodeTwinAssembler::assemble(
    const CaptureManifest& manifest,
    const std::span<const EpisodeTwinProduct> products,
    const EngineModel& seedEngineModel,
    EpisodeTwin& episodeOutput,
    EngineModel& engineOutput) {
    EpisodeTwin candidate = manifest.seedEpisode;
    EngineModel engineCandidate = seedEngineModel;
    candidate.id = manifest.id;
    candidate.coordinateConvention = manifest.coordinateConvention;
    candidate.artifacts.clear();
    candidate.artifacts.reserve(products.size());

    bool calibratedRGBD = false;
    bool robotStateTrace = false;
    bool robotCommandTrace = false;
    bool semanticGraph = false;
    bool manipulatedPose = false;
    bool manipulatedRenderGeometry = false;
    bool manipulatedCollisionGeometry = false;

    const auto findAsset = [&candidate](const std::string& id) {
        return std::ranges::find_if(
            candidate.assets,
            [&id](const WorldAsset& value) { return value.id == id; });
    };
    const auto findSensor = [&candidate](const std::string& id) {
        return std::ranges::find_if(
            candidate.sensors,
            [&id](const SensorSpec& value) { return value.id == id; });
    };
    const auto applyCalibration =
        [](SensorSpec& sensor, const CaptureCalibration& calibration) {
            if (calibration.hasPose) {
                sensor.localPose = calibration.worldFromSensor;
            }
            if (calibration.hasResolution) {
                sensor.width = calibration.width;
                sensor.height = calibration.height;
            }
            if (calibration.hasIntrinsics) {
                sensor.intrinsics = calibration.intrinsics;
            }
            if (calibration.hasDistortion) {
                sensor.distortion = calibration.distortion;
            }
        };
    const auto targetId = [](const EpisodeTwinProduct& product) {
        return product.payload.targetId.empty()
                   ? product.artifact.assetId
                   : product.payload.targetId;
    };

    for (const EpisodeTwinProduct& product : products) {
        candidate.artifacts.push_back(product.artifact);
        const EpisodeTwinProductPayload& payload = product.payload;
        const std::string target = targetId(product);
        const bool deterministicPhysicalProduct =
            product.artifact.producer ==
                EpisodeArtifactProducer::deterministicTool ||
            product.artifact.producer == EpisodeArtifactProducer::measured;
        if (manifest.profile == CaptureProfile::frankaFixedRGBD &&
            (payload.kind == EpisodeTwinProductKind::sensorCalibration ||
             payload.kind == EpisodeTwinProductKind::segmentation ||
             payload.kind == EpisodeTwinProductKind::objectPoseTrack ||
             payload.kind == EpisodeTwinProductKind::renderGeometry ||
             payload.kind == EpisodeTwinProductKind::collisionGeometry ||
             payload.kind == EpisodeTwinProductKind::physicalPrior) &&
            !deterministicPhysicalProduct) {
            return failure(
                EpisodeTwinCompilerStatus::assemblyFailure,
                "physical product " + product.artifact.id +
                    " must come from a measured or deterministic producer");
        }

        switch (payload.kind) {
        case EpisodeTwinProductKind::artifactOnly:
        case EpisodeTwinProductKind::segmentation:
        case EpisodeTwinProductKind::taskEvents:
            break;
        case EpisodeTwinProductKind::synchronizedSensorStream:
        case EpisodeTwinProductKind::sensorCalibration: {
            if (target.empty()) {
                if (payload.kind ==
                    EpisodeTwinProductKind::sensorCalibration) {
                    return failure(
                        EpisodeTwinCompilerStatus::assemblyFailure,
                        "sensor calibration product has no target sensor");
                }
                break;
            }
            const auto sensor = findSensor(target);
            if (sensor == candidate.sensors.end()) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "capture product " + product.artifact.id +
                        " targets unknown sensor " + target);
            }
            applyCalibration(*sensor, payload.calibration);
            calibratedRGBD =
                calibratedRGBD ||
                (sensor->kind == MR_WORLD_SENSOR_RGBD &&
                 payload.calibration.hasResolution &&
                 payload.calibration.hasIntrinsics &&
                 payload.calibration.hasPose);
            break;
        }
        case EpisodeTwinProductKind::robotStateTrace:
            robotStateTrace = true;
            break;
        case EpisodeTwinProductKind::robotCommandTrace:
            robotCommandTrace = true;
            break;
        case EpisodeTwinProductKind::semanticGraph:
            semanticGraph = true;
            break;
        case EpisodeTwinProductKind::objectPoseTrack: {
            const auto asset = findAsset(target);
            if (target.empty() || asset == candidate.assets.end() ||
                !payload.hasWorldPose) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "object pose product " + product.artifact.id +
                        " has no valid target pose");
            }
            asset->initialPose = payload.worldPose;
            manipulatedPose =
                manipulatedPose ||
                target == candidate.task.manipulatedAssetId;
            break;
        }
        case EpisodeTwinProductKind::renderGeometry: {
            const auto asset = findAsset(target);
            if (target.empty() || asset == candidate.assets.end() ||
                payload.renderRepresentation == MR_WORLD_RENDER_NONE) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "render geometry product " + product.artifact.id +
                        " has no valid target representation");
            }
            asset->render = payload.renderRepresentation;
            manipulatedRenderGeometry =
                manipulatedRenderGeometry ||
                target == candidate.task.manipulatedAssetId;
            break;
        }
        case EpisodeTwinProductKind::collisionGeometry: {
            const auto asset = findAsset(target);
            if (target.empty() || asset == candidate.assets.end() ||
                payload.collisionRepresentation ==
                    MR_WORLD_COLLISION_NONE) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "collision geometry product " + product.artifact.id +
                        " has no valid target representation");
            }
            asset->collision = payload.collisionRepresentation;
            bool collisionPatched =
                manifest.profile != CaptureProfile::frankaFixedRGBD;
            if (payload.hasCollisionBox) {
                collisionPatched = false;
                for (const std::uint32_t shapeIndex :
                     asset->shapeIndices) {
                    if (shapeIndex >= engineCandidate.shapes.size()) {
                        return failure(
                            EpisodeTwinCompilerStatus::assemblyFailure,
                            "collision product " + product.artifact.id +
                                " targets an unknown engine shape");
                    }
                    MRShapeGPU& shape =
                        engineCandidate.shapes[shapeIndex];
                    if (shape.shapeType != MR_SHAPE_BOX) {
                        continue;
                    }
                    shape.dimensions =
                        payload.collisionBoxHalfExtents;
                    shape.contactRestAndBoundingRadius.z = std::sqrt(
                        shape.dimensions.x * shape.dimensions.x +
                        shape.dimensions.y * shape.dimensions.y +
                        shape.dimensions.z * shape.dimensions.z
                    );
                    collisionPatched = true;
                }
            }
            if (!collisionPatched) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "collision product " + product.artifact.id +
                        " did not cook a supported engine collision shape");
            }
            manipulatedCollisionGeometry =
                manipulatedCollisionGeometry ||
                target == candidate.task.manipulatedAssetId;
            break;
        }
        case EpisodeTwinProductKind::physicalPrior: {
            const auto asset = findAsset(target);
            if (target.empty() || asset == candidate.assets.end() ||
                !payload.hasPhysicalPrior) {
                return failure(
                    EpisodeTwinCompilerStatus::assemblyFailure,
                    "physical prior product " + product.artifact.id +
                        " has no valid target prior");
            }
            asset->massScale = payload.massScale;
            asset->frictionScale = payload.frictionScale;
            asset->restitutionScale = payload.restitutionScale;
            asset->dampingScale = payload.dampingScale;
            break;
        }
        }
    }

    if (manifest.profile == CaptureProfile::frankaFixedRGBD) {
        std::vector<std::string> missing;
        if (!calibratedRGBD) {
            missing.emplace_back("calibrated fixed RGB-D");
        }
        if (!robotStateTrace) {
            missing.emplace_back("robot state trace");
        }
        if (!robotCommandTrace) {
            missing.emplace_back("robot command trace");
        }
        if (!semanticGraph) {
            missing.emplace_back("semantic entity/support graph");
        }
        if (!manipulatedPose) {
            missing.emplace_back("manipulated-object pose track");
        }
        if (!manipulatedRenderGeometry) {
            missing.emplace_back("manipulated-object render geometry");
        }
        if (!manipulatedCollisionGeometry) {
            missing.emplace_back("manipulated-object collision geometry");
        }
        if (!missing.empty()) {
            std::string message =
                "physical capture is missing required twin products: ";
            for (std::size_t index = 0u; index < missing.size(); ++index) {
                if (index != 0u) {
                    message += ", ";
                }
                message += missing[index];
            }
            return failure(
                EpisodeTwinCompilerStatus::assemblyFailure,
                std::move(message));
        }
    }

    episodeOutput = std::move(candidate);
    engineOutput = std::move(engineCandidate);
    return {};
}

EpisodeTwinCompiler::EpisodeTwinCompiler(EpisodeTwinCompilerConfig config)
    : config_(std::move(config)) {}

void EpisodeTwinCompiler::addProvider(
    std::shared_ptr<EpisodeTwinStageProvider> provider) {
    if (provider != nullptr) {
        providers_.push_back(std::move(provider));
    }
}

EpisodeTwinCompilerResult EpisodeTwinCompiler::compile(
    const CaptureManifest& manifest, const EngineModel& engineModel,
    const WorldProgram& worldProgram, CompiledEpisodeTwin& output) {
    std::string reason;
    if (!manifest.valid(&reason)) {
        return failure(EpisodeTwinCompilerStatus::invalidManifest,
                       std::move(reason));
    }
    if (config_.artifactStore.empty()) {
        return failure(EpisodeTwinCompilerStatus::invalidConfiguration,
                       "episode compiler artifact store path is empty");
    }
    if (config_.requireExpectedHashes) {
        for (const CaptureStream& stream : manifest.streams) {
            if (stream.expectedContentHash.empty()) {
                return failure(EpisodeTwinCompilerStatus::invalidManifest,
                               "capture stream " + stream.id +
                                   " has no required content hash");
            }
        }
        for (const CaptureProduct& product : manifest.products) {
            if (product.expectedContentHash.empty()) {
                return failure(EpisodeTwinCompilerStatus::invalidManifest,
                               "capture product " + product.id +
                                   " has no required content hash");
            }
        }
    }

    EpisodeArtifactStore store(config_.artifactStore);
    if (!store.prepare(&reason)) {
        return failure(EpisodeTwinCompilerStatus::ioFailure, std::move(reason));
    }

    constexpr std::array stages{
        EpisodeTwinStage::ingest,
        EpisodeTwinStage::selectFrames,
        EpisodeTwinStage::discoverEntities,
        EpisodeTwinStage::segment,
        EpisodeTwinStage::reconstructGeometry,
        EpisodeTwinStage::trackPoses,
        EpisodeTwinStage::inferPhysics,
        EpisodeTwinStage::assembleReplay,
        EpisodeTwinStage::alignReplay,
        EpisodeTwinStage::publish,
    };
    std::vector<EpisodeArtifact> artifacts;
    std::vector<EpisodeTwinProduct> products;
    std::vector<EpisodeTwinStageReceipt> receipts;
    std::unordered_set<std::string> artifactIds;

    try {
        for (const EpisodeTwinStage stage : stages) {
            std::vector<CaptureProduct> declaredProducts;
            if (stage == EpisodeTwinStage::ingest) {
                declaredProducts.reserve(manifest.streams.size() +
                                         manifest.products.size());
                for (const CaptureStream& stream : manifest.streams) {
                    declaredProducts.push_back(streamProduct(stream));
                }
            }
            for (const CaptureProduct& product : manifest.products) {
                if (product.stage == stage) {
                    declaredProducts.push_back(product);
                }
            }

            std::vector<EpisodeArtifact> declaredArtifacts;
            std::vector<EpisodeTwinProduct> declaredTwinProducts;
            declaredArtifacts.reserve(declaredProducts.size());
            declaredTwinProducts.reserve(declaredProducts.size());
            for (const CaptureProduct& product : declaredProducts) {
                EpisodeArtifactImport imported;
                if (!store.importProduct(product, imported, &reason)) {
                    return failure(EpisodeTwinCompilerStatus::artifactFailure,
                                   std::move(reason));
                }
                declaredArtifacts.push_back(std::move(imported.artifact));
                declaredTwinProducts.push_back(std::move(imported.product));
            }

            const std::string key =
                makeStageKey(stage, manifest, engineModel, artifacts,
                             declaredArtifacts, declaredProducts, providers_);
            EpisodeTwinStageReceipt receipt;
            if (config_.resume &&
                loadReceipt(store.root(), stage, key, receipt)) {
                for (const EpisodeArtifact& artifact : receipt.artifacts) {
                    if (!artifactIds.insert(artifact.id).second) {
                        return failure(
                            EpisodeTwinCompilerStatus::artifactFailure,
                            "cached stage produced duplicate artifact id: " +
                                artifact.id);
                    }
                    artifacts.push_back(artifact);
                }
                products.insert(
                    products.end(),
                    receipt.products.begin(),
                    receipt.products.end());
                receipts.push_back(std::move(receipt));
                continue;
            }

            receipt.stage = stage;
            receipt.stageKey = key;
            receipt.artifacts = std::move(declaredArtifacts);
            receipt.products = std::move(declaredTwinProducts);
            for (const auto& provider : providers_) {
                if (provider == nullptr ||
                    !provider->supports(stage, manifest)) {
                    continue;
                }
                std::vector<CaptureProduct> outputs;
                provider->execute({stage, key, &manifest, artifacts}, outputs);
                for (const CaptureProduct& product : outputs) {
                    if (product.stage != stage) {
                        return failure(
                            EpisodeTwinCompilerStatus::providerFailure,
                            "provider " + provider->id() +
                                " returned a product for the wrong stage");
                    }
                    EpisodeArtifactImport imported;
                    if (!store.importProduct(product, imported, &reason)) {
                        return failure(
                            EpisodeTwinCompilerStatus::artifactFailure,
                            std::move(reason));
                    }
                    receipt.artifacts.push_back(std::move(imported.artifact));
                    receipt.products.push_back(std::move(imported.product));
                }
            }
            for (const EpisodeArtifact& artifact : receipt.artifacts) {
                if (!artifactIds.insert(artifact.id).second) {
                    return failure(EpisodeTwinCompilerStatus::artifactFailure,
                                   "stage produced duplicate artifact id: " +
                                       artifact.id);
                }
                artifacts.push_back(artifact);
            }
            products.insert(
                products.end(),
                receipt.products.begin(),
                receipt.products.end());
            if (!saveReceipt(store.root(), receipt, &reason)) {
                return failure(EpisodeTwinCompilerStatus::ioFailure,
                               std::move(reason));
            }
            receipts.push_back(std::move(receipt));
        }
    } catch (const std::exception& error) {
        return failure(EpisodeTwinCompilerStatus::providerFailure,
                       error.what());
    } catch (...) {
        return failure(EpisodeTwinCompilerStatus::providerFailure,
                       "episode stage provider raised an unknown exception");
    }

    CompiledEpisodeTwin candidate;
    EngineModel assembledEngine;
    const EpisodeTwinCompilerResult assembled =
        EpisodeTwinAssembler::assemble(
            manifest,
            products,
            engineModel,
            candidate.episode,
            assembledEngine);
    if (!assembled.succeeded()) {
        return assembled;
    }
    candidate.products = std::move(products);

    const WorldCompileResult twin = compileEpisodeTwin(
        candidate.episode, assembledEngine, candidate.worldTemplate);
    if (!twin.succeeded()) {
        return failure(EpisodeTwinCompilerStatus::episodeCompileFailure,
                       twin.message);
    }
    const WorldCompileResult family = compileWorldFamily(
        candidate.worldTemplate, worldProgram, candidate.worldFamily);
    if (!family.succeeded()) {
        return failure(EpisodeTwinCompilerStatus::familyCompileFailure,
                       family.message);
    }
    const WorldPackResult pack =
        compileWorldPack(candidate.worldFamily, candidate.worldPack);
    if (!pack.succeeded()) {
        return failure(EpisodeTwinCompilerStatus::packCompileFailure,
                       pack.message);
    }
    candidate.receipts = std::move(receipts);
    output = std::move(candidate);
    return {};
}

const char* episodeTwinStageName(const EpisodeTwinStage stage) noexcept {
    switch (stage) {
    case EpisodeTwinStage::ingest:
        return "ingest";
    case EpisodeTwinStage::selectFrames:
        return "select_frames";
    case EpisodeTwinStage::discoverEntities:
        return "discover_entities";
    case EpisodeTwinStage::segment:
        return "segment";
    case EpisodeTwinStage::reconstructGeometry:
        return "reconstruct_geometry";
    case EpisodeTwinStage::trackPoses:
        return "track_poses";
    case EpisodeTwinStage::inferPhysics:
        return "infer_physics";
    case EpisodeTwinStage::assembleReplay:
        return "assemble_replay";
    case EpisodeTwinStage::alignReplay:
        return "align_replay";
    case EpisodeTwinStage::publish:
        return "publish";
    }
}

const char*
episodeTwinCompilerStatusName(const EpisodeTwinCompilerStatus status) noexcept {
    switch (status) {
    case EpisodeTwinCompilerStatus::success:
        return "success";
    case EpisodeTwinCompilerStatus::invalidManifest:
        return "invalid_manifest";
    case EpisodeTwinCompilerStatus::invalidConfiguration:
        return "invalid_configuration";
    case EpisodeTwinCompilerStatus::artifactFailure:
        return "artifact_failure";
    case EpisodeTwinCompilerStatus::providerFailure:
        return "provider_failure";
    case EpisodeTwinCompilerStatus::assemblyFailure:
        return "assembly_failure";
    case EpisodeTwinCompilerStatus::episodeCompileFailure:
        return "episode_compile_failure";
    case EpisodeTwinCompilerStatus::familyCompileFailure:
        return "family_compile_failure";
    case EpisodeTwinCompilerStatus::packCompileFailure:
        return "pack_compile_failure";
    case EpisodeTwinCompilerStatus::ioFailure:
        return "io_failure";
    case EpisodeTwinCompilerStatus::internalFailure:
        return "internal_failure";
    }
}

} // namespace metalrobo
