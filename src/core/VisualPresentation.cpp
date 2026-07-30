#include "metalrobo/VisualPresentation.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kPackMagic{
    'M', 'R', 'V', 'P', 'A', 'C', 'K', '2',
};
constexpr std::array<char, 8u> kEnvironmentMagic{
    'M', 'R', 'E', 'N', 'V', 'P', 'K', '2',
};
constexpr std::uint64_t kPackAlignment = 4096u;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct DiskPackHeaderV2 {
    std::array<char, 8u> magic{};
    std::uint32_t schemaVersion = 0u;
    std::uint32_t sectionCount = 0u;
    std::uint64_t directoryOffset = 0u;
    std::uint64_t fileSize = 0u;
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> contentHash{};
    std::array<std::uint8_t, 24u> reserved{};
};

struct DiskPackSectionV2 {
    std::uint32_t kind = 0u;
    std::uint32_t index = 0u;
    std::uint64_t fileOffset = 0u;
    std::uint64_t byteCount = 0u;
    std::uint64_t elementCount = 0u;
    std::uint32_t elementStride = 0u;
    std::uint32_t reserved = 0u;
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> contentHash{};
};

static_assert(std::is_trivially_copyable_v<DiskPackHeaderV2>);
static_assert(std::is_trivially_copyable_v<DiskPackSectionV2>);
static_assert(sizeof(DiskPackHeaderV2) == 88u);
static_assert(sizeof(DiskPackSectionV2) == 72u);

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

std::string jsonEscape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (character < 0x20u) {
                result += "\\u00";
                result.push_back(hex[character >> 4u]);
                result.push_back(hex[character & 0x0fu]);
            } else {
                result.push_back(static_cast<char>(character));
            }
            break;
        }
    }
    return result;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite4(const mr_float4& value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

bool unitQuaternion(const mr_float4& value) {
    if (!finite4(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    return std::abs(squared - 1.0) <= 1.0e-3;
}

class HashBuilder {
public:
    void append(const void* data, const std::size_t size) {
        const auto* bytes =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= kFnvPrime;
        }
    }

    template <typename Value>
    void scalar(const Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        append(&value, sizeof(value));
    }

    template <typename Value>
    void values(const std::span<const Value> values) {
        const std::uint64_t count = values.size();
        scalar(count);
        if (!values.empty()) {
            append(values.data(), values.size_bytes());
        }
    }

    void string(const std::string& value) {
        values<char>(value);
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFnvOffset;
};

class Sha256Builder {
public:
    Sha256Builder() {
        CC_SHA256_Init(&context_);
    }

    void append(const void* data, const std::size_t size) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(data);
        std::size_t offset = 0u;
        while (offset < size) {
            const std::size_t chunk = std::min<std::size_t>(
                size - offset,
                std::numeric_limits<CC_LONG>::max()
            );
            CC_SHA256_Update(
                &context_,
                bytes + offset,
                static_cast<CC_LONG>(chunk)
            );
            offset += chunk;
        }
    }

    template <typename Value>
    void scalar(const Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        append(&value, sizeof(value));
    }

    template <typename Value>
    void values(const std::span<const Value> values) {
        const std::uint64_t count = values.size();
        scalar(count);
        if (!values.empty()) {
            append(values.data(), values.size_bytes());
        }
    }

    void string(const std::string& value) {
        values<char>(value);
    }

    [[nodiscard]] std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH>
    finishBytes() {
        std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> digest{};
        CC_SHA256_Final(digest.data(), &context_);
        return digest;
    }

private:
    CC_SHA256_CTX context_{};
};

std::string hexSha256(
    const std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH>& digest
) {
    std::ostringstream stream;
    stream << "sha256:";
    for (const std::uint8_t value : digest) {
        stream << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<unsigned>(value);
    }
    return stream.str();
}

std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> sha256Bytes(
    const void* data,
    const std::size_t size
) {
    Sha256Builder hash;
    if (size != 0u) {
        hash.append(data, size);
    }
    return hash.finishBytes();
}

std::uint64_t alignUp(
    const std::uint64_t value,
    const std::uint64_t alignment
) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

class BinaryWriter {
public:
    template <typename Value>
    void scalar(const Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        const auto* first =
            reinterpret_cast<const std::uint8_t*>(&value);
        bytes_.insert(bytes_.end(), first, first + sizeof(value));
    }

    template <typename Value>
    void values(const std::vector<Value>& values) {
        static_assert(std::is_trivially_copyable_v<Value>);
        scalar(static_cast<std::uint64_t>(values.size()));
        const auto* first =
            reinterpret_cast<const std::uint8_t*>(values.data());
        bytes_.insert(
            bytes_.end(),
            first,
            first + values.size() * sizeof(Value)
        );
    }

    void string(const std::string& value) {
        scalar(static_cast<std::uint64_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const {
        return bytes_;
    }

private:
    std::vector<std::uint8_t> bytes_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::span<const std::uint8_t> bytes)
        : bytes_(bytes) {}

    template <typename Value>
    bool scalar(Value& value) {
        static_assert(std::is_trivially_copyable_v<Value>);
        if (sizeof(Value) > bytes_.size() - offset_) {
            return false;
        }
        std::memcpy(&value, bytes_.data() + offset_, sizeof(Value));
        offset_ += sizeof(Value);
        return true;
    }

    template <typename Value>
    bool values(std::vector<Value>& values) {
        static_assert(std::is_trivially_copyable_v<Value>);
        std::uint64_t count = 0u;
        if (!scalar(count) ||
            count > std::numeric_limits<std::size_t>::max() /
                sizeof(Value)) {
            return false;
        }
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(Value);
        if (bytes > bytes_.size() - offset_) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        if (bytes != 0u) {
            std::memcpy(values.data(), bytes_.data() + offset_, bytes);
        }
        offset_ += bytes;
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t size = 0u;
        if (!scalar(size) ||
            size > bytes_.size() - offset_) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            static_cast<std::size_t>(size)
        );
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    [[nodiscard]] bool complete() const noexcept {
        return offset_ == bytes_.size();
    }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t offset_ = 0u;
};

void writeTextureMetadata(
    BinaryWriter& writer,
    const VisualTextureImageV2& texture
) {
    writer.string(texture.id);
    writer.string(texture.contentHash);
    writer.scalar(texture.width);
    writer.scalar(texture.height);
    writer.scalar(texture.mipCount);
    writer.scalar(texture.arrayLength);
    writer.scalar(texture.pixelFormat);
    writer.scalar(texture.dimension);
    writer.scalar(texture.flags);
    writer.values(texture.subresources);
}

bool readTextureMetadata(
    BinaryReader& reader,
    VisualTextureImageV2& texture
) {
    return reader.string(texture.id) &&
        reader.string(texture.contentHash) &&
        reader.scalar(texture.width) &&
        reader.scalar(texture.height) &&
        reader.scalar(texture.mipCount) &&
        reader.scalar(texture.arrayLength) &&
        reader.scalar(texture.pixelFormat) &&
        reader.scalar(texture.dimension) &&
        reader.scalar(texture.flags) &&
        reader.values(texture.subresources);
}

void writeBinding(
    BinaryWriter& writer,
    const VisualSymbolicBindingV2& binding
) {
    writer.string(binding.node);
    writer.string(binding.link);
    writer.scalar(binding.instanceIndex);
    writer.scalar(binding.bodyIndex);
    writer.scalar(binding.binding);
}

bool readBinding(
    BinaryReader& reader,
    VisualSymbolicBindingV2& binding
) {
    return reader.string(binding.node) &&
        reader.string(binding.link) &&
        reader.scalar(binding.instanceIndex) &&
        reader.scalar(binding.bodyIndex) &&
        reader.scalar(binding.binding);
}

std::vector<std::uint8_t> serializeAssetMetadata(
    const VisualAssetPackV2& pack
) {
    BinaryWriter writer;
    writer.scalar(pack.schemaVersion);
    writer.string(pack.id);
    writer.string(pack.sourceUri);
    writer.string(pack.sourceContentHash);
    writer.string(pack.contentHash);
    writer.string(pack.license);
    writer.string(pack.preprocessingProvenance);
    writer.string(pack.coordinateConvention);
    writer.scalar(static_cast<std::uint64_t>(pack.textures.size()));
    writer.scalar(
        static_cast<std::uint64_t>(pack.symbolicBindings.size())
    );
    return writer.bytes();
}

std::vector<std::uint8_t> serializeTextureDescriptors(
    const VisualAssetPackV2& pack
) {
    BinaryWriter writer;
    for (const VisualTextureImageV2& texture : pack.textures) {
        writeTextureMetadata(writer, texture);
    }
    return writer.bytes();
}

std::vector<std::uint8_t> serializeSymbolicBindings(
    const VisualAssetPackV2& pack
) {
    BinaryWriter writer;
    for (const VisualSymbolicBindingV2& binding :
         pack.symbolicBindings) {
        writeBinding(writer, binding);
    }
    return writer.bytes();
}

std::string sha256ContentHash(const VisualAssetPackV2& pack) {
    Sha256Builder hash;
    hash.scalar(pack.schemaVersion);
    hash.string(pack.id);
    hash.string(pack.sourceUri);
    hash.string(pack.sourceContentHash);
    hash.string(pack.license);
    hash.string(pack.preprocessingProvenance);
    hash.string(pack.coordinateConvention);
    hash.values<MRVisualVertexGPUV2>(pack.vertices);
    hash.values<std::uint32_t>(pack.indices);
    hash.values<MRVisualPrimitiveGPUV2>(pack.primitives);
    hash.values<MRVisualInstanceGPUV2>(pack.instances);
    hash.values<MRVisualMaterialGPUV2>(pack.materials);
    hash.values<MRVisualTextureBindingGPUV2>(
        pack.textureBindings
    );
    for (const VisualTextureImageV2& texture : pack.textures) {
        hash.string(texture.id);
        hash.string(texture.contentHash);
        hash.scalar(texture.width);
        hash.scalar(texture.height);
        hash.scalar(texture.mipCount);
        hash.scalar(texture.arrayLength);
        hash.scalar(texture.pixelFormat);
        hash.scalar(texture.dimension);
        hash.scalar(texture.flags);
        hash.values<VisualTextureSubresourceV2>(
            texture.subresources
        );
        hash.values<std::uint8_t>(texture.data);
    }
    for (const VisualSymbolicBindingV2& binding :
         pack.symbolicBindings) {
        hash.string(binding.node);
        hash.string(binding.link);
        hash.scalar(binding.instanceIndex);
        hash.scalar(binding.bodyIndex);
        hash.scalar(binding.binding);
    }
    return hexSha256(hash.finishBytes());
}

bool validMaterial(const MRVisualMaterialGPUV2& material) {
    return finite4(material.baseColorAndOpacity) &&
        finite4(material.emissionAndStrength) &&
        finite4(material.surface) &&
        finite4(material.coatingAndAlphaCutoff) &&
        material.baseColorAndOpacity.w >= 0.0f &&
        material.baseColorAndOpacity.w <= 1.0f &&
        material.surface.x >= 0.0f &&
        material.surface.x <= 1.0f &&
        material.surface.y >= 0.0f &&
        material.surface.y <= 1.0f &&
        material.coatingAndAlphaCutoff.x >= 0.0f &&
        material.coatingAndAlphaCutoff.x <= 1.0f &&
        material.coatingAndAlphaCutoff.y >= 0.0f &&
        material.coatingAndAlphaCutoff.y <= 1.0f &&
        material.flags.x <= MR_VISUAL_ALPHA_BLEND;
}

MRVisualLightGPUV1 defaultLight() {
    MRVisualLightGPUV1 light{};
    light.positionAndRange = {0.5f, -0.5f, 2.5f, 20.0f};
    light.directionAndSpot = {
        -0.35f, 0.35f, -0.87f, -1.0f,
    };
    light.colorAndIntensity = {1.0f, 0.97f, 0.92f, 1200.0f};
    light.shape = {1.0f, 1.0f, -1.0f, 0.08f};
    light.identity = {
        MR_VISUAL_LIGHT_DIRECTIONAL,
        MR_VISUAL_LIGHT_UNIT_LUX,
        100u,
        1u,
    };
    light.shadow = {1u, 0u, 4u, 0u};
    return light;
}

struct SectionSourceV2 {
    VisualAssetSectionKindV2 kind =
        VisualAssetSectionKindV2::metadata;
    std::uint32_t index = 0u;
    const void* data = nullptr;
    std::uint64_t byteCount = 0u;
    std::uint64_t elementCount = 0u;
    std::uint32_t elementStride = 0u;
};

std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH>
parseSha256(const std::string& value) {
    std::array<std::uint8_t, CC_SHA256_DIGEST_LENGTH> result{};
    if (!value.starts_with("sha256:") ||
        value.size() != 7u + 2u * result.size()) {
        return result;
    }
    const auto nibble = [](const char character) -> std::uint8_t {
        if (character >= '0' && character <= '9') {
            return static_cast<std::uint8_t>(character - '0');
        }
        if (character >= 'a' && character <= 'f') {
            return static_cast<std::uint8_t>(
                character - 'a' + 10
            );
        }
        if (character >= 'A' && character <= 'F') {
            return static_cast<std::uint8_t>(
                character - 'A' + 10
            );
        }
        return 0xffu;
    };
    for (std::size_t index = 0u; index < result.size(); ++index) {
        const std::uint8_t high =
            nibble(value[7u + 2u * index]);
        const std::uint8_t low =
            nibble(value[8u + 2u * index]);
        if (high > 0x0fu || low > 0x0fu) {
            result.fill(0u);
            return result;
        }
        result[index] =
            static_cast<std::uint8_t>((high << 4u) | low);
    }
    return result;
}

bool writePadding(
    std::ofstream& stream,
    std::uint64_t& position,
    const std::uint64_t target
) {
    const std::array<char, kPackAlignment> zeroes{};
    while (position < target) {
        const std::uint64_t count =
            std::min<std::uint64_t>(
                target - position,
                zeroes.size()
            );
        stream.write(
            zeroes.data(),
            static_cast<std::streamsize>(count)
        );
        if (!stream) {
            return false;
        }
        position += count;
    }
    return true;
}

bool writeSectionedPack(
    const std::array<char, 8u>& magic,
    const std::uint32_t schemaVersion,
    const std::string& contentHash,
    const std::span<const SectionSourceV2> sources,
    const std::filesystem::path& path,
    std::string* reason
) {
    if (path.empty() ||
        sources.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return fail(reason, "sectioned pack destination is invalid");
    }
    DiskPackHeaderV2 header;
    header.magic = magic;
    header.schemaVersion = schemaVersion;
    header.sectionCount =
        static_cast<std::uint32_t>(sources.size());
    header.directoryOffset = sizeof(DiskPackHeaderV2);
    header.contentHash = parseSha256(contentHash);

    std::vector<DiskPackSectionV2> directory(sources.size());
    std::uint64_t cursor = alignUp(
        sizeof(DiskPackHeaderV2) +
            directory.size() * sizeof(DiskPackSectionV2),
        kPackAlignment
    );
    for (std::size_t index = 0u;
         index < sources.size();
         ++index) {
        const SectionSourceV2& source = sources[index];
        DiskPackSectionV2& section = directory[index];
        section.kind = static_cast<std::uint32_t>(source.kind);
        section.index = source.index;
        section.fileOffset = cursor;
        section.byteCount = source.byteCount;
        section.elementCount = source.elementCount;
        section.elementStride = source.elementStride;
        section.contentHash = sha256Bytes(
            source.data,
            static_cast<std::size_t>(source.byteCount)
        );
        cursor = alignUp(
            cursor + source.byteCount,
            kPackAlignment
        );
    }
    header.fileSize = cursor;

    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(
            path.parent_path(),
            error
        );
        if (error) {
            return fail(reason, "could not create pack directory");
        }
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    std::ofstream stream(
        temporary,
        std::ios::binary | std::ios::trunc
    );
    if (!stream) {
        return fail(reason, "could not open pack output");
    }
    stream.write(
        reinterpret_cast<const char*>(&header),
        sizeof(header)
    );
    stream.write(
        reinterpret_cast<const char*>(directory.data()),
        static_cast<std::streamsize>(
            directory.size() * sizeof(DiskPackSectionV2)
        )
    );
    std::uint64_t position =
        sizeof(header) +
        directory.size() * sizeof(DiskPackSectionV2);
    for (std::size_t index = 0u;
         index < sources.size();
         ++index) {
        const SectionSourceV2& source = sources[index];
        const DiskPackSectionV2& section = directory[index];
        if (!writePadding(stream, position, section.fileOffset)) {
            break;
        }
        if (source.byteCount != 0u) {
            stream.write(
                static_cast<const char*>(source.data),
                static_cast<std::streamsize>(source.byteCount)
            );
            position += source.byteCount;
        }
    }
    if (stream) {
        writePadding(stream, position, header.fileSize);
        stream.flush();
    }
    stream.close();
    if (!stream) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not stream sectioned pack");
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not publish sectioned pack");
    }
    return true;
}

bool readExact(
    std::ifstream& stream,
    const std::uint64_t offset,
    void* destination,
    const std::size_t size
) {
    stream.seekg(static_cast<std::streamoff>(offset));
    if (!stream) {
        return false;
    }
    if (size != 0u) {
        stream.read(
            static_cast<char*>(destination),
            static_cast<std::streamsize>(size)
        );
    }
    return static_cast<bool>(stream);
}

std::optional<std::vector<DiskPackSectionV2>> readDirectory(
    std::ifstream& stream,
    const std::array<char, 8u>& magic,
    const std::uint32_t version,
    DiskPackHeaderV2& header,
    std::string* reason
) {
    if (!readExact(stream, 0u, &header, sizeof(header)) ||
        header.magic != magic ||
        header.schemaVersion != version ||
        header.sectionCount == 0u ||
        header.directoryOffset != sizeof(DiskPackHeaderV2)) {
        fail(reason, "sectioned pack header is invalid");
        return std::nullopt;
    }
    stream.seekg(0, std::ios::end);
    const std::streamoff fileSize = stream.tellg();
    if (fileSize < 0 ||
        static_cast<std::uint64_t>(fileSize) != header.fileSize) {
        fail(reason, "sectioned pack file size is invalid");
        return std::nullopt;
    }
    std::vector<DiskPackSectionV2> directory(
        header.sectionCount
    );
    if (!readExact(
            stream,
            header.directoryOffset,
            directory.data(),
            directory.size() * sizeof(DiskPackSectionV2)
        )) {
        fail(reason, "sectioned pack directory is truncated");
        return std::nullopt;
    }
    for (const DiskPackSectionV2& section : directory) {
        if (section.fileOffset > header.fileSize ||
            section.byteCount >
                header.fileSize - section.fileOffset) {
            fail(reason, "sectioned pack section is out of range");
            return std::nullopt;
        }
    }
    return directory;
}

const DiskPackSectionV2* findSection(
    const std::span<const DiskPackSectionV2> sections,
    const VisualAssetSectionKindV2 kind,
    const std::uint32_t index = 0u
) {
    const auto found = std::ranges::find_if(
        sections,
        [kind, index](const DiskPackSectionV2& section) {
            return section.kind ==
                    static_cast<std::uint32_t>(kind) &&
                section.index == index;
        }
    );
    return found == sections.end() ? nullptr : &*found;
}

bool readByteSection(
    std::ifstream& stream,
    const std::span<const DiskPackSectionV2> sections,
    const VisualAssetSectionKindV2 kind,
    std::vector<std::uint8_t>& output,
    std::string* reason,
    const std::uint32_t index = 0u
) {
    const DiskPackSectionV2* section =
        findSection(sections, kind, index);
    if (section == nullptr ||
        section->byteCount >
            std::numeric_limits<std::size_t>::max()) {
        return fail(reason, "pack byte section is invalid");
    }
    output.resize(
        static_cast<std::size_t>(section->byteCount)
    );
    if (!readExact(
            stream,
            section->fileOffset,
            output.data(),
            output.size()
        ) ||
        sha256Bytes(output.data(), output.size()) !=
            section->contentHash) {
        return fail(reason, "pack byte section checksum failed");
    }
    return true;
}

template <typename Value>
bool readVectorSection(
    std::ifstream& stream,
    const std::span<const DiskPackSectionV2> sections,
    const VisualAssetSectionKindV2 kind,
    std::vector<Value>& output,
    std::string* reason
) {
    const DiskPackSectionV2* section =
        findSection(sections, kind);
    if (section == nullptr ||
        section->elementStride != sizeof(Value) ||
        section->elementCount >
            std::numeric_limits<std::size_t>::max() /
                sizeof(Value) ||
        section->byteCount !=
            section->elementCount * sizeof(Value)) {
        return fail(reason, "pack vector section is invalid");
    }
    output.resize(
        static_cast<std::size_t>(section->elementCount)
    );
    if (!readExact(
            stream,
            section->fileOffset,
            output.data(),
            static_cast<std::size_t>(section->byteCount)
        ) ||
        sha256Bytes(output.data(), output.size() * sizeof(Value)) !=
            section->contentHash) {
        return fail(reason, "pack vector section checksum failed");
    }
    return true;
}

} // namespace

bool VisualTextureImageV2::valid(std::string* reason) const {
    const auto bytesPerPixel = [this]() -> std::uint32_t {
        switch (pixelFormat) {
        case VisualTexturePixelFormatV2::rgba8Unorm:
        case VisualTexturePixelFormatV2::rgba8UnormSrgb:
        case VisualTexturePixelFormatV2::rg11b10Float:
        case VisualTexturePixelFormatV2::rg16Float:
            return 4u;
        case VisualTexturePixelFormatV2::rgba16Float:
            return 8u;
        }
        return 0u;
    }();
    const std::uint32_t expectedSlices =
        dimension == VisualTextureDimensionV2::cube ? 6u : 1u;
    if (id.empty() || contentHash.empty() ||
        width == 0u || height == 0u ||
        mipCount == 0u || arrayLength != expectedSlices ||
        bytesPerPixel == 0u ||
        subresources.size() !=
            static_cast<std::size_t>(mipCount) * arrayLength ||
        data.empty() ||
        (flags & ~(MR_VISUAL_TEXTURE_SRGB |
                   MR_VISUAL_TEXTURE_CLAMP_U |
                   MR_VISUAL_TEXTURE_CLAMP_V)) != 0u) {
        return fail(reason, "visual texture metadata is invalid");
    }
    std::uint64_t expectedOffset = 0u;
    for (std::uint32_t slice = 0u;
         slice < arrayLength;
         ++slice) {
        for (std::uint32_t level = 0u;
             level < mipCount;
             ++level) {
            const VisualTextureSubresourceV2& subresource =
                subresources[
                    static_cast<std::size_t>(slice) * mipCount +
                    level
                ];
            const std::uint32_t levelWidth =
                std::max(width >> level, 1u);
            const std::uint32_t levelHeight =
                std::max(height >> level, 1u);
            const std::uint64_t minimumRow =
                static_cast<std::uint64_t>(levelWidth) *
                bytesPerPixel;
            const std::uint64_t minimumImage =
                static_cast<std::uint64_t>(
                    subresource.bytesPerRow
                ) * levelHeight;
            if (subresource.mipLevel != level ||
                subresource.arraySlice != slice ||
                subresource.width != levelWidth ||
                subresource.height != levelHeight ||
                subresource.dataOffset != expectedOffset ||
                subresource.bytesPerRow < minimumRow ||
                subresource.bytesPerImage < minimumImage ||
                subresource.dataSize !=
                    subresource.bytesPerImage ||
                subresource.dataSize > data.size() ||
                subresource.dataOffset >
                    data.size() - subresource.dataSize) {
                return fail(
                    reason,
                    "visual texture subresource layout is invalid"
                );
            }
            expectedOffset += subresource.dataSize;
        }
    }
    if (expectedOffset != data.size()) {
        return fail(reason, "visual texture payload size is invalid");
    }
    return true;
}

bool VisualSymbolicBindingV2::valid(
    const std::uint32_t instanceCount,
    std::string* reason
) const {
    const bool bodyBinding =
        binding == MR_VISUAL_BINDING_RIGID_BODY ||
        binding == MR_VISUAL_BINDING_ARTICULATED_LINK;
    if (node.empty() || instanceIndex >= instanceCount ||
        binding > MR_VISUAL_BINDING_ARTICULATED_LINK ||
        (bodyBinding && (link.empty() ||
                         bodyIndex == MR_INVALID_INDEX)) ||
        (!bodyBinding && bodyIndex != MR_INVALID_INDEX)) {
        return fail(reason, "symbolic visual binding is invalid");
    }
    return true;
}

bool VisualAssetPackV2::valid(std::string* reason) const {
    if (schemaVersion != kVisualAssetPackVersion ||
        id.empty() || sourceUri.empty() ||
        sourceContentHash.empty() || contentHash.empty() ||
        license.empty() || preprocessingProvenance.empty() ||
        coordinateConvention.empty() || vertices.empty() ||
        indices.empty() || primitives.empty() ||
        instances.empty() || materials.empty()) {
        return fail(reason, "visual asset pack metadata is incomplete");
    }
    for (std::size_t vertexIndex = 0u;
         vertexIndex < vertices.size();
         ++vertexIndex) {
        const MRVisualVertexGPUV2& vertex = vertices[vertexIndex];
        const float normalLengthSquared =
            vertex.normalAndTangentSign.x *
                vertex.normalAndTangentSign.x +
            vertex.normalAndTangentSign.y *
                vertex.normalAndTangentSign.y +
            vertex.normalAndTangentSign.z *
                vertex.normalAndTangentSign.z;
        const float tangentLengthSquared =
            vertex.tangent.x * vertex.tangent.x +
            vertex.tangent.y * vertex.tangent.y +
            vertex.tangent.z * vertex.tangent.z;
        const float tangentProjection =
            vertex.normalAndTangentSign.x * vertex.tangent.x +
            vertex.normalAndTangentSign.y * vertex.tangent.y +
            vertex.normalAndTangentSign.z * vertex.tangent.z;
        const char* invalidField =
            !finite4(vertex.position) ||
                    vertex.position.w != 1.0f
                ? "position"
                : !finite4(vertex.normalAndTangentSign) ||
                          std::abs(
                              vertex.normalAndTangentSign.w
                          ) != 1.0f ||
                          std::abs(normalLengthSquared - 1.0f) >
                              2.0e-3f
                    ? "normal"
                    : !finite4(vertex.tangent) ||
                              std::abs(
                                  tangentLengthSquared - 1.0f
                              ) > 2.0e-3f ||
                              std::abs(tangentProjection) >
                                  2.0e-3f
                        ? "tangent"
                        : !finite4(vertex.texcoord01)
                            ? "texcoord"
                            : !finite4(vertex.color)
                                ? "color"
                                : nullptr;
        if (invalidField != nullptr) {
            return fail(
                reason,
                "visual asset pack has an invalid vertex at index " +
                    std::to_string(vertexIndex) + " (" +
                    invalidField + ")"
            );
        }
    }
    if (std::ranges::any_of(
            indices,
            [this](const std::uint32_t index) {
                return index >= vertices.size();
            }
        )) {
        return fail(reason, "visual asset pack index is out of range");
    }
    if (!std::ranges::all_of(materials, validMaterial)) {
        return fail(reason, "visual asset pack has an invalid material");
    }
    for (const VisualTextureImageV2& texture : textures) {
        if (!texture.valid(reason)) {
            return false;
        }
    }
    for (const MRVisualTextureBindingGPUV2& binding :
         textureBindings) {
        if (binding.resource.x >= textures.size() ||
            binding.resource.y >= 108u ||
            binding.resource.z > 1u ||
            !finite4(binding.uvTransform0) ||
            !finite4(binding.uvTransform1)) {
            return fail(reason, "visual texture binding is invalid");
        }
    }
    const auto bindingValid = [this](const std::uint32_t binding) {
        return binding == MR_INVALID_INDEX ||
            binding < textureBindings.size();
    };
    for (const MRVisualMaterialGPUV2& material : materials) {
        const std::array textureIndices{
            material.textureIndices0.x,
            material.textureIndices0.y,
            material.textureIndices0.z,
            material.textureIndices0.w,
            material.textureIndices1.x,
            material.textureIndices1.y,
            material.textureIndices1.z,
            material.reserved.x,
            material.reserved.y,
            material.reserved.z,
            material.reserved.w,
        };
        if (!std::ranges::all_of(textureIndices, bindingValid)) {
            return fail(
                reason,
                "visual material texture binding is out of range"
            );
        }
    }
    for (std::size_t index = 0u; index < instances.size(); ++index) {
        const MRVisualInstanceGPUV2& instance = instances[index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(instance.geometry.x) +
            instance.geometry.y;
        if (!finite4(instance.translationAndScale) ||
            !unitQuaternion(instance.orientation) ||
            !(instance.translationAndScale.w > 0.0f) ||
            instance.binding.z >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            end > primitives.size()) {
            return fail(reason, "visual asset pack has an invalid instance");
        }
    }
    for (std::size_t index = 0u; index < primitives.size(); ++index) {
        const MRVisualPrimitiveGPUV2& primitive = primitives[index];
        const std::uint64_t end =
            static_cast<std::uint64_t>(primitive.geometry.x) +
            primitive.geometry.y;
        if (primitive.geometry.y == 0u ||
            primitive.geometry.y % 3u != 0u ||
            end > indices.size() ||
            primitive.geometry.z >= materials.size() ||
            primitive.geometry.w >= instances.size() ||
            !finite4(primitive.boundsMinimum) ||
            !finite4(primitive.boundsMaximum) ||
            primitive.boundsMinimum.x > primitive.boundsMaximum.x ||
            primitive.boundsMinimum.y > primitive.boundsMaximum.y ||
            primitive.boundsMinimum.z > primitive.boundsMaximum.z) {
            return fail(reason, "visual asset pack has an invalid primitive");
        }
        const MRVisualInstanceGPUV2& instance =
            instances[primitive.geometry.w];
        if (index < instance.geometry.x ||
            index >= instance.geometry.x + instance.geometry.y) {
            return fail(
                reason,
                "visual primitive is outside its instance range"
            );
        }
    }
    for (const VisualSymbolicBindingV2& binding : symbolicBindings) {
        if (!binding.valid(
                static_cast<std::uint32_t>(instances.size()),
                reason
            )) {
            return false;
        }
    }
    if (contentHash != sha256ContentHash(*this)) {
        return fail(reason, "visual asset pack content hash is invalid");
    }
    return true;
}

bool VisualEnvironmentPackV2::valid(std::string* reason) const {
    if (schemaVersion != kVisualEnvironmentPackVersion ||
        id.empty() || sourceUri.empty() ||
        sourceContentHash.empty() || contentHash.empty() ||
        sourceColorSpace.empty() ||
        preprocessingProvenance.empty() ||
        specularFaceSize == 0u || diffuseFaceSize == 0u ||
        brdfLutSize == 0u ||
        !diffuseIrradiance.valid(reason) ||
        !prefilteredSpecular.valid(reason) ||
        !brdfLut.valid(reason) ||
        diffuseIrradiance.dimension !=
            VisualTextureDimensionV2::cube ||
        prefilteredSpecular.dimension !=
            VisualTextureDimensionV2::cube ||
        brdfLut.dimension !=
            VisualTextureDimensionV2::texture2D ||
        diffuseIrradiance.width != diffuseFaceSize ||
        prefilteredSpecular.width != specularFaceSize ||
        brdfLut.width != brdfLutSize ||
        contentHash !=
            computeVisualEnvironmentPackContentHash(*this)) {
        return reason != nullptr && !reason->empty()
            ? false
            : fail(reason, "visual environment pack is invalid");
    }
    return true;
}

bool VisualEnvironmentReferenceV2::valid(
    std::string* reason
) const {
    if (id.empty() || contentHash.empty() ||
        !finite(intensity) || intensity < 0.0f ||
        !finite(rotationRadians) ||
        (packPath.empty() &&
         !contentHash.starts_with("builtin:"))) {
        return fail(reason, "visual environment is invalid");
    }
    return true;
}

bool VisualLightRigV1::valid(std::string* reason) const {
    if (id.empty() || contentHash.empty() || lights.empty()) {
        return fail(reason, "visual light rig is incomplete");
    }
    std::set<std::uint32_t> identities;
    for (const MRVisualLightGPUV1& light : lights) {
        if (!finite4(light.positionAndRange) ||
            !finite4(light.directionAndSpot) ||
            !finite4(light.colorAndIntensity) ||
            !finite4(light.shape) ||
            light.identity.x > MR_VISUAL_LIGHT_RECTANGLE ||
            light.identity.y > MR_VISUAL_LIGHT_UNIT_NIT ||
            light.identity.w == 0u ||
            !identities.insert(light.identity.w).second ||
            light.positionAndRange.w < 0.0f ||
            light.colorAndIntensity.x < 0.0f ||
            light.colorAndIntensity.y < 0.0f ||
            light.colorAndIntensity.z < 0.0f ||
            light.colorAndIntensity.w < 0.0f) {
            return fail(reason, "visual light rig has an invalid light");
        }
    }
    return true;
}

bool VisualRenderSceneV3::valid(std::string* reason) const {
    if (id.empty() || assetCount == 0u ||
        assetCount >= MR_INVALID_INDEX ||
        (gaussians.empty() && visualPacks.empty())) {
        return fail(reason, "V3 visual render scene is incomplete");
    }
    for (const VisualAssetReferenceV3& reference : visualPacks) {
        if (reference.packPath.empty() ||
            reference.contentHash.empty() ||
            reference.assetIndex >= assetCount ||
            reference.semanticId == 0u ||
            reference.semanticId == MR_INVALID_INDEX ||
            reference.instanceId == 0u ||
            reference.instanceId == MR_INVALID_INDEX) {
            return fail(reason, "V3 visual pack reference is invalid");
        }
    }
    for (const MRHybridGaussianGPU& gaussian : gaussians) {
        if (!finite4(gaussian.meanAndOpacity) ||
            !finite4(gaussian.scaleAndImportance) ||
            !unitQuaternion(gaussian.orientation) ||
            !finite4(gaussian.colorAndEmission) ||
            gaussian.binding.x >= assetCount ||
            gaussian.binding.z == 0u ||
            gaussian.binding.z == MR_INVALID_INDEX ||
            gaussian.binding.w > MR_HYBRID_GAUSSIAN_WORLD ||
            (gaussian.binding.w ==
                 MR_HYBRID_GAUSSIAN_BODY_LOCAL &&
             gaussian.binding.y >= bodyCount)) {
            return fail(reason, "V3 visual scene has an invalid Gaussian");
        }
    }
    for (const MRVisualSensorBindingGPU& sensor : sensorBindings) {
        const bool bodyBound =
            sensor.identity.x == MR_VISUAL_BINDING_RIGID_BODY ||
            sensor.identity.x ==
                MR_VISUAL_BINDING_ARTICULATED_LINK;
        if (sensor.identity.x >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            (bodyBound && sensor.identity.y >= bodyCount) ||
            !finite4(sensor.timing) ||
            !finite4(sensor.rangeAndResponse) ||
            !(sensor.timing.x > 0.0f) ||
            !(sensor.rangeAndResponse.y >
              sensor.rangeAndResponse.x)) {
            return fail(reason, "V3 visual sensor binding is invalid");
        }
    }
    if (!environment.valid(reason) ||
        !lightRig.valid(reason)) {
        return false;
    }
    if (fingerprint != 0u &&
        fingerprint != computeVisualRenderSceneV3Fingerprint(*this)) {
        return fail(reason, "V3 visual scene fingerprint does not match");
    }
    return true;
}

bool VisualSceneManifestV3::valid(std::string* reason) const {
    const std::set<std::string> uniquePacks{
        visualPackHashes.begin(),
        visualPackHashes.end(),
    };
    if (schemaVersion != kVisualSceneManifestV3Version ||
        id.empty() ||
        coordinateConvention != "x-forward,y-left,z-up" ||
        worldFingerprint == 0u ||
        fingerprint == 0u ||
        visualPackHashes.empty() ||
        uniquePacks.size() != visualPackHashes.size() ||
        std::ranges::any_of(
            visualPackHashes,
            [](const std::string& value) {
                return value.empty();
            }
        ) ||
        environmentMapHash.empty() ||
        lightRigHash.empty() ||
        preprocessingProvenance.empty() ||
        !renderScene.valid(reason)) {
        return reason != nullptr && !reason->empty()
            ? false
            : fail(reason, "V3 visual scene manifest is incomplete");
    }
    if (renderScene.fingerprint == 0u ||
        environmentMapHash != renderScene.environment.contentHash ||
        lightRigHash != renderScene.lightRig.contentHash ||
        fingerprint != computeVisualSceneManifestV3Fingerprint(*this)) {
        return fail(
            reason,
            "V3 visual scene manifest provenance does not match its "
            "render scene"
        );
    }
    return true;
}

VisualRendererProfileV1 VisualRendererProfileV1::sensorFast() {
    VisualRendererProfileV1 result;
    result.id = "sensor_fast";
    result.kind = MR_VISUAL_RENDERER_SENSOR_FAST;
    result.temporalSamples = 2u;
    result.rollingShutterBands = 16u;
    result.shadowMapResolution = 128u;
    result.areaLightSamples = 4u;
    result.rayQueryVisibility = false;
    result.retainObservation = false;
    result.fingerprint =
        computeVisualRendererProfileFingerprint(result);
    return result;
}

VisualRendererProfileV1 VisualRendererProfileV1::sensorReference() {
    VisualRendererProfileV1 result;
    result.id = "sensor_reference";
    result.kind = MR_VISUAL_RENDERER_SENSOR_REFERENCE;
    result.temporalSamples = 8u;
    result.rollingShutterBands = 0u;
    result.shadowMapResolution = 512u;
    result.areaLightSamples = 8u;
    result.rayQueryVisibility = true;
    result.retainObservation = true;
    result.fingerprint =
        computeVisualRendererProfileFingerprint(result);
    return result;
}

bool VisualRendererProfileV1::valid(std::string* reason) const {
    if (id.empty() || kind > MR_VISUAL_RENDERER_SENSOR_REFERENCE ||
        temporalSamples == 0u || temporalSamples > 64u ||
        rollingShutterBands > 4096u ||
        shadowMapResolution == 0u ||
        shadowMapResolution > 8192u ||
        areaLightSamples == 0u || areaLightSamples > 64u ||
        (kind == MR_VISUAL_RENDERER_SENSOR_FAST &&
         rayQueryVisibility) ||
        (kind == MR_VISUAL_RENDERER_SENSOR_REFERENCE &&
         !rayQueryVisibility)) {
        return fail(reason, "visual renderer profile is invalid");
    }
    if (fingerprint != 0u &&
        fingerprint !=
            computeVisualRendererProfileFingerprint(*this)) {
        return fail(
            reason,
            "visual renderer profile fingerprint does not match"
        );
    }
    return true;
}

bool VisualSensorProfileV2::valid(std::string* reason) const {
    if (id.empty() || !finite(nominalRateHz) ||
        !(nominalRateHz > 0.0) ||
        !finite(exposureSeconds) || exposureSeconds < 0.0 ||
        !finite(shutterReadoutSeconds) ||
        shutterReadoutSeconds < 0.0 ||
        !finite(frameJitterSeconds) ||
        frameJitterSeconds < 0.0 ||
        !finite(minimumDepthMeters) ||
        minimumDepthMeters < 0.0 ||
        !finite(maximumDepthMeters) ||
        !(maximumDepthMeters > minimumDepthMeters) ||
        !finite(depthQuantumMeters) ||
        !(depthQuantumMeters > 0.0) ||
        !finite(latencySeconds) || latencySeconds < 0.0 ||
        shutterModel > MR_VISUAL_SHUTTER_ROLLING ||
        shutterDirection > MR_VISUAL_SHUTTER_RIGHT_TO_LEFT ||
        (shutterModel == MR_VISUAL_SHUTTER_GLOBAL &&
         shutterReadoutSeconds != 0.0)) {
        return fail(reason, "V2 visual sensor profile is invalid");
    }
    return true;
}

bool VisualMotionSampleBatchV1::valid(std::string* reason) const {
    std::size_t perSample = 0u;
    if (schemaVersion != kVisualMotionSampleBatchVersion ||
        environmentCount == 0u || bodyCount == 0u ||
        sampleCount < 2u ||
        !finite(exposureOpenSeconds) ||
        !finite(exposureCloseSeconds) ||
        exposureOpenSeconds > exposureCloseSeconds ||
        timestampsSeconds.size() != sampleCount ||
        environmentCount >
            std::numeric_limits<std::size_t>::max() / bodyCount) {
        return fail(reason, "visual motion sample metadata is invalid");
    }
    perSample =
        static_cast<std::size_t>(environmentCount) * bodyCount;
    if (sampleCount >
            std::numeric_limits<std::size_t>::max() / perSample ||
        bodyStates.size() != perSample * sampleCount ||
        timestampsSeconds.front() > exposureOpenSeconds ||
        timestampsSeconds.back() < exposureCloseSeconds ||
        !std::ranges::is_sorted(timestampsSeconds)) {
        return fail(reason, "visual motion sample coverage is invalid");
    }
    for (const double timestamp : timestampsSeconds) {
        if (!finite(timestamp)) {
            return fail(reason, "visual motion timestamp is invalid");
        }
    }
    for (const MRBodyStateGPU& body : bodyStates) {
        if (!finite4(body.position) ||
            !unitQuaternion(body.orientation) ||
            !finite4(body.linearVelocityAndInverseMass) ||
            !finite4(body.angularVelocity)) {
            return fail(reason, "visual motion body state is invalid");
        }
    }
    return true;
}

std::span<const MRBodyStateGPU> VisualMotionSampleBatchV1::sample(
    const std::uint32_t sampleIndex
) const noexcept {
    if (sampleIndex >= sampleCount) {
        return {};
    }
    const std::size_t count =
        static_cast<std::size_t>(environmentCount) * bodyCount;
    return {
        bodyStates.data() + count * sampleIndex,
        count,
    };
}

bool writeVisualAssetPack(
    const VisualAssetPackV2& pack,
    const std::filesystem::path& path,
    std::string* reason
) {
    if (!pack.valid(reason)) {
        return false;
    }
    const std::vector<std::uint8_t> metadata =
        serializeAssetMetadata(pack);
    const std::vector<std::uint8_t> textureDescriptors =
        serializeTextureDescriptors(pack);
    const std::vector<std::uint8_t> symbolicBindings =
        serializeSymbolicBindings(pack);
    std::vector<SectionSourceV2> sections{
        {
            VisualAssetSectionKindV2::metadata,
            0u,
            metadata.data(),
            metadata.size(),
            1u,
            1u,
        },
        {
            VisualAssetSectionKindV2::vertices,
            0u,
            pack.vertices.data(),
            pack.vertices.size() * sizeof(MRVisualVertexGPUV2),
            pack.vertices.size(),
            sizeof(MRVisualVertexGPUV2),
        },
        {
            VisualAssetSectionKindV2::indices,
            0u,
            pack.indices.data(),
            pack.indices.size() * sizeof(std::uint32_t),
            pack.indices.size(),
            sizeof(std::uint32_t),
        },
        {
            VisualAssetSectionKindV2::primitives,
            0u,
            pack.primitives.data(),
            pack.primitives.size() *
                sizeof(MRVisualPrimitiveGPUV2),
            pack.primitives.size(),
            sizeof(MRVisualPrimitiveGPUV2),
        },
        {
            VisualAssetSectionKindV2::instances,
            0u,
            pack.instances.data(),
            pack.instances.size() * sizeof(MRVisualInstanceGPUV2),
            pack.instances.size(),
            sizeof(MRVisualInstanceGPUV2),
        },
        {
            VisualAssetSectionKindV2::materials,
            0u,
            pack.materials.data(),
            pack.materials.size() * sizeof(MRVisualMaterialGPUV2),
            pack.materials.size(),
            sizeof(MRVisualMaterialGPUV2),
        },
        {
            VisualAssetSectionKindV2::textureBindings,
            0u,
            pack.textureBindings.data(),
            pack.textureBindings.size() *
                sizeof(MRVisualTextureBindingGPUV2),
            pack.textureBindings.size(),
            sizeof(MRVisualTextureBindingGPUV2),
        },
        {
            VisualAssetSectionKindV2::textureDescriptors,
            0u,
            textureDescriptors.data(),
            textureDescriptors.size(),
            pack.textures.size(),
            0u,
        },
        {
            VisualAssetSectionKindV2::symbolicBindings,
            0u,
            symbolicBindings.data(),
            symbolicBindings.size(),
            pack.symbolicBindings.size(),
            0u,
        },
    };
    sections.reserve(sections.size() + pack.textures.size());
    for (std::uint32_t index = 0u;
         index < pack.textures.size();
         ++index) {
        const VisualTextureImageV2& texture = pack.textures[index];
        sections.push_back({
            VisualAssetSectionKindV2::texturePayload,
            index,
            texture.data.data(),
            texture.data.size(),
            texture.data.size(),
            1u,
        });
    }
    return writeSectionedPack(
        kPackMagic,
        pack.schemaVersion,
        pack.contentHash,
        sections,
        path,
        reason
    );
}

bool readVisualAssetPack(
    const std::filesystem::path& path,
    VisualAssetPackV2& output,
    std::string* reason
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail(reason, "could not open visual asset pack");
    }
    DiskPackHeaderV2 header;
    const auto diskDirectory = readDirectory(
        stream,
        kPackMagic,
        kVisualAssetPackVersion,
        header,
        reason
    );
    if (!diskDirectory.has_value()) {
        return false;
    }
    const DiskPackSectionV2* metadataSection =
        findSection(
            *diskDirectory,
            VisualAssetSectionKindV2::metadata
        );
    if (metadataSection == nullptr ||
        metadataSection->byteCount >
            std::numeric_limits<std::size_t>::max()) {
        return fail(reason, "visual pack metadata section is invalid");
    }
    std::vector<std::uint8_t> metadata(
        static_cast<std::size_t>(metadataSection->byteCount)
    );
    if (!readExact(
            stream,
            metadataSection->fileOffset,
            metadata.data(),
            metadata.size()
        ) ||
        sha256Bytes(metadata.data(), metadata.size()) !=
            metadataSection->contentHash) {
        return fail(reason, "visual pack metadata checksum failed");
    }
    BinaryReader reader{metadata};
    VisualAssetPackV2 candidate;
    std::uint64_t textureCount = 0u;
    std::uint64_t bindingCount = 0u;
    if (!reader.scalar(candidate.schemaVersion) ||
        !reader.string(candidate.id) ||
        !reader.string(candidate.sourceUri) ||
        !reader.string(candidate.sourceContentHash) ||
        !reader.string(candidate.contentHash) ||
        !reader.string(candidate.license) ||
        !reader.string(candidate.preprocessingProvenance) ||
        !reader.string(candidate.coordinateConvention) ||
        !reader.scalar(textureCount) ||
        !reader.scalar(bindingCount) ||
        textureCount > std::numeric_limits<std::uint32_t>::max() ||
        bindingCount > std::numeric_limits<std::uint32_t>::max() ||
        !reader.complete()) {
        return fail(reason, "visual asset metadata is truncated");
    }
    std::vector<std::uint8_t> textureDescriptorBytes;
    std::vector<std::uint8_t> symbolicBindingBytes;
    const DiskPackSectionV2* textureDescriptorSection =
        findSection(
            *diskDirectory,
            VisualAssetSectionKindV2::textureDescriptors
        );
    const DiskPackSectionV2* symbolicBindingSection =
        findSection(
            *diskDirectory,
            VisualAssetSectionKindV2::symbolicBindings
        );
    if (textureDescriptorSection == nullptr ||
        textureDescriptorSection->elementCount != textureCount ||
        textureDescriptorSection->elementStride != 0u ||
        symbolicBindingSection == nullptr ||
        symbolicBindingSection->elementCount != bindingCount ||
        symbolicBindingSection->elementStride != 0u) {
        return fail(
            reason,
            "visual descriptor section layout is invalid"
        );
    }
    if (!readByteSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::textureDescriptors,
            textureDescriptorBytes,
            reason
        ) ||
        !readByteSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::symbolicBindings,
            symbolicBindingBytes,
            reason
        )) {
        return false;
    }
    candidate.textures.resize(
        static_cast<std::size_t>(textureCount)
    );
    BinaryReader textureReader{textureDescriptorBytes};
    for (VisualTextureImageV2& texture : candidate.textures) {
        if (!readTextureMetadata(textureReader, texture)) {
            return fail(reason, "visual texture metadata is truncated");
        }
    }
    if (!textureReader.complete()) {
        return fail(reason, "visual texture metadata has trailing data");
    }
    candidate.symbolicBindings.resize(
        static_cast<std::size_t>(bindingCount)
    );
    BinaryReader bindingReader{symbolicBindingBytes};
    for (VisualSymbolicBindingV2& binding :
         candidate.symbolicBindings) {
        if (!readBinding(bindingReader, binding)) {
            return fail(reason, "visual binding payload is truncated");
        }
    }
    if (!bindingReader.complete() ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::vertices,
            candidate.vertices,
            reason
        ) ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::indices,
            candidate.indices,
            reason
        ) ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::primitives,
            candidate.primitives,
            reason
        ) ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::instances,
            candidate.instances,
            reason
        ) ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::materials,
            candidate.materials,
            reason
        ) ||
        !readVectorSection(
            stream,
            *diskDirectory,
            VisualAssetSectionKindV2::textureBindings,
            candidate.textureBindings,
            reason
        )) {
        return false;
    }
    candidate.sections.reserve(diskDirectory->size());
    for (const DiskPackSectionV2& disk : *diskDirectory) {
        candidate.sections.push_back({
            static_cast<VisualAssetSectionKindV2>(disk.kind),
            disk.index,
            disk.fileOffset,
            disk.byteCount,
            disk.elementCount,
            disk.elementStride,
            hexSha256(disk.contentHash),
        });
    }
    for (std::uint32_t index = 0u;
         index < candidate.textures.size();
         ++index) {
        const DiskPackSectionV2* section = findSection(
            *diskDirectory,
            VisualAssetSectionKindV2::texturePayload,
            index
        );
        if (section == nullptr ||
            section->byteCount >
                std::numeric_limits<std::size_t>::max()) {
            return fail(reason, "visual texture section is invalid");
        }
        VisualTextureImageV2& texture =
            candidate.textures[index];
        texture.data.resize(
            static_cast<std::size_t>(section->byteCount)
        );
        if (!readExact(
                stream,
                section->fileOffset,
                texture.data.data(),
                texture.data.size()
            ) ||
            sha256Bytes(texture.data.data(), texture.data.size()) !=
                section->contentHash) {
            return fail(reason, "visual texture checksum failed");
        }
    }
    if (hexSha256(header.contentHash) != candidate.contentHash ||
        !candidate.valid(reason)) {
        return false;
    }
    output = std::move(candidate);
    return true;
}

bool readVisualAssetPackIndex(
    const std::filesystem::path& path,
    VisualAssetPackV2& output,
    std::string* reason
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail(reason, "could not open visual asset pack");
    }
    DiskPackHeaderV2 header;
    const auto directory = readDirectory(
        stream,
        kPackMagic,
        kVisualAssetPackVersion,
        header,
        reason
    );
    if (!directory.has_value()) {
        return false;
    }
    const DiskPackSectionV2* metadataSection = findSection(
        *directory,
        VisualAssetSectionKindV2::metadata
    );
    if (metadataSection == nullptr ||
        metadataSection->byteCount >
            std::numeric_limits<std::size_t>::max()) {
        return fail(reason, "visual pack metadata section is invalid");
    }
    std::vector<std::uint8_t> metadata(
        static_cast<std::size_t>(metadataSection->byteCount)
    );
    if (!readExact(
            stream,
            metadataSection->fileOffset,
            metadata.data(),
            metadata.size()
        ) ||
        sha256Bytes(metadata.data(), metadata.size()) !=
            metadataSection->contentHash) {
        return fail(reason, "visual pack metadata checksum failed");
    }

    VisualAssetPackV2 candidate;
    BinaryReader reader{metadata};
    std::uint64_t textureCount = 0u;
    std::uint64_t symbolicBindingCount = 0u;
    if (!reader.scalar(candidate.schemaVersion) ||
        !reader.string(candidate.id) ||
        !reader.string(candidate.sourceUri) ||
        !reader.string(candidate.sourceContentHash) ||
        !reader.string(candidate.contentHash) ||
        !reader.string(candidate.license) ||
        !reader.string(candidate.preprocessingProvenance) ||
        !reader.string(candidate.coordinateConvention) ||
        !reader.scalar(textureCount) ||
        !reader.scalar(symbolicBindingCount) ||
        textureCount >
            std::numeric_limits<std::uint32_t>::max() ||
        symbolicBindingCount >
            std::numeric_limits<std::uint32_t>::max() ||
        !reader.complete()) {
        return fail(reason, "visual asset metadata is truncated");
    }
    std::vector<std::uint8_t> textureDescriptorBytes;
    std::vector<std::uint8_t> symbolicBindingBytes;
    const DiskPackSectionV2* textureDescriptorSection =
        findSection(
            *directory,
            VisualAssetSectionKindV2::textureDescriptors
        );
    const DiskPackSectionV2* symbolicBindingSection =
        findSection(
            *directory,
            VisualAssetSectionKindV2::symbolicBindings
        );
    if (textureDescriptorSection == nullptr ||
        textureDescriptorSection->elementCount != textureCount ||
        textureDescriptorSection->elementStride != 0u ||
        symbolicBindingSection == nullptr ||
        symbolicBindingSection->elementCount !=
            symbolicBindingCount ||
        symbolicBindingSection->elementStride != 0u) {
        return fail(
            reason,
            "visual descriptor section layout is invalid"
        );
    }
    if (!readByteSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::textureDescriptors,
            textureDescriptorBytes,
            reason
        ) ||
        !readByteSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::symbolicBindings,
            symbolicBindingBytes,
            reason
        )) {
        return false;
    }
    candidate.textures.resize(
        static_cast<std::size_t>(textureCount)
    );
    BinaryReader textureReader{textureDescriptorBytes};
    for (VisualTextureImageV2& texture : candidate.textures) {
        if (!readTextureMetadata(textureReader, texture)) {
            return fail(reason, "visual texture metadata is truncated");
        }
    }
    if (!textureReader.complete()) {
        return fail(reason, "visual texture metadata has trailing data");
    }
    candidate.symbolicBindings.resize(
        static_cast<std::size_t>(symbolicBindingCount)
    );
    BinaryReader bindingReader{symbolicBindingBytes};
    for (VisualSymbolicBindingV2& binding :
         candidate.symbolicBindings) {
        if (!readBinding(bindingReader, binding)) {
            return fail(reason, "visual binding payload is truncated");
        }
    }
    if (!bindingReader.complete() ||
        !readVectorSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::primitives,
            candidate.primitives,
            reason
        ) ||
        !readVectorSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::instances,
            candidate.instances,
            reason
        ) ||
        !readVectorSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::materials,
            candidate.materials,
            reason
        ) ||
        !readVectorSection(
            stream,
            *directory,
            VisualAssetSectionKindV2::textureBindings,
            candidate.textureBindings,
            reason
        )) {
        return false;
    }

    const DiskPackSectionV2* vertexSection = findSection(
        *directory,
        VisualAssetSectionKindV2::vertices
    );
    const DiskPackSectionV2* indexSection = findSection(
        *directory,
        VisualAssetSectionKindV2::indices
    );
    if (vertexSection == nullptr ||
        vertexSection->elementStride !=
            sizeof(MRVisualVertexGPUV2) ||
        vertexSection->elementCount == 0u ||
        vertexSection->byteCount !=
            vertexSection->elementCount *
                sizeof(MRVisualVertexGPUV2) ||
        indexSection == nullptr ||
        indexSection->elementStride != sizeof(std::uint32_t) ||
        indexSection->elementCount == 0u ||
        indexSection->elementCount % 3u != 0u ||
        indexSection->byteCount !=
            indexSection->elementCount * sizeof(std::uint32_t)) {
        return fail(reason, "visual geometry section layout is invalid");
    }

    const auto textureMetadataValid = [](
        const VisualTextureImageV2& texture,
        const std::uint64_t payloadBytes
    ) {
        const std::uint32_t bytesPerPixel =
            texture.pixelFormat ==
                    VisualTexturePixelFormatV2::rgba16Float
                ? 8u
                : 4u;
        const std::uint32_t slices =
            texture.dimension == VisualTextureDimensionV2::cube
            ? 6u
            : 1u;
        if (texture.id.empty() || texture.contentHash.empty() ||
            texture.width == 0u || texture.height == 0u ||
            texture.mipCount == 0u ||
            texture.arrayLength != slices ||
            texture.subresources.size() !=
                static_cast<std::size_t>(texture.mipCount) *
                    slices) {
            return false;
        }
        std::uint64_t expectedOffset = 0u;
        for (std::uint32_t slice = 0u;
             slice < slices;
             ++slice) {
            for (std::uint32_t mip = 0u;
                 mip < texture.mipCount;
                 ++mip) {
                const auto& subresource =
                    texture.subresources[
                        static_cast<std::size_t>(slice) *
                            texture.mipCount +
                        mip
                    ];
                const std::uint32_t width =
                    std::max(texture.width >> mip, 1u);
                const std::uint32_t height =
                    std::max(texture.height >> mip, 1u);
                if (subresource.mipLevel != mip ||
                    subresource.arraySlice != slice ||
                    subresource.width != width ||
                    subresource.height != height ||
                    subresource.dataOffset != expectedOffset ||
                    subresource.bytesPerRow <
                        static_cast<std::uint64_t>(width) *
                            bytesPerPixel ||
                    subresource.bytesPerImage <
                        static_cast<std::uint64_t>(
                            subresource.bytesPerRow
                        ) * height ||
                    subresource.dataSize !=
                        subresource.bytesPerImage) {
                    return false;
                }
                expectedOffset += subresource.dataSize;
            }
        }
        return expectedOffset == payloadBytes;
    };

    for (std::uint32_t index = 0u;
         index < candidate.textures.size();
         ++index) {
        const DiskPackSectionV2* payload = findSection(
            *directory,
            VisualAssetSectionKindV2::texturePayload,
            index
        );
        if (payload == nullptr ||
            !textureMetadataValid(
                candidate.textures[index],
                payload->byteCount
            )) {
            return fail(reason, "visual texture index is invalid");
        }
    }
    for (const MRVisualTextureBindingGPUV2& binding :
         candidate.textureBindings) {
        if (binding.resource.x >= candidate.textures.size() ||
            binding.resource.y >= 108u ||
            binding.resource.z > 1u ||
            !finite4(binding.uvTransform0) ||
            !finite4(binding.uvTransform1)) {
            return fail(reason, "visual texture binding is invalid");
        }
    }
    const auto validBinding = [&candidate](
        const std::uint32_t value
    ) {
        return value == MR_INVALID_INDEX ||
            value < candidate.textureBindings.size();
    };
    for (const MRVisualMaterialGPUV2& material :
         candidate.materials) {
        if (!validMaterial(material) ||
            !validBinding(material.textureIndices0.x) ||
            !validBinding(material.textureIndices0.y) ||
            !validBinding(material.textureIndices0.z) ||
            !validBinding(material.textureIndices0.w) ||
            !validBinding(material.textureIndices1.x) ||
            !validBinding(material.textureIndices1.y) ||
            !validBinding(material.textureIndices1.z) ||
            !validBinding(material.reserved.x) ||
            !validBinding(material.reserved.y)) {
            return fail(reason, "visual material index is invalid");
        }
    }
    for (std::size_t instanceIndex = 0u;
         instanceIndex < candidate.instances.size();
         ++instanceIndex) {
        const MRVisualInstanceGPUV2& instance =
            candidate.instances[instanceIndex];
        if (!finite4(instance.translationAndScale) ||
            !unitQuaternion(instance.orientation) ||
            !(instance.translationAndScale.w > 0.0f) ||
            instance.binding.z >
                MR_VISUAL_BINDING_ARTICULATED_LINK ||
            static_cast<std::uint64_t>(instance.geometry.x) +
                    instance.geometry.y >
                candidate.primitives.size()) {
            return fail(reason, "visual instance index is invalid");
        }
    }
    for (std::size_t primitiveIndex = 0u;
         primitiveIndex < candidate.primitives.size();
         ++primitiveIndex) {
        const MRVisualPrimitiveGPUV2& primitive =
            candidate.primitives[primitiveIndex];
        if (primitive.geometry.y == 0u ||
            primitive.geometry.y % 3u != 0u ||
            static_cast<std::uint64_t>(primitive.geometry.x) +
                    primitive.geometry.y >
                indexSection->elementCount ||
            primitive.geometry.z >= candidate.materials.size() ||
            primitive.geometry.w >= candidate.instances.size() ||
            !finite4(primitive.boundsMinimum) ||
            !finite4(primitive.boundsMaximum)) {
            return fail(reason, "visual primitive index is invalid");
        }
    }
    for (const VisualSymbolicBindingV2& binding :
         candidate.symbolicBindings) {
        if (!binding.valid(
                static_cast<std::uint32_t>(
                    candidate.instances.size()
                ),
                reason
            )) {
            return false;
        }
    }

    candidate.sections.reserve(directory->size());
    for (const DiskPackSectionV2& disk : *directory) {
        candidate.sections.push_back({
            static_cast<VisualAssetSectionKindV2>(disk.kind),
            disk.index,
            disk.fileOffset,
            disk.byteCount,
            disk.elementCount,
            disk.elementStride,
            hexSha256(disk.contentHash),
        });
    }
    if (hexSha256(header.contentHash) != candidate.contentHash) {
        return fail(reason, "visual pack content hash is inconsistent");
    }
    output = std::move(candidate);
    return true;
}

std::string computeVisualAssetPackContentHash(
    const VisualAssetPackV2& pack
) {
    return sha256ContentHash(pack);
}

bool appendVisualAssetPackReference(
    const std::filesystem::path& packPath,
    const std::uint32_t assetIndex,
    const std::uint32_t semanticId,
    const std::uint32_t instanceId,
    VisualRenderSceneV3& scene,
    std::string* reason
) {
    VisualAssetPackV2 pack;
    if (!readVisualAssetPackIndex(packPath, pack, reason) ||
        assetIndex >= scene.assetCount ||
        semanticId == 0u || semanticId == MR_INVALID_INDEX ||
        instanceId == 0u || instanceId == MR_INVALID_INDEX) {
        return reason != nullptr && !reason->empty()
            ? false
            : fail(reason, "visual pack scene binding is invalid");
    }
    scene.visualPacks.push_back({
        packPath,
        pack.contentHash,
        assetIndex,
        semanticId,
        instanceId,
    });
    scene.fingerprint = 0u;
    return true;
}

std::string computeVisualEnvironmentPackContentHash(
    const VisualEnvironmentPackV2& pack
) {
    Sha256Builder hash;
    hash.scalar(pack.schemaVersion);
    hash.string(pack.id);
    hash.string(pack.sourceUri);
    hash.string(pack.sourceContentHash);
    hash.string(pack.sourceColorSpace);
    hash.string(pack.preprocessingProvenance);
    hash.scalar(pack.specularFaceSize);
    hash.scalar(pack.diffuseFaceSize);
    hash.scalar(pack.brdfLutSize);
    const auto texture = [&hash](
        const VisualTextureImageV2& value
    ) {
        hash.string(value.id);
        hash.string(value.contentHash);
        hash.scalar(value.width);
        hash.scalar(value.height);
        hash.scalar(value.mipCount);
        hash.scalar(value.arrayLength);
        hash.scalar(value.pixelFormat);
        hash.scalar(value.dimension);
        hash.scalar(value.flags);
        hash.values<VisualTextureSubresourceV2>(
            value.subresources
        );
        hash.values<std::uint8_t>(value.data);
    };
    texture(pack.diffuseIrradiance);
    texture(pack.prefilteredSpecular);
    texture(pack.brdfLut);
    return hexSha256(hash.finishBytes());
}

bool writeVisualEnvironmentPack(
    const VisualEnvironmentPackV2& pack,
    const std::filesystem::path& path,
    std::string* reason
) {
    if (!pack.valid(reason)) {
        return false;
    }
    BinaryWriter writer;
    writer.scalar(pack.schemaVersion);
    writer.string(pack.id);
    writer.string(pack.sourceUri);
    writer.string(pack.sourceContentHash);
    writer.string(pack.contentHash);
    writer.string(pack.sourceColorSpace);
    writer.string(pack.preprocessingProvenance);
    writer.scalar(pack.specularFaceSize);
    writer.scalar(pack.diffuseFaceSize);
    writer.scalar(pack.brdfLutSize);
    writeTextureMetadata(writer, pack.diffuseIrradiance);
    writeTextureMetadata(writer, pack.prefilteredSpecular);
    writeTextureMetadata(writer, pack.brdfLut);
    const std::vector<std::uint8_t> metadata = writer.bytes();
    const std::array<const VisualTextureImageV2*, 3u> textures{
        &pack.diffuseIrradiance,
        &pack.prefilteredSpecular,
        &pack.brdfLut,
    };
    std::vector<SectionSourceV2> sections;
    sections.reserve(4u);
    sections.push_back({
        VisualAssetSectionKindV2::metadata,
        0u,
        metadata.data(),
        metadata.size(),
        1u,
        1u,
    });
    for (std::uint32_t index = 0u;
         index < textures.size();
         ++index) {
        sections.push_back({
            VisualAssetSectionKindV2::texturePayload,
            index,
            textures[index]->data.data(),
            textures[index]->data.size(),
            textures[index]->data.size(),
            1u,
        });
    }
    return writeSectionedPack(
        kEnvironmentMagic,
        pack.schemaVersion,
        pack.contentHash,
        sections,
        path,
        reason
    );
}

bool readVisualEnvironmentPack(
    const std::filesystem::path& path,
    VisualEnvironmentPackV2& output,
    std::string* reason
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail(reason, "could not open visual environment pack");
    }
    DiskPackHeaderV2 header;
    const auto directory = readDirectory(
        stream,
        kEnvironmentMagic,
        kVisualEnvironmentPackVersion,
        header,
        reason
    );
    if (!directory.has_value()) {
        return false;
    }
    const DiskPackSectionV2* metadataSection =
        findSection(
            *directory,
            VisualAssetSectionKindV2::metadata
        );
    if (metadataSection == nullptr ||
        metadataSection->byteCount >
            std::numeric_limits<std::size_t>::max()) {
        return fail(reason, "environment metadata section is invalid");
    }
    std::vector<std::uint8_t> metadata(
        static_cast<std::size_t>(metadataSection->byteCount)
    );
    if (!readExact(
            stream,
            metadataSection->fileOffset,
            metadata.data(),
            metadata.size()
        ) ||
        sha256Bytes(metadata.data(), metadata.size()) !=
            metadataSection->contentHash) {
        return fail(reason, "environment metadata checksum failed");
    }
    BinaryReader reader{metadata};
    VisualEnvironmentPackV2 candidate;
    if (!reader.scalar(candidate.schemaVersion) ||
        !reader.string(candidate.id) ||
        !reader.string(candidate.sourceUri) ||
        !reader.string(candidate.sourceContentHash) ||
        !reader.string(candidate.contentHash) ||
        !reader.string(candidate.sourceColorSpace) ||
        !reader.string(candidate.preprocessingProvenance) ||
        !reader.scalar(candidate.specularFaceSize) ||
        !reader.scalar(candidate.diffuseFaceSize) ||
        !reader.scalar(candidate.brdfLutSize) ||
        !readTextureMetadata(
            reader,
            candidate.diffuseIrradiance
        ) ||
        !readTextureMetadata(
            reader,
            candidate.prefilteredSpecular
        ) ||
        !readTextureMetadata(reader, candidate.brdfLut) ||
        !reader.complete()) {
        return fail(reason, "environment metadata is truncated");
    }
    const std::array<VisualTextureImageV2*, 3u> textures{
        &candidate.diffuseIrradiance,
        &candidate.prefilteredSpecular,
        &candidate.brdfLut,
    };
    candidate.sections.reserve(directory->size());
    for (const DiskPackSectionV2& disk : *directory) {
        candidate.sections.push_back({
            static_cast<VisualAssetSectionKindV2>(disk.kind),
            disk.index,
            disk.fileOffset,
            disk.byteCount,
            disk.elementCount,
            disk.elementStride,
            hexSha256(disk.contentHash),
        });
    }
    for (std::uint32_t index = 0u;
         index < textures.size();
         ++index) {
        const DiskPackSectionV2* section = findSection(
            *directory,
            VisualAssetSectionKindV2::texturePayload,
            index
        );
        if (section == nullptr ||
            section->byteCount >
                std::numeric_limits<std::size_t>::max()) {
            return fail(reason, "environment texture section is invalid");
        }
        textures[index]->data.resize(
            static_cast<std::size_t>(section->byteCount)
        );
        if (!readExact(
                stream,
                section->fileOffset,
                textures[index]->data.data(),
                textures[index]->data.size()
            ) ||
            sha256Bytes(
                textures[index]->data.data(),
                textures[index]->data.size()
            ) != section->contentHash) {
            return fail(reason, "environment texture checksum failed");
        }
    }
    if (hexSha256(header.contentHash) != candidate.contentHash ||
        !candidate.valid(reason)) {
        return false;
    }
    output = std::move(candidate);
    return true;
}

bool readVisualEnvironmentPackIndex(
    const std::filesystem::path& path,
    VisualEnvironmentPackV2& output,
    std::string* reason
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return fail(reason, "could not open visual environment pack");
    }
    DiskPackHeaderV2 header;
    const auto directory = readDirectory(
        stream,
        kEnvironmentMagic,
        kVisualEnvironmentPackVersion,
        header,
        reason
    );
    if (!directory.has_value()) {
        return false;
    }
    const DiskPackSectionV2* metadataSection = findSection(
        *directory,
        VisualAssetSectionKindV2::metadata
    );
    if (metadataSection == nullptr ||
        metadataSection->byteCount >
            std::numeric_limits<std::size_t>::max()) {
        return fail(reason, "environment metadata section is invalid");
    }
    std::vector<std::uint8_t> metadata(
        static_cast<std::size_t>(metadataSection->byteCount)
    );
    if (!readExact(
            stream,
            metadataSection->fileOffset,
            metadata.data(),
            metadata.size()
        ) ||
        sha256Bytes(metadata.data(), metadata.size()) !=
            metadataSection->contentHash) {
        return fail(reason, "environment metadata checksum failed");
    }
    VisualEnvironmentPackV2 candidate;
    BinaryReader reader{metadata};
    if (!reader.scalar(candidate.schemaVersion) ||
        !reader.string(candidate.id) ||
        !reader.string(candidate.sourceUri) ||
        !reader.string(candidate.sourceContentHash) ||
        !reader.string(candidate.contentHash) ||
        !reader.string(candidate.sourceColorSpace) ||
        !reader.string(candidate.preprocessingProvenance) ||
        !reader.scalar(candidate.specularFaceSize) ||
        !reader.scalar(candidate.diffuseFaceSize) ||
        !reader.scalar(candidate.brdfLutSize) ||
        !readTextureMetadata(
            reader,
            candidate.diffuseIrradiance
        ) ||
        !readTextureMetadata(
            reader,
            candidate.prefilteredSpecular
        ) ||
        !readTextureMetadata(reader, candidate.brdfLut) ||
        !reader.complete()) {
        return fail(reason, "environment metadata is truncated");
    }
    candidate.sections.reserve(directory->size());
    for (const DiskPackSectionV2& disk : *directory) {
        candidate.sections.push_back({
            static_cast<VisualAssetSectionKindV2>(disk.kind),
            disk.index,
            disk.fileOffset,
            disk.byteCount,
            disk.elementCount,
            disk.elementStride,
            hexSha256(disk.contentHash),
        });
    }
    const std::array<const VisualTextureImageV2*, 3u> textures{
        &candidate.diffuseIrradiance,
        &candidate.prefilteredSpecular,
        &candidate.brdfLut,
    };
    for (std::uint32_t index = 0u;
         index < textures.size();
         ++index) {
        const DiskPackSectionV2* payload = findSection(
            *directory,
            VisualAssetSectionKindV2::texturePayload,
            index
        );
        const VisualTextureImageV2& texture = *textures[index];
        if (payload == nullptr || texture.id.empty() ||
            texture.width == 0u || texture.height == 0u ||
            texture.mipCount == 0u ||
            texture.subresources.size() !=
                static_cast<std::size_t>(texture.mipCount) *
                    texture.arrayLength ||
            texture.subresources.empty()) {
            return fail(reason, "environment texture index is invalid");
        }
        const VisualTextureSubresourceV2& last =
            texture.subresources.back();
        if (last.dataOffset + last.dataSize !=
            payload->byteCount) {
            return fail(
                reason,
                "environment texture payload layout is invalid"
            );
        }
    }
    if (candidate.schemaVersion !=
            kVisualEnvironmentPackVersion ||
        candidate.id.empty() ||
        candidate.sourceContentHash.empty() ||
        candidate.specularFaceSize == 0u ||
        candidate.diffuseFaceSize == 0u ||
        candidate.brdfLutSize == 0u ||
        candidate.diffuseIrradiance.dimension !=
            VisualTextureDimensionV2::cube ||
        candidate.prefilteredSpecular.dimension !=
            VisualTextureDimensionV2::cube ||
        candidate.brdfLut.dimension !=
            VisualTextureDimensionV2::texture2D ||
        hexSha256(header.contentHash) != candidate.contentHash) {
        return fail(reason, "environment pack index is invalid");
    }
    output = std::move(candidate);
    return true;
}

VisualEnvironmentReferenceV2 makeNeutralStudioEnvironmentV2() {
    VisualEnvironmentReferenceV2 result;
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.scalar(result.intensity);
    hash.scalar(result.rotationRadians);
    result.fingerprint = hash.finish();
    return result;
}

VisualLightRigV1 makeStudioKeyLightRigV1() {
    VisualLightRigV1 result;
    result.lights = {defaultLight()};
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.values<MRVisualLightGPUV1>(result.lights);
    result.fingerprint = hash.finish();
    return result;
}

VisualLightRigV1 makeIndoorAreaLightRigV1() {
    VisualLightRigV1 result;
    result.id = "indoor_area";
    result.contentHash = "builtin:indoor-area-v1";
    MRVisualLightGPUV1 light = defaultLight();
    light.positionAndRange = {0.6f, -0.4f, 2.6f, 20.0f};
    light.directionAndSpot = {-0.25f, 0.20f, -0.95f, -1.0f};
    light.colorAndIntensity = {1.0f, 0.94f, 0.86f, 950.0f};
    light.shape = {0.9f, 0.7f, -1.0f, 0.08f};
    light.identity.x = MR_VISUAL_LIGHT_RECTANGLE;
    light.identity.y = MR_VISUAL_LIGHT_UNIT_NIT;
    light.shadow.x = 1u;
    result.lights = {light};
    HashBuilder hash;
    hash.string(result.id);
    hash.string(result.contentHash);
    hash.values<MRVisualLightGPUV1>(result.lights);
    result.fingerprint = hash.finish();
    return result;
}

std::uint64_t computeVisualRenderSceneV3Fingerprint(
    const VisualRenderSceneV3& scene
) {
    HashBuilder hash;
    hash.string(scene.id);
    hash.scalar(scene.assetCount);
    hash.scalar(scene.bodyCount);
    hash.values<MRHybridGaussianGPU>(scene.gaussians);
    for (const VisualAssetReferenceV3& reference :
         scene.visualPacks) {
        hash.string(reference.packPath.string());
        hash.string(reference.contentHash);
        hash.scalar(reference.assetIndex);
        hash.scalar(reference.semanticId);
        hash.scalar(reference.instanceId);
    }
    hash.values<MRVisualSensorBindingGPU>(scene.sensorBindings);
    hash.string(scene.environment.id);
    hash.string(scene.environment.packPath.string());
    hash.string(scene.environment.contentHash);
    hash.scalar(scene.environment.intensity);
    hash.scalar(scene.environment.rotationRadians);
    hash.string(scene.lightRig.id);
    hash.string(scene.lightRig.contentHash);
    hash.values<MRVisualLightGPUV1>(scene.lightRig.lights);
    return hash.finish();
}

std::uint64_t computeVisualSceneManifestV3Fingerprint(
    const VisualSceneManifestV3& manifest
) {
    HashBuilder hash;
    hash.scalar(manifest.schemaVersion);
    hash.string(manifest.id);
    hash.string(manifest.coordinateConvention);
    hash.scalar(manifest.worldFingerprint);
    for (const std::string& packHash : manifest.visualPackHashes) {
        hash.string(packHash);
    }
    hash.string(manifest.environmentMapHash);
    hash.string(manifest.lightRigHash);
    hash.string(manifest.preprocessingProvenance);
    hash.scalar(manifest.renderScene.fingerprint);
    return hash.finish();
}

bool writeVisualSceneManifestV3(
    const VisualSceneManifestV3& manifest,
    const std::filesystem::path& path,
    std::string* reason
) {
    std::string manifestReason;
    if (path.empty() || !manifest.valid(&manifestReason)) {
        return fail(
            reason,
            "V3 visual scene manifest input is invalid: " +
                manifestReason
        );
    }
    std::error_code error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(
            path.parent_path(),
            error
        );
        if (error) {
            return fail(
                reason,
                "could not create V3 visual manifest directory"
            );
        }
    }
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(manifest.fingerprint);
    std::ofstream output(
        temporary,
        std::ios::binary | std::ios::trunc
    );
    if (!output) {
        return fail(reason, "could not open V3 visual scene manifest");
    }
    output
        << "{\n"
        << "  \"schema_version\": 3,\n"
        << "  \"id\": \"" << jsonEscape(manifest.id) << "\",\n"
        << "  \"coordinate_convention\": \""
        << jsonEscape(manifest.coordinateConvention) << "\",\n"
        << "  \"world_fingerprint\": "
        << manifest.worldFingerprint << ",\n"
        << "  \"fingerprint\": " << manifest.fingerprint << ",\n"
        << "  \"visual_pack_hashes\": [";
    for (std::size_t index = 0u;
         index < manifest.visualPackHashes.size();
         ++index) {
        if (index != 0u) {
            output << ',';
        }
        output << '"' << jsonEscape(manifest.visualPackHashes[index])
               << '"';
    }
    const VisualRenderSceneV3& scene = manifest.renderScene;
    output
        << "],\n"
        << "  \"environment_map_hash\": \""
        << jsonEscape(manifest.environmentMapHash) << "\",\n"
        << "  \"light_rig_hash\": \""
        << jsonEscape(manifest.lightRigHash) << "\",\n"
        << "  \"preprocessing_provenance\": \""
        << jsonEscape(manifest.preprocessingProvenance) << "\",\n"
        << "  \"render_scene\": {\n"
        << "    \"id\": \"" << jsonEscape(scene.id) << "\",\n"
        << "    \"asset_count\": " << scene.assetCount << ",\n"
        << "    \"body_count\": " << scene.bodyCount << ",\n"
        << "    \"gaussian_count\": " << scene.gaussians.size() << ",\n"
        << "    \"visual_pack_count\": "
        << scene.visualPacks.size() << ",\n"
        << "    \"sensor_binding_count\": "
        << scene.sensorBindings.size() << ",\n"
        << "    \"light_count\": " << scene.lightRig.lights.size() << ",\n"
        << "    \"environment_id\": \""
        << jsonEscape(scene.environment.id) << "\",\n"
        << "    \"light_rig_id\": \""
        << jsonEscape(scene.lightRig.id) << "\",\n"
        << "    \"fingerprint\": " << scene.fingerprint << "\n"
        << "  }\n"
        << "}\n";
    output.close();
    if (!output) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not write V3 visual scene manifest");
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not publish V3 visual scene manifest");
    }
    return true;
}

std::uint64_t computeVisualRendererProfileFingerprint(
    const VisualRendererProfileV1& profile
) {
    HashBuilder hash;
    hash.string(profile.id);
    hash.scalar(profile.kind);
    hash.scalar(profile.temporalSamples);
    hash.scalar(profile.rollingShutterBands);
    hash.scalar(profile.shadowMapResolution);
    hash.scalar(profile.areaLightSamples);
    hash.scalar(profile.rayQueryVisibility);
    hash.scalar(profile.retainObservation);
    return hash.finish();
}

const char* visualAssetCookStatusName(
    const VisualAssetCookStatus status
) noexcept {
    switch (status) {
    case VisualAssetCookStatus::success:
        return "success";
    case VisualAssetCookStatus::ioFailure:
        return "io_failure";
    case VisualAssetCookStatus::unsupportedFormat:
        return "unsupported_format";
    case VisualAssetCookStatus::malformedAsset:
        return "malformed_asset";
    case VisualAssetCookStatus::unsupportedFeature:
        return "unsupported_feature";
    case VisualAssetCookStatus::invalidGeometry:
        return "invalid_geometry";
    case VisualAssetCookStatus::invalidMaterial:
        return "invalid_material";
    case VisualAssetCookStatus::invalidTexture:
        return "invalid_texture";
    case VisualAssetCookStatus::invalidBinding:
        return "invalid_binding";
    case VisualAssetCookStatus::capacityOverflow:
        return "capacity_overflow";
    case VisualAssetCookStatus::writeFailure:
        return "write_failure";
    case VisualAssetCookStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
