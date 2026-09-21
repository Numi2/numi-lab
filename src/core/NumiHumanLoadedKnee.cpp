#include "metalrobo/NumiHumanLoadedKnee.hpp"

#include "metalrobo/MatterSnapshotArchive.hpp"
#include "metalrobo/MetalWorld.hpp"

#include "numi/matter/detail.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/stdio.h>
#include <type_traits>
#include <utility>
#include <unistd.h>

namespace metalrobo {
namespace {

constexpr std::string_view kAuthoringSchema =
    "HumanPack.loaded-anatomy-knee.v1";
constexpr std::string_view kAuthoringCompiler =
    "numilab-human.loaded-anatomy-knee.1";
constexpr std::string_view kAcceptanceSchema =
    "numilab.human.loaded-knee-runtime-acceptance.v1";
constexpr std::string_view kMassDomain =
    "numi.lab.loaded-knee.mass-closure.v1";
constexpr std::string_view kCookedNodeMassDomain =
    "numi.lab.loaded-knee.cooked-node-mass.v1";
constexpr std::string_view kSnapshotDomain =
    "numi.lab.loaded-knee.matter-snapshot-authority.v1";
constexpr std::string_view kTransactionDomain =
    "numi.lab.loaded-knee.matter-transaction.v1";
constexpr std::string_view kMaterialExecutionDomain =
    "numi.lab.loaded-knee.material-execution.v1";
constexpr std::uint64_t kExactFEBioMatterSourceBytes = 2'702u;

constexpr std::array<std::string_view, kNumiHumanLoadedKneeRegionCount>
    kRegionNames{{"ACL", "LCL", "MCL", "PCL", "PTL", "QAT"}};
constexpr std::array<std::string_view, kNumiHumanLoadedKneeContactPairCount>
    kContactPairNames{{
        "TBC-L_To_FMC",
        "TBC-L_To_MNS-L",
        "PTC_To_FMC",
        "MNS-L_To_FMC",
        "MNS-M_To_TBC-M",
        "MNS-M_To_FMC",
        "TBC-M_To_FMC",
    }};
constexpr std::array<std::string_view, kNumiHumanLoadedKneeContactPairCount>
    kContactMasterSurfaces{{
        "TBC-L_@_FMC_ContactFaces",
        "TBC-L_@_MNS-L_ContactFaces",
        "PTC_@_FMC_ContactFaces",
        "MNS-L_@_FMC_ContactFaces",
        "MNS-M_@_TBC-M_ContactFaces",
        "MNS-M_@_FMC_ContactFaces",
        "TBC-M_@_FMC_ContactFaces",
    }};
constexpr std::array<std::string_view, kNumiHumanLoadedKneeContactPairCount>
    kContactSlaveSurfaces{{
        "FMC_@_TBC-L_ContactFaces",
        "MNS-L_@_TBC-L_ContactFaces",
        "FMC_@_PTC_ContactFaces",
        "FMC_@_MNS-L_ContactFaces",
        "TBC-M_@_MNS-M_ContactFaces",
        "FMC_@_MNS-M_ContactFaces",
        "FMC_@_TBC-M_ContactFaces",
    }};
constexpr std::array<std::string_view, kNumiHumanLoadedKneePassiveOwnerCount>
    kPassiveNames{{"ACL", "LCL", "MCL", "PCL", "PTL"}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kActuatorIndices{{405u, 413u, 414u, 415u}};
constexpr std::array<std::string_view,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kRouteNames{{"recfem_l", "vasint_l", "vaslat_l", "vasmed_l"}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kLoadEndpoints{{810u, 826u, 828u, 830u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kLoadRouteNodes{{1798u, 1830u, 1836u, 1842u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kLoadSourceSites{{1548u, 1715u, 1717u, 1719u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kLoadBodies{{128u, 145u, 145u, 145u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kAnchorEndpoints{{811u, 827u, 829u, 831u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kAnchorRouteNodes{{1803u, 1835u, 1841u, 1847u}};
constexpr std::array<std::uint32_t,
                     kNumiHumanLoadedKneeActiveReplacementCount>
    kAnchorSourceSites{{1751u, 1765u, 1766u, 1767u}};

struct ExpectedMaterial {
    double c1;
    double c2;
    double c3;
    double c4;
    double c5;
    double lambdaMaximum;
    double bulk;
    double initialStretch;
    std::array<double, 3u> homogeneousFiberWorld;
};

constexpr std::array<ExpectedMaterial, kNumiHumanLoadedKneeRegionCount>
    kExpectedMaterials{{
        {1.95e6, 0.0, 0.0139e6, 116.22, 535.039e6, 1.046,
         146.41e6, 1.016,
         {0.023879600688815117, 0.5208771824836731,
          0.8532975912094116}},
        {1.44e6, 0.0, 0.57e6, 48.0, 467.1e6, 1.063,
         793.65e6, 1.027,
         {-0.1950301229953766, -0.1955101639032364,
          0.961113452911377}},
        {1.44e6, 0.0, 0.57e6, 48.0, 467.1e6, 1.063,
         793.65e6, 1.034,
         {-0.28888770937919617, -0.0032872306182980537,
          0.9573573470115662}},
        {3.25e6, 0.0, 0.1196e6, 87.178, 431.063e6, 1.035,
         243.90e6, 1.0,
         {0.02508438006043434, 0.7640138864517212,
          -0.644711971282959}},
        {2.75e6, 0.0, 0.065e6, 115.89, 777.56e6, 1.042,
         206.61e6, 1.0,
         {-0.1883269101381302, -0.4512072503566742,
          0.872321605682373}},
        {2.75e6, 0.0, 0.065e6, 115.89, 777.56e6, 1.042,
         206.61e6, 1.0,
         {-0.035553932189941406, 0.22192594408988953,
          0.974415123462677}},
    }};

[[nodiscard]] bool fail(std::string& error, std::string message) {
    error = std::move(message);
    return false;
}

[[nodiscard]] bool digestPresent(
    const NumiHumanLoadedKneeDigest& digest
) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](const auto value) {
        return value != 0u;
    });
}

[[nodiscard]] bool finitePositive(const double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool closeTo(
    const double value,
    const double expected,
    const double relative = 2.0e-6
) noexcept {
    if (!std::isfinite(value) || !std::isfinite(expected)) return false;
    return std::abs(value - expected) <=
        relative * std::max({1.0, std::abs(value), std::abs(expected)});
}

[[nodiscard]] int hexNibble(const char value) noexcept {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return 10 + value - 'a';
    return -1;
}

[[nodiscard]] NumiHumanLoadedKneeDigest digestFromHex(
    const std::string_view text
) noexcept {
    NumiHumanLoadedKneeDigest result{};
    if (text.size() != 2u * result.size()) return {};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        const int high = hexNibble(text[2u * index]);
        const int low = hexNibble(text[2u * index + 1u]);
        if (high < 0 || low < 0) return {};
        result[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return result;
}

const NumiHumanLoadedKneeDigest kKneePayloadSHA256 = digestFromHex(
    "2e38201528de25911ea496164602ea7e823cd9c2e3c94efa550ee52689324ae5");
const NumiHumanLoadedKneeDigest kTendonPayloadSHA256 = digestFromHex(
    "72ca0ec4ef53f647784a89f761e78495d4997c7d72cfe6a1d9eedb24dd97feaa");
const NumiHumanLoadedKneeDigest kTopologyIdentitySHA256 = digestFromHex(
    "0569112a4d39eccc84bcb81a19a870b690cb2b3d642a74708d41881923f3c349");
const NumiHumanLoadedKneeDigest kExecutableFEMTopologySHA256 = digestFromHex(
    "56af24e63c9a4c2ecafb2c94bab98fa95bbcbc77085f425c84b9c1a95e3a9040");
const NumiHumanLoadedKneeDigest kSourceGlobalNodeIndexSHA256 = digestFromHex(
    "e451aa9e773a90dfb0cde77d4fb1555c7f1393c0362c7fbb3614909045a6602b");
const NumiHumanLoadedKneeDigest kAnchorOwnershipSHA256 = digestFromHex(
    "04a5c374f15f8681af40ae2e07fe35258f0bbc805d4d9d8291eae1f891d23de6");
const NumiHumanLoadedKneeDigest kSourceRigidPayloadSHA256 = digestFromHex(
    "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44");
const NumiHumanLoadedKneeDigest kEqualityPayloadSHA256 = digestFromHex(
    "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a");
const NumiHumanLoadedKneeDigest kAuthoringProfileFileSHA256 = digestFromHex(
    "ec311c48c7ab58dede83da228686d6574d8ebd1c47d22539dbee09967131d3f5");
const NumiHumanLoadedKneeDigest kAuthoringProfileIdentitySHA256 = digestFromHex(
    "c4bbde523018016b94c5e176fd01e519cfbb2a811f84005f964d0718753b6437");
const NumiHumanLoadedKneeDigest kExactFEBioMatterSourceSHA256 = digestFromHex(
    "c5166dfcdbcb8402cd107d087ac5071e548f8154b41f088bee3924e40ac9871e");

class SHA256Writer final {
public:
    SHA256Writer() noexcept : valid_(CC_SHA256_Init(&context_) == 1) {}

    [[nodiscard]] bool append(const void* bytes, std::size_t count) noexcept {
        if (!valid_ || (bytes == nullptr && count != 0u)) return false;
        const auto* cursor = static_cast<const std::uint8_t*>(bytes);
        while (count != 0u) {
            const auto chunk = static_cast<CC_LONG>(std::min<std::size_t>(
                count, std::numeric_limits<CC_LONG>::max()));
            if (CC_SHA256_Update(&context_, cursor, chunk) != 1) {
                valid_ = false;
                return false;
            }
            cursor += chunk;
            count -= chunk;
        }
        return true;
    }

    [[nodiscard]] bool appendU32(const std::uint32_t value) noexcept {
        std::array<std::uint8_t, 4u> bytes{};
        for (std::size_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (8u * index)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool appendU64(const std::uint64_t value) noexcept {
        std::array<std::uint8_t, 8u> bytes{};
        for (std::size_t index = 0u; index < bytes.size(); ++index) {
            bytes[index] = static_cast<std::uint8_t>(
                (value >> (8u * index)) & 0xffu);
        }
        return append(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool appendFloat(const float value) noexcept {
        return appendU32(std::bit_cast<std::uint32_t>(value));
    }

    [[nodiscard]] bool appendDouble(const double value) noexcept {
        return appendU64(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] bool appendString(const std::string_view value) noexcept {
        return appendU64(static_cast<std::uint64_t>(value.size())) &&
            append(value.data(), value.size());
    }

    [[nodiscard]] bool appendDigest(
        const NumiHumanLoadedKneeDigest& value
    ) noexcept {
        return append(value.data(), value.size());
    }

    template <typename T>
    [[nodiscard]] bool appendVector(
        const std::string_view typeName,
        const std::vector<T>& value
    ) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!appendString(typeName) ||
            !appendU64(static_cast<std::uint64_t>(sizeof(T))) ||
            !appendU64(static_cast<std::uint64_t>(value.size()))) return false;
        return value.empty() ||
            append(value.data(), value.size() * sizeof(T));
    }

    [[nodiscard]] bool finish(
        NumiHumanLoadedKneeDigest& output
    ) noexcept {
        if (!valid_ || CC_SHA256_Final(output.data(), &context_) != 1) {
            output.fill(0u);
            valid_ = false;
            return false;
        }
        valid_ = false;
        return true;
    }

private:
    CC_SHA256_CTX context_{};
    bool valid_ = false;
};

class ScopedFileDescriptor final {
public:
    explicit ScopedFileDescriptor(const int value) noexcept : value_(value) {}
    ~ScopedFileDescriptor() {
        if (value_ >= 0) ::close(value_);
    }
    ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
    ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

    [[nodiscard]] int get() const noexcept { return value_; }

    [[nodiscard]] bool closeChecked(
        const std::string_view context,
        std::string& error
    ) {
        if (value_ < 0) return true;
        const int value = std::exchange(value_, -1);
        if (::close(value) == 0) return true;
        const int closeError = errno;
        error = std::string(context) + ": " + std::strerror(closeError);
        return false;
    }

private:
    int value_ = -1;
};

class ScopedStagingPath final {
public:
    ScopedStagingPath(const int directory, std::string name) noexcept
        : directory_(directory), name_(std::move(name)) {}
    ~ScopedStagingPath() {
        if (active_) (void)::unlinkat(directory_, name_.c_str(), 0);
    }
    ScopedStagingPath(const ScopedStagingPath&) = delete;
    ScopedStagingPath& operator=(const ScopedStagingPath&) = delete;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void published() noexcept { active_ = false; }

private:
    int directory_ = -1;
    std::string name_;
    bool active_ = true;
};

[[nodiscard]] std::string errnoMessage(
    const std::string_view context,
    const int errorNumber
) {
    return std::string(context) + ": " + std::strerror(errorNumber);
}

[[nodiscard]] bool existingImmutableFileMatches(
    const int directory,
    const std::string& filename,
    const std::span<const std::byte> expected,
    bool& exists,
    std::string& error
) {
    const int value = ::openat(
        directory, filename.c_str(),
        O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
    if (value < 0) {
        const int openError = errno;
        if (openError == ENOENT) {
            exists = false;
            return true;
        }
        return fail(error, errnoMessage(
            "immutable loaded-knee destination could not be opened safely",
            openError));
    }
    exists = true;
    ScopedFileDescriptor descriptor(value);
    struct stat metadata {};
    if (::fstat(descriptor.get(), &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        static_cast<std::uint64_t>(metadata.st_size) != expected.size()) {
        return fail(error,
            "immutable loaded-knee destination is not a regular file with identical bytes");
    }
    std::array<std::byte, 64u * 1024u> buffer{};
    std::size_t offset = 0u;
    while (offset < expected.size()) {
        const std::size_t remaining = expected.size() - offset;
        const auto count = ::read(
            descriptor.get(), buffer.data(),
            std::min(remaining, buffer.size()));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            return fail(error,
                "immutable loaded-knee destination read was incomplete");
        }
        const std::size_t readCount = static_cast<std::size_t>(count);
        if (!std::equal(
                buffer.begin(), buffer.begin() + readCount,
                expected.begin() + offset)) {
            return fail(error,
                "immutable loaded-knee destination already exists with different bytes");
        }
        offset += readCount;
    }
    struct stat pathMetadata {};
    if (::fstatat(
            directory, filename.c_str(), &pathMetadata,
            AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(pathMetadata.st_mode) ||
        pathMetadata.st_dev != metadata.st_dev ||
        pathMetadata.st_ino != metadata.st_ino ||
        pathMetadata.st_size != metadata.st_size) {
        return fail(error,
            "immutable loaded-knee destination changed while it was verified");
    }
    error.clear();
    return true;
}

[[nodiscard]] std::string uniqueStagingName() {
    std::array<unsigned char, 16u> random{};
    ::arc4random_buf(random.data(), random.size());
    constexpr char hexadecimal[] = "0123456789abcdef";
    std::string name = ".numi-loaded-knee-stage-";
    name += std::to_string(static_cast<long long>(::getpid()));
    name += '-';
    name.reserve(name.size() + 2u * random.size());
    for (const auto value : random) {
        name.push_back(hexadecimal[value >> 4u]);
        name.push_back(hexadecimal[value & 0x0fu]);
    }
    return name;
}

[[nodiscard]] bool readVerifiedImmutableFile(
    const std::filesystem::path& path,
    const NumiHumanLoadedKneeDigest& expectedSHA256,
    const std::uint64_t expectedByteCount,
    std::vector<char>& output,
    std::string& error
) {
    if (!digestPresent(expectedSHA256) || expectedByteCount == 0u ||
        expectedByteCount > std::numeric_limits<std::size_t>::max()) {
        return fail(error, "immutable loaded-knee file expectation is invalid");
    }
    const ScopedFileDescriptor descriptor(::open(
        path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.get() < 0) {
        return fail(error,
            "immutable loaded-knee file could not be opened without following a link");
    }
    struct stat metadata {};
    if (::fstat(descriptor.get(), &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        static_cast<std::uint64_t>(metadata.st_size) != expectedByteCount) {
        return fail(error,
            "immutable loaded-knee file is not a regular file with the expected byte count");
    }
    std::vector<char> candidate(
        static_cast<std::size_t>(expectedByteCount));
    std::size_t offset = 0u;
    while (offset < candidate.size()) {
        const auto count = ::read(
            descriptor.get(), candidate.data() + offset,
            candidate.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            return fail(error,
                "immutable loaded-knee file read was incomplete");
        }
        offset += static_cast<std::size_t>(count);
    }
    SHA256Writer writer;
    NumiHumanLoadedKneeDigest observed{};
    if (!writer.append(candidate.data(), candidate.size()) ||
        !writer.finish(observed) || observed != expectedSHA256) {
        return fail(error,
            "immutable loaded-knee file SHA-256 does not match");
    }
    output = std::move(candidate);
    error.clear();
    return true;
}

struct DecodedExecutableTopology {
    std::vector<std::uint32_t> sourceGlobalNodeIndices;
    std::vector<std::array<std::uint32_t, 4u>> tetrahedra;
    NumiHumanLoadedKneeDigest sourceGlobalNodeIndexSHA256{};
    NumiHumanLoadedKneeDigest executableFEMTopologySHA256{};
    NumiHumanLoadedKneeDigest anchorOwnershipSHA256{};
};

[[nodiscard]] bool decodeExecutableTopology(
    const NumiHumanKneePayload& payload,
    DecodedExecutableTopology& output,
    std::string& error
) {
    if (payload.nodes.size() != kNumiHumanLoadedKneeNodeCount ||
        payload.tetrahedra.size() !=
            kNumiHumanLoadedKneeTetrahedronCount) {
        return fail(error,
            "loaded-knee executable topology requires the exact ABI3 payload");
    }
    DecodedExecutableTopology candidate;
    candidate.sourceGlobalNodeIndices.reserve(
        kNumiHumanLoadedKneeLoadedNodeCount);
    candidate.tetrahedra.reserve(
        kNumiHumanLoadedKneeLoadedTetrahedronCount);
    std::vector<std::uint32_t> globalToRuntime(
        payload.nodes.size(), NUMI_HUMAN_KNEE_INVALID_INDEX);
    SHA256Writer globalNodeWriter;
    SHA256Writer topologyWriter;
    SHA256Writer anchorWriter;
    for (const std::string_view name : kRegionNames) {
        const auto region = std::find_if(
            payload.regions.begin(), payload.regions.end(),
            [name](const NumiHumanKneeRegion& value) {
                return value.name == name;
            });
        if (region == payload.regions.end() ||
            region->firstNode > payload.nodes.size() ||
            region->nodeCount > payload.nodes.size() - region->firstNode ||
            region->firstTetrahedron > payload.tetrahedra.size() ||
            region->tetrahedronCount >
                payload.tetrahedra.size() - region->firstTetrahedron) {
            return fail(error,
                "loaded-knee ABI3 profile-region span is invalid");
        }
        for (std::uint32_t local = 0u; local < region->nodeCount; ++local) {
            const std::uint32_t global = region->firstNode + local;
            if (globalToRuntime[global] != NUMI_HUMAN_KNEE_INVALID_INDEX) {
                return fail(error,
                    "loaded-knee ABI3 profile regions overlap node ownership");
            }
            const std::uint32_t runtime = static_cast<std::uint32_t>(
                candidate.sourceGlobalNodeIndices.size());
            globalToRuntime[global] = runtime;
            candidate.sourceGlobalNodeIndices.push_back(global);
            const auto& node = payload.nodes[global];
            const std::uint32_t flags = node.rigidlyAttached ? 1u : 0u;
            if (!globalNodeWriter.appendU32(global) ||
                !anchorWriter.appendU32(global) ||
                !anchorWriter.appendU32(node.anchorBodyIndex) ||
                !anchorWriter.appendU32(flags) ||
                !anchorWriter.appendFloat(node.anchorLocal[0u]) ||
                !anchorWriter.appendFloat(node.anchorLocal[1u]) ||
                !anchorWriter.appendFloat(node.anchorLocal[2u])) {
                return fail(error,
                    "loaded-knee ABI3 node/anchor identity hashing failed");
            }
        }
        for (std::uint32_t local = 0u;
             local < region->tetrahedronCount; ++local) {
            const auto& source = payload.tetrahedra[
                region->firstTetrahedron + local];
            std::array<std::uint32_t, 4u> remapped{};
            for (std::size_t corner = 0u; corner < remapped.size(); ++corner) {
                if (source[corner] >= globalToRuntime.size() ||
                    globalToRuntime[source[corner]] ==
                        NUMI_HUMAN_KNEE_INVALID_INDEX) {
                    return fail(error,
                        "loaded-knee ABI3 tetrahedron is outside the permanent profile order");
                }
                remapped[corner] = globalToRuntime[source[corner]];
                if (!topologyWriter.appendU32(remapped[corner])) {
                    return fail(error,
                        "loaded-knee ABI3 executable topology hashing failed");
                }
            }
            candidate.tetrahedra.push_back(remapped);
        }
    }
    if (candidate.sourceGlobalNodeIndices.size() !=
            kNumiHumanLoadedKneeLoadedNodeCount ||
        candidate.tetrahedra.size() !=
            kNumiHumanLoadedKneeLoadedTetrahedronCount ||
        !globalNodeWriter.finish(
            candidate.sourceGlobalNodeIndexSHA256) ||
        !topologyWriter.finish(candidate.executableFEMTopologySHA256) ||
        !anchorWriter.finish(candidate.anchorOwnershipSHA256)) {
        return fail(error,
            "loaded-knee ABI3 executable topology identity is incomplete");
    }
    output = std::move(candidate);
    return true;
}

[[nodiscard]] bool validIdentity(
    const NumiHumanLoadedKneeImmutableIdentityV1& identity
) noexcept {
    return !identity.schema.empty() && digestPresent(identity.fileSHA256) &&
        digestPresent(identity.identitySHA256);
}

[[nodiscard]] bool validOwnerStrings(
    const NumiHumanLoadedKneeRegionV1& region
) noexcept {
    return !region.semanticID.empty() &&
        !region.physicalVolumeOwnerID.empty() &&
        !region.mechanicalMassOwnerID.empty() &&
        !region.materialOwnerID.empty() &&
        !region.stateOwnerID.empty();
}

[[nodiscard]] bool sameContactPair(
    const NumiHumanLoadedKneeContactPairV1& left,
    const NumiHumanLoadedKneeContactPairV1& right
) noexcept {
    return left.semanticID == right.semanticID &&
        left.sourcePairName == right.sourcePairName &&
        left.masterSurface == right.masterSurface &&
        left.slaveSurface == right.slaveSurface &&
        left.ownerID == right.ownerID;
}

[[nodiscard]] bool sameReplacement(
    const NumiHumanLoadedKneeActiveReplacementV1& left,
    const NumiHumanLoadedKneeActiveReplacementV1& right
) noexcept {
    return left.semanticID == right.semanticID &&
        left.sourceActuatorIndex == right.sourceActuatorIndex &&
        left.sourceRouteName == right.sourceRouteName &&
        left.loadEndpointIndex == right.loadEndpointIndex &&
        left.loadRouteNodeIndex == right.loadRouteNodeIndex &&
        left.loadSourceSiteIndex == right.loadSourceSiteIndex &&
        left.loadBodyIndex == right.loadBodyIndex &&
        left.anchorEndpointIndex == right.anchorEndpointIndex &&
        left.anchorRouteNodeIndex == right.anchorRouteNodeIndex &&
        left.anchorSourceSiteIndex == right.anchorSourceSiteIndex &&
        left.anchorBodyIndex == right.anchorBodyIndex &&
        left.sourceOwnerID == right.sourceOwnerID &&
        left.candidateOwnerID == right.candidateOwnerID &&
        left.mode == right.mode &&
        left.replacementFraction == right.replacementFraction;
}

[[nodiscard]] bool samePassiveOwner(
    const NumiHumanLoadedKneePassiveOwnerV1& left,
    const NumiHumanLoadedKneePassiveOwnerV1& right
) noexcept {
    return left.semanticID == right.semanticID &&
        left.sourceRegionName == right.sourceRegionName &&
        left.ownerID == right.ownerID;
}

[[nodiscard]] bool sameExpressionIntrinsic(
    const numi::matter::Expr& left,
    const numi::matter::Expr& right
) noexcept {
    return left.kind == right.kind &&
        left.dimension == right.dimension &&
        left.constant == right.constant && left.index == right.index &&
        left.integer == right.integer && left.arguments == right.arguments;
}

[[nodiscard]] bool sameMixedMaterialIntrinsic(
    const numi::matter::MixedMaterialSource& left,
    const numi::matter::MixedMaterialSource& right
) noexcept {
    return left.bulkModulus == right.bulkModulus &&
        left.thermalExpansion == right.thermalExpansion &&
        left.biotCoefficient == right.biotCoefficient &&
        left.referenceTemperature == right.referenceTemperature &&
        left.heatCapacity == right.heatCapacity &&
        left.thermalConductivity == right.thermalConductivity &&
        left.heatSource == right.heatSource &&
        left.jouleHeatFraction == right.jouleHeatFraction &&
        left.poreStorage == right.poreStorage &&
        left.poreMobility == right.poreMobility &&
        left.poreSource == right.poreSource &&
        left.electricalConductivity == right.electricalConductivity &&
        left.activationDiffusivity == right.activationDiffusivity &&
        left.activationOnRate == right.activationOnRate &&
        left.activationOffRate == right.activationOffRate &&
        left.fibreDirection == right.fibreDirection &&
        left.maximumActiveTension == right.maximumActiveTension &&
        left.activationThreshold == right.activationThreshold &&
        left.activationSlope == right.activationSlope &&
        left.cohesiveStrength == right.cohesiveStrength &&
        left.fractureEnergy == right.fractureEnergy;
}

// Material name, parameter values, identifiability, and the compiler-owned
// fingerprint are the only permitted regional overlays. Everything else must
// remain exactly the pinned FEBio exp-linear source IR.
[[nodiscard]] std::string exactFEBioIntrinsicMismatch(
    const numi::matter::MaterialProgram& observed,
    const numi::matter::MaterialProgram& source
) {
    if (observed.parameters.size() != source.parameters.size() ||
        observed.internalState.size() != source.internalState.size() ||
        observed.expressions.nodes.size() != source.expressions.nodes.size()) {
        return "program extent observed(parameters=" +
            std::to_string(observed.parameters.size()) +
            ",state=" + std::to_string(observed.internalState.size()) +
            ",expressions=" +
            std::to_string(observed.expressions.nodes.size()) +
            ") source(parameters=" +
            std::to_string(source.parameters.size()) +
            ",state=" + std::to_string(source.internalState.size()) +
            ",expressions=" +
            std::to_string(source.expressions.nodes.size()) + ")";
    }
    if (
        observed.energyRoot != source.energyRoot ||
        observed.dissipationRoot != source.dissipationRoot ||
        observed.validityRoot != source.validityRoot ||
        observed.stateUpdateRoots != source.stateUpdateRoots ||
        observed.stateImplicitRoots != source.stateImplicitRoots) {
        return "expression roots";
    }
    if (
        observed.supportedRepresentations !=
            source.supportedRepresentations ||
        observed.hint != source.hint ||
        !sameMixedMaterialIntrinsic(observed.mixed, source.mixed) ||
        observed.learned.has_value() || source.learned.has_value()) {
        return "representation, hint, mixed, or learned metadata";
    }
    if (
        observed.staticFriction != source.staticFriction ||
        observed.dynamicFriction != source.dynamicFriction ||
        observed.restitution != source.restitution ||
        observed.adhesion != source.adhesion ||
        observed.minimumDeterminant != source.minimumDeterminant ||
        observed.maximumDeterminant != source.maximumDeterminant ||
        observed.maximumStress != source.maximumStress ||
        observed.maximumEnergyDensity != source.maximumEnergyDensity) {
        return "contact or validity limits";
    }
    for (std::size_t index = 0u; index < source.parameters.size(); ++index) {
        const auto& actual = observed.parameters[index];
        const auto& expected = source.parameters[index];
        if (actual.name != expected.name ||
            actual.dimension != expected.dimension ||
            actual.lower != expected.lower || actual.upper != expected.upper ||
            actual.logarithmic != expected.logarithmic) {
            return "parameter metadata at " + std::to_string(index);
        }
    }
    for (std::size_t index = 0u; index < source.internalState.size(); ++index) {
        const auto& actual = observed.internalState[index];
        const auto& expected = source.internalState[index];
        if (actual.name != expected.name ||
            actual.dimension != expected.dimension ||
            actual.initialValue != expected.initialValue ||
            actual.transfer != expected.transfer) {
            return "internal-state metadata at " + std::to_string(index);
        }
    }
    for (std::size_t index = 0u;
         index < source.expressions.nodes.size(); ++index) {
        if (!sameExpressionIntrinsic(observed.expressions.nodes[index],
                                     source.expressions.nodes[index])) {
            return "expression node at " + std::to_string(index);
        }
    }
    return {};
}

[[nodiscard]] bool sameInstructionVector(
    const std::vector<NMExpressionInstructionGPU>& left,
    const std::vector<NMExpressionInstructionGPU>& right
) noexcept {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(
            left.data(), right.data(),
            left.size() * sizeof(NMExpressionInstructionGPU)) == 0);
}

[[nodiscard]] bool sameScalarBytecode(
    const numi::matter::ScalarBytecode& left,
    const numi::matter::ScalarBytecode& right
) noexcept {
    return left.maximumStack == right.maximumStack &&
        sameInstructionVector(left.instructions, right.instructions);
}

template <std::size_t Count>
[[nodiscard]] bool sameScalarBytecodeArray(
    const std::array<numi::matter::ScalarBytecode, Count>& left,
    const std::array<numi::matter::ScalarBytecode, Count>& right
) noexcept {
    for (std::size_t index = 0u; index < Count; ++index) {
        if (!sameScalarBytecode(left[index], right[index])) return false;
    }
    return true;
}

[[nodiscard]] bool sameScalarBytecodeVector(
    const std::vector<numi::matter::ScalarBytecode>& left,
    const std::vector<numi::matter::ScalarBytecode>& right
) noexcept {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        if (!sameScalarBytecode(left[index], right[index])) return false;
    }
    return true;
}

[[nodiscard]] bool sameCompiledIntrinsic(
    const numi::matter::ConstitutiveProgram& left,
    const numi::matter::ConstitutiveProgram& right
) noexcept {
    const auto sameOptional = [](const auto& a, const auto& b) {
        return a.has_value() == b.has_value() &&
            (!a.has_value() || sameScalarBytecode(*a, *b));
    };
    return sameScalarBytecodeArray(left.stress, right.stress) &&
        sameScalarBytecodeArray(left.tangentVector, right.tangentVector) &&
        sameScalarBytecodeArray(left.viscousStress, right.viscousStress) &&
        sameScalarBytecodeArray(
            left.viscousTangentVector, right.viscousTangentVector) &&
        sameScalarBytecodeVector(left.stateUpdates, right.stateUpdates) &&
        sameScalarBytecodeVector(
            left.implicitResiduals, right.implicitResiduals) &&
        sameScalarBytecodeVector(
            left.implicitJacobians, right.implicitJacobians) &&
        sameScalarBytecodeVector(
            left.implicitDeformationDirections,
            right.implicitDeformationDirections) &&
        sameScalarBytecodeVector(
            left.stressStateDerivatives, right.stressStateDerivatives) &&
        sameOptional(left.dissipation, right.dissipation) &&
        sameOptional(left.validity, right.validity);
}

[[nodiscard]] float sourcePressureFloat(const double pascals) noexcept {
    return 1.0e6f * static_cast<float>(pascals / 1.0e6);
}

[[nodiscard]] bool exactFloat(
    const float observed,
    const float expected
) noexcept {
    return std::bit_cast<std::uint32_t>(observed) ==
        std::bit_cast<std::uint32_t>(expected);
}

[[nodiscard]] bool finiteFloat4XYZ(const nm_float4& value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && exactFloat(value.w, 0.0f);
}

[[nodiscard]] bool sameFloat4Bits(
    const nm_float4& left,
    const nm_float4& right
) noexcept {
    return exactFloat(left.x, right.x) && exactFloat(left.y, right.y) &&
        exactFloat(left.z, right.z) && exactFloat(left.w, right.w);
}

[[nodiscard]] bool appendScalarBytecodeAuthority(
    SHA256Writer& writer,
    const std::string_view role,
    const numi::matter::ScalarBytecode& bytecode
) noexcept {
    return writer.appendString(role) &&
        writer.appendU32(bytecode.maximumStack) &&
        writer.appendVector(
            "NMExpressionInstructionGPU", bytecode.instructions);
}

[[nodiscard]] bool appendConstitutiveAuthority(
    SHA256Writer& writer,
    const numi::matter::ConstitutiveProgram& program
) noexcept {
    if (!writer.appendString(program.material.name) ||
        !writer.appendU64(program.fingerprint) ||
        !writer.appendString("NMMaterialGPU") ||
        !writer.appendU64(sizeof(program.gpu)) ||
        !writer.append(&program.gpu, sizeof(program.gpu)) ||
        !writer.appendVector("NMParameterRangeGPU", program.parameters)) {
        return false;
    }
    for (const auto& parameter : program.material.parameters) {
        if (!writer.appendString(parameter.name) ||
            !writer.appendDouble(parameter.defaultValue) ||
            !writer.appendU32(parameter.identifiable ? 1u : 0u)) return false;
    }
    for (std::size_t index = 0u; index < program.stress.size(); ++index) {
        if (!appendScalarBytecodeAuthority(
                writer, "stress-" + std::to_string(index),
                program.stress[index]) ||
            !appendScalarBytecodeAuthority(
                writer, "tangent-" + std::to_string(index),
                program.tangentVector[index]) ||
            !appendScalarBytecodeAuthority(
                writer, "viscous-stress-" + std::to_string(index),
                program.viscousStress[index]) ||
            !appendScalarBytecodeAuthority(
                writer, "viscous-tangent-" + std::to_string(index),
                program.viscousTangentVector[index])) return false;
    }
    if (program.dissipation.has_value() &&
        !appendScalarBytecodeAuthority(
            writer, "dissipation", *program.dissipation)) return false;
    return !program.validity.has_value() ||
        appendScalarBytecodeAuthority(writer, "validity", *program.validity);
}

[[nodiscard]] bool validateMaterial(
    const NumiHumanLoadedKneeMaterialV1& material,
    const ExpectedMaterial& expected
) noexcept {
    for (std::size_t axis = 0u;
         axis < material.homogeneousFiberWorld.size(); ++axis) {
        const double value = material.homogeneousFiberWorld[axis];
        if (!std::isfinite(value) ||
            std::abs(value - expected.homogeneousFiberWorld[axis]) > 2.0e-7)
            return false;
    }
    return material.sourceType == "trans iso Mooney-Rivlin" &&
        material.calibrationStatus == "source_population_prior" &&
        closeTo(material.c1Pascals, expected.c1) &&
        closeTo(material.c2Pascals, expected.c2) &&
        closeTo(material.c3Pascals, expected.c3) &&
        closeTo(material.c4, expected.c4) &&
        closeTo(material.c5Pascals, expected.c5) &&
        closeTo(material.lambdaMaximum, expected.lambdaMaximum) &&
        closeTo(material.bulkModulusPascals, expected.bulk) &&
        closeTo(material.initialStretch, expected.initialStretch);
}

[[nodiscard]] bool validateDecodedPayload(
    const NumiHumanKneePayload& payload,
    std::string& error
) {
    if (payload.payloadAbi != 3u ||
        payload.side != NumiHumanKneeSide::left ||
        payload.nodes.size() != kNumiHumanLoadedKneeNodeCount ||
        payload.tetrahedra.size() != kNumiHumanLoadedKneeTetrahedronCount ||
        payload.surfaces.size() != kNumiHumanLoadedKneeSurfaceCount ||
        payload.faces.size() != kNumiHumanLoadedKneeSurfaceFaceCount ||
        payload.nodeSets.size() != kNumiHumanLoadedKneeNodeSetCount ||
        payload.memberships.size() !=
            kNumiHumanLoadedKneeNodeSetMembershipCount ||
        payload.surfacePairs.size() !=
            kNumiHumanLoadedKneeSourceSurfacePairCount) {
        return fail(error,
            "decoded left NHKNEE1 ABI3 topology does not match the authoring contract");
    }

    std::uint64_t loadedNodes = 0u;
    std::uint64_t loadedTetrahedra = 0u;
    for (std::size_t index = 0u; index < kRegionNames.size(); ++index) {
        const auto found = std::find_if(
            payload.regions.begin(), payload.regions.end(),
            [index](const NumiHumanKneeRegion& region) {
                return region.name == kRegionNames[index];
            });
        if (found == payload.regions.end()) {
            return fail(error,
                "decoded NHKNEE1 payload is missing a required loaded region");
        }
        const bool tendon = found->name == "PTL" || found->name == "QAT";
        if ((tendon && found->kind != NumiHumanKneeRegionKind::tendon) ||
            (!tendon && found->kind != NumiHumanKneeRegionKind::ligament)) {
            return fail(error, "decoded loaded-knee tissue kind is invalid");
        }
        const auto& expected = kExpectedMaterials[index];
        const auto& material = found->material;
        bool exactFiber = true;
        for (std::size_t axis = 0u;
             axis < material.homogeneousFiberWorld.size(); ++axis) {
            exactFiber = exactFiber && std::isfinite(
                material.homogeneousFiberWorld[axis]) &&
                std::abs(static_cast<double>(
                    material.homogeneousFiberWorld[axis]) -
                    expected.homogeneousFiberWorld[axis]) <= 2.0e-7;
        }
        if (!material.hasHomogeneousFiber ||
            !material.hasIsochoricInSituStretch ||
            !closeTo(1.0e6 * material.c1MPa, expected.c1) ||
            !closeTo(1.0e6 * material.c2MPa, expected.c2) ||
            !closeTo(1.0e6 * material.c3MPa, expected.c3) ||
            !closeTo(material.c4, expected.c4) ||
            !closeTo(1.0e6 * material.c5MPa, expected.c5) ||
            !closeTo(material.lambdaMaximum, expected.lambdaMaximum) ||
            !closeTo(1.0e6 * material.bulkModulusMPa, expected.bulk) ||
            !closeTo(material.initialStretch, expected.initialStretch) ||
            !exactFiber) {
            return fail(error,
                "decoded NHKNEE1 loaded-region material identity drifted");
        }
        loadedNodes += found->nodeCount;
        loadedTetrahedra += found->tetrahedronCount;
    }
    if (loadedNodes != kNumiHumanLoadedKneeLoadedNodeCount ||
        loadedTetrahedra != kNumiHumanLoadedKneeLoadedTetrahedronCount) {
        return fail(error,
            "decoded NHKNEE1 loaded-region topology totals drifted");
    }

    std::vector<std::array<std::string_view, 3u>> articularPairs;
    for (const auto& pair : payload.surfacePairs) {
        if (pair.masterSurface >= payload.surfaces.size() ||
            pair.slaveSurface >= payload.surfaces.size()) {
            return fail(error, "decoded NHKNEE1 surface-pair index is invalid");
        }
        const auto& master = payload.surfaces[pair.masterSurface];
        const auto& slave = payload.surfaces[pair.slaveSurface];
        if (master.regionIndex >= payload.regions.size() ||
            slave.regionIndex >= payload.regions.size()) {
            return fail(error, "decoded NHKNEE1 surface region is invalid");
        }
        const auto isArticular = [](const NumiHumanKneeRegionKind kind) {
            return kind == NumiHumanKneeRegionKind::cartilage ||
                kind == NumiHumanKneeRegionKind::meniscus;
        };
        if (isArticular(payload.regions[master.regionIndex].kind) &&
            isArticular(payload.regions[slave.regionIndex].kind)) {
            articularPairs.push_back({
                pair.name, master.name, slave.name});
        }
    }
    if (articularPairs.size() != kContactPairNames.size()) {
        return fail(error,
            "decoded NHKNEE1 seven-pair articular identity or order drifted");
    }
    for (std::size_t index = 0u; index < articularPairs.size(); ++index) {
        if (articularPairs[index][0u] != kContactPairNames[index] ||
            articularPairs[index][1u] != kContactMasterSurfaces[index] ||
            articularPairs[index][2u] != kContactSlaveSurfaces[index]) {
            return fail(error,
                "decoded NHKNEE1 seven-pair surface identity or order drifted");
        }
    }
    DecodedExecutableTopology executable;
    if (!decodeExecutableTopology(payload, executable, error)) return false;
    if (executable.sourceGlobalNodeIndexSHA256 !=
            kSourceGlobalNodeIndexSHA256 ||
        executable.executableFEMTopologySHA256 !=
            kExecutableFEMTopologySHA256 ||
        executable.anchorOwnershipSHA256 != kAnchorOwnershipSHA256) {
        return fail(error,
            "decoded NHKNEE1 executable node, tetrahedron, or anchor identity drifted");
    }
    return true;
}

[[nodiscard]] bool appendBodyMassAuthority(
    SHA256Writer& writer,
    const MRBodyPropertiesGPU& body
) noexcept {
    return writer.appendFloat(body.massAndInverseMass.x) &&
        writer.appendFloat(body.massAndInverseMass.y) &&
        writer.appendFloat(body.centerOfMass.x) &&
        writer.appendFloat(body.centerOfMass.y) &&
        writer.appendFloat(body.centerOfMass.z) &&
        writer.appendFloat(body.inertiaRow0.x) &&
        writer.appendFloat(body.inertiaRow0.y) &&
        writer.appendFloat(body.inertiaRow0.z) &&
        writer.appendFloat(body.inertiaRow1.x) &&
        writer.appendFloat(body.inertiaRow1.y) &&
        writer.appendFloat(body.inertiaRow1.z) &&
        writer.appendFloat(body.inertiaRow2.x) &&
        writer.appendFloat(body.inertiaRow2.y) &&
        writer.appendFloat(body.inertiaRow2.z);
}

[[nodiscard]] bool digestMassClosure(
    const std::span<const NumiHumanTissueMassPartition> partitions,
    NumiHumanLoadedKneeDigest& output
) noexcept {
    SHA256Writer writer;
    if (!writer.appendString(kMassDomain) ||
        !writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) ||
        !writer.appendU64(partitions.size())) {
        return false;
    }
    for (const auto& partition : partitions) {
        if (!writer.appendU32(partition.donorBody) ||
            !writer.appendU32(partition.nodeCount) ||
            !writer.appendDouble(partition.tissueMassKg)) return false;
        for (const double value : partition.tissueFirstMomentKgM) {
            if (!writer.appendDouble(value)) return false;
        }
        for (const double value : partition.tissueSecondMomentKgM2) {
            if (!writer.appendDouble(value)) return false;
        }
        for (const double value : partition.remainingCOMOffsetM) {
            if (!writer.appendDouble(value)) return false;
        }
        if (!appendBodyMassAuthority(writer, partition.sourceBody) ||
            !appendBodyMassAuthority(writer, partition.remainingBody) ||
            !writer.appendDouble(partition.packedMomentRelativeError)) {
            return false;
        }
    }
    return writer.finish(output) && digestPresent(output);
}

[[nodiscard]] bool digestCookedNodeMass(
    const std::span<const NumiHumanTissueMassNode> nodes,
    NumiHumanLoadedKneeDigest& output
) noexcept {
    SHA256Writer writer;
    if (!writer.appendString(kCookedNodeMassDomain) ||
        !writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) ||
        !writer.appendU64(nodes.size())) return false;
    for (const auto& node : nodes) {
        if (!writer.appendU32(node.nodeIndex) ||
            !writer.appendU32(node.donorBody) ||
            !writer.appendDouble(node.massKg)) return false;
        for (const double value : node.localPosition) {
            if (!writer.appendDouble(value)) return false;
        }
    }
    return writer.finish(output) && digestPresent(output);
}

[[nodiscard]] bool appendStringVectorAuthority(
    SHA256Writer& writer,
    const std::string_view typeName,
    const std::vector<std::string>& values
) noexcept {
    if (!writer.appendString(typeName) ||
        !writer.appendU64(static_cast<std::uint64_t>(values.size()))) {
        return false;
    }
    for (const auto& value : values) {
        if (!writer.appendString(value)) return false;
    }
    return true;
}

[[nodiscard]] std::array<double, 3u> inverseRotatePoint(
    const mr_float4 orientation,
    const std::array<double, 3u>& point
) noexcept {
    const double x = -static_cast<double>(orientation.x);
    const double y = -static_cast<double>(orientation.y);
    const double z = -static_cast<double>(orientation.z);
    const double w = static_cast<double>(orientation.w);
    const double tx = 2.0 * (y * point[2u] - z * point[1u]);
    const double ty = 2.0 * (z * point[0u] - x * point[2u]);
    const double tz = 2.0 * (x * point[1u] - y * point[0u]);
    return {
        point[0u] + w * tx + (y * tz - z * ty),
        point[1u] + w * ty + (z * tx - x * tz),
        point[2u] + w * tz + (x * ty - y * tx),
    };
}

} // namespace

bool validateNumiHumanLoadedKneeExecutedAnchorRowV1(
    const std::uint32_t expectedSourceGlobalNodeIndex,
    const NumiHumanKneeNode& sourceNode,
    const std::array<double, 3u>& donorCOMOffsetM,
    const NMFEMNodeStateGPU& matterNode,
    const NumiHumanLoadedKneeExecutedAnchorV1& executedAnchor,
    std::string& error
) {
    try {
        const bool attached = sourceNode.rigidlyAttached;
        const bool exactInactive =
            !attached &&
            executedAnchor.bodyIndex == NUMI_HUMAN_KNEE_INVALID_INDEX &&
            executedAnchor.flags == 0u &&
            std::bit_cast<std::uint32_t>(
                executedAnchor.localPoint[0u]) == 0u &&
            std::bit_cast<std::uint32_t>(
                executedAnchor.localPoint[1u]) == 0u &&
            std::bit_cast<std::uint32_t>(
                executedAnchor.localPoint[2u]) == 0u &&
            matterNode.restAndFixed.w == 0.0f;
        const bool exactActive =
            attached &&
            executedAnchor.bodyIndex == sourceNode.anchorBodyIndex &&
            executedAnchor.flags == 1u &&
            std::isfinite(executedAnchor.localPoint[0u]) &&
            std::isfinite(executedAnchor.localPoint[1u]) &&
            std::isfinite(executedAnchor.localPoint[2u]) &&
            matterNode.restAndFixed.w == 2.0f;
        std::array<float, 3u> expectedExecutedLocal =
            sourceNode.anchorLocal;
        if (attached) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                if (!std::isfinite(sourceNode.anchorLocal[axis]) ||
                    !std::isfinite(donorCOMOffsetM[axis])) {
                    return fail(error,
                        "loaded-knee executed anchor row is not finite");
                }
                expectedExecutedLocal[axis] -= static_cast<float>(
                    donorCOMOffsetM[axis]);
            }
        }
        const bool exactExecutedLocal = !attached ||
            (std::bit_cast<std::uint32_t>(
                 executedAnchor.localPoint[0u]) ==
                 std::bit_cast<std::uint32_t>(expectedExecutedLocal[0u]) &&
             std::bit_cast<std::uint32_t>(
                 executedAnchor.localPoint[1u]) ==
                 std::bit_cast<std::uint32_t>(expectedExecutedLocal[1u]) &&
             std::bit_cast<std::uint32_t>(
                 executedAnchor.localPoint[2u]) ==
                 std::bit_cast<std::uint32_t>(expectedExecutedLocal[2u]));
        if (executedAnchor.sourceGlobalNodeIndex !=
                expectedSourceGlobalNodeIndex ||
            !(exactInactive || exactActive) || !exactExecutedLocal) {
            return fail(error,
                "loaded-knee executed anchor row is not the immutable ABI3/rebased coordinate");
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee executed anchor row validation failed";
        return false;
    }
}

bool validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
    const std::uint32_t expectedExecutableNodeIndex,
    const std::uint32_t expectedObjectIndex,
    const NumiHumanKneeNode& sourceNode,
    const NumiHumanLoadedKneeExecutedAnchorV1& executedAnchor,
    const NMFEMNodeStateGPU& matterNode,
    const NMFEMHumanAttachmentGPU& matterAttachment,
    std::string& error
) {
    try {
        if (expectedExecutableNodeIndex >=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            expectedObjectIndex >= kNumiHumanLoadedKneeRegionCount ||
            !sourceNode.rigidlyAttached ||
            executedAnchor.flags != 1u ||
            executedAnchor.bodyIndex != sourceNode.anchorBodyIndex ||
            !std::isfinite(executedAnchor.localPoint[0u]) ||
            !std::isfinite(executedAnchor.localPoint[1u]) ||
            !std::isfinite(executedAnchor.localPoint[2u]) ||
            matterNode.restAndFixed.w != 2.0f) {
            return fail(error,
                "loaded-knee compiled attachment is not an active direct Matter row");
        }
        const std::uint32_t expectedStableIdentifier =
            kNumiHumanLoadedKneeAttachmentStableIdentifierBase +
            expectedExecutableNodeIndex + 1u;
        const bool exactLocalPoint =
            std::bit_cast<std::uint32_t>(matterAttachment.localPoint.x) ==
                std::bit_cast<std::uint32_t>(
                    executedAnchor.localPoint[0u]) &&
            std::bit_cast<std::uint32_t>(matterAttachment.localPoint.y) ==
                std::bit_cast<std::uint32_t>(
                    executedAnchor.localPoint[1u]) &&
            std::bit_cast<std::uint32_t>(matterAttachment.localPoint.z) ==
                std::bit_cast<std::uint32_t>(
                    executedAnchor.localPoint[2u]) &&
            std::bit_cast<std::uint32_t>(matterAttachment.localPoint.w) == 0u;
        if (matterAttachment.identity.x != expectedExecutableNodeIndex ||
            matterAttachment.identity.y != executedAnchor.bodyIndex ||
            matterAttachment.identity.z != expectedObjectIndex ||
            matterAttachment.identity.w != expectedStableIdentifier ||
            !exactLocalPoint) {
            return fail(error,
                "loaded-knee compiled attachment differs from the profile-order direct Matter binding");
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee compiled attachment row validation failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeExecutedAnchorsV1(
    const std::span<const NumiHumanLoadedKneeExecutedAnchorV1>
        executedAnchors,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        if (executedAnchors.empty()) {
            return fail(error,
                "loaded-knee executed-anchor identity is empty");
        }
        SHA256Writer writer;
        for (const auto& anchor : executedAnchors) {
            if (!writer.appendU32(anchor.sourceGlobalNodeIndex) ||
                !writer.appendU32(anchor.bodyIndex) ||
                !writer.appendU32(anchor.flags) ||
                !writer.appendFloat(anchor.localPoint[0u]) ||
                !writer.appendFloat(anchor.localPoint[1u]) ||
                !writer.appendFloat(anchor.localPoint[2u])) {
                return fail(error,
                    "loaded-knee executed-anchor identity hashing failed");
            }
        }
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error,
                "loaded-knee executed-anchor SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee executed-anchor SHA-256 failed";
        return false;
    }
}

bool validateNumiHumanLoadedKneeAuthoringV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanKneePayload* decodedPayload,
    std::string& error
) {
    try {
        if (authoring.formatVersion !=
                kNumiHumanLoadedKneeAuthoringVersionV1 ||
            authoring.schema != kAuthoringSchema ||
            authoring.compiler != kAuthoringCompiler ||
            authoring.status != "candidate" || authoring.side != "left" ||
            (authoring.sourceOwnershipStatus != "blocked" &&
             authoring.sourceOwnershipStatus != "partial") ||
            authoring.productionPromotion ||
            !digestPresent(authoring.manifestSHA256) ||
            authoring.manifestCanonicalization !=
                "utf8-json-sorted-keys-compact-ensure_ascii=false-allow_nan=false" ||
            authoring.manifestHashExclusion !=
                "top-level manifest_sha256") {
            return fail(error,
                "loaded-knee HumanPack header or immutable receipt identity is invalid");
        }
        if (!validIdentity(authoring.ownershipInput) ||
            !validIdentity(authoring.authoringProfileInput) ||
            !validIdentity(authoring.tendonPayloadInput) ||
            !validIdentity(authoring.xReferenceInput) ||
            !validIdentity(authoring.labExportInput) ||
            authoring.ownershipInput.schema != "HumanPack.ownership.v1" ||
            authoring.authoringProfileInput.schema !=
                "numi.human.loaded-anatomy-knee-authoring.v1" ||
            authoring.authoringProfileInput.fileSHA256 !=
                kAuthoringProfileFileSHA256 ||
            authoring.authoringProfileInput.identitySHA256 !=
                kAuthoringProfileIdentitySHA256 ||
            authoring.tendonPayloadInput.schema !=
                "numi.human.tendon-attachment-envelope-payload.v3" ||
            authoring.xReferenceInput.schema !=
                "numi.human.loaded-anatomy-knee-x-ref-f32le.v1" ||
            authoring.labExportInput.schema !=
                "numi.lab.loaded-knee-authoring-export.v1" ||
            !digestPresent(authoring.sourceModelFingerprintSHA256) ||
            authoring.ownershipInput.identitySHA256 !=
                authoring.ownershipManifestSHA256) {
            return fail(error,
                "loaded-knee HumanPack referenced identity is incomplete");
        }
        if (authoring.kneePayload.schema !=
                "numi.human.open-knee-oks003-payload.v3" ||
            authoring.kneePayload.magic != "NHKNEE1" ||
            authoring.kneePayload.abi != 3u ||
            authoring.kneePayload.byteCount !=
                kNumiHumanLoadedKneePayloadBytesV1 ||
            authoring.kneePayload.fileSHA256 != kKneePayloadSHA256) {
            return fail(error,
                "loaded-knee HumanPack does not bind the pinned left NHKNEE1 ABI3 payload");
        }
        if (authoring.tendonPayload.schema !=
                "numi.human.tendon-attachment-envelope-payload.v3" ||
            authoring.tendonPayload.magic != "NHTENDON3" ||
            authoring.tendonPayload.abi != 3u ||
            authoring.tendonPayload.byteCount != 238'288u ||
            authoring.tendonPayload.fileSHA256 != kTendonPayloadSHA256) {
            return fail(error,
                "loaded-knee HumanPack does not bind the qualified NHTENDON3 payload");
        }
        if (authoring.sourceRigidPayloadSHA256 !=
                kSourceRigidPayloadSHA256 ||
            authoring.equalityPayloadSHA256 != kEqualityPayloadSHA256 ||
            !digestPresent(authoring.sourceModelFingerprintSHA256) ||
            authoring.sourceDefaultPoseID !=
                "numi-human:unprojected-myosim-default-body-pose" ||
            !digestPresent(authoring.sourceDefaultPoseSHA256) ||
            authoring.projectedReferencePoseID !=
                "numi-human:equality-projected-default-reference-body-pose" ||
            !digestPresent(authoring.projectedReferencePoseSHA256) ||
            authoring.sourceToReferenceMappingID !=
                "numi-lab.open-knee-restWorld-to-equality-projected-default-body-poses.1" ||
            authoring.sourceToReferenceMappingAlgorithm.empty() ||
            !digestPresent(
                authoring.sourceToReferenceMappingCodeSHA256)) {
            return fail(error,
                "loaded-knee source rigid, equality, pose, or mapping provenance differs");
        }
        if (authoring.subjectID != "OpenKnee:oks003:left" ||
            authoring.coverageLeafSHA256s.empty() ||
            std::any_of(authoring.coverageLeafSHA256s.begin(),
                        authoring.coverageLeafSHA256s.end(),
                        [](const auto& digest) {
                            return !digestPresent(digest);
                        }) ||
            authoring.datasetID != "OpenKnee:oks003" ||
            !digestPresent(authoring.licenseFileSHA256)) {
            return fail(error,
                "loaded-knee HumanPack semantic or source scope is incomplete");
        }

        const auto& topology = authoring.topology;
        if (topology.identitySHA256 != kTopologyIdentitySHA256 ||
            topology.executableFEMTopologySHA256 !=
                kExecutableFEMTopologySHA256 ||
            topology.sourceGlobalNodeIndexSHA256 !=
                kSourceGlobalNodeIndexSHA256 ||
            topology.anchorOwnershipSHA256 != kAnchorOwnershipSHA256 ||
            topology.nodeCount != kNumiHumanLoadedKneeNodeCount ||
            topology.tetrahedronCount !=
                kNumiHumanLoadedKneeTetrahedronCount ||
            topology.surfaceCount != kNumiHumanLoadedKneeSurfaceCount ||
            topology.surfaceFaceCount !=
                kNumiHumanLoadedKneeSurfaceFaceCount ||
            topology.nodeSetCount != kNumiHumanLoadedKneeNodeSetCount ||
            topology.nodeSetMembershipCount !=
                kNumiHumanLoadedKneeNodeSetMembershipCount ||
            topology.sourceSurfacePairCount !=
                kNumiHumanLoadedKneeSourceSurfacePairCount ||
            topology.loadedNodeCount !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            topology.loadedTetrahedronCount !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            !std::equal(topology.loadedRegionNames.begin(),
                        topology.loadedRegionNames.end(),
                        kRegionNames.begin())) {
            return fail(error,
                "loaded-knee HumanPack topology identity or exact counts drifted");
        }

        const auto& coordinates = authoring.coordinates;
        if (!digestPresent(coordinates.sourceStateSHA256) ||
            !digestPresent(coordinates.referenceStateSHA256) ||
            coordinates.sourceStateNodeCount !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            coordinates.referenceStateNodeCount !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            coordinates.sourceStateEncoding != "float32-le-xyz" ||
            coordinates.referenceStateEncoding != "float32-le-xyz" ||
            coordinates.sourceFrameID != "myosim-world-m" ||
            coordinates.referenceFrameID != "myosim-world-m" ||
            coordinates.referenceStateClass != "projected-rest-candidate" ||
            coordinates.constructionID !=
                "numi-lab.open-knee-moving-enthesis-projected-rest.1" ||
            coordinates.unloadedReferenceQualified ||
            coordinates.prestrainResetStatus != "unresolved" ||
            !coordinates.prestrainResetMethodID.empty() ||
            coordinates.volumetricPrestressStatus != "not_applied" ||
            coordinates.currentStateOwner !=
                "separate-lab-runtime-acceptance-receipt" ||
            coordinates.currentStateHashAlgorithm != "sha256" ||
            coordinates.currentStateHashScope.empty() ||
            !coordinates.currentStateRequired ||
            coordinates.authoredAcceptedCurrentStatePresent ||
            digestPresent(coordinates.authoredAcceptedCurrentStateSHA256)) {
            return fail(error,
                "loaded-knee coordinate contract overclaims or omits runtime x_current authority");
        }

        if (!closeTo(authoring.densitySourceValue, 1.0e-9, 1.0e-12) ||
            authoring.densitySourceUnit != "tonne_per_mm3" ||
            !closeTo(authoring.densityConversionFactorToKgPerM3, 1.0e12) ||
            !closeTo(authoring.densityRuntimeKgPerM3, 1000.0) ||
            authoring.densityCalibrationStatus != "source_population_prior") {
            return fail(error,
                "loaded-knee density conversion is absent or not a source population prior");
        }

        for (std::size_t index = 0u; index < authoring.regions.size();
             ++index) {
            const auto& region = authoring.regions[index];
            const std::uint32_t expectedDonor =
                region.sourceRegionName == "PTL"
                ? NUMI_HUMAN_KNEE_TIBIA_BODY
                : NUMI_HUMAN_KNEE_FEMUR_BODY;
            const bool expectedActiveOwner = region.sourceRegionName == "QAT"
                ? region.activeForceOwnerStatus == "candidate" &&
                    region.activeForceOwnerID ==
                        "humanpack:loaded-anatomy-knee:left/QAT/active-force"
                : region.activeForceOwnerStatus == "none" &&
                    region.activeForceOwnerID.empty();
            const std::string ownerPrefix =
                "humanpack:loaded-anatomy-knee:left/region/" +
                region.sourceRegionName;
            if (region.sourceRegionName != kRegionNames[index] ||
                region.semanticID !=
                    "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/Elements[name=" +
                        region.sourceRegionName + "][1]" ||
                region.donorBodyIndex != expectedDonor ||
                region.donorBodyIndex == NUMI_HUMAN_KNEE_PATELLA_BODY ||
                !expectedActiveOwner ||
                region.physicalVolumeOwnerID !=
                    ownerPrefix + "/physical-volume" ||
                region.mechanicalMassOwnerID !=
                    ownerPrefix + "/mechanical-mass" ||
                region.materialOwnerID != ownerPrefix + "/material" ||
                region.stateOwnerID !=
                    "humanpack:loaded-anatomy-knee:left/full-state" ||
                !digestPresent(region.topologyIdentitySHA256) ||
                !validOwnerStrings(region) ||
                !validateMaterial(region.material, kExpectedMaterials[index])) {
                return fail(error,
                    "loaded-knee region material or ownership row is invalid");
            }
        }

        for (std::size_t index = 0u; index < authoring.contactPairs.size();
             ++index) {
            const auto& pair = authoring.contactPairs[index];
            if (pair.sourcePairName != kContactPairNames[index] ||
                pair.semanticID !=
                    "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/SurfacePair[name=" +
                        pair.sourcePairName + "][1]" ||
                pair.ownerID.empty() ||
                pair.ownerID !=
                    "humanpack:loaded-anatomy-knee:left/contact/" +
                        pair.sourcePairName + "/state" ||
                pair.masterSurface != kContactMasterSurfaces[index] ||
                pair.slaveSurface != kContactSlaveSurfaces[index]) {
                return fail(error,
                    "loaded-knee articular pair identity or order is invalid");
            }
        }

        std::set<std::uint32_t> endpoints;
        for (std::size_t index = 0u;
             index < authoring.activeReplacements.size(); ++index) {
            const auto& replacement = authoring.activeReplacements[index];
            const std::string semantic =
                "myosim_fullbody:composed/myofullbody/muscles/" +
                std::string(kRouteNames[index]);
            if (replacement.semanticID != semantic ||
                replacement.sourceActuatorIndex != kActuatorIndices[index] ||
                replacement.sourceRouteName != kRouteNames[index] ||
                replacement.loadEndpointIndex != kLoadEndpoints[index] ||
                replacement.loadRouteNodeIndex != kLoadRouteNodes[index] ||
                replacement.loadSourceSiteIndex != kLoadSourceSites[index] ||
                replacement.loadBodyIndex != kLoadBodies[index] ||
                replacement.anchorEndpointIndex != kAnchorEndpoints[index] ||
                replacement.anchorRouteNodeIndex !=
                    kAnchorRouteNodes[index] ||
                replacement.anchorSourceSiteIndex !=
                    kAnchorSourceSites[index] ||
                replacement.anchorBodyIndex !=
                    NUMI_HUMAN_KNEE_TIBIA_BODY ||
                replacement.sourceOwnerID != semantic + "/owner/source-jt" ||
                replacement.candidateOwnerID !=
                    "humanpack:loaded-anatomy-knee:left/QAT/active-force" ||
                replacement.mode != "replacement" ||
                replacement.replacementFraction != 1.0 ||
                !endpoints.insert(replacement.loadEndpointIndex).second ||
                !endpoints.insert(replacement.anchorEndpointIndex).second) {
                return fail(error,
                    "loaded-knee quadriceps replacement row or exact endpoint tuple is invalid");
            }
        }

        for (std::size_t index = 0u; index < authoring.passiveOwners.size();
             ++index) {
            const auto& owner = authoring.passiveOwners[index];
            if (owner.sourceRegionName != kPassiveNames[index] ||
                owner.semanticID !=
                    "humanpack:loaded-anatomy-knee:left/region/" +
                        owner.sourceRegionName + "/passive-force" ||
                owner.ownerID !=
                    "humanpack:loaded-anatomy-knee:left/region/" +
                        owner.sourceRegionName + "/material") {
                return fail(error,
                    "loaded-knee passive ligament owner identity or order is invalid");
            }
        }
        constexpr std::array<std::string_view, 6u> requiredComponents{{
            "articulated-q-v-root-time",
            "articular-contact-history",
            "fem-position-velocity-mass-material-status",
            "passive-ligament-state",
            "solver-adaptive-state",
            "tendon-transfer-replacement-state",
        }};
        if (!digestPresent(authoring.authoredMassPartitionSHA256) ||
            !digestPresent(authoring.authoredRawF32NodeMassSHA256) ||
            authoring.fullStateSemanticID !=
                "humanpack:loaded-anatomy-knee:left/full-state-authority" ||
            authoring.fullStateOwnerID !=
                "numi-lab:human-matter/accepted-step-transaction" ||
            authoring.fullStateSchema !=
                "NumiLab.HumanMatterAcceptedState.v1" ||
            !digestPresent(authoring.fullStateIdentitySHA256) ||
            authoring.requiredSnapshotComponents.size() !=
                requiredComponents.size() ||
            !std::equal(authoring.requiredSnapshotComponents.begin(),
                        authoring.requiredSnapshotComponents.end(),
                        requiredComponents.begin())) {
            return fail(error,
                "loaded-knee full-state or authored mass authority is invalid");
        }
        if (!authoring.candidateOnly || authoring.productionQualified ||
            authoring.unloadedReferenceQualified ||
            authoring.subjectCalibratedMaterials ||
            !authoring.donorMassSubtractionRequired ||
            !authoring.massMomentClosureRequired ||
            authoring.boundary !=
                kNumiHumanLoadedKneeHumanPackBoundaryV1) {
            return fail(error,
                "loaded-knee HumanPack qualification boundary is invalid");
        }
        if (decodedPayload != nullptr &&
            !validateDecodedPayload(*decodedPayload, error)) return false;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee HumanPack validation failed";
        return false;
    }
}

bool verifyNumiHumanLoadedKneeImmutableFileV1(
    const std::filesystem::path& path,
    const NumiHumanLoadedKneeDigest& expectedSHA256,
    const std::uint64_t expectedByteCount,
    std::string& error
) {
    try {
        std::vector<char> bytes;
        return readVerifiedImmutableFile(
            path, expectedSHA256, expectedByteCount, bytes, error);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool writeNumiHumanLoadedKneeImmutableFileV1(
    const std::filesystem::path& path,
    const std::span<const std::byte> bytes,
    std::string& error
) {
    try {
        if (path.empty()) {
            return fail(error, "immutable loaded-knee output path is empty");
        }
        const std::filesystem::path filenamePath = path.filename();
        const std::string filename = filenamePath.native();
        if (filename.empty() || filename == "." || filename == ".." ||
            filename.find('\0') != std::string::npos) {
            return fail(error,
                "immutable loaded-knee output filename is invalid");
        }

        std::filesystem::path parent = path.parent_path();
        if (parent.empty()) parent = ".";
        std::error_code directoryError;
        std::filesystem::create_directories(parent, directoryError);
        if (directoryError) {
            return fail(error,
                "immutable loaded-knee output directory could not be created: " +
                directoryError.message());
        }

        ScopedFileDescriptor directory(::open(
            parent.c_str(),
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        if (directory.get() < 0) {
            const int openError = errno;
            return fail(error, errnoMessage(
                "immutable loaded-knee output directory could not be opened safely",
                openError));
        }

        bool exists = false;
        if (!existingImmutableFileMatches(
                directory.get(), filename, bytes, exists, error)) {
            return false;
        }
        if (exists) {
            error.clear();
            return true;
        }

        constexpr std::size_t kMaximumStagingAttempts = 128u;
        int stagingValue = -1;
        std::string stagingName;
        for (std::size_t attempt = 0u;
             attempt < kMaximumStagingAttempts; ++attempt) {
            stagingName = uniqueStagingName();
            stagingValue = ::openat(
                directory.get(), stagingName.c_str(),
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (stagingValue >= 0) break;
            const int stagingError = errno;
            if (stagingError != EEXIST) {
                return fail(error, errnoMessage(
                    "immutable loaded-knee staging file could not be created safely",
                    stagingError));
            }
        }
        if (stagingValue < 0) {
            return fail(error,
                "immutable loaded-knee staging filename retries were exhausted");
        }

        ScopedStagingPath staging(
            directory.get(), std::move(stagingName));
        ScopedFileDescriptor stagingDescriptor(stagingValue);
        constexpr std::size_t kWriteChunkBytes = 16u * 1024u * 1024u;
        std::size_t offset = 0u;
        while (offset < bytes.size()) {
            const std::size_t countToWrite = std::min(
                bytes.size() - offset, kWriteChunkBytes);
            const auto count = ::write(
                stagingDescriptor.get(),
                reinterpret_cast<const char*>(bytes.data()) + offset,
                countToWrite);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                const int writeError = count < 0 ? errno : EIO;
                return fail(error, errnoMessage(
                    "immutable loaded-knee staging write was incomplete",
                    writeError));
            }
            offset += static_cast<std::size_t>(count);
        }
        while (::fsync(stagingDescriptor.get()) != 0) {
            const int syncError = errno;
            if (syncError == EINTR) continue;
            return fail(error, errnoMessage(
                "immutable loaded-knee staging file could not be synchronized",
                syncError));
        }
        if (!stagingDescriptor.closeChecked(
                "immutable loaded-knee staging file could not be closed",
                error)) {
            return false;
        }

        if (::renameatx_np(
                directory.get(), staging.name().c_str(),
                directory.get(), filename.c_str(), RENAME_EXCL) != 0) {
            const int renameError = errno;
            return fail(error, errnoMessage(
                "immutable loaded-knee no-replace publication failed",
                renameError));
        }
        staging.published();
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "immutable loaded-knee output publication failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeSourceModelV1(
    const EngineModel& model,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        std::string modelError;
        if (!model.valid(&modelError)) {
            return fail(error,
                "loaded-knee source EngineModel is invalid: " + modelError);
        }
        SHA256Writer writer;
        if (!writer.appendString(
                "numi.lab.loaded-knee.source-engine-model.v1") ||
            !writer.appendU32(kNumiHumanLoadedKneeAuthoringVersionV1) ||
            !writer.appendString("MRWorldGPU") ||
            !writer.appendU64(sizeof(model.world)) ||
            !writer.append(&model.world, sizeof(model.world)) ||
            !writer.appendVector("MRArticulationGPU", model.articulations) ||
            !writer.appendVector("MRJointDescriptorGPU", model.joints) ||
            !writer.appendString("FunctionBasedJointProgram") ||
            !writer.appendU64(model.functionBasedJointPrograms.size())) {
            return fail(error,
                "loaded-knee source EngineModel header hashing failed");
        }
        for (const auto& program : model.functionBasedJointPrograms) {
            MROpenSimSpatialTransformGPU transform{};
            if (packOpenSimSpatialTransformGPU(
                    program.transform, transform) !=
                    OpenSimSpatialTransformStatus::success ||
                !writer.appendU32(program.jointIndex) ||
                !writer.appendU64(sizeof(transform)) ||
                !writer.append(&transform, sizeof(transform))) {
                return fail(error,
                    "loaded-knee source FunctionBased program hashing failed");
            }
        }
        if (!writer.appendVector("MRDofPropertiesGPU", model.dofs) ||
            !writer.appendVector(
                "MRActuatorProfileGPU", model.actuatorProfiles) ||
            !writer.appendVector("MRBodyPropertiesGPU", model.bodies) ||
            !writer.appendVector("MRShapeGPU", model.shapes) ||
            !writer.appendVector("MRMaterialGPU", model.materials) ||
            !writer.appendVector(
                "MRGeometryHeaderGPU", model.geometryHeaders) ||
            !writer.appendVector("mr_float4", model.geometryVertices) ||
            !writer.appendVector("uint32", model.geometryIndices) ||
            !writer.appendVector("MRConvexFaceGPU", model.convexFaces) ||
            !writer.appendVector(
                "MRConvexHalfEdgeGPU", model.convexHalfEdges) ||
            !writer.appendVector("MRMeshBVHNodeGPU", model.meshBvhNodes) ||
            !writer.appendVector("MRMeshTriangleGPU", model.meshTriangles) ||
            !writer.appendVector(
                "CollisionPairExclusion", model.collisionExclusions) ||
            !writer.appendString("ConstraintIR") ||
            !writer.appendU32(model.constraintProgram.abiVersion) ||
            !writer.appendVector(
                "MRConstraintBlockGPU", model.constraintProgram.blocks) ||
            !writer.appendVector(
                "MRConstraintEndpointGPU",
                model.constraintProgram.endpoints) ||
            !writer.appendVector(
                "MRConstraintRowGPU", model.constraintProgram.rows) ||
            !writer.appendVector(
                "MRConstraintConeGPU", model.constraintProgram.cones) ||
            !writer.appendVector(
                "float", model.constraintProgram.warmImpulses) ||
            !writer.appendVector("defaultQ", model.defaultQ) ||
            !writer.appendVector("defaultV", model.defaultV) ||
            !appendStringVectorAuthority(
                writer, "bodyNames", model.bodyNames) ||
            !appendStringVectorAuthority(
                writer, "jointNames", model.jointNames) ||
            !appendStringVectorAuthority(
                writer, "dofNames", model.dofNames) ||
            !appendStringVectorAuthority(
                writer, "shapeNames", model.shapeNames) ||
            !writer.appendString(model.name)) {
            return fail(error,
                "loaded-knee source EngineModel authority hashing failed");
        }
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error,
                "loaded-knee source EngineModel SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee source EngineModel SHA-256 failed";
        return false;
    }
}

bool prepareNumiHumanLoadedKneeMassV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const EngineModel& source,
    const std::span<const std::array<float, 3u>> referenceCoordinates,
    const std::span<const MRBodyStateGPU> referenceBodies,
    const std::span<const NumiHumanTissueMassNode> cookedNodes,
    NumiHumanLoadedKneeMassEvidenceV1& output,
    std::string& error
) {
    try {
        if (!validateNumiHumanLoadedKneeAuthoringV1(
                authoring, nullptr, error)) return false;
        if (cookedNodes.size() != kNumiHumanLoadedKneeLoadedNodeCount ||
            referenceCoordinates.size() != cookedNodes.size() ||
            referenceBodies.size() != source.bodies.size() ||
            !digestPresent(authoring.coordinates.referenceStateSHA256)) {
            return fail(error,
                "loaded-knee mass partition requires exact reference coordinates, body poses, and 62402 cooked nodes");
        }
        NumiHumanLoadedKneeDigest observedReferenceSHA256{};
        if (!digestNumiHumanLoadedKneeCoordinatesV1(
                referenceCoordinates, observedReferenceSHA256, error) ||
            observedReferenceSHA256 !=
                authoring.coordinates.referenceStateSHA256) {
            return fail(error,
                "loaded-knee cooked mass is not bound to the authored x_ref bytes");
        }
        std::vector<NumiHumanTissueMassNode> derivedNodes(
            cookedNodes.begin(), cookedNodes.end());
        std::vector<bool> covered(cookedNodes.size(), false);
        constexpr std::array<std::uint32_t,
                             kNumiHumanLoadedKneeRegionCount>
            kProfileRegionNodeCounts{{
                15792u, 2960u, 15693u, 3714u, 9280u, 14963u}};
        const auto expectedDonorForNode = [
            &authoring, &kProfileRegionNodeCounts](
            const std::uint32_t nodeIndex) {
            std::uint32_t first = 0u;
            for (std::size_t region = 0u;
                 region < kProfileRegionNodeCounts.size(); ++region) {
                const std::uint32_t count =
                    kProfileRegionNodeCounts[region];
                if (nodeIndex >= first && nodeIndex - first < count)
                    return authoring.regions[region].donorBodyIndex;
                first += count;
            }
            return NUMI_HUMAN_KNEE_INVALID_INDEX;
        };
        for (auto& node : derivedNodes) {
            if (node.nodeIndex >= referenceCoordinates.size() ||
                covered[node.nodeIndex] ||
                node.donorBody >= referenceBodies.size() ||
                node.donorBody != expectedDonorForNode(node.nodeIndex)) {
                return fail(error,
                    "loaded-knee cooked mass ownership does not exactly cover the authored profile-region donor policy");
            }
            covered[node.nodeIndex] = true;
            const auto& body = referenceBodies[node.donorBody];
            const double quaternionNormSquared =
                static_cast<double>(body.orientation.x) * body.orientation.x +
                static_cast<double>(body.orientation.y) * body.orientation.y +
                static_cast<double>(body.orientation.z) * body.orientation.z +
                static_cast<double>(body.orientation.w) * body.orientation.w;
            if (!closeTo(quaternionNormSquared, 1.0, 2.0e-5) ||
                !std::isfinite(body.position.x) ||
                !std::isfinite(body.position.y) ||
                !std::isfinite(body.position.z)) {
                return fail(error,
                    "loaded-knee x_ref donor pose is invalid");
            }
            const auto& coordinate = referenceCoordinates[node.nodeIndex];
            node.localPosition = inverseRotatePoint(body.orientation, {
                static_cast<double>(coordinate[0u]) - body.position.x,
                static_cast<double>(coordinate[1u]) - body.position.y,
                static_cast<double>(coordinate[2u]) - body.position.z,
            });
        }
        if (std::find(covered.begin(), covered.end(), false) !=
            covered.end()) {
            return fail(error,
                "loaded-knee cooked mass ownership omitted x_ref rows");
        }
        const auto compiled = compileNumiHumanTissueMassPartition(
            source.bodies, derivedNodes);
        if (!compiled.succeeded() || compiled.partitions.empty()) {
            return fail(error, "loaded-knee donor subtraction failed: " +
                compiled.error);
        }
        EngineModel rebased;
        std::string rebaseError;
        if (!rebaseNumiHumanTissueMassPartition(
                source, compiled.partitions, rebased, rebaseError)) {
            return fail(error, "loaded-knee donor rebase failed: " +
                rebaseError);
        }
        double maximumError = 0.0;
        std::uint64_t partitionedNodeCount = 0u;
        if (source.bodies.size() <= NUMI_HUMAN_KNEE_PATELLA_BODY ||
            !closeTo(source.bodies[NUMI_HUMAN_KNEE_FEMUR_BODY]
                         .massAndInverseMass.x, 8.4, 2.0e-5) ||
            !closeTo(source.bodies[NUMI_HUMAN_KNEE_TIBIA_BODY]
                         .massAndInverseMass.x, 3.8, 2.0e-5) ||
            !closeTo(source.bodies[NUMI_HUMAN_KNEE_PATELLA_BODY]
                         .massAndInverseMass.x, 0.02785286, 2.0e-5)) {
            return fail(error,
                "loaded-knee donor source masses do not match femur145/tibia150/patella156 authority");
        }
        std::set<std::uint32_t> donorBodies;
        for (const auto& partition : compiled.partitions) {
            donorBodies.insert(partition.donorBody);
            partitionedNodeCount += partition.nodeCount;
            maximumError = std::max(
                maximumError, partition.packedMomentRelativeError);
            if (!finitePositive(partition.tissueMassKg) ||
                !std::isfinite(partition.packedMomentRelativeError) ||
                partition.packedMomentRelativeError >
                    16.0 * std::numeric_limits<float>::epsilon()) {
                return fail(error,
                    "loaded-knee donor mass/moment closure exceeded the FP32 ABI bound");
            }
        }
        if (partitionedNodeCount != cookedNodes.size()) {
            return fail(error,
                "loaded-knee mass partition lost cooked node ownership");
        }
        if (donorBodies != std::set<std::uint32_t>{
                NUMI_HUMAN_KNEE_FEMUR_BODY,
                NUMI_HUMAN_KNEE_TIBIA_BODY} ||
            donorBodies.contains(NUMI_HUMAN_KNEE_PATELLA_BODY)) {
            return fail(error,
                "loaded-knee mass ownership must subtract only femur145 and tibia150 donors");
        }
        NumiHumanLoadedKneeDigest digest{};
        if (!digestMassClosure(compiled.partitions, digest)) {
            return fail(error,
                "loaded-knee mass/moment closure SHA-256 failed");
        }
        NumiHumanLoadedKneeDigest sourceModelSHA256{};
        if (!digestNumiHumanLoadedKneeSourceModelV1(
                source, sourceModelSHA256, error) ||
            sourceModelSHA256 != authoring.sourceModelFingerprintSHA256) {
            return fail(error,
                "loaded-knee source EngineModel does not match the Lab authoring export");
        }
        NumiHumanLoadedKneeMassEvidenceV1 candidate;
        candidate.partitions = compiled.partitions;
        candidate.cookedNodes = std::move(derivedNodes);
        candidate.rebasedModel = std::move(rebased);
        candidate.closureSHA256 = digest;
        if (!digestCookedNodeMass(
                candidate.cookedNodes, candidate.cookedNodeMassSHA256)) {
            return fail(error,
                "loaded-knee authored x_ref node-mass SHA-256 failed");
        }
        candidate.referenceCoordinateSHA256 = observedReferenceSHA256;
        candidate.authoredRawF32NodeMassSHA256 =
            authoring.authoredRawF32NodeMassSHA256;
        candidate.sourceRigidModelFingerprint =
            engineModelFingerprint(source);
        if (candidate.sourceRigidModelFingerprint == 0u) {
            return fail(error,
                "loaded-knee source rigid model identity is invalid");
        }
        candidate.maximumPackedMomentRelativeError = maximumError;
        output = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee mass preparation failed";
        return false;
    }
}

bool bindNumiHumanLoadedKneeMassToMatterWorldV1(
    const numi::matter::CompiledWorld& world,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
) {
    try {
        const std::span<const NumiHumanTissueMassNode> cookedNodes =
            evidence.cookedNodes;
        if (cookedNodes.size() != kNumiHumanLoadedKneeLoadedNodeCount ||
            world.physicsFingerprint == 0u ||
            world.fem.nodes.size() != cookedNodes.size() ||
            world.dispatch.femNodeCount != cookedNodes.size() ||
            world.fem.tetrahedra.size() !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            world.dispatch.tetrahedronCount !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            evidence.partitions.empty() ||
            !digestPresent(evidence.closureSHA256) ||
            !digestPresent(evidence.authoredRawF32NodeMassSHA256) ||
            evidence.rebasedModel.bodies.size() <=
                NUMI_HUMAN_KNEE_PATELLA_BODY) {
            return fail(error,
                "loaded-knee Matter world cannot be bound to donor mass evidence");
        }
        std::vector<bool> seen(cookedNodes.size(), false);
        SHA256Writer rawMassWriter;
        for (const auto& node : world.fem.nodes) {
            if (!std::isfinite(node.positionAndMass.w) ||
                node.positionAndMass.w < 0.0f ||
                !rawMassWriter.appendFloat(node.positionAndMass.w)) {
                return fail(error,
                    "loaded-knee executable raw f32 node mass is invalid");
            }
        }
        for (const auto& node : cookedNodes) {
            if (node.nodeIndex >= world.fem.nodes.size() ||
                seen[node.nodeIndex] ||
                std::bit_cast<std::uint32_t>(
                    static_cast<float>(node.massKg)) !=
                    std::bit_cast<std::uint32_t>(world.fem.nodes[
                        node.nodeIndex].positionAndMass.w)) {
                return fail(error,
                    "loaded-knee cooked node mass does not match the executable Matter arena");
            }
            seen[node.nodeIndex] = true;
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
            return fail(error,
                "loaded-knee cooked node order does not cover the executable Matter arena");
        }

        std::vector<MRBodyPropertiesGPU> sourceBodies =
            evidence.rebasedModel.bodies;
        for (const auto& partition : evidence.partitions) {
            if (partition.donorBody >= sourceBodies.size()) {
                return fail(error,
                    "loaded-knee mass evidence donor is outside the rebased model");
            }
            sourceBodies[partition.donorBody] = partition.sourceBody;
        }
        const auto recomputed = compileNumiHumanTissueMassPartition(
            sourceBodies, cookedNodes);
        NumiHumanLoadedKneeDigest recomputedClosure{};
        if (!recomputed.succeeded() ||
            !digestMassClosure(recomputed.partitions, recomputedClosure) ||
            recomputedClosure != evidence.closureSHA256) {
            return fail(error,
                "loaded-knee executable node masses do not reproduce donor moment closure");
        }
        NumiHumanLoadedKneeDigest nodeMassSHA256{};
        NumiHumanLoadedKneeDigest rawF32NodeMassSHA256{};
        if (!digestCookedNodeMass(cookedNodes, nodeMassSHA256) ||
            nodeMassSHA256 != evidence.cookedNodeMassSHA256 ||
            !rawMassWriter.finish(rawF32NodeMassSHA256) ||
            rawF32NodeMassSHA256 !=
                evidence.authoredRawF32NodeMassSHA256) {
            return fail(error,
                "loaded-knee executable node mass differs from the Human-authored raw f32 distribution");
        }
        NumiHumanLoadedKneeMassEvidenceV1 candidate = evidence;
        candidate.cookedNodeMassSHA256 = nodeMassSHA256;
        candidate.executedRawF32NodeMassSHA256 =
            rawF32NodeMassSHA256;
        candidate.executedSourcePhysicsFingerprint = world.physicsFingerprint;
        evidence = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee Matter mass binding failed";
        return false;
    }
}

bool bindNumiHumanLoadedKneeExecutableTopologyV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanKneePayload& decodedPayload,
    const numi::matter::CompiledWorld& world,
    const std::span<const NumiHumanLoadedKneeExecutedAnchorV1>
        executedAnchors,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
) {
    try {
        if (!validateNumiHumanLoadedKneeAuthoringV1(
                authoring, &decodedPayload, error)) return false;
        if (world.fem.nodes.size() !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            world.dispatch.femNodeCount !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            world.fem.tetrahedra.size() !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            world.dispatch.tetrahedronCount !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            executedAnchors.size() !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            evidence.referenceCoordinateSHA256 !=
                authoring.coordinates.referenceStateSHA256) {
            return fail(error,
                "loaded-knee executable topology binding has the wrong world, anchor, or x_ref extent");
        }

        DecodedExecutableTopology decoded;
        if (!decodeExecutableTopology(decodedPayload, decoded, error)) {
            return false;
        }
        if (decoded.sourceGlobalNodeIndexSHA256 !=
                authoring.topology.sourceGlobalNodeIndexSHA256 ||
            decoded.executableFEMTopologySHA256 !=
                authoring.topology.executableFEMTopologySHA256 ||
            decoded.anchorOwnershipSHA256 !=
                authoring.topology.anchorOwnershipSHA256) {
            return fail(error,
                "loaded-knee decoded ABI3 topology does not match HumanPack");
        }

        if (world.objects.size() != kNumiHumanLoadedKneeRegionCount) {
            return fail(error,
                "loaded-knee executable world does not contain exactly the six profile-order FEM objects");
        }
        std::uint32_t expectedNodeOffset = 0u;
        std::uint32_t expectedTetrahedronOffset = 0u;
        for (std::size_t objectIndex = 0u;
             objectIndex < world.objects.size(); ++objectIndex) {
            const auto payloadRegion = std::find_if(
                decodedPayload.regions.begin(), decodedPayload.regions.end(),
                [objectIndex](const NumiHumanKneeRegion& region) {
                    return region.name == kRegionNames[objectIndex];
                });
            const auto& object = world.objects[objectIndex];
            if (payloadRegion == decodedPayload.regions.end() ||
                object.representation != NM_REPRESENTATION_FEM ||
                object.stateOffset != expectedNodeOffset ||
                object.stateCount != payloadRegion->nodeCount ||
                object.elementOffset != expectedTetrahedronOffset ||
                object.elementCount != payloadRegion->tetrahedronCount) {
                return fail(error,
                    "loaded-knee compiled FEM object spans left profile/object/local order");
            }
            expectedNodeOffset += payloadRegion->nodeCount;
            expectedTetrahedronOffset += payloadRegion->tetrahedronCount;
        }
        if (expectedNodeOffset != kNumiHumanLoadedKneeLoadedNodeCount ||
            expectedTetrahedronOffset !=
                kNumiHumanLoadedKneeLoadedTetrahedronCount ||
            world.dispatch.femHumanAttachmentCount !=
                world.fem.humanAttachments.size()) {
            return fail(error,
                "loaded-knee compiled FEM spans or direct-attachment dispatch count are inconsistent");
        }

        SHA256Writer referenceWriter;
        SHA256Writer executableTopologyWriter;
        SHA256Writer executedAnchorWriter;
        std::map<std::uint32_t, std::array<double, 3u>> donorCOMOffsets;
        for (const auto& partition : evidence.partitions) {
            if (!donorCOMOffsets.emplace(
                    partition.donorBody,
                    partition.remainingCOMOffsetM).second) {
                return fail(error,
                    "loaded-knee executable anchor binding has repeated donor offsets");
            }
        }
        std::size_t objectIndex = 0u;
        std::size_t attachmentIndex = 0u;
        for (std::size_t index = 0u; index < world.fem.nodes.size(); ++index) {
            const auto& node = world.fem.nodes[index];
            if (!std::isfinite(node.restAndFixed.x) ||
                !std::isfinite(node.restAndFixed.y) ||
                !std::isfinite(node.restAndFixed.z) ||
                !referenceWriter.appendFloat(node.restAndFixed.x) ||
                !referenceWriter.appendFloat(node.restAndFixed.y) ||
                !referenceWriter.appendFloat(node.restAndFixed.z)) {
                return fail(error,
                    "loaded-knee executable Matter x_ref arena is invalid");
            }

            const std::uint32_t sourceGlobal =
                decoded.sourceGlobalNodeIndices[index];
            const auto& sourceNode = decodedPayload.nodes[sourceGlobal];
            const auto& anchor = executedAnchors[index];
            while (objectIndex + 1u < world.objects.size() &&
                   index >= static_cast<std::size_t>(
                       world.objects[objectIndex].stateOffset) +
                       world.objects[objectIndex].stateCount) {
                ++objectIndex;
            }
            const auto& object = world.objects[objectIndex];
            if (index < object.stateOffset ||
                index >= static_cast<std::size_t>(object.stateOffset) +
                    object.stateCount) {
                return fail(error,
                    "loaded-knee executable node escaped its profile-order FEM object span");
            }
            std::array<double, 3u> donorCOMOffset{};
            if (const auto offset = donorCOMOffsets.find(
                    sourceNode.anchorBodyIndex);
                offset != donorCOMOffsets.end()) {
                donorCOMOffset = offset->second;
            }
            if (!validateNumiHumanLoadedKneeExecutedAnchorRowV1(
                    sourceGlobal, sourceNode, donorCOMOffset, node, anchor,
                    error) ||
                !executedAnchorWriter.appendU32(sourceGlobal) ||
                !executedAnchorWriter.appendU32(anchor.bodyIndex) ||
                !executedAnchorWriter.appendU32(anchor.flags) ||
                !executedAnchorWriter.appendFloat(anchor.localPoint[0u]) ||
                !executedAnchorWriter.appendFloat(anchor.localPoint[1u]) ||
                !executedAnchorWriter.appendFloat(anchor.localPoint[2u])) {
                return fail(error,
                    "loaded-knee executed anchor table is not the ABI3 profile-order ownership map");
            }

            if (attachmentIndex < world.fem.humanAttachments.size() &&
                world.fem.humanAttachments[attachmentIndex].identity.x <
                    index) {
                return fail(error,
                    "loaded-knee compiled attachment row is out of profile-order or unclaimed");
            }
            if (sourceNode.rigidlyAttached) {
                if (attachmentIndex >= world.fem.humanAttachments.size() ||
                    !validateNumiHumanLoadedKneeCompiledAttachmentRowV1(
                        static_cast<std::uint32_t>(index),
                        static_cast<std::uint32_t>(objectIndex),
                        sourceNode, anchor, node,
                        world.fem.humanAttachments[attachmentIndex], error)) {
                    return fail(error,
                        "loaded-knee active ABI3 anchor is missing its exact compiled Matter attachment row");
                }
                ++attachmentIndex;
            } else if (attachmentIndex <
                           world.fem.humanAttachments.size() &&
                       world.fem.humanAttachments[attachmentIndex]
                               .identity.x == index) {
                return fail(error,
                    "loaded-knee inactive ABI3 node owns an unexpected compiled Matter attachment row");
            }
        }
        if (attachmentIndex != world.fem.humanAttachments.size()) {
            return fail(error,
                "loaded-knee compiled Matter attachment table contains trailing or unbound rows");
        }

        for (std::size_t index = 0u;
             index < world.fem.tetrahedra.size(); ++index) {
            const auto& observed = world.fem.tetrahedra[index].nodes;
            const auto& expected = decoded.tetrahedra[index];
            const std::array<std::uint32_t, 4u> actual{{
                observed.x, observed.y, observed.z, observed.w}};
            if (actual != expected) {
                return fail(error,
                    "loaded-knee Matter tetrahedron connectivity/order differs from HumanPack");
            }
            for (const std::uint32_t node : actual) {
                if (!executableTopologyWriter.appendU32(node)) {
                    return fail(error,
                        "loaded-knee Matter topology hashing failed");
                }
            }
        }

        NumiHumanLoadedKneeDigest observedReference{};
        NumiHumanLoadedKneeDigest observedTopology{};
        NumiHumanLoadedKneeDigest observedAnchors{};
        if (!referenceWriter.finish(observedReference) ||
            observedReference != evidence.referenceCoordinateSHA256 ||
            !executableTopologyWriter.finish(observedTopology) ||
            observedTopology !=
                authoring.topology.executableFEMTopologySHA256 ||
            !executedAnchorWriter.finish(observedAnchors) ||
            !digestPresent(observedAnchors)) {
            return fail(error,
                "loaded-knee executable x_ref, topology, or anchor identity differs");
        }

        NumiHumanLoadedKneeMassEvidenceV1 candidate = evidence;
        candidate.sourceGlobalNodeIndexSHA256 =
            decoded.sourceGlobalNodeIndexSHA256;
        candidate.executableFEMTopologySHA256 = observedTopology;
        candidate.sourceAnchorOwnershipSHA256 =
            decoded.anchorOwnershipSHA256;
        candidate.executedAnchorOwnershipSHA256 = observedAnchors;
        evidence = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee executable topology binding failed";
        return false;
    }
}

bool bindNumiHumanLoadedKneeMaterialExecutionV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const std::filesystem::path& exactMatterSourcePath,
    const numi::matter::CompiledWorld& world,
    const std::span<const NumiHumanLoadedKneeExecutedAnchorV1>
        executedAnchors,
    const std::span<const NumiHumanLoadedKneePassiveOwnerV1>
        executedPassiveOwners,
    const std::span<const NMNumiHumanPassiveLigamentGPU>
        executedPassiveLigaments,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
) {
    try {
        if (!validateNumiHumanLoadedKneeAuthoringV1(
                authoring, nullptr, error)) return false;
        std::vector<char> exactMatterSourceBytes;
        if (!readVerifiedImmutableFile(
                exactMatterSourcePath, kExactFEBioMatterSourceSHA256,
                kExactFEBioMatterSourceBytes, exactMatterSourceBytes,
                error)) {
            return fail(error,
                "loaded-knee material source is not the pinned exact FEBio exp-linear asset");
        }
        const auto parsed = numi::matter::parseMatter(std::string_view(
            exactMatterSourceBytes.data(), exactMatterSourceBytes.size()));
        if (!parsed.succeeded() ||
            parsed.material.name !=
                "open_knee_ligament_febio_exp_linear_v1" ||
            std::count_if(
                parsed.material.expressions.nodes.begin(),
                parsed.material.expressions.nodes.end(),
                [](const numi::matter::Expr& expression) {
                    return expression.kind ==
                        numi::matter::ExprKind::fiberExpLinear;
                }) != 1) {
            return fail(error,
                "loaded-knee material source did not parse as the exact FEBio fibre primitive");
        }

        NumiHumanLoadedKneeDigest observedExecutedAnchors{};
        if (executedAnchors.size() != world.fem.nodes.size() ||
            !digestNumiHumanLoadedKneeExecutedAnchorsV1(
                executedAnchors, observedExecutedAnchors, error) ||
            observedExecutedAnchors !=
                evidence.executedAnchorOwnershipSHA256 ||
            evidence.sourceGlobalNodeIndexSHA256 !=
                authoring.topology.sourceGlobalNodeIndexSHA256 ||
            evidence.executableFEMTopologySHA256 !=
                authoring.topology.executableFEMTopologySHA256 ||
            evidence.sourceAnchorOwnershipSHA256 !=
                authoring.topology.anchorOwnershipSHA256) {
            return fail(error,
                "loaded-knee material geometry is not bound to the topology-authenticated adapter anchors");
        }

        std::string layoutError;
        if (!numi::matter::validateCompiledWorldLayout(
                world, &layoutError) || world.physicsFingerprint == 0u ||
            world.constitutive.size() !=
                kNumiHumanLoadedKneeRegionCount ||
            world.materials.size() !=
                kNumiHumanLoadedKneeRegionCount ||
            world.objects.size() != kNumiHumanLoadedKneeRegionCount ||
            executedPassiveOwners.size() !=
                kNumiHumanLoadedKneePassiveOwnerCount ||
            executedPassiveLigaments.size() !=
                kNumiHumanLoadedKneePassiveOwnerCount ||
            evidence.executedSourcePhysicsFingerprint !=
                world.physicsFingerprint ||
            !digestPresent(evidence.closureSHA256) ||
            !digestPresent(evidence.executedRawF32NodeMassSHA256)) {
            return fail(error,
                "loaded-knee material binding requires the exact validated six-program Matter world and prior mass binding: " +
                    layoutError);
        }

        SHA256Writer writer;
        bool composed = writer.appendString(kMaterialExecutionDomain) &&
            writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) &&
            writer.appendDigest(kExactFEBioMatterSourceSHA256) &&
            writer.appendDigest(authoring.manifestSHA256) &&
            writer.appendU64(world.physicsFingerprint) &&
            writer.appendU64(world.fingerprint);
        if (!composed) {
            return fail(error,
                "loaded-knee material execution identity initialization failed");
        }

        const auto parameterIndex = [&parsed](const std::string_view name) {
            const auto found = std::find_if(
                parsed.material.parameters.begin(),
                parsed.material.parameters.end(),
                [name](const numi::matter::Parameter& parameter) {
                    return parameter.name == name;
                });
            return found == parsed.material.parameters.end()
                ? std::numeric_limits<std::size_t>::max()
                : static_cast<std::size_t>(std::distance(
                    parsed.material.parameters.begin(), found));
        };
        const std::array<std::string_view, 13u> exactParameterNames{{
            "density", "c1", "c3", "c4", "c5", "lambda_max",
            "fiber_scale", "bulk", "initial_stretch", "fiber_x",
            "fiber_y", "fiber_z", "numerical_viscosity"}};
        for (std::size_t index = 0u;
             index < exactParameterNames.size(); ++index) {
            if (parameterIndex(exactParameterNames[index]) != index) {
                return fail(error,
                    "loaded-knee pinned FEBio material parameter order drifted");
            }
        }

        const auto expectedParameterValue = [
            &authoring, &exactParameterNames](
            const std::size_t region,
            const std::size_t parameter) -> std::optional<float> {
            const auto& material = authoring.regions[region].material;
            const auto name = exactParameterNames[parameter];
            if (name == "density")
                return static_cast<float>(authoring.densityRuntimeKgPerM3);
            if (name == "c1") return sourcePressureFloat(material.c1Pascals);
            if (name == "c3") return sourcePressureFloat(material.c3Pascals);
            if (name == "c4") return static_cast<float>(material.c4);
            if (name == "c5") return sourcePressureFloat(material.c5Pascals);
            if (name == "lambda_max")
                return static_cast<float>(material.lambdaMaximum);
            if (name == "fiber_scale")
                return region < kNumiHumanLoadedKneePassiveOwnerCount
                    ? 0.0f : 1.0f;
            if (name == "bulk")
                return sourcePressureFloat(material.bulkModulusPascals);
            if (name == "initial_stretch")
                return region < kNumiHumanLoadedKneePassiveOwnerCount
                    ? 1.0f
                    : static_cast<float>(material.initialStretch);
            if (name == "numerical_viscosity") return 25.0f;
            return std::nullopt;
        };

        const auto& firstProgram = world.constitutive.front();
        for (std::size_t region = 0u;
             region < kNumiHumanLoadedKneeRegionCount; ++region) {
            const auto& program = world.constitutive[region];
            const std::string expectedName =
                "open_knee_" + authoring.regions[region].sourceRegionName +
                "_live_human_febio_exp_linear_v1";
            if (program.material.name != expectedName) {
                return fail(error,
                    "loaded-knee compiled material name is not exact FEBio regional identity at " +
                        std::to_string(region));
            }

            // compileWorld retains the compiler-expanded expression graph in
            // ConstitutiveProgram::material. Recompile an independently
            // reconstructed regional source program before comparing that
            // graph; comparing it directly with the 303-node parsed source
            // would reject every valid differentiated program.
            auto expectedMaterial = parsed.material;
            expectedMaterial.name = expectedName;
            expectedMaterial.fingerprint = 0u;
            for (std::size_t parameter = 0u;
                 parameter < expectedMaterial.parameters.size(); ++parameter) {
                auto& expectedParameter =
                    expectedMaterial.parameters[parameter];
                expectedParameter.identifiable = false;
                const auto expected = expectedParameterValue(
                    region, parameter);
                if (expected.has_value()) {
                    expectedParameter.defaultValue = *expected;
                } else {
                    // Fibre axes are deterministically rotated into the
                    // projected reference frame by the topology/anchor cook.
                    // Their exact executed values are checked below; retain
                    // those values while independently compiling the pinned
                    // constitutive source structure.
                    expectedParameter.defaultValue =
                        program.material.parameters[parameter].defaultValue;
                }
            }
            const auto expectedCompiled =
                numi::matter::detail::compileConstitutive(
                    expectedMaterial, NM_EXPRESSION_STACK_CAPACITY);
            if (!expectedCompiled.succeeded()) {
                return fail(error,
                    "loaded-knee independent exact FEBio compilation failed at " +
                        std::to_string(region));
            }
            const std::string intrinsicMismatch =
                exactFEBioIntrinsicMismatch(
                    program.material, expectedCompiled.program.material);
            if (!intrinsicMismatch.empty()) {
                return fail(error,
                    "loaded-knee compiled material IR is not the pinned FEBio intrinsic at " +
                        std::to_string(region) + ": " + intrinsicMismatch);
            }
            if (!sameCompiledIntrinsic(
                    program, expectedCompiled.program)) {
                return fail(error,
                    "loaded-knee compiled material bytecode is not the independently compiled pinned FEBio intrinsic at " +
                        std::to_string(region));
            }
            if (region != 0u &&
                !sameCompiledIntrinsic(program, firstProgram)) {
                return fail(error,
                    "loaded-knee regional compiled bytecode differs at " +
                        std::to_string(region));
            }
            if (program.parameters.size() !=
                    parsed.material.parameters.size() ||
                program.gpu.parameterCount != program.parameters.size() ||
                program.gpu.constitutiveKind != NM_CONSTITUTIVE_BYTECODE ||
                std::memcmp(&program.gpu, &world.materials[region],
                            sizeof(program.gpu)) != 0) {
                return fail(error,
                    "loaded-knee compiled material descriptor differs at " +
                        std::to_string(region));
            }

            const auto& object = world.objects[region];
            if (object.representation != NM_REPRESENTATION_FEM ||
                object.materialIndex != region ||
                object.elementCount == 0u ||
                object.elementOffset > world.fem.tetrahedra.size() ||
                object.elementCount >
                    world.fem.tetrahedra.size() - object.elementOffset) {
                return fail(error,
                    "loaded-knee continuum object is not bound to its profile-order material");
            }
            for (std::uint32_t local = 0u;
                 local < object.elementCount; ++local) {
                const auto& identity = world.fem.tetrahedra[
                    object.elementOffset + local].identity;
                if (identity.x != region || identity.y != region) {
                    return fail(error,
                        "loaded-knee tetrahedron material/object ownership left profile order");
                }
            }

            double fibreNormSquared = 0.0;
            for (std::size_t parameter = 0u;
                 parameter < program.material.parameters.size(); ++parameter) {
                const auto& authoredParameter =
                    program.material.parameters[parameter];
                const auto& compiledParameter = program.parameters[parameter];
                const auto& globalParameter = world.parameters[
                    program.gpu.parameterOffset + parameter];
                if (authoredParameter.identifiable ||
                    std::memcmp(&compiledParameter, &globalParameter,
                                sizeof(compiledParameter)) != 0 ||
                    !exactFloat(
                        compiledParameter.valueAndBounds.y,
                        static_cast<float>(
                            parsed.material.parameters[parameter].lower)) ||
                    !exactFloat(
                        compiledParameter.valueAndBounds.z,
                        static_cast<float>(
                            parsed.material.parameters[parameter].upper)) ||
                    !exactFloat(
                        compiledParameter.valueAndBounds.w,
                        parsed.material.parameters[parameter].logarithmic
                            ? 1.0f : 0.0f)) {
                    return fail(error,
                        "loaded-knee compiled material parameter metadata or overlay diverged");
                }
                const auto expected = expectedParameterValue(
                    region, parameter);
                if (expected.has_value()) {
                    if (!exactFloat(
                            compiledParameter.valueAndBounds.x, *expected) ||
                        !exactFloat(
                            static_cast<float>(
                                authoredParameter.defaultValue), *expected)) {
                        return fail(error,
                            "loaded-knee compiled material value differs from the Human source row");
                    }
                } else {
                    const float value =
                        compiledParameter.valueAndBounds.x;
                    if (!std::isfinite(value) ||
                        !exactFloat(
                            static_cast<float>(
                                authoredParameter.defaultValue), value)) {
                        return fail(error,
                            "loaded-knee projected fibre parameter is invalid");
                    }
                    fibreNormSquared +=
                        static_cast<double>(value) * value;
                }
            }
            if (std::abs(fibreNormSquared - 1.0) > 2.0e-5) {
                return fail(error,
                    "loaded-knee projected constitutive fibre is not unit length");
            }
            composed = composed &&
                writer.appendString(authoring.regions[region].semanticID) &&
                writer.appendString(authoring.regions[region].materialOwnerID) &&
                writer.appendString(
                    authoring.regions[region].activeForceOwnerStatus) &&
                writer.appendString(
                    authoring.regions[region].activeForceOwnerID) &&
                appendConstitutiveAuthority(writer, program) &&
                writer.appendString("NMContinuumObjectGPU") &&
                writer.appendU64(sizeof(object)) &&
                writer.append(&object, sizeof(object));
        }

        constexpr std::array<std::uint32_t,
                             kNumiHumanLoadedKneePassiveOwnerCount>
            firstBodies{{
                NUMI_HUMAN_KNEE_FEMUR_BODY,
                NUMI_HUMAN_KNEE_FEMUR_BODY,
                NUMI_HUMAN_KNEE_FEMUR_BODY,
                NUMI_HUMAN_KNEE_FEMUR_BODY,
                NUMI_HUMAN_KNEE_PATELLA_BODY}};
        std::array<NMNumiHumanPassiveLigamentGPU,
                   kNumiHumanLoadedKneePassiveOwnerCount>
            expectedPassiveRows{};
        for (std::size_t index = 0u;
             index < expectedPassiveRows.size(); ++index) {
            const auto& object = world.objects[index];
            if (object.stateCount == 0u ||
                object.stateOffset > world.fem.nodes.size() ||
                object.stateCount >
                    world.fem.nodes.size() - object.stateOffset) {
                return fail(error,
                    "loaded-knee passive geometry object node span is invalid");
            }
            std::array<float, 3u> firstLocal{};
            std::array<float, 3u> secondLocal{};
            std::array<float, 3u> firstReference{};
            std::array<float, 3u> secondReference{};
            std::uint32_t firstCount = 0u;
            std::uint32_t secondCount = 0u;
            for (std::uint32_t local = 0u;
                 local < object.stateCount; ++local) {
                const std::uint32_t nodeIndex = object.stateOffset + local;
                const auto& anchor = executedAnchors[nodeIndex];
                if (anchor.flags == 0u) continue;
                if (anchor.flags != 1u) {
                    return fail(error,
                        "loaded-knee passive geometry has an invalid adapter-anchor flag");
                }
                std::array<float, 3u>* localSum = nullptr;
                std::array<float, 3u>* referenceSum = nullptr;
                std::uint32_t* count = nullptr;
                if (anchor.bodyIndex == firstBodies[index]) {
                    localSum = &firstLocal;
                    referenceSum = &firstReference;
                    count = &firstCount;
                } else if (anchor.bodyIndex ==
                           NUMI_HUMAN_KNEE_TIBIA_BODY) {
                    localSum = &secondLocal;
                    referenceSum = &secondReference;
                    count = &secondCount;
                } else {
                    continue;
                }
                const auto& reference = world.fem.nodes[nodeIndex].restAndFixed;
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    (*localSum)[axis] += anchor.localPoint[axis];
                    (*referenceSum)[axis] += (&reference.x)[axis];
                }
                ++*count;
            }
            if (firstCount == 0u || secondCount == 0u) {
                return fail(error,
                    "loaded-knee passive geometry lacks both exact entheses");
            }
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                firstLocal[axis] *= 1.0f / static_cast<float>(firstCount);
                secondLocal[axis] *= 1.0f / static_cast<float>(secondCount);
                firstReference[axis] *=
                    1.0f / static_cast<float>(firstCount);
                secondReference[axis] *=
                    1.0f / static_cast<float>(secondCount);
            }
            const std::array<float, 3u> referenceAxis{{
                secondReference[0u] - firstReference[0u],
                secondReference[1u] - firstReference[1u],
                secondReference[2u] - firstReference[2u]}};
            const float referenceLength = std::sqrt(
                referenceAxis[0u] * referenceAxis[0u] +
                referenceAxis[1u] * referenceAxis[1u] +
                referenceAxis[2u] * referenceAxis[2u]);
            double volume = 0.0;
            for (std::uint32_t local = 0u;
                 local < object.elementCount; ++local) {
                const auto& nodes = world.fem.tetrahedra[
                    object.elementOffset + local].nodes;
                const std::array<std::uint32_t, 4u> indices{{
                    nodes.x, nodes.y, nodes.z, nodes.w}};
                std::array<std::array<float, 3u>, 4u> points{};
                for (std::size_t corner = 0u;
                     corner < points.size(); ++corner) {
                    if (indices[corner] >= world.fem.nodes.size()) {
                        return fail(error,
                            "loaded-knee passive geometry tetrahedron is out of range");
                    }
                    const auto& point =
                        world.fem.nodes[indices[corner]].restAndFixed;
                    points[corner] = {point.x, point.y, point.z};
                }
                std::array<float, 3u> ab{};
                std::array<float, 3u> ac{};
                std::array<float, 3u> ad{};
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    ab[axis] = points[1u][axis] - points[0u][axis];
                    ac[axis] = points[2u][axis] - points[0u][axis];
                    ad[axis] = points[3u][axis] - points[0u][axis];
                }
                const std::array<float, 3u> cross{{
                    ac[1u] * ad[2u] - ac[2u] * ad[1u],
                    ac[2u] * ad[0u] - ac[0u] * ad[2u],
                    ac[0u] * ad[1u] - ac[1u] * ad[0u]}};
                const float triple = ab[0u] * cross[0u] +
                    ab[1u] * cross[1u] + ab[2u] * cross[2u];
                volume += std::abs(static_cast<double>(triple)) / 6.0;
            }
            const float effectiveArea = static_cast<float>(
                volume / static_cast<double>(referenceLength));
            const auto& material = authoring.regions[index].material;
            auto& expected = expectedPassiveRows[index];
            expected.firstBodyIndex = firstBodies[index];
            expected.secondBodyIndex = NUMI_HUMAN_KNEE_TIBIA_BODY;
            expected.flags = NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE;
            expected.firstLocalPoint = {
                firstLocal[0u], firstLocal[1u], firstLocal[2u], 0.0f};
            expected.secondLocalPoint = {
                secondLocal[0u], secondLocal[1u], secondLocal[2u], 0.0f};
            expected.material = {
                sourcePressureFloat(material.c3Pascals),
                static_cast<float>(material.c4),
                sourcePressureFloat(material.c5Pascals),
                static_cast<float>(material.lambdaMaximum)};
            expected.reference = {
                referenceLength, effectiveArea,
                static_cast<float>(material.initialStretch), 0.0f};
            if (!finiteFloat4XYZ(expected.firstLocalPoint) ||
                !finiteFloat4XYZ(expected.secondLocalPoint) ||
                !std::isfinite(referenceLength) ||
                referenceLength <= 1.0e-4f ||
                !std::isfinite(effectiveArea) ||
                effectiveArea <= 1.0e-8f || effectiveArea >= 0.01f) {
                return fail(error,
                    "loaded-knee independently derived passive geometry is invalid");
            }
        }
        for (std::size_t index = 0u;
             index < kNumiHumanLoadedKneePassiveOwnerCount; ++index) {
            const auto& owner = executedPassiveOwners[index];
            const auto& row = executedPassiveLigaments[index];
            const auto& expected = expectedPassiveRows[index];
            if (!samePassiveOwner(owner, authoring.passiveOwners[index]) ||
                owner.sourceRegionName != kPassiveNames[index] ||
                row.firstBodyIndex != expected.firstBodyIndex ||
                row.secondBodyIndex != expected.secondBodyIndex ||
                row.flags != expected.flags ||
                row.reserved0 != expected.reserved0 ||
                !sameFloat4Bits(
                    row.firstLocalPoint, expected.firstLocalPoint) ||
                !sameFloat4Bits(
                    row.secondLocalPoint, expected.secondLocalPoint) ||
                !sameFloat4Bits(row.material, expected.material) ||
                !sameFloat4Bits(row.reference, expected.reference)) {
                return fail(error,
                    "loaded-knee reduced passive row differs from Human material, geometry, or owner mapping");
            }
            composed = composed && writer.appendString(owner.semanticID) &&
                writer.appendString(owner.sourceRegionName) &&
                writer.appendString(owner.ownerID) &&
                writer.appendString("NMNumiHumanPassiveLigamentGPU") &&
                writer.appendU64(sizeof(row)) &&
                writer.append(&row, sizeof(row));
        }

        NumiHumanLoadedKneeDigest materialExecution{};
        if (!composed || !writer.finish(materialExecution) ||
            !digestPresent(materialExecution)) {
            return fail(error,
                "loaded-knee material execution SHA-256 failed");
        }
        NumiHumanLoadedKneeMassEvidenceV1 candidate = evidence;
        candidate.materialExecutionSHA256 = materialExecution;
        evidence = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee material execution binding failed";
        return false;
    }
}

bool bindNumiHumanLoadedKneeMassToRigidExecutionV1(
    const EngineModel& executedModel,
    NumiHumanLoadedKneeMassEvidenceV1& evidence,
    std::string& error
) {
    try {
        const std::uint64_t expected =
            engineModelFingerprint(evidence.rebasedModel);
        const std::uint64_t executed =
            engineModelFingerprint(executedModel);
        if (expected == 0u || executed == 0u || executed != expected) {
            return fail(error,
                "loaded-knee articulated execution did not use the rebased donor model");
        }
        NumiHumanLoadedKneeMassEvidenceV1 candidate = evidence;
        candidate.executedRigidModelFingerprint = executed;
        evidence = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee rigid-execution mass binding failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeXCurrentV1(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    const std::uint32_t firstNode,
    const std::uint32_t nodeCount,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        if (!snapshot.available || nodeCount == 0u ||
            firstNode > snapshot.femNodes.size() ||
            nodeCount > snapshot.femNodes.size() - firstNode) {
            return fail(error, "loaded-knee x_current snapshot range is invalid");
        }
        SHA256Writer writer;
        for (std::uint32_t index = 0u; index < nodeCount; ++index) {
            const auto& node = snapshot.femNodes[firstNode + index];
            if (!std::isfinite(node.positionAndMass.x) ||
                !std::isfinite(node.positionAndMass.y) ||
                !std::isfinite(node.positionAndMass.z) ||
                !writer.appendFloat(node.positionAndMass.x) ||
                !writer.appendFloat(node.positionAndMass.y) ||
                !writer.appendFloat(node.positionAndMass.z)) {
                return fail(error,
                    "loaded-knee x_current contains nonfinite or unhashable coordinates");
            }
        }
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error, "loaded-knee x_current SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool digestNumiHumanLoadedKneeCoordinatesV1(
    const std::span<const std::array<float, 3u>> coordinates,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        if (coordinates.size() != kNumiHumanLoadedKneeLoadedNodeCount) {
            return fail(error,
                "loaded-knee coordinate identity requires exactly 62402 XYZ rows");
        }
        SHA256Writer writer;
        for (const auto& coordinate : coordinates) {
            for (const float value : coordinate) {
                if (!std::isfinite(value) || !writer.appendFloat(value)) {
                    return fail(error,
                        "loaded-knee coordinate identity contains nonfinite or unhashable values");
                }
            }
        }
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error, "loaded-knee coordinate SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee coordinate SHA-256 failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeCoordinateDerivationV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanLoadedKneeDigest& initialCoordinateSHA256,
    const NumiHumanLoadedKneeDigest& referencePoseSHA256,
    const NumiHumanLoadedKneeDigest& initialPoseSHA256,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        if (!digestPresent(authoring.coordinates.sourceStateSHA256) ||
            !digestPresent(authoring.coordinates.referenceStateSHA256) ||
            !digestPresent(initialCoordinateSHA256) ||
            !digestPresent(referencePoseSHA256) ||
            !digestPresent(initialPoseSHA256) ||
            referencePoseSHA256 !=
                authoring.projectedReferencePoseSHA256 ||
            !digestPresent(authoring.sourceDefaultPoseSHA256) ||
            !digestPresent(
                authoring.sourceToReferenceMappingCodeSHA256) ||
            authoring.coordinates.constructionID.empty()) {
            return fail(error,
                "loaded-knee A-to-B-to-C coordinate derivation identity is incomplete");
        }
        SHA256Writer writer;
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.appendString(
                "numi.lab.loaded-knee.coordinate-derivation.v1") ||
            !writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) ||
            !writer.appendString(authoring.coordinates.constructionID) ||
            !writer.appendString(authoring.sourceDefaultPoseID) ||
            !writer.appendDigest(authoring.sourceDefaultPoseSHA256) ||
            !writer.appendString(authoring.projectedReferencePoseID) ||
            !writer.appendString(authoring.sourceToReferenceMappingID) ||
            !writer.appendString(
                authoring.sourceToReferenceMappingAlgorithm) ||
            !writer.appendDigest(
                authoring.sourceToReferenceMappingCodeSHA256) ||
            !writer.appendDigest(authoring.sourceRigidPayloadSHA256) ||
            !writer.appendDigest(authoring.equalityPayloadSHA256) ||
            !writer.appendDigest(authoring.coordinates.sourceStateSHA256) ||
            !writer.appendDigest(authoring.coordinates.referenceStateSHA256) ||
            !writer.appendDigest(initialCoordinateSHA256) ||
            !writer.appendDigest(referencePoseSHA256) ||
            !writer.appendDigest(initialPoseSHA256) ||
            !writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error,
                "loaded-knee A-to-B-to-C coordinate derivation SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee coordinate derivation SHA-256 failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeExternalComponentV1(
    const std::string_view componentID,
    const std::span<const std::byte> bytes,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        constexpr std::array<std::string_view, 3u> componentIDs{{
            "articulated-q-v-root-time",
            "muscle-activation-tendon-transfer-corrections",
            "adapter-passive-articular-accepted-state",
        }};
        if (bytes.empty() ||
            std::find(componentIDs.begin(), componentIDs.end(),
                      componentID) == componentIDs.end()) {
            return fail(error,
                "loaded-knee external full-state component is empty or undeclared");
        }
        SHA256Writer writer;
        NumiHumanLoadedKneeDigest candidate{};
        if (!writer.appendString(
                "numi.lab.loaded-knee.external-full-state.v1") ||
            !writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) ||
            !writer.appendString(componentID) ||
            !writer.appendU64(bytes.size()) ||
            !writer.append(bytes.data(), bytes.size()) ||
            !writer.finish(candidate) || !digestPresent(candidate)) {
            return fail(error,
                "loaded-knee external full-state component SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee external full-state component SHA-256 failed";
        return false;
    }
}

bool digestNumiHumanLoadedKneeMatterSnapshotV1(
    const numi::matter::RuntimeStateSnapshot& snapshot,
    NumiHumanLoadedKneeDigest& output,
    std::string& error
) {
    try {
        if (!snapshot.available || snapshot.sourcePhysicsFingerprint == 0u ||
            snapshot.deviceProgramFingerprint == 0u) {
            return fail(error,
                "loaded-knee full Matter snapshot authority is unavailable");
        }
        SHA256Writer writer;
        const bool scalar = writer.appendString(kSnapshotDomain) &&
            writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) &&
            writer.appendU64(snapshot.sourcePhysicsFingerprint) &&
            writer.appendU64(snapshot.deviceProgramFingerprint) &&
            writer.appendU32(snapshot.controlStep) &&
            writer.appendU32(snapshot.physicsSubstep) &&
            writer.appendU32(snapshot.identificationGeneration) &&
            writer.appendU32(snapshot.identificationCheckpoint) &&
            writer.appendU32(snapshot.identificationAdvanced ? 1u : 0u) &&
            writer.appendU64(snapshot.sutureProxyBindingRevision) &&
            writer.appendU32(snapshot.coupledTimestepMultiplier) &&
            writer.appendU32(snapshot.coupledTimestepDivisor) &&
            writer.appendU32(snapshot.fgmresIterationBudgetOverride) &&
            writer.appendU32(snapshot.newtonIterationBudgetOverride) &&
            writer.appendU32(snapshot.allocationGeneration) &&
            writer.appendU32(snapshot.learnedWeightRevision) &&
            writer.appendU32(snapshot.materialStateStride);
        const bool vectors = writer.appendVector(
                "u32:sutureProxyEdges", snapshot.sutureProxyEdges) &&
            writer.appendVector("NMParticleStateGPU", snapshot.particles) &&
            writer.appendVector("NMFEMNodeStateGPU", snapshot.femNodes) &&
            writer.appendVector("NMFEMFieldStateGPU", snapshot.femFields) &&
            writer.appendVector("nm_float4:vascularState",
                                snapshot.vascularState) &&
            writer.appendVector("NMVascularClockGPU", snapshot.vascularClock) &&
            writer.appendVector("NMFEMTopologyNodeGPU",
                                snapshot.femTopologyNodes) &&
            writer.appendVector("NMTetrahedronGPU",
                                snapshot.femTopologyTetrahedra) &&
            writer.appendVector("NMCohesiveFaceGPU", snapshot.cohesiveFaces) &&
            writer.appendVector("NMPunctureChannelGPU",
                                snapshot.punctureChannels) &&
            writer.appendVector("NMFEMTopologyStateGPU",
                                snapshot.topologyStates) &&
            writer.appendVector("NMMatterStatusGPU", snapshot.statuses) &&
            writer.appendVector("NMSolverCertificateGPU",
                                snapshot.solverCertificates) &&
            writer.appendVector("u32:mpmActiveNodeIndices",
                                snapshot.mpmActiveNodeIndices) &&
            writer.appendVector("u32:mpmNodeToActive",
                                snapshot.mpmNodeToActive) &&
            writer.appendVector("u32:mpmActiveNodeCounts",
                                snapshot.mpmActiveNodeCounts) &&
            writer.appendVector("f32:rigidGeneralizedCandidate",
                                snapshot.rigidGeneralizedCandidate) &&
            writer.appendVector("f32:learnedWeights", snapshot.learnedWeights) &&
            writer.appendVector("NMAdaptiveStateGPU", snapshot.adaptive) &&
            writer.appendVector("NMSchedulerStateGPU", snapshot.schedulers) &&
            writer.appendVector("NMRigidReactionGPU", snapshot.reactions) &&
            writer.appendVector("NMRigidStateGPU", snapshot.rigidStates) &&
            writer.appendVector("NMContactSampleGPU", snapshot.contactSamples) &&
            writer.appendVector("nm_float4:contactHistories",
                                snapshot.contactHistories) &&
            writer.appendVector("nm_float4:humanSupportHistories",
                                snapshot.humanSupportHistories) &&
            writer.appendVector("NMHumanSupportConsequenceGPU",
                                snapshot.humanSupportConsequences) &&
            writer.appendVector("NMDeformableContactHistoryGPU",
                                snapshot.deformableContactHistories) &&
            writer.appendVector("f32:particleMaterialState",
                                snapshot.particleMaterialState) &&
            writer.appendVector("f32:femMaterialState",
                                snapshot.femMaterialState) &&
            writer.appendVector("NMIdentificationDistributionGPU",
                                snapshot.identification) &&
            writer.appendVector("f32:environmentParameters",
                                snapshot.environmentParameters);
        NumiHumanLoadedKneeDigest candidate{};
        if (!scalar || !vectors || !writer.finish(candidate) ||
            !digestPresent(candidate)) {
            return fail(error,
                "loaded-knee full Matter snapshot SHA-256 failed");
        }
        output = candidate;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool acceptNumiHumanLoadedKneeRuntimeStateV1(
    const NumiHumanLoadedKneeAuthoringV1& authoring,
    const NumiHumanLoadedKneeMassEvidenceV1& mass,
    const NumiHumanLoadedKneeRuntimeEvidenceV1& evidence,
    const NumiHumanLoadedKneeAcceptanceReceiptV1* previous,
    NumiHumanLoadedKneeAcceptanceReceiptV1& output,
    std::string& error
) {
    try {
        if (!validateNumiHumanLoadedKneeAuthoringV1(
                authoring, nullptr, error)) return false;
        if (mass.partitions.empty() ||
            mass.cookedNodes.size() !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            !digestPresent(mass.closureSHA256) ||
            !digestPresent(mass.cookedNodeMassSHA256) ||
            mass.authoredRawF32NodeMassSHA256 !=
                authoring.authoredRawF32NodeMassSHA256 ||
            mass.executedRawF32NodeMassSHA256 !=
                authoring.authoredRawF32NodeMassSHA256 ||
            mass.referenceCoordinateSHA256 !=
                authoring.coordinates.referenceStateSHA256 ||
            mass.sourceGlobalNodeIndexSHA256 !=
                authoring.topology.sourceGlobalNodeIndexSHA256 ||
            mass.executableFEMTopologySHA256 !=
                authoring.topology.executableFEMTopologySHA256 ||
            mass.sourceAnchorOwnershipSHA256 !=
                authoring.topology.anchorOwnershipSHA256 ||
            !digestPresent(mass.executedAnchorOwnershipSHA256) ||
            !digestPresent(mass.materialExecutionSHA256) ||
            mass.sourceRigidModelFingerprint == 0u ||
            mass.executedSourcePhysicsFingerprint == 0u ||
            mass.executedRigidModelFingerprint == 0u ||
            engineModelFingerprint(mass.rebasedModel) !=
                mass.executedRigidModelFingerprint ||
            !std::isfinite(mass.maximumPackedMomentRelativeError) ||
            mass.maximumPackedMomentRelativeError >
                16.0 * std::numeric_limits<float>::epsilon()) {
            return fail(error,
                "loaded-knee runtime acceptance lacks donor mass/moment closure");
        }
        NumiHumanLoadedKneeDigest expectedCoordinateDerivation{};
        if (!digestNumiHumanLoadedKneeCoordinateDerivationV1(
                authoring, evidence.executedInitialCoordinateSHA256,
                evidence.referencePoseSHA256,
                evidence.initialPoseSHA256,
                expectedCoordinateDerivation, error)) return false;
        if (evidence.formatVersion !=
                kNumiHumanLoadedKneeAcceptanceVersionV1 ||
            evidence.acceptedStepIndex == 0u ||
            evidence.acceptedStepIndex > kNumiHumanLoadedKneeAcceptedStepCount ||
            evidence.timestepNanoseconds !=
                kNumiHumanLoadedKneeStepNanoseconds ||
            evidence.acceptedTimestampNanoseconds !=
                static_cast<std::uint64_t>(evidence.acceptedStepIndex) *
                    kNumiHumanLoadedKneeStepNanoseconds ||
            evidence.snapshot.controlStep !=
                evidence.acceptedStepIndex - 1u ||
            evidence.snapshot.physicsSubstep != 0u ||
            evidence.loadedFEMNodeFirst != 0u ||
            evidence.loadedFEMNodeCount !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            evidence.snapshot.femNodes.size() !=
                kNumiHumanLoadedKneeLoadedNodeCount ||
            evidence.executedSourceCoordinateSHA256 !=
                authoring.coordinates.sourceStateSHA256 ||
            evidence.executedReferenceCoordinateSHA256 !=
                authoring.coordinates.referenceStateSHA256 ||
            !digestPresent(evidence.executedInitialCoordinateSHA256) ||
            evidence.coordinateDerivationSHA256 !=
                expectedCoordinateDerivation ||
            evidence.executedMassClosureSHA256 != mass.closureSHA256 ||
            evidence.executedRawF32NodeMassSHA256 !=
                mass.executedRawF32NodeMassSHA256 ||
            evidence.executedSourceGlobalNodeIndexSHA256 !=
                mass.sourceGlobalNodeIndexSHA256 ||
            evidence.executedFEMTopologySHA256 !=
                mass.executableFEMTopologySHA256 ||
            evidence.executedSourceAnchorOwnershipSHA256 !=
                mass.sourceAnchorOwnershipSHA256 ||
            evidence.executedAnchorOwnershipSHA256 !=
                mass.executedAnchorOwnershipSHA256 ||
            evidence.executedMaterialExecutionSHA256 !=
                mass.materialExecutionSHA256 ||
            evidence.executedSourceRigidModelFingerprint !=
                mass.sourceRigidModelFingerprint ||
            evidence.executedSourcePhysicsFingerprint !=
                mass.executedSourcePhysicsFingerprint ||
            evidence.executedRigidModelFingerprint !=
                mass.executedRigidModelFingerprint ||
            !digestPresent(evidence.articulatedQVRootTimeSHA256) ||
            !digestPresent(evidence.muscleTendonStateSHA256) ||
            !digestPresent(evidence.adapterAcceptedStateSHA256) ||
            evidence.snapshot.sourcePhysicsFingerprint !=
                mass.executedSourcePhysicsFingerprint) {
            return fail(error,
                "loaded-knee runtime evidence does not describe an exact 50us accepted state");
        }
        if (std::any_of(
                evidence.contactPairNormalForceNewtons.begin(),
                evidence.contactPairNormalForceNewtons.end(),
                [](const double value) {
                    return !std::isfinite(value) || value < 0.0;
                }) ||
            std::none_of(
                evidence.contactPairNormalForceNewtons.begin(),
                evidence.contactPairNormalForceNewtons.end(),
                [](const double value) { return value > 0.0; }) ||
            std::any_of(
                evidence.prescribedClosurePairForceNewtons.begin(),
                evidence.prescribedClosurePairForceNewtons.end(),
                [](const double value) { return !finitePositive(value); })) {
            return fail(error,
                "loaded-knee contact requires finite nonnegative actual pairs, an active pair, and seven positive prescribed-closure operators");
        }
        double measuredContactSum = 0.0;
        double measuredContactL1 = 0.0;
        for (const double value :
             evidence.contactPairNormalForceNewtons) {
            measuredContactSum += value;
            measuredContactL1 += std::abs(value);
        }
        constexpr double epsilon =
            static_cast<double>(std::numeric_limits<float>::epsilon());
        constexpr double gamma =
            kNumiHumanLoadedKneeContactPairCount * epsilon /
            (1.0 - kNumiHumanLoadedKneeContactPairCount * epsilon);
        const double aggregateBound = gamma * measuredContactL1 +
            2.0 * epsilon * std::max(
                1.0, std::abs(evidence.contactAggregateNormalForceNewtons));
        if (!std::isfinite(evidence.contactAggregateNormalForceNewtons) ||
            std::abs(evidence.contactAggregateNormalForceNewtons -
                     measuredContactSum) > aggregateBound) {
            return fail(error,
                "loaded-knee per-pair contact forces do not reproduce the GPU aggregate within a dimension-aware FP32 bound");
        }
        for (std::size_t index = 0u;
             index < evidence.executedContactPairs.size(); ++index) {
            if (!sameContactPair(evidence.executedContactPairs[index],
                                 authoring.contactPairs[index])) {
                return fail(error,
                    "loaded-knee executed articular pair identity/order does not match HumanPack");
            }
        }
        for (std::size_t index = 0u;
             index < evidence.executedActiveReplacements.size(); ++index) {
            if (!sameReplacement(evidence.executedActiveReplacements[index],
                                 authoring.activeReplacements[index])) {
                return fail(error,
                    "loaded-knee executed quadriceps replacement tuple does not match HumanPack");
            }
        }
        for (std::size_t index = 0u;
             index < evidence.executedPassiveOwners.size(); ++index) {
            if (!samePassiveOwner(evidence.executedPassiveOwners[index],
                                  authoring.passiveOwners[index])) {
                return fail(error,
                    "loaded-knee executed passive-owner identity/order does not match HumanPack");
            }
        }
        NumiHumanLoadedKneeDigest previousRoot{};
        if (previous == nullptr) {
            if (evidence.acceptedStepIndex != 1u) {
                return fail(error,
                    "loaded-knee transaction chain must start at accepted step one");
            }
        } else {
            if (previous->formatVersion !=
                    kNumiHumanLoadedKneeAcceptanceVersionV1 ||
                previous->schema != kAcceptanceSchema ||
                previous->sourceOwnershipStatus !=
                    authoring.sourceOwnershipStatus ||
                previous->acceptedStepIndex + 1u !=
                    evidence.acceptedStepIndex ||
                previous->acceptedTimestampNanoseconds +
                        kNumiHumanLoadedKneeStepNanoseconds !=
                    evidence.acceptedTimestampNanoseconds ||
                previous->timestepNanoseconds !=
                    evidence.timestepNanoseconds ||
                previous->humanPackManifestSHA256 !=
                    authoring.manifestSHA256 ||
                previous->massClosureSHA256 != mass.closureSHA256 ||
                previous->rawF32NodeMassSHA256 !=
                    mass.executedRawF32NodeMassSHA256 ||
                previous->sourceGlobalNodeIndexSHA256 !=
                    mass.sourceGlobalNodeIndexSHA256 ||
                previous->executableFEMTopologySHA256 !=
                    mass.executableFEMTopologySHA256 ||
                previous->sourceAnchorOwnershipSHA256 !=
                    mass.sourceAnchorOwnershipSHA256 ||
                previous->executedAnchorOwnershipSHA256 !=
                    mass.executedAnchorOwnershipSHA256 ||
                previous->materialExecutionSHA256 !=
                    mass.materialExecutionSHA256 ||
                previous->xSourceSHA256 !=
                    evidence.executedSourceCoordinateSHA256 ||
                previous->xReferenceSHA256 !=
                    evidence.executedReferenceCoordinateSHA256 ||
                previous->xInitialSHA256 !=
                    evidence.executedInitialCoordinateSHA256 ||
                previous->coordinateDerivationSHA256 !=
                    evidence.coordinateDerivationSHA256 ||
                !digestPresent(previous->articulatedQVRootTimeSHA256) ||
                !digestPresent(previous->muscleTendonStateSHA256) ||
                !digestPresent(previous->adapterAcceptedStateSHA256) ||
                !digestPresent(previous->fullAcceptedStateSHA256) ||
                previous->sourceRigidModelFingerprint !=
                    mass.sourceRigidModelFingerprint ||
                previous->executedRigidModelFingerprint !=
                    mass.executedRigidModelFingerprint ||
                !digestPresent(previous->transactionSHA256) ||
                previous->globalSevenOwnerAcceptedStateRootAvailable ||
                previous->productionQualified || !previous->candidateOnly) {
                return fail(error,
                    "loaded-knee previous transaction receipt is not the exact accepted predecessor");
            }
            previousRoot = previous->transactionSHA256;
        }

        NumiHumanLoadedKneeDigest xCurrent{};
        if (!digestNumiHumanLoadedKneeXCurrentV1(
                evidence.snapshot, evidence.loadedFEMNodeFirst,
                evidence.loadedFEMNodeCount, xCurrent, error)) return false;
        NumiHumanLoadedKneeDigest fullSnapshot{};
        if (!digestNumiHumanLoadedKneeMatterSnapshotV1(
                evidence.snapshot, fullSnapshot, error)) return false;
        if (previous != nullptr &&
            previous->fullMatterSnapshotSHA256 == fullSnapshot) {
            return fail(error,
                "loaded-knee accepted state did not advance full Matter authority");
        }

        SHA256Writer fullStateWriter;
        NumiHumanLoadedKneeDigest fullAcceptedState{};
        if (!fullStateWriter.appendString(
                "numi.lab.loaded-knee.full-accepted-state.v1") ||
            !fullStateWriter.appendU32(
                kNumiHumanLoadedKneeAcceptanceVersionV1) ||
            !fullStateWriter.appendString(authoring.sourceOwnershipStatus) ||
            !fullStateWriter.appendDigest(mass.materialExecutionSHA256) ||
            !fullStateWriter.appendDigest(fullSnapshot) ||
            !fullStateWriter.appendDigest(
                evidence.articulatedQVRootTimeSHA256) ||
            !fullStateWriter.appendDigest(
                evidence.muscleTendonStateSHA256) ||
            !fullStateWriter.appendDigest(
                evidence.adapterAcceptedStateSHA256) ||
            !fullStateWriter.appendU32(evidence.acceptedStepIndex) ||
            !fullStateWriter.appendU64(
                evidence.acceptedTimestampNanoseconds) ||
            !fullStateWriter.finish(fullAcceptedState) ||
            !digestPresent(fullAcceptedState) ||
            (previous != nullptr &&
             previous->fullAcceptedStateSHA256 == fullAcceptedState)) {
            return fail(error,
                "loaded-knee full accepted owner-state SHA-256 failed or did not advance");
        }

        SHA256Writer writer;
        bool composed = writer.appendString(kTransactionDomain) &&
            writer.appendU32(kNumiHumanLoadedKneeAcceptanceVersionV1) &&
            writer.appendDigest(authoring.manifestSHA256) &&
            writer.appendString(authoring.sourceOwnershipStatus) &&
            writer.appendDigest(evidence.executedSourceCoordinateSHA256) &&
            writer.appendDigest(evidence.executedReferenceCoordinateSHA256) &&
            writer.appendDigest(evidence.executedInitialCoordinateSHA256) &&
            writer.appendDigest(evidence.coordinateDerivationSHA256) &&
            writer.appendDigest(mass.closureSHA256) &&
            writer.appendDigest(mass.cookedNodeMassSHA256) &&
            writer.appendDigest(mass.executedRawF32NodeMassSHA256) &&
            writer.appendDigest(mass.sourceGlobalNodeIndexSHA256) &&
            writer.appendDigest(mass.executableFEMTopologySHA256) &&
            writer.appendDigest(mass.sourceAnchorOwnershipSHA256) &&
            writer.appendDigest(mass.executedAnchorOwnershipSHA256) &&
            writer.appendDigest(mass.materialExecutionSHA256) &&
            writer.appendU64(mass.sourceRigidModelFingerprint) &&
            writer.appendU64(mass.executedSourcePhysicsFingerprint) &&
            writer.appendU64(mass.executedRigidModelFingerprint) &&
            writer.appendDigest(previousRoot) && writer.appendDigest(xCurrent) &&
            writer.appendDigest(fullSnapshot) &&
            writer.appendDigest(fullAcceptedState) &&
            writer.appendU32(evidence.acceptedStepIndex) &&
            writer.appendU64(evidence.acceptedTimestampNanoseconds) &&
            writer.appendU64(evidence.timestepNanoseconds);
        for (const double force : evidence.contactPairNormalForceNewtons) {
            composed = composed && writer.appendDouble(force);
        }
        composed = composed &&
            writer.appendDouble(
                evidence.contactAggregateNormalForceNewtons);
        for (const double force :
             evidence.prescribedClosurePairForceNewtons) {
            composed = composed && writer.appendDouble(force);
        }
        NumiHumanLoadedKneeDigest transaction{};
        if (!composed || !writer.finish(transaction) ||
            !digestPresent(transaction) ||
            (previous != nullptr && transaction == previousRoot)) {
            return fail(error,
                "loaded-knee domain-separated transaction SHA-256 failed or did not advance");
        }

        NumiHumanLoadedKneeAcceptanceReceiptV1 candidate;
        candidate.schema = std::string(kAcceptanceSchema);
        candidate.sourceOwnershipStatus = authoring.sourceOwnershipStatus;
        candidate.humanPackManifestSHA256 = authoring.manifestSHA256;
        candidate.massClosureSHA256 = mass.closureSHA256;
        candidate.rawF32NodeMassSHA256 =
            mass.executedRawF32NodeMassSHA256;
        candidate.sourceGlobalNodeIndexSHA256 =
            mass.sourceGlobalNodeIndexSHA256;
        candidate.executableFEMTopologySHA256 =
            mass.executableFEMTopologySHA256;
        candidate.sourceAnchorOwnershipSHA256 =
            mass.sourceAnchorOwnershipSHA256;
        candidate.executedAnchorOwnershipSHA256 =
            mass.executedAnchorOwnershipSHA256;
        candidate.materialExecutionSHA256 =
            mass.materialExecutionSHA256;
        candidate.previousTransactionSHA256 = previousRoot;
        candidate.xSourceSHA256 =
            evidence.executedSourceCoordinateSHA256;
        candidate.xReferenceSHA256 =
            evidence.executedReferenceCoordinateSHA256;
        candidate.xInitialSHA256 =
            evidence.executedInitialCoordinateSHA256;
        candidate.coordinateDerivationSHA256 =
            evidence.coordinateDerivationSHA256;
        candidate.xCurrentSHA256 = xCurrent;
        candidate.fullMatterSnapshotSHA256 = fullSnapshot;
        candidate.articulatedQVRootTimeSHA256 =
            evidence.articulatedQVRootTimeSHA256;
        candidate.muscleTendonStateSHA256 =
            evidence.muscleTendonStateSHA256;
        candidate.adapterAcceptedStateSHA256 =
            evidence.adapterAcceptedStateSHA256;
        candidate.fullAcceptedStateSHA256 = fullAcceptedState;
        candidate.transactionSHA256 = transaction;
        candidate.sourceRigidModelFingerprint =
            mass.sourceRigidModelFingerprint;
        candidate.executedRigidModelFingerprint =
            mass.executedRigidModelFingerprint;
        candidate.acceptedStepIndex = evidence.acceptedStepIndex;
        candidate.acceptedTimestampNanoseconds =
            evidence.acceptedTimestampNanoseconds;
        candidate.timestepNanoseconds = evidence.timestepNanoseconds;
        output = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    } catch (...) {
        error = "loaded-knee runtime acceptance failed";
        return false;
    }
}

bool verifyNumiHumanLoadedKneeRestoreReplayV1(
    const std::span<const NumiHumanLoadedKneeRuntimeEvidenceV1> accepted,
    const std::span<const NumiHumanLoadedKneeRuntimeEvidenceV1> replayed,
    std::string& error
) {
    try {
        if (accepted.size() != kNumiHumanLoadedKneeAcceptedStepCount ||
            replayed.size() != accepted.size()) {
            return fail(error,
                "loaded-knee restore/replay requires exactly eight accepted states");
        }
        for (std::size_t index = 0u; index < accepted.size(); ++index) {
            const auto& observed = accepted[index];
            const auto& replay = replayed[index];
            bool sameExecutedPairs = true;
            for (std::size_t pair = 0u;
                 pair < observed.executedContactPairs.size(); ++pair) {
                sameExecutedPairs = sameExecutedPairs && sameContactPair(
                    observed.executedContactPairs[pair],
                    replay.executedContactPairs[pair]);
            }
            bool sameExecutedReplacements = true;
            for (std::size_t replacement = 0u;
                 replacement < observed.executedActiveReplacements.size();
                 ++replacement) {
                sameExecutedReplacements = sameExecutedReplacements &&
                    sameReplacement(
                        observed.executedActiveReplacements[replacement],
                        replay.executedActiveReplacements[replacement]);
            }
            bool sameExecutedPassiveOwners = true;
            for (std::size_t owner = 0u;
                 owner < observed.executedPassiveOwners.size(); ++owner) {
                sameExecutedPassiveOwners = sameExecutedPassiveOwners &&
                    samePassiveOwner(
                        observed.executedPassiveOwners[owner],
                        replay.executedPassiveOwners[owner]);
            }
            if (observed.formatVersion !=
                    kNumiHumanLoadedKneeAcceptanceVersionV1 ||
                observed.formatVersion != replay.formatVersion ||
                observed.acceptedStepIndex != index + 1u ||
                replay.acceptedStepIndex != index + 1u ||
                observed.acceptedTimestampNanoseconds !=
                    (index + 1u) * kNumiHumanLoadedKneeStepNanoseconds ||
                replay.acceptedTimestampNanoseconds !=
                    observed.acceptedTimestampNanoseconds ||
                observed.timestepNanoseconds !=
                    kNumiHumanLoadedKneeStepNanoseconds ||
                observed.timestepNanoseconds != replay.timestepNanoseconds ||
                observed.snapshot.controlStep != index ||
                replay.snapshot.controlStep != index ||
                observed.snapshot.physicsSubstep != 0u ||
                replay.snapshot.physicsSubstep != 0u ||
                observed.loadedFEMNodeFirst != 0u ||
                replay.loadedFEMNodeFirst != 0u ||
                observed.loadedFEMNodeCount !=
                    kNumiHumanLoadedKneeLoadedNodeCount ||
                replay.loadedFEMNodeCount !=
                    kNumiHumanLoadedKneeLoadedNodeCount ||
                observed.snapshot.femNodes.size() !=
                    kNumiHumanLoadedKneeLoadedNodeCount ||
                replay.snapshot.femNodes.size() !=
                    kNumiHumanLoadedKneeLoadedNodeCount ||
                observed.loadedFEMNodeFirst != replay.loadedFEMNodeFirst ||
                observed.loadedFEMNodeCount != replay.loadedFEMNodeCount ||
                observed.executedSourceCoordinateSHA256 !=
                    replay.executedSourceCoordinateSHA256 ||
                observed.executedReferenceCoordinateSHA256 !=
                    replay.executedReferenceCoordinateSHA256 ||
                observed.executedInitialCoordinateSHA256 !=
                    replay.executedInitialCoordinateSHA256 ||
                observed.referencePoseSHA256 != replay.referencePoseSHA256 ||
                observed.initialPoseSHA256 != replay.initialPoseSHA256 ||
                observed.coordinateDerivationSHA256 !=
                    replay.coordinateDerivationSHA256 ||
                observed.executedMassClosureSHA256 !=
                    replay.executedMassClosureSHA256 ||
                observed.executedRawF32NodeMassSHA256 !=
                    replay.executedRawF32NodeMassSHA256 ||
                observed.executedSourceGlobalNodeIndexSHA256 !=
                    replay.executedSourceGlobalNodeIndexSHA256 ||
                observed.executedFEMTopologySHA256 !=
                    replay.executedFEMTopologySHA256 ||
                observed.executedSourceAnchorOwnershipSHA256 !=
                    replay.executedSourceAnchorOwnershipSHA256 ||
                observed.executedAnchorOwnershipSHA256 !=
                    replay.executedAnchorOwnershipSHA256 ||
                !digestPresent(
                    observed.executedMaterialExecutionSHA256) ||
                observed.executedMaterialExecutionSHA256 !=
                    replay.executedMaterialExecutionSHA256 ||
                observed.executedSourceRigidModelFingerprint !=
                    replay.executedSourceRigidModelFingerprint ||
                observed.executedSourcePhysicsFingerprint !=
                    replay.executedSourcePhysicsFingerprint ||
                observed.executedRigidModelFingerprint !=
                    replay.executedRigidModelFingerprint ||
                observed.contactPairNormalForceNewtons !=
                    replay.contactPairNormalForceNewtons ||
                observed.contactAggregateNormalForceNewtons !=
                    replay.contactAggregateNormalForceNewtons ||
                observed.prescribedClosurePairForceNewtons !=
                    replay.prescribedClosurePairForceNewtons ||
                !sameExecutedPairs || !sameExecutedReplacements ||
                !sameExecutedPassiveOwners ||
                observed.articulatedQVRootTimeSHA256 !=
                    replay.articulatedQVRootTimeSHA256 ||
                observed.muscleTendonStateSHA256 !=
                    replay.muscleTendonStateSHA256 ||
                observed.adapterAcceptedStateSHA256 !=
                    replay.adapterAcceptedStateSHA256 ||
                !sameMatterSnapshotAuthority(
                    observed.snapshot, replay.snapshot)) {
                return fail(error,
                    "loaded-knee restore/replay changed a declared full-state owner at step " +
                    std::to_string(index + 1u));
            }
            NumiHumanLoadedKneeDigest acceptedDigest{};
            NumiHumanLoadedKneeDigest replayDigest{};
            if (!digestNumiHumanLoadedKneeMatterSnapshotV1(
                    observed.snapshot, acceptedDigest, error) ||
                !digestNumiHumanLoadedKneeMatterSnapshotV1(
                    replay.snapshot, replayDigest, error) ||
                acceptedDigest != replayDigest) {
                return fail(error,
                    "loaded-knee restore/replay snapshot SHA-256 mismatch");
            }
            if (index != 0u &&
                accepted[index - 1u].executedMaterialExecutionSHA256 !=
                    observed.executedMaterialExecutionSHA256) {
                return fail(error,
                    "loaded-knee accepted trajectory changed immutable material execution at step " +
                    std::to_string(index + 1u));
            }
            if (index != 0u &&
                sameMatterSnapshotAuthority(
                    accepted[index - 1u].snapshot,
                    observed.snapshot) &&
                accepted[index - 1u].articulatedQVRootTimeSHA256 ==
                    observed.articulatedQVRootTimeSHA256 &&
                accepted[index - 1u].muscleTendonStateSHA256 ==
                    observed.muscleTendonStateSHA256 &&
                accepted[index - 1u].adapterAcceptedStateSHA256 ==
                    observed.adapterAcceptedStateSHA256) {
                return fail(error,
                    "loaded-knee accepted trajectory contains a repeated full state");
            }
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

} // namespace metalrobo
