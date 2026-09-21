#include "metalrobo/NumiHumanSourceConstraints.hpp"

#include "metalrobo/NumiHumanRuntimeIdentity.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace metalrobo {
namespace {

static_assert(std::endian::native == std::endian::little,
              "Numi Human source programs require little-endian decoding");

#pragma pack(push, 1)
struct SourceConstraintHeaderV1 {
    std::array<char, 8u> magic{};
    std::uint32_t abi = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t count = 0u;
    std::uint32_t recordBytes = 0u;
    std::uint32_t sourceCount = 0u;
    std::uint32_t policy = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    NumiHumanLoadedKneeDigest sourceSHA256{};
};
#pragma pack(pop)

static_assert(sizeof(SourceConstraintHeaderV1) == 80u);

constexpr std::array<char, 8u> kEqualityMagic{
    'N', 'H', 'E', 'Q', '2', '\0', '\0', '\0'};
constexpr std::array<char, 8u> kLimitMagic{
    'N', 'H', 'L', 'I', 'M', '1', '\0', '\0'};

class ScopedDescriptor final {
public:
    explicit ScopedDescriptor(const int descriptor) noexcept
        : descriptor_(descriptor) {}
    ~ScopedDescriptor() {
        if (descriptor_ >= 0) ::close(descriptor_);
    }
    ScopedDescriptor(const ScopedDescriptor&) = delete;
    ScopedDescriptor& operator=(const ScopedDescriptor&) = delete;
    [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
    int descriptor_ = -1;
};

[[nodiscard]] bool fail(std::string& error, const std::string& message) {
    error = message;
    return false;
}

[[nodiscard]] bool digestPresent(
    const NumiHumanLoadedKneeDigest& digest) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](const auto value) {
        return value != 0u;
    });
}

[[nodiscard]] bool finiteNormalOrZero(const nm_float4& value) noexcept {
    const auto valid = [](const float scalar) {
        return std::isfinite(scalar) &&
            std::fpclassify(scalar) != FP_SUBNORMAL;
    };
    return valid(value.x) && valid(value.y) && valid(value.z) &&
        valid(value.w);
}

[[nodiscard]] NumiHumanLoadedKneeDigest sha256(
    const std::span<const std::byte> bytes) noexcept {
    NumiHumanLoadedKneeDigest digest{};
    if (bytes.size() > std::numeric_limits<CC_LONG>::max()) return {};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    return digest;
}

[[nodiscard]] bool checkedPayloadBytes(
    const std::uint32_t count,
    const std::uint32_t recordBytes,
    std::size_t& output) noexcept {
    const std::size_t countValue = count;
    const std::size_t recordValue = recordBytes;
    if (countValue >
        (std::numeric_limits<std::size_t>::max() -
         sizeof(SourceConstraintHeaderV1)) /
            recordValue) {
        return false;
    }
    output = sizeof(SourceConstraintHeaderV1) + countValue * recordValue;
    return true;
}

[[nodiscard]] bool readImmutableFile(
    const std::filesystem::path& path,
    const std::size_t expectedBytes,
    const NumiHumanLoadedKneeDigest& expectedSHA256,
    std::vector<std::byte>& output,
    std::string& error,
    const std::string_view label) {
    if (path.empty() || expectedBytes == 0u ||
        !digestPresent(expectedSHA256)) {
        return fail(error, std::string(label) +
            " immutable-file expectation is invalid");
    }
    const ScopedDescriptor descriptor(
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.get() < 0) {
        return fail(error, std::string(label) +
            " could not be opened without following a link");
    }
    struct stat metadata {};
    if (::fstat(descriptor.get(), &metadata) != 0 ||
        !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        static_cast<std::uint64_t>(metadata.st_size) != expectedBytes) {
        return fail(error, std::string(label) +
            " is not a regular file with the exact expected byte count");
    }
    std::vector<std::byte> candidate(expectedBytes);
    std::size_t offset = 0u;
    while (offset < candidate.size()) {
        const ssize_t count = ::read(
            descriptor.get(), candidate.data() + offset,
            candidate.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            return fail(error, std::string(label) +
                " immutable read was incomplete");
        }
        offset += static_cast<std::size_t>(count);
    }
    if (sha256(candidate) != expectedSHA256) {
        return fail(error, std::string(label) +
            " SHA-256 does not match the authenticated expectation");
    }
    output = std::move(candidate);
    return true;
}

[[nodiscard]] bool finiteEqualityRows(
    const std::span<const NMHumanJointEqualityGPU> rows,
    const std::uint32_t nv,
    std::string& error) {
    std::vector<bool> dependent(nv, false);
    for (const auto& row : rows) {
        const bool fixed = row.indices.z == NM_INVALID_INDEX;
        if (!finiteNormalOrZero(row.referencesAndCoefficients0) ||
            !finiteNormalOrZero(row.coefficients1) ||
            !finiteNormalOrZero(row.solref) ||
            !finiteNormalOrZero(row.solimp0) ||
            !finiteNormalOrZero(row.solimp1) ||
            !finiteNormalOrZero(row.sourceInverseWeights) ||
            row.indices.y < 6u || row.indices.y >= nv ||
            row.indices.x != row.indices.y + 1u ||
            (fixed
                 ? row.indices.w != NM_INVALID_INDEX
                 : (row.indices.w < 6u || row.indices.w >= nv ||
                    row.indices.z != row.indices.w + 1u ||
                    row.indices.y == row.indices.w)) ||
            dependent[row.indices.y] || row.coefficients1.w != 0.0f ||
            row.solref.z != 0.0f || row.solref.w != 0.0f ||
            !((row.solref.x > 0.0f && row.solref.y > 0.0f) ||
              (row.solref.x <= 0.0f && row.solref.y <= 0.0f)) ||
            row.solimp1.y != 0.0f || row.solimp1.z != 0.0f ||
            row.solimp1.w != 0.0f ||
            !(row.sourceInverseWeights.x > 0.0f) ||
            (fixed ? row.sourceInverseWeights.y != 0.0f
                   : !(row.sourceInverseWeights.y > 0.0f)) ||
            row.sourceInverseWeights.z != 0.0f ||
            row.sourceInverseWeights.w != 0.0f) {
            return fail(error,
                "NHEQ2 contains an invalid scalar row or source inverse weight");
        }
        dependent[row.indices.y] = true;
    }
    for (const auto& row : rows) {
        if (row.indices.w != NM_INVALID_INDEX && dependent[row.indices.w]) {
            return fail(error,
                "NHEQ2 dependent chains require a separately qualified source profile");
        }
    }
    return true;
}

[[nodiscard]] bool finiteLimitRows(
    const std::span<const NMHumanJointLimitGPU> rows,
    const std::uint32_t nv,
    std::string& error) {
    std::vector<bool> limited(nv, false);
    for (std::size_t index = 0u; index < rows.size(); ++index) {
        const auto& row = rows[index];
        bool duplicateSource = false;
        for (std::size_t prior = 0u; prior < index; ++prior)
            duplicateSource = duplicateSource ||
                rows[prior].indices.z == row.indices.z;
        if (!finiteNormalOrZero(row.rangeMarginInverseWeight) ||
            !finiteNormalOrZero(row.solref) ||
            !finiteNormalOrZero(row.solimp0) ||
            !finiteNormalOrZero(row.solimp1) ||
            row.indices.y < 6u || row.indices.y >= nv ||
            row.indices.x != row.indices.y + 1u ||
            row.indices.z == NM_INVALID_INDEX || row.indices.w != 0u ||
            duplicateSource || limited[row.indices.y] ||
            row.rangeMarginInverseWeight.x >
                row.rangeMarginInverseWeight.y ||
            !(row.rangeMarginInverseWeight.w >=
              std::numeric_limits<float>::min()) ||
            row.solref.z != 0.0f || row.solref.w != 0.0f ||
            !((row.solref.x > 0.0f && row.solref.y > 0.0f) ||
              (row.solref.x <= 0.0f && row.solref.y <= 0.0f)) ||
            row.solimp1.y != 0.0f || row.solimp1.z != 0.0f ||
            row.solimp1.w != 0.0f) {
            return fail(error,
                "NHLIM1 contains an invalid scalar row or source identity");
        }
        limited[row.indices.y] = true;
    }
    return true;
}

[[nodiscard]] bool validExpectation(
    const NumiHumanSourceConstraintExpectationV1& expectation,
    std::string& error) {
    if (!digestPresent(expectation.equalityFileSHA256) ||
        !digestPresent(expectation.limitFileSHA256) ||
        !digestPresent(expectation.sourceArchiveSHA256) ||
        expectation.nv < 7u || expectation.nv > 160u ||
        expectation.nq != expectation.nv + 1u ||
        expectation.equalityRowCount == 0u ||
        expectation.equalityRowCount > expectation.nv ||
        expectation.limitRowCount == 0u ||
        expectation.limitRowCount > expectation.nv - 6u ||
        expectation.policy !=
            NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC ||
        (expectation.flags & ~NM_HUMAN_EQUALITY_REFSAFE) != 0u) {
        return fail(error,
            "source-constraint expectation is incomplete or unsupported");
    }
    return true;
}

[[nodiscard]] bool decodePayloads(
    const std::vector<std::byte>& equalityBytes,
    const std::vector<std::byte>& limitBytes,
    const NumiHumanSourceConstraintExpectationV1& expectation,
    NumiHumanSourceConstraintProgramV1& output,
    std::string& error) {
    if (equalityBytes.size() < sizeof(SourceConstraintHeaderV1) ||
        limitBytes.size() < sizeof(SourceConstraintHeaderV1)) {
        return fail(error, "source-constraint program is truncated");
    }
    SourceConstraintHeaderV1 equalityHeader{};
    SourceConstraintHeaderV1 limitHeader{};
    std::memcpy(&equalityHeader, equalityBytes.data(), sizeof(equalityHeader));
    std::memcpy(&limitHeader, limitBytes.data(), sizeof(limitHeader));
    const auto headerMatches = [&](const SourceConstraintHeaderV1& header,
                                   const std::array<char, 8u>& magic,
                                   const std::uint32_t abi,
                                   const std::uint32_t count,
                                   const std::uint32_t recordBytes) {
        return header.magic == magic && header.abi == abi &&
            header.nq == expectation.nq && header.nv == expectation.nv &&
            header.count == count && header.sourceCount == count &&
            header.recordBytes == recordBytes &&
            header.policy == expectation.policy &&
            header.flags == expectation.flags && header.reserved0 == 0u &&
            header.reserved1 == 0u &&
            header.sourceSHA256 == expectation.sourceArchiveSHA256;
    };
    if (!headerMatches(equalityHeader, kEqualityMagic, 2u,
                       expectation.equalityRowCount,
                       sizeof(NMHumanJointEqualityGPU))) {
        return fail(error,
            "NHEQ2 header, source identity, dimensions, or policy differs");
    }
    if (!headerMatches(limitHeader, kLimitMagic,
                       NM_HUMAN_LIMIT_ABI_VERSION,
                       expectation.limitRowCount,
                       sizeof(NMHumanJointLimitGPU))) {
        return fail(error,
            "NHLIM1 header, source identity, dimensions, or policy differs");
    }
    std::size_t expectedEqualityBytes = 0u;
    std::size_t expectedLimitBytes = 0u;
    if (!checkedPayloadBytes(equalityHeader.count,
                             equalityHeader.recordBytes,
                             expectedEqualityBytes) ||
        !checkedPayloadBytes(limitHeader.count, limitHeader.recordBytes,
                             expectedLimitBytes) ||
        equalityBytes.size() != expectedEqualityBytes ||
        limitBytes.size() != expectedLimitBytes) {
        return fail(error,
            "source-constraint payload byte count differs from its exact envelope");
    }
    NumiHumanSourceConstraintProgramV1 candidate;
    candidate.equalityPayloadBytes = equalityBytes;
    candidate.limitPayloadBytes = limitBytes;
    candidate.jointEqualities.resize(equalityHeader.count);
    candidate.jointLimits.resize(limitHeader.count);
    std::memcpy(candidate.jointEqualities.data(),
                equalityBytes.data() + sizeof(equalityHeader),
                candidate.jointEqualities.size() *
                    sizeof(NMHumanJointEqualityGPU));
    std::memcpy(candidate.jointLimits.data(),
                limitBytes.data() + sizeof(limitHeader),
                candidate.jointLimits.size() * sizeof(NMHumanJointLimitGPU));
    if (!finiteEqualityRows(candidate.jointEqualities, expectation.nv,
                            error) ||
        !finiteLimitRows(candidate.jointLimits, expectation.nv, error)) {
        return false;
    }
    candidate.equalityDispatch.count = equalityHeader.count;
    candidate.equalityDispatch.qCount = equalityHeader.nq;
    candidate.equalityDispatch.dofCount = equalityHeader.nv;
    candidate.equalityDispatch.flags = equalityHeader.flags;
    candidate.equalityDispatch.policy = equalityHeader.policy;
    candidate.limitDispatch.count = limitHeader.count;
    candidate.limitDispatch.qCount = limitHeader.nq;
    candidate.limitDispatch.dofCount = limitHeader.nv;
    candidate.limitDispatch.flags = limitHeader.flags;
    candidate.limitDispatch.policy = limitHeader.policy;
    candidate.equalityFingerprint = numiHumanRuntimePayloadFingerprint(
        candidate.equalityPayloadBytes);
    candidate.limitFingerprint = numiHumanRuntimePayloadFingerprint(
        candidate.limitPayloadBytes);
    candidate.equalityFileSHA256 = sha256(candidate.equalityPayloadBytes);
    candidate.limitFileSHA256 = sha256(candidate.limitPayloadBytes);
    candidate.sourceArchiveSHA256 = equalityHeader.sourceSHA256;
    output = std::move(candidate);
    error.clear();
    return true;
}

[[nodiscard]] bool dispatchEmpty(
    const NMHumanEqualityDispatchGPU& dispatch) noexcept {
    return dispatch.count == 0u && dispatch.qCount == 0u &&
        dispatch.dofCount == 0u && dispatch.flags == 0u &&
        dispatch.policy == 0u && dispatch.reserved0 == 0u &&
        dispatch.reserved1 == 0u && dispatch.reserved2 == 0u &&
        dispatch.time.x == 0.0f && dispatch.time.y == 0.0f &&
        dispatch.time.z == 0.0f && dispatch.time.w == 0.0f;
}

[[nodiscard]] bool validateDecodedProgram(
    const NumiHumanSourceConstraintProgramV1& program,
    std::string& error) {
    if (program.version != kNumiHumanSourceConstraintProgramVersionV1 ||
        !digestPresent(program.equalityFileSHA256) ||
        !digestPresent(program.limitFileSHA256) ||
        !digestPresent(program.sourceArchiveSHA256) ||
        program.equalityFingerprint == 0u || program.limitFingerprint == 0u ||
        program.equalityDispatch.count != program.jointEqualities.size() ||
        program.limitDispatch.count != program.jointLimits.size() ||
        program.equalityDispatch.dofCount < 7u ||
        program.equalityDispatch.dofCount > 160u ||
        program.equalityDispatch.qCount !=
            program.equalityDispatch.dofCount + 1u ||
        program.equalityDispatch.count == 0u ||
        program.equalityDispatch.count >
            program.equalityDispatch.dofCount ||
        program.limitDispatch.count == 0u ||
        program.limitDispatch.count >
            program.limitDispatch.dofCount - 6u ||
        program.equalityDispatch.qCount != program.limitDispatch.qCount ||
        program.equalityDispatch.dofCount != program.limitDispatch.dofCount ||
        program.equalityDispatch.policy != program.limitDispatch.policy ||
        program.equalityDispatch.flags != program.limitDispatch.flags ||
        program.equalityDispatch.policy !=
            NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC ||
        (program.equalityDispatch.flags &
         ~NM_HUMAN_EQUALITY_REFSAFE) != 0u ||
        program.equalityDispatch.reserved0 != 0u ||
        program.equalityDispatch.reserved1 != 0u ||
        program.equalityDispatch.reserved2 != 0u ||
        program.limitDispatch.reserved0 != 0u ||
        program.limitDispatch.reserved1 != 0u ||
        program.limitDispatch.reserved2 != 0u ||
        program.equalityDispatch.time.x != 0.0f ||
        program.equalityDispatch.time.y != 0.0f ||
        program.equalityDispatch.time.z != 0.0f ||
        program.equalityDispatch.time.w != 0.0f ||
        program.limitDispatch.time.x != 0.0f ||
        program.limitDispatch.time.y != 0.0f ||
        program.limitDispatch.time.z != 0.0f ||
        program.limitDispatch.time.w != 0.0f) {
        return fail(error,
            "decoded source-constraint program metadata is inconsistent");
    }
    std::size_t equalityBytes = 0u;
    std::size_t limitBytes = 0u;
    if (!checkedPayloadBytes(program.equalityDispatch.count,
                             sizeof(NMHumanJointEqualityGPU),
                             equalityBytes) ||
        !checkedPayloadBytes(program.limitDispatch.count,
                             sizeof(NMHumanJointLimitGPU), limitBytes) ||
        program.equalityPayloadBytes.size() != equalityBytes ||
        program.limitPayloadBytes.size() != limitBytes ||
        sha256(program.equalityPayloadBytes) !=
            program.equalityFileSHA256 ||
        sha256(program.limitPayloadBytes) != program.limitFileSHA256 ||
        numiHumanRuntimePayloadFingerprint(program.equalityPayloadBytes) !=
            program.equalityFingerprint ||
        numiHumanRuntimePayloadFingerprint(program.limitPayloadBytes) !=
            program.limitFingerprint ||
        std::memcmp(
            program.equalityPayloadBytes.data() +
                sizeof(SourceConstraintHeaderV1),
            program.jointEqualities.data(),
            program.jointEqualities.size() *
                sizeof(NMHumanJointEqualityGPU)) != 0 ||
        std::memcmp(
            program.limitPayloadBytes.data() +
                sizeof(SourceConstraintHeaderV1),
            program.jointLimits.data(),
            program.jointLimits.size() * sizeof(NMHumanJointLimitGPU)) != 0) {
        return fail(error,
            "decoded source-constraint bytes or row identity changed after admission");
    }
    SourceConstraintHeaderV1 equalityHeader{};
    SourceConstraintHeaderV1 limitHeader{};
    std::memcpy(&equalityHeader, program.equalityPayloadBytes.data(),
                sizeof(equalityHeader));
    std::memcpy(&limitHeader, program.limitPayloadBytes.data(),
                sizeof(limitHeader));
    const auto headerMatchesDispatch = [](
        const SourceConstraintHeaderV1& header,
        const NMHumanEqualityDispatchGPU& dispatch,
        const std::array<char, 8u>& magic,
        const std::uint32_t abi,
        const std::uint32_t recordBytes) {
        return header.magic == magic && header.abi == abi &&
            header.nq == dispatch.qCount && header.nv == dispatch.dofCount &&
            header.count == dispatch.count &&
            header.sourceCount == dispatch.count &&
            header.recordBytes == recordBytes &&
            header.policy == dispatch.policy && header.flags == dispatch.flags &&
            header.reserved0 == 0u && header.reserved1 == 0u;
    };
    if (!headerMatchesDispatch(
            equalityHeader, program.equalityDispatch, kEqualityMagic, 2u,
            sizeof(NMHumanJointEqualityGPU)) ||
        !headerMatchesDispatch(
            limitHeader, program.limitDispatch, kLimitMagic,
            NM_HUMAN_LIMIT_ABI_VERSION,
            sizeof(NMHumanJointLimitGPU)) ||
        equalityHeader.sourceSHA256 != program.sourceArchiveSHA256 ||
        limitHeader.sourceSHA256 != program.sourceArchiveSHA256) {
        return fail(error,
            "decoded source-constraint header identity changed after admission");
    }
    if (!finiteEqualityRows(program.jointEqualities,
                            program.equalityDispatch.dofCount, error) ||
        !finiteLimitRows(program.jointLimits,
                         program.limitDispatch.dofCount, error)) {
        return false;
    }
    return true;
}

} // namespace

bool loadNumiHumanSourceConstraintProgramV1(
    const std::filesystem::path& jointEqualityPath,
    const std::filesystem::path& jointLimitPath,
    const NumiHumanSourceConstraintExpectationV1& expectation,
    NumiHumanSourceConstraintProgramV1& output,
    std::string& error) {
    output = {};
    if (!validExpectation(expectation, error)) return false;
    std::size_t equalityBytes = 0u;
    std::size_t limitBytes = 0u;
    if (!checkedPayloadBytes(expectation.equalityRowCount,
                             sizeof(NMHumanJointEqualityGPU),
                             equalityBytes) ||
        !checkedPayloadBytes(expectation.limitRowCount,
                             sizeof(NMHumanJointLimitGPU), limitBytes)) {
        return fail(error, "source-constraint expected byte count overflows");
    }
    std::vector<std::byte> equalityPayload;
    std::vector<std::byte> limitPayload;
    if (!readImmutableFile(jointEqualityPath, equalityBytes,
                           expectation.equalityFileSHA256, equalityPayload,
                           error, "NHEQ2") ||
        !readImmutableFile(jointLimitPath, limitBytes,
                           expectation.limitFileSHA256, limitPayload, error,
                           "NHLIM1")) {
        return false;
    }
    return decodePayloads(equalityPayload, limitPayload, expectation, output,
                          error);
}

bool loadNumiHumanLoadedKneeSourceConstraintProgramV1(
    const std::filesystem::path& jointEqualityPath,
    const std::filesystem::path& jointLimitPath,
    const NumiHumanLoadedKneeBindingAdmissionV1& authenticatedBase,
    const NumiHumanLoadedKneeSourceComplianceAdmissionV1&
        authenticatedSourceCompliance,
    NumiHumanSourceConstraintProgramV1& output,
    std::string& error) {
    output = {};
    std::string baseError;
    if (!validateNumiHumanLoadedKneeAuthoringV1(
            authenticatedBase.authoring, nullptr, baseError)) {
        return fail(error,
            "loaded-knee base admission is invalid: " + baseError);
    }
    if (!digestPresent(authenticatedBase.authoring.manifestSHA256) ||
        !digestPresent(authenticatedBase.manifestFileSHA256) ||
        !digestPresent(authenticatedBase.bindingSHA256) ||
        !digestPresent(authenticatedBase.bindingFileSHA256) ||
        !digestPresent(authenticatedBase.ownershipFileSHA256) ||
        !digestPresent(authenticatedSourceCompliance.bindingSHA256) ||
        !digestPresent(authenticatedSourceCompliance.fileSHA256) ||
        !digestPresent(
            authenticatedSourceCompliance.baseManifestSHA256) ||
        !digestPresent(
            authenticatedSourceCompliance.baseManifestFileSHA256) ||
        !digestPresent(
            authenticatedSourceCompliance.jointEqualityFileSHA256) ||
        !digestPresent(
            authenticatedSourceCompliance.jointLimitFileSHA256) ||
        !digestPresent(
            authenticatedSourceCompliance.sourceArchiveSHA256)) {
        return fail(error,
            "loaded-knee base or source-compliance admission lacks an authenticated identity");
    }
    if (authenticatedSourceCompliance.baseManifestSHA256 !=
            authenticatedBase.authoring.manifestSHA256 ||
        authenticatedSourceCompliance.baseManifestFileSHA256 !=
            authenticatedBase.manifestFileSHA256) {
        return fail(error,
            "loaded-knee source-compliance admission is linked to a different base manifest");
    }
    const NumiHumanSourceConstraintExpectationV1 expectation{
        .equalityFileSHA256 =
            authenticatedSourceCompliance.jointEqualityFileSHA256,
        .limitFileSHA256 =
            authenticatedSourceCompliance.jointLimitFileSHA256,
        .sourceArchiveSHA256 =
            authenticatedSourceCompliance.sourceArchiveSHA256,
        .nq = authenticatedSourceCompliance.nq,
        .nv = authenticatedSourceCompliance.nv,
        .equalityRowCount =
            authenticatedSourceCompliance.jointEqualityRowCount,
        .limitRowCount = authenticatedSourceCompliance.jointLimitRowCount,
        .policy = authenticatedSourceCompliance.policy,
        .flags = authenticatedSourceCompliance.flags,
    };
    if (!validExpectation(expectation, error)) {
        error = "loaded-knee authenticated source-compliance metadata is invalid: " +
            error;
        return false;
    }
    return loadNumiHumanSourceConstraintProgramV1(
        jointEqualityPath, jointLimitPath, expectation, output, error);
}

bool configureNumiHumanSourceConstraintsV1(
    const NumiHumanSourceConstraintProgramV1& program,
    numi::matter::RuntimeConfiguration& configuration,
    std::string& error) {
    if (!validateDecodedProgram(program, error)) return false;
    if (!configuration.humanJointEqualities.empty() ||
        !configuration.humanJointLimits.empty() ||
        configuration.humanEqualitySourceFingerprint != 0u ||
        configuration.humanLimitSourceFingerprint != 0u ||
        !dispatchEmpty(configuration.humanEqualityDispatch) ||
        !dispatchEmpty(configuration.humanLimitDispatch)) {
        return fail(error,
            "Matter RuntimeConfiguration already owns a Human source-constraint program");
    }
    configuration.humanJointEqualities = program.jointEqualities;
    configuration.humanEqualityDispatch = program.equalityDispatch;
    configuration.humanEqualitySourceFingerprint =
        program.equalityFingerprint;
    configuration.humanJointLimits = program.jointLimits;
    configuration.humanLimitDispatch = program.limitDispatch;
    configuration.humanLimitSourceFingerprint = program.limitFingerprint;
    error.clear();
    return true;
}

} // namespace metalrobo
