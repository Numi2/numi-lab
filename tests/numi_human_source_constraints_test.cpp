#include "metalrobo/NumiHumanSourceConstraints.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Header {
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
    metalrobo::NumiHumanLoadedKneeDigest sourceSHA256{};
};
#pragma pack(pop)

static_assert(sizeof(Header) == 80u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

metalrobo::NumiHumanLoadedKneeDigest sha256(
    const std::vector<std::byte>& bytes) {
    metalrobo::NumiHumanLoadedKneeDigest result{};
    require(bytes.size() <= std::numeric_limits<CC_LONG>::max(),
            "test payload exceeds CommonCrypto bounds");
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), result.data());
    return result;
}

template <typename Row>
std::vector<std::byte> payload(const Header& header, const Row& row) {
    std::vector<std::byte> result(sizeof(header) + sizeof(row));
    std::memcpy(result.data(), &header, sizeof(header));
    std::memcpy(result.data() + sizeof(header), &row, sizeof(row));
    return result;
}

void writeBytes(const std::filesystem::path& path,
                const std::vector<std::byte>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.is_open(), "could not create test payload");
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    require(output.good(), "could not write test payload");
}

struct TemporaryDirectory {
    TemporaryDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() /
             "numi-source-constraints-XXXXXX")
                .string();
        std::vector<char> storage(pattern.begin(), pattern.end());
        storage.push_back('\0');
        const char* created = ::mkdtemp(storage.data());
        require(created != nullptr, "could not create test directory");
        path = created;
    }
    ~TemporaryDirectory() { std::filesystem::remove_all(path); }
    std::filesystem::path path;
};

struct Fixture {
    std::vector<std::byte> equality;
    std::vector<std::byte> limit;
    metalrobo::NumiHumanSourceConstraintExpectationV1 expectation;
};

Fixture fixture() {
    metalrobo::NumiHumanLoadedKneeDigest source{};
    for (std::size_t index = 0u; index < source.size(); ++index)
        source[index] = static_cast<std::uint8_t>(index + 1u);

    Header equalityHeader;
    equalityHeader.magic = {'N', 'H', 'E', 'Q', '2', '\0', '\0', '\0'};
    equalityHeader.abi = 2u;
    equalityHeader.nq = 9u;
    equalityHeader.nv = 8u;
    equalityHeader.count = 1u;
    equalityHeader.recordBytes = sizeof(NMHumanJointEqualityGPU);
    equalityHeader.sourceCount = 1u;
    equalityHeader.policy = NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC;
    equalityHeader.flags = NM_HUMAN_EQUALITY_REFSAFE;
    equalityHeader.sourceSHA256 = source;
    NMHumanJointEqualityGPU equality{};
    equality.indices = {7u, 6u, NM_INVALID_INDEX, NM_INVALID_INDEX};
    equality.referencesAndCoefficients0 = {0.0f, 0.0f, 1.0f, 0.0f};
    equality.coefficients1 = {0.0f, 0.0f, 0.0f, 0.0f};
    equality.solref = {-0.02f, -1.0f, 0.0f, 0.0f};
    equality.solimp0 = {0.9f, 0.95f, 0.001f, 0.5f};
    equality.solimp1 = {2.0f, 0.0f, 0.0f, 0.0f};
    equality.sourceInverseWeights = {1.0f, 0.0f, 0.0f, 0.0f};

    Header limitHeader;
    limitHeader.magic = {'N', 'H', 'L', 'I', 'M', '1', '\0', '\0'};
    limitHeader.abi = NM_HUMAN_LIMIT_ABI_VERSION;
    limitHeader.nq = equalityHeader.nq;
    limitHeader.nv = equalityHeader.nv;
    limitHeader.count = 1u;
    limitHeader.recordBytes = sizeof(NMHumanJointLimitGPU);
    limitHeader.sourceCount = 1u;
    limitHeader.policy = equalityHeader.policy;
    limitHeader.flags = equalityHeader.flags;
    limitHeader.sourceSHA256 = source;
    NMHumanJointLimitGPU limit{};
    limit.indices = {8u, 7u, 42u, 0u};
    limit.rangeMarginInverseWeight = {-1.0f, 1.0f, 0.01f, 1.0f};
    limit.solref = {-0.02f, -1.0f, 0.0f, 0.0f};
    limit.solimp0 = {0.9f, 0.95f, 0.001f, 0.5f};
    limit.solimp1 = {2.0f, 0.0f, 0.0f, 0.0f};

    Fixture result;
    result.equality = payload(equalityHeader, equality);
    result.limit = payload(limitHeader, limit);
    result.expectation.equalityFileSHA256 = sha256(result.equality);
    result.expectation.limitFileSHA256 = sha256(result.limit);
    result.expectation.sourceArchiveSHA256 = source;
    result.expectation.nq = equalityHeader.nq;
    result.expectation.nv = equalityHeader.nv;
    result.expectation.equalityRowCount = equalityHeader.count;
    result.expectation.limitRowCount = limitHeader.count;
    result.expectation.policy = equalityHeader.policy;
    result.expectation.flags = equalityHeader.flags;
    return result;
}

void expectRejected(
    const TemporaryDirectory& directory,
    const std::string& name,
    const std::vector<std::byte>& equality,
    const std::vector<std::byte>& limit,
    const metalrobo::NumiHumanSourceConstraintExpectationV1& expectation) {
    const auto equalityPath = directory.path / (name + ".nheq");
    const auto limitPath = directory.path / (name + ".nhlim");
    writeBytes(equalityPath, equality);
    writeBytes(limitPath, limit);
    metalrobo::NumiHumanSourceConstraintProgramV1 output;
    output.version = 99u;
    output.jointEqualities.resize(1u);
    std::string error;
    require(!metalrobo::loadNumiHumanSourceConstraintProgramV1(
                equalityPath, limitPath, expectation, output, error),
            name + " mutation was admitted");
    require(!error.empty(), name + " rejection omitted its typed reason");
    require(output.version ==
                metalrobo::kNumiHumanSourceConstraintProgramVersionV1 &&
                output.jointEqualities.empty() && output.jointLimits.empty(),
            name + " rejection published partial decoded output");
}

} // namespace

int main() {
    try {
        TemporaryDirectory directory;
        const Fixture valid = fixture();
        const auto equalityPath = directory.path / "valid.nheq";
        const auto limitPath = directory.path / "valid.nhlim";
        writeBytes(equalityPath, valid.equality);
        writeBytes(limitPath, valid.limit);

        metalrobo::NumiHumanSourceConstraintProgramV1 program;
        std::string error;
        require(metalrobo::loadNumiHumanSourceConstraintProgramV1(
                    equalityPath, limitPath, valid.expectation, program,
                    error),
                "valid source programs were rejected: " + error);
        require(program.jointEqualities.size() == 1u &&
                    program.jointLimits.size() == 1u &&
                    program.equalityFingerprint != 0u &&
                    program.limitFingerprint != 0u,
                "valid source programs decoded incompletely");

        numi::matter::RuntimeConfiguration configuration;
        require(metalrobo::configureNumiHumanSourceConstraintsV1(
                    program, configuration, error),
                "valid RuntimeConfiguration binding failed: " + error);
        require(configuration.humanJointEqualities.data() ==
                    program.jointEqualities.data() &&
                    configuration.humanJointLimits.data() ==
                    program.jointLimits.data() &&
                    configuration.humanEqualityDispatch.count == 1u &&
                    configuration.humanLimitDispatch.count == 1u &&
                    configuration.humanEqualitySourceFingerprint ==
                        program.equalityFingerprint &&
                    configuration.humanLimitSourceFingerprint ==
                        program.limitFingerprint,
                "RuntimeConfiguration did not borrow the exact decoded rows");
        require(!metalrobo::configureNumiHumanSourceConstraintsV1(
                    program, configuration, error),
                "existing RuntimeConfiguration source owner was overwritten");

        auto wrongDigest = valid.expectation;
        wrongDigest.equalityFileSHA256[0u] ^= 1u;
        expectRejected(directory, "wrong-digest", valid.equality, valid.limit,
                       wrongDigest);

        auto badMagic = valid.equality;
        badMagic[0u] = std::byte{'X'};
        auto badMagicExpectation = valid.expectation;
        badMagicExpectation.equalityFileSHA256 = sha256(badMagic);
        expectRejected(directory, "bad-magic", badMagic, valid.limit,
                       badMagicExpectation);

        auto badReserved = valid.limit;
        Header badReservedHeader{};
        std::memcpy(&badReservedHeader, badReserved.data(),
                    sizeof(badReservedHeader));
        badReservedHeader.reserved0 = 1u;
        std::memcpy(badReserved.data(), &badReservedHeader,
                    sizeof(badReservedHeader));
        auto badReservedExpectation = valid.expectation;
        badReservedExpectation.limitFileSHA256 = sha256(badReserved);
        expectRejected(directory, "bad-reserved", valid.equality, badReserved,
                       badReservedExpectation);

        auto badInverseWeight = valid.equality;
        NMHumanJointEqualityGPU badEquality{};
        std::memcpy(&badEquality, badInverseWeight.data() + sizeof(Header),
                    sizeof(badEquality));
        badEquality.sourceInverseWeights.x = 0.0f;
        std::memcpy(badInverseWeight.data() + sizeof(Header), &badEquality,
                    sizeof(badEquality));
        auto badInverseExpectation = valid.expectation;
        badInverseExpectation.equalityFileSHA256 = sha256(badInverseWeight);
        expectRejected(directory, "bad-inverse-weight", badInverseWeight,
                       valid.limit, badInverseExpectation);

        auto subnormalEquality = valid.equality;
        std::memcpy(&badEquality,
                    subnormalEquality.data() + sizeof(Header),
                    sizeof(badEquality));
        badEquality.solimp0.w = std::numeric_limits<float>::denorm_min();
        std::memcpy(subnormalEquality.data() + sizeof(Header), &badEquality,
                    sizeof(badEquality));
        auto subnormalEqualityExpectation = valid.expectation;
        subnormalEqualityExpectation.equalityFileSHA256 =
            sha256(subnormalEquality);
        expectRejected(directory, "subnormal-equality", subnormalEquality,
                       valid.limit, subnormalEqualityExpectation);

        auto subnormalLimit = valid.limit;
        NMHumanJointLimitGPU badLimit{};
        std::memcpy(&badLimit, subnormalLimit.data() + sizeof(Header),
                    sizeof(badLimit));
        badLimit.solimp0.w = std::numeric_limits<float>::denorm_min();
        std::memcpy(subnormalLimit.data() + sizeof(Header), &badLimit,
                    sizeof(badLimit));
        auto subnormalExpectation = valid.expectation;
        subnormalExpectation.limitFileSHA256 = sha256(subnormalLimit);
        expectRejected(directory, "subnormal-limit", valid.equality,
                       subnormalLimit, subnormalExpectation);

        auto trailingEquality = valid.equality;
        trailingEquality.push_back(std::byte{0u});
        auto trailingExpectation = valid.expectation;
        trailingExpectation.equalityFileSHA256 = sha256(trailingEquality);
        expectRejected(directory, "trailing-byte", trailingEquality,
                       valid.limit, trailingExpectation);

        const auto symlinkPath = directory.path / "redirected.nheq";
        require(::symlink(equalityPath.c_str(), symlinkPath.c_str()) == 0,
                "could not create symlink mutation");
        metalrobo::NumiHumanSourceConstraintProgramV1 symlinkOutput;
        require(!metalrobo::loadNumiHumanSourceConstraintProgramV1(
                    symlinkPath, limitPath, valid.expectation, symlinkOutput,
                    error),
                "symlinked NHEQ2 input was admitted");

        auto changedRows = program;
        changedRows.jointEqualities[0u].sourceInverseWeights.x = 2.0f;
        numi::matter::RuntimeConfiguration changedConfiguration;
        require(!metalrobo::configureNumiHumanSourceConstraintsV1(
                    changedRows, changedConfiguration, error),
                "post-admission NHEQ2 row mutation was configured");

        auto unsupportedExpectation = valid.expectation;
        unsupportedExpectation.policy = 0u;
        expectRejected(directory, "unsupported-policy", valid.equality,
                       valid.limit, unsupportedExpectation);

        std::cout << "numi_human_source_constraints_test=passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi_human_source_constraints_test=failed reason=\""
                  << error.what() << "\"\n";
        return 1;
    }
}
