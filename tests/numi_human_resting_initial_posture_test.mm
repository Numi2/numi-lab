#import <Foundation/Foundation.h>

#include "apps/NumiHumanRestingInitialPosture.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

using numi::human::resting_initial_posture::InitialPosture;
using numi::human::resting_initial_posture::readRestingInitialPosture;

constexpr char kRigidSHA[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
std::size_t checks = 0u;

void require(bool condition, const char* label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}

class TemporaryLog final {
public:
    explicit TemporaryLog(const std::string& contents) {
        std::vector<char> pattern{
            '/', 't', 'm', 'p', '/', 'n', 'u', 'm', 'i', '-', 'i', 'n', 'i',
            't', '-', 'p', 'o', 's', 't', 'u', 'r', 'e', '-', 't', 'e', 's',
            't', '-', 'X', 'X', 'X', 'X', 'X', 'X', '\0'};
        const int descriptor = ::mkstemp(pattern.data());
        if (descriptor < 0) throw std::runtime_error("mkstemp failed");
        path_ = pattern.data();
        std::size_t written = 0u;
        while (written < contents.size()) {
            const ssize_t count = ::write(descriptor,
                contents.data() + written, contents.size() - written);
            if (count < 0) {
                const int savedErrno = errno;
                ::close(descriptor);
                ::unlink(path_.c_str());
                throw std::runtime_error(
                    std::string("write failed: ") + std::strerror(savedErrno));
            }
            if (count == 0) {
                ::close(descriptor);
                ::unlink(path_.c_str());
                throw std::runtime_error("short write to temporary log");
            }
            written += static_cast<std::size_t>(count);
        }
        if (::close(descriptor) != 0) {
            ::unlink(path_.c_str());
            throw std::runtime_error("close failed for temporary log");
        }
    }

    ~TemporaryLog() {
        if (!path_.empty()) ::unlink(path_.c_str());
    }

    TemporaryLog(const TemporaryLog&) = delete;
    TemporaryLog& operator=(const TemporaryLog&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

class OversizedTemporaryLog final {
public:
    OversizedTemporaryLog() {
        std::vector<char> pattern{
            '/', 't', 'm', 'p', '/', 'n', 'u', 'm', 'i', '-', 'i', 'n', 'i',
            't', '-', 'p', 'o', 's', 't', 'u', 'r', 'e', '-', 'l', 'a', 'r',
            'g', 'e', '-', 'X', 'X', 'X', 'X', 'X', 'X', '\0'};
        const int descriptor = ::mkstemp(pattern.data());
        if (descriptor < 0) throw std::runtime_error("mkstemp failed");
        path_ = pattern.data();
        constexpr off_t kOversizedLength =
            static_cast<off_t>(256ull * 1024ull * 1024ull + 1ull);
        if (::ftruncate(descriptor, kOversizedLength) != 0) {
            const int savedErrno = errno;
            ::close(descriptor);
            ::unlink(path_.c_str());
            throw std::runtime_error(
                std::string("ftruncate failed: ") + std::strerror(savedErrno));
        }
        if (::close(descriptor) != 0) {
            ::unlink(path_.c_str());
            throw std::runtime_error("close failed for oversized temporary log");
        }
    }

    ~OversizedTemporaryLog() {
        if (!path_.empty()) ::unlink(path_.c_str());
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

std::string sourceLine(const std::string& hash = kRigidSHA) {
    return "rigid_source_rest_frames=passed source_body_count=86 rigid_sha256=" +
        hash + " boundary=consumed_reference_frame_join";
}

std::string footerLine(const std::string& state = "completed",
                       const std::string& assistance = "false") {
    return "resting_integrated_body=" + state +
        " simulated_s=20 wall_s=2 root_assistance=" + assistance;
}

std::string validJSON(const std::string& q =
                          "[1.2345678806304932,-0.0,-2.5,0,0,0,1,0.125]",
                      const std::string& schema =
                          "numi.human.legacy-stand-terminal.v1",
                      const std::string& stepCount = "10000",
                      const std::string& timestep = "0.002",
                      const std::string& rootAssistance = "false",
                      const std::string& extra = "") {
    return "{\"schema\":\"" + schema + "\",\"step_count\":" + stepCount +
        ",\"timestep_seconds\":" + timestep +
        ",\"root_assistance\":" + rootAssistance + ",\"q\":" + q +
        extra + "}";
}

std::string validLog(const std::string& json,
                     const std::string& source = sourceLine(),
                     const std::string& footer = footerLine()) {
    return "native startup\n" + source + "\n" +
        "stand_terminal_state=" + json + "\n" +
        footer + "\n";
}

void expectRejected(const char* label, const std::string& log,
                    const std::string& expectedSHA = kRigidSHA,
                    std::size_t nq = 8u) {
    TemporaryLog input(log);
    bool rejected = false;
    try {
        (void)readRestingInitialPosture(input.path(), expectedSHA, nq);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, label);
}

void testAcceptsExactPostureOnly() {
    const std::string json = validJSON(
        "[1.2345678806304932,-0.0,-2.5,0,0,0,1,0.125]",
        "numi.human.legacy-stand-terminal.v1",
        "10000", "0.002", "false",
        ",\"initial_q\":[9,9,9],\"v\":[1,2,3],"
        "\"initial_v\":[4,5],\"muscle_activation\":[0.7]");
    TemporaryLog input(validLog(json));
    const InitialPosture value =
        readRestingInitialPosture(input.path(), kRigidSHA, 8u);
    const std::array<float, 8u> expected{
        1.2345678806304932f, -0.0f, -2.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.125f};
    require(value.q.size() == expected.size(),
            "valid posture preserves q count");
    require(std::memcmp(value.q.data(), expected.data(), sizeof(expected)) == 0,
            "valid posture preserves Float32 q values without projection");
    require(value.stepCount == 10000u,
            "valid posture returns terminal source step count");
    require(value.timestepSeconds == 0.002,
            "valid posture returns terminal source timestep");
}

void testAcceptsRawNativeFloat32DecimalAfterNSNumberFormatting() {
    const std::string json = validJSON(
        "[-0.23904088139533997,-0.0,-2.5,0,0,0,1,0.125]");
    TemporaryLog input(validLog(json));
    const InitialPosture value =
        readRestingInitialPosture(input.path(), kRigidSHA, 8u);
    const std::array<float, 8u> expected{
        -0.23904088139533997f, -0.0f, -2.5f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.125f};
    require(std::memcmp(value.q.data(), expected.data(), sizeof(expected)) == 0,
            "raw native decimal survives lossy NSNumber string formatting exactly");
}

void testRejectsFileAdmissionFailures() {
    expectRejected("rejects an empty log", "", kRigidSHA, 8u);
    bool directoryRejected = false;
    try {
        (void)readRestingInitialPosture("/tmp", kRigidSHA, 8u);
    } catch (const std::runtime_error&) {
        directoryRejected = true;
    }
    require(directoryRejected, "rejects a directory as the posture log");

    OversizedTemporaryLog oversized;
    bool sizeRejected = false;
    try {
        (void)readRestingInitialPosture(oversized.path(), kRigidSHA, 8u);
    } catch (const std::runtime_error&) {
        sizeRejected = true;
    }
    require(sizeRejected, "rejects a file larger than 256 MiB before reading");
}

void testRejectsSourceIdentityFailures() {
    const std::string json = validJSON();
    expectRejected("rejects absent rigid-source record",
        validLog(json, "ordinary log line"));
    expectRejected("rejects mismatched rigid-source hash",
        validLog(json, sourceLine(std::string(64u, 'a'))));
    expectRejected("rejects duplicate rigid-source records",
        "native startup\n" + sourceLine() + "\n" + sourceLine() + "\n" +
            "stand_terminal_state=" + json + "\n" + footerLine() + "\n");
    expectRejected("rejects duplicate rigid SHA token on one source record",
        validLog(json, sourceLine() + " rigid_sha256=" + kRigidSHA));
    expectRejected("rejects SHA substring in a longer token",
        validLog(json, sourceLine() + "x", kRigidSHA));
    expectRejected("rejects malformed caller SHA",
        validLog(json), "not-a-sha", 8u);
}

void testRejectsTerminalRecordFailures() {
    expectRejected("rejects wrong terminal schema",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.other.v1")));
    expectRejected("rejects malformed terminal JSON",
        validLog("{not json"));
    expectRejected("rejects duplicate terminal records",
        validLog(validJSON()) +
            "stand_terminal_state=" + validJSON() + "\n");
    expectRejected("rejects missing terminal q",
        validLog("{\"schema\":\"numi.human.legacy-stand-terminal.v1\","
                 "\"step_count\":10000,\"timestep_seconds\":0.002,"
                 "\"root_assistance\":false}"));
}

void testRejectsCompletionAndAssistanceFailures() {
    const std::string json = validJSON();
    expectRejected("rejects missing completion footer",
        "native startup\n" + sourceLine() + "\n" +
            "stand_terminal_state=" + json + "\n");
    expectRejected("rejects failed completion footer",
        validLog(json, sourceLine(), footerLine("failed")));
    expectRejected("rejects duplicate completion footers",
        validLog(json) + footerLine() + "\n");
    expectRejected("rejects footer with root assistance",
        validLog(json, sourceLine(), footerLine("completed", "true")));
    expectRejected("rejects terminal root assistance",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "10000", "0.002", "true")));
    expectRejected("rejects non-boolean root assistance",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "10000", "0.002", "0")));
    expectRejected("rejects missing terminal root assistance",
        validLog("{\"schema\":\"numi.human.legacy-stand-terminal.v1\","
                 "\"step_count\":10000,\"timestep_seconds\":0.002,"
                 "\"q\":[1,2,3,0,0,0,1,0]}"));
}

void testRejectsAmbiguousJSONAndRecordOrder() {
    const std::string json = validJSON();
    expectRejected("rejects completion before terminal",
        sourceLine() + "\n" + footerLine() + "\nstand_terminal_state=" + json + "\n");
    expectRejected("rejects terminal before source",
        "stand_terminal_state=" + json + "\n" + sourceLine() + "\n" + footerLine() + "\n");
    for (const std::string member : {
             std::string("\"root_assistance\":false"),
             std::string("\"step_count\":10000"),
             std::string("\"q\":[1,2,3,0,0,0,1,0]"),
             std::string("\"\\u0071\":[1,2,3,0,0,0,1,0]")}) {
        expectRejected("rejects duplicate decoded terminal member",
            validLog(json.substr(0u, json.size() - 1u) + "," + member + "}"));
    }
}

void testRejectsInvalidDimensionsAndScalars() {
    expectRejected("rejects requested q size below root quaternion",
        validLog(validJSON()), kRigidSHA, 6u);
    expectRejected("rejects wrong q count",
        validLog(validJSON("[1,2,3,0,0,0,1]")));
    expectRejected("rejects boolean q entry",
        validLog(validJSON("[1,2,true,0,0,0,1,0]")));
    expectRejected("rejects a noncanonical decimal q value",
        validLog(validJSON("[1.23456789,2,3,0,0,0,1,0]")));
    expectRejected("rejects q underflow to Float32 zero",
        validLog(validJSON("[1e-50,2,3,0,0,0,1,0]")));
    expectRejected("rejects JSON NaN",
        validLog(validJSON("[1,2,NaN,0,0,0,1,0]")));
    expectRejected("rejects finite double that overflows Float32",
        validLog(validJSON("[1,2,1e40,0,0,0,1,0]")));
    expectRejected("rejects zero step count",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "0")));
    expectRejected("rejects boolean step count",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "true")));
    expectRejected("rejects zero timestep",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "10000", "0")));
    expectRejected("rejects nonfinite timestep",
        validLog(validJSON("[1,2,3,0,0,0,1,0]",
                           "numi.human.legacy-stand-terminal.v1",
                           "10000", "1e400")));
    expectRejected("rejects non-unit root quaternion",
        validLog(validJSON("[1,2,3,0,0,0,1.01,0]")));
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 1 && argc != 4) {
                throw std::runtime_error(
                    "usage: test [real-native-log expected-rigid-sha nq]");
            }
            testAcceptsExactPostureOnly();
            testAcceptsRawNativeFloat32DecimalAfterNSNumberFormatting();
            testRejectsFileAdmissionFailures();
            testRejectsSourceIdentityFailures();
            testRejectsTerminalRecordFailures();
            testRejectsCompletionAndAssistanceFailures();
            testRejectsInvalidDimensionsAndScalars();
            testRejectsAmbiguousJSONAndRecordOrder();
            if (argc == 4) {
                const std::size_t requestedNQ =
                    static_cast<std::size_t>(std::stoull(argv[3]));
                const InitialPosture real =
                    readRestingInitialPosture(argv[1], argv[2], requestedNQ);
                std::cout << "real retained posture accepted step_count="
                          << real.stepCount << " timestep_seconds="
                          << real.timestepSeconds << " nq=" << real.q.size()
                          << "\n";
            }
            std::cout << "Resting initial-posture reader: " << checks
                      << " checks passed; no physics/restart state consumed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
