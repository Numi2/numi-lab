#include "NumiHumanRestingBedSurface.hpp"
#include <Foundation/Foundation.h>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* kSkin = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char* kSupport = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr const char* kCapture = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
unsigned checks = 0;

void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}

struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "numi-bed-admission-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        char* created = mkdtemp(buffer.data());
        if (!created) throw std::runtime_error("could not create temporary test directory");
        path = created;
    }
    ~TempDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

std::string validManifest() {
    std::string out = R"JSON({
      "schema":"numi.human.resting-supine-source-scene.v1",
      "source":{"skin":{"sha256":")JSON";
    out += kSkin;
    out += R"JSON("}},
      "outputs":{"support_contact":{"sha256":")JSON";
    out += kSupport;
    out += R"JSON("}},
      "bed":{"heightfield":{
        "frame":"fixed_world","triangulation":"00_10_01__10_11_01",
        "nx":3,"ny":3,"origin_xy_m":[-0.005,-0.005],"spacing_xy_m":[0.005,0.005],
        "heights_m":[0,0,0,0,0.01,0,0,0,0],
        "source_capture_path":"/tmp/source.mrvpack","source_capture_sha256":")JSON";
    out += kCapture;
    out += R"JSON("
      }}
    })JSON";
    return out;
}

void replaceOnce(std::string& text, const std::string& before, const std::string& after) {
    const auto at = text.find(before);
    if (at == std::string::npos) throw std::runtime_error("test fixture token not found: " + before);
    text.replace(at, before.size(), after);
}

void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream << text;
}

void accepts(const std::filesystem::path& path, const std::string& label) {
    try {
        NumiHumanRestingBedSurface parsed(path, kSkin, kSupport);
        check(parsed.gpu.counts.x == 3 && parsed.gpu.counts.y == 3, label + " dimensions");
        check(parsed.heights.size() == 9 && parsed.heights[4] == 0.01f, label + " height payload");
        check(parsed.sourceCaptureSHA256 == kCapture, label + " capture identity");
    } catch (const std::exception& error) {
        throw std::runtime_error(label + " unexpectedly rejected: " + error.what());
    }
}

void rejects(const std::filesystem::path& path, const std::string& manifest,
             const std::string& label, const std::string& expectedSkin = kSkin,
             const std::string& expectedSupport = kSupport) {
    write(path, manifest);
    bool failed = false;
    try { NumiHumanRestingBedSurface parsed(path, expectedSkin, expectedSupport); }
    catch (const std::exception&) { failed = true; }
    check(failed, label + " must be rejected");
}

void runFixtureTests(const std::filesystem::path& root) {
    const auto path = root / "fixture.json";
    const std::string original = validManifest();
    write(path, original);
    accepts(path, "minimal valid manifest");

    rejects(path, original, "mismatched skin identity", std::string(64, 'd'), kSupport);
    rejects(path, original, "mismatched support identity", kSkin, std::string(64, 'e'));
    auto changed = original; replaceOnce(changed, "\"nx\":3", "\"nx\":1025");
    rejects(path, changed, "grid extent cap");
    changed = original; replaceOnce(changed, "\"nx\":3", "\"nx\":3.5");
    rejects(path, changed, "fractional node count");
    changed = original; replaceOnce(changed, "\"nx\":3", "\"nx\":513");
    replaceOnce(changed, "\"ny\":3", "\"ny\":512");
    rejects(path, changed, "total node capacity");
    changed = original; replaceOnce(changed, "\"spacing_xy_m\":[0.005,0.005]", "\"spacing_xy_m\":[0,0.005]");
    rejects(path, changed, "zero spacing");
    changed = original; replaceOnce(changed, "\"spacing_xy_m\":[0.005,0.005]", "\"spacing_xy_m\":[0.0005,0.005]");
    rejects(path, changed, "spacing below declared bounds");
    changed = original; replaceOnce(changed, "\"origin_xy_m\":[-0.005,-0.005]", "\"origin_xy_m\":[100000000,-0.005]");
    rejects(path, changed, "Float32 nonmonotonic grid");
    changed = original; replaceOnce(changed, "\"origin_xy_m\":[-0.005,-0.005]", "\"origin_xy_m\":[1e100,-0.005]");
    rejects(path, changed, "finite Float32 origin requirement");
    changed = original; replaceOnce(changed, "\"heights_m\":[0,0,0,0,0.01,0,0,0,0]", "\"heights_m\":[0,0,0,0,0.01,0,0,0]");
    rejects(path, changed, "height count mismatch");
    changed = original; replaceOnce(changed, "0.01", "1e100");
    rejects(path, changed, "finite Float32 height requirement");
    changed = original; replaceOnce(changed, "0.01", "0.51");
    rejects(path, changed, "height range");
    changed = original; replaceOnce(changed, "\"heights_m\":[0,0,0,0,0.01,0,0,0,0]", "\"heights_m\":[0.001,0,0,0,0.01,0,0,0,0]");
    rejects(path, changed, "zero boundary rim");
    changed = original; replaceOnce(changed, "00_10_01__10_11_01", "other_diagonal");
    rejects(path, changed, "unsupported diagonal");
    changed = original; replaceOnce(changed, kCapture, "not-a-sha");
    rejects(path, changed, "capture SHA syntax");
}

void runFrozenManifest(const std::filesystem::path& path, const std::string& skin,
                      const std::string& support) {
    NumiHumanRestingBedSurface parsed(path, skin, support);
    check(parsed.gpu.counts.x == 261 && parsed.gpu.counts.y == 461,
          "frozen 1196 manifest node dimensions");
    check(parsed.heights.size() == 261u * 461u, "frozen 1196 manifest height count");
    check(parsed.sourceCaptureSHA256 == "68dd2aecebd5febb561738e586cb9093813aee61309ccf1c0ceae5a535751bef",
          "frozen 1196 manifest capture identity");
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            TempDirectory temp;
            runFixtureTests(temp.path);
            if (argc == 1) {
                std::cout << "checks=" << checks << " status=passed scope=native_bed_manifest_reader" << std::endl;
                return 0;
            }
            if (argc != 5 || std::string(argv[1]) != "--manifest")
                throw std::runtime_error("usage: bed-admission-check [--manifest PATH SKIN_SHA SUPPORT_SHA]");
            runFrozenManifest(argv[2], argv[3], argv[4]);
            std::cout << "checks=" << checks << " status=passed scope=native_bed_manifest_reader including_frozen_manifest" << std::endl;
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
}
