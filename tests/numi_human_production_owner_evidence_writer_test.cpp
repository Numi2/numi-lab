#include "metalrobo/NumiHumanProductionOwnerEvidenceWriter.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <unistd.h>

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr,
        "numanx production-owner evidence writer: %s\n", message.c_str());
    std::exit(1);
}

void require(const bool condition, const std::string& message) {
    if (!condition) fail(message);
}

class TemporaryDirectory final {
public:
    TemporaryDirectory() {
        std::array<char, 128u> pattern{};
        const int written = std::snprintf(pattern.data(), pattern.size(),
            "/private/tmp/numi-owner-evidence-writer.%ld.XXXXXX",
            static_cast<long>(::getpid()));
        require(written > 0 &&
                static_cast<std::size_t>(written) < pattern.size(),
            "temporary path construction failed");
        char* created = ::mkdtemp(pattern.data());
        require(created != nullptr, "temporary directory creation failed");
        path_ = created;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "published evidence file did not open");
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

} // namespace

int main() {
    TemporaryDirectory temporary;
    const auto nested = temporary.path() / "first" / "second" / "third";
    const auto target = nested / "evidence.json";
    const std::string first = "{\"evidence\":\"first\"}\n";
    std::string error;
    require(metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            target, first, error),
        "first durable publication failed: " + error);
    require(std::filesystem::is_directory(nested),
        "nested evidence directory was not created");
    require(readFile(target) == first,
        "published evidence bytes differ from source");

    const std::string replacement = "{\"evidence\":\"replacement\"}\n";
    require(!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            target, replacement, error),
        "no-replace writer overwrote an existing target");
    require(error == "production-owner evidence target already exists",
        "existing-target failure was not typed exactly");
    require(readFile(target) == first,
        "failed replacement changed published evidence bytes");

    std::size_t entryCount = 0u;
    for (const auto& entry : std::filesystem::directory_iterator(nested)) {
        ++entryCount;
        require(entry.path() == target,
            "temporary evidence file remained after publication");
    }
    require(entryCount == 1u,
        "evidence directory contains an unexpected entry");

    const auto blockingFile = temporary.path() / "not-a-directory";
    {
        std::ofstream output(blockingFile, std::ios::binary);
        require(output.good(), "blocking file creation failed");
        output << "block";
    }
    require(!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            blockingFile / "evidence.json", first, error),
        "writer traversed a regular file as a directory");
    require(!error.empty(), "writer failure did not report an error");

    const auto realDirectory = temporary.path() / "real-directory";
    const auto symbolicDirectory = temporary.path() / "symbolic-directory";
    std::filesystem::create_directory(realDirectory);
    std::error_code symlinkError;
    std::filesystem::create_directory_symlink(
        realDirectory, symbolicDirectory, symlinkError);
    require(!symlinkError, "directory symlink setup failed");
    require(!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            symbolicDirectory / "evidence.json", first, error),
        "writer followed a symbolic directory component");
    require(!std::filesystem::exists(realDirectory / "evidence.json"),
        "symlink traversal published evidence outside the requested path");

    const auto traversalTarget = temporary.path() / "child" / ".." /
        "escaped-evidence.json";
    require(!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
            traversalTarget, first, error),
        "writer accepted a parent-directory traversal component");
    require(error ==
            "production-owner evidence directory may not contain '..'",
        "parent-directory traversal failure was not typed exactly");
    require(!std::filesystem::exists(
            temporary.path() / "escaped-evidence.json"),
        "parent-directory traversal published an escaped artifact");
    require(!std::filesystem::exists(temporary.path() / "child"),
        "parent-directory traversal created a partial directory");

    std::puts("numanx_production_owner_evidence_writer=pass");
    return 0;
}
