#include "metalrobo/VisualPresentation.hpp"

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void printUsage() {
    std::cerr
        << "usage: metalrobo_environment_cook INPUT.hdr|exr OUTPUT.mrenv "
           "[--id ID] [--face-size N] "
           "[--source-color-space auto|linear-rec709|acescg]\n";
}

bool parseFaceSize(
    const std::string_view text,
    std::uint32_t& value
) {
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} &&
           result.ptr == last &&
           value != 0u;
}

} // namespace

int main(const int argc, const char* const* argv) {
    if (argc < 3) {
        printUsage();
        return 2;
    }

    metalrobo::VisualEnvironmentCookOptions options{};
    const std::filesystem::path input = argv[1];
    const std::filesystem::path output = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto nextValue = [&]() -> const char* {
            if (index + 1 >= argc) {
                return nullptr;
            }
            return argv[++index];
        };

        if (argument == "--id") {
            const char* const value = nextValue();
            if (value == nullptr) {
                printUsage();
                return 2;
            }
            options.id = value;
        } else if (argument == "--face-size") {
            const char* const value = nextValue();
            if (value == nullptr ||
                !parseFaceSize(value, options.faceSize)) {
                std::cerr << "invalid --face-size\n";
                return 2;
            }
        } else if (argument == "--source-color-space") {
            const char* const value = nextValue();
            if (value == nullptr) {
                printUsage();
                return 2;
            }
            options.sourceColorSpace = value;
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            printUsage();
            return 2;
        }
    }

    metalrobo::VisualEnvironmentPackV2 pack{};
    const auto diagnostics =
        metalrobo::cookVisualEnvironment(input, pack, options);
    if (!diagnostics.succeeded()) {
        std::cerr << diagnostics.message << '\n';
        return 1;
    }

    std::string reason;
    if (!metalrobo::writeVisualEnvironmentPack(
            pack,
            output,
            &reason
        )) {
        std::cerr << reason << '\n';
        return 1;
    }

    std::cout << pack.contentHash << '\n';
    return 0;
}
