#include "metalrobo/VisualPresentation.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "usage: metalrobo_visual_cook [--urdf] INPUT OUTPUT "
           "[--id ID] [--license SPDX] [--provenance TEXT] "
           "[--link NAME=BODY_INDEX ...] "
           "[--rigid NAME=BODY_INDEX ...]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    bool urdf = false;
    std::vector<std::string> positional;
    metalrobo::VisualAssetCookOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--urdf") {
            urdf = true;
        } else if (argument == "--link" || argument == "--rigid") {
            if (index + 1 >= argc) {
                usage();
                return 2;
            }
            const std::string binding{argv[++index]};
            const std::size_t separator = binding.find('=');
            if (separator == std::string::npos || separator == 0u ||
                separator + 1u == binding.size()) {
                usage();
                return 2;
            }
            try {
                const unsigned long body = std::stoul(
                    binding.substr(separator + 1u)
                );
                if (body > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::out_of_range{"body index"};
                }
                auto& bindings = argument == "--link"
                    ? options.linkBodyIndices
                    : options.rigidBodyIndices;
                bindings.emplace(
                    binding.substr(0u, separator),
                    static_cast<std::uint32_t>(body)
                );
            } catch (...) {
                usage();
                return 2;
            }
        } else if (
            argument == "--id" ||
            argument == "--license" ||
            argument == "--provenance"
        ) {
            if (index + 1 >= argc) {
                usage();
                return 2;
            }
            const std::string value{argv[++index]};
            if (argument == "--id") {
                options.id = value;
            } else if (argument == "--license") {
                options.license = value;
            } else {
                options.preprocessingProvenance = value;
            }
        } else if (argument.starts_with("--")) {
            usage();
            return 2;
        } else {
            positional.emplace_back(argument);
        }
    }
    if (positional.size() != 2u) {
        usage();
        return 2;
    }

    const std::filesystem::path input{positional[0]};
    const std::filesystem::path output{positional[1]};
    if (urdf) {
        std::vector<metalrobo::VisualAssetPackV2> packs;
        const auto diagnostics =
            metalrobo::cookUrdfVisualDescription(
                input,
                packs,
                options
            );
        if (!diagnostics.succeeded()) {
            std::cerr
                << metalrobo::visualAssetCookStatusName(
                       diagnostics.status
                   )
                << ": " << diagnostics.message << '\n';
            return 1;
        }
        for (std::size_t index = 0u;
             index < packs.size();
             ++index) {
            std::string reason;
            const std::filesystem::path path =
                output /
                (
                    packs[index].id + "_" +
                    std::to_string(index) + ".mrvpack"
                );
            if (!metalrobo::writeVisualAssetPack(
                    packs[index],
                    path,
                    &reason
                )) {
                std::cerr << "write_failure: " << reason << '\n';
                return 1;
            }
        }
        std::cout
            << "cooked " << packs.size()
            << " URDF visual packs; source="
            << diagnostics.sourceHash << '\n';
        return 0;
    }

    metalrobo::VisualAssetPackV2 pack;
    const auto diagnostics =
        metalrobo::cookVisualAsset(input, pack, options);
    if (!diagnostics.succeeded()) {
        std::cerr
            << metalrobo::visualAssetCookStatusName(
                   diagnostics.status
               )
            << ": " << diagnostics.message << '\n';
        return 1;
    }
    std::string reason;
    if (!metalrobo::writeVisualAssetPack(pack, output, &reason)) {
        std::cerr << "write_failure: " << reason << '\n';
        return 1;
    }
    std::cout
        << "cooked " << diagnostics.vertexCount << " vertices, "
        << diagnostics.indexCount << " indices, "
        << diagnostics.materialCount << " materials, "
        << diagnostics.textureCount << " textures; pack="
        << diagnostics.packHash << '\n';
    return 0;
}
