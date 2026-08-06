#include "metalrobo/G1.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/MotionCompiler.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path interactionPack;
    std::filesystem::path output;
    std::string robot;
    std::string id;
    std::string clip;
    std::string anchorBody;
    std::vector<std::string> trackedBodies;
};

void usage() {
    std::cerr
        << "usage: metalrobo_motion_compile --robot unitree-g1 "
           "--interaction-pack FILE --output FILE --id ID "
           "[--clip ID] --anchor-body BODY "
           "--tracked-bodies BODY[,BODY...]\n";
}

std::vector<std::string> split(const std::string_view value) {
    std::vector<std::string> result;
    std::size_t begin = 0u;
    while (begin <= value.size()) {
        const std::size_t end = value.find(',', begin);
        const std::string_view token = value.substr(
            begin,
            end == std::string_view::npos
                ? value.size() - begin
                : end - begin
        );
        if (token.empty()) {
            throw std::runtime_error("tracked body list contains an empty name");
        }
        result.emplace_back(token);
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1u;
    }
    return result;
}

Options parse(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument{argv[index]};
        if (index + 1 >= argc) {
            usage();
            throw std::runtime_error("missing value for " + argument);
        }
        const std::string value{argv[++index]};
        if (argument == "--interaction-pack") {
            options.interactionPack = value;
        } else if (argument == "--output") {
            options.output = value;
        } else if (argument == "--robot") {
            options.robot = value;
        } else if (argument == "--id") {
            options.id = value;
        } else if (argument == "--clip") {
            options.clip = value;
        } else if (argument == "--anchor-body") {
            options.anchorBody = value;
        } else if (argument == "--tracked-bodies") {
            options.trackedBodies = split(value);
        } else {
            usage();
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.interactionPack.empty() || options.output.empty() ||
        options.robot.empty() || options.id.empty() ||
        options.anchorBody.empty() || options.trackedBodies.empty()) {
        usage();
        throw std::runtime_error("required motion compiler option is missing");
    }
    return options;
}

metalrobo::EngineModel model(const std::string_view robot) {
    if (robot == "unitree-g1") {
        return metalrobo::makeUnitreeG1EngineModel();
    }
    throw std::runtime_error("unsupported robot: " + std::string{robot});
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc == 2 &&
            (std::string_view{argv[1]} == "--help" ||
             std::string_view{argv[1]} == "-h")) {
            usage();
            return 0;
        }
        const Options options = parse(argc, argv);
        metalrobo::InteractionPack interactions;
        const metalrobo::LearningPackResult read =
            metalrobo::readInteractionPack(
                options.interactionPack,
                interactions
            );
        if (!read.succeeded()) {
            throw std::runtime_error(
                std::string{"InteractionPack read ["} +
                metalrobo::learningPackStatusName(read.status) +
                "]: " + read.message
            );
        }
        metalrobo::MotionPack motion;
        const metalrobo::MotionCompileResult compiled =
            metalrobo::compileInteractionMotionPack(
                interactions,
                model(options.robot),
                {
                    .id = options.id,
                    .clipId = options.clip,
                    .anchorBody = options.anchorBody,
                    .trackedBodies = options.trackedBodies,
                },
                motion
            );
        if (!compiled.succeeded()) {
            throw std::runtime_error(
                std::string{"motion compile ["} +
                metalrobo::motionCompileStatusName(compiled.status) +
                "]: " + compiled.message
            );
        }
        const metalrobo::LearningPackResult written =
            metalrobo::writeMotionPack(motion, options.output);
        if (!written.succeeded()) {
            throw std::runtime_error(
                std::string{"MotionPack write ["} +
                metalrobo::learningPackStatusName(written.status) +
                "]: " + written.message
            );
        }
        const std::size_t frameCount =
            motion.clips.front().features.size() / motion.featureCount;
        std::cout
            << "motion_pack=\"" << options.output.string() << "\""
            << " robot=" << options.robot
            << " clip=\"" << motion.clips.front().id << "\""
            << " frames=" << frameCount
            << " features=" << motion.featureCount
            << " content_hash=" << written.contentHash
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_motion_compile: " << error.what() << '\n';
        return 1;
    }
}
