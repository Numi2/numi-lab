#include "metalrobo/EpisodeTwinCompiler.hpp"

#include "metalrobo/FrankaWorld.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

metalrobo::EngineModel resolveEngineModel(const std::string& id) {
    if (id == "franka_pick_place") {
        return metalrobo::makeFrankaPickPlaceEngineModel();
    }
    throw std::runtime_error("unregistered engine model: " + id);
}

metalrobo::WorldProgram resolveWorldProgram(const std::string& id) {
    if (id == "franka_pick_place") {
        return metalrobo::makeFrankaPickPlaceWorldProgram();
    }
    throw std::runtime_error("unregistered world program: " + id);
}

void printUsage() {
    std::cerr << "usage: metalrobo_episode_compile "
                 "<capture.json> <output.mrworld> "
                 "[--store <directory>] [--no-resume] "
                 "[--require-hashes]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            printUsage();
            return 2;
        }
        const std::filesystem::path manifestPath{argv[1]};
        const std::filesystem::path outputPath{argv[2]};
        metalrobo::EpisodeTwinCompilerConfig config;
        config.artifactStore = outputPath.parent_path() /
                               (outputPath.stem().string() + ".artifacts");
        for (int index = 3; index < argc; ++index) {
            const std::string argument{argv[index]};
            if (argument == "--store") {
                if (++index >= argc) {
                    printUsage();
                    return 2;
                }
                config.artifactStore = argv[index];
            } else if (argument == "--no-resume") {
                config.resume = false;
            } else if (argument == "--require-hashes") {
                config.requireExpectedHashes = true;
            } else {
                printUsage();
                return 2;
            }
        }

        metalrobo::CaptureManifest manifest;
        const metalrobo::EpisodeTwinCompilerResult loaded =
            metalrobo::loadCaptureManifestJSON(manifestPath, manifest);
        if (!loaded.succeeded()) {
            throw std::runtime_error(
                std::string{"manifest load ["} +
                metalrobo::episodeTwinCompilerStatusName(loaded.status) +
                "]: " + loaded.message);
        }

        metalrobo::EpisodeTwinCompiler compiler{config};
        metalrobo::CompiledEpisodeTwin compiled;
        const metalrobo::EpisodeTwinCompilerResult result = compiler.compile(
            manifest, resolveEngineModel(manifest.engineModelId),
            resolveWorldProgram(manifest.worldProgramId), compiled);
        if (!result.succeeded()) {
            throw std::runtime_error(
                std::string{"episode compile ["} +
                metalrobo::episodeTwinCompilerStatusName(result.status) +
                "]: " + result.message);
        }
        const metalrobo::WorldPackResult written =
            metalrobo::writeWorldPack(compiled.worldPack, outputPath);
        if (!written.succeeded()) {
            throw std::runtime_error(
                std::string{"world-pack write ["} +
                metalrobo::worldPackStatusName(written.status) +
                "]: " + written.message);
        }

        std::size_t cacheHits = 0u;
        for (const auto& receipt : compiled.receipts) {
            cacheHits += receipt.cacheHit ? 1u : 0u;
            std::cout << "stage="
                      << metalrobo::episodeTwinStageName(receipt.stage)
                      << " key=" << receipt.stageKey
                      << " cache=" << (receipt.cacheHit ? "hit" : "miss")
                      << " outputs=" << receipt.artifacts.size() << '\n';
        }
        std::cout << "world_pack=\"" << outputPath.string() << "\""
                  << " episode=\"" << compiled.episode.id << "\""
                  << " artifacts=" << compiled.episode.artifacts.size()
                  << " assets=" << compiled.worldTemplate.assets.size()
                  << " bodies="
                  << compiled.worldTemplate.engineModel.bodies.size()
                  << " variations="
                  << compiled.worldFamily.program.variations.size()
                  << " content_hash=" << compiled.worldPack.contentHash
                  << " resumed_stages=" << cacheHits << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_episode_compile: " << error.what() << '\n';
        return 1;
    }
}
