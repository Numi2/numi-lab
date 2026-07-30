#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/WorldPack.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(
    const metalrobo::WorldCompileResult& result,
    const char* stage
) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{stage} + ": " + result.message
        );
    }
}

void require(
    const metalrobo::WorldPackResult& result,
    const char* stage
) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{stage} + " [" +
            metalrobo::worldPackStatusName(result.status) +
            "]: " + result.message
        );
    }
}

void printPack(
    const std::filesystem::path& path,
    const metalrobo::MRWorldPack& pack
) {
    const auto& world = pack.family.worldTemplate;
    std::cout
        << "path=\"" << path.string() << "\""
        << " content_hash=" << pack.contentHash
        << " family_fingerprint=" << pack.family.fingerprint
        << " template_fingerprint=" << world.fingerprint
        << " assets=" << world.assets.size()
        << " bodies=" << world.engineModel.bodies.size()
        << " articulations=" << world.engineModel.articulations.size()
        << " variations=" << pack.family.program.variations.size()
        << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string{argv[1]} == "--inspect") {
            metalrobo::MRWorldPack pack;
            require(
                metalrobo::readWorldPack(argv[2], pack),
                "read"
            );
            printPack(argv[2], pack);
            return 0;
        }
        const bool tactile =
            argc == 3 &&
            std::string{argv[1]} == "--franka-tactile";
        if (argc != 2 && !tactile) {
            std::cerr
                << "usage: metalrobo_world_pack <output.mrworld>\n"
                << "       metalrobo_world_pack --franka-tactile "
                   "<output.mrworld>\n"
                << "       metalrobo_world_pack --inspect "
                   "<input.mrworld>\n";
            return 2;
        }

        const metalrobo::EpisodeTwin episode =
            tactile
            ? metalrobo::makeFrankaTactileEpisodeTwin()
            : metalrobo::makeFrankaPickPlaceEpisodeTwin();
        metalrobo::WorldTemplate worldTemplate;
        require(
            metalrobo::compileEpisodeTwin(
                episode,
                tactile
                ? metalrobo::makeFrankaTactileEngineModel()
                : metalrobo::makeFrankaPickPlaceEngineModel(),
                worldTemplate
            ),
            "episode compile"
        );
        metalrobo::WorldFamily family;
        require(
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            ),
            "family compile"
        );
        metalrobo::MRWorldPack pack;
        require(
            metalrobo::compileWorldPack(family, pack),
            "pack compile"
        );
        const std::filesystem::path output{
            tactile ? argv[2] : argv[1]
        };
        require(metalrobo::writeWorldPack(pack, output), "write");
        printPack(output, pack);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_world_pack: "
                  << error.what() << '\n';
        return 1;
    }
}
