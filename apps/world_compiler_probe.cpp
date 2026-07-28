#include "metalrobo/Franka.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/WorldCompiler.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
bool sameBytes(
    const std::vector<T>& left,
    const std::vector<T>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(T)
         ) == 0);
}

} // namespace

int main() {
    try {
        const metalrobo::EpisodeTwin episode =
            metalrobo::makeFrankaPickPlaceEpisodeTwin();
        metalrobo::WorldTemplate worldTemplate;
        const metalrobo::WorldCompileResult twinResult =
            metalrobo::compileEpisodeTwin(
                episode,
                metalrobo::makeFrankaPandaEngineModel(),
                worldTemplate
            );
        if (!twinResult.succeeded()) {
            throw std::runtime_error(twinResult.message);
        }

        metalrobo::WorldFamily family;
        const metalrobo::WorldCompileResult familyResult =
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            );
        if (!familyResult.succeeded()) {
            throw std::runtime_error(familyResult.message);
        }

        constexpr std::uint32_t kWorldCount = 4096u;
        constexpr std::uint64_t kSeed = 0x123456789abcdef0ull;
        const metalrobo::WorldInstanceBatch first =
            family.sample(kWorldCount, kSeed);
        const metalrobo::WorldInstanceBatch replay =
            family.sample(kWorldCount, kSeed);
        std::string reason;
        if (!first.valid(&reason)) {
            throw std::runtime_error(reason);
        }
        if (!sameBytes(first.instances, replay.instances) ||
            !sameBytes(first.assets, replay.assets) ||
            !sameBytes(first.sensors, replay.sensors) ||
            !sameBytes(first.appearances, replay.appearances)) {
            throw std::runtime_error(
                "world-family sampling is not deterministic"
            );
        }
        const std::uint32_t objectIndex =
            worldTemplate.assetIndex("pick_object");
        const float firstObjectX =
            first.assets[objectIndex].positionAndScale.x;
        const float lastObjectX =
            first.assets[
                (kWorldCount - 1u) * worldTemplate.assets.size() +
                objectIndex
            ].positionAndScale.x;
        if (std::abs(firstObjectX - lastObjectX) < 1.0e-6f) {
            throw std::runtime_error(
                "world-family object configurations collapsed"
            );
        }

        std::cout
            << "episode=\"" << episode.id << "\""
            << " template_fingerprint=" << worldTemplate.fingerprint
            << " family_fingerprint=" << family.fingerprint
            << " worlds=" << first.instances.size()
            << " asset_instances=" << first.assets.size()
            << " sensor_instances=" << first.sensors.size()
            << " variations=" << family.program.variations.size()
            << " capabilities=" << worldTemplate.capabilities
            << " deterministic=yes"
            << " gpu_ready=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_world_compiler_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
