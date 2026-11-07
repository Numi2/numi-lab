#include "metalrobo/LearningPacks.hpp"
#include <filesystem>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace metalrobo;
    if (argc != 4) {
        std::cerr << "usage: actuator_pack_migrate main|numisolver INPUT OUTPUT\n"
                     "Specify the verified source branch for version-2 actuator tag 7.\n";
        return 2;
    }
    const std::string_view source{argv[1]};
    const auto origin = source == "main" ? RobotActuatorLegacyOrigin::mainMillard :
        source == "numisolver" ? RobotActuatorLegacyOrigin::numisolverMeasuredSurface :
        RobotActuatorLegacyOrigin::unspecified;
    if (origin == RobotActuatorLegacyOrigin::unspecified) {
        std::cerr << "invalid source branch\n"; return 2;
    }
    const std::filesystem::path outputPath{argv[3]};
    std::error_code error;
    if (std::filesystem::exists(outputPath, error) || error) {
        std::cerr << "output already exists or cannot be inspected\n"; return 1;
    }
    RobotActuatorPack pack;
    auto result = readRobotActuatorPack(argv[2], pack, origin);
    if (!result.succeeded()) { std::cerr << result.message << '\n'; return 1; }
    result = writeRobotActuatorPack(pack, outputPath);
    if (!result.succeeded()) { std::cerr << result.message << '\n'; return 1; }
    RobotActuatorPack check;
    result = readRobotActuatorPack(outputPath, check);
    if (!result.succeeded()) { std::cerr << "output verification failed: " << result.message << '\n'; return 1; }
    std::cout << "wrote version-3 actuator pack; source=" << source
              << " actuators=" << check.actuators.size() << '\n';
}
