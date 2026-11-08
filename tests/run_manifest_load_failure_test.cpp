#include "metalrobo/c_api.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const fs::path fixture = argv[1];
    const fs::path temp = fs::temp_directory_path() /
        ("numi-owner-error-" + std::to_string(getpid()));
    if (!fs::create_directory(temp)) return 2;
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); } } cleanup{temp};
    const std::array<std::string,4> paths = {
        (fixture/"task.taskpack").string(), (fixture/"actuators.actuatorpack").string(),
        (fixture/"sensors.sensorpack").string(), (fixture/"reality.realitypack").string()
    };
    const std::array<const char*,4> names = {
        "TaskPack", "RobotActuatorPack", "SensorProgramPack", "RealityProgramPack"
    };
    for (uint32_t source : {MR_RUN_SOURCE_WORLD_PACK, MR_RUN_SOURCE_IMPORTED_URDF}) {
        for (std::size_t missing = 0; missing < paths.size(); ++missing) {
            auto selected = paths;
            selected[missing] = (temp / ("missing-" + std::to_string(missing))).string();
            std::string world = (fixture/"world.mrworld").string();
            MRRunManifestC manifest{};
            manifest.source = source;
            manifest.world_pack_path = world.c_str();
            manifest.urdf_path = "/unused-because-pack-load-fails.urdf";
            manifest.task_pack_path = selected[0].c_str();
            manifest.robot_actuator_pack_path = selected[1].c_str();
            manifest.sensor_program_pack_path = selected[2].c_str();
            manifest.reality_program_pack_path = selected[3].c_str();
            if (auto* result = mr_create_task_rollout(&manifest)) {
                mr_task_rollout_destroy(result);
                std::cerr << "missing owner unexpectedly accepted\n";
                return 1;
            }
            const std::string error = mr_last_error();
            if (error.find(names[missing]) == std::string::npos ||
                error.find("io_failure") == std::string::npos ||
                error.find(selected[missing]) == std::string::npos ||
                error.find("learning pack is missing or has an invalid length") == std::string::npos) {
                std::cerr << "bad diagnostic for " << names[missing] << ": " << error << "\n";
                return 1;
            }
        }
    }
    const fs::path corrupt = temp/"corrupt.taskpack";
    fs::copy_file(paths[0], corrupt);
    { std::fstream f(corrupt, std::ios::in | std::ios::out | std::ios::binary); f.put('X'); }
    std::string corruptPath = corrupt.string();
    MRRunManifestC manifest{};
    manifest.source = MR_RUN_SOURCE_WORLD_PACK;
    manifest.world_pack_path = "/unused.world";
    manifest.task_pack_path = corruptPath.c_str();
    manifest.robot_actuator_pack_path = paths[1].c_str();
    manifest.sensor_program_pack_path = paths[2].c_str();
    manifest.reality_program_pack_path = paths[3].c_str();
    auto* result = mr_create_task_rollout(&manifest);
    if (result) { mr_task_rollout_destroy(result); return 1; }
    const std::string error = mr_last_error();
    if (error.find("TaskPack") == std::string::npos ||
        error.find("corrupt_payload") == std::string::npos ||
        error.find(corruptPath) == std::string::npos ||
        error.find("header or kind is invalid") == std::string::npos) {
        std::cerr << "bad corrupt-pack diagnostic: " << error << "\n";
        return 1;
    }
    std::cout << "run owner pack diagnostics passed\n";
    return 0;
}
