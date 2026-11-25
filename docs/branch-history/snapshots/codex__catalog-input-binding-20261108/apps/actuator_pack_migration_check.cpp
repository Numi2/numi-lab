#include "metalrobo/LearningPacks.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace metalrobo;

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}
bool setVersion(const fs::path& path, const char version) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    const char encoded[4]{version,0,0,0};
    file.seekp(8);
    file.write(encoded, sizeof(encoded));
    return static_cast<bool>(file);
}
}
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "expected historical fixture directory\n"; return 2; }
    const fs::path dir = fs::temp_directory_path() /
        ("numi-actuator-pack-migration-" + std::to_string(getpid()));
    std::error_code error;
    if (!fs::create_directory(dir, error)) return 1;
    struct Cleanup { fs::path path; ~Cleanup() {
        std::error_code ignored; fs::remove_all(path, ignored);
    }} cleanup{dir};
    RobotActuatorPack pack;
    pack.id = "fixture";
    RobotActuatorSpec actuator;
    actuator.id = "drive"; actuator.target = "physical-owner";
    actuator.kind = RobotActuatorKind::millardExcitation;
    pack.actuators.push_back(actuator);
    const auto legacy = dir/"legacy.pack";
    if (!check(writeRobotActuatorPack(pack, legacy).succeeded() && setVersion(legacy, 2),
               "could not create legacy wire fixture")) return 1;
    RobotActuatorPack output;
    output.id = "unchanged";
    auto result = readRobotActuatorPack(legacy, output);
    if (!check(result.status == LearningPackStatus::unsupportedVersion &&
               result.message.find("ambiguous") != std::string::npos &&
               output.id == "unchanged", "ambiguous input did not fail atomically")) return 1;
    for (const auto [origin, expected, name] : {
        std::tuple{RobotActuatorLegacyOrigin::mainMillard, RobotActuatorKind::millardExcitation, "main"},
        std::tuple{RobotActuatorLegacyOrigin::numisolverMeasuredSurface, RobotActuatorKind::measuredSurface, "numisolver"}
    }) {
        result = readRobotActuatorPack(legacy, output, origin);
        if (!check(result.succeeded() && output.actuators.size() == 1 &&
                   output.actuators[0].kind == expected, "legacy provenance mapping failed")) return 1;
        const auto migrated = dir/(std::string(name)+".pack");
        if (!check(writeRobotActuatorPack(output, migrated).succeeded(), "migration write failed")) return 1;
        RobotActuatorPack roundtrip;
        result = readRobotActuatorPack(migrated, roundtrip);
        if (!check(result.succeeded() && roundtrip.id == pack.id &&
                   roundtrip.actuators.size() == 1 &&
                   roundtrip.actuators[0].kind == expected &&
                   roundtrip.actuators[0].target == actuator.target,
                   "version-3 migration round trip failed")) return 1;
    }
    pack.actuators[0].kind = RobotActuatorKind::jointPosition;
    if (!check(writeRobotActuatorPack(pack, legacy).succeeded() && setVersion(legacy, 2),
               "could not create unambiguous legacy fixture")) return 1;
    output = {};
    result = readRobotActuatorPack(legacy, output);
    if (!check(result.succeeded() && output.actuators.size() == 1 &&
               output.actuators[0].kind == RobotActuatorKind::jointPosition,
               "unambiguous legacy input failed")) return 1;
    TaskPack task;
    task.id = "legacy-task-fixture";
    task.actions.push_back({"drive"});
    task.maximumEpisodeSteps = 321;
    const auto taskPath = dir/"task.pack";
    for (char legacyVersion : {char{17}, char{19}}) {
        if (!check(writeTaskPack(task, taskPath).succeeded() &&
                   setVersion(taskPath, legacyVersion), "task fixture write failed")) return 1;
        TaskPack restored;
        result = readTaskPack(taskPath, restored);
        if (!check(result.succeeded() && restored.id == task.id &&
                   restored.maximumEpisodeSteps == 321 && restored.actions.size() == 1 &&
                   restored.actions[0].actuator == "drive", "legacy task decode failed")) return 1;
        const auto migratedTask = dir/("task-" + std::to_string(legacyVersion) + ".pack");
        if (!check(writeTaskPack(restored, migratedTask).succeeded(), "task migration write failed")) return 1;
        TaskPack roundtrip;
        result = readTaskPack(migratedTask, roundtrip);
        if (!check(result.succeeded() && roundtrip.id == task.id &&
                   roundtrip.maximumEpisodeSteps == 321 &&
                   roundtrip.actions.size() == 1 && roundtrip.actions[0].actuator == "drive",
                   "task version-20 round trip failed")) return 1;
    }
    if (!check(setVersion(taskPath, 18), "unsupported task fixture write failed")) return 1;
    TaskPack sentinel;
    sentinel.id = "unchanged";
    result = readTaskPack(taskPath, sentinel);
    if (!check(!result.succeeded() && sentinel.id == "unchanged",
               "unsupported task version did not fail atomically")) return 1;
    const fs::path historical(argv[1]);
    for (const char* branch : {"main", "numisolver"}) {
        TaskPack legacyTask;
        result = readTaskPack(historical/branch/"task.pack", legacyTask);
        if (!check(result.succeeded() && legacyTask.id == "historical-task" &&
                   legacyTask.maximumEpisodeSteps == 321 &&
                   legacyTask.actions.size() == 1 &&
                   legacyTask.actions[0].actuator == "drive",
                   "historical producer task failed to decode")) return 1;
        const auto upgraded = dir/(std::string(branch)+"-historical-task.pack");
        if (!check(writeTaskPack(legacyTask, upgraded).succeeded(),
                   "historical task migration failed")) return 1;
        TaskPack restored;
        result = readTaskPack(upgraded, restored);
        if (!check(result.succeeded() && restored.id == legacyTask.id &&
                   restored.maximumEpisodeSteps == legacyTask.maximumEpisodeSteps &&
                   restored.actions[0].actuator == "drive",
                   "historical task migration round trip failed")) return 1;
    }
    RobotActuatorPack historicalMain;
    historicalMain.id = "unchanged";
    result = readRobotActuatorPack(historical/"main"/"actuator.pack", historicalMain);
    if (!check(result.status == LearningPackStatus::unsupportedVersion &&
               historicalMain.id == "unchanged",
               "historical main tag 7 did not reject ambiguity")) return 1;
    result = readRobotActuatorPack(historical/"main"/"actuator.pack", historicalMain,
                                   RobotActuatorLegacyOrigin::mainMillard);
    if (!check(result.succeeded() && historicalMain.actuators.size() == 1 &&
               historicalMain.actuators[0].kind == RobotActuatorKind::millardExcitation,
               "historical main actuator migration failed")) return 1;
    const auto migratedHistorical = dir/"main-historical-actuator.pack";
    RobotActuatorPack historicalRoundtrip;
    if (!check(writeRobotActuatorPack(historicalMain, migratedHistorical).succeeded() &&
               readRobotActuatorPack(migratedHistorical, historicalRoundtrip).succeeded() &&
               historicalRoundtrip.id == historicalMain.id &&
               historicalRoundtrip.actuators.size() == 1 &&
               historicalRoundtrip.actuators[0].kind == RobotActuatorKind::millardExcitation &&
               historicalRoundtrip.actuators[0].target == historicalMain.actuators[0].target,
               "historical main actuator migration round trip failed")) return 1;
    RobotActuatorPack historicalNumisolver;
    result = readRobotActuatorPack(historical/"numisolver"/"actuator.pack", historicalNumisolver);
    if (!check(result.succeeded() && historicalNumisolver.actuators.size() == 1 &&
               historicalNumisolver.actuators[0].kind == RobotActuatorKind::bodyWrench,
               "historical numisolver unambiguous actuator failed")) return 1;
    std::cout << "synthetic and historical producer pack migration passed\n";
}
