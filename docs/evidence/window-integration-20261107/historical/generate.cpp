#include "metalrobo/LearningPacks.hpp"
#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    using namespace metalrobo;
    if (argc != 2) return 2;
    const std::filesystem::path out(argv[1]);
    std::filesystem::create_directories(out);
    TaskPack task;
    task.id = "historical-task";
    task.actions.push_back({"drive"});
    task.maximumEpisodeSteps = 321u;
    auto result = writeTaskPack(task, out/"task.pack");
    if (!result.succeeded()) { std::cerr << result.message << '\n'; return 1; }
    RobotActuatorPack robot;
    robot.id = "historical-robot";
    RobotActuatorSpec actuator;
    actuator.id = "drive";
    actuator.target = "owner";
#ifdef NUMISOLVER_LINEAGE
    actuator.kind = RobotActuatorKind::bodyWrench;
#else
    actuator.kind = RobotActuatorKind::millardExcitation;
#endif
    robot.actuators.push_back(actuator);
    result = writeRobotActuatorPack(robot, out/"actuator.pack");
    if (!result.succeeded()) { std::cerr << result.message << '\n'; return 1; }
#ifdef NUMISOLVER_LINEAGE
    robot.actuators[0].kind = RobotActuatorKind::measuredSurface;
    result = writeRobotActuatorPack(robot, out/"measured.pack");
    if (result.succeeded()) {
        std::cerr << "unexpected historical measured-surface write success\n"; return 1;
    }
    std::cout << "measured-surface v2 writer rejection: " << result.message << '\n';
#endif
    std::cout << "historical branch task and actuator written\n";
}
