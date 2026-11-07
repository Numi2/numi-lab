#include "metalrobo/RunProgram.hpp"

#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/LocomotionWorld.hpp"
#include "metalrobo/TaskProgram.hpp"
#include <array>
#include <cstdio>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: depth_contract_check temporary.sensorpack\n";
    return 1;
  }
  using namespace metalrobo;
  auto world = makeUnitreeG1LocomotionWorld(LocomotionSurface::ground);
  const auto robot = builtinRobotPack("unitree_g1");
  if (!robot) return 1;
  TaskObservationProgram obs;
  TaskResetProgram reset;
  const auto task = makeUnitreeG1BallDodgeTaskPack(LocomotionSurface::ground, obs, reset);
  std::array<LocomotionDynamicSphere, 6> spheres{};
  for (int i=0;i<6;i++) {
    spheres[i].position = {-2.0f, float(i)*0.25f, 1.0f, 1.0f};
    spheres[i].linearVelocity = {3.0f, 0.0f, 1.0f, 0.0f};
    spheres[i].radius = 0.1f + 0.02f * i;
    spheres[i].mass = 0.1f + 0.1f * i;
    spheres[i].launchStep = 100u * (i+1);
  }
  appendLocomotionDynamicSpheres(world, spheres);
  CompiledLocomotionWorld oracle, raw;
  auto a = compileLocomotionWorld(world, world.articulationIndex, task, robot->actuators, obs, reset, oracle);
  if (!a.succeeded()) { std::cerr << a.world.message << " " << a.task.message << "\n"; return 2; }
  if (oracle.task.header().schedule.w & MR_TASK_PROGRAM_UNSEGMENTED_DEPTH) return 3;
  obs.visual.useUnsegmentedDepth = true;
  auto b = compileLocomotionWorld(world, world.articulationIndex, task, robot->actuators, obs, reset, raw);
  if (!b.succeeded()) { std::cerr << b.world.message << " " << b.task.message << "\n"; return 4; }
  if (!(raw.task.header().schedule.w & MR_TASK_PROGRAM_UNSEGMENTED_DEPTH) ||
      raw.task.layout().actorObservationSize != oracle.task.layout().actorObservationSize ||
      raw.task.observationFingerprint() == oracle.task.observationFingerprint() ||
      raw.task.fingerprint() == oracle.task.fingerprint()) return 5;
  const char* path = argv[1];
  SensorProgramPack packed{.id="test", .observation=obs}, read;
  if (!writeSensorProgramPack(packed,path).succeeded() || !readSensorProgramPack(path,read).succeeded() ||
      !read.observation.visual.useUnsegmentedDepth || !read.observation.visual.includeDerivedFeatures) return 6;
  std::printf("PASS actor=%u oracle_obs=%llu raw_obs=%llu persisted=1\n",raw.task.layout().actorObservationSize,
    (unsigned long long)oracle.task.observationFingerprint(), (unsigned long long)raw.task.observationFingerprint());
}
