#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/LearningPacks.hpp"
#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/RunProgram.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace metalrobo;
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: probe write|read directory");
        const bool write = std::string(argv[1]) == "write";
        if (!write && std::string(argv[1]) != "read")
            throw std::runtime_error("invalid phase");
        RunManifest m;
        m.id = "franka_depth_contract";
        m.robot = *builtinRobotPack("franka_panda");
        m.scene = makeFrankaPickPlaceScenePack();
        m.sensors.id = "default";
        m.task = makeFrankaPickPlaceTaskPack(m.sensors.observation, m.reality.reset);
        m.reality.id = "nominal";
        m.profile.id = "test";
        m.profile.environmentCount = 1;
        m.profile.controlSteps = 1;
        m.profile.physicsSubsteps = 1;
        m.profile.controlTimestepSeconds = .02f;
        m.teacher.id = "none";
        CompiledRun base;
        auto rs = compileRun(m, base);
        if (!rs.succeeded())
            throw std::runtime_error("base run: " + rs.message);
        for (int mode = 0; mode < 4; ++mode) {
            auto obs = m.sensors.observation;
            if (mode >= 2) {
                auto objectTrack = [](const TaskObservationOperatorSpec& op) {
                    return op.source == TaskObservationSource::objectTrack;
                };
                std::erase_if(obs.actorFrame, objectTrack);
                std::erase_if(obs.critic, objectTrack);
            }
            if ((mode % 2) != 0) {
                obs.visual.width = 4;
                obs.visual.height = 2;
                obs.visual.frameOffsets = {0};
                for (unsigned i = 0; i < 8; ++i)
                    obs.actorFrame.push_back(
                        {.source = TaskObservationSource::maskedDepth, .component = i});
            }
            CompiledTaskProgram task;
            auto ts = compileTaskProgram(m.task, m.robot.actuators, obs, m.reality.reset,
                                         base.world(), task);
            if (!ts.succeeded())
                throw std::runtime_error("task mode " + std::to_string(mode) + ": " +
                                         ts.message);
            const auto path =
                std::filesystem::path(argv[2]) / (std::to_string(mode) + ".npolicy");
            PolicyPack policy;
            if (write) {
                policy.id = "cpu_only_zero_policy";
                policy.layers.push_back(
                    {.inputCount = task.layout().actorObservationSize,
                     .outputCount = task.layout().actionCount,
                     .activation = PolicyActivation::identity,
                     .weights = std::vector<float>(task.layout().actorObservationSize *
                                                       task.layout().actionCount,
                                                   0.f),
                     .bias = std::vector<float>(task.layout().actionCount, 0.f)});
                bindPolicyPack(policy, task);
                auto st = writePolicyPack(policy, path);
                if (!st.succeeded())
                    throw std::runtime_error("write failed: " + st.message);
            } else {
                auto st = readPolicyPack(path, policy);
                if (!st.succeeded())
                    throw std::runtime_error("read failed: " + st.message);
            }
            if (!write) {
                const PolicyContract old = policy.contract;
                if (old.worldFingerprint != task.worldFingerprint() ||
                    old.actionFingerprint != task.actionFingerprint()) {
                    throw std::runtime_error("fixture world or action changed");
                }
                if (old.taskFingerprint != task.fingerprint()) {
                    CompiledPolicyProgram unchanged;
                    const auto incompatible =
                        compilePolicyProgram(policy, task, unchanged);
                    if (incompatible.status !=
                        PolicyCompileStatus::incompatibleContract) {
                        throw std::runtime_error(
                            "different old task unexpectedly accepted");
                    }
                    // Normalize only the task identity on this in-memory test copy.
                    // Retain the old observation fingerprint under test.
                    bindPolicyPack(policy, task);
                    policy.contract.observationFingerprint = old.observationFingerprint;
                    std::cout << "mode=" << mode << " test_only_task_rebind=1\n";
                }
                // Mode 3 has no object track but does contain masked depth;
                // its older sample semantics are independently incompatible.
                if ((mode == 2 &&
                     old.observationFingerprint != task.observationFingerprint()) ||
                    (mode != 2 &&
                     old.observationFingerprint == task.observationFingerprint())) {
                    throw std::runtime_error("observation fingerprint migration gate");
                }
            }
            CompiledPolicyProgram out;
            auto ps = compilePolicyProgram(policy, task, out);
            std::cout << "mode=" << mode
                      << " phase=" << (write ? "baseline" : "candidate")
                      << " size=" << task.layout().actorObservationSize
                      << " old_obs=" << policy.contract.observationFingerprint
                      << " task_obs=" << task.observationFingerprint()
                      << " status=" << static_cast<int>(ps.status)
                      << " message=" << ps.message << '\n';
            if (write && !ps.succeeded())
                throw std::runtime_error("baseline self compile");
            if (!write &&
                ((mode == 2 && !ps.succeeded()) ||
                 (mode != 2 && ps.status != PolicyCompileStatus::incompatibleContract)))
                throw std::runtime_error("migration gate");
            if (!write && mode != 2) {
                bindPolicyPack(policy, task);
                CompiledPolicyProgram rebound;
                auto c = compilePolicyProgram(policy, task, rebound);
                if (!c.succeeded())
                    throw std::runtime_error("new contract failure");
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "failure: " << e.what() << '\n';
        return 2;
    }
}
