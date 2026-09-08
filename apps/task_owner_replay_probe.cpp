// Native-owner regression for external actions and complete resident replay.
#include "metalrobo/c_api.h"
#include "metalrobo/MetalWorld.hpp"
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void checkProgramFingerprint() {
    metalrobo::MetalWorldMulticopterProgram a{};
    a.model.rotorCount = 4;
    a.articulationIndex = a.bodyIndex = a.firstAction = 0;
    auto b = a;
    const size_t start = offsetof(metalrobo::MetalWorldMulticopterProgram, firstAction) + sizeof(a.firstAction);
    const size_t end = offsetof(metalrobo::MetalWorldMulticopterProgram, windVelocity);
    require(end > start, "test requires the aligned wind padding");
    auto* aa = reinterpret_cast<unsigned char*>(&a);
    auto* bb = reinterpret_cast<unsigned char*>(&b);
    for (size_t i = start; i < end; ++i) { aa[i] = 0xa5; bb[i] = 0x5a; }
    const auto expected = a.fingerprint();
    require(expected != 0 && b.fingerprint() == expected, "host padding changed program identity");
    for (unsigned member = 0; member < 7; ++member) {
        auto changed = a;
        switch (member) {
        case 0: changed.model.coefficients.x = 1; break;
        case 1: changed.rotors[0].positionAndReactionSign.x = 1; break;
        case 2: changed.mixer.hoverAndScales.x = 1; break;
        case 3: changed.articulationIndex = 1; break;
        case 4: changed.bodyIndex = 1; break;
        case 5: changed.firstAction = 1; break;
        case 6: changed.windVelocity.x = 1; break;
        }
        require(changed.fingerprint() != expected, "authored member missing from identity");
    }
    b.model.rotorCount = 0;
    require(b.fingerprint() == 0, "invalid program admitted");
    std::cout << "PROGRAM_IDENTITY padding_independent=yes authored_members=7\n";
}
struct Run {
    std::vector<uint64_t> digests;
    std::vector<float> q;
    uint32_t contacts = 0;
};
Run execute(uint32_t source, uint32_t environments, const char* metallib, float sign) {
    MRRunManifestC manifest{};
    manifest.source = source;
    manifest.surface = MR_LOCOMOTION_SURFACE_GROUND;
    manifest.task = MR_UNITREE_G1_TASK_VELOCITY;
    manifest.profile.environment_count = environments;
    manifest.profile.physics_substeps = 1;
    manifest.profile.velocity_iterations = 16;
    manifest.profile.final_velocity_iterations = 32;
    manifest.profile.control_timestep_seconds = 0.001f;
    manifest.profile.seed = 0x4e554d49;
    manifest.metallib_path = metallib;
    std::unique_ptr<MRTaskRolloutHandle, decltype(&mr_task_rollout_destroy)> world(
        mr_create_task_rollout(&manifest), &mr_task_rollout_destroy);
    require(bool(world), std::string("construction: ") + mr_last_error());
    require(mr_task_rollout_set_state_readback(world.get(), 1) == 0, "readback setup");
    auto layout = mr_task_rollout_layout(world.get());
    require(layout.action_count > 0, "empty action table");
    std::vector<float> actions(size_t(environments) * layout.action_count);
    Run run;
    for (uint32_t step = 0; step < 4; ++step) {
        for (uint32_t env = 0; env < environments; ++env)
            actions[size_t(env) * layout.action_count] = sign * 0.01f * float(step + 1);
        MRTaskRolloutAdvanceC advance{};
        require(mr_task_rollout_advance(world.get(), actions.data(), actions.size(),
            nullptr, 0, 1, 100 + step, 0, &advance) == 0,
            "advance failed");
        require(advance.control_step_count == 1 &&
            advance.successful_environment_steps == environments &&
            advance.failed_environment_steps == 0 && advance.first_gpu_status_code == 0,
            "GPU or numerical step failure");
        run.contacts = std::max(run.contacts, advance.maximum_active_contacts);
        const auto digest = mr_task_rollout_resident_state_fingerprint(world.get());
        require(digest != 0 && digest == mr_task_rollout_resident_state_fingerprint(world.get()),
            "unstable idle state");
        run.digests.push_back(digest);
    }
    const float* q = mr_task_rollout_final_q(world.get());
    require(q != nullptr && layout.nq > 0, "missing physical state");
    run.q.assign(q, q + size_t(environments) * layout.nq);
    for (float value : run.q) require(std::isfinite(value), "nonfinite physical state");
    return run;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: metalrobo_task_owner_replay_probe METALLIB");
        checkProgramFingerprint();
        const std::array<std::pair<const char*, uint32_t>, 3> robots{{
            {"franka", MR_RUN_SOURCE_FRANKA_PICK_PLACE},
            {"g1", MR_RUN_SOURCE_UNITREE_G1}, {"x500", MR_RUN_SOURCE_PX4_X500}}};
        for (uint32_t environments : {1u, 2u}) {
            for (const auto& [name, source] : robots) {
                const auto original = execute(source, environments, argv[1], 1);
                const auto replay = execute(source, environments, argv[1], 1);
                const auto changed = execute(source, environments, argv[1], -1);
                require(original.digests == replay.digests, std::string(name) + " resident replay differs");
                require(original.q.size() == replay.q.size() &&
                    std::memcmp(original.q.data(), replay.q.data(), original.q.size() * sizeof(float)) == 0,
                    std::string(name) + " physical replay differs");
                require(original.digests != changed.digests, "changed actions had no effect");
                std::cout << "NATIVE_OWNER_REPLAY robot=" << name << " environments=" << environments
                    << " accepted_steps_per_run=4 runs=3 contacts=" << original.contacts
                    << " resident_replay=exact physical_replay=exact\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native owner regression failed: " << error.what() << '\n';
        return 1;
    }
}
