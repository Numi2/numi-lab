// Compile the exact C API manifest authoring path in the historical owner.
// This executable only compiles CPU contracts and does not create a Metal
// device or submit a GPU workload.
#include "/Users/numi/numi-g1-policy-origin-20261107/src/c_api.cpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    MRTaskRolloutConfigC config{};
    config.environment_count = 8u;
    config.physics_substeps = 4u;
    config.velocity_iterations = 4u;
    config.final_velocity_iterations = 2u;
    config.control_timestep_seconds = 0.02f;
    config.seed = 4081007u;
    auto manifest = makeUnitreeG1RunManifest(
        metalrobo::LocomotionSurface::ground,
        metalrobo::UnitreeG1Task::g1LegsLocomotion, config);
    manifest.teacher.id = "no_teacher";
    manifest.profile.capacities = manifest.task.capacities;
    metalrobo::CompiledRun run;
    auto status = metalrobo::compileRun(manifest, run);
    if (!status.succeeded()) {
        std::cerr << status.element << ": " << status.message << "\n";
        return 3;
    }
    metalrobo::PolicyPack pack;
    auto read = metalrobo::readPolicyPack(argv[1], pack);
    if (!read.succeeded()) { std::cerr << read.message << "\n"; return 4; }
    const auto& task = run.task();
    std::cout << "policy world=" << pack.contract.worldFingerprint
              << " task=" << pack.contract.taskFingerprint
              << " observation=" << pack.contract.observationFingerprint
              << " action=" << pack.contract.actionFingerprint << "\n";
    std::cout << "executor world=" << task.worldFingerprint()
              << " task=" << task.fingerprint()
              << " observation=" << task.observationFingerprint()
              << " action=" << task.actionFingerprint() << "\n";
    metalrobo::CompiledPolicyProgram compiled;
    auto check = metalrobo::compilePolicyProgram(pack, task, compiled);
    if (!check.succeeded()) {
        std::cerr << check.element << ": " << check.message << "\n";
        return 5;
    }
    std::cout << "PASS exact policy contract\n";
}
