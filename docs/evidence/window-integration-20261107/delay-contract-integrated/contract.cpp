// Compile the exact C API manifest authoring path in the historical owner.
// This executable only compiles CPU contracts and does not create a Metal
// device or submit a GPU workload.
#include "/Users/numi/numi-lab-release-integration-20261107/src/c_api.cpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    MRTaskRolloutConfigC config{};
    config.environment_count = 8u;
    config.physics_substeps = 4u;
    config.velocity_iterations = 4u;
    config.final_velocity_iterations = 2u;
    config.control_timestep_seconds = 0.02f;
    config.seed = 4081010u;
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
    if (check.status != metalrobo::PolicyCompileStatus::incompatibleContract) {
        std::cerr << "expected incompatibility, got: " << check.message << "\n";
        return 5;
    }
    if (pack.contract.worldFingerprint != task.worldFingerprint() ||
        pack.contract.taskFingerprint != task.fingerprint() ||
        pack.contract.actionFingerprint != task.actionFingerprint() ||
        pack.contract.observationFingerprint == task.observationFingerprint()) {
        std::cerr << "contract difference was not isolated to observation\n";
        return 6;
    }
    std::cout << "PASS old policy rejected by changed observation semantics\n";
    return 0;
    std::cout << "policy_program_fingerprint=" << compiled.fingerprint() << "\n";
    std::cout << "PASS exact policy contract\n";
}
