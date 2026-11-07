// Compile the exact C API manifest authoring path in the historical owner.
// This executable only compiles CPU contracts and does not create a Metal
// device or submit a GPU workload.
#include "/Users/numi/numi-lab-release-integration-20261107/src/c_api.cpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
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
    const auto& task = run.task();
    metalrobo::PolicyPack pack;
    if (std::string(argv[1]) == "write") {
        pack.id = "synthetic-integrated-prefixed-delay-policy";
        pack.revision = 1u;
        const auto& layout = task.layout();
        pack.layers = {{.inputCount = layout.actorObservationSize, .outputCount = layout.actionCount,
            .activation = metalrobo::PolicyActivation::identity,
            .weights = std::vector<float>(static_cast<std::size_t>(layout.actorObservationSize) * layout.actionCount, 0.0f),
            .bias = std::vector<float>(layout.actionCount, 0.0f)}};
        metalrobo::bindPolicyPack(pack, task);
        auto written = metalrobo::writePolicyPack(pack, argv[2]);
        if (!written.succeeded()) { std::cerr << written.message << "\n"; return 4; }
    } else {
        auto read = metalrobo::readPolicyPack(argv[2], pack);
        if (!read.succeeded()) { std::cerr << read.message << "\n"; return 4; }
    }
    metalrobo::CompiledPolicyProgram compiled;
    auto check = metalrobo::compilePolicyProgram(pack, task, compiled);
    std::cout << "policy=" << pack.contract.worldFingerprint << "," << pack.contract.taskFingerprint << "," << pack.contract.observationFingerprint << "," << pack.contract.actionFingerprint << "\n";
    std::cout << "executor=" << task.worldFingerprint() << "," << task.fingerprint() << "," << task.observationFingerprint() << "," << task.actionFingerprint() << "\n";
    std::cout << "compile_status=" << static_cast<unsigned>(check.status) << " message=" << check.message << "\n";
    if (std::string(argv[1]) == "write") return check.succeeded() ? 0 : 5;
    return check.status == metalrobo::PolicyCompileStatus::incompatibleContract &&
        pack.contract.worldFingerprint == task.worldFingerprint() &&
        pack.contract.taskFingerprint == task.fingerprint() &&
        pack.contract.actionFingerprint == task.actionFingerprint() &&
        pack.contract.observationFingerprint != task.observationFingerprint() ? 0 : 6;
}
