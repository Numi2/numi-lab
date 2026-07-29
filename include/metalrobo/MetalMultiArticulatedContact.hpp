#pragma once

#include "metalrobo/MetalQualityContactSolver.hpp"
#include "metalrobo/MultiArticulatedContact.hpp"
#include "metalrobo/ParallelABASchedule.hpp"
#include "metalrobo/multi_contact_shared.h"
#include "metalrobo/parallel_aba_shared.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalMultiArticulatedContactDiagnostics;

// Immutable articulation topology and frontier schedule shared by standalone
// Metal and the upcoming MLX encoder adapter. Model validation and schedule
// cooking happen once rather than inside every contact submission.
class CompiledMetalMultiArticulatedContactProgram {
public:
    CompiledMetalMultiArticulatedContactProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] const ParallelABASchedule& abaSchedule()
        const noexcept;
    [[nodiscard]] std::span<const float>
    generalizedEqualityJacobian() const noexcept;
    [[nodiscard]] std::uint32_t equalityRowCount()
        const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;

private:
    friend MetalMultiArticulatedContactDiagnostics
    compileMetalMultiArticulatedContactProgram(
        const EngineModel&,
        CompiledMetalMultiArticulatedContactProgram&
    );

    EngineModel model_;
    ParallelABASchedule schedule_;
    std::vector<float> generalizedEqualityJacobian_;
    std::uint64_t fingerprint_ = 0u;
};

struct MetalMultiArticulatedContactInput {
    std::size_t environmentCount = 0u;
    std::size_t contactCount = 0u;
    std::size_t sceneBodyCount = 0u;
    // Environment-major global articulated state.
    std::span<const float> q{};
    std::span<const float> freeArticulationVelocity{};
    // Environment-major maximal-coordinate state.
    std::span<const MRBodyStateGPU> sceneBodies{};
    // Environment-major contacts. Endpoint kind/body topology must be
    // identical for the same contact slot across cloned environments; local
    // points, frame, targets, regularization, warm start and friction may vary.
    std::span<const MultiArticulatedIslandContact> contacts{};
};

struct MetalMultiArticulatedContactConfig {
    MetalQualityContactSolverConfig quality{};
    float delassusSymmetryTolerance = 2.0e-4f;
    float delassusDiagonalTolerance = 2.0e-5f;
    ConstraintIREvaluationConfig equalityEvaluation{};
    float equalityPivotTolerance = 2.0e-6f;
    float equalityResidualTolerance = 2.0e-4f;
    std::string metallibPath;
};

enum class MetalMultiArticulatedContactStatus :
    std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidModel,
    unsupportedTopology,
    invalidDimensions,
    capacityOverflow,
    arithmeticOverflow,
    nonfiniteInput,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuEnvironmentFailure,
    nonfiniteResult,
    internalFailure,
};

struct MetalMultiArticulatedContactLayout {
    MRMultiContactDispatchGPU dispatch{};
    std::vector<MRArticulatedOperatorDispatchGPU>
        pointDispatches;
    std::vector<MRMultiContactJacobianSliceGPU>
        pointSlices;
    std::vector<MRMultiInverseMassDispatchGPU>
        inverseMassDispatches;
    std::size_t pointQueryElements = 0u;
    std::size_t pointJacobianElements = 0u;
    std::size_t packedVelocityElements = 0u;
    std::size_t jacobianElements = 0u;
    std::size_t responseRowElements = 0u;
    std::size_t delassusElements = 0u;
    std::size_t rowElements = 0u;
    std::size_t equalityOperatorElements = 0u;
    std::size_t equalityCouplingElements = 0u;
    std::size_t equalityImpulseElements = 0u;
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalMultiArticulatedContactResult {
    MetalMultiArticulatedContactLayout layout;
    // Environment-major [articulated nv, dynamic scene-body 6D blocks].
    std::vector<float> nextVelocity;
    std::vector<float> impulses;
    std::vector<float> delassus;
    std::vector<float> freeContactVelocity;
    std::vector<float> equalityImpulses;
    std::vector<MRMultiContactStatusGPU> statuses;
    std::vector<MRMultiContactEqualityStatusGPU>
        equalityStatuses;
    std::vector<MRArticulatedOperatorStatusGPU>
        pointStatuses;
    std::vector<MRInverseMassStatusGPU>
        inverseMassStatuses;
    std::vector<MRMetalQualityStatusGPU> qualityStatuses;
};

struct MetalMultiArticulatedContactDiagnostics {
    MetalMultiArticulatedContactStatus status =
        MetalMultiArticulatedContactStatus::success;
    MetalMultiArticulatedContactLayout layout;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_MULTI_CONTACT_SUCCESS;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MetalMultiArticulatedContactStatus::success;
    }
};

// Transactionally validates and snapshots a multi-articulation model and its
// deterministic frontier schedule. Failure leaves output unchanged.
[[nodiscard]] MetalMultiArticulatedContactDiagnostics
compileMetalMultiArticulatedContactProgram(
    const EngineModel& model,
    CompiledMetalMultiArticulatedContactProgram& output
);

// Executes point-Jacobian projection, global contact/equality row assembly,
// articulated and 6D scene-body inverse response, exact generalized-equality
// Schur reduction, projected Delassus construction, exact-cone Metal quality
// solve, equality reconstruction/certification, and transactional velocity
// publication in one command buffer. There is no CPU collision/solver
// fallback and no intermediate wait.
[[nodiscard]] MetalMultiArticulatedContactDiagnostics
solveMetalMultiArticulatedContacts(
    const EngineModel& model,
    const MetalMultiArticulatedContactInput& input,
    MetalMultiArticulatedContactResult& output,
    const MetalMultiArticulatedContactConfig& config = {}
);

// Reuses the immutable model and parallel-ABA schedule cooked above.
[[nodiscard]] MetalMultiArticulatedContactDiagnostics
solveMetalMultiArticulatedContacts(
    const CompiledMetalMultiArticulatedContactProgram& program,
    const MetalMultiArticulatedContactInput& input,
    MetalMultiArticulatedContactResult& output,
    const MetalMultiArticulatedContactConfig& config = {}
);

[[nodiscard]] const char*
metalMultiArticulatedContactStatusName(
    MetalMultiArticulatedContactStatus status
) noexcept;

} // namespace metalrobo
